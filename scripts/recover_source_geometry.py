"""Recover complete instance-segmentation geometry for the current source state.

Phase 4A recorded 1,022 current annotations as carrying a bounding box and no
usable geometry. Phase 5A establishes that this was a consumption gap: those
instances are ``mask``-type boxes whose geometry travels inline, base64-wrapped
zlib-compressed COCO run-length encoding, in a field the earlier walk ignored.

This script re-walks the provider's source inventory read-only and recovers every
annotation's geometry in *original image coordinates*, classifying each one and
re-measuring it against the provider's own declared area and box.

Writes:
    data/interim/source_geometry.jsonl        git-ignored; full geometry
    reports/source_geometry_summary.json      committed; counts only, no payloads

Usage:
    uv run python scripts/recover_source_geometry.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from construction_safety_vision.config import ConfigError, load_experiment_config
from construction_safety_vision.data.roboflow import (
    DatasetCoordinates,
    MissingApiKeyError,
    RoboflowClient,
    RoboflowError,
    api_key_from_env,
)
from construction_safety_vision.data.source import InventoryError, collect_source_images
from construction_safety_vision.data.sourcegeometry import (
    GEOMETRY_KINDS,
    UNSUPPORTED,
    GeometryRecoveryError,
    RecoveredImage,
    recover_image,
)
from construction_safety_vision.env import load_project_env
from construction_safety_vision.paths import ProjectPaths
from construction_safety_vision.provenance import git_commit

EXPECTED_SOURCE_IMAGES = 436
"""Independent source population established in phase 3 and re-verified in 4A."""

EXPECTED_ANNOTATIONS = 2031
"""Current-state annotation total measured in phase 4A."""

GEOMETRY_FILENAME = "source_geometry.jsonl"
"""Interim file holding the recovered geometry. Git-ignored: it is bulk data."""

SUMMARY_FILENAME = "source_geometry_summary.json"
"""Committed summary of the recovery. Counts and identities only."""


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write rows as newline-delimited JSON with LF endings.

    Args:
        path: Destination file.
        rows: Rows to serialise.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows)
    path.write_text(f"{body}\n" if body else "", encoding="utf-8", newline="\n")


def write_json(path: Path, payload: dict) -> None:
    """Write a JSON document with LF endings and stable key order.

    Args:
        path: Destination file.
        payload: Document to serialise.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    path.write_text(f"{text}\n", encoding="utf-8", newline="\n")


def summarise(images: list[RecoveredImage]) -> dict:
    """Aggregate the recovery into committable counts.

    Args:
        images: Every recovered source image.

    Returns:
        A JSON-serialisable summary carrying no geometry payload.
    """
    geometry: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    per_split: Counter[str] = Counter()
    annotations_per_split: Counter[str] = Counter()
    unsupported: list[dict] = []
    provider_types: Counter[str] = Counter()
    deltas_by_kind: dict[str, list[float]] = {}

    for image in images:
        per_split[image.split] += 1
        annotations_per_split[image.split] += len(image.annotations)
        for annotation in image.annotations:
            geometry[annotation.geometry_kind] += 1
            labels[annotation.label] += 1
            provider_types[annotation.provider_type or "<none>"] += 1
            if annotation.bbox_delta_px is not None:
                deltas_by_kind.setdefault(annotation.geometry_kind, []).append(
                    annotation.bbox_delta_px
                )
            if annotation.geometry_kind == UNSUPPORTED:
                unsupported.append(
                    {
                        "image_id": annotation.image_id,
                        "annotation_id": annotation.annotation_id,
                        "label": annotation.label,
                        "provider_type": annotation.provider_type,
                        "bbox": [round(v, 3) for v in annotation.bbox],
                        "confidence": annotation.confidence,
                        "notes": annotation.notes,
                    }
                )

    total = sum(geometry.values())
    stored_box_agreement = {
        kind: {
            "measured": len(values),
            "agree_within_1px": sum(1 for v in values if v <= 1.0),
            "max_delta_px": round(max(values), 3),
            "mean_delta_px": round(sum(values) / len(values), 4),
        }
        for kind, values in sorted(deltas_by_kind.items())
        if values
    }
    return {
        "source_images": len(images),
        "annotations": total,
        "geometry_kind_counts": {kind: geometry.get(kind, 0) for kind in GEOMETRY_KINDS},
        "annotations_with_mask_geometry": total - geometry.get(UNSUPPORTED, 0),
        "unsupported_annotations": unsupported,
        "class_counts": dict(sorted(labels.items())),
        "provider_type_counts": dict(sorted(provider_types.items())),
        "images_per_provider_split": dict(sorted(per_split.items())),
        "annotations_per_provider_split": dict(sorted(annotations_per_split.items())),
        "stored_box_vs_recovered_geometry": stored_box_agreement,
        "coordinate_frame": "original source image pixels",
        "recovery_method": (
            "read-only GET {workspace}/{project}/images/{id}; polygon boxes from 'points', "
            "mask boxes from base64(zlib(COCO compressed RLE counts)) in 'mask', decoded "
            "against the full original canvas in [height, width] order"
        ),
        "verification": (
            "every recovered instance re-measured with pycocotools; measured area and "
            "derived bbox checked against the provider's declared area and box"
        ),
        "git_commit": git_commit(),
    }


def main(argv: list[str] | None = None) -> int:
    """Recover current-state geometry for the whole source population.

    Args:
        argv: Command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="Configuration file to use.")
    args = parser.parse_args(argv)

    paths = ProjectPaths.from_root()
    load_project_env(paths.root)
    try:
        config = load_experiment_config(args.config or (paths.configs / "project.yaml"))
        api_key = api_key_from_env()
    except (ConfigError, MissingApiKeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    dataset = config.dataset
    coordinates = DatasetCoordinates(
        workspace=dataset.workspace,
        project=dataset.project_slug,
        version=dataset.version,
        export_format=dataset.export_format,
    )
    client = RoboflowClient(api_key)

    print(f"walking source inventory of {coordinates.project_path} ...", flush=True)
    try:
        walk = collect_source_images(client, coordinates)
    except (RoboflowError, InventoryError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3
    print(f"source images listed: {len(walk.records)}")

    print("recovering annotation geometry ...", flush=True)
    images: list[RecoveredImage] = []
    for index, record in enumerate(walk.records, start=1):
        try:
            images.append(recover_image(client.image_details(coordinates, record.image_id)))
        except (RoboflowError, GeometryRecoveryError) as exc:
            print(f"ERROR on {record.image_id}: {client.redact(str(exc))}", file=sys.stderr)
            return 4
        if index % 50 == 0 or index == len(walk.records):
            print(f"  {index}/{len(walk.records)}", flush=True)

    images.sort(key=lambda image: image.image_id)
    summary = summarise(images)

    write_jsonl(
        paths.data_interim / GEOMETRY_FILENAME,
        [
            {
                "image_id": image.image_id,
                "name": image.name,
                "split": image.split,
                "width": image.width,
                "height": image.height,
                "annotations": [a.geometry_entry() for a in image.annotations],
            }
            for image in images
        ],
    )
    write_json(paths.reports / SUMMARY_FILENAME, summary)

    print("\n--- recovery summary ---")
    print(f"source images:        {summary['source_images']}")
    print(f"annotations:          {summary['annotations']}")
    for kind, count in summary["geometry_kind_counts"].items():
        print(f"  {kind:<24} {count}")
    print(f"with mask geometry:   {summary['annotations_with_mask_geometry']}")
    print(f"class counts:         {summary['class_counts']}")
    print("stored box vs recovered geometry:")
    for kind, stats in summary["stored_box_vs_recovered_geometry"].items():
        print(
            f"  {kind:<10} agree<=1px {stats['agree_within_1px']}/{stats['measured']}  "
            f"max {stats['max_delta_px']} px  mean {stats['mean_delta_px']} px"
        )

    problems: list[str] = []
    if summary["source_images"] != EXPECTED_SOURCE_IMAGES:
        problems.append(
            f"expected {EXPECTED_SOURCE_IMAGES} source images, recovered {summary['source_images']}"
        )
    if summary["annotations"] != EXPECTED_ANNOTATIONS:
        problems.append(
            f"expected {EXPECTED_ANNOTATIONS} annotations, recovered {summary['annotations']}"
        )
    if problems:
        # Not fatal: the provider project is live and may legitimately have moved
        # on. It must be visible and reconciled, never silently absorbed.
        print("\nWARNING: recovery differs from the phase 4A measurement:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)

    print(f"\nwrote {paths.data_interim / GEOMETRY_FILENAME} (git-ignored)")
    print(f"wrote {paths.reports / SUMMARY_FILENAME}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
