"""Recover the 436-image source inventory and its authoritative annotations.

Phase 3 established that the exported 742 images are not 742 independent
samples. This script recovers the population that *is* independent - the source
images the provider holds - together with the annotation geometry in original
image coordinates.

Writes:
    reports/source_image_manifest.jsonl   committed; no geometry, no URLs, no paths
    data/interim/source_annotations.jsonl git-ignored; per-instance geometry summary

Usage:
    uv run python scripts/fetch_source_inventory.py
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
from construction_safety_vision.data.source import (
    InventoryError,
    SourceImageRecord,
    collect_source_images,
    split_counts,
)
from construction_safety_vision.env import load_project_env
from construction_safety_vision.paths import ProjectPaths

EXPECTED_SOURCE_IMAGES = 436
"""Source population established in phase 3."""

EXPECTED_SPLITS = {"train": 306, "valid": 87, "test": 43}
"""Provider split allocation established in phase 3."""


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write rows as newline-delimited JSON with LF endings.

    Args:
        path: Destination file.
        rows: Rows to serialise.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows)
    path.write_text(f"{body}\n" if body else "", encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> int:
    """Recover the inventory.

    Args:
        argv: Command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="Configuration file to use.")
    parser.add_argument(
        "--skip-details",
        action="store_true",
        help="List images without fetching per-image annotation geometry.",
    )
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

    records: list[SourceImageRecord] = walk.records
    unique_ids = {r.image_id for r in records}
    print(
        f"records recovered: {len(records)}   unique provider ids: {len(unique_ids)}   "
        f"pages: {walk.pages_fetched}   repeated ids returned by provider: "
        f"{len(walk.repeated_ids)}"
    )

    if not args.skip_details:
        print("fetching authoritative annotations per image ...", flush=True)
        for index, record in enumerate(records, start=1):
            try:
                record.attach_details(client.image_details(coordinates, record.image_id))
            except RoboflowError as exc:
                print(f"ERROR on {record.image_id}: {exc}", file=sys.stderr)
                return 4
            if index % 50 == 0 or index == len(records):
                print(f"  {index}/{len(records)}", flush=True)

    observed_splits = split_counts(records)
    geometry: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    detail_mismatch = 0
    for record in records:
        geometry.update(record.geometry_types)
        for annotation in record.annotations:
            labels[annotation.label] += 1
        if record.annotations and len(record.annotations) != record.annotation_count:
            detail_mismatch += 1

    write_jsonl(
        paths.reports / "source_image_manifest.jsonl",
        [r.manifest_entry() for r in sorted(records, key=lambda r: r.image_id)],
    )
    write_jsonl(
        paths.data_interim / "source_annotations.jsonl",
        [e for r in sorted(records, key=lambda r: r.image_id) for e in r.annotation_entries()],
    )

    print()
    print(f"source records          : {len(records)} (expected {EXPECTED_SOURCE_IMAGES})")
    print(f"unique provider ids     : {len(unique_ids)}")
    print(f"provider splits         : {observed_splits} (expected {EXPECTED_SPLITS})")
    print(f"instances (search)      : {sum(r.annotation_count for r in records)}")
    print(f"instances (details)     : {sum(len(r.annotations) for r in records)}")
    print(f"geometry types          : {dict(geometry)}")
    print(f"labels                  : {dict(labels.most_common())}")
    print(f"images with zero labels : {sum(1 for r in records if r.annotation_count == 0)}")
    print(f"search/details mismatch : {detail_mismatch}")
    print(f"provider page repeats   : {len(walk.repeated_ids)}")
    print("wrote reports/source_image_manifest.jsonl and data/interim/source_annotations.jsonl")

    problems = []
    if len(records) != EXPECTED_SOURCE_IMAGES:
        problems.append(f"expected {EXPECTED_SOURCE_IMAGES} records, got {len(records)}")
    if len(unique_ids) != len(records):
        problems.append("duplicate provider image ids")
    if observed_splits != EXPECTED_SPLITS:
        problems.append(f"split allocation differs from phase 3: {observed_splits}")
    if problems:
        print(f"\nDISCREPANCIES: {problems}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
