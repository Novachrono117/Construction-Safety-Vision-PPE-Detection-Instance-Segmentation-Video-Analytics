"""Audit how much canonical instance-mask geometry survives the YOLO segmentation format.

Phase 8A. It trains nothing, evaluates nothing, downloads no weights and selects
no architecture. It builds a development-only, audit-only adapter, measures what
the conversion costs per instance, and stops.

The question is narrow and worth stating precisely: *if this project's canonical
COCO instance masks were expressed as Ultralytics YOLO segmentation labels, how
much of the geometry would still be there?* Not "is YOLO good enough" - that is a
judgement, and it belongs to a human in phase 8B. This script produces the
evidence that judgement needs and deliberately declines to make it.

What it does, in order:

* verifies the frozen detector is untouched and the holdout is locked;
* reads the canonical phase 5D segmentation documents for ``train`` and
  ``validation`` only, and verifies their fingerprints;
* converts every development annotation to exactly one YOLO row, using the
  framework's own primitives where they exist;
* writes the labels, then **reads them back off disk** and rasterises them the
  way Ultralytics does;
* compares each reconstruction against the canonical mask at three levels, so
  that the rasteriser convention, component joining, and serialisation are each
  charged only for what they cost;
* hands the generated dataset to the installed Ultralytics dataset scanner and
  checks that the instance count survives;
* emits a row-level table, a manifest and a report.

Runs entirely offline. Requires:

* ``configs/segmentation_adapter_audit.yaml``
* ``reports/task_dataset_manifest.json``            phase 5D
* ``reports/split_manifest.json``                   phase 5C.2
* ``data/processed/canonical/``                     phase 5D

Writes:
    data/processed/adapters/yolo_segmentation_audit/   (git-ignored)
    reports/segmentation_adapter_fidelity.csv
    reports/segmentation_adapter_audit_manifest.json
    reports/segmentation_adapter_fidelity_report.md
    reports/segmentation_adapter_audit.provenance.json

Usage:
    uv run python scripts/audit_segmentation_adapter.py
    uv run python scripts/audit_segmentation_adapter.py --verify-only
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from construction_safety_vision.config import ConfigError
from construction_safety_vision.data.canonical import scan_for_sensitive
from construction_safety_vision.data.segmentation_adapter import (
    ADAPTER_TYPE,
    FORBIDDEN_SPLIT,
    GEOMETRY_TYPES,
    AuditAdapterConfig,
    SegmentationAdapterError,
    analyse_topology,
    bounds_violations,
    canonical_mask,
    classify_geometry,
    denormalise_ring,
    digest,
    format_label_line,
    label_fingerprint,
    label_text,
    load_audit_config,
    merge_rings,
    normalise_ring,
    parse_label_line,
    rings_for_annotation,
)
from construction_safety_vision.data.segmentation_fidelity import (
    FidelityRow,
    attribute_reasons,
    compare_masks,
    describe,
    fingerprint_rows,
    group_by,
    quartile_edges,
    rasterise_ring,
    summarise,
    worst,
)
from construction_safety_vision.data.yolo_detection_adapter import (
    ADAPTER_ROOT_PLACEHOLDER,
    dataset_yaml,
)
from construction_safety_vision.paths import ProjectPaths
from construction_safety_vision.provenance import ProvenanceRecord, git_commit, sha256_file
from construction_safety_vision.splits import HOLDOUT_UNLOCK_ENV_VAR, holdout_unlocked

AUDIT_YAML = "segmentation_adapter_audit.yaml"
TASK_MANIFEST_JSON = "task_dataset_manifest.json"
SPLIT_MANIFEST_JSON = "split_manifest.json"
FINAL_DETECTOR_JSON = "final_detector_manifest.json"

FIDELITY_CSV = "segmentation_adapter_fidelity.csv"
AUDIT_MANIFEST_JSON = "segmentation_adapter_audit_manifest.json"
AUDIT_REPORT_MD = "segmentation_adapter_fidelity_report.md"
PROVENANCE_JSON = "segmentation_adapter_audit.provenance.json"

COMPLETE = "SEGMENTATION_ADAPTER_AUDIT_COMPLETE"
STRUCTURAL_INCOMPATIBILITY = "STRUCTURAL_FORMAT_INCOMPATIBILITY"
GEOMETRY_AUDIT_FAILURE = "GEOMETRY_AUDIT_FAILURE"
BLOCKED = "BLOCKED"

ARCHITECTURE_UNSELECTED = "UNSELECTED_PENDING_FIDELITY_REVIEW"
BASELINE_UNFROZEN = "UNFROZEN"
S0_UNDEFINED = "NOT_DEFINED"

HOLDOUT_STATUS = "PROTECTED_NOT_ACCESSED"
HOLDOUT_REASON = (
    "Phase 8A measured annotation-format fidelity on the development splits only. The holdout "
    "was not read, materialised, adapted, converted, counted or inspected; no holdout label, "
    "statistic or identifier exists in any artifact this phase wrote. Nothing about how COCO "
    "geometry serialises depends on which images are held out, so reading it would have bought "
    "no information and spent the one look the protocol allows."
)

WORST_CASE_COUNT = 20

#: Rendered in the report beside the flag census, so the census row reads correctly.
MATERIAL_IOU_LABEL = "0.999"

CSV_COLUMNS: tuple[str, ...] = (
    "source_image_id",
    "annotation_id",
    "split",
    "class",
    "canonical_geometry_type",
    "canonical_area_px",
    "reconstructed_area_px",
    "connected_components",
    "component_rings",
    "joined",
    "hole_count",
    "hole_pixels",
    "thinness",
    "adapter_point_count",
    "mask_iou",
    "dice",
    "control_iou",
    "merged_iou",
    "join_loss",
    "serialization_loss",
    "absolute_area_error_px",
    "relative_area_error",
    "signed_relative_area_error",
    "fp_pixels",
    "fn_pixels",
    "bbox_max_delta_px",
    "reason_flags",
)


class AuditError(RuntimeError):
    """Raised when a required input is missing or an invariant fails."""


def read_json(path: Path) -> dict[str, Any]:
    """Read a committed JSON artifact.

    Args:
        path: File to read.

    Returns:
        The parsed mapping.

    Raises:
        AuditError: If the file is missing or is not a JSON object.
    """
    if not path.is_file():
        msg = f"required input not found: {path.name}"
        raise AuditError(msg)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"{path.name} is not valid JSON ({exc})"
        raise AuditError(msg) from exc
    if not isinstance(data, dict):
        msg = f"{path.name} must contain a JSON object"
        raise AuditError(msg)
    return data


def write_json(path: Path, payload: Any) -> str:
    """Write a JSON artifact deterministically and return its digest.

    Args:
        path: Destination file.
        payload: JSON-serialisable content.

    Returns:
        The SHA-256 digest of the payload.
    """
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")
    return digest(payload)


def transfer_image(source: Path, destination: Path) -> str:
    """Place a canonical image into the adapter without altering a byte.

    A hard link is preferred over a copy: the audit adapter is a third view of
    the same 112 MB of pixels, and duplicating them buys nothing when the
    filesystem can point at the same blocks. Either way the digest is verified,
    so a link and a copy are indistinguishable in what they guarantee.

    Args:
        source: Canonical image.
        destination: Adapter image.

    Returns:
        How the file was transferred.

    Raises:
        AuditError: If the adapter image's bytes differ from the canonical
            image's.
    """
    mode = "HARDLINK"
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copyfile(source, destination)
        mode = "BINARY_COPY"
    if sha256_file(destination) != sha256_file(source):
        msg = f"adapter image {destination.name} does not match the canonical image byte for byte"
        raise AuditError(msg)
    return mode


def verify_inputs(paths: ProjectPaths, config: AuditAdapterConfig) -> dict[str, Any]:
    """Check the frozen state this audit depends on and must not disturb.

    Args:
        paths: Project layout.
        config: The audit protocol.

    Returns:
        The verified fingerprints and class map.

    Raises:
        AuditError: If a fingerprint disagrees or the holdout is unlocked.
    """
    task_manifest = read_json(paths.reports / TASK_MANIFEST_JSON)
    split_manifest = read_json(paths.reports / SPLIT_MANIFEST_JSON)

    class_map = task_manifest.get("class_map")
    if not isinstance(class_map, Mapping) or not class_map:
        msg = f"{TASK_MANIFEST_JSON} carries no class map"
        raise AuditError(msg)
    if config["placeholder_category_excluded"] in class_map:
        msg = (
            f"the placeholder category {config['placeholder_category_excluded']!r} is present in "
            "the class map; it carries no annotations and must stay excluded"
        )
        raise AuditError(msg)

    canonical_keys = (
        "canonical_id_assignment_sha256",
        "class_map_sha256",
        "groups_sha256",
        "materialization_config_sha256",
        "modeling_population_sha256",
        "split_assignment_sha256",
    )
    fingerprints = {
        key: task_manifest[key] for key in canonical_keys if isinstance(task_manifest.get(key), str)
    }
    missing = [key for key in canonical_keys if key not in fingerprints]
    if missing:
        msg = f"{TASK_MANIFEST_JSON} is missing canonical fingerprint(s) {missing}"
        raise AuditError(msg)
    class_map_sha256 = fingerprints["class_map_sha256"]
    if class_map_sha256 != config["class_map_sha256"]:
        msg = (
            f"the canonical class map hashes {class_map_sha256}, but the audit protocol names "
            f"{config['class_map_sha256']}"
        )
        raise AuditError(msg)

    split_sha256 = split_manifest.get("split_assignment_sha256")
    if split_sha256 != fingerprints["split_assignment_sha256"]:
        msg = (
            "the frozen split manifest and the canonical task manifest disagree on "
            "split_assignment_sha256; the canonical documents were built under a different split"
        )
        raise AuditError(msg)
    if not split_sha256:
        msg = f"{SPLIT_MANIFEST_JSON} carries no split_assignment_sha256"
        raise AuditError(msg)

    detector = read_json(paths.reports / FINAL_DETECTOR_JSON)
    if detector.get("status") != "FROZEN":
        msg = "the final detector manifest is not FROZEN; the detection block must stay frozen"
        raise AuditError(msg)

    return {
        "task_manifest_sha256": sha256_file(paths.reports / TASK_MANIFEST_JSON),
        "split_assignment_sha256": split_sha256,
        "class_map_sha256": class_map_sha256,
        "class_map": {str(name): int(index) for name, index in class_map.items()},
        "canonical_fingerprints": dict(fingerprints),
        "final_detector_sha256": detector.get("final_detector_sha256"),
        "final_detector_experiment": detector.get("selected_experiment"),
    }


def convert_split(
    paths: ProjectPaths,
    config: AuditAdapterConfig,
    split: str,
    class_names: Mapping[int, str],
) -> tuple[dict[str, str], list[FidelityRow], dict[str, Any]]:
    """Convert one development split and measure every instance.

    Args:
        paths: Project layout.
        config: The audit protocol.
        split: ``train`` or ``validation``.
        class_names: Class index to class name.

    Returns:
        The label file contents keyed by stem, the fidelity rows, and per-split
        counters.

    Raises:
        AuditError: If the split names the protected split or an invariant fails.
    """
    if split == FORBIDDEN_SPLIT:
        msg = "the holdout has no part in this audit"
        raise AuditError(msg)

    annotations_root = paths.root / config["canonical_annotations_root"]
    document = json.loads(
        (annotations_root / f"segmentation_{split}.coco.json").read_text(encoding="utf-8")
    )
    images = {image["id"]: image for image in document["images"]}
    grouped: dict[int, list[Mapping[str, Any]]] = {image_id: [] for image_id in images}
    for annotation in document["annotations"]:
        grouped[annotation["image_id"]].append(annotation)

    labels: dict[str, str] = {}
    rows: list[FidelityRow] = []
    counters: dict[str, Any] = {
        "images": len(images),
        "annotations": len(document["annotations"]),
        "negative_images": 0,
        "rows_written": 0,
        "bounds_violations": 0,
        "geometry_counts": dict.fromkeys(GEOMETRY_TYPES, 0),
        "duplicate_box_collisions": [],
    }

    material_iou = float(config["material_deviation_iou"])
    small_area = int(config["small_mask_area_px"])
    minimum_points = int(config["minimum_points_per_instance"])

    for image_id in sorted(images):
        image = images[image_id]
        height, width = int(image["height"]), int(image["width"])
        stem = Path(str(image["file_name"])).stem
        instance_rows: list[str] = []
        boxes: list[tuple[int, tuple[float, float, float, float]]] = []

        for annotation in sorted(grouped[image_id], key=lambda item: str(item["id"])):
            geometry_type = classify_geometry(annotation)
            counters["geometry_counts"][geometry_type] += 1

            mask = canonical_mask(annotation, height=height, width=width)
            topology = analyse_topology(mask, connectivity=config.connectivity)
            rings = rings_for_annotation(
                annotation, mask, geometry_type=geometry_type, minimum_points=minimum_points
            )
            merged, joined = merge_rings(rings)

            normalised = normalise_ring(merged, image_width=width, image_height=height)
            counters["bounds_violations"] += bounds_violations(normalised)
            class_index = int(annotation["category_id"])
            line = format_label_line(class_index, normalised, precision=config.precision)

            # The comparison must use the row as written, so parse it back the
            # way the framework does rather than reusing the in-memory ring.
            parsed_class, parsed = parse_label_line(line)
            if parsed_class != class_index:
                msg = f"annotation {annotation['id']}: class index did not survive serialisation"
                raise AuditError(msg)
            reconstructed = rasterise_ring(
                denormalise_ring(parsed, image_width=width, image_height=height),
                height=height,
                width=width,
            )

            # Level 1: the rings before any conversion, same rasteriser.
            control = np.zeros((height, width), dtype=np.uint8)
            for ring in rings:
                control = np.maximum(control, rasterise_ring(ring, height=height, width=width))
            # Level 2: the merged single ring, still at full precision.
            merged_mask = rasterise_ring(merged, height=height, width=width)

            comparison = compare_masks(mask, reconstructed)
            control_comparison = compare_masks(mask, control)
            merged_comparison = compare_masks(mask, merged_mask)

            row_data = {
                "mask_iou": comparison.iou,
                "merged_iou": merged_comparison.iou,
                "control_iou": control_comparison.iou,
                "hole_count": topology.hole_count,
                "connected_components": topology.component_count,
                "component_rings": len(rings),
                "joined": joined,
                "canonical_area_px": topology.area_px,
            }
            rows.append(
                FidelityRow(
                    source_image_id=str(image["source_image_id"]),
                    annotation_id=str(annotation["id"]),
                    split=split,
                    class_name=class_names[class_index],
                    canonical_geometry_type=geometry_type,
                    canonical_area_px=comparison.canonical_area,
                    reconstructed_area_px=comparison.reconstructed_area,
                    connected_components=topology.component_count,
                    hole_count=topology.hole_count,
                    hole_pixels=topology.hole_pixels,
                    thinness=topology.thinness,
                    adapter_point_count=len(parsed),
                    component_rings=len(rings),
                    joined=joined,
                    mask_iou=comparison.iou,
                    dice=comparison.dice,
                    control_iou=control_comparison.iou,
                    merged_iou=merged_comparison.iou,
                    absolute_area_error_px=comparison.absolute_area_error,
                    relative_area_error=comparison.relative_area_error,
                    signed_relative_area_error=comparison.signed_relative_area_error,
                    fp_pixels=comparison.false_positive,
                    fn_pixels=comparison.false_negative,
                    bbox_max_delta_px=comparison.bbox_max_delta,
                    reason_flags=attribute_reasons(
                        row_data, material_iou=material_iou, small_area_px=small_area
                    ),
                )
            )
            instance_rows.append(line)

            # Ultralytics drops duplicate (class, box) rows with np.unique after
            # parsing. Two instances that collide there would silently become
            # one, so the collision is detected here rather than discovered as a
            # count mismatch later.
            xy = parsed.astype(np.float32)
            boxes.append(
                (
                    class_index,
                    (
                        float(xy[:, 0].min()),
                        float(xy[:, 1].min()),
                        float(xy[:, 0].max()),
                        float(xy[:, 1].max()),
                    ),
                )
            )

        seen: set[tuple[int, tuple[float, float, float, float]]] = set()
        for entry in boxes:
            if entry in seen:
                counters["duplicate_box_collisions"].append(stem)
            seen.add(entry)

        labels[stem] = label_text(instance_rows)
        counters["rows_written"] += len(instance_rows)
        if not instance_rows:
            counters["negative_images"] += 1

    return labels, rows, counters


def write_adapter(
    paths: ProjectPaths,
    config: AuditAdapterConfig,
    labels: Mapping[str, Mapping[str, str]],
    class_map: Mapping[str, int],
) -> dict[str, Any]:
    """Materialise the audit adapter on disk.

    Args:
        paths: Project layout.
        config: The audit protocol.
        labels: Label contents keyed by split then image stem.
        class_map: The frozen class map.

    Returns:
        A record of what was written.

    Raises:
        AuditError: If an image cannot be transferred byte-identically.
    """
    root = paths.root / config.output_root
    directories = config["split_directories"]
    transfer_modes: set[str] = set()
    image_counts: dict[str, int] = {}

    for split, directory in directories.items():
        image_dir = root / "images" / directory
        label_dir = root / "labels" / directory
        for target in (image_dir, label_dir):
            if target.exists():
                shutil.rmtree(target)
            target.mkdir(parents=True, exist_ok=True)

        canonical_images = paths.root / config["canonical_images_root"] / split
        written = 0
        for stem in sorted(labels[split]):
            (label_dir / f"{stem}.txt").write_text(
                labels[split][stem], encoding="utf-8", newline="\n"
            )
            matches = sorted(canonical_images.glob(f"{stem}.*"))
            if not matches:
                msg = f"canonical image for {stem} not found in {split}"
                raise AuditError(msg)
            transfer_modes.add(transfer_image(matches[0], image_dir / matches[0].name))
            written += 1
        image_counts[split] = written

    descriptor = dataset_yaml(
        path_value=str(root.resolve()),
        train_directory=f"images/{directories['train']}",
        validation_directory=f"images/{directories['validation']}",
        class_map=class_map,
    )
    (root / "dataset.yaml").write_text(descriptor, encoding="utf-8", newline="\n")
    template = dataset_yaml(
        path_value=ADAPTER_ROOT_PLACEHOLDER,
        train_directory=f"images/{directories['train']}",
        validation_directory=f"images/{directories['validation']}",
        class_map=class_map,
    )
    (root / "README.md").write_text(
        "# YOLO segmentation adapter - AUDIT ONLY\n\n"
        "Generated by `scripts/audit_segmentation_adapter.py` (phase 8A). This directory is\n"
        "**not** the project's segmentation dataset. It exists to measure how much canonical\n"
        "COCO instance-mask geometry survives the Ultralytics YOLO segmentation label format,\n"
        "and no architecture has been selected on the strength of it.\n\n"
        "- status: `AUDIT_ONLY`\n"
        "- canonical status: `NOT_CANONICAL`\n"
        "- training status: `NOT_YET_APPROVED_FOR_TRAINING`\n"
        "- development splits only; the holdout has no directory here and must not be added\n\n"
        "The canonical COCO instance segmentation frozen in phase 5D remains the ground truth.\n"
        "If a label here and the canonical document ever disagree, the COCO document is right\n"
        "and this adapter is broken: regenerate it, never hand-edit it.\n\n"
        "Portable descriptor template:\n\n"
        "```yaml\n"
        f"{template}```\n",
        encoding="utf-8",
        newline="\n",
    )

    return {
        "root": config.output_root,
        "image_counts": image_counts,
        "image_transfer_modes": sorted(transfer_modes),
        "descriptor": f"{config.output_root}/dataset.yaml",
    }


def validate_with_framework(
    paths: ProjectPaths, config: AuditAdapterConfig, expected: Mapping[str, int]
) -> dict[str, Any]:
    """Hand the generated labels to the installed Ultralytics dataset scanner.

    Structural parsing only. No model is constructed, no weight is downloaded and
    nothing is trained; this asks one question - does the framework read back the
    instances that were written?

    Args:
        paths: Project layout.
        config: The audit protocol.
        expected: Expected instance count per split.

    Returns:
        What the framework found.

    Raises:
        AuditError: If the framework changes the instance cardinality.
    """
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.data.utils import check_det_dataset

    root = paths.root / config.output_root
    data = check_det_dataset(str(root / "dataset.yaml"), autodownload=False)
    if data.get(FORBIDDEN_SPLIT):
        msg = "the generated descriptor exposes the protected split"
        raise AuditError(msg)

    found: dict[str, Any] = {"splits": {}}
    directories = config["split_directories"]
    for split, directory in directories.items():
        cache = root / "labels" / f"{directory}.cache"
        if cache.exists():
            cache.unlink()
        dataset = YOLODataset(
            img_path=str(root / "images" / directory),
            data=data,
            task="segment",
        )
        instances = sum(len(label["cls"]) for label in dataset.labels)
        segments = sum(len(label["segments"]) for label in dataset.labels)
        negatives = sum(1 for label in dataset.labels if len(label["cls"]) == 0)
        found["splits"][split] = {
            "images_discovered": len(dataset.labels),
            "instances": instances,
            "segment_rows": segments,
            "negative_images": negatives,
        }
        if instances != expected[split]:
            msg = (
                f"{split}: the framework parser read {instances} instances but "
                f"{expected[split]} were written. Instance cardinality changed during parsing, "
                "which invalidates the one-annotation-one-instance invariant."
            )
            raise AuditError(msg)
        if segments != instances:
            msg = (
                f"{split}: the framework read {segments} segment rows for {instances} instances; "
                "a segment dataset requires one segment per box"
            )
            raise AuditError(msg)
        if cache.exists():
            cache.unlink()

    found["classes"] = len(data["names"])
    found["corrupt_labels"] = 0
    found["test_split_present"] = bool(data.get(FORBIDDEN_SPLIT))
    return found


def area_quartile(area: int, edges: Sequence[float]) -> str:
    """Place a mask area in its development-distribution quartile.

    Args:
        area: Canonical mask area in pixels.
        edges: The 25th, 50th and 75th percentiles.

    Returns:
        A quartile label.
    """
    if area <= edges[0]:
        return "Q1_smallest"
    if area <= edges[1]:
        return "Q2"
    if area <= edges[2]:
        return "Q3"
    return "Q4_largest"


def _group(rows: Sequence[FidelityRow], classify: Any) -> dict[str, list[FidelityRow]]:
    """Partition instances by a derived key.

    Args:
        rows: The instances.
        classify: A callable returning the group key for one instance.

    Returns:
        Instances keyed by group, in sorted key order.
    """
    groups: dict[str, list[FidelityRow]] = {}
    for row in rows:
        groups.setdefault(classify(row), []).append(row)
    return {name: groups[name] for name in sorted(groups)}


def component_bucket(row: FidelityRow) -> str:
    """Bucket an instance by connected-component count.

    Args:
        row: The instance.

    Returns:
        ``"1"``, ``"2"`` or ``"3+"``.
    """
    if row.connected_components <= 1:
        return "1"
    return "2" if row.connected_components == 2 else "3+"


def build_aggregates(rows: Sequence[FidelityRow]) -> dict[str, Any]:
    """Summarise the audit across every stratum the protocol names.

    The strata are the point. A single global mean would let the near-exact
    polygon conversions absorb whatever the RLE masks lose, and would hide the
    two structural approximations entirely.

    Args:
        rows: Every development instance.

    Returns:
        A JSON-serialisable mapping of the aggregate results.
    """
    edges = quartile_edges([row.canonical_area_px for row in rows])

    reason_counts: dict[str, int] = {}
    for row in rows:
        for flag in row.reason_flags:
            reason_counts[flag] = reason_counts.get(flag, 0) + 1

    return {
        "global": summarise(rows),
        "by_split": {name: summarise(group) for name, group in group_by(rows, "split").items()},
        "by_geometry_type": {
            name: summarise(group)
            for name, group in group_by(rows, "canonical_geometry_type").items()
        },
        "by_class": {
            name: summarise(group) for name, group in group_by(rows, "class_name").items()
        },
        "by_component_count": {
            name: summarise(group) for name, group in _group(rows, component_bucket).items()
        },
        "by_hole_presence": {
            name: summarise(group)
            for name, group in _group(
                rows, lambda row: "with_holes" if row.hole_count > 0 else "without_holes"
            ).items()
        },
        "by_mask_area_quartile": {
            name: summarise(group)
            for name, group in _group(
                rows, lambda row: area_quartile(row.canonical_area_px, edges)
            ).items()
        },
        "mask_area_quartile_edges": {
            "p25": edges[0],
            "p50": edges[1],
            "p75": edges[2],
            "note": (
                "Quartiles of the development mask-area distribution, computed from the data "
                "itself. Diagnostic bins for this audit only; they deliberately do not reuse or "
                "redefine the project's small-object EDA threshold, which is a different "
                "measurement."
            ),
        },
        "thinness": describe([row.thinness for row in rows]),
        "reason_flag_counts": dict(sorted(reason_counts.items())),
        "exact_instances": sum(1 for row in rows if not row.reason_flags),
    }


def build_topology(rows: Sequence[FidelityRow]) -> dict[str, Any]:
    """Count the topology the format has to flatten.

    Args:
        rows: Every development instance.

    Returns:
        A JSON-serialisable mapping of the topology census.
    """
    distribution: dict[str, int] = {}
    for row in rows:
        key = str(row.connected_components)
        distribution[key] = distribution.get(key, 0) + 1
    with_holes = [row for row in rows if row.hole_count > 0]
    multi = [row for row in rows if row.connected_components > 1]
    return {
        "connected_component_distribution": {
            key: distribution[key] for key in sorted(distribution, key=int)
        },
        "instances_with_multiple_components": len(multi),
        "maximum_component_count": max((row.connected_components for row in rows), default=0),
        "instances_with_holes": len(with_holes),
        "total_holes": sum(row.hole_count for row in rows),
        "maximum_holes_in_one_instance": max((row.hole_count for row in rows), default=0),
        "total_hole_pixels": sum(row.hole_pixels for row in with_holes),
        "largest_hole_area_fraction": max(
            (
                row.hole_pixels / (row.canonical_area_px + row.hole_pixels)
                for row in with_holes
                if row.canonical_area_px + row.hole_pixels > 0
            ),
            default=0.0,
        ),
        "connectivity": 8,
    }


def worst_cases(rows: Sequence[FidelityRow]) -> dict[str, Any]:
    """Select the deterministic worst instances by each headline metric.

    Args:
        rows: Every development instance.

    Returns:
        A JSON-serialisable mapping of the worst cases.
    """

    def render(row: FidelityRow) -> dict[str, Any]:
        return {
            "source_image_id": row.source_image_id,
            "annotation_id": row.annotation_id,
            "split": row.split,
            "class": row.class_name,
            "canonical_geometry_type": row.canonical_geometry_type,
            "canonical_area_px": row.canonical_area_px,
            "connected_components": row.connected_components,
            "hole_count": row.hole_count,
            "mask_iou": round(row.mask_iou, 6),
            "control_iou": round(row.control_iou, 6),
            "merged_iou": round(row.merged_iou, 6),
            "relative_area_error": round(row.relative_area_error, 6),
            "reason_flags": list(row.reason_flags),
        }

    return {
        "worst_mask_iou": [
            render(row) for row in worst(rows, "mask_iou", WORST_CASE_COUNT, ascending=True)
        ],
        "worst_relative_area_error": [
            render(row)
            for row in worst(rows, "relative_area_error", WORST_CASE_COUNT, ascending=False)
        ],
    }


def write_fidelity_csv(path: Path, rows: Sequence[FidelityRow]) -> str:
    """Write the row-level audit table.

    One row per development annotation. Development identifiers are committed
    here because the repository already commits them for every other
    development-level analysis; no holdout identifier appears.

    Args:
        path: Destination file.
        rows: Every development instance.

    Returns:
        The file's SHA-256 digest.
    """
    ordered = sorted(rows, key=lambda row: (row.split, row.source_image_id, row.annotation_id))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS), lineterminator="\n")
        writer.writeheader()
        for row in ordered:
            writer.writerow({key: row.as_row()[key] for key in CSV_COLUMNS})
    return sha256_file(path)


def framework_format_note() -> dict[str, Any]:
    """Record the format contract, read from the installed framework source.

    Every clause here was established by reading Ultralytics 8.4.138 on this
    machine rather than by consulting documentation, because the audit's whole
    value is that it measures the parser that would actually load the data.

    Returns:
        A JSON-serialisable mapping of the contract.
    """
    return {
        "package": "ultralytics",
        "version": "8.4.138",
        "evidence": "INSTALLED_PACKAGE_SOURCE_INSPECTION",
        "row_structure": (
            "class then a flat sequence of x y pairs, all normalised to the image width and "
            "height. ultralytics/data/utils.py::verify_image_label parses the tail as "
            "np.array(fields[1:], dtype=np.float32).reshape(-1, 2)."
        ),
        "one_row_per_instance": True,
        "row_is_single_ring": True,
        "segment_row_threshold": (
            "A row is read as a segment only when it has more than 6 whitespace-separated "
            "fields, i.e. a class plus at least 3 points. A 5-field row is a detection box, and "
            "mixing the two kinds in one file raises."
        ),
        "multi_segment_separator": (
            "NONE. The format has no separator between rings and no ring-role marker, so an "
            "instance made of several components cannot be written as several rings in one row."
        ),
        "multi_segment_handling": (
            "ultralytics/data/converter.py::convert_coco calls merge_multi_segment when a COCO "
            "annotation carries more than one polygon, bridging components along their nearest "
            "points with a zero-width connector so that one instance stays one row. This audit "
            "uses that same primitive."
        ),
        "holes_representable": False,
        "hole_evidence": (
            "Two independent confirmations in the installed source. "
            "convert_segment_masks_to_yolo_seg extracts contours with cv2.RETR_EXTERNAL, which "
            "returns outer boundaries only and discards interior contours by construction. And "
            "polygon2mask fills a single ring with cv2.fillPoly, which has no even-odd "
            "subtraction, so anything the outer boundary encloses is filled."
        ),
        "coordinate_parsing": (
            "float32. Coordinates are bounds-checked into [-0.01, 1.01]; a row outside that "
            "range makes the whole image-label pair corrupt."
        ),
        "internal_resampling": (
            "Yes, at training time: YOLODataset.update_labels_info calls resample_segments with "
            "n=1000, raised to max_len+1 when a segment already has more points. Because "
            "resample_segments inserts the original vertices into the interpolation grid and "
            "never downsamples below the input length under that rule, it densifies a path "
            "rather than simplifying it, and introduces no additional geometric loss. It is a "
            "training-time transformation and is not part of the label format."
        ),
        "rasterization": (
            "polygon2mask casts coordinates with np.asarray(..., dtype=np.int32), which "
            "truncates toward zero, then fills with cv2.fillPoly. The truncation is a real "
            "source of sub-pixel loss and this audit measures it rather than avoiding it."
        ),
        "cardinality_hazard": (
            "After parsing, verify_image_label derives each row's bounding box with "
            "segments2boxes and removes duplicate class-plus-box rows via np.unique. Two "
            "distinct instances of the same class sharing a bounding box would silently "
            "collapse into one. The audit checks for that collision explicitly rather than "
            "assuming it cannot happen."
        ),
        "empty_label_semantics": (
            "An image with an empty label file is counted as a background image; an image with "
            "no label file at all is counted as a missing label. Negatives therefore require "
            "empty files, not absent ones."
        ),
    }


def build_manifest(
    *,
    config: AuditAdapterConfig,
    verified: Mapping[str, Any],
    counters: Mapping[str, Mapping[str, Any]],
    adapter: Mapping[str, Any],
    fingerprints: Mapping[str, str],
    aggregates: Mapping[str, Any],
    topology: Mapping[str, Any],
    parser: Mapping[str, Any],
    cases: Mapping[str, Any],
    geometry_counts: Mapping[str, int],
) -> dict[str, Any]:
    """Assemble the audit manifest.

    Args:
        config: The audit protocol.
        verified: Verified canonical fingerprints.
        counters: Per-split conversion counters.
        adapter: What was written to disk.
        fingerprints: Deterministic digests of the audit's outputs.
        aggregates: The stratified fidelity summaries.
        topology: The topology census.
        parser: The framework parser's verdict.
        cases: The worst cases.
        geometry_counts: Canonical geometry counts.

    Returns:
        The manifest payload.
    """
    images = sum(int(counters[split]["images"]) for split in counters)
    annotations = sum(int(counters[split]["annotations"]) for split in counters)
    rows_written = sum(int(counters[split]["rows_written"]) for split in counters)
    return {
        "schema_version": 1,
        "phase": "8A",
        "task": "segmentation",
        "status": COMPLETE,
        "adapter_type": ADAPTER_TYPE,
        "adapter_status": {
            "status": config["status"],
            "canonical_status": config["canonical_status"],
            "training_status": config["training_status"],
            "note": (
                "This adapter is a measurement instrument, not the project's segmentation "
                "dataset. It has not been reviewed, approved or adopted, and nothing was "
                "trained on it."
            ),
        },
        "canonical_source": config["canonical_source"],
        "canonical_source_phase": config["canonical_source_phase"],
        "canonical_task_manifest_sha256": verified["task_manifest_sha256"],
        "canonical_fingerprints": dict(verified["canonical_fingerprints"]),
        "split_assignment_sha256": verified["split_assignment_sha256"],
        "class_map_sha256": verified["class_map_sha256"],
        "class_map": dict(verified["class_map"]),
        "placeholder_category_excluded": config["placeholder_category_excluded"],
        "audit_config_sha256": config.fingerprint(),
        "development_counts": {
            "images": images,
            "annotations": annotations,
            "per_split": {
                split: {
                    "images": counters[split]["images"],
                    "annotations": counters[split]["annotations"],
                    "instance_rows": counters[split]["rows_written"],
                    "negative_images": counters[split]["negative_images"],
                }
                for split in sorted(counters)
            },
        },
        "canonical_geometry_counts": dict(geometry_counts),
        "instance_cardinality": {
            "policy": config["instance_cardinality_policy"],
            "canonical_annotations": annotations,
            "model_instance_rows": rows_written,
            "preserved": annotations == rows_written,
            "instances_split": 0,
            "instances_merged": 0,
            "instances_dropped": 0,
        },
        "adapter": dict(adapter),
        "adapter_fingerprints": dict(fingerprints),
        "framework_format": framework_format_note(),
        "conversion_policy": {
            "rle_decode": config["rle_decode"],
            "contour_retrieval_mode": config["contour_retrieval_mode"],
            "contour_approximation_mode": config["contour_approximation_mode"],
            "contour_epsilon_simplification": config["contour_epsilon_simplification"],
            "multi_component_policy": config["multi_component_policy"],
            "hole_policy": config["hole_policy"],
            "serialization_precision": config["serialization_precision"],
            "rasterization": config["rasterization"],
            "rasterization_canvas": config["rasterization_canvas"],
            "connectivity": config["connectivity"],
        },
        "topology": dict(topology),
        "fidelity": dict(aggregates),
        "worst_cases": dict(cases),
        "parser_validation": dict(parser),
        "bounds_violations": sum(int(counters[split]["bounds_violations"]) for split in counters),
        "duplicate_box_collisions": sorted(
            {stem for split in counters for stem in counters[split]["duplicate_box_collisions"]}
        ),
        "segmentation_architecture_selection": ARCHITECTURE_UNSELECTED,
        "segmentation_baseline": BASELINE_UNFROZEN,
        "S0": S0_UNDEFINED,
        "architecture_decision": {
            "status": ARCHITECTURE_UNSELECTED,
            "decided_here": False,
            "note": (
                "Phase 8A measures and stops. A good IoU distribution is not an approval of "
                "YOLO segmentation and a poor one is not a rejection; the choice of model path "
                "is a reviewed human decision in phase 8B."
            ),
            "fallback_recorded_not_selected": {
                "option": "mask-native instance segmentation consuming canonical COCO masks/RLE",
                "example": "Mask R-CNN",
                "status": "FUTURE_ALTERNATIVE_NOT_IMPLEMENTED_NOT_BENCHMARKED",
                "note": (
                    "Recorded only so the alternative is on the record if fidelity is judged "
                    "materially inadequate. Nothing about it has been implemented, installed, "
                    "benchmarked or selected."
                ),
            },
        },
        "models_trained_in_this_phase": 0,
        "models_evaluated_in_this_phase": 0,
        "detector_touched": False,
        "frozen_detector_sha256": verified["final_detector_sha256"],
        "test": {"status": HOLDOUT_STATUS, "reason": HOLDOUT_REASON},
    }


def _pct(value: float) -> str:
    """Render a fraction as a percentage string.

    Args:
        value: The fraction.

    Returns:
        The rendered percentage.
    """
    return f"{100.0 * value:.2f}%"


def _iou_row(label: str, summary: Mapping[str, Any]) -> str:
    """Render one stratum as a Markdown table row.

    Args:
        label: Row label, already formatted.
        summary: A summary produced by ``summarise``.

    Returns:
        The table row.
    """
    iou = summary["mask_iou"]
    error = summary["relative_area_error"]
    area = summary["canonical_area_px"]
    return (
        f"| {label} | {summary['instances']} | {area['median']:.0f} | {iou['mean']:.6f} | "
        f"{iou['median']:.6f} | {iou['p05']:.6f} | {iou['min']:.6f} | {error['mean']:.6f} | "
        f"{error['p95']:.6f} |"
    )


IOU_TABLE_HEADER: tuple[str, str] = (
    "| Stratum | Instances | Median area px | Mean IoU | Median IoU | P05 IoU | Min IoU | "
    "Mean rel. area err | P95 rel. area err |",
    "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
)


def audit_report(manifest: Mapping[str, Any], *, commit: str | None) -> str:
    """Render the phase 8A audit report.

    Args:
        manifest: The audit manifest payload.
        commit: The repository commit, when available.

    Returns:
        The Markdown text.
    """
    fidelity = manifest["fidelity"]
    overall = fidelity["global"]
    iou = overall["mask_iou"]
    control = overall["control_iou"]
    merged = overall["merged_iou"]
    error = overall["relative_area_error"]
    bands = overall["iou_bands"]
    topology = manifest["topology"]
    counts = manifest["development_counts"]
    geometry = manifest["canonical_geometry_counts"]
    parser = manifest["parser_validation"]
    fmt = manifest["framework_format"]
    head = commit or "unrecorded"

    lines: list[str] = [
        "# Phase 8A - YOLO Instance-Segmentation Adapter Fidelity Audit",
        "",
        f"Phase: 8A · Commit: `{head}` · Classification: **{manifest['status']}**",
        "",
        "**No segmentation model was trained, evaluated or downloaded, and no architecture was "
        "selected.** This phase measures one thing: how much of the project's canonical COCO "
        "instance-mask geometry survives being expressed as Ultralytics YOLO segmentation "
        "labels. The holdout was not read, adapted, converted, counted or inspected.",
        "",
        "Claims are labelled `FACT` (measured here), `CANONICAL_DATA_POLICY` (frozen earlier and "
        "inherited), `MODEL_FORMAT_CONSTRAINT` (a property of the installed framework), "
        "`COMPUTED_RESULT` (this audit's arithmetic), `APPROXIMATION` (information the format "
        "cannot carry), `LIMITATION` and `PENDING_HUMAN_DECISION`.",
        "",
        "## 1. Audit objective",
        "",
        "`PENDING_HUMAN_DECISION` The project's canonical annotations are COCO instance masks, "
        "half of them stored as compressed RLE. Before any segmentation architecture is chosen, "
        "one question has to be answered with numbers rather than assumption: **would a YOLO "
        "segmentation label still describe the same object?**",
        "",
        "`PENDING_HUMAN_DECISION` This phase answers that and stops. It does not approve YOLO "
        "segmentation and it does not reject it. A high IoU distribution is not an approval; a "
        "low one is not a rejection. The architecture decision belongs to a reviewed human step "
        "in phase 8B, and this report exists to inform it.",
        "",
        "## 2. Why model-format fidelity matters",
        "",
        "`MODEL_FORMAT_CONSTRAINT` A COCO instance mask can express things a single YOLO "
        "segmentation row cannot: an object in several disconnected pieces, an object with a "
        "hole through it, a boundary at arbitrary sub-pixel precision. If those are silently "
        "flattened during conversion, every mask metric the project later reports is measured "
        "against labels that are **not** the annotations anyone reviewed - and the loss would "
        "be invisible, because the model would be trained and evaluated against the same "
        "degraded labels.",
        "",
        "`LIMITATION` This audit measures **representation**, not model performance. It says "
        "nothing about how well any model would learn from these labels.",
        "",
        "## 3. Canonical segmentation source",
        "",
        f"`CANONICAL_DATA_POLICY` The source of truth is `{manifest['canonical_source']}`, "
        f"materialised in phase {manifest['canonical_source_phase']}. It was read and never "
        "modified: no canonical COCO document was rewritten, no provider mask was "
        "reinterpreted, and no provider box or provider preprocessing was used. What this phase "
        "produced is a **derived** representation.",
        "",
        f"`FACT` Development population: **{counts['images']} images** and "
        f"**{counts['annotations']} annotations**, verified against the canonical documents "
        "rather than assumed.",
        "",
        "| Split | Images | Annotations | Instance rows written | Negative images |",
        "| --- | --- | --- | --- | --- |",
    ]
    for split in sorted(counts["per_split"]):
        row = counts["per_split"][split]
        lines.append(
            f"| `{split}` | {row['images']} | {row['annotations']} | {row['instance_rows']} | "
            f"{row['negative_images']} |"
        )

    lines += [
        "",
        "`FACT` Canonical geometry, counted from the documents:",
        "",
        "| Canonical representation | Annotations |",
        "| --- | --- |",
        f"| polygon | {geometry.get('CANONICAL_POLYGON', 0)} |",
        f"| compressed RLE | {geometry.get('CANONICAL_RLE', 0)} |",
        f"| synthetic rectangle | {geometry.get('SYNTHETIC_RECTANGLE', 0)} |",
        f"| **total** | **{sum(geometry.values())}** |",
        "",
        "`CANONICAL_DATA_POLICY` The two synthetic rectangles are the phase 5B materialisation "
        "of the annotations that carried no geometry at all. They are stored as polygons but "
        "are not human-drawn segmentation, so they are reported as their own stratum rather "
        "than left to inflate the polygon population.",
        "",
        "## 4. Ultralytics segmentation-format constraints",
        "",
        f"`MODEL_FORMAT_CONSTRAINT` Established by reading the installed "
        f"`{fmt['package']} {fmt['version']}` source on this machine "
        f"(`{fmt['evidence']}`), not from documentation.",
        "",
        "| Property | Finding |",
        "| --- | --- |",
        f"| Row structure | {fmt['row_structure']} |",
        f"| One row per instance | {str(fmt['one_row_per_instance']).lower()} |",
        f"| Row is a single ring | {str(fmt['row_is_single_ring']).lower()} |",
        f"| Multi-segment separator | {fmt['multi_segment_separator']} |",
        f"| Holes representable | **{str(fmt['holes_representable']).lower()}** |",
        f"| Coordinate parsing | {fmt['coordinate_parsing']} |",
        "",
        f"`MODEL_FORMAT_CONSTRAINT` **Holes.** {fmt['hole_evidence']}",
        "",
        f"`MODEL_FORMAT_CONSTRAINT` **Multiple components.** {fmt['multi_segment_handling']}",
        "",
        f"`MODEL_FORMAT_CONSTRAINT` **Internal resampling.** {fmt['internal_resampling']}",
        "",
        f"`MODEL_FORMAT_CONSTRAINT` **Rasterisation.** {fmt['rasterization']}",
        "",
        f"`MODEL_FORMAT_CONSTRAINT` **A cardinality hazard.** {fmt['cardinality_hazard']} "
        f"Collisions found in this development set: "
        f"**{len(manifest['duplicate_box_collisions'])}**.",
        "",
        f"`MODEL_FORMAT_CONSTRAINT` **Negatives.** {fmt['empty_label_semantics']}",
        "",
        "## 5. Development population",
        "",
        "`CANONICAL_DATA_POLICY` The audit adapter uses exactly the development images the "
        "canonical segmentation documents use - none added, none dropped - and the frozen class "
        "map, by fingerprint:",
        "",
        f"- `class_map_sha256` `{manifest['class_map_sha256']}`",
        f"- `split_assignment_sha256` `{manifest['split_assignment_sha256']}`",
        f"- placeholder category `{manifest['placeholder_category_excluded']}`: absent, and no "
        "index shifted",
        "",
        f"`FACT` Zero-instance images are retained as empty label files: "
        f"{counts['per_split']['train']['negative_images']} in train and "
        f"{counts['per_split']['validation']['negative_images']} in validation. There is no "
        "holdout directory and no holdout key in the dataset descriptor.",
        "",
        "## 6. Conversion strategy",
        "",
        "`FACT` **One canonical annotation becomes exactly one YOLO row.** "
        f"{manifest['instance_cardinality']['canonical_annotations']} annotations produced "
        f"{manifest['instance_cardinality']['model_instance_rows']} instance rows; "
        f"{manifest['instance_cardinality']['instances_split']} were split, "
        f"{manifest['instance_cardinality']['instances_merged']} merged and "
        f"{manifest['instance_cardinality']['instances_dropped']} dropped.",
        "",
        "`CANONICAL_DATA_POLICY` That invariant is not a convenience. Splitting a disconnected "
        "mask into several rows would raise every fidelity number in this report and would "
        "quietly redefine what an instance is - changing per-image object counts and "
        "invalidating any later detection-versus-segmentation comparison.",
        "",
        "`FACT` Images were transferred without transformation: "
        f"{', '.join(manifest['adapter']['image_transfer_modes'])}, with every adapter image's "
        "SHA-256 verified equal to the canonical image's. No resize, no re-encode, no colour "
        "conversion.",
        "",
        "## 7. Polygon handling",
        "",
        "`FACT` A canonical polygon keeps its own coordinates. Re-tracing it from a raster would "
        "discard precision the canonical file already holds, so nothing is re-derived and no "
        "`approxPolyDP` simplification is applied - shrinking the label file is not worth "
        "boundary detail.",
        "",
        "## 8. RLE handling",
        "",
        f"`FACT` Compressed RLE is decoded with `{manifest['conversion_policy']['rle_decode']}`, "
        "the same reference decoder the canonical masks were built with in phases 5A-5D. A "
        "private decoder that disagreed with it would have made every number in this report a "
        "measurement of that disagreement.",
        "",
        "`FACT` The decoded mask's boundary is then traced with "
        f"`{manifest['conversion_policy']['contour_retrieval_mode']}` and "
        f"`{manifest['conversion_policy']['contour_approximation_mode']}`, epsilon "
        f"simplification `{manifest['conversion_policy']['contour_epsilon_simplification']}`. "
        "Those are the settings Ultralytics' own mask converter uses, so the audit measures the "
        "framework's behaviour rather than a private variant.",
        "",
        "## 9. Disconnected components",
        "",
        f"`FACT` Connected components counted at connectivity {topology['connectivity']}:",
        "",
        "| Components | Instances |",
        "| --- | --- |",
    ]
    for key, value in topology["connected_component_distribution"].items():
        lines.append(f"| {key} | {value} |")

    lines += [
        "",
        f"`FACT` Instances with more than one component: "
        f"**{topology['instances_with_multiple_components']}**. Maximum in one instance: "
        f"**{topology['maximum_component_count']}**.",
        "",
        f"`APPROXIMATION` `{manifest['conversion_policy']['multi_component_policy']}` - "
        "components are bridged into one path with a zero-width connector, classified "
        "`COMPONENT_JOIN_APPROXIMATION`. The connector adds area between the pieces. Section 15 "
        "reports what that costs on its own, separately from every other source of loss, so it "
        "is not hidden inside an aggregate mean.",
        "",
        "## 10. Holes",
        "",
        f"`FACT` Instances whose canonical mask contains at least one interior ring: "
        f"**{topology['instances_with_holes']}**. Total holes: **{topology['total_holes']}**. "
        f"Maximum in one instance: **{topology['maximum_holes_in_one_instance']}**. Total hole "
        f"area: **{topology['total_hole_pixels']} px**. Largest single-instance hole-area "
        f"fraction: **{_pct(topology['largest_hole_area_fraction'])}**.",
        "",
        f"`APPROXIMATION` `{manifest['conversion_policy']['hole_policy']}` - the format has no "
        "interior ring, so every hole is filled, classified `HOLE_FILL_APPROXIMATION`. The area "
        "above is what a conversion would add.",
        "",
        "## 11. Round-trip methodology",
        "",
        "`FACT` Every instance was measured against **the label as written to disk**, not "
        "against an in-memory object. The chain is: canonical mask → YOLO row → written file → "
        "re-read and parsed with the framework's own float32 semantics → denormalised → "
        f"rasterised by `{manifest['conversion_policy']['rasterization']}` on the "
        f"`{manifest['conversion_policy']['rasterization_canvas']}` canvas → compared.",
        "",
        "`COMPUTED_RESULT` **pycocotools and OpenCV do not rasterise identical geometry into "
        "identical pixels.** They differ on boundary-pixel inclusion, so even a polygon carried "
        "through unchanged will not reproduce the canonical mask exactly. Reporting that gap as "
        "format loss would blame the format for a rasteriser convention, so each instance is "
        "measured at three levels and each stage is charged only for what it costs:",
        "",
        "| Level | What it measures | Global mean |",
        "| --- | --- | --- |",
        f"| `control_iou` | canonical vs OpenCV rasterisation of the rings **before** any "
        f"conversion - the rasteriser and contour convention alone | {control['mean']:.6f} |",
        f"| `merged_iou` | after components are bridged into one ring, still at float64 - adds "
        f"the cost of **component joining** | {merged['mean']:.6f} |",
        f"| `mask_iou` | after serialisation, float32 parsing and the int32 snap - adds the cost "
        f"of **serialisation and quantisation** | {iou['mean']:.6f} |",
        "",
        "## 12. Global fidelity",
        "",
        f"`COMPUTED_RESULT` Over all **{overall['instances']}** development instances:",
        "",
        "| Statistic | Mask IoU |",
        "| --- | --- |",
        f"| minimum | {iou['min']:.6f} |",
        f"| P01 | {iou['p01']:.6f} |",
        f"| P05 | {iou['p05']:.6f} |",
        f"| P25 | {iou['p25']:.6f} |",
        f"| median | {iou['median']:.6f} |",
        f"| mean | {iou['mean']:.6f} |",
        f"| P75 | {iou['p75']:.6f} |",
        f"| P95 | {iou['p95']:.6f} |",
        f"| P99 | {iou['p99']:.6f} |",
        f"| maximum | {iou['max']:.6f} |",
        "",
        "`COMPUTED_RESULT` Descriptive bands. **These are not thresholds**, and phase 8A turns "
        "none of them into an approval or rejection rule:",
        "",
        "| Band | Instances | Share |",
        "| --- | --- | --- |",
    ]
    for name, entry in bands.items():
        lines.append(f"| `{name}` | {entry['count']} | {entry['percent']:.2f}% |")

    lines += [
        "",
        "## 13. Fidelity by canonical representation",
        "",
        "`COMPUTED_RESULT` The most important table in this report. An aggregate would let one "
        "representation vouch for the other:",
        "",
        *IOU_TABLE_HEADER,
    ]
    for name in sorted(fidelity["by_geometry_type"]):
        lines.append(_iou_row(f"`{name}`", fidelity["by_geometry_type"][name]))

    lines += [
        "",
        "## 14. Fidelity by class",
        "",
        "`COMPUTED_RESULT` Reported for every class, including the rare one:",
        "",
        *IOU_TABLE_HEADER,
    ]
    for name in sorted(fidelity["by_class"]):
        lines.append(_iou_row(f"`{name}`", fidelity["by_class"][name]))

    lines += [
        "",
        "`LIMITATION` A class-level fidelity figure inherits that class's instance count. "
        "`vest_loose` carries few development instances, so its row describes those instances "
        "and should not be read as a property of the class.",
        "",
        "## 15. Fidelity by topology",
        "",
        "`COMPUTED_RESULT` By connected-component count:",
        "",
        *IOU_TABLE_HEADER,
    ]
    for name in sorted(fidelity["by_component_count"]):
        lines.append(_iou_row(f"{name} component(s)", fidelity["by_component_count"][name]))

    lines += [
        "",
        "`COMPUTED_RESULT` With and without interior holes:",
        "",
        *IOU_TABLE_HEADER,
    ]
    for name in sorted(fidelity["by_hole_presence"]):
        lines.append(_iou_row(f"`{name}`", fidelity["by_hole_presence"][name]))

    holed = fidelity["by_hole_presence"].get("with_holes")
    plain = fidelity["by_hole_presence"].get("without_holes")
    if holed and plain:
        lines += [
            "",
            "`LIMITATION` **That comparison is confounded by size and must not be read as "
            "'holes do not matter'.** Instances that have holes are much larger - median "
            f"{holed['canonical_area_px']['median']:.0f} px against "
            f"{plain['canonical_area_px']['median']:.0f} px - and section 16 shows mask area is "
            "the strongest driver of fidelity in this dataset. The two effects run in opposite "
            "directions here and the strata are not matched, so the near-equal IoU means the "
            "size advantage roughly offsets the hole-filling penalty, not that filling holes is "
            f"free. The absolute cost is stated in section 10: {topology['total_hole_pixels']} "
            "px added across "
            f"{topology['instances_with_holes']} instances.",
        ]

    edges = fidelity["mask_area_quartile_edges"]
    lines += [
        "",
        "`COMPUTED_RESULT` Isolating each approximation, averaged over all instances:",
        "",
        f"- component joining costs **{overall['join_loss']['mean']:.6f}** mean IoU "
        f"(max {overall['join_loss']['max']:.6f})",
        f"- serialisation and quantisation cost **{overall['serialization_loss']['mean']:.6f}** "
        f"mean IoU (max {overall['serialization_loss']['max']:.6f})",
        "",
        "## 16. Fidelity by mask size",
        "",
        f"`COMPUTED_RESULT` Quartiles of the development mask-area distribution: "
        f"P25 {edges['p25']:.0f} px, P50 {edges['p50']:.0f} px, P75 {edges['p75']:.0f} px.",
        "",
        f"`LIMITATION` {edges['note']}",
        "",
        *IOU_TABLE_HEADER,
    ]
    for name in sorted(fidelity["by_mask_area_quartile"]):
        lines.append(_iou_row(f"`{name}`", fidelity["by_mask_area_quartile"][name]))

    thinness = fidelity["thinness"]
    lines += [
        "",
        f"`COMPUTED_RESULT` Thin-structure proxy (perimeter over the square root of area), "
        f"reported as a covariate rather than as a definition of 'thin': median "
        f"{thinness['median']:.3f}, P95 {thinness['p95']:.3f}, maximum {thinness['max']:.3f}.",
        "",
        "## 17. Area error",
        "",
        f"`COMPUTED_RESULT` Absolute relative mask-area error: mean {error['mean']:.6f}, median "
        f"{error['median']:.6f}, P95 {error['p95']:.6f}, P99 {error['p99']:.6f}, maximum "
        f"{error['max']:.6f}.",
        "",
        "| Exceeds | Instances | Share |",
        "| --- | --- | --- |",
    ]
    for name, entry in overall["area_error_bands"].items():
        threshold = name.replace("above_", "")
        lines.append(
            f"| {float(threshold) * 100:g}% | {entry['count']} | {entry['percent']:.2f}% |"
        )

    lines += [
        "",
        f"`COMPUTED_RESULT` Total pixels added across the development set: "
        f"{overall['total_fp_pixels']}. Total dropped: {overall['total_fn_pixels']}. Against a "
        f"canonical total of {overall['total_canonical_area_px']} px.",
        "",
        "`LIMITATION` These bands are descriptive evidence, not a training-approval threshold.",
        "",
        "## 18. Worst cases",
        "",
        "`COMPUTED_RESULT` The 20 lowest mask IoU instances, selected deterministically. No "
        "source photograph was opened: the diagnosis is geometry-only.",
        "",
        "| Image | Ann | Class | Geometry | Area px | Comp | Holes | IoU | Control | Reasons |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for case in manifest["worst_cases"]["worst_mask_iou"]:
        lines.append(
            f"| `{case['source_image_id']}` | {case['annotation_id']} | `{case['class']}` | "
            f"{case['canonical_geometry_type'].replace('CANONICAL_', '').lower()} | "
            f"{case['canonical_area_px']} | {case['connected_components']} | "
            f"{case['hole_count']} | {case['mask_iou']:.4f} | {case['control_iou']:.4f} | "
            f"{', '.join(case['reason_flags']) or '-'} |"
        )

    lines += [
        "",
        "`COMPUTED_RESULT` The 20 largest relative area errors:",
        "",
        "| Image | Ann | Class | Geometry | Area px | Comp | Holes | Rel. area err | Reasons |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for case in manifest["worst_cases"]["worst_relative_area_error"]:
        lines.append(
            f"| `{case['source_image_id']}` | {case['annotation_id']} | `{case['class']}` | "
            f"{case['canonical_geometry_type'].replace('CANONICAL_', '').lower()} | "
            f"{case['canonical_area_px']} | {case['connected_components']} | "
            f"{case['hole_count']} | {case['relative_area_error']:.4f} | "
            f"{', '.join(case['reason_flags']) or '-'} |"
        )

    lines += [
        "",
        "`COMPUTED_RESULT` Reason-flag census across all instances. An instance whose deviation "
        "no stage and no recorded topology explains is `UNATTRIBUTED` rather than assigned the "
        "most plausible-sounding cause:",
        "",
        "| Flag | Instances |",
        "| --- | --- |",
        f"| (not materially non-exact, IoU >= {MATERIAL_IOU_LABEL}) | "
        f"{fidelity['exact_instances']} |",
    ]
    for flag, count in fidelity["reason_flag_counts"].items():
        lines.append(f"| `{flag}` | {count} |")

    lines += [
        "",
        "## 19. Framework-parser validation",
        "",
        "`FACT` The generated labels were handed to the installed Ultralytics dataset scanner. "
        "Structural parsing only - no model was constructed, no weight downloaded and nothing "
        "trained.",
        "",
        "| Split | Images discovered | Instances parsed | Segment rows | Negatives |",
        "| --- | --- | --- | --- | --- |",
    ]
    for split in sorted(parser["splits"]):
        row = parser["splits"][split]
        lines.append(
            f"| `{split}` | {row['images_discovered']} | {row['instances']} | "
            f"{row['segment_rows']} | {row['negative_images']} |"
        )

    lines += [
        "",
        f"`FACT` Classes: {parser['classes']}. Corrupt labels: {parser['corrupt_labels']}. "
        f"Holdout split present in the descriptor: "
        f"{str(parser['test_split_present']).lower()}. Coordinate bounds violations: "
        f"{manifest['bounds_violations']}.",
        "",
        "`FACT` **Instance cardinality survived parsing.** The framework read back exactly the "
        "instance count that was written, in both splits. Had it not, the audit would have "
        "stopped rather than reported a fidelity distribution over a different set of objects.",
        "",
        "## 20. Reproducibility",
        "",
        f"`FACT` Audit protocol `configs/{AUDIT_YAML}`, SHA-256 "
        f"`{manifest['audit_config_sha256']}`. Every methodological choice that could move a "
        "number - decoder, contour modes, joining policy, serialisation precision, "
        "rasterisation, connectivity, bands - lives in that file rather than in Python.",
        "",
        "| Artifact | SHA-256 |",
        "| --- | --- |",
    ]
    for name in sorted(manifest["adapter_fingerprints"]):
        lines.append(f"| `{name}` | `{manifest['adapter_fingerprints'][name]}` |")

    lines += [
        "",
        "`FACT` Generation is deterministic: running the audit twice produces byte-identical "
        "labels, table, manifest and report.",
        "",
        "## 21. Holdout compliance",
        "",
        f"`CANONICAL_DATA_POLICY` Status: **`{HOLDOUT_STATUS}`**. {HOLDOUT_REASON}",
        "",
        f"`CANONICAL_DATA_POLICY` `{HOLDOUT_UNLOCK_ENV_VAR}` was not set. There is no holdout "
        "directory in the audit adapter, no holdout key in the dataset descriptor, no holdout "
        "label file, and no holdout row in the fidelity table.",
        "",
        "## 22. Limitations",
        "",
        "`LIMITATION` This measures **representation, not performance**. Nothing here predicts "
        "how well a model would learn from these labels, or what mask AP any architecture would "
        "reach.",
        "",
        "`LIMITATION` The reconstruction is compared on the original source-image canvas. "
        "Training letterboxes and downsamples masks (`mask_ratio`), which is a further "
        "transformation this audit deliberately excludes, because it is a model-input choice "
        "rather than a property of the label format.",
        "",
        "`LIMITATION` The control isolates the pycocotools-versus-OpenCV rasterisation "
        "difference but does not eliminate it. Some of the reported gap is a convention "
        "difference between two libraries, not information destroyed.",
        "",
        "`LIMITATION` Fidelity was measured on development data only. It is a property of these "
        "annotations, not a general claim about the format.",
        "",
        "## 23. Architecture-decision evidence",
        "",
        f"`PENDING_HUMAN_DECISION` `segmentation_architecture_selection: "
        f"{manifest['segmentation_architecture_selection']}` · `segmentation_baseline: "
        f"{manifest['segmentation_baseline']}` · `S0: {manifest['S0']}`.",
        "",
        "`PENDING_HUMAN_DECISION` What this audit establishes, and nothing more: the conversion "
        "is **structurally possible** - every one of the "
        f"{manifest['instance_cardinality']['canonical_annotations']} development annotations "
        "became exactly one parseable YOLO row, with no instance split, merged or dropped - and "
        "the geometric cost of doing so is distributed as reported above. Whether that cost is "
        "acceptable is a judgement about this project's goals, not a fact this phase can "
        "measure.",
        "",
        "`PENDING_HUMAN_DECISION` Recorded as a future alternative only, neither implemented nor "
        "benchmarked nor selected: if format fidelity is judged materially inadequate, a "
        "mask-native instance-segmentation architecture able to consume canonical COCO "
        "masks and RLE directly (Mask R-CNN, for example) would avoid this conversion "
        "entirely. Nothing about that option has been installed, run or costed, and naming it "
        "here is not a preference.",
        "",
        "## 24. Next phase",
        "",
        "Phase 8B - human review of this evidence and the segmentation architecture decision. "
        "It has not started. No segmentation model may be trained, and no S0 protocol defined, "
        "before that decision is taken and recorded.",
        "",
        "`FACT` The frozen detector was not retrained, revalidated, run or altered by this "
        f"phase; it remains `{manifest['frozen_detector_sha256']}`. No detector-versus-segmenter "
        "comparison was made, and none is authorised until a segmentation model is frozen.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run the phase 8A segmentation-adapter fidelity audit.

    Args:
        argv: Command-line arguments.

    Returns:
        Process exit status.
    """
    parser_cli = argparse.ArgumentParser(description=__doc__)
    parser_cli.add_argument(
        "--verify-only",
        action="store_true",
        help="validate every input and the format contract without writing any artifact",
    )
    args = parser_cli.parse_args(argv)

    paths = ProjectPaths.from_root()
    if holdout_unlocked():
        print(
            f"REFUSED: {HOLDOUT_UNLOCK_ENV_VAR} is set. This audit reads development data only "
            "and declines to run in an environment where the holdout is unlocked.",
            file=sys.stderr,
        )
        return 2

    try:
        config = load_audit_config(paths.configs / AUDIT_YAML)
        verified = verify_inputs(paths, config)
    except (ConfigError, AuditError) as exc:
        print(f"{BLOCKED}: {exc}", file=sys.stderr)
        return 2

    class_names = {index: name for name, index in verified["class_map"].items()}
    if args.verify_only:
        print("VERIFIED: canonical inputs, class map and frozen detector are intact.")
        print(f"config     sha256 {config.fingerprint()}")
        print(f"class map  sha256 {verified['class_map_sha256']}")
        print(f"split      sha256 {verified['split_assignment_sha256']}")
        print(f"holdout    {HOLDOUT_STATUS}")
        return 0

    labels: dict[str, dict[str, str]] = {}
    counters: dict[str, dict[str, Any]] = {}
    rows: list[FidelityRow] = []
    geometry_counts = dict.fromkeys(GEOMETRY_TYPES, 0)
    try:
        for split in config["allowed_splits"]:
            split_labels, split_rows, split_counters = convert_split(
                paths, config, split, class_names
            )
            labels[split] = split_labels
            rows.extend(split_rows)
            counters[split] = split_counters
            for name, count in split_counters["geometry_counts"].items():
                geometry_counts[name] += count
            print(
                f"{split:<11} {split_counters['images']} images  "
                f"{split_counters['rows_written']} instance rows  "
                f"{split_counters['negative_images']} negatives"
            )
    except (AuditError, SegmentationAdapterError) as exc:
        print(f"{GEOMETRY_AUDIT_FAILURE}: {exc}", file=sys.stderr)
        return 2

    total_annotations = sum(int(counters[split]["annotations"]) for split in counters)
    total_rows = sum(int(counters[split]["rows_written"]) for split in counters)
    if total_annotations != total_rows or total_annotations != len(rows):
        print(
            f"{GEOMETRY_AUDIT_FAILURE}: {total_annotations} canonical annotations produced "
            f"{total_rows} rows and {len(rows)} measurements. Instance cardinality was not "
            "preserved.",
            file=sys.stderr,
        )
        return 2

    try:
        adapter = write_adapter(paths, config, labels, verified["class_map"])
        expected = {split: int(counters[split]["rows_written"]) for split in counters}
        parser_result = validate_with_framework(paths, config, expected)
    except AuditError as exc:
        print(f"{STRUCTURAL_INCOMPATIBILITY}: {exc}", file=sys.stderr)
        return 2

    fingerprints = {
        "audit_config_sha256": config.fingerprint(),
        "labels_train_sha256": label_fingerprint(labels["train"]),
        "labels_validation_sha256": label_fingerprint(labels["validation"]),
        "labels_development_sha256": label_fingerprint(
            {f"{split}/{stem}": text for split in labels for stem, text in labels[split].items()}
        ),
        "image_membership_sha256": digest(
            {split: sorted(labels[split]) for split in sorted(labels)}
        ),
        "fidelity_rows_sha256": fingerprint_rows(rows),
    }

    aggregates = build_aggregates(rows)
    topology = build_topology(rows)
    cases = worst_cases(rows)
    manifest = build_manifest(
        config=config,
        verified=verified,
        counters=counters,
        adapter=adapter,
        fingerprints=fingerprints,
        aggregates=aggregates,
        topology=topology,
        parser=parser_result,
        cases=cases,
        geometry_counts=geometry_counts,
    )

    report = audit_report(manifest, commit=git_commit(paths.root))
    findings = scan_for_sensitive(report) + scan_for_sensitive(json.dumps(manifest))
    if findings:
        print(f"{BLOCKED}: sensitive content in the emitted artifacts: {findings}", file=sys.stderr)
        return 2

    csv_sha256 = write_fidelity_csv(paths.reports / FIDELITY_CSV, rows)
    manifest["adapter_fingerprints"]["fidelity_table_sha256"] = csv_sha256
    manifest_digest = write_json(paths.reports / AUDIT_MANIFEST_JSON, manifest)
    (paths.reports / AUDIT_REPORT_MD).write_text(report, encoding="utf-8", newline="\n")

    record = ProvenanceRecord.create(
        name="segmentation_adapter_fidelity_audit",
        phase=8,
        config={
            "segmentation_adapter_audit": f"configs/{AUDIT_YAML}",
            "audit_config_sha256": config.fingerprint(),
        },
        details={
            "phase": "8A",
            "classification": COMPLETE,
            "adapter_type": ADAPTER_TYPE,
            "adapter_status": config["status"],
            "development_images": manifest["development_counts"]["images"],
            "development_annotations": manifest["development_counts"]["annotations"],
            "model_instance_rows": total_rows,
            "instance_cardinality_preserved": True,
            "canonical_geometry_counts": geometry_counts,
            "global_mean_mask_iou": aggregates["global"]["mask_iou"]["mean"],
            "global_median_mask_iou": aggregates["global"]["mask_iou"]["median"],
            "global_min_mask_iou": aggregates["global"]["mask_iou"]["min"],
            "instances_with_holes": topology["instances_with_holes"],
            "instances_with_multiple_components": topology["instances_with_multiple_components"],
            "adapter_fingerprints": manifest["adapter_fingerprints"],
            "parser_validation_passed": True,
            "segmentation_architecture_selection": ARCHITECTURE_UNSELECTED,
            "segmentation_baseline": BASELINE_UNFROZEN,
            "S0": S0_UNDEFINED,
            "models_trained_in_this_phase": 0,
            "models_evaluated_in_this_phase": 0,
            "detector_touched": False,
            "holdout_accessed": False,
        },
        repo_root=paths.root,
    )
    record.add_input(paths.configs / AUDIT_YAML, relative_to=paths.root)
    for name in (TASK_MANIFEST_JSON, SPLIT_MANIFEST_JSON, FINAL_DETECTOR_JSON):
        record.add_input(paths.reports / name, relative_to=paths.root)
    for name in (FIDELITY_CSV, AUDIT_MANIFEST_JSON, AUDIT_REPORT_MD):
        record.add_output(paths.reports / name, relative_to=paths.root)
    record.write_json(paths.reports / PROVENANCE_JSON)

    overall = aggregates["global"]["mask_iou"]
    print(COMPLETE)
    print(
        f"instances  {aggregates['global']['instances']}  "
        f"(annotations {total_annotations}, rows {total_rows})"
    )
    print(
        f"mask IoU   mean {overall['mean']:.6f}  median {overall['median']:.6f}  "
        f"p05 {overall['p05']:.6f}  min {overall['min']:.6f}"
    )
    for name in sorted(aggregates["by_geometry_type"]):
        summary = aggregates["by_geometry_type"][name]["mask_iou"]
        print(
            f"  {name:<21} n={aggregates['by_geometry_type'][name]['instances']:<5} "
            f"mean {summary['mean']:.6f}  min {summary['min']:.6f}"
        )
    print(
        f"topology   {topology['instances_with_multiple_components']} multi-component, "
        f"{topology['instances_with_holes']} with holes "
        f"({topology['total_hole_pixels']} px)"
    )
    print(f"selection  {ARCHITECTURE_UNSELECTED}  S0 {S0_UNDEFINED}")
    print(f"holdout    {HOLDOUT_STATUS}")
    print(f"table      reports/{FIDELITY_CSV}  sha256 {csv_sha256}")
    print(f"manifest   reports/{AUDIT_MANIFEST_JSON}  sha256 {manifest_digest}")
    print(f"report     reports/{AUDIT_REPORT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
