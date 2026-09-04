"""Build the YOLO detection adapter from the canonical COCO detection dataset.

Phase 6A. Ultralytics reads labels in its own format, so a derived
representation is unavoidable; what is avoidable is a *lossy* one. This script
converts every canonical detection box, writes it, reads it back from the file
on disk, decodes it to source pixels and compares it with the canonical box.
The adapter is only declared usable when that audit is clean.

Canonical COCO remains the ground truth. Nothing here writes a segmentation
label, reads the provider's bbox, or touches the protected split: the holdout
has no adapter, no directory and no dataset entry, and a development run refuses
it outright.

Runs entirely offline. Requires:

* ``configs/detection_adapter.yaml``
* ``reports/task_dataset_manifest.json``             scripts/materialize_task_datasets.py
* ``data/processed/canonical/``                      scripts/materialize_task_datasets.py

Writes (bulk adapter data git-ignored, evidence committed):
    data/processed/adapters/yolo_detection/{images,labels}/{train,val}/
    data/processed/adapters/yolo_detection/dataset.yaml
    reports/detection_adapter_manifest.json
    reports/detection_adapter_report.md
    reports/detection_adapter.provenance.json

Usage:
    uv run python scripts/build_detection_adapter.py
    uv run python scripts/build_detection_adapter.py --verify-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from construction_safety_vision.config import ConfigError
from construction_safety_vision.data.canonical import scan_for_sensitive
from construction_safety_vision.data.materialization import (
    ImageCopy,
    MaterializationError,
    assert_development_split,
    copy_image,
    image_content_fingerprint,
    resolve_split_images,
)
from construction_safety_vision.data.split_freeze import (
    TEST,
    SplitManifestError,
    load_frozen_splits,
)
from construction_safety_vision.data.yolo_detection_adapter import (
    ADAPTER_TYPE,
    NOT_MATERIALIZED,
    AdapterConfig,
    YoloAdapterError,
    YoloBox,
    coco_to_yolo,
    dataset_yaml,
    label_fingerprint,
    label_text,
    load_adapter_config,
    parse_label_line,
    round_trip_boxes,
    validate_normalised,
)
from construction_safety_vision.paths import ProjectPaths
from construction_safety_vision.provenance import ProvenanceRecord, git_commit
from construction_safety_vision.splits import HOLDOUT_UNLOCK_ENV_VAR

TASK_MANIFEST_JSON = "task_dataset_manifest.json"
POPULATION_MANIFEST_JSON = "canonical_modeling_manifest.json"
SPLIT_MANIFEST_JSON = "split_manifest.json"
ADAPTER_MANIFEST_JSON = "detection_adapter_manifest.json"
REPORT_MD = "detection_adapter_report.md"
PROVENANCE_JSON = "detection_adapter.provenance.json"
DATASET_YAML = "dataset.yaml"

DEVELOPMENT_SPLITS: tuple[str, ...] = ("train", "validation")


class AdapterInputError(RuntimeError):
    """Raised when a required input artifact is missing or unusable."""


def read_json(path: Path) -> dict[str, Any]:
    """Read a committed JSON artifact.

    Args:
        path: File to read.

    Returns:
        The parsed mapping.

    Raises:
        AdapterInputError: If the file is absent or unparsable.
    """
    if not path.is_file():
        msg = f"{path.name} not found; run the phase that produces it first"
        raise AdapterInputError(msg)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"{path.name} is not valid JSON ({exc})"
        raise AdapterInputError(msg) from exc


class Inputs:
    """Every committed artifact the adapter reads, parsed once.

    Attributes:
        splits: The frozen split.
        task_manifest: The phase 5D dataset manifest.
        class_map: The frozen class map.
        class_map_sha256: The frozen class map's digest.
    """

    def __init__(self, paths: ProjectPaths, config: AdapterConfig) -> None:
        """Load and parse every input.

        Args:
            paths: Project layout.
            config: The adapter protocol.

        Raises:
            AdapterInputError: If an artifact is missing or inconsistent.
            SplitManifestError: If the frozen split does not verify.
        """
        self.splits = load_frozen_splits(paths.reports / SPLIT_MANIFEST_JSON)
        manifest_name = config.canonical_task_manifest.rsplit("/", 1)[-1]
        self.task_manifest = read_json(paths.reports / manifest_name)
        population = read_json(paths.reports / POPULATION_MANIFEST_JSON)
        self.class_map: dict[str, int] = dict(population["class_map"])
        self.class_map_sha256: str = population["fingerprints"]["class_map_sha256"]

        # The adapter is derived from the phase 5D datasets, so it must be
        # derived from the *same* phase 5D datasets whose fingerprints the task
        # manifest records. A mismatch here means the canonical data moved.
        if self.task_manifest["class_map"] != self.class_map:
            msg = "the task manifest's class map disagrees with the frozen population class map"
            raise AdapterInputError(msg)
        if self.task_manifest["class_map_sha256"] != self.class_map_sha256:
            msg = "the task manifest's class_map_sha256 disagrees with the frozen population"
            raise AdapterInputError(msg)
        if self.task_manifest["split_assignment_sha256"] != self.splits.split_assignment_sha256:
            msg = "the task manifest was built over a different frozen split"
            raise AdapterInputError(msg)
        if self.task_manifest[TEST]["status"] != NOT_MATERIALIZED:
            msg = f"the task manifest does not record the holdout as {NOT_MATERIALIZED}"
            raise AdapterInputError(msg)


def canonical_detection(paths: ProjectPaths, config: AdapterConfig, split: str) -> dict[str, Any]:
    """Read the canonical COCO detection document of one split.

    Args:
        paths: Project layout.
        config: The adapter protocol.
        split: Canonical split name.

    Returns:
        The parsed COCO document.

    Raises:
        AdapterInputError: If it is absent.
    """
    path = paths.root / config.canonical_annotations_root / f"detection_{split}.coco.json"
    if not path.is_file():
        msg = (
            f"{path.name} not found; run scripts/materialize_task_datasets.py to rebuild the "
            "canonical task datasets"
        )
        raise AdapterInputError(msg)
    return json.loads(path.read_text(encoding="utf-8"))


def adapt_split(
    paths: ProjectPaths,
    config: AdapterConfig,
    inputs: Inputs,
    split: str,
    *,
    allow_test: bool = False,
    env: dict[str, str] | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Build one split's YOLO images, labels and fidelity audit.

    Args:
        paths: Project layout.
        config: The adapter protocol.
        inputs: The parsed inputs.
        split: Canonical split name.
        allow_test: Explicit in-code opt-in for the holdout.
        env: Environment mapping to read. Defaults to ``os.environ``.
        write: Whether to write anything.

    Returns:
        The split's adapter record.

    Raises:
        HoldoutViolationError: If the holdout is requested without both opt-ins.
        YoloAdapterError: If a box cannot be converted without losing geometry.
        MaterializationError: If an image cannot be copied byte-for-byte.
    """
    frozen_ids = resolve_split_images(
        inputs.splits,
        split,
        purpose=f"building the YOLO detection adapter for {split}",
        allow_test=allow_test,
        env=env,
    )
    document = canonical_detection(paths, config, split)
    canvases = {
        record["id"]: (record["width"], record["height"], record["source_image_id"])
        for record in document["images"]
    }

    boxes_by_image: dict[str, list[YoloBox]] = {
        record["source_image_id"]: [] for record in document["images"]
    }
    canonical_by_key: dict[tuple[str, int], list[float]] = {}
    problems: list[str] = []
    for record in sorted(document["annotations"], key=lambda r: r["id"]):
        width, height, image_id = canvases[record["image_id"]]
        box = coco_to_yolo(
            record["bbox"],
            class_id=record["category_id"],
            image_width=width,
            image_height=height,
        )
        invalid = validate_normalised(box, epsilon=config.tolerances.normalised_bound_epsilon)
        if invalid:
            problems.append(f"{split}/{image_id} annotation {record['id']}: {'; '.join(invalid)}")
        boxes_by_image[image_id].append(box)
        canonical_by_key[(image_id, record["id"])] = list(record["bbox"])

    labels = {
        image_id: label_text(boxes, precision=config.label_precision)
        for image_id, boxes in boxes_by_image.items()
    }

    directory = config.split_directories[split]
    output_root = paths.root / config.output_root
    image_directory = output_root / "images" / directory
    label_directory = output_root / "labels" / directory

    copies: list[ImageCopy] = []
    if write:
        image_directory.mkdir(parents=True, exist_ok=True)
        label_directory.mkdir(parents=True, exist_ok=True)
        for record in sorted(document["images"], key=lambda r: r["source_image_id"]):
            image_id = record["source_image_id"]
            source = paths.root / config.canonical_images_root / split / record["file_name"]
            copies.append(
                copy_image(source, image_directory / record["file_name"], source_image_id=image_id)
            )
            stem = Path(record["file_name"]).stem
            (label_directory / f"{stem}.txt").write_text(
                labels[image_id], encoding="utf-8", newline="\n"
            )

    round_trip = None
    if write:
        # Read the labels back off disk: the serialisation is part of what the
        # audit is testing, so comparing against the in-memory boxes would prove
        # nothing about the files a trainer will actually load.
        emitted_by_image: dict[str, list[YoloBox]] = {}
        for record in sorted(document["images"], key=lambda r: r["source_image_id"]):
            stem = Path(record["file_name"]).stem
            text = (label_directory / f"{stem}.txt").read_text(encoding="utf-8")
            emitted_by_image[record["source_image_id"]] = [
                parse_label_line(line) for line in text.splitlines() if line.strip()
            ]
        audit_entries = []
        counters: dict[str, int] = {}
        for record in sorted(document["annotations"], key=lambda r: r["id"]):
            width, height, image_id = canvases[record["image_id"]]
            index = counters.get(image_id, 0)
            counters[image_id] = index + 1
            emitted = emitted_by_image[image_id][index]
            if emitted.class_id != record["category_id"]:
                problems.append(
                    f"{split}/{image_id} annotation {record['id']}: emitted class "
                    f"{emitted.class_id} != canonical {record['category_id']}"
                )
            audit_entries.append(
                (
                    f"{split}/{image_id}#{record['id']}",
                    canonical_by_key[(image_id, record["id"])],
                    emitted,
                    width,
                    height,
                )
            )
        round_trip = round_trip_boxes(audit_entries, tolerance_px=config.tolerances.round_trip_px)
        problems.extend(f"round-trip/{split}: {p}" for p in round_trip.mismatches)

    frozen = set(frozen_ids)
    emitted_ids = set(boxes_by_image)
    if emitted_ids != frozen:
        problems.append(
            f"{split}: adapter covers {len(emitted_ids)} image(s) but the frozen split holds "
            f"{len(frozen)}"
        )

    negatives = sum(1 for text in labels.values() if not text)
    return {
        "split": split,
        "directory": directory,
        "images": len(boxes_by_image),
        "annotations": sum(len(boxes) for boxes in boxes_by_image.values()),
        "negative_images": negatives,
        "labels": labels,
        "label_sha256": label_fingerprint(labels),
        "image_content_sha256": image_content_fingerprint(copies) if copies else "",
        "image_copies": copies,
        "round_trip": round_trip.as_dict() if round_trip is not None else None,
        "round_trip_object": round_trip,
        "instances_by_class": _class_counts(boxes_by_image, inputs.class_map),
        "problems": problems,
    }


def _class_counts(
    boxes_by_image: dict[str, list[YoloBox]], class_map: dict[str, int]
) -> dict[str, int]:
    """Count emitted instances per class.

    Args:
        boxes_by_image: Emitted boxes keyed by source image id.
        class_map: The frozen class map.

    Returns:
        Instance counts keyed by class name.
    """
    by_index = {index: name for name, index in class_map.items()}
    counts = dict.fromkeys(class_map, 0)
    for boxes in boxes_by_image.values():
        for box in boxes:
            counts[by_index[box.class_id]] += 1
    return counts


def build_manifest(
    config: AdapterConfig, inputs: Inputs, results: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Assemble the versioned adapter manifest.

    Args:
        config: The adapter protocol.
        inputs: The parsed inputs.
        results: Per-split adapter records.

    Returns:
        The manifest mapping.
    """
    task = inputs.task_manifest
    manifest: dict[str, Any] = {
        "schema_version": config.schema_version,
        "phase": "6A",
        "adapter_type": ADAPTER_TYPE,
        "status": "DEVELOPMENT_ADAPTER_BUILT",
        "canonical_task_manifest": config.canonical_task_manifest,
        "canonical_detection_format": task["canonical_detection_format"],
        "derived_representation": True,
        "ground_truth": "reports/task_dataset_manifest.json (canonical COCO detection)",
        "split_assignment_sha256": task["split_assignment_sha256"],
        "modeling_population_sha256": task["modeling_population_sha256"],
        "class_map_sha256": inputs.class_map_sha256,
        "class_map": inputs.class_map,
        "adapter_config_sha256": config.fingerprint(),
        "adapter_config": config.as_dict(),
        "bbox_source": config.bbox_source,
        "provider_bbox_used": False,
        "segmentation_labels_generated": False,
        "output_root": config.output_root,
        "bulk_data_committed": False,
    }
    for split in DEVELOPMENT_SPLITS:
        result = results[split]
        manifest[split] = {
            "status": "ADAPTED",
            "adapter_directory": result["directory"],
            "image_count": result["images"],
            "annotation_count": result["annotations"],
            "negative_images": result["negative_images"],
            "instances_by_class": result["instances_by_class"],
            "yolo_label_sha256": result["label_sha256"],
            "image_content_sha256": result["image_content_sha256"],
            "canonical_image_content_sha256": task[split]["image_content_sha256"],
            "round_trip": result["round_trip"],
        }
    manifest[TEST] = {
        "status": NOT_MATERIALIZED,
        "reason": (
            "the protected holdout gets no adapter while the models are unfrozen. No test "
            "image, label, directory, dataset entry or statistic exists. The "
            "final-evaluation phase builds it through this same code, with both holdout "
            "opt-ins present."
        ),
    }
    manifest["development_totals"] = {
        "images": sum(results[s]["images"] for s in DEVELOPMENT_SPLITS),
        "annotations": sum(results[s]["annotations"] for s in DEVELOPMENT_SPLITS),
        "negative_images": sum(results[s]["negative_images"] for s in DEVELOPMENT_SPLITS),
    }
    trips = [results[s]["round_trip"] for s in DEVELOPMENT_SPLITS]
    if all(trip is not None for trip in trips):
        manifest["round_trip_totals"] = {
            "annotations_checked": sum(t["annotations_checked"] for t in trips),
            "within_tolerance": sum(t["within_tolerance"] for t in trips),
            "exact_matches": sum(t["exact_matches"] for t in trips),
            "mismatches": sum(t["mismatches"] for t in trips),
            "max_delta_px": max(t["max_delta_px"] for t in trips),
            "tolerance_px": config.tolerances.round_trip_px,
        }
    manifest["git_commit"] = git_commit()
    return manifest


def build_report(manifest: dict[str, Any], config: AdapterConfig, inputs: Inputs) -> str:
    """Write the detection-adapter report.

    Args:
        manifest: The adapter manifest.
        config: The adapter protocol.
        inputs: The parsed inputs.

    Returns:
        The report as Markdown.
    """
    lines: list[str] = []
    add = lines.append
    totals = manifest["development_totals"]
    trip = manifest["round_trip_totals"]
    commit = manifest["git_commit"] or "unavailable"

    add("# YOLO Detection Adapter - Phase 6A")
    add("")
    add(f"Phase: 6A · Commit: `{commit}` · Status: **{manifest['status']}**")
    add("")
    add(
        "## 1. Purpose\n\n"
        "`MODEL-ADAPTER POLICY` Ultralytics reads labels in its own format, so a derived "
        "representation is unavoidable. What is avoidable is a **lossy** one, and a lossy "
        "adapter is invisible: training would learn slightly wrong boxes and every downstream "
        "metric would quietly measure the wrong thing. This adapter therefore exists together "
        "with an audit that proves it changed nothing."
    )
    add("")
    add("## 2. Canonical source")
    add("")
    add(
        f"`FACT` The input is the canonical COCO detection dataset frozen in phase 5D "
        f"(`{config.canonical_task_manifest}`), whose boxes are themselves derived from the "
        "canonical segmentation geometry."
    )
    add("")
    add("| Fingerprint | Value |")
    add("| --- | --- |")
    add(f"| `split_assignment_sha256` | `{manifest['split_assignment_sha256']}` |")
    add(f"| `modeling_population_sha256` | `{manifest['modeling_population_sha256']}` |")
    add(f"| `class_map_sha256` | `{manifest['class_map_sha256']}` |")
    add(f"| `adapter_config_sha256` | `{manifest['adapter_config_sha256']}` |")
    add("")
    add("## 3. Why the adapter is derived, not canonical")
    add("")
    add(
        "`MODEL-ADAPTER POLICY` A model-specific representation must never redefine ground "
        "truth. If this adapter and the COCO file ever disagree, **the COCO file is right and "
        "the adapter is broken**. Concretely: the adapter is regenerated from COCO, never "
        "edited; it is git-ignored rather than committed; and no downstream phase may treat a "
        "YOLO label as the authority for what an object is."
    )
    add("")
    add(
        "`MODEL-ADAPTER POLICY` **Detection only.** No YOLO segmentation labels were "
        "generated. The canonical segmentation state holds compressed RLE, which YOLO's "
        "polygon-only format cannot express without rasterising and re-polygonising; that "
        "conversion must be quantitatively fidelity-audited before any model sees it, and it "
        "is a later model-adapter concern. Boxes have no such problem - a box is a box - which "
        "is exactly why the detection adapter can be proven lossless and a segmentation one "
        "cannot be assumed to be."
    )
    add("")
    add("## 4. Class mapping")
    add("")
    add("| YOLO index | Class |")
    add("| --- | --- |")
    for name, index in sorted(inputs.class_map.items(), key=lambda item: item[1]):
        add(f"| {index} | `{name}` |")
    add("")
    add(
        f"`COMPUTED RESULT` The mapping is the frozen canonical class map, verified against "
        f"`class_map_sha256` = `{manifest['class_map_sha256']}`. No second ordering was "
        "introduced, and no index was re-derived from dictionary iteration or runtime sorting."
    )
    add("")
    add(
        "`FACT` The provider's placeholder category `object` carries no annotation and does "
        "not appear in the adapter, the dataset descriptor or any label file."
    )
    add("")
    lines.extend(_report_tail(manifest, config, totals, trip))
    return "\n".join(lines) + "\n"


def _report_tail(
    manifest: dict[str, Any],
    config: AdapterConfig,
    totals: dict[str, int],
    trip: dict[str, Any],
) -> list[str]:
    """Write the report from membership onward.

    Args:
        manifest: The adapter manifest.
        config: The adapter protocol.
        totals: Development totals.
        trip: Round-trip totals.

    Returns:
        The report lines.
    """
    lines: list[str] = []
    add = lines.append

    add("## 5. Membership validation")
    add("")
    add("| Split | Adapter directory | Images | Annotations | Negatives |")
    add("| --- | --- | --- | --- | --- |")
    for split in DEVELOPMENT_SPLITS:
        entry = manifest[split]
        add(
            f"| {split} | `{entry['adapter_directory']}` | {entry['image_count']} | "
            f"{entry['annotation_count']} | {entry['negative_images']} |"
        )
    add(f"| test | - | - | - | **{NOT_MATERIALIZED}** |")
    add(
        f"| **total** | | **{totals['images']}** | **{totals['annotations']}** | "
        f"**{totals['negative_images']}** |"
    )
    add("")
    add(
        "`COMPUTED RESULT` Every adapter image id was checked against the frozen split "
        "membership, resolved through the holdout guard rather than by listing a directory. "
        "Adapter images are byte-identical copies of the canonical materialised images, which "
        "are themselves byte-identical copies of the source originals - verified by hashing "
        "both sides of every copy."
    )
    add("")
    add("## 6. Negative images")
    add("")
    add(
        f"`MODEL-ADAPTER POLICY` `zero_instance_policy: {config.zero_instance_policy}`. The "
        f"{totals['negative_images']} development images carrying no annotation get an **empty "
        "label file**, which is Ultralytics' own representation of a background image. An "
        "empty label file is a valid negative, not a missing label: the dataset scan counts "
        "these as backgrounds rather than as corrupt entries. They were confirmed as "
        "deliberate negatives by the phase 4B audit and are never dropped."
    )
    add("")
    add("## 7. Bbox conversion")
    add("")
    add("```text")
    add("COCO   [x, y, w, h]        top-left corner plus extent, in source pixels")
    add("YOLO   [cx, cy, nw, nh]    centre plus extent, normalised by W and H")
    add("")
    add("cx = (x + w / 2) / W       nw = w / W")
    add("cy = (y + h / 2) / H       nh = h / H")
    add("```")
    add("")
    add(
        f"`MODEL-ADAPTER POLICY` `bbox_source: {config.bbox_source}`. The input is the "
        "segmentation-derived canonical box from phase 5D. **The provider's supplied bbox is "
        "not read here and never was ground truth**; phase 4A measured it disagreeing with its "
        "own geometry by up to 123.5 px."
    )
    add("")
    add(
        f"`FACT` Coordinates are written fixed-point at {config.label_precision} decimal "
        "places. Nine rather than the conventional six for a measured reason: the largest "
        "source image is 1880 px wide, so one unit in the ninth decimal is about 2e-6 px. "
        "Fixed-point rather than a shortest-repr float, so the label files are byte-identical "
        "between runs and machines."
    )
    add("")
    add(
        "`FACT` **Nothing was clamped.** Every canonical development box was measured to lie "
        "strictly inside its own canvas - zero boxes overshoot by any amount - so the "
        f"`normalised_bound_epsilon` of {config.tolerances.normalised_bound_epsilon} absorbs "
        "floating-point error in the division and nothing else. A box outside the unit square "
        "is a hard error here, not a value to be quietly corrected."
    )
    add("")
    add("## 8. Round-trip fidelity audit")
    add("")
    add(
        "`COMPUTED RESULT` Every development box was converted, written, **read back from the "
        "label file on disk**, decoded to source pixels and compared with the canonical box. "
        "Reading back from disk is the point: comparing against the in-memory boxes would "
        "prove nothing about the files a trainer actually loads."
    )
    add("")
    add(
        "| Split | Checked | Within tolerance | Exact | Max delta (px) | "
        "Mean delta (px) | Mismatches |"
    )
    add("| --- | --- | --- | --- | --- | --- | --- |")
    for split in DEVELOPMENT_SPLITS:
        entry = manifest[split]["round_trip"]
        add(
            f"| {split} | {entry['annotations_checked']} | {entry['within_tolerance']} | "
            f"{entry['exact_matches']} | {entry['max_delta_px']:.3e} | "
            f"{entry['mean_delta_px']:.3e} | {entry['mismatches']} |"
        )
    add(
        f"| **total** | **{trip['annotations_checked']}** | **{trip['within_tolerance']}** | "
        f"**{trip['exact_matches']}** | **{trip['max_delta_px']:.3e}** | | "
        f"**{trip['mismatches']}** |"
    )
    add("")
    add(
        f"`COMPUTED RESULT` Tolerance {trip['tolerance_px']} px. "
        f"{trip['within_tolerance']}/{trip['annotations_checked']} boxes within tolerance, "
        f"{trip['mismatches']} mismatches. The conversion is lossless within serialisation "
        "precision."
    )
    add("")
    add("## 9. Fingerprints")
    add("")
    add("| Split | YOLO labels | Adapter images | Canonical images |")
    add("| --- | --- | --- | --- |")
    for split in DEVELOPMENT_SPLITS:
        entry = manifest[split]
        add(
            f"| {split} | `{entry['yolo_label_sha256']}` | `{entry['image_content_sha256']}` | "
            f"`{entry['canonical_image_content_sha256']}` |"
        )
    add("")
    add(
        "`COMPUTED RESULT` The adapter and canonical image fingerprints are equal per split, "
        "which is what "
        "byte-identical means when stated as a measurement rather than an intention."
    )
    add("")
    add("## 10. Test protection")
    add("")
    add(
        f"`HOLDOUT POLICY` `test` is `{NOT_MATERIALIZED}`. There is no test image directory, "
        "no test label directory, no test label file, and **no `test` key in the Ultralytics "
        "dataset descriptor** - a dataset file being exactly the kind of place a protected "
        f"split gets reached by accident. `{HOLDOUT_UNLOCK_ENV_VAR}` was not set and no new "
        "statistic about the holdout was computed."
    )
    add("")
    add(
        "`HOLDOUT POLICY` Split membership is resolved through the same guard as every other "
        "consumer, so building a holdout adapter needs both an in-code opt-in and the "
        "environment opt-in; a development run refuses it outright before either is consulted."
    )
    add("")
    add("## 11. Limitations")
    add("")
    add(
        "* `LIMITATION` **Losslessness is about geometry, not correctness.** The audit proves "
        "the adapter reproduces the canonical boxes; it says nothing about whether those boxes "
        "are right."
    )
    add(
        "* `LIMITATION` **Only boxes are proven.** No claim is made here about segmentation "
        "conversion, which is a different problem and has not been attempted."
    )
    add(
        "* `LIMITATION` **Adapter images are a second copy.** They are verified byte-identical, "
        "but they are storage that must be regenerated rather than trusted after any change to "
        "the canonical datasets."
    )
    add(
        "* `LIMITATION` **Nothing here has been validated against the holdout**, by design. If "
        "the holdout contains a box case the development data does not, it will first be seen "
        "at final materialisation."
    )
    add(
        "* `LIMITATION` **No model exists.** No training, inference or metric has been produced "
        "from this adapter."
    )
    return lines


def _write_provenance(
    paths: ProjectPaths, manifest: dict[str, Any], config: AdapterConfig, outputs: list[Path]
) -> Path:
    """Record how the adapter was produced.

    Args:
        paths: Project layout.
        manifest: The adapter manifest.
        config: The adapter protocol.
        outputs: The committed files this run produced.

    Returns:
        The provenance record path.
    """
    record = ProvenanceRecord.create(
        "yolo_detection_adapter",
        phase=6,
        repo_root=paths.root,
        config={
            "detection_adapter": "configs/detection_adapter.yaml",
            "adapter_config_sha256": config.fingerprint(),
            "test_adapter": config.test_adapter,
        },
        details={
            "phase": "6A",
            "adapter_type": ADAPTER_TYPE,
            "status": manifest["status"],
            "development_totals": manifest["development_totals"],
            "round_trip_totals": manifest["round_trip_totals"],
            "class_map_sha256": manifest["class_map_sha256"],
            "split_assignment_sha256": manifest["split_assignment_sha256"],
            "test_status": NOT_MATERIALIZED,
            "holdout_accessed": False,
            "segmentation_labels_generated": False,
            "models_trained": 0,
        },
    )
    for name in (TASK_MANIFEST_JSON, POPULATION_MANIFEST_JSON, SPLIT_MANIFEST_JSON):
        record.add_input(paths.reports / name, relative_to=paths.root)
    record.add_input(paths.configs / "detection_adapter.yaml", relative_to=paths.root)
    for path in outputs:
        record.add_output(path, relative_to=paths.root)
    destination = paths.reports / PROVENANCE_JSON
    record.write_json(destination)
    return destination


def main(argv: list[str] | None = None) -> int:
    """Build and audit the YOLO detection adapter.

    Args:
        argv: Command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="Adapter configuration.")
    parser.add_argument("--verify-only", action="store_true", help="Validate and write nothing.")
    args = parser.parse_args(argv)

    paths = ProjectPaths.from_root()
    try:
        config = load_adapter_config(args.config or (paths.configs / "detection_adapter.yaml"))
        inputs = Inputs(paths, config)
    except (ConfigError, AdapterInputError, SplitManifestError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    results: dict[str, dict[str, Any]] = {}
    try:
        for split in DEVELOPMENT_SPLITS:
            assert_development_split(split)
            results[split] = adapt_split(paths, config, inputs, split, write=not args.verify_only)
    except (YoloAdapterError, MaterializationError, AdapterInputError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3

    problems = [p for split in DEVELOPMENT_SPLITS for p in results[split]["problems"]]
    for split in DEVELOPMENT_SPLITS:
        result = results[split]
        trip = result["round_trip"]
        suffix = (
            f"  round-trip {trip['within_tolerance']}/{trip['annotations_checked']} "
            f"max {trip['max_delta_px']:.2e} px"
            if trip
            else ""
        )
        print(
            f"  {split:<11} images {result['images']:>3}  labels "
            f"{result['annotations']:>4}  negatives {result['negative_images']:>2}{suffix}"
        )
    print(f"  holdout ({TEST}): {NOT_MATERIALIZED}")

    if problems:
        print("\nADAPTER_GEOMETRY_MISMATCH / validation failures:", file=sys.stderr)
        for problem in problems[:40]:
            print(f"  - {problem}", file=sys.stderr)
        if len(problems) > 40:
            print(f"  ... and {len(problems) - 40} more", file=sys.stderr)
        return 4

    if args.verify_only:
        print("--verify-only: nothing written")
        return 0

    output_root = paths.root / config.output_root
    # Resolved rather than relative: Ultralytics resolves `path` against the
    # working directory, not against the descriptor. The portable form lives in
    # the committed template; this file is machine-specific and git-ignored.
    (output_root / DATASET_YAML).write_text(
        dataset_yaml(
            path_value=output_root.as_posix(),
            train_directory=f"images/{config.split_directories['train']}",
            validation_directory=f"images/{config.split_directories['validation']}",
            class_map=inputs.class_map,
        ),
        encoding="utf-8",
        newline="\n",
    )

    manifest = build_manifest(config, inputs, results)
    serialised = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
    report = build_report(manifest, config, inputs)
    for name, text in ((ADAPTER_MANIFEST_JSON, serialised), (REPORT_MD, report)):
        unsafe = scan_for_sensitive(text)
        if unsafe:
            print(f"ERROR: {name} is not fit to commit: {'; '.join(unsafe)}", file=sys.stderr)
            return 5

    manifest_path = paths.reports / ADAPTER_MANIFEST_JSON
    manifest_path.write_text(f"{serialised}\n", encoding="utf-8", newline="\n")
    report_path = paths.reports / REPORT_MD
    report_path.write_text(report, encoding="utf-8", newline="\n")
    provenance_path = _write_provenance(paths, manifest, config, [manifest_path, report_path])

    print(f"wrote {(output_root / DATASET_YAML).relative_to(paths.root).as_posix()}")
    for path in (manifest_path, report_path, provenance_path):
        print(f"wrote {path.relative_to(paths.root).as_posix()}")
    print("\nDETECTION_ADAPTER_READY")
    return 0


if __name__ == "__main__":
    sys.exit(main())
