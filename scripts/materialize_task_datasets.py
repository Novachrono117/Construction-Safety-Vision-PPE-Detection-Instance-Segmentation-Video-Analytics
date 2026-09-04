"""Materialise the canonical COCO detection and segmentation views of a split.

Phase 5D. Turns the membership frozen in phase 5C.2 into data a training
pipeline can read, and changes nothing about that data on the way: images are
copied byte-for-byte, annotations keep their canonical geometry, and the class
order is the frozen one.

By default it materialises **train and validation only**. The holdout is not a
development split; the code path that would materialise it is the same one used
here, and it refuses to run without both holdout opt-ins. Nothing about the
protected split is measured, inspected or written by a default run.

What this does not do: train, evaluate, run inference, resize an image, write a
YOLO label, or read the provider's rejected split.

Runs entirely offline. Requires:

* ``configs/task_materialization.yaml``
* ``reports/split_manifest.json``                    scripts/freeze_split.py
* ``reports/canonical_modeling_manifest.json``       scripts/build_modeling_population.py
* ``reports/canonical_modeling_population.csv``      scripts/build_modeling_population.py
* ``reports/canonical_annotation_actions.csv``       scripts/build_modeling_population.py
* ``data/interim/modeling_annotations.jsonl``        scripts/build_modeling_population.py
* ``data/external/source_images/``                   scripts/download_source_images.py

Writes (bulk data git-ignored, evidence committed):
    data/processed/canonical/images/{train,validation}/
    data/processed/canonical/annotations/{detection,segmentation}_{split}.coco.json
    reports/task_dataset_manifest.json
    reports/task_materialization_report.md
    reports/task_materialization.provenance.json

Usage:
    uv run python scripts/materialize_task_datasets.py
    uv run python scripts/materialize_task_datasets.py --verify-only
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image

from construction_safety_vision.config import ConfigError
from construction_safety_vision.data.canonical import scan_for_sensitive
from construction_safety_vision.data.coco_materialization import (
    DETECTION,
    SEGMENTATION,
    TASKS,
    CanonicalAnnotation,
    CanonicalImage,
    CocoMaterializationError,
    alignment_report,
    bbox_agreement,
    build_detection_coco,
    build_segmentation_coco,
    document_fingerprint,
    round_trip_geometry,
    validate_coco,
)
from construction_safety_vision.data.materialization import (
    DEVELOPMENT_SPLITS,
    CanonicalIds,
    ImageCopy,
    MaterializationConfig,
    MaterializationError,
    assert_development_split,
    build_canonical_ids,
    copy_image,
    digest,
    image_content_fingerprint,
    load_materialization_config,
    load_population,
    materialized_name,
    resolve_split_images,
)
from construction_safety_vision.data.population import ELIGIBLE, KEEP_ANNOTATION
from construction_safety_vision.data.population import MATERIALISE_RECTANGLE as MATERIALISE
from construction_safety_vision.data.split_freeze import (
    TEST,
    SplitManifestError,
    load_frozen_splits,
)
from construction_safety_vision.paths import ProjectPaths, long_path
from construction_safety_vision.provenance import ProvenanceRecord, git_commit
from construction_safety_vision.splits import HOLDOUT_UNLOCK_ENV_VAR

SOURCE_IMAGE_DIR = "source_images"
INTERIM_ANNOTATIONS = "modeling_annotations.jsonl"
POPULATION_CSV = "canonical_modeling_population.csv"
ACTIONS_CSV = "canonical_annotation_actions.csv"
POPULATION_MANIFEST_JSON = "canonical_modeling_manifest.json"
SPLIT_MANIFEST_JSON = "split_manifest.json"
DATASET_MANIFEST_JSON = "task_dataset_manifest.json"
REPORT_MD = "task_materialization_report.md"
PROVENANCE_JSON = "task_materialization.provenance.json"

ELIGIBLE_ANNOTATION_ACTIONS = frozenset({KEEP_ANNOTATION, MATERIALISE})
NOT_MATERIALIZED = "NOT_MATERIALIZED_PROTECTED_HOLDOUT"


class MaterializationInputError(RuntimeError):
    """Raised when a required input artifact is missing or unusable."""


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a committed CSV artifact.

    Args:
        path: File to read.

    Returns:
        The parsed rows.

    Raises:
        MaterializationInputError: If the file is absent.
    """
    if not path.is_file():
        msg = f"{path.name} not found; run the phase that produces it first"
        raise MaterializationInputError(msg)
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    """Read a committed JSON artifact.

    Args:
        path: File to read.

    Returns:
        The parsed mapping.

    Raises:
        MaterializationInputError: If the file is absent or unparsable.
    """
    if not path.is_file():
        msg = f"{path.name} not found; run the phase that produces it first"
        raise MaterializationInputError(msg)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"{path.name} is not valid JSON ({exc})"
        raise MaterializationInputError(msg) from exc


def load_annotations(path: Path, wanted: set[str]) -> dict[str, list[CanonicalAnnotation]]:
    """Load the canonical annotations of the requested images only.

    The interim file holds the whole modelling population in one stream. Each
    record's image id is checked **before** anything else is read from it, so an
    annotation belonging to a split this run is not materialising is discarded
    without its geometry ever being looked at.

    Args:
        path: ``data/interim/modeling_annotations.jsonl``.
        wanted: Source image ids this run is allowed to read.

    Returns:
        Canonical annotations keyed by source image id.

    Raises:
        MaterializationInputError: If the file is absent.
    """
    if not path.is_file():
        msg = f"{path.name} not found; run scripts/build_modeling_population.py first"
        raise MaterializationInputError(msg)
    by_image: dict[str, list[CanonicalAnnotation]] = {}
    with Path(long_path(path)).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            image_id = record["source_image_id"]
            if image_id not in wanted:
                continue
            by_image.setdefault(image_id, []).append(
                CanonicalAnnotation(
                    source_image_id=image_id,
                    annotation_id=record["annotation_id"],
                    label=record["label"],
                    class_index=int(record["class_index"]),
                    segmentation=record["segmentation"],
                    area=float(record["area"]),
                    canonical_bbox=list(record["bbox"]),
                    geometry_origin=record["geometry_origin"],
                )
            )
    return by_image


class Inputs:
    """Every committed artifact the materialisation reads, parsed once.

    Attributes:
        splits: The frozen split.
        class_map: The frozen class map.
        population: The phase 5B population manifest.
        dimensions: Declared ``(width, height)`` keyed by source image id.
        groups: Group id keyed by source image id.
        ids: The deterministic global COCO id assignment.
        expected_annotations: Canonical annotation count keyed by source image id.
    """

    def __init__(self, paths: ProjectPaths, config: MaterializationConfig) -> None:
        """Load and parse every input.

        Args:
            paths: Project layout.
            config: The materialisation protocol.

        Raises:
            MaterializationInputError: If an artifact is missing or malformed.
            SplitManifestError: If the frozen split does not verify.
        """
        self.splits = load_frozen_splits(paths.reports / SPLIT_MANIFEST_JSON)
        self.population = read_json(paths.reports / POPULATION_MANIFEST_JSON)
        self.class_map: dict[str, int] = dict(self.population["class_map"])

        population = load_population(
            read_csv(paths.reports / POPULATION_CSV), eligible_status=ELIGIBLE
        )
        self.dimensions = population.dimensions
        self.groups = population.groups
        eligible = list(population.eligible)

        keys: list[tuple[str, str]] = []
        self.expected_annotations: dict[str, int] = dict.fromkeys(eligible, 0)
        for row in read_csv(paths.reports / ACTIONS_CSV):
            if row["action"] not in ELIGIBLE_ANNOTATION_ACTIONS:
                continue
            keys.append((row["source_image_id"], row["annotation_id"]))
            self.expected_annotations[row["source_image_id"]] += 1

        self.ids: CanonicalIds = build_canonical_ids(eligible, keys)
        _ = config


def materialize_split(
    paths: ProjectPaths,
    config: MaterializationConfig,
    inputs: Inputs,
    split: str,
    *,
    allow_test: bool = False,
    env: dict[str, str] | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Materialise one split's images and both of its task representations.

    This is the only materialisation path in the project. The holdout, when the
    final-evaluation phase eventually materialises it, runs through this exact
    function with this exact configuration - there is no split-specific branch
    that could give it different treatment.

    Args:
        paths: Project layout.
        config: The materialisation protocol.
        inputs: The parsed inputs.
        split: Split being materialised.
        allow_test: Explicit in-code opt-in for the holdout.
        env: Environment mapping to read. Defaults to ``os.environ``.
        write: Whether to write anything. ``False`` verifies and writes nothing.

    Returns:
        The split's materialisation record.

    Raises:
        HoldoutViolationError: If the holdout is requested without both opt-ins.
        MaterializationError: If an image cannot be copied byte-for-byte.
        CocoMaterializationError: If a task view cannot be built.
    """
    image_ids = resolve_split_images(
        inputs.splits,
        split,
        purpose=f"materialising the canonical {split} task datasets",
        allow_test=allow_test,
        env=env,
    )
    wanted = set(image_ids)
    annotations_by_image = load_annotations(paths.data_interim / INTERIM_ANNOTATIONS, wanted)

    output_root = paths.root / config.output_root
    image_directory = output_root / "images" / split
    # A dry run has written nothing, so the file-existence check resolves against
    # the canonical originals instead. Both use the same names by construction.
    source_directory = paths.data_external / SOURCE_IMAGE_DIR
    annotation_directory = output_root / "annotations"

    copies: list[ImageCopy] = []
    images: list[CanonicalImage] = []
    dimension_problems: list[str] = []
    for image_id in image_ids:
        source = source_directory / f"{image_id}.jpg"
        name = materialized_name(image_id, source)
        destination = image_directory / name
        if write:
            copies.append(copy_image(source, destination, source_image_id=image_id))
        declared_width, declared_height = inputs.dimensions[image_id]
        probe = destination if write else source
        with Image.open(long_path(probe)) as handle:
            decoded_width, decoded_height = handle.size
        if (decoded_width, decoded_height) != (declared_width, declared_height):
            dimension_problems.append(
                f"{image_id}: decoded {decoded_width}x{decoded_height} but the population "
                f"declares {declared_width}x{declared_height}"
            )
        images.append(
            CanonicalImage(
                source_image_id=image_id,
                file_name=name,
                width=declared_width,
                height=declared_height,
                group_id=inputs.groups[image_id],
            )
        )

    annotations = [
        annotation
        for image_id in image_ids
        for annotation in annotations_by_image.get(image_id, [])
    ]
    expected_annotation_count = sum(inputs.expected_annotations[i] for i in image_ids)
    split_sha = inputs.splits.split_assignment_sha256
    documents = {
        DETECTION: build_detection_coco(
            images,
            annotations,
            ids=inputs.ids,
            class_map=inputs.class_map,
            split=split,
            config_fingerprint=config.fingerprint(),
            split_sha256=split_sha,
        ),
        SEGMENTATION: build_segmentation_coco(
            images,
            annotations,
            ids=inputs.ids,
            class_map=inputs.class_map,
            split=split,
            config_fingerprint=config.fingerprint(),
            split_sha256=split_sha,
        ),
    }

    written: dict[str, Path] = {}
    if write:
        annotation_directory.mkdir(parents=True, exist_ok=True)
        for task, document in documents.items():
            path = annotation_directory / f"{task}_{split}.coco.json"
            serialised = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True)
            path.write_text(f"{serialised}\n", encoding="utf-8", newline="\n")
            written[task] = path

    problems = list(dimension_problems)
    for task, document in documents.items():
        problems.extend(
            f"{task}/{split}: {problem}"
            for problem in validate_coco(
                document,
                task=task,
                split=split,
                image_directory=image_directory if write else source_directory,
                expected_images=len(image_ids),
                expected_annotations=expected_annotation_count,
                class_map=inputs.class_map,
                tolerances=config.tolerances,
            )
        )

    alignment = alignment_report(documents[DETECTION], documents[SEGMENTATION], split)
    problems.extend(f"alignment/{split}: {problem}" for problem in alignment["problems"])

    agreement = bbox_agreement(annotations, config.tolerances.bbox_agreement_px)
    problems.extend(f"bbox/{split}: {problem}" for problem in agreement["problems"])

    round_trip = None
    if write:
        emitted = json.loads(written[SEGMENTATION].read_text(encoding="utf-8"))
        round_trip = round_trip_geometry(
            annotations,
            {record["id"]: record for record in emitted["annotations"]},
            inputs.ids,
            {image.source_image_id: (image.width, image.height) for image in images},
            config.tolerances,
        )
        problems.extend(f"geometry/{split}: {problem}" for problem in round_trip.mismatches)

    return {
        "split": split,
        "images": len(images),
        "annotations": len(annotations),
        "expected_annotations": expected_annotation_count,
        "zero_instance_images": sum(
            1 for image in images if not annotations_by_image.get(image.source_image_id)
        ),
        "instances_by_class": _class_counts(annotations, inputs.class_map),
        "image_copies": copies,
        "image_content_sha256": image_content_fingerprint(copies) if copies else "",
        "documents": documents,
        "written": dict(written),
        "document_sha256": {
            task: document_fingerprint(document) for task, document in documents.items()
        },
        "file_sha256": {
            task: digest(path.read_text(encoding="utf-8")) for task, path in written.items()
        },
        "alignment": alignment,
        "bbox_agreement": agreement,
        "round_trip": round_trip.as_dict() if round_trip is not None else None,
        "round_trip_object": round_trip,
        "problems": problems,
    }


def _class_counts(
    annotations: list[CanonicalAnnotation], class_map: dict[str, int]
) -> dict[str, int]:
    """Count instances per class.

    Args:
        annotations: The split's annotations.
        class_map: The frozen class map.

    Returns:
        Instance counts keyed by class name.
    """
    counts = dict.fromkeys(class_map, 0)
    for annotation in annotations:
        counts[annotation.label] += 1
    return counts


def build_dataset_manifest(
    config: MaterializationConfig,
    inputs: Inputs,
    results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Assemble the versioned development-dataset manifest.

    Carries no timestamp, no absolute path, no holdout identifier and no holdout
    content: the protected split appears only as a status.

    Args:
        config: The materialisation protocol.
        inputs: The parsed inputs.
        results: Per-split materialisation records.

    Returns:
        The manifest mapping.
    """
    fingerprints = inputs.population["fingerprints"]
    manifest: dict[str, Any] = {
        "schema_version": config.schema_version,
        "phase": "5D",
        "status": "DEVELOPMENT_DATASETS_MATERIALIZED",
        "canonical_split_manifest": config.canonical_split_manifest,
        "split_assignment_sha256": inputs.splits.split_assignment_sha256,
        "modeling_population_sha256": fingerprints["modeling_population_sha256"],
        "groups_sha256": fingerprints["groups_sha256"],
        "class_map_sha256": fingerprints["class_map_sha256"],
        "class_map": inputs.class_map,
        "materialization_config_sha256": config.fingerprint(),
        "materialization_config": config.as_dict(),
        "canonical_id_assignment_sha256": inputs.ids.fingerprint(),
        "canonical_detection_format": "COCO",
        "canonical_segmentation_format": "COCO_INSTANCE_SEGMENTATION",
        "model_specific_adapter": "NOT_YET_SELECTED",
        "output_root": config.output_root,
        "bulk_data_committed": False,
    }
    for split in DEVELOPMENT_SPLITS:
        result = results[split]
        manifest[split] = {
            "status": "MATERIALIZED",
            "image_count": result["images"],
            "annotation_count": result["annotations"],
            "zero_instance_images": result["zero_instance_images"],
            "instances_by_class": result["instances_by_class"],
            "image_content_sha256": result["image_content_sha256"],
            "detection_document_sha256": result["document_sha256"][DETECTION],
            "segmentation_document_sha256": result["document_sha256"][SEGMENTATION],
            "detection_file_sha256": result["file_sha256"][DETECTION],
            "segmentation_file_sha256": result["file_sha256"][SEGMENTATION],
            "alignment": {
                key: value for key, value in result["alignment"].items() if key != "problems"
            },
            "geometry_round_trip": result["round_trip"],
            "bbox_agreement": {
                key: value for key, value in result["bbox_agreement"].items() if key != "problems"
            },
        }
    manifest[TEST] = {
        "status": NOT_MATERIALIZED,
        "reason": (
            "the protected holdout is not materialised while the models are unfrozen. No "
            "image, annotation, identifier or statistic of it is recorded here. The "
            "final-evaluation phase materialises it through this same code path, with both "
            "holdout opt-ins present."
        ),
    }
    manifest["development_totals"] = {
        "images": sum(results[s]["images"] for s in DEVELOPMENT_SPLITS),
        "annotations": sum(results[s]["annotations"] for s in DEVELOPMENT_SPLITS),
    }
    manifest["git_commit"] = git_commit()
    return manifest


def _percentage(part: int, whole: int) -> str:
    """Format a share as a percentage string.

    Args:
        part: Numerator.
        whole: Denominator.

    Returns:
        The formatted share, or ``"-"`` when the denominator is zero.
    """
    return "-" if whole <= 0 else f"{100.0 * part / whole:.1f}%"


def build_report(
    manifest: dict[str, Any],
    config: MaterializationConfig,
    inputs: Inputs,
    results: dict[str, dict[str, Any]],
) -> str:
    """Write the task-materialisation report.

    Args:
        manifest: The dataset manifest that was written.
        config: The materialisation protocol.
        inputs: The parsed inputs.
        results: Per-split materialisation records.

    Returns:
        The report as Markdown.
    """
    lines: list[str] = []
    add = lines.append
    totals = manifest["development_totals"]
    commit = manifest["git_commit"] or "unavailable"

    add("# Canonical Task Dataset Materialisation - Phase 5D")
    add("")
    add(f"Phase: 5D · Commit: `{commit}` · Status: **{manifest['status']}**")
    add("")
    add(
        "This phase turned the membership frozen in phase 5C.2 into two task views of the "
        "same data. The property it claims is that it **added nothing**: the same source "
        "bytes, the same annotations, the same class order. Anything it changed about the "
        "data would be a defect, so every such property below is measured rather than "
        "asserted."
    )
    add("")
    add(
        "**Only `train` and `validation` were materialised.** The holdout was not read, not "
        "measured, not copied and not counted from its labels; the counts quoted for it "
        "anywhere in this repository are the phase 5C.2 protocol facts, not new findings."
    )
    add("")
    add(
        "Claims are labelled `FACT` (measured from a committed artifact), `COMPUTED RESULT` "
        "(produced by this run), `CANONICALIZATION POLICY` (a decision about the ground "
        "truth), `HOLDOUT POLICY` and `FUTURE MODEL ADAPTER REQUIREMENT`."
    )
    add("")
    add("## 1. Canonical input state")
    add("")
    add(
        "`FACT` The annotation source is the phase 5A canonical snapshot "
        "(`CURRENT_COMPLETE_GEOMETRY`) as filtered by the phase 5B modelling population: 433 "
        "eligible images and 2031 retained annotations in **original image coordinates**. The "
        "v4 export is not used; its geometry is expressed after a stretch resize to 640x640."
    )
    add("")
    add("| Fingerprint | Value |")
    add("| --- | --- |")
    add(f"| `modeling_population_sha256` | `{manifest['modeling_population_sha256']}` |")
    add(f"| `groups_sha256` | `{manifest['groups_sha256']}` |")
    add(f"| `class_map_sha256` | `{manifest['class_map_sha256']}` |")
    add(f"| `split_assignment_sha256` | `{manifest['split_assignment_sha256']}` |")
    add(f"| `materialization_config_sha256` | `{manifest['materialization_config_sha256']}` |")
    add(f"| `canonical_id_assignment_sha256` | `{manifest['canonical_id_assignment_sha256']}` |")
    add("")
    add("## 2. Frozen split reference")
    add("")
    add(
        f"`FACT` Membership comes from `{config.canonical_split_manifest}` and from nothing "
        "else. The provider's split is read nowhere - not as an input, an initialisation or a "
        "destination - and it is not present in any artifact this phase consults."
    )
    add("")
    add("## 3. Why only train and validation")
    add("")
    add(
        "`HOLDOUT POLICY` The holdout is protected data, and materialising it now would create "
        "an opportunity to look at it that the protocol does not permit. The code path that "
        "will materialise it exists and is exercised by tests against synthetic fixtures, but "
        "it requires `allow_test=True` **and** "
        f"`{HOLDOUT_UNLOCK_ENV_VAR}=1`, and it was not invoked. The configuration records this "
        f"as `test_materialization: {config.test_materialization}`."
    )
    add("")
    add(
        "`HOLDOUT POLICY` The final-evaluation phase runs the **same function with the same "
        "configuration**. There is no split-specific branch, so the holdout cannot receive "
        "different preprocessing than the data the models were developed on - which is the "
        "point of materialising it late rather than differently."
    )
    add("")
    add("## 4. Canonical task formats")
    add("")
    add(f"* `canonical_detection_format`: **{manifest['canonical_detection_format']}**")
    add(f"* `canonical_segmentation_format`: **{manifest['canonical_segmentation_format']}**")
    add(f"* `model_specific_adapter`: **{manifest['model_specific_adapter']}**")
    add("")
    add(
        "`CANONICALIZATION POLICY` COCO for both, and **no YOLO labels yet**. The canonical "
        "annotation state holds polygon geometry *and* compressed RLE; COCO carries both "
        "natively, while YOLO's segmentation format carries polygons only. Writing YOLO now "
        "would mean rasterising and re-polygonising every RLE mask - an approximation applied "
        "to the ground truth, before a model has even been chosen. A model-specific "
        "representation must not redefine ground truth."
    )
    add("")
    add("## 5. Image copy policy")
    add("")
    add(
        f"`CANONICALIZATION POLICY` `image_copy_mode: {config.image_copy_mode}`. No resize, no "
        "crop, no re-encode, no EXIF-driven rotation, no colour-space change, no augmentation. "
        "Images are transferred with binary copy semantics and never routed through an image "
        "library, so no decoder can quietly re-encode them."
    )
    add("")
    add(
        f"`CANONICALIZATION POLICY` `image_naming: {config.image_naming}`. The provider's "
        "filenames are descriptive, long enough to hit the Windows path limit on this machine, "
        "and not guaranteed unique; the canonical id is short, unique and already the join key."
    )
    add("")
    lines.extend(_report_middle(manifest, config, inputs, results, totals))
    lines.extend(_report_tail(manifest, config, results))
    return "\n".join(lines) + "\n"


def _report_middle(
    manifest: dict[str, Any],
    config: MaterializationConfig,
    inputs: Inputs,
    results: dict[str, dict[str, Any]],
    totals: dict[str, int],
) -> list[str]:
    """Write the report sections covering the class map through the counts.

    Args:
        manifest: The dataset manifest.
        config: The materialisation protocol.
        inputs: The parsed inputs.
        results: Per-split materialisation records.
        totals: Development totals.

    Returns:
        The report lines.
    """
    lines: list[str] = []
    add = lines.append

    add("## 6. Class map")
    add("")
    add("| Class | COCO `category_id` |")
    add("| --- | --- |")
    for name, index in sorted(inputs.class_map.items(), key=lambda item: item[1]):
        add(f"| `{name}` | {index} |")
    add("")
    add(
        f"`CANONICALIZATION POLICY` `category_id_policy: {config.category_id_policy}` - the "
        "COCO category id **is** the frozen class index. A second numbering would create two "
        "orderings to keep in step, and `class_map_sha256` would stop describing what the "
        "datasets contain. The order was not re-derived from dictionary iteration, provider "
        "category order or runtime sorting."
    )
    add("")
    add(
        "`FACT` The provider's placeholder category `object` carries no annotation and is "
        "absent from both task views; its exclusion does not shift any other class index."
    )
    add("")
    add("## 7. Segmentation geometry preservation")
    add("")
    add(
        f"`CANONICALIZATION POLICY` `segmentation_geometry_policy: "
        f"{config.segmentation_geometry_policy}`. A polygon stays a polygon; a compressed RLE "
        'stays a compressed RLE, emitted as `{"size": [h, w], "counts": ...}`. No RLE is '
        "converted to a polygon anywhere in this phase."
    )
    add("")
    add(
        "`FACT` Two annotations on image `OQJwjQoYsf1KUgr9G0V8` carry "
        "`geometry_origin = SYNTHETIC_FROM_PROVIDER_BBOX`: they had no provider segmentation "
        "and phase 5B materialised them as four-corner rectangles clipped to the canvas. They "
        "are emitted as those exact rectangles and flagged in the emitted records. **They are "
        "not human-drawn masks** and must never be reported as such."
    )
    add("")
    add("## 8. Detection bbox derivation")
    add("")
    add(
        f"`CANONICALIZATION POLICY` `detection_bbox_source: {config.detection_bbox_source}`. "
        "Every detection box is computed from the canonical segmentation by the geometry "
        "implementation audited in phase 4A, which handles polygons, compressed RLE and the "
        "synthetic rectangles. **The provider's supplied bbox is not used as ground truth "
        "anywhere**: phase 4A measured it disagreeing with its own geometry by up to 123.5 px "
        "inside the v4 export."
    )
    add("")
    add(
        "`COMPUTED RESULT` As a corroboration, each derived box was compared against the box "
        "phase 5A measured independently from the same geometry:"
    )
    add("")
    add("| Split | Annotations compared | Max delta (px) | Tolerance (px) | Within tolerance |")
    add("| --- | --- | --- | --- | --- |")
    for split in DEVELOPMENT_SPLITS:
        agreement = results[split]["bbox_agreement"]
        add(
            f"| {split} | {agreement['annotations_compared']} | {agreement['max_delta_px']} | "
            f"{agreement['tolerance_px']} | {agreement['within_tolerance']} |"
        )
    add("")
    add(
        "`CANONICALIZATION POLICY` The detection view keeps the **segmentation-derived "
        "`area`** rather than recomputing it from the box. That is COCO's own convention - the "
        "reference dataset's detection annotations carry mask area - and it is what keeps the "
        "small/medium/large size buckets identical between the two tasks. A `bbox_area` field "
        "is emitted alongside for anything that wants the rectangle's area explicitly."
    )
    add("")
    add("## 9. Output counts")
    add("")
    add("| Split | Images | Annotations | Zero-instance images | Status |")
    add("| --- | --- | --- | --- | --- |")
    for split in DEVELOPMENT_SPLITS:
        result = results[split]
        add(
            f"| {split} | {result['images']} | {result['annotations']} | "
            f"{result['zero_instance_images']} | MATERIALIZED |"
        )
    add(f"| test | - | - | - | **{NOT_MATERIALIZED}** |")
    add(
        f"| **development total** | **{totals['images']}** | **{totals['annotations']}** | "
        f"**{sum(results[s]['zero_instance_images'] for s in DEVELOPMENT_SPLITS)}** | |"
    )
    add("")
    add(
        f"`COMPUTED RESULT` {totals['images']} images and {totals['annotations']} annotations, "
        "derived from the frozen manifest and verified against the canonical population - not "
        "copied from a brief."
    )
    add("")
    add("`COMPUTED RESULT` Instances per class in the materialised development data:")
    add("")
    add("| Class | train | validation | development total |")
    add("| --- | --- | --- | --- |")
    for name in sorted(inputs.class_map, key=lambda n: inputs.class_map[n]):
        train_count = results["train"]["instances_by_class"][name]
        validation_count = results["validation"]["instances_by_class"][name]
        add(f"| `{name}` | {train_count} | {validation_count} | {train_count + validation_count} |")
    add("")
    add(
        "`FACT` Zero-instance images are **retained as image records with no annotations**. "
        "The phase 4B audit confirmed them as deliberate negatives rather than unlabelled "
        "images, and a task view that silently dropped them would change the dataset."
    )
    add("")
    add("## 10. Cross-task alignment")
    add("")
    add(
        "| Split | Image ids match | Annotation ids match | Categories match | "
        "Boxes match | Aligned |"
    )
    add("| --- | --- | --- | --- | --- | --- |")
    for split in DEVELOPMENT_SPLITS:
        alignment = results[split]["alignment"]
        add(
            f"| {split} | {alignment['image_ids_match']} | {alignment['annotation_ids_match']} | "
            f"{alignment['categories_match']} | {alignment['boxes_match']} | "
            f"{alignment['aligned']} |"
        )
    add("")
    add(
        "`COMPUTED RESULT` For each development split the two views hold the same COCO image "
        "ids, the same source image ids, the same annotation ids, the same categories, the "
        "same class per annotation and the same boxes. No image or annotation exists in one "
        "view and not the other. The only intended difference is that the segmentation view "
        "carries mask geometry and the detection view does not."
    )
    add("")
    add("## 11. Geometry round-trip validation")
    add("")
    add(
        "`COMPUTED RESULT` Preservation is claimed, so it is measured. The emitted "
        "segmentation file is **read back from disk** and compared against the canonical "
        "state, which tests the serialisation as well as the construction. RLE is compared by "
        "**decoded mask**, not by comparing `counts` strings: identical strings would prove "
        "only that a copy happened, whereas decoding both sides proves the file describes the "
        "same pixels."
    )
    add("")
    add(
        "| Split | Polygons | RLE | Synthetic rectangles | Matches | Mismatches | "
        "Max polygon delta (px) |"
    )
    add("| --- | --- | --- | --- | --- | --- | --- |")
    for split in DEVELOPMENT_SPLITS:
        trip = results[split]["round_trip"]
        add(
            f"| {split} | {trip['polygon_annotations_checked']} "
            f"| {trip['rle_annotations_checked']} "
            f"| {trip['synthetic_rectangles_checked']} | {trip['matches']} | "
            f"{trip['mismatches']} | {trip['max_polygon_coordinate_delta_px']} |"
        )
    add("")
    return lines


def _report_tail(
    manifest: dict[str, Any], config: MaterializationConfig, results: dict[str, dict[str, Any]]
) -> list[str]:
    """Write the report sections from the fingerprints onward.

    Args:
        manifest: The dataset manifest.
        config: The materialisation protocol.
        results: Per-split materialisation records.

    Returns:
        The report lines.
    """
    lines: list[str] = []
    add = lines.append

    add("## 12. Fingerprints")
    add("")
    add("| Split | Task | Document SHA-256 | Image content SHA-256 |")
    add("| --- | --- | --- | --- |")
    for split in DEVELOPMENT_SPLITS:
        result = results[split]
        for task in TASKS:
            content = result["image_content_sha256"] if task == TASKS[0] else ""
            add(f"| {split} | {task} | `{result['document_sha256'][task]}` | `{content}` |")
    add("")
    add(
        "`FACT` The document fingerprint covers the categories, the image table and the "
        "annotations - the ground truth - and excludes the `info` block, which describes the "
        "protocol rather than the data. The image-content fingerprint covers the bytes of "
        "every materialised image, so it moves when an image changes and not when a directory "
        "is re-listed."
    )
    add("")
    add("## 13. Reproducibility")
    add("")
    add(
        "`FACT` Materialisation is a pure function of the canonical population, the frozen "
        "split manifest, the frozen class map, "
        "`configs/task_materialization.yaml` and this code. No timestamp is written into any "
        "emitted file, COCO ids come from a sorted canonical ordering rather than filesystem "
        "traversal, and JSON is emitted with sorted keys, so two runs produce byte-identical "
        "output. Only the provenance record's `created_at` differs between runs, which is what "
        "a run record is for."
    )
    add("")
    add(
        "`FACT` COCO numeric ids are assigned over the **whole modelling population**, not per "
        "split. That is what lets the holdout be materialised later, by this same code, "
        "without renumbering anything that already exists, and it is why an image carries one "
        "id in both task views."
    )
    add("")
    add("## 14. Protected-test policy")
    add("")
    add(
        f"`HOLDOUT POLICY` `test` is `{NOT_MATERIALIZED}`. This phase created no test image "
        "directory, no test COCO file, no test contact sheet, no test class diagnostic and no "
        "test geometry statistic. It computed nothing new about the holdout at all: the "
        f"manifest records only a status and a reason. `{HOLDOUT_UNLOCK_ENV_VAR}` was not set."
    )
    add("")
    add(
        "`HOLDOUT POLICY` Reading the holdout still requires both opt-ins, and the "
        "materialisation path routes through the same guard as every other consumer. Tests "
        "exercise that path against synthetic fixtures only."
    )
    add("")
    add("## 15. Future model-adapter requirement")
    add("")
    add(
        f"`FUTURE MODEL ADAPTER REQUIREMENT` `model_specific_adapter: "
        f"{manifest['model_specific_adapter']}`. If a YOLO segmentation stack is chosen later, "
        "then **before any training run**, the adapter must:"
    )
    add("")
    add("1. convert the canonical masks to the model-required polygon format;")
    add("2. rasterise the converted polygons;")
    add("3. compare the rasterised result against the canonical masks;")
    add("4. report per-instance mask IoU and area error;")
    add("5. identify disconnected-component and hole cases, which polygons cannot express;")
    add("6. reject or reconsider the adapter if the geometry loss is material.")
    add("")
    add(
        "That audit is a phase 8 / model-adapter concern and **was not performed here**. No "
        "conversion was attempted, so this phase makes no claim about how lossy it would be."
    )
    add("")
    add("## 16. Limitations")
    add("")
    add(
        "* `LIMITATION` **Preservation is verified, fitness is not.** This phase proves the "
        "emitted datasets carry the canonical geometry unchanged. It says nothing about "
        "whether that geometry is correct, or whether the classes are separable."
    )
    add(
        "* `LIMITATION` **Nothing is validated against the holdout.** No property measured "
        "here has been checked on the protected split, by design. If the holdout contains a "
        "geometry case the development data does not, it will first be seen at final "
        "materialisation."
    )
    add(
        "* `LIMITATION` **Round-trip equality is not accuracy.** Decoded masks matching proves "
        "the emitted file describes the same pixels as the canonical state; it does not prove "
        "those pixels outline the object correctly."
    )
    add(
        "* `LIMITATION` **The category id starts at 0.** COCO permits it and the frozen class "
        "map requires it, but some tooling reserves 0 for background. A model adapter may need "
        "to re-index, and if it does, that re-indexing must be recorded rather than done "
        "silently."
    )
    add(
        f"* `LIMITATION` **Bulk data is not committed.** `{config.output_root}` is git-ignored. "
        "What is committed is the manifest and this report, which together make the datasets "
        "re-derivable; the data itself is reproduced by re-running the command."
    )
    add(
        "* `LIMITATION` **No model exists.** No training, no inference and no metric has been "
        "produced, on any split."
    )
    add("")
    add("## 17. What this phase did not do")
    add("")
    add(
        "No model was trained and no framework for training one was installed. No inference "
        "was run. No YOLO detection or segmentation label was written. No image was resized, "
        "cropped, re-encoded or augmented. No real holdout image or annotation was "
        "materialised, read or measured, and no new statistic about the holdout was computed. "
        "Phase 6 has not been started."
    )
    return lines


def _write_provenance(
    paths: ProjectPaths,
    manifest: dict[str, Any],
    config: MaterializationConfig,
    outputs: list[Path],
) -> Path:
    """Record how the materialisation was produced.

    Args:
        paths: Project layout.
        manifest: The dataset manifest.
        config: The materialisation protocol.
        outputs: The committed files this run produced.

    Returns:
        The provenance record path.
    """
    record = ProvenanceRecord.create(
        "task_dataset_materialization",
        phase=5,
        repo_root=paths.root,
        config={
            "task_materialization": "configs/task_materialization.yaml",
            "materialization_config_sha256": config.fingerprint(),
            "development_splits": list(config.development_splits),
            "test_materialization": config.test_materialization,
        },
        details={
            "phase": "5D",
            "status": manifest["status"],
            "split_assignment_sha256": manifest["split_assignment_sha256"],
            "class_map_sha256": manifest["class_map_sha256"],
            "development_totals": manifest["development_totals"],
            "canonical_detection_format": manifest["canonical_detection_format"],
            "canonical_segmentation_format": manifest["canonical_segmentation_format"],
            "model_specific_adapter": manifest["model_specific_adapter"],
            "test_status": NOT_MATERIALIZED,
            "holdout_accessed": False,
            "models_trained": 0,
            "yolo_labels_written": 0,
        },
    )
    for name in (
        SPLIT_MANIFEST_JSON,
        POPULATION_MANIFEST_JSON,
        POPULATION_CSV,
        ACTIONS_CSV,
    ):
        record.add_input(paths.reports / name, relative_to=paths.root)
    record.add_input(paths.configs / "task_materialization.yaml", relative_to=paths.root)
    for path in outputs:
        record.add_output(path, relative_to=paths.root)
    destination = paths.reports / PROVENANCE_JSON
    record.write_json(destination)
    return destination


def main(argv: list[str] | None = None) -> int:
    """Materialise the canonical development task datasets.

    Args:
        argv: Command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="Materialisation config.")
    parser.add_argument("--verify-only", action="store_true", help="Validate and write nothing.")
    args = parser.parse_args(argv)

    paths = ProjectPaths.from_root()
    try:
        config = load_materialization_config(
            args.config or (paths.configs / "task_materialization.yaml")
        )
        inputs = Inputs(paths, config)
    except (ConfigError, MaterializationInputError, SplitManifestError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    results: dict[str, dict[str, Any]] = {}
    try:
        for split in config.development_splits:
            assert_development_split(split)
            results[split] = materialize_split(
                paths, config, inputs, split, write=not args.verify_only
            )
    except (MaterializationError, CocoMaterializationError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3

    problems = [
        problem for split in config.development_splits for problem in results[split]["problems"]
    ]
    for split in config.development_splits:
        result = results[split]
        copies = result["image_copies"]
        identical = sum(1 for copy in copies if copy.byte_identical)
        print(
            f"  {split:<11} images {result['images']:>3}  annotations "
            f"{result['annotations']:>4}  byte-identical {identical}/{len(copies)}"
        )
    totals_images = sum(results[s]["images"] for s in config.development_splits)
    totals_annotations = sum(results[s]["annotations"] for s in config.development_splits)
    print(f"  development total: {totals_images} images, {totals_annotations} annotations")
    print(f"  holdout ({TEST}): {NOT_MATERIALIZED}")

    if problems:
        print("\nGEOMETRY_MISMATCH / validation failures:", file=sys.stderr)
        for problem in problems[:40]:
            print(f"  - {problem}", file=sys.stderr)
        if len(problems) > 40:
            print(f"  ... and {len(problems) - 40} more", file=sys.stderr)
        return 4

    if args.verify_only:
        print("--verify-only: nothing written")
        return 0

    manifest = build_dataset_manifest(config, inputs, results)
    serialised = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
    report = build_report(manifest, config, inputs, results)
    for name, text in ((DATASET_MANIFEST_JSON, serialised), (REPORT_MD, report)):
        unsafe = scan_for_sensitive(text)
        if unsafe:
            print(f"ERROR: {name} is not fit to commit: {'; '.join(unsafe)}", file=sys.stderr)
            return 5

    manifest_path = paths.reports / DATASET_MANIFEST_JSON
    manifest_path.write_text(f"{serialised}\n", encoding="utf-8", newline="\n")
    report_path = paths.reports / REPORT_MD
    report_path.write_text(report, encoding="utf-8", newline="\n")
    provenance_path = _write_provenance(paths, manifest, config, [manifest_path, report_path])

    for split in config.development_splits:
        for task, path in results[split]["written"].items():
            print(f"wrote {path.relative_to(paths.root).as_posix()} ({task})")
    for path in (manifest_path, report_path, provenance_path):
        print(f"wrote {path.relative_to(paths.root).as_posix()}")
    print("\nTASK_DATASETS_READY")
    return 0


if __name__ == "__main__":
    sys.exit(main())
