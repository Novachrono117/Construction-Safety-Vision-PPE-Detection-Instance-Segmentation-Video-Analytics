"""Compare the frozen detector and segmenter on validation.

Phase 10B. It runs controlled inference with both frozen models under the two
protocols phase 10A froze, scores their boxes against the same canonical ground
truth with the same external evaluator, and computes the frozen spatial
quantities and their box counterparts. It trains nothing, modifies neither
model, benchmarks no latency and never touches the holdout.

Four things here are deliberate.

**Precision parity is proved at runtime, not assumed from the config.** Before
any comparison number exists, both models are probed for their backend
precision flag, parameter dtype, input dtype and autocast state, and the phase
stops as ``PRECISION_PROTOCOL_MISMATCH`` if the two differ or either departs
from the frozen FP32 intent. A latency or accuracy comparison across two
precisions would measure the precision.

**The two inference protocols never mix.** AP inference runs at conf 0.001,
because average precision needs the low-scoring tail; operational inference
runs at conf 0.25, because the spatial analysis needs the model to commit. They
are separate passes and their outputs are never crossed.

**The segmenter's boxes are the segmenter's own.** Nothing here derives a box
from a mask. The recognition axis compares each model's actual predicted boxes
against common canonical ground truth.

**Masks are used on the original canvas or not at all.** A predicted mask whose
shape does not match its source image is not silently resized into agreement;
the instance is excluded and the exclusion is counted.

Requires:

* ``configs/detector_segmenter_comparison.yaml``               phase 10A
* ``reports/final_detector_manifest.json``                     phase 7D
* ``reports/final_segmenter_manifest.json``                    phase 8G
* ``data/processed/canonical/`` detection + images              phase 5D

Writes:
    reports/detector_segmenter_box_comparison.json
    reports/detector_segmenter_box_comparison.csv
    reports/detector_segmenter_spatial_comparison.json
    reports/detector_segmenter_validation_comparison.md
    reports/detector_segmenter_comparison_examples.csv
    reports/detector_segmenter_validation_comparison.provenance.json
    artifacts/comparison/                                      (git-ignored)

Usage:
    uv run python scripts/compare_detector_segmenter.py
    uv run python scripts/compare_detector_segmenter.py --verify-only
    uv run python scripts/compare_detector_segmenter.py --preflight-only
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from construction_safety_vision.canonical_evaluation import CanonicalEvaluationError
from construction_safety_vision.config import ConfigError
from construction_safety_vision.data.canonical import scan_for_sensitive
from construction_safety_vision.detection_freeze import (
    FinalDetectorError,
    load_final_detector,
)
from construction_safety_vision.detection_freeze import (
    load_checkpoint_path as detector_checkpoint,
)
from construction_safety_vision.detector_segmenter_analysis import (
    ASSOCIATION_DISAGREEMENT,
    FEATURE_SEMANTICS,
    GEOMETRY_ISOLATING,
    HELMET_CLASSES,
    METRIC_PRECISION,
    NO_BOX_PROXY,
    PERSON_CLASS,
    PIPELINE_LEVEL,
    PPE_CLASSES,
    RARE_CLASS,
    RARE_CLASS_STATUS,
    STATISTIC_NAMES,
    TAXONOMY_EXCEPTION,
    TAXONOMY_NON_EXHAUSTIVE,
    Instance,
    associate,
    association_category,
    box_containment,
    box_intersection,
    centroid_displacement,
    decompose_all_class_delta,
    describe,
    instance_area_pixels,
    mask_containment,
    mask_intersection,
    mask_to_box_fill_ratio,
    result_fingerprint,
    shape_extent,
    tally,
    validate_box_comparison,
    validate_spatial_comparison,
)
from construction_safety_vision.detector_segmenter_comparison import (
    ASSOCIATION_CATEGORIES,
    EVALUATION_SPLIT,
    HOLDOUT_STATUS,
    IOU_THRESHOLDS,
    MAX_DETS,
    PRECISION,
    ComparisonProtocolError,
    load_comparison_protocol,
    membership_fingerprint,
)
from construction_safety_vision.paths import ProjectPaths, long_path
from construction_safety_vision.provenance import ProvenanceRecord, git_commit, sha256_file
from construction_safety_vision.segmentation_freeze import (
    FinalSegmenterError,
    load_final_segmenter,
)
from construction_safety_vision.segmentation_freeze import (
    load_checkpoint_path as segmenter_checkpoint,
)
from construction_safety_vision.splits import HOLDOUT_UNLOCK_ENV_VAR, holdout_unlocked

CONFIG_YAML = "detector_segmenter_comparison.yaml"
PROTOCOL_JSON = "detector_segmenter_comparison_protocol.json"

BOX_JSON = "detector_segmenter_box_comparison.json"
BOX_CSV = "detector_segmenter_box_comparison.csv"
SPATIAL_JSON = "detector_segmenter_spatial_comparison.json"
REPORT_MD = "detector_segmenter_validation_comparison.md"
EXAMPLES_CSV = "detector_segmenter_comparison_examples.csv"
PROVENANCE_JSON = "detector_segmenter_validation_comparison.provenance.json"

RUNTIME_ROOT = "artifacts/comparison"

SCHEMA_VERSION = 1
PHASE = "10B"

COMPLETE = "DETECTOR_SEGMENTER_VALIDATION_COMPARISON_COMPLETE"
PRECISION_MISMATCH = "PRECISION_PROTOCOL_MISMATCH"
MODEL_IDENTITY_MISMATCH = "MODEL_IDENTITY_MISMATCH"
BOX_EVALUATION_FAILED = "CANONICAL_BOX_EVALUATION_FAILED"
SPATIAL_ANALYSIS_FAILED = "SPATIAL_ANALYSIS_FAILED"
PROTOCOL_VIOLATION = "PROTOCOL_VIOLATION"
BLOCKED = "BLOCKED"

EXPECTED_PROTOCOL = "d92a157637baf52f9a7248bf24257d5a177e8496b7727ffb5afc874ab96d1c4d"
DETECTOR = "D2"
SEGMENTER = "S1"

EXAMPLE_COUNT = 20
"""How many deterministic disagreement examples the manifest records."""

HOLDOUT_REASON = (
    "Phase 10B ran controlled inference with both frozen models on the frozen validation "
    "split only. The holdout was not read, materialised, adapted, counted, predicted on or "
    "inspected; no holdout identifier, image, prediction or statistic exists in any artifact "
    "this phase wrote."
)

HISTORICAL: tuple[str, ...] = (
    "reports/final_detector_manifest.json",
    "reports/detection_selection_report.md",
    "reports/final_segmenter_manifest.json",
    "reports/segmentation_selection_report.md",
    "reports/segmentation_experiment_comparison.csv",
    "reports/segmentation_experiment_results.json",
    "reports/detection_experiment_results.json",
    "configs/detector_segmenter_comparison.yaml",
    "reports/detector_segmenter_comparison_protocol.json",
    "reports/detector_segmenter_comparison_protocol.md",
    "reports/detector_segmenter_comparison_membership.csv",
    "reports/detector_segmenter_latency_membership.csv",
    "reports/segmentation_S0_result_manifest.json",
    "reports/segmentation_S0_canonical_evaluation.json",
    "reports/segmentation_S1_result_manifest.json",
    "reports/segmentation_S1_canonical_evaluation.json",
    "configs/segmentation_canonical_evaluation.yaml",
    "reports/split_manifest.json",
    "reports/task_dataset_manifest.json",
)
"""Every artifact this phase must leave byte-identical."""


class ComparisonError(RuntimeError):
    """Raised when a required input is missing or an invariant fails."""


class PrecisionError(ComparisonError):
    """Raised when the two models do not resolve to the frozen precision."""


class ModelIdentityError(ComparisonError):
    """Raised when a frozen model is not the one the protocol names."""


# --- helpers ------------------------------------------------------------------


def read_json(path: Path) -> dict[str, Any]:
    """Read a committed JSON artifact.

    Args:
        path: File to read.

    Returns:
        The parsed mapping.

    Raises:
        ComparisonError: If the file is missing or is not a JSON object.
    """
    if not path.is_file():
        msg = f"required input not found: {path.name}"
        raise ComparisonError(msg)
    data = json.loads(Path(long_path(path)).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        msg = f"{path.name} must contain a JSON object"
        raise ComparisonError(msg)
    return data


def write_json(path: Path, payload: Any) -> str:
    """Write a JSON artifact deterministically and return its digest.

    Args:
        path: Destination file.
        payload: JSON-serialisable content.

    Returns:
        The SHA-256 digest of the bytes written.
    """
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return sha256_file(path)


def historical_digests(paths: ProjectPaths) -> dict[str, str]:
    """Digest every artifact this phase must not change.

    Args:
        paths: Project layout.

    Returns:
        Digests keyed by repository-relative name.

    Raises:
        ComparisonError: If one is absent.
    """
    digests: dict[str, str] = {}
    for name in HISTORICAL:
        path = paths.root / name
        if not path.is_file():
            msg = f"historical artifact missing: {name}"
            raise ComparisonError(msg)
        digests[name] = sha256_file(path)
    return digests


def rounded(value: Any) -> Any:
    """Round a metric to the reported precision, passing non-numbers through.

    Args:
        value: A metric or a sentinel.

    Returns:
        The rounded value, or the input unchanged.
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return round(float(value), METRIC_PRECISION)
    return value


# --- the precision preflight ------------------------------------------------------


def precision_preflight(
    paths: ProjectPaths, checkpoints: Mapping[str, Path], protocol: Any, image: Path
) -> dict[str, Any]:
    """Prove both models resolve to the frozen precision, structurally.

    The config expresses FP32 through the installed framework's precision
    control, but a configuration value is an intention. This loads both models,
    runs one prediction each with a forward pre-hook attached, and records what
    the runtime actually did: the backend's precision flag, every parameter
    dtype, the dtype of the tensor that reached the network, and whether
    autocast was active.

    Args:
        paths: Project layout.
        checkpoints: Resolved checkpoint paths, keyed by experiment.
        protocol: The frozen comparison protocol.
        image: One validation image to probe with.

    Returns:
        The captured evidence for both models.

    Raises:
        PrecisionError: If the two differ, or either departs from FP32.
    """
    import torch
    from ultralytics import YOLO

    inference = protocol.ap_inference
    captured: dict[str, Any] = {}

    for experiment, path in checkpoints.items():
        model = YOLO(str(path))
        seen: dict[str, Any] = {}

        def hook(_module: Any, args: Any, _seen: dict[str, Any] = seen) -> None:
            tensor = args[0]
            _seen.setdefault("input_dtype", str(tensor.dtype))
            _seen.setdefault("autocast_enabled_during_forward", torch.is_autocast_enabled("cuda"))

        call = {
            "source": str(image),
            "imgsz": inference["imgsz"],
            "conf": inference["conf"],
            "iou": inference["iou"],
            "max_det": inference["max_det"],
            "augment": inference["augment"],
            inference["precision_argument"]: inference["precision_value"],
            "verbose": False,
            "save": False,
        }
        model.predict(**call)
        backend = model.predictor.model
        handle = backend.model.register_forward_pre_hook(hook)
        model.predict(**call)
        handle.remove()

        captured[experiment] = {
            "backend_fp16_flag": bool(getattr(backend, "fp16", False)),
            "resolved_precision_argument": inference["precision_argument"],
            "resolved_precision_value": getattr(
                model.predictor.args, inference["precision_argument"], None
            ),
            "parameter_dtypes": sorted({str(p.dtype) for p in backend.model.parameters()}),
            "float_buffer_dtypes": sorted(
                {str(b.dtype) for b in backend.model.buffers() if b.is_floating_point()}
            ),
            "input_dtype": seen.get("input_dtype"),
            "autocast_enabled_during_forward": seen.get("autocast_enabled_during_forward"),
            "quantization_config_present": getattr(backend.model, "qconfig", None) is not None,
            "device_type": str(next(backend.model.parameters()).device).split(":")[0],
        }
        del model
        torch.cuda.empty_cache()

    comparable = {experiment: dict(evidence) for experiment, evidence in captured.items()}
    if comparable[DETECTOR] != comparable[SEGMENTER]:
        differing = sorted(
            key
            for key in comparable[DETECTOR]
            if comparable[DETECTOR][key] != comparable[SEGMENTER][key]
        )
        msg = (
            f"{PRECISION_MISMATCH}: the two models resolve differently in {differing}. "
            "Comparing them would measure the precision as well as the models."
        )
        raise PrecisionError(msg)

    evidence = captured[DETECTOR]
    if evidence["backend_fp16_flag"] is not False:
        msg = f"{PRECISION_MISMATCH}: the backend enabled FP16 despite the frozen FP32 intent"
        raise PrecisionError(msg)
    if evidence["parameter_dtypes"] != ["torch.float32"]:
        msg = f"{PRECISION_MISMATCH}: parameters are {evidence['parameter_dtypes']}, not FP32 only"
        raise PrecisionError(msg)
    if evidence["input_dtype"] != "torch.float32":
        msg = f"{PRECISION_MISMATCH}: the input tensor is {evidence['input_dtype']}, not FP32"
        raise PrecisionError(msg)
    if evidence["autocast_enabled_during_forward"] is not False:
        msg = (
            f"{PRECISION_MISMATCH}: autocast was active during the forward pass, which would "
            "silently run parts of the network at a lower precision"
        )
        raise PrecisionError(msg)
    if evidence["quantization_config_present"]:
        msg = f"{PRECISION_MISMATCH}: a quantization configuration is attached to the model"
        raise PrecisionError(msg)

    return {
        "status": "EFFECTIVE_PRECISION_PARITY_VERIFIED",
        "frozen_intent": PRECISION,
        "identical_across_models": True,
        "evidence": captured,
        "probe_method": (
            "One prediction per model with a forward pre-hook on the network, capturing the "
            "dtype of the tensor that actually reached it and the autocast state at that "
            "moment. Configuration values were not trusted."
        ),
        "models_executed_for_probe": len(captured),
    }


# --- the canonical population ------------------------------------------------------


def load_population(paths: ProjectPaths, protocol: Any) -> dict[str, Any]:
    """Read the canonical validation document and verify its membership.

    Args:
        paths: Project layout.
        protocol: The frozen comparison protocol.

    Returns:
        The document, the image order and the verified fingerprint.

    Raises:
        ComparisonError: If the document or its membership has moved.
    """
    declared = protocol["population"]
    document = paths.root / str(declared["source"])
    digest = sha256_file(document)
    if digest != declared["source_sha256"]:
        msg = f"the canonical validation document changed: {digest}"
        raise ComparisonError(msg)

    payload = json.loads(Path(long_path(document)).read_text(encoding="utf-8"))
    entries = sorted(payload["images"], key=lambda entry: str(entry["file_name"]))
    image_ids = [Path(str(entry["file_name"])).stem for entry in entries]
    fingerprint = membership_fingerprint(image_ids)
    if fingerprint != declared["membership_sha256"]:
        msg = f"the validation membership changed: {fingerprint}"
        raise ComparisonError(msg)
    if len(image_ids) != int(declared["images"]):
        msg = f"expected {declared['images']} validation images, found {len(image_ids)}"
        raise ComparisonError(msg)

    images_root = paths.data_processed / "canonical" / "images" / EVALUATION_SPLIT
    files: dict[str, Path] = {}
    sizes: dict[str, tuple[int, int]] = {}
    ids: dict[str, int] = {}
    for entry in entries:
        stem = Path(str(entry["file_name"])).stem
        candidate = images_root / str(entry["file_name"])
        if not candidate.is_file():
            msg = f"validation image not on this machine: {entry['file_name']}"
            raise ComparisonError(msg)
        files[stem] = candidate
        sizes[stem] = (int(entry["height"]), int(entry["width"]))
        ids[stem] = int(entry["id"])

    return {
        "document": payload,
        "document_path": str(declared["source"]),
        "document_sha256": digest,
        "image_order": image_ids,
        "files": files,
        "sizes": sizes,
        "coco_ids": ids,
        "membership_sha256": fingerprint,
        "annotations": len(payload["annotations"]),
    }


# --- inference ------------------------------------------------------------------------


def predict(
    checkpoint: Path,
    population: Mapping[str, Any],
    settings: Mapping[str, Any],
    *,
    want_masks: bool,
) -> dict[str, Any]:
    """Run one model over the validation split under one frozen protocol.

    Both models see the same images in the same order, at the same settings.
    Masks, when requested, are taken on the original image canvas through the
    framework's native mask path; an instance whose mask does not match its
    source image is excluded rather than resized into agreement.

    Args:
        checkpoint: The frozen checkpoint to run.
        population: The verified validation population.
        settings: The frozen inference block to run under.
        want_masks: Whether to collect instance masks.

    Returns:
        Instances keyed by image id, plus counts and any exclusions.

    Raises:
        ComparisonError: If prediction fails on an image.
    """
    from ultralytics import YOLO

    model = YOLO(str(checkpoint))
    names: dict[int, str] = {}
    per_image: dict[str, list[Instance]] = {}
    total = 0
    masks_reconstructed = 0
    excluded: list[dict[str, Any]] = []

    for stem in population["image_order"]:
        path = population["files"][stem]
        height, width = population["sizes"][stem]
        call = {
            "source": str(path),
            "imgsz": settings["imgsz"],
            "conf": settings["conf"],
            "iou": settings["iou"],
            "max_det": settings["max_det"],
            "augment": settings["augment"],
            settings["precision_argument"]: settings["precision_value"],
            "verbose": False,
            "save": False,
            "stream": False,
        }
        if want_masks:
            call["retina_masks"] = True
        try:
            outputs = model.predict(**call)
        except Exception as exc:
            msg = f"prediction failed on {stem} ({exc})"
            raise ComparisonError(msg) from exc

        result = outputs[0]
        if not names:
            names = (
                dict(result.names)
                if isinstance(result.names, dict)
                else dict(enumerate(result.names))
            )
        instances: list[Instance] = []
        boxes = result.boxes
        count = 0 if boxes is None else len(boxes)
        masks = None
        if want_masks and result.masks is not None:
            masks = result.masks.data.cpu().numpy().astype(bool)

        for index in range(count):
            class_index = int(boxes.cls[index].item())
            score = float(boxes.conf[index].item())
            x1, y1, x2, y2 = (float(value) for value in boxes.xyxy[index].tolist())
            mask = None
            if want_masks:
                if masks is None or index >= len(masks):
                    excluded.append(
                        {"image_id": stem, "class": names[class_index], "reason": "NO_MASK"}
                    )
                    continue
                candidate = masks[index]
                if candidate.shape != (height, width):
                    excluded.append(
                        {
                            "image_id": stem,
                            "class": names[class_index],
                            "reason": "MASK_NOT_ON_ORIGINAL_CANVAS",
                            "mask_shape": list(candidate.shape),
                            "image_shape": [height, width],
                        }
                    )
                    continue
                mask = candidate
                masks_reconstructed += 1
            instances.append(
                Instance(
                    image_id=stem,
                    class_name=str(names[class_index]),
                    score=score,
                    box=(x1, y1, x2, y2),
                    mask=mask,
                )
            )
            total += 1
        per_image[stem] = instances

    del model
    return {
        "per_image": per_image,
        "total": total,
        "names": names,
        "masks_reconstructed": masks_reconstructed,
        "excluded": excluded,
        "images": len(population["image_order"]),
    }


# --- the common canonical box evaluator -------------------------------------------------


def evaluate_boxes(
    population: Mapping[str, Any],
    predictions: Mapping[str, Any],
    class_map: Mapping[str, int],
) -> dict[str, Any]:
    """Score one model's predicted boxes against the canonical boxes.

    Args:
        population: The verified validation population.
        predictions: One model's AP-protocol predictions.
        class_map: Class name to canonical category id.

    Returns:
        Global and per-class canonical box AP.

    Raises:
        CanonicalEvaluationError: If COCOeval departs from the frozen semantics.
    """
    import contextlib

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    detections: list[dict[str, Any]] = []
    for stem, instances in predictions["per_image"].items():
        image_id = population["coco_ids"][stem]
        for instance in instances:
            x1, y1, x2, y2 = instance.box
            detections.append(
                {
                    "image_id": image_id,
                    "category_id": int(class_map[instance.class_name]),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": instance.score,
                }
            )

    with contextlib.redirect_stdout(io.StringIO()):
        truth = COCO()
        truth.dataset = json.loads(json.dumps(population["document"]))
        truth.createIndex()
        predicted = truth.loadRes(list(detections)) if detections else None

    names = {index: name for name, index in class_map.items()}
    if predicted is None:
        return {
            "all_class_map50_95": 0.0,
            "all_class_map50": 0.0,
            "per_class": {name: {"AP@0.50:0.95": 0.0, "AP@0.50": 0.0} for name in names.values()},
            "detections_scored": 0,
        }

    with contextlib.redirect_stdout(io.StringIO()):
        evaluator = COCOeval(truth, predicted, "bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

    used_thresholds = tuple(round(float(value), 2) for value in evaluator.params.iouThrs)
    if used_thresholds != IOU_THRESHOLDS:
        msg = f"COCOeval used IoU thresholds {used_thresholds}, not the frozen {IOU_THRESHOLDS}"
        raise CanonicalEvaluationError(msg)
    if tuple(int(value) for value in evaluator.params.maxDets) != MAX_DETS:
        msg = f"COCOeval used maxDets {evaluator.params.maxDets}, not the frozen {MAX_DETS}"
        raise CanonicalEvaluationError(msg)

    precision = evaluator.eval["precision"]
    per_class: dict[str, dict[str, Any]] = {}
    for position, category_id in enumerate(evaluator.params.catIds):
        name = names.get(int(category_id), str(category_id))
        window = precision[:, :, position, 0, 2]
        at_fifty = precision[0, :, position, 0, 2]
        per_class[name] = {
            "AP@0.50:0.95": rounded(np.mean(window[window > -1])) if (window > -1).any() else None,
            "AP@0.50": rounded(np.mean(at_fifty[at_fifty > -1])) if (at_fifty > -1).any() else None,
        }

    return {
        "all_class_map50_95": rounded(evaluator.stats[0]),
        "all_class_map50": rounded(evaluator.stats[1]),
        "per_class": dict(sorted(per_class.items())),
        "detections_scored": len(detections),
        "cocoeval": {
            "implementation": "pycocotools.cocoeval.COCOeval",
            "iou_type": "bbox",
            "iou_thresholds": [float(value) for value in used_thresholds],
            "max_dets": list(MAX_DETS),
            "area_range": "all",
        },
    }


# --- the frozen spatial analysis ------------------------------------------------------


def spatial_analysis(
    segmenter: Mapping[str, Any],
    detector: Mapping[str, Any],
    *,
    containment_floor: float,
) -> dict[str, Any]:
    """Compute the frozen spatial quantities, their box proxies and association.

    Args:
        segmenter: The segmenter's operational predictions, with masks.
        detector: The detector's operational predictions.
        containment_floor: The frozen association containment floor.

    Returns:
        Per-feature statistics, proxy comparisons, association tallies and the
        deterministic disagreement records.
    """
    areas: list[float] = []
    fill_ratios: list[float] = []
    extents: list[float] = []
    displacements: list[float] = []
    box_areas: list[float] = []
    per_class_fill: dict[str, list[float]] = {}
    per_class_extent: dict[str, list[float]] = {}

    mask_intersections: list[float] = []
    box_intersections: list[float] = []
    mask_containments: list[float] = []
    box_containments: list[float] = []
    coverage_proxy_mask: list[float] = []
    coverage_proxy_box: list[float] = []

    box_overlap_without_mask_overlap = 0
    candidate_pairs = 0

    categories_isolating: list[str] = []
    categories_pipeline: list[str] = []
    per_relationship: dict[str, list[str]] = {"helmet_to_person": [], "vest_to_person": []}
    per_class_categories: dict[str, list[str]] = {}
    records: list[dict[str, Any]] = []

    for stem, instances in segmenter["per_image"].items():
        people = [item for item in instances if item.class_name == PERSON_CLASS]
        ppe = [item for item in instances if item.class_name in PPE_CLASSES]
        detector_people = [
            item for item in detector["per_image"].get(stem, []) if item.class_name == PERSON_CLASS
        ]

        for instance in instances:
            areas.append(float(instance_area_pixels(instance)))
            box_areas.append(instance.box_area())
            ratio = mask_to_box_fill_ratio(instance)
            if ratio is not None:
                fill_ratios.append(ratio)
                per_class_fill.setdefault(instance.class_name, []).append(ratio)
            extent = shape_extent(instance)
            if extent is not None:
                extents.append(extent)
                per_class_extent.setdefault(instance.class_name, []).append(extent)
            displacement = centroid_displacement(instance)
            if displacement is not None:
                displacements.append(displacement)

        for item in ppe:
            relationship = (
                "helmet_to_person" if item.class_name in HELMET_CLASSES else "vest_to_person"
            )
            for person in people:
                candidate_pairs += 1
                shared = mask_intersection(item, person)
                overlap = box_intersection(item, person)
                mask_intersections.append(float(shared))
                box_intersections.append(overlap)
                containment = mask_containment(item, person)
                proxy = box_containment(item, person)
                if containment is not None:
                    mask_containments.append(containment)
                if proxy is not None:
                    box_containments.append(proxy)
                if overlap > 0 and shared == 0:
                    box_overlap_without_mask_overlap += 1

            mask_choice = associate(
                item, people, containment_floor=containment_floor, use_masks=True
            )
            box_choice_isolating = associate(
                item, people, containment_floor=containment_floor, use_masks=False
            )
            box_choice_pipeline = associate(
                item, detector_people, containment_floor=containment_floor, use_masks=False
            )

            isolating = association_category(mask_choice, box_choice_isolating)
            pipeline = association_category(mask_choice, box_choice_pipeline)
            categories_isolating.append(isolating)
            categories_pipeline.append(pipeline)
            per_relationship[relationship].append(isolating)
            per_class_categories.setdefault(item.class_name, []).append(isolating)

            if isolating != "BOX_AND_MASK_AGREE":
                records.append(
                    {
                        "image_id": stem,
                        "ppe_class": item.class_name,
                        "ppe_score": round(item.score, 4),
                        "relationship": relationship,
                        "category_geometry_isolating": isolating,
                        "category_pipeline_level": pipeline,
                        "mask_containment_best": rounded(
                            max((mask_containment(item, person) or 0.0) for person in people)
                            if people
                            else 0.0
                        ),
                        "box_containment_best": rounded(
                            max((box_containment(item, person) or 0.0) for person in people)
                            if people
                            else 0.0
                        ),
                        "mask_to_box_fill_ratio": rounded(mask_to_box_fill_ratio(item)),
                        "person_candidates": len(people),
                        # Recorded so a later phase can reproduce an exception down to which
                        # person each rule chose, without re-running either model.
                        "mask_selected_person_index": mask_choice,
                        "box_selected_person_index_geometry_isolating": box_choice_isolating,
                        "box_selected_person_index_pipeline_level": box_choice_pipeline,
                    }
                )

    # VISIBLE_PPE_COVERAGE_PROXY: associated PPE mask area over person mask area,
    # and the box counterpart on the same pairs. The literal reading of the frozen
    # sentence; see the module docstring for why it is the one interpretive one.
    for instances in segmenter["per_image"].values():
        people = [item for item in instances if item.class_name == PERSON_CLASS]
        ppe = [item for item in instances if item.class_name in PPE_CLASSES]
        for person in people:
            person_mask_area = person.mask_area()
            person_box_area = person.box_area()
            if person_mask_area > 0:
                shared = sum(mask_intersection(item, person) for item in ppe)
                coverage_proxy_mask.append(shared / person_mask_area)
            if person_box_area > 0:
                overlap = sum(box_intersection(item, person) for item in ppe)
                coverage_proxy_box.append(overlap / person_box_area)

    def paired(mask_values: Sequence[float], box_values: Sequence[float]) -> dict[str, Any]:
        return {
            "mask_measurement": describe(mask_values),
            "box_proxy": describe(box_values),
        }

    return {
        "spatial_features": {
            "INSTANCE_AREA_PIXELS": {
                "semantics": FEATURE_SEMANTICS["INSTANCE_AREA_PIXELS"],
                "statistics": describe(areas),
            },
            "MASK_TO_BOX_FILL_RATIO": {
                "semantics": FEATURE_SEMANTICS["MASK_TO_BOX_FILL_RATIO"],
                "statistics": describe(fill_ratios),
                "per_class": {
                    name: describe(values) for name, values in sorted(per_class_fill.items())
                },
            },
            "MASK_CENTROID": {
                "semantics": FEATURE_SEMANTICS["MASK_CENTROID"],
                "statistics": describe(displacements),
                "statistic_is": "DISPLACEMENT_FROM_THE_MODELS_OWN_BOX_CENTRE_IN_PIXELS",
            },
            "SHAPE_EXTENT": {
                "semantics": FEATURE_SEMANTICS["SHAPE_EXTENT"],
                "statistics": describe(extents),
                "per_class": {
                    name: describe(values) for name, values in sorted(per_class_extent.items())
                },
            },
            "PERSON_PPE_MASK_INTERSECTION": {
                "semantics": FEATURE_SEMANTICS["PERSON_PPE_MASK_INTERSECTION"],
                "statistics": describe(mask_intersections),
            },
            "PERSON_PPE_MASK_CONTAINMENT": {
                "semantics": FEATURE_SEMANTICS["PERSON_PPE_MASK_CONTAINMENT"],
                "statistics": describe(mask_containments),
            },
            "VISIBLE_PPE_COVERAGE_PROXY": {
                "semantics": FEATURE_SEMANTICS["VISIBLE_PPE_COVERAGE_PROXY"],
                "statistics": describe(coverage_proxy_mask),
                "definition_is_qualitative_in_the_frozen_protocol": True,
                "implemented_as": (
                    "Summed PPE-mask intersection with a person mask, divided by that person's "
                    "own mask area. The literal reading of the frozen sentence, which fixes the "
                    "inputs and the direction but not an exact formula."
                ),
                "is_not_a_compliance_measure": True,
            },
        },
        "box_proxies": {
            "INSTANCE_AREA_PIXELS": {
                "proxy": "BOX_AREA_PIXELS",
                "statistics": paired(areas, box_areas),
            },
            "MASK_TO_BOX_FILL_RATIO": {"proxy": NO_BOX_PROXY, "statistics": None},
            "MASK_CENTROID": {
                "proxy": "BOX_CENTER",
                "statistics": {"displacement_pixels": describe(displacements)},
            },
            "SHAPE_EXTENT": {"proxy": NO_BOX_PROXY, "statistics": None},
            "PERSON_PPE_MASK_INTERSECTION": {
                "proxy": "BOX_INTERSECTION_AREA",
                "statistics": paired(mask_intersections, box_intersections),
            },
            "PERSON_PPE_MASK_CONTAINMENT": {
                "proxy": "BOX_INTERSECTION_OVER_PPE_BOX_AREA",
                "statistics": paired(mask_containments, box_containments),
            },
            "VISIBLE_PPE_COVERAGE_PROXY": {
                "proxy": "BOX_OVERLAP_DERIVED_COVERAGE",
                "statistics": paired(coverage_proxy_mask, coverage_proxy_box),
            },
        },
        "association": {
            "containment_floor": containment_floor,
            "deterministic": True,
            "geometry_isolating": {
                "box_source": GEOMETRY_ISOLATING,
                **tally(categories_isolating),
            },
            "pipeline_level": {
                "box_source": PIPELINE_LEVEL,
                **tally(categories_pipeline),
            },
            "per_relationship": {
                name: tally(values) for name, values in sorted(per_relationship.items())
            },
            "per_class": {
                name: tally(values) for name, values in sorted(per_class_categories.items())
            },
            "frozen_categories": list(ASSOCIATION_CATEGORIES),
            "frozen_categories_unchanged": True,
            "row_level_person_identity_recorded": True,
            "row_level_person_identity_note": (
                "Each disagreement record carries the person index the mask rule chose and the "
                "one each box rule chose, so an exception can be reproduced down to the person "
                "without re-running either model."
            ),
            "geometry_isolating_versus_pipeline_level": (
                "The geometry-isolating reading holds the model and its instances fixed and "
                "varies only the shape representation, so its exceptions are attributable to "
                "mask-versus-box geometry. The pipeline-level reading compares the frozen "
                "detector's outputs with the frozen segmenter's, so its exceptions reflect "
                "BOTH representational geometry AND the fact that different models produced "
                "different instances. The pipeline-level exceptions must not be attributed "
                "solely to geometry."
            ),
            "fifth_peer_category_added": False,
            "association_taxonomy_status": (
                TAXONOMY_NON_EXHAUSTIVE
                if (
                    TAXONOMY_EXCEPTION in categories_isolating
                    or TAXONOMY_EXCEPTION in categories_pipeline
                )
                else "FROZEN_TAXONOMY_EXHAUSTIVE_FOR_OBSERVED_DATA"
            ),
            "taxonomy_exception_type": TAXONOMY_EXCEPTION,
            "taxonomy_exception_reading": ASSOCIATION_DISAGREEMENT,
            "taxonomy_exception_note": (
                "Phase 10A froze four categories on the implicit assumption that a rule either "
                "associates or it does not. Two rules can both associate and pick different "
                "people, and none of the four is true of that. It is recorded as an exception "
                "to the taxonomy's coverage - excluded from the classified denominator, "
                "counted on its own - rather than as a fifth peer category, because adding a "
                "category after seeing data is what a frozen taxonomy exists to prevent. The "
                "phase 10A protocol is historical and was not modified. This is a "
                "protocol-design limitation found during execution; it invalidates no metric, "
                "no prediction and no raw association decision."
            ),
            "not_an_error": (
                "A different-person outcome is an ASSOCIATION_RULE_DISAGREEMENT, not an "
                "association error. The project holds no person-PPE association ground truth, "
                "so neither rule's answer can be called wrong."
            ),
        },
        "geometry_disagreement": {
            "candidate_pairs_examined": candidate_pairs,
            "box_overlap_without_mask_overlap": box_overlap_without_mask_overlap,
            "box_overlap_without_mask_overlap_fraction": (
                rounded(box_overlap_without_mask_overlap / candidate_pairs)
                if candidate_pairs
                else None
            ),
            "note": (
                "Pairs where the two boxes overlap but the two masks share no pixel at all. "
                "Reported as a count and a fraction rather than binned, because phase 10A "
                "declared no threshold for 'strong' or 'minimal' overlap."
            ),
        },
        "records": records,
    }


# --- artifacts ------------------------------------------------------------------


def build_box_artifact(
    *,
    protocol: Any,
    models: Mapping[str, Any],
    population: Mapping[str, Any],
    results: Mapping[str, Any],
    counts: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble the canonical box comparison artifact.

    Args:
        protocol: The frozen comparison protocol.
        models: Both models' verified identities.
        population: The verified validation population.
        results: Each model's canonical box evaluation.
        counts: Each model's AP-protocol prediction counts.

    Returns:
        The artifact, with deltas recomputed rather than restated.
    """
    inference = protocol.ap_inference
    detector = results[DETECTOR]
    segmenter = results[SEGMENTER]

    def delta(left: Any, right: Any) -> Any:
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            return round(float(right) - float(left), METRIC_PRECISION)
        return None

    per_class = {}
    for name in sorted(set(detector["per_class"]) | set(segmenter["per_class"])):
        d_value = detector["per_class"].get(name, {})
        s_value = segmenter["per_class"].get(name, {})
        per_class[name] = {
            "D2_AP@0.50:0.95": d_value.get("AP@0.50:0.95"),
            "S1_AP@0.50:0.95": s_value.get("AP@0.50:0.95"),
            "delta_AP@0.50:0.95": delta(d_value.get("AP@0.50:0.95"), s_value.get("AP@0.50:0.95")),
            "D2_AP@0.50": d_value.get("AP@0.50"),
            "S1_AP@0.50": s_value.get("AP@0.50"),
            "delta_AP@0.50": delta(d_value.get("AP@0.50"), s_value.get("AP@0.50")),
            "status": RARE_CLASS_STATUS if name == RARE_CLASS else "COMPARISON_REPORTED",
        }

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "analysis": "CANONICAL_BOX_COMPARISON",
        "protocol_fingerprint": protocol.fingerprint(),
        "protocol_config": f"configs/{CONFIG_YAML}",
        "detector": dict(models[DETECTOR]),
        "segmenter": dict(models[SEGMENTER]),
        "population": {
            "split": EVALUATION_SPLIT,
            "images": len(population["image_order"]),
            "annotations": population["annotations"],
            "membership_sha256": population["membership_sha256"],
            "identical_images_for_both_models": True,
            "document": population["document_path"],
            "document_sha256": population["document_sha256"],
        },
        "inference": {
            "purpose": "AP_CURVE",
            "imgsz": inference["imgsz"],
            "conf": inference["conf"],
            "iou": inference["iou"],
            "max_det": inference["max_det"],
            "augment": inference["augment"],
            "tta": inference["tta"],
            "precision": inference["precision"],
        },
        "cocoeval": detector["cocoeval"],
        "ground_truth": "CANONICAL_COCO_DETECTION_BOXES",
        "segmenter_boxes_derived_from_masks": False,
        "segmenter_boxes_source": "THE_SEGMENTERS_OWN_PREDICTED_BOXES",
        "prediction_counts": dict(counts),
        "metrics": {
            "CANONICAL_BOX_MAP50_95": {
                "D2": detector["all_class_map50_95"],
                "S1": segmenter["all_class_map50_95"],
                "delta": delta(detector["all_class_map50_95"], segmenter["all_class_map50_95"]),
            },
            "CANONICAL_BOX_MAP50": {
                "D2": detector["all_class_map50"],
                "S1": segmenter["all_class_map50"],
                "delta": delta(detector["all_class_map50"], segmenter["all_class_map50"]),
            },
        },
        "per_class": per_class,
        "all_class_delta_decomposition": decompose_all_class_delta(per_class),
        "not_comparable_to_native_metrics": (
            "These are canonical-evaluator figures. They are NOT the native framework box "
            "metrics either model reported in its own experiment phase, and the two must never "
            "be differenced: different evaluator implementation, different ground-truth "
            "document and a different confidence. Only the D2-versus-S1 delta computed here, "
            "by one evaluator over one ground truth, is a comparison."
        ),
        "detections_scored": {
            DETECTOR: detector["detections_scored"],
            SEGMENTER: segmenter["detections_scored"],
        },
        "interpretation": (
            "Descriptive. The question is how much object-localisation capability the frozen "
            "segmenter retains relative to the frozen detector while also producing masks. "
            "Neither model changes as a result, and no winner is declared."
        ),
        "winner_declared": False,
        "aggregate_score": False,
        "models_trained": 0,
        "thresholds_tuned": 0,
        "latency_measured": False,
        "rare_class": {
            "name": RARE_CLASS,
            "status": RARE_CLASS_STATUS,
            "detail": (
                "vest_loose holds one validation source image and eight instances under the "
                "frozen split. Its AP and its delta are reported in full and carry high "
                "sampling uncertainty; neither selects or ranks anything."
            ),
        },
        "test": {"status": HOLDOUT_STATUS, "reason": HOLDOUT_REASON},
        "holdout_accessed": False,
    }
    payload["box_comparison_sha256"] = result_fingerprint(
        {
            "protocol_fingerprint": payload["protocol_fingerprint"],
            "detector_checkpoint": models[DETECTOR]["checkpoint_sha256"],
            "segmenter_checkpoint": models[SEGMENTER]["checkpoint_sha256"],
            "membership": population["membership_sha256"],
            "inference": payload["inference"],
            "metrics": payload["metrics"],
            "per_class": payload["per_class"],
            "detections_scored": payload["detections_scored"],
        }
    )
    return payload


def build_spatial_artifact(
    *,
    protocol: Any,
    models: Mapping[str, Any],
    population: Mapping[str, Any],
    analysis: Mapping[str, Any],
    counts: Mapping[str, Any],
    segmenter_predictions: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble the spatial comparison artifact.

    Args:
        protocol: The frozen comparison protocol.
        models: Both models' verified identities.
        population: The verified validation population.
        analysis: The computed spatial analysis.
        counts: Operational prediction counts per model.
        segmenter_predictions: The segmenter's operational predictions.

    Returns:
        The artifact, without the row-level records.
    """
    inference = protocol.operational_inference
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "analysis": "SPATIAL_INFORMATION_COMPARISON",
        "protocol_fingerprint": protocol.fingerprint(),
        "protocol_config": f"configs/{CONFIG_YAML}",
        "detector": dict(models[DETECTOR]),
        "segmenter": dict(models[SEGMENTER]),
        "population": {
            "split": EVALUATION_SPLIT,
            "images": len(population["image_order"]),
            "membership_sha256": population["membership_sha256"],
            "identical_images_for_both_models": True,
        },
        "inference": {
            "purpose": "OPERATIONAL",
            "imgsz": inference["imgsz"],
            "conf": inference["conf"],
            "iou": inference["iou"],
            "max_det": inference["max_det"],
            "augment": inference["augment"],
            "tta": inference["tta"],
            "precision": inference["precision"],
            "retina_masks_for_segmenter": True,
        },
        "prediction_counts": dict(counts),
        "mask_reconstruction": {
            "masks_on_original_canvas": segmenter_predictions["masks_reconstructed"],
            "excluded_predictions": len(segmenter_predictions["excluded"]),
            "exclusions": segmenter_predictions["excluded"],
            "policy": (
                "A predicted mask whose shape does not match its source image is excluded from "
                "every mask-derived measurement and counted here, rather than resized into "
                "agreement - which would make each measurement partly a measurement of the "
                "resize."
            ),
        },
        "spatial_features": dict(analysis["spatial_features"]),
        "box_proxies": dict(analysis["box_proxies"]),
        "association": dict(analysis["association"]),
        "geometry_disagreement": dict(analysis["geometry_disagreement"]),
        "statistic_set": list(STATISTIC_NAMES),
        "inferential_tests_run": 0,
        "post_hoc_bins_created": False,
        "compliance_accuracy_claimed": False,
        "new_metrics_introduced": 0,
        "thresholds_tuned": 0,
        "latency_measured": False,
        "models_trained": 0,
        "rare_class": {
            "name": RARE_CLASS,
            "status": RARE_CLASS_STATUS,
            "detail": (
                "vest_loose is reported in every per-class breakdown and decides nothing. Its "
                "validation support is one image and eight canonical instances."
            ),
        },
        "interpretation": {
            "REPRESENTATION_GAIN": (
                "Quantities a mask makes computable that a box cannot express at all: "
                "MASK_TO_BOX_FILL_RATIO and SHAPE_EXTENT, both frozen as "
                "NO_BOX_ONLY_EQUIVALENT."
            ),
            "PROXY_REFINEMENT": (
                "Quantities a box can already approximate, where the mask changes the value: "
                "area, centroid, intersection, containment and the coverage proxy."
            ),
            "ASSOCIATION_DIFFERENCE": (
                "Cases where the mask rule and the box rule reach different conclusions about "
                "which person a PPE instance belongs to."
            ),
            "no_compliance_claim": (
                "No safety-compliance accuracy is claimed anywhere. The project holds no "
                "compliance ground truth, so there is nothing such a claim could be measured "
                "against."
            ),
        },
        "test": {"status": HOLDOUT_STATUS, "reason": HOLDOUT_REASON},
        "holdout_accessed": False,
    }
    payload["spatial_comparison_sha256"] = result_fingerprint(
        {
            "protocol_fingerprint": payload["protocol_fingerprint"],
            "detector_checkpoint": models[DETECTOR]["checkpoint_sha256"],
            "segmenter_checkpoint": models[SEGMENTER]["checkpoint_sha256"],
            "membership": population["membership_sha256"],
            "inference": payload["inference"],
            "prediction_counts": payload["prediction_counts"],
            "spatial_features": payload["spatial_features"],
            "box_proxies": payload["box_proxies"],
            "association": payload["association"],
            "geometry_disagreement": payload["geometry_disagreement"],
        }
    )
    return payload


def build_box_csv(payload: Mapping[str, Any]) -> str:
    """Render the box comparison as a table.

    Args:
        payload: The box comparison artifact.

    Returns:
        CSV text with a trailing newline.
    """
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=["metric", "scope", "D2", "S1", "delta_S1_minus_D2", "status"],
        lineterminator="\n",
    )
    writer.writeheader()
    for name in ("CANONICAL_BOX_MAP50_95", "CANONICAL_BOX_MAP50"):
        block = payload["metrics"][name]
        writer.writerow(
            {
                "metric": name,
                "scope": "all_class",
                "D2": block["D2"],
                "S1": block["S1"],
                "delta_S1_minus_D2": block["delta"],
                "status": "COMPARISON_REPORTED",
            }
        )
    for name, row in sorted(payload["per_class"].items()):
        writer.writerow(
            {
                "metric": "CANONICAL_BOX_AP50_95",
                "scope": name,
                "D2": row["D2_AP@0.50:0.95"],
                "S1": row["S1_AP@0.50:0.95"],
                "delta_S1_minus_D2": row["delta_AP@0.50:0.95"],
                "status": row["status"],
            }
        )
    return buffer.getvalue()


def build_examples_csv(records: Sequence[Mapping[str, Any]]) -> str:
    """Render the deterministic disagreement example manifest.

    Selected by a stable ordering of the records themselves, never by looking
    at images and picking interesting ones.

    Args:
        records: Every association disagreement found.

    Returns:
        CSV text with a trailing newline.
    """
    fields = [
        "image_id",
        "ppe_class",
        "ppe_score",
        "relationship",
        "category_geometry_isolating",
        "category_pipeline_level",
        "mask_containment_best",
        "box_containment_best",
        "mask_to_box_fill_ratio",
        "person_candidates",
    ]
    ordered = sorted(
        records,
        key=lambda row: (
            row["category_geometry_isolating"],
            -abs((row["mask_containment_best"] or 0.0) - (row["box_containment_best"] or 0.0)),
            row["image_id"],
            row["ppe_class"],
        ),
    )[:EXAMPLE_COUNT]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in ordered:
        writer.writerow({name: row[name] for name in fields})
    return buffer.getvalue()


def build_report(box: Mapping[str, Any], spatial: Mapping[str, Any], *, commit: str | None) -> str:
    """Render the combined comparison report.

    Args:
        box: The canonical box comparison artifact.
        spatial: The spatial comparison artifact.
        commit: Repository commit, when available.

    Returns:
        The report text.
    """
    detector = box["detector"]
    segmenter = box["segmenter"]
    metrics = box["metrics"]
    features = spatial["spatial_features"]
    proxies = spatial["box_proxies"]
    association = spatial["association"]
    disagreement = spatial["geometry_disagreement"]

    lines: list[str] = []
    add = lines.append

    def stat_row(label: str, block: Mapping[str, Any]) -> str:
        return (
            f"| {label} | {block['count']} | {block['mean']} | {block['median']} | "
            f"{block['std']} | {block['p25']} | {block['p75']} | {block['p90']} | "
            f"{block['p95']} | {block['min']} | {block['max']} |"
        )

    header = (
        "| Quantity | n | mean | median | std | P25 | P75 | P90 | P95 | min | max |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    )

    add("# Detector versus segmenter on validation")
    add("")
    add(
        f"Phase {box['phase']} · `FROZEN_PROTOCOL` `{box['protocol_fingerprint']}` · "
        f"validation only · no latency measured"
    )
    add("")
    add(
        "**Every number here is a validation number.** The holdout has never been evaluated. "
        "Neither model was trained, modified or re-thresholded, and no latency or memory "
        "benchmark was run - that is phase 10C."
    )
    add("")
    if commit:
        add(f"Repository commit at production time: `{commit}`.")
        add("")

    add("## 1. Scientific comparison question")
    add("")
    add(
        "> What additional spatial and operational information does the frozen instance "
        "segmentation model provide beyond the frozen bounding-box detector?"
    )
    add("")
    add(
        "`FROZEN_PROTOCOL`. The two models do not produce the same output, so this is not a "
        "contest and no winner is declared. The cost half of the question - latency and memory "
        "- is `LATENCY_PENDING` and belongs to phase 10C."
    )
    add("")

    add("## 2. Frozen model identities")
    add("")
    add("| | Detector | Segmenter |")
    add("| --- | --- | --- |")
    add(f"| Experiment | **{detector['experiment']}** | **{segmenter['experiment']}** |")
    add(f"| Model | {detector['model']} | {segmenter['model']} |")
    add(f"| imgsz | {detector['imgsz']} | {segmenter['imgsz']} |")
    add(f"| Checkpoint | `{detector['checkpoint_sha256']}` | `{segmenter['checkpoint_sha256']}` |")
    add("")

    add("## 3-4. Frozen protocol and validation population")
    add("")
    add(
        f"Protocol fingerprint `{box['protocol_fingerprint']}`, frozen in phase 10A before any "
        f"of this ran. Population: **{box['population']['images']} validation images**, "
        f"{box['population']['annotations']} canonical annotations, membership "
        f"`{box['population']['membership_sha256']}`. Both models saw the same images "
        f"(`identical_images_for_both_models: "
        f"{box['population']['identical_images_for_both_models']}`)."
    )
    add("")

    add("## 5. Precision verification")
    add("")
    precision = box["precision_preflight"]
    evidence = precision["evidence"][detector["experiment"]]
    add(
        f"`{precision['status']}`. Both models were probed at runtime before any comparison "
        f"number existed: backend FP16 flag `{evidence['backend_fp16_flag']}`, parameter dtypes "
        f"`{evidence['parameter_dtypes']}`, input tensor dtype `{evidence['input_dtype']}`, "
        f"autocast during forward `{evidence['autocast_enabled_during_forward']}`, quantization "
        f"config present `{evidence['quantization_config_present']}`. Identical for both "
        f"(`identical_across_models: {precision['identical_across_models']}`)."
    )
    add("")
    add(f"{precision['probe_method']}")
    add("")

    add("## 6-9. Canonical box evaluation")
    add("")
    add(
        f"`CANONICAL_BOX_EVALUATION`. Both models' **own predicted boxes** "
        f"(`segmenter_boxes_derived_from_masks: "
        f"{box['segmenter_boxes_derived_from_masks']}`) scored against the same canonical "
        f"ground truth by one external evaluator: `{box['cocoeval']['implementation']}`, "
        f"`iouType='{box['cocoeval']['iou_type']}'`, IoU {box['cocoeval']['iou_thresholds'][0]}"
        f":{box['cocoeval']['iou_thresholds'][-1]}, maxDets {box['cocoeval']['max_dets']}. "
        f"AP inference at conf {box['inference']['conf']}, imgsz {box['inference']['imgsz']}, "
        f"{box['inference']['precision']}."
    )
    add("")
    add("| Metric | D2 | S1 | delta (S1 - D2) |")
    add("| --- | --- | --- | --- |")
    for name in ("CANONICAL_BOX_MAP50_95", "CANONICAL_BOX_MAP50"):
        block = metrics[name]
        add(f"| **{name}** | {block['D2']} | {block['S1']} | **{block['delta']:+f}** |")
    add("")
    add("`COMPUTED_RESULT`. Per class, AP@0.50:0.95:")
    add("")
    add("| Class | D2 | S1 | delta | status |")
    add("| --- | --- | --- | --- | --- |")
    for name, row in sorted(box["per_class"].items()):
        add(
            f"| {name} | {row['D2_AP@0.50:0.95']} | {row['S1_AP@0.50:0.95']} | "
            f"{row['delta_AP@0.50:0.95']:+f} | `{row['status']}` |"
        )
    add("")
    add(f"{box['interpretation']}")
    add("")
    decomposition = box["all_class_delta_decomposition"]
    add(
        f"**The aggregate delta is not an across-the-board improvement.** The all-class figure "
        f"is the unweighted mean of the five per-class APs "
        f"(`all_class_is_unweighted_mean_of_per_class: "
        f"{decomposition['all_class_is_unweighted_mean_of_per_class']}`), so each class "
        f"contributes its own delta divided by {decomposition['class_count']}: "
        + ", ".join(
            f"`{name}` {value:+f}" for name, value in sorted(decomposition["contributions"].items())
        )
        + "."
    )
    add("")
    add(
        f"`LIMITATION`. **`{decomposition['rare_class']}` alone contributes "
        f"{decomposition['rare_class_contribution']:+f}, which is larger than the entire "
        f"aggregate delta of {box['metrics']['CANONICAL_BOX_MAP50_95']['delta']:+f}** - and it "
        f"is the class the project already classifies `{RARE_CLASS_STATUS}`, with one "
        f"validation source image. Excluding it, the mean delta over the other "
        f"{decomposition['class_count'] - 1} classes is "
        f"**{decomposition['delta_excluding_rare_class']:+f}**: the segmenter would sit "
        f"*below* the detector. Classes that declined: "
        + ", ".join(f"`{name}`" for name in decomposition["declined_classes"])
        + ". Quoting the aggregate improvement without this would be exactly the "
        "metric-shopping the project's phase 7A policy forbids."
    )
    add("")
    add(f"`LIMITATION`. {box['not_comparable_to_native_metrics']}")
    add("")

    add("## 10. Operational inference population")
    add("")
    counts = spatial["prediction_counts"]
    reconstruction = spatial["mask_reconstruction"]
    add(
        f"`OPERATIONAL_ANALYSIS` at conf {spatial['inference']['conf']}, imgsz "
        f"{spatial['inference']['imgsz']}, {spatial['inference']['precision']} - a separate "
        "pass from the AP protocol, never mixed with it."
    )
    add("")
    add("| | D2 | S1 |")
    add("| --- | --- | --- |")
    add(f"| Total predictions | {counts['D2']['total']} | {counts['S1']['total']} |")
    for name in sorted(counts["S1"]["per_class"]):
        add(
            f"| {name} | {counts['D2']['per_class'].get(name, 0)} | "
            f"{counts['S1']['per_class'].get(name, 0)} |"
        )
    add("")
    add(
        f"S1 masks reconstructed on the original canvas: "
        f"**{reconstruction['masks_on_original_canvas']}**; predictions excluded from "
        f"mask-derived analysis: **{reconstruction['excluded_predictions']}**. "
        f"{reconstruction['policy']}"
    )
    add("")

    add("## 11. Spatial information available only from masks")
    add("")
    add(
        "`SPATIAL_INFORMATION_GAIN`. Two of the seven frozen quantities have no box-only "
        "equivalent at all - they are what a box fundamentally cannot express."
    )
    add("")
    add(header)
    add(stat_row("`MASK_TO_BOX_FILL_RATIO`", features["MASK_TO_BOX_FILL_RATIO"]["statistics"]))
    add(stat_row("`SHAPE_EXTENT`", features["SHAPE_EXTENT"]["statistics"]))
    add("")
    add(
        "A fill ratio well below 1 means the predicted box is mostly not the object. A shape "
        "extent below 1 means the instance does not fill even its own tight rectangle. Both "
        "are `MASK_MEASUREMENT` with `BOX_PROXY` `NO_BOX_ONLY_EQUIVALENT`; the distributions "
        "are reported continuously, because phase 10A declared no bins."
    )
    add("")
    add("Per class, fill ratio:")
    add("")
    add(header)
    for name, block in sorted(features["MASK_TO_BOX_FILL_RATIO"]["per_class"].items()):
        add(stat_row(f"`{name}`", block))
    add("")

    add("## 12. Mask measurements versus box proxies")
    add("")
    add("`MASK_MEASUREMENT` against `BOX_PROXY`, on the same instances and the same pairs.")
    add("")
    for name in (
        "INSTANCE_AREA_PIXELS",
        "PERSON_PPE_MASK_INTERSECTION",
        "PERSON_PPE_MASK_CONTAINMENT",
        "VISIBLE_PPE_COVERAGE_PROXY",
    ):
        block = proxies[name]
        add(f"**`{name}`** vs `{block['proxy']}`")
        add("")
        add(header)
        add(stat_row("mask measurement", block["statistics"]["mask_measurement"]))
        add(stat_row("box proxy", block["statistics"]["box_proxy"]))
        add("")
    displacement = proxies["MASK_CENTROID"]["statistics"]["displacement_pixels"]
    add("**`MASK_CENTROID`** vs `BOX_CENTER` - the displacement between them, in pixels:")
    add("")
    add(header)
    add(stat_row("centroid displacement", displacement))
    add("")

    add("## 13-14. Person-PPE spatial association and disagreement")
    add("")
    add(
        f"`OPERATIONAL_ANALYSIS`. The frozen deterministic rule at containment floor "
        f"**{association['containment_floor']}**, applied to two box sources and reported "
        "separately."
    )
    add("")
    frozen_names = list(association["frozen_categories"])
    for key, label in (
        ("geometry_isolating", "Geometry-isolating (S1's own boxes)"),
        ("pipeline_level", "Pipeline-level (D2's boxes)"),
    ):
        block = association[key]
        add(
            f"**{label}** - `{block['box_source']}`, {block['total_relationships']} candidate "
            f"relationships, of which **{block['classified_relationships']} are classified by "
            f"the frozen taxonomy** and {block['taxonomy_exceptions']} fall outside it "
            f"(coverage {block['taxonomy_coverage']})."
        )
        add("")
        add(f"| Frozen category | count | % of classified ({block['classified_relationships']}) |")
        add("| --- | --- | --- |")
        for name in frozen_names:
            add(
                f"| `{name}` | {block['frozen_category_counts'][name]} | "
                f"{block['frozen_category_percentages'][name]} |"
            )
        add(
            f"| *(outside the taxonomy)* `{block['taxonomy_exception']['type']}` | "
            f"{block['taxonomy_exceptions']} | *excluded from the denominator* |"
        )
        add("")
    add(
        f"Counts are over all relationships; percentages use "
        f"`{association['geometry_isolating']['percentage_denominator']}` as their denominator, "
        "stated here because mixing an undeclared state into it would quietly change what the "
        "frozen percentages mean. Taxonomy coverage is protocol bookkeeping - how much of the "
        "observed data the frozen taxonomy describes - and not a spatial-performance metric."
    )
    add("")
    add(f"`LIMITATION` `{association['association_taxonomy_status']}`.")
    add("")
    add(f"{association['taxonomy_exception_note']}")
    add("")
    add(f"{association['not_an_error']}")
    add("")
    add(f"`LIMITATION`. {association['geometry_isolating_versus_pipeline_level']}")
    add("")
    add(f"{association['row_level_person_identity_note']}")
    add("")
    add("Per relationship, over the frozen categories:")
    add("")
    add(
        "| Relationship | "
        + " | ".join(f"`{name}`" for name in frozen_names)
        + " | classified | exceptions | total |"
    )
    add("| --- | " + " | ".join("---" for _ in frozen_names) + " | --- | --- | --- |")
    for name, block in sorted(association["per_relationship"].items()):
        add(
            f"| {name} | "
            + " | ".join(str(block["frozen_category_counts"][key]) for key in frozen_names)
            + f" | {block['classified_relationships']} | {block['taxonomy_exceptions']} "
            f"| {block['total_relationships']} |"
        )
    add("")

    add("## 15. Per-class context")
    add("")
    add(
        "| Class | "
        + " | ".join(f"`{name}`" for name in frozen_names)
        + " | classified | exceptions | total |"
    )
    add("| --- | " + " | ".join("---" for _ in frozen_names) + " | --- | --- | --- |")
    for name, block in sorted(association["per_class"].items()):
        add(
            f"| {name} | "
            + " | ".join(str(block["frozen_category_counts"][key]) for key in frozen_names)
            + f" | {block['classified_relationships']} | {block['taxonomy_exceptions']} "
            f"| {block['total_relationships']} |"
        )
    add("")

    add("## 16-17. Representation gain and proxy refinement")
    add("")
    interpretation = spatial["interpretation"]
    add(f"* **REPRESENTATION GAIN** - {interpretation['REPRESENTATION_GAIN']}")
    add(f"* **PROXY REFINEMENT** - {interpretation['PROXY_REFINEMENT']}")
    add(f"* **ASSOCIATION DIFFERENCE** - {interpretation['ASSOCIATION_DIFFERENCE']}")
    add("")
    add(
        f"`COMPUTED_RESULT`. Of {disagreement['candidate_pairs_examined']} PPE-person candidate "
        f"pairs, **{disagreement['box_overlap_without_mask_overlap']}** had overlapping boxes "
        f"but masks sharing no pixel at all "
        f"({disagreement['box_overlap_without_mask_overlap_fraction']} of pairs). "
        f"{disagreement['note']}"
    )
    add("")

    add("## 18-19. Operational interpretation and the coverage proxy's limitation")
    add("")
    coverage = features["VISIBLE_PPE_COVERAGE_PROXY"]
    add(f"`OPERATIONAL_PROXY` `{coverage['semantics']}`. {coverage['implemented_as']}")
    add("")
    add(
        f"`LIMITATION`. This is the one frozen quantity whose definition is qualitative "
        f"(`definition_is_qualitative_in_the_frozen_protocol: "
        f"{coverage['definition_is_qualitative_in_the_frozen_protocol']}`): the protocol fixed "
        "its inputs and its direction but not an exact formula, so the implementation is a "
        "literal reading rather than a derivation. It is **not a compliance measure** "
        f"(`is_not_a_compliance_measure: {coverage['is_not_a_compliance_measure']}`), and "
        f"{interpretation['no_compliance_claim']} Nothing in this report should rest on this "
        "quantity alone."
    )
    add("")

    add("## 20. Recognition and spatial trade-off")
    add("")
    sensitivity = box.get("supported_class_sensitivity")
    if sensitivity:
        add(
            f"`{sensitivity['label']}`. Because the all-class figure is an unweighted mean, the "
            "same comparison is shown over the classes the project's **pre-existing** support "
            f"rule already admits ({sensitivity['admitted_classes']}), setting aside "
            f"{', '.join(sensitivity['excluded_classes'])}:"
        )
        add("")
        add("| | D2 | S1 | delta |")
        add("| --- | --- | --- | --- |")
        add(
            f"| All-class canonical box mAP@0.50:0.95 | {metrics['CANONICAL_BOX_MAP50_95']['D2']} "
            f"| {metrics['CANONICAL_BOX_MAP50_95']['S1']} | "
            f"**{metrics['CANONICAL_BOX_MAP50_95']['delta']:+f}** |"
        )
        add(
            f"| Supported-class macro (descriptive) | {sensitivity['D2_supported_macro']} | "
            f"{sensitivity['S1_supported_macro']} | **{sensitivity['delta']:+f}** |"
        )
        add("")
        add(
            f"`LIMITATION`. This is **not** a frozen phase 10A metric "
            f"(`is_a_frozen_phase_10a_metric: "
            f"{sensitivity['is_a_frozen_phase_10a_metric']}`), **not** a selection rule "
            f"(`is_a_selection_rule: {sensitivity['is_a_selection_rule']}`) and **not** a "
            f"significance test (`is_a_significance_test: "
            f"{sensitivity['is_a_significance_test']}`). It changes no frozen number "
            f"(`changes_any_frozen_number: {sensitivity['changes_any_frozen_number']}`). "
            f"{sensitivity['why']}"
        )
        add("")
    add(f"`COMPUTED_RESULT`. {box.get('localization_conclusion', '')}")
    add("")
    add(
        "Alongside that, the segmenter produces instance masks, which make two quantities "
        "computable that a box cannot express at all and change the value of five more. "
        "Whether the trade is worth making also depends on cost, which this phase did not "
        "measure."
    )
    add("")
    add(
        "`LIMITATION`. One run of each model was ever trained, so run-to-run variance is "
        "UNKNOWN for both and a small localisation delta is not evidence of an ordering."
    )
    add("")

    add("## 21. Holdout compliance")
    add("")
    add(f"`HOLDOUT_POLICY` `{box['test']['status']}`. {box['test']['reason']}")
    add("")

    add("## 22. Limitations")
    add("")
    add("`LIMITATION`.")
    add("")
    add("* Validation only. Nothing here says anything about test performance.")
    add(
        "* The spatial quantities are computed from **predictions**, not ground truth. They "
        "describe what the models assert, not what is true in the scene."
    )
    add(
        "* There is no person-PPE association ground truth, so the association analysis "
        "reports agreement between two geometric rules and **no accuracy**."
    )
    add(
        "* The frozen four-category taxonomy does not cover both rules associating to "
        "different people; that case is counted separately rather than absorbed."
    )
    add(
        "* `VISIBLE_PPE_COVERAGE_PROXY` rests on a qualitative frozen definition and is "
        "labelled an interpretive proxy throughout."
    )
    add(
        "* No inferential test is reported. None was predeclared, and choosing one now would "
        "be choosing it after seeing the data."
    )
    add(f"* `{RARE_CLASS}` remains `{RARE_CLASS_STATUS}` and decides nothing.")
    add("")

    add("## 23. Pending latency comparison")
    add("")
    add(
        f"`LATENCY_PENDING`. No latency, throughput or memory benchmark was executed "
        f"(`latency_measured: {box['latency_measured']}`). The cost half of the scientific "
        "question is phase 10C, under the benchmark protocol phase 10A already froze. Any "
        "framework speed line emitted incidentally during this phase's inference is "
        "`INCIDENTAL_NOT_10C_BENCHMARK` and was not recorded or used."
    )
    add("")

    add("## 24. Next phase")
    add("")
    add(
        "Phase 10C runs the frozen latency and memory benchmark; phase 10D synthesises benefit "
        "against cost. No aggregate score will be produced in either."
    )
    add("")

    return "\n".join(lines) + "\n"


# --- entry point ----------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run the controlled validation comparison.

    Args:
        argv: Command-line arguments.

    Returns:
        Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="check preconditions only")
    parser.add_argument(
        "--preflight-only", action="store_true", help="run the precision preflight and stop"
    )
    args = parser.parse_args(argv)

    paths = ProjectPaths.from_root()
    if holdout_unlocked():
        print(
            f"REFUSED: {HOLDOUT_UNLOCK_ENV_VAR} is set. This phase uses the validation split "
            "only and declines to run in an environment where the holdout is unlocked.",
            file=sys.stderr,
        )
        return 2

    try:
        historical = historical_digests(paths)
        protocol = load_comparison_protocol(paths.configs / CONFIG_YAML)
        if protocol.fingerprint() != EXPECTED_PROTOCOL:
            msg = (
                f"the comparison protocol hashes to {protocol.fingerprint()}, not the frozen "
                f"{EXPECTED_PROTOCOL}"
            )
            raise ComparisonError(msg)
        frozen_manifest = read_json(paths.reports / PROTOCOL_JSON)
        if frozen_manifest["protocol_fingerprint"] != EXPECTED_PROTOCOL:
            msg = "the frozen protocol manifest names a different configuration"
            raise ComparisonError(msg)

        detector_identity = load_final_detector(paths.reports)
        segmenter_identity = load_final_segmenter(paths.reports)
        checkpoints = {
            DETECTOR: detector_checkpoint(detector_identity, paths.root),
            SEGMENTER: segmenter_checkpoint(segmenter_identity, paths.root),
        }
        for experiment, identity, declared in (
            (DETECTOR, detector_identity, protocol["detector"]["checkpoint_sha256"]),
            (SEGMENTER, segmenter_identity, protocol["segmenter"]["checkpoint_sha256"]),
        ):
            if identity.checkpoint_sha256 != declared:
                msg = f"the frozen {experiment} checkpoint is not the one the protocol names"
                raise ModelIdentityError(msg)

        models = {
            DETECTOR: {
                "experiment": detector_identity.selected_experiment,
                "model": detector_identity.model,
                "imgsz": detector_identity.imgsz,
                "checkpoint_sha256": detector_identity.checkpoint_sha256,
                "identity_fingerprint": detector_identity.fingerprint,
                "trained_in_this_phase": False,
                "modified_in_this_phase": False,
            },
            SEGMENTER: {
                "experiment": segmenter_identity.selected_experiment,
                "model": segmenter_identity.model,
                "imgsz": segmenter_identity.imgsz,
                "overlap_mask": segmenter_identity.overlap_mask,
                "mask_ratio": segmenter_identity.mask_ratio,
                "checkpoint_sha256": segmenter_identity.checkpoint_sha256,
                "identity_fingerprint": segmenter_identity.fingerprint,
                "final_segmenter_sha256": segmenter_identity.fingerprint,
                "trained_in_this_phase": False,
                "modified_in_this_phase": False,
            },
        }

        population = load_population(paths, protocol)
        audit = read_json(paths.reports / "segmentation_adapter_audit_manifest.json")
        class_map = {name: int(index) for name, index in audit["class_map"].items()}
    except (
        ConfigError,
        ComparisonProtocolError,
        FinalDetectorError,
        FinalSegmenterError,
        ComparisonError,
    ) as exc:
        classification = MODEL_IDENTITY_MISMATCH if isinstance(exc, ModelIdentityError) else BLOCKED
        print(f"{classification}: {exc}", file=sys.stderr)
        return 2

    print(f"protocol   {protocol.fingerprint()}  VERIFIED")
    print(
        f"detector   {models[DETECTOR]['experiment']}  "
        f"{models[DETECTOR]['checkpoint_sha256'][:16]}..."
    )
    print(
        f"segmenter  {models[SEGMENTER]['experiment']}  "
        f"{models[SEGMENTER]['checkpoint_sha256'][:16]}..."
    )
    print(
        f"population {len(population['image_order'])} validation images  "
        f"{population['membership_sha256'][:16]}..."
    )

    if args.verify_only:
        print("VERIFIED: preconditions hold. Nothing inferred or written.")
        print(f"holdout    {HOLDOUT_STATUS}")
        return 0

    # --- precision parity, before any comparison number exists -------------------
    probe_image = population["files"][population["image_order"][0]]
    try:
        precision = precision_preflight(paths, checkpoints, protocol, probe_image)
    except PrecisionError as exc:
        print(f"{PRECISION_MISMATCH}: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"{BLOCKED}: the precision preflight could not run ({exc})", file=sys.stderr)
        return 2
    evidence = precision["evidence"][DETECTOR]
    print(
        f"precision  {precision['status']}  params {evidence['parameter_dtypes']}  input "
        f"{evidence['input_dtype']}  autocast {evidence['autocast_enabled_during_forward']}"
    )

    if args.preflight_only:
        print("PREFLIGHT ONLY: precision parity verified. Nothing else run.")
        return 0

    # --- AP protocol -----------------------------------------------------------------
    try:
        ap_predictions = {
            experiment: predict(
                checkpoints[experiment], population, protocol.ap_inference, want_masks=False
            )
            for experiment in (DETECTOR, SEGMENTER)
        }
    except ComparisonError as exc:
        print(f"{BOX_EVALUATION_FAILED}: {exc}", file=sys.stderr)
        return 2
    print(
        f"AP pass    D2 {ap_predictions[DETECTOR]['total']} predictions  "
        f"S1 {ap_predictions[SEGMENTER]['total']} predictions  (conf 0.001)"
    )

    try:
        box_results = {
            experiment: evaluate_boxes(population, ap_predictions[experiment], class_map)
            for experiment in (DETECTOR, SEGMENTER)
        }
    except CanonicalEvaluationError as exc:
        print(f"{BOX_EVALUATION_FAILED}: {exc}", file=sys.stderr)
        return 2
    print(
        f"box AP     D2 {box_results[DETECTOR]['all_class_map50_95']}  "
        f"S1 {box_results[SEGMENTER]['all_class_map50_95']}  (canonical, mAP@0.50:0.95)"
    )

    # --- operational protocol ------------------------------------------------------------
    try:
        operational = {
            DETECTOR: predict(
                checkpoints[DETECTOR],
                population,
                protocol.operational_inference,
                want_masks=False,
            ),
            SEGMENTER: predict(
                checkpoints[SEGMENTER],
                population,
                protocol.operational_inference,
                want_masks=True,
            ),
        }
    except ComparisonError as exc:
        print(f"{SPATIAL_ANALYSIS_FAILED}: {exc}", file=sys.stderr)
        return 2

    def per_class_counts(predictions: Mapping[str, Any]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for instances in predictions["per_image"].values():
            for instance in instances:
                counts[instance.class_name] = counts.get(instance.class_name, 0) + 1
        return dict(sorted(counts.items()))

    counts = {
        experiment: {
            "total": operational[experiment]["total"],
            "per_class": per_class_counts(operational[experiment]),
        }
        for experiment in (DETECTOR, SEGMENTER)
    }
    counts[SEGMENTER]["masks_reconstructed"] = operational[SEGMENTER]["masks_reconstructed"]
    counts[SEGMENTER]["excluded_from_mask_analysis"] = len(operational[SEGMENTER]["excluded"])
    print(
        f"op pass    D2 {counts[DETECTOR]['total']} predictions  "
        f"S1 {counts[SEGMENTER]['total']} predictions  "
        f"{counts[SEGMENTER]['masks_reconstructed']} masks  (conf 0.25)"
    )

    try:
        analysis = spatial_analysis(
            operational[SEGMENTER],
            operational[DETECTOR],
            containment_floor=float(protocol["association"]["containment_floor"]),
        )
    except Exception as exc:
        print(f"{SPATIAL_ANALYSIS_FAILED}: {exc}", file=sys.stderr)
        return 2
    print(
        f"spatial    {analysis['geometry_disagreement']['candidate_pairs_examined']} "
        f"candidate pairs  "
        f"{analysis['association']['geometry_isolating']['total']} associations examined"
    )

    # --- artifacts -------------------------------------------------------------------------
    box_payload = build_box_artifact(
        protocol=protocol,
        models=models,
        population=population,
        results=box_results,
        counts={
            experiment: {"total": ap_predictions[experiment]["total"]}
            for experiment in (DETECTOR, SEGMENTER)
        },
    )
    box_payload["precision_preflight"] = precision
    box_payload["box_comparison_sha256"] = result_fingerprint(
        {
            "protocol_fingerprint": box_payload["protocol_fingerprint"],
            "detector_checkpoint": models[DETECTOR]["checkpoint_sha256"],
            "segmenter_checkpoint": models[SEGMENTER]["checkpoint_sha256"],
            "membership": population["membership_sha256"],
            "inference": box_payload["inference"],
            "metrics": box_payload["metrics"],
            "per_class": box_payload["per_class"],
            "detections_scored": box_payload["detections_scored"],
        }
    )

    spatial_payload = build_spatial_artifact(
        protocol=protocol,
        models=models,
        population=population,
        analysis=analysis,
        counts=counts,
        segmenter_predictions=operational[SEGMENTER],
    )
    spatial_payload["precision_preflight_status"] = precision["status"]

    problems = validate_box_comparison(
        box_payload,
        protocol_fingerprint=protocol.fingerprint(),
        detector_sha256=models[DETECTOR]["checkpoint_sha256"],
        segmenter_sha256=models[SEGMENTER]["checkpoint_sha256"],
        membership_sha256=population["membership_sha256"],
    ) + validate_spatial_comparison(
        spatial_payload,
        protocol_fingerprint=protocol.fingerprint(),
        membership_sha256=population["membership_sha256"],
    )
    if problems:
        print(f"{PROTOCOL_VIOLATION}: the emitted artifacts do not validate:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 2

    report = build_report(box_payload, spatial_payload, commit=git_commit(paths.root))
    box_csv = build_box_csv(box_payload)
    examples_csv = build_examples_csv(analysis["records"])

    findings = (
        scan_for_sensitive(report)
        + scan_for_sensitive(box_csv)
        + scan_for_sensitive(examples_csv)
        + scan_for_sensitive(json.dumps(box_payload))
        + scan_for_sensitive(json.dumps(spatial_payload))
    )
    if findings:
        print(f"{BLOCKED}: sensitive content in the emitted artifacts: {findings}", file=sys.stderr)
        return 2

    box_sha = write_json(paths.reports / BOX_JSON, box_payload)
    spatial_sha = write_json(paths.reports / SPATIAL_JSON, spatial_payload)
    (paths.reports / REPORT_MD).write_text(report, encoding="utf-8", newline="\n")
    (paths.reports / BOX_CSV).write_text(box_csv, encoding="utf-8", newline="\n")
    (paths.reports / EXAMPLES_CSV).write_text(examples_csv, encoding="utf-8", newline="\n")

    # Row-level records stay in ignored runtime space; the committed artifacts carry
    # the aggregates and the fingerprints that reproduce them.
    runtime = paths.root / RUNTIME_ROOT
    runtime.mkdir(parents=True, exist_ok=True)
    write_json(runtime / "spatial_records.json", analysis["records"])

    after = historical_digests(paths)
    changed = sorted(name for name in historical if historical[name] != after.get(name))
    if changed:
        print(f"{PROTOCOL_VIOLATION}: historical artifacts changed: {changed}", file=sys.stderr)
        return 2

    record = ProvenanceRecord.create(
        name="detector_segmenter_validation_comparison",
        phase=10,
        config={
            "detector_segmenter_comparison": f"configs/{CONFIG_YAML}",
            "protocol_fingerprint": protocol.fingerprint(),
        },
        details={
            "phase": PHASE,
            "classification": COMPLETE,
            "detector_checkpoint_sha256": models[DETECTOR]["checkpoint_sha256"],
            "segmenter_checkpoint_sha256": models[SEGMENTER]["checkpoint_sha256"],
            "membership_sha256": population["membership_sha256"],
            "precision_status": precision["status"],
            "canonical_box_map50_95": box_payload["metrics"]["CANONICAL_BOX_MAP50_95"],
            "canonical_box_map50": box_payload["metrics"]["CANONICAL_BOX_MAP50"],
            "box_comparison_sha256": box_payload["box_comparison_sha256"],
            "spatial_comparison_sha256": spatial_payload["spatial_comparison_sha256"],
            "operational_counts": counts,
            "models_trained": 0,
            "models_modified": 0,
            "thresholds_tuned": 0,
            "latency_measured": False,
            "memory_measured": False,
            "holdout_accessed": False,
            "historical_artifacts_unchanged": True,
        },
        repo_root=paths.root,
    )
    record.add_input(paths.configs / CONFIG_YAML, relative_to=paths.root)
    for name in ("final_detector_manifest.json", "final_segmenter_manifest.json", PROTOCOL_JSON):
        record.add_input(paths.reports / name, relative_to=paths.root)
    for name in (BOX_JSON, BOX_CSV, SPATIAL_JSON, REPORT_MD, EXAMPLES_CSV):
        record.add_output(paths.reports / name, relative_to=paths.root)
    record.write_json(paths.reports / PROVENANCE_JSON)

    print(COMPLETE)
    print(
        f"box        D2 {box_payload['metrics']['CANONICAL_BOX_MAP50_95']['D2']}  "
        f"S1 {box_payload['metrics']['CANONICAL_BOX_MAP50_95']['S1']}  "
        f"delta {box_payload['metrics']['CANONICAL_BOX_MAP50_95']['delta']}"
    )
    print(f"box sha    {box_payload['box_comparison_sha256']}")
    print(f"spatial sha {spatial_payload['spatial_comparison_sha256']}")
    print(
        f"artifacts  reports/{BOX_JSON} ({box_sha[:16]}...)  "
        f"reports/{SPATIAL_JSON} ({spatial_sha[:16]}...)"
    )
    print(f"report     reports/{REPORT_MD}")
    print("latency    NOT_MEASURED (phase 10C)")
    print(f"holdout    {HOLDOUT_STATUS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
