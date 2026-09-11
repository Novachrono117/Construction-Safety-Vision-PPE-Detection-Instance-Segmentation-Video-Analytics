"""Execution of the frozen phase 11A holdout protocol - phase 11B.

Phase 11A wrote down, in advance, everything that would be done to the holdout.
This module does exactly that and nothing else. It is separated from
:mod:`construction_safety_vision.final_holdout_evaluation`, which is a phase 11A
artifact and stays byte-identical, and from the runner, which is the
authorisation boundary.

Six things are deliberate.

**The holdout is reached once, through the existing guard.** Membership comes
from :class:`~construction_safety_vision.data.split_freeze.FrozenSplits` and the
images are materialised by the phase 5D function with the phase 5D
configuration. There is no split-specific code path: the holdout is written by
the same function that wrote ``train`` and ``validation``.

**Predictions are persisted and fingerprinted before any metric exists.** Every
number reported here is then computed from the persisted file, never from a live
model. That is what makes a report rebuild possible without a second inference
run, and it is why the two failures have separate names in the frozen policy.

**Three inference passes, all declared in advance.** The detector's AP pass and
the segmenter's AP pass at conf 0.001, and the segmenter's operational pass at
conf 0.25 for the phase 8C direct-IoU diagnostic. Nothing else runs a model.

**Quantities the protocol gave a confidence but no inference block - the
confusion matrix, object-level TP/FP/FN and the qualitative ranking - are
derived from the declared AP passes filtered at the frozen 0.25.** That is the
framework's own convention, it keeps the two models symmetric, and the
resolution was written before any holdout number existed. It is recorded as
``PRE_EXECUTION_PROTOCOL_GAP_RESOLUTION`` rather than presented as something
phase 11A had already settled.

**Nothing is tuned.** No threshold is swept, no operating point is chosen with a
result in view, no category is added to a frozen taxonomy, and no model is
selected, retrained or replaced.

**No holdout identifier reaches a committed artifact.** Identifiers exist inside
the git-ignored prediction and selection files, because a qualitative example
has to name an instance; the committed reports carry counts, metrics and
fingerprints only.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from construction_safety_vision.final_holdout_evaluation import (
    CLASS_MAP_SHA256,
    CLASSES,
    CONFUSION_MATRIX_CONF,
    CONFUSION_MATRIX_IOU,
    IOU_THRESHOLDS,
    MAX_DETS,
    OBJECT_MATCH_IOU,
    OPERATIONAL_CONF,
    QUALITATIVE_CATEGORIES,
    SEGMENTATION_FAILURE_CATEGORIES,
    TEST_ANNOTATIONS,
    TEST_IMAGES,
    QualitativeCandidate,
    select_qualitative_examples,
)
from construction_safety_vision.paths import ProjectPaths

PHASE = "11B"

TEST_SPLIT = "test"

METRIC_PRECISION = 6

WELL_HANDLED = "WELL_HANDLED_INSTANCE"

TAXONOMY_CASCADE_RESOLUTION: dict[str, object] = {
    "label": "FROZEN_TAXONOMY_AMBIGUOUS_CASCADE_RESOLVED_BEFORE_EXECUTION",
    "resolved_before_any_holdout_number_existed": True,
    "phase_11a_protocol_edited": False,
    "ambiguities": [
        (
            "CLASSIFICATION_MISMATCH is written as 'a prediction reaches box IoU 0.50 but "
            "carries a different class'. Read literally in a first-match-wins cascade it would "
            "also fire on a correctly detected instance that happens to be overlapped at 0.50 "
            "by some other-class prediction."
        ),
        (
            "LOCALIZATION_FAILURE is written as 'a same-class prediction exists and overlaps, "
            "but its box IoU is below 0.50'. Read literally it is unreachable, because "
            "DETECTION_MISS is evaluated first and already claims every instance that no "
            "prediction reaches at 0.50."
        ),
    ],
    "resolution": [
        "DETECTION_MISS: no prediction of any class reaches box IoU 0.50.",
        ("CLASSIFICATION_MISMATCH: something reaches 0.50 but no SAME-CLASS prediction does."),
        (
            "LOCALIZATION_FAILURE: a same-class prediction reaches 0.50, yet this instance "
            "secured no one-to-one match because that prediction was awarded elsewhere."
        ),
        "MASK_QUALITY_FAILURE: matched at box IoU 0.50 or above, mask IoU below 0.50.",
    ],
    "categories_added": 0,
    "why": (
        "A frozen taxonomy that cannot be applied as written is a defect in the taxonomy, not "
        "a licence to change what is measured. The cascade is resolved here in the only "
        "reading under which all four frozen categories are reachable and mutually exclusive, "
        "the resolution is recorded rather than applied silently, no category was added or "
        "removed, and the phase 11A protocol document was not modified."
    ),
}
"""How the frozen four-category cascade was made applicable, decided in advance."""

GAP_RESOLUTION = "PRE_EXECUTION_PROTOCOL_GAP_RESOLUTION"
"""How a frozen confidence with no frozen inference block was resolved."""

DERIVED_FROM_AP_PASS = "DERIVED_FROM_DECLARED_AP_PASS_FILTERED_AT_FROZEN_OPERATING_CONFIDENCE"

QUALITATIVE_DESCENDING: dict[str, bool] = {
    "HIGHEST_CONFIDENCE_FALSE_POSITIVE": True,
    "HIGHEST_CONFIDENCE_CLASSIFICATION_MISMATCH": True,
    "LOWEST_IOU_MATCHED_INSTANCE": False,
    "LARGEST_MISSED_INSTANCE": True,
    "SEGMENTATION_UNDER_COVERAGE": False,
    "SEGMENTATION_OVER_COVERAGE": True,
}
"""The frozen sort direction of every qualitative category."""


class HoldoutExecutionError(RuntimeError):
    """Raised when phase 11B cannot proceed under the frozen protocol."""


class TestDataIntegrityError(HoldoutExecutionError):
    """Raised when the holdout fails its predeclared technical validation."""


class PredictionExecutionError(HoldoutExecutionError):
    """Raised when inference failed before a complete prediction set existed."""


def rounded(value: Any) -> Any:
    """Round a metric to the project's reported precision.

    Args:
        value: A number, or ``None``.

    Returns:
        The rounded float, or ``None`` unchanged.
    """
    if value is None:
        return None
    return round(float(value), METRIC_PRECISION)


def digest_payload(payload: Any) -> str:
    """Hash a payload's semantic content, deterministically.

    Args:
        payload: Any JSON-shaped value.

    Returns:
        A SHA-256 hex digest over its canonical serialisation.
    """
    text = json.dumps(
        json.loads(json.dumps(payload, sort_keys=True, default=str)),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def membership_fingerprint(image_ids: Sequence[str]) -> str:
    """Fingerprint a set of images as a set, not as an ordering.

    Args:
        image_ids: Image identifiers.

    Returns:
        A SHA-256 hex digest over the sorted, newline-joined ids.
    """
    return hashlib.sha256(("\n".join(sorted(image_ids)) + "\n").encode("utf-8")).hexdigest()


# --- the holdout population ------------------------------------------------------------


@dataclass(frozen=True)
class HoldoutPopulation:
    """The materialised holdout, verified against the frozen split.

    Attributes:
        detection: The canonical COCO detection document.
        segmentation: The canonical COCO instance-segmentation document.
        image_order: Deterministic image order.
        files: Image path keyed by source image id.
        sizes: ``(height, width)`` keyed by source image id.
        coco_ids: Canonical COCO image id keyed by source image id.
        class_map: Class name to canonical category id.
        membership_sha256: Digest over the sorted holdout image ids.
        integrity: The predeclared technical validation's outcome.
    """

    detection: Mapping[str, Any]
    segmentation: Mapping[str, Any]
    image_order: tuple[str, ...]
    files: Mapping[str, Path]
    sizes: Mapping[str, tuple[int, int]]
    coco_ids: Mapping[str, int]
    class_map: Mapping[str, int]
    membership_sha256: str
    integrity: Mapping[str, Any]

    @property
    def images(self) -> int:
        """How many holdout images were loaded.

        Returns:
            The image count.
        """
        return len(self.image_order)

    @property
    def annotations(self) -> int:
        """How many canonical holdout annotations were loaded.

        Returns:
            The annotation count.
        """
        return len(self.segmentation["annotations"])


def load_class_map(paths: ProjectPaths) -> dict[str, int]:
    """Read the frozen class map and verify its digest.

    Args:
        paths: Project layout.

    Returns:
        Class name to canonical category id.

    Raises:
        HoldoutExecutionError: If the class map is not the frozen one.
    """
    manifest = json.loads(
        (paths.reports / "task_dataset_manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("class_map_sha256") != CLASS_MAP_SHA256:
        msg = "the committed class map is not the one the frozen protocol names"
        raise HoldoutExecutionError(msg)
    class_map = {str(name): int(index) for name, index in manifest["class_map"].items()}
    if tuple(sorted(class_map, key=lambda name: class_map[name])) != CLASSES:
        msg = "the class map does not carry the five frozen classes in index order"
        raise HoldoutExecutionError(msg)
    return class_map


def build_population(
    paths: ProjectPaths,
    *,
    detection: Mapping[str, Any],
    segmentation: Mapping[str, Any],
    class_map: Mapping[str, int],
    integrity: Mapping[str, Any],
) -> HoldoutPopulation:
    """Assemble the holdout population from its two canonical documents.

    Args:
        paths: Project layout.
        detection: The canonical COCO detection document.
        segmentation: The canonical COCO instance-segmentation document.
        class_map: The frozen class map.
        integrity: The technical validation's outcome.

    Returns:
        The population.

    Raises:
        TestDataIntegrityError: If an image file named by the document is absent.
    """
    entries = sorted(detection["images"], key=lambda entry: str(entry["file_name"]))
    image_root = paths.data_processed / "canonical" / "images" / TEST_SPLIT

    order: list[str] = []
    files: dict[str, Path] = {}
    sizes: dict[str, tuple[int, int]] = {}
    coco_ids: dict[str, int] = {}
    missing: list[str] = []
    for entry in entries:
        stem = Path(str(entry["file_name"])).stem
        candidate = image_root / str(entry["file_name"])
        if not candidate.is_file():
            missing.append(stem)
            continue
        order.append(stem)
        files[stem] = candidate
        sizes[stem] = (int(entry["height"]), int(entry["width"]))
        coco_ids[stem] = int(entry["id"])

    if missing:
        msg = f"{len(missing)} holdout image(s) named by the canonical document are not on disk"
        raise TestDataIntegrityError(msg)

    return HoldoutPopulation(
        detection=detection,
        segmentation=segmentation,
        image_order=tuple(order),
        files=files,
        sizes=sizes,
        coco_ids=coco_ids,
        class_map=dict(class_map),
        membership_sha256=membership_fingerprint(order),
        integrity=dict(integrity),
    )


# --- inference -------------------------------------------------------------------------


@dataclass(frozen=True)
class PredictedInstance:
    """One prediction on the original image canvas.

    Attributes:
        image_id: Source image id.
        class_name: One of the five frozen classes.
        score: The model's confidence.
        box: ``(x1, y1, x2, y2)`` in original image coordinates.
        mask_rle: COCO RLE of the instance mask, or ``None`` for a box-only
            prediction.
    """

    image_id: str
    class_name: str
    score: float
    box: tuple[float, float, float, float]
    mask_rle: Mapping[str, Any] | None = None


def encode_mask(mask: np.ndarray) -> dict[str, Any]:
    """Encode a binary mask as COCO RLE on its own canvas.

    Args:
        mask: A boolean or 0/1 array.

    Returns:
        A COCO RLE with ``size`` and a string ``counts``.
    """
    from pycocotools import mask as mask_utils

    encoded = mask_utils.encode(np.asfortranarray(np.asarray(mask, dtype=np.uint8)))
    counts = encoded["counts"]
    return {
        "size": [int(encoded["size"][0]), int(encoded["size"][1])],
        "counts": counts.decode("utf-8") if isinstance(counts, bytes) else counts,
    }


def decode_mask(rle: Mapping[str, Any]) -> np.ndarray:
    """Decode a COCO RLE back to a boolean mask.

    Args:
        rle: A COCO RLE.

    Returns:
        The boolean mask.
    """
    from pycocotools import mask as mask_utils

    payload: dict[str, Any] = {"size": list(rle["size"]), "counts": rle["counts"]}
    if isinstance(payload["counts"], str):
        payload["counts"] = payload["counts"].encode("utf-8")
    return mask_utils.decode(payload).astype(bool)


def predict(
    checkpoint: Path,
    population: HoldoutPopulation,
    settings: Mapping[str, Any],
    *,
    want_masks: bool,
) -> dict[str, Any]:
    """Run one frozen model over the whole holdout under one frozen block.

    Mirrors the phase 10B inference path exactly, including its refusal to
    resize a mask into agreement with its source image.

    Args:
        checkpoint: The frozen checkpoint to run.
        population: The holdout population.
        settings: The frozen inference block.
        want_masks: Whether to collect instance masks.

    Returns:
        Instances keyed by image id, plus counts and any exclusions.

    Raises:
        PredictionExecutionError: If inference fails on an image, before a
            complete prediction set exists.
    """
    from ultralytics import YOLO

    model = YOLO(str(checkpoint))
    names: dict[int, str] = {}
    per_image: dict[str, list[PredictedInstance]] = {}
    total = 0
    masks_reconstructed = 0
    excluded: list[dict[str, Any]] = []

    for stem in population.image_order:
        height, width = population.sizes[stem]
        call: dict[str, Any] = {
            "source": str(population.files[stem]),
            "imgsz": int(settings["imgsz"]),
            "conf": float(settings["conf"]),
            "iou": float(settings["iou"]),
            "max_det": int(settings["max_det"]),
            "augment": bool(settings["augment"]),
            "agnostic_nms": bool(settings["agnostic_nms"]),
            str(settings["precision_argument"]): settings["precision_value"],
            "verbose": False,
            "save": False,
            "stream": False,
        }
        if want_masks:
            call["retina_masks"] = True
        try:
            outputs = model.predict(**call)
        except Exception as exc:
            msg = f"holdout inference failed before a complete prediction set existed ({exc})"
            raise PredictionExecutionError(msg) from exc

        result = outputs[0]
        if not names:
            names = (
                dict(result.names)
                if isinstance(result.names, dict)
                else dict(enumerate(result.names))
            )
        boxes = result.boxes
        count = 0 if boxes is None else len(boxes)
        masks = None
        if want_masks and result.masks is not None:
            masks = result.masks.data.cpu().numpy().astype(bool)

        instances: list[PredictedInstance] = []
        for index in range(count):
            class_index = int(boxes.cls[index].item())
            score = float(boxes.conf[index].item())
            x1, y1, x2, y2 = (float(value) for value in boxes.xyxy[index].tolist())
            mask_rle: dict[str, Any] | None = None
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
                mask_rle = encode_mask(candidate)
                masks_reconstructed += 1
            instances.append(
                PredictedInstance(
                    image_id=stem,
                    class_name=str(names[class_index]),
                    score=score,
                    box=(x1, y1, x2, y2),
                    mask_rle=mask_rle,
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
        "images": len(population.image_order),
    }


# --- persistence and fingerprints ------------------------------------------------------


def prediction_document(
    predictions: Mapping[str, Any],
    population: HoldoutPopulation,
    *,
    model: Mapping[str, Any],
    protocol_fingerprint: str,
    settings: Mapping[str, Any],
    pass_name: str,
) -> dict[str, Any]:
    """Render one prediction pass as the authoritative persisted document.

    Args:
        predictions: One pass's output.
        population: The holdout population.
        model: The frozen model's identity.
        protocol_fingerprint: The frozen phase 11A protocol fingerprint.
        settings: The inference block that produced it.
        pass_name: Which declared pass this is.

    Returns:
        A JSON-serialisable document carrying every prediction in order.
    """
    images = []
    for stem in population.image_order:
        images.append(
            {
                "image_id": stem,
                "coco_image_id": population.coco_ids[stem],
                "height": population.sizes[stem][0],
                "width": population.sizes[stem][1],
                "predictions": [
                    {
                        "class": instance.class_name,
                        "score": instance.score,
                        "box_xyxy": list(instance.box),
                        "mask_rle": instance.mask_rle,
                    }
                    for instance in predictions["per_image"][stem]
                ],
            }
        )
    return {
        "phase": PHASE,
        "pass": pass_name,
        "split": TEST_SPLIT,
        "model": dict(model),
        "protocol_fingerprint": protocol_fingerprint,
        "split_reference": {
            "holdout_membership_sha256": population.membership_sha256,
            "images": population.images,
            "annotations": population.annotations,
        },
        "inference": dict(settings),
        "counts": {
            "images": predictions["images"],
            "predictions": predictions["total"],
            "masks_reconstructed": predictions["masks_reconstructed"],
            "excluded": len(predictions["excluded"]),
        },
        "excluded": list(predictions["excluded"]),
        "images_with_predictions": images,
    }


def prediction_fingerprint(document: Mapping[str, Any]) -> str:
    """Fingerprint a persisted prediction set.

    Covers model identity, protocol, split reference, inference settings and the
    full ordered set of predictions with class, score and geometry. Excludes
    every machine-specific field, so the same pass fingerprints identically on
    another machine.

    Args:
        document: The persisted prediction document.

    Returns:
        A SHA-256 hex digest.
    """
    scope = {
        "pass": document["pass"],
        "split": document["split"],
        "model": document["model"],
        "protocol_fingerprint": document["protocol_fingerprint"],
        "split_reference": document["split_reference"],
        "inference": document["inference"],
        "predictions": [
            {
                "image_id": entry["image_id"],
                "predictions": entry["predictions"],
            }
            for entry in document["images_with_predictions"]
        ],
    }
    return digest_payload(scope)


def load_predictions(document: Mapping[str, Any]) -> dict[str, list[PredictedInstance]]:
    """Rebuild prediction instances from a persisted document.

    Every metric in this phase is computed from the output of this function, so
    that a report rebuild never needs a model.

    Args:
        document: The persisted prediction document.

    Returns:
        Instances keyed by source image id, in persisted order.
    """
    per_image: dict[str, list[PredictedInstance]] = {}
    for entry in document["images_with_predictions"]:
        stem = str(entry["image_id"])
        per_image[stem] = [
            PredictedInstance(
                image_id=stem,
                class_name=str(record["class"]),
                score=float(record["score"]),
                box=(
                    float(record["box_xyxy"][0]),
                    float(record["box_xyxy"][1]),
                    float(record["box_xyxy"][2]),
                    float(record["box_xyxy"][3]),
                ),
                mask_rle=record.get("mask_rle"),
            )
            for record in entry["predictions"]
        ]
    return per_image


def at_operating_point(
    per_image: Mapping[str, Sequence[PredictedInstance]],
) -> dict[str, list[PredictedInstance]]:
    """Filter a persisted AP prediction set to the frozen operating confidence.

    The protocol gives the confusion matrix, the object-level counts and the
    qualitative ranking a confidence of 0.25 without declaring a separate
    inference block for them. They are therefore derived from the declared AP
    pass, which is the framework's own convention and keeps both models
    symmetric. No threshold is swept and none is chosen with a result in view.

    Args:
        per_image: The AP pass's instances.

    Returns:
        The same instances, keeping only those at or above the frozen operating
        confidence.
    """
    return {
        stem: [instance for instance in instances if instance.score >= OPERATIONAL_CONF]
        for stem, instances in per_image.items()
    }


# --- canonical evaluation --------------------------------------------------------------


def _per_class_ap(evaluator: Any, class_map: Mapping[str, int]) -> dict[str, dict[str, Any]]:
    """Extract per-class AP from an accumulated COCOeval.

    Args:
        evaluator: An accumulated ``COCOeval``.
        class_map: Class name to canonical category id.

    Returns:
        Per-class ``AP@0.50:0.95`` and ``AP@0.50``.
    """
    precision = evaluator.eval["precision"]
    names = {index: name for name, index in class_map.items()}
    per_class: dict[str, dict[str, Any]] = {}
    for position, category_id in enumerate(evaluator.params.catIds):
        name = names.get(int(category_id), str(category_id))
        window = precision[:, :, position, 0, 2]
        at_fifty = precision[0, :, position, 0, 2]
        per_class[name] = {
            "AP@0.50:0.95": rounded(np.mean(window[window > -1])) if (window > -1).any() else None,
            "AP@0.50": rounded(np.mean(at_fifty[at_fifty > -1])) if (at_fifty > -1).any() else None,
        }
    return dict(sorted(per_class.items()))


def _operating_point_counts(
    evaluator: Any, class_map: Mapping[str, int], *, confidence: float
) -> dict[str, Any]:
    """Count TP, FP and FN from the COCOeval accumulation at the frozen point.

    Precision and recall are not threshold-independent, so they are read at the
    project's single operational confidence, at IoU 0.50, over exactly the
    detections the AP figures were computed from.

    Args:
        evaluator: An evaluated ``COCOeval``.
        class_map: Class name to canonical category id.
        confidence: The frozen operating confidence.

    Returns:
        Global and per-class precision, recall and counts.
    """
    names = {index: name for name, index in class_map.items()}
    category_ids = list(evaluator.params.catIds)
    area_ranges = len(evaluator.params.areaRng)
    image_ids = list(evaluator.params.imgIds)

    tally = {name: {"tp": 0, "fp": 0, "gt": 0} for name in names.values()}
    for category_position, category_id in enumerate(category_ids):
        name = names.get(int(category_id), str(category_id))
        base = category_position * area_ranges * len(image_ids)
        for image_position in range(len(image_ids)):
            entry = evaluator.evalImgs[base + image_position]
            if entry is None:
                continue
            gt_ignore = np.asarray(entry["gtIgnore"], dtype=bool)
            tally[name]["gt"] += int(np.count_nonzero(~gt_ignore))
            scores = np.asarray(entry["dtScores"], dtype=float)
            if scores.size == 0:
                continue
            matches = np.asarray(entry["dtMatches"], dtype=float)[0]
            ignored = np.asarray(entry["dtIgnore"], dtype=bool)[0]
            keep = (scores >= confidence) & (~ignored)
            tally[name]["tp"] += int(np.count_nonzero(keep & (matches > 0)))
            tally[name]["fp"] += int(np.count_nonzero(keep & (matches == 0)))

    per_class: dict[str, dict[str, Any]] = {}
    for name in sorted(tally):
        counts = tally[name]
        predicted = counts["tp"] + counts["fp"]
        per_class[name] = {
            "true_positives": counts["tp"],
            "false_positives": counts["fp"],
            "false_negatives": counts["gt"] - counts["tp"],
            "ground_truth": counts["gt"],
            "precision": rounded(counts["tp"] / predicted) if predicted else None,
            "recall": rounded(counts["tp"] / counts["gt"]) if counts["gt"] else None,
        }
    total_tp = sum(counts["tp"] for counts in tally.values())
    total_fp = sum(counts["fp"] for counts in tally.values())
    total_gt = sum(counts["gt"] for counts in tally.values())
    return {
        "label": "CANONICAL_TEST_PRECISION_RECALL_AT_FROZEN_OPERATING_POINT",
        "operating_confidence": confidence,
        "iou_threshold_for_counting": OBJECT_MATCH_IOU,
        "is_threshold_independent": False,
        "tuned_on_test": False,
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_gt - total_tp,
        "ground_truth": total_gt,
        "precision": rounded(total_tp / (total_tp + total_fp)) if (total_tp + total_fp) else None,
        "recall": rounded(total_tp / total_gt) if total_gt else None,
        "per_class": per_class,
    }


def evaluate_canonical(
    ground_truth: Mapping[str, Any],
    detections: Sequence[Mapping[str, Any]],
    *,
    class_map: Mapping[str, int],
    iou_type: str,
) -> dict[str, Any]:
    """Score detections against canonical ground truth with COCO semantics.

    One external evaluator judges both models, because they run through
    different framework validation paths and their native numbers are not
    guaranteed to be computed identically.

    Args:
        ground_truth: The canonical COCO document.
        detections: COCO-format detections.
        class_map: Class name to canonical category id.
        iou_type: ``bbox`` or ``segm``.

    Returns:
        Global and per-class AP, the operating-point counts, and what COCOeval
        actually used.

    Raises:
        HoldoutExecutionError: If COCOeval departs from the frozen semantics.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    names = {index: name for name, index in class_map.items()}
    with contextlib.redirect_stdout(io.StringIO()):
        truth = COCO()
        truth.dataset = json.loads(json.dumps(ground_truth))
        truth.createIndex()
        predicted = truth.loadRes(list(detections)) if detections else None

    if predicted is None:
        empty = {name: {"AP@0.50:0.95": 0.0, "AP@0.50": 0.0} for name in names.values()}
        return {
            "all_class_map50_95": 0.0,
            "all_class_map50": 0.0,
            "per_class": dict(sorted(empty.items())),
            "detections_scored": 0,
            "no_detections": True,
            "operating_point": None,
            "cocoeval": {
                "implementation": "pycocotools.cocoeval.COCOeval",
                "iou_type": iou_type,
                "iou_thresholds": list(IOU_THRESHOLDS),
                "max_dets": list(MAX_DETS),
                "area_range": "all",
            },
        }

    with contextlib.redirect_stdout(io.StringIO()):
        evaluator = COCOeval(truth, predicted, iou_type)
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

    used_thresholds = tuple(round(float(value), 2) for value in evaluator.params.iouThrs)
    used_max_dets = tuple(int(value) for value in evaluator.params.maxDets)
    if used_thresholds != IOU_THRESHOLDS:
        msg = f"COCOeval used IoU thresholds {used_thresholds}, not the frozen {IOU_THRESHOLDS}"
        raise HoldoutExecutionError(msg)
    if used_max_dets != MAX_DETS:
        msg = f"COCOeval used maxDets {used_max_dets}, not the frozen {MAX_DETS}"
        raise HoldoutExecutionError(msg)

    return {
        "all_class_map50_95": rounded(evaluator.stats[0]),
        "all_class_map50": rounded(evaluator.stats[1]),
        "per_class": _per_class_ap(evaluator, class_map),
        "detections_scored": len(detections),
        "no_detections": False,
        "operating_point": _operating_point_counts(
            evaluator, class_map, confidence=OPERATIONAL_CONF
        ),
        "cocoeval": {
            "implementation": "pycocotools.cocoeval.COCOeval",
            "iou_type": iou_type,
            "iou_thresholds": [float(value) for value in used_thresholds],
            "max_dets": list(used_max_dets),
            "area_range": "all",
        },
    }


def box_detections(
    per_image: Mapping[str, Sequence[PredictedInstance]],
    population: HoldoutPopulation,
) -> list[dict[str, Any]]:
    """Build COCO bbox detections from each model's own predicted boxes.

    Never derived from a mask: re-deriving the segmenter's boxes from its masks
    would measure a post-processing choice this project invented rather than the
    model.

    Args:
        per_image: Persisted instances keyed by image id.
        population: The holdout population.

    Returns:
        COCO-format bbox detections.
    """
    detections: list[dict[str, Any]] = []
    for stem in population.image_order:
        image_id = population.coco_ids[stem]
        for instance in per_image.get(stem, ()):
            x1, y1, x2, y2 = instance.box
            detections.append(
                {
                    "image_id": image_id,
                    "category_id": int(population.class_map[instance.class_name]),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": instance.score,
                }
            )
    return detections


def mask_detections(
    per_image: Mapping[str, Sequence[PredictedInstance]],
    population: HoldoutPopulation,
) -> list[dict[str, Any]]:
    """Build COCO segm detections from persisted instance masks.

    Args:
        per_image: Persisted instances keyed by image id.
        population: The holdout population.

    Returns:
        COCO-format segm detections.

    Raises:
        HoldoutExecutionError: If a persisted mask is not on its image's canvas.
    """
    detections: list[dict[str, Any]] = []
    for stem in population.image_order:
        image_id = population.coco_ids[stem]
        height, width = population.sizes[stem]
        for instance in per_image.get(stem, ()):
            if instance.mask_rle is None:
                continue
            size = [int(value) for value in instance.mask_rle["size"]]
            if size != [height, width]:
                msg = (
                    f"a persisted mask sits on canvas {size} rather than the original "
                    f"[{height}, {width}]; predictions and canonical masks must share it"
                )
                raise HoldoutExecutionError(msg)
            detections.append(
                {
                    "image_id": image_id,
                    "category_id": int(population.class_map[instance.class_name]),
                    "segmentation": dict(instance.mask_rle),
                    "score": instance.score,
                }
            )
    return detections


def support(population: HoldoutPopulation) -> dict[str, dict[str, Any]]:
    """Apply the frozen phase 7A support rule to the holdout's own support.

    The rule is reused unchanged and names no class; the admitted set is its
    output. It is applied only after the evaluation and decides nothing.

    Args:
        population: The holdout population.

    Returns:
        Per class, its counts and whether the rule admits it.
    """
    names = {index: name for name, index in population.class_map.items()}
    counts: dict[str, dict[str, Any]] = {
        name: {"images": set(), "instances": 0} for name in population.class_map
    }
    for annotation in population.segmentation["annotations"]:
        name = names[int(annotation["category_id"])]
        counts[name]["instances"] += 1
        counts[name]["images"].add(int(annotation["image_id"]))
    resolved: dict[str, dict[str, Any]] = {}
    for name in sorted(counts):
        images = len(counts[name]["images"])
        instances = int(counts[name]["instances"])
        resolved[name] = {
            "images": images,
            "instances": instances,
            "supported": images >= 5 and instances >= 20,
            "status": "COMPARISON_SUPPORTED"
            if (images >= 5 and instances >= 20)
            else "DESCRIPTIVE_HIGH_UNCERTAINTY",
        }
    return resolved


def supported_macro(
    per_class: Mapping[str, Mapping[str, Any]], admitted: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Average per-class AP over the classes the frozen rule admits.

    Args:
        per_class: Canonical per-class AP.
        admitted: The support rule's verdict per class.

    Returns:
        The metric, the admitted classes and the arithmetic behind it.
    """
    names = [name for name in sorted(admitted) if admitted[name]["supported"]]
    values = [
        float(per_class[name]["AP@0.50:0.95"])
        for name in names
        if name in per_class and per_class[name]["AP@0.50:0.95"] is not None
    ]
    return {
        "value": rounded(sum(values) / len(values)) if values else None,
        "admitted_classes": names,
        "contributing_values": [rounded(value) for value in values],
        "rule": "COMPARISON_SUPPORTED requires >= 5 positive images AND >= 20 instances",
        "rule_origin": "PREEXISTING_PHASE_7A_SUPPORT_RULE_REUSED_UNCHANGED",
        "names_no_class": True,
        "role_on_test": "DESCRIPTIVE_CAVEAT_ONLY_NEVER_A_SELECTION_RULE",
    }


# --- the frozen confusion matrix -------------------------------------------------------


def confusion_matrix(
    per_image: Mapping[str, Sequence[PredictedInstance]],
    population: HoldoutPopulation,
) -> dict[str, Any]:
    """Compute the frozen framework confusion matrix over predicted boxes.

    The semantics were read from the installed framework in phase 11A and are
    applied here unchanged: confidence 0.25, IoU 0.45, class-agnostic matching
    with the class pair then recorded, and a ``(nc+1, nc+1)`` matrix with rows
    predicted and columns ground truth.

    Args:
        per_image: Persisted instances keyed by image id.
        population: The holdout population.

    Returns:
        The matrix, its normalised variant and the settings used.
    """
    import torch
    from ultralytics.utils.metrics import ConfusionMatrix

    names = {int(index): str(name) for name, index in population.class_map.items()}
    matrix = ConfusionMatrix(names=dict(sorted(names.items())), task="detect")

    truth: dict[str, list[tuple[int, list[float]]]] = {stem: [] for stem in population.image_order}
    by_coco_id = {value: key for key, value in population.coco_ids.items()}
    for annotation in population.detection["annotations"]:
        stem = by_coco_id[int(annotation["image_id"])]
        x, y, width, height = (float(value) for value in annotation["bbox"])
        truth[stem].append((int(annotation["category_id"]), [x, y, x + width, y + height]))

    for stem in population.image_order:
        instances = list(per_image.get(stem, ()))
        detections = {
            "cls": torch.tensor(
                [population.class_map[instance.class_name] for instance in instances],
                dtype=torch.float32,
            ),
            "conf": torch.tensor([instance.score for instance in instances], dtype=torch.float32),
            "bboxes": torch.tensor(
                [list(instance.box) for instance in instances], dtype=torch.float32
            ).reshape(-1, 4),
        }
        batch = {
            "cls": torch.tensor([entry[0] for entry in truth[stem]], dtype=torch.float32),
            "bboxes": torch.tensor(
                [entry[1] for entry in truth[stem]], dtype=torch.float32
            ).reshape(-1, 4),
        }
        matrix.process_batch(
            detections, batch, conf=CONFUSION_MATRIX_CONF, iou_thres=CONFUSION_MATRIX_IOU
        )

    raw = matrix.matrix
    columns = raw.sum(0).reshape(1, -1)
    normalised = np.divide(raw, columns, out=np.zeros_like(raw, dtype=float), where=columns > 0)
    labels = [*[names[index] for index in sorted(names)], "background"]
    return {
        "implementation": "ultralytics.utils.metrics.ConfusionMatrix",
        "implementation_version": "ultralytics==8.4.138",
        "task": "detect",
        "conf": CONFUSION_MATRIX_CONF,
        "iou_threshold": CONFUSION_MATRIX_IOU,
        "matching": "CLASS_AGNOSTIC_IOU_THEN_CLASS_PAIR_RECORDED",
        "orientation": "ROWS_ARE_PREDICTED_COLUMNS_ARE_GROUND_TRUTH",
        "labels": labels,
        "matrix": [[int(value) for value in row] for row in raw],
        "normalised": [[rounded(value) for value in row] for row in normalised],
        "threshold_chosen_after_seeing_test": False,
        "_matrix_object": matrix,
    }


# --- object-level TP / FP / FN and the frozen failure taxonomy -------------------------


def box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    """Intersection over union of two axis-aligned boxes.

    Args:
        first: ``(x1, y1, x2, y2)``.
        second: ``(x1, y1, x2, y2)``.

    Returns:
        The IoU in ``[0, 1]``.
    """
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = width * height
    if intersection <= 0.0:
        return 0.0
    union = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    union += max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union -= intersection
    return intersection / union if union > 0 else 0.0


@dataclass(frozen=True)
class GroundTruthInstance:
    """One canonical holdout instance.

    Attributes:
        annotation_id: Canonical annotation id.
        image_id: Source image id.
        class_name: Its class.
        box: ``(x1, y1, x2, y2)`` derived from its segmentation.
        mask: Its canonical mask on the original canvas, or ``None``.
    """

    annotation_id: int
    image_id: str
    class_name: str
    box: tuple[float, float, float, float]
    mask: np.ndarray | None = None


def canonical_instances(
    population: HoldoutPopulation, *, with_masks: bool
) -> dict[str, list[GroundTruthInstance]]:
    """Decode the canonical holdout instances.

    Args:
        population: The holdout population.
        with_masks: Whether to decode instance masks as well as boxes.

    Returns:
        Instances keyed by source image id, in canonical annotation-id order.
    """
    from pycocotools import mask as mask_utils

    names = {index: name for name, index in population.class_map.items()}
    by_coco_id = {value: key for key, value in population.coco_ids.items()}
    boxes = {
        int(annotation["id"]): [float(value) for value in annotation["bbox"]]
        for annotation in population.detection["annotations"]
    }

    instances: dict[str, list[GroundTruthInstance]] = {stem: [] for stem in population.image_order}
    for annotation in sorted(
        population.segmentation["annotations"], key=lambda entry: int(entry["id"])
    ):
        stem = by_coco_id[int(annotation["image_id"])]
        height, width = population.sizes[stem]
        mask = None
        if with_masks:
            segmentation = annotation["segmentation"]
            if isinstance(segmentation, list):
                rle = mask_utils.merge(mask_utils.frPyObjects(segmentation, height, width))
            elif isinstance(segmentation.get("counts"), list):
                rle = mask_utils.frPyObjects(segmentation, height, width)
            else:
                rle = dict(segmentation)
                if isinstance(rle["counts"], str):
                    rle["counts"] = rle["counts"].encode("utf-8")
            decoded = mask_utils.decode(rle).astype(bool)
            mask = decoded.any(axis=2) if decoded.ndim == 3 else decoded
        x, y, box_width, box_height = boxes[int(annotation["id"])]
        instances[stem].append(
            GroundTruthInstance(
                annotation_id=int(annotation["id"]),
                image_id=stem,
                class_name=names[int(annotation["category_id"])],
                box=(x, y, x + box_width, y + box_height),
                mask=mask,
            )
        )
    return instances


@dataclass
class ObjectOutcome:
    """One canonical instance's object-level outcome.

    Attributes:
        instance: The canonical instance.
        prediction_index: Index of the prediction matched to it, or ``-1``.
        box_iou: Its matched box IoU, or ``0.0``.
        mask_iou: Its matched mask IoU, or ``None`` when no mask exists.
        category: Its frozen failure category, or ``WELL_HANDLED_INSTANCE``.
    """

    instance: GroundTruthInstance
    prediction_index: int = -1
    box_iou: float = 0.0
    mask_iou: float | None = None
    category: str = WELL_HANDLED


def object_level_outcomes(
    per_image: Mapping[str, Sequence[PredictedInstance]],
    truth: Mapping[str, Sequence[GroundTruthInstance]],
    population: HoldoutPopulation,
    *,
    with_predicted_masks: bool,
) -> dict[str, Any]:
    """Apply the frozen object-level matching and failure taxonomy.

    Class-aware, one-to-one per image and class, at IoU 0.50, over predictions
    at the frozen operating confidence. Predictions are consumed in descending
    confidence and then highest IoU, exactly as the protocol declares.

    Args:
        per_image: Operating-point instances keyed by image id.
        truth: Canonical instances keyed by image id.
        population: The holdout population.
        with_predicted_masks: Whether mask-quality failures can be evaluated.

    Returns:
        The counts, the outcomes and the taxonomy census.
    """
    outcomes: list[ObjectOutcome] = []
    false_positives: list[dict[str, Any]] = []

    for stem in population.image_order:
        predictions = list(per_image.get(stem, ()))
        order = sorted(
            range(len(predictions)),
            key=lambda index: (-predictions[index].score, index),
        )
        instances = list(truth.get(stem, ()))
        records = [ObjectOutcome(instance=instance) for instance in instances]
        claimed: set[int] = set()

        for prediction_index in order:
            prediction = predictions[prediction_index]
            best_position = -1
            best_iou = 0.0
            for position, record in enumerate(records):
                if position in claimed:
                    continue
                if record.instance.class_name != prediction.class_name:
                    continue
                value = box_iou(record.instance.box, prediction.box)
                if value >= OBJECT_MATCH_IOU and value > best_iou:
                    best_position, best_iou = position, value
            if best_position >= 0:
                claimed.add(best_position)
                records[best_position].prediction_index = prediction_index
                records[best_position].box_iou = best_iou
            else:
                false_positives.append(
                    {
                        "image_id": stem,
                        "class": prediction.class_name,
                        "score": prediction.score,
                        "prediction_index": prediction_index,
                    }
                )

        for record in records:
            record.category = _classify(
                record, predictions, with_predicted_masks=with_predicted_masks
            )
        outcomes.extend(records)

    census = dict.fromkeys((*SEGMENTATION_FAILURE_CATEGORIES, WELL_HANDLED), 0)
    for outcome in outcomes:
        census[outcome.category] += 1

    true_positives = sum(1 for outcome in outcomes if outcome.prediction_index >= 0)
    return {
        "class_aware": True,
        "iou_threshold": OBJECT_MATCH_IOU,
        "confidence_threshold": OPERATIONAL_CONF,
        "assignment": "ONE_TO_ONE_PER_IMAGE_PER_CLASS",
        "assignment_algorithm": "GREEDY_DESCENDING_CONFIDENCE_THEN_HIGHEST_IOU",
        "true_positives": true_positives,
        "false_positives": len(false_positives),
        "false_negatives": len(outcomes) - true_positives,
        "ground_truth": len(outcomes),
        "precision": rounded(true_positives / (true_positives + len(false_positives)))
        if (true_positives + len(false_positives))
        else None,
        "recall": rounded(true_positives / len(outcomes)) if outcomes else None,
        "taxonomy_census": census,
        "taxonomy_partitions_every_instance": False,
        "unclassified_label": WELL_HANDLED,
        "taxonomy_cascade_resolution": TAXONOMY_CASCADE_RESOLUTION,
        "_outcomes": outcomes,
        "_false_positives": false_positives,
    }


def _classify(
    record: ObjectOutcome,
    predictions: Sequence[PredictedInstance],
    *,
    with_predicted_masks: bool,
) -> str:
    """Assign one canonical instance its frozen failure category.

    Categories are evaluated in the frozen order and the first that matches
    wins. They deliberately do not partition every instance; the remainder is
    ``WELL_HANDLED_INSTANCE``.

    Args:
        record: The instance's outcome so far.
        predictions: That image's operating-point predictions.
        with_predicted_masks: Whether mask-quality failures can be evaluated.

    Returns:
        The frozen category name.
    """
    overlapping = [
        (index, box_iou(record.instance.box, prediction.box))
        for index, prediction in enumerate(predictions)
    ]
    any_class_at_threshold = [index for index, value in overlapping if value >= OBJECT_MATCH_IOU]
    if not any_class_at_threshold:
        return "DETECTION_MISS"
    same_class_at_threshold = [
        index
        for index in any_class_at_threshold
        if predictions[index].class_name == record.instance.class_name
    ]
    if not same_class_at_threshold:
        return "CLASSIFICATION_MISMATCH"
    if record.prediction_index < 0:
        return "LOCALIZATION_FAILURE"
    if with_predicted_masks and record.instance.mask is not None:
        mask_rle = predictions[record.prediction_index].mask_rle
        if mask_rle is not None:
            value = _mask_iou(record.instance.mask, decode_mask(mask_rle))
            record.mask_iou = value
            if value < OBJECT_MATCH_IOU:
                return "MASK_QUALITY_FAILURE"
    return WELL_HANDLED


def _mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    """Intersection over union of two boolean masks.

    Args:
        first: A boolean mask.
        second: A boolean mask of the same shape.

    Returns:
        The IoU in ``[0, 1]``.
    """
    intersection = int(np.count_nonzero(first & second))
    if intersection == 0:
        return 0.0
    return intersection / int(np.count_nonzero(first | second))


# --- deterministic qualitative selection -----------------------------------------------


def qualitative_candidates(
    model: str,
    objects: Mapping[str, Any],
    per_image: Mapping[str, Sequence[PredictedInstance]],
    truth: Mapping[str, Sequence[GroundTruthInstance]],
    *,
    with_masks: bool,
) -> list[QualitativeCandidate]:
    """Build every candidate for the frozen qualitative categories.

    No image is opened here. The ranking is computed from the persisted
    predictions and the canonical annotations alone, and only the instances it
    selects are ever rendered.

    Args:
        model: Which model the candidates belong to.
        objects: That model's object-level outcome payload.
        per_image: Operating-point instances keyed by image id.
        truth: Canonical instances keyed by image id.
        with_masks: Whether the two segmentation categories apply.

    Returns:
        Every qualifying candidate, in any order.
    """
    _ = model, truth
    candidates: list[QualitativeCandidate] = []

    for record in objects["_false_positives"]:
        candidates.append(
            QualitativeCandidate(
                category="HIGHEST_CONFIDENCE_FALSE_POSITIVE",
                rank_value=float(record["score"]),
                canonical_annotation_id=-1,
                canonical_prediction_index=int(record["prediction_index"]),
                source_image_id=str(record["image_id"]),
            )
        )

    for outcome in objects["_outcomes"]:
        stem = outcome.instance.image_id
        predictions = list(per_image.get(stem, ()))
        if outcome.category == "CLASSIFICATION_MISMATCH":
            best_index, best_score = -1, -1.0
            for index, prediction in enumerate(predictions):
                if box_iou(outcome.instance.box, prediction.box) < OBJECT_MATCH_IOU:
                    continue
                if prediction.score > best_score:
                    best_index, best_score = index, prediction.score
            if best_index >= 0:
                candidates.append(
                    QualitativeCandidate(
                        category="HIGHEST_CONFIDENCE_CLASSIFICATION_MISMATCH",
                        rank_value=float(best_score),
                        canonical_annotation_id=outcome.instance.annotation_id,
                        canonical_prediction_index=best_index,
                        source_image_id=stem,
                    )
                )
        if outcome.prediction_index >= 0:
            candidates.append(
                QualitativeCandidate(
                    category="LOWEST_IOU_MATCHED_INSTANCE",
                    rank_value=float(outcome.box_iou),
                    canonical_annotation_id=outcome.instance.annotation_id,
                    canonical_prediction_index=outcome.prediction_index,
                    source_image_id=stem,
                )
            )
        if outcome.category == "DETECTION_MISS":
            area = (
                float(np.count_nonzero(outcome.instance.mask))
                if outcome.instance.mask is not None
                else _box_area(outcome.instance.box)
            )
            candidates.append(
                QualitativeCandidate(
                    category="LARGEST_MISSED_INSTANCE",
                    rank_value=area,
                    canonical_annotation_id=outcome.instance.annotation_id,
                    canonical_prediction_index=-1,
                    source_image_id=stem,
                )
            )
        if not with_masks or outcome.prediction_index < 0:
            continue
        if outcome.instance.mask is None:
            continue
        mask_rle = predictions[outcome.prediction_index].mask_rle
        if mask_rle is None:
            continue
        canonical_area = int(np.count_nonzero(outcome.instance.mask))
        if canonical_area == 0:
            continue
        ratio = float(np.count_nonzero(decode_mask(mask_rle))) / canonical_area
        for category in ("SEGMENTATION_UNDER_COVERAGE", "SEGMENTATION_OVER_COVERAGE"):
            candidates.append(
                QualitativeCandidate(
                    category=category,
                    rank_value=ratio,
                    canonical_annotation_id=outcome.instance.annotation_id,
                    canonical_prediction_index=outcome.prediction_index,
                    source_image_id=stem,
                )
            )
    return candidates


def _box_area(box: Sequence[float]) -> float:
    """Area of an axis-aligned box.

    Args:
        box: ``(x1, y1, x2, y2)``.

    Returns:
        The area in square pixels, never negative.
    """
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def qualitative_selection(
    candidates: Sequence[QualitativeCandidate], *, applicable: Mapping[str, bool]
) -> dict[str, Any]:
    """Run the frozen deterministic selection and record every shortfall.

    Args:
        candidates: Every qualifying candidate.
        applicable: Per category, whether it applies to this model at all.

    Returns:
        The selection, its shortfalls and the frozen rule that produced it.
    """
    selected = select_qualitative_examples(candidates, descending=QUALITATIVE_DESCENDING)
    payload: dict[str, Any] = {}
    shortfalls: dict[str, Any] = {}
    for category in QUALITATIVE_CATEGORIES:
        if not applicable.get(category, True):
            payload[category] = {"applicable": False, "selected": []}
            continue
        chosen = selected.get(category, [])
        available = sum(1 for candidate in candidates if candidate.category == category)
        payload[category] = {
            "applicable": True,
            "available": available,
            "selected": [
                {
                    "rank": position,
                    "rank_value": rounded(candidate.rank_value),
                    "canonical_annotation_id": candidate.canonical_annotation_id,
                    "canonical_prediction_index": candidate.canonical_prediction_index,
                    "source_image_id": candidate.source_image_id,
                }
                for position, candidate in enumerate(chosen, start=1)
            ],
        }
        if len(chosen) < 3:
            shortfalls[category] = {"selected": len(chosen), "quota": 3, "available": available}
    return {
        "policy": "DETERMINISTIC_RANKING_FROZEN_BEFORE_ANY_TEST_ACCESS",
        "examples_per_category": 3,
        "manual_cherry_picking_permitted": False,
        "images_inspected_to_design_this_rule": 0,
        "images_browsed_before_selection": 0,
        "topping_up_from_another_category_permitted": False,
        "categories": payload,
        "shortfalls": shortfalls,
    }


# --- the phase 8C direct instance-mask IoU diagnostic ----------------------------------


def direct_mask_iou(
    per_image: Mapping[str, Sequence[PredictedInstance]],
    truth: Mapping[str, Sequence[GroundTruthInstance]],
    population: HoldoutPopulation,
) -> dict[str, Any]:
    """Run the frozen phase 8C diagnostic, unchanged, on the holdout.

    The matching rule, the operating confidence and the ground-truth source are
    phase 8C's. No new rule is invented for the holdout, because inventing one
    now would let it be chosen with the final result in view.

    Args:
        per_image: The segmenter's operational instances keyed by image id.
        truth: Canonical instances keyed by image id, with masks decoded.
        population: The holdout population.

    Returns:
        The global and per-class diagnostics, with their frozen labels.
    """
    from construction_safety_vision.mask_iou_evaluation import MaskInstance, evaluate

    samples = []
    for stem in population.image_order:
        ground_truth = [
            MaskInstance(
                class_id=int(population.class_map[instance.class_name]), mask=instance.mask
            )
            for instance in truth.get(stem, ())
            if instance.mask is not None
        ]
        predictions = [
            MaskInstance(
                class_id=int(population.class_map[instance.class_name]),
                mask=decode_mask(instance.mask_rle),
            )
            for instance in per_image.get(stem, ())
            if instance.mask_rle is not None
        ]
        samples.append((stem, ground_truth, predictions))

    names = {index: name for name, index in population.class_map.items()}
    result = evaluate(samples).as_dict(names)
    result["role"] = "SECONDARY_CANONICAL_DIAGNOSTIC"
    result["is_primary_segmenter_metric"] = False
    result["is_an_average_precision"] = False
    result["ground_truth_source"] = "CANONICAL_COCO_INSTANCE_SEGMENTATION"
    result["headlines_are_not_interchangeable"] = (
        "matched_mask_iou_mean describes mask quality where the model produced an overlapping "
        "same-class instance; gt_normalized_mask_iou divides the same IoU sum by every "
        "canonical instance, so misses lower it. Neither is a COCO AP."
    )
    return result


# --- artifact builders ------------------------------------------------------------------


def strip_private(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Drop the in-memory helpers a result payload carries.

    Args:
        payload: A result mapping.

    Returns:
        The same mapping without any key beginning with an underscore.
    """
    return {key: value for key, value in payload.items() if not str(key).startswith("_")}


IDENTIFIER_KEYS: frozenset[str] = frozenset(
    {
        "image_id",
        "source_image_id",
        "coco_image_id",
        "canonical_annotation_id",
        "canonical_prediction_index",
        "prediction_index",
        "images_with_predictions",
        "excluded",
    }
)
"""Keys whose values identify a holdout image or one of its instances."""


def redact_identifiers(payload: Any) -> Any:
    """Remove every holdout identifier from a payload bound for the repository.

    A committed artifact may carry counts, metrics and fingerprints. It may not
    carry the identity of a holdout image or of the instances inside it, because
    publishing those is a leak even when no pixel is published with them.

    Args:
        payload: Any JSON-shaped value.

    Returns:
        The payload with every identifier key removed.
    """
    if isinstance(payload, Mapping):
        return {
            key: redact_identifiers(value)
            for key, value in payload.items()
            if key not in IDENTIFIER_KEYS
        }
    if isinstance(payload, list):
        return [redact_identifiers(value) for value in payload]
    return payload


def build_detector_result(
    *,
    model: Mapping[str, Any],
    protocol_fingerprint: str,
    prediction_sha256: str,
    population: HoldoutPopulation,
    boxes: Mapping[str, Any],
    admitted: Mapping[str, Mapping[str, Any]],
    objects: Mapping[str, Any],
    matrix: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble the detector's committed holdout result.

    Args:
        model: The frozen detector's identity.
        protocol_fingerprint: The frozen phase 11A protocol fingerprint.
        prediction_sha256: The detector's prediction fingerprint.
        population: The holdout population.
        boxes: The canonical bbox evaluation.
        admitted: The support rule's verdict on the holdout's own support.
        objects: The object-level outcome payload.
        matrix: The frozen confusion matrix.

    Returns:
        The result payload, free of holdout identifiers.
    """
    payload = {
        "schema_version": 1,
        "phase": PHASE,
        "protocol": "FINAL_HOLDOUT_ONE_SHOT_EVALUATION",
        "protocol_fingerprint": protocol_fingerprint,
        "split": TEST_SPLIT,
        "split_status": "OBSERVED_ONCE_FINAL",
        "model": dict(model),
        "prediction_fingerprint": prediction_sha256,
        "metrics_derived_from": "PERSISTED_PREDICTIONS_ONLY",
        "model_invoked_during_metric_computation": False,
        "population": {
            "images": population.images,
            "annotations": population.annotations,
            "membership_sha256": population.membership_sha256,
            "evaluated": "ALL_FROZEN_TEST_IMAGES",
            "sampling_permitted": False,
            "exclusions_permitted": False,
        },
        "canonical_box": {
            "CANONICAL_TEST_BOX_MAP50_95": boxes["all_class_map50_95"],
            "CANONICAL_TEST_BOX_MAP50": boxes["all_class_map50"],
            "per_class": boxes["per_class"],
            "detections_scored": boxes["detections_scored"],
            "evaluator": boxes["cocoeval"],
        },
        "canonical_precision_recall": strip_private(boxes["operating_point"]),
        "support_on_holdout": dict(admitted),
        "supported_macro": supported_macro(boxes["per_class"], admitted),
        "object_level": strip_private(objects),
        "confusion_matrix": strip_private(matrix),
        "threshold_tuned_on_test": False,
        "model_selection_follows": False,
        "winner_declared": False,
        "composite_score": False,
    }
    return redact_identifiers(payload)


def build_segmenter_result(
    *,
    model: Mapping[str, Any],
    protocol_fingerprint: str,
    prediction_sha256: str,
    operational_prediction_sha256: str,
    population: HoldoutPopulation,
    masks: Mapping[str, Any],
    boxes: Mapping[str, Any],
    detector_boxes: Mapping[str, Any],
    admitted: Mapping[str, Mapping[str, Any]],
    objects: Mapping[str, Any],
    matrix: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble the segmenter's committed holdout result.

    Args:
        model: The frozen segmenter's identity.
        protocol_fingerprint: The frozen phase 11A protocol fingerprint.
        prediction_sha256: The segmenter's AP prediction fingerprint.
        operational_prediction_sha256: Its operational prediction fingerprint.
        population: The holdout population.
        masks: The canonical mask evaluation.
        boxes: The canonical bbox evaluation of its own predicted boxes.
        detector_boxes: The detector's canonical bbox evaluation.
        admitted: The support rule's verdict on the holdout's own support.
        objects: The object-level outcome payload.
        matrix: The frozen confusion matrix.

    Returns:
        The result payload, free of holdout identifiers.
    """
    delta = None
    if boxes["all_class_map50_95"] is not None and detector_boxes["all_class_map50_95"] is not None:
        delta = rounded(boxes["all_class_map50_95"] - detector_boxes["all_class_map50_95"])
    delta50 = None
    if boxes["all_class_map50"] is not None and detector_boxes["all_class_map50"] is not None:
        delta50 = rounded(boxes["all_class_map50"] - detector_boxes["all_class_map50"])

    payload = {
        "schema_version": 1,
        "phase": PHASE,
        "protocol": "FINAL_HOLDOUT_ONE_SHOT_EVALUATION",
        "protocol_fingerprint": protocol_fingerprint,
        "split": TEST_SPLIT,
        "split_status": "OBSERVED_ONCE_FINAL",
        "model": dict(model),
        "prediction_fingerprint": prediction_sha256,
        "operational_prediction_fingerprint": operational_prediction_sha256,
        "metrics_derived_from": "PERSISTED_PREDICTIONS_ONLY",
        "model_invoked_during_metric_computation": False,
        "population": {
            "images": population.images,
            "annotations": population.annotations,
            "membership_sha256": population.membership_sha256,
            "evaluated": "ALL_FROZEN_TEST_IMAGES",
        },
        "canonical_mask": {
            "CANONICAL_TEST_MASK_MAP50_95": masks["all_class_map50_95"],
            "CANONICAL_TEST_MASK_MAP50": masks["all_class_map50"],
            "per_class": masks["per_class"],
            "detections_scored": masks["detections_scored"],
            "evaluator": masks["cocoeval"],
        },
        "canonical_mask_precision_recall": strip_private(masks["operating_point"]),
        "canonical_box": {
            "S1_CANONICAL_TEST_BOX_MAP50_95": boxes["all_class_map50_95"],
            "S1_CANONICAL_TEST_BOX_MAP50": boxes["all_class_map50"],
            "per_class": boxes["per_class"],
            "detections_scored": boxes["detections_scored"],
            "evaluator": boxes["cocoeval"],
            "boxes_source": "EACH_MODELS_OWN_PREDICTED_BOXES",
            "boxes_derived_from_masks": False,
        },
        "canonical_box_precision_recall": strip_private(boxes["operating_point"]),
        "localisation_comparison_with_detector": {
            "label": "DESCRIPTIVE_ONLY",
            "detector_map50_95": detector_boxes["all_class_map50_95"],
            "segmenter_map50_95": boxes["all_class_map50_95"],
            "delta_map50_95": delta,
            "detector_map50": detector_boxes["all_class_map50"],
            "segmenter_map50": boxes["all_class_map50"],
            "delta_map50": delta50,
            "winner_declared": False,
            "composite_score": False,
            "model_selection_follows": False,
            "significance_test": False,
            "note": (
                "One evaluator over one ground truth. The two models do not solve the same "
                "output task, so this is a description of localisation, never a ranking."
            ),
        },
        "supported_macro_mask": supported_macro(masks["per_class"], admitted),
        "supported_macro_box": supported_macro(boxes["per_class"], admitted),
        "support_on_holdout": dict(admitted),
        "object_level": strip_private(objects),
        "confusion_matrix": strip_private(matrix),
        "mask_and_box_never_merged": True,
        "threshold_tuned_on_test": False,
        "winner_declared": False,
        "composite_score": False,
    }
    return redact_identifiers(payload)


def build_direct_iou_result(
    *,
    model: Mapping[str, Any],
    protocol_fingerprint: str,
    direct_iou_protocol_fingerprint: str,
    prediction_sha256: str,
    population: HoldoutPopulation,
    diagnostic: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble the committed direct-IoU diagnostic result.

    Args:
        model: The frozen segmenter's identity.
        protocol_fingerprint: The frozen phase 11A protocol fingerprint.
        direct_iou_protocol_fingerprint: Phase 8C's frozen fingerprint.
        prediction_sha256: The operational prediction fingerprint.
        population: The holdout population.
        diagnostic: The diagnostic's output.

    Returns:
        The result payload, free of holdout identifiers.
    """
    payload = {
        "schema_version": 1,
        "phase": PHASE,
        "protocol": "DIRECT_INSTANCE_MASK_IOU_DIAGNOSTIC",
        "protocol_fingerprint": protocol_fingerprint,
        "direct_iou_protocol_fingerprint": direct_iou_protocol_fingerprint,
        "unchanged_from_phase_8c": True,
        "new_rule_invented": False,
        "split": TEST_SPLIT,
        "model": dict(model),
        "prediction_fingerprint": prediction_sha256,
        "metrics_derived_from": "PERSISTED_PREDICTIONS_ONLY",
        "population": {
            "images": population.images,
            "annotations": population.annotations,
            "membership_sha256": population.membership_sha256,
        },
        "inference": {
            "imgsz": 768,
            "conf": OPERATIONAL_CONF,
            "iou": 0.70,
            "max_det": 300,
            "retina_masks": True,
            "augment": False,
            "agnostic_nms": False,
            "half": False,
        },
        "matching": {
            "algorithm": "SCIPY_LINEAR_SUM_ASSIGNMENT_MAXIMIZE_MASK_IOU",
            "scope": "PER_IMAGE_PER_CLASS_ONE_TO_ONE",
            "zero_overlap_policy": "ASSIGNED_PAIRS_WITH_ZERO_IOU_ARE_NOT_MATCHES",
            "unmatched_gt_policy": "CONTRIBUTES_ZERO_TO_GT_NORMALIZED_MASK_IOU",
            "deterministic": True,
        },
        "diagnostic": dict(diagnostic),
        "combined_score": False,
        "threshold_swept": False,
    }
    return redact_identifiers(payload)


def result_fingerprint(payload: Mapping[str, Any]) -> str:
    """Fingerprint a committed result document.

    Covers model identity, the split reference, the evaluator protocol, the
    inference settings, the metric outputs and the corresponding prediction
    fingerprint. Excludes everything machine-specific.

    Args:
        payload: The result payload.

    Returns:
        A SHA-256 hex digest.
    """
    return digest_payload(payload)


# --- loading the holdout, once ----------------------------------------------------------


MATERIALIZER = "scripts/materialize_task_datasets.py"
"""The phase 5D script. The holdout runs through its function, not a copy."""


def _materializer(paths: ProjectPaths) -> Any:
    """Import the phase 5D materialisation script as a module.

    The holdout must be written by the same function that wrote ``train`` and
    ``validation``, so the script is imported rather than reimplemented. A second
    copy of that logic could drift and give the protected split different
    treatment.

    Args:
        paths: Project layout.

    Returns:
        The imported module.
    """
    import importlib.util

    source = paths.root / MATERIALIZER
    spec = importlib.util.spec_from_file_location("_phase_5d_materializer", source)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        msg = f"could not import {MATERIALIZER}"
        raise HoldoutExecutionError(msg)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_holdout(
    paths: ProjectPaths, *, allow_test: bool, env: dict[str, str] | None = None
) -> HoldoutPopulation:
    """Materialise and load the holdout, once, through the frozen guard.

    The images and both canonical documents are produced by the phase 5D
    function with the phase 5D configuration; the predeclared technical
    validation is that function's own validator suite, and any problem it
    reports stops the phase rather than shrinking the population.

    Args:
        paths: Project layout.
        allow_test: The in-code opt-in.
        env: Environment mapping to read. Defaults to ``os.environ``.

    Returns:
        The verified holdout population.

    Raises:
        TestDataIntegrityError: If the predeclared technical validation fails.
    """
    from construction_safety_vision.data.materialization import load_materialization_config

    module = _materializer(paths)
    config = load_materialization_config(paths.configs / "task_materialization.yaml")
    inputs = module.Inputs(paths, config)
    record = module.materialize_split(
        paths, config, inputs, TEST_SPLIT, allow_test=allow_test, env=env, write=True
    )
    if record["problems"]:
        msg = (
            f"TEST_DATA_INTEGRITY_FAILURE: {len(record['problems'])} problem(s) in the frozen "
            f"holdout; the first is {record['problems'][0]}"
        )
        raise TestDataIntegrityError(msg)

    class_map = load_class_map(paths)
    documents = record["documents"]
    integrity = {
        "criterion": "FILE_UNREADABLE_OR_UNDECODABLE_OR_ANNOTATION_DOCUMENT_MALFORMED",
        "performed_before_model_execution": True,
        "problems": 0,
        "silent_removal_permitted": False,
        "images_materialised": int(record["images"]),
        "annotations_materialised": int(record["annotations"]),
        "expected_annotations": int(record["expected_annotations"]),
        "zero_instance_images": int(record["zero_instance_images"]),
        "byte_identical_copies": sum(1 for copy in record["image_copies"] if copy.byte_identical),
        "image_copies": len(record["image_copies"]),
        "detection_segmentation_alignment_problems": len(record["alignment"]["problems"]),
        "bbox_agreement_problems": len(record["bbox_agreement"]["problems"]),
        "bbox_agreement_max_delta_px": record["bbox_agreement"]["max_delta_px"],
        "geometry_round_trip_mismatches": (
            int(record["round_trip"]["mismatches"]) if record["round_trip"] else 0
        ),
        "geometry_round_trip_annotations_checked": (
            int(record["round_trip"]["annotations_checked"]) if record["round_trip"] else 0
        ),
        "materialised_by": MATERIALIZER,
        "split_specific_branch_used": False,
        "instances_by_class": dict(record["instances_by_class"]),
    }
    population = build_population(
        paths,
        detection=documents["detection"],
        segmentation=documents["segmentation"],
        class_map=class_map,
        integrity=integrity,
    )
    if population.images != TEST_IMAGES or population.annotations != TEST_ANNOTATIONS:
        msg = (
            f"the materialised holdout holds {population.images} images and "
            f"{population.annotations} annotations, not the frozen "
            f"{TEST_IMAGES} / {TEST_ANNOTATIONS}"
        )
        raise TestDataIntegrityError(msg)
    return population


# --- the frozen inference blocks --------------------------------------------------------


def inference_block(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Turn a frozen protocol inference block into a prediction call.

    Args:
        raw: The protocol's ``detector_inference`` or ``segmenter_inference``.

    Returns:
        The settings the prediction path consumes.
    """
    return {
        "imgsz": int(raw["imgsz"]),
        "conf": float(raw["conf"]),
        "iou": float(raw["iou"]),
        "max_det": int(raw["max_det"]),
        "augment": bool(raw["augment"]),
        "agnostic_nms": bool(raw["agnostic_nms"]),
        "precision_argument": "quantize",
        "precision_value": int(raw["quantize"]),
        "precision": str(raw["precision"]),
        "half": bool(raw["half"]),
    }


def direct_iou_block(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Turn the frozen direct-IoU inference block into a prediction call.

    Args:
        raw: The protocol's ``direct_iou.inference``.

    Returns:
        The settings the prediction path consumes.
    """
    return {
        "imgsz": int(raw["imgsz"]),
        "conf": float(raw["conf"]),
        "iou": float(raw["iou"]),
        "max_det": int(raw["max_det"]),
        "augment": bool(raw["augment"]),
        "agnostic_nms": bool(raw["agnostic_nms"]),
        "precision_argument": "quantize",
        "precision_value": 32,
        "precision": "FP32",
        "half": bool(raw["half"]),
    }


# --- validators over the produced results -----------------------------------------------


FORBIDDEN_RESULT_KEYS: tuple[str, ...] = (
    "winner_declared",
    "composite_score",
    "weighted_ranking",
    "aggregate_benefit_score",
    "model_selection_follows",
    "threshold_tuned_on_test",
    "significance_test",
    "combined_score",
    "threshold_swept",
    "boxes_derived_from_masks",
    "model_invoked_during_metric_computation",
    "manual_cherry_picking_permitted",
    "topping_up_from_another_category_permitted",
    "sampling_permitted",
    "exclusions_permitted",
    "threshold_chosen_after_seeing_test",
    "new_rule_invented",
)
"""Every one must be false wherever it appears, at any depth, in a result."""


def _walk(payload: Any, path: str = "") -> list[tuple[str, str, Any]]:
    """Yield every key/value pair in a nested payload.

    Args:
        payload: Any JSON-shaped value.
        path: Accumulated dotted path.

    Returns:
        ``(path, key, value)`` for every mapping entry reached.
    """
    found: list[tuple[str, str, Any]] = []
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            where = f"{path}.{key}" if path else str(key)
            found.append((where, str(key), value))
            found.extend(_walk(value, where))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            found.extend(_walk(value, f"{path}[{index}]"))
    return found


def validate_result(
    payload: Mapping[str, Any],
    *,
    protocol_fingerprint: str,
    prediction_sha256: str,
    holdout_identifiers: Sequence[str],
) -> list[str]:
    """Check one committed result against the rules phase 11A froze.

    Args:
        payload: The result payload.
        protocol_fingerprint: The frozen protocol fingerprint.
        prediction_sha256: The prediction fingerprint it must name.
        holdout_identifiers: Every frozen holdout image id, so a leak is caught
            rather than assumed absent.

    Returns:
        One description per problem found.
    """
    problems: list[str] = []
    if payload.get("protocol_fingerprint") != protocol_fingerprint:
        problems.append("the result does not name the frozen protocol fingerprint")
    if payload.get("prediction_fingerprint") != prediction_sha256:
        problems.append("the result does not name its prediction fingerprint")
    if payload.get("split") != TEST_SPLIT:
        problems.append("the result is not about the holdout")
    if payload.get("metrics_derived_from") != "PERSISTED_PREDICTIONS_ONLY":
        problems.append("the result does not declare that it derives from persisted predictions")

    for where, key, value in _walk(payload):
        if key in FORBIDDEN_RESULT_KEYS and value not in (False, None):
            problems.append(f"{where} must be false, found {value!r}")

    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    leaked = sorted(identifier for identifier in holdout_identifiers if identifier in text)
    if leaked:
        problems.append(f"{len(leaked)} holdout identifier(s) reached a committed result")
    return problems


# --- the frozen phase 11B pipeline ------------------------------------------------------


RUNTIME_ROOT = "artifacts/final_test"
"""Git-ignored. Predictions, the ledger, the selection and the figures live here."""

DETECTOR_PASS = "DETECTOR_AP_PASS"
SEGMENTER_PASS = "SEGMENTER_AP_PASS"
SEGMENTER_OPERATIONAL_PASS = "SEGMENTER_OPERATIONAL_PASS_FOR_DIRECT_IOU"


def _write_json(path: Path, payload: Any) -> str:
    """Write a JSON document with LF endings and return its digest.

    Args:
        path: Destination file.
        payload: The document.

    Returns:
        The SHA-256 of the bytes written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def resolve_models(paths: ProjectPaths, protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve both frozen checkpoints by digest and verify their identity.

    Args:
        paths: Project layout.
        protocol: The frozen protocol document.

    Returns:
        Per model, its identity and the resolved checkpoint path.

    Raises:
        HoldoutExecutionError: If a checkpoint is absent or is not the frozen one.
    """
    from construction_safety_vision.detection_freeze import (
        load_checkpoint_path as detector_checkpoint,
    )
    from construction_safety_vision.detection_freeze import load_final_detector
    from construction_safety_vision.segmentation_freeze import (
        load_checkpoint_path as segmenter_checkpoint,
    )
    from construction_safety_vision.segmentation_freeze import load_final_segmenter

    detector = load_final_detector(paths.reports)
    segmenter = load_final_segmenter(paths.reports)
    resolved = {
        "D2": {
            "path": detector_checkpoint(detector, paths.root),
            "identity": {
                "experiment": detector.selected_experiment,
                "model": detector.model,
                "imgsz": detector.imgsz,
                "checkpoint_sha256": detector.checkpoint_sha256,
                "checkpoint_bytes": protocol["detector"]["checkpoint_bytes"],
                "identity_fingerprint": detector.fingerprint,
                "frozen_in_phase": protocol["detector"]["frozen_in_phase"],
                "trained_in_this_phase": False,
                "modified_in_this_phase": False,
            },
            "declared": protocol["detector"]["checkpoint_sha256"],
        },
        "S1": {
            "path": segmenter_checkpoint(segmenter, paths.root),
            "identity": {
                "experiment": segmenter.selected_experiment,
                "model": segmenter.model,
                "imgsz": segmenter.imgsz,
                "mask_ratio": segmenter.mask_ratio,
                "overlap_mask": segmenter.overlap_mask,
                "checkpoint_sha256": segmenter.checkpoint_sha256,
                "checkpoint_bytes": protocol["segmenter"]["checkpoint_bytes"],
                "identity_fingerprint": segmenter.fingerprint,
                "final_segmenter_sha256": segmenter.fingerprint,
                "frozen_in_phase": protocol["segmenter"]["frozen_in_phase"],
                "trained_in_this_phase": False,
                "modified_in_this_phase": False,
            },
            "declared": protocol["segmenter"]["checkpoint_sha256"],
        },
    }
    for experiment, entry in resolved.items():
        if entry["identity"]["checkpoint_sha256"] != entry["declared"]:
            msg = f"MODEL_IDENTITY_MISMATCH: the frozen {experiment} is not the protocol's"
            raise HoldoutExecutionError(msg)
        if entry["identity"]["imgsz"] != 768:
            msg = f"MODEL_IDENTITY_MISMATCH: {experiment} is not frozen at imgsz 768"
            raise HoldoutExecutionError(msg)
    return resolved


def run_pass(
    checkpoint: Path,
    population: HoldoutPopulation,
    settings: Mapping[str, Any],
    *,
    want_masks: bool,
    model: Mapping[str, Any],
    protocol_fingerprint: str,
    pass_name: str,
    destination: Path,
) -> dict[str, Any]:
    """Run, persist and fingerprint one declared inference pass.

    Persistence happens here, before any metric exists, so that a later
    reporting failure is recovered by rebuilding rather than by running a model
    a second time.

    Args:
        checkpoint: The frozen checkpoint.
        population: The holdout population.
        settings: The frozen inference block.
        want_masks: Whether to collect instance masks.
        model: The frozen model's identity.
        protocol_fingerprint: The frozen protocol fingerprint.
        pass_name: Which declared pass this is.
        destination: Where to persist the prediction document.

    Returns:
        The persisted document, its fingerprint and the pass's counts.
    """
    raw = predict(checkpoint, population, settings, want_masks=want_masks)
    document = prediction_document(
        raw,
        population,
        model=model,
        protocol_fingerprint=protocol_fingerprint,
        settings=settings,
        pass_name=pass_name,
    )
    fingerprint = prediction_fingerprint(document)
    document["prediction_sha256"] = fingerprint
    file_sha256 = _write_json(destination, document)
    return {
        "document": document,
        "fingerprint": fingerprint,
        "file_sha256": file_sha256,
        "path": destination,
        "counts": document["counts"],
    }


def execute(
    paths: ProjectPaths,
    protocol: Mapping[str, Any],
    protocol_fingerprint: str,
    ledger: Any,
    *,
    allow_test: bool,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run the frozen phase 11B pipeline in the frozen order.

    Args:
        paths: Project layout.
        protocol: The frozen protocol document.
        protocol_fingerprint: Its verified fingerprint.
        ledger: The append-only one-shot ledger.
        allow_test: The in-code opt-in, already checked by the runner.
        env: Environment mapping to read. Defaults to ``os.environ``.

    Returns:
        Everything the report and provenance artifacts need.
    """
    runtime = paths.root / RUNTIME_ROOT

    models = resolve_models(paths, protocol)
    ledger.advance(
        "MODEL_IDENTITY_VERIFIED",
        detail="both frozen checkpoints resolved by digest and verified against the protocol",
    )

    population = load_holdout(paths, allow_test=allow_test, env=env)
    ledger.advance(
        "TEST_LOADED",
        detail=(
            f"holdout materialised through {MATERIALIZER}: {population.images} images, "
            f"{population.annotations} annotations, 0 integrity problems"
        ),
    )

    detector_settings = inference_block(protocol["detector_inference"])
    segmenter_settings = inference_block(protocol["segmenter_inference"])
    operational_settings = direct_iou_block(protocol["direct_iou"]["inference"])

    ledger.advance("DETECTOR_PREDICTION_STARTED", detail=f"{DETECTOR_PASS} at conf 0.001")
    detector_pass = run_pass(
        models["D2"]["path"],
        population,
        detector_settings,
        want_masks=False,
        model=models["D2"]["identity"],
        protocol_fingerprint=protocol_fingerprint,
        pass_name=DETECTOR_PASS,
        destination=runtime / "D2" / "predictions" / "ap_predictions.json",
    )
    ledger.advance(
        "DETECTOR_PREDICTION_COMPLETE",
        detail=(
            f"{detector_pass['counts']['predictions']} predictions persisted and fingerprinted "
            f"as {detector_pass['fingerprint']}"
        ),
    )

    ledger.advance(
        "SEGMENTER_PREDICTION_STARTED", detail=f"{SEGMENTER_PASS} and {SEGMENTER_OPERATIONAL_PASS}"
    )
    segmenter_pass = run_pass(
        models["S1"]["path"],
        population,
        segmenter_settings,
        want_masks=True,
        model=models["S1"]["identity"],
        protocol_fingerprint=protocol_fingerprint,
        pass_name=SEGMENTER_PASS,
        destination=runtime / "S1" / "predictions" / "ap_predictions.json",
    )
    operational_pass = run_pass(
        models["S1"]["path"],
        population,
        operational_settings,
        want_masks=True,
        model=models["S1"]["identity"],
        protocol_fingerprint=protocol_fingerprint,
        pass_name=SEGMENTER_OPERATIONAL_PASS,
        destination=runtime / "S1" / "predictions" / "operational_predictions.json",
    )
    ledger.advance(
        "SEGMENTER_PREDICTION_COMPLETE",
        detail=(
            f"{segmenter_pass['counts']['predictions']} AP predictions "
            f"({segmenter_pass['counts']['masks_reconstructed']} masks) and "
            f"{operational_pass['counts']['predictions']} operational predictions persisted"
        ),
    )
    ledger.advance(
        "PREDICTIONS_PERSISTED",
        detail=(
            "prediction immutability barrier established; every remaining number derives from "
            "the persisted files and no model is invoked again"
        ),
    )

    # --- past the barrier: everything below reads the persisted files only ---------
    detector_instances = load_predictions(detector_pass["document"])
    segmenter_instances = load_predictions(segmenter_pass["document"])
    operational_instances = load_predictions(operational_pass["document"])

    truth = canonical_instances(population, with_masks=True)
    admitted = support(population)

    detector_boxes = evaluate_canonical(
        population.detection,
        box_detections(detector_instances, population),
        class_map=population.class_map,
        iou_type="bbox",
    )
    segmenter_boxes = evaluate_canonical(
        population.detection,
        box_detections(segmenter_instances, population),
        class_map=population.class_map,
        iou_type="bbox",
    )
    segmenter_masks = evaluate_canonical(
        population.segmentation,
        mask_detections(segmenter_instances, population),
        class_map=population.class_map,
        iou_type="segm",
    )

    diagnostic = direct_mask_iou(operational_instances, truth, population)

    detector_operating = at_operating_point(detector_instances)
    segmenter_operating = at_operating_point(segmenter_instances)

    detector_matrix = confusion_matrix(detector_operating, population)
    segmenter_matrix = confusion_matrix(segmenter_operating, population)

    detector_objects = object_level_outcomes(
        detector_operating, truth, population, with_predicted_masks=False
    )
    segmenter_objects = object_level_outcomes(
        segmenter_operating, truth, population, with_predicted_masks=True
    )

    detector_selection = qualitative_selection(
        qualitative_candidates("D2", detector_objects, detector_operating, truth, with_masks=False),
        applicable={
            category: category not in ("SEGMENTATION_UNDER_COVERAGE", "SEGMENTATION_OVER_COVERAGE")
            for category in QUALITATIVE_CATEGORIES
        },
    )
    segmenter_selection = qualitative_selection(
        qualitative_candidates(
            "S1", segmenter_objects, segmenter_operating, truth, with_masks=True
        ),
        applicable=dict.fromkeys(QUALITATIVE_CATEGORIES, True),
    )
    ledger.advance(
        "METRICS_COMPUTED",
        detail="canonical AP, operating-point counts, direct IoU, confusion matrices and the "
        "deterministic qualitative ranking, all from the persisted predictions",
    )

    return {
        "models": models,
        "population": population,
        "truth": truth,
        "support": admitted,
        "passes": {
            DETECTOR_PASS: detector_pass,
            SEGMENTER_PASS: segmenter_pass,
            SEGMENTER_OPERATIONAL_PASS: operational_pass,
        },
        "detector_boxes": detector_boxes,
        "segmenter_boxes": segmenter_boxes,
        "segmenter_masks": segmenter_masks,
        "direct_iou": diagnostic,
        "detector_matrix": detector_matrix,
        "segmenter_matrix": segmenter_matrix,
        "detector_objects": detector_objects,
        "segmenter_objects": segmenter_objects,
        "detector_selection": detector_selection,
        "segmenter_selection": segmenter_selection,
        "operating_point_instances": {
            "D2": detector_operating,
            "S1": segmenter_operating,
        },
        "runtime_root": runtime,
    }


# --- the bounded validation-versus-test comparison -------------------------------------


def validation_versus_test(paths: ProjectPaths, bundle: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Pair each already-existing validation metric with its holdout counterpart.

    Only metrics that existed before the holdout was read, read from their
    committed artifacts **by field** rather than transcribed, and paired only
    where the evaluator, the ground-truth document type and the confidence
    match. No new metric is invented, no subgroup is mined and no significance
    test is reported, because none was predeclared.

    Args:
        paths: Project layout.
        bundle: The pipeline's output.

    Returns:
        One row per permitted comparison.
    """
    box = json.loads(
        (paths.reports / "detector_segmenter_box_comparison.json").read_text(encoding="utf-8")
    )
    canonical = json.loads(
        (paths.reports / "segmentation_S1_canonical_evaluation.json").read_text(encoding="utf-8")
    )
    direct = json.loads(
        (paths.reports / "segmentation_S1_mask_iou.json").read_text(encoding="utf-8")
    )
    diagnostic = bundle["direct_iou"]["global"]

    pairs: list[tuple[str, Any, Any]] = [
        (
            "D2 canonical box mAP@0.50:0.95",
            box["metrics"]["CANONICAL_BOX_MAP50_95"]["D2"],
            bundle["detector_boxes"]["all_class_map50_95"],
        ),
        (
            "D2 canonical box mAP@0.50",
            box["metrics"]["CANONICAL_BOX_MAP50"]["D2"],
            bundle["detector_boxes"]["all_class_map50"],
        ),
        (
            "S1 canonical box mAP@0.50:0.95",
            box["metrics"]["CANONICAL_BOX_MAP50_95"]["S1"],
            bundle["segmenter_boxes"]["all_class_map50_95"],
        ),
        (
            "S1 canonical box mAP@0.50",
            box["metrics"]["CANONICAL_BOX_MAP50"]["S1"],
            bundle["segmenter_boxes"]["all_class_map50"],
        ),
        (
            "S1 canonical mask mAP@0.50:0.95",
            canonical["canonical"]["all_class_map50_95"],
            bundle["segmenter_masks"]["all_class_map50_95"],
        ),
        (
            "S1 canonical mask mAP@0.50",
            canonical["canonical"]["all_class_map50"],
            bundle["segmenter_masks"]["all_class_map50"],
        ),
        (
            "S1 direct `matched_mask_iou_mean`",
            direct["global"]["matched_mask_iou_mean"],
            diagnostic["matched_mask_iou_mean"],
        ),
        (
            "S1 direct `gt_normalized_mask_iou`",
            direct["global"]["gt_normalized_mask_iou"],
            diagnostic["gt_normalized_mask_iou"],
        ),
        (
            "S1 direct `gt_match_coverage`",
            direct["global"]["gt_match_coverage"],
            diagnostic["gt_match_coverage"],
        ),
        (
            "S1 direct `gt_iou50_coverage`",
            direct["global"]["gt_iou50_coverage"],
            diagnostic["gt_iou50_coverage"],
        ),
        (
            "S1 direct `gt_iou75_coverage`",
            direct["global"]["gt_iou75_coverage"],
            diagnostic["gt_iou75_coverage"],
        ),
    ]
    rows: list[dict[str, Any]] = []
    for metric, validation, test in pairs:
        difference = (
            rounded(abs(float(test) - float(validation)))
            if validation is not None and test is not None
            else None
        )
        rows.append(
            {
                "metric": metric,
                "validation": rounded(validation),
                "test": rounded(test),
                "absolute_difference": difference,
                "label": "DESCRIPTIVE_GENERALIZATION_COMPARISON",
                "significance_test": False,
                "metric_existed_before_test_access": True,
            }
        )
    return rows


def box_decomposition(bundle: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Decompose the D2-versus-S1 all-class box delta into its per-class parts.

    The all-class figure is the unweighted mean of five per-class APs, so it
    decomposes exactly. Publishing the aggregate without this is the error the
    decomposition exists to prevent.

    Args:
        bundle: The pipeline's output.

    Returns:
        One row per class.
    """
    detector = bundle["detector_boxes"]["per_class"]
    segmenter = bundle["segmenter_boxes"]["per_class"]
    rows: list[dict[str, Any]] = []
    for name in CLASSES:
        left = detector.get(name, {}).get("AP@0.50:0.95")
        right = segmenter.get(name, {}).get("AP@0.50:0.95")
        delta = (
            rounded(float(right) - float(left)) if left is not None and right is not None else None
        )
        rows.append(
            {
                "class": name,
                "detector": left,
                "segmenter": right,
                "delta": delta,
                "contribution": rounded(delta / len(CLASSES)) if delta is not None else None,
            }
        )
    return rows


# --- artifact writing -------------------------------------------------------------------


REPORT_DETECTOR = "final_test_detector.json"
REPORT_SEGMENTER = "final_test_segmenter.json"
REPORT_DIRECT_IOU = "final_test_direct_iou.json"
REPORT_MARKDOWN = "final_test_evaluation.md"
REPORT_PROVENANCE = "final_test_evaluation.provenance.json"
REPORT_PER_CLASS = "final_test_per_class.csv"

FIGURE_ROOT = "figures/final_test"
"""Committed figures. Confusion matrices only - they carry no imagery and no id."""

TEST_EVALUATION_COMPLETE = "TEST_EVALUATION_COMPLETE"


def per_class_csv(bundle: Mapping[str, Any]) -> str:
    """Render the per-class table both models' results share.

    Args:
        bundle: The pipeline's output.

    Returns:
        CSV text with LF endings.
    """
    header = (
        "class,support_images,support_instances,support_status,"
        "d2_box_ap50_95,d2_box_ap50,s1_box_ap50_95,s1_box_ap50,"
        "s1_mask_ap50_95,s1_mask_ap50,"
        "s1_direct_matched_mask_iou_mean,s1_direct_gt_normalized_mask_iou"
    )
    lines = [header]
    direct = bundle["direct_iou"]["per_class"]
    for name in CLASSES:
        support_row = bundle["support"][name]
        detector = bundle["detector_boxes"]["per_class"].get(name, {})
        segmenter = bundle["segmenter_boxes"]["per_class"].get(name, {})
        masks = bundle["segmenter_masks"]["per_class"].get(name, {})
        diagnostic = direct.get(name, {})

        def cell(value: Any) -> str:
            return "" if value is None else f"{float(value):.6f}"

        lines.append(
            ",".join(
                [
                    name,
                    str(support_row["images"]),
                    str(support_row["instances"]),
                    support_row["status"],
                    cell(detector.get("AP@0.50:0.95")),
                    cell(detector.get("AP@0.50")),
                    cell(segmenter.get("AP@0.50:0.95")),
                    cell(segmenter.get("AP@0.50")),
                    cell(masks.get("AP@0.50:0.95")),
                    cell(masks.get("AP@0.50")),
                    cell(diagnostic.get("matched_mask_iou_mean")),
                    cell(diagnostic.get("gt_normalized_mask_iou")),
                ]
            )
        )
    return "\n".join(lines) + "\n"


def _confusion_figures(bundle: Mapping[str, Any], destination: Path) -> list[Path]:
    """Render both models' confusion-matrix figures.

    Args:
        bundle: The pipeline's output.
        destination: Committed figure directory.

    Returns:
        The figures written.
    """
    written: list[Path] = []
    for experiment, key in (("D2", "detector_matrix"), ("S1", "segmenter_matrix")):
        target = destination / experiment.lower()
        target.mkdir(parents=True, exist_ok=True)
        matrix = bundle[key]["_matrix_object"]
        for normalise in (False, True):
            matrix.plot(normalize=normalise, save_dir=str(target))
        for name in ("confusion_matrix.png", "confusion_matrix_normalized.png"):
            candidate = target / name
            if candidate.is_file():
                written.append(candidate)
    return written


def _qualitative_figures(bundle: Mapping[str, Any], destination: Path) -> dict[str, Any]:
    """Render only the deterministically selected qualitative examples.

    No other holdout image is opened. The figures land under the git-ignored run
    directory, because committing holdout imagery or a holdout identifier is
    prohibited by the frozen protocol.

    Args:
        bundle: The pipeline's output.
        destination: Git-ignored figure directory.

    Returns:
        How many figures were rendered, per model.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    from PIL import Image

    population = bundle["population"]
    truth = bundle["truth"]
    rendered: dict[str, int] = {}

    for experiment, selection_key, instances_key in (
        ("D2", "detector_selection", "D2"),
        ("S1", "segmenter_selection", "S1"),
    ):
        target = destination / experiment.lower()
        target.mkdir(parents=True, exist_ok=True)
        predictions = bundle["operating_point_instances"][instances_key]
        count = 0
        for category, block in bundle[selection_key]["categories"].items():
            if not block["applicable"]:
                continue
            for entry in block["selected"]:
                stem = entry["source_image_id"]
                with Image.open(population.files[stem]) as handle:
                    image = handle.convert("RGB")
                figure, axis = plt.subplots(1, 1, figsize=(10, 8))
                axis.imshow(image)
                axis.axis("off")
                annotation_id = entry["canonical_annotation_id"]
                for instance in truth.get(stem, ()):
                    if instance.annotation_id != annotation_id:
                        continue
                    x1, y1, x2, y2 = instance.box
                    axis.add_patch(
                        patches.Rectangle(
                            (x1, y1),
                            x2 - x1,
                            y2 - y1,
                            linewidth=2,
                            edgecolor="#00b050",
                            facecolor="none",
                        )
                    )
                index = entry["canonical_prediction_index"]
                if 0 <= index < len(predictions.get(stem, [])):
                    prediction = predictions[stem][index]
                    x1, y1, x2, y2 = prediction.box
                    axis.add_patch(
                        patches.Rectangle(
                            (x1, y1),
                            x2 - x1,
                            y2 - y1,
                            linewidth=2,
                            edgecolor="#d62728",
                            facecolor="none",
                            linestyle="--",
                        )
                    )
                    if prediction.mask_rle is not None:
                        axis.imshow(
                            np.ma.masked_where(
                                ~decode_mask(prediction.mask_rle),
                                np.ones(decode_mask(prediction.mask_rle).shape),
                            ),
                            alpha=0.35,
                            cmap="autumn",
                        )
                axis.set_title(
                    f"{experiment} - {category} - rank {entry['rank']} - "
                    f"value {float(entry['rank_value']):.6f}\n"
                    "green solid: canonical ground truth   red dashed: prediction",
                    fontsize=9,
                )
                figure.tight_layout()
                figure.savefig(
                    target / f"{category.lower()}_{entry['rank']}.png",
                    dpi=120,
                    bbox_inches="tight",
                )
                plt.close(figure)
                count += 1
        rendered[experiment] = count
    return rendered


def write_artifacts(
    paths: ProjectPaths,
    bundle: Mapping[str, Any],
    ledger: Any,
    *,
    protocol_fingerprint: str,
    direct_iou_protocol_fingerprint: str,
    holdout_identifiers: Sequence[str],
) -> dict[str, Any]:
    """Write every phase 11B artifact and verify what may be committed.

    Args:
        paths: Project layout.
        bundle: The pipeline's output.
        ledger: The append-only one-shot ledger.
        protocol_fingerprint: The frozen protocol fingerprint.
        direct_iou_protocol_fingerprint: Phase 8C's frozen fingerprint.
        holdout_identifiers: Every frozen holdout image id, used only to prove
            that none of them reached a committed artifact.

    Returns:
        The fingerprints, the paths written and the validation outcome.

    Raises:
        HoldoutExecutionError: If a committed artifact fails validation or would
            publish a holdout identifier.
    """
    from construction_safety_vision.data.canonical import scan_for_sensitive
    from construction_safety_vision.final_holdout_results import render

    population = bundle["population"]
    runtime = bundle["runtime_root"]

    detector_result = build_detector_result(
        model=bundle["models"]["D2"]["identity"],
        protocol_fingerprint=protocol_fingerprint,
        prediction_sha256=bundle["passes"][DETECTOR_PASS]["fingerprint"],
        population=population,
        boxes=bundle["detector_boxes"],
        admitted=bundle["support"],
        objects=bundle["detector_objects"],
        matrix=bundle["detector_matrix"],
    )
    segmenter_result = build_segmenter_result(
        model=bundle["models"]["S1"]["identity"],
        protocol_fingerprint=protocol_fingerprint,
        prediction_sha256=bundle["passes"][SEGMENTER_PASS]["fingerprint"],
        operational_prediction_sha256=bundle["passes"][SEGMENTER_OPERATIONAL_PASS]["fingerprint"],
        population=population,
        masks=bundle["segmenter_masks"],
        boxes=bundle["segmenter_boxes"],
        detector_boxes=bundle["detector_boxes"],
        admitted=bundle["support"],
        objects=bundle["segmenter_objects"],
        matrix=bundle["segmenter_matrix"],
    )
    direct_result = build_direct_iou_result(
        model=bundle["models"]["S1"]["identity"],
        protocol_fingerprint=protocol_fingerprint,
        direct_iou_protocol_fingerprint=direct_iou_protocol_fingerprint,
        prediction_sha256=bundle["passes"][SEGMENTER_OPERATIONAL_PASS]["fingerprint"],
        population=population,
        diagnostic=bundle["direct_iou"],
    )

    detector_result["qualitative"] = redact_identifiers(bundle["detector_selection"])
    segmenter_result["qualitative"] = redact_identifiers(bundle["segmenter_selection"])

    detector_sha256 = result_fingerprint(detector_result)
    segmenter_sha256 = result_fingerprint(segmenter_result)
    direct_sha256 = result_fingerprint(direct_result)
    detector_result["result_sha256"] = detector_sha256
    segmenter_result["result_sha256"] = segmenter_sha256
    direct_result["result_sha256"] = direct_sha256

    # The selection carries identifiers, so it is fingerprinted whole and kept
    # out of the repository; the committed report quotes the fingerprint.
    selection = {
        "phase": PHASE,
        "protocol_fingerprint": protocol_fingerprint,
        "D2": bundle["detector_selection"],
        "S1": bundle["segmenter_selection"],
    }
    qualitative_sha256 = digest_payload(selection)
    selection["qualitative_selection_sha256"] = qualitative_sha256
    _write_json(runtime / "qualitative_selection.json", selection)

    problems: list[str] = []
    for payload, prediction_sha256 in (
        (detector_result, bundle["passes"][DETECTOR_PASS]["fingerprint"]),
        (segmenter_result, bundle["passes"][SEGMENTER_PASS]["fingerprint"]),
        (direct_result, bundle["passes"][SEGMENTER_OPERATIONAL_PASS]["fingerprint"]),
    ):
        problems.extend(
            validate_result(
                payload,
                protocol_fingerprint=protocol_fingerprint,
                prediction_sha256=prediction_sha256,
                holdout_identifiers=holdout_identifiers,
            )
        )
    if problems:
        msg = f"a committed result does not validate: {problems}"
        raise HoldoutExecutionError(msg)

    figures = _confusion_figures(bundle, paths.reports / FIGURE_ROOT)
    qualitative_figures = _qualitative_figures(bundle, runtime / "figures")

    report_payload = {
        "classification": TEST_EVALUATION_COMPLETE,
        "protocol_fingerprint": protocol_fingerprint,
        "ledger": ledger.as_record(),
        "counts": {
            "models_executed": 2,
            "inference_passes": 3,
            "models_trained": 0,
            "thresholds_tuned": 0,
            "latency_measurements": 0,
            "new_spatial_metrics": 0,
            "prediction_reruns": 0,
        },
        "population": {
            "images": population.images,
            "annotations": population.annotations,
            "membership_sha256": population.membership_sha256,
        },
        "integrity": population.integrity,
        "support": bundle["support"],
        "detector": detector_result,
        "segmenter": segmenter_result,
        "direct_iou": direct_result,
        "box_decomposition": box_decomposition(bundle),
        "validation_versus_test": validation_versus_test(paths, bundle),
        "prediction_passes": [
            {
                "pass": name,
                "model": bundle["passes"][name]["document"]["model"]["experiment"],
                "conf": bundle["passes"][name]["document"]["inference"]["conf"],
                "masks": bool(bundle["passes"][name]["counts"]["masks_reconstructed"]),
                "predictions": bundle["passes"][name]["counts"]["predictions"],
                "fingerprint": bundle["passes"][name]["fingerprint"],
            }
            for name in (DETECTOR_PASS, SEGMENTER_PASS, SEGMENTER_OPERATIONAL_PASS)
        ],
        "qualitative_selection_sha256": qualitative_sha256,
        "fingerprints": {
            "detector_test_prediction_sha256": bundle["passes"][DETECTOR_PASS]["fingerprint"],
            "segmenter_test_prediction_sha256": bundle["passes"][SEGMENTER_PASS]["fingerprint"],
            "segmenter_operational_test_prediction_sha256": bundle["passes"][
                SEGMENTER_OPERATIONAL_PASS
            ]["fingerprint"],
            "detector_test_result_sha256": detector_sha256,
            "segmenter_test_result_sha256": segmenter_sha256,
            "direct_iou_test_result_sha256": direct_sha256,
            "qualitative_selection_sha256": qualitative_sha256,
        },
    }
    markdown = render(report_payload)

    committed: list[tuple[Path, str]] = [
        (paths.reports / REPORT_DETECTOR, json.dumps(detector_result, sort_keys=True)),
        (paths.reports / REPORT_SEGMENTER, json.dumps(segmenter_result, sort_keys=True)),
        (paths.reports / REPORT_DIRECT_IOU, json.dumps(direct_result, sort_keys=True)),
        (paths.reports / REPORT_MARKDOWN, markdown),
        (paths.reports / REPORT_PER_CLASS, per_class_csv(bundle)),
    ]
    for path, text in committed:
        unsafe = scan_for_sensitive(text)
        if unsafe:
            msg = f"{path.name} is not fit to commit: {unsafe}"
            raise HoldoutExecutionError(msg)
        leaked = sorted(identifier for identifier in holdout_identifiers if identifier in text)
        if leaked:
            msg = f"{path.name} would publish {len(leaked)} holdout identifier(s)"
            raise HoldoutExecutionError(msg)

    _write_json(paths.reports / REPORT_DETECTOR, detector_result)
    _write_json(paths.reports / REPORT_SEGMENTER, segmenter_result)
    _write_json(paths.reports / REPORT_DIRECT_IOU, direct_result)
    (paths.reports / REPORT_MARKDOWN).write_text(markdown, encoding="utf-8", newline="\n")
    (paths.reports / REPORT_PER_CLASS).write_text(
        per_class_csv(bundle), encoding="utf-8", newline="\n"
    )

    return {
        "report_payload": report_payload,
        "fingerprints": report_payload["fingerprints"],
        "figures": [str(path.relative_to(paths.root)) for path in figures],
        "qualitative_figures": qualitative_figures,
        "committed": [str(path.relative_to(paths.root)) for path, _ in committed],
        "selection_path": str((runtime / "qualitative_selection.json").relative_to(paths.root)),
    }


# --- leak proof and provenance ----------------------------------------------------------


def frozen_holdout_identifiers(
    paths: ProjectPaths, *, allow_test: bool, env: dict[str, str] | None = None
) -> tuple[str, ...]:
    """Return the frozen holdout image ids, for leak checking only.

    They are read through the same guarded accessor everything else uses, they
    never leave this process, and their only purpose is to prove that no
    committed artifact contains one. Proving the absence of a leak requires
    knowing what a leak would look like.

    Args:
        paths: Project layout.
        allow_test: The in-code opt-in.
        env: Environment mapping to read. Defaults to ``os.environ``.

    Returns:
        The holdout's source image ids.
    """
    from construction_safety_vision.data.split_freeze import load_frozen_splits

    splits = load_frozen_splits(paths.reports / "split_manifest.json")
    return splits.image_ids(
        TEST_SPLIT,
        purpose="proving no holdout identifier reached a committed artifact",
        allow_test=allow_test,
        env=env,
    )


def write_ledger(paths: ProjectPaths, ledger: Any) -> Path:
    """Persist the append-only one-shot ledger.

    Called on every exit, success or failure, so the evidence of what happened
    survives even when the phase stops for human review.

    Args:
        paths: Project layout.
        ledger: The ledger to persist.

    Returns:
        The path written.
    """
    destination = paths.root / RUNTIME_ROOT / "one_shot_ledger.json"
    _write_json(destination, ledger.as_record())
    return destination


def write_provenance(
    paths: ProjectPaths,
    bundle: Mapping[str, Any],
    written: Mapping[str, Any],
    ledger: Any,
    *,
    protocol_fingerprint: str,
) -> Path:
    """Write the committed provenance record and the git-ignored ledger.

    Args:
        paths: Project layout.
        bundle: The pipeline's output.
        written: The artifact writer's output.
        ledger: The append-only one-shot ledger.
        protocol_fingerprint: The frozen protocol fingerprint.

    Returns:
        The provenance record's path.
    """
    from construction_safety_vision.provenance import ProvenanceRecord

    population = bundle["population"]
    _write_json(bundle["runtime_root"] / "one_shot_ledger.json", ledger.as_record())

    record = ProvenanceRecord.create(
        "final_holdout_evaluation",
        phase=11,
        repo_root=paths.root,
        config={
            "protocol": "configs/final_holdout_evaluation.yaml",
            "protocol_fingerprint": protocol_fingerprint,
        },
        details={
            "classification": TEST_EVALUATION_COMPLETE,
            "phase": PHASE,
            "holdout": "OBSERVED_ONCE_FINAL",
            "holdout_reads_permitted": 1,
            "holdout_reads_performed": 1,
            "attempt": ledger.attempt,
            "attempt_counter_resettable": False,
            "ledger_state": ledger.state,
            "ledger": ledger.as_record(),
            "models_executed": 2,
            "models_trained": 0,
            "inference_passes": 3,
            "prediction_reruns": 0,
            "thresholds_tuned": 0,
            "thresholds_swept": 0,
            "latency_measurements_taken": 0,
            "memory_measurements_taken": 0,
            "new_spatial_metrics": 0,
            "association_analysis_repeated": False,
            "significance_tests": 0,
            "winner_declared": False,
            "composite_score": False,
            "model_selection_follows": False,
            "test_accessed": True,
            "test_predictions_produced": True,
            "test_metrics_computed": True,
            "test_images_read": population.images,
            "test_annotations_read": population.annotations,
            "test_identifiers_committed": 0,
            "test_imagery_committed": False,
            "predictions_committed": False,
            "metrics_derived_from": "PERSISTED_PREDICTIONS_ONLY",
            "model_invoked_during_metric_computation": False,
            "model_invoked_during_report_generation": False,
            "post_test_policy": {
                "state": "FINAL_TEST_OBSERVED",
                "model_selection": "CLOSED",
                "hyperparameter_tuning": "CLOSED",
                "threshold_tuning": "CLOSED",
                "data_cleaning_for_performance": "CLOSED",
            },
            "protocol_gap_resolution": {
                "label": GAP_RESOLUTION,
                "gap": (
                    "The frozen protocol gives the confusion matrix, the object-level TP/FP/FN "
                    "counts and the qualitative ranking a confidence of 0.25 without declaring "
                    "a separate inference block for them."
                ),
                "resolution": DERIVED_FROM_AP_PASS,
                "resolved_before_any_holdout_number_existed": True,
                "applied_identically_to_both_models": True,
                "phase_11a_declared_this_resolution": False,
                "why": (
                    "It is the framework's own convention - its validator computes the matrix "
                    "from the validation prediction set and filters at 0.25 internally - it "
                    "keeps the two models symmetric, and it adds no inference pass beyond the "
                    "three the protocol declares."
                ),
            },
            "qualitative_publication_note": {
                "label": "PROTOCOL_COVERAGE_NOTE",
                "tension": (
                    "The frozen protocol lists approved qualitative figures among its committed "
                    "outputs and separately prohibits committing holdout identifiers, imagery "
                    "or predictions."
                ),
                "resolution": "PROHIBITION_WINS_FIGURES_AND_SELECTION_STAY_GIT_IGNORED",
                "committed_instead": "the selection rule, the ranked values and the fingerprint",
                "protocol_edited_to_accommodate_this": False,
            },
            "fingerprints": dict(written["fingerprints"]),
            "population": {
                "images": population.images,
                "annotations": population.annotations,
                "membership_sha256": population.membership_sha256,
                "evaluated": "ALL_FROZEN_TEST_IMAGES",
                "sampled": False,
                "manually_excluded": 0,
            },
            "integrity": dict(population.integrity),
            "figures_committed": list(written["figures"]),
            "qualitative_figures_written_git_ignored": dict(written["qualitative_figures"]),
            "runtime_artifacts_git_ignored": RUNTIME_ROOT,
        },
    )
    for name in (
        "configs/final_holdout_evaluation.yaml",
        "reports/final_holdout_evaluation_protocol.json",
        "reports/final_detector_manifest.json",
        "reports/final_segmenter_manifest.json",
        "reports/split_manifest.json",
        "reports/task_dataset_manifest.json",
        "configs/segmentation_mask_iou_evaluation.yaml",
        "configs/task_materialization.yaml",
    ):
        record.add_input(paths.root / name, relative_to=paths.root)
    for name in written["committed"]:
        record.add_output(paths.root / name, relative_to=paths.root)
    for name in written["figures"]:
        record.add_output(paths.root / name, relative_to=paths.root)

    destination = paths.reports / REPORT_PROVENANCE
    record.write_json(destination)
    return destination
