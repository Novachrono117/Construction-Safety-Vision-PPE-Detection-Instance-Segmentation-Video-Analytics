"""A common canonical evaluator for comparing segmentation experiments.

Phase 8E. It exists because of a problem phase 8D uncovered: the framework's own
mask average precision is computed against the target its training flags
construct. S0 trained with ``overlap_mask: true``, where the smaller instance
owns any shared pixel; the S1 candidate flips that flag. Their native mask AP
would therefore be measured against **different ground truth**, and comparing
the two numbers would compare the targets as much as the models.

So both are evaluated here against one fixed reference that no training flag can
move: the canonical phase 5D COCO instance segmentation, scored with
``pycocotools`` ``COCOeval`` at ``iouType='segm'``.

Four things are deliberate.

**Canonical masks only.** Never the YOLO adapter, whose own round-trip loss
phase 8A measured, and never a target that a training flag reshapes. A
``ground_truth_document`` pointing at an adapter is refused at parse time.

**Standard COCO semantics, unmodified.** IoU thresholds 0.50:0.05:0.95 and
``maxDets`` [1, 10, 100]. Redefining either after a candidate exists is the
classic way to move a result, so both are pinned in the configuration and
checked against what ``COCOeval`` actually used.

**A low fixed confidence, and it is not an operating point.** AP integrates
precision over recall, so it needs the low-scoring tail that an operational
threshold would discard. 0.001 is there to preserve that curve. The separate
phase 8C direct-IoU diagnostic keeps its own frozen operational threshold of
0.25, and the two protocols are never mixed.

**Binary-mask RLE, from the reference implementation.** Predictions are encoded
with ``pycocotools.mask.encode`` on the original image canvas - exact, no
polygon approximation, and the same library that decodes the canonical ground
truth, so the two sides of every IoU come from one implementation.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from construction_safety_vision.config import ConfigError, check_keys
from construction_safety_vision.detection_comparison import (
    SUPPORT_MIN_INSTANCES,
    SUPPORT_MIN_POSITIVE_IMAGES,
)

CONFIG_SCHEMA_VERSION = 1
"""Schema version of ``configs/segmentation_canonical_evaluation.yaml``."""

PROTOCOL_NAME = "CANONICAL_COCO_MASK_AP_EVALUATION"
"""What this protocol is, recorded in every artifact that reports it."""

GROUND_TRUTH_SOURCE = "CANONICAL_COCO_INSTANCE_SEGMENTATION"
"""The only admissible ground truth, and the reason the evaluator exists."""

IOU_TYPE = "segm"
"""COCOeval mode. Mask IoU, never box IoU."""

IOU_THRESHOLDS: tuple[float, ...] = tuple(round(0.50 + 0.05 * step, 2) for step in range(10))
"""The standard COCO sweep, 0.50:0.05:0.95, pinned so it cannot move later."""

MAX_DETS: tuple[int, ...] = (1, 10, 100)
"""COCOeval's standard detection caps.

Note the deliberate asymmetry with the prediction stage, which allows up to 300
candidates per image: the model is given room to propose, and the metric applies
its conventional cap of 100 when scoring. Recording both stops a reader
inferring that one number constrained the other.
"""

PRIMARY_METRIC = "CANONICAL_SUPPORTED_MACRO_MASK_MAP50_95"
"""The frozen S0-versus-S1 selection metric."""

ALL_CLASS_METRIC = "CANONICAL_ALL_CLASS_MASK_MAP50_95"
"""The mandatory all-class figure, reported beside the selection metric."""

ALL_CLASS_METRIC_50 = "CANONICAL_ALL_CLASS_MASK_MAP50"
"""The mandatory all-class figure at a single IoU threshold."""

NATIVE_METRIC_STATUS = "NOT_CROSS_TARGET_COMPARABLE_FOR_S0_S1_SELECTION"
"""What the framework's own mask AP may and may not be used for."""

PRACTICAL_EQUIVALENCE_MARGIN = 0.005
"""Absolute margin on the primary metric. An engineering threshold, not a test."""

IMPROVES = "S1_IMPROVES_S0_BEYOND_MARGIN"
EQUIVALENT = "PRACTICALLY_EQUIVALENT"
BELOW = "S1_BELOW_S0"
DISAGREEMENT = "CROSS_METRIC_DIRECTION_DISAGREEMENT"

SELECTION_CASES: tuple[str, ...] = (IMPROVES, EQUIVALENT, BELOW)
"""Every verdict the frozen comparison may reach."""

FORBIDDEN_SPLIT = "test"
EVALUATION_SPLIT = "validation"

METRIC_PRECISION = 6
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class CanonicalEvaluationError(RuntimeError):
    """Raised when the canonical evaluation cannot run as specified."""


# --- protocol -----------------------------------------------------------------


@dataclass(frozen=True)
class CanonicalEvaluationConfig:
    """The frozen canonical COCO mask-AP evaluation protocol.

    Attributes:
        raw: The validated configuration mapping, exactly as parsed.
    """

    raw: dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        """Read a configuration value.

        Args:
            key: Configuration key.

        Returns:
            The value.
        """
        return self.raw[key]

    @property
    def inference(self) -> dict[str, Any]:
        """The frozen prediction settings.

        Returns:
            The inference block.
        """
        return dict(self.raw["inference"])

    @property
    def cocoeval(self) -> dict[str, Any]:
        """The frozen COCOeval settings.

        Returns:
            The evaluator block.
        """
        return dict(self.raw["cocoeval"])

    def as_dict(self) -> dict[str, Any]:
        """Serialise the protocol for hashing and reporting.

        Returns:
            A JSON-serialisable mapping.
        """
        return json.loads(json.dumps(self.raw, sort_keys=True))

    def fingerprint(self) -> str:
        """Hash the protocol, so a result can name the rules that produced it.

        Returns:
            A SHA-256 hex digest over the configuration content only.
        """
        text = json.dumps(self.as_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


_CONFIG_KEYS: tuple[str, ...] = (
    "schema_version",
    "protocol",
    "purpose",
    "why_a_common_evaluator_is_required",
    "is_not",
    "split",
    "ground_truth_source",
    "ground_truth_document",
    "class_map_sha256",
    "split_assignment_sha256",
    "inference",
    "cocoeval",
    "metrics",
    "rare_class",
    "rare_class_status",
    "test_policy",
)

_INFERENCE_KEYS: tuple[str, ...] = (
    "imgsz",
    "conf",
    "iou",
    "max_det",
    "retina_masks",
    "augment",
    "agnostic_nms",
    "half",
    "classes",
    "checkpoint",
    "threshold_policy",
)

_COCOEVAL_KEYS: tuple[str, ...] = (
    "implementation",
    "iou_type",
    "iou_thresholds",
    "max_dets",
    "area_range",
    "mask_encoding",
    "category_id_mapping",
)

_METRIC_KEYS: tuple[str, ...] = (
    "primary",
    "all_class",
    "per_class",
    "secondary_direct_iou",
    "native_framework_status",
    "composite_score",
)


def _contains_forbidden_split(value: Any) -> bool:
    """Report whether a parsed value mentions the protected split.

    Args:
        value: Any parsed configuration value.

    Returns:
        ``True`` when the protected split appears as a string or a key.
    """
    if isinstance(value, str):
        return value.strip().lower() == FORBIDDEN_SPLIT
    if isinstance(value, Mapping):
        return any(
            _contains_forbidden_split(key) or _contains_forbidden_split(item)
            for key, item in value.items()
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_contains_forbidden_split(item) for item in value)
    return False


def load_canonical_evaluation_config(path: str | Path) -> CanonicalEvaluationConfig:
    """Load and validate the frozen canonical evaluation protocol.

    Args:
        path: Configuration file.

    Returns:
        The parsed protocol.

    Raises:
        ConfigError: If the file is missing, malformed, names the protected
            split, evaluates against anything but the canonical masks, alters
            the standard COCO semantics, or introduces a composite score.
    """
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        msg = f"Canonical evaluation protocol not found: {config_path.name}"
        raise ConfigError(msg)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        msg = f"Canonical evaluation protocol is not valid YAML: {config_path.name} ({exc})"
        raise ConfigError(msg) from exc
    if not isinstance(raw, Mapping):
        msg = f"Canonical evaluation protocol must be a mapping: {config_path.name}"
        raise ConfigError(msg)

    check_keys(raw, required=_CONFIG_KEYS, context=config_path.name)

    if int(raw["schema_version"]) != CONFIG_SCHEMA_VERSION:
        msg = f"schema_version must be {CONFIG_SCHEMA_VERSION}, got {raw['schema_version']!r}"
        raise ConfigError(msg)
    if str(raw["protocol"]) != PROTOCOL_NAME:
        msg = f"protocol must be {PROTOCOL_NAME!r}, got {raw['protocol']!r}"
        raise ConfigError(msg)
    if str(raw["split"]) != EVALUATION_SPLIT:
        msg = f"split must be {EVALUATION_SPLIT!r}, got {raw['split']!r}"
        raise ConfigError(msg)
    if _contains_forbidden_split(dict(raw)):
        msg = "the canonical evaluation protocol references the protected split"
        raise ConfigError(msg)

    if str(raw["ground_truth_source"]) != GROUND_TRUTH_SOURCE:
        msg = (
            f"ground_truth_source must be {GROUND_TRUTH_SOURCE!r}. The whole point of this "
            "evaluator is a reference no training flag can move."
        )
        raise ConfigError(msg)
    document = str(raw["ground_truth_document"])
    if "adapter" in document.lower():
        msg = (
            f"ground_truth_document {document!r} points at an adapter. Evaluating against a "
            "derived representation would fold its measured loss into every model's score."
        )
        raise ConfigError(msg)

    for name in ("class_map_sha256", "split_assignment_sha256"):
        if not SHA256_PATTERN.match(str(raw[name])):
            msg = f"{name} is not a SHA-256 digest: {raw[name]!r}"
            raise ConfigError(msg)

    inference = raw["inference"]
    if not isinstance(inference, Mapping):
        msg = "inference: must be a mapping"
        raise ConfigError(msg)
    check_keys(inference, required=_INFERENCE_KEYS, context="inference")
    if float(inference["conf"]) != 0.001:
        msg = (
            f"inference.conf must be 0.001, got {inference['conf']!r}. Average precision "
            "integrates over the score curve and needs the low-scoring tail; a higher value "
            "would silently truncate it."
        )
        raise ConfigError(msg)
    if inference["augment"] is not False:
        msg = "inference.augment must be false: no test-time augmentation is authorised"
        raise ConfigError(msg)
    if inference["retina_masks"] is not True:
        msg = (
            "inference.retina_masks must be true. The canonical masks live on the original "
            "image canvas, and the native mask path puts predictions on the same canvas without "
            "this project resampling them itself."
        )
        raise ConfigError(msg)

    evaluator = raw["cocoeval"]
    if not isinstance(evaluator, Mapping):
        msg = "cocoeval: must be a mapping"
        raise ConfigError(msg)
    check_keys(evaluator, required=_COCOEVAL_KEYS, context="cocoeval")
    if str(evaluator["iou_type"]) != IOU_TYPE:
        msg = f"cocoeval.iou_type must be {IOU_TYPE!r}, got {evaluator['iou_type']!r}"
        raise ConfigError(msg)
    declared = tuple(round(float(value), 2) for value in evaluator["iou_thresholds"])
    if declared != IOU_THRESHOLDS:
        msg = (
            f"cocoeval.iou_thresholds must be exactly {list(IOU_THRESHOLDS)}. Changing the "
            "sweep after a candidate exists is the classic way to move a result."
        )
        raise ConfigError(msg)
    if tuple(int(value) for value in evaluator["max_dets"]) != MAX_DETS:
        msg = f"cocoeval.max_dets must be exactly {list(MAX_DETS)}"
        raise ConfigError(msg)

    metrics = raw["metrics"]
    if not isinstance(metrics, Mapping):
        msg = "metrics: must be a mapping"
        raise ConfigError(msg)
    check_keys(metrics, required=_METRIC_KEYS, context="metrics")
    if str(metrics["primary"]) != PRIMARY_METRIC:
        msg = f"metrics.primary must be {PRIMARY_METRIC!r}, got {metrics['primary']!r}"
        raise ConfigError(msg)
    if str(metrics["native_framework_status"]) != NATIVE_METRIC_STATUS:
        msg = f"metrics.native_framework_status must be {NATIVE_METRIC_STATUS!r}"
        raise ConfigError(msg)
    if metrics["composite_score"] is not False:
        msg = (
            "metrics.composite_score must be false. If the canonical AP and the direct IoU "
            "diagnostic ever disagree, the answer is to record the disagreement, not to invent "
            "a weighting that resolves it."
        )
        raise ConfigError(msg)

    return CanonicalEvaluationConfig(raw=json.loads(json.dumps(raw)))


# --- mask serialisation -------------------------------------------------------


def encode_mask(mask: np.ndarray) -> dict[str, Any]:
    """Encode a binary mask as COCO RLE on its own canvas.

    Uses ``pycocotools.mask.encode``, the reference implementation and the same
    one that decodes the canonical ground truth, so both sides of every IoU come
    from one library. Deliberately not a polygon: a polygon would approximate the
    boundary before the metric ever saw it.

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

    payload = {"size": list(rle["size"]), "counts": rle["counts"]}
    if isinstance(payload["counts"], str):
        payload["counts"] = payload["counts"].encode("utf-8")
    return mask_utils.decode(payload).astype(bool)


def prediction_record(
    *, image_id: int, category_id: int, mask: np.ndarray, score: float
) -> dict[str, Any]:
    """Build one COCO detection record for the evaluator.

    Args:
        image_id: The canonical image id the mask belongs to.
        category_id: The canonical class index.
        mask: The predicted mask, on the original image canvas.
        score: The prediction's confidence.

    Returns:
        A COCO-format detection.
    """
    return {
        "image_id": int(image_id),
        "category_id": int(category_id),
        "segmentation": encode_mask(mask),
        "score": float(score),
    }


def verify_prediction_geometry(
    detections: Sequence[Mapping[str, Any]], images: Mapping[int, tuple[int, int]]
) -> None:
    """Confirm every predicted mask sits on its image's original canvas.

    Args:
        detections: COCO-format detections.
        images: ``image_id -> (height, width)`` from the canonical document.

    Raises:
        CanonicalEvaluationError: If a detection names an unknown image, or its
            mask canvas differs from that image's original size. Silently
            resizing here would make every IoU a measurement of the resize.
    """
    for detection in detections:
        image_id = int(detection["image_id"])
        if image_id not in images:
            msg = f"prediction names an image outside the canonical document: {image_id}"
            raise CanonicalEvaluationError(msg)
        height, width = images[image_id]
        size = [int(value) for value in detection["segmentation"]["size"]]
        if size != [height, width]:
            msg = (
                f"prediction mask canvas {size} does not match the original image size "
                f"[{height}, {width}] for image {image_id}. Predictions and canonical masks "
                "must share the original canvas; resizing one to fit the other would make the "
                "metric a measurement of the resize."
            )
            raise CanonicalEvaluationError(msg)


# --- evaluation ---------------------------------------------------------------


def _clean(value: Any) -> float | None:
    """Turn a COCOeval score into a reportable number.

    Args:
        value: A raw score.

    Returns:
        The rounded value, or ``None`` where COCOeval signalled "not defined"
        with ``-1`` rather than a real score.
    """
    number = float(value)
    if number < 0:
        return None
    return round(number, METRIC_PRECISION)


def evaluate_canonical(
    ground_truth: Mapping[str, Any],
    detections: Sequence[Mapping[str, Any]],
    *,
    class_map: Mapping[str, int],
) -> dict[str, Any]:
    """Score detections against canonical masks with standard COCO semantics.

    Args:
        ground_truth: The canonical COCO document.
        detections: COCO-format detections on the original canvas.
        class_map: Class name to canonical category id.

    Returns:
        Global and per-class canonical mask AP, plus what COCOeval actually used.

    Raises:
        CanonicalEvaluationError: If COCOeval departs from the frozen semantics.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    images = {
        int(entry["id"]): (int(entry["height"]), int(entry["width"]))
        for entry in ground_truth["images"]
    }
    verify_prediction_geometry(detections, images)

    with contextlib.redirect_stdout(io.StringIO()):
        truth = COCO()
        truth.dataset = json.loads(json.dumps(ground_truth))
        truth.createIndex()
        # An empty detection set is a legitimate outcome, not a crash: a model
        # that predicts nothing scores zero rather than failing to be measured.
        predicted = truth.loadRes(list(detections)) if detections else None

    if predicted is None:
        names = {index: name for name, index in class_map.items()}
        return {
            "all_class_map50_95": 0.0,
            "all_class_map50": 0.0,
            "per_class": {name: {"AP@0.50:0.95": 0.0, "AP@0.50": 0.0} for name in names.values()},
            "detections": 0,
            "cocoeval": {
                "iou_thresholds": list(IOU_THRESHOLDS),
                "max_dets": list(MAX_DETS),
                "iou_type": IOU_TYPE,
            },
            "no_detections": True,
        }

    with contextlib.redirect_stdout(io.StringIO()):
        evaluator = COCOeval(truth, predicted, IOU_TYPE)
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

    used_thresholds = tuple(round(float(value), 2) for value in evaluator.params.iouThrs)
    used_max_dets = tuple(int(value) for value in evaluator.params.maxDets)
    if used_thresholds != IOU_THRESHOLDS:
        msg = f"COCOeval used IoU thresholds {used_thresholds}, not the frozen {IOU_THRESHOLDS}"
        raise CanonicalEvaluationError(msg)
    if used_max_dets != MAX_DETS:
        msg = f"COCOeval used maxDets {used_max_dets}, not the frozen {MAX_DETS}"
        raise CanonicalEvaluationError(msg)

    # precision has shape [T, R, K, A, M]: IoU threshold, recall, category, area
    # range, detection cap. Area index 0 is "all", cap index 2 is 100.
    precision = evaluator.eval["precision"]
    category_ids = list(evaluator.params.catIds)
    names = {index: name for name, index in class_map.items()}

    per_class: dict[str, dict[str, Any]] = {}
    for position, category_id in enumerate(category_ids):
        name = names.get(int(category_id), str(category_id))
        window = precision[:, :, position, 0, 2]
        at_fifty = precision[0, :, position, 0, 2]
        per_class[name] = {
            "AP@0.50:0.95": _clean(np.mean(window[window > -1])) if (window > -1).any() else None,
            "AP@0.50": _clean(np.mean(at_fifty[at_fifty > -1])) if (at_fifty > -1).any() else None,
        }

    return {
        "all_class_map50_95": _clean(evaluator.stats[0]),
        "all_class_map50": _clean(evaluator.stats[1]),
        "per_class": dict(sorted(per_class.items())),
        "detections": len(detections),
        "cocoeval": {
            "iou_thresholds": [float(value) for value in used_thresholds],
            "max_dets": list(used_max_dets),
            "iou_type": IOU_TYPE,
            "area_range": "all",
            "implementation": "pycocotools.cocoeval.COCOeval",
        },
        "no_detections": False,
    }


def validation_support(
    ground_truth: Mapping[str, Any], class_map: Mapping[str, int]
) -> dict[str, dict[str, Any]]:
    """Apply the frozen phase 7A support rule to the canonical validation split.

    The rule is imported rather than restated, and it names no class: the
    admitted set is its output, computed from the data actually evaluated.

    Args:
        ground_truth: The canonical COCO document.
        class_map: Class name to canonical category id.

    Returns:
        Per class, its counts and whether the rule admits it.
    """
    names = {index: name for name, index in class_map.items()}
    counts = {name: {"images": set(), "instances": 0} for name in class_map}
    for annotation in ground_truth["annotations"]:
        name = names[int(annotation["category_id"])]
        counts[name]["instances"] += 1
        counts[name]["images"].add(int(annotation["image_id"]))
    resolved: dict[str, dict[str, Any]] = {}
    for name in sorted(counts):
        images = len(counts[name]["images"])
        instances = counts[name]["instances"]
        resolved[name] = {
            "images": images,
            "instances": instances,
            "supported": images >= SUPPORT_MIN_POSITIVE_IMAGES
            and instances >= SUPPORT_MIN_INSTANCES,
        }
    return resolved


def supported_macro(
    per_class: Mapping[str, Mapping[str, Any]], support: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Average canonical per-class AP over the classes the frozen rule admits.

    Args:
        per_class: Canonical per-class AP.
        support: The support rule's verdict per class.

    Returns:
        The metric, the admitted classes and the arithmetic behind it.
    """
    admitted = [name for name in sorted(support) if support[name]["supported"]]
    values = [
        float(per_class[name]["AP@0.50:0.95"])
        for name in admitted
        if name in per_class and per_class[name]["AP@0.50:0.95"] is not None
    ]
    return {
        "metric": PRIMARY_METRIC,
        "value": round(sum(values) / len(values), METRIC_PRECISION) if values else None,
        "admitted_classes": admitted,
        "contributing_values": [round(value, METRIC_PRECISION) for value in values],
        "rule": (
            f"COMPARISON_SUPPORTED requires >= {SUPPORT_MIN_POSITIVE_IMAGES} positive validation "
            f"source images AND >= {SUPPORT_MIN_INSTANCES} validation instances"
        ),
        "rule_origin": "FROZEN_IN_PHASE_7A_REUSED_UNCHANGED",
        "names_no_class": True,
    }


def classify_delta(reference: float, candidate: float) -> dict[str, Any]:
    """Apply the frozen practical-equivalence margin to a candidate.

    Args:
        reference: The reference experiment's primary metric.
        candidate: The candidate's.

    Returns:
        The delta and its verdict.
    """
    delta = round(candidate - reference, METRIC_PRECISION)
    if delta > PRACTICAL_EQUIVALENCE_MARGIN:
        verdict = IMPROVES
    elif delta < -PRACTICAL_EQUIVALENCE_MARGIN:
        verdict = BELOW
    else:
        verdict = EQUIVALENT
    return {
        "reference": round(reference, METRIC_PRECISION),
        "candidate": round(candidate, METRIC_PRECISION),
        "delta": delta,
        "margin": PRACTICAL_EQUIVALENCE_MARGIN,
        "verdict": verdict,
        "margin_is_not_a_significance_test": True,
        "practically_equivalent_prefers_reference": verdict == EQUIVALENT,
    }
