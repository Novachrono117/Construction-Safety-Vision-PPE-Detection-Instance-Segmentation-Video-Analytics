"""The direct instance-mask IoU diagnostic.

Phase 8C. The academic deliverable requires an explicit IoU result, and mask
average precision does not supply one: AP is an averaged, ranking-sensitive
summary over IoU thresholds, so a reader cannot recover from it how similar a
predicted mask actually was to the ground truth it covered. This module computes
that directly, under a protocol frozen before S0 trained.

What it is **not**: a checkpoint-selection metric, a tuning signal, a threshold
search, or a replacement for mask mAP. It reports at one predeclared operating
point and decides nothing.

Four decisions carry the diagnostic's meaning, and all four are frozen in
``configs/segmentation_mask_iou_evaluation.yaml`` rather than chosen here.

**Ground truth is the canonical COCO instance segmentation, never the YOLO
adapter.** The adapter is a lossy derived view whose own round-trip error was
measured in phase 8A at mean mask IoU 0.973066. Scoring predictions against it
would fold that approximation into the model's result and quietly flatter or
penalise the model by an amount nobody could separate afterwards.

**Matching is one-to-one, per image and per class, and maximises total IoU.**
Greedy descending-IoU matching is a different rule that can return a worse total
assignment, so which rule is used has to be part of the protocol rather than an
implementation detail. The assignment is solved with
:func:`scipy.optimize.linear_sum_assignment`, the reference implementation.

**A zero-overlap assignment is not a match.** The assignment problem will happily
pair a ground-truth instance with a prediction they share no pixel with when the
alternative is leaving both unassigned. Counting that as a matched instance would
inflate coverage with pairs that have nothing to do with each other, so such
pairs are filtered out after the assignment and counted as an unmatched
ground truth plus an unmatched prediction.

**An unmatched ground-truth instance contributes zero, it is not dropped.** That
is the difference between the two headline numbers:
:data:`MATCHED_MASK_IOU_MEAN` describes mask quality *where the model found
something*, and :data:`GT_NORMALIZED_MASK_IOU` divides by every canonical
instance, so missed objects reduce it. Reporting only the first would describe a
model that finds one object perfectly as near-perfect.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from construction_safety_vision.config import ConfigError, check_keys

CONFIG_SCHEMA_VERSION = 1
"""Schema version of ``configs/segmentation_mask_iou_evaluation.yaml``."""

PROTOCOL_NAME = "DIRECT_INSTANCE_MASK_IOU_DIAGNOSTIC"
"""What this protocol is, recorded in every artifact that reports it."""

GROUND_TRUTH_SOURCE = "CANONICAL_COCO_INSTANCE_SEGMENTATION"
"""The only admissible ground truth. Never the YOLO segmentation adapter."""

MATCHING_ALGORITHM = "SCIPY_LINEAR_SUM_ASSIGNMENT_MAXIMIZE_MASK_IOU"
"""One-to-one assignment maximising total mask IoU, per image and per class."""

MATCHING_SCOPE = "PER_IMAGE_PER_CLASS_ONE_TO_ONE"
"""Predictions may only match ground truth of the same class in the same image."""

ZERO_OVERLAP_POLICY = "ASSIGNED_PAIRS_WITH_ZERO_IOU_ARE_NOT_MATCHES"
"""A pair sharing no pixel is discarded rather than reported as a match."""

UNMATCHED_GT_POLICY = "CONTRIBUTES_ZERO_TO_GT_NORMALIZED_MASK_IOU"
"""A missed instance lowers the headline diagnostic instead of vanishing from it."""

FORBIDDEN_SPLIT = "test"
"""The protected split, which this protocol may not name or evaluate."""

EVALUATION_SPLIT = "validation"
"""The only split the diagnostic runs on."""

MATCHED_MASK_IOU_MEAN = "matched_mask_iou_mean"
"""Mean IoU over assigned pairs that actually overlap."""

GT_NORMALIZED_MASK_IOU = "gt_normalized_mask_iou"
"""Summed assigned IoU divided by every canonical ground-truth instance."""

GT_MATCH_COVERAGE = "gt_match_coverage"
"""Fraction of ground truth with an overlapping same-class assignment."""

GT_IOU50_COVERAGE = "gt_iou50_coverage"
"""Fraction of ground truth assigned at IoU >= 0.50."""

GT_IOU75_COVERAGE = "gt_iou75_coverage"
"""Fraction of ground truth assigned at IoU >= 0.75."""

HEADLINE_DIAGNOSTICS: tuple[str, ...] = (
    MATCHED_MASK_IOU_MEAN,
    GT_NORMALIZED_MASK_IOU,
    GT_MATCH_COVERAGE,
    GT_IOU50_COVERAGE,
    GT_IOU75_COVERAGE,
)
"""Every diagnostic this protocol reports, frozen before S0 ran.

Reported side by side and never collapsed into one score: each answers a
different question, and a single blended number would let a coverage failure
hide behind good mask quality on the instances that were found.
"""

COVERAGE_THRESHOLDS: tuple[float, ...] = (0.50, 0.75)
"""IoU levels the coverage diagnostics report at."""

METRIC_PRECISION = 6
"""Decimal places reported. Matches the detection phases."""

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
"""A digest is 64 lowercase hexadecimal characters, or it is not a digest."""


class MaskIoUEvaluationError(RuntimeError):
    """Raised when the diagnostic cannot run as specified."""


# --- protocol -----------------------------------------------------------------


@dataclass(frozen=True)
class MaskIoUEvaluationConfig:
    """The frozen direct mask-IoU diagnostic protocol.

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
    "is_not",
    "experiment",
    "split",
    "ground_truth_source",
    "ground_truth_document",
    "canonical_task_manifest",
    "class_map_sha256",
    "split_assignment_sha256",
    "checkpoint",
    "inference",
    "matching",
    "diagnostics",
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
    "threshold_policy",
)

_MATCHING_KEYS: tuple[str, ...] = (
    "algorithm",
    "implementation",
    "scope",
    "objective",
    "zero_overlap_policy",
    "unmatched_gt_policy",
    "deterministic",
)

_DIAGNOSTIC_KEYS: tuple[str, ...] = (
    "headline",
    "coverage_thresholds",
    "per_class",
    "counts",
    "combined_score",
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


def load_mask_iou_config(path: str | Path) -> MaskIoUEvaluationConfig:
    """Load and validate the frozen direct mask-IoU protocol.

    Parsing is strict: an unknown key raises rather than being ignored, so a
    typo cannot silently change what a reported diagnostic measured.

    Args:
        path: Configuration file.

    Returns:
        The parsed protocol.

    Raises:
        ConfigError: If the file is missing, malformed, names the protected
            split, evaluates against the derived adapter instead of the
            canonical masks, declares a matching rule other than the frozen one,
            or introduces a combined score.
    """
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        msg = f"Direct mask-IoU protocol not found: {config_path.name}"
        raise ConfigError(msg)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        msg = f"Direct mask-IoU protocol is not valid YAML: {config_path.name} ({exc})"
        raise ConfigError(msg) from exc
    if not isinstance(raw, Mapping):
        msg = f"Direct mask-IoU protocol must be a mapping: {config_path.name}"
        raise ConfigError(msg)

    check_keys(raw, required=_CONFIG_KEYS, context=config_path.name)

    if int(raw["schema_version"]) != CONFIG_SCHEMA_VERSION:
        msg = f"schema_version must be {CONFIG_SCHEMA_VERSION}, got {raw['schema_version']!r}"
        raise ConfigError(msg)
    if str(raw["protocol"]) != PROTOCOL_NAME:
        msg = f"protocol must be {PROTOCOL_NAME!r}, got {raw['protocol']!r}"
        raise ConfigError(msg)
    if str(raw["split"]) != EVALUATION_SPLIT:
        msg = (
            f"split must be {EVALUATION_SPLIT!r}, got {raw['split']!r}. This diagnostic runs on "
            "the development validation split only."
        )
        raise ConfigError(msg)
    if _contains_forbidden_split(dict(raw)):
        msg = (
            "the direct mask-IoU protocol references the protected split. The holdout takes no "
            "part in a diagnostic run before the models are frozen."
        )
        raise ConfigError(msg)

    if str(raw["ground_truth_source"]) != GROUND_TRUTH_SOURCE:
        msg = (
            f"ground_truth_source must be {GROUND_TRUTH_SOURCE!r}, got "
            f"{raw['ground_truth_source']!r}. Scoring against the derived YOLO adapter would "
            "fold its own measured approximation into the model's result."
        )
        raise ConfigError(msg)
    document = str(raw["ground_truth_document"])
    if "adapter" in document.lower():
        msg = (
            f"ground_truth_document {document!r} points at an adapter. The canonical COCO "
            "instance segmentation is the only admissible ground truth."
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
    if inference["augment"] is not False:
        msg = "inference.augment must be false: no test-time augmentation is authorised"
        raise ConfigError(msg)
    if inference["retina_masks"] is not True:
        msg = (
            "inference.retina_masks must be true. The canonical masks live on the original image "
            "canvas, and the native mask path is what puts predictions on the same canvas "
            "without this project resampling them itself."
        )
        raise ConfigError(msg)

    matching = raw["matching"]
    if not isinstance(matching, Mapping):
        msg = "matching: must be a mapping"
        raise ConfigError(msg)
    check_keys(matching, required=_MATCHING_KEYS, context="matching")
    for name, expected in (
        ("algorithm", MATCHING_ALGORITHM),
        ("scope", MATCHING_SCOPE),
        ("zero_overlap_policy", ZERO_OVERLAP_POLICY),
        ("unmatched_gt_policy", UNMATCHED_GT_POLICY),
    ):
        if str(matching[name]) != expected:
            msg = f"matching.{name} must be {expected!r}, got {matching[name]!r}"
            raise ConfigError(msg)
    if matching["deterministic"] is not True:
        msg = "matching.deterministic must be true"
        raise ConfigError(msg)

    diagnostics = raw["diagnostics"]
    if not isinstance(diagnostics, Mapping):
        msg = "diagnostics: must be a mapping"
        raise ConfigError(msg)
    check_keys(diagnostics, required=_DIAGNOSTIC_KEYS, context="diagnostics")
    if tuple(diagnostics["headline"]) != HEADLINE_DIAGNOSTICS:
        msg = f"diagnostics.headline must be exactly {list(HEADLINE_DIAGNOSTICS)}"
        raise ConfigError(msg)
    if tuple(float(value) for value in diagnostics["coverage_thresholds"]) != COVERAGE_THRESHOLDS:
        msg = f"diagnostics.coverage_thresholds must be exactly {list(COVERAGE_THRESHOLDS)}"
        raise ConfigError(msg)
    if diagnostics["combined_score"] is not False:
        msg = (
            "diagnostics.combined_score must be false. The five diagnostics answer different "
            "questions; blending them would let a coverage failure hide behind good mask "
            "quality on the instances that were found."
        )
        raise ConfigError(msg)

    return MaskIoUEvaluationConfig(raw=json.loads(json.dumps(raw)))


# --- geometry -----------------------------------------------------------------


@dataclass(frozen=True)
class MaskInstance:
    """One instance mask on the original image canvas.

    Attributes:
        class_id: Frozen class index.
        mask: Boolean array, ``True`` inside the instance.
    """

    class_id: int
    mask: np.ndarray

    def area(self) -> int:
        """Count the instance's pixels.

        Returns:
            The number of set pixels.
        """
        return int(np.count_nonzero(self.mask))


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    """Intersection over union of two boolean masks.

    Args:
        first: A boolean mask.
        second: A boolean mask of the same shape.

    Returns:
        The IoU in ``[0, 1]``. Two empty masks give ``0.0`` rather than an
        undefined ``0/0``: an empty prediction has not found anything, and
        reporting perfect agreement for it would be wrong.

    Raises:
        MaskIoUEvaluationError: If the shapes differ, which would mean a
            prediction and its ground truth were rasterised on different
            canvases.
    """
    left = np.asarray(first, dtype=bool)
    right = np.asarray(second, dtype=bool)
    if left.shape != right.shape:
        msg = (
            f"cannot compare masks of shape {left.shape} and {right.shape}: predictions and "
            "canonical masks must share the original image canvas"
        )
        raise MaskIoUEvaluationError(msg)
    intersection = int(np.count_nonzero(left & right))
    if intersection == 0:
        return 0.0
    union = int(np.count_nonzero(left | right))
    return intersection / union


def iou_matrix(
    ground_truth: Sequence[MaskInstance], predictions: Sequence[MaskInstance]
) -> np.ndarray:
    """Build the pairwise mask-IoU matrix for one image and one class.

    Args:
        ground_truth: Canonical instances.
        predictions: Predicted instances.

    Returns:
        A ``(len(ground_truth), len(predictions))`` array of IoU values.
    """
    matrix = np.zeros((len(ground_truth), len(predictions)), dtype=np.float64)
    for row, truth in enumerate(ground_truth):
        for column, prediction in enumerate(predictions):
            matrix[row, column] = mask_iou(truth.mask, prediction.mask)
    return matrix


def assign_maximum_total_iou(matrix: np.ndarray) -> list[tuple[int, int]]:
    """Solve the one-to-one assignment that maximises total IoU.

    Uses :func:`scipy.optimize.linear_sum_assignment`, the reference
    implementation, rather than a greedy pass: greedy matching takes the single
    best pair first and can be forced into a worse total, and which rule is used
    is part of the frozen protocol rather than an implementation detail.

    Args:
        matrix: A ``(n_gt, n_pred)`` array of IoU values.

    Returns:
        ``(gt_index, prediction_index)`` pairs, sorted by ground-truth index so
        the output order does not depend on the solver's internals. Pairs with
        zero IoU are **kept** here; filtering them is the caller's step, so the
        assignment and the match policy stay separable and separately testable.
    """
    from scipy.optimize import linear_sum_assignment

    if matrix.size == 0:
        return []
    rows, columns = linear_sum_assignment(matrix, maximize=True)
    return sorted(zip(rows.tolist(), columns.tolist(), strict=True))


# --- matching -----------------------------------------------------------------


@dataclass(frozen=True)
class MatchedPair:
    """One assigned ground-truth/prediction pair.

    Attributes:
        image_id: The image both belong to.
        class_id: The class both carry; matching never crosses classes.
        gt_index: Index into that image-and-class's ground-truth instances.
        prediction_index: Index into its predicted instances.
        iou: Their mask IoU.
    """

    image_id: str
    class_id: int
    gt_index: int
    prediction_index: int
    iou: float


@dataclass(frozen=True)
class ImageMatching:
    """The assignment outcome for one image.

    Attributes:
        image_id: The image.
        pairs: Assigned pairs with non-zero overlap, in deterministic order.
        gt_counts: Ground-truth instance count per class.
        prediction_counts: Predicted instance count per class.
    """

    image_id: str
    pairs: tuple[MatchedPair, ...]
    gt_counts: dict[int, int]
    prediction_counts: dict[int, int]


def _grouped(instances: Iterable[MaskInstance]) -> dict[int, list[MaskInstance]]:
    """Group instances by class, preserving their input order.

    Args:
        instances: Instances to group.

    Returns:
        Instances keyed by class index.
    """
    grouped: dict[int, list[MaskInstance]] = {}
    for instance in instances:
        grouped.setdefault(int(instance.class_id), []).append(instance)
    return grouped


def match_image(
    image_id: str,
    ground_truth: Sequence[MaskInstance],
    predictions: Sequence[MaskInstance],
) -> ImageMatching:
    """Match one image's predictions to its canonical instances.

    Same class only, one-to-one, maximising total IoU within each class. A pair
    the solver assigned but which shares no pixel is dropped: the assignment
    problem pairs those when the alternative is leaving both unassigned, and
    calling that a match would inflate coverage with unrelated objects.

    Args:
        image_id: The image both sets belong to.
        ground_truth: Canonical instances on the original canvas.
        predictions: Predicted instances on the same canvas.

    Returns:
        The image's assignment outcome.
    """
    truth_by_class = _grouped(ground_truth)
    predictions_by_class = _grouped(predictions)

    pairs: list[MatchedPair] = []
    for class_id in sorted(set(truth_by_class) | set(predictions_by_class)):
        class_truth = truth_by_class.get(class_id, [])
        class_predictions = predictions_by_class.get(class_id, [])
        if not class_truth or not class_predictions:
            continue
        matrix = iou_matrix(class_truth, class_predictions)
        for gt_index, prediction_index in assign_maximum_total_iou(matrix):
            value = float(matrix[gt_index, prediction_index])
            if value <= 0.0:
                continue
            pairs.append(
                MatchedPair(
                    image_id=image_id,
                    class_id=class_id,
                    gt_index=gt_index,
                    prediction_index=prediction_index,
                    iou=value,
                )
            )

    return ImageMatching(
        image_id=image_id,
        pairs=tuple(sorted(pairs, key=lambda pair: (pair.class_id, pair.gt_index))),
        gt_counts={class_id: len(items) for class_id, items in truth_by_class.items()},
        prediction_counts={
            class_id: len(items) for class_id, items in predictions_by_class.items()
        },
    )


# --- aggregation --------------------------------------------------------------


@dataclass
class Tally:
    """Running counts and IoU sums for one class or for everything.

    Attributes:
        gt_count: Canonical instances seen.
        prediction_count: Predicted instances seen.
        matched_count: Assigned pairs with non-zero overlap.
        iou_sum: Their summed IoU.
        iou50: Assigned pairs at IoU >= 0.50.
        iou75: Assigned pairs at IoU >= 0.75.
    """

    gt_count: int = 0
    prediction_count: int = 0
    matched_count: int = 0
    iou_sum: float = 0.0
    iou50: int = 0
    iou75: int = 0

    def summary(self) -> dict[str, Any]:
        """Render the frozen diagnostics from the counts.

        ``matched_mask_iou_mean`` is ``None`` rather than ``0.0`` when nothing
        matched: a mean over an empty set does not exist, and writing zero would
        be indistinguishable from a model whose masks were all wrong.

        Returns:
            The diagnostics, rounded to the reported precision.
        """
        matched_mean = (
            round(self.iou_sum / self.matched_count, METRIC_PRECISION)
            if self.matched_count
            else None
        )
        if self.gt_count:
            normalised = round(self.iou_sum / self.gt_count, METRIC_PRECISION)
            coverage = round(self.matched_count / self.gt_count, METRIC_PRECISION)
            coverage50 = round(self.iou50 / self.gt_count, METRIC_PRECISION)
            coverage75 = round(self.iou75 / self.gt_count, METRIC_PRECISION)
        else:
            normalised = coverage = coverage50 = coverage75 = None
        return {
            "gt_count": self.gt_count,
            "prediction_count": self.prediction_count,
            "matched_count": self.matched_count,
            "unmatched_gt": self.gt_count - self.matched_count,
            "unmatched_predictions": self.prediction_count - self.matched_count,
            MATCHED_MASK_IOU_MEAN: matched_mean,
            GT_NORMALIZED_MASK_IOU: normalised,
            GT_MATCH_COVERAGE: coverage,
            GT_IOU50_COVERAGE: coverage50,
            GT_IOU75_COVERAGE: coverage75,
        }


@dataclass
class DiagnosticResult:
    """The complete diagnostic outcome.

    Attributes:
        overall: Counts across every class.
        per_class: Counts keyed by class index.
        images: How many images were evaluated.
    """

    overall: Tally = field(default_factory=Tally)
    per_class: dict[int, Tally] = field(default_factory=dict)
    images: int = 0

    def as_dict(self, class_names: Mapping[int, str]) -> dict[str, Any]:
        """Render the result for the machine-readable artifact.

        Args:
            class_names: Class index to name.

        Returns:
            Global and per-class diagnostics.
        """
        return {
            "images": self.images,
            "global": self.overall.summary(),
            "per_class": {
                class_names[class_id]: self.per_class[class_id].summary()
                for class_id in sorted(self.per_class)
            },
        }


def accumulate(matchings: Iterable[ImageMatching]) -> DiagnosticResult:
    """Fold per-image assignments into global and per-class diagnostics.

    Args:
        matchings: One entry per evaluated image.

    Returns:
        The accumulated result.
    """
    result = DiagnosticResult()
    for matching in matchings:
        result.images += 1
        for class_id, count in matching.gt_counts.items():
            result.per_class.setdefault(class_id, Tally()).gt_count += count
            result.overall.gt_count += count
        for class_id, count in matching.prediction_counts.items():
            result.per_class.setdefault(class_id, Tally()).prediction_count += count
            result.overall.prediction_count += count
        for pair in matching.pairs:
            tally = result.per_class.setdefault(pair.class_id, Tally())
            for target in (tally, result.overall):
                target.matched_count += 1
                target.iou_sum += pair.iou
                if pair.iou >= COVERAGE_THRESHOLDS[0]:
                    target.iou50 += 1
                if pair.iou >= COVERAGE_THRESHOLDS[1]:
                    target.iou75 += 1
    return result


def evaluate(
    samples: Iterable[tuple[str, Sequence[MaskInstance], Sequence[MaskInstance]]],
) -> DiagnosticResult:
    """Run the diagnostic over a sequence of images.

    Args:
        samples: ``(image_id, ground_truth, predictions)`` triples.

    Returns:
        The accumulated result.
    """
    return accumulate(
        match_image(image_id, ground_truth, predictions)
        for image_id, ground_truth, predictions in samples
    )
