"""Spatial measurements, box proxies and association, for phase 10B.

Every definition here implements one the phase 10A protocol froze. Nothing is
redefined, nothing is added, and the seven spatial features and their box
counterparts are exactly the frozen set - a test asserts that.

Three things are worth stating plainly.

**A mask measurement and its box proxy are computed from the same instance.**
The comparison asks what the mask adds, so both sides come from one prediction:
the model's own mask and the model's own box. Comparing a mask from one model
against a box from another would confound the geometry with the model.

**The association analysis is run twice, from two box sources, and both are
labelled.** The frozen ``box_rule`` says "the models' predicted boxes", which
does not by itself say whose. Applying it to the segmenter's own boxes isolates
the geometry - same model, same instances, only the shape representation
differs - and is therefore the primary reading. Applying it to the detector's
boxes describes what a box-only pipeline would actually produce, and is
reported beside it. Neither is presented as the other.

**``VISIBLE_PPE_COVERAGE_PROXY`` is the one frozen quantity whose definition is
qualitative.** The protocol describes it as "an overlap-derived operational
proxy for how much visible PPE support a person instance has", which fixes its
inputs and its direction but not an exact formula. It is implemented as the
literal reading of that sentence - associated PPE mask area over person mask
area - and it carries ``INTERPRETIVE_OPERATIONAL_PROXY`` everywhere it appears.
It is not a compliance measure, and the project holds no compliance ground
truth that could make it one.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from construction_safety_vision.detector_segmenter_comparison import (
    ASSOCIATION_CATEGORIES,
    SPATIAL_FEATURES,
)

METRIC_PRECISION = 6
"""Decimal places every reported statistic is rounded to."""

PURE_GEOMETRIC = "PURE_GEOMETRIC_MEASUREMENT"
"""A quantity computed from pixels or coordinates alone."""

INTERPRETIVE_PROXY = "INTERPRETIVE_OPERATIONAL_PROXY"
"""A quantity whose meaning depends on an operational reading, not just geometry."""

FEATURE_SEMANTICS: dict[str, str] = {
    "INSTANCE_AREA_PIXELS": PURE_GEOMETRIC,
    "MASK_TO_BOX_FILL_RATIO": PURE_GEOMETRIC,
    "MASK_CENTROID": PURE_GEOMETRIC,
    "SHAPE_EXTENT": PURE_GEOMETRIC,
    "PERSON_PPE_MASK_INTERSECTION": PURE_GEOMETRIC,
    "PERSON_PPE_MASK_CONTAINMENT": PURE_GEOMETRIC,
    "VISIBLE_PPE_COVERAGE_PROXY": INTERPRETIVE_PROXY,
}
"""Which frozen features are geometry and which require an operational reading."""

NO_BOX_PROXY = "NO_BOX_ONLY_EQUIVALENT"

PERSON_CLASS = "person"
HELMET_CLASSES: tuple[str, ...] = ("helmet_loose", "helmet_on_head")
VEST_CLASSES: tuple[str, ...] = ("vest_loose", "vest_on_body")
PPE_CLASSES: tuple[str, ...] = (*HELMET_CLASSES, *VEST_CLASSES)

RARE_CLASS = "vest_loose"
RARE_CLASS_STATUS = "DESCRIPTIVE_HIGH_UNCERTAINTY"

AGREE, BOX_ONLY, MASK_ONLY, NEITHER = ASSOCIATION_CATEGORIES

STATISTIC_NAMES: tuple[str, ...] = (
    "count",
    "mean",
    "median",
    "std",
    "p25",
    "p75",
    "p90",
    "p95",
    "min",
    "max",
)
"""The descriptive set reported for every continuous frozen quantity.

No inferential test appears anywhere: none was predeclared, and computing one
now would be choosing the test after seeing the data.
"""

GEOMETRY_ISOLATING = "SEGMENTER_OWN_BOXES_GEOMETRY_ISOLATING"
"""Box source that holds the model constant and varies only the shape representation."""

PIPELINE_LEVEL = "DETECTOR_BOXES_PIPELINE_LEVEL"
"""Box source that describes what a box-only pipeline would actually produce."""


class SpatialAnalysisError(RuntimeError):
    """Raised when a spatial measurement cannot be computed as frozen."""


# --- one predicted instance -------------------------------------------------------


@dataclass(frozen=True)
class Instance:
    """One operational prediction on the original image canvas.

    Attributes:
        image_id: Canonical source image id.
        class_name: One of the five frozen classes.
        score: The model's confidence.
        box: ``(x1, y1, x2, y2)`` in original image coordinates.
        mask: Boolean array on the original canvas, or ``None`` for a
            detector prediction, which has no mask by construction.
    """

    image_id: str
    class_name: str
    score: float
    box: tuple[float, float, float, float]
    mask: np.ndarray | None = field(default=None, repr=False)

    def box_area(self) -> float:
        """Area of the predicted box.

        Returns:
            The area in square pixels, never negative.
        """
        x1, y1, x2, y2 = self.box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def box_center(self) -> tuple[float, float]:
        """Centre of the predicted box.

        Returns:
            ``(x, y)`` in original image coordinates.
        """
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def mask_area(self) -> int:
        """Foreground pixel count of the predicted mask.

        Returns:
            The count, or 0 when the instance carries no mask.

        Raises:
            SpatialAnalysisError: If the instance carries no mask.
        """
        if self.mask is None:
            msg = f"{self.class_name} instance on {self.image_id} carries no mask"
            raise SpatialAnalysisError(msg)
        return int(np.count_nonzero(self.mask))


# --- the seven frozen spatial features ----------------------------------------------


def instance_area_pixels(instance: Instance) -> int:
    """``INSTANCE_AREA_PIXELS``: foreground pixels of the predicted mask.

    Args:
        instance: A segmenter prediction.

    Returns:
        The pixel count on the original canvas.
    """
    return instance.mask_area()


def mask_to_box_fill_ratio(instance: Instance) -> float | None:
    """``MASK_TO_BOX_FILL_RATIO``: mask area over the model's own box area.

    Args:
        instance: A segmenter prediction.

    Returns:
        The ratio, or ``None`` when the predicted box has no area, because a
        ratio over zero is undefined rather than zero or infinite.
    """
    area = instance.box_area()
    if area <= 0:
        return None
    return instance.mask_area() / area


def mask_centroid(instance: Instance) -> tuple[float, float] | None:
    """``MASK_CENTROID``: centroid of the mask's foreground pixels.

    Args:
        instance: A segmenter prediction.

    Returns:
        ``(x, y)`` in original image coordinates, or ``None`` for an empty mask.
    """
    if instance.mask is None:
        msg = f"{instance.class_name} instance on {instance.image_id} carries no mask"
        raise SpatialAnalysisError(msg)
    rows, columns = np.nonzero(instance.mask)
    if rows.size == 0:
        return None
    return (float(columns.mean()), float(rows.mean()))


def shape_extent(instance: Instance) -> float | None:
    """``SHAPE_EXTENT``: foreground support over the mask's own tight rectangle.

    Deliberately the mask's own bounding rectangle, not the model's predicted
    box: this describes how far the shape departs from filling a rectangle,
    which is a property of the shape rather than of the box regression.

    Args:
        instance: A segmenter prediction.

    Returns:
        The ratio in ``(0, 1]``, or ``None`` for an empty mask.
    """
    if instance.mask is None:
        msg = f"{instance.class_name} instance on {instance.image_id} carries no mask"
        raise SpatialAnalysisError(msg)
    rows, columns = np.nonzero(instance.mask)
    if rows.size == 0:
        return None
    height = int(rows.max() - rows.min()) + 1
    width = int(columns.max() - columns.min()) + 1
    return float(rows.size) / float(height * width)


def mask_intersection(ppe: Instance, person: Instance) -> int:
    """``PERSON_PPE_MASK_INTERSECTION``: shared foreground pixels.

    Args:
        ppe: A PPE instance carrying a mask.
        person: A person instance carrying a mask.

    Returns:
        The count of pixels in both masks.

    Raises:
        SpatialAnalysisError: If either instance carries no mask, or the two
            masks sit on different canvases - which would make the count a
            measurement of the mismatch.
    """
    if ppe.mask is None or person.mask is None:
        msg = "mask intersection needs two masks"
        raise SpatialAnalysisError(msg)
    if ppe.mask.shape != person.mask.shape:
        msg = (
            f"masks sit on different canvases ({ppe.mask.shape} and {person.mask.shape}); "
            "both must be on the original image canvas"
        )
        raise SpatialAnalysisError(msg)
    return int(np.count_nonzero(np.logical_and(ppe.mask, person.mask)))


def mask_containment(ppe: Instance, person: Instance) -> float | None:
    """``PERSON_PPE_MASK_CONTAINMENT``: intersection over the PPE mask's own area.

    Args:
        ppe: A PPE instance carrying a mask.
        person: A person instance carrying a mask.

    Returns:
        The fraction of the PPE mask inside the person mask, or ``None`` when
        the PPE mask is empty.
    """
    area = ppe.mask_area()
    if area == 0:
        return None
    return mask_intersection(ppe, person) / area


def box_intersection(first: Instance, second: Instance) -> float:
    """``BOX_INTERSECTION_AREA``: the box proxy for mask intersection.

    Args:
        first: One instance.
        second: Another.

    Returns:
        The overlapping area in square pixels.
    """
    ax1, ay1, ax2, ay2 = first.box
    bx1, by1, bx2, by2 = second.box
    width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    height = max(0.0, min(ay2, by2) - max(ay1, by1))
    return width * height


def box_containment(ppe: Instance, person: Instance) -> float | None:
    """``BOX_INTERSECTION_OVER_PPE_BOX_AREA``: the box proxy for containment.

    Args:
        ppe: A PPE instance.
        person: A person instance.

    Returns:
        The fraction of the PPE box inside the person box, or ``None`` when the
        PPE box has no area.
    """
    area = ppe.box_area()
    if area <= 0:
        return None
    return box_intersection(ppe, person) / area


def centroid_displacement(instance: Instance) -> float | None:
    """Distance between the model's box centre and its mask centroid.

    The box centre is the frozen proxy for ``MASK_CENTROID``, so this is the
    magnitude of the refinement the mask provides for that one quantity.

    Args:
        instance: A segmenter prediction.

    Note:
        The centroid is a mean over pixel indices, so it lives at pixel
        centres, while the box is in continuous coordinates. A mask that
        perfectly fills its box therefore reports about 0.71 px rather than
        exactly 0 - a half-pixel in each axis. The offset is constant and
        far below the displacements this quantity exists to describe, but it
        is stated so nobody reads a small non-zero value as a real shift.

    Returns:
        The Euclidean distance in pixels, or ``None`` for an empty mask.
    """
    centroid = mask_centroid(instance)
    if centroid is None:
        return None
    cx, cy = instance.box_center()
    return float(np.hypot(centroid[0] - cx, centroid[1] - cy))


# --- the frozen association rule ------------------------------------------------------


def associate(
    ppe: Instance,
    people: Sequence[Instance],
    *,
    containment_floor: float,
    use_masks: bool,
) -> int | None:
    """Apply the frozen association rule to one PPE instance.

    The rule, as frozen: associate the PPE instance with the person it overlaps
    most, subject to the containment floor. Ties break on the highest overlap
    and then the lowest person index, so the outcome does not depend on
    iteration order.

    Args:
        ppe: The PPE instance.
        people: Candidate person instances, in a stable order.
        containment_floor: The frozen minimum containment.
        use_masks: Whether to use mask geometry or the box proxy.

    Returns:
        The index of the associated person, or ``None`` when no candidate
        clears the floor.
    """
    best_index: int | None = None
    best_value = -1.0
    for index, person in enumerate(people):
        value = mask_containment(ppe, person) if use_masks else box_containment(ppe, person)
        if value is None or value < containment_floor:
            continue
        if value > best_value:
            best_value = value
            best_index = index
    return best_index


TAXONOMY_EXCEPTION = "BOTH_RULES_ASSOCIATE_DIFFERENT_PERSON"
"""The state phase 10A's four categories do not describe.

Both rules produce an association, but to **different** people. That is not
agreement, and it is neither box-only nor mask-only, so none of the four frozen
categories is true of it.

It is deliberately **not** a fifth peer category. Adding one after seeing data
is precisely what a frozen taxonomy exists to prevent, and it would also
misrepresent the situation: the four categories were predeclared and this state
was not, so it is an exception to their coverage rather than a sibling of them.
It is recorded as :data:`TAXONOMY_EXCEPTION_STATUS`, counted separately, and
excluded from the classified denominator.
"""

TAXONOMY_EXCEPTION_STATUS = "UNCLASSIFIED_BY_FROZEN_FOUR_CATEGORY_TAXONOMY"
"""What an exception is: outside the frozen taxonomy, not a new member of it."""

TAXONOMY_NON_EXHAUSTIVE = "FROZEN_TAXONOMY_NON_EXHAUSTIVE_FOR_OBSERVED_DATA"
"""Recorded when at least one observed relationship falls outside the four."""

TAXONOMY_EXHAUSTIVE = "FROZEN_TAXONOMY_EXHAUSTIVE_FOR_OBSERVED_DATA"
"""Recorded when every observed relationship fits one of the four."""

ASSOCIATION_DISAGREEMENT = "ASSOCIATION_RULE_DISAGREEMENT"
"""What a different-person outcome is.

Two deterministic geometric rules reached different conclusions. There is no
person-PPE association ground truth in this project, so neither conclusion can
be called wrong and the word "error" is not available.
"""


def association_category(mask_choice: int | None, box_choice: int | None) -> str:
    """Classify one candidate relationship.

    Returns one of the four frozen categories, or :data:`TAXONOMY_EXCEPTION`
    for the state the frozen four do not cover.

    Args:
        mask_choice: The person the mask rule chose, or ``None``.
        box_choice: The person the box rule chose, or ``None``.

    Returns:
        The category.
    """
    if mask_choice is None and box_choice is None:
        return NEITHER
    if mask_choice is not None and box_choice is None:
        return MASK_ONLY
    if mask_choice is None and box_choice is not None:
        return BOX_ONLY
    return AGREE if mask_choice == box_choice else TAXONOMY_EXCEPTION


# --- descriptive statistics -------------------------------------------------------------


def describe(values: Sequence[float]) -> dict[str, Any]:
    """Summarise a continuous quantity with the frozen statistic set.

    No inferential test is computed: none was predeclared, and choosing one now
    would be choosing it after seeing the data.

    Args:
        values: The observations.

    Returns:
        The descriptive statistics, with ``None`` where an empty sample makes a
        statistic undefined.
    """
    array = np.asarray([float(value) for value in values], dtype=float)
    if array.size == 0:
        return {name: (0 if name == "count" else None) for name in STATISTIC_NAMES}

    def rounded(value: float) -> float:
        return round(float(value), METRIC_PRECISION)

    return {
        "count": int(array.size),
        "mean": rounded(array.mean()),
        "median": rounded(np.median(array)),
        "std": rounded(array.std(ddof=1)) if array.size > 1 else None,
        "p25": rounded(np.percentile(array, 25)),
        "p75": rounded(np.percentile(array, 75)),
        "p90": rounded(np.percentile(array, 90)),
        "p95": rounded(np.percentile(array, 95)),
        "min": rounded(array.min()),
        "max": rounded(array.max()),
    }


def tally(categories: Sequence[str]) -> dict[str, Any]:
    """Count outcomes, separating the frozen four from what they do not cover.

    The frozen categories keep their own counts and their own denominator: a
    percentage among *classified* relationships is a statement about the
    taxonomy that was declared, and mixing an undeclared state into that
    denominator would quietly change what those percentages mean.

    Taxonomy coverage is reported separately. It is protocol bookkeeping - how
    much of the observed data the frozen taxonomy describes - and not a
    spatial-performance metric.

    Args:
        categories: One entry per candidate relationship.

    Returns:
        Frozen-category counts, percentages among classified relationships,
        the exception count, and the coverage arithmetic.
    """
    counts = {
        name: sum(1 for item in categories if item == name) for name in ASSOCIATION_CATEGORIES
    }
    exceptions = sum(1 for item in categories if item == TAXONOMY_EXCEPTION)
    total = len(categories)
    classified = sum(counts.values())
    return {
        "total_relationships": total,
        "classified_relationships": classified,
        "taxonomy_exceptions": exceptions,
        "taxonomy_coverage": round(classified / total, 6) if total else None,
        "taxonomy_status": (TAXONOMY_NON_EXHAUSTIVE if exceptions else TAXONOMY_EXHAUSTIVE),
        "frozen_category_counts": counts,
        "frozen_category_percentages": {
            name: round(100.0 * count / classified, 3) if classified else None
            for name, count in counts.items()
        },
        "percentage_denominator": "classified_relationships",
        "taxonomy_exception": {
            "type": TAXONOMY_EXCEPTION,
            "status": TAXONOMY_EXCEPTION_STATUS,
            "count": exceptions,
            "is_a_fifth_frozen_category": False,
            "reading": ASSOCIATION_DISAGREEMENT,
            "detail": (
                "Both rules associated, to different people. None of the four frozen "
                "categories is true of that, so it is recorded as an exception to their "
                "coverage rather than as a fifth peer category, and it is excluded from the "
                "classified denominator. It is a rule disagreement, not an error: there is no "
                "person-PPE association ground truth to be wrong against."
            ),
        },
        "counts_partition": classified + exceptions == total,
    }


def decompose_all_class_delta(
    per_class: Mapping[str, Mapping[str, Any]], *, rare_class: str = RARE_CLASS
) -> dict[str, Any]:
    """Decompose the all-class box delta into the per-class movements behind it.

    ``COCOeval``'s all-class AP is the unweighted mean of the per-class APs, so
    each class contributes its own delta divided by the class count. That makes
    the decomposition exact rather than indicative, and it is checked here: if
    the contributions do not sum to the reported delta, the assumption is wrong
    and the caller is told rather than shown a plausible number.

    The rare class is separated out because a class the project has already
    classified ``DESCRIPTIVE_HIGH_UNCERTAINTY`` can move an unweighted mean
    without that movement meaning anything.

    Args:
        per_class: The per-class comparison, keyed by class name.
        rare_class: The class whose support is too small to carry weight.

    Returns:
        Each class's contribution, and the delta with the rare class removed.
    """
    deltas = {
        name: float(row["delta_AP@0.50:0.95"])
        for name, row in per_class.items()
        if isinstance(row.get("delta_AP@0.50:0.95"), (int, float))
    }
    count = len(deltas)
    if count == 0:
        return {"contributions": {}, "class_count": 0}

    contributions = {name: round(value / count, METRIC_PRECISION) for name, value in deltas.items()}
    without_rare = {name: value for name, value in deltas.items() if name != rare_class}
    improved = sorted(name for name, value in deltas.items() if value > 0)
    declined = sorted(name for name, value in deltas.items() if value < 0)

    return {
        "class_count": count,
        "all_class_is_unweighted_mean_of_per_class": True,
        "contributions": contributions,
        "total_contribution": round(sum(contributions.values()), METRIC_PRECISION),
        "improved_classes": improved,
        "declined_classes": declined,
        "rare_class": rare_class,
        "rare_class_delta": round(deltas.get(rare_class, 0.0), METRIC_PRECISION),
        "rare_class_contribution": contributions.get(rare_class),
        "delta_excluding_rare_class": round(
            sum(without_rare.values()) / len(without_rare), METRIC_PRECISION
        )
        if without_rare
        else None,
        "rare_class_contribution_exceeds_total": (
            abs(contributions.get(rare_class, 0.0)) > abs(sum(contributions.values()))
        ),
        "note": (
            "The all-class figure is an unweighted mean, so a single class can move it. The "
            "rare class is separated out because its validation support is one image; a "
            "movement there is not evidence about the models. Why any individual class moved "
            "is UNKNOWN - this phase ran no experiment isolating a cause."
        ),
    }


def tally_from_counts(counts: Mapping[str, int]) -> dict[str, Any]:
    """Re-aggregate a tally from already-recorded per-outcome counts.

    Exists so a recorded result can be restructured without re-running the
    models that produced it. Counts are the whole input, the arithmetic is
    identical to :func:`tally`, and a test asserts the two agree - so this is a
    change of presentation, never of measurement.

    Args:
        counts: Outcome counts, which may include the taxonomy exception.

    Returns:
        The same shape :func:`tally` produces.
    """
    categories: list[str] = []
    for name in (*ASSOCIATION_CATEGORIES, TAXONOMY_EXCEPTION):
        categories.extend([name] * int(counts.get(name, 0)))
    return tally(categories)


def supported_class_sensitivity(
    per_class: Mapping[str, Mapping[str, Any]], supported: Sequence[str]
) -> dict[str, Any]:
    """Recompute the localisation comparison over adequately supported classes.

    A descriptive sensitivity check, not a metric. It exists because the
    all-class figure is an unweighted mean, so a class the project has already
    classified as too sparsely supported to trust can move it. Applying the
    project's *pre-existing* support rule shows what the comparison looks like
    without that class.

    It selects nothing. It is not a phase 10A frozen metric, not a selection
    rule and not a significance test, and it changes no frozen number.

    Args:
        per_class: The per-class canonical comparison.
        supported: The classes the pre-existing support rule admits.

    Returns:
        Each model's macro over the supported classes, and their delta.
    """
    admitted = [name for name in sorted(supported) if name in per_class]
    if not admitted:
        return {"admitted_classes": [], "value": None}

    def macro(key: str) -> float:
        return sum(float(per_class[name][key]) for name in admitted) / len(admitted)

    detector = macro("D2_AP@0.50:0.95")
    segmenter = macro("S1_AP@0.50:0.95")
    return {
        "label": "POST_HOC_DESCRIPTIVE_SUPPORT_SENSITIVITY",
        "admitted_classes": admitted,
        "excluded_classes": sorted(set(per_class) - set(admitted)),
        "support_rule_origin": "PREEXISTING_PHASE_7A_SUPPORT_RULE_REUSED_UNCHANGED",
        "D2_supported_macro": round(detector, METRIC_PRECISION),
        "S1_supported_macro": round(segmenter, METRIC_PRECISION),
        "delta": round(segmenter - detector, METRIC_PRECISION),
        "is_a_frozen_phase_10a_metric": False,
        "is_a_selection_rule": False,
        "is_a_significance_test": False,
        "changes_any_frozen_number": False,
        "why": (
            "The all-class canonical figure is an unweighted mean over five classes, so a "
            "class with one validation image can move it more than the aggregate itself. This "
            "shows the same comparison over the classes the project's pre-existing support "
            "rule already admits. It is descriptive and selects nothing."
        ),
    }


# --- fingerprints -------------------------------------------------------------------------


def result_fingerprint(values: Mapping[str, Any]) -> str:
    """Hash a result's semantic content.

    Covers model identity, protocol, membership and the computed values, and
    excludes anything machine-specific, so the same comparison on another
    machine fingerprints the same.

    Args:
        values: The identity and result fields.

    Returns:
        A SHA-256 hex digest.
    """
    text = json.dumps(
        json.loads(json.dumps(values, sort_keys=True, default=str)),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --- validators --------------------------------------------------------------------------


def validate_box_comparison(
    payload: Mapping[str, Any],
    *,
    protocol_fingerprint: str,
    detector_sha256: str,
    segmenter_sha256: str,
    membership_sha256: str,
) -> list[str]:
    """Check the committed box-comparison artifact.

    Args:
        payload: The parsed artifact.
        protocol_fingerprint: The frozen phase 10A protocol fingerprint.
        detector_sha256: The frozen detector's checkpoint digest.
        segmenter_sha256: The frozen segmenter's checkpoint digest.
        membership_sha256: The validation membership fingerprint.

    Returns:
        One description per problem found.
    """
    problems: list[str] = []
    if payload.get("protocol_fingerprint") != protocol_fingerprint:
        problems.append("the protocol fingerprint is not the frozen phase 10A one")
    if payload.get("detector", {}).get("checkpoint_sha256") != detector_sha256:
        problems.append("the detector checkpoint is not the frozen one")
    if payload.get("segmenter", {}).get("checkpoint_sha256") != segmenter_sha256:
        problems.append("the segmenter checkpoint is not the frozen one")
    if payload.get("population", {}).get("membership_sha256") != membership_sha256:
        problems.append("the validation membership does not match")

    inference = payload.get("inference", {})
    if float(inference.get("conf", -1)) != 0.001:
        problems.append("the AP inference confidence must be 0.001")
    if float(inference.get("iou", -1)) != 0.70:
        problems.append("the NMS IoU must be 0.70")
    if int(inference.get("imgsz", 0)) != 768:
        problems.append("imgsz must be 768")
    if inference.get("precision") != "FP32":
        problems.append("both models must run in FP32")

    evaluator = payload.get("cocoeval", {})
    if evaluator.get("iou_type") != "bbox":
        problems.append("the recognition axis is a box comparison")

    if payload.get("segmenter_boxes_derived_from_masks") is not False:
        problems.append("the segmenter's boxes must be its own predictions")
    if payload.get("winner_declared") is not False:
        problems.append("no winner may be declared")

    sensitivity = payload.get("supported_class_sensitivity")
    if sensitivity is not None:
        if sensitivity.get("label") != "POST_HOC_DESCRIPTIVE_SUPPORT_SENSITIVITY":
            problems.append("the support sensitivity must carry its post-hoc descriptive label")
        for flag in (
            "is_a_frozen_phase_10a_metric",
            "is_a_selection_rule",
            "is_a_significance_test",
            "changes_any_frozen_number",
        ):
            if sensitivity.get(flag) is not False:
                problems.append(f"supported_class_sensitivity.{flag} must be false")
        if RARE_CLASS in sensitivity.get("admitted_classes", []):
            problems.append(
                f"{RARE_CLASS} must be excluded from the support sensitivity: it is the class "
                "the sensitivity exists to set aside"
            )
    if payload.get("aggregate_score") is not False:
        problems.append("no aggregate score may be computed")

    metrics = payload.get("metrics", {})
    for name in ("CANONICAL_BOX_MAP50_95", "CANONICAL_BOX_MAP50"):
        block = metrics.get(name, {})
        for model in ("D2", "S1"):
            if not isinstance(block.get(model), (int, float)):
                problems.append(f"{name} is missing a value for {model}")
        if isinstance(block.get("D2"), (int, float)) and isinstance(block.get("S1"), (int, float)):
            expected = round(block["S1"] - block["D2"], METRIC_PRECISION)
            if round(float(block.get("delta", -99)), METRIC_PRECISION) != expected:
                problems.append(f"{name} delta does not recompute")

    problems.extend(_holdout_problems(payload, label="the box comparison"))
    if payload.get("latency_measured", False):
        problems.append("phase 10B measures no latency")
    return problems


def validate_spatial_comparison(
    payload: Mapping[str, Any],
    *,
    protocol_fingerprint: str,
    membership_sha256: str,
) -> list[str]:
    """Check the committed spatial-comparison artifact.

    Args:
        payload: The parsed artifact.
        protocol_fingerprint: The frozen phase 10A protocol fingerprint.
        membership_sha256: The validation membership fingerprint.

    Returns:
        One description per problem found.
    """
    problems: list[str] = []
    if payload.get("protocol_fingerprint") != protocol_fingerprint:
        problems.append("the protocol fingerprint is not the frozen phase 10A one")
    if payload.get("population", {}).get("membership_sha256") != membership_sha256:
        problems.append("the validation membership does not match")

    inference = payload.get("inference", {})
    if float(inference.get("conf", -1)) != 0.25:
        problems.append("the operational confidence must be 0.25")
    if inference.get("precision") != "FP32":
        problems.append("both models must run in FP32")

    features = payload.get("spatial_features", {})
    if set(features) != set(SPATIAL_FEATURES):
        problems.append("exactly the seven frozen spatial features must be reported")
    for name, block in features.items():
        if block.get("semantics") != FEATURE_SEMANTICS.get(name):
            problems.append(f"{name} carries the wrong semantic class")
    if features.get("VISIBLE_PPE_COVERAGE_PROXY", {}).get("semantics") != INTERPRETIVE_PROXY:
        problems.append("VISIBLE_PPE_COVERAGE_PROXY must remain an interpretive proxy")

    proxies = payload.get("box_proxies", {})
    for name in ("MASK_TO_BOX_FILL_RATIO", "SHAPE_EXTENT"):
        if proxies.get(name, {}).get("proxy") != NO_BOX_PROXY:
            problems.append(f"{name} must keep {NO_BOX_PROXY}")
        if proxies.get(name, {}).get("statistics") is not None:
            problems.append(f"{name} has no box proxy, so none may be computed for it")

    association = payload.get("association", {})
    if float(association.get("containment_floor", -1)) != 0.50:
        problems.append("the containment floor must be the frozen 0.50")
    if association.get("deterministic") is not True:
        problems.append("the association rule must be deterministic")

    # The frozen four may not gain a member, lose one, or be renamed.
    if list(association.get("frozen_categories", [])) != list(ASSOCIATION_CATEGORIES):
        problems.append("the frozen four association categories were changed")
    if association.get("fifth_peer_category_added") is not False:
        problems.append("no fifth peer category may be added to the frozen taxonomy")
    if association.get("taxonomy_exception_type") != TAXONOMY_EXCEPTION:
        problems.append(f"the taxonomy exception must be {TAXONOMY_EXCEPTION!r}")
    if association.get("taxonomy_exception_reading") != ASSOCIATION_DISAGREEMENT:
        problems.append(
            f"a different-person outcome must be read as {ASSOCIATION_DISAGREEMENT!r}, "
            "never as an error: there is no association ground truth"
        )
    if association.get("association_taxonomy_status") not in (
        TAXONOMY_NON_EXHAUSTIVE,
        TAXONOMY_EXHAUSTIVE,
    ):
        problems.append("the association taxonomy status is not one of the two recorded values")

    for source in ("geometry_isolating", "pipeline_level"):
        block = association.get(source, {})
        if not block:
            continue
        if set(block.get("frozen_category_counts", {})) != set(ASSOCIATION_CATEGORIES):
            problems.append(f"{source} does not count exactly the frozen four categories")
        if TAXONOMY_EXCEPTION in block.get("frozen_category_counts", {}):
            problems.append(f"{source} counts the exception among the frozen categories")
        classified = block.get("classified_relationships")
        exceptions = block.get("taxonomy_exceptions")
        total = block.get("total_relationships")
        if not isinstance(classified, int) or not isinstance(exceptions, int):
            problems.append(f"{source} is missing its coverage counts")
            continue
        if classified + exceptions != total:
            problems.append(f"{source}: classified plus exceptions does not equal the total")
        if sum(block["frozen_category_counts"].values()) != classified:
            problems.append(f"{source}: the frozen counts do not sum to the classified total")
        if block.get("counts_partition") is not True:
            problems.append(f"{source}: the outcomes do not partition the relationships")
        if block.get("percentage_denominator") != "classified_relationships":
            problems.append(f"{source}: the percentage denominator must be stated and explicit")
        expected = round(classified / total, 6) if total else None
        if block.get("taxonomy_coverage") != expected:
            problems.append(f"{source}: taxonomy coverage does not recompute")

    if payload.get("compliance_accuracy_claimed", True):
        problems.append("no compliance accuracy may be claimed")
    if payload.get("post_hoc_bins_created", True):
        problems.append("no qualitative bins may be created after seeing results")
    if payload.get("latency_measured", False):
        problems.append("phase 10B measures no latency")

    rare = payload.get("rare_class", {})
    if rare.get("status") != RARE_CLASS_STATUS:
        problems.append(f"{RARE_CLASS} must keep {RARE_CLASS_STATUS}")

    problems.extend(_holdout_problems(payload, label="the spatial comparison"))
    return problems


def _holdout_problems(payload: Mapping[str, Any], *, label: str) -> list[str]:
    """Check an artifact declares and honours the holdout policy.

    Args:
        payload: The artifact.
        label: Its name, for the message.

    Returns:
        One description per problem.
    """
    problems: list[str] = []
    if payload.get("test", {}).get("status") != "PROTECTED_NOT_ACCESSED":
        problems.append(f"{label} does not declare the holdout protected")
    if payload.get("holdout_accessed", True):
        problems.append(f"{label} records holdout access")
    if payload.get("population", {}).get("split") != "validation":
        problems.append(f"{label} must run on the validation split")
    return problems
