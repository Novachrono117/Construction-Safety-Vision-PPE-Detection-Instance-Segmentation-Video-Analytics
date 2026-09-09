"""Error attribution for the frozen S0 segmentation baseline.

Phase 8D. It explains nothing by itself: it turns one already-computed
diagnostic into per-instance evidence, so that a later reviewed decision about a
second experiment rests on something other than a hunch about why the mask
metric trails the box metric.

Nothing here trains, re-validates, or invents a metric. The inference settings
and the matching rule are the ones phase 8C froze; only the *unit of reporting*
changes, from a population aggregate to one row per canonical instance.

Three things are deliberate.

**The taxonomy is frozen before any image is opened.** An error category invented
while looking at failures describes the failures that were looked at. The bands
below are fixed thresholds over quantities the diagnostic already produces, and
the mechanism flags an automated pass may assign are separated from the ones only
a person can.

**Automated flags are candidates, not findings.** ``UNDERSEGMENTATION`` from an
area ratio is a shape statistic; whether the model actually missed a visible
region is a judgement. So the two are recorded in different fields and never
merged, and an instance nothing explains stays ``UNATTRIBUTED`` rather than being
given the most plausible-sounding label.

**The review set is chosen from the table, not from the images.** Deterministic,
recorded before inspection, and never swapped for a more interesting picture -
which is the difference between error analysis and illustration.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

# --- frozen thresholds --------------------------------------------------------

IOU_LOW = 0.50
"""Below this, a matched mask is classified ``LOW_OVERLAP_MASK``."""

IOU_HIGH = 0.75
"""At or above this, a matched mask is classified ``HIGH_QUALITY_MASK``."""

TINY_MASK_AREA_PX = 1024
"""Canonical area below which an instance joins the tiny-mask stratum.

Not a new number: it is the ``small_mask_area_px`` phase 8A already declared and
used to justify its own ``SMALL_MASK_SENSITIVITY`` flag. Reused so the two phases
partition the same instances the same way.
"""

ADAPTER_FIDELITY_RISK_IOU = 0.90
"""Adapter round-trip IoU below which an instance is a fidelity-risk candidate.

The band phase 8A published (47 development instances fell below it), reused
rather than redrawn so the stratum means the same thing in both phases.
"""

AREA_RATIO_TOLERANCE = 0.25
"""Predicted-versus-canonical area difference that separates support from contour.

Above it in either direction the predicted mask covers materially more or less
of the object than the canonical one, which is a support error. Within it, a
matched mask that still scores poorly is disagreeing about the boundary rather
than about what the object is.
"""

# --- band taxonomy ------------------------------------------------------------

DETECTION_MISS = "DETECTION_MISS"
LOW_OVERLAP_MASK = "LOW_OVERLAP_MASK"
MODERATE_MASK = "MODERATE_MASK"
HIGH_QUALITY_MASK = "HIGH_QUALITY_MASK"

OUTCOME_BANDS: tuple[str, ...] = (
    DETECTION_MISS,
    LOW_OVERLAP_MASK,
    MODERATE_MASK,
    HIGH_QUALITY_MASK,
)
"""Every canonical instance falls in exactly one of these, and they partition."""

# --- mechanism flags ----------------------------------------------------------

UNDERSEGMENTATION = "UNDERSEGMENTATION"
OVERSEGMENTATION = "OVERSEGMENTATION"
BOUNDARY_ERROR = "BOUNDARY_ERROR"
INSTANCE_SEPARATION_ERROR = "INSTANCE_SEPARATION_ERROR"
OCCLUSION_ASSOCIATED = "OCCLUSION_ASSOCIATED"
TINY_MASK_ASSOCIATED = "TINY_MASK_ASSOCIATED"
ADAPTER_FIDELITY_RISK = "ADAPTER_FIDELITY_RISK"
UNATTRIBUTED = "UNATTRIBUTED"

MECHANISM_FLAGS: tuple[str, ...] = (
    UNDERSEGMENTATION,
    OVERSEGMENTATION,
    BOUNDARY_ERROR,
    INSTANCE_SEPARATION_ERROR,
    OCCLUSION_ASSOCIATED,
    TINY_MASK_ASSOCIATED,
    ADAPTER_FIDELITY_RISK,
    UNATTRIBUTED,
)
"""Mechanisms an instance may carry. Several may coexist."""

AUTOMATIC_FLAGS: tuple[str, ...] = (
    UNDERSEGMENTATION,
    OVERSEGMENTATION,
    BOUNDARY_ERROR,
    TINY_MASK_ASSOCIATED,
    ADAPTER_FIDELITY_RISK,
    UNATTRIBUTED,
)
"""The subset an automated pass may assign from measured quantities alone.

``INSTANCE_SEPARATION_ERROR`` and ``OCCLUSION_ASSOCIATED`` are absent on purpose:
neither is decidable from an IoU and an area ratio. Assigning them
automatically would manufacture a mechanism census out of arithmetic.
"""

MANUAL_ONLY_FLAGS: tuple[str, ...] = (INSTANCE_SEPARATION_ERROR, OCCLUSION_ASSOCIATED)
"""Mechanisms that require a person to have looked at the image."""

FORBIDDEN_FLAGS: tuple[str, ...] = (
    "MODEL_CAPACITY_LIMIT",
    "INSUFFICIENT_TRAINING",
    "NEEDS_MORE_DATA",
    "ARCHITECTURE_LIMIT",
)
"""Labels this phase cannot establish and therefore may not record.

Each names a cause that only a controlled experiment could demonstrate. Writing
one down after looking at a handful of images would turn a guess into a finding.
"""


class ErrorAnalysisError(RuntimeError):
    """Raised when the error analysis cannot proceed as specified."""


# --- one canonical instance ---------------------------------------------------


@dataclass(frozen=True)
class InstanceRow:
    """One canonical validation instance and how S0 handled it.

    Attributes:
        image_id: Source image stem.
        annotation_id: Canonical annotation id.
        class_name: Frozen class name.
        canonical_area_px: Canonical mask area.
        adapter_mask_iou: The phase 8A round-trip IoU, when the instance appears
            in that table.
        adapter_geometry_type: Its canonical representation in phase 8A.
        connected_components: Component count measured in phase 8A.
        hole_count: Interior holes measured in phase 8A.
        matched: Whether an overlapping same-class prediction was assigned.
        mask_iou: The assigned pair's IoU, or ``0.0`` when unmatched.
        confidence: The matched prediction's confidence, when matched.
        predicted_area_px: The matched prediction's area, when matched.
        false_positive_px: Predicted pixels outside the canonical mask.
        false_negative_px: Canonical pixels the prediction missed.
        band: The outcome band.
        automatic_flags: Mechanism candidates derived from measured quantities.
        contested_fraction: Share of this instance's canonical pixels that also
            lie inside another class's canonical mask in the same image.
        overlap_resolved_iou: IoU of the same prediction against the
            overlap-resolved target the framework actually trained on, where
            contested pixels belong to the smaller instance.
        false_negative_contested_share: Share of the missed canonical pixels that
            lie in the contested region.
    """

    image_id: str
    annotation_id: int
    class_name: str
    canonical_area_px: int
    adapter_mask_iou: float | None
    adapter_geometry_type: str | None
    connected_components: int | None
    hole_count: int | None
    matched: bool
    mask_iou: float
    confidence: float | None
    predicted_area_px: int | None
    false_positive_px: int | None
    false_negative_px: int | None
    band: str
    automatic_flags: tuple[str, ...]
    contested_fraction: float
    overlap_resolved_iou: float
    false_negative_contested_share: float | None

    def as_dict(self) -> dict[str, Any]:
        """Serialise for the row-level table.

        Returns:
            A JSON- and CSV-friendly mapping.
        """
        return {
            "image_id": self.image_id,
            "annotation_id": self.annotation_id,
            "class": self.class_name,
            "canonical_area_px": self.canonical_area_px,
            "adapter_mask_iou": self.adapter_mask_iou,
            "adapter_geometry_type": self.adapter_geometry_type,
            "connected_components": self.connected_components,
            "hole_count": self.hole_count,
            "matched": self.matched,
            "mask_iou": round(self.mask_iou, 6),
            "confidence": None if self.confidence is None else round(self.confidence, 6),
            "predicted_area_px": self.predicted_area_px,
            "false_positive_px": self.false_positive_px,
            "false_negative_px": self.false_negative_px,
            "relative_area_error": self.relative_area_error(),
            "gt_normalized_contribution": round(self.mask_iou, 6),
            "band": self.band,
            "automatic_flags": ";".join(self.automatic_flags),
            "contested_fraction": round(self.contested_fraction, 6),
            "overlap_resolved_iou": round(self.overlap_resolved_iou, 6),
            "false_negative_contested_share": (
                None
                if self.false_negative_contested_share is None
                else round(self.false_negative_contested_share, 6)
            ),
        }

    def relative_area_error(self) -> float | None:
        """Signed predicted-versus-canonical area difference.

        Returns:
            ``(predicted - canonical) / canonical``, or ``None`` when unmatched.
        """
        if not self.matched or self.predicted_area_px is None or not self.canonical_area_px:
            return None
        return round((self.predicted_area_px - self.canonical_area_px) / self.canonical_area_px, 6)


def outcome_band(*, matched: bool, mask_iou: float) -> str:
    """Classify one canonical instance's outcome.

    Args:
        matched: Whether an overlapping same-class prediction was assigned.
        mask_iou: The assigned pair's IoU.

    Returns:
        The band. The four bands partition every canonical instance, so the
        counts must sum to the ground-truth total - which is what makes the
        census checkable rather than merely plausible.
    """
    if not matched or mask_iou <= 0.0:
        return DETECTION_MISS
    if mask_iou < IOU_LOW:
        return LOW_OVERLAP_MASK
    if mask_iou < IOU_HIGH:
        return MODERATE_MASK
    return HIGH_QUALITY_MASK


def automatic_flags(
    *,
    band: str,
    canonical_area_px: int,
    adapter_mask_iou: float | None,
    relative_area_error: float | None,
) -> tuple[str, ...]:
    """Assign the mechanism candidates that measured quantities alone support.

    Deliberately conservative. A shape statistic can say a predicted mask covers
    a quarter less area than the canonical one; it cannot say the model failed to
    see a visible region rather than the annotation being generous. So these are
    candidates for a person to confirm, and an instance nothing explains is
    ``UNATTRIBUTED`` rather than assigned the nearest label.

    Args:
        band: The outcome band.
        canonical_area_px: Canonical mask area.
        adapter_mask_iou: The phase 8A round-trip IoU, when known.
        relative_area_error: Signed predicted-versus-canonical area difference.

    Returns:
        The flags, in the frozen order.
    """
    flags: list[str] = []
    if canonical_area_px < TINY_MASK_AREA_PX:
        flags.append(TINY_MASK_ASSOCIATED)
    if adapter_mask_iou is not None and adapter_mask_iou < ADAPTER_FIDELITY_RISK_IOU:
        flags.append(ADAPTER_FIDELITY_RISK)

    if band in (LOW_OVERLAP_MASK, MODERATE_MASK) and relative_area_error is not None:
        if relative_area_error < -AREA_RATIO_TOLERANCE:
            flags.append(UNDERSEGMENTATION)
        elif relative_area_error > AREA_RATIO_TOLERANCE:
            flags.append(OVERSEGMENTATION)
        else:
            flags.append(BOUNDARY_ERROR)

    if not flags:
        flags.append(UNATTRIBUTED)
    return tuple(name for name in MECHANISM_FLAGS if name in flags)


# --- aggregation --------------------------------------------------------------


@dataclass
class GroupSummary:
    """Outcome counts for one class, stratum or the whole split.

    Attributes:
        instances: Canonical instances in the group.
        bands: Count per outcome band.
        iou_sum: Summed IoU over matched instances.
        matched: Matched instance count.
    """

    instances: int = 0
    bands: dict[str, int] = field(default_factory=lambda: dict.fromkeys(OUTCOME_BANDS, 0))
    iou_sum: float = 0.0
    matched: int = 0

    def add(self, row: InstanceRow) -> None:
        """Fold one instance into the group.

        Args:
            row: The instance.
        """
        self.instances += 1
        self.bands[row.band] += 1
        if row.matched:
            self.matched += 1
            self.iou_sum += row.mask_iou

    def as_dict(self) -> dict[str, Any]:
        """Render the group's diagnostics.

        Returns:
            Counts and the two IoU summaries, kept apart for the same reason
            phase 8C kept them apart: one describes mask quality where the model
            found something, the other spreads the same sum over every instance.
        """
        return {
            "instances": self.instances,
            "matched": self.matched,
            "misses": self.instances - self.matched,
            "bands": dict(self.bands),
            "matched_mask_iou_mean": (
                round(self.iou_sum / self.matched, 6) if self.matched else None
            ),
            "gt_normalized_mask_iou": (
                round(self.iou_sum / self.instances, 6) if self.instances else None
            ),
            "gt_match_coverage": (
                round(self.matched / self.instances, 6) if self.instances else None
            ),
        }


def summarise(rows: Iterable[InstanceRow], key: Any) -> dict[str, dict[str, Any]]:
    """Group instances by a key and summarise each group.

    Args:
        rows: The instance rows.
        key: A callable returning the group name for a row, or ``None`` to skip.

    Returns:
        Group diagnostics keyed by group name, in sorted order.
    """
    groups: dict[str, GroupSummary] = {}
    for row in rows:
        name = key(row)
        if name is None:
            continue
        groups.setdefault(str(name), GroupSummary()).add(row)
    return {name: groups[name].as_dict() for name in sorted(groups)}


def area_quartile_edges(rows: Sequence[InstanceRow]) -> list[float]:
    """Compute canonical-area quartile boundaries for this validation split.

    Args:
        rows: The instance rows.

    Returns:
        The three interior boundaries.
    """
    import numpy as np

    areas = np.array([row.canonical_area_px for row in rows], dtype=np.float64)
    return [round(float(value), 3) for value in np.percentile(areas, [25, 50, 75])]


def area_quartile(row: InstanceRow, edges: Sequence[float]) -> str:
    """Name the area quartile an instance falls in.

    Args:
        row: The instance.
        edges: The three interior boundaries.

    Returns:
        ``Q1`` to ``Q4``.
    """
    area = float(row.canonical_area_px)
    if area <= edges[0]:
        return "Q1_smallest"
    if area <= edges[1]:
        return "Q2"
    if area <= edges[2]:
        return "Q3"
    return "Q4_largest"


def fidelity_band(row: InstanceRow) -> str | None:
    """Name the phase 8A adapter-fidelity band an instance falls in.

    Args:
        row: The instance.

    Returns:
        The band, or ``None`` when the instance has no recorded fidelity.
    """
    if row.adapter_mask_iou is None:
        return None
    if row.adapter_mask_iou < ADAPTER_FIDELITY_RISK_IOU:
        return "adapter_iou_lt_0.90"
    if row.adapter_mask_iou < 0.95:
        return "adapter_iou_0.90_to_0.95"
    if row.adapter_mask_iou < 0.99:
        return "adapter_iou_0.95_to_0.99"
    return "adapter_iou_ge_0.99"


# --- deterministic review selection -------------------------------------------


def _sort_key(row: InstanceRow) -> tuple[Any, ...]:
    """Total order over instances, so selection cannot depend on input order.

    Args:
        row: The instance.

    Returns:
        A tuple ordering by IoU, then by identity to break every tie.
    """
    return (row.mask_iou, row.image_id, row.annotation_id)


def select_review_set(
    rows: Sequence[InstanceRow],
    *,
    focus_class: str,
    supported_classes: Sequence[str],
    worst_focus: int = 10,
    focus_misses: int = 10,
    best_focus: int = 5,
    worst_other: int = 5,
    fidelity_risk: int = 5,
) -> dict[str, list[dict[str, Any]]]:
    """Choose the instances to look at, from the table, before opening an image.

    Every selection is a deterministic slice of a totally ordered list, so the
    set is reproducible and cannot drift toward whichever example turned out to
    be the most photogenic.

    Args:
        rows: The instance rows.
        focus_class: The predeclared focus class.
        supported_classes: The classes the frozen support rule admits.
        worst_focus: How many worst matched focus-class instances to take.
        focus_misses: How many focus-class misses to take.
        best_focus: How many best focus-class instances to take.
        worst_other: How many worst instances per other supported class.
        fidelity_risk: How many adapter-fidelity-risk instances to take.

    Returns:
        The review set, keyed by selection rule, each entry naming its instance.
    """

    def identify(row: InstanceRow, rule: str) -> dict[str, Any]:
        return {
            "rule": rule,
            "image_id": row.image_id,
            "annotation_id": row.annotation_id,
            "class": row.class_name,
            "band": row.band,
            "mask_iou": round(row.mask_iou, 6),
            "canonical_area_px": row.canonical_area_px,
            "adapter_mask_iou": row.adapter_mask_iou,
        }

    focus = [row for row in rows if row.class_name == focus_class]
    focus_matched = sorted((row for row in focus if row.matched), key=_sort_key)
    focus_missed = sorted((row for row in focus if not row.matched), key=_sort_key)

    selection: dict[str, list[dict[str, Any]]] = {
        f"worst_matched_{focus_class}": [
            identify(row, f"worst_matched_{focus_class}") for row in focus_matched[:worst_focus]
        ],
        f"misses_{focus_class}": [
            identify(row, f"misses_{focus_class}") for row in focus_missed[:focus_misses]
        ],
        f"best_matched_{focus_class}": [
            identify(row, f"best_matched_{focus_class}")
            for row in list(reversed(focus_matched))[:best_focus]
        ],
    }

    for name in sorted(set(supported_classes) - {focus_class}):
        others = sorted((row for row in rows if row.class_name == name), key=_sort_key)
        selection[f"worst_{name}"] = [
            identify(row, f"worst_{name}") for row in others[:worst_other]
        ]

    risky = sorted(
        (row for row in rows if ADAPTER_FIDELITY_RISK in row.automatic_flags), key=_sort_key
    )
    selection["adapter_fidelity_risk"] = [
        identify(row, "adapter_fidelity_risk") for row in risky[:fidelity_risk]
    ]
    return selection


def review_identifiers(selection: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[str]:
    """List every instance in the review set, deduplicated and ordered.

    A single instance can satisfy two rules; it is inspected once.

    Args:
        selection: The review set.

    Returns:
        ``"<image_id>#<annotation_id>"`` keys, sorted.
    """
    keys = {
        f"{entry['image_id']}#{entry['annotation_id']}"
        for entries in selection.values()
        for entry in entries
    }
    return sorted(keys)


def validate_manual_attributions(
    attributions: Mapping[str, Mapping[str, Any]],
    selection: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    """Check declared human judgements against the review set they claim to describe.

    Args:
        attributions: Declared judgements keyed by ``"<image_id>#<annotation_id>"``.
        selection: The deterministic review set.

    Raises:
        ErrorAnalysisError: If a judgement names an instance outside the review
            set, uses a flag outside the taxonomy, uses a flag this phase cannot
            establish, or carries no evidence note.
    """
    allowed = set(review_identifiers(selection))
    for key, entry in sorted(attributions.items()):
        if key not in allowed:
            msg = (
                f"manual attribution {key} is not in the deterministic review set. Judgements "
                "may only describe instances the selection rules chose before any image was "
                "opened."
            )
            raise ErrorAnalysisError(msg)
        flags = tuple(entry.get("flags", ()))
        if not flags:
            msg = f"manual attribution {key} carries no flag"
            raise ErrorAnalysisError(msg)
        for flag in flags:
            if flag in FORBIDDEN_FLAGS:
                msg = (
                    f"manual attribution {key} uses {flag!r}, which names a cause only a "
                    "controlled experiment could establish. This phase runs none."
                )
                raise ErrorAnalysisError(msg)
            if flag not in MECHANISM_FLAGS:
                msg = f"manual attribution {key} uses {flag!r}, which is outside the taxonomy"
                raise ErrorAnalysisError(msg)
        if UNATTRIBUTED in flags and len(flags) > 1:
            msg = f"manual attribution {key} is both UNATTRIBUTED and something else"
            raise ErrorAnalysisError(msg)
        if not str(entry.get("evidence", "")).strip():
            msg = f"manual attribution {key} records no evidence note"
            raise ErrorAnalysisError(msg)


def flag_census(
    attributions: Mapping[str, Mapping[str, Any]],
) -> dict[str, int]:
    """Count how often each mechanism was assigned by human review.

    Args:
        attributions: Declared judgements.

    Returns:
        Counts per flag, in the frozen order, including zeros.
    """
    counts = dict.fromkeys(MECHANISM_FLAGS, 0)
    for entry in attributions.values():
        for flag in entry.get("flags", ()):
            counts[flag] += 1
    return counts


def overlap_resolved_targets(masks: Sequence[Any]) -> dict[int, Any]:
    """Reproduce the training target the framework builds when ``overlap_mask`` is on.

    Read from the installed ``polygons2masks_overlap``: instances are sorted by
    **descending** area and painted with a running maximum, so the *later* -
    smaller - instance owns any pixel two instances share. A vest therefore owns
    the pixels of the person wearing it, and that person's target is the
    remainder.

    This matters because the canonical COCO masks the direct IoU diagnostic
    scores against do **not** resolve overlap: they give every instance its full
    extent. Scoring a model against a target it was never trained on charges it
    for a difference the protocol created, so both are computed and reported
    side by side rather than one being presented as the truth.

    Args:
        masks: The image's canonical instance masks, in annotation order.

    Returns:
        The overlap-resolved mask for each index.
    """
    import numpy as np

    if not masks:
        return {}
    order = sorted(range(len(masks)), key=lambda index: -int(np.count_nonzero(masks[index])))
    owner = np.zeros(np.asarray(masks[0]).shape, dtype=np.int32)
    for rank, index in enumerate(order, start=1):
        owner = np.maximum(owner, np.asarray(masks[index]).astype(np.int32) * rank)
    return {index: (owner == rank) for rank, index in enumerate(order, start=1)}
