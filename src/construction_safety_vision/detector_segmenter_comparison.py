"""The frozen detector-versus-segmenter comparison protocol.

Phase 10A. Written before any comparison runs, which is the only thing that
makes it a protocol. It defines what will be measured, on what, at which
settings, and - just as importantly - what the measurements will not be allowed
to claim.

The scientific question is not "which model is better". D2 emits a class, a
confidence and a box; S1 emits those plus an instance mask. They do not solve
the same output task, so a single ranking would be meaningless. The question is
**what the mask adds and what it costs**, along four axes: recognition,
spatial information, computational cost, and operational person-PPE reasoning.

Five decisions are fixed here rather than once results exist.

**Two confidence thresholds, never mixed.** Average precision integrates over
the score curve and needs the low-scoring tail, so the AP protocol uses 0.001.
Operational analysis needs a working point, so it uses 0.25. Each belongs to
one protocol; a value from one may never be reported under the other.

**S1's boxes are S1's own.** The segmenter predicts boxes directly, and those
are what the comparison uses. Deriving boxes from S1's masks after inference
would make the box comparison measure a post-processing choice this project
invented, not the model.

**Mask measurements are paired with box proxies, or declared to have none.**
That pairing is the whole answer to "what does segmentation add": a quantity
with a good box proxy adds little, and one with no box equivalent is the actual
gain. Deciding which is which after seeing the numbers would be circular.

**Latency is measured under one symmetric design.** Same machine, same
precision, same warmup, same iteration count, same images in the same order,
same synchronisation, and an interleaved execution order in both directions -
because benchmarking one model to completion and then the other measures the
laptop's thermal state as much as the models.

**No aggregate score.** Benefit and cost are reported separately. A weighted
index invented once the numbers are visible would hide the trade-off this phase
exists to expose.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from construction_safety_vision.config import ConfigError, check_keys

CONFIG_SCHEMA_VERSION = 1
"""Schema version of ``configs/detector_segmenter_comparison.yaml``."""

PROTOCOL_NAME = "DETECTOR_VERSUS_SEGMENTER_CONTROLLED_COMPARISON"
"""What this protocol is, recorded in every artifact that reports it."""

STATUS_FROZEN_NOT_EXECUTED = "FROZEN_NOT_EXECUTED"
"""The protocol's state at the end of phase 10A."""

DETECTOR_EXPERIMENT = "D2"
SEGMENTER_EXPERIMENT = "S1"

DETECTOR_ROLE = "FINAL_OBJECT_DETECTOR"
SEGMENTER_ROLE = "FINAL_INSTANCE_SEGMENTER"

COMPARISON_IMGSZ = 768
"""Both frozen models take the same input size, so resolution is held constant."""

EVALUATION_SPLIT = "validation"
FORBIDDEN_SPLIT = "test"
HOLDOUT_STATUS = "PROTECTED_NOT_ACCESSED"

AP_CONFIDENCE = 0.001
"""AP integrates over the score curve and needs the low-scoring tail."""

OPERATIONAL_CONFIDENCE = 0.25
"""A working point for operational analysis. Never used for AP."""

NMS_IOU = 0.70
MAX_DET = 300

PRECISION = "FP32"
"""The one precision both models are measured in.

``half`` is deprecated in the installed ultralytics 8.4.138 in favour of
``quantize``, and leaving ``quantize`` unset delegates the choice to the
runtime - which could differ between two models and would silently make the
latency comparison a precision comparison. So it is pinned explicitly.
"""

PRECISION_ARGUMENT = "quantize"
PRECISION_VALUE = 32

IOU_TYPE = "bbox"
"""COCOeval mode for the recognition axis. Box IoU, never mask IoU."""

IOU_THRESHOLDS: tuple[float, ...] = tuple(round(0.50 + 0.05 * step, 2) for step in range(10))
MAX_DETS: tuple[int, ...] = (1, 10, 100)

CANONICAL_BOX_METRIC = "CANONICAL_BOX_MAP50_95"
CANONICAL_BOX_METRIC_50 = "CANONICAL_BOX_MAP50"

LATENCY_BATCH = 1
WARMUP_ITERATIONS = 20
TIMED_ITERATIONS_PER_IMAGE = 30
BENCHMARK_IMAGE_COUNT = 20

MODEL_INFERENCE_LATENCY = "MODEL_INFERENCE_LATENCY_MS"
END_TO_END_LATENCY = "END_TO_END_MODEL_OUTPUT_LATENCY_MS"

LATENCY_STATISTICS: tuple[str, ...] = (
    "mean",
    "median",
    "std",
    "p50",
    "p90",
    "p95",
    "p99",
    "min",
    "max",
)

BENCHMARK_LABEL = "CONTROLLED_LOCAL_HARDWARE_BENCHMARK"
"""What the latency numbers will be valid for: this machine, this runtime."""

SELECTION_RULE = "STABLE_SHA256_RANK_OF_IMAGE_ID"
"""How benchmark images are chosen: by digest, without looking at any image."""

SPATIAL_FEATURES: tuple[str, ...] = (
    "INSTANCE_AREA_PIXELS",
    "MASK_TO_BOX_FILL_RATIO",
    "MASK_CENTROID",
    "SHAPE_EXTENT",
    "PERSON_PPE_MASK_INTERSECTION",
    "PERSON_PPE_MASK_CONTAINMENT",
    "VISIBLE_PPE_COVERAGE_PROXY",
)
"""Quantities a predicted mask makes available. Not safety-compliance truth."""

BOX_PROXY_NONE = "NO_BOX_ONLY_EQUIVALENT"
"""Marker for a mask quantity a bounding box cannot approximate."""

ASSOCIATION_CATEGORIES: tuple[str, ...] = (
    "BOX_AND_MASK_AGREE",
    "BOX_ONLY_ASSOCIATION",
    "MASK_ONLY_ASSOCIATION",
    "NEITHER_ASSOCIATION",
)
"""How a person-PPE candidate pair may be classified. Descriptive, not accuracy."""

ASSOCIATION_LABEL = "SPATIAL_ASSOCIATION_ANALYSIS"
"""What the association analysis is called.

Deliberately not ``COMPLIANCE_CLASSIFICATION_ACCURACY``: the project has no
canonical compliance ground truth, so there is nothing to be accurate against.
The provider's ``helmet_on_head`` and ``vest_on_body`` already encode a
provider-level state, and inventing a compliance label on top of them would be
manufacturing ground truth.
"""

COMPOSITE_SCORE_FORBIDDEN = False
"""No weighted benefit-versus-cost index, now or later."""

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ComparisonProtocolError(ConfigError):
    """Raised when the comparison protocol is missing, malformed or unsafe."""


# --- deterministic benchmark membership -------------------------------------------


def stable_rank(image_id: str) -> str:
    """Rank key for one image, derived only from its identifier.

    Args:
        image_id: The canonical source image id.

    Returns:
        A hex digest used as the sort key.
    """
    return hashlib.sha256(image_id.encode("utf-8")).hexdigest()


def select_benchmark_images(
    image_ids: Sequence[str], count: int = BENCHMARK_IMAGE_COUNT
) -> tuple[str, ...]:
    """Choose the latency benchmark subset without looking at any image.

    Ranking by the digest of the identifier means the subset is reproducible
    from the frozen membership alone and cannot have been picked for being
    easy, crowded or visually interesting - which is exactly the bias a latency
    benchmark is prone to.

    Args:
        image_ids: The validation membership.
        count: How many images to select.

    Returns:
        The selected ids, in the frozen benchmark order.

    Raises:
        ComparisonProtocolError: If there are not enough images, or the
            membership carries a duplicate.
    """
    unique = list(dict.fromkeys(image_ids))
    if len(unique) != len(image_ids):
        msg = "the validation membership carries a duplicate image id"
        raise ComparisonProtocolError(msg)
    if count > len(unique):
        msg = f"cannot select {count} benchmark images from a membership of {len(unique)}"
        raise ComparisonProtocolError(msg)
    return tuple(sorted(unique, key=stable_rank)[:count])


def membership_fingerprint(image_ids: Sequence[str]) -> str:
    """Fingerprint a set of images as a set, not as an ordering.

    Args:
        image_ids: Image identifiers.

    Returns:
        A SHA-256 hex digest over the sorted, newline-joined ids.
    """
    text = "\n".join(sorted(image_ids)) + "\n"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ordered_fingerprint(image_ids: Sequence[str]) -> str:
    """Fingerprint a sequence of images as an ordering.

    The benchmark order is part of the protocol - two models must see the same
    images in the same sequence - so it gets a digest that changes when the
    order does, unlike :func:`membership_fingerprint`.

    Args:
        image_ids: Image identifiers, in order.

    Returns:
        A SHA-256 hex digest over the newline-joined ids as given.
    """
    text = "\n".join(image_ids) + "\n"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --- the protocol -------------------------------------------------------------------


@dataclass(frozen=True)
class ComparisonProtocol:
    """The frozen detector-versus-segmenter comparison protocol.

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
    def ap_inference(self) -> dict[str, Any]:
        """The frozen AP-curve inference settings.

        Returns:
            The inference block.
        """
        return dict(self.raw["ap_inference"])

    @property
    def operational_inference(self) -> dict[str, Any]:
        """The frozen operational inference settings.

        Returns:
            The inference block.
        """
        return dict(self.raw["operational_inference"])

    @property
    def latency(self) -> dict[str, Any]:
        """The frozen latency benchmark protocol.

        Returns:
            The latency block.
        """
        return dict(self.raw["latency_protocol"])

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
    "status",
    "scientific_question",
    "comparison_axes",
    "detector",
    "segmenter",
    "comparison_input_imgsz",
    "population",
    "ap_inference",
    "operational_inference",
    "threshold_policy",
    "canonical_box_evaluator",
    "box_comparability",
    "spatial_features",
    "spatial_feature_status",
    "box_proxies",
    "box_proxy_note",
    "association",
    "spatial_value_metrics",
    "latency_protocol",
    "memory_protocol",
    "model_complexity",
    "reporting",
    "interpretation",
    "limitations",
    "test_policy",
)

_MODEL_KEYS: tuple[str, ...] = (
    "experiment",
    "role",
    "model",
    "imgsz",
    "checkpoint_sha256",
    "manifest",
    "outputs",
)

_INFERENCE_KEYS: tuple[str, ...] = (
    "purpose",
    "imgsz",
    "conf",
    "iou",
    "max_det",
    "augment",
    "tta",
    "precision",
    "precision_argument",
    "precision_value",
)

_LATENCY_KEYS: tuple[str, ...] = (
    "label",
    "batch",
    "imgsz",
    "precision",
    "precision_argument",
    "precision_value",
    "warmup_iterations",
    "timed_iterations_per_image",
    "benchmark_image_count",
    "benchmark_selection_rule",
    "benchmark_membership_sha256",
    "benchmark_membership_artifact",
    "benchmark_selection_note",
    "execution_order",
    "synchronization",
    "timing_primitive",
    "timing_boundaries",
    "statistics",
    "derived_quantities",
    "fairness_invariants",
    "abort_on_deviation",
    "excluded_from_timed_region",
    "limitations",
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


def _check_model(block: Any, *, name: str, experiment: str, role: str) -> None:
    """Validate one frozen model's declaration.

    Args:
        block: The parsed model block.
        name: Block name, for messages.
        experiment: The experiment id it must name.
        role: The role it must declare.

    Raises:
        ComparisonProtocolError: If the block is malformed or names the wrong
            model.
    """
    if not isinstance(block, Mapping):
        msg = f"{name}: must be a mapping"
        raise ComparisonProtocolError(msg)
    try:
        check_keys(block, required=_MODEL_KEYS, context=name)
    except ConfigError as exc:
        raise ComparisonProtocolError(str(exc)) from exc
    if str(block["experiment"]) != experiment:
        msg = f"{name}.experiment must be {experiment!r}, got {block['experiment']!r}"
        raise ComparisonProtocolError(msg)
    if str(block["role"]) != role:
        msg = f"{name}.role must be {role!r}"
        raise ComparisonProtocolError(msg)
    if int(block["imgsz"]) != COMPARISON_IMGSZ:
        msg = (
            f"{name}.imgsz must be {COMPARISON_IMGSZ}: comparing the two models at different "
            "resolutions would make the result a resolution comparison"
        )
        raise ComparisonProtocolError(msg)
    if not SHA256_PATTERN.match(str(block["checkpoint_sha256"])):
        msg = f"{name}.checkpoint_sha256 is not a SHA-256 digest"
        raise ComparisonProtocolError(msg)


def _check_inference(block: Any, *, name: str, conf: float) -> None:
    """Validate one inference block against its frozen confidence.

    Args:
        block: The parsed inference block.
        name: Block name, for messages.
        conf: The confidence this protocol must declare.

    Raises:
        ComparisonProtocolError: If any frozen setting has moved.
    """
    if not isinstance(block, Mapping):
        msg = f"{name}: must be a mapping"
        raise ComparisonProtocolError(msg)
    try:
        check_keys(block, required=_INFERENCE_KEYS, context=name)
    except ConfigError as exc:
        raise ComparisonProtocolError(str(exc)) from exc
    if float(block["conf"]) != conf:
        msg = (
            f"{name}.conf must be {conf}. The AP protocol and the operational protocol keep "
            "their own thresholds and are never mixed."
        )
        raise ComparisonProtocolError(msg)
    if float(block["iou"]) != NMS_IOU:
        msg = f"{name}.iou must be {NMS_IOU}"
        raise ComparisonProtocolError(msg)
    if int(block["max_det"]) != MAX_DET:
        msg = f"{name}.max_det must be {MAX_DET}"
        raise ComparisonProtocolError(msg)
    if int(block["imgsz"]) != COMPARISON_IMGSZ:
        msg = f"{name}.imgsz must be {COMPARISON_IMGSZ}"
        raise ComparisonProtocolError(msg)
    if block["augment"] is not False or block["tta"] is not False:
        msg = f"{name}: no test-time augmentation is authorised"
        raise ComparisonProtocolError(msg)
    if str(block["precision"]) != PRECISION:
        msg = f"{name}.precision must be {PRECISION!r} for both models"
        raise ComparisonProtocolError(msg)
    if str(block["precision_argument"]) != PRECISION_ARGUMENT:
        msg = f"{name}.precision_argument must be {PRECISION_ARGUMENT!r}"
        raise ComparisonProtocolError(msg)
    if int(block["precision_value"]) != PRECISION_VALUE:
        msg = f"{name}.precision_value must be {PRECISION_VALUE}"
        raise ComparisonProtocolError(msg)


def _check_latency(block: Any) -> None:
    """Validate the latency benchmark protocol.

    Args:
        block: The parsed latency block.

    Raises:
        ComparisonProtocolError: If any frozen setting has moved, or a required
            control is absent.
    """
    if not isinstance(block, Mapping):
        msg = "latency_protocol: must be a mapping"
        raise ComparisonProtocolError(msg)
    try:
        check_keys(block, required=_LATENCY_KEYS, context="latency_protocol")
    except ConfigError as exc:
        raise ComparisonProtocolError(str(exc)) from exc

    if str(block["label"]) != BENCHMARK_LABEL:
        msg = (
            f"latency_protocol.label must be {BENCHMARK_LABEL!r}: the result is valid for this "
            "machine and runtime, not as a universal model latency"
        )
        raise ComparisonProtocolError(msg)
    if int(block["batch"]) != LATENCY_BATCH:
        msg = f"latency_protocol.batch must be {LATENCY_BATCH}"
        raise ComparisonProtocolError(msg)
    if int(block["imgsz"]) != COMPARISON_IMGSZ:
        msg = f"latency_protocol.imgsz must be {COMPARISON_IMGSZ}"
        raise ComparisonProtocolError(msg)
    if str(block["precision"]) != PRECISION:
        msg = (
            f"latency_protocol.precision must be {PRECISION!r}. Benchmarking one model in FP16 "
            "and the other in FP32 would measure the precision, not the models."
        )
        raise ComparisonProtocolError(msg)
    if int(block["warmup_iterations"]) != WARMUP_ITERATIONS:
        msg = f"latency_protocol.warmup_iterations must be {WARMUP_ITERATIONS}"
        raise ComparisonProtocolError(msg)
    if int(block["timed_iterations_per_image"]) != TIMED_ITERATIONS_PER_IMAGE:
        msg = (
            f"latency_protocol.timed_iterations_per_image must be "
            f"{TIMED_ITERATIONS_PER_IMAGE}. Changing it after observing variance is the move "
            "this file exists to prevent."
        )
        raise ComparisonProtocolError(msg)
    if int(block["benchmark_image_count"]) != BENCHMARK_IMAGE_COUNT:
        msg = f"latency_protocol.benchmark_image_count must be {BENCHMARK_IMAGE_COUNT}"
        raise ComparisonProtocolError(msg)
    if str(block["benchmark_selection_rule"]) != SELECTION_RULE:
        msg = f"latency_protocol.benchmark_selection_rule must be {SELECTION_RULE!r}"
        raise ComparisonProtocolError(msg)

    synchronization = block["synchronization"]
    if not isinstance(synchronization, Mapping) or synchronization.get("cuda_synchronize") is not (
        True
    ):
        msg = (
            "latency_protocol.synchronization.cuda_synchronize must be true. CUDA work is "
            "asynchronous, so an unsynchronised wall-clock reading measures dispatch, not "
            "execution."
        )
        raise ComparisonProtocolError(msg)
    for edge in ("before_timed_region", "after_timed_region"):
        if synchronization.get(edge) is not True:
            msg = f"latency_protocol.synchronization.{edge} must be true"
            raise ComparisonProtocolError(msg)
    if synchronization.get("same_primitive_for_both_models") is not True:
        msg = "latency_protocol.synchronization.same_primitive_for_both_models must be true"
        raise ComparisonProtocolError(msg)

    boundaries = block["timing_boundaries"]
    if not isinstance(boundaries, Mapping):
        msg = "latency_protocol.timing_boundaries: must be a mapping"
        raise ComparisonProtocolError(msg)
    for name in (MODEL_INFERENCE_LATENCY, END_TO_END_LATENCY):
        if name not in boundaries:
            msg = f"latency_protocol.timing_boundaries must define {name}"
            raise ComparisonProtocolError(msg)
    end_to_end = boundaries[END_TO_END_LATENCY]
    if not isinstance(end_to_end, Mapping):
        msg = f"latency_protocol.timing_boundaries.{END_TO_END_LATENCY}: must be a mapping"
        raise ComparisonProtocolError(msg)
    if end_to_end.get("segmenter_mask_reconstruction_included") is not True:
        msg = (
            "the segmenter's mask reconstruction must be inside its end-to-end measurement. "
            "Excluding it would hide the cost this comparison exists to quantify."
        )
        raise ComparisonProtocolError(msg)

    if not SHA256_PATTERN.match(str(block["benchmark_membership_sha256"])):
        msg = "latency_protocol.benchmark_membership_sha256 is not a SHA-256 digest"
        raise ComparisonProtocolError(msg)
    if block["abort_on_deviation"] is not True:
        msg = (
            "latency_protocol.abort_on_deviation must be true: a benchmark that continues after "
            "one model deviates is not a controlled comparison"
        )
        raise ComparisonProtocolError(msg)

    missing_statistics = [name for name in LATENCY_STATISTICS if name not in block["statistics"]]
    if missing_statistics:
        msg = f"latency_protocol.statistics must include {missing_statistics}"
        raise ComparisonProtocolError(msg)

    order = block["execution_order"]
    if not isinstance(order, Mapping) or order.get("interleaved") is not True:
        msg = (
            "latency_protocol.execution_order.interleaved must be true. Running one model to "
            "completion and then the other measures the machine's thermal state as well."
        )
        raise ComparisonProtocolError(msg)
    if order.get("symmetric") is not True:
        msg = "latency_protocol.execution_order.symmetric must be true"
        raise ComparisonProtocolError(msg)
    if order.get("randomized") is not False:
        msg = "latency_protocol.execution_order.randomized must be false"
        raise ComparisonProtocolError(msg)


def load_comparison_protocol(path: str | Path) -> ComparisonProtocol:
    """Load and validate the frozen comparison protocol.

    Args:
        path: Configuration file.

    Returns:
        The parsed protocol.

    Raises:
        ComparisonProtocolError: If the file is missing, malformed, names the
            protected split, mixes the two confidence thresholds, derives the
            segmenter's boxes from its masks, introduces an aggregate score, or
            carries a result.
    """
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        msg = f"Comparison protocol not found: {config_path.name}"
        raise ComparisonProtocolError(msg)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        msg = f"Comparison protocol is not valid YAML: {config_path.name} ({exc})"
        raise ComparisonProtocolError(msg) from exc
    if not isinstance(raw, Mapping):
        msg = f"Comparison protocol must be a mapping: {config_path.name}"
        raise ComparisonProtocolError(msg)

    try:
        check_keys(raw, required=_CONFIG_KEYS, context=config_path.name)
    except ConfigError as exc:
        raise ComparisonProtocolError(str(exc)) from exc

    if int(raw["schema_version"]) != CONFIG_SCHEMA_VERSION:
        msg = f"schema_version must be {CONFIG_SCHEMA_VERSION}"
        raise ComparisonProtocolError(msg)
    if str(raw["protocol"]) != PROTOCOL_NAME:
        msg = f"protocol must be {PROTOCOL_NAME!r}"
        raise ComparisonProtocolError(msg)
    if str(raw["status"]) != STATUS_FROZEN_NOT_EXECUTED:
        msg = f"status must be {STATUS_FROZEN_NOT_EXECUTED!r} while no comparison has run"
        raise ComparisonProtocolError(msg)
    if _contains_forbidden_split(dict(raw)):
        msg = "the comparison protocol references the protected split"
        raise ComparisonProtocolError(msg)
    if int(raw["comparison_input_imgsz"]) != COMPARISON_IMGSZ:
        msg = f"comparison_input_imgsz must be {COMPARISON_IMGSZ}"
        raise ComparisonProtocolError(msg)

    _check_model(
        raw["detector"], name="detector", experiment=DETECTOR_EXPERIMENT, role=DETECTOR_ROLE
    )
    _check_model(
        raw["segmenter"], name="segmenter", experiment=SEGMENTER_EXPERIMENT, role=SEGMENTER_ROLE
    )

    population = raw["population"]
    if not isinstance(population, Mapping):
        msg = "population: must be a mapping"
        raise ComparisonProtocolError(msg)
    if str(population.get("split")) != EVALUATION_SPLIT:
        msg = f"population.split must be {EVALUATION_SPLIT!r}"
        raise ComparisonProtocolError(msg)
    if population.get("identical_images_for_both_models") is not True:
        msg = "population.identical_images_for_both_models must be true"
        raise ComparisonProtocolError(msg)
    if not SHA256_PATTERN.match(str(population.get("membership_sha256", ""))):
        msg = "population.membership_sha256 is not a SHA-256 digest"
        raise ComparisonProtocolError(msg)

    _check_inference(raw["ap_inference"], name="ap_inference", conf=AP_CONFIDENCE)
    _check_inference(
        raw["operational_inference"], name="operational_inference", conf=OPERATIONAL_CONFIDENCE
    )

    evaluator = raw["canonical_box_evaluator"]
    if not isinstance(evaluator, Mapping):
        msg = "canonical_box_evaluator: must be a mapping"
        raise ComparisonProtocolError(msg)
    if str(evaluator.get("iou_type")) != IOU_TYPE:
        msg = f"canonical_box_evaluator.iou_type must be {IOU_TYPE!r}"
        raise ComparisonProtocolError(msg)
    declared = tuple(round(float(value), 2) for value in evaluator.get("iou_thresholds", ()))
    if declared != IOU_THRESHOLDS:
        msg = f"canonical_box_evaluator.iou_thresholds must be exactly {list(IOU_THRESHOLDS)}"
        raise ComparisonProtocolError(msg)
    if tuple(int(value) for value in evaluator.get("max_dets", ())) != MAX_DETS:
        msg = f"canonical_box_evaluator.max_dets must be exactly {list(MAX_DETS)}"
        raise ComparisonProtocolError(msg)
    if "adapter" in str(evaluator.get("ground_truth_document", "")).lower():
        msg = "canonical_box_evaluator.ground_truth_document points at an adapter"
        raise ComparisonProtocolError(msg)

    comparability = raw["box_comparability"]
    if not isinstance(comparability, Mapping):
        msg = "box_comparability: must be a mapping"
        raise ComparisonProtocolError(msg)
    if comparability.get("segmenter_boxes_derived_from_masks") is not False:
        msg = (
            "box_comparability.segmenter_boxes_derived_from_masks must be false. The comparison "
            "uses each model's actual predicted boxes; re-deriving the segmenter's from its "
            "masks would measure a post-processing choice this project invented."
        )
        raise ComparisonProtocolError(msg)

    features = raw["spatial_features"]
    if not isinstance(features, Mapping):
        msg = "spatial_features: must be a mapping"
        raise ComparisonProtocolError(msg)
    missing_features = [name for name in SPATIAL_FEATURES if name not in features]
    if missing_features:
        msg = f"spatial_features must define {missing_features}"
        raise ComparisonProtocolError(msg)

    proxies = raw["box_proxies"]
    if not isinstance(proxies, Mapping):
        msg = "box_proxies: must be a mapping"
        raise ComparisonProtocolError(msg)
    unpaired = [name for name in SPATIAL_FEATURES if name not in proxies]
    if unpaired:
        msg = (
            f"every spatial feature needs a declared box proxy or {BOX_PROXY_NONE}: {unpaired}. "
            "That pairing is what answers 'what does segmentation add'."
        )
        raise ComparisonProtocolError(msg)

    association = raw["association"]
    if not isinstance(association, Mapping):
        msg = "association: must be a mapping"
        raise ComparisonProtocolError(msg)
    if str(association.get("label")) != ASSOCIATION_LABEL:
        msg = (
            f"association.label must be {ASSOCIATION_LABEL!r}. The project has no canonical "
            "compliance ground truth, so an accuracy framing would be measuring against a "
            "label that does not exist."
        )
        raise ComparisonProtocolError(msg)
    if tuple(association.get("disagreement_categories", ())) != ASSOCIATION_CATEGORIES:
        msg = f"association.disagreement_categories must be exactly {list(ASSOCIATION_CATEGORIES)}"
        raise ComparisonProtocolError(msg)
    for forbidden in ("appearance_embeddings", "tracking", "learned_association"):
        if association.get(forbidden) is not False:
            msg = f"association.{forbidden} must be false: the comparison stays geometric"
            raise ComparisonProtocolError(msg)
    if association.get("classes_collapsed") is not False:
        msg = "association.classes_collapsed must be false: the five frozen classes stay distinct"
        raise ComparisonProtocolError(msg)

    if not raw["spatial_value_metrics"]:
        msg = (
            "spatial_value_metrics must list the descriptive comparisons before any result is "
            "inspected. Defining them afterwards would be choosing what to show."
        )
        raise ComparisonProtocolError(msg)

    _check_latency(raw["latency_protocol"])

    memory = raw["memory_protocol"]
    if not isinstance(memory, Mapping):
        msg = "memory_protocol: must be a mapping"
        raise ComparisonProtocolError(msg)
    if memory.get("reset_peak_stats_after_warmup") is not True:
        msg = "memory_protocol.reset_peak_stats_after_warmup must be true"
        raise ComparisonProtocolError(msg)
    if memory.get("training_memory_reused") is not False:
        msg = "memory_protocol.training_memory_reused must be false"
        raise ComparisonProtocolError(msg)

    reporting = raw["reporting"]
    if not isinstance(reporting, Mapping):
        msg = "reporting: must be a mapping"
        raise ComparisonProtocolError(msg)
    if reporting.get("aggregate_benefit_score") is not COMPOSITE_SCORE_FORBIDDEN:
        msg = (
            "reporting.aggregate_benefit_score must be false. Collapsing benefit and cost into "
            "one index would hide the trade-off this comparison exists to expose."
        )
        raise ComparisonProtocolError(msg)
    for name in (CANONICAL_BOX_METRIC, CANONICAL_BOX_METRIC_50):
        if name not in reporting.get("box_metrics", ()):
            msg = f"reporting.box_metrics must include {name}"
            raise ComparisonProtocolError(msg)

    if str(raw["test_policy"]) != HOLDOUT_STATUS:
        msg = f"test_policy must be {HOLDOUT_STATUS!r}"
        raise ComparisonProtocolError(msg)

    for forbidden in ("results", "measurements", "latency_results", "box_results"):
        if forbidden in raw:
            msg = f"the protocol must carry no {forbidden}: nothing has been executed"
            raise ComparisonProtocolError(msg)

    return ComparisonProtocol(raw=json.loads(json.dumps(raw)))


# --- the protocol validator ------------------------------------------------------------


def validate_protocol_manifest(
    manifest: Mapping[str, Any],
    *,
    protocol_fingerprint: str,
    detector_sha256: str,
    segmenter_sha256: str,
    membership_sha256: str,
    benchmark_sha256: str,
) -> list[str]:
    """Check a frozen comparison-protocol manifest against the frozen identities.

    Args:
        manifest: The parsed manifest.
        protocol_fingerprint: The committed configuration's fingerprint.
        detector_sha256: The frozen detector's checkpoint digest.
        segmenter_sha256: The frozen segmenter's checkpoint digest.
        membership_sha256: The validation membership fingerprint.
        benchmark_sha256: The benchmark subset's ordered fingerprint.

    Returns:
        One description per problem found, empty when the manifest is valid.
    """
    problems: list[str] = []

    if manifest.get("status") != STATUS_FROZEN_NOT_EXECUTED:
        problems.append(f"status must be {STATUS_FROZEN_NOT_EXECUTED!r}")
    if manifest.get("protocol") != PROTOCOL_NAME:
        problems.append(f"protocol must be {PROTOCOL_NAME!r}")
    if manifest.get("protocol_fingerprint") != protocol_fingerprint:
        problems.append("protocol_fingerprint does not match the committed configuration")

    detector = manifest.get("detector", {})
    segmenter = manifest.get("segmenter", {})
    if detector.get("checkpoint_sha256") != detector_sha256:
        problems.append("the detector checkpoint is not the frozen one")
    if segmenter.get("checkpoint_sha256") != segmenter_sha256:
        problems.append("the segmenter checkpoint is not the frozen one")
    if detector.get("experiment") != DETECTOR_EXPERIMENT:
        problems.append(f"the detector must be {DETECTOR_EXPERIMENT}")
    if segmenter.get("experiment") != SEGMENTER_EXPERIMENT:
        problems.append(f"the segmenter must be {SEGMENTER_EXPERIMENT}")
    for name, block in (("detector", detector), ("segmenter", segmenter)):
        if int(block.get("imgsz", 0)) != COMPARISON_IMGSZ:
            problems.append(f"{name}.imgsz must be {COMPARISON_IMGSZ}")

    population = manifest.get("population", {})
    if population.get("split") != EVALUATION_SPLIT:
        problems.append(f"the comparison population must be the {EVALUATION_SPLIT} split")
    if population.get("membership_sha256") != membership_sha256:
        problems.append("the validation membership fingerprint does not match")
    if population.get("identical_images_for_both_models") is not True:
        problems.append("both models must see identical images")

    ap = manifest.get("ap_inference", {})
    operational = manifest.get("operational_inference", {})
    if float(ap.get("conf", -1)) != AP_CONFIDENCE:
        problems.append(f"the AP confidence must be {AP_CONFIDENCE}")
    if float(operational.get("conf", -1)) != OPERATIONAL_CONFIDENCE:
        problems.append(f"the operational confidence must be {OPERATIONAL_CONFIDENCE}")
    for name, block in (("ap_inference", ap), ("operational_inference", operational)):
        if float(block.get("iou", -1)) != NMS_IOU:
            problems.append(f"{name}.iou must be {NMS_IOU}")
        if block.get("augment") is not False or block.get("tta") is not False:
            problems.append(f"{name} must disable augmentation and TTA")
        if block.get("precision") != PRECISION:
            problems.append(f"{name}.precision must be {PRECISION}")

    latency = manifest.get("latency_protocol", {})
    if int(latency.get("batch", 0)) != LATENCY_BATCH:
        problems.append(f"the latency batch must be {LATENCY_BATCH}")
    if int(latency.get("warmup_iterations", 0)) != WARMUP_ITERATIONS:
        problems.append(f"warmup must be exactly {WARMUP_ITERATIONS} iterations")
    if int(latency.get("timed_iterations_per_image", 0)) != TIMED_ITERATIONS_PER_IMAGE:
        problems.append(f"timed iterations must be exactly {TIMED_ITERATIONS_PER_IMAGE}")
    if latency.get("precision") != PRECISION:
        problems.append(f"the latency precision must be {PRECISION} for both models")
    if latency.get("benchmark_membership_sha256") != benchmark_sha256:
        problems.append("the benchmark membership or its order does not match")
    synchronization = latency.get("synchronization", {})
    if synchronization.get("cuda_synchronize") is not True:
        problems.append("CUDA synchronization is required around every timed region")
    boundaries = latency.get("timing_boundaries", {})
    if END_TO_END_LATENCY not in boundaries or MODEL_INFERENCE_LATENCY not in boundaries:
        problems.append("both timing boundaries must be defined")
    elif boundaries[END_TO_END_LATENCY].get("segmenter_mask_reconstruction_included") is not True:
        problems.append("mask reconstruction must be inside the segmenter's end-to-end timing")

    if manifest.get("reporting", {}).get("aggregate_benefit_score") is not False:
        problems.append("no aggregate benefit score may be defined")

    test = manifest.get("test", {})
    if test.get("status") != HOLDOUT_STATUS:
        problems.append(f"test.status must be {HOLDOUT_STATUS!r}")
    body = {key: value for key, value in manifest.items() if key != "test"}
    if _contains_forbidden_split(body):
        problems.append("the manifest names the protected split outside its holdout declaration")

    for forbidden in ("results", "measurements", "latency_results", "box_results"):
        if forbidden in manifest:
            problems.append(f"the manifest carries {forbidden}: nothing has been executed")

    return problems
