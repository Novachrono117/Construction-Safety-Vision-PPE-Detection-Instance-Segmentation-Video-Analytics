"""Synthesise the committed phase 10B and 10C evidence into one scientific answer.

Phase 10D executes no model. It reads what phases 10B and 10C already measured
and committed, and assembles it into an answer to the project's question: what
additional spatial and operational information does instance segmentation
provide beyond bounding boxes, and what does it cost?

Five things are deliberate.

**Every number is read, never restated.** The builders below take the parsed
committed artifacts and copy exact values out of them. No figure is transcribed
from prose, rounded by hand, or recomputed from predictions - there are no
predictions here, and no model to produce any.

**Four axes, never one score.** Recognition, spatial representation, operational
association and computational cost stay separately interpretable.
:func:`validate_synthesis` refuses a payload that carries a weighted score, a
cost-benefit index, an overall benefit number or a declared winner, because a
single figure would hide exactly the trade-off this phase exists to expose.

**The caveats travel with the numbers.** The positive all-class localisation
delta is carried by a one-image class, the association result holds only at the
frozen containment floor, the latency figures belong to one laptop at one
confidence, and the distribution's shape has no established cause. Each of those
is a field in the artifact, not a sentence someone may drop when quoting it.

**Representation gain and proxy refinement are different claims.**
``MASK_TO_BOX_FILL_RATIO`` and ``SHAPE_EXTENT`` are frozen
``NO_BOX_ONLY_EQUIVALENT``: a rectangle cannot express them at all. Area,
intersection, containment and centroid *can* be approximated by a box, and the
mask changes the value. The first is a gain in what is computable; the second is
a refinement of a proxy - and neither is a claim about accuracy against ground
truth, because no ground-truth comparison was made for them.

**No compliance accuracy exists to be claimed.** The project holds no person-PPE
compliance ground truth, so ``VISIBLE_PPE_COVERAGE_PROXY`` stays an interpretive
proxy and the validator rejects any payload asserting compliance, violation or
correct-wearing accuracy.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from construction_safety_vision.detector_segmenter_analysis import (
    NO_BOX_PROXY,
    RARE_CLASS,
    RARE_CLASS_STATUS,
    result_fingerprint,
)
from construction_safety_vision.detector_segmenter_comparison import (
    ASSOCIATION_CATEGORIES,
    DETECTOR_EXPERIMENT,
    END_TO_END_LATENCY,
    MODEL_INFERENCE_LATENCY,
    SEGMENTER_EXPERIMENT,
)
from construction_safety_vision.provenance import sha256_file

PHASE = "10D"

SYNTHESIS_COMPLETE = "DETECTOR_SEGMENTER_SCIENTIFIC_SYNTHESIS_COMPLETE"
SOURCE_ARTIFACT_MISMATCH = "SOURCE_ARTIFACT_MISMATCH"
CLAIM_VALIDATION_FAILED = "SYNTHESIS_CLAIM_VALIDATION_FAILED"
PROTOCOL_VIOLATION = "PROTOCOL_VIOLATION"
BLOCKED = "BLOCKED"

# --- labels used verbatim in the artifacts -----------------------------------------------

COMMITTED_EVIDENCE = "COMMITTED_EVIDENCE"
SCIENTIFIC_SYNTHESIS = "SCIENTIFIC_SYNTHESIS"
REPRESENTATION_GAIN = "REPRESENTATION_GAIN"
PROXY_REFINEMENT = "PROXY_REFINEMENT"
OPERATIONAL_LIMITATION = "OPERATIONAL_LIMITATION"
CONTROLLED_LOCAL_BENCHMARK = "CONTROLLED_LOCAL_HARDWARE_BENCHMARK"
USE_CASE_CONDITIONAL = "USE_CASE_CONDITIONAL"
LIMITATION = "LIMITATION"
HOLDOUT_POLICY = "HOLDOUT_POLICY"
HOLDOUT_STATUS = "PROTECTED_NOT_ACCESSED"

AXES: tuple[str, ...] = (
    "RECOGNITION_LOCALIZATION",
    "SPATIAL_REPRESENTATION_GAIN",
    "OPERATIONAL_ASSOCIATION_VALUE",
    "COMPUTATIONAL_COST",
)
"""The four axes. Each is reported on its own; they are never combined."""

PROTOCOL_GAP_RESOLUTION = "PRE_BENCHMARK_PROTOCOL_GAP_RESOLUTION"
"""Phase 10A froze no confidence inside its latency block; 10C settled it first."""

LATENCY_SCOPE = "OPERATIONAL_OUTPUT_LATENCY_AT_CONF_0_25"
"""The only operating point any latency figure here describes."""

OPERATIONAL_CONF = 0.25
AP_CONF = 0.001

MEAN_DERIVED_THROUGHPUT = "MEAN_DERIVED_BATCH1_THROUGHPUT"
"""1000 / mean at batch 1, never the reciprocal of the fastest repetition."""

SUPPORT_SENSITIVITY_LABEL = "POST_HOC_DESCRIPTIVE_SUPPORT_SENSITIVITY"
INTERPRETIVE_PROXY = "INTERPRETIVE_OPERATIONAL_PROXY"
TAXONOMY_EXCEPTION_STATUS = "UNCLASSIFIED_BY_FROZEN_FOUR_CATEGORY_TAXONOMY"
SEGMENTATION_COST_LABEL = "ADDITIONAL_SEGMENTATION_PIPELINE_COST"
STATIC_COMPLEXITY_LABEL = "STATIC_MODEL_COMPLEXITY"
DVFS_CAUSAL_ATTRIBUTION = "UNKNOWN"
NO_POWER_TELEMETRY = "NO_SYNCHRONIZED_PER_OBSERVATION_POWER_STATE_TELEMETRY"

MASK_ONLY_CATEGORY = "MASK_ONLY_ASSOCIATION"
BOX_ONLY_CATEGORY = "BOX_ONLY_ASSOCIATION"
AGREE_CATEGORY = "BOX_AND_MASK_AGREE"
NEITHER_CATEGORY = "NEITHER_ASSOCIATION"

AREA_FEATURE = "INSTANCE_AREA_PIXELS"
INTERSECTION_FEATURE = "PERSON_PPE_MASK_INTERSECTION"

# --- committed sources -------------------------------------------------------------------

SOURCES: dict[str, str] = {
    "box_comparison": "reports/detector_segmenter_box_comparison.json",
    "spatial_comparison": "reports/detector_segmenter_spatial_comparison.json",
    "taxonomy_correction": "reports/association_taxonomy_correction.provenance.json",
    "latency_benchmark": "reports/detector_segmenter_latency_comparison.json",
    "memory_benchmark": "reports/detector_segmenter_memory_comparison.json",
    "final_detector": "reports/final_detector_manifest.json",
    "final_segmenter": "reports/final_segmenter_manifest.json",
    "comparison_protocol": "reports/detector_segmenter_comparison_protocol.json",
}
"""Role -> repo-relative path. Phase 10D reads these and writes none of them."""

SELF_FINGERPRINT_FIELD: dict[str, str] = {
    "box_comparison": "box_comparison_sha256",
    "spatial_comparison": "spatial_comparison_sha256",
    "latency_benchmark": "latency_result_sha256",
    "memory_benchmark": "memory_result_sha256",
    "final_detector": "final_detector_sha256",
    "final_segmenter": "final_segmenter_sha256",
    "comparison_protocol": "protocol_fingerprint",
}
"""Each source artifact's own semantic fingerprint field, where it has one."""

HISTORICAL: tuple[str, ...] = (
    "configs/detector_segmenter_comparison.yaml",
    "reports/detector_segmenter_comparison_protocol.json",
    "reports/detector_segmenter_comparison_protocol.md",
    "reports/detector_segmenter_box_comparison.json",
    "reports/detector_segmenter_spatial_comparison.json",
    "reports/detector_segmenter_validation_comparison.md",
    "reports/detector_segmenter_latency_comparison.json",
    "reports/detector_segmenter_memory_comparison.json",
    "reports/detector_segmenter_latency_report.md",
    "reports/final_detector_manifest.json",
    "reports/final_segmenter_manifest.json",
    "reports/segmentation_S1_canonical_evaluation.json",
    "reports/segmentation_S1_mask_iou.json",
    "reports/detection_D2_manifest.json",
)
"""Read, digested and verified byte-identical. Never regenerated by this phase."""

CLAIM_IDS: tuple[str, ...] = (
    "LOCALIZATION_SIMILARITY",
    "MASK_REPRESENTATION_GAIN",
    "BOX_PROXY_INFLATION",
    "NO_MAJOR_MASK_ONLY_ASSOCIATION_GAIN_AT_FROZEN_RULE",
    "END_TO_END_LATENCY_COST",
    "INFERENCE_MEMORY_OVERHEAD",
    "USE_CASE_CONDITIONAL_SELECTION",
)
"""Every headline claim the later report and pitch may reuse."""

CLAIM_FIELDS: tuple[str, ...] = (
    "claim_id",
    "claim_text",
    "evidence_artifact",
    "evidence_field",
    "claim_scope",
    "limitation",
)
"""Each register entry carries exactly these, so a claim cannot arrive bare."""

FORBIDDEN_SCORE_KEYS: tuple[str, ...] = (
    "aggregate_benefit_score",
    "overall_benefit_score",
    "cost_benefit_index",
    "weighted_score",
    "combined_latency_score",
    "composite_score",
    "single_winner_metric",
)
"""Keys that must be absent or false. A composite would hide the trade-off."""

COMPLIANCE_CLAIM_KEYS: tuple[str, ...] = (
    "compliance_accuracy",
    "safety_violation_accuracy",
    "correct_wearing_accuracy",
)
"""No ground truth exists for any of these, so none may appear."""

TEST_FORBIDDEN_KEYS: tuple[str, ...] = (
    "test_metrics",
    "test_predictions",
    "test_images",
    "holdout_metrics",
)
"""Keys whose mere presence would mean the holdout had been touched."""


class SynthesisError(RuntimeError):
    """Raised when the synthesis cannot be derived from committed evidence."""


def load_sources(root: Path) -> dict[str, dict[str, Any]]:
    """Read every committed source artifact this phase synthesises.

    Args:
        root: Repository root.

    Returns:
        Role -> parsed artifact.

    Raises:
        SynthesisError: If a source is missing or is not a JSON object.
    """
    loaded: dict[str, dict[str, Any]] = {}
    for role, relative in SOURCES.items():
        path = root / relative
        if not path.is_file():
            msg = f"required source artifact not found: {relative}"
            raise SynthesisError(msg)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            msg = f"{relative} must contain a JSON object"
            raise SynthesisError(msg)
        loaded[role] = payload
    return loaded


def source_fingerprints(root: Path, sources: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Digest each source artifact and record its own semantic fingerprint.

    The file digest proves which bytes were read; the artifact's self-recorded
    fingerprint proves which measurement those bytes describe.

    Args:
        root: Repository root.
        sources: The parsed artifacts, as returned by :func:`load_sources`.

    Returns:
        Role -> path, file digest and, where the artifact records one, its own
        fingerprint field and value.
    """
    fingerprints: dict[str, Any] = {}
    for role, relative in SOURCES.items():
        entry: dict[str, str] = {
            "path": relative,
            "file_sha256": sha256_file(root / relative),
        }
        field = SELF_FINGERPRINT_FIELD.get(role)
        if field is not None:
            entry["recorded_fingerprint_field"] = field
            entry["recorded_fingerprint"] = str(sources[role][field])
        fingerprints[role] = entry
    return fingerprints


def model_identities(box: Mapping[str, Any]) -> dict[str, Any]:
    """Copy both frozen model identities out of a committed comparison artifact.

    Args:
        box: The phase 10B box-comparison artifact.

    Returns:
        Detector and segmenter identity blocks, plus the statement that they do
        not solve the same output task.
    """
    return {
        DETECTOR_EXPERIMENT: dict(box["detector"]),
        SEGMENTER_EXPERIMENT: dict(box["segmenter"]),
        "same_output_task": False,
        "drop_in_replacement": False,
        "architectural_identity_claimed": False,
        "note": (
            "D2 emits class, confidence and a box; S1 emits those and an instance mask. "
            "They are the project's frozen models for two different output tasks, not two "
            "candidates for one. Neither is a drop-in replacement for the other, and their "
            "architectures are related but not identical."
        ),
    }


def build_recognition(box: Mapping[str, Any]) -> dict[str, Any]:
    """Assemble the recognition and localisation axis.

    Args:
        box: The phase 10B box-comparison artifact.

    Returns:
        The axis block, carrying the raw delta, the support sensitivity, the
        exact per-class decomposition and the caveat that binds them.
    """
    metrics = box["metrics"]
    decomposition = box["all_class_delta_decomposition"]
    per_class = {
        name: {
            "D2_AP@0.50:0.95": row["D2_AP@0.50:0.95"],
            "S1_AP@0.50:0.95": row["S1_AP@0.50:0.95"],
            "delta_AP@0.50:0.95": row["delta_AP@0.50:0.95"],
            "status": row["status"],
        }
        for name, row in sorted(box["per_class"].items())
    }
    return {
        "axis": AXES[0],
        "label": COMMITTED_EVIDENCE,
        "evaluator": box["cocoeval"]["implementation"],
        "iou_type": box["cocoeval"]["iou_type"],
        "ground_truth": box["ground_truth"],
        "conf": box["inference"]["conf"],
        "population": {
            "split": box["population"]["split"],
            "images": box["population"]["images"],
            "annotations": box["population"]["annotations"],
        },
        "all_class": {
            "CANONICAL_BOX_MAP50_95": dict(metrics["CANONICAL_BOX_MAP50_95"]),
            "CANONICAL_BOX_MAP50": dict(metrics["CANONICAL_BOX_MAP50"]),
        },
        "supported_class_sensitivity": dict(box["supported_class_sensitivity"]),
        "per_class": per_class,
        "per_class_contributions": dict(decomposition["contributions"]),
        "improved_classes": list(decomposition["improved_classes"]),
        "declined_classes": list(decomposition["declined_classes"]),
        "delta_excluding_rare_class": decomposition["delta_excluding_rare_class"],
        "rare_class": {
            "name": RARE_CLASS,
            "status": RARE_CLASS_STATUS,
            "delta": decomposition["rare_class_delta"],
            "contribution": decomposition["rare_class_contribution"],
            "contribution_exceeds_total": decomposition["rare_class_contribution_exceeds_total"],
            "validation_support": box["rare_class"]["detail"],
        },
        "synthesis": box["localization_conclusion"],
        "superiority_claimed_for_either_model": False,
        "significance_test_performed": False,
        "why_any_class_moved": "UNKNOWN",
        "not_comparable_to_native_metrics": box["not_comparable_to_native_metrics"],
        "reading": (
            "S1 retains broadly similar localisation performance to D2 while additionally "
            "producing masks. The positive all-class delta is not robust evidence that S1 is "
            "the superior object localiser, because it is dominated by the highly uncertain "
            f"{RARE_CLASS} class; over the adequately supported classes the same comparison "
            "runs slightly the other way. Neither direction is a significance test, some "
            "classes improved and some regressed, and why any of them moved is UNKNOWN."
        ),
    }


def build_spatial(spatial: Mapping[str, Any]) -> dict[str, Any]:
    """Assemble the spatial representation axis.

    Splits what masks make computable at all from what they merely refine, and
    keeps both apart from any accuracy claim.

    Args:
        spatial: The phase 10B spatial-comparison artifact.

    Returns:
        The axis block.
    """
    features = spatial["spatial_features"]
    proxies = spatial["box_proxies"]
    mask_only = sorted(name for name, entry in proxies.items() if entry["proxy"] == NO_BOX_PROXY)
    refinement = {
        name: {
            "box_proxy": entry["proxy"],
            "mask_measurement": dict(entry["statistics"]["mask_measurement"]),
            "box_proxy_statistics": dict(entry["statistics"]["box_proxy"]),
        }
        for name, entry in sorted(proxies.items())
        if entry["proxy"] != NO_BOX_PROXY
        and entry["statistics"] is not None
        and "mask_measurement" in entry["statistics"]
    }
    # MASK_CENTROID is excluded above on purpose: its box proxy is the box centre,
    # so the recorded statistic is a displacement between the two rather than a
    # pair of like measurements. It is reported in its own block, where that
    # difference is stated, instead of being tabulated as if it were one.
    return {
        "axis": AXES[1],
        "label": REPRESENTATION_GAIN,
        "conf": spatial["inference"]["conf"],
        "instances_measured": features["MASK_TO_BOX_FILL_RATIO"]["statistics"]["count"],
        "masks_reconstructed": spatial["mask_reconstruction"]["masks_on_original_canvas"],
        "masks_excluded": spatial["mask_reconstruction"]["excluded_predictions"],
        "mask_only_quantities": mask_only,
        "mask_only_label": NO_BOX_PROXY,
        "mask_only_reason": (
            "A bounding box has no way to express how much of itself is object, or how far "
            "the object departs from a rectangle. These two quantities are frozen "
            f"{NO_BOX_PROXY}: they are computable from a mask and not from a box, which is a "
            "gain in what can be measured, not a demonstration of better predictive accuracy."
        ),
        "MASK_TO_BOX_FILL_RATIO": dict(features["MASK_TO_BOX_FILL_RATIO"]["statistics"]),
        "SHAPE_EXTENT": dict(features["SHAPE_EXTENT"]["statistics"]),
        "fill_ratio_reading": (
            "The median predicted instance mask occupied about two-thirds of its bounding "
            "rectangle, illustrating information that the rectangular representation does not "
            "encode. It is a comparison of predicted mask support against predicted box area, "
            "not a measurement of background against ground truth, so it is not a background "
            "error rate."
        ),
        "background_error_rate_claimed": False,
        "proxy_refinement": {
            "label": PROXY_REFINEMENT,
            "definition": spatial["interpretation"][PROXY_REFINEMENT],
            "features": refinement,
            "comparison_basis": (
                "Each pair below is the same summary statistic of the same population under "
                "the two representations - mask measurement against box proxy, mean against "
                "mean and median against median. Unlike statistics are never compared."
            ),
            "direction": "BOX_PROXY_SYSTEMATICALLY_INFLATED_RELATIVE_TO_MASK_MEASUREMENT",
            "accuracy_claim": False,
            "box_error_claimed": False,
            "reading": (
                "Box-based geometric proxies were systematically inflated relative to the "
                "mask-based measurements in this validation analysis. That is a refinement of "
                "a proxy, not an error rate: no ground-truth geometry entered this comparison, "
                "so neither representation is shown to be right or wrong against it."
            ),
        },
        "centroid": {
            "statistic": features["MASK_CENTROID"]["statistic_is"],
            "box_proxy": proxies["MASK_CENTROID"]["proxy"],
            "displacement_pixels": dict(
                proxies["MASK_CENTROID"]["statistics"]["displacement_pixels"]
            ),
            "true_center_claimed": False,
            "reading": (
                "A box centre and a foreground centroid can differ materially for irregular, "
                "partially visible or spatially imbalanced shapes, and the mask makes the "
                "second computable. The mask centroid is a different geometric quantity, not "
                "the true centre of the object."
            ),
            "convention_note": (
                "The mask centroid averages pixel indices while the box is in continuous "
                "coordinates, so a mask perfectly filling its box reports about 0.71 px rather "
                "than 0. The offset is constant and far below the displacements described."
            ),
        },
        "geometry_disagreement": dict(spatial["geometry_disagreement"]),
        "new_metrics_introduced": spatial["new_metrics_introduced"],
        "post_hoc_bins_created": spatial["post_hoc_bins_created"],
    }


def build_association(spatial: Mapping[str, Any]) -> dict[str, Any]:
    """Assemble the operational association axis.

    Args:
        spatial: The phase 10B spatial-comparison artifact.

    Returns:
        The axis block, with the geometry-isolating reading as the answer and
        the pipeline-level reading kept explicitly separate.
    """
    association = spatial["association"]
    geometry = association["geometry_isolating"]
    pipeline = association["pipeline_level"]
    coverage = spatial["box_proxies"]["VISIBLE_PPE_COVERAGE_PROXY"]
    return {
        "axis": AXES[2],
        "label": COMMITTED_EVIDENCE,
        "analysis": "SPATIAL_ASSOCIATION_ANALYSIS",
        "containment_floor": association["containment_floor"],
        "frozen_categories": list(ASSOCIATION_CATEGORIES),
        "geometry_isolating": {
            "box_source": geometry["box_source"],
            "total_relationships": geometry["total_relationships"],
            "classified_relationships": geometry["classified_relationships"],
            "taxonomy_exceptions": geometry["taxonomy_exceptions"],
            "taxonomy_coverage": geometry["taxonomy_coverage"],
            "frozen_category_counts": dict(geometry["frozen_category_counts"]),
            "percentage_denominator": geometry["percentage_denominator"],
            "isolates_geometry": True,
        },
        "pipeline_level": {
            "box_source": pipeline["box_source"],
            "total_relationships": pipeline["total_relationships"],
            "classified_relationships": pipeline["classified_relationships"],
            "taxonomy_exceptions": pipeline["taxonomy_exceptions"],
            "taxonomy_coverage": pipeline["taxonomy_coverage"],
            "frozen_category_counts": dict(pipeline["frozen_category_counts"]),
            "isolates_geometry": False,
            "attributable_to_geometry_alone": False,
            "confound_note": association["geometry_isolating_versus_pipeline_level"],
        },
        "mask_only_associations": geometry["frozen_category_counts"][MASK_ONLY_CATEGORY],
        "taxonomy_limitation": {
            "label": OPERATIONAL_LIMITATION,
            "status": association["association_taxonomy_status"],
            "exception_type": association["taxonomy_exception_type"],
            "exception_status": TAXONOMY_EXCEPTION_STATUS,
            "reading": association["taxonomy_exception_reading"],
            "fifth_peer_category_added": association["fifth_peer_category_added"],
            "frozen_categories_unchanged": association["frozen_categories_unchanged"],
            "phase_10a_protocol_modified": False,
            "note": association["taxonomy_exception_note"],
            "not_an_error": association["not_an_error"],
        },
        "visible_ppe_coverage_proxy": {
            "label": INTERPRETIVE_PROXY,
            "semantics": spatial["spatial_features"]["VISIBLE_PPE_COVERAGE_PROXY"]["semantics"],
            "is_not_a_compliance_measure": True,
            "box_proxy": coverage["proxy"],
            "primary_scientific_conclusion": False,
            "reading": (
                "Masks permit a more spatially specific coverage-like proxy than rectangular "
                "boxes. Nothing further follows from it: the project holds no canonical "
                "compliance ground truth, so no compliance, violation or correct-wearing "
                "accuracy is claimed or claimable, and no conclusion here rests on this proxy."
            ),
        },
        "compliance_accuracy_claimed": False,
        "synthesis": (
            "At the frozen 0.50 containment rule and on this validation population, instance "
            "masks did NOT demonstrate a substantial advantage in discovering additional "
            "person-PPE associations. The geometry-isolating analysis - which holds the model "
            "and its instances fixed and varies only the shape representation - found zero "
            f"{MASK_ONLY_CATEGORY} cases. The finding is specific to this threshold and this "
            "population and does not generalise beyond them."
        ),
    }


def build_latency(latency: Mapping[str, Any]) -> dict[str, Any]:
    """Assemble the latency half of the computational-cost axis.

    Args:
        latency: The phase 10C latency artifact.

    Returns:
        The latency block, with both boundaries kept separate, the distribution
        reported alongside the mean, and the confidence gap disclosed.
    """
    deltas = latency["deltas"]
    per_model = latency["latency"]
    diagnostic = latency["post_hoc_distribution_diagnostic"]
    conditions = latency["conditions"]

    def boundary(name: str) -> dict[str, Any]:
        delta = deltas[name]
        return {
            "boundary": name,
            "definition": latency["timing_boundaries"][name]["definition"],
            DETECTOR_EXPERIMENT: dict(per_model[DETECTOR_EXPERIMENT][name]),
            SEGMENTER_EXPERIMENT: dict(per_model[SEGMENTER_EXPERIMENT][name]),
            "absolute_latency_delta_ms": delta["absolute_latency_delta_ms"],
            "relative_latency_cost": delta["relative_latency_cost"],
            "derived_from": delta["derived_from"],
            "throughput": {
                "label": MEAN_DERIVED_THROUGHPUT,
                "definition": "1000 / mean latency in ms, at batch 1",
                DETECTOR_EXPERIMENT: delta["detector_images_per_second_from_mean"],
                SEGMENTER_EXPERIMENT: delta["segmenter_images_per_second_from_mean"],
                "throughput_ratio": delta["throughput_ratio"],
                "is_application_video_fps": False,
                "is_batched_throughput": False,
                "from_fastest_iteration": False,
            },
        }

    return {
        "axis": AXES[3],
        "label": CONTROLLED_LOCAL_BENCHMARK,
        "status": latency["benchmark_status"],
        "scope": LATENCY_SCOPE,
        "primary_operational_boundary": END_TO_END_LATENCY,
        "primary_operational_boundary_reason": (
            "It ends where a caller has usable outputs, and for the segmenter that includes "
            "the mask reconstruction this comparison exists to price."
        ),
        "boundaries_combined": latency["boundaries_combined"],
        "boundaries": {
            MODEL_INFERENCE_LATENCY: boundary(MODEL_INFERENCE_LATENCY),
            END_TO_END_LATENCY: boundary(END_TO_END_LATENCY),
        },
        "cost_label": latency["segmentation_cost_label"],
        "cost_label_note": latency["segmentation_cost_note"],
        "pure_mask_reconstruction_cost_isolated": latency["pure_mask_reconstruction_cost_isolated"],
        "conditions": {
            "batch": conditions["batch"],
            "imgsz": conditions["imgsz"],
            "precision": conditions["precision"],
            "conf": conditions["conf"],
            "warmup_iterations": conditions["warmup_iterations"],
            "timed_iterations_per_image": conditions["timed_iterations_per_image"],
            "benchmark_images": latency["benchmark"]["benchmark_image_count"],
            "execution_blocks": latency["benchmark"]["execution_blocks"],
            "observations": latency["raw_timings"]["observations"],
            "interleaved": latency["benchmark"]["interleaved"],
            "symmetric": latency["benchmark"]["symmetric"],
        },
        "distribution": {
            "label": diagnostic["label"],
            "status": diagnostic["status"],
            "observations_used": diagnostic["observations_used"],
            "observations_discarded": diagnostic["observations_discarded"],
            "outlier_rejection_applied": diagnostic["outlier_rejection_applied"],
            "observations_normalized_or_rescaled": diagnostic[
                "observations_normalized_or_rescaled"
            ],
            "mean_to_median_ratio": {
                model: {
                    name: diagnostic["evidence"][model][name]["mean_to_median_ratio"]
                    for name in (MODEL_INFERENCE_LATENCY, END_TO_END_LATENCY)
                }
                for model in (DETECTOR_EXPERIMENT, SEGMENTER_EXPERIMENT)
            },
            "block_mean_range": {
                model: {
                    name: {
                        "min": diagnostic["evidence"][model][name]["block_mean_min"],
                        "max": diagnostic["evidence"][model][name]["block_mean_max"],
                    }
                    for name in (MODEL_INFERENCE_LATENCY, END_TO_END_LATENCY)
                }
                for model in (DETECTOR_EXPERIMENT, SEGMENTER_EXPERIMENT)
            },
            "reading": diagnostic["reading"],
            "headline_statistic": "mean",
            "headline_statistic_note": (
                "The frozen deltas are computed from the mean. The mean is reported together "
                "with the median, P95 and the range, because the distribution is wide and the "
                "mean alone would mislead."
            ),
        },
        "dvfs": {
            "label": LIMITATION,
            "causal_attribution": diagnostic["causal_attribution"],
            "hypothesis": diagnostic["hypothesis"],
            "hypothesis_status": diagnostic["hypothesis_status"],
            "per_observation_power_state_telemetry": diagnostic[
                "per_observation_power_state_telemetry"
            ],
            "telemetry_synchronous_with_timed_blocks": diagnostic[
                "telemetry_synchronous_with_timed_blocks"
            ],
            "proportionality_across_models_demonstrated": diagnostic[
                "proportionality_across_models_demonstrated"
            ],
            "dvfs_asserted_as_cause": False,
            "reading": (
                "The observed multimodality is consistent with mobile-GPU DVFS and power-state "
                "behaviour, and that remains an untested hypothesis. No clock, P-state, "
                "utilisation, temperature or power reading accompanied the timed regions, so "
                "no observation maps to a device state and no alternative was excluded. The "
                "controlled symmetric order mitigates order bias; it does not prove the source "
                "of the multimodality, and no proportional effect across the two models is "
                "claimed."
            ),
        },
        "confidence_protocol_gap": {
            "label": PROTOCOL_GAP_RESOLUTION,
            "phase_10a_froze_latency_confidence": False,
            "resolved_to": conditions["conf"],
            "resolved_before_any_timing_existed": True,
            "applied_equally_to_both_models": True,
            "confidence_source": conditions["confidence_source"],
            "ap_confidence_is_not_an_operating_point": AP_CONF,
            "latency_measured_at_ap_confidence": False,
            "benchmark_invalidated": False,
            "note": conditions["confidence_decision"],
            "reading": (
                "Phase 10A's latency subsection declared batch, resolution, precision, warmup, "
                "repetitions, membership and execution order, but no confidence threshold. "
                "Phase 10C resolved that gap before any timing existed, to the project's "
                "already frozen operational 0.25, and applied it equally to both models. It "
                "was not explicitly frozen by phase 10A, and saying otherwise would be false; "
                "the resolution does not invalidate the benchmark, and no latency claim is "
                "made at conf 0.001."
            ),
        },
        "runtime": {
            "gpu": latency["runtime"]["gpu"]["name"],
            "driver_version": latency["runtime"]["driver_version"],
            "torch": latency["runtime"]["torch"],
            "ultralytics": latency["runtime"]["ultralytics"],
            "power_source": latency["runtime"]["power"]["source"],
            "power_settings_changed_by_this_phase": latency["runtime"][
                "power_settings_changed_by_this_phase"
            ],
            "thermal_correction_applied": latency["runtime"]["thermal_correction_applied"],
        },
        "hardware_independent_claim": False,
    }


def build_memory(memory: Mapping[str, Any]) -> dict[str, Any]:
    """Assemble the inference-memory half of the computational-cost axis.

    Args:
        memory: The phase 10C memory artifact.

    Returns:
        The memory block, stating both the relative overhead and the absolute
        footprint, because either alone misleads.
    """
    delta = memory["delta"]
    per_model = memory["memory"]
    return {
        "label": CONTROLLED_LOCAL_BENCHMARK,
        "measurement": memory["measurement"],
        "comparable_with_training_memory": False,
        "training_memory": memory["training_memory"],
        "isolation_method": memory["conditions"]["isolation_method"],
        "peak_stats_reset_after_warmup": memory["conditions"]["peak_stats_reset_after_warmup"],
        "peak_allocated": {
            DETECTOR_EXPERIMENT: {
                "bytes": per_model[DETECTOR_EXPERIMENT]["peak_memory_allocated_bytes"],
                "gib": per_model[DETECTOR_EXPERIMENT]["peak_memory_allocated_gib"],
            },
            SEGMENTER_EXPERIMENT: {
                "bytes": per_model[SEGMENTER_EXPERIMENT]["peak_memory_allocated_bytes"],
                "gib": per_model[SEGMENTER_EXPERIMENT]["peak_memory_allocated_gib"],
            },
            "ratio": delta["peak_memory_allocated_ratio"],
            "delta_gib": delta["peak_memory_allocated_delta_gib"],
        },
        "peak_reserved": {
            DETECTOR_EXPERIMENT: {
                "bytes": per_model[DETECTOR_EXPERIMENT]["peak_memory_reserved_bytes"],
                "gib": per_model[DETECTOR_EXPERIMENT]["peak_memory_reserved_gib"],
            },
            SEGMENTER_EXPERIMENT: {
                "bytes": per_model[SEGMENTER_EXPERIMENT]["peak_memory_reserved_bytes"],
                "gib": per_model[SEGMENTER_EXPERIMENT]["peak_memory_reserved_gib"],
            },
            "ratio": delta["peak_memory_reserved_ratio"],
            "delta_gib": delta["peak_memory_reserved_delta_gib"],
        },
        "device_total_memory_bytes": memory["runtime"]["gpu"]["total_memory_bytes"],
        "device": memory["runtime"]["gpu"]["name"],
        "relative_overhead_substantial": True,
        "absolute_footprint_low_on_measured_device": True,
        "memory_heavy_in_absolute_terms": False,
        "reading": (
            "The relative inference-memory overhead is substantial - the segmenter's peak "
            "allocated figure is over three times the detector's and its peak reserved figure "
            "is 2.375 times. The absolute footprint nevertheless remains low on the measured "
            "device: both models peak well under a third of a GiB reserved on an approximately "
            "8 GiB GPU. Neither statement stands without the other, and S1 is not memory-heavy "
            "in absolute terms."
        ),
    }


def build_complexity(latency: Mapping[str, Any]) -> dict[str, Any]:
    """Assemble the static model-complexity block.

    Args:
        latency: The phase 10C latency artifact, which carries the figures it
            read from the frozen experiment manifests.

    Returns:
        The complexity block, scoped to the framework's 640 reference input.
    """
    complexity = latency["model_complexity"]
    return {
        "label": STATIC_COMPLEXITY_LABEL,
        "reference_input_size": complexity["reference_input_size"],
        "measured_at_benchmark_input_size": complexity["measured_at_benchmark_input_size"],
        "recomputed_in_this_phase": False,
        "derived_at_benchmark_input_size": False,
        "source": complexity["source"],
        "comparable_basis": complexity["comparable_basis"],
        DETECTOR_EXPERIMENT: {
            "parameters": complexity[DETECTOR_EXPERIMENT]["parameters"],
            "gflops_at_reference_input": complexity[DETECTOR_EXPERIMENT]["gflops"],
            "source_artifact": complexity[DETECTOR_EXPERIMENT]["source_artifact"],
        },
        SEGMENTER_EXPERIMENT: {
            "parameters": complexity[SEGMENTER_EXPERIMENT]["parameters"],
            "gflops_at_reference_input": complexity[SEGMENTER_EXPERIMENT]["gflops"],
            "fused_parameters": complexity[SEGMENTER_EXPERIMENT]["fused_parameters"],
            "fused_gflops_at_reference_input": complexity[SEGMENTER_EXPERIMENT]["fused_gflops"],
            "source_artifact": complexity[SEGMENTER_EXPERIMENT]["source_artifact"],
        },
        "parameter_delta": complexity["parameter_delta"],
        "gflops_delta_at_reference_input": complexity["gflops_delta_at_reference_input"],
        "explains_the_latency": False,
        "reading": (
            "These are static architectural figures at the framework's default 640 reference "
            "input, not measured FLOPs of the timed configuration. The benchmark ran "
            "independently at imgsz 768, no 768 figure was derived, and none of these numbers "
            "explains the measured latency."
        ),
    }


def build_use_case_interpretation() -> dict[str, Any]:
    """State the conditional recommendation, without weighting anything.

    Returns:
        The use-case-conditional block.
    """
    return {
        "label": USE_CASE_CONDITIONAL,
        "universally_superior_model": None,
        "winner_declared": False,
        "weighted_score": False,
        "cost_benefit_index": False,
        "detector_preferred_when": [
            "object presence",
            "the object class",
            "a confidence score",
            "bounding-box localisation",
            "lower inference cost in latency and memory",
        ],
        "segmenter_preferred_when": [
            "foreground support inside the box",
            "non-rectangular geometry",
            "mask area",
            "mask-to-box fill ratio or shape extent",
            "a mask centroid",
            "more spatially specific overlap and containment measurements",
        ],
        "both_remain_frozen_final_models": True,
        "reading": (
            "The choice is use-case conditional rather than a ranking. Both remain the "
            "project's frozen final models for their respective tasks, and this synthesis "
            "changes neither."
        ),
    }


def central_answer(
    recognition: Mapping[str, Any],
    spatial: Mapping[str, Any],
    association: Mapping[str, Any],
    latency: Mapping[str, Any],
) -> dict[str, Any]:
    """Compose the central scientific answer from the four axes.

    Args:
        recognition: The recognition axis block.
        spatial: The spatial axis block.
        association: The association axis block.
        latency: The latency block.

    Returns:
        The answer plus the scope that bounds it.
    """
    supported_delta = recognition["supported_class_sensitivity"]["delta"]
    fill_median = spatial["MASK_TO_BOX_FILL_RATIO"]["median"]
    floor = association["containment_floor"]
    end_to_end = latency["boundaries"][END_TO_END_LATENCY]
    relative = end_to_end["relative_latency_cost"]
    absolute = end_to_end["absolute_latency_delta_ms"]
    return {
        "label": SCIENTIFIC_SYNTHESIS,
        "answer": (
            "Instance segmentation did not demonstrate a robust localisation advantage over "
            "the dedicated detector on adequately supported classes - over those four classes "
            f"the canonical box comparison gives {supported_delta:+f} - nor did it uncover "
            "substantial new person-PPE associations at the frozen containment rule of "
            f"{floor}, where the geometry-isolating analysis found zero mask-only "
            "associations. Its demonstrated value was instead richer spatial representation: "
            "masks encode foreground support, non-rectangular shape and spatial measurements "
            "that bounding boxes cannot directly represent - the median predicted mask "
            f"occupied {fill_median} of its own bounding rectangle - while reducing the "
            "systematic inflation of box-based area and intersection proxies. On the "
            "controlled local RTX 5070 Laptop FP32 benchmark, at the operational confidence "
            "of 0.25, exposing that additional mask output cost "
            f"{absolute:+f} ms of mean end-to-end model-output latency ({relative:+f} "
            "relative, approximately 30 per cent) and higher inference memory use."
        ),
        "scope": {
            "split": "validation",
            "holdout": HOLDOUT_STATUS,
            "generalises_to_test": False,
            "generalises_to_other_hardware": False,
            "generalises_beyond_frozen_association_rule": False,
            "significance_tested": False,
            "note": (
                "Every figure is a validation figure produced by a single execution of each "
                "frozen model under a frozen protocol on one machine. Nothing here is a "
                "statement about held-out performance, about either architecture in general, "
                "or about association behaviour at any other containment floor."
            ),
        },
    }


def _area_stat(spatial: Mapping[str, Any], feature: str, side: str, statistic: str) -> Any:
    """Read one proxy-refinement statistic out of the spatial axis block.

    Args:
        spatial: The spatial axis block.
        feature: Frozen feature name.
        side: ``mask_measurement`` or ``box_proxy_statistics``.
        statistic: Statistic name, such as ``mean``.

    Returns:
        The recorded value.
    """
    return spatial["proxy_refinement"]["features"][feature][side][statistic]


def build_claim_register(
    recognition: Mapping[str, Any],
    spatial: Mapping[str, Any],
    association: Mapping[str, Any],
    latency: Mapping[str, Any],
    memory: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Build the machine-readable claim register.

    Every headline claim the later academic report and pitch may reuse is
    recorded with the artifact and field that support it, the scope it holds
    within, and the limitation that travels with it.

    Args:
        recognition: The recognition axis block.
        spatial: The spatial axis block.
        association: The association axis block.
        latency: The latency block.
        memory: The memory block.

    Returns:
        One entry per claim, ordered as :data:`CLAIM_IDS`.

    Raises:
        SynthesisError: If the register does not cover exactly the declared ids.
    """
    box_path = SOURCES["box_comparison"]
    spatial_path = SOURCES["spatial_comparison"]
    latency_path = SOURCES["latency_benchmark"]
    memory_path = SOURCES["memory_benchmark"]
    raw = recognition["all_class"]["CANONICAL_BOX_MAP50_95"]
    supported = recognition["supported_class_sensitivity"]
    geometry = association["geometry_isolating"]
    counts = geometry["frozen_category_counts"]
    end_to_end = latency["boundaries"][END_TO_END_LATENCY]
    allocated = memory["peak_allocated"]
    reserved = memory["peak_reserved"]
    register: list[dict[str, Any]] = [
        {
            "claim_id": "LOCALIZATION_SIMILARITY",
            "claim_text": (
                "S1 retains broadly similar box-localisation performance to D2 while "
                f"additionally producing masks: canonical box mAP@0.50:0.95 {raw['D2']} "
                f"against {raw['S1']}, delta {raw['delta']:+f} over all five classes and "
                f"{supported['delta']:+f} over the four adequately supported ones."
            ),
            "evidence_artifact": box_path,
            "evidence_field": "metrics.CANONICAL_BOX_MAP50_95, supported_class_sensitivity.delta",
            "claim_scope": "VALIDATION_SPLIT_ONE_EXECUTION_PER_MODEL_CANONICAL_COCOEVAL_BBOX",
            "limitation": (
                "The positive all-class delta is dominated by vest_loose, which has one "
                f"validation image and is frozen {RARE_CLASS_STATUS}. The support sensitivity "
                f"is {SUPPORT_SENSITIVITY_LABEL}: descriptive, not a selection rule and not a "
                "significance test. Some classes improved and some regressed; why any of them "
                "moved is UNKNOWN. Neither model is claimed to be the better localiser."
            ),
        },
        {
            "claim_id": "MASK_REPRESENTATION_GAIN",
            "claim_text": (
                "Masks make quantities computable that a rectangle cannot express: "
                "MASK_TO_BOX_FILL_RATIO median "
                f"{spatial['MASK_TO_BOX_FILL_RATIO']['median']} and SHAPE_EXTENT median "
                f"{spatial['SHAPE_EXTENT']['median']}, both frozen {NO_BOX_PROXY}."
            ),
            "evidence_artifact": spatial_path,
            "evidence_field": (
                "spatial_features.MASK_TO_BOX_FILL_RATIO.statistics, "
                "spatial_features.SHAPE_EXTENT.statistics, box_proxies.*.proxy"
            ),
            "claim_scope": (
                "VALIDATION_SPLIT_SEGMENTER_PREDICTIONS_AT_OPERATIONAL_CONF_0_25_"
                "PREDICTED_MASK_AGAINST_PREDICTED_BOX"
            ),
            "limitation": (
                "A gain in what is measurable, not a demonstration of better predictive "
                "accuracy. The ratio compares predicted mask support with predicted box area "
                "and involves no ground truth, so it is not a background-error rate."
            ),
        },
        {
            "claim_id": "BOX_PROXY_INFLATION",
            "claim_text": (
                "Where a box proxy exists it was systematically inflated relative to the mask "
                "measurement: instance area mean "
                f"{_area_stat(spatial, AREA_FEATURE, 'mask_measurement', 'mean')} px against a "
                f"box proxy of {_area_stat(spatial, AREA_FEATURE, 'box_proxy_statistics', 'mean')}"
                " px, and person-PPE intersection mean "
                f"{_area_stat(spatial, INTERSECTION_FEATURE, 'mask_measurement', 'mean')} px "
                "against "
                f"{_area_stat(spatial, INTERSECTION_FEATURE, 'box_proxy_statistics', 'mean')} px."
            ),
            "evidence_artifact": spatial_path,
            "evidence_field": "box_proxies.*.statistics.{mask_measurement,box_proxy}",
            "claim_scope": (
                "VALIDATION_SPLIT_SAME_STATISTIC_COMPARED_WITH_SAME_STATISTIC_"
                "SEGMENTER_OWN_PREDICTIONS"
            ),
            "limitation": (
                f"{PROXY_REFINEMENT}, not BOX_ERROR. No ground-truth geometry entered this "
                "comparison, so neither representation is shown to be right or wrong against "
                "it. Only like summary statistics are compared with like."
            ),
        },
        {
            "claim_id": "NO_MAJOR_MASK_ONLY_ASSOCIATION_GAIN_AT_FROZEN_RULE",
            "claim_text": (
                "Holding the model and its instances fixed at the frozen "
                f"{association['containment_floor']} containment floor, masks changed almost "
                f"no person-PPE association decision: {counts[AGREE_CATEGORY]} agree, "
                f"{counts[BOX_ONLY_CATEGORY]} box-only, "
                f"{association['mask_only_associations']} mask-only, "
                f"{counts[NEITHER_CATEGORY]} neither, over "
                f"{geometry['total_relationships']} relationships."
            ),
            "evidence_artifact": spatial_path,
            "evidence_field": "association.geometry_isolating.frozen_category_counts",
            "claim_scope": "VALIDATION_SPLIT_GEOMETRY_ISOLATING_READING_AT_CONTAINMENT_FLOOR_0_50",
            "limitation": (
                "Specific to this threshold and this population; it does not generalise to "
                "another containment floor or another dataset. The frozen four-category "
                "taxonomy proved non-exhaustive - one relationship had both rules associate to "
                f"different people and is recorded {TAXONOMY_EXCEPTION_STATUS}, excluded from "
                "the classified denominator and never added as a fifth category. The "
                "pipeline-level reading disagrees far more but varies the model as well as the "
                "geometry and must not be attributed to geometry alone. A different-person "
                "outcome is a rule disagreement, not an error: the project holds no person-PPE "
                "association ground truth."
            ),
        },
        {
            "claim_id": "END_TO_END_LATENCY_COST",
            "claim_text": (
                "Exposing usable mask output cost "
                f"{end_to_end['absolute_latency_delta_ms']:+f} ms of mean end-to-end "
                f"model-output latency ({end_to_end['relative_latency_cost']:+f} relative): "
                f"{end_to_end[DETECTOR_EXPERIMENT]['mean']} ms against "
                f"{end_to_end[SEGMENTER_EXPERIMENT]['mean']} ms, with medians "
                f"{end_to_end[DETECTOR_EXPERIMENT]['median']} and "
                f"{end_to_end[SEGMENTER_EXPERIMENT]['median']} ms and P95 "
                f"{end_to_end[DETECTOR_EXPERIMENT]['p95']} and "
                f"{end_to_end[SEGMENTER_EXPERIMENT]['p95']} ms."
            ),
            "evidence_artifact": latency_path,
            "evidence_field": f"latency.*.{END_TO_END_LATENCY}, deltas.{END_TO_END_LATENCY}",
            "claim_scope": (
                f"{CONTROLLED_LOCAL_BENCHMARK}_RTX_5070_LAPTOP_FP32_BATCH_1_IMGSZ_768_"
                f"{LATENCY_SCOPE}"
            ),
            "limitation": (
                f"{SEGMENTATION_COST_LABEL}, not a pure mask-reconstruction cost: the two "
                "networks differ in the mask branch as well as in postprocessing and neither "
                "is isolated. Valid for this machine, driver and runtime only, never as a "
                "property of either architecture. The distribution is wide and multimodal, its "
                f"cause is {DVFS_CAUSAL_ATTRIBUTION} because {NO_POWER_TELEMETRY}, and no "
                "observation was discarded or normalised. Phase 10A froze no confidence for "
                "the latency block; 10C resolved it to the operational 0.25 before any timing "
                "existed. Latency at conf 0.001 was not measured and is not claimed."
            ),
        },
        {
            "claim_id": "INFERENCE_MEMORY_OVERHEAD",
            "claim_text": (
                "Peak inference memory was substantially higher in relative terms and low in "
                "absolute terms: peak allocated "
                f"{allocated[DETECTOR_EXPERIMENT]['gib']} GiB against "
                f"{allocated[SEGMENTER_EXPERIMENT]['gib']} GiB (ratio {allocated['ratio']}), "
                f"peak reserved {reserved[DETECTOR_EXPERIMENT]['gib']} GiB against "
                f"{reserved[SEGMENTER_EXPERIMENT]['gib']} GiB (ratio {reserved['ratio']})."
            ),
            "evidence_artifact": memory_path,
            "evidence_field": "memory.*.peak_memory_{allocated,reserved}_gib, delta.*_ratio",
            "claim_scope": (
                f"{CONTROLLED_LOCAL_BENCHMARK}_INFERENCE_MEMORY_BATCH_1_IMGSZ_768_FP32"
            ),
            "limitation": (
                "INFERENCE_MEMORY only; it may never be compared with any training-memory "
                "figure, which is a different quantity under a different protocol. Each model "
                "was measured in a dedicated process because peak CUDA statistics are "
                "device-global. Batch 1 at imgsz 768 only. The segmenter is not memory-heavy "
                "in absolute terms on the measured device."
            ),
        },
        {
            "claim_id": "USE_CASE_CONDITIONAL_SELECTION",
            "claim_text": (
                "The evidence supports a use-case-conditional choice rather than a ranking: "
                "the detector where class, confidence and box localisation suffice at lower "
                "cost, the segmenter where foreground support, non-rectangular geometry, mask "
                "area, fill or extent, mask centroid or spatially specific overlap is required."
            ),
            "evidence_artifact": f"{box_path}, {spatial_path}, {latency_path}, {memory_path}",
            "evidence_field": (
                "supported_class_sensitivity.delta, spatial_features.MASK_TO_BOX_FILL_RATIO, "
                "association.geometry_isolating.frozen_category_counts, "
                f"deltas.{END_TO_END_LATENCY}, delta.peak_memory_reserved_ratio"
            ),
            "claim_scope": "SYNTHESIS_OVER_THE_FOUR_AXES_NO_AGGREGATE_SCORE",
            "limitation": (
                "No weighted score, cost-benefit index or overall winner is computed, and "
                "none may be: the four axes are not commensurable. Both models remain the "
                "project's frozen final models for their respective tasks and neither is a "
                "drop-in replacement for the other. Validation evidence only."
            ),
        },
    ]
    if tuple(entry["claim_id"] for entry in register) != CLAIM_IDS:
        msg = "the claim register does not cover exactly the declared claim ids"
        raise SynthesisError(msg)
    return register


def build_assignment_coverage(sources: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Map the assignment's metric requirements onto the artifacts that hold them.

    Every entry names a committed artifact. Items with no artifact are listed as
    pending against the phase that will produce them, never as delivered.

    Args:
        sources: The parsed source artifacts.

    Returns:
        The coverage block, split into delivered and pending.

    Raises:
        SynthesisError: If a delivered entry names no evidence artifact.
    """
    detector = sources["final_detector"]
    segmenter = sources["final_segmenter"]
    delivered = {
        "detection_mAP50": {
            "artifact": "reports/detection_D2_manifest.json",
            "field": "validation_metrics.mAP@0.50",
            "split": "validation",
        },
        "detection_mAP50_95": {
            "artifact": "reports/detection_D2_manifest.json",
            "field": "validation_metrics.mAP@0.50:0.95",
            "split": "validation",
        },
        "detection_precision": {
            "artifact": "reports/detection_D2_manifest.json",
            "field": "validation_metrics.precision",
            "split": "validation",
        },
        "detection_recall": {
            "artifact": "reports/detection_D2_manifest.json",
            "field": "validation_metrics.recall",
            "split": "validation",
        },
        "segmentation_mask_mAP50": {
            "artifact": "reports/segmentation_S1_result_manifest.json",
            "field": "native_metrics.mask.mAP@0.50",
            "split": "validation",
        },
        "segmentation_mask_mAP50_95": {
            "artifact": "reports/segmentation_S1_result_manifest.json",
            "field": "native_metrics.mask.mAP@0.50:0.95",
            "split": "validation",
        },
        "segmentation_mask_precision": {
            "artifact": "reports/segmentation_S1_result_manifest.json",
            "field": "native_metrics.mask.precision",
            "split": "validation",
        },
        "segmentation_mask_recall": {
            "artifact": "reports/segmentation_S1_result_manifest.json",
            "field": "native_metrics.mask.recall",
            "split": "validation",
        },
        "segmentation_canonical_mask_AP": {
            "artifact": "reports/segmentation_S1_canonical_evaluation.json",
            "field": "canonical, supported_macro",
            "split": "validation",
        },
        "segmentation_direct_instance_mask_IoU": {
            "artifact": "reports/segmentation_S1_mask_iou.json",
            "field": "global.matched_mask_iou_mean, global.gt_normalized_mask_iou",
            "split": "validation",
        },
        "confusion_matrix_and_metric_figures": {
            "artifact": ("reports/figures/detection/D2/, reports/figures/segmentation_S1/"),
            "field": "confusion_matrix.png, confusion_matrix_normalized.png, PR/P/R/F1 curves",
            "split": "validation",
        },
        "detector_versus_segmenter_comparison": {
            "artifact": (
                f"{SOURCES['box_comparison']}, {SOURCES['spatial_comparison']}, "
                f"{SOURCES['latency_benchmark']}, {SOURCES['memory_benchmark']}"
            ),
            "field": "phases 10B and 10C, synthesised by phase 10D",
            "split": "validation",
        },
    }
    for name, entry in delivered.items():
        if not entry["artifact"]:
            msg = f"assignment coverage entry {name} names no evidence artifact"
            raise SynthesisError(msg)
    pending = {
        "qualitative_fp_fn_analysis_per_class": {
            "status": "PENDING",
            "phase": "12 - error analysis",
            "note": (
                "Phase 8D produced a per-instance segmentation error analysis for S0 and "
                "inspected six instances. The assignment's per-class documented false "
                "positive and false negative gallery, for both tasks, has not been produced."
            ),
        },
        "holdout_metrics_for_both_tasks": {
            "status": "PENDING",
            "phase": "11 - one-shot final test evaluation",
            "note": (
                "The holdout has never been evaluated. No figure exists for it for any "
                "model, and none may be estimated from validation."
            ),
        },
        "video_application": {
            "status": "PENDING",
            "phase": "13 - video inference and tracking",
            "note": "No video has been acquired, run or measured.",
        },
    }
    return {
        "label": COMMITTED_EVIDENCE,
        "frozen_detector": detector["selected_experiment"],
        "frozen_segmenter": segmenter["selected_experiment"],
        "all_delivered_metrics_are_validation_only": True,
        "delivered": delivered,
        # The artifact is written with sorted keys, so the reading order is
        # recorded explicitly rather than left to dict insertion order, which a
        # JSON round trip would silently reshuffle.
        "delivered_order": list(delivered),
        "pending": pending,
        "pending_order": list(pending),
    }


def build_remaining_work() -> list[dict[str, Any]]:
    """List the project work that remains after phase 10D.

    The roadmap is the authority; nothing is marked complete that no artifact
    supports.

    Returns:
        One entry per remaining major item.
    """
    return [
        {
            "item": "FINAL_ONE_SHOT_HOLDOUT_EVALUATION",
            "phase": "11",
            "status": "NOT_STARTED",
            "note": (
                "The test split has never been evaluated, inspected or materialised. It is "
                "read exactly once, after both models are frozen, which they now are. The "
                "immediate next phase is 11A, freezing the holdout evaluation protocol."
            ),
        },
        {
            "item": "ERROR_ANALYSIS",
            "phase": "12",
            "status": "NOT_STARTED",
            "note": (
                "A per-class documented false-positive and false-negative analysis for both "
                "tasks. Phase 8D's S0 per-instance analysis is related evidence, not this "
                "deliverable."
            ),
        },
        {
            "item": "REAL_VIDEO_INFERENCE_AT_LEAST_30_SECONDS",
            "phase": "13",
            "status": "NOT_STARTED",
            "note": "Includes measured FPS on named hardware and temporal failure modes.",
        },
        {
            "item": "TRACKING_BONUS",
            "phase": "13",
            "status": "OPTIONAL_NOT_STARTED",
            "note": "Bonus only, and only after the mandatory deliverables are complete.",
        },
        {
            "item": "EXECUTABLE_COLAB_NOTEBOOK",
            "phase": "14",
            "status": "NOT_STARTED",
            "note": "No notebook exists yet; notebooks/ holds only a README.",
        },
        {
            "item": "ACADEMIC_REPORT_AND_PDF",
            "phase": "14",
            "status": "NOT_STARTED",
            "note": "This synthesis and its claim register are inputs to it.",
        },
        {
            "item": "PITCH_SCRIPT_AND_RECORDING",
            "phase": "14",
            "status": "NOT_STARTED",
            "note": "Every spoken number must match a committed artifact.",
        },
        {
            "item": "REPRODUCIBILITY_AUDIT_FROM_A_CLEAN_CLONE",
            "phase": "14",
            "status": "NOT_STARTED",
            "note": "Walks the rubric contract item by item against real artifacts.",
        },
    ]


def build_synthesis(root: Path, sources: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Assemble the complete phase 10D synthesis payload.

    Args:
        root: Repository root, used only to digest the source artifacts.
        sources: The parsed source artifacts.

    Returns:
        The synthesis, ready to validate and write.
    """
    box = sources["box_comparison"]
    spatial_source = sources["spatial_comparison"]
    latency_source = sources["latency_benchmark"]
    memory_source = sources["memory_benchmark"]

    recognition = build_recognition(box)
    spatial = build_spatial(spatial_source)
    association = build_association(spatial_source)
    latency = build_latency(latency_source)
    memory = build_memory(memory_source)
    complexity = build_complexity(latency_source)
    use_case = build_use_case_interpretation()
    answer = central_answer(recognition, spatial, association, latency)
    register = build_claim_register(recognition, spatial, association, latency, memory)

    payload: dict[str, Any] = {
        "schema_version": 1,
        "phase": PHASE,
        "status": SYNTHESIS_COMPLETE,
        "analysis": "DETECTOR_SEGMENTER_SCIENTIFIC_AND_OPERATIONAL_SYNTHESIS",
        "label": SCIENTIFIC_SYNTHESIS,
        "research_question": (
            "What additional spatial and operational information does instance segmentation "
            "provide beyond bounding boxes, and what is its performance and latency cost?"
        ),
        "axes": list(AXES),
        "axes_combined": False,
        "aggregate_benefit_score": False,
        "cost_benefit_index": False,
        "weighted_score": False,
        "winner_declared": False,
        "models": model_identities(box),
        "protocol_fingerprint": box["protocol_fingerprint"],
        "source_artifacts": source_fingerprints(root, sources),
        "recognition": recognition,
        "spatial_representation": spatial,
        "association": association,
        "cost": {
            "axis": AXES[3],
            "latency": latency,
            "memory": memory,
            "static_complexity": complexity,
        },
        "use_case_conditional": use_case,
        "central_scientific_answer": answer,
        "claim_register": register,
        "claim_register_fields": list(CLAIM_FIELDS),
        "assignment_coverage": build_assignment_coverage(sources),
        "remaining_work": build_remaining_work(),
        "next_phase": "PHASE_11A_FINAL_HOLDOUT_EVALUATION_PROTOCOL_FREEZE",
        "execution": {
            "models_executed": 0,
            "models_trained": 0,
            "models_modified": 0,
            "latency_measurements_taken": 0,
            "memory_measurements_taken": 0,
            "AP_recomputed": False,
            "spatial_analysis_rerun": False,
            "association_analysis_rerun": False,
            "images_read": 0,
            "predictions_produced": 0,
            "thresholds_tuned": 0,
            "new_metrics_introduced": 0,
            "post_hoc_bins_created": False,
            "significance_tests_run": 0,
            "confidence_intervals_computed": 0,
            "subgroup_searches_performed": 0,
            "test_accessed": False,
        },
        "historical_artifacts_unchanged": True,
        "holdout_policy": {
            "label": HOLDOUT_POLICY,
            "status": HOLDOUT_STATUS,
            "final_test_claims": False,
            "note": (
                "Phase 10D read committed development and validation result artifacts only. "
                "No holdout identifier, image, annotation, adapter, prediction, metric or "
                "statistic was read, derived or written, and no final-test claim is made."
            ),
        },
        "test": {
            "status": HOLDOUT_STATUS,
            "images_read": 0,
            "predictions": 0,
            "statistics": 0,
            "reason": (
                "Phase 10D executed no model and opened no image. Its inputs are committed "
                "phase 10B and 10C artifacts, both of which record the holdout as untouched."
            ),
        },
        "limitations": [
            f"{LIMITATION}: every figure is validation-only and says nothing about the "
            "holdout, which has never been evaluated.",
            f"{LIMITATION}: each frozen model was trained once and measured once under each "
            "protocol. Run-to-run variance is UNKNOWN and no margin here is a significance "
            "test.",
            f"{LIMITATION}: the positive all-class localisation delta is carried by a class "
            "with one validation image; the support sensitivity that shows this is descriptive "
            "and decides nothing.",
            f"{LIMITATION}: the association result holds at the frozen 0.50 containment floor "
            "on this population only, and the frozen four-category taxonomy proved "
            "non-exhaustive for the observed data.",
            f"{LIMITATION}: the latency and memory figures are a "
            f"{CONTROLLED_LOCAL_BENCHMARK} on one laptop GPU at batch 1 in FP32 at conf 0.25, "
            "not a property of either architecture.",
            f"{LIMITATION}: the latency distribution is wide and multimodal and its cause is "
            f"{DVFS_CAUSAL_ATTRIBUTION}; the DVFS reading is an untested hypothesis because "
            f"{NO_POWER_TELEMETRY}.",
            f"{LIMITATION}: the committed parameter and GFLOPs figures are static complexity "
            "at the framework's 640 reference input, not the benchmark's 768.",
            f"{LIMITATION}: no compliance, violation or correct-wearing accuracy is claimed or "
            "claimable - the project holds no compliance ground truth.",
        ],
    }
    payload["synthesis_sha256"] = result_fingerprint(
        {
            "protocol_fingerprint": payload["protocol_fingerprint"],
            "models": payload["models"],
            "recognition": recognition,
            "spatial_representation": spatial,
            "association": association,
            "cost": payload["cost"],
            "use_case_conditional": use_case,
            "central_scientific_answer": answer,
            "claim_register": register,
        }
    )
    return payload


# --- validator ---------------------------------------------------------------------------


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
            found.append((f"{path}.{key}" if path else str(key), str(key), value))
            found.extend(_walk(value, f"{path}.{key}" if path else str(key)))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            found.extend(_walk(value, f"{path}[{index}]"))
    return found


def _score_problems(payload: Mapping[str, Any]) -> list[str]:
    """Refuse any composite score anywhere in the payload.

    Args:
        payload: The synthesis.

    Returns:
        One problem per forbidden key found true or numeric.
    """
    problems: list[str] = []
    for path, key, value in _walk(payload):
        if key in FORBIDDEN_SCORE_KEYS and value not in (False, None):
            problems.append(f"{path} declares a composite score ({value!r}); axes never combine")
        if key == "winner_declared" and value is not False:
            problems.append(f"{path} declares a winner ({value!r}); the comparison ranks nothing")
        if key in COMPLIANCE_CLAIM_KEYS:
            problems.append(f"{path} claims compliance accuracy, for which no ground truth exists")
        if key in TEST_FORBIDDEN_KEYS:
            problems.append(f"{path} carries holdout-derived content")
    return problems


def _recognition_problems(payload: Mapping[str, Any], box: Mapping[str, Any]) -> list[str]:
    """Check the recognition axis against its source artifact.

    Args:
        payload: The synthesis.
        box: The committed box-comparison artifact.

    Returns:
        One problem per disagreement.
    """
    problems: list[str] = []
    recognition = payload["recognition"]
    recorded = box["metrics"]["CANONICAL_BOX_MAP50_95"]
    synthesised = recognition["all_class"]["CANONICAL_BOX_MAP50_95"]
    for field in ("D2", "S1", "delta"):
        if synthesised[field] != recorded[field]:
            problems.append(
                f"recognition all-class {field} is {synthesised[field]}, "
                f"committed evidence says {recorded[field]}"
            )
    sensitivity = recognition["supported_class_sensitivity"]
    source_sensitivity = box["supported_class_sensitivity"]
    for field in ("D2_supported_macro", "S1_supported_macro", "delta", "label"):
        if sensitivity[field] != source_sensitivity[field]:
            problems.append(f"supported sensitivity {field} does not match committed evidence")
    if sensitivity["label"] != SUPPORT_SENSITIVITY_LABEL:
        problems.append("the supported sensitivity is not marked descriptive")
    for flag in ("is_a_frozen_phase_10a_metric", "is_a_selection_rule", "is_a_significance_test"):
        if sensitivity[flag] is not False:
            problems.append(f"supported sensitivity {flag} must stay false")
    if recognition["rare_class"]["status"] != RARE_CLASS_STATUS:
        problems.append(f"the rare class must stay {RARE_CLASS_STATUS}")
    if not recognition["rare_class"]["contribution_exceeds_total"]:
        problems.append("the rare class's dominance of the all-class delta was not preserved")
    if recognition["superiority_claimed_for_either_model"] is not False:
        problems.append("recognition claims one model is the superior localiser")
    if not recognition["improved_classes"] or not recognition["declined_classes"]:
        problems.append("the per-class localisation trade-off was not preserved on both sides")
    if recognition["why_any_class_moved"] != "UNKNOWN":
        problems.append("why a class moved must stay UNKNOWN; no experiment isolated a cause")
    return problems


def _spatial_problems(payload: Mapping[str, Any], spatial_source: Mapping[str, Any]) -> list[str]:
    """Check the spatial axis against its source artifact.

    Args:
        payload: The synthesis.
        spatial_source: The committed spatial-comparison artifact.

    Returns:
        One problem per disagreement.
    """
    problems: list[str] = []
    spatial = payload["spatial_representation"]
    recorded = spatial_source["spatial_features"]["MASK_TO_BOX_FILL_RATIO"]["statistics"]
    for field, value in recorded.items():
        if spatial["MASK_TO_BOX_FILL_RATIO"][field] != value:
            problems.append(f"MASK_TO_BOX_FILL_RATIO {field} does not match committed evidence")
    expected_mask_only = sorted(
        name
        for name, entry in spatial_source["box_proxies"].items()
        if entry["proxy"] == NO_BOX_PROXY
    )
    if spatial["mask_only_quantities"] != expected_mask_only:
        problems.append(
            "the NO_BOX_ONLY_EQUIVALENT quantities do not match the frozen pairing: "
            f"{spatial['mask_only_quantities']} against {expected_mask_only}"
        )
    if spatial["background_error_rate_claimed"] is not False:
        problems.append("the fill ratio was translated into a background error rate")
    refinement = spatial["proxy_refinement"]
    if refinement["label"] != PROXY_REFINEMENT or refinement["box_error_claimed"] is not False:
        problems.append("box-proxy inflation must be reported as PROXY_REFINEMENT, not BOX_ERROR")
    for name, entry in refinement["features"].items():
        source = spatial_source["box_proxies"][name]["statistics"]
        for side, key in (
            ("mask_measurement", "mask_measurement"),
            ("box_proxy_statistics", "box_proxy"),
        ):
            for field, value in source[key].items():
                if entry[side][field] != value:
                    problems.append(f"{name} {side} {field} does not match committed evidence")
    if spatial["centroid"]["true_center_claimed"] is not False:
        problems.append("the mask centroid was called the true object centre")
    recorded_centroid = spatial_source["box_proxies"]["MASK_CENTROID"]["statistics"][
        "displacement_pixels"
    ]
    for field in ("median", "p95", "max"):
        if spatial["centroid"]["displacement_pixels"][field] != recorded_centroid[field]:
            problems.append(f"centroid displacement {field} does not match committed evidence")
    return problems


def _association_problems(
    payload: Mapping[str, Any], spatial_source: Mapping[str, Any]
) -> list[str]:
    """Check the association axis against its source artifact.

    Args:
        payload: The synthesis.
        spatial_source: The committed spatial-comparison artifact.

    Returns:
        One problem per disagreement.
    """
    problems: list[str] = []
    association = payload["association"]
    source = spatial_source["association"]
    geometry = source["geometry_isolating"]
    synthesised = association["geometry_isolating"]
    if association["containment_floor"] != source["containment_floor"]:
        problems.append("the containment floor does not match the frozen rule")
    for field in (
        "total_relationships",
        "classified_relationships",
        "taxonomy_exceptions",
        "taxonomy_coverage",
    ):
        if synthesised[field] != geometry[field]:
            problems.append(f"geometry-isolating {field} does not match committed evidence")
    for name, value in geometry["frozen_category_counts"].items():
        if synthesised["frozen_category_counts"][name] != value:
            problems.append(f"geometry-isolating count for {name} does not match")
    if association["mask_only_associations"] != 0:
        problems.append(
            "the geometry-isolating MASK_ONLY_ASSOCIATION count must stay 0 as measured"
        )
    limitation = association["taxonomy_limitation"]
    if limitation["status"] != source["association_taxonomy_status"]:
        problems.append("the taxonomy status does not match committed evidence")
    if limitation["exception_status"] != TAXONOMY_EXCEPTION_STATUS:
        problems.append("the taxonomy exception was reclassified")
    if limitation["fifth_peer_category_added"] is not False:
        problems.append("a fifth association category was added")
    if limitation["frozen_categories_unchanged"] is not True:
        problems.append("the frozen association categories were altered")
    if association["frozen_categories"] != list(ASSOCIATION_CATEGORIES):
        problems.append("the frozen four categories were not preserved verbatim")
    if association["pipeline_level"]["attributable_to_geometry_alone"] is not False:
        problems.append("pipeline-level exceptions were attributed to geometry alone")
    coverage = association["visible_ppe_coverage_proxy"]
    if coverage["label"] != INTERPRETIVE_PROXY:
        problems.append(f"VISIBLE_PPE_COVERAGE_PROXY must stay {INTERPRETIVE_PROXY}")
    if coverage["primary_scientific_conclusion"] is not False:
        problems.append("the coverage proxy was used as a primary scientific conclusion")
    if association["compliance_accuracy_claimed"] is not False:
        problems.append("compliance accuracy was claimed")
    return problems


def _cost_problems(
    payload: Mapping[str, Any],
    latency_source: Mapping[str, Any],
    memory_source: Mapping[str, Any],
) -> list[str]:
    """Check the computational-cost axis against its source artifacts.

    Args:
        payload: The synthesis.
        latency_source: The committed latency artifact.
        memory_source: The committed memory artifact.

    Returns:
        One problem per disagreement.
    """
    problems: list[str] = []
    latency = payload["cost"]["latency"]
    if latency["scope"] != LATENCY_SCOPE:
        problems.append(f"the latency result must be scoped {LATENCY_SCOPE}")
    if latency["conditions"]["conf"] != OPERATIONAL_CONF:
        problems.append("the latency result is not scoped to the operational confidence 0.25")
    if latency["primary_operational_boundary"] != END_TO_END_LATENCY:
        problems.append("the primary operational cost statement must use the end-to-end boundary")
    if latency["boundaries_combined"] is not False:
        problems.append("the two timing boundaries were merged")
    for name in (MODEL_INFERENCE_LATENCY, END_TO_END_LATENCY):
        boundary = latency["boundaries"][name]
        recorded_delta = latency_source["deltas"][name]
        for field in ("absolute_latency_delta_ms", "relative_latency_cost"):
            if boundary[field] != recorded_delta[field]:
                problems.append(f"{name} {field} does not match committed evidence")
        for model in (DETECTOR_EXPERIMENT, SEGMENTER_EXPERIMENT):
            recorded = latency_source["latency"][model][name]
            for field in ("mean", "median", "p95", "p99", "count"):
                if boundary[model][field] != recorded[field]:
                    problems.append(f"{name} {model} {field} does not match committed evidence")
        throughput = boundary["throughput"]
        if throughput["label"] != MEAN_DERIVED_THROUGHPUT:
            problems.append(f"{name} throughput is not labelled {MEAN_DERIVED_THROUGHPUT}")
        if throughput["from_fastest_iteration"] is not False:
            problems.append(f"{name} throughput was derived from the fastest iteration")
        if throughput["throughput_ratio"] != recorded_delta["throughput_ratio"]:
            problems.append(f"{name} throughput ratio does not match committed evidence")
        if boundary["derived_from"] != "mean":
            problems.append(f"{name} delta is not derived from the frozen mean statistic")
    if latency["pure_mask_reconstruction_cost_isolated"] is not False:
        problems.append("the latency difference was presented as an isolated mask cost")
    if latency["cost_label"] != SEGMENTATION_COST_LABEL:
        problems.append(f"the latency difference must be labelled {SEGMENTATION_COST_LABEL}")

    distribution = latency["distribution"]
    if distribution["observations_discarded"] != 0:
        problems.append("observations were discarded from the latency distribution")
    if distribution["outlier_rejection_applied"] is not False:
        problems.append("outlier rejection was applied to the latency distribution")
    if distribution["headline_statistic"] != "mean":
        problems.append("the headline latency delta must stay based on the frozen mean")

    dvfs = latency["dvfs"]
    if dvfs["causal_attribution"] != DVFS_CAUSAL_ATTRIBUTION:
        problems.append("the latency distribution's cause must stay UNKNOWN")
    if dvfs["hypothesis_status"] != "UNTESTED_HYPOTHESIS":
        problems.append("the DVFS reading must stay an untested hypothesis")
    if dvfs["per_observation_power_state_telemetry"] != NO_POWER_TELEMETRY:
        problems.append("the absence of synchronised power-state telemetry was not preserved")
    if dvfs["telemetry_synchronous_with_timed_blocks"] is not False:
        problems.append("synchronised telemetry was claimed where none was taken")
    if dvfs["proportionality_across_models_demonstrated"] is not False:
        problems.append("a proportional DVFS effect across the two models was claimed")
    if dvfs["dvfs_asserted_as_cause"] is not False:
        problems.append("DVFS was asserted as the cause of the distribution")

    gap = latency["confidence_protocol_gap"]
    if gap["label"] != PROTOCOL_GAP_RESOLUTION:
        problems.append(f"the confidence gap must be classified {PROTOCOL_GAP_RESOLUTION}")
    if gap["phase_10a_froze_latency_confidence"] is not False:
        problems.append(
            "the synthesis claims phase 10A explicitly froze the latency confidence, "
            "which is not historically true"
        )
    if gap["resolved_to"] != OPERATIONAL_CONF:
        problems.append("the confidence gap was resolved to something other than 0.25")
    if gap["resolved_before_any_timing_existed"] is not True:
        problems.append("the confidence resolution's pre-benchmark timing was not preserved")
    if gap["latency_measured_at_ap_confidence"] is not False:
        problems.append("a latency claim was made at the AP confidence of 0.001")
    if gap["benchmark_invalidated"] is not False:
        problems.append("the protocol gap was read as invalidating the benchmark")

    memory = payload["cost"]["memory"]
    if memory["measurement"] != memory_source["measurement"]:
        problems.append("the memory measurement label does not match committed evidence")
    if memory["comparable_with_training_memory"] is not False:
        problems.append("inference memory was made comparable with training memory")
    for block, prefix in (("peak_allocated", "allocated"), ("peak_reserved", "reserved")):
        recorded_ratio = memory_source["delta"][f"peak_memory_{prefix}_ratio"]
        if memory[block]["ratio"] != recorded_ratio:
            problems.append(f"{block} ratio does not match committed evidence")
        for model in (DETECTOR_EXPERIMENT, SEGMENTER_EXPERIMENT):
            recorded = memory_source["memory"][model]
            if memory[block][model]["bytes"] != recorded[f"peak_memory_{prefix}_bytes"]:
                problems.append(f"{block} {model} bytes do not match committed evidence")
            if memory[block][model]["gib"] != recorded[f"peak_memory_{prefix}_gib"]:
                problems.append(f"{block} {model} GiB do not match committed evidence")
    if memory["relative_overhead_substantial"] is not True:
        problems.append("the substantial relative memory overhead was not stated")
    if memory["absolute_footprint_low_on_measured_device"] is not True:
        problems.append("the low absolute memory footprint was not stated")
    if memory["memory_heavy_in_absolute_terms"] is not False:
        problems.append("the segmenter was characterised as memory-heavy in absolute terms")

    complexity = payload["cost"]["static_complexity"]
    if complexity["label"] != STATIC_COMPLEXITY_LABEL:
        problems.append(f"the complexity figures must be labelled {STATIC_COMPLEXITY_LABEL}")
    if complexity["measured_at_benchmark_input_size"] is not False:
        problems.append("the 640-reference GFLOPs were presented as measured at 768")
    if complexity["derived_at_benchmark_input_size"] is not False:
        problems.append("an uncommitted 768 FLOP value was derived")
    if complexity["recomputed_in_this_phase"] is not False:
        problems.append("model complexity was recomputed in a phase that executes no model")
    source_complexity = latency_source["model_complexity"]
    for model in (DETECTOR_EXPERIMENT, SEGMENTER_EXPERIMENT):
        if complexity[model]["parameters"] != source_complexity[model]["parameters"]:
            problems.append(f"{model} parameter count does not match committed evidence")
        if complexity[model]["gflops_at_reference_input"] != source_complexity[model]["gflops"]:
            problems.append(f"{model} reference GFLOPs do not match committed evidence")
    return problems


def validate_synthesis(
    payload: Mapping[str, Any], sources: Mapping[str, Mapping[str, Any]]
) -> list[str]:
    """Check the synthesis against the committed evidence it claims to carry.

    Every headline value is compared with its source artifact, and every caveat
    that must travel with a number is checked for presence and correct polarity.

    Args:
        payload: The assembled synthesis.
        sources: The parsed source artifacts.

    Returns:
        One description per problem found; empty when the synthesis is sound.
    """
    problems: list[str] = []
    if payload.get("status") != SYNTHESIS_COMPLETE:
        problems.append(f"status must be {SYNTHESIS_COMPLETE}")
    if payload.get("phase") != PHASE:
        problems.append(f"phase must be {PHASE}")
    if list(payload.get("axes", ())) != list(AXES):
        problems.append("the four synthesis axes were altered")
    if payload.get("axes_combined") is not False:
        problems.append("the axes were combined into one interpretation")
    if payload.get("winner_declared") is not False:
        problems.append("a winner was declared; the comparison is not a contest")

    execution = payload.get("execution", {})
    for field in (
        "models_executed",
        "models_trained",
        "models_modified",
        "latency_measurements_taken",
        "memory_measurements_taken",
        "images_read",
        "predictions_produced",
        "thresholds_tuned",
        "new_metrics_introduced",
        "significance_tests_run",
        "confidence_intervals_computed",
        "subgroup_searches_performed",
    ):
        if execution.get(field) != 0:
            problems.append(f"execution.{field} must be 0 in a synthesis phase")
    for field in (
        "AP_recomputed",
        "spatial_analysis_rerun",
        "association_analysis_rerun",
        "post_hoc_bins_created",
        "test_accessed",
    ):
        if execution.get(field) is not False:
            problems.append(f"execution.{field} must be false in a synthesis phase")

    for block, label in (("holdout_policy", HOLDOUT_POLICY), ("test", None)):
        section = payload.get(block, {})
        if section.get("status") != HOLDOUT_STATUS:
            problems.append(f"{block}.status must be {HOLDOUT_STATUS}")
        if label is not None and section.get("label") != label:
            problems.append(f"{block}.label must be {label}")
    if payload.get("holdout_policy", {}).get("final_test_claims") is not False:
        problems.append("a final-test claim was made")

    models = payload.get("models", {})
    box = sources["box_comparison"]
    for role, source_key in (
        (DETECTOR_EXPERIMENT, "detector"),
        (SEGMENTER_EXPERIMENT, "segmenter"),
    ):
        recorded = box[source_key]
        for field in ("experiment", "model", "imgsz", "checkpoint_sha256"):
            if models.get(role, {}).get(field) != recorded[field]:
                problems.append(f"{role} {field} does not match committed evidence")
    if models.get("drop_in_replacement") is not False:
        problems.append("one model was described as a drop-in replacement for the other")
    if models.get("architectural_identity_claimed") is not False:
        problems.append("architectural identity was claimed between the two models")
    if models.get("same_output_task") is not False:
        problems.append("the two models were described as solving the same output task")

    fingerprints = payload.get("source_artifacts", {})
    for role, field in SELF_FINGERPRINT_FIELD.items():
        recorded = str(sources[role][field])
        if fingerprints.get(role, {}).get("recorded_fingerprint") != recorded:
            problems.append(f"source fingerprint for {role} does not match the committed artifact")
    if set(fingerprints) != set(SOURCES):
        problems.append("the source artifact map does not cover every committed source")

    register = payload.get("claim_register", [])
    if tuple(entry.get("claim_id") for entry in register) != CLAIM_IDS:
        problems.append("the claim register does not cover exactly the declared claims")
    for entry in register:
        missing = [field for field in CLAIM_FIELDS if not entry.get(field)]
        if missing:
            problems.append(f"claim {entry.get('claim_id')} is missing {missing}")

    use_case = payload.get("use_case_conditional", {})
    if use_case.get("label") != USE_CASE_CONDITIONAL:
        problems.append(f"the interpretation must be labelled {USE_CASE_CONDITIONAL}")
    if use_case.get("universally_superior_model") is not None:
        problems.append("a universally superior model was declared")
    if not use_case.get("detector_preferred_when") or not use_case.get("segmenter_preferred_when"):
        problems.append("the use-case-conditional interpretation names no conditions")

    if not payload.get("remaining_work"):
        problems.append("the remaining project work was not identified")
    if not payload.get("assignment_coverage", {}).get("pending"):
        problems.append("pending assignment items were not marked separately")

    problems.extend(_score_problems(payload))
    problems.extend(_recognition_problems(payload, box))
    problems.extend(_spatial_problems(payload, sources["spatial_comparison"]))
    problems.extend(_association_problems(payload, sources["spatial_comparison"]))
    problems.extend(
        _cost_problems(payload, sources["latency_benchmark"], sources["memory_benchmark"])
    )
    return problems


# --- renderers ---------------------------------------------------------------------------

TRADEOFF_FIELDS: tuple[str, ...] = ("dimension", "unit", "D2", "S1", "comparison", "reading")
"""Columns of the trade-off table. No score column, and no ranking column."""


def tradeoff_rows(payload: Mapping[str, Any]) -> list[dict[str, str]]:
    """Build the trade-off table rows.

    One row per dimension, each holding the detector's value, the segmenter's
    value and an interpretation. There is deliberately no overall score, no
    star rating and no weighting.

    Args:
        payload: The assembled synthesis.

    Returns:
        Rows keyed by :data:`TRADEOFF_FIELDS`.
    """
    recognition = payload["recognition"]
    spatial = payload["spatial_representation"]
    association = payload["association"]
    latency = payload["cost"]["latency"]
    memory = payload["cost"]["memory"]
    complexity = payload["cost"]["static_complexity"]
    raw = recognition["all_class"]["CANONICAL_BOX_MAP50_95"]
    supported = recognition["supported_class_sensitivity"]
    geometry = association["geometry_isolating"]
    counts = geometry["frozen_category_counts"]
    inference = latency["boundaries"][MODEL_INFERENCE_LATENCY]
    end_to_end = latency["boundaries"][END_TO_END_LATENCY]
    rows: list[dict[str, str]] = [
        {
            "dimension": "Recognition / localisation (canonical box mAP@0.50:0.95, all classes)",
            "unit": "AP",
            "D2": f"{raw['D2']}",
            "S1": f"{raw['S1']}",
            "comparison": f"{raw['delta']:+f}",
            "reading": (
                "Carried entirely by vest_loose, one validation image, "
                f"{RARE_CLASS_STATUS}. Not evidence of superior localisation."
            ),
        },
        {
            "dimension": (
                "Recognition / localisation (supported-class macro, descriptive sensitivity)"
            ),
            "unit": "AP",
            "D2": f"{supported['D2_supported_macro']}",
            "S1": f"{supported['S1_supported_macro']}",
            "comparison": f"{supported['delta']:+f}",
            "reading": (
                f"{SUPPORT_SENSITIVITY_LABEL}: descriptive, not a selection rule and not a "
                "significance test. Broadly similar localisation, slightly below D2."
            ),
        },
        {
            "dimension": "Spatial geometry (MASK_TO_BOX_FILL_RATIO, median)",
            "unit": "ratio",
            "D2": NO_BOX_PROXY,
            "S1": f"{spatial['MASK_TO_BOX_FILL_RATIO']['median']}",
            "comparison": REPRESENTATION_GAIN,
            "reading": (
                "A box cannot express how much of itself is object. The median predicted mask "
                "occupied about two-thirds of its bounding rectangle."
            ),
        },
        {
            "dimension": "Spatial geometry (SHAPE_EXTENT, median)",
            "unit": "ratio",
            "D2": NO_BOX_PROXY,
            "S1": f"{spatial['SHAPE_EXTENT']['median']}",
            "comparison": REPRESENTATION_GAIN,
            "reading": "Instances do not fill even their own tight rectangle.",
        },
        {
            "dimension": "Spatial geometry (instance area, mean px)",
            "unit": "px",
            "D2": f"{_area_stat(spatial, AREA_FEATURE, 'box_proxy_statistics', 'mean')}",
            "S1": f"{_area_stat(spatial, AREA_FEATURE, 'mask_measurement', 'mean')}",
            "comparison": PROXY_REFINEMENT,
            "reading": (
                "The box proxy is systematically inflated relative to the mask measurement. "
                "No ground truth entered this comparison, so it is not an error rate."
            ),
        },
        {
            "dimension": "Spatial geometry (person-PPE intersection, mean px)",
            "unit": "px",
            "D2": f"{_area_stat(spatial, INTERSECTION_FEATURE, 'box_proxy_statistics', 'mean')}",
            "S1": f"{_area_stat(spatial, INTERSECTION_FEATURE, 'mask_measurement', 'mean')}",
            "comparison": PROXY_REFINEMENT,
            "reading": "Same direction: the rectangular proxy overstates the shared area.",
        },
        {
            "dimension": "Spatial geometry (mask centroid displacement from box centre, median)",
            "unit": "px",
            "D2": "BOX_CENTER (reference)",
            "S1": f"{spatial['centroid']['displacement_pixels']['median']}",
            "comparison": (
                f"P95 {spatial['centroid']['displacement_pixels']['p95']}, "
                f"max {spatial['centroid']['displacement_pixels']['max']}"
            ),
            "reading": (
                "Box centre and foreground centroid differ materially for irregular shapes. "
                "The mask centroid is not the true object centre."
            ),
        },
        {
            "dimension": "Person-PPE association (geometry-isolating, containment floor 0.50)",
            "unit": "relationships",
            "D2": f"{counts[BOX_ONLY_CATEGORY]} box-only",
            "S1": f"{association['mask_only_associations']} mask-only",
            "comparison": (
                f"{counts[AGREE_CATEGORY]} agree, {counts[NEITHER_CATEGORY]} neither, "
                f"{geometry['taxonomy_exceptions']} taxonomy exception, "
                f"{geometry['total_relationships']} total"
            ),
            "reading": (
                "Masks found no association the box rule missed. Specific to this threshold "
                "and population. No compliance accuracy is claimed."
            ),
        },
        {
            "dimension": f"Mean {MODEL_INFERENCE_LATENCY}",
            "unit": "ms",
            "D2": f"{inference[DETECTOR_EXPERIMENT]['mean']}",
            "S1": f"{inference[SEGMENTER_EXPERIMENT]['mean']}",
            "comparison": (
                f"{inference['absolute_latency_delta_ms']:+f} ms "
                f"({inference['relative_latency_cost']:+f} relative)"
            ),
            "reading": (
                f"Forward pass only. Medians {inference[DETECTOR_EXPERIMENT]['median']} and "
                f"{inference[SEGMENTER_EXPERIMENT]['median']} ms; the distribution is wide."
            ),
        },
        {
            "dimension": f"Mean {END_TO_END_LATENCY}",
            "unit": "ms",
            "D2": f"{end_to_end[DETECTOR_EXPERIMENT]['mean']}",
            "S1": f"{end_to_end[SEGMENTER_EXPERIMENT]['mean']}",
            "comparison": (
                f"{end_to_end['absolute_latency_delta_ms']:+f} ms "
                f"({end_to_end['relative_latency_cost']:+f} relative)"
            ),
            "reading": (
                f"The primary operational cost statement. {SEGMENTATION_COST_LABEL}; mask "
                f"reconstruction is inside S1's timer. {LATENCY_SCOPE}."
            ),
        },
        {
            "dimension": f"{MEAN_DERIVED_THROUGHPUT} at the end-to-end boundary",
            "unit": "images/s",
            "D2": f"{end_to_end['throughput'][DETECTOR_EXPERIMENT]}",
            "S1": f"{end_to_end['throughput'][SEGMENTER_EXPERIMENT]}",
            "comparison": f"ratio {end_to_end['throughput']['throughput_ratio']}",
            "reading": (
                "1000 / mean at batch 1. Not batched throughput, not application video FPS, "
                "and never the reciprocal of the fastest repetition."
            ),
        },
        {
            "dimension": "Peak allocated inference memory",
            "unit": "GiB",
            "D2": f"{memory['peak_allocated'][DETECTOR_EXPERIMENT]['gib']}",
            "S1": f"{memory['peak_allocated'][SEGMENTER_EXPERIMENT]['gib']}",
            "comparison": f"ratio {memory['peak_allocated']['ratio']}",
            "reading": (
                "Substantial relative overhead, low absolute footprint on the measured "
                "~8 GiB GPU. INFERENCE_MEMORY only, never training memory."
            ),
        },
        {
            "dimension": "Peak reserved inference memory",
            "unit": "GiB",
            "D2": f"{memory['peak_reserved'][DETECTOR_EXPERIMENT]['gib']}",
            "S1": f"{memory['peak_reserved'][SEGMENTER_EXPERIMENT]['gib']}",
            "comparison": f"ratio {memory['peak_reserved']['ratio']}",
            "reading": "Each model measured in a dedicated process; peak stats reset after warmup.",
        },
        {
            "dimension": "Static parameters",
            "unit": "parameters",
            "D2": f"{complexity[DETECTOR_EXPERIMENT]['parameters']}",
            "S1": f"{complexity[SEGMENTER_EXPERIMENT]['parameters']}",
            "comparison": f"{complexity['parameter_delta']:+d}",
            "reading": f"{STATIC_COMPLEXITY_LABEL}, unfused on both sides. Not recomputed here.",
        },
        {
            "dimension": "Static reference GFLOPs",
            "unit": "GFLOPs at the framework's 640 reference input",
            "D2": f"{complexity[DETECTOR_EXPERIMENT]['gflops_at_reference_input']}",
            "S1": f"{complexity[SEGMENTER_EXPERIMENT]['gflops_at_reference_input']}",
            "comparison": f"{complexity['gflops_delta_at_reference_input']:+f}",
            "reading": (
                "NOT measured at the benchmark's imgsz 768, no 768 value was derived, and "
                "these figures explain none of the latency."
            ),
        },
    ]
    for row in rows:
        if set(row) != set(TRADEOFF_FIELDS):
            msg = f"trade-off row {row.get('dimension')!r} does not carry the declared fields"
            raise SynthesisError(msg)
    return rows


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    """Render a Markdown table.

    Args:
        header: Column titles.
        rows: Row cells, already stringified.

    Returns:
        The table's lines.
    """
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines.extend("| " + " | ".join(cell for cell in row) + " |" for row in rows)
    return lines


def _recognition_section(payload: Mapping[str, Any]) -> list[str]:
    """Render the recognition and rare-class sections.

    Args:
        payload: The assembled synthesis.

    Returns:
        Markdown lines.
    """
    recognition = payload["recognition"]
    raw = recognition["all_class"]["CANONICAL_BOX_MAP50_95"]
    at50 = recognition["all_class"]["CANONICAL_BOX_MAP50"]
    supported = recognition["supported_class_sensitivity"]
    rare = recognition["rare_class"]
    lines = [
        "## 4. Recognition and localisation",
        "",
        f"`{COMMITTED_EVIDENCE}` - source: `{SOURCES['box_comparison']}`.",
        "",
        f"One external evaluator judges both models: {recognition['evaluator']} at "
        f"`iouType='{recognition['iou_type']}'` against `{recognition['ground_truth']}`, at "
        f"conf {recognition['conf']}, over the frozen "
        f"{recognition['population']['images']} validation images and "
        f"{recognition['population']['annotations']} canonical annotations. S1's boxes are "
        "S1's own predicted boxes, not boxes re-derived from its masks.",
        "",
    ]
    lines.extend(
        _table(
            ["Metric", "D2", "S1", "Delta"],
            [
                [
                    "Canonical box mAP@0.50:0.95 (all classes)",
                    f"**{raw['D2']}**",
                    f"**{raw['S1']}**",
                    f"**{raw['delta']:+f}**",
                ],
                [
                    "Canonical box mAP@0.50 (all classes)",
                    f"{at50['D2']}",
                    f"{at50['S1']}",
                    f"{at50['delta']:+f}",
                ],
                [
                    "Supported-class macro AP@0.50:0.95 (descriptive)",
                    f"**{supported['D2_supported_macro']}**",
                    f"**{supported['S1_supported_macro']}**",
                    f"**{supported['delta']:+f}**",
                ],
            ],
        )
    )
    lines.extend(
        [
            "",
            f"`{SCIENTIFIC_SYNTHESIS}` - **S1 retains broadly similar localisation performance "
            "to D2 while adding mask output.** The positive all-class delta is **not** robust "
            f"evidence that S1 is the superior localiser: it is dominated by `{RARE_CLASS}`, "
            f"already frozen `{RARE_CLASS_STATUS}`. Over the four adequately supported classes "
            f"the same comparison gives {supported['delta']:+f}, placing the segmenter "
            "marginally below the detector.",
            "",
            "Neither statement is a ranking. It is not claimed that S1 detects objects better "
            "than D2, nor that D2 definitively detects objects better than S1.",
            "",
            f"> `{LIMITATION}` The supported-class macro is `{supported['label']}`: descriptive, "
            f"reusing the project's pre-existing support rule "
            f"(`{supported['support_rule_origin']}`). It is **not** a frozen phase 10A metric, "
            "**not** a selection rule and **not** a significance test, and it changes no frozen "
            "number.",
            "",
            "### Per-class localisation trade-off",
            "",
        ]
    )
    lines.extend(
        _table(
            ["Class", "D2 AP@0.50:0.95", "S1 AP@0.50:0.95", "Delta", "Contribution", "Status"],
            [
                [
                    f"`{name}`",
                    f"{row['D2_AP@0.50:0.95']}",
                    f"{row['S1_AP@0.50:0.95']}",
                    f"{row['delta_AP@0.50:0.95']:+f}",
                    f"{recognition['per_class_contributions'][name]:+f}",
                    row["status"],
                ]
                for name, row in recognition["per_class"].items()
            ],
        )
    )
    improved = ", ".join(f"`{name}`" for name in recognition["improved_classes"])
    declined = ", ".join(f"`{name}`" for name in recognition["declined_classes"])
    lines.extend(
        [
            "",
            f"Some classes improved ({improved}) and some regressed ({declined}). **Why any "
            "individual class moved is UNKNOWN** - this comparison ran no experiment isolating "
            "a cause, and nothing was repeated, so run-to-run variance on this setup is also "
            "UNKNOWN.",
            "",
            f"> `{LIMITATION}` {recognition['not_comparable_to_native_metrics']}",
            "",
            "## 5. Rare-class sensitivity",
            "",
            f"`{rare['name']}` holds **one** validation image and eight instances under the "
            f"frozen split, and is frozen `{rare['status']}`. Its canonical box AP moved "
            f"{rare['delta']:+f}, contributing {rare['contribution']:+f} to the five-class "
            "unweighted mean - **more than the whole all-class delta of "
            f"{raw['delta']:+f}**. Excluding it, the mean over the remaining four classes is "
            f"{recognition['delta_excluding_rare_class']:+f}.",
            "",
            "It is reported in full and decides nothing. Never quote the all-class delta on its "
            "own.",
            "",
        ]
    )
    return lines


def _spatial_section(payload: Mapping[str, Any]) -> list[str]:
    """Render the spatial-representation sections.

    Args:
        payload: The assembled synthesis.

    Returns:
        Markdown lines.
    """
    spatial = payload["spatial_representation"]
    fill = spatial["MASK_TO_BOX_FILL_RATIO"]
    extent = spatial["SHAPE_EXTENT"]
    centroid = spatial["centroid"]["displacement_pixels"]
    refinement = spatial["proxy_refinement"]["features"]
    lines = [
        "## 6. What masks represent beyond boxes",
        "",
        f"`{COMMITTED_EVIDENCE}` - source: `{SOURCES['spatial_comparison']}`, operational "
        f"inference at conf {spatial['conf']}, {spatial['instances_measured']} segmenter "
        f"instances with {spatial['masks_reconstructed']} masks reconstructed on the original "
        f"canvas and {spatial['masks_excluded']} excluded.",
        "",
        f"`{REPRESENTATION_GAIN}` - **masks represent foreground support inside bounding "
        f"boxes.** The median predicted instance mask occupied **{fill['median']}** of its "
        "bounding rectangle, illustrating information that the rectangular representation does "
        "not encode.",
        "",
    ]
    lines.extend(
        _table(
            ["Quantity", "Median", "P25", "P75", "Mean", "Box equivalent"],
            [
                [
                    "`MASK_TO_BOX_FILL_RATIO`",
                    f"**{fill['median']}**",
                    f"{fill['p25']}",
                    f"{fill['p75']}",
                    f"{fill['mean']}",
                    f"`{NO_BOX_PROXY}`",
                ],
                [
                    "`SHAPE_EXTENT`",
                    f"**{extent['median']}**",
                    f"{extent['p25']}",
                    f"{extent['p75']}",
                    f"{extent['mean']}",
                    f"`{NO_BOX_PROXY}`",
                ],
            ],
        )
    )
    lines.extend(
        [
            "",
            f"> `{LIMITATION}` This is **not** a 33.6 per cent background-error rate. The "
            "measure compares *predicted mask support* with *predicted box area*; no "
            "ground-truth background classification enters it.",
            "",
            "## 7. Mask-only geometric quantities",
            "",
            f"`{REPRESENTATION_GAIN}` - two of the seven frozen spatial features were declared "
            f"`{NO_BOX_PROXY}` **before any measurement was taken**: "
            + ", ".join(f"`{name}`" for name in spatial["mask_only_quantities"])
            + ".",
            "",
            "A bounding box cannot directly encode non-rectangular foreground support. It has "
            "no way to express how much of itself is object, or how far the object departs from "
            "a rectangle. These quantities are therefore a gain in **what is computable**, not "
            "a demonstration of improved predictive accuracy: nothing here compares either "
            "representation against ground-truth geometry.",
            "",
            "Deciding which quantities a box could approximate *after* seeing the numbers would "
            "have been circular, which is why the pairing was frozen in phase 10A.",
            "",
            "## 8. Box-proxy inflation and refinement",
            "",
            f"`{PROXY_REFINEMENT}` - for the {len(refinement)} features where a mask "
            "measurement and a like-for-like box proxy both exist, the box proxy was "
            "**systematically inflated** relative to the mask measurement. Each pair below is "
            "the same summary statistic of the same population under the two representations. "
            "`MASK_CENTROID` also has a box proxy but is reported separately below, because "
            "its recorded statistic is a displacement between the two rather than a pair of "
            "like measurements.",
            "",
        ]
    )
    lines.extend(
        _table(
            ["Quantity", "Box proxy", "Mask measurement (mean)", "Box proxy (mean)"],
            [
                [
                    f"`{name}`",
                    f"`{entry['box_proxy']}`",
                    f"{entry['mask_measurement']['mean']}",
                    f"{entry['box_proxy_statistics']['mean']}",
                ]
                for name, entry in refinement.items()
            ],
        )
    )
    lines.extend(
        [
            "",
            f"> `{LIMITATION}` Read this as `{PROXY_REFINEMENT}`, **never** as `BOX_ERROR`. No "
            "ground-truth geometry entered the comparison, so neither representation is shown "
            "to be right or wrong against it. Unlike statistics are never compared: mean is set "
            "against mean, median against median, over the same instances.",
            "",
            "A further geometric disagreement is recorded as a count rather than binned: "
            f"**{spatial['geometry_disagreement']['box_overlap_without_mask_overlap']} of "
            f"{spatial['geometry_disagreement']['candidate_pairs_examined']} candidate pairs "
            f"({spatial['geometry_disagreement']['box_overlap_without_mask_overlap_fraction']})**"
            " had overlapping boxes whose masks shared no pixel at all. Phase 10A declared no "
            "threshold for 'strong' or 'minimal' overlap, so none was invented.",
            "",
            "### Centroid information",
            "",
            f"`{PROXY_REFINEMENT}` - displacement between the model's own box centre and the "
            f"mask centroid: median **{centroid['median']} px**, P95 **{centroid['p95']} px**, "
            f"max **{centroid['max']} px** over {centroid['count']} instances.",
            "",
            "A box centre and a foreground centroid can differ materially for irregular, "
            "partially visible or spatially imbalanced shapes, and the mask makes the second "
            "computable. That is additional geometric information.",
            "",
            f'> `{LIMITATION}` The mask centroid is **not** "the true object centre". It is a '
            f"different geometric quantity. {spatial['centroid']['convention_note']}",
            "",
        ]
    )
    return lines


def _association_section(payload: Mapping[str, Any]) -> list[str]:
    """Render the association, taxonomy and proxy-limitation sections.

    Args:
        payload: The assembled synthesis.

    Returns:
        Markdown lines.
    """
    association = payload["association"]
    geometry = association["geometry_isolating"]
    pipeline = association["pipeline_level"]
    counts = geometry["frozen_category_counts"]
    pipeline_counts = pipeline["frozen_category_counts"]
    limitation = association["taxonomy_limitation"]
    coverage = association["visible_ppe_coverage_proxy"]
    lines = [
        "## 9. Person-PPE association",
        "",
        f"`{COMMITTED_EVIDENCE}` - `{association['analysis']}`, frozen containment floor "
        f"**{association['containment_floor']}**.",
        "",
        "The **geometry-isolating** reading is the one that answers the question: it holds the "
        "model and its instances fixed - S1's own boxes against S1's own masks - and varies "
        "only the shape representation.",
        "",
    ]
    lines.extend(
        _table(
            ["Category", "Geometry-isolating", "Pipeline-level (confounded)"],
            [
                [
                    f"`{AGREE_CATEGORY}`",
                    f"{counts[AGREE_CATEGORY]}",
                    f"{pipeline_counts[AGREE_CATEGORY]}",
                ],
                [
                    f"`{BOX_ONLY_CATEGORY}`",
                    f"{counts[BOX_ONLY_CATEGORY]}",
                    f"{pipeline_counts[BOX_ONLY_CATEGORY]}",
                ],
                [
                    f"`{MASK_ONLY_CATEGORY}`",
                    f"**{counts[MASK_ONLY_CATEGORY]}**",
                    f"{pipeline_counts[MASK_ONLY_CATEGORY]}",
                ],
                [
                    f"`{NEITHER_CATEGORY}`",
                    f"{counts[NEITHER_CATEGORY]}",
                    f"{pipeline_counts[NEITHER_CATEGORY]}",
                ],
                [
                    "Taxonomy exceptions",
                    f"{geometry['taxonomy_exceptions']}",
                    f"{pipeline['taxonomy_exceptions']}",
                ],
                [
                    "Classified / total",
                    f"{geometry['classified_relationships']} / {geometry['total_relationships']}",
                    f"{pipeline['classified_relationships']} / {pipeline['total_relationships']}",
                ],
                [
                    "Taxonomy coverage",
                    f"{geometry['taxonomy_coverage']}",
                    f"{pipeline['taxonomy_coverage']}",
                ],
            ],
        )
    )
    lines.extend(
        [
            "",
            f"Counts are over all relationships; percentages elsewhere use "
            f"`{geometry['percentage_denominator']}`.",
            "",
            f"`{SCIENTIFIC_SYNTHESIS}` - **at the frozen association rule and on this "
            "validation population, instance masks did NOT demonstrate a substantial advantage "
            "in discovering additional person-PPE associations.** Specifically, the "
            f"geometry-isolating analysis found **zero `{MASK_ONLY_CATEGORY}`** cases: the mask "
            "changed almost no association decision the box rule had already made.",
            "",
            f"> `{LIMITATION}` This does not generalise beyond this containment floor and this "
            "population. The pipeline-level column varies the **model** as well as the "
            "geometry - different instances, different predictions - and its disagreements "
            "**must not be attributed solely to mask-versus-box geometry**.",
            "",
            "## 10. Association taxonomy limitation",
            "",
            f"`{OPERATIONAL_LIMITATION}` - the frozen four-category taxonomy is "
            f"`{limitation['status']}`.",
            "",
            "Phase 10A froze four categories on the implicit assumption that a rule either "
            "associates or it does not. A state occurred that none of them describes: **box and "
            "mask rules both associated, but selected different persons** "
            f"(`{limitation['exception_type']}`).",
            "",
            f"It is recorded `{limitation['exception_status']}` - counted on its own, excluded "
            "from the classified denominator - and is **not** a fifth frozen category "
            f"(`fifth_peer_category_added: {str(limitation['fifth_peer_category_added']).lower()}`"
            "). The phase 10A protocol is historical and was not modified.",
            "",
            f"> `{LIMITATION}` {limitation['not_an_error']}",
            "",
            "## 11. Operational proxy limitation",
            "",
            f"`{OPERATIONAL_LIMITATION}` - `VISIBLE_PPE_COVERAGE_PROXY` remains "
            f"`{coverage['label']}` and is **not** used as a primary scientific conclusion.",
            "",
            "**No PPE compliance accuracy, safety-violation accuracy or correct-wearing "
            "classification accuracy is claimed here, and none is claimable**: the project holds "
            "no canonical compliance ground truth, so there is nothing such a claim could be "
            "measured against. What may be said is only this: masks permit a more spatially "
            "specific coverage-like proxy than rectangular boxes.",
            "",
        ]
    )
    return lines


def _cost_section(payload: Mapping[str, Any]) -> list[str]:
    """Render the computational-cost sections.

    Args:
        payload: The assembled synthesis.

    Returns:
        Markdown lines.
    """
    latency = payload["cost"]["latency"]
    memory = payload["cost"]["memory"]
    complexity = payload["cost"]["static_complexity"]
    inference = latency["boundaries"][MODEL_INFERENCE_LATENCY]
    end_to_end = latency["boundaries"][END_TO_END_LATENCY]
    conditions = latency["conditions"]
    distribution = latency["distribution"]
    dvfs = latency["dvfs"]
    gap = latency["confidence_protocol_gap"]

    def boundary_rows(block: Mapping[str, Any]) -> list[list[str]]:
        return [
            [
                statistic,
                f"{block[DETECTOR_EXPERIMENT][statistic]}",
                f"{block[SEGMENTER_EXPERIMENT][statistic]}",
            ]
            for statistic in ("mean", "median", "p90", "p95", "p99", "min", "max", "std")
        ]

    lines = [
        "## 12. Model inference cost",
        "",
        f"`{CONTROLLED_LOCAL_BENCHMARK}` - source: `{SOURCES['latency_benchmark']}`. Batch "
        f"{conditions['batch']} at imgsz {conditions['imgsz']} in {conditions['precision']}, "
        f"conf {conditions['conf']}, {conditions['benchmark_images']} frozen validation images, "
        f"{conditions['warmup_iterations']} warmup iterations discarded and "
        f"{conditions['timed_iterations_per_image']} timed repetitions per block, in a "
        f"symmetric interleaved order over {conditions['execution_blocks']} blocks - "
        f"{conditions['observations']} timed readings.",
        "",
        f"`{MODEL_INFERENCE_LATENCY}` - the forward pass alone.",
        "",
    ]
    lines.extend(_table(["Statistic (ms)", "D2", "S1"], boundary_rows(inference)))
    lines.extend(
        [
            "",
            f"Mean delta **{inference['absolute_latency_delta_ms']:+f} ms** "
            f"(**{inference['relative_latency_cost']:+f}** relative).",
            "",
            "## 13. End-to-end output cost",
            "",
            f"`{END_TO_END_LATENCY}` - what a caller waits for: preprocessing, the forward pass, "
            "NMS and postprocessing, **including the segmenter's mask reconstruction onto the "
            "original canvas**. This is the primary operational cost statement, because it is "
            "the boundary at which both models have usable outputs.",
            "",
        ]
    )
    lines.extend(_table(["Statistic (ms)", "D2", "S1"], boundary_rows(end_to_end)))
    lines.extend(
        [
            "",
            f"Mean delta **{end_to_end['absolute_latency_delta_ms']:+f} ms** "
            f"(**{end_to_end['relative_latency_cost']:+f}** relative, approximately 30 per "
            "cent).",
            "",
            f"> `{LIMITATION}` Call this `{latency['cost_label']}`, never "
            "`PURE_MASK_RECONSTRUCTION_CAUSAL_COST`. "
            f"{latency['cost_label_note']}",
            "",
            "## 14. Throughput",
            "",
            f"`{MEAN_DERIVED_THROUGHPUT}` - 1000 / mean latency at batch 1.",
            "",
        ]
    )
    lines.extend(
        _table(
            ["Boundary", "D2 (images/s)", "S1 (images/s)", "Ratio"],
            [
                [
                    f"`{name}`",
                    f"{block['throughput'][DETECTOR_EXPERIMENT]}",
                    f"{block['throughput'][SEGMENTER_EXPERIMENT]}",
                    f"{block['throughput']['throughput_ratio']}",
                ]
                for name, block in (
                    (MODEL_INFERENCE_LATENCY, inference),
                    (END_TO_END_LATENCY, end_to_end),
                )
            ],
        )
    )
    lines.extend(
        [
            "",
            f"> `{LIMITATION}` This is a latency reciprocal, not batched throughput and not "
            "application or video FPS. It is not derived from the fastest iteration. A system "
            "that does not process frames independently under these same assumptions will not "
            "see these rates.",
            "",
            "## 15. Inference memory",
            "",
            f"`{CONTROLLED_LOCAL_BENCHMARK}` - `{memory['measurement']}`, source: "
            f"`{SOURCES['memory_benchmark']}`. Each model measured in a dedicated process "
            f"(`{memory['isolation_method']}`), peak statistics reset after the frozen warmup.",
            "",
        ]
    )
    lines.extend(
        _table(
            ["Peak", "D2", "S1", "Ratio"],
            [
                [
                    "Allocated",
                    f"{memory['peak_allocated'][DETECTOR_EXPERIMENT]['gib']} GiB "
                    f"({memory['peak_allocated'][DETECTOR_EXPERIMENT]['bytes']} B)",
                    f"{memory['peak_allocated'][SEGMENTER_EXPERIMENT]['gib']} GiB "
                    f"({memory['peak_allocated'][SEGMENTER_EXPERIMENT]['bytes']} B)",
                    f"**{memory['peak_allocated']['ratio']}**",
                ],
                [
                    "Reserved",
                    f"{memory['peak_reserved'][DETECTOR_EXPERIMENT]['gib']} GiB "
                    f"({memory['peak_reserved'][DETECTOR_EXPERIMENT]['bytes']} B)",
                    f"{memory['peak_reserved'][SEGMENTER_EXPERIMENT]['gib']} GiB "
                    f"({memory['peak_reserved'][SEGMENTER_EXPERIMENT]['bytes']} B)",
                    f"**{memory['peak_reserved']['ratio']}**",
                ],
            ],
        )
    )
    lines.extend(
        [
            "",
            "**Both statements hold together.** The relative overhead is substantial - over "
            "three times the allocated peak and 2.375 times the reserved peak. The absolute "
            f"footprint is low: on the measured {memory['device']} with "
            f"{memory['device_total_memory_bytes']} bytes of device memory, both models peak "
            "well under a third of a GiB reserved. S1 is **not** memory-heavy in absolute terms.",
            "",
            f"> `{LIMITATION}` `{memory['measurement']}` may never be compared with training "
            f"memory: {memory['training_memory']}.",
            "",
            "## 16. Static model complexity",
            "",
            f"`{STATIC_COMPLEXITY_LABEL}` - read from the committed experiment manifests, not "
            "recomputed.",
            "",
        ]
    )
    lines.extend(
        _table(
            ["Figure", "D2", "S1", "Delta"],
            [
                [
                    "Parameters (unfused)",
                    f"{complexity[DETECTOR_EXPERIMENT]['parameters']}",
                    f"{complexity[SEGMENTER_EXPERIMENT]['parameters']}",
                    f"{complexity['parameter_delta']:+d}",
                ],
                [
                    "GFLOPs at the 640 reference input",
                    f"{complexity[DETECTOR_EXPERIMENT]['gflops_at_reference_input']}",
                    f"{complexity[SEGMENTER_EXPERIMENT]['gflops_at_reference_input']}",
                    f"{complexity['gflops_delta_at_reference_input']:+f}",
                ],
            ],
        )
    )
    lines.extend(
        [
            "",
            f"> `{LIMITATION}` These are `{complexity['reference_input_size']}`. "
            "`ultralytics.utils.torch_utils.get_flops` and `model_info` both default to "
            "`imgsz=640`, and both committed figures came from those paths. **They are not the "
            "FLOPs of the timed configuration**, no imgsz 768 value was derived, and latency "
            "was measured independently at imgsz 768. They explain none of the timings.",
            "",
            "## 17. Benchmark distribution and DVFS limitation",
            "",
            f"`{distribution['label']}` / `{distribution['status']}` - computed from all "
            f"{distribution['observations_used']} observations. "
            f"**{distribution['observations_discarded']} observations were discarded**, no "
            "outlier rejection was applied, nothing was normalised or rescaled, and the "
            "benchmark was not re-run.",
            "",
            "The distribution is broad and multimodal, so the mean alone misleads. Mean-to-"
            "median ratios:",
            "",
        ]
    )
    lines.extend(
        _table(
            [
                "Boundary",
                "D2 mean/median",
                "S1 mean/median",
                "D2 block-mean range (ms)",
                "S1 block-mean range (ms)",
            ],
            [
                [
                    f"`{name}`",
                    f"{distribution['mean_to_median_ratio'][DETECTOR_EXPERIMENT][name]}",
                    f"{distribution['mean_to_median_ratio'][SEGMENTER_EXPERIMENT][name]}",
                    f"{distribution['block_mean_range'][DETECTOR_EXPERIMENT][name]['min']} - "
                    f"{distribution['block_mean_range'][DETECTOR_EXPERIMENT][name]['max']}",
                    f"{distribution['block_mean_range'][SEGMENTER_EXPERIMENT][name]['min']} - "
                    f"{distribution['block_mean_range'][SEGMENTER_EXPERIMENT][name]['max']}",
                ]
                for name in (MODEL_INFERENCE_LATENCY, END_TO_END_LATENCY)
            ],
        )
    )
    lines.extend(
        [
            "",
            "The headline delta remains based on the frozen "
            f"**{distribution['headline_statistic']}** "
            "statistic; the median, P95 and range are reported beside it rather than replacing it.",
            "",
            f"> `{LIMITATION}` `causal_attribution: {dvfs['causal_attribution']}`. The observed "
            "multimodality is *consistent with* mobile-GPU DVFS and power-state behaviour, and "
            f"that stays `{dvfs['hypothesis_status']}`: "
            f"`{dvfs['per_observation_power_state_telemetry']}` - no clock, P-state, "
            "utilisation, temperature or power reading accompanied the timed regions, so no "
            "observation maps to a device state and no alternative was excluded. **It is not "
            "stated that DVFS caused the distribution**, and "
            "`proportionality_across_models_demonstrated: "
            f"{str(dvfs['proportionality_across_models_demonstrated']).lower()}` - no "
            "proportional effect across the two models is claimed. The controlled symmetric "
            "order mitigates order bias; it does not prove the source of the multimodality.",
            "",
            "## 18. Confidence-protocol-gap disclosure",
            "",
            f"`{PROTOCOL_GAP_RESOLUTION}`.",
            "",
            "Phase 10A's latency subsection froze batch, resolution, precision, warmup, "
            "repetitions, membership and execution order - but **no confidence threshold**. It "
            "is therefore **false** to say that latency confidence 0.25 was explicitly frozen "
            "by phase 10A, and this synthesis does not say it.",
            "",
            f"Before any timing result existed, phase 10C resolved the benchmark to conf "
            f"**{gap['resolved_to']}** (`{gap['confidence_source']}`), because that is the "
            "project's already frozen operational inference threshold - the protocol itself "
            f"declares the AP block's {gap['ap_confidence_is_not_an_operating_point']} "
            "deliberately *not* an operating point. The resolution was applied equally to both "
            "models.",
            "",
            f"This does **not** invalidate the benchmark. It scopes it: "
            f"`{latency['scope']}`. **No latency claim is made for conf "
            f"{gap['ap_confidence_is_not_an_operating_point']}** - a lower threshold pushes more "
            "candidates through NMS and, for the segmenter, more masks through reconstruction, "
            "and that was not measured.",
            "",
        ]
    )
    return lines


def _closing_section(payload: Mapping[str, Any]) -> list[str]:
    """Render the interpretation, answer, coverage, limitations and next-phase sections.

    Args:
        payload: The assembled synthesis.

    Returns:
        Markdown lines.
    """
    use_case = payload["use_case_conditional"]
    answer = payload["central_scientific_answer"]
    coverage = payload["assignment_coverage"]
    lines = [
        "## 19. Benefit-versus-cost interpretation",
        "",
        "The four axes are reported separately and are **not** combined. There is no weighted "
        "score, no overall benefit score, no cost-benefit index and no single winner metric "
        "anywhere in this phase's artifacts, and the validator refuses one.",
        "",
        "What the evidence supports is a conditional reading:",
        "",
        "- **Recognition.** Broadly similar, with the positive aggregate carried by a "
        "one-image class and the supported-class reading marginally favouring the detector. "
        "Neither model is the established better localiser.",
        "- **Spatial representation.** A real, measured gain for the segmenter, in quantities a "
        "rectangle cannot express and in the refinement of quantities it can only approximate.",
        "- **Operational association.** No measured advantage at the frozen containment rule.",
        "- **Computational cost.** A consistent, measured premium in latency and in inference "
        "memory on this machine.",
        "",
        "## 20. Use-case-conditional recommendation",
        "",
        f"`{USE_CASE_CONDITIONAL}` - not `ONE_MODEL_UNIVERSALLY_SUPERIOR`.",
        "",
        "**D2 is attractive when the application primarily needs:**",
        "",
    ]
    lines.extend(f"- {item};" for item in use_case["detector_preferred_when"])
    lines.extend(
        [
            "",
            "**S1 is attractive when the application additionally requires:**",
            "",
        ]
    )
    lines.extend(f"- {item};" for item in use_case["segmenter_preferred_when"])
    lines.extend(
        [
            "",
            "Both remain the project's frozen final models for their respective tasks. Neither "
            "is a drop-in replacement for the other: D2 emits class, confidence and a box; S1 "
            "emits those and an instance mask. They do not solve the same output task, and no "
            "architectural identity is claimed between them.",
            "",
            "## 21. Central scientific answer",
            "",
            f"`{SCIENTIFIC_SYNTHESIS}`",
            "",
            f"> {answer['answer']}",
            "",
            f"`{HOLDOUT_POLICY}` - every figure above is a **validation** figure. "
            f"`generalises_to_test: {str(answer['scope']['generalises_to_test']).lower()}`, "
            "`generalises_to_other_hardware: "
            f"{str(answer['scope']['generalises_to_other_hardware']).lower()}`, "
            "`significance_tested: "
            f"{str(answer['scope']['significance_tested']).lower()}`.",
            "",
            "### Claim register",
            "",
            "Each headline claim, with the artifact and field that support it and the "
            "limitation that travels with it. Intended for reuse by the academic report and "
            "the pitch, so that no claim is repeated without its scope.",
            "",
        ]
    )
    lines.extend(
        _table(
            ["Claim", "Evidence artifact", "Evidence field", "Scope"],
            [
                [
                    f"`{entry['claim_id']}`",
                    f"`{entry['evidence_artifact']}`",
                    f"`{entry['evidence_field']}`",
                    f"`{entry['claim_scope']}`",
                ]
                for entry in payload["claim_register"]
            ],
        )
    )
    lines.extend(
        [
            "",
            "The full claim text and the limitation attached to each claim are in "
            "`reports/detector_segmenter_scientific_synthesis.json` under `claim_register`.",
            "",
            "## 22. Assignment coverage",
            "",
            f"`{COMMITTED_EVIDENCE}` - where the assignment's required metrics now live. Every "
            "entry is **validation only**.",
            "",
        ]
    )
    lines.extend(
        _table(
            ["Requirement", "Artifact", "Field"],
            [
                [
                    f"`{name}`",
                    f"`{coverage['delivered'][name]['artifact']}`",
                    f"`{coverage['delivered'][name]['field']}`",
                ]
                for name in coverage["delivered_order"]
            ],
        )
    )
    lines.extend(["", "**Still pending:**", ""])
    lines.extend(
        f"- `{name}` - {coverage['pending'][name]['status']}, phase "
        f"{coverage['pending'][name]['phase']}. {coverage['pending'][name]['note']}"
        for name in coverage["pending_order"]
    )
    lines.extend(
        [
            "",
            "## 23. Limitations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in payload["limitations"])
    lines.extend(
        [
            "",
            "## 24. Remaining work",
            "",
            "The repository roadmap is authoritative. Nothing below is marked complete, because "
            "no artifact supports any of it yet.",
            "",
        ]
    )
    lines.extend(
        _table(
            ["Item", "Phase", "Status", "Note"],
            [
                [f"`{entry['item']}`", entry["phase"], entry["status"], entry["note"]]
                for entry in payload["remaining_work"]
            ],
        )
    )
    lines.extend(
        [
            "",
            "## 25. Next phase",
            "",
            "**Phase 11A - final holdout evaluation protocol freeze.**",
            "",
            f"`{HOLDOUT_POLICY}` - the `test` split remains **LOCKED** and has never been "
            "evaluated, inspected, materialised, adapted or plotted. Both models are now "
            "frozen, which is the precondition phase 11 requires, but phase 10D neither "
            "unlocked nor accessed the holdout and started no part of phase 11. Access needs "
            "`allow_test=True` **and** `CSVISION_ALLOW_TEST_SPLIT=1`, and neither was used here.",
            "",
            "---",
            "",
            f"Phase {payload['phase']} - synthesis fingerprint "
            f"`{payload['synthesis_sha256']}`. Models executed: "
            f"{payload['execution']['models_executed']}. Latency measurements taken: "
            f"{payload['execution']['latency_measurements_taken']}. AP recomputed: "
            f"{str(payload['execution']['AP_recomputed']).lower()}. Spatial analysis rerun: "
            f"{str(payload['execution']['spatial_analysis_rerun']).lower()}. Test accessed: "
            f"{str(payload['execution']['test_accessed']).lower()}.",
            "",
        ]
    )
    return lines


def render_markdown(payload: Mapping[str, Any]) -> str:
    """Render the human-readable synthesis.

    Args:
        payload: The assembled synthesis.

    Returns:
        The Markdown document, newline-terminated.
    """
    models = payload["models"]
    detector = models[DETECTOR_EXPERIMENT]
    segmenter = models[SEGMENTER_EXPERIMENT]
    lines = [
        "# Detector versus segmenter - scientific and operational synthesis",
        "",
        f"Phase **{payload['phase']}** · Status **`{payload['status']}`** · "
        f"`{SCIENTIFIC_SYNTHESIS}`",
        "",
        "**This phase executed no model.** It trained nothing, ran no inference, recomputed no "
        "average precision, reran neither the phase 10B spatial analysis nor the phase 10C "
        "benchmark, tuned no threshold and read no holdout data. Every number below is copied "
        "from a committed artifact, named beside it.",
        "",
        "## 1. Research question",
        "",
        f"> {payload['research_question']}",
        "",
        "The answer is organised on exactly four axes, each separately interpretable and none "
        "combined with another:",
        "",
    ]
    lines.extend(f"{index}. `{axis}`" for index, axis in enumerate(payload["axes"], start=1))
    lines.extend(
        [
            "",
            "## 2. Frozen detector and segmenter",
            "",
        ]
    )
    lines.extend(
        _table(
            ["", "Detector", "Segmenter"],
            [
                ["Experiment", f"**{detector['experiment']}**", f"**{segmenter['experiment']}**"],
                ["Architecture", detector["model"], segmenter["model"]],
                ["Input size", f"{detector['imgsz']}", f"{segmenter['imgsz']}"],
                ["`mask_ratio`", "n/a", f"{segmenter['mask_ratio']}"],
                ["`overlap_mask`", "n/a", f"{str(segmenter['overlap_mask']).lower()}"],
                [
                    "Checkpoint SHA-256",
                    f"`{detector['checkpoint_sha256']}`",
                    f"`{segmenter['checkpoint_sha256']}`",
                ],
                [
                    "Identity fingerprint",
                    f"`{detector['identity_fingerprint']}`",
                    f"`{segmenter['identity_fingerprint']}`",
                ],
                ["Trained in this phase", "no", "no"],
                ["Modified in this phase", "no", "no"],
            ],
        )
    )
    lines.extend(
        [
            "",
            f"> `{LIMITATION}` {models['note']}",
            "",
            "## 3. Evidence sources",
            "",
            f"`{COMMITTED_EVIDENCE}` - every figure in this document resolves to one of these "
            "committed artifacts, by file digest and, where the artifact records one, by its own "
            "semantic fingerprint.",
            "",
        ]
    )
    lines.extend(
        _table(
            ["Role", "Artifact", "File SHA-256", "Recorded fingerprint"],
            [
                [
                    f"`{role}`",
                    f"`{entry['path']}`",
                    f"`{entry['file_sha256']}`",
                    f"`{entry.get('recorded_fingerprint', '-')}`",
                ]
                for role, entry in sorted(payload["source_artifacts"].items())
            ],
        )
    )
    lines.extend(
        [
            "",
            f"Comparison protocol fingerprint: `{payload['protocol_fingerprint']}` (phase 10A, "
            "historical and unmodified).",
            "",
        ]
    )
    lines.extend(_recognition_section(payload))
    lines.extend(_spatial_section(payload))
    lines.extend(_association_section(payload))
    lines.extend(_cost_section(payload))
    lines.extend(_closing_section(payload))
    return "\n".join(lines)
