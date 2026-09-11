"""Render the human-readable phase 11A protocol document.

Separated from the freeze script so the rendering is importable and testable
without running the freeze, and so a prose correction is a change to one
function rather than to the script that also writes artifacts.

The document states a protocol and no result. It contains no holdout
identifier, no holdout image, no holdout annotation and no holdout metric,
because none exists.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from construction_safety_vision.final_holdout_evaluation import (
    FAILURE_STATES,
    QUALITATIVE_CATEGORIES,
)

OBJECT_DEFINITION_ORDER: tuple[str, ...] = (
    "true_positive",
    "false_positive",
    "false_negative",
)
"""The manifest is written with sorted keys, so reading order is declared here."""

PREDECLARED_PROTOCOL = "PREDECLARED_PROTOCOL"
FINAL_HOLDOUT = "FINAL_HOLDOUT"
ONE_SHOT_EVALUATION = "ONE_SHOT_EVALUATION"
FROZEN_MODEL = "FROZEN_MODEL"
CANONICAL_EVALUATION = "CANONICAL_EVALUATION"
QUALITATIVE_SELECTION = "QUALITATIVE_SELECTION"
FAILURE_POLICY = "FAILURE_POLICY"
HOLDOUT_POLICY = "HOLDOUT_POLICY"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    """Render a Markdown table.

    Args:
        header: Column titles.
        rows: Row cells, already stringified.

    Returns:
        The table's lines.
    """
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def _models(manifest: Mapping[str, Any]) -> list[str]:
    """Render the frozen model identities.

    Args:
        manifest: The protocol manifest.

    Returns:
        Markdown lines.
    """
    detector = manifest["detector"]
    segmenter = manifest["segmenter"]
    lines = [
        "## 3. Frozen model identities",
        "",
        f"`{FROZEN_MODEL}` - both were frozen long before this protocol was written, and "
        "neither is retrained, revalidated, re-thresholded or substituted by the final "
        "evaluation. Each is resolved **by digest** through its existing freeze accessor, "
        "never by path.",
        "",
    ]
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
                    "Checkpoint bytes",
                    f"{detector['checkpoint_bytes']}",
                    f"{segmenter['checkpoint_bytes']}",
                ],
                [
                    "Identity fingerprint",
                    f"`{detector['identity_fingerprint']}`",
                    f"`{segmenter['identity_fingerprint']}`",
                ],
                ["Frozen in phase", detector["frozen_in_phase"], segmenter["frozen_in_phase"]],
                ["Accessor", f"`{detector['accessor']}`", f"`{segmenter['accessor']}`"],
                ["Executed in this phase", "no", "no"],
            ],
        )
    )
    lines.extend(
        [
            "",
            "No alternative checkpoint may be substituted in phase 11B - not `last.pt`, not D0, "
            "D1 or S0, and not a retrained copy carrying the same experiment name. A missing "
            "binary is `BLOCKED_MISSING_MODEL_ARTIFACT` and is **never** a reason to retrain: a "
            "re-run produces different bytes under the same name, which is precisely the "
            "substitution the digest accessor exists to catch.",
            "",
        ]
    )
    return lines


def _authorization(manifest: Mapping[str, Any]) -> list[str]:
    """Render the one-shot principle and the authorisation gates.

    Args:
        manifest: The protocol manifest.

    Returns:
        Markdown lines.
    """
    one_shot = manifest["one_shot"]
    auth = manifest["authorization"]
    lines = [
        "## 4. The one-shot principle",
        "",
        f"`{ONE_SHOT_EVALUATION}` - `{one_shot['policy']}`, "
        f"`reads_permitted: {one_shot['reads_permitted']}`.",
        "",
        "The holdout is the project's only remaining unbiased estimate. Every read spends some "
        "of that, and a second read spends the rest, because the second is necessarily informed "
        "by the first.",
        "",
        "**The test split is not:**",
        "",
    ]
    lines.extend(f"- {item};" for item in one_shot["the_test_split_is_not"])
    lines.extend(
        [
            "",
            "**Forbidden from the moment phase 11B begins** - not from the moment it finishes, "
            "because seeing a partial result is still seeing a result:",
            "",
        ]
    )
    lines.extend(f"- {item};" for item in one_shot["prohibited_after_execution_begins"])
    lines.extend(
        [
            "",
            f"> {one_shot['rationale']}",
            "",
            "## 5. Authorization gates",
            "",
            f"`{HOLDOUT_POLICY}` - `{auth['policy']}`. The project's existing dual gate is "
            "preserved exactly and is not re-implemented: the check delegates to "
            f"`{auth['guard']}`, so this repository has one guard rather than two that can "
            "drift apart.",
            "",
        ]
    )
    lines.extend(
        _table(
            ["Gate", "Requirement", "Sufficient alone?"],
            [
                ["Code", f"`{auth['code_opt_in']}`", "**no**"],
                ["Environment", f"`{auth['environment_opt_in']}`", "**no**"],
            ],
        )
    )
    lines.extend(
        [
            "",
            "A stray environment variable cannot unlock a development script, and a stray "
            "in-code flag cannot unlock a shell session.",
            "",
            "Two further restrictions apply:",
            "",
            f"- **Access is restricted to the declared runner** (`{auth['restricted_to_runner']}`)"
            ". A generic development script holding both opt-ins is still refused, because the "
            "accessor checks the declared purpose and caller "
            f"(`generic_development_access_permitted: "
            f"{str(auth['generic_development_access_permitted']).lower()}`).",
            "- **No code may satisfy its own precondition.** The runner may read the environment "
            f"gate and refuse; it may never write it (`runner_may_write_environment: "
            f"{str(auth['runner_may_write_environment']).lower()}`, "
            "`automatic_environment_unlock_permitted: "
            f"{str(auth['automatic_environment_unlock_permitted']).lower()}`). Only a person "
            "sets it, deliberately, immediately before phase 11B.",
            "",
            f"**Phase 11A activated neither gate** (`activated_in_this_phase: "
            f"{str(auth['activated_in_this_phase']).lower()}`), and "
            f"`{auth['environment_gate_variable']}` was verified unset at process, user and "
            "machine scope while this protocol was written.",
            "",
        ]
    )
    return lines


def _population(manifest: Mapping[str, Any]) -> list[str]:
    """Render the test-population policy.

    Args:
        manifest: The protocol manifest.

    Returns:
        Markdown lines.
    """
    population = manifest["test_population"]
    validation = population["technical_validation"]
    lines = [
        "## 6. Test population policy",
        "",
        f"`{FINAL_HOLDOUT}` - aggregate facts only. These counts were frozen in phase 5C.2, "
        "long before any model existed, and are read from the committed manifest's **aggregate "
        f"count fields** (`{', '.join(population['aggregate_source_fields'])}`). The membership "
        "sections were never opened, so **no holdout identifier entered this phase** "
        f"(`membership_section_read: "
        f"{str(population['membership_section_read']).lower()}`).",
        "",
    ]
    lines.extend(
        _table(
            ["Quantity", "Value"],
            [
                ["Images", f"**{population['images']}**"],
                ["Annotations", f"**{population['annotations']}**"],
                ["Indivisible groups", f"{population['groups']}"],
                ["Zero-instance images retained", f"{population['negative_images']}"],
                ["`split_assignment_sha256`", f"`{manifest['split_assignment_sha256']}`"],
                ["`holdout_sha256`", f"`{manifest['holdout_sha256']}`"],
                ["`class_map_sha256`", f"`{manifest['class_map_sha256']}`"],
            ],
        )
    )
    lines.extend(
        [
            "",
            f"Phase 11B evaluates **`{population['evaluate']}`**: no exclusions, no sampling, no "
            "manual removals, no stratification.",
            "",
            "The single exception is technical and is declared here rather than improvised "
            f"later: a file that cannot be decoded at all is not evaluable. The check runs "
            f"**before any model executes** (`{validation['criterion']}`), a failure is "
            f"classified `{validation['on_failure']}`, and "
            f"`silent_removal_permitted: {str(validation['silent_removal_permitted']).lower()}` "
            "- the count and the affected identifiers are recorded in the ledger and the phase "
            "stops for human review rather than the population shrinking quietly.",
            "",
        ]
    )
    return lines


def _metrics(manifest: Mapping[str, Any]) -> list[str]:
    """Render the metric and evaluator sections.

    Args:
        manifest: The protocol manifest.

    Returns:
        Markdown lines.
    """
    detector_inf = manifest["detector_inference"]
    segmenter_inf = manifest["segmenter_inference"]
    bbox = manifest["canonical_bbox_evaluator"]
    segm = manifest["canonical_segm_evaluator"]
    detector_metrics = manifest["detector_metrics"]
    segmenter_metrics = manifest["segmenter_metrics"]
    semantics = manifest["precision_recall_semantics"]
    lines = [
        "## 7. Detector metrics",
        "",
        f"`{CANONICAL_EVALUATION}` - primary:",
        "",
    ]
    lines.extend(f"- `{name}`" for name in detector_metrics["primary"])
    lines.extend(["", "Also reported:", ""])
    lines.extend(f"- `{name}`" for name in detector_metrics["secondary"])
    lines.extend(
        [
            "",
            "Per class, for **all five** frozen classes with no collapsing: "
            + ", ".join(f"`{m}`" for m in detector_metrics["per_class"])
            + ".",
            "",
            "## 8. Segmenter metrics",
            "",
            f"`{CANONICAL_EVALUATION}` - primary:",
            "",
        ]
    )
    lines.extend(f"- `{name}`" for name in segmenter_metrics["primary"])
    lines.extend(["", "Also reported:", ""])
    lines.extend(f"- `{name}`" for name in segmenter_metrics["secondary"])
    lines.extend(
        [
            "",
            "Per class: " + ", ".join(f"`{m}`" for m in segmenter_metrics["per_class"]) + ".",
            "",
            "**The segmenter's own predicted boxes** are additionally scored through the *same* "
            "bbox evaluator the detector uses, so the final localisation comparison is one "
            "evaluator over one ground truth:",
            "",
        ]
    )
    lines.extend(f"- `{name}`" for name in segmenter_metrics["box_metrics"])
    lines.extend(
        [
            "",
            f"`box_metrics_derived_from_masks: "
            f"{str(segmenter_metrics['box_metrics_derived_from_masks']).lower()}` - re-deriving "
            "them from its masks would improve their geometric consistency with the mask branch "
            "and would then measure a post-processing choice this project invented, not the "
            "model.",
            "",
            "## 9. Canonical evaluators and inference settings",
            "",
            "One external evaluator judges both models, because they run through different "
            "framework validation paths and their native numbers are not guaranteed to be "
            "computed identically.",
            "",
        ]
    )
    lines.extend(
        _table(
            ["", "Bounding box", "Instance mask"],
            [
                ["Implementation", f"`{bbox['implementation']}`", f"`{segm['implementation']}`"],
                ["`iouType`", f"`{bbox['iou_type']}`", f"`{segm['iou_type']}`"],
                ["IoU sweep", "0.50:0.05:0.95", "0.50:0.05:0.95"],
                ["`maxDets`", f"{bbox['max_dets']}", f"{segm['max_dets']}"],
                [
                    "Ground truth",
                    f"`{bbox['ground_truth_source']}`",
                    f"`{segm['ground_truth_source']}`",
                ],
                ["Mirrors", f"`{bbox['mirrors']}`", f"`{segm['mirrors']}`"],
            ],
        )
    )
    lines.extend(["", "Inference, frozen and identical in structure for both models:", ""])
    lines.extend(
        _table(
            ["Setting", "Detector", "Segmenter"],
            [
                ["`imgsz`", f"{detector_inf['imgsz']}", f"{segmenter_inf['imgsz']}"],
                ["`conf`", f"**{detector_inf['conf']}**", f"**{segmenter_inf['conf']}**"],
                ["NMS `iou`", f"{detector_inf['iou']}", f"{segmenter_inf['iou']}"],
                ["`max_det`", f"{detector_inf['max_det']}", f"{segmenter_inf['max_det']}"],
                ["Precision", f"{detector_inf['precision']}", f"{segmenter_inf['precision']}"],
                ["`augment` / TTA", "false / false", "false / false"],
                ["`retina_masks`", "n/a", f"{str(segmenter_inf['retina_masks']).lower()}"],
            ],
        )
    )
    lines.extend(
        [
            "",
            "**conf 0.001 is deliberately NOT an operating point.** Average precision integrates "
            "precision over recall, so it needs the low-scoring tail that an operational "
            "threshold discards; truncating it would silently cap recall for reasons that have "
            "nothing to do with the model. No sweep is authorised and no confidence may be tuned "
            "on the holdout.",
            "",
            "`retina_masks` puts predicted masks on the **original canvas**, where the canonical "
            "ground-truth masks live. Resizing either side to meet the other would make every "
            "IoU partly a measurement of the resize. This behaviour may not be changed.",
            "",
            "## 11. Precision and recall semantics",
            "",
            "Two different quantities carry these names in this project, and they are never "
            "mixed or differenced.",
            "",
            f"**Canonical** (`{semantics['canonical']['label']}`): computed from the canonical "
            f"accumulation at IoU {semantics['canonical']['iou_threshold_for_counting']}, at the "
            f"frozen operating confidence **{semantics['canonical']['operating_confidence']}** "
            f"(`{semantics['canonical']['operating_confidence_origin']}`). It is "
            f"`is_threshold_independent: "
            f"{str(semantics['canonical']['is_threshold_independent']).lower()}` and "
            f"`tuned_on_test: {str(semantics['canonical']['tuned_on_test']).lower()}`.",
            "",
            f"**Native framework**: `{semantics['native_framework']['status']}`, with the caveat "
            f"`{semantics['native_framework']['caveat']}` - Ultralytics reports one "
            "precision/recall pair at the F1-maximising point rather than at a fixed confidence, "
            "so it is not comparable with the canonical pair and the two are never differenced.",
            "",
        ]
    )
    return lines


def _diagnostics(manifest: Mapping[str, Any]) -> list[str]:
    """Render the direct-IoU, confusion-matrix and FP/FN sections.

    Args:
        manifest: The protocol manifest.

    Returns:
        Markdown lines.
    """
    direct = manifest["direct_iou"]
    matrix = manifest["confusion_matrix"]
    obj = manifest["object_level_matching"]
    lines = [
        "## 10. Direct mask-IoU diagnostic",
        "",
        f"`{PREDECLARED_PROTOCOL}` - the **exact phase 8C protocol**, reused unchanged "
        f"(`protocol_fingerprint` `{direct['protocol_fingerprint']}`, "
        f"`unchanged_from_phase_8c: {str(direct['unchanged_from_phase_8c']).lower()}`, "
        f"`new_rule_invented: {str(direct['new_rule_invented']).lower()}`). Inventing a new "
        "matching rule now would let it be chosen with the final result in view.",
        "",
        f"Role: **`{direct['role']}`**. It is "
        f"`is_primary_segmenter_metric: "
        f"{str(direct['is_primary_segmenter_metric']).lower()}` and "
        f"`is_an_average_precision: {str(direct['is_an_average_precision']).lower()}`.",
        "",
        "Reported at minimum: "
        + ", ".join(f"`{name}`" for name in direct["headline"])
        + ", plus the per-class breakdown.",
        "",
        f"> This diagnostic keeps its own **operational confidence of "
        f"{direct['inference']['conf']}**, exactly as in phase 8C, while the canonical AP "
        "protocols keep 0.001. The two are never mixed, averaged or swapped, and a figure "
        "produced at one may never be reported under the other's name.",
        "",
        f"> {direct['headlines_are_not_interchangeable']}",
        "",
        "## 12. Confusion-matrix semantics",
        "",
        f"`{PREDECLARED_PROTOCOL}` - no canonical confusion-matrix protocol existed before this "
        f"phase (`prior_canonical_protocol_existed: "
        f"{str(matrix['prior_canonical_protocol_existed']).lower()}`). One is frozen here on the "
        f"basis `{matrix['basis']}`: the semantics already in force during validation, **read "
        f"from the installed `{matrix['implementation_version']}` source**. No holdout data was "
        "consulted while deciding it, and freezing it changes nothing and invents nothing - "
        "these are the exact values that produced every committed validation matrix in this "
        "repository.",
        "",
    ]
    lines.extend(
        _table(
            ["Setting", "Value", "Where it comes from"],
            [
                [
                    "Implementation",
                    f"`{matrix['implementation']}`",
                    matrix["implementation_version"],
                ],
                ["Confidence", f"**{matrix['conf']}**", f"`{matrix['conf_source']}`"],
                [
                    "IoU threshold",
                    f"**{matrix['iou_threshold']}**",
                    f"`{matrix['iou_threshold_source']}`",
                ],
                ["Matching", f"`{matrix['matching']}`", "read from source"],
                ["Assignment", f"`{matrix['assignment']}`", "read from source"],
                ["Shape", f"{matrix['matrix_dimension']}x{matrix['matrix_dimension']}", "`nc + 1`"],
                ["Orientation", f"`{matrix['orientation']}`", "read from source"],
            ],
        )
    )
    lines.extend(
        [
            "",
            f"**Background handling.** {matrix['background_handling']}",
            "",
            "Class ordering is the canonical class-map index order: "
            + ", ".join(f"`{name}`" for name in matrix["class_order"])
            + ". A normalised variant is reported alongside the raw counts.",
            "",
            f"A matrix is produced for **both** models, using identical detection semantics over "
            "each one's predicted boxes, so the two are directly comparable. "
            f"{matrix['segmenter_matrix_note']}",
            "",
            f"`threshold_chosen_after_seeing_test: "
            f"{str(matrix['threshold_chosen_after_seeing_test']).lower()}`.",
            "",
            "## 13. FP / FN semantics",
            "",
            f"`{PREDECLARED_PROTOCOL}` - frozen separately from the confusion matrix, because "
            "the qualitative selection needs per-instance outcomes rather than a matrix cell, "
            'and because a **class-aware** rule is the right one for "which objects did the '
            'model get wrong".',
            "",
        ]
    )
    lines.extend(
        _table(
            ["Property", "Value"],
            [
                ["Class-aware", f"{str(obj['class_aware']).lower()}"],
                ["IoU threshold", f"{obj['iou_threshold']}"],
                ["Assignment", f"`{obj['assignment']}`"],
                ["Algorithm", f"`{obj['assignment_algorithm']}`"],
                ["Confidence threshold", f"{obj['confidence_threshold']}"],
            ],
        )
    )
    lines.extend(["", "Definitions, fixed in advance:", ""])
    lines.extend(
        f"- **{name.replace('_', ' ')}** - {obj['definitions'][name]}"
        for name in OBJECT_DEFINITION_ORDER
    )
    taxonomy = obj["segmentation_failure_taxonomy"]
    lines.extend(
        [
            "",
            "For the segmentation qualitative work a miss and a bad mask are different failures "
            "and are not pooled. Categories are evaluated in the order listed and the first that "
            "matches wins:",
            "",
        ]
    )
    lines.extend(
        f"- **`{name}`** - {taxonomy[name]}"
        for name in (
            "DETECTION_MISS",
            "CLASSIFICATION_MISMATCH",
            "LOCALIZATION_FAILURE",
            "MASK_QUALITY_FAILURE",
        )
    )
    lines.extend(
        [
            "",
            f"An instance matching none of them is `{taxonomy['unclassified_label']}`; the "
            "categories describe failures and do not claim to partition every instance "
            f"(`categories_partition_every_ground_truth_instance: "
            f"{str(taxonomy['categories_partition_every_ground_truth_instance']).lower()}`).",
            "",
            f"`redefinition_after_seeing_test_permitted: "
            f"{str(obj['redefinition_after_seeing_test_permitted']).lower()}`.",
            "",
        ]
    )
    return lines


def _qualitative(manifest: Mapping[str, Any]) -> list[str]:
    """Render the deterministic qualitative-selection section.

    Args:
        manifest: The protocol manifest.

    Returns:
        Markdown lines.
    """
    block = manifest["qualitative_selection"]
    lines = [
        "## 14. Qualitative selection",
        "",
        f"`{QUALITATIVE_SELECTION}` - `{block['policy']}`.",
        "",
        "This is the point of the whole phase in miniature: the examples that appear in the "
        "academic report are chosen by a rule frozen **before anyone has seen a holdout image**. "
        "Browsing first and choosing second is exactly what this prevents.",
        "",
        f"`manual_cherry_picking_permitted: "
        f"{str(block['manual_cherry_picking_permitted']).lower()}` · "
        f"`random_sampling_permitted: {str(block['random_sampling_permitted']).lower()}` · "
        f"`images_inspected_to_design_this_rule: "
        f"{block['images_inspected_to_design_this_rule']}`.",
        "",
        f"**{block['examples_per_category']} examples per category**, ranked as follows:",
        "",
    ]
    lines.extend(
        _table(
            ["Category", "Applies to", "Ranked by", "Order"],
            [
                [
                    f"`{name}`",
                    ", ".join(block["categories"][name]["applies_to"]),
                    f"`{block['categories'][name]['rank_by']}`",
                    block["categories"][name]["order"],
                ]
                for name in QUALITATIVE_CATEGORIES
            ],
        )
    )
    lines.extend(
        [
            "",
            "**Tie-breaking** is a total order, so the same holdout produces the same gallery on "
            "any machine:",
            "",
        ]
    )
    lines.extend(f"{index}. {rule}" for index, rule in enumerate(block["tie_breaking"], start=1))
    lines.extend(
        [
            "",
            "Every ranked quantity is a float, so ties are possible; the chain ends in an "
            "identifier, which is unique.",
            "",
            f"`same_image_may_occupy_multiple_categories: "
            f"{str(block['same_image_may_occupy_multiple_categories']).lower()}` · "
            f"`same_instance_may_occupy_multiple_categories: "
            f"{str(block['same_instance_may_occupy_multiple_categories']).lower()}`. "
            f"{block['duplicate_instance_policy']}",
            "",
            f"A category with fewer qualifying instances than the quota publishes what it has and "
            f"records the shortfall (`{block['underfilled_category_policy']}`); "
            f"`topping_up_from_another_category_permitted: "
            f"{str(block['topping_up_from_another_category_permitted']).lower()}`.",
            "",
            "**Human visual interpretation may occur only after the deterministic selection has "
            "been generated.** The ranking decides which instances are looked at; a person then "
            "explains what they show.",
            "",
        ]
    )
    return lines


def _persistence(manifest: Mapping[str, Any]) -> list[str]:
    """Render persistence, fingerprints, ledger and failure policy.

    Args:
        manifest: The protocol manifest.

    Returns:
        Markdown lines.
    """
    persistence = manifest["prediction_persistence"]
    fingerprints = manifest["fingerprints"]
    ledger = manifest["one_shot_ledger"]
    policy = manifest["failure_policy"]
    lines = [
        "## 15. Prediction persistence",
        "",
        "Raw predictions go to deterministic, **git-ignored** runtime locations and are "
        "persisted **before any metric is computed** "
        f"(`persisted_before_metric_computation: "
        f"{str(persistence['persisted_before_metric_computation']).lower()}`), so a later report "
        "rebuild can regenerate every number without touching a model.",
        "",
    ]
    lines.extend(
        _table(
            ["Artifact", "Location"],
            [
                ["Detector raw predictions", f"`{persistence['detector_raw']}`"],
                ["Segmenter raw predictions", f"`{persistence['segmenter_raw']}`"],
                ["Canonical bbox predictions", f"`{persistence['canonical_bbox_predictions']}`"],
                ["Canonical mask predictions", f"`{persistence['canonical_mask_predictions']}`"],
                ["Direct-IoU representation", f"`{persistence['direct_iou_representation']}`"],
                ["One-shot ledger", f"`{persistence['ledger']}`"],
            ],
        )
    )
    lines.extend(["", "Committed to the repository:", ""])
    lines.extend(f"- {item};" for item in persistence["committed_outputs"])
    lines.extend(
        [
            "",
            f"**Not** committed: bulk prediction tensors, holdout imagery, holdout identifiers "
            f"(`bulk_tensors_committed: "
            f"{str(persistence['bulk_tensors_committed']).lower()}`, "
            f"`test_imagery_committed: "
            f"{str(persistence['test_imagery_committed']).lower()}`).",
            "",
            "## 16. Prediction fingerprints",
            "",
            f"`{fingerprints['prediction']['detector']}` and "
            f"`{fingerprints['prediction']['segmenter']}` cover:",
            "",
        ]
    )
    lines.extend(f"- {item};" for item in fingerprints["prediction"]["covers"])
    lines.extend(
        [
            "",
            f"> {fingerprints['prediction']['purpose']}",
            "",
            "## 17. Result fingerprints",
            "",
        ]
    )
    lines.extend(
        _table(
            ["Result", "Fingerprint field"],
            [
                ["Detector", f"`{fingerprints['result']['detector']}`"],
                ["Segmenter", f"`{fingerprints['result']['segmenter']}`"],
                ["Direct IoU", f"`{fingerprints['result']['direct_iou']}`"],
                ["Qualitative selection", f"`{fingerprints['result']['qualitative']}`"],
            ],
        )
    )
    lines.extend(["", "Each covers:", ""])
    lines.extend(f"- {item};" for item in fingerprints["result"]["covers"])
    lines.extend(["", "Each **excludes**:", ""])
    lines.extend(f"- {item};" for item in fingerprints["excludes"])
    lines.extend(
        [
            "",
            f"`deterministic_recomputation_required: "
            f"{str(fingerprints['deterministic_recomputation_required']).lower()}`. "
            f"{fingerprints['rebuild_policy']['note']}",
            "",
            "## 18. The one-shot ledger",
            "",
            f"`{ONE_SHOT_EVALUATION}` - append-only, at `{ledger['artifact']}`, summarised into "
            f"`{ledger['committed_summary']}`. "
            f"`attempt_counter_resettable: "
            f"{str(ledger['attempt_counter_resettable']).lower()}` and "
            f"`mutable_history: {str(ledger['mutable_history']).lower()}`.",
            "",
            "States:",
            "",
        ]
    )
    lines.extend(f"{index}. `{state}`" for index, state in enumerate(ledger["states"], start=1))
    lines.extend(["", "Recorded at each step:", ""])
    lines.extend(f"- {item};" for item in ledger["records"])
    rerun = ledger["rerun_policy"]
    lines.extend(
        [
            "",
            f"**Rerun policy.** `automatic_rerun_permitted: "
            f"{str(rerun['automatic_rerun_permitted']).lower()}`, "
            f"`human_authorization_required: "
            f"{str(rerun['human_authorization_required']).lower()}`, "
            f"`attempt_must_be_numbered: "
            f"{str(rerun['attempt_must_be_numbered']).lower()}`, "
            f"`second_run_may_be_presented_as_the_first: "
            f"{str(rerun['second_run_may_be_presented_as_the_first']).lower()}`.",
            "",
            "## 19. Failure and recovery policy",
            "",
            f"`{FAILURE_POLICY}` - named in advance, so a failure is classified rather than "
            'improvised, and so the difference between "the report failed to write" and "the '
            'model failed to run" is settled before either can happen.',
            "",
        ]
    )
    lines.extend(
        _table(
            ["State", "Meaning"],
            [[f"`{name}`", policy["states"][name].strip()] for name in FAILURE_STATES],
        )
    )
    partial = policy["partial_prediction_policy"]
    lines.extend(
        [
            "",
            "**The two cases that matter most.** If a complete, fingerprinted prediction set "
            "exists and only report writing failed, **do not rerun inference** - rebuild the "
            "metrics and reports from the persisted predictions. If inference crashed *before* a "
            "complete prediction set existed, preserve the evidence, stop, and require human "
            f"review: `automatic_restart_permitted: "
            f"{str(partial['automatic_restart_permitted']).lower()}`, "
            f"`present_second_run_as_original_permitted: "
            f"{str(partial['present_second_run_as_original_permitted']).lower()}`.",
            "",
            f"`silent_rerun_permitted: {str(policy['silent_rerun_permitted']).lower()}`.",
            "",
        ]
    )
    return lines


def _closing(manifest: Mapping[str, Any]) -> list[str]:
    """Render the comparison policy, prohibitions, plan and status.

    Args:
        manifest: The protocol manifest.

    Returns:
        Markdown lines.
    """
    comparison = manifest["validation_versus_test"]
    reports = manifest["reports"]
    figures = manifest["figures"]
    final = manifest["final_comparison"]
    cost = manifest["computational_cost"]
    spatial = manifest["spatial_analysis_on_test"]
    lines = [
        "## 20. Validation-versus-test comparison",
        "",
        f"`{comparison['label']}` - permitted, and bounded. Only these may be put side by side:",
        "",
    ]
    lines.extend(f"- {item};" for item in comparison["permitted_content"])
    lines.extend(
        [
            "",
            f"`metrics_must_already_exist: "
            f"{str(comparison['metrics_must_already_exist']).lower()}` and "
            f"`new_metric_invented_for_the_comparison: "
            f"{str(comparison['new_metric_invented_for_the_comparison']).lower()}`.",
            "",
            f"**No significance test** (`significance_test: "
            f"{str(comparison['significance_test']).lower()}`). "
            f"{comparison['significance_test_note']}",
            "",
            "Prohibited:",
            "",
        ]
    )
    lines.extend(f"- {item};" for item in comparison["prohibited"])
    lines.extend(
        [
            "",
            "## 21. Prohibited post-test actions",
            "",
            "Every one of these becomes forbidden the moment phase 11B begins:",
            "",
        ]
    )
    lines.extend(f"- {item};" for item in manifest["prohibited"])
    lines.extend(
        [
            "",
            f"**Computational cost is not re-measured.** `rerun_on_test: "
            f"{str(cost['rerun_on_test']).lower()}` - {cost['rationale'].strip()}",
            "",
            f"**The phase 10B exploratory spatial study is not repeated.** "
            f"`repeat_full_phase_10b_exploration: "
            f"{str(spatial['repeat_full_phase_10b_exploration']).lower()}`, "
            f"`new_spatial_metrics_permitted: "
            f"{str(spatial['new_spatial_metrics_permitted']).lower()}`. "
            f"{spatial['rationale'].strip()}",
            "",
            "## 22. Reporting plan",
            "",
            f"`{PREDECLARED_PROTOCOL}` - schemas only. "
            f"`created_in_phase_11a: {str(reports['created_in_phase_11a']).lower()}` and "
            f"`placeholder_values_permitted: "
            f"{str(reports['placeholder_values_permitted']).lower()}`: **no artifact below "
            "exists yet, and none may be created with invented values.**",
            "",
        ]
    )
    lines.extend(
        _table(
            ["Artifact", "Content"],
            [
                [f"`{reports['detector']}`", "detector canonical results and per-class table"],
                [f"`{reports['segmenter']}`", "segmenter canonical mask and box results"],
                [f"`{reports['direct_iou']}`", "the secondary direct mask-IoU diagnostic"],
                [f"`{reports['human_readable']}`", "the human-readable final evaluation"],
                [f"`{reports['provenance']}`", "one-shot ledger summary and fingerprints"],
            ],
        )
    )
    lines.extend(["", "Required sections of the final report:", ""])
    lines.extend(
        f"{index}. {section}" for index, section in enumerate(reports["required_sections"], start=1)
    )
    lines.extend(["", "**Permitted figures:**", ""])
    lines.extend(f"- {item};" for item in figures["permitted"])
    lines.extend(
        [
            "",
            f"Any holdout image shown in the academic report must come from the deterministic "
            f"selection (`test_imagery_source: {figures['test_imagery_source']}`). "
            f"`browsing_then_choosing_permitted: "
            f"{str(figures['browsing_then_choosing_permitted']).lower()}`, "
            f"`exploratory_visual_mining_permitted: "
            f"{str(figures['exploratory_visual_mining_permitted']).lower()}`.",
            "",
            "### The final detector-versus-segmenter comparison",
            "",
            f"`{final['label']}`, on the phase 10D axes: "
            + ", ".join(f"`{axis}`" for axis in final["axes"])
            + ".",
            "",
            f"`winner_declared: {str(final['winner_declared']).lower()}` · "
            f"`composite_score: {str(final['composite_score']).lower()}` · "
            f"`weighted_ranking: {str(final['weighted_ranking']).lower()}` · "
            f"`model_selection_follows: "
            f"{str(final['model_selection_follows']).lower()}`. The comparison remains what it "
            "has been since phase 10A: descriptive, not a contest.",
            "",
            "## 23. Assignment coverage",
            "",
            "This protocol closes the remaining evaluation requirements of criterion C4 once "
            "phase 11B executes it:",
            "",
        ]
    )
    lines.extend(
        _table(
            ["Requirement", "Covered by"],
            [
                ["mAP@0.50 and mAP@0.50:0.95, detection", "`CANONICAL_TEST_BOX_MAP50{,_95}`"],
                ["mAP@0.50 and mAP@0.50:0.95, segmentation", "`CANONICAL_TEST_MASK_MAP50{,_95}`"],
                ["IoU", "the direct instance-mask IoU diagnostic (phase 8C protocol)"],
                ["precision and recall", "canonical, at the frozen operating point"],
                ["confusion matrix", "frozen framework semantics, both models"],
                ["qualitative FP/FN analysis", "deterministic selection, six categories"],
                ["single holdout evaluation", "the one-shot ledger"],
            ],
        )
    )
    lines.extend(
        [
            "",
            "## 24. Phase 11B execution contract",
            "",
            "The runner executes exactly these steps, in this order:",
            "",
        ]
    )
    lines.extend(
        f"{index}. `{step}`"
        for index, step in enumerate(manifest["phase_11b_execution_order"], start=1)
    )
    lines.extend(
        [
            "",
            f"Implemented at `{manifest['runner']}`, which **refuses to execute** while phase 11B "
            "is unauthorised. It reads the environment gate and never writes it.",
            "",
            "## 25. Holdout status",
            "",
            f"`{HOLDOUT_POLICY}` - **`{manifest['test']['status']}`**.",
            "",
            f"> {manifest['test']['reason']}",
            "",
        ]
    )
    lines.extend(
        _table(
            ["Count", "Value"],
            [
                ["Models executed", f"{manifest['models_executed']}"],
                ["Test predictions produced", f"{manifest['test_predictions_produced']}"],
                ["Test metrics computed", f"{manifest['test_metrics_computed']}"],
                ["Test images read", f"{manifest['test_images_read']}"],
                ["Test annotations read", f"{manifest['test_annotations_read']}"],
                ["Test identifiers recorded", f"{manifest['test_identifiers_recorded']}"],
                ["Latency measurements taken", f"{manifest['latency_measurements_taken']}"],
                ["Thresholds tuned", f"{manifest['thresholds_tuned']}"],
                ["Figures generated", f"{manifest['figures_generated']}"],
            ],
        )
    )
    lines.extend(
        [
            "",
            "---",
            "",
            f"Phase {manifest['phase']} · `{manifest['status']}` · protocol fingerprint "
            f"`{manifest['protocol_fingerprint']}` · executes in phase "
            f"{manifest['executes_in_phase']}.",
            "",
        ]
    )
    return lines


def render(manifest: Mapping[str, Any]) -> str:
    """Render the whole protocol document.

    Args:
        manifest: The assembled protocol manifest.

    Returns:
        The Markdown document, newline-terminated.
    """
    lines = [
        "# Final holdout evaluation - frozen protocol",
        "",
        f"Phase **{manifest['phase']}** · Status **`{manifest['status']}`** · "
        f"`{PREDECLARED_PROTOCOL}`",
        "",
        "**This phase evaluated nothing.** It predeclares, in full, what the single holdout "
        "evaluation will measure, at which settings, by which evaluator, how failures are "
        "handled and what may never be done afterwards - and then stops. No holdout identifier, "
        "image, annotation, prediction or metric exists in any artifact it wrote, and neither "
        "authorisation gate was activated.",
        "",
        "## 1. Objective",
        "",
        f"> {manifest['objective'].strip()}",
        "",
        "## 2. Why the holdout is still protected",
        "",
        "Every design decision this project has made - the split, the adapters, both "
        "architectures, both resolutions, the `overlap_mask` treatment, every threshold, every "
        "evaluator and every diagnostic - was made on `train` and `validation` alone. That is "
        "what gives the holdout its value: it is the only data that has never informed a choice.",
        "",
        "The protocol below is written **now**, while no holdout number exists and none can "
        'exist, precisely so that the answer to "why this threshold, this rule, these '
        'examples?" is always "because it was frozen before anyone could see the result". A '
        "protocol written afterwards would be indistinguishable from a set of choices that "
        "happened to flatter the outcome.",
        "",
    ]
    lines.extend(_models(manifest))
    lines.extend(_authorization(manifest))
    lines.extend(_population(manifest))
    lines.extend(_metrics(manifest))
    lines.extend(_diagnostics(manifest))
    lines.extend(_qualitative(manifest))
    lines.extend(_persistence(manifest))
    lines.extend(_closing(manifest))
    return "\n".join(lines)
