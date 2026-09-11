"""Render the human-readable phase 11B holdout evaluation report.

Separated from the runner so the rendering is importable and testable without
executing anything, and so a prose correction is a change to one function rather
than to the code that also reads the holdout.

The document states results. It carries no holdout image identifier, no holdout
annotation identifier and no holdout imagery, because publishing those is a leak
even when no pixel travels with them: every number here is an aggregate, and the
deterministic qualitative selection is described by its rule, its ranked values
and its fingerprint rather than by the instances it names.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from construction_safety_vision.final_holdout_evaluation import (
    AP_CONF,
    CLASSES,
    OPERATIONAL_CONF,
    QUALITATIVE_CATEGORIES,
    RARE_CLASS,
)

ONE_SHOT = "ONE_SHOT_FINAL_HOLDOUT_EVALUATION"
OBSERVED = "FINAL_TEST_OBSERVED"
DESCRIPTIVE = "DESCRIPTIVE_GENERALIZATION_COMPARISON"


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


def _number(value: Any) -> str:
    """Render a metric for the report, or mark it undefined.

    Args:
        value: A number, or ``None``.

    Returns:
        The six-decimal rendering, or ``n/a``.
    """
    if value is None:
        return "n/a"
    return f"{float(value):.6f}"


def _delta(value: Any) -> str:
    """Render a signed difference.

    Args:
        value: A number, or ``None``.

    Returns:
        The signed six-decimal rendering, or ``n/a``.
    """
    if value is None:
        return "n/a"
    return f"{float(value):+.6f}"


def _policy(payload: Mapping[str, Any]) -> list[str]:
    """Render the holdout policy and the one-shot execution status.

    Args:
        payload: The assembled report payload.

    Returns:
        Markdown lines.
    """
    ledger = payload["ledger"]
    return [
        "# Final holdout evaluation - one-shot results",
        "",
        f"**Phase 11B. `{ONE_SHOT}`. `{OBSERVED}`.**",
        "",
        "## 1. Holdout policy",
        "",
        "The `test` split was a locked holdout from phase 5C.2 until this phase. It took no "
        "part in any design, selection, tuning or diagnostic decision: not in model selection, "
        "not in architecture selection, not in hyperparameter or augmentation tuning, not in "
        "threshold tuning, not in qualitative debugging, and it was not looked at. Both models "
        "were frozen long before - the detector in phase 7D, the segmenter in phase 8G - and "
        "the whole validation comparison was published in phases 10A-10D.",
        "",
        "It has now been read **once**, under the protocol phase 11A froze before any holdout "
        "number could exist. Every rule below - which checkpoints, at which settings, judged by "
        "which evaluator, with which confusion-matrix semantics, and which qualitative examples "
        "- was written down in advance. Nothing was adapted after a number was seen.",
        "",
        "## 2. One-shot execution status",
        "",
        *_table(
            ["", "Value"],
            [
                ["Classification", f"`{payload['classification']}`"],
                ["Attempt", f"{ledger['attempt']}"],
                ["Attempt justification", f"`{ledger['justification']}`"],
                ["Ledger final state", f"`{ledger['state']}`"],
                ["Attempt counter resettable", "no"],
                ["Holdout reads permitted", "1"],
                ["Holdout reads performed", "1"],
                ["Models executed", f"{payload['counts']['models_executed']}"],
                ["Inference passes", f"{payload['counts']['inference_passes']}"],
                ["Models trained", f"{payload['counts']['models_trained']}"],
                ["Thresholds tuned", f"{payload['counts']['thresholds_tuned']}"],
                ["Latency measurements taken", f"{payload['counts']['latency_measurements']}"],
                ["New spatial metrics", f"{payload['counts']['new_spatial_metrics']}"],
                ["Prediction reruns", f"{payload['counts']['prediction_reruns']}"],
                ["Protocol fingerprint", f"`{payload['protocol_fingerprint']}`"],
            ],
        ),
        "",
        "Predictions were persisted and fingerprinted **before any metric was computed**, and "
        "every figure in this report was then derived from those persisted files. No model was "
        "invoked during metric computation or report generation. That ordering is what makes a "
        "report rebuild possible without a second inference run, and it is why an artifact-write "
        "failure and a prediction failure have separate names in the frozen policy.",
        "",
    ]


def _models(payload: Mapping[str, Any]) -> list[str]:
    """Render the frozen model identities and the prediction fingerprints.

    Args:
        payload: The assembled report payload.

    Returns:
        Markdown lines.
    """
    detector = payload["detector"]["model"]
    segmenter = payload["segmenter"]["model"]
    return [
        "## 3. Final model identities",
        "",
        "Both were resolved **by digest**, never by path, through their existing freeze "
        "accessors. Neither was retrained, revalidated, re-thresholded or substituted.",
        "",
        *_table(
            ["", "Detector", "Segmenter"],
            [
                ["Experiment", f"**{detector['experiment']}**", f"**{segmenter['experiment']}**"],
                ["Architecture", detector["model"], segmenter["model"]],
                ["Input size", f"{detector['imgsz']}", f"{segmenter['imgsz']}"],
                ["`mask_ratio`", "n/a", f"{segmenter.get('mask_ratio', 'n/a')}"],
                ["`overlap_mask`", "n/a", f"{str(segmenter.get('overlap_mask')).lower()}"],
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
                ["Frozen in phase", detector["frozen_in_phase"], segmenter["frozen_in_phase"]],
                ["Trained in this phase", "no", "no"],
                ["Modified in this phase", "no", "no"],
            ],
        ),
        "",
        "### Prediction fingerprints",
        "",
        *_table(
            ["Pass", "Model", "Confidence", "Masks", "Predictions", "Fingerprint"],
            [
                [
                    f"`{entry['pass']}`",
                    entry["model"],
                    f"{entry['conf']}",
                    "yes" if entry["masks"] else "no",
                    f"{entry['predictions']}",
                    f"`{entry['fingerprint']}`",
                ]
                for entry in payload["prediction_passes"]
            ],
        ),
        "",
        f"Three inference passes ran, all declared in advance: each model's AP pass at conf "
        f"{AP_CONF}, and the segmenter's operational pass at conf {OPERATIONAL_CONF} for the "
        "phase 8C direct-IoU diagnostic. The two confidences are never mixed, averaged or "
        "swapped: average precision integrates over the score curve and needs its low-scoring "
        "tail, which is why it is deliberately **not** an operating point.",
        "",
    ]


def _population(payload: Mapping[str, Any]) -> list[str]:
    """Render the holdout population and its integrity check.

    Args:
        payload: The assembled report payload.

    Returns:
        Markdown lines.
    """
    population = payload["population"]
    integrity = payload["integrity"]
    rows = [
        ["Images", f"{population['images']}"],
        ["Annotations", f"{population['annotations']}"],
        ["Evaluated", "all frozen holdout images"],
        ["Sampled", "no"],
        ["Manually excluded", "0"],
        ["Membership fingerprint", f"`{population['membership_sha256']}`"],
        ["Integrity problems", f"{integrity['problems']}"],
        [
            "Byte-identical image copies",
            f"{integrity['byte_identical_copies']}/{integrity['image_copies']}",
        ],
        [
            "Detection/segmentation alignment problems",
            f"{integrity['detection_segmentation_alignment_problems']}",
        ],
        ["Geometry round-trip mismatches", f"{integrity['geometry_round_trip_mismatches']}"],
        ["Zero-instance images", f"{integrity['zero_instance_images']}"],
    ]
    return [
        "## 4. Test population",
        "",
        "The holdout was materialised by the **phase 5D function with the phase 5D "
        "configuration** - the same function that wrote `train` and `validation`. There is no "
        "split-specific branch that could give the protected split different treatment, and the "
        "predeclared technical validation ran before any model did.",
        "",
        *_table(["", "Value"], rows),
        "",
        "The counts match the aggregate figures frozen in phase 5C.2, long before any model "
        "existed. Per-image membership is not published here: a holdout identifier in a "
        "committed artifact is a leak even when no pixel travels with it.",
        "",
        "### Class support on the holdout",
        "",
        "The **phase 7A support rule is reused unchanged** - `COMPARISON_SUPPORTED` requires at "
        "least 5 positive images and at least 20 instances. It names no class; the admitted set "
        "is its output. It was applied only after the evaluation, its thresholds were not "
        "re-parameterised in response to what the holdout turned out to contain, and on the "
        "holdout it is a descriptive caveat that decides nothing.",
        "",
        *_table(
            ["Class", "Images", "Instances", "Status"],
            [
                [
                    f"`{name}`",
                    f"{payload['support'][name]['images']}",
                    f"{payload['support'][name]['instances']}",
                    f"`{payload['support'][name]['status']}`",
                ]
                for name in CLASSES
            ],
        ),
        "",
    ]


def _per_class_rows(per_class: Mapping[str, Mapping[str, Any]]) -> list[list[str]]:
    """Render per-class AP rows in the frozen class order.

    Args:
        per_class: Per-class AP.

    Returns:
        Table rows.
    """
    return [
        [
            f"`{name}`",
            _number(per_class.get(name, {}).get("AP@0.50:0.95")),
            _number(per_class.get(name, {}).get("AP@0.50")),
        ]
        for name in CLASSES
    ]


def _operating_rows(block: Mapping[str, Any]) -> list[list[str]]:
    """Render per-class operating-point precision and recall.

    Args:
        block: The operating-point payload.

    Returns:
        Table rows.
    """
    per_class = block["per_class"]
    return [
        [
            f"`{name}`",
            _number(per_class.get(name, {}).get("precision")),
            _number(per_class.get(name, {}).get("recall")),
            f"{per_class.get(name, {}).get('true_positives')}",
            f"{per_class.get(name, {}).get('false_positives')}",
            f"{per_class.get(name, {}).get('false_negatives')}",
        ]
        for name in CLASSES
    ]


def _detector(payload: Mapping[str, Any]) -> list[str]:
    """Render the detector's canonical holdout results.

    Args:
        payload: The assembled report payload.

    Returns:
        Markdown lines.
    """
    result = payload["detector"]
    box = result["canonical_box"]
    operating = result["canonical_precision_recall"]
    admitted = ", ".join("`" + name + "`" for name in result["supported_macro"]["admitted_classes"])
    return [
        "## 5. Detector canonical bounding-box results",
        "",
        "One external evaluator judges both models' boxes - `pycocotools.cocoeval.COCOeval` at "
        "`iouType='bbox'` against the canonical phase 5D holdout boxes, IoU 0.50:0.05:0.95, "
        "`maxDets` [1, 10, 100] - because the two models run through different framework "
        "validation paths and their native box numbers are not guaranteed to be computed "
        "identically.",
        "",
        *_table(
            ["Metric", "Value"],
            [
                [
                    "`CANONICAL_TEST_BOX_MAP50_95`",
                    f"**{_number(box['CANONICAL_TEST_BOX_MAP50_95'])}**",
                ],
                ["`CANONICAL_TEST_BOX_MAP50`", _number(box["CANONICAL_TEST_BOX_MAP50"])],
                ["Detections scored", f"{box['detections_scored']}"],
                ["Inference confidence", f"{AP_CONF}"],
            ],
        ),
        "",
        "### 5.1 Detector per-class results",
        "",
        *_table(["Class", "AP@0.50:0.95", "AP@0.50"], _per_class_rows(box["per_class"])),
        "",
        "### 5.2 Detector precision and recall at the frozen operating point",
        "",
        f"Precision and recall are **not threshold-independent**. They are reported at the "
        f"project's single operational confidence of {OPERATIONAL_CONF}, counted at IoU "
        f"{operating['iou_threshold_for_counting']} over exactly the detections the AP figures "
        "were computed from. That operating point has been frozen since phase 10A and was not "
        "chosen with any holdout number in view, and no threshold was swept on the holdout.",
        "",
        *_table(
            ["", "Precision", "Recall", "TP", "FP", "FN"],
            [
                [
                    "**all classes**",
                    _number(operating["precision"]),
                    _number(operating["recall"]),
                    f"{operating['true_positives']}",
                    f"{operating['false_positives']}",
                    f"{operating['false_negatives']}",
                ],
                *_operating_rows(operating),
            ],
        ),
        "",
        f"Descriptive supported macro AP@0.50:0.95 over the classes the frozen rule admits: "
        f"**{_number(result['supported_macro']['value'])}** "
        f"({admitted}). "
        "This is a descriptive figure on the holdout, never a selection metric.",
        "",
    ]


def _segmenter(payload: Mapping[str, Any]) -> list[str]:
    """Render the segmenter's canonical mask and box results.

    Args:
        payload: The assembled report payload.

    Returns:
        Markdown lines.
    """
    result = payload["segmenter"]
    mask = result["canonical_mask"]
    box = result["canonical_box"]
    mask_operating = result["canonical_mask_precision_recall"]
    comparison = result["localisation_comparison_with_detector"]
    return [
        "## 6. Segmenter canonical mask results",
        "",
        "`pycocotools.cocoeval.COCOeval` at `iouType='segm'` against the canonical phase 5D "
        "holdout masks, on the original image canvas. **Mask and box never merge**: the box "
        "figures below are reported in their own section and no composite exists.",
        "",
        *_table(
            ["Metric", "Value"],
            [
                [
                    "`CANONICAL_TEST_MASK_MAP50_95`",
                    f"**{_number(mask['CANONICAL_TEST_MASK_MAP50_95'])}**",
                ],
                ["`CANONICAL_TEST_MASK_MAP50`", _number(mask["CANONICAL_TEST_MASK_MAP50"])],
                ["Detections scored", f"{mask['detections_scored']}"],
                ["Inference confidence", f"{AP_CONF}"],
            ],
        ),
        "",
        "### 6.1 Segmenter per-class mask results",
        "",
        *_table(["Class", "AP@0.50:0.95", "AP@0.50"], _per_class_rows(mask["per_class"])),
        "",
        "### 6.2 Segmenter mask precision and recall at the frozen operating point",
        "",
        *_table(
            ["", "Precision", "Recall", "TP", "FP", "FN"],
            [
                [
                    "**all classes**",
                    _number(mask_operating["precision"]),
                    _number(mask_operating["recall"]),
                    f"{mask_operating['true_positives']}",
                    f"{mask_operating['false_positives']}",
                    f"{mask_operating['false_negatives']}",
                ],
                *_operating_rows(mask_operating),
            ],
        ),
        "",
        f"Descriptive supported macro mask AP@0.50:0.95: "
        f"**{_number(result['supported_macro_mask']['value'])}**.",
        "",
        "## 7. Segmenter canonical bounding-box results",
        "",
        "The segmenter's boxes are **its own predicted boxes**, never re-derived from its masks: "
        "re-deriving them would improve their geometric consistency with the mask branch and "
        "would then be measuring a post-processing choice this project invented rather than the "
        "model. They go through the **same** canonical bbox evaluator the detector's boxes go "
        "through.",
        "",
        *_table(
            ["Metric", "Value"],
            [
                [
                    "`S1_CANONICAL_TEST_BOX_MAP50_95`",
                    f"**{_number(box['S1_CANONICAL_TEST_BOX_MAP50_95'])}**",
                ],
                ["`S1_CANONICAL_TEST_BOX_MAP50`", _number(box["S1_CANONICAL_TEST_BOX_MAP50"])],
                ["Detections scored", f"{box['detections_scored']}"],
                ["Boxes derived from masks", "no"],
            ],
        ),
        "",
        *_table(["Class", "AP@0.50:0.95", "AP@0.50"], _per_class_rows(box["per_class"])),
        "",
        "## 8. Detector versus segmenter localisation on the holdout",
        "",
        "`DESCRIPTIVE_ONLY`. The two models do not solve the same output task - the detector "
        "emits class, confidence and a box; the segmenter emits those **plus a mask** - so this "
        "is a description of localisation and never a ranking. **No winner is declared, no "
        "composite or weighted score exists, no significance test was run, and no model "
        "selection follows.**",
        "",
        *_table(
            ["Canonical box metric", "D2", "S1", "Delta"],
            [
                [
                    "mAP@0.50:0.95",
                    _number(comparison["detector_map50_95"]),
                    _number(comparison["segmenter_map50_95"]),
                    _delta(comparison["delta_map50_95"]),
                ],
                [
                    "mAP@0.50",
                    _number(comparison["detector_map50"]),
                    _number(comparison["segmenter_map50"]),
                    _delta(comparison["delta_map50"]),
                ],
            ],
        ),
        "",
        "### 8.1 Per-class decomposition",
        "",
        "The all-class figure is the unweighted mean of five per-class APs, so it decomposes "
        "exactly. Quoting the aggregate without this table is the error the decomposition exists "
        f"to prevent - especially because `{RARE_CLASS}` carries very little support.",
        "",
        *_table(
            ["Class", "D2 AP@0.50:0.95", "S1 AP@0.50:0.95", "Delta", "Contribution to delta"],
            [
                [
                    f"`{row['class']}`",
                    _number(row["detector"]),
                    _number(row["segmenter"]),
                    _delta(row["delta"]),
                    _delta(row["contribution"]),
                ]
                for row in payload["box_decomposition"]
            ],
        ),
        "",
        "Computational cost is **not** re-measured here. Phase 10C measured latency and "
        "inference memory under a frozen symmetric protocol on this machine; those are "
        "properties of the model, the runtime and the hardware, not of which split the images "
        "came from, and phase 11B ran no benchmark of any kind.",
        "",
    ]


def _direct_iou(payload: Mapping[str, Any]) -> list[str]:
    """Render the direct instance-mask IoU diagnostic.

    Args:
        payload: The assembled report payload.

    Returns:
        Markdown lines.
    """
    result = payload["direct_iou"]
    diagnostic = result["diagnostic"]
    overall = diagnostic["global"]
    per_class = diagnostic["per_class"]
    return [
        "## 9. Direct instance-mask IoU diagnostic",
        "",
        "`SECONDARY_CANONICAL_DIAGNOSTIC`. This is the **exact phase 8C protocol, reused by "
        f"fingerprint and unchanged** (`{result['direct_iou_protocol_fingerprint']}`). No new "
        "matching rule was invented for the holdout, because inventing one now would let it be "
        "chosen with the final result in view. It scores against the canonical COCO masks at the "
        f"diagnostic's own operational confidence of {OPERATIONAL_CONF}, matched one-to-one per "
        "image and per class by `scipy.optimize.linear_sum_assignment` maximising total IoU.",
        "",
        "**It is not an average precision and it is not the primary segmenter metric.**",
        "",
        *_table(
            ["Diagnostic", "Value"],
            [
                ["`matched_mask_iou_mean`", f"**{_number(overall['matched_mask_iou_mean'])}**"],
                [
                    "`gt_normalized_mask_iou`",
                    f"**{_number(overall['gt_normalized_mask_iou'])}**",
                ],
                ["`gt_match_coverage`", _number(overall["gt_match_coverage"])],
                ["`gt_iou50_coverage`", _number(overall["gt_iou50_coverage"])],
                ["`gt_iou75_coverage`", _number(overall["gt_iou75_coverage"])],
                ["Canonical instances", f"{overall['gt_count']}"],
                ["Predictions", f"{overall['prediction_count']}"],
                ["Matched", f"{overall['matched_count']}"],
                ["Unmatched ground truth", f"{overall['unmatched_gt']}"],
                ["Unmatched predictions", f"{overall['unmatched_predictions']}"],
            ],
        ),
        "",
        "**The two headlines are not interchangeable.** `matched_mask_iou_mean` describes mask "
        "quality *where the model produced an overlapping same-class instance*; "
        "`gt_normalized_mask_iou` divides the same IoU sum by *every* canonical instance, so the "
        "misses lower it. Neither may be quoted as the project's IoU without saying which.",
        "",
        *_table(
            ["Class", "GT", "Pred", "Matched", "matched IoU", "GT-normalised", "IoU>=0.50"],
            [
                [
                    f"`{name}`",
                    f"{per_class.get(name, {}).get('gt_count', 0)}",
                    f"{per_class.get(name, {}).get('prediction_count', 0)}",
                    f"{per_class.get(name, {}).get('matched_count', 0)}",
                    _number(per_class.get(name, {}).get("matched_mask_iou_mean")),
                    _number(per_class.get(name, {}).get("gt_normalized_mask_iou")),
                    _number(per_class.get(name, {}).get("gt_iou50_coverage")),
                ]
                for name in CLASSES
            ],
        ),
        "",
    ]


def _confusion(payload: Mapping[str, Any]) -> list[str]:
    """Render both models' frozen confusion matrices.

    Args:
        payload: The assembled report payload.

    Returns:
        Markdown lines.
    """
    lines = [
        "## 10. Confusion matrix",
        "",
        "The semantics were **read from the installed framework** in phase 11A rather than "
        "assumed, and they are the exact values that produced every committed validation matrix "
        "in this repository: `ultralytics.utils.metrics.ConfusionMatrix`, confidence "
        f"{payload['detector']['confusion_matrix']['conf']}, IoU "
        f"{payload['detector']['confusion_matrix']['iou_threshold']}, class-agnostic matching "
        "with the class pair then recorded, a 6x6 matrix with **rows predicted and columns "
        "ground truth**, and the final row and column the unmatched background bucket. A matched "
        "pair whose classes disagree lands off-diagonal and counts as **both** a false positive "
        "and a false negative, which is the framework's own behaviour and is not modified here.",
        "",
        "Neither threshold was altered after the results were seen.",
        "",
    ]
    for key, title in (("detector", "Detector (D2)"), ("segmenter", "Segmenter (S1)")):
        matrix = payload[key]["confusion_matrix"]
        labels = matrix["labels"]
        lines.extend(
            [
                f"### 10.{1 if key == 'detector' else 2} {title}",
                "",
                *_table(
                    ["predicted \\ ground truth", *[f"`{label}`" for label in labels]],
                    [
                        [f"`{labels[index]}`", *[f"{value}" for value in row]]
                        for index, row in enumerate(matrix["matrix"])
                    ],
                ),
                "",
            ]
        )
    return lines


def _object_level(payload: Mapping[str, Any]) -> list[str]:
    """Render object-level TP/FP/FN and the frozen failure taxonomy.

    Args:
        payload: The assembled report payload.

    Returns:
        Markdown lines.
    """
    detector = payload["detector"]["object_level"]
    segmenter = payload["segmenter"]["object_level"]
    return [
        "## 11. Object-level TP / FP / FN",
        "",
        "Frozen **separately** from the confusion matrix and not to be conflated with it: "
        f"class-aware, IoU {detector['iou_threshold']}, one-to-one per image and per class, at "
        f"the operational confidence {detector['confidence_threshold']}, assigned greedily by "
        "descending confidence and then highest IoU. The qualitative work needs per-instance "
        "outcomes, not a matrix cell.",
        "",
        *_table(
            ["", "TP", "FP", "FN", "Ground truth", "Precision", "Recall"],
            [
                [
                    "**D2**",
                    f"{detector['true_positives']}",
                    f"{detector['false_positives']}",
                    f"{detector['false_negatives']}",
                    f"{detector['ground_truth']}",
                    _number(detector["precision"]),
                    _number(detector["recall"]),
                ],
                [
                    "**S1**",
                    f"{segmenter['true_positives']}",
                    f"{segmenter['false_positives']}",
                    f"{segmenter['false_negatives']}",
                    f"{segmenter['ground_truth']}",
                    _number(segmenter["precision"]),
                    _number(segmenter["recall"]),
                ],
            ],
        ),
        "",
        "### 11.1 Frozen segmentation failure taxonomy",
        "",
        "The four categories are evaluated **in the order listed, first match wins**, and they "
        "explicitly do **not** partition every instance - the remainder is "
        "`WELL_HANDLED_INSTANCE`. No category was added in response to what the holdout "
        "contained.",
        "",
        *_table(
            ["Category", "D2", "S1"],
            [
                [
                    f"`{name}`",
                    f"{detector['taxonomy_census'][name]}",
                    f"{segmenter['taxonomy_census'][name]}",
                ]
                for name in (
                    "DETECTION_MISS",
                    "CLASSIFICATION_MISMATCH",
                    "LOCALIZATION_FAILURE",
                    "MASK_QUALITY_FAILURE",
                    "WELL_HANDLED_INSTANCE",
                )
            ],
        ),
        "",
        "`MASK_QUALITY_FAILURE` is structurally unavailable to the detector, which emits no "
        "mask; its zero is a property of the output type, not a performance statement.",
        "",
    ]


def _qualitative(payload: Mapping[str, Any]) -> list[str]:
    """Render the deterministic qualitative selection.

    Args:
        payload: The assembled report payload.

    Returns:
        Markdown lines.
    """
    lines = [
        "## 12. Deterministically selected qualitative FP/FN examples",
        "",
        "**No holdout image was browsed before selection.** The examples were chosen by the rule "
        "phase 11A froze, from the persisted predictions and the canonical annotations alone: six "
        "categories, three examples each, each ranked by a declared quantity in a declared "
        "direction, with a tie-breaking chain that ends in an identifier so the order is total on "
        "any machine. One canonical instance appears in at most one category. An underfilled "
        "category publishes what it has and records the shortfall; it is never topped up from "
        "another. Human interpretation happens **after** the ranking, never before it.",
        "",
        f"Qualitative selection fingerprint: `{payload['qualitative_selection_sha256']}`.",
        "",
        "The selected instances and the rendered figures are deliberately **not committed**: a "
        "holdout image identifier in this repository would be a leak, and holdout imagery more "
        "so. They exist under the git-ignored run directory for human review. The frozen "
        "protocol lists qualitative figures among its permitted outputs *and* prohibits "
        "committing holdout identifiers or imagery; where those two clauses meet, the "
        "prohibition wins and the tension is recorded rather than silently resolved.",
        "",
    ]
    for key, title in (("detector", "Detector (D2)"), ("segmenter", "Segmenter (S1)")):
        selection = payload[key]["qualitative"]
        rows = []
        for category in QUALITATIVE_CATEGORIES:
            block = selection["categories"][category]
            if not block["applicable"]:
                rows.append([f"`{category}`", "n/a", "n/a", "not applicable to this model"])
                continue
            values = ", ".join(f"{float(entry['rank_value']):.6f}" for entry in block["selected"])
            shortfall = selection["shortfalls"].get(category)
            note = (
                f"shortfall: {shortfall['selected']} of {shortfall['quota']}"
                if shortfall
                else "quota met"
            )
            rows.append(
                [
                    f"`{category}`",
                    f"{block['available']}",
                    f"{len(block['selected'])}",
                    f"{note}" + (f" - ranked values {values}" if values else ""),
                ]
            )
        lines.extend(
            [
                f"### 12.{1 if key == 'detector' else 2} {title}",
                "",
                *_table(["Category", "Qualifying", "Selected", "Note"], rows),
                "",
            ]
        )
    return lines


def _validation_versus_test(payload: Mapping[str, Any]) -> list[str]:
    """Render the bounded validation-versus-test comparison.

    Args:
        payload: The assembled report payload.

    Returns:
        Markdown lines.
    """
    return [
        "## 13. Relationship to the validation conclusions",
        "",
        f"`{DESCRIPTIVE}`. Only metrics that **already existed before the holdout was read**, "
        "their holdout counterparts and the absolute difference. No new metric was invented for "
        "this comparison, no subgroup was mined, and **no significance test is reported, because "
        "none was predeclared and none may be added afterwards**. Each model was trained once and "
        "evaluated once per split, so run-to-run variance is UNKNOWN and a gap of any size is an "
        "observation rather than a result.",
        "",
        *_table(
            ["Metric", "Validation", "Test", "Absolute difference"],
            [
                [
                    row["metric"],
                    _number(row["validation"]),
                    _number(row["test"]),
                    _number(row["absolute_difference"]),
                ]
                for row in payload["validation_versus_test"]
            ],
        ),
        "",
        "The validation figures are quoted from their committed artifacts by field. They were "
        "produced by the same evaluators and the same inference settings as their holdout "
        "counterparts, which is what makes the pair comparable at all; where an evaluator or a "
        "confidence differs, the two numbers are not placed in the same row.",
        "",
        "**Why a gap exists in either direction is UNKNOWN.** Nothing here tested an explanation, "
        "and no experiment may now be run to produce one for a reported holdout number.",
        "",
    ]


def _limitations(payload: Mapping[str, Any]) -> list[str]:
    """Render the final limitations and the no-tuning statement.

    Args:
        payload: The assembled report payload.

    Returns:
        Markdown lines.
    """
    support = payload["support"]
    rare = support[RARE_CLASS]
    return [
        "## 14. Final limitations",
        "",
        f"- **`{RARE_CLASS}` carries {rare['instances']} instances over {rare['images']} holdout "
        f"image(s)** and is classified `{rare['status']}` by the unchanged phase 7A rule. Its "
        "per-class figures are reported in full and in exactly the same detail as every other "
        "class, and they decide nothing. No support threshold was invented or relaxed after the "
        "holdout's support was seen.",
        "- **Each model was trained once and evaluated once per split.** Run-to-run variance is "
        "UNKNOWN on this setup, so no difference reported here is an effect size and none is "
        "backed by a significance test.",
        "- **The holdout holds 65 images.** Every figure carries the sampling uncertainty of a "
        "set that size, and per-class figures carry more of it than the aggregates.",
        "- **Precision and recall depend on an operating point** and are reported at the "
        "project's frozen operational confidence. They are not threshold-independent properties "
        "of either model, and the framework's own F1-maximising pair is a different quantity that "
        "is never differenced against them.",
        "- **The split is a group-aware and class-aware constrained split**, not a perfectly "
        "stratified one. Nothing establishes that two images in different splits do not share a "
        "site, a day, a camera or a worker, so the holdout estimates generalisation to held-out "
        "images from this dataset, not to a new construction site.",
        "- **The canonical and the native framework metrics are different quantities** produced "
        "by different implementations against different ground-truth documents at different "
        "confidences. They are never differenced.",
        "- **Computational cost was not measured on the holdout.** The committed evidence remains "
        "phase 10C's controlled local hardware benchmark, valid for that machine, that runtime "
        "and that protocol.",
        "- **The segmentation adapter is lossy.** Phase 8A measured its round-trip at mean mask "
        "IoU 0.973066 with no instance exact; some portion of the segmenter's mask error is "
        "attributable to it, and nothing here separates the two.",
        "",
        "## 15. No test-driven tuning",
        "",
        "**Nothing in this repository was tuned, selected or changed in response to a holdout "
        "number.** The holdout was read once, under a protocol frozen before it could be read. "
        "Specifically, during and after this phase:",
        "",
        "- no model was trained, fine-tuned or retrained;",
        "- no model, architecture or checkpoint was selected;",
        "- no hyperparameter was changed;",
        "- no threshold was tuned or swept, and no operating point was chosen with a result in "
        "view;",
        "- no dataset cleaning, annotation correction or class regrouping was motivated by a test "
        "outcome;",
        "- no inference was re-run, and no second attempt exists;",
        "- no metric was added after the results were seen, and no evaluator was swapped;",
        "- no qualitative example was replaced for being unattractive or inconvenient;",
        "- no winner was declared, and no composite, weighted or aggregate score was computed.",
        "",
        f"`{OBSERVED}`. Model selection, hyperparameter tuning, threshold tuning and "
        "performance-motivated data cleaning are **CLOSED**. The results above may inform "
        "reporting, discussion and limitations. They may not trigger a D3, an S2, retraining, "
        "threshold optimisation, class regrouping, data filtering or a new model selection "
        "within the reported experiment.",
        "",
    ]


def render(payload: Mapping[str, Any]) -> str:
    """Render the complete phase 11B report.

    Args:
        payload: The assembled report payload.

    Returns:
        The Markdown document.
    """
    lines: list[str] = []
    lines.extend(_policy(payload))
    lines.extend(_models(payload))
    lines.extend(_population(payload))
    lines.extend(_detector(payload))
    lines.extend(_segmenter(payload))
    lines.extend(_direct_iou(payload))
    lines.extend(_confusion(payload))
    lines.extend(_object_level(payload))
    lines.extend(_qualitative(payload))
    lines.extend(_validation_versus_test(payload))
    lines.extend(_limitations(payload))
    lines.extend(
        [
            "---",
            "",
            "### Artifact fingerprints",
            "",
            *_table(
                ["Artifact", "Fingerprint"],
                [
                    [
                        "`detector_test_prediction_sha256`",
                        f"`{payload['fingerprints']['detector_test_prediction_sha256']}`",
                    ],
                    [
                        "`segmenter_test_prediction_sha256`",
                        f"`{payload['fingerprints']['segmenter_test_prediction_sha256']}`",
                    ],
                    [
                        "`detector_test_result_sha256`",
                        f"`{payload['fingerprints']['detector_test_result_sha256']}`",
                    ],
                    [
                        "`segmenter_test_result_sha256`",
                        f"`{payload['fingerprints']['segmenter_test_result_sha256']}`",
                    ],
                    [
                        "`direct_iou_test_result_sha256`",
                        f"`{payload['fingerprints']['direct_iou_test_result_sha256']}`",
                    ],
                    [
                        "`qualitative_selection_sha256`",
                        f"`{payload['fingerprints']['qualitative_selection_sha256']}`",
                    ],
                    ["protocol fingerprint", f"`{payload['protocol_fingerprint']}`"],
                ],
            ),
            "",
            "Every reported figure derives from the two persisted prediction sets those "
            "fingerprints cover. A report whose prediction fingerprint does not match the "
            "ledger's is not a rebuild; it is a second run.",
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"
