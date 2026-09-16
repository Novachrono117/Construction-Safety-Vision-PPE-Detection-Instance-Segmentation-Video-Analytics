"""Readable Phase 12C evidence and offline validation of its deterministic selections."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from construction_safety_vision.data.canonical import scan_for_sensitive
from construction_safety_vision.data.split_freeze import load_frozen_splits
from construction_safety_vision.delivery_status import historical_changes
from construction_safety_vision.provenance import sha256_file
from construction_safety_vision.qualitative_gallery import (
    CHECKPOINTS,
    COMPLETE,
    CONFIG,
    LICENSE_URL,
    PROVENANCE,
    REPORT,
    SOURCE_URL,
    choose_examples,
    choose_hero,
    choose_masks,
    fingerprint,
    load_policy,
    mask_category,
    match_objects,
)


def example_label(row: dict[str, Any] | None) -> str:
    """Identify an example without an ambiguous image-only reference."""
    if row is None:
        return "NO_VALID_EXAMPLE"
    key = (
        f"prediction {row['prediction_index']}"
        if row["category"] == "FALSE_POSITIVE"
        else f"GT {row['annotation_id']}"
    )
    return f"`{row['image_id']}` / {key}"


def render_report(payload: dict[str, Any]) -> str:
    """Write observations strictly from the chosen examples and measured local evidence."""
    lines = [
        "# Validation qualitative FP/FN and mask gallery",
        "",
        f"Phase 12C · **{payload['classification']}** · validation only",
        "",
        "These are deterministic illustrations from the complete frozen validation population. "
        "They do not reopen model selection or tuning, and are not new aggregate performance "
        "results.",
        "",
        "## Reading the figures",
        "",
        "Cyan dashed boxes identify canonical ground truth; solid amber boxes identify "
        "predictions. "
        "The selected error has a thicker outline. Panels show all ground-truth and predicted "
        "boxes "
        "of the named class, on the entire source canvas. In S1 FP panels the selected "
        "predicted mask "
        "is translucent. FN panels show the ground-truth object even when no prediction exists.",
        "",
        "[D2 detector FP/FN gallery](figures/qualitative/validation_fp_fn_d2.png) · "
        "[S1 segmenter FP/FN gallery](figures/qualitative/validation_fp_fn_s1.png)",
        "",
        "![D2 validation FP/FN gallery](figures/qualitative/validation_fp_fn_d2.png)",
        "",
        "![S1 validation FP/FN gallery](figures/qualitative/validation_fp_fn_s1.png)",
        "",
        "## Evidence and fixed operating point",
        "",
        f"- Complete population: {payload['population_images']} validation images / "
        f"{payload['population_annotations']} canonical annotations.",
        "- Frozen D2 / YOLO11n and S1 / YOLO11n-seg, both imgsz 768. S1 was trained with "
        "mask_ratio 4 and overlap_mask false.",
        "- Confidence 0.25, NMS IoU 0.70, max_det 300, FP32, batch 1, augment/TTA false; S1 "
        "retina_masks true.",
        "- One-to-one, class-aware box matching at IoU 0.50. Predictions are consumed by "
        "descending confidence, "
        "then original per-image prediction index; highest IoU wins, with canonical annotation "
        "ID breaking GT ties.",
        "- The delivery matcher adapts the established object_level_outcomes greedy loop "
        "without invoking "
        "a holdout execution route or computing aggregate precision/recall. Synthetic parity "
        "tests cover its semantics.",
        "- FP rank: confidence descending, prediction index ascending, image ID ascending. "
        "FN rank: canonical object area descending, annotation ID ascending, image ID ascending.",
        "- Existing validation comparison artifacts contain aggregates and spatial records, "
        "not complete "
        "D2/S1 predicted boxes/masks. Therefore MODEL_INFERENCE_REQUIRED=true. The new caches are "
        "DELIVERY_VISUALIZATION_ONLY and remain ignored under artifacts/qualitative_validation/.",
        "",
        "| Model | predict() calls | Validation images | Forward calls including framework "
        "warmup | Effective input |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for model, state in payload["execution"]["models"].items():
        lines.append(
            f"| {model} | {state['predict_invocations']} | {state['images_completed']} | "
            f"{state['forward_calls_including_framework_warmup']} | "
            f"{', '.join(state['effective_input_dtypes'])} |"
        )
    if "prior_execution" in payload["execution"]:
        lines += [
            "",
            "### Disclosed delivery execution correction",
            "",
            "The table above describes the accepted cached pass. Before it, **one predict() call "
            "per model** used a list of 65 paths. The pinned framework routes lists through "
            "LoadPilAndNumpy, which batches the whole list despite batch=1. The recorded two "
            "forward hooks per model are consistent with that route (warmup plus one image batch). "
            "Those initial predictions are archived and excluded from every final "
            "gallery selection.",
            "",
            "The input route was corrected to the verified validation directory; its loader "
            "honors batch=1, and a new forward hook now rejects any other effective batch size. "
            "A synthetic loader regression test preceded the correction pass. "
            "**Total phase counts: D2=2 and S1=2 predict() invocations**, each covering "
            "65 validation images. Both executions "
            "are retained in provenance. This was an engineering correction to the predeclared "
            "batch contract, not a result-driven rerun: no checkpoint, confidence, NMS, image "
            "size, selection rule or scientific result changed. No initial example was retained "
            "by aesthetic choice; all final selections derive from the corrected "
            "complete population.",
            "",
            "Classification of excluded execution: DELIVERY_INPUT_BATCHING_DEVIATION_NOT_USED.",
            "",
            "The corrected execution also emitted an NMS time-limit warning. In the pinned "
            "implementation, the time guard occurs after assigning the current image's output; "
            "with effective batch 1 there is no subsequent image in that batch to omit. All "
            "65 image outputs per model were persisted. The time limit and NMS parameters were "
            "not changed, and no rerun was performed to remove the warning.",
        ]
    lines += [
        "",
        "Forward calls include the framework's setup warmup, explicitly recorded; they are not "
        "additional predict() invocations or additional selected-image passes. Neither model "
        "was trained.",
        "",
        "## Per-class availability and selected examples",
        "",
        "Availability counts describe candidate pools for illustration, not a headline metric. "
        "A missing error type is reported as NO_VALID_EXAMPLE; it is never fabricated.",
        "",
        "| Model | Class | Error | Available | Selected example | Descriptive observation |",
        "| --- | --- | --- | ---: | --- | --- |",
    ]
    for model, selections in payload["selections"].items():
        for selection in selections:
            r = selection["example"]
            if r is None:
                observation = "No qualifying error at this operating point."
            elif r["category"] == "FALSE_POSITIVE":
                observation = (
                    f"Prediction at confidence {r['score']:.3f} remains unmatched under "
                    "the one-to-one rule. This does not assert that no real object exists."
                )
            else:
                observation = (
                    "This canonical object has no assigned same-class prediction at box IoU 0.50."
                )
                if r.get("classification_mismatch"):
                    observation += (
                        " An overlapping prediction reaches 0.50 only with a different class "
                        "(CLASSIFICATION_MISMATCH)."
                    )
            lines.append(
                f"| {model} | {selection['class']} | {selection['category']} | "
                f"{selection['available']} | {example_label(r)} | {observation} |"
            )
    lines += [
        "",
        "**FACT:** the selected examples expose unmatched predictions and missed canonical "
        "objects. "
        "**UNKNOWN:** the causes of these errors. No claim about lighting, occlusion, training "
        "mechanisms or causal superiority follows from these images. **HYPOTHESIS (untested):** "
        "localization or class ambiguity could contribute to an individual unmatched case; the "
        "present delivery phase does not test that explanation.",
        "",
        "### vest_loose",
        "",
        "The frozen validation split contains one image and eight instances of vest_loose. "
        "Its selected failures are shown under exactly the same rules. Absence of a D2 FP, if "
        "reported, must not be read as success when objects are missed. This support is too small "
        "for model ranking or a general class-level claim.",
        "",
        "## Mask-specific evidence",
        "",
        "![Mask quality examples](figures/qualitative/mask_quality_gallery.png)",
        "",
        "Pairs first satisfy the same class-aware box matching rule. The Phase 8D mask bands "
        "are reused: HIGH_QUALITY_MASK (IoU >= 0.75), LOW_OVERLAP_MASK (0 < IoU < 0.50), "
        "and MODERATE_MASK (0.50 <= IoU < 0.75). The existing area tolerance is 25%. "
        "GOOD_MASK_MATCH is a display alias for HIGH_QUALITY_MASK. Under/over-coverage are "
        "display aliases for the existing under/oversegmentation candidate flags on lower-quality "
        "matched masks. They describe area disagreement, not a causal diagnosis or a new taxonomy.",
        "",
        "Good-mask rank is IoU descending then canonical area descending; under/over rank by "
        "signed relative area error ascending/descending. Annotation ID, image ID and prediction "
        "index break ties. The two coverage examples need not be PPE; person is a required "
        "class too.",
        "",
        "| Display | Example | Class | Mask IoU | Relative mask-area error |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for selected in payload["mask_selections"]:
        r = selected["example"]
        if r:
            lines.append(
                f"| {selected['category']} | {example_label(r)} | {r['class']} | "
                f"{r['mask_iou']:.6f} | {r['relative_area_error']:+.6f} |"
            )
        else:
            lines.append(f"| {selected['category']} | NO_VALID_EXAMPLE | - | - | - |")
    lines += [
        "",
        "In the disagreement panel cyan marks canonical foreground missing from the prediction; "
        "amber marks predicted foreground outside the canonical mask. Boxes cannot directly "
        "express that foreground support. These examples do not establish globally superior "
        "localization.",
        "",
        "## Box-versus-mask hero candidate",
        "",
        f"Status: **{payload['hero']['status']}**. No manual replacement occurred.",
        "",
        "Eligibility was declared before inference: the same non-person PPE instance must be "
        "correctly box-matched by both models; S1 mask IoU >= 0.75; GT area >= 0.5% of the "
        "image; mask/predicted-box fill between 0.10 and 0.85; image width >= 640. Rank by "
        "fewest total D2/S1 FP+FN on the scene, largest relative GT area, highest mask IoU, "
        "then annotation/image/prediction IDs. This intentionally selects a readable successful "
        "illustration, not a representative estimate of average performance. All predictions "
        "on the selected scene remain visible.",
        "",
    ]
    if payload["hero"]["example"]:
        lines += [
            "![Box versus mask candidate](figures/qualitative/box_vs_mask_hero_candidate.png)",
            "",
            f"Selected anchor: {example_label(payload['hero']['example'])}. "
            f"Eligible candidates: {payload['hero']['eligible_candidates']}. Final README hero "
            f"selection belongs to a later phase.",
            "",
        ]
    lines += [
        "## Requirement and delivery status",
        "",
        f"GAP-005: **{payload['gap_005_status']}** (previously OPEN). Assignment R13: "
        f"**{payload['assignment_r13_status']}**.",
        "",
        "Coverage is assessed per class and error across the two required systems; every "
        "model/class/error "
        "slot is still attempted and disclosed. If either FP or FN is unavailable across both "
        "models "
        "for any class, strict per-class coverage remains partial. The presentation work can be "
        "complete with that explicitly justified evidence limitation. This does not change the "
        "immutable Phase 12A audit or mark unrelated academic deliverables complete.",
        "",
        "## Existing confusion evidence",
        "",
        "- [D2 validation confusion matrix](figures/detection/D2/confusion_matrix.png)",
        "- [S1 validation confusion matrix](figures/segmentation_S1/confusion_matrix.png)",
        "- [Existing final-test report](final_test_evaluation.md) for already published "
        "aggregate evidence only.",
        "",
        "No confusion matrix was recomputed. Historical framework confusion semantics differ "
        "from this gallery's box-IoU 0.50 matching and must not be equated.",
        "",
        "## Attribution and reuse",
        "",
        f"Dataset: [Construction PPE Compliance Detection, agis-workspace-8gs52 / Roboflow "
        f"Universe]({SOURCE_URL}). "
        f"Images and annotations are licensed under [CC BY 4.0]({LICENSE_URL}), as recorded in "
        "[dataset provenance](dataset_provenance.md). Source image coordinates and canonical "
        "annotations follow the frozen live-source decision; the v4 export is the acquisition "
        "reference.",
        "",
        "Changes: GT and model overlays, mask-disagreement overlays, rescaling for figure layout; "
        "no source-scene cropping. Credit and a modification notice accompany every PNG; this "
        "report and PNG metadata provide the source and license links. No endorsement is implied. "
        "The source imagery is not relicensed under the repository's AGPL-3.0 code license. "
        "This use follows the official CC BY sharing/adaptation terms.",
        "",
        "## Rebuilding and verification",
        "",
        "```text",
        "uv run python scripts/build_qualitative_gallery.py",
        "uv run python scripts/validate_qualitative_gallery.py",
        "uv run python scripts/validate_qualitative_gallery.py --with-data",
        "```",
        "",
        "The builder defaults to the existing caches and never executes a model. The one-time "
        "`--infer` route refuses an existing execution ledger, including a failed one. "
        "The validator's default mode reads committed metadata only; `--with-data` verifies "
        "the validation caches, annotations and image hashes and re-derives selections, "
        "without inference.",
        "",
        "Every image is admitted through the frozen validation accessor. The verified split is "
        "a disjoint partition, so positive validation membership excludes test without retrieving "
        "test IDs. No holdout image, annotation, prediction or final-test accessor is used. "
        "Per-instance box/mask overlaps exist solely to classify/render these cases; no new "
        "headline metric, model comparison, training or tuning was performed.",
        "",
    ]
    return "\n".join(lines)


def validate_gallery(root: Path, *, verify_provenance: bool = True) -> list[str]:
    """Re-derive selected errors from committed box evidence and verify source membership."""
    problems = []
    payload = json.loads((root / REPORT).read_text(encoding="utf-8"))
    policy = load_policy(root / CONFIG)
    if payload["policy"] != policy or payload["policy_sha256"] != fingerprint(policy):
        problems.append("Gallery policy mismatch")
    prior = payload["execution"].get("prior_execution")
    if prior is not None:
        if prior["policy_sha256"] != fingerprint(policy):
            problems.append("Initial execution did not use the same selection policy")
        for model, digest in CHECKPOINTS.items():
            state = prior["models"][model]
            if (
                state["checkpoint_sha256"] != digest
                or state["predict_invocations"] != 1
                or state["images_completed"] != 65
            ):
                problems.append("Initial execution accounting mismatch")
        correction = payload["execution"]["execution_correction"]
        if correction["initial_predictions_used_in_gallery"] is not False:
            problems.append("Excluded batching-deviation predictions were used")
    allowed = set(
        load_frozen_splits(root / "reports/split_manifest.json").image_ids(
            "validation", purpose="Phase 12C metadata membership validation"
        )
    )
    if payload["split"] != "validation" or set(payload["images"]) != allowed:
        problems.append("Source population differs from frozen validation")
    for stem, image in payload["images"].items():
        expected_path = f"data/processed/canonical/images/validation/{image['file_name']}"
        if (
            image["split"] != "validation"
            or image["source_image_id"] != stem
            or Path(image["file_name"]).stem != stem
            or Path(image["file_name"]).name != image["file_name"]
            or image["path"] != expected_path
        ):
            problems.append("Invalid validation source metadata or path")
    records = {
        m: match_objects(payload["ground_truth"], p, allowed)
        for m, p in payload["predictions"].items()
    }
    for model in CHECKPOINTS:
        if choose_examples(records[model]) != payload["selections"][model]:
            problems.append(f"{model}: FP/FN selection or matching changed")
        state = payload["execution"]["models"][model]
        if state["checkpoint_sha256"] != CHECKPOINTS[model] or state["status"] != "COMPLETE":
            problems.append(f"{model}: frozen identity/execution mismatch")
        if state["images_completed"] != len(allowed) or state["predict_invocations"] != 1:
            problems.append(f"{model}: invocation accounting mismatch")
        if state["effective_input_dtypes"] != ["torch.float32"] or state["autocast_enabled"]:
            problems.append(f"{model}: FP32 evidence missing")
        if state.get("effective_input_batch_sizes") != [1]:
            problems.append(f"{model}: effective batch=1 evidence missing")
    pairs = {(r["image_id"], r["annotation_id"]): r for r in payload["mask_pairs"]}
    matched = {
        (r["image_id"], r["annotation_id"])
        for r in records["S1"]
        if r["category"] == "TRUE_POSITIVE"
    }
    if set(pairs) != matched or len(pairs) != len(payload["mask_pairs"]):
        problems.append("Mask pair membership incomplete")
    for r in records["S1"]:
        if r["category"] != "TRUE_POSITIVE":
            continue
        pair = pairs.get((r["image_id"], r["annotation_id"]))
        if pair is None:
            continue
        if any(pair.get(k) != v for k, v in r.items()):
            problems.append("Mask pair does not identify its authoritative box match")
        g, p, intersection, union = (
            pair[k]
            for k in (
                "gt_mask_pixels",
                "predicted_mask_pixels",
                "intersection_pixels",
                "union_pixels",
            )
        )
        if not 0 <= intersection <= min(g, p) or union != g + p - intersection:
            problems.append("Invalid per-instance mask evidence")
        iou, relative = intersection / union, (p - g) / g
        if not math.isclose(iou, pair["mask_iou"]) or not math.isclose(
            relative, pair["relative_area_error"]
        ):
            problems.append("Mask diagnostic arithmetic mismatch")
        if mask_category(iou, relative) != pair["mask_category"]:
            problems.append("Mask display classification mismatch")
        r.update(pair)
    if choose_masks(records["S1"]) != payload["mask_selections"]:
        problems.append("Mask selection changed")
    if choose_hero(records, payload["images"], policy["hero"]) != payload["hero"]:
        problems.append("Hero selection changed")
    coverage = all(
        any(
            s["status"] == "SELECTED"
            for rows in payload["selections"].values()
            for s in rows
            if s["class"] == name and s["category"] == category
        )
        for name in ("helmet_loose", "helmet_on_head", "person", "vest_loose", "vest_on_body")
        for category in ("FALSE_POSITIVE", "FALSE_NEGATIVE")
    )
    if payload["gap_005_status"] != ("RESOLVED" if coverage else "PARTIALLY_RESOLVED"):
        problems.append("GAP-005 coverage status is unsupported")
    if payload["assignment_r13_status"] != ("COMPLETE" if coverage else "PARTIAL"):
        problems.append("Assignment coverage status is unsupported")
    for field in (
        "scientific_results_modified",
        "headline_metrics_computed",
        "holdout_content_accessed",
        "manual_example_replacement",
    ):
        if payload[field] is not False:
            problems.append(f"Invalid scientific boundary: {field}")
    if (
        payload["attribution"]["source"] != SOURCE_URL
        or payload["attribution"]["license_url"] != LICENSE_URL
    ):
        problems.append("Dataset attribution missing")
    if payload["attribution"]["image_license_is_agpl"] is not False:
        problems.append("Source imagery must remain CC BY 4.0")
    if historical_changes(root):
        problems.append("Historical scientific artifact changed")
    if payload["classification"] != COMPLETE:
        problems.append("Gallery incomplete")
    for model in CHECKPOINTS:
        if not any(s["status"] == "SELECTED" for s in payload["selections"][model]):
            problems.append(f"{model}: no visual error evidence")
    for relative in (REPORT, "reports/qualitative_validation_gallery.md"):
        problems.extend(scan_for_sensitive((root / relative).read_text(encoding="utf-8")))
    if render_report(payload) != (root / "reports/qualitative_validation_gallery.md").read_text(
        encoding="utf-8"
    ):
        problems.append("Report differs from generated observations")
    if verify_provenance:
        provenance = json.loads((root / PROVENANCE).read_text(encoding="utf-8"))
        # Approved Phase 12C provenance remains immutable as live delivery files
        # evolve. Its gallery assets and selection code still verify on disk.
        later_delivery_paths = {
            "README.md",
            "reports/README.md",
            "reports/roadmap.md",
            "reports/delivery_gap_resolution_status.json",
            "src/construction_safety_vision/delivery_status.py",
            "src/construction_safety_vision/qualitative_gallery_report.py",
        }
        for entry in provenance["inputs"] + provenance["outputs"]:
            relative = entry["path"]
            if relative.startswith(("data/", "artifacts/")):
                continue  # --with-data is the explicit local cache/content check.
            if relative in later_delivery_paths:
                body = subprocess.check_output(
                    ["git", "show", f"fd8c13581e31bbadd4369ae5b0a56f227a03d934:{relative}"],
                    cwd=root,
                )
                digest = hashlib.sha256(body).hexdigest()
            else:
                digest = sha256_file(root / relative)
            if digest != entry["sha256"]:
                problems.append(f"Provenance digest mismatch: {relative}")
        if provenance["details"]["execution"] != payload["execution"]:
            problems.append("Execution provenance mismatch")
    return problems
