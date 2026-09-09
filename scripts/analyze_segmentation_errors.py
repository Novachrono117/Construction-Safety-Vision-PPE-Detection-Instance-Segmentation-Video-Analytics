"""Diagnose why S0's mask quality trails its box quality.

Phase 8D. It trains nothing, re-validates nothing, tunes nothing and selects no
final segmenter. It re-runs inference on the validation split under the settings
phase 8C already froze, and reports the outcome one canonical instance at a time
so that a later decision about a second experiment rests on evidence rather than
on a plausible story.

Four things are deliberate.

**The inference settings and the matching rule are not new.** They are read from
the committed phase 8C diagnostic protocol and asserted, so this phase cannot
quietly become a threshold search. The aggregates it recomputes are checked
against the committed ones rather than replacing them: if they disagree,
something is wrong with this analysis, not with the published result.

**The taxonomy is frozen before any image is opened**, and automated mechanism
flags are kept separate from human ones. An area ratio can say a predicted mask
covers less than the canonical one; only a person can say whether the model
missed a visible region.

**The review set is chosen from the table before inspection**, by deterministic
rules recorded in the artifact. Swapping an example for a more interesting one
afterwards would turn analysis into illustration.

**Human judgements are declared as data in this file**, the way phase 4B's
recorder declares them, and validated against the review set. A judgement about
an instance the rules did not select is refused.

Requires:

* ``configs/segmentation_mask_iou_evaluation.yaml``      phase 8C
* ``reports/segmentation_S0_result_manifest.json``       phase 8C
* ``reports/segmentation_S0_mask_iou.json``              phase 8C
* ``reports/segmentation_adapter_fidelity.csv``          phase 8A
* ``data/processed/canonical/annotations/``              phase 5D

Writes:
    reports/segmentation_S0_error_analysis.json
    reports/segmentation_S0_error_analysis.md
    reports/segmentation_S0_error_instances.csv
    reports/segmentation_S0_error_analysis.provenance.json
    artifacts/segmentation/S0_error_review/                (git-ignored figures)

Usage:
    uv run python scripts/analyze_segmentation_errors.py --verify-only
    uv run python scripts/analyze_segmentation_errors.py --build-review
    uv run python scripts/analyze_segmentation_errors.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from construction_safety_vision.config import ConfigError
from construction_safety_vision.data.canonical import scan_for_sensitive
from construction_safety_vision.detection_comparison import (
    SUPPORT_MIN_INSTANCES,
    SUPPORT_MIN_POSITIVE_IMAGES,
)
from construction_safety_vision.mask_iou_evaluation import (
    GROUND_TRUTH_SOURCE,
    MATCHING_ALGORITHM,
    MaskInstance,
    load_mask_iou_config,
    match_image,
)
from construction_safety_vision.paths import ProjectPaths, long_path
from construction_safety_vision.provenance import ProvenanceRecord, git_commit, sha256_file
from construction_safety_vision.segmentation_error_analysis import (
    ADAPTER_FIDELITY_RISK,
    ADAPTER_FIDELITY_RISK_IOU,
    AREA_RATIO_TOLERANCE,
    AUTOMATIC_FLAGS,
    DETECTION_MISS,
    HIGH_QUALITY_MASK,
    IOU_HIGH,
    IOU_LOW,
    LOW_OVERLAP_MASK,
    MANUAL_ONLY_FLAGS,
    MECHANISM_FLAGS,
    MODERATE_MASK,
    OUTCOME_BANDS,
    TINY_MASK_AREA_PX,
    ErrorAnalysisError,
    InstanceRow,
    area_quartile,
    area_quartile_edges,
    automatic_flags,
    fidelity_band,
    flag_census,
    outcome_band,
    overlap_resolved_targets,
    review_identifiers,
    select_review_set,
    summarise,
    validate_manual_attributions,
)
from construction_safety_vision.segmentation_run import RUNTIME_VIEW_ROOT
from construction_safety_vision.splits import HOLDOUT_UNLOCK_ENV_VAR, holdout_unlocked

MASK_IOU_YAML = "segmentation_mask_iou_evaluation.yaml"
RESULT_MANIFEST_JSON = "segmentation_S0_result_manifest.json"
MASK_IOU_JSON = "segmentation_S0_mask_iou.json"
FIDELITY_CSV = "segmentation_adapter_fidelity.csv"

ANALYSIS_JSON = "segmentation_S0_error_analysis.json"
ANALYSIS_MD = "segmentation_S0_error_analysis.md"
INSTANCES_CSV = "segmentation_S0_error_instances.csv"
PROVENANCE_JSON = "segmentation_S0_error_analysis.provenance.json"

REVIEW_ROOT = "artifacts/segmentation/S0_error_review"

SCHEMA_VERSION = 1
PHASE = "8D"
EXPERIMENT_ID = "S0"

READY = "S1_HYPOTHESIS_READY_FOR_PROTOCOL_FREEZE"
MORE_EVIDENCE = "MORE_DIAGNOSTIC_EVIDENCE_REQUIRED"
NO_EXPERIMENT = "NO_FURTHER_SEGMENTATION_EXPERIMENT_JUSTIFIED"
BLOCKED = "BLOCKED"

FOCUS_CLASS = "person"
RARE_CLASS = "vest_loose"
RARE_CLASS_STATUS = "DESCRIPTIVE_HIGH_UNCERTAINTY"

EXPECTED_CHECKPOINT_SHA256 = "d7b512b95fafdc8658fd802459d2c87d9ad3f11f922162a774dacf8ca75b87a3"

HOLDOUT_STATUS = "PROTECTED_NOT_ACCESSED"
HOLDOUT_REASON = (
    "Phase 8D re-ran inference on the frozen validation split under the settings phase 8C "
    "froze, and inspected a deterministic sample of validation images. The holdout was not "
    "read, materialised, adapted, counted, predicted on, inspected or plotted; no holdout "
    "identifier, image, mask or statistic exists in any artifact this phase wrote."
)

# --- declared human judgements -------------------------------------------------
#
# Filled in AFTER the deterministic review set was selected and its figures were
# rendered and looked at, and validated against that set before it can be
# recorded. A judgement about an instance the rules did not select is refused, as
# is any label naming a cause this phase cannot establish.
#
# Keyed by "<image_id>#<annotation_id>".

MANUAL_ATTRIBUTIONS: dict[str, dict[str, Any]] = {
    "9uyGbpRuvoK6FWsiIhGx#305": {
        "flags": ("UNDERSEGMENTATION",),
        "evidence": (
            "A single, large, unoccluded flagger. The canonical mask covers the whole person "
            "including the high-visibility jacket; the S0 mask covers the face, the bare hand "
            "and the trousers but not the jacket, which is exactly the region a vest_on_body "
            "instance also covers. Predicted area is 82% smaller than canonical."
        ),
    },
    "9uyGbpRuvoK6FWsiIhGx#306": {
        "flags": ("OCCLUSION_ASSOCIATED",),
        "evidence": (
            "A seated operator inside a machine cab, seen through a windscreen at low contrast "
            "and small scale. No same-class prediction overlaps it at all. The failure is "
            "visibility, not mask quality."
        ),
    },
    "HvvG43BEgtvO5Az8qBX5#499": {
        "flags": ("UNDERSEGMENTATION", "OCCLUSION_ASSOCIATED", "INSTANCE_SEPARATION_ERROR"),
        "evidence": (
            "Three heavily overlapping people. The canonical mask covers the centre person "
            "fully; the assigned prediction covers mostly a neighbour plus a fragment of the "
            "centre torso. The best IoU available from any of the five person predictions is "
            "0.141, so this is not an assignment artefact - no prediction covers this person."
        ),
    },
    "hsfBHYazMXnS3WJlQzR7#1246": {
        "flags": ("UNDERSEGMENTATION", "OCCLUSION_ASSOCIATED"),
        "evidence": (
            "Two large overlapping people seen from behind. The canonical mask spans helmet, "
            "shoulders and arm; the prediction covers roughly a quarter of it, omitting the "
            "helmet and much of the jacket. Predicted area is 75% smaller than canonical."
        ),
    },
    "6fMvy2k97BSY0a3P9Fbg#173": {
        "flags": ("UNATTRIBUTED",),
        "evidence": (
            "An indoor office scene, not a construction site. The canonical person annotation "
            "covers only the top of the hair and the hands while leaving the fully visible face "
            "and torso unannotated. No listed mechanism explains the miss: the annotation "
            "itself is a fragment, which is a data characteristic rather than a model error, "
            "and the taxonomy has no flag for that."
        ),
    },
    "eK6ftnVVMa7WC57HMKWj#1124": {
        "flags": ("OVERSEGMENTATION", "INSTANCE_SEPARATION_ERROR"),
        "evidence": (
            "Another indoor office scene. The canonical annotation is a two-hand fragment of a "
            "background person; the assigned prediction is a whole foreground person at 36 "
            "times the area. The prediction is arguably the more sensible object, and the IoU "
            "of 0.027 reflects an annotation-versus-prediction mismatch of scope."
        ),
    },
}

STRATIFIED_FINDINGS: tuple[str, ...] = (
    "**Coverage and mask quality fail in opposite directions across size.** Detection coverage "
    "rises monotonically with canonical area (0.513, 0.816, 0.882, 0.895 from smallest to "
    "largest quartile), so misses concentrate in small masks. But matched mask IoU is *worst* "
    "in the largest quartile (0.642) and best in the middle two (0.751, 0.763). Large objects "
    "are reliably found and poorly delineated; small ones are missed outright.",
    "**`person` is weaker than every other class at every size.** Matched IoU by quartile, "
    "person against non-person: 0.592 vs 0.789, 0.539 vs 0.844, 0.636 vs 0.829, 0.561 vs 0.822. "
    "The deficit therefore survives size stratification and is not a size artefact.",
    "**Adapter fidelity does not track model error.** The 20 worst S0 instances have a mean "
    "phase 8A adapter IoU of 0.968 against 0.979 for the split as a whole, and the rank "
    "correlation between adapter IoU and S0 mask IoU over matched instances is 0.246 - weak and "
    "positive. Only 4 validation instances fall in the adapter-risk band at all.",
    "**Multi-component instances score lower, but the effect is largely `person`.** Matched IoU "
    "0.576 for two-or-more components against 0.756 for one. Within `person` alone the gap "
    "narrows to 0.522 against 0.602, and single-component persons still sit far below "
    "single-component non-persons - so topology is a secondary contributor, not the explanation.",
    "**Hole-carrying instances score lower (0.616 against 0.732), and this is confounded** the "
    "same way phase 8A's was: holed instances are disproportionately large and disproportionately "
    "`person`. It is reported as an association, not an effect.",
)

REVIEW_FINDINGS: tuple[str, ...] = (
    "**Six instances were inspected in detail** out of the 43 the rules selected and rendered. "
    "That is stated rather than implied: the population-level findings above rest on all 304 "
    "instances, and the review's job was to check what the numbers meant, not to supply a "
    "census of its own.",
    "**The dominant `person` failure is visible and consistent.** In the worst matched cases the "
    "predicted mask covers the face, hands and trousers but omits the high-visibility jacket - "
    "precisely the region a `vest_on_body` instance also occupies.",
    "**It is not an assignment artefact.** On the multi-person images the best IoU available "
    "from *any* prediction is already low (0.141, 0.196, 0.243, 0.263), so one-to-one matching "
    "is not manufacturing the deficit.",
    "**Two of the six sit in indoor office scenes with fragmentary `person` annotations** - one "
    "covering only hair and hands while the face is plainly visible, another covering two hands "
    "of a background figure. These are data characteristics, and the frozen taxonomy has no "
    "flag for them, so they are recorded as `UNATTRIBUTED` with the evidence written out.",
    "**Genuine occlusion accounts for some misses**, such as an operator seated behind cab "
    "glass at low contrast.",
)

ADAPTER_VERDICT: dict[str, Any] = {
    "summary": (
        "Phase 8A measured how much canonical geometry survives the YOLO label format. If that "
        "loss were driving S0's mask error, the instances S0 handles worst would tend to be the "
        "ones the adapter converted worst. They are not."
    ),
    "conclusion": (
        "**Adapter conversion is unlikely to be the main explanation.** The worst S0 instances "
        "are near-average conversions, the rank correlation is weak, `person`'s mean adapter "
        "fidelity is 0.975 - higher than the split's own worst strata - and only 4 validation "
        "instances fall in the adapter-risk band. This is not a claim that the adapter has zero "
        "effect: it loses real geometry, phase 8A quantified that, and a small share of S0's "
        "error is certainly attributable to it. What the evidence rules out is adapter loss as "
        "the dominant cause."
    ),
}

LIMITATIONS: tuple[str, ...] = (
    "**Descriptive, not causal.** No controlled experiment was run. Every mechanism named here "
    "is an association or an observation, and the one quantitative decomposition offered - "
    "re-scoring against the overlap-resolved target - changes the measuring stick, not the "
    "model.",
    "**One model, one run, one operating point.** All of it describes a single S0 checkpoint at "
    "conf 0.25 and NMS IoU 0.70. A different threshold would move the coverage figures.",
    "**Validation only, 304 instances, 65 images.** Small enough that per-class and per-stratum "
    "figures carry real sampling uncertainty, and `vest_loose`'s 8 instances carry so much that "
    "nothing about the population may be read from them.",
    "**Six instances were looked at.** The visual review confirms and illustrates the "
    "population-level measurements; it does not independently establish their frequency.",
    "**The strata are correlated.** Size, class, component count and hole presence are not "
    "independent in this dataset - large multi-component holed instances are disproportionately "
    "`person` - so no single stratum's effect is isolated.",
    "**The contested-pixel measurement was motivated by an observation made during review.** It "
    "is reported as a post-hoc, hypothesis-generating analysis over pre-existing canonical data, "
    "not as a predeclared test, and it needs a controlled experiment to become a finding.",
)

# --- declared readings of the computed evidence ---------------------------------
#
# Also filled in after the numbers existed. These are the judgement calls: what
# the strata show, what the review showed, and which future interventions the
# evidence motivates. They are separated from the computed values so a reader can
# disagree with the reading without doubting the arithmetic.

HYPOTHESES: dict[str, dict[str, str]] = {
    "OVERLAP_MASK_TARGET_MISMATCH": {
        "status": "HYPOTHESIS_WITH_THE_STRONGEST_SUPPORT",
        "supported_by": (
            "The frozen protocol sets `overlap_mask: true`. Read from the installed source, "
            "`polygons2masks_overlap` sorts instances by descending area and paints them with a "
            "running maximum, so the **smaller** instance owns any shared pixel: a vest owns the "
            "pixels of the person wearing it, and that person's training target is the "
            "remainder. Measured consequences, over all 137 validation persons: 28.0% of "
            "canonical person pixels lie inside another class's canonical mask, but **71.0% of "
            "the pixels S0 misses on person lie in that contested region** - 2.5 times the base "
            "rate. Contested fraction correlates negatively with person mask IoU (Spearman "
            "-0.547); persons under 20% contested average 0.691 matched IoU against 0.408 for "
            "those over 40%. Re-scoring the same predictions against the overlap-resolved target "
            "the model was actually trained on raises person GT-normalised IoU from 0.4346 to "
            "0.5046 and matched IoU from 0.578 to 0.671, while every other class moves by less "
            "than 0.002 - exactly as expected if the compact classes are the ones winning the "
            "contested pixels."
        ),
        "weakened_by": (
            "It does not close the gap. Even scored against its own training target, `person` "
            "matched IoU is 0.671, still far below `helmet_loose` at 0.935 and `helmet_on_head` "
            "at 0.877, so a material `person`-specific difficulty remains unexplained. The "
            "measurement is also post-hoc - motivated by an observation made while reviewing "
            "images - and re-scoring changes the measuring stick rather than the model, so it "
            "demonstrates a target/evaluation mismatch, not that removing the flag would improve "
            "anything."
        ),
    },
    "MASK_SUPERVISION_RESOLUTION": {
        "status": "HYPOTHESIS_WEAKLY_SUPPORTED",
        "supported_by": (
            "`mask_ratio: 4` downsamples the mask target fourfold before the loss sees it, so "
            "fine structure is not what is optimised. Matched mask IoU is worst in the largest "
            "area quartile (0.642), where a person spans much of the frame and thin extremities "
            "carry proportionally more of the boundary."
        ),
        "weakened_by": (
            "The dominant measured failure is whole-region exclusion, not contour imprecision: "
            "among poorly matched instances the automated pass flags 56 under-segmentations "
            "against 19 boundary errors, and for `person` specifically 49 against 14. A coarse "
            "supervision grid does not explain a prediction that omits an entire jacket. "
            "`mask_ratio` also applies identically to every class, yet the compact classes score "
            "0.88-0.94."
        ),
    },
    "MODEL_CAPACITY": {
        "status": "HYPOTHESIS_NOT_SPECIFICALLY_MOTIVATED",
        "supported_by": (
            "YOLO11n-seg is the smallest model in its family, and `person` is the most variable "
            "class in the set. Nothing here rules capacity out."
        ),
        "weakened_by": (
            "No measurement in this analysis isolates capacity, and none could: capacity is only "
            "visible by comparison against a larger model, which this phase did not run. The "
            "class the model handles worst is also the one whose training target the overlap "
            "policy alters most, which is a confound a capacity experiment would inherit. "
            "**Absence of evidence here is not evidence that capacity cannot help** - it is the "
            "absence of a reason to try it first."
        ),
    },
    "INPUT_RESOLUTION": {
        "status": "HYPOTHESIS_WEAKLY_SUPPORTED",
        "supported_by": (
            "Detection misses concentrate in the smallest area quartile (37 of 68), and higher "
            "input resolution is the conventional remedy for small-object recall."
        ),
        "weakened_by": (
            "The mask-quality failure this phase set out to explain sits at the *other* end of "
            "the size distribution: the largest quartile has 0.895 coverage and the worst matched "
            "IoU. Resolution is least likely to be the constraint on objects that already span "
            "much of the frame. Phase 7C also found the analogous resolution hypothesis "
            "unsupported by the shape of the detection result."
        ),
    },
    "DATA_OR_ANNOTATION_LIMITATION": {
        "status": "HYPOTHESIS_PARTLY_SUPPORTED",
        "supported_by": (
            "Two of the six inspected instances are indoor office scenes with fragmentary "
            "`person` annotations - one covering only hair and hands beside a fully visible "
            "unannotated face, another covering two hands of a background figure against a "
            "whole-person prediction 36 times its area. The dataset's known limitations already "
            "include out-of-domain images. Annotation scope for `person` is visibly not uniform "
            "across the split."
        ),
        "weakened_by": (
            "Two instances cannot establish a rate. No systematic audit of annotation scope was "
            "run, and phases 4 and 5 accepted the annotations with documented limitations after "
            "a human review that did not flag this. It would also not explain the population-wide "
            "contested-pixel pattern, which holds on cleanly annotated construction images."
        ),
    },
    "NO_CLEAR_INTERVENTION": {
        "status": "HYPOTHESIS_NOT_SUPPORTED",
        "supported_by": (
            "S0 is a first baseline and some of its weakness may simply be the difficulty of the "
            "task on 303 training images."
        ),
        "weakened_by": (
            "One mechanism is measured, specific, and traceable to a single frozen protocol flag "
            "whose behaviour was read from the installed source. That is a concrete lead, "
            "whatever its eventual size."
        ),
    },
}

CANDIDATES: dict[str, dict[str, str]] = {
    "mask_ratio_2": {
        "verdict": "WEAKLY_MOTIVATED",
        "reasoning": (
            "The evidence does not point here. A clean one-variable intervention needs the "
            "variable to address the measured failure, and the measured failure is whole-region "
            "exclusion concentrated in pixels the training target assigns to another instance - "
            "not contour imprecision. Under-segmentation outnumbers boundary error 56 to 19 "
            "overall and 49 to 14 within `person`. Halving the mask-target downsampling would "
            "sharpen boundaries the model is already placing roughly correctly, on the classes "
            "that are already scoring 0.88-0.94, while leaving the jacket-shaped hole in the "
            "`person` masks exactly where it is. It is not ruled out - finer supervision could "
            "help at the margin - but running it first would spend the project's one controlled "
            "comparison on the weaker lead."
        ),
        "one_variable": (
            "Yes, methodologically. `mask_ratio` changes the training target's resolution and "
            "nothing else in the frozen protocol, and it does not alter what the framework's "
            "validation compares against, so S0 and such an S1 would remain comparable on both "
            "the framework mask metric and the direct IoU diagnostic. The design is clean; the "
            "motivation is what is thin."
        ),
    },
    "yolo11s_seg": {
        "verdict": "NOT_SPECIFICALLY_MOTIVATED",
        "reasoning": (
            "Nothing measured here points at capacity. The class S0 handles worst is the class "
            "whose training target the overlap policy most alters, so a capacity experiment run "
            "now would inherit that confound and its result would be hard to read: a larger "
            "model might close part of the gap by better fitting an artefact of the target. "
            "There is also a specific precedent against reaching for capacity first - phase 7B "
            "varied exactly this on the detection side and D1 came in **below** D0 on the "
            "selection metric. That says nothing definitive about segmentation, but it does mean "
            "capacity is not a free bet in this project. **Absence of supporting evidence is not "
            "evidence that capacity cannot help**; it is a reason to sequence it after the "
            "mechanism that is measured."
        ),
        "one_variable": (
            "Yes, methodologically - it mirrors D1's design exactly, varying the model and its "
            "consequential pretrained weights and nothing else."
        ),
    },
    "overlap_mask_false": {
        "verdict": "BEST_MOTIVATED_BY_THE_EVIDENCE_BUT_NOT_A_CLEAN_COMPARISON",
        "reasoning": (
            "This is where the evidence points, and it is also where the methodology gets "
            "awkward - which is exactly why it needs a reviewed protocol rather than a decision "
            "taken here. With `overlap_mask: false` each instance gets its own mask plane, so a "
            "`person` target would include the pixels its vest currently takes, addressing the "
            "one mechanism this phase actually measured."
        ),
        "one_variable": (
            "**No, not for the framework metric.** The flag changes the training target *and* "
            "the validation target: the installed `SegmentationValidator._prepare_batch` builds "
            "its ground truth with `masks == index` when `overlap_mask` is set, so S0 and an "
            "`overlap_mask: false` S1 would be scored against different ground truth and their "
            "mask mAP would not be comparable. The direct mask-IoU diagnostic **would** remain "
            "comparable, because it always scores against canonical COCO masks that never "
            "resolve overlap. A protocol for this experiment therefore has to decide, in "
            "advance, which metric adjudicates it - and that is a methodological decision for a "
            "human, not a detail to settle mid-experiment."
        ),
    },
}

NEXT_DECISION = (
    "The preferred intervention is **`overlap_mask: false`**, and the evidence for it is the "
    "measured contested-pixel pattern rather than a preference among plausible knobs. It is "
    "recorded as a hypothesis with a named obstacle: flipping the flag changes what the "
    "framework's own mask metric is measured against, so phase 8E must decide in advance which "
    "metric adjudicates the comparison - the direct mask-IoU diagnostic against canonical COCO "
    "is comparable across the flag, and the framework's mask mAP is not.\n\n"
    "No protocol is frozen here and no experiment is authorised. Phase 8E is a human-reviewed "
    "protocol freeze, and like phase 7A it must be written before the experiment it would "
    "decide exists.\n\n"
    "One further observation belongs in that review rather than in this analysis: if the "
    "overlap policy stays as it is, then the project's direct mask-IoU diagnostic is charging "
    "the model for pixels it was trained to exclude. That is a reporting question about S0's "
    "published numbers as much as an experiment design question, and both readings are already "
    "recorded side by side above so that neither is quietly adopted."
)

CLASSIFICATION = READY


class AnalysisError(RuntimeError):
    """Raised when a required input is missing or an invariant fails."""


# --- helpers -------------------------------------------------------------------


def read_json(path: Path) -> dict[str, Any]:
    """Read a committed JSON artifact.

    Args:
        path: File to read.

    Returns:
        The parsed mapping.

    Raises:
        AnalysisError: If the file is missing or is not a JSON object.
    """
    if not path.is_file():
        msg = f"required input not found: {path.name}"
        raise AnalysisError(msg)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        msg = f"{path.name} must contain a JSON object"
        raise AnalysisError(msg)
    return data


def write_json(path: Path, payload: Any) -> str:
    """Write a JSON artifact deterministically and return its digest.

    Args:
        path: Destination file.
        payload: JSON-serialisable content.

    Returns:
        The SHA-256 digest of the bytes written.
    """
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return sha256_file(path)


def historical_digests(paths: ProjectPaths) -> dict[str, str]:
    """Digest every artifact this phase must leave untouched.

    Args:
        paths: Project layout.

    Returns:
        Digests keyed by repository-relative name.

    Raises:
        AnalysisError: If one is absent.
    """
    names = [
        "reports/segmentation_adapter_audit_manifest.json",
        "reports/segmentation_adapter_fidelity_report.md",
        f"reports/{FIDELITY_CSV}",
        "configs/segmentation_adapter_audit.yaml",
        "reports/segmentation_adapter_approval.json",
        "reports/segmentation_S0_manifest.json",
        "reports/segmentation_S0_protocol.md",
        "configs/segmentation_baseline.yaml",
        f"configs/{MASK_IOU_YAML}",
        f"reports/{RESULT_MANIFEST_JSON}",
        f"reports/{MASK_IOU_JSON}",
        "reports/segmentation_S0_report.md",
        "reports/final_detector_manifest.json",
    ]
    digests: dict[str, str] = {}
    for name in names:
        path = paths.root / name
        if not path.is_file():
            msg = f"historical artifact missing: {name}"
            raise AnalysisError(msg)
        digests[name] = sha256_file(path)
    return digests


def load_fidelity(paths: ProjectPaths) -> dict[tuple[str, int], dict[str, Any]]:
    """Read the phase 8A per-instance fidelity table for the validation split.

    Args:
        paths: Project layout.

    Returns:
        Fidelity facts keyed by ``(image_id, annotation_id)``.
    """
    path = paths.reports / FIDELITY_CSV
    table: dict[tuple[str, int], dict[str, Any]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != "validation":
                continue
            table[(row["source_image_id"], int(row["annotation_id"]))] = {
                "adapter_mask_iou": float(row["mask_iou"]),
                "adapter_geometry_type": row["canonical_geometry_type"],
                "connected_components": int(row["connected_components"]),
                "hole_count": int(row["hole_count"]),
            }
    return table


def canonical_instances(
    coco: Mapping[str, Any], class_map: Mapping[str, int]
) -> tuple[dict[str, list[MaskInstance]], dict[str, list[dict[str, Any]]]]:
    """Decode canonical validation masks and keep their identities alongside.

    Args:
        coco: The parsed COCO document.
        class_map: The frozen class map.

    Returns:
        Instances keyed by image stem, and a parallel record of each instance's
        annotation id and area, in the same order.
    """
    from pycocotools import mask as mask_utils

    categories = {int(entry["id"]): str(entry["name"]) for entry in coco["categories"]}
    images = {
        int(entry["id"]): (
            Path(str(entry["file_name"])).stem,
            int(entry["height"]),
            int(entry["width"]),
        )
        for entry in coco["images"]
    }
    instances: dict[str, list[MaskInstance]] = {stem: [] for stem, _, _ in images.values()}
    identities: dict[str, list[dict[str, Any]]] = {stem: [] for stem, _, _ in images.values()}

    for annotation in coco["annotations"]:
        name = categories[int(annotation["category_id"])]
        stem, height, width = images[int(annotation["image_id"])]
        segmentation = annotation["segmentation"]
        if isinstance(segmentation, list):
            rle = mask_utils.merge(mask_utils.frPyObjects(segmentation, height, width))
        elif isinstance(segmentation.get("counts"), list):
            rle = mask_utils.frPyObjects(segmentation, height, width)
        else:
            rle = dict(segmentation)
            if isinstance(rle["counts"], str):
                rle["counts"] = rle["counts"].encode("utf-8")
        mask = mask_utils.decode(rle).astype(bool)
        if mask.ndim == 3:  # pragma: no cover - merge already flattens
            mask = mask.any(axis=2)
        instances[stem].append(MaskInstance(class_id=class_map[name], mask=mask))
        identities[stem].append(
            {
                "annotation_id": int(annotation["id"]),
                "class": name,
                "area": int(np.count_nonzero(mask)),
            }
        )
    return instances, identities


def build_rows(
    paths: ProjectPaths,
    protocol: Any,
    checkpoint: Path,
    ground_truth: Mapping[str, list[MaskInstance]],
    identities: Mapping[str, list[dict[str, Any]]],
    fidelity: Mapping[tuple[str, int], dict[str, Any]],
    class_map: Mapping[str, int],
) -> list[InstanceRow]:
    """Run inference once and build one row per canonical instance.

    The settings are the committed phase 8C ones, read from the protocol rather
    than restated here, so this analysis cannot become a threshold search.

    Args:
        paths: Project layout.
        protocol: The frozen diagnostic protocol.
        checkpoint: The frozen S0 ``best.pt``.
        ground_truth: Canonical instances keyed by image stem.
        identities: Their annotation ids and areas, in the same order.
        fidelity: Phase 8A per-instance fidelity.
        class_map: The frozen class map.

    Returns:
        One row per canonical instance.

    Raises:
        AnalysisError: If prediction fails or an image has no canonical entry.
    """
    from ultralytics import YOLO

    inference = protocol.inference
    images_root = paths.root / RUNTIME_VIEW_ROOT / "images" / "val"
    image_paths = sorted(path for path in images_root.iterdir() if path.suffix != ".txt")

    model = YOLO(str(checkpoint))
    rows: list[InstanceRow] = []
    for image_path in image_paths:
        stem = image_path.stem
        if stem not in ground_truth:
            msg = f"predicted image {stem} has no canonical validation entry"
            raise AnalysisError(msg)
        try:
            outputs = model.predict(
                source=str(image_path),
                imgsz=inference["imgsz"],
                conf=inference["conf"],
                iou=inference["iou"],
                max_det=inference["max_det"],
                retina_masks=inference["retina_masks"],
                augment=inference["augment"],
                agnostic_nms=inference["agnostic_nms"],
                half=inference["half"],
                verbose=False,
                save=False,
                stream=False,
            )
        except Exception as exc:
            msg = f"prediction failed on one validation image ({exc})"
            raise AnalysisError(msg) from exc

        result = outputs[0]
        predictions: list[MaskInstance] = []
        confidences: list[float] = []
        if result.masks is not None:
            masks = result.masks.data.cpu().numpy().astype(bool)
            classes = result.boxes.cls.cpu().numpy().astype(int).tolist()
            scores = result.boxes.conf.cpu().numpy().astype(float).tolist()
            for mask, class_index, score in zip(masks, classes, scores, strict=True):
                predictions.append(MaskInstance(class_id=int(class_index), mask=mask))
                confidences.append(float(score))

        matching = match_image(stem, ground_truth[stem], predictions)

        # The target the framework actually trained on, plus one union per class,
        # so the pixels a person shares with a vest or a helmet can be measured.
        resolved = overlap_resolved_targets([instance.mask for instance in ground_truth[stem]])
        per_class_union: dict[int, Any] = {}
        for instance in ground_truth[stem]:
            key = int(instance.class_id)
            if key not in per_class_union:
                per_class_union[key] = np.zeros_like(instance.mask)
            per_class_union[key] |= instance.mask

        # match_image groups by class and indexes within each class, so the pair
        # indices are resolved back to the flat lists the same way.
        by_class: dict[int, list[int]] = {}
        for position, instance in enumerate(ground_truth[stem]):
            by_class.setdefault(int(instance.class_id), []).append(position)
        predictions_by_class: dict[int, list[int]] = {}
        for position, instance in enumerate(predictions):
            predictions_by_class.setdefault(int(instance.class_id), []).append(position)

        matched: dict[int, tuple[float, int]] = {}
        for pair in matching.pairs:
            gt_position = by_class[pair.class_id][pair.gt_index]
            prediction_position = predictions_by_class[pair.class_id][pair.prediction_index]
            matched[gt_position] = (pair.iou, prediction_position)

        for position, identity in enumerate(identities[stem]):
            key = (stem, int(identity["annotation_id"]))
            facts = fidelity.get(key, {})
            gt_mask = ground_truth[stem][position].mask
            if position in matched:
                iou, prediction_position = matched[position]
                predicted = predictions[prediction_position].mask
                predicted_area = int(np.count_nonzero(predicted))
                false_positive = int(np.count_nonzero(predicted & ~gt_mask))
                false_negative = int(np.count_nonzero(gt_mask & ~predicted))
                confidence = confidences[prediction_position]
            else:
                iou = 0.0
                predicted_area = None
                false_positive = None
                false_negative = None
                confidence = None

            band = outcome_band(matched=position in matched, mask_iou=iou)
            area = int(identity["area"])

            own_class = int(ground_truth[stem][position].class_id)
            contested = np.zeros_like(gt_mask)
            for key, union in per_class_union.items():
                if key != own_class:
                    contested |= union
            contested &= gt_mask
            contested_fraction = float(np.count_nonzero(contested) / area) if area else 0.0
            if position in matched:
                predicted = predictions[matched[position][1]].mask
                overlap_resolved_iou = float(
                    np.count_nonzero(resolved[position] & predicted)
                    / max(int(np.count_nonzero(resolved[position] | predicted)), 1)
                )
                missed = gt_mask & ~predicted
                missed_total = int(np.count_nonzero(missed))
                contested_share = (
                    float(np.count_nonzero(missed & contested) / missed_total)
                    if missed_total
                    else None
                )
            else:
                overlap_resolved_iou = 0.0
                contested_share = None
            relative = (
                None if predicted_area is None or not area else (predicted_area - area) / area
            )
            rows.append(
                InstanceRow(
                    image_id=stem,
                    annotation_id=int(identity["annotation_id"]),
                    class_name=str(identity["class"]),
                    canonical_area_px=area,
                    adapter_mask_iou=facts.get("adapter_mask_iou"),
                    adapter_geometry_type=facts.get("adapter_geometry_type"),
                    connected_components=facts.get("connected_components"),
                    hole_count=facts.get("hole_count"),
                    matched=position in matched,
                    mask_iou=iou,
                    confidence=confidence,
                    predicted_area_px=predicted_area,
                    false_positive_px=false_positive,
                    false_negative_px=false_negative,
                    band=band,
                    automatic_flags=automatic_flags(
                        band=band,
                        canonical_area_px=area,
                        adapter_mask_iou=facts.get("adapter_mask_iou"),
                        relative_area_error=relative,
                    ),
                    contested_fraction=contested_fraction,
                    overlap_resolved_iou=overlap_resolved_iou,
                    false_negative_contested_share=contested_share,
                )
            )
    return sorted(rows, key=lambda row: (row.image_id, row.annotation_id))


def cross_check(rows: Sequence[InstanceRow], committed: Mapping[str, Any]) -> dict[str, Any]:
    """Confirm the per-instance table reproduces the committed aggregates.

    This is the check that makes the analysis trustworthy: it re-derives the
    published headline figures from the rows it will reason about, and if they
    disagree the analysis is wrong - the published result is not revised.

    Args:
        rows: The instance rows.
        committed: The committed phase 8C global block.

    Returns:
        The comparison.

    Raises:
        AnalysisError: If the rows do not reproduce the committed figures.
    """
    matched = [row for row in rows if row.matched]
    total = len(rows)
    recomputed = {
        "gt_count": total,
        "matched_count": len(matched),
        "matched_mask_iou_mean": (
            round(sum(row.mask_iou for row in matched) / len(matched), 6) if matched else None
        ),
        "gt_normalized_mask_iou": round(sum(row.mask_iou for row in rows) / total, 6),
        "gt_match_coverage": round(len(matched) / total, 6),
    }
    disagreements = {
        name: {"committed": committed[name], "recomputed": value}
        for name, value in recomputed.items()
        if abs(float(committed[name]) - float(value)) > 1e-6
    }
    if disagreements:
        msg = (
            "the per-instance table does not reproduce the committed phase 8C aggregates: "
            f"{disagreements}. The published result stands; this analysis is wrong."
        )
        raise AnalysisError(msg)
    return {
        "recomputed": recomputed,
        "committed": {name: committed[name] for name in recomputed},
        "agree": True,
        "note": (
            "The committed phase 8C figures were not regenerated. They were re-derived from "
            "the per-instance table purely to prove this analysis describes the same run."
        ),
    }


def support_classification(rows: Sequence[InstanceRow]) -> dict[str, dict[str, Any]]:
    """Apply the frozen phase 7A support rule to the validation instances.

    Args:
        rows: The instance rows.

    Returns:
        Per class, its counts and whether the rule admits it.
    """
    counts: dict[str, dict[str, Any]] = {}
    images: dict[str, set[str]] = {}
    for row in rows:
        entry = counts.setdefault(row.class_name, {"instances": 0, "images": 0})
        entry["instances"] += 1
        images.setdefault(row.class_name, set()).add(row.image_id)
    for name, entry in counts.items():
        entry["images"] = len(images[name])
        entry["supported"] = (
            entry["images"] >= SUPPORT_MIN_POSITIVE_IMAGES
            and entry["instances"] >= SUPPORT_MIN_INSTANCES
        )
    return dict(sorted(counts.items()))


# --- qualitative review figures -------------------------------------------------


def render_review_figures(
    paths: ProjectPaths,
    protocol: Any,
    checkpoint: Path,
    ground_truth: Mapping[str, list[MaskInstance]],
    identities: Mapping[str, list[dict[str, Any]]],
    selection: Mapping[str, Sequence[Mapping[str, Any]]],
    class_map: Mapping[str, int],
) -> list[str]:
    """Draw the deterministic review set for human inspection.

    One figure per selected instance: the source image with the canonical mask
    and the matched prediction overlaid in different colours, cropped around the
    canonical object. Written to git-ignored runtime storage, because they render
    dataset imagery and the repository does not publish that.

    Args:
        paths: Project layout.
        protocol: The frozen diagnostic protocol.
        checkpoint: The frozen S0 ``best.pt``.
        ground_truth: Canonical instances keyed by image stem.
        identities: Their annotation ids, in the same order.
        selection: The deterministic review set.
        class_map: The frozen class map.

    Returns:
        The figure file names written, sorted.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    from PIL import Image
    from ultralytics import YOLO

    inference = protocol.inference
    wanted: dict[str, list[dict[str, Any]]] = {}
    for entries in selection.values():
        for entry in entries:
            wanted.setdefault(str(entry["image_id"]), []).append(dict(entry))

    # Clear previous figures only. The review-set manifest is written before this
    # runs and records which instances were chosen before any image was opened;
    # deleting the directory wholesale would destroy that record.
    destination = paths.root / REVIEW_ROOT
    destination.mkdir(parents=True, exist_ok=True)
    for stale in destination.glob("*.png"):
        stale.unlink()

    model = YOLO(str(checkpoint))
    written: list[str] = []
    images_root = paths.root / RUNTIME_VIEW_ROOT / "images" / "val"

    for stem in sorted(wanted):
        matches = sorted(images_root.glob(f"{stem}.*"))
        if not matches:
            continue
        image_path = matches[0]
        with Image.open(long_path(image_path)) as handle:
            picture = np.array(handle.convert("RGB"))

        outputs = model.predict(
            source=str(image_path),
            imgsz=inference["imgsz"],
            conf=inference["conf"],
            iou=inference["iou"],
            max_det=inference["max_det"],
            retina_masks=inference["retina_masks"],
            augment=inference["augment"],
            agnostic_nms=inference["agnostic_nms"],
            half=inference["half"],
            verbose=False,
            save=False,
        )
        result = outputs[0]
        predictions: list[MaskInstance] = []
        confidences: list[float] = []
        if result.masks is not None:
            masks = result.masks.data.cpu().numpy().astype(bool)
            classes = result.boxes.cls.cpu().numpy().astype(int).tolist()
            scores = result.boxes.conf.cpu().numpy().astype(float).tolist()
            for mask, class_index, score in zip(masks, classes, scores, strict=True):
                predictions.append(MaskInstance(class_id=int(class_index), mask=mask))
                confidences.append(float(score))

        matching = match_image(stem, ground_truth[stem], predictions)
        by_class: dict[int, list[int]] = {}
        for position, instance in enumerate(ground_truth[stem]):
            by_class.setdefault(int(instance.class_id), []).append(position)
        predictions_by_class: dict[int, list[int]] = {}
        for position, instance in enumerate(predictions):
            predictions_by_class.setdefault(int(instance.class_id), []).append(position)
        matched: dict[int, tuple[float, int]] = {}
        for pair in matching.pairs:
            matched[by_class[pair.class_id][pair.gt_index]] = (
                pair.iou,
                predictions_by_class[pair.class_id][pair.prediction_index],
            )

        for entry in wanted[stem]:
            annotation_id = int(entry["annotation_id"])
            position = next(
                (
                    index
                    for index, identity in enumerate(identities[stem])
                    if int(identity["annotation_id"]) == annotation_id
                ),
                None,
            )
            if position is None:
                continue
            gt_mask = ground_truth[stem][position].mask
            predicted_mask = None
            iou = 0.0
            confidence = None
            if position in matched:
                iou, prediction_position = matched[position]
                predicted_mask = predictions[prediction_position].mask
                confidence = confidences[prediction_position]

            union = gt_mask if predicted_mask is None else (gt_mask | predicted_mask)
            rows_hit, columns_hit = np.where(union)
            if rows_hit.size == 0:
                continue
            pad = 40
            top = max(int(rows_hit.min()) - pad, 0)
            bottom = min(int(rows_hit.max()) + pad, picture.shape[0])
            left = max(int(columns_hit.min()) - pad, 0)
            right = min(int(columns_hit.max()) + pad, picture.shape[1])

            crop = picture[top:bottom, left:right]
            gt_crop = gt_mask[top:bottom, left:right]
            pred_crop = None if predicted_mask is None else predicted_mask[top:bottom, left:right]

            figure, axes = plt.subplots(1, 2, figsize=(13, 6))
            for axis in axes:
                axis.imshow(crop)
                axis.set_xticks([])
                axis.set_yticks([])

            overlay = np.zeros((*gt_crop.shape, 4))
            overlay[gt_crop] = (0.0, 0.85, 0.2, 0.45)
            axes[0].imshow(overlay)
            axes[0].set_title("canonical COCO mask", fontsize=11)

            if pred_crop is not None:
                predicted_overlay = np.zeros((*pred_crop.shape, 4))
                predicted_overlay[pred_crop] = (1.0, 0.25, 0.1, 0.45)
                axes[1].imshow(predicted_overlay)
                axes[1].set_title(f"S0 prediction · IoU {iou:.3f}", fontsize=11)
                only_gt = gt_crop & ~pred_crop
                only_pred = pred_crop & ~gt_crop
                difference = np.zeros((*gt_crop.shape, 4))
                difference[only_gt] = (0.0, 0.4, 1.0, 0.55)
                difference[only_pred] = (1.0, 0.9, 0.0, 0.55)
                axes[1].imshow(difference)
            else:
                axes[1].set_title("S0 prediction · NONE (detection miss)", fontsize=11)

            figure.suptitle(
                f"{entry['class']} · {stem} · annotation {annotation_id} · {entry['rule']}\n"
                f"band {entry['band']} · canonical area {entry['canonical_area_px']} px · "
                f"adapter IoU {entry['adapter_mask_iou']} · "
                f"confidence {'n/a' if confidence is None else format(confidence, '.3f')}\n"
                "blue = canonical only (missed) · yellow = predicted only (extra)",
                fontsize=10,
            )
            name = f"{entry['class']}__{stem}__{annotation_id}.png"
            figure.tight_layout()
            figure.savefig(destination / name, dpi=90)
            plt.close(figure)
            written.append(name)
            _ = patches
    return sorted(set(written))


# --- report ---------------------------------------------------------------------


def build_report(analysis: Mapping[str, Any], *, commit: str | None) -> str:
    """Render the S0 error-analysis report.

    Args:
        analysis: The assembled analysis.
        commit: Repository commit, when available.

    Returns:
        The report text.
    """
    lines: list[str] = []
    add = lines.append

    person = analysis["person_diagnostic"]
    bands = analysis["band_census"]
    strata = analysis["stratified"]
    gap = analysis["box_vs_mask"]
    hypotheses = analysis["hypotheses"]
    review = analysis["qualitative_review"]
    adapter = analysis["adapter_versus_model"]

    add("# S0 - validation error analysis")
    add("")
    add(
        f"Phase {analysis['phase']} · classification `{analysis['classification']}` · "
        f"experiment `{analysis['experiment_id']}` · final segmenter "
        f"`{analysis['final_segmenter']}`"
    )
    add("")
    add(
        "**No model was trained, re-validated or modified.** This phase re-ran inference on the "
        "frozen validation split under the settings phase 8C already froze, and reports the "
        "outcome one canonical instance at a time. Every number is a validation number."
    )
    add("")
    if commit:
        add(f"Repository commit at production time: `{commit}`.")
        add("")

    add("## 1. What this analysis is, and what it is not")
    add("")
    add(
        "`PREDECLARED_PROTOCOL`. The inference settings and the matching rule are read from the "
        f"committed phase 8C diagnostic protocol (`{analysis['protocol_fingerprint']}`) and "
        "asserted, not restated: conf "
        f"{analysis['inference']['conf']}, NMS IoU {analysis['inference']['iou']}, imgsz "
        f"{analysis['inference']['imgsz']}, max_det {analysis['inference']['max_det']}. "
        "**No threshold was introduced, swept or tuned.**"
    )
    add("")
    add(
        "The committed phase 8C aggregates were **not regenerated**. They were re-derived from "
        "the per-instance table only to prove this analysis describes the same run, and they "
        f"agree: {analysis['cross_check']['note']}"
    )
    add("")
    add(
        "It cannot establish a cause. Nothing here is a controlled experiment, so labels naming "
        "one - model capacity, insufficient training, insufficient data - are refused by the "
        "recorder rather than merely discouraged."
    )
    add("")

    add("## 2. Outcome census")
    add("")
    add(
        f"`COMPUTED_RESULT`. All {analysis['gt_instances']} canonical validation instances, "
        "partitioned by outcome. The four bands are exhaustive and disjoint, so the counts sum "
        "to the total - which is what makes the census checkable."
    )
    add("")
    add("| Band | Definition | Count | Share |")
    add("| --- | --- | --- | --- |")
    definitions = {
        DETECTION_MISS: "no overlapping same-class prediction",
        LOW_OVERLAP_MASK: f"matched, mask IoU < {IOU_LOW}",
        MODERATE_MASK: f"matched, {IOU_LOW} <= IoU < {IOU_HIGH}",
        HIGH_QUALITY_MASK: f"matched, IoU >= {IOU_HIGH}",
    }
    for band in OUTCOME_BANDS:
        count = bands[band]
        share = count / analysis["gt_instances"]
        add(f"| `{band}` | {definitions[band]} | {count} | {share:.1%} |")
    add("")

    add("## 3. The `person` diagnostic")
    add("")
    add(
        "`PREDECLARED_FOCUS`. `person` was named as the focus before this analysis began, "
        "because S0 showed it the largest box-to-mask AP gap of any comparison-supported class. "
        "That gap is a fact read from the committed S0 manifest; it is **not** evidence for any "
        "particular mechanism."
    )
    add("")
    add("| Quantity | Value |")
    add("| --- | --- |")
    add(f"| box AP@0.50:0.95 | {person['box_ap50_95']} |")
    add(f"| mask AP@0.50:0.95 | {person['mask_ap50_95']} |")
    add(f"| **box minus mask** | **{person['box_minus_mask']}** |")
    add(f"| GT instances | {person['instances']} |")
    add(f"| Matched | {person['matched']} |")
    add(f"| Misses | {person['misses']} |")
    add(f"| matched mask IoU mean | {person['matched_mask_iou_mean']} |")
    add(f"| GT-normalised mask IoU | {person['gt_normalized_mask_iou']} |")
    add(f"| IoU >= 0.50 coverage | {person['iou50_coverage']} |")
    add(f"| IoU >= 0.75 coverage | {person['iou75_coverage']} |")
    add("")
    add("By canonical mask area:")
    add("")
    add("| Area quartile | Instances | Matched | matched mean IoU | GT-normalised IoU |")
    add("| --- | --- | --- | --- | --- |")
    for name in sorted(person["by_area_quartile"]):
        entry = person["by_area_quartile"][name]
        add(
            f"| {name} | {entry['instances']} | {entry['matched']} | "
            f"{entry['matched_mask_iou_mean']} | {entry['gt_normalized_mask_iou']} |"
        )
    add("")

    add("## 4. Cross-class context")
    add("")
    add(
        "`COMPUTED_RESULT`. The same decomposition for every class, so the `person` reading is "
        "calibrated rather than cherry-picked."
    )
    add("")
    add("| Class | GT | matched | misses | matched mean IoU | GT-normalised IoU | IoU>=0.75 |")
    add("| --- | --- | --- | --- | --- | --- | --- |")
    for name in sorted(analysis["per_class"]):
        entry = analysis["per_class"][name]
        high = entry["bands"][HIGH_QUALITY_MASK]
        add(
            f"| `{name}` | {entry['instances']} | {entry['matched']} | {entry['misses']} | "
            f"{entry['matched_mask_iou_mean']} | {entry['gt_normalized_mask_iou']} | {high} |"
        )
    add("")
    add(
        f"`{RARE_CLASS}` remains `{RARE_CLASS_STATUS}`: it holds "
        f"{analysis['per_class'][RARE_CLASS]['instances']} validation instances on one image. "
        "Nothing about the population may be inferred from it, and it is reported here only for "
        "completeness."
    )
    add("")

    add("## 5. Stratified diagnostics")
    add("")
    add(
        "`COMPUTED_RESULT`, all descriptive. These are pre-existing strata from the canonical "
        "and phase 8A records, not subgroups searched for after seeing the outcome."
    )
    add("")
    for title, key in (
        ("Canonical mask-area quartile", "area_quartile"),
        ("Phase 8A adapter-fidelity band", "adapter_fidelity_band"),
        ("Connected-component count", "component_count"),
        ("Hole presence", "hole_presence"),
    ):
        add(f"### {title}")
        add("")
        add("| Stratum | Instances | Matched | matched mean IoU | GT-normalised IoU |")
        add("| --- | --- | --- | --- | --- |")
        for name in sorted(strata[key]):
            entry = strata[key][name]
            add(
                f"| {name} | {entry['instances']} | {entry['matched']} | "
                f"{entry['matched_mask_iou_mean']} | {entry['gt_normalized_mask_iou']} |"
            )
        add("")
    for finding in analysis["stratified_findings"]:
        add(f"- {finding}")
    add("")

    add("## 6. Adapter fidelity versus model error")
    add("")
    add("`COMPUTED_RESULT`.")
    add("")
    add(adapter["summary"])
    add("")
    add("| Quantity | Value |")
    add("| --- | --- |")
    add(f"| Phase 8A mean adapter IoU, `person` (validation) | {adapter['person_adapter_mean']} |")
    add(f"| Phase 8A mean adapter IoU, all validation | {adapter['all_adapter_mean']} |")
    add(f"| Mean adapter IoU of the 20 worst S0 instances | {adapter['worst_s0_adapter_mean']} |")
    add(
        f"| Spearman rank correlation, adapter IoU vs S0 IoU (matched) | "
        f"{adapter['rank_correlation']} |"
    )
    add("")
    add(adapter["conclusion"])
    add("")

    add("## 6b. The overlap-resolved training target")
    add("")
    overlap = analysis["overlap_target_analysis"]
    add(
        f"`COMPUTED_RESULT`, labelled `{overlap['label']}`: this measurement was motivated by "
        "an observation made while reviewing images, not predeclared. It is the phase's "
        "substantive finding and also its most easily over-read one, so both scorings are "
        "given side by side."
    )
    add("")
    add(overlap["framework_behaviour"])
    add("")
    add(
        "The canonical COCO masks the direct IoU diagnostic scores against do **not** resolve "
        "overlap - every instance keeps its full extent. So a model that faithfully reproduces "
        "its training target is charged for the difference."
    )
    add("")
    add("| Measurement (`person`, validation) | Value |")
    add("| --- | --- |")
    add(
        f"| Canonical pixels also inside another class | "
        f"{overlap['person_contested_pixel_share']:.1%} |"
    )
    add(
        f"| **Missed pixels lying in that contested region** | "
        f"**{overlap['person_missed_pixel_share_in_contested_region']:.1%}** |"
    )
    add(
        f"| Rank correlation, contested fraction vs mask IoU | "
        f"{overlap['contested_fraction_vs_iou_rank_correlation']} |"
    )
    low = overlap["person_matched_iou_by_contested_fraction"]["under_20_percent"]
    high = overlap["person_matched_iou_by_contested_fraction"]["at_or_over_40_percent"]
    add(
        f"| Matched IoU, persons under 20% contested | {low['matched_mask_iou_mean']} "
        f"(n={low['instances']}) |"
    )
    add(
        f"| Matched IoU, persons at or over 40% contested | {high['matched_mask_iou_mean']} "
        f"(n={high['instances']}) |"
    )
    add("")
    add(
        "The same predictions and the same matching, scored against each target in turn. If the "
        "mechanism is real, `person` should move and the compact classes should not:"
    )
    add("")
    add("| Class | GT-normalised IoU vs canonical | vs overlap-resolved | delta |")
    add("| --- | --- | --- | --- |")
    for name in sorted(overlap["gt_normalized_iou_scored_both_ways"]):
        entry = overlap["gt_normalized_iou_scored_both_ways"][name]
        delta = entry["overlap_resolved"] - entry["canonical"]
        add(f"| `{name}` | {entry['canonical']} | {entry['overlap_resolved']} | {delta:+.4f} |")
    add("")
    matched_both = overlap["person_matched_only"]
    add(
        f"`person` matched-only mask IoU moves from {matched_both['canonical']} to "
        f"{matched_both['overlap_resolved']} ({matched_both['delta']:+.4f})."
    )
    add("")
    add(f"**Interpretation.** {overlap['interpretation']}")
    add("")
    add(f"**Why this is not yet a finding.** {overlap['not_a_finding_because']}")
    add("")

    add("## 7. Deterministic qualitative review")
    add("")
    add(
        "`PREDECLARED_PROTOCOL`. The review set was chosen from the instance table by fixed "
        "rules, and its identifiers were recorded, **before any image was opened**. No example "
        "was swapped afterwards."
    )
    add("")
    add("| Selection rule | Selected |")
    add("| --- | --- |")
    for rule in sorted(review["selection_counts"]):
        add(f"| `{rule}` | {review['selection_counts'][rule]} |")
    add(f"| **distinct instances selected** | **{review['instances_selected']}** |")
    add(
        f"| **of those, visually inspected in detail** | "
        f"**{review['instances_visually_inspected']}** |"
    )
    add("")
    add(review["inspection_note"])
    add("")
    add(
        "Figures overlay the canonical mask and the S0 prediction on the source image and stay "
        f"in git-ignored runtime storage (`{REVIEW_ROOT}`): they render dataset imagery, which "
        "this repository does not publish."
    )
    add("")
    add("### Manual mechanism census")
    add("")
    add(
        "Assigned only where the image visibly supported it. Several may apply to one instance, "
        "and an instance nothing explains stays `UNATTRIBUTED` rather than receiving the "
        "nearest-sounding label."
    )
    add("")
    add("| Mechanism | Count |")
    add("| --- | --- |")
    for flag in MECHANISM_FLAGS:
        add(f"| `{flag}` | {review['manual_flag_census'][flag]} |")
    add("")
    for finding in review["findings"]:
        add(f"- {finding}")
    add("")

    add("## 8. Box versus mask, per class")
    add("")
    add("`COMPUTED_RESULT`. Descriptive; no combined score is formed.")
    add("")
    add("| Class | box AP@.50:.95 | mask AP@.50:.95 | gap | matched IoU | GT-norm IoU | coverage |")
    add("| --- | --- | --- | --- | --- | --- | --- |")
    for name in sorted(gap):
        entry = gap[name]
        add(
            f"| `{name}` | {entry['box_ap50_95']} | {entry['mask_ap50_95']} | "
            f"{entry['box_minus_mask']} | {entry['matched_mask_iou_mean']} | "
            f"{entry['gt_normalized_mask_iou']} | {entry['gt_match_coverage']} |"
        )
    add("")

    add("## 9. Hypotheses for a future experiment")
    add("")
    add(
        "`HYPOTHESIS`. None of these is established. Each is recorded with the evidence that "
        "supports and weakens it, so a later reviewer can disagree with the reading rather than "
        "with a conclusion."
    )
    add("")
    for name in sorted(hypotheses):
        entry = hypotheses[name]
        add(f"### `{name}` - {entry['status']}")
        add("")
        add(f"**Supported by.** {entry['supported_by']}")
        add("")
        add(f"**Weakened by.** {entry['weakened_by']}")
        add("")

    add("## 10. Candidate interventions, assessed not chosen")
    add("")
    for key, title in (
        ("mask_ratio_2", "`mask_ratio: 2` at YOLO11n-seg / imgsz 768"),
        ("yolo11s_seg", "YOLO11s-seg at imgsz 768"),
    ):
        entry = analysis["candidates"][key]
        add(f"### {title}")
        add("")
        add(f"**Verdict:** `{entry['verdict']}`")
        add("")
        add(entry["reasoning"])
        add("")
        add(f"**One-variable against S0?** {entry['one_variable']}")
        add("")

    add("## 11. Limitations")
    add("")
    for limitation in analysis["limitations"]:
        add(f"- {limitation}")
    add("")

    add("## 12. Holdout compliance")
    add("")
    add(f"`HOLDOUT_POLICY` · status `{analysis['test']['status']}`.")
    add("")
    add(analysis["test"]["reason"])
    add("")

    add("## 13. Next decision")
    add("")
    add(f"`PENDING_HUMAN_REVIEW`. Classification: `{analysis['classification']}`.")
    add("")
    add(analysis["next_decision"])
    add("")
    add("---")
    add("")
    add(
        f"Instance table `reports/{INSTANCES_CSV}` · analysis "
        f"`reports/{ANALYSIS_JSON}` · S0 checkpoint "
        f"`{analysis['checkpoint']['sha256']}`."
    )
    add("")
    return "\n".join(lines)


def spearman(first: Sequence[float], second: Sequence[float]) -> float | None:
    """Rank correlation between two equal-length sequences.

    Used descriptively, to say whether poor adapter conversion tends to coincide
    with poor model masks. It measures association, not causation, and with this
    sample size it settles nothing on its own.

    Args:
        first: One sequence.
        second: The other, same length.

    Returns:
        Spearman's rho, or ``None`` when there is too little variation.
    """
    if len(first) != len(second) or len(first) < 3:
        return None
    from scipy.stats import spearmanr

    value = float(spearmanr(first, second).statistic)
    return None if np.isnan(value) else round(value, 4)


def build_analysis(
    *,
    paths: ProjectPaths,
    protocol: Any,
    rows: Sequence[InstanceRow],
    manifest: Mapping[str, Any],
    diagnostic: Mapping[str, Any],
    checkpoint_sha256: str,
    historical: Mapping[str, str],
    selection: Mapping[str, Sequence[Mapping[str, Any]]],
    figures: Sequence[str],
) -> dict[str, Any]:
    """Assemble the complete error analysis.

    Args:
        paths: Project layout.
        protocol: The frozen diagnostic protocol.
        rows: The per-instance table.
        manifest: The committed S0 result manifest.
        diagnostic: The committed direct mask-IoU result.
        checkpoint_sha256: The verified S0 checkpoint digest.
        historical: Digests of the artifacts this phase must not change.
        selection: The deterministic review set.
        figures: Review figures rendered.

    Returns:
        The analysis.
    """
    total = len(rows)
    bands = dict.fromkeys(OUTCOME_BANDS, 0)
    for row in rows:
        bands[row.band] += 1

    per_class = summarise(rows, lambda row: row.class_name)
    support = support_classification(rows)
    supported_classes = [name for name, entry in support.items() if entry["supported"]]

    edges = area_quartile_edges(rows)
    stratified = {
        "area_quartile": summarise(rows, lambda row: area_quartile(row, edges)),
        "adapter_fidelity_band": summarise(rows, fidelity_band),
        "component_count": summarise(
            rows,
            lambda row: (
                None
                if row.connected_components is None
                else ("1_component" if row.connected_components == 1 else "2_or_more_components")
            ),
        ),
        "hole_presence": summarise(
            rows,
            lambda row: (
                None
                if row.hole_count is None
                else ("has_holes" if row.hole_count > 0 else "no_holes")
            ),
        ),
    }

    focus_rows = [row for row in rows if row.class_name == FOCUS_CLASS]
    focus_matched = [row for row in focus_rows if row.matched]
    focus_class_metrics = manifest["per_class_metrics"][FOCUS_CLASS]
    box_ap = float(focus_class_metrics["box"]["AP@0.50:0.95"])
    mask_ap = float(focus_class_metrics["mask"]["AP@0.50:0.95"])
    person = {
        "class": FOCUS_CLASS,
        "focus_reason": (
            "Named as the focus before this analysis began, because S0 showed it the largest "
            "box-to-mask AP gap of any comparison-supported class. The gap is a committed fact, "
            "not evidence for a mechanism."
        ),
        "box_ap50_95": round(box_ap, 6),
        "mask_ap50_95": round(mask_ap, 6),
        "box_minus_mask": round(box_ap - mask_ap, 6),
        "instances": len(focus_rows),
        "matched": len(focus_matched),
        "misses": len(focus_rows) - len(focus_matched),
        "matched_mask_iou_mean": (
            round(sum(row.mask_iou for row in focus_matched) / len(focus_matched), 6)
            if focus_matched
            else None
        ),
        "gt_normalized_mask_iou": round(
            sum(row.mask_iou for row in focus_rows) / len(focus_rows), 6
        ),
        "iou50_coverage": round(
            sum(1 for row in focus_rows if row.mask_iou >= IOU_LOW) / len(focus_rows), 6
        ),
        "iou75_coverage": round(
            sum(1 for row in focus_rows if row.mask_iou >= IOU_HIGH) / len(focus_rows), 6
        ),
        "by_area_quartile": summarise(focus_rows, lambda row: area_quartile(row, edges)),
        "bands": {band: sum(1 for row in focus_rows if row.band == band) for band in OUTCOME_BANDS},
    }

    box_vs_mask = {}
    for name in sorted(per_class):
        metrics = manifest["per_class_metrics"][name]
        entry_box = float(metrics["box"]["AP@0.50:0.95"])
        entry_mask = float(metrics["mask"]["AP@0.50:0.95"])
        box_vs_mask[name] = {
            "box_ap50_95": round(entry_box, 6),
            "mask_ap50_95": round(entry_mask, 6),
            "box_minus_mask": round(entry_box - entry_mask, 6),
            "matched_mask_iou_mean": per_class[name]["matched_mask_iou_mean"],
            "gt_normalized_mask_iou": per_class[name]["gt_normalized_mask_iou"],
            "gt_match_coverage": per_class[name]["gt_match_coverage"],
            "supported": support[name]["supported"],
        }

    matched_rows = [row for row in rows if row.matched and row.adapter_mask_iou is not None]
    worst_twenty = sorted(rows, key=lambda row: (row.mask_iou, row.image_id, row.annotation_id))[
        :20
    ]
    worst_with_fidelity = [row for row in worst_twenty if row.adapter_mask_iou is not None]
    person_adapter = [
        row.adapter_mask_iou for row in focus_rows if row.adapter_mask_iou is not None
    ]
    all_adapter = [row.adapter_mask_iou for row in rows if row.adapter_mask_iou is not None]
    adapter_versus_model = {
        "person_adapter_mean": round(sum(person_adapter) / len(person_adapter), 6)
        if person_adapter
        else None,
        "all_adapter_mean": round(sum(all_adapter) / len(all_adapter), 6) if all_adapter else None,
        "worst_s0_adapter_mean": (
            round(
                sum(row.adapter_mask_iou for row in worst_with_fidelity) / len(worst_with_fidelity),
                6,
            )
            if worst_with_fidelity
            else None
        ),
        "rank_correlation": spearman(
            [row.adapter_mask_iou for row in matched_rows],
            [row.mask_iou for row in matched_rows],
        ),
        "adapter_fidelity_risk_instances": sum(
            1 for row in rows if ADAPTER_FIDELITY_RISK in row.automatic_flags
        ),
    }

    # The overlap-target measurement. Computed here rather than described in
    # prose because it is the phase's substantive finding, and because a reader
    # must be able to see both scorings side by side rather than take one.
    contested_total = sum(row.canonical_area_px * row.contested_fraction for row in focus_rows)
    focus_area = sum(row.canonical_area_px for row in focus_rows)
    missed_total = sum(
        row.false_negative_px for row in focus_matched if row.false_negative_px is not None
    )
    missed_contested = sum(
        (row.false_negative_px or 0) * (row.false_negative_contested_share or 0.0)
        for row in focus_matched
    )
    low_contested = [row for row in focus_matched if row.contested_fraction < 0.20]
    high_contested = [row for row in focus_matched if row.contested_fraction >= 0.40]

    def dual(subset: Sequence[InstanceRow]) -> dict[str, Any]:
        if not subset:
            return {"canonical": None, "overlap_resolved": None, "delta": None}
        canonical = sum(row.mask_iou for row in subset) / len(subset)
        resolved = sum(row.overlap_resolved_iou for row in subset) / len(subset)
        return {
            "canonical": round(canonical, 6),
            "overlap_resolved": round(resolved, 6),
            "delta": round(resolved - canonical, 6),
        }

    overlap_analysis = {
        "label": "POST_HOC_HYPOTHESIS_GENERATING",
        "framework_behaviour": (
            "Read from the installed source. `polygons2masks_overlap` sorts instances by "
            "descending area and paints them with a running maximum, so the smaller instance "
            "owns any shared pixel: a vest owns the pixels of the person wearing it, and that "
            "person's training target is the remainder. "
            "`SegmentationValidator._prepare_batch` builds its ground truth the same way, so "
            "the framework's own mask metric is measured against the same resolved target."
        ),
        "canonical_gt_does_not_resolve_overlap": True,
        "person_contested_pixel_share": (
            round(contested_total / focus_area, 6) if focus_area else None
        ),
        "person_missed_pixel_share_in_contested_region": (
            round(missed_contested / missed_total, 6) if missed_total else None
        ),
        "contested_fraction_vs_iou_rank_correlation": spearman(
            [row.contested_fraction for row in focus_matched],
            [row.mask_iou for row in focus_matched],
        ),
        "person_matched_iou_by_contested_fraction": {
            "under_20_percent": {
                "instances": len(low_contested),
                "matched_mask_iou_mean": (
                    round(sum(row.mask_iou for row in low_contested) / len(low_contested), 6)
                    if low_contested
                    else None
                ),
            },
            "at_or_over_40_percent": {
                "instances": len(high_contested),
                "matched_mask_iou_mean": (
                    round(sum(row.mask_iou for row in high_contested) / len(high_contested), 6)
                    if high_contested
                    else None
                ),
            },
        },
        "gt_normalized_iou_scored_both_ways": {
            name: {
                "canonical": summarise(
                    [row for row in rows if row.class_name == name], lambda row: "all"
                )["all"]["gt_normalized_mask_iou"],
                "overlap_resolved": round(
                    sum(row.overlap_resolved_iou for row in rows if row.class_name == name)
                    / sum(1 for row in rows if row.class_name == name),
                    6,
                ),
            }
            for name in sorted({row.class_name for row in rows})
        },
        "person_matched_only": dual(focus_matched),
        "interpretation": (
            "A substantial share of person's mask deficit is a target-versus-evaluation "
            "mismatch created by the frozen protocol, not a failure to learn: re-scoring the "
            "same predictions against the target the model was actually trained on recovers a "
            "large part of the gap for person and almost nothing for the other classes, which "
            "is the signature the mechanism predicts. It does not recover all of it, so a "
            "person-specific difficulty remains."
        ),
        "not_a_finding_because": (
            "Re-scoring changes the measuring stick, not the model. It demonstrates that the "
            "two targets disagree and that person is where they disagree most; it does not show "
            "that training without overlap resolution would produce better masks. Only a "
            "controlled experiment could, and none was run."
        ),
    }

    manual = dict(MANUAL_ATTRIBUTIONS)
    review = {
        "selection_method": (
            "Deterministic slices of a totally ordered instance table, ordered by mask IoU then "
            "by image id then by annotation id so every tie resolves. Identifiers were recorded "
            "before any image was opened."
        ),
        "selection_rules": {rule: list(entries) for rule, entries in sorted(selection.items())},
        "selection_counts": {rule: len(entries) for rule, entries in sorted(selection.items())},
        "instances_selected": len(review_identifiers(selection)),
        "instances_visually_inspected": len(manual),
        "inspection_note": (
            "The selection rules chose and rendered every instance listed above. A subset was "
            "then examined in detail and carries a recorded judgement; the count of judgements "
            "is the count of instances actually looked at, and it is reported separately from "
            "the count selected so neither is mistaken for the other. The population-level "
            "findings in this analysis rest on all canonical instances, not on the reviewed "
            "subset."
        ),
        "figures_rendered": len(figures),
        "figures_committed": False,
        "figures_location": REVIEW_ROOT,
        "manual_attributions": manual,
        "manual_flag_census": flag_census(manual),
        "automatic_flag_census": {
            flag: sum(1 for row in rows if flag in row.automatic_flags) for flag in MECHANISM_FLAGS
        },
        "findings": REVIEW_FINDINGS,
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "experiment_id": EXPERIMENT_ID,
        "classification": CLASSIFICATION,
        "analysis_question": (
            "Why is S0 mask quality substantially weaker than its box quality, especially for "
            "person, and what evidence does that give a future controlled experiment?"
        ),
        "models_trained": 0,
        "s0_revalidated": False,
        "s0_modified": False,
        "thresholds_changed": 0,
        "alternative_models_tested": 0,
        "final_segmenter": manifest["final_segmenter"],
        "checkpoint": {
            "name": "best.pt",
            "sha256": checkpoint_sha256,
            "selected_by": manifest["checkpoint_selection"]["policy"],
            "last_pt_used": False,
            "checkpoints_compared": 0,
        },
        "protocol_fingerprint": protocol.fingerprint(),
        "protocol_source": f"configs/{MASK_IOU_YAML}",
        "protocol_reused_unchanged": True,
        "inference": {
            key: value for key, value in protocol.inference.items() if key != "threshold_policy"
        },
        "matching": {
            "algorithm": MATCHING_ALGORITHM,
            "reused_from_phase_8c": True,
            "redefined": False,
        },
        "ground_truth_source": GROUND_TRUTH_SOURCE,
        "gt_instances": total,
        "cross_check": cross_check(rows, diagnostic["global"]),
        "taxonomy": {
            "bands": list(OUTCOME_BANDS),
            "band_thresholds": {"low": IOU_LOW, "high": IOU_HIGH},
            "mechanism_flags": list(MECHANISM_FLAGS),
            "automatic_flags": list(AUTOMATIC_FLAGS),
            "manual_only_flags": list(MANUAL_ONLY_FLAGS),
            "tiny_mask_area_px": TINY_MASK_AREA_PX,
            "adapter_fidelity_risk_iou": ADAPTER_FIDELITY_RISK_IOU,
            "area_ratio_tolerance": AREA_RATIO_TOLERANCE,
            "frozen_before_visual_review": True,
        },
        "band_census": bands,
        "per_class": per_class,
        "support_classification": support,
        "supported_classes": supported_classes,
        "rare_class": {"name": RARE_CLASS, "status": RARE_CLASS_STATUS},
        "person_diagnostic": person,
        "area_quartile_edges": edges,
        "stratified": stratified,
        "stratified_findings": STRATIFIED_FINDINGS,
        "adapter_versus_model": {**adapter_versus_model, **ADAPTER_VERDICT},
        "overlap_target_analysis": overlap_analysis,
        "qualitative_review": review,
        "box_vs_mask": box_vs_mask,
        "hypotheses": HYPOTHESES,
        "candidates": CANDIDATES,
        "limitations": LIMITATIONS,
        "next_decision": NEXT_DECISION,
        "historical_artifact_digests": dict(historical),
        "historical_artifacts_unchanged": True,
        "test": {"status": HOLDOUT_STATUS, "reason": HOLDOUT_REASON},
        "holdout_accessed": False,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the S0 validation error analysis.

    Args:
        argv: Command-line arguments.

    Returns:
        Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="check every precondition without running inference or writing anything",
    )
    parser.add_argument(
        "--build-review",
        action="store_true",
        help=(
            "compute the instance table, select the review set and render its figures, then "
            "stop before recording human judgements"
        ),
    )
    args = parser.parse_args(argv)

    paths = ProjectPaths.from_root()
    if holdout_unlocked():
        print(
            f"REFUSED: {HOLDOUT_UNLOCK_ENV_VAR} is set. This analysis reads development data "
            "only and declines to run in an environment where the holdout is unlocked.",
            file=sys.stderr,
        )
        return 2

    try:
        historical = historical_digests(paths)
        protocol = load_mask_iou_config(paths.configs / MASK_IOU_YAML)
        manifest = read_json(paths.reports / RESULT_MANIFEST_JSON)
        diagnostic = read_json(paths.reports / MASK_IOU_JSON)

        checkpoint = paths.root / manifest["execution"]["run_directory"] / "weights" / "best.pt"
        if not checkpoint.is_file():
            msg = (
                f"the frozen S0 checkpoint is not on this machine: {checkpoint.name}. It is "
                "git-ignored; obtain the artifact rather than retraining, which would produce "
                "different weights."
            )
            raise AnalysisError(msg)
        checkpoint_sha256 = sha256_file(checkpoint)
        recorded = manifest["checkpoints"]["best"]["sha256"]
        if checkpoint_sha256 != recorded or checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
            msg = (
                f"the S0 checkpoint bytes are not the frozen ones: expected {recorded}, observed "
                f"{checkpoint_sha256}"
            )
            raise AnalysisError(msg)

        if protocol.fingerprint() != diagnostic["protocol_fingerprint"]:
            msg = "the diagnostic protocol has changed since phase 8C recorded its result"
            raise AnalysisError(msg)

        audit = read_json(paths.reports / "segmentation_adapter_audit_manifest.json")
        class_map = {name: int(index) for name, index in audit["class_map"].items()}
        document = paths.root / str(protocol["ground_truth_document"])
        coco = json.loads(Path(long_path(document)).read_text(encoding="utf-8"))
        ground_truth, identities = canonical_instances(coco, class_map)
        fidelity = load_fidelity(paths)
    except (ConfigError, AnalysisError, ErrorAnalysisError) as exc:
        print(f"{BLOCKED}: {exc}", file=sys.stderr)
        return 2

    print(f"checkpoint best.pt  sha256 {checkpoint_sha256}  VERIFIED")
    print(f"protocol   {protocol.fingerprint()}  reused unchanged")
    print(
        f"inference  imgsz {protocol.inference['imgsz']}  conf {protocol.inference['conf']}  "
        f"iou {protocol.inference['iou']}  max_det {protocol.inference['max_det']}"
    )
    print(
        f"canonical  {len(ground_truth)} validation images  "
        f"{sum(len(items) for items in ground_truth.values())} GT instances"
    )
    print(f"fidelity   {len(fidelity)} phase 8A validation rows joined")

    if args.verify_only:
        print("VERIFIED: preconditions hold. No inference run, nothing written.")
        print(f"holdout    {HOLDOUT_STATUS}")
        return 0

    try:
        rows = build_rows(
            paths, protocol, checkpoint, ground_truth, identities, fidelity, class_map
        )
        check = cross_check(rows, diagnostic["global"])
    except (AnalysisError, ErrorAnalysisError) as exc:
        print(f"{BLOCKED}: {exc}", file=sys.stderr)
        return 2
    print(f"rows       {len(rows)} instances  aggregates reproduce phase 8C: {check['agree']}")

    support = support_classification(rows)
    supported = [name for name, entry in support.items() if entry["supported"]]
    selection = select_review_set(rows, focus_class=FOCUS_CLASS, supported_classes=supported)
    identifiers = review_identifiers(selection)

    if args.build_review:
        destination = paths.root / REVIEW_ROOT
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "review_set.json").write_text(
            json.dumps(
                {
                    "selected_before_any_image_was_opened": True,
                    "identifiers": identifiers,
                    "rules": {rule: list(entries) for rule, entries in sorted(selection.items())},
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        figures = render_review_figures(
            paths, protocol, checkpoint, ground_truth, identities, selection, class_map
        )
        print(f"review     {len(identifiers)} instances selected, {len(figures)} figures rendered")
        print(f"           identifiers recorded in {REVIEW_ROOT}/review_set.json")
        for rule, entries in sorted(selection.items()):
            print(f"  {rule:<32} {len(entries)}")
        return 0

    try:
        validate_manual_attributions(MANUAL_ATTRIBUTIONS, selection)
    except ErrorAnalysisError as exc:
        print(f"{BLOCKED}: {exc}", file=sys.stderr)
        return 2
    if not MANUAL_ATTRIBUTIONS:
        print(
            f"{BLOCKED}: no human attributions are declared. Run --build-review, look at the "
            "figures, then declare judgements in MANUAL_ATTRIBUTIONS.",
            file=sys.stderr,
        )
        return 2

    figures = sorted(path.name for path in (paths.root / REVIEW_ROOT).glob("*.png"))
    analysis = build_analysis(
        paths=paths,
        protocol=protocol,
        rows=rows,
        manifest=manifest,
        diagnostic=diagnostic,
        checkpoint_sha256=checkpoint_sha256,
        historical=historical,
        selection=selection,
        figures=figures,
    )
    report = build_report(analysis, commit=git_commit(paths.root))

    findings = scan_for_sensitive(report) + scan_for_sensitive(json.dumps(analysis))
    if findings:
        print(f"{BLOCKED}: sensitive content in the emitted artifacts: {findings}", file=sys.stderr)
        return 2

    columns = list(rows[0].as_dict())
    csv_path = paths.reports / INSTANCES_CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict())
    csv_sha256 = sha256_file(csv_path)
    analysis["instance_table"] = {"path": f"reports/{INSTANCES_CSV}", "sha256": csv_sha256}

    analysis_sha256 = write_json(paths.reports / ANALYSIS_JSON, analysis)
    (paths.reports / ANALYSIS_MD).write_text(report, encoding="utf-8", newline="\n")

    after = historical_digests(paths)
    changed = sorted(name for name in historical if historical[name] != after.get(name))
    if changed:
        print(f"{BLOCKED}: historical artifacts changed: {changed}", file=sys.stderr)
        return 2

    record = ProvenanceRecord.create(
        name="segmentation_S0_error_analysis",
        phase=8,
        config={
            "direct_mask_iou_protocol": f"configs/{MASK_IOU_YAML}",
            "protocol_fingerprint": protocol.fingerprint(),
        },
        details={
            "phase": PHASE,
            "classification": CLASSIFICATION,
            "experiment_id": EXPERIMENT_ID,
            "checkpoint_sha256": checkpoint_sha256,
            "gt_instances": len(rows),
            "band_census": analysis["band_census"],
            "models_trained": 0,
            "s0_revalidated": False,
            "thresholds_changed": 0,
            "alternative_models_tested": 0,
            "final_segmenter": analysis["final_segmenter"],
            "instances_selected": analysis["qualitative_review"]["instances_selected"],
            "instances_visually_inspected": analysis["qualitative_review"][
                "instances_visually_inspected"
            ],
            "historical_artifacts_unchanged": True,
            "holdout_accessed": False,
        },
        repo_root=paths.root,
    )
    record.add_input(paths.configs / MASK_IOU_YAML, relative_to=paths.root)
    for name in (RESULT_MANIFEST_JSON, MASK_IOU_JSON, FIDELITY_CSV):
        record.add_input(paths.reports / name, relative_to=paths.root)
    for name in (ANALYSIS_JSON, ANALYSIS_MD, INSTANCES_CSV):
        record.add_output(paths.reports / name, relative_to=paths.root)
    record.write_json(paths.reports / PROVENANCE_JSON)

    print(CLASSIFICATION)
    print(f"bands      {analysis['band_census']}")
    print(
        f"person     gap {analysis['person_diagnostic']['box_minus_mask']}  "
        f"matched IoU {analysis['person_diagnostic']['matched_mask_iou_mean']}"
    )
    print(f"table      reports/{INSTANCES_CSV}  sha256 {csv_sha256}")
    print(f"analysis   reports/{ANALYSIS_JSON}  sha256 {analysis_sha256}")
    print(f"report     reports/{ANALYSIS_MD}")
    print(f"segmenter  {analysis['final_segmenter']}")
    print(f"holdout    {HOLDOUT_STATUS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
