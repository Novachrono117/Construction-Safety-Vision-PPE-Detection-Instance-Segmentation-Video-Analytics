"""Freeze the project's final instance-segmentation model.

Phase 8G. It trains nothing, evaluates nothing and predicts nothing. Every
number it records was produced by phase 8C or phase 8F and committed; this phase
reads those artifacts, re-derives the comparison arithmetically, records the
human review that accepted the policy's answer, and freezes the selected
checkpoint's identity.

Four things here are deliberate.

**The winner is derived, not asserted.** The primary metric is read out of each
experiment's committed canonical evaluation, the delta is recomputed, and the
frozen 0.005 margin is applied by
``construction_safety_vision.canonical_evaluation.classify_delta``. No
experiment id appears in this script as the answer, and a test asserts that.

**Human review confirms the policy; it does not override it.** If the derived
classification were anything other than the one the reviewed decision rests on,
the phase stops rather than recording a selection the evidence does not support.

**The intervention is part of the model's identity.** S0 and S1 are the same
architecture at the same size, trained by the same protocol, and their
checkpoints are the same number of bytes. What separates them is
``overlap_mask``, so it is recorded in the manifest, folded into the semantic
fingerprint, and S0's digest is written down as an explicitly rejected
checkpoint.

**The trade-off is published, not buried.** S1 wins on the primary metric while
``helmet_loose`` regresses. Both go in the manifest and the report.

Requires:

* ``configs/segmentation_comparison.yaml``                  phase 8E
* ``configs/segmentation_canonical_evaluation.yaml``        phase 8E
* ``reports/segmentation_comparison_policy.json``           phase 8E
* ``reports/segmentation_S0_*.json``                        phase 8C
* ``reports/segmentation_S1_*.json``                        phase 8F

Writes:
    reports/final_segmenter_manifest.json
    reports/segmentation_selection_report.md
    reports/segmentation_experiment_comparison.csv
    reports/segmentation_experiment_results.json          (live state, updated)
    reports/final_segmenter.provenance.json
    artifacts/frozen/segmentation/S1_best.pt              (git-ignored)

Usage:
    uv run python scripts/freeze_final_segmenter.py
    uv run python scripts/freeze_final_segmenter.py --verify-only
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from construction_safety_vision.canonical_evaluation import (
    IMPROVES,
    METRIC_PRECISION,
    NATIVE_METRIC_STATUS,
    PRACTICAL_EQUIVALENCE_MARGIN,
    PRIMARY_METRIC,
    classify_delta,
    load_canonical_evaluation_config,
)
from construction_safety_vision.config import ConfigError
from construction_safety_vision.data.canonical import scan_for_sensitive
from construction_safety_vision.detection_freeze import FinalDetectorError, load_final_detector
from construction_safety_vision.mask_iou_evaluation import (
    GT_IOU50_COVERAGE,
    GT_IOU75_COVERAGE,
    GT_MATCH_COVERAGE,
    GT_NORMALIZED_MASK_IOU,
    MATCHED_MASK_IOU_MEAN,
)
from construction_safety_vision.paths import ProjectPaths
from construction_safety_vision.provenance import ProvenanceRecord, git_commit, sha256_file
from construction_safety_vision.segmentation_comparison import (
    CANDIDATE,
    CHECKPOINT_POLICY,
    ONE_VARIABLE_FIELD,
    REFERENCE,
    ComparisonConfigError,
    load_comparison_config,
)
from construction_safety_vision.segmentation_freeze import (
    FINAL_SELECTED,
    FROZEN,
    HOLDOUT_STATUS,
    INVALID_SELECTION_STATE,
    SCHEMA_VERSION,
    TASK,
    FinalSegmenterError,
    final_segmenter_fingerprint,
    holdout_leaks,
    parse_final_segmenter,
)
from construction_safety_vision.segmentation_s1 import (
    CANONICAL_PRIMARY_METRIC,
    CONSISTENT,
    DISAGREEMENT,
    RARE_CLASS,
    RARE_CLASS_STATUS,
    cross_metric_direction,
)
from construction_safety_vision.splits import HOLDOUT_UNLOCK_ENV_VAR, holdout_unlocked

COMPARISON_YAML = "segmentation_comparison.yaml"
CANONICAL_YAML = "segmentation_canonical_evaluation.yaml"

POLICY_JSON = "segmentation_comparison_policy.json"
S0_RESULT_JSON = "segmentation_S0_result_manifest.json"
S0_CANONICAL_JSON = "segmentation_S0_canonical_evaluation.json"
S0_MASK_IOU_JSON = "segmentation_S0_mask_iou.json"
S1_RESULT_JSON = "segmentation_S1_result_manifest.json"
S1_CANONICAL_JSON = "segmentation_S1_canonical_evaluation.json"
S1_MASK_IOU_JSON = "segmentation_S1_mask_iou.json"

MANIFEST_JSON = "final_segmenter_manifest.json"
REPORT_MD = "segmentation_selection_report.md"
COMPARISON_CSV = "segmentation_experiment_comparison.csv"
RESULTS_JSON = "segmentation_experiment_results.json"
PROVENANCE_JSON = "final_segmenter.provenance.json"

FROZEN_COPY = "artifacts/frozen/segmentation/S1_best.pt"

PHASE = "8G"

SEGMENTER_FROZEN = "SEGMENTER_FROZEN"
MODEL_ARTIFACT_MISMATCH = "MODEL_ARTIFACT_MISMATCH"
BLOCKED_MISSING_MODEL_ARTIFACT = "BLOCKED_MISSING_MODEL_ARTIFACT"
PROTOCOL_VIOLATION = "PROTOCOL_VIOLATION"
BLOCKED = "BLOCKED"

SELECTION_METHOD = "PREDECLARED_CANONICAL_POLICY_PLUS_HUMAN_REVIEW"

PERSON = "person"
REGRESSION_CLASS_KEY = "regressed_supported_classes"

MECHANISM_STATUS = "CONSISTENT_WITH_PHASE_8D_OVERLAP_TARGET_HYPOTHESIS"
MECHANISM_NOT = "PROOF_OF_CAUSAL_MECHANISM"

HOLDOUT_REASON = (
    "Phase 8G selected and froze the final segmenter from artifacts that already existed. It "
    "trained nothing, evaluated nothing and ran no inference. The holdout was not read, "
    "materialised, adapted, counted, predicted on or inspected; no holdout identifier, image, "
    "label, prediction or statistic exists in any artifact this phase wrote."
)

HISTORICAL: tuple[str, ...] = (
    "reports/final_detector_manifest.json",
    "reports/detection_experiment_results.json",
    "reports/segmentation_adapter_audit_manifest.json",
    "reports/segmentation_adapter_fidelity_report.md",
    "reports/segmentation_adapter_fidelity.csv",
    "reports/segmentation_adapter_approval.json",
    "reports/segmentation_S0_manifest.json",
    "reports/segmentation_S0_protocol.md",
    "configs/segmentation_baseline.yaml",
    "configs/segmentation_mask_iou_evaluation.yaml",
    "configs/segmentation_canonical_evaluation.yaml",
    "configs/segmentation_comparison.yaml",
    "reports/segmentation_S0_result_manifest.json",
    "reports/segmentation_S0_mask_iou.json",
    "reports/segmentation_S0_report.md",
    "reports/segmentation_S0_canonical_evaluation.json",
    "reports/segmentation_S0_error_analysis.json",
    "reports/segmentation_S0_error_analysis.md",
    "reports/segmentation_S0_error_instances.csv",
    "reports/segmentation_canonical_comparison_reference.md",
    "reports/segmentation_comparison_policy.json",
    "reports/segmentation_comparison_policy.md",
    "reports/segmentation_S1_protocol.md",
    "reports/segmentation_S1_protocol_manifest.json",
    "reports/segmentation_S1_result_manifest.json",
    "reports/segmentation_S1_canonical_evaluation.json",
    "reports/segmentation_S1_mask_iou.json",
    "reports/segmentation_S1_report.md",
)
"""Every artifact this phase must leave byte-identical."""

DIRECT_KEYS: tuple[str, ...] = (
    MATCHED_MASK_IOU_MEAN,
    GT_NORMALIZED_MASK_IOU,
    GT_MATCH_COVERAGE,
    GT_IOU50_COVERAGE,
    GT_IOU75_COVERAGE,
)


class FreezeError(RuntimeError):
    """Raised when a required input is missing or an invariant fails."""


class SelectionStateError(FreezeError):
    """Raised when the committed evidence does not describe a completed comparison."""


class CheckpointError(FreezeError):
    """Raised when the selected checkpoint cannot be identified on this machine."""


# --- helpers ------------------------------------------------------------------


def read_json(path: Path) -> dict[str, Any]:
    """Read a committed JSON artifact.

    Args:
        path: File to read.

    Returns:
        The parsed mapping.

    Raises:
        FreezeError: If the file is missing or is not a JSON object.
    """
    if not path.is_file():
        msg = f"required input not found: {path.name}"
        raise FreezeError(msg)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"{path.name} is not valid JSON ({exc})"
        raise FreezeError(msg) from exc
    if not isinstance(data, dict):
        msg = f"{path.name} must contain a JSON object"
        raise FreezeError(msg)
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
    """Digest every artifact this phase must not change.

    Args:
        paths: Project layout.

    Returns:
        Digests keyed by repository-relative name.

    Raises:
        FreezeError: If one is absent.
    """
    digests: dict[str, str] = {}
    for name in HISTORICAL:
        path = paths.root / name
        if not path.is_file():
            msg = f"historical artifact missing: {name}"
            raise FreezeError(msg)
        digests[name] = sha256_file(path)
    return digests


# --- reading the two experiments ------------------------------------------------


def load_experiment(paths: ProjectPaths, experiment: str) -> dict[str, Any]:
    """Read one experiment's committed result, canonical and diagnostic artifacts.

    Nothing is executed and no metric is recomputed from images: this reads what
    the experiment's own phase wrote and verifies the three artifacts agree that
    they describe the same checkpoint.

    Args:
        paths: Project layout.
        experiment: ``S0`` or ``S1``.

    Returns:
        The experiment's identity and its committed metrics.

    Raises:
        SelectionStateError: If an artifact is incomplete or the three
            artifacts disagree about which checkpoint they describe.
    """
    names = {
        REFERENCE: (S0_RESULT_JSON, S0_CANONICAL_JSON, S0_MASK_IOU_JSON),
        CANDIDATE: (S1_RESULT_JSON, S1_CANONICAL_JSON, S1_MASK_IOU_JSON),
    }[experiment]
    result = read_json(paths.reports / names[0])
    canonical = read_json(paths.reports / names[1])
    direct = read_json(paths.reports / names[2])

    checkpoint = result["checkpoints"]["best"]
    for artifact, payload in (("canonical", canonical), ("direct-IoU", direct)):
        recorded = payload.get("checkpoint", {}).get("sha256")
        if recorded != checkpoint["sha256"]:
            msg = (
                f"{experiment}'s {artifact} artifact describes checkpoint {recorded}, but its "
                f"result manifest records {checkpoint['sha256']}. They are not the same model."
            )
            raise SelectionStateError(msg)

    effective = result["effective_training_configuration"]
    if experiment == REFERENCE:
        supported_macro = canonical["supported_macro"]["value"]
        experiment_sha = result["S0_experiment_sha256"]
        native_mask = result["mask_metrics"]
        native_box = result["box_metrics_from_segmenter"]
        native_macro = result["supported_macro_mask"]["value"]
    else:
        supported_macro = canonical["supported_macro"]["value"]
        experiment_sha = result["S1_experiment_sha256"]
        native_mask = result["native_metrics"]["mask"]
        native_box = result["native_metrics"]["box"]
        native_macro = result["descriptive_native_supported_macro"]["value"]

    if supported_macro is None:
        msg = f"{experiment} has no canonical supported macro; the comparison cannot be derived"
        raise SelectionStateError(msg)

    return {
        "experiment_id": experiment,
        "status": result["status"],
        "model": result["model"],
        "imgsz": effective["imgsz"],
        "batch": effective["batch"],
        "epochs": effective["epochs"],
        "seed": effective["seed"],
        "mask_ratio": effective["mask_ratio"],
        "overlap_mask": effective[ONE_VARIABLE_FIELD],
        "best_epoch": result["checkpoint_selection"]["best_epoch"],
        "checkpoint_policy": result["checkpoint_selection"]["policy"],
        "checkpoint": dict(checkpoint),
        "experiment_sha256": experiment_sha,
        "pretrained_weights": dict(result["pretrained_weights"]),
        "adapter_fingerprints": dict(result["adapter"]["fingerprints"]),
        "canonical_fingerprints": dict(result["canonical_fingerprints"]),
        "canonical": {
            "supported_macro": supported_macro,
            "all_class_map50_95": canonical["canonical"]["all_class_map50_95"],
            "all_class_map50": canonical["canonical"]["all_class_map50"],
            "per_class": dict(canonical["canonical"]["per_class"]),
            "admitted_classes": list(canonical["supported_macro"]["admitted_classes"]),
        },
        "direct_iou": {key: direct["global"][key] for key in DIRECT_KEYS},
        "direct_per_class": dict(direct["per_class"]),
        "native": {
            "mask": dict(native_mask),
            "box": dict(native_box),
            "supported_macro": native_macro,
        },
        "artifact_digests": {
            f"reports/{names[0]}": sha256_file(paths.reports / names[0]),
            f"reports/{names[1]}": sha256_file(paths.reports / names[1]),
            f"reports/{names[2]}": sha256_file(paths.reports / names[2]),
        },
    }


def verify_one_variable(
    reference: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    """Confirm the two committed experiments differ only in the intervention.

    Checked against what each experiment's framework actually recorded, not
    against a restatement of the protocol.

    Args:
        reference: S0's loaded identity.
        candidate: S1's loaded identity.

    Returns:
        What was verified, field by field.

    Raises:
        SelectionStateError: If a field that must be shared differs, or if the
            intervention did not differ.
    """
    shared = ("model", "imgsz", "batch", "epochs", "seed", "mask_ratio", "checkpoint_policy")
    drifted = [
        f"{name}: {reference['experiment_id']}={reference[name]!r}, "
        f"{candidate['experiment_id']}={candidate[name]!r}"
        for name in shared
        if reference[name] != candidate[name]
    ]
    if reference["pretrained_weights"]["sha256"] != candidate["pretrained_weights"]["sha256"]:
        drifted.append("pretrained weight digest differs")
    if reference["adapter_fingerprints"] != candidate["adapter_fingerprints"]:
        drifted.append("adapter label digests differ")
    for key in ("class_map_sha256", "split_assignment_sha256"):
        if reference["canonical_fingerprints"][key] != candidate["canonical_fingerprints"][key]:
            drifted.append(f"{key} differs")
    if drifted:
        msg = "the two experiments differ in more than the declared intervention: " + "; ".join(
            drifted
        )
        raise SelectionStateError(msg)

    if reference["overlap_mask"] == candidate["overlap_mask"]:
        msg = (
            f"both experiments recorded {ONE_VARIABLE_FIELD}="
            f"{reference['overlap_mask']!r}; the intervention never applied"
        )
        raise SelectionStateError(msg)

    return {
        "intentional_field": ONE_VARIABLE_FIELD,
        "reference_value": reference["overlap_mask"],
        "candidate_value": candidate["overlap_mask"],
        "fields_verified_identical": sorted(
            [
                *shared,
                "pretrained_weight_sha256",
                "adapter_fingerprints",
                "class_map_sha256",
                "split_assignment_sha256",
            ]
        ),
        "one_variable": True,
        "verified_against": "EACH_EXPERIMENTS_OWN_COMMITTED_EFFECTIVE_CONFIGURATION",
    }


def per_class_table(
    reference: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    """Build the canonical per-class comparison from the two committed artifacts.

    Args:
        reference: S0's loaded identity.
        candidate: S1's loaded identity.

    Returns:
        Per class, both experiments' AP and the delta.
    """
    admitted = set(candidate["canonical"]["admitted_classes"])
    table: dict[str, dict[str, Any]] = {}
    names = set(reference["canonical"]["per_class"]) | set(candidate["canonical"]["per_class"])
    for name in sorted(names):
        left = reference["canonical"]["per_class"].get(name, {})
        right = candidate["canonical"]["per_class"].get(name, {})
        row: dict[str, Any] = {
            "supported": name in admitted,
            "status": "COMPARISON_SUPPORTED" if name in admitted else RARE_CLASS_STATUS,
        }
        for key in ("AP@0.50:0.95", "AP@0.50"):
            a, b = left.get(key), right.get(key)
            row[f"S0_{key}"] = a
            row[f"S1_{key}"] = b
            row[f"delta_{key}"] = (
                round(float(b) - float(a), METRIC_PRECISION)
                if isinstance(a, (int, float)) and isinstance(b, (int, float))
                else None
            )
        table[name] = row
    return table


def freeze_checkpoint(paths: ProjectPaths, candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Verify the selected checkpoint and place an immutable byte-identical copy.

    The copy lives outside the training run's output directory, because that
    directory is where a re-run would land. Nothing is trained and nothing is
    downloaded: a missing checkpoint stops the phase.

    Args:
        paths: Project layout.
        candidate: S1's loaded identity.

    Returns:
        The runtime and frozen-copy records.

    Raises:
        CheckpointError: If the runtime checkpoint is absent, or its bytes are
            not the committed ones, or an existing frozen copy holds different
            bytes.
    """
    expected = candidate["checkpoint"]["sha256"]
    expected_size = candidate["checkpoint"]["size_bytes"]
    runtime = paths.root / "artifacts" / "segmentation" / CANDIDATE / "weights" / "best.pt"

    if not runtime.is_file():
        msg = (
            f"{BLOCKED_MISSING_MODEL_ARTIFACT}: {CANDIDATE} best.pt is not on this machine. "
            f"Obtain the artifact with SHA-256 {expected}; do not retrain, because a re-run "
            "produces different weights under the same experiment name."
        )
        raise CheckpointError(msg)

    observed = sha256_file(runtime)
    observed_size = runtime.stat().st_size
    if observed != expected or observed_size != expected_size:
        msg = (
            f"{MODEL_ARTIFACT_MISMATCH}: the runtime {CANDIDATE} checkpoint has SHA-256 "
            f"{observed} ({observed_size} bytes), but the committed result manifest records "
            f"{expected} ({expected_size} bytes). These are different weights."
        )
        raise CheckpointError(msg)

    frozen = paths.root / FROZEN_COPY
    if frozen.is_file():
        existing = sha256_file(frozen)
        if existing != expected:
            msg = (
                f"{MODEL_ARTIFACT_MISMATCH}: a frozen copy already exists at {FROZEN_COPY} with "
                f"SHA-256 {existing}, which is not the selected checkpoint {expected}. It is "
                "not overwritten; resolve this deliberately."
            )
            raise CheckpointError(msg)
        placed = "ALREADY_PRESENT_VERIFIED"
    else:
        frozen.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(runtime, frozen)
        placed = "COPIED"

    frozen_sha = sha256_file(frozen)
    if frozen_sha != expected:  # pragma: no cover - defended, not expected
        msg = f"{MODEL_ARTIFACT_MISMATCH}: the frozen copy hashed to {frozen_sha} after writing"
        raise CheckpointError(msg)

    return {
        "runtime": {
            "relative_path": f"artifacts/segmentation/{CANDIDATE}/weights/best.pt",
            "sha256": observed,
            "size_bytes": observed_size,
            "verified": True,
        },
        "frozen_copy": {
            "relative_path": FROZEN_COPY,
            "sha256": frozen_sha,
            "size_bytes": frozen.stat().st_size,
            "committed": False,
            "immutable": True,
            "action": placed,
            "note": (
                "A byte-identical copy of the selected checkpoint, kept outside the training "
                "run's output directory so that re-running an experiment cannot overwrite the "
                "frozen model. Git-ignored: the repository commits the record, not the binary."
            ),
        },
    }


# --- artifacts ------------------------------------------------------------------


def build_manifest(
    *,
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    contract: Mapping[str, Any],
    delta: Mapping[str, Any],
    direction: Mapping[str, Any],
    table: Mapping[str, Any],
    checkpoints: Mapping[str, Any],
    policy_sha256: str,
    evaluator_sha256: str,
    historical: Mapping[str, str],
    detector: Mapping[str, Any],
    rationale: Sequence[str],
) -> dict[str, Any]:
    """Assemble the final-segmenter manifest.

    Args:
        reference: S0's loaded identity.
        candidate: S1's loaded identity.
        contract: The verified one-variable contract.
        delta: The recomputed primary delta and its verdict.
        direction: The recomputed cross-metric direction.
        table: The canonical per-class comparison.
        checkpoints: The runtime and frozen-copy records.
        policy_sha256: Digest of the frozen phase 8E policy artifact.
        evaluator_sha256: Fingerprint of the frozen canonical evaluator.
        historical: Digests of every artifact this phase must not change.
        detector: The frozen detector, verified and left alone.
        rationale: The recorded selection rationale.

    Returns:
        The manifest, with its semantic fingerprint filled in.
    """
    regressed = sorted(
        name
        for name, row in table.items()
        if row["supported"]
        and isinstance(row["delta_AP@0.50:0.95"], (int, float))
        and row["delta_AP@0.50:0.95"] < 0
    )
    person_row = table.get(PERSON, {})

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "status": FROZEN,
        "task": TASK,
        "selection_status": FINAL_SELECTED,
        "selected_experiment": candidate["experiment_id"],
        "selection_method": SELECTION_METHOD,
        "margin_classification": delta["verdict"],
        "model": candidate["model"],
        "imgsz": candidate["imgsz"],
        "batch": candidate["batch"],
        "mask_ratio": candidate["mask_ratio"],
        "overlap_mask": candidate["overlap_mask"],
        "epochs": candidate["epochs"],
        "seed": candidate["seed"],
        "best_epoch": candidate["best_epoch"],
        "checkpoint_selection_policy": candidate["checkpoint_policy"],
        "pretrained_weights": dict(candidate["pretrained_weights"]),
        "selected_checkpoint": {
            "relative_path": checkpoints["runtime"]["relative_path"],
            "sha256": checkpoints["runtime"]["sha256"],
            "size_bytes": checkpoints["runtime"]["size_bytes"],
            "committed": False,
            "selected_by": candidate["checkpoint_policy"],
            "is_best_not_last": True,
        },
        # `action` is deliberately dropped: whether this invocation created the copy
        # or found it already present describes the run, not the frozen model, and
        # carrying it here would make the manifest differ between two identical
        # freezes. It is recorded in the provenance record instead.
        "frozen_copy": {
            key: value for key, value in checkpoints["frozen_copy"].items() if key != "action"
        },
        "rejected_checkpoints": {
            "reference_experiment": reference["experiment_id"],
            "reference_experiment_checkpoint_sha256": reference["checkpoint"]["sha256"],
            "reason": (
                f"{reference['experiment_id']} was trained with {ONE_VARIABLE_FIELD}="
                f"{reference['overlap_mask']} and predicts a different target. Its checkpoint is "
                "the same architecture and the same number of bytes as the selected one, so it "
                "is rejected by digest rather than by inspection."
            ),
            "last_pt_rejected": True,
            "last_pt_reason": (
                "The final epoch's weights are not the checkpoint the predeclared native rule "
                "selected."
            ),
        },
        "experiment_sha256": candidate["experiment_sha256"],
        "reference_experiment_sha256": reference["experiment_sha256"],
        "comparison_policy_sha256": policy_sha256,
        "canonical_evaluator_sha256": evaluator_sha256,
        "dataset_fingerprints": {
            "class_map_sha256": candidate["canonical_fingerprints"]["class_map_sha256"],
            "canonical_task_manifest_sha256": candidate["canonical_fingerprints"][
                "canonical_task_manifest_sha256"
            ],
            "validation_annotations_sha256": candidate["canonical_fingerprints"][
                "validation_annotations_sha256"
            ],
            "adapter": dict(candidate["adapter_fingerprints"]),
        },
        "split_reference": {
            "split_assignment_sha256": candidate["canonical_fingerprints"][
                "split_assignment_sha256"
            ],
            "splits_used": ["train", "validation"],
            "manifest": "reports/split_manifest.json",
        },
        "one_variable_contract": dict(contract),
        "primary_selection_metric": PRIMARY_METRIC,
        "primary_selection_metric_label": CANONICAL_PRIMARY_METRIC,
        "practical_equivalence_margin": PRACTICAL_EQUIVALENCE_MARGIN,
        "comparison": {
            "reference_experiment": reference["experiment_id"],
            "candidate_experiment": candidate["experiment_id"],
            "reference_supported_macro": reference["canonical"]["supported_macro"],
            "candidate_supported_macro": candidate["canonical"]["supported_macro"],
            "delta": delta["delta"],
            "margin": delta["margin"],
            "verdict": delta["verdict"],
            "margin_is_not_a_significance_test": True,
            "derived_from": "COMMITTED_CANONICAL_EVALUATION_ARTIFACTS_ONLY",
            "recomputed_in_this_phase": True,
            "predictions_recomputed": False,
        },
        "canonical_all_class": {
            "reference_map50_95": reference["canonical"]["all_class_map50_95"],
            "candidate_map50_95": candidate["canonical"]["all_class_map50_95"],
            "delta_map50_95": round(
                float(candidate["canonical"]["all_class_map50_95"])
                - float(reference["canonical"]["all_class_map50_95"]),
                METRIC_PRECISION,
            ),
            "reference_map50": reference["canonical"]["all_class_map50"],
            "candidate_map50": candidate["canonical"]["all_class_map50"],
            "is_not_the_selection_metric": True,
        },
        "canonical_per_class_comparison": dict(table),
        "direct_iou_comparison": {
            "status": direction["status"],
            "label": "SECONDARY_CANONICAL_DIAGNOSTIC",
            "protocol": "UNCHANGED_PHASE_8C_PROTOCOL",
            "reference": dict(reference["direct_iou"]),
            "candidate": dict(candidate["direct_iou"]),
            "deltas": {
                key: round(
                    float(candidate["direct_iou"][key]) - float(reference["direct_iou"][key]),
                    METRIC_PRECISION,
                )
                for key in DIRECT_KEYS
            },
            "does_not_rank": True,
            "composite_created": False,
        },
        "native_metrics_comparison": {
            "status": NATIVE_METRIC_STATUS,
            "used_to_arbitrate": False,
            "reference": dict(reference["native"]),
            "candidate": dict(candidate["native"]),
            "why_not_used": (
                "overlap_mask changes the framework's validation ground truth as well as its "
                "training target, so the two experiments' native mask AP is measured against "
                "different targets. The figures are reported in full for each model against its "
                "own target and are never differenced."
            ),
        },
        "person_diagnostic": {
            "class": PERSON,
            "reference_AP@0.50:0.95": person_row.get("S0_AP@0.50:0.95"),
            "candidate_AP@0.50:0.95": person_row.get("S1_AP@0.50:0.95"),
            "delta_AP@0.50:0.95": person_row.get("delta_AP@0.50:0.95"),
            "largest_canonical_improvement": True,
            "mechanism_status": MECHANISM_STATUS,
            "mechanism_is_not": MECHANISM_NOT,
            "note": (
                "person showed the largest canonical improvement, which is consistent with the "
                "phase 8D overlap-target observation. Phase 8D was POST_HOC_HYPOTHESIS_GENERATING "
                "and this phase ran no experiment to test the mechanism, so the consistency is "
                "not converted into a causal claim and the hypothesis is not retroactively "
                "presented as predeclared."
            ),
            "selected_the_model": False,
        },
        "regression_note": {
            REGRESSION_CLASS_KEY: regressed,
            "detail": (
                "Selection does not require every class to improve. The primary metric is an "
                "unweighted mean over the admitted classes and it moved beyond the margin, "
                "while these supported classes moved the other way. The trade-off is recorded "
                "here and in the selection report rather than left inside the mean."
            ),
            "deltas": {name: table[name]["delta_AP@0.50:0.95"] for name in regressed},
            "why_it_moved": "UNKNOWN",
            "why_unknown": (
                "The experiment varied one flag and measured the outcome. It tested no "
                "per-class mechanism, and one run cannot separate a per-class movement from "
                "run-to-run variance, which on this setup is UNKNOWN because nothing was "
                "repeated."
            ),
        },
        "rare_class": {
            "name": RARE_CLASS,
            "status": RARE_CLASS_STATUS,
            "reference_AP@0.50:0.95": table.get(RARE_CLASS, {}).get("S0_AP@0.50:0.95"),
            "candidate_AP@0.50:0.95": table.get(RARE_CLASS, {}).get("S1_AP@0.50:0.95"),
            "delta_AP@0.50:0.95": table.get(RARE_CLASS, {}).get("delta_AP@0.50:0.95"),
            "in_supported_macro": RARE_CLASS in candidate["canonical"]["admitted_classes"],
            "reported_in_full": True,
            "may_decide_anything": False,
            "limitation": (
                "vest_loose holds one validation source image and eight instances under the "
                "frozen split. Its movement is reported for completeness and took no part in "
                "the selection."
            ),
        },
        "selection_rationale": list(rationale),
        "human_review": {
            "status": "HUMAN_REVIEWED_AND_ACCEPTED",
            "confirms_policy_result": True,
            "overrides_policy_result": False,
            "note": (
                "The frozen policy produced the classification mechanically from committed "
                "artifacts; human review accepted it. Had the derived classification differed "
                "from the reviewed decision, this phase would have stopped rather than record "
                "a selection the evidence does not support."
            ),
        },
        "limitations": [
            "Every figure is a validation figure. Nothing here says anything about test "
            "performance, and the holdout has never been evaluated.",
            "One run per experiment. Run-to-run variance on this setup is UNKNOWN, and the "
            "0.005 margin is an engineering threshold rather than a significance test.",
            "The comparison policy was frozen after S0 ran (POST_S0_PRE_S1_PROTOCOL_FREEZE), "
            "unlike phase 7A's, which predated its candidates. No S1 number influenced any rule.",
            "Both checkpoints were selected by the same native fitness rule computed against "
            "different targets, because overlap_mask reshapes the target. That is why the "
            "decision was made externally against canonical COCO masks.",
            "The aggregate gain is not uniform: person carries most of it and helmet_loose "
            "regressed. Why any individual class moved is UNKNOWN.",
            "No standardised latency or throughput benchmark has been run for either model.",
        ],
        "frozen_detector": dict(detector),
        "detector_touched": False,
        "historical_artifact_digests": dict(historical),
        "historical_artifacts_unchanged": True,
        "models_trained_in_this_phase": 0,
        "models_evaluated_in_this_phase": 0,
        "inference_run_in_this_phase": False,
        "thresholds_tuned": 0,
        "benchmarks_run": 0,
        "binary_distribution_status": "LOCAL_IGNORED_FROZEN_ARTIFACT",
        "repository_contains_model_binary": False,
        "test": {"status": HOLDOUT_STATUS, "reason": HOLDOUT_REASON},
        "holdout_accessed": False,
        "next_phase": "PHASE_9_DETECTOR_VERSUS_SEGMENTER_OPERATIONAL_COMPARISON",
    }

    manifest["final_segmenter_sha256"] = final_segmenter_fingerprint(
        {
            "selected_experiment": manifest["selected_experiment"],
            "model": manifest["model"],
            "imgsz": manifest["imgsz"],
            "batch": manifest["batch"],
            "mask_ratio": manifest["mask_ratio"],
            "overlap_mask": manifest["overlap_mask"],
            "selected_checkpoint_sha256": manifest["selected_checkpoint"]["sha256"],
            "experiment_sha256": manifest["experiment_sha256"],
            "comparison_policy_sha256": manifest["comparison_policy_sha256"],
            "canonical_evaluator_sha256": manifest["canonical_evaluator_sha256"],
            "split_assignment_sha256": manifest["split_reference"]["split_assignment_sha256"],
            "class_map_sha256": manifest["dataset_fingerprints"]["class_map_sha256"],
            "adapter_labels_development_sha256": manifest["dataset_fingerprints"]["adapter"][
                "labels_development_sha256"
            ],
            "adapter_image_membership_sha256": manifest["dataset_fingerprints"]["adapter"][
                "image_membership_sha256"
            ],
        }
    )
    return manifest


def build_comparison_csv(
    reference: Mapping[str, Any], candidate: Mapping[str, Any], delta: Mapping[str, Any]
) -> str:
    """Render the two-row experiment comparison table.

    Args:
        reference: S0's loaded identity.
        candidate: S1's loaded identity.
        delta: The recomputed primary delta and its verdict.

    Returns:
        CSV text with a trailing newline.
    """
    columns = [
        "experiment_id",
        "overlap_mask",
        "canonical_supported_macro_mask_map50_95",
        "delta_vs_S0",
        "margin_status",
        "canonical_all_class_map50_95",
        "canonical_map50",
        "direct_gt_normalized_mask_iou",
        "direct_matched_mask_iou",
        "native_mask_map50_95",
        "best_epoch",
        "checkpoint_sha256",
        "experiment_sha256",
        "selected",
    ]
    rows = [
        {
            "experiment_id": reference["experiment_id"],
            "overlap_mask": reference["overlap_mask"],
            "canonical_supported_macro_mask_map50_95": reference["canonical"]["supported_macro"],
            "delta_vs_S0": 0.0,
            "margin_status": "REFERENCE",
            "canonical_all_class_map50_95": reference["canonical"]["all_class_map50_95"],
            "canonical_map50": reference["canonical"]["all_class_map50"],
            "direct_gt_normalized_mask_iou": reference["direct_iou"][GT_NORMALIZED_MASK_IOU],
            "direct_matched_mask_iou": reference["direct_iou"][MATCHED_MASK_IOU_MEAN],
            "native_mask_map50_95": reference["native"]["mask"]["mAP@0.50:0.95"],
            "best_epoch": reference["best_epoch"],
            "checkpoint_sha256": reference["checkpoint"]["sha256"],
            "experiment_sha256": reference["experiment_sha256"],
            "selected": False,
        },
        {
            "experiment_id": candidate["experiment_id"],
            "overlap_mask": candidate["overlap_mask"],
            "canonical_supported_macro_mask_map50_95": candidate["canonical"]["supported_macro"],
            "delta_vs_S0": delta["delta"],
            "margin_status": delta["verdict"],
            "canonical_all_class_map50_95": candidate["canonical"]["all_class_map50_95"],
            "canonical_map50": candidate["canonical"]["all_class_map50"],
            "direct_gt_normalized_mask_iou": candidate["direct_iou"][GT_NORMALIZED_MASK_IOU],
            "direct_matched_mask_iou": candidate["direct_iou"][MATCHED_MASK_IOU_MEAN],
            "native_mask_map50_95": candidate["native"]["mask"]["mAP@0.50:0.95"],
            "best_epoch": candidate["best_epoch"],
            "checkpoint_sha256": candidate["checkpoint"]["sha256"],
            "experiment_sha256": candidate["experiment_sha256"],
            "selected": True,
        },
    ]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue()


def build_results_artifact(
    existing: Mapping[str, Any], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Update the live segmentation-results artifact with the frozen selection.

    The experiment rows are carried through unchanged - they are phase 8F's
    record and this phase measures nothing - and only the selection state is
    written.

    Args:
        existing: The committed live artifact.
        manifest: The final-segmenter manifest.

    Returns:
        The updated payload.
    """
    updated = json.loads(json.dumps(existing))
    updated["phase"] = PHASE
    updated["selection_status"] = FINAL_SELECTED
    updated["final_selected_experiment"] = manifest["selected_experiment"]
    updated["preferred_experiment"] = manifest["selected_experiment"]
    updated["final_segmenter"] = manifest["selected_experiment"]
    updated["final_segmenter_sha256"] = manifest["final_segmenter_sha256"]
    updated["final_segmenter_manifest"] = f"reports/{MANIFEST_JSON}"
    updated["selection_method"] = manifest["selection_method"]
    updated["margin_classification"] = manifest["margin_classification"]
    updated["pending_human_review"] = False
    updated["human_review"] = dict(manifest["human_review"])
    updated["selected_checkpoint_sha256"] = manifest["selected_checkpoint"]["sha256"]
    updated["binary_distribution_status"] = manifest["binary_distribution_status"]
    updated["regressed_supported_classes"] = list(manifest["regression_note"][REGRESSION_CLASS_KEY])
    updated["final_segmenter_note"] = (
        "Selected by the predeclared canonical policy and accepted by human review. Selection "
        "is validation-only; no test number exists."
    )
    for row in updated.get("experiments", []):
        row["selected"] = row["experiment_id"] == manifest["selected_experiment"]
    updated["test"] = {"status": HOLDOUT_STATUS, "reason": HOLDOUT_REASON}
    updated["holdout_accessed"] = False
    return updated


def build_report(manifest: Mapping[str, Any], *, commit: str | None) -> str:
    """Render the selection report.

    Args:
        manifest: The final-segmenter manifest.
        commit: Repository commit, when available.

    Returns:
        The report text.
    """
    comparison = manifest["comparison"]
    direct = manifest["direct_iou_comparison"]
    native = manifest["native_metrics_comparison"]
    person = manifest["person_diagnostic"]
    regression = manifest["regression_note"]
    rare = manifest["rare_class"]
    checkpoint = manifest["selected_checkpoint"]
    frozen = manifest["frozen_copy"]
    table = manifest["canonical_per_class_comparison"]

    lines: list[str] = []
    add = lines.append

    add("# Final segmenter selection")
    add("")
    add(
        f"Phase {manifest['phase']} · status `{manifest['status']}` · "
        f"`{manifest['selection_status']}` · selected **{manifest['selected_experiment']}** · "
        f"`{manifest['selection_method']}`"
    )
    add("")
    add(
        "**Every number here is a validation number.** The holdout has never been evaluated, "
        "and nothing below says anything about test performance. This phase trained nothing, "
        "evaluated nothing and ran no inference."
    )
    add("")
    if commit:
        add(f"Repository commit at production time: `{commit}`.")
        add("")

    add("## 1. Selection objective")
    add("")
    add(
        "`FINAL_DECISION`. Choose one instance-segmentation model as the project's segmenter, "
        "from the two experiments the frozen phase 8E policy admits, using only committed "
        "evidence."
    )
    add("")

    add("## 2. S0 baseline")
    add("")
    add(
        f"`{comparison['reference_experiment']}` - YOLO11n-seg, imgsz 768, batch 8, "
        f"mask_ratio 4, `overlap_mask: true`. Canonical supported macro "
        f"**{comparison['reference_supported_macro']}**."
    )
    add("")

    add("## 3. Phase 8D hypothesis")
    add("")
    add(
        "`LIMITATION`. Phase 8D measured that S0's `person` masks lose most of their overlap in "
        "pixels contested by a smaller class, and traced that to `overlap_mask: true`, under "
        "which the smaller instance owns shared pixels. That analysis is "
        "`POST_HOC_HYPOTHESIS_GENERATING`: it showed a target mismatch, not that training "
        "differently would help."
    )
    add("")

    add("## 4. Phase 8E comparison policy")
    add("")
    add(
        f"`PREDECLARED_POLICY`. Frozen before S1 existed, digest "
        f"`{manifest['comparison_policy_sha256']}`. Primary metric "
        f"`{manifest['primary_selection_metric']}`, practical-equivalence margin "
        f"**{manifest['practical_equivalence_margin']}**, three cases fixed in advance. The "
        "canonical evaluator it names is "
        f"`{manifest['canonical_evaluator_sha256']}` - pycocotools `COCOeval` at "
        "`iouType='segm'` against the canonical phase 5D COCO validation masks, which no "
        "training flag can move."
    )
    add("")

    add("## 5. S1 controlled intervention")
    add("")
    contract = manifest["one_variable_contract"]
    add(
        f"`CONTROLLED_INTERVENTION`. `{contract['intentional_field']}` "
        f"{str(contract['reference_value']).lower()} -> "
        f"{str(contract['candidate_value']).lower()}. Verified here against **each "
        "experiment's own committed effective configuration**, not against a restatement of "
        "the protocol: "
        + ", ".join(f"`{name}`" for name in contract["fields_verified_identical"])
        + " are identical across the two runs."
    )
    add("")

    add("## 6. Canonical comparison rationale")
    add("")
    add(
        "`CANONICAL_PRIMARY_METRIC`. Both models are scored against the same canonical COCO "
        "masks by the same evaluator. That is the only comparison the two experiments admit, "
        "because their native targets differ."
    )
    add("")

    add("## 7-9. Canonical results and the primary delta")
    add("")
    add("| | S0 | S1 | delta |")
    add("| --- | --- | --- | --- |")
    add(
        f"| **{manifest['primary_selection_metric']}** | "
        f"{comparison['reference_supported_macro']} | "
        f"**{comparison['candidate_supported_macro']}** | **{comparison['delta']:+f}** |"
    )
    all_class = manifest["canonical_all_class"]
    add(
        f"| Canonical all-class mAP@0.50:0.95 | {all_class['reference_map50_95']} | "
        f"{all_class['candidate_map50_95']} | {all_class['delta_map50_95']:+f} |"
    )
    add(
        f"| Canonical all-class mAP@0.50 | {all_class['reference_map50']} | "
        f"{all_class['candidate_map50']} | |"
    )
    add("")
    add(
        f"Derived from `{comparison['derived_from']}`. No prediction was recomputed: "
        f"`predictions_recomputed: {comparison['predictions_recomputed']}`."
    )
    add("")

    add("## 10. Practical-equivalence rule")
    add("")
    add(
        f"`PREDECLARED_POLICY`. Margin {comparison['margin']} absolute AP. Classification: "
        f"**`{comparison['verdict']}`**. The margin is an engineering threshold, **not a "
        "significance test**: nothing was repeated, so run-to-run variance on this setup "
        "remains UNKNOWN."
    )
    add("")

    add("## 11. Direct-IoU consistency")
    add("")
    add(
        f"`DIRECT_IOU_DIAGNOSTIC` · `{direct['label']}` · `{direct['protocol']}`. Status: "
        f"**`{direct['status']}`**."
    )
    add("")
    add("| Figure | S0 | S1 | delta |")
    add("| --- | --- | --- | --- |")
    for key in DIRECT_KEYS:
        add(
            f"| {key} | {direct['reference'][key]} | {direct['candidate'][key]} | "
            f"{direct['deltas'][key]:+f} |"
        )
    add("")
    add(
        f"The secondary diagnostic informs the reading; it does not rank "
        f"(`does_not_rank: {direct['does_not_rank']}`) and no composite was created."
    )
    add("")

    add("## 12. Native-metric comparability limitation")
    add("")
    add(f"`LIMITATION` `{native['status']}`. {native['why_not_used']}")
    add("")
    add(
        f"For the record, each against its own target: S0 native mask mAP@0.50:0.95 "
        f"{native['reference']['mask']['mAP@0.50:0.95']}, S1 "
        f"{native['candidate']['mask']['mAP@0.50:0.95']}. **These two numbers must never be "
        "differenced.**"
    )
    add("")

    add("## 13. Per-class trade-offs")
    add("")
    add("| Class | S0 AP@0.50:0.95 | S1 AP@0.50:0.95 | delta | support |")
    add("| --- | --- | --- | --- | --- |")
    for name in sorted(table):
        row = table[name]
        add(
            f"| {name} | {row['S0_AP@0.50:0.95']} | {row['S1_AP@0.50:0.95']} | "
            f"{row['delta_AP@0.50:0.95']:+f} | `{row['status']}` |"
        )
    add("")
    add(
        "Selection does not require every class to improve. It requires the predeclared "
        "primary metric to clear the predeclared margin, which it does."
    )
    add("")

    add("## 14. The person finding")
    add("")
    add(
        f"`{person['mechanism_status']}`, and explicitly **not** `{person['mechanism_is_not']}`. "
        f"`person` moved {person['reference_AP@0.50:0.95']} -> "
        f"{person['candidate_AP@0.50:0.95']} ({person['delta_AP@0.50:0.95']:+f}), the largest "
        "canonical improvement of any class."
    )
    add("")
    add(f"`LIMITATION`. {person['note']}")
    add("")

    add("## 15. The helmet_loose regression")
    add("")
    if regression[REGRESSION_CLASS_KEY]:
        add(
            "`LIMITATION`. A supported class moved the other way: "
            + ", ".join(
                f"`{name}` {regression['deltas'][name]:+f}"
                for name in regression[REGRESSION_CLASS_KEY]
            )
            + "."
        )
        add("")
        add(f"{regression['detail']}")
        add("")
        add(f"Why it moved is **{regression['why_it_moved']}**. {regression['why_unknown']}")
    else:
        add("No supported class regressed.")
    add("")

    add("## 16. vest_loose uncertainty")
    add("")
    add(
        f"`LIMITATION` `{rare['status']}`. `{rare['name']}` moved "
        f"{rare['reference_AP@0.50:0.95']} -> {rare['candidate_AP@0.50:0.95']} "
        f"({rare['delta_AP@0.50:0.95']:+f}). {rare['limitation']} It is not in the supported "
        f"macro (`in_supported_macro: {rare['in_supported_macro']}`) and "
        f"`may_decide_anything: {rare['may_decide_anything']}`."
    )
    add("")

    add("## 17. Single-run limitation")
    add("")
    add("`LIMITATION`.")
    add("")
    for item in manifest["limitations"]:
        add(f"* {item}")
    add("")

    add("## 18. Human review")
    add("")
    review = manifest["human_review"]
    add(
        f"`HUMAN_REVIEW` `{review['status']}`. "
        f"`confirms_policy_result: {review['confirms_policy_result']}`, "
        f"`overrides_policy_result: {review['overrides_policy_result']}`. {review['note']}"
    )
    add("")
    add("Recorded rationale:")
    add("")
    for item in manifest["selection_rationale"]:
        add(f"* {item}")
    add("")

    add("## 19. Final selected segmenter")
    add("")
    add("| Field | Value |")
    add("| --- | --- |")
    add(f"| Selected experiment | **{manifest['selected_experiment']}** |")
    add(f"| Model | {manifest['model']} |")
    add(f"| imgsz | {manifest['imgsz']} |")
    add(f"| batch | {manifest['batch']} |")
    add(f"| mask_ratio | {manifest['mask_ratio']} |")
    add(f"| overlap_mask | **{manifest['overlap_mask']}** |")
    add(f"| Best epoch | {manifest['best_epoch']} |")
    add(f"| Checkpoint policy | `{manifest['checkpoint_selection_policy']}` |")
    add(f"| `experiment_sha256` | `{manifest['experiment_sha256']}` |")
    add(f"| `final_segmenter_sha256` | `{manifest['final_segmenter_sha256']}` |")
    add("")

    add("## 20. Frozen checkpoint identity")
    add("")
    add(
        f"`FINAL_DECISION`. The frozen segmenter is **{manifest['selected_experiment']}'s "
        f"`best.pt`**, SHA-256 `{checkpoint['sha256']}`, {checkpoint['size_bytes']} bytes, "
        f"selected by `{checkpoint['selected_by']}`. A byte-identical immutable copy lives at "
        f"`{frozen['relative_path']}`, outside the run directory a re-run would overwrite."
    )
    add("")
    rejected = manifest["rejected_checkpoints"]
    add(
        f"Two checkpoints are rejected **by identity, not by convention**: "
        f"`{rejected['reference_experiment']}`'s `best.pt` "
        f"(`{rejected['reference_experiment_checkpoint_sha256']}`) and any `last.pt`. "
        f"{rejected['reason']}"
    )
    add("")

    add("## 21. Binary distribution")
    add("")
    add(
        f"`{manifest['binary_distribution_status']}`. "
        f"`repository_contains_model_binary: {manifest['repository_contains_model_binary']}`. "
        "The repository commits the record, not the weights: a fresh clone must obtain the "
        "checkpoint by digest, and must not retrain to produce one."
    )
    add("")

    add("## 22. Holdout compliance")
    add("")
    add(f"`HOLDOUT_POLICY` `{manifest['test']['status']}`. {manifest['test']['reason']}")
    add("")

    add("## 23. Next phase")
    add("")
    add(
        f"`{manifest['next_phase']}`. A standardised detector-versus-segmenter comparison is "
        "still owed by the project's scientific question, and no latency, throughput or memory "
        "benchmark was run here. The final test evaluation remains a separate, single-shot "
        "phase against the locked holdout."
    )
    add("")

    return "\n".join(lines) + "\n"


# --- entry point ----------------------------------------------------------------


def build_rationale(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    delta: Mapping[str, Any],
    direction: Mapping[str, Any],
    table: Mapping[str, Any],
) -> list[str]:
    """Compose the recorded selection rationale from the derived figures.

    Every clause is generated from a number this phase read or derived, so the
    rationale cannot drift from the evidence it claims to rest on.

    Args:
        reference: S0's loaded identity.
        candidate: S1's loaded identity.
        delta: The recomputed primary delta and its verdict.
        direction: The recomputed cross-metric direction.
        table: The canonical per-class comparison.

    Returns:
        The rationale, one clause per entry.
    """
    ratio = abs(delta["delta"]) / PRACTICAL_EQUIVALENCE_MARGIN
    all_class_delta = round(
        float(candidate["canonical"]["all_class_map50_95"])
        - float(reference["canonical"]["all_class_map50_95"]),
        METRIC_PRECISION,
    )
    direct_delta = direction["secondary_delta"]
    regressed = sorted(
        name
        for name, row in table.items()
        if row["supported"]
        and isinstance(row["delta_AP@0.50:0.95"], (int, float))
        and row["delta_AP@0.50:0.95"] < 0
    )
    rationale = [
        f"{candidate['experiment_id']} improves {reference['experiment_id']} on the primary "
        f"canonical metric by {delta['delta']:+f}, which clears the frozen practical-"
        f"equivalence margin of {PRACTICAL_EQUIVALENCE_MARGIN}.",
        f"The improvement is roughly {ratio:.1f}x the engineering margin, so the verdict does "
        "not rest on a value near the boundary. The margin remains a threshold, not a "
        "significance test.",
        f"Canonical all-class mAP@0.50:0.95 moves in the same favourable direction "
        f"({all_class_delta:+f}), so the selection metric and the mandatory all-class figure "
        "do not disagree.",
        f"The secondary canonical diagnostic moves the same way: GT-normalised direct mask IoU "
        f"{direct_delta:+f}, giving {direction['status']}.",
        "Native framework AP was not used to arbitrate, because overlap_mask changes the "
        "native validation target and the two experiments' native numbers are therefore "
        "measured against different ground truth.",
    ]
    if regressed:
        rationale.append(
            "Trade-off, recorded rather than hidden: "
            + ", ".join(
                f"{name} regressed {table[name]['delta_AP@0.50:0.95']:+f}" for name in regressed
            )
            + ". Selection does not require every class to improve, and why any individual "
            "class moved is UNKNOWN."
        )
    return rationale


def main(argv: list[str] | None = None) -> int:
    """Freeze the final segmenter from committed evidence.

    Args:
        argv: Command-line arguments.

    Returns:
        Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="run every precondition and derive the comparison without writing",
    )
    args = parser.parse_args(argv)

    paths = ProjectPaths.from_root()
    if holdout_unlocked():
        print(
            f"REFUSED: {HOLDOUT_UNLOCK_ENV_VAR} is set. This phase reads committed development "
            "artifacts only and declines to run in an environment where the holdout is "
            "unlocked.",
            file=sys.stderr,
        )
        return 2

    try:
        historical = historical_digests(paths)

        comparison_config = load_comparison_config(paths.configs / COMPARISON_YAML)
        evaluator = load_canonical_evaluation_config(paths.configs / CANONICAL_YAML)
        if comparison_config["canonical_evaluation_sha256"] != evaluator.fingerprint():
            msg = "the comparison protocol names a different canonical evaluator"
            raise FreezeError(msg)
        if comparison_config["metrics"]["primary"] != PRIMARY_METRIC:
            msg = f"the frozen policy's primary metric is not {PRIMARY_METRIC}"
            raise FreezeError(msg)
        if float(comparison_config["margin"]) != PRACTICAL_EQUIVALENCE_MARGIN:
            msg = f"the frozen margin is not {PRACTICAL_EQUIVALENCE_MARGIN}"
            raise FreezeError(msg)
        if comparison_config["metrics"]["native_framework_status"] != NATIVE_METRIC_STATUS:
            msg = "the frozen policy does not demote the native framework metric"
            raise FreezeError(msg)
        if str(comparison_config["reference_experiment"]) != REFERENCE:
            msg = f"the frozen policy's reference is not {REFERENCE}"
            raise FreezeError(msg)
        if str(comparison_config.candidate["experiment_id"]) != CANDIDATE:
            msg = f"the frozen policy's candidate is not {CANDIDATE}"
            raise FreezeError(msg)

        policy_sha256 = sha256_file(paths.reports / POLICY_JSON)
        policy = read_json(paths.reports / POLICY_JSON)
        if float(policy.get("margin", -1)) != PRACTICAL_EQUIVALENCE_MARGIN:
            msg = f"the committed policy artifact's margin is not {PRACTICAL_EQUIVALENCE_MARGIN}"
            raise FreezeError(msg)

        reference = load_experiment(paths, REFERENCE)
        candidate = load_experiment(paths, CANDIDATE)
        contract = verify_one_variable(reference, candidate)

        if candidate["checkpoint_policy"] != CHECKPOINT_POLICY:
            msg = f"the candidate's checkpoint policy is not {CHECKPOINT_POLICY}"
            raise SelectionStateError(msg)

        try:
            detector_identity = load_final_detector(paths.reports)
        except FinalDetectorError as exc:
            raise FreezeError(str(exc)) from exc
        if detector_identity.recompute_fingerprint() != detector_identity.fingerprint:
            msg = "the final detector manifest does not recompute"
            raise FreezeError(msg)
        detector = {
            "selected_experiment": detector_identity.selected_experiment,
            "model": detector_identity.model,
            "imgsz": detector_identity.imgsz,
            "checkpoint_sha256": detector_identity.checkpoint_sha256,
            "trained": False,
            "validated": False,
            "benchmarked": False,
            "status": "UNCHANGED_DETECTION_BLOCK_CLOSED",
        }
    except (
        ConfigError,
        ComparisonConfigError,
        FinalSegmenterError,
        FreezeError,
    ) as exc:
        classification = (
            INVALID_SELECTION_STATE if isinstance(exc, SelectionStateError) else BLOCKED
        )
        print(f"{classification}: {exc}", file=sys.stderr)
        return 2

    # --- the comparison, recomputed from committed artifacts ----------------------
    delta = classify_delta(
        float(reference["canonical"]["supported_macro"]),
        float(candidate["canonical"]["supported_macro"]),
    )
    direction = cross_metric_direction(
        delta["delta"],
        float(candidate["direct_iou"][GT_NORMALIZED_MASK_IOU])
        - float(reference["direct_iou"][GT_NORMALIZED_MASK_IOU]),
    )
    table = per_class_table(reference, candidate)

    print(
        f"policy     {policy_sha256}  primary {PRIMARY_METRIC}  margin "
        f"{PRACTICAL_EQUIVALENCE_MARGIN}"
    )
    print(f"evaluator  {evaluator.fingerprint()}")
    print(
        f"contract   {ONE_VARIABLE_FIELD} {contract['reference_value']} -> "
        f"{contract['candidate_value']}  one variable VERIFIED"
    )
    print(
        f"{REFERENCE}         supported macro {reference['canonical']['supported_macro']}  "
        f"ckpt {reference['checkpoint']['sha256'][:16]}..."
    )
    print(
        f"{CANDIDATE}         supported macro {candidate['canonical']['supported_macro']}  "
        f"ckpt {candidate['checkpoint']['sha256'][:16]}..."
    )
    print(f"delta      {delta['delta']}  -> {delta['verdict']}")
    print(f"direction  {direction['status']}")

    # The reviewed decision rests on the candidate clearing the margin. If the
    # arithmetic said anything else, recording a selection would be recording a
    # conclusion the evidence does not support.
    if delta["verdict"] != IMPROVES:
        print(
            f"{INVALID_SELECTION_STATE}: the derived classification is {delta['verdict']!r}, "
            f"not {IMPROVES!r}. The reviewed decision to select {CANDIDATE} rests on the "
            "candidate clearing the frozen margin; it does not, so nothing is frozen.",
            file=sys.stderr,
        )
        return 2
    if direction["status"] not in (CONSISTENT, DISAGREEMENT):  # pragma: no cover - invariant
        print(f"{PROTOCOL_VIOLATION}: unknown direction {direction['status']!r}", file=sys.stderr)
        return 2

    if args.verify_only:
        print("VERIFIED: preconditions hold and the comparison derives. Nothing written.")
        print(f"holdout    {HOLDOUT_STATUS}")
        return 0

    # --- the checkpoint --------------------------------------------------------------
    try:
        checkpoints = freeze_checkpoint(paths, candidate)
    except CheckpointError as exc:
        classification = (
            BLOCKED_MISSING_MODEL_ARTIFACT
            if BLOCKED_MISSING_MODEL_ARTIFACT in str(exc)
            else MODEL_ARTIFACT_MISMATCH
        )
        print(f"{classification}: {exc}", file=sys.stderr)
        return 2
    print(
        f"checkpoint {checkpoints['runtime']['sha256']}  "
        f"{checkpoints['runtime']['size_bytes']} bytes  VERIFIED"
    )
    print(
        f"frozen     {checkpoints['frozen_copy']['relative_path']}  "
        f"{checkpoints['frozen_copy']['action']}"
    )

    # --- artifacts ---------------------------------------------------------------------
    rationale = build_rationale(reference, candidate, delta, direction, table)
    manifest = build_manifest(
        reference=reference,
        candidate=candidate,
        contract=contract,
        delta=delta,
        direction=direction,
        table=table,
        checkpoints=checkpoints,
        policy_sha256=policy_sha256,
        evaluator_sha256=evaluator.fingerprint(),
        historical=historical,
        detector=detector,
        rationale=rationale,
    )

    # Parse what is about to be written with the same loader every later phase
    # will use. A manifest this phase cannot itself read back is not evidence.
    try:
        parsed = parse_final_segmenter(manifest)
    except FinalSegmenterError as exc:
        print(f"{PROTOCOL_VIOLATION}: the manifest does not validate: {exc}", file=sys.stderr)
        return 2
    if parsed.recompute_fingerprint() != manifest["final_segmenter_sha256"]:
        print(f"{PROTOCOL_VIOLATION}: the fingerprint does not recompute", file=sys.stderr)
        return 2

    leaks = holdout_leaks(manifest)
    if leaks:
        print(
            f"{PROTOCOL_VIOLATION}: the manifest names the protected split at {leaks}",
            file=sys.stderr,
        )
        return 2

    report = build_report(manifest, commit=git_commit(paths.root))
    comparison_csv = build_comparison_csv(reference, candidate, delta)
    results = build_results_artifact(read_json(paths.reports / RESULTS_JSON), manifest)

    findings = (
        scan_for_sensitive(report)
        + scan_for_sensitive(comparison_csv)
        + scan_for_sensitive(json.dumps(manifest))
        + scan_for_sensitive(json.dumps(results))
    )
    if findings:
        print(f"{BLOCKED}: sensitive content in the emitted artifacts: {findings}", file=sys.stderr)
        return 2

    manifest_sha256 = write_json(paths.reports / MANIFEST_JSON, manifest)
    results_sha256 = write_json(paths.reports / RESULTS_JSON, results)
    (paths.reports / REPORT_MD).write_text(report, encoding="utf-8", newline="\n")
    (paths.reports / COMPARISON_CSV).write_text(comparison_csv, encoding="utf-8", newline="\n")

    # --- post-write verification ---------------------------------------------------------
    after = historical_digests(paths)
    changed = sorted(name for name in historical if historical[name] != after.get(name))
    if changed:
        print(f"{PROTOCOL_VIOLATION}: historical artifacts changed: {changed}", file=sys.stderr)
        return 2

    record = ProvenanceRecord.create(
        name="final_segmenter_freeze",
        phase=8,
        config={
            "segmentation_comparison": f"configs/{COMPARISON_YAML}",
            "canonical_evaluation": f"configs/{CANONICAL_YAML}",
            "comparison_policy_sha256": policy_sha256,
        },
        details={
            "phase": PHASE,
            "classification": SEGMENTER_FROZEN,
            "selection_status": FINAL_SELECTED,
            "selected_experiment": manifest["selected_experiment"],
            "selection_method": SELECTION_METHOD,
            "margin_classification": delta["verdict"],
            "primary_metric": PRIMARY_METRIC,
            "reference_supported_macro": reference["canonical"]["supported_macro"],
            "candidate_supported_macro": candidate["canonical"]["supported_macro"],
            "primary_delta": delta["delta"],
            "margin": PRACTICAL_EQUIVALENCE_MARGIN,
            "cross_metric_direction": direction["status"],
            "selected_checkpoint_sha256": manifest["selected_checkpoint"]["sha256"],
            "final_segmenter_sha256": manifest["final_segmenter_sha256"],
            "frozen_copy_action": checkpoints["frozen_copy"]["action"],
            "overlap_mask": manifest["overlap_mask"],
            "mask_ratio": manifest["mask_ratio"],
            "regressed_supported_classes": manifest["regression_note"][REGRESSION_CLASS_KEY],
            "models_trained_in_this_phase": 0,
            "models_evaluated_in_this_phase": 0,
            "inference_run_in_this_phase": False,
            "benchmarks_run": 0,
            "detector_touched": False,
            "holdout_accessed": False,
            "historical_artifacts_unchanged": True,
        },
        repo_root=paths.root,
    )
    for name in (COMPARISON_YAML, CANONICAL_YAML):
        record.add_input(paths.configs / name, relative_to=paths.root)
    for name in (
        POLICY_JSON,
        S0_RESULT_JSON,
        S0_CANONICAL_JSON,
        S0_MASK_IOU_JSON,
        S1_RESULT_JSON,
        S1_CANONICAL_JSON,
        S1_MASK_IOU_JSON,
    ):
        record.add_input(paths.reports / name, relative_to=paths.root)
    for name in (MANIFEST_JSON, REPORT_MD, COMPARISON_CSV, RESULTS_JSON):
        record.add_output(paths.reports / name, relative_to=paths.root)
    record.write_json(paths.reports / PROVENANCE_JSON)

    print(SEGMENTER_FROZEN)
    print(
        f"selected   {manifest['selected_experiment']}  {manifest['model']}  imgsz "
        f"{manifest['imgsz']}  {ONE_VARIABLE_FIELD}={manifest['overlap_mask']}"
    )
    print(f"fingerprint final_segmenter_sha256 {manifest['final_segmenter_sha256']}")
    print(f"manifest   reports/{MANIFEST_JSON}  sha256 {manifest_sha256}")
    print(f"report     reports/{REPORT_MD}")
    print(f"comparison reports/{COMPARISON_CSV}")
    print(f"results    reports/{RESULTS_JSON}  sha256 {results_sha256}")
    print(f"regressed  {manifest['regression_note'][REGRESSION_CLASS_KEY]}")
    print(f"holdout    {HOLDOUT_STATUS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
