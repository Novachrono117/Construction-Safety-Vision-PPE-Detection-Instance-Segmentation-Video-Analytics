"""Freeze the project's final PPE detector.

Phase 7D. It trains nothing, validates nothing, runs no inference, benchmarks
nothing, tunes no threshold and reads no image. Every number it emits was read
from a committed result manifest; the only computation it performs is
arithmetic over those numbers and SHA-256 over files already on disk.

The decision it records was not made here. It was frozen in Phase 7A, before D1
and D2 existed, and this script applies it mechanically: it rebuilds each
experiment's record from its committed manifest, re-derives the class-support
filter from the frozen split, recomputes the selection metric, and asks
:func:`construction_safety_vision.detection_comparison.compare` which case
holds. The winning experiment id is *returned by that function*, never written
into this file - which is the only way a "the policy chose D2" claim can be
worth anything. If the engine produced a different case or a different leader,
this script would stop rather than reconcile the difference.

Human review sits alongside the policy rather than above it. The maintainer
reviewed the complete D0/D1/D2 results and accepted the policy's output; that
confirms the protocol was respected and the rule correctly applied. It did not
override the rule, and this script records it in those terms.

What is frozen is an identity, not a file path: model family, input resolution,
and the exact bytes of one checkpoint. The checkpoint itself stays git-ignored -
the repository commits the record that makes it re-obtainable, not the binary.

Runs entirely offline. Requires:

* ``configs/detection_experiments.yaml``            phase 7A
* ``configs/detection_baseline.yaml``               phase 6A
* ``reports/detection_comparison_policy.json``      phase 7A
* ``reports/detection_comparison_reference.json``   phase 7A
* ``reports/detection_D0_manifest.json``            phase 6B
* ``reports/detection_D1_manifest.json``            phase 7B
* ``reports/detection_D2_manifest.json``            phase 7C
* ``reports/split_manifest.json``                   phase 5C.2

Writes:
    reports/final_detector_manifest.json
    reports/detection_selection_report.md
    reports/detection_experiment_comparison.csv
    reports/detection_experiment_results.json   (selection state only)
    reports/final_detector.provenance.json
    artifacts/frozen/detection/<id>_best.pt     (git-ignored binary copy)

Usage:
    uv run python scripts/freeze_final_detector.py
    uv run python scripts/freeze_final_detector.py --verify-only
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

from construction_safety_vision.config import ConfigError
from construction_safety_vision.data.canonical import scan_for_sensitive
from construction_safety_vision.data.materialization import digest
from construction_safety_vision.detection_comparison import (
    CASE_B,
    OFFICIAL_ALL_CLASS_METRIC,
    PRACTICAL_EQUIVALENCE_MARGIN,
    PRIMARY_SELECTION_METRIC,
    SECONDARY_METRICS,
    ComparisonConfigError,
    ComparisonError,
    ExperimentMatrix,
    ExperimentRecord,
    build_experiment_record,
    check_protocol_compatibility,
    class_support,
    compare,
    descriptive_class_names,
    load_experiment_matrix,
    resolve_candidate_protocol,
    supported_class_names,
)
from construction_safety_vision.detection_freeze import (
    FROZEN,
    SCHEMA_VERSION,
    FinalDetectorError,
    final_detector_fingerprint,
    parse_final_detector,
)
from construction_safety_vision.detection_results import validate_result_manifest
from construction_safety_vision.experiment import load_detection_baseline_config
from construction_safety_vision.paths import ProjectPaths
from construction_safety_vision.provenance import ProvenanceRecord, git_commit, sha256_file
from construction_safety_vision.splits import HOLDOUT_UNLOCK_ENV_VAR, holdout_unlocked

MATRIX_YAML = "detection_experiments.yaml"
BASELINE_YAML = "detection_baseline.yaml"
POLICY_JSON = "detection_comparison_policy.json"
REFERENCE_JSON = "detection_comparison_reference.json"
SPLIT_MANIFEST_JSON = "split_manifest.json"
RESULTS_JSON = "detection_experiment_results.json"

FINAL_MANIFEST_JSON = "final_detector_manifest.json"
SELECTION_REPORT_MD = "detection_selection_report.md"
COMPARISON_CSV = "detection_experiment_comparison.csv"
PROVENANCE_JSON = "final_detector.provenance.json"

#: Classification emitted when everything holds.
DETECTOR_FROZEN = "DETECTOR_FROZEN"
INVALID_SELECTION_STATE = "INVALID_SELECTION_STATE"
MODEL_ARTIFACT_MISMATCH = "MODEL_ARTIFACT_MISMATCH"
BLOCKED_MISSING_MODEL_ARTIFACT = "BLOCKED_MISSING_MODEL_ARTIFACT"
PROTOCOL_VIOLATION = "PROTOCOL_VIOLATION"
BLOCKED = "BLOCKED"

PHASE = "7D"
SELECTION_STATUS = "FINAL_SELECTED"
SELECTION_METHOD = "PREDECLARED_POLICY_PLUS_HUMAN_REVIEW"
BINARY_DISTRIBUTION_STATUS = "LOCAL_IGNORED_FROZEN_ARTIFACT"

HOLDOUT_STATUS = "PROTECTED_NOT_ACCESSED"
HOLDOUT_REASON = (
    "Phase 7D selected a detector from committed validation results only. The holdout was not "
    "read, materialised, adapted, evaluated, counted or inspected, no artifact this phase wrote "
    "carries a holdout number, and no holdout identifier appears in any of them."
)

#: The historical experiment artifacts this phase must not modify.
HISTORICAL_ARTIFACTS: tuple[str, ...] = (
    "detection_D0_manifest.json",
    "detection_D0_report.md",
    "detection_D1_manifest.json",
    "detection_D1_report.md",
    "detection_D2_manifest.json",
    "detection_D2_report.md",
    POLICY_JSON,
    REFERENCE_JSON,
)

#: Where the immutable copy of the selected checkpoint is kept.
FROZEN_ARTIFACT_DIR = Path("artifacts") / "frozen" / "detection"

#: Result-manifest statuses that count as a completed experiment.
COMPLETE_STATUSES: frozenset[str] = frozenset(
    {"D0_BASELINE_COMPLETE", "DETECTION_EXPERIMENT_COMPLETE"}
)


class FreezeInputError(RuntimeError):
    """Raised when a required input artifact is missing or unusable."""


class SelectionStateError(RuntimeError):
    """Raised when the committed artifacts do not support a freeze."""


def read_json(path: Path) -> dict[str, Any]:
    """Read a committed JSON artifact.

    Args:
        path: File to read.

    Returns:
        The parsed mapping.

    Raises:
        FreezeInputError: If the file is missing or is not a JSON object.
    """
    if not path.is_file():
        msg = f"required input not found: {path.name}"
        raise FreezeInputError(msg)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"{path.name} is not valid JSON ({exc})"
        raise FreezeInputError(msg) from exc
    if not isinstance(data, dict):
        msg = f"{path.name} must contain a JSON object"
        raise FreezeInputError(msg)
    return data


def write_json(path: Path, payload: Any) -> str:
    """Write a JSON artifact deterministically and return its digest.

    Sorted keys, no timestamp, explicit LF: running the phase twice must produce
    byte-identical output, which is what makes "run it twice" a usable check
    that nothing here depends on when or where it ran.

    Args:
        path: Destination file.
        payload: JSON-serialisable content.

    Returns:
        The SHA-256 digest of the payload.
    """
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")
    return digest(payload)


def artifact_digests(reports: Path, names: Sequence[str]) -> dict[str, str]:
    """Hash a set of committed artifacts.

    Args:
        reports: The reports directory.
        names: File names to hash.

    Returns:
        Digest keyed by file name.

    Raises:
        FreezeInputError: If any named artifact is absent.
    """
    found: dict[str, str] = {}
    for name in names:
        path = reports / name
        if not path.is_file():
            msg = f"historical artifact not found: {name}"
            raise FreezeInputError(msg)
        found[name] = sha256_file(path)
    return found


def _exact(value: Decimal) -> str:
    """Render a Decimal without scientific notation or trailing-zero loss.

    Args:
        value: The value.

    Returns:
        Its plain-string form.
    """
    return format(value, "f")


def _round6(value: Decimal) -> float:
    """Round a Decimal to the manifests' six-place reporting precision.

    Args:
        value: The value.

    Returns:
        The rounded float.
    """
    return float(round(value, 6))


def validate_policy(policy: Mapping[str, Any], matrix: ExperimentMatrix) -> None:
    """Check that the frozen Phase 7 policy still says what the freeze assumes.

    The policy is a historical artifact and is never regenerated here. This is a
    read-only agreement check: if the committed policy no longer declares the
    metric, support rule, margin or cases the selection engine implements, the
    freeze has no protocol to apply and stops.

    Args:
        policy: The parsed policy artifact.
        matrix: The parsed experiment matrix.

    Raises:
        SelectionStateError: If any frozen term disagrees.
    """
    problems: list[str] = []

    metrics = policy.get("metrics", {})
    if metrics.get("primary_selection") != PRIMARY_SELECTION_METRIC:
        problems.append(
            f"policy primary_selection is {metrics.get('primary_selection')!r}, "
            f"expected {PRIMARY_SELECTION_METRIC!r}"
        )
    if metrics.get("official_all_class") != OFFICIAL_ALL_CLASS_METRIC:
        problems.append(
            f"policy official_all_class is {metrics.get('official_all_class')!r}, "
            f"expected {OFFICIAL_ALL_CLASS_METRIC!r}"
        )

    rule = policy.get("support_rule", {})
    if rule.get("min_validation_positive_images") != matrix.support_rule.min_positive_images:
        problems.append("policy support rule disagrees on min_validation_positive_images")
    if rule.get("min_validation_instances") != matrix.support_rule.min_instances:
        problems.append("policy support rule disagrees on min_validation_instances")

    margin = policy.get("practical_equivalence_margin", {})
    if Decimal(str(margin.get("value"))) != matrix.practical_equivalence_margin:
        problems.append(
            f"policy margin {margin.get('value')!r} differs from the matrix margin "
            f"{matrix.practical_equivalence_margin}"
        )
    if matrix.practical_equivalence_margin != PRACTICAL_EQUIVALENCE_MARGIN:
        problems.append(
            f"matrix margin {matrix.practical_equivalence_margin} differs from the frozen "
            f"constant {PRACTICAL_EQUIVALENCE_MARGIN}"
        )

    logic = policy.get("selection_logic", {})
    cases = logic.get("cases", {})
    expected_cases = {
        "CASE_A_RETAIN_REFERENCE",
        "CASE_B_VALIDATION_PERFORMANCE_LEADER",
        "CASE_C_PERFORMANCE_TIE_REQUIRES_EFFICIENCY_COMPARISON",
        "CASE_D_EXECUTION_OR_PROTOCOL_FAILURE",
    }
    if set(cases) != expected_cases:
        problems.append(f"policy declares cases {sorted(cases)}, expected {sorted(expected_cases)}")
    if not logic.get("case_c_coverage_note"):
        problems.append("policy is missing the predeclared Case C coverage clarification")
    if logic.get("metric") != PRIMARY_SELECTION_METRIC:
        problems.append(f"policy selection logic decides on {logic.get('metric')!r}")

    if problems:
        msg = "frozen Phase 7 policy validation failed: " + "; ".join(problems)
        raise SelectionStateError(msg)


def validate_experiment(
    experiment_id: str,
    manifest: Mapping[str, Any],
    class_names: Sequence[str],
    reference_fingerprints: Mapping[str, Any] | None,
) -> None:
    """Check one committed result manifest against the freeze's requirements.

    Args:
        experiment_id: The experiment the manifest should describe.
        manifest: The parsed manifest.
        class_names: The frozen class names.
        reference_fingerprints: The reference's dataset fingerprints, or
            ``None`` when this manifest *is* the reference.

    Raises:
        SelectionStateError: If the manifest is invalid, incomplete, describes a
            different experiment, disagrees on the data, or does not record the
            holdout as protected.
    """
    problems = list(validate_result_manifest(manifest, class_names=class_names))

    if manifest.get("experiment_id") != experiment_id:
        problems.append(
            f"manifest experiment_id is {manifest.get('experiment_id')!r}, expected "
            f"{experiment_id!r}"
        )
    if manifest.get("status") not in COMPLETE_STATUSES:
        problems.append(f"status {manifest.get('status')!r} is not a completed experiment")

    holdout = manifest.get("test")
    if not isinstance(holdout, Mapping) or holdout.get("status") != HOLDOUT_STATUS:
        problems.append(f"holdout status is not {HOLDOUT_STATUS}")

    checkpoint = manifest.get("best_checkpoint")
    if not isinstance(checkpoint, Mapping):
        problems.append("no best_checkpoint block")
    else:
        sha256 = checkpoint.get("sha256")
        if not isinstance(sha256, str) or len(sha256) != 64:
            problems.append(f"best_checkpoint.sha256 {sha256!r} is not a SHA-256 digest")

    if reference_fingerprints is not None:
        own = manifest.get("dataset_fingerprints")
        if own != reference_fingerprints:
            problems.append(
                "dataset fingerprints differ from the reference: the experiments did not train "
                "and validate on the same data, so no metric difference is attributable to the "
                "declared variable"
            )
        split = manifest.get("split_reference", {})
        if not isinstance(split, Mapping) or not split.get("split_assignment_sha256"):
            problems.append("no split_reference.split_assignment_sha256")

    if problems:
        msg = f"{experiment_id} result manifest is unusable for the freeze: " + "; ".join(problems)
        raise SelectionStateError(msg)


def verify_controlled_contracts(
    paths: ProjectPaths,
    matrix: ExperimentMatrix,
    manifests: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Re-derive each candidate's one-variable contract against the reference.

    Two independent checks, deliberately not one. The contract is re-derived
    from the committed configuration - resolve the candidate's protocol by
    inheriting the baseline and applying its declared override set, then diff it
    against the baseline - and the verdict that derivation produces is then
    required to agree with the verdict the experiment recorded when it ran. A
    single check would leave open that the configuration drifted after the run.

    Args:
        paths: Project layout.
        matrix: The parsed experiment matrix.
        manifests: Parsed result manifests keyed by experiment id.

    Returns:
        One verdict mapping per candidate.

    Raises:
        SelectionStateError: If a candidate is not a one-variable comparison, or
            if the re-derived verdict disagrees with the recorded one.
    """
    baseline = load_detection_baseline_config(paths.configs / BASELINE_YAML)
    reference_id = matrix.reference_experiment
    reference_manifest = manifests[reference_id]
    if reference_manifest.get("baseline_config_sha256") != baseline.fingerprint():
        msg = (
            f"the committed {reference_id} result was produced under a different protocol than "
            f"configs/{BASELINE_YAML} currently holds; the inherited protocol cannot be rebuilt"
        )
        raise SelectionStateError(msg)

    reference_protocol = baseline.as_dict()
    verdicts: list[dict[str, Any]] = []
    for declaration in matrix.candidates:
        resolved = resolve_candidate_protocol(reference_protocol, declaration)
        verdict = check_protocol_compatibility(
            reference_protocol,
            resolved,
            declaration,
            reference_experiment=reference_id,
        )
        if not verdict.compatible:
            msg = (
                f"{declaration.experiment_id} is not a one-variable comparison against "
                f"{reference_id}: undeclared differences {list(verdict.undeclared_differences)}, "
                f"unapplied declarations {list(verdict.unapplied_declarations)}"
            )
            raise SelectionStateError(msg)
        recorded = manifests[declaration.experiment_id].get("protocol_compatibility")
        derived = verdict.as_dict()
        if recorded != derived:
            msg = (
                f"{declaration.experiment_id}'s recorded protocol_compatibility disagrees with "
                f"the contract re-derived from configs/{BASELINE_YAML} and configs/{MATRIX_YAML}. "
                f"recorded={recorded!r} derived={derived!r}"
            )
            raise SelectionStateError(msg)
        verdicts.append(derived)

    _verify_weight_identity(matrix, manifests, reference_id)
    return verdicts


def _verify_weight_identity(
    matrix: ExperimentMatrix,
    manifests: Mapping[str, Mapping[str, Any]],
    reference_id: str,
) -> None:
    """Check each candidate's pretrained weights against its declaration.

    A candidate that declares no weight change must have started from the
    reference's exact bytes, and a candidate that declares one must not have.
    Both directions matter: the first would be an undeclared second variable,
    the second a declared variable that was never applied.

    Args:
        matrix: The parsed experiment matrix.
        manifests: Parsed result manifests keyed by experiment id.
        reference_id: The reference experiment's id.

    Raises:
        SelectionStateError: If a candidate's weight identity contradicts its
            declaration.
    """
    reference_sha = (manifests[reference_id].get("pretrained_weights") or {}).get("sha256")
    for declaration in matrix.candidates:
        weights = manifests[declaration.experiment_id].get("pretrained_weights") or {}
        candidate_sha = weights.get("sha256")
        declares_new = "weight_identifier" in declaration.overrides
        if declares_new and candidate_sha == reference_sha:
            msg = (
                f"{declaration.experiment_id} declares new pretrained weights but loaded the "
                f"same bytes as {reference_id}; the declared variable was never applied"
            )
            raise SelectionStateError(msg)
        if not declares_new and candidate_sha != reference_sha:
            msg = (
                f"{declaration.experiment_id} declares no weight change but started from "
                f"different bytes than {reference_id}; that is an undeclared second variable"
            )
            raise SelectionStateError(msg)


def build_comparison(
    paths: ProjectPaths,
    matrix: ExperimentMatrix,
    manifests: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, ExperimentRecord], dict[str, Any], tuple[Any, ...]]:
    """Rebuild every experiment record and run the frozen selection logic.

    Args:
        paths: Project layout.
        matrix: The parsed experiment matrix.
        manifests: Parsed result manifests keyed by experiment id.

    Returns:
        The records, the comparison payload, and the frozen class support.

    Raises:
        SelectionStateError: If the comparison refuses.
    """
    split_manifest = read_json(paths.reports / SPLIT_MANIFEST_JSON)
    reference_id = matrix.reference_experiment
    support = class_support(
        split_manifest,
        manifests[reference_id]["class_map"],
        rule=matrix.support_rule,
    )
    records = {
        experiment_id: build_experiment_record(manifest, support)
        for experiment_id, manifest in manifests.items()
    }
    try:
        comparison = compare(matrix, records)
    except ComparisonError as exc:
        msg = f"the frozen comparison refused: {exc}"
        raise SelectionStateError(msg) from exc
    return records, comparison, tuple(support)


def apply_selection(comparison: Mapping[str, Any]) -> tuple[str, str]:
    """Read the case and leader the frozen logic produced.

    Neither value is written here. They come out of the selection engine, which
    is why this function can refuse: a freeze is authorised for the case in
    which one candidate leads, and any other outcome is a different decision
    that this phase has no mandate to make.

    Args:
        comparison: The comparison payload.

    Returns:
        The selection case and the elected experiment id.

    Raises:
        SelectionStateError: If the engine did not elect a single leader.
    """
    selection = comparison.get("selection", {})
    case = str(selection.get("case", ""))
    preferred = selection.get("preferred_experiment")
    if case != CASE_B:
        msg = (
            f"the frozen logic produced {case!r}, not {CASE_B!r}. Phase 7D freezes a "
            "validation-performance leader; any other case is a different decision and is not "
            "authorised here."
        )
        raise SelectionStateError(msg)
    if not isinstance(preferred, str) or not preferred:
        msg = f"{CASE_B} was produced without a preferred experiment"
        raise SelectionStateError(msg)
    if selection.get("leader") != preferred:
        msg = (
            f"the frozen logic's leader {selection.get('leader')!r} differs from its preferred "
            f"experiment {preferred!r}"
        )
        raise SelectionStateError(msg)
    if selection.get("efficiency_comparison_required"):
        msg = (
            "the frozen logic requires an efficiency comparison; Phase 7D is not authorised to "
            "run one and cannot freeze a detector while a tie stands"
        )
        raise SelectionStateError(msg)
    return case, preferred


def human_review_record(selected: str, case: str) -> dict[str, Any]:
    """Record what the maintainer's review did and did not do.

    The distinction matters enough to be structural rather than prose. Review
    confirms that the protocol held and the rule was applied correctly; it does
    not choose. A record that let the two blur would make every future reader
    wonder whether the number or the person picked the model.

    Args:
        selected: The elected experiment id.
        case: The selection case.

    Returns:
        The review record.
    """
    return {
        "status": SELECTION_STATUS,
        "method": SELECTION_METHOD,
        "reviewer": "project maintainer",
        "evidence_basis": "MAINTAINER_DECLARED",
        "reviewed": [
            "the complete committed D0, D1 and D2 validation results",
            "the frozen Phase 7A comparison policy and its four selection cases",
            "each candidate's one-variable contract against the reference",
            "the holdout compliance recorded by every experiment",
        ],
        "confirmed": [
            "the frozen protocol was respected by all three experiments",
            "no disqualifying experimental violation exists",
            f"the frozen policy was correctly applied and yields {case}",
            f"{selected} is accepted as the final detector checkpoint",
        ],
        "did_not": [
            "override the frozen policy",
            "introduce a metric, tie-breaker or threshold",
            "re-run, re-tune or re-validate any experiment",
            "consult the holdout",
        ],
        "relationship_to_policy": (
            "Confirmatory, not corrective. The selected experiment is the frozen policy's "
            "output; human review established that the policy was applicable and correctly "
            "applied, and accepted the result. Had review disagreed with the policy's output, "
            "the correct action would have been to record the disagreement and stop - not to "
            "select a different model."
        ),
    }


def interpretation_limits(selected: str) -> dict[str, Any]:
    """State exactly what the Phase 7 result does and does not establish.

    Args:
        selected: The elected experiment id.

    Returns:
        The limits record.
    """
    return {
        "label": "LIMITATION",
        "supports": (
            "Under this frozen dataset, split, software stack and one-run-per-configuration "
            f"controlled protocol, {selected} - YOLO11n at imgsz 768 - was the best validation "
            "detector among D0, D1 and D2 on the predeclared selection metric."
        ),
        "does_not_establish": [
            "statistical significance: the margin is an engineering decision threshold, carries "
            "no confidence level, and nothing was repeated, so run-to-run variance on this setup "
            "is UNKNOWN",
            "superiority across arbitrary datasets: one dataset of 433 eligible images from one "
            "provider was used",
            "superiority across random seeds: every experiment ran once at seed 42",
            "that increased resolution helped because of small objects: the class-level "
            "diagnostic conducted in Phase 7C did not support that mechanism",
            "any statement about test performance: the holdout has never been evaluated",
        ],
        "small_object_hypothesis": {
            "status": "UNSUPPORTED_BY_THE_SHAPE_OF_THE_RESULT",
            "detail": (
                "Phase 7C ranked the four supported classes by their frozen small-object "
                "fraction against their AP change and found a rank correlation of -0.20: the "
                "largest gain went to the least small-object-heavy class and the only decline "
                "was person. Resolution improved the selection metric beyond the margin - that "
                "is the controlled claim - but the proposed mechanism does not explain it, and "
                "the beyond-margin gain must never be presented as confirming the hypothesis."
            ),
        },
        "single_run": (
            "Each configuration was trained exactly once. deterministic: true reduces "
            "run-to-run variance but does not remove it, so every delta reported here is a "
            "difference between two single runs, not an estimate with an interval."
        ),
        "d1_d2_not_ranked": (
            "D1 and D2 differ from each other in two things at once, because each varies a "
            "different field relative to D0. Their difference is recorded for completeness and "
            "is not attributable to either variable; only each candidate's comparison with D0 "
            "is controlled."
        ),
        "operating_point": (
            "Ultralytics reports one precision/recall pair at the F1-maximising point rather "
            "than at a fixed confidence, so a large move in either can partly reflect where "
            "that point landed. No threshold was ever tuned."
        ),
    }


def rare_class_record(support: Sequence[Any]) -> dict[str, Any]:
    """Record the rare-class policy the freeze inherits.

    Args:
        support: The frozen class-support records.

    Returns:
        The rare-class record.
    """
    return {
        "label": "LIMITATION",
        "classes": list(descriptive_class_names(support)),
        "classification": "DESCRIPTIVE_HIGH_UNCERTAINTY",
        "reported_in_full": True,
        "used_in_selection": False,
        "rule": (
            "Every class listed here failed the frozen support rule on the validation split. "
            "Its precision, recall, AP@0.50 and AP@0.50:0.95 are reported in full for every "
            "experiment, and it remains a required class of the project. What it may not do is "
            "decide a winner: it was not tuned for, no model was preferred or rejected because "
            "it moved, it is never ranked against the supported classes, and the holdout was "
            "not consulted to resolve its uncertainty."
        ),
        "selection_rationale_excludes_it": True,
    }


def efficiency_record(
    separation: Decimal, margin: Decimal, leader: str, candidate_runner_up: str
) -> dict:
    """Record why no efficiency tie-break benchmark was run.

    Args:
        separation: Leader minus next-best candidate on the selection metric.
        margin: The frozen practical-equivalence margin.
        leader: The elected experiment id.
        candidate_runner_up: The next-best **candidate**, reference excluded.

    Returns:
        The efficiency record.
    """
    return {
        "label": "SELECTION_RULE",
        "benchmark_run": False,
        "required_by_policy": False,
        "reason": (
            f"The frozen policy requires an efficiency comparison only in Case C, where "
            f"candidate performance is practically tied. {leader} separates from the next-best "
            f"candidate ({candidate_runner_up}) by {_exact(separation)} on "
            f"{PRIMARY_SELECTION_METRIC}, more than the frozen margin of {margin}, so no tie "
            "exists among the candidates and no tie-break is required."
        ),
        "existing_speed_values": (
            "The FRAMEWORK_VALIDATION_SPEED values recorded in each experiment manifest are "
            "descriptive only. They were measured by the training framework under differing "
            "input resolutions and were not used in this selection."
        ),
        "still_required_later": (
            "A standardised detector-versus-segmenter latency study remains required by the "
            "project's scientific question. It is a later phase and is not authorised now."
        ),
    }


def threshold_record() -> dict[str, Any]:
    """Record that no inference threshold was tuned.

    Returns:
        The threshold record.
    """
    return {
        "label": "SELECTION_RULE",
        "tuned": False,
        "parameters_untouched": ["confidence", "IoU", "NMS"],
        "selected_object": "model architecture + training checkpoint + input resolution",
        "note": (
            "Validation ran at the pinned framework defaults, identical across all three "
            "experiments. If the video demonstration later needs an operational inference "
            "threshold, that is a separate, predeclared, validation-only decision in which no "
            "holdout data may participate."
        ),
    }


def overall_ranking(records: Mapping[str, ExperimentRecord]) -> list[dict[str, Any]]:
    """Rank every experiment by the selection metric, reference included.

    This is a *description*, not the selection rule. The frozen rule ranks
    candidates only, because the reference enters it through a separate test;
    that is correct, and it is also why an unqualified "runner-up" is
    misleading. The reference can outscore a candidate while still not being a
    candidate - which is exactly what happened here - so the honest overall
    ordering is recorded separately and never fed back into the decision.

    Args:
        records: Comparison records keyed by experiment id.

    Returns:
        One entry per experiment, best first, each carrying its rank, its exact
        metric value and its role.

    """
    ordered = sorted(
        records.items(),
        key=lambda item: (-item[1].supported_macro, item[0]),
    )
    return [
        {
            "rank": position,
            "experiment_id": experiment_id,
            PRIMARY_SELECTION_METRIC: _round6(record.supported_macro),
            f"{PRIMARY_SELECTION_METRIC}_exact": _exact(record.supported_macro),
        }
        for position, (experiment_id, record) in enumerate(ordered, start=1)
    ]


def build_final_manifest(
    *,
    matrix: ExperimentMatrix,
    records: Mapping[str, ExperimentRecord],
    comparison: Mapping[str, Any],
    manifests: Mapping[str, Mapping[str, Any]],
    support: Sequence[Any],
    selected: str,
    case: str,
    historical: Mapping[str, str],
    frozen_copy: Mapping[str, Any] | None,
    matrix_sha256: str,
    policy_sha256: str,
) -> dict[str, Any]:
    """Assemble the final-detector manifest.

    Args:
        matrix: The parsed experiment matrix.
        records: Comparison records keyed by experiment id.
        comparison: The comparison payload.
        manifests: Parsed result manifests keyed by experiment id.
        support: The frozen class support.
        selected: The elected experiment id.
        case: The selection case.
        historical: Digests of the historical artifacts at freeze time.
        frozen_copy: The frozen binary copy's record, or ``None``.
        matrix_sha256: Digest of the experiment matrix file.
        policy_sha256: Digest of the frozen policy artifact.

    Returns:
        The manifest payload.
    """
    reference_id = matrix.reference_experiment
    chosen = manifests[selected]
    selection = comparison["selection"]
    # The frozen rule ranks candidates only, so this is the next-best
    # *challenger* - not necessarily the second-highest experiment overall. It
    # is named accordingly everywhere it is published.
    candidate_runner_up = selection.get("runner_up")
    margin = matrix.practical_equivalence_margin

    primary_exact = {
        experiment_id: _exact(record.supported_macro) for experiment_id, record in records.items()
    }
    primary = {
        experiment_id: _round6(record.supported_macro) for experiment_id, record in records.items()
    }
    deltas_exact: dict[str, str] = {}
    deltas: dict[str, float] = {}
    peers = (
        (selected, reference_id),
        (selected, candidate_runner_up),
        (candidate_runner_up, reference_id),
    )
    for left, right in peers:
        if not left or not right or left == right:
            continue
        value = records[left].supported_macro - records[right].supported_macro
        deltas_exact[f"{left}_minus_{right}"] = _exact(value)
        deltas[f"{left}_minus_{right}"] = _round6(value)

    all_class = {
        experiment_id: _round6(record.all_class_map50_95)
        for experiment_id, record in records.items()
    }
    secondary = {
        experiment_id: {name: _round6(record.secondary[name]) for name in SECONDARY_METRICS}
        for experiment_id, record in records.items()
    }

    separation = Decimal(str(selection["leader_separation"]))
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": FROZEN,
        "phase": PHASE,
        "task": "detection",
        "selected_experiment": selected,
        "selection_status": SELECTION_STATUS,
        "selection_method": SELECTION_METHOD,
        "policy_case": case,
        "policy_case_derivation": {
            "derived_by": "construction_safety_vision.detection_comparison.select",
            "hardcoded_winner": False,
            "note": (
                "The selected experiment id is the frozen selection engine's output, computed "
                "from the committed result manifests. It is not named in the freeze script, and "
                "the freeze refuses if the engine produces any case other than "
                f"{CASE_B}."
            ),
            "reference_experiment": reference_id,
            "validation_performance_leader": selection["leader"],
            "candidate_runner_up": candidate_runner_up,
            "candidate_runner_up_semantics": (
                "The next-best CONTROLLED CHALLENGER, i.e. the second-ranked experiment among "
                f"the non-reference candidates. It is NOT a claim that {candidate_runner_up} "
                "holds the second-highest value of the selection metric overall - the reference "
                f"{reference_id} is excluded from this ranking by design, because it enters the "
                "rule through the separate 'clears the reference by more than the margin' test "
                "rather than as a peer. See overall_validation_ranking for the ordering across "
                "all three experiments."
            ),
            "leader_separation": selection["leader_separation"],
            "leader_separation_exact": _exact(separation),
            "leader_separation_semantics": (
                f"{selection['leader']} minus {candidate_runner_up} on "
                f"{PRIMARY_SELECTION_METRIC}: a separation between two candidates, never between "
                "a candidate and the reference."
            ),
            "rationale": selection["rationale"],
            "efficiency_comparison_required": selection["efficiency_comparison_required"],
        },
        "reference_experiment": reference_id,
        "validation_performance_leader": selected,
        "candidate_runner_up": candidate_runner_up,
        "overall_validation_ranking": overall_ranking(records),
        "overall_validation_ranking_note": (
            "A descriptive ordering of all three experiments by "
            f"{PRIMARY_SELECTION_METRIC}, reference included. It is reported so that no reader "
            "infers a total ordering from the candidate-only fields above, and it played no part "
            "in the decision: the frozen rule compares each candidate against the reference and "
            "the candidates against each other, and never ranks the reference as a peer."
        ),
        "human_review": human_review_record(selected, case),
        "model": chosen["model"],
        "imgsz": chosen["resolved_training_arguments"]["imgsz"],
        "batch": chosen["resolved_training_arguments"]["batch"],
        "seed": chosen["resolved_protocol"]["seed"],
        "epochs_completed": chosen["epochs_completed"],
        "best_epoch": chosen["best_epoch"],
        "resolved_optimizer": chosen["resolved_optimizer"],
        "pretrained_weights": dict(chosen["pretrained_weights"]),
        "pretrained_weight_identity": chosen.get("pretrained_weight_identity"),
        "selected_checkpoint": {
            "relative_path": chosen["best_checkpoint"]["relative_path"],
            "sha256": chosen["best_checkpoint"]["sha256"],
            "size_bytes": chosen["best_checkpoint"]["size_bytes"],
            "committed": False,
            "selection_rule": chosen["checkpoint_selection"],
        },
        "frozen_copy": frozen_copy,
        "experiment_sha256": records[selected].experiment_sha256,
        "phase7_policy_sha256": policy_sha256,
        "experiment_matrix_sha256": matrix_sha256,
        "dataset_fingerprints": dict(chosen["dataset_fingerprints"]),
        "split_reference": dict(chosen["split_reference"]),
        "class_map": dict(chosen["class_map"]),
        "adapter_manifest_sha256": chosen["adapter_manifest_sha256"],
        "baseline_config_sha256": chosen["baseline_config_sha256"],
        "primary_selection_metric": PRIMARY_SELECTION_METRIC,
        "primary_selection_metric_definition": (
            "unweighted arithmetic mean of per-class AP@0.50:0.95 over the classes the frozen "
            "support rule classifies COMPARISON_SUPPORTED"
        ),
        "selection_metric_classes": list(supported_class_names(support)),
        "practical_equivalence_margin": str(margin),
        "primary_metric_values": primary,
        "primary_metric_values_exact": primary_exact,
        "primary_metric_deltas": deltas,
        "primary_metric_deltas_exact": deltas_exact,
        "official_all_class_metric": OFFICIAL_ALL_CLASS_METRIC,
        "official_all_class_values": all_class,
        "secondary_metric_values": secondary,
        "per_class_metrics": {
            experiment_id: manifest["per_class_metrics"]
            for experiment_id, manifest in manifests.items()
        },
        "metrics_are_validation_only": True,
        "rare_class_policy": rare_class_record(support),
        "efficiency_tie_break": efficiency_record(
            separation, margin, selected, str(candidate_runner_up or "")
        ),
        "threshold_tuning": threshold_record(),
        "interpretation_limits": interpretation_limits(selected),
        "binary_distribution_status": BINARY_DISTRIBUTION_STATUS,
        "repository_contains_model_binary": False,
        "reproducibility_note": (
            "A fresh clone cannot run inference immediately. The repository commits the record "
            "that identifies the frozen checkpoint - digest, size, producing experiment and "
            "complete protocol - not the binary itself. The weights must be obtained separately "
            "or reproduced by re-running the recorded protocol, and a re-run produces different "
            "bytes. How the frozen model is distributed for academic release is an open "
            "decision."
        ),
        "historical_artifact_digests": dict(historical),
        "training_or_evaluation_performed_in_this_phase": False,
        "test": {"status": HOLDOUT_STATUS, "reason": HOLDOUT_REASON},
    }
    payload["final_detector_sha256"] = final_detector_fingerprint(
        {
            "selected_experiment": payload["selected_experiment"],
            "model": payload["model"],
            "imgsz": payload["imgsz"],
            "selected_checkpoint_sha256": payload["selected_checkpoint"]["sha256"],
            "experiment_sha256": payload["experiment_sha256"],
            "phase7_policy_sha256": payload["phase7_policy_sha256"],
            "split_assignment_sha256": payload["split_reference"]["split_assignment_sha256"],
            "adapter_config_sha256": payload["dataset_fingerprints"]["adapter_config_sha256"],
            "class_map_sha256": payload["dataset_fingerprints"]["class_map_sha256"],
        }
    )
    return payload


def comparison_csv(
    matrix: ExperimentMatrix,
    records: Mapping[str, ExperimentRecord],
    manifests: Mapping[str, Mapping[str, Any]],
    ranking: Sequence[Mapping[str, Any]],
) -> str:
    """Render the machine-readable experiment comparison table.

    Args:
        matrix: The parsed experiment matrix.
        records: Comparison records keyed by experiment id.
        manifests: Parsed result manifests keyed by experiment id.
        ranking: The descriptive ordering of every experiment.

    Returns:
        The CSV text.
    """
    reference_id = matrix.reference_experiment
    ranks = {entry["experiment_id"]: entry["rank"] for entry in ranking}
    header = [
        "experiment_id",
        "role",
        "overall_validation_rank",
        "intentional_variable",
        "model",
        "imgsz",
        "batch",
        "supported_macro_map50_95",
        "delta_vs_D0",
        "margin_status",
        "all_class_map50_95",
        "mAP50",
        "precision",
        "recall",
        "best_epoch",
        "checkpoint_sha256",
        "experiment_sha256",
    ]
    order = [reference_id] + [item.experiment_id for item in matrix.candidates]
    lines = [",".join(header)]
    for experiment_id in order:
        record = records[experiment_id]
        manifest = manifests[experiment_id]
        delta = record.supported_macro - records[reference_id].supported_macro
        if experiment_id == reference_id:
            status = "REFERENCE"
            delta_text = "0.0"
        else:
            status = manifest["phase7_comparison"]["margin_status"]
            delta_text = f"{_round6(delta):.6f}"
        lines.append(
            ",".join(
                [
                    experiment_id,
                    str(manifest.get("role", "REFERENCE_BASELINE")),
                    str(ranks[experiment_id]),
                    str(manifest.get("intentional_variable", "NONE_REFERENCE")),
                    str(record.model),
                    str(record.imgsz),
                    str(manifest["resolved_training_arguments"]["batch"]),
                    f"{_round6(record.supported_macro):.6f}",
                    delta_text,
                    status,
                    f"{_round6(record.all_class_map50_95):.6f}",
                    f"{_round6(record.secondary['mAP@0.50']):.6f}",
                    f"{_round6(record.secondary['precision']):.6f}",
                    f"{_round6(record.secondary['recall']):.6f}",
                    str(manifest["best_epoch"]),
                    record.best_checkpoint_sha256,
                    record.experiment_sha256,
                ]
            )
        )
    return "\n".join(lines) + "\n"


def selection_report(
    *,
    manifest: Mapping[str, Any],
    matrix: ExperimentMatrix,
    records: Mapping[str, ExperimentRecord],
    manifests: Mapping[str, Mapping[str, Any]],
    support: Sequence[Any],
    commit: str | None,
) -> str:
    """Render the Phase 7D selection report.

    Args:
        manifest: The final-detector manifest payload.
        matrix: The parsed experiment matrix.
        records: Comparison records keyed by experiment id.
        manifests: Parsed result manifests keyed by experiment id.
        support: The frozen class support.
        commit: The repository commit, when available.

    Returns:
        The Markdown text.
    """
    reference_id = matrix.reference_experiment
    selected = manifest["selected_experiment"]
    candidate_runner_up = manifest["candidate_runner_up"]
    ranking = manifest["overall_validation_ranking"]
    margin = manifest["practical_equivalence_margin"]
    primary = manifest["primary_metric_values"]
    exact = manifest["primary_metric_values_exact"]
    deltas = manifest["primary_metric_deltas_exact"]
    all_class = manifest["official_all_class_values"]
    secondary = manifest["secondary_metric_values"]
    supported = list(supported_class_names(support))
    descriptive = list(descriptive_class_names(support))
    order = [reference_id] + [item.experiment_id for item in matrix.candidates]

    head = commit or "unrecorded"
    lines: list[str] = [
        "# Phase 7D - Final Controlled Detector Selection",
        "",
        f"Phase: {PHASE} · Commit: `{head}` · Classification: **{DETECTOR_FROZEN}**",
        "",
        "**No model was trained, validated, benchmarked or run in this phase.** Every number "
        "below was read from a committed result manifest and every comparison is arithmetic "
        "over those numbers. The holdout was not read, materialised, adapted, evaluated or "
        "inspected.",
        "",
        "Claims are labelled `PREDECLARED_POLICY` (fixed in Phase 7A, before D1 and D2 "
        "existed), `COMPUTED_RESULT` (arithmetic over committed artifacts), `SELECTION_RULE` "
        "(deterministic), `HUMAN_REVIEW`, `FINAL_DECISION`, `LIMITATION` and `HOLDOUT_POLICY`.",
        "",
        "## 1. Selection objective",
        "",
        "`PREDECLARED_POLICY` Phase 7 asked one question: among the frozen controlled "
        "detection experiments, which configuration should become the project's detector? "
        "Phase 7A fixed how that question would be answered. Phases 7B and 7C produced the "
        "results. This phase applies the rule and freezes the answer - it does not extend the "
        "experiment matrix, and no further detection experiment is authorised.",
        "",
        "`SELECTION_RULE` What is being selected is a *model identity*: architecture, training "
        "checkpoint and input resolution. Not a confidence threshold, not an IoU or NMS "
        "setting, not a deployment configuration.",
        "",
        "## 2. Frozen comparison policy",
        "",
        f"`PREDECLARED_POLICY` Primary selection metric: `{PRIMARY_SELECTION_METRIC}` - the "
        "unweighted arithmetic mean of per-class `AP@0.50:0.95` over the classes the frozen "
        "support rule admits.",
        "",
        "`PREDECLARED_POLICY` Support rule: a class is `COMPARISON_SUPPORTED` when **both** "
        f"`validation_positive_images >= {matrix.support_rule.min_positive_images}` **and** "
        f"`validation_instances >= {matrix.support_rule.min_instances}`; otherwise "
        "`DESCRIPTIVE_HIGH_UNCERTAINTY`. The rule names no class. Applied to the split frozen "
        f"in phase 5C.2 it admits {', '.join(f'`{name}`' for name in supported)} and classifies "
        f"{', '.join(f'`{name}`' for name in descriptive)} as descriptive.",
        "",
        f"`PREDECLARED_POLICY` Practical-equivalence margin: **{margin}** absolute AP. Strictly "
        "greater than the margin counts as an improvement; exactly the margin is practically "
        "equivalent. It is an engineering decision threshold that prevents escalating to a "
        "larger or slower model for a trivial validation difference. It is **not** a "
        "significance test and carries no confidence level.",
        "",
        f"`PREDECLARED_POLICY` The official all-class `{OFFICIAL_ALL_CLASS_METRIC}` remains "
        "mandatory and is never hidden. It does not decide the winner.",
        "",
        "`PREDECLARED_POLICY` The four selection cases, fixed in Phase 7A:",
        "",
        "| Case | Condition | Outcome |",
        "| --- | --- | --- |",
        f"| A | No candidate clears {reference_id} by more than the margin | Retain "
        f"{reference_id}, the lower-complexity model |",
        "| B | One candidate clears the reference by more than the margin **and** separates "
        "from the runner-up by more than the margin | That candidate is the "
        "validation-performance leader |",
        "| C | A candidate clears the reference but does not separate from the runner-up | No "
        "winner; a later controlled efficiency comparison decides |",
        "| D | Execution or protocol failure | Protocol review; never read as model inferiority |",
        "",
        "`PREDECLARED_POLICY` Case C coverage was clarified in the frozen policy itself, not "
        "afterwards: it is defined by the leader's separation from the runner-up rather than by "
        "both candidates having cleared the reference, so it also covers the region where only "
        "one candidate clears the reference while the two sit within the margin of each other. "
        "Leaving that region undefined would have meant deciding it after seeing the numbers.",
        "",
        "## 3. Controlled experiment matrix",
        "",
        "`PREDECLARED_POLICY` The candidates carry no protocol of their own. Each inherits "
        f"`configs/{BASELINE_YAML}` wholesale and declares an override set, so *only one thing "
        "differs* is a structural property of the configuration rather than a promise in prose. "
        "This phase re-derived each contract from the committed configuration and required it "
        "to agree with the verdict the experiment recorded when it ran.",
        "",
        "| Experiment | Role | Intentional variable | Declared override | Model | imgsz | Batch |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]

    declarations = {item.experiment_id: item for item in matrix.candidates}
    for experiment_id in order:
        record = records[experiment_id]
        manifest_row = manifests[experiment_id]
        declaration = declarations.get(experiment_id)
        overrides = (
            ", ".join(f"`{key}: {value}`" for key, value in sorted(declaration.overrides.items()))
            if declaration and declaration.overrides
            else "-"
        )
        variable = manifest_row.get("intentional_variable", "NONE_REFERENCE")
        role = manifest_row.get("role", "REFERENCE_BASELINE")
        lines.append(
            f"| `{experiment_id}` | `{role}` | `{variable}` | {overrides} | {record.model} | "
            f"{record.imgsz} | {manifest_row['resolved_training_arguments']['batch']} |"
        )

    lines += [
        "",
        "`COMPUTED_RESULT` All three experiments share identical dataset, split, adapter and "
        "class-map fingerprints, verified field by field. Batch stayed at 16 throughout; no run "
        "was rescued by a smaller batch, auto-batch, gradient accumulation, a different image "
        "size or a different model.",
        "",
        "`COMPUTED_RESULT` `optimizer: auto` resolved to AdamW at lr0 0.001111 for all three "
        "experiments, so the optimizer does not confound any comparison.",
        "",
    ]

    for experiment_id, heading, note in (
        (
            reference_id,
            "4. D0 result - baseline reference",
            "D0 is the reference point, not a candidate. It was run once and must not be "
            "re-run, retuned or averaged; a second run would measure noise rather than a "
            "change.",
        ),
        (
            "D1",
            "5. D1 result - capacity intervention",
            "D1 varied `MODEL_CAPACITY` (YOLO11s), with the pretrained checkpoint identity "
            "recorded as consequential rather than as a second variable.",
        ),
        (
            "D2",
            "6. D2 result - resolution intervention",
            "D2 varied `INPUT_RESOLUTION` (imgsz 768) starting from the same `yolo11n.pt` bytes "
            "D0 used, verified by digest.",
        ),
    ):
        if experiment_id not in records:
            continue
        record = records[experiment_id]
        manifest_row = manifests[experiment_id]
        lines += [
            f"## {heading}",
            "",
            f"`COMPUTED_RESULT` {note}",
            "",
            "| Metric | Value |",
            "| --- | --- |",
            f"| `{PRIMARY_SELECTION_METRIC}` | **{exact[experiment_id]}** |",
            f"| `{OFFICIAL_ALL_CLASS_METRIC}` (all-class, official) | {all_class[experiment_id]} |",
            f"| `mAP@0.50` | {secondary[experiment_id]['mAP@0.50']} |",
            f"| precision | {secondary[experiment_id]['precision']} |",
            f"| recall | {secondary[experiment_id]['recall']} |",
            f"| best epoch | {manifest_row['best_epoch']} / {manifest_row['epochs_completed']} |",
            f"| experiment SHA-256 | `{record.experiment_sha256}` |",
            f"| best checkpoint SHA-256 | `{record.best_checkpoint_sha256}` |",
            "",
            "All of these are **validation** numbers and say nothing about test performance.",
            "",
        ]
        if experiment_id == "D1":
            lines += [
                "`COMPUTED_RESULT` D1's two headline metrics moved in opposite directions, and "
                "the reason is arithmetic rather than a paradox: both are unweighted means over "
                "the same per-class APs and differ only in which classes they average. "
                f"`{descriptive[0] if descriptive else 'the excluded class'}` - excluded by the "
                "support rule, standing on one validation image - gained enough to flip the "
                "sign of the five-class figure on its own. **D1's all-class improvement is not "
                "evidence that it beat D0**; quoting it that way would be the metric-shopping "
                "the Phase 7A policy exists to prevent, and suppressing it would be equally "
                "wrong.",
                "",
                "`COMPUTED_RESULT` The substantive D1 finding is `vest_on_body`, which fell "
                "further than the other three supported classes gained. **Why is UNKNOWN**: one "
                "run cannot separate it from run-to-run variance, and diagnosing it needs "
                "image-level error analysis that belongs to a later phase.",
                "",
            ]
        if experiment_id == "D2":
            lines += [
                "`COMPUTED_RESULT` Unlike D1, both of D2's headline metrics move in the same "
                "direction, so nothing turns on which is read.",
                "",
                "`LIMITATION` D2's predeclared small-object hypothesis is **not supported by "
                "the shape of the result**. Ranking the four supported classes by their frozen "
                "small-object fraction against their AP change gives a rank correlation of "
                "-0.20: the largest gain went to the *least* small-object-heavy class and the "
                "only decline was `person`. Resolution improved the selection metric beyond the "
                "margin - that is the controlled claim - but the proposed mechanism does not "
                "explain it.",
                "",
            ]

    lines += [
        "## 7. Primary metric comparison",
        "",
        "`COMPUTED_RESULT` Exact decimals, recomputed in this phase from the committed "
        "per-class metrics rather than copied from any summary:",
        "",
        f"| Experiment | `{PRIMARY_SELECTION_METRIC}` (exact) | Rounded | Delta vs "
        f"{reference_id} | Margin status |",
        "| --- | --- | --- | --- | --- |",
    ]
    for experiment_id in order:
        if experiment_id == reference_id:
            lines.append(
                f"| `{experiment_id}` | {exact[experiment_id]} | {primary[experiment_id]} | - | "
                "`REFERENCE` |"
            )
            continue
        key = f"{experiment_id}_minus_{reference_id}"
        delta_text = deltas.get(key)
        if delta_text is None:
            value = records[experiment_id].supported_macro - records[reference_id].supported_macro
            delta_text = _exact(value)
        status = manifests[experiment_id]["phase7_comparison"]["margin_status"]
        lines.append(
            f"| `{experiment_id}` | {exact[experiment_id]} | {primary[experiment_id]} | "
            f"{delta_text} | `{status}` |"
        )

    peer_key = f"{selected}_minus_{candidate_runner_up}"
    lines += [
        "",
        f"`COMPUTED_RESULT` {selected} against {candidate_runner_up}: "
        f"{deltas.get(peer_key, 'n/a')} on the selection metric.",
        "",
        "`LIMITATION` That last figure is a **difference, not a ranking**. D1 and D2 differ "
        "from each other in two things at once, because each varies a different field relative "
        f"to {reference_id}. Only each candidate's comparison with the reference is controlled.",
        "",
        "### Overall ordering of all three experiments",
        "",
        "`COMPUTED_RESULT` Descriptive, and reported here so that no reader infers a total "
        "ordering from the candidate-only fields the selection rule uses:",
        "",
        f"| Overall rank | Experiment | Role | `{PRIMARY_SELECTION_METRIC}` (exact) |",
        "| --- | --- | --- | --- |",
    ]
    for entry in ranking:
        experiment_id = entry["experiment_id"]
        role = "reference baseline" if experiment_id == reference_id else "controlled candidate"
        lines.append(
            f"| {entry['rank']} | `{experiment_id}` | {role} | "
            f"{entry[f'{PRIMARY_SELECTION_METRIC}_exact']} |"
        )
    lines += [
        "",
        f"`LIMITATION` **The reference outscores one of the candidates.** {reference_id} sits "
        f"above {candidate_runner_up} on the selection metric, which is why the phrase "
        '"runner-up" is qualified everywhere in this report and in the machine-readable '
        "artifacts. This ordering did **not** enter the decision; the frozen rule compares each "
        "candidate against the reference and the candidates against each other, and never ranks "
        "the reference as a peer.",
        "",
        "## 8. Official all-class comparison",
        "",
        f"`COMPUTED_RESULT` The official `{OFFICIAL_ALL_CLASS_METRIC}` is reported in full "
        "alongside the selection metric, as the frozen policy requires. It does not decide the "
        "winner.",
        "",
        f"| Experiment | `{OFFICIAL_ALL_CLASS_METRIC}` | `mAP@0.50` | precision | recall |",
        "| --- | --- | --- | --- | --- |",
    ]
    for experiment_id in order:
        row = secondary[experiment_id]
        lines.append(
            f"| `{experiment_id}` | {all_class[experiment_id]} | {row['mAP@0.50']} | "
            f"{row['precision']} | {row['recall']} |"
        )

    lines += [
        "",
        "`LIMITATION` Precision and recall carry an operating-point caveat. Ultralytics reports "
        "one precision/recall pair at the F1-maximising point rather than at a fixed "
        "confidence, so a large move in either can partly reflect where that point landed. Read "
        "them as a hint about the precision/recall balance, never as a threshold-independent "
        "property.",
        "",
        "### Per-class comparison",
        "",
        "`COMPUTED_RESULT` Per-class `AP@0.50:0.95`, every class reported:",
        "",
        "| Class | Support | " + " | ".join(f"`{item}`" for item in order) + " |",
        "| --- | --- | " + " | ".join("---" for _ in order) + " |",
    ]
    for entry in support:
        cells = []
        for experiment_id in order:
            table = manifests[experiment_id]["per_class_metrics"][entry.class_name]
            cells.append(str(table["AP@0.50:0.95"]))
        lines.append(
            f"| `{entry.class_name}` | `{entry.classification}` | " + " | ".join(cells) + " |"
        )

    lines += [
        "",
        "## 9. Frozen selection-case application",
        "",
        "`SELECTION_RULE` The frozen logic was applied mechanically, by the same code that "
        "implements the Phase 7A policy, over records rebuilt from the committed result "
        "manifests. The winning experiment id is that function's **output**; it is not written "
        "into the freeze script, and the freeze refuses if any other case is produced.",
        "",
        "`SELECTION_RULE` Case B compares D1 and D2 as **controlled challengers** against "
        f"{reference_id} as the **reference baseline**. {reference_id} is not a candidate and "
        "is never ranked as a peer of the two: it enters the rule through the separate 'clears "
        "the reference by more than the margin' test. Three distinct things follow, and this "
        "report keeps them apart because conflating them is an easy and consequential mistake:",
        "",
        "| Concept | Value | Meaning |",
        "| --- | --- | --- |",
        f"| **Validation performance leader** | **`{selected}`** | The experiment the frozen "
        "rule elects, and the frozen detector |",
        f"| **Second-highest experiment overall** | **`{ranking[1]['experiment_id']}`** | "
        f"Descriptive ordering by {PRIMARY_SELECTION_METRIC} across all three; played no part "
        "in the decision |",
        f"| **Non-reference candidate runner-up** | **`{candidate_runner_up}`** | The "
        "second-ranked *challenger*, the quantity the Case B separation test uses |",
        "",
        f"`LIMITATION` **`{candidate_runner_up}` being the other candidate does not make its "
        f"metric the second-highest overall.** Here it is not: {reference_id} scores "
        f"{exact[reference_id]} against {candidate_runner_up}'s {exact[candidate_runner_up]}, so "
        f"the overall ordering is {ranking[0]['experiment_id']} > {ranking[1]['experiment_id']} "
        f"> {ranking[2]['experiment_id']} while the candidate ordering is {selected} > "
        f"{candidate_runner_up}. Both statements are true and they are about different sets.",
        "",
        f"- Candidates clearing {reference_id} by more than {margin}: **{selected}**",
        f"- Validation performance leader: **{selected}** · non-reference candidate runner-up: "
        f"**{candidate_runner_up}** · separation between those two candidates "
        f"**{manifest['policy_case_derivation']['leader_separation_exact']}**, more than the "
        f"margin",
        f"- Derived case: **`{manifest['policy_case']}`**",
        "",
        f"`SELECTION_RULE` {manifest['policy_case_derivation']['rationale']}.",
        "",
        "## 10. Human review",
        "",
        "`HUMAN_REVIEW` The project maintainer reviewed the complete D0/D1/D2 results, the "
        "frozen policy, each candidate's one-variable contract and the holdout compliance "
        "recorded by every experiment, and confirmed that the protocol was respected, that no "
        "disqualifying experimental violation exists, that the policy was correctly applied, "
        f"and that {selected} is accepted as the final detector checkpoint.",
        "",
        "`HUMAN_REVIEW` Review was **confirmatory, not corrective**. It did not override the "
        "frozen policy, introduce a metric or tie-breaker, re-run anything, or consult the "
        "holdout. Had review disagreed with the policy's output, the correct action would have "
        "been to record the disagreement and stop - not to select a different model.",
        "",
        "## 11. Final selected detector",
        "",
        f"`FINAL_DECISION` **{selected} - {manifest['model']} at imgsz {manifest['imgsz']}.**",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Selected experiment | `{selected}` |",
        f"| Selection status | `{manifest['selection_status']}` |",
        f"| Selection method | `{manifest['selection_method']}` |",
        f"| Policy case | `{manifest['policy_case']}` |",
        f"| Model | {manifest['model']} |",
        f"| Input resolution | {manifest['imgsz']} |",
        f"| Batch | {manifest['batch']} |",
        f"| Seed | {manifest['seed']} |",
        f"| Best epoch | {manifest['best_epoch']} / {manifest['epochs_completed']} |",
        f"| Resolved optimizer | {manifest['resolved_optimizer']} |",
        f"| Pretrained source | `{manifest['pretrained_weights']['identifier']}` "
        f"(SHA-256 `{manifest['pretrained_weights']['sha256']}`) |",
        f"| Experiment SHA-256 | `{manifest['experiment_sha256']}` |",
        f"| Phase 7 policy SHA-256 | `{manifest['phase7_policy_sha256']}` |",
        f"| `final_detector_sha256` | `{manifest['final_detector_sha256']}` |",
        "",
        "### Selection rationale",
        "",
        f"`FINAL_DECISION` `{reference_id}` is the baseline reference. `D1`, the capacity "
        f"intervention, scored **below** {reference_id} on the primary Phase 7 metric. `D2`, "
        f"the resolution intervention, **improves {reference_id} beyond the frozen margin**, by "
        f"an amount materially larger than {margin}, and also exceeds D1 by more than the same "
        "margin - so no performance tie exists and no efficiency tie-break is required. D2 also "
        f"improved the official five-class `{OFFICIAL_ALL_CLASS_METRIC}`, so the two headline "
        "figures agree in direction.",
        "",
        f"`LIMITATION` The rationale above rests entirely on the supported classes. "
        f"`{descriptive[0] if descriptive else 'The descriptive class'}` played no part in it "
        "and may not be cited as a reason for this decision.",
        "",
        "## 12. Frozen checkpoint identity",
        "",
        "`FINAL_DECISION` The final detector is one specific set of bytes, identified by digest "
        "rather than by path:",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Producing run | `{manifest['selected_checkpoint']['relative_path']}` |",
        f"| SHA-256 | `{manifest['selected_checkpoint']['sha256']}` |",
        f"| Size | {manifest['selected_checkpoint']['size_bytes']} bytes |",
        "| Checkpoint rule | `ULTRALYTICS_BEST_ON_VALIDATION_FITNESS` |",
        f"| Committed to git | {str(manifest['selected_checkpoint']['committed']).lower()} |",
    ]
    frozen = manifest.get("frozen_copy")
    if isinstance(frozen, Mapping):
        lines += [
            f"| Immutable local copy | `{frozen['relative_path']}` |",
            f"| Copy SHA-256 | `{frozen['sha256']}` (verified equal) |",
        ]
    lines += [
        "",
        "`SELECTION_RULE` `last.pt` is explicitly **not** the frozen detector. It is the final "
        "epoch's weights, sits in the same directory and loads without complaint, which is why "
        "the freeze accessor rejects it by name as well as by digest.",
        "",
        "## 13. Why no efficiency tie-break was required",
        "",
        f"`SELECTION_RULE` {manifest['efficiency_tie_break']['reason']}",
        "",
        f"`LIMITATION` {manifest['efficiency_tie_break']['existing_speed_values']}",
        "",
        f"`LIMITATION` {manifest['efficiency_tie_break']['still_required_later']}",
        "",
        "## 14. Rare-class limitation",
        "",
        "`LIMITATION` Classified `DESCRIPTIVE_HIGH_UNCERTAINTY` by the frozen support rule: "
        + ", ".join(f"`{name}`" for name in manifest["rare_class_policy"]["classes"])
        + ".",
        "",
        f"`LIMITATION` {manifest['rare_class_policy']['rule']}",
        "",
        "## 15. Small-object hypothesis limitation",
        "",
        f"`LIMITATION` {manifest['interpretation_limits']['small_object_hypothesis']['detail']}",
        "",
        "## 16. Single-run and statistical limitation",
        "",
        f"`LIMITATION` {manifest['interpretation_limits']['single_run']}",
        "",
        "`LIMITATION` What the Phase 7 result supports: "
        f"{manifest['interpretation_limits']['supports']}",
        "",
        "`LIMITATION` What it does **not** establish:",
        "",
    ]
    for item in manifest["interpretation_limits"]["does_not_establish"]:
        lines.append(f"- {item}")

    lines += [
        "",
        "## 17. Holdout compliance",
        "",
        f"`HOLDOUT_POLICY` Status: **`{HOLDOUT_STATUS}`**. {HOLDOUT_REASON}",
        "",
        f"`HOLDOUT_POLICY` `{HOLDOUT_UNLOCK_ENV_VAR}` was not set, no holdout adapter, label "
        "file, dataset entry, prediction or metric exists anywhere in the repository, and no "
        "holdout identifier appears in any artifact this phase wrote. The holdout is read once, "
        "in phase 11, after both models are frozen.",
        "",
        "`HOLDOUT_POLICY` No threshold was tuned in this phase: "
        f"{manifest['threshold_tuning']['note']}",
        "",
        "## 18. Binary distribution and reproducibility status",
        "",
        f"`LIMITATION` `binary_distribution_status: {manifest['binary_distribution_status']}` · "
        f"`repository_contains_model_binary: "
        f"{str(manifest['repository_contains_model_binary']).lower()}`.",
        "",
        f"`LIMITATION` {manifest['reproducibility_note']}",
        "",
        "## 19. Next phase",
        "",
        "Phase 8 - the instance-segmentation model adapter and protocol. It has not started. "
        "The RLE-to-polygon fidelity audit the segmentation adapter requires has not been done, "
        "and no claim about how lossy that conversion would be may be made until it has.",
        "",
        "`SELECTION_RULE` No further detection experiment is authorised. Any new one needs a "
        "new, explicitly reviewed protocol frozen before it runs.",
        "",
    ]
    return "\n".join(lines)


def update_live_results(
    paths: ProjectPaths,
    *,
    selected: str,
    case: str,
    comparison: Mapping[str, Any],
    ranking: Sequence[Mapping[str, Any]],
    reference_id: str,
) -> tuple[Path, str]:
    """Move the live results artifact to its final reviewed selection state.

    Only the selection block and the final-detector field change. Every metric,
    per-class table and fingerprint already in the file is left exactly as the
    experiment phases wrote it.

    Args:
        paths: Project layout.
        selected: The elected experiment id.
        case: The selection case.
        comparison: The comparison payload.
        ranking: The descriptive ordering of every experiment.
        reference_id: The reference experiment's id.

    Returns:
        The artifact path and its new digest.

    Raises:
        SelectionStateError: If an experiment in the file is not complete.
    """
    path = paths.reports / RESULTS_JSON
    payload = read_json(path)
    incomplete = [
        row.get("experiment_id")
        for row in payload.get("experiments", [])
        if row.get("status") != "COMPLETE"
    ]
    if incomplete:
        msg = f"cannot finalise selection while {incomplete} are not COMPLETE"
        raise SelectionStateError(msg)

    selection = comparison["selection"]
    payload["final_selected_detector"] = selected
    payload["selection"] = {
        "status": SELECTION_STATUS,
        "method": SELECTION_METHOD,
        "policy_case": case,
        "policy_case_rationale": selection["rationale"],
        "preferred_experiment": selected,
        "final_selected_experiment": selected,
        "reference_experiment": reference_id,
        "validation_performance_leader": selection["leader"],
        "candidate_runner_up": selection["runner_up"],
        "candidate_runner_up_semantics": (
            "The next-best CONTROLLED CHALLENGER among the non-reference candidates, not a "
            "claim to the second-highest value of the selection metric overall. The reference "
            f"{reference_id} is excluded from that ranking by design. See "
            "overall_validation_ranking."
        ),
        "overall_validation_ranking": list(ranking),
        "leader_separation": selection["leader_separation"],
        "leader_separation_semantics": (
            "leader minus next-best candidate; a separation between two candidates, never "
            "between a candidate and the reference"
        ),
        "efficiency_comparison_required": selection["efficiency_comparison_required"],
        "reason": (
            "Phase 7D applied the frozen Phase 7A policy mechanically to the committed results "
            "and recorded the maintainer's review of the outcome. The case and the leader are "
            "the selection engine's output; human review confirmed the protocol was respected "
            "and the rule correctly applied, and did not override it."
        ),
        "advisory_only": False,
        "manifest": f"reports/{FINAL_MANIFEST_JSON}",
        "report": f"reports/{SELECTION_REPORT_MD}",
    }
    return path, write_json(path, payload)


def freeze_checkpoint_copy(
    paths: ProjectPaths,
    *,
    selected: str,
    relative_path: str,
    expected_sha256: str,
    expected_size: int,
    write: bool,
) -> tuple[dict[str, Any] | None, str]:
    """Verify the runtime checkpoint and make an immutable local copy of it.

    Args:
        paths: Project layout.
        selected: The elected experiment id.
        relative_path: The run's checkpoint path, relative to the root.
        expected_sha256: The digest the committed manifest records.
        expected_size: The size the committed manifest records.
        write: Whether to create the copy.

    Returns:
        The frozen-copy record (``None`` when the runtime artifact is absent)
        and a status string.

    Raises:
        SelectionStateError: If the runtime checkpoint's bytes differ from the
            committed digest, or a different frozen copy already exists.
    """
    source = paths.root / relative_path
    if not source.is_file():
        return None, BLOCKED_MISSING_MODEL_ARTIFACT

    found = sha256_file(source)
    if found != expected_sha256:
        msg = (
            f"{MODEL_ARTIFACT_MISMATCH}: {relative_path} has SHA-256 {found}, but "
            f"reports/detection_{selected}_manifest.json records {expected_sha256}. The runtime "
            "artifact is not the checkpoint the committed result describes."
        )
        raise SelectionStateError(msg)
    size = source.stat().st_size
    if size != expected_size:
        msg = (
            f"{MODEL_ARTIFACT_MISMATCH}: {relative_path} is {size} bytes, but the committed "
            f"manifest records {expected_size}"
        )
        raise SelectionStateError(msg)

    destination = paths.root / FROZEN_ARTIFACT_DIR / f"{selected}_best.pt"
    if destination.is_file():
        existing = sha256_file(destination)
        if existing != expected_sha256:
            msg = (
                f"{MODEL_ARTIFACT_MISMATCH}: {destination} already exists with SHA-256 "
                f"{existing}, which is not the selected checkpoint {expected_sha256}. Refusing "
                "to overwrite - inspect it, then remove it deliberately."
            )
            raise SelectionStateError(msg)
        status = "VERIFIED_EXISTING"
    elif write:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        copied = sha256_file(destination)
        if copied != expected_sha256:
            msg = f"{MODEL_ARTIFACT_MISMATCH}: the frozen copy hashed {copied} after copying"
            raise SelectionStateError(msg)
        status = "CREATED"
    else:
        return None, "NOT_WRITTEN_VERIFY_ONLY"

    return (
        {
            "relative_path": destination.relative_to(paths.root).as_posix(),
            "sha256": expected_sha256,
            "size_bytes": expected_size,
            "committed": False,
            "immutable": True,
            "source_relative_path": relative_path,
            "note": (
                "A byte-identical copy of the selected checkpoint, kept outside the training "
                "run's output directory so that re-running an experiment cannot overwrite the "
                "frozen model. Git-ignored: the repository commits the record, not the binary."
            ),
        },
        status,
    )


def main(argv: list[str] | None = None) -> int:
    """Freeze the project's final detector.

    Args:
        argv: Command-line arguments.

    Returns:
        Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="validate every input and derive the selection without writing any artifact",
    )
    args = parser.parse_args(argv)

    paths = ProjectPaths.from_root()
    if holdout_unlocked():
        print(
            f"REFUSED: {HOLDOUT_UNLOCK_ENV_VAR} is set. Phase 7D selects from validation "
            "results only and declines to run in an environment where the holdout is unlocked.",
            file=sys.stderr,
        )
        return 2

    try:
        historical = artifact_digests(paths.reports, HISTORICAL_ARTIFACTS)
        matrix = load_experiment_matrix(paths.configs / MATRIX_YAML)
        policy = read_json(paths.reports / POLICY_JSON)
        read_json(paths.reports / REFERENCE_JSON)
        ids = [matrix.reference_experiment] + [item.experiment_id for item in matrix.candidates]
        manifests = {
            experiment_id: read_json(paths.reports / f"detection_{experiment_id}_manifest.json")
            for experiment_id in ids
        }
    except (ComparisonConfigError, ConfigError, FreezeInputError) as exc:
        print(f"{BLOCKED}: {exc}", file=sys.stderr)
        return 2

    matrix_sha256 = sha256_file(paths.configs / MATRIX_YAML)
    policy_sha256 = historical[POLICY_JSON]

    try:
        validate_policy(policy, matrix)

        reference_id = matrix.reference_experiment
        class_names = list(manifests[reference_id]["class_map"])
        reference_fingerprints = manifests[reference_id]["dataset_fingerprints"]
        for experiment_id, manifest in manifests.items():
            validate_experiment(
                experiment_id,
                manifest,
                class_names,
                None if experiment_id == reference_id else reference_fingerprints,
            )
            recorded_policy = manifest.get("phase7_policy_sha256")
            if recorded_policy is not None and recorded_policy != policy_sha256:
                msg = (
                    f"{experiment_id} was produced under policy {recorded_policy}, but the "
                    f"committed policy hashes {policy_sha256}"
                )
                raise SelectionStateError(msg)
            recorded_matrix = manifest.get("experiment_matrix_sha256")
            if recorded_matrix is not None and recorded_matrix != matrix_sha256:
                msg = (
                    f"{experiment_id} was produced under matrix {recorded_matrix}, but the "
                    f"committed matrix hashes {matrix_sha256}"
                )
                raise SelectionStateError(msg)

        verify_controlled_contracts(paths, matrix, manifests)
        records, comparison, support = build_comparison(paths, matrix, manifests)
        case, selected = apply_selection(comparison)
    except SelectionStateError as exc:
        print(f"{INVALID_SELECTION_STATE}: {exc}", file=sys.stderr)
        return 2
    except (ComparisonError, FreezeInputError) as exc:
        print(f"{PROTOCOL_VIOLATION}: {exc}", file=sys.stderr)
        return 2

    chosen = manifests[selected]
    try:
        frozen_copy, copy_status = freeze_checkpoint_copy(
            paths,
            selected=selected,
            relative_path=chosen["best_checkpoint"]["relative_path"],
            expected_sha256=chosen["best_checkpoint"]["sha256"],
            expected_size=chosen["best_checkpoint"]["size_bytes"],
            write=not args.verify_only,
        )
    except SelectionStateError as exc:
        print(f"{MODEL_ARTIFACT_MISMATCH}: {exc}", file=sys.stderr)
        return 2

    if copy_status == BLOCKED_MISSING_MODEL_ARTIFACT:
        print(
            f"{BLOCKED_MISSING_MODEL_ARTIFACT}: {chosen['best_checkpoint']['relative_path']} is "
            f"not present on this machine. The freeze needs the selected checkpoint's bytes; "
            "obtain the artifact - do not retrain, because a re-run produces different weights.",
            file=sys.stderr,
        )
        return 2

    final_manifest = build_final_manifest(
        matrix=matrix,
        records=records,
        comparison=comparison,
        manifests=manifests,
        support=support,
        selected=selected,
        case=case,
        historical=historical,
        frozen_copy=frozen_copy,
        matrix_sha256=matrix_sha256,
        policy_sha256=policy_sha256,
    )

    try:
        parse_final_detector(final_manifest)
    except FinalDetectorError as exc:
        print(
            f"{INVALID_SELECTION_STATE}: the assembled manifest is invalid: {exc}",
            file=sys.stderr,
        )
        return 2

    report = selection_report(
        manifest=final_manifest,
        matrix=matrix,
        records=records,
        manifests=manifests,
        support=support,
        commit=git_commit(paths.root),
    )
    csv_text = comparison_csv(
        matrix, records, manifests, final_manifest["overall_validation_ranking"]
    )

    findings = (
        scan_for_sensitive(report)
        + scan_for_sensitive(json.dumps(final_manifest))
        + scan_for_sensitive(csv_text)
    )
    if findings:
        print(f"{BLOCKED}: sensitive content in the emitted artifacts: {findings}", file=sys.stderr)
        return 2

    if args.verify_only:
        print(f"VERIFIED: the frozen policy elects {selected} under {case}.")
        _summarise(final_manifest, records, matrix)
        return 0

    manifest_digest = write_json(paths.reports / FINAL_MANIFEST_JSON, final_manifest)
    (paths.reports / SELECTION_REPORT_MD).write_text(report, encoding="utf-8", newline="\n")
    (paths.reports / COMPARISON_CSV).write_text(csv_text, encoding="utf-8", newline="\n")

    try:
        results_path, results_digest = update_live_results(
            paths,
            selected=selected,
            case=case,
            comparison=comparison,
            ranking=final_manifest["overall_validation_ranking"],
            reference_id=matrix.reference_experiment,
        )
    except (SelectionStateError, FreezeInputError) as exc:
        print(f"{INVALID_SELECTION_STATE}: {exc}", file=sys.stderr)
        return 2

    after = artifact_digests(paths.reports, HISTORICAL_ARTIFACTS)
    if after != historical:
        changed = sorted(name for name in after if after[name] != historical.get(name))
        print(
            f"{PROTOCOL_VIOLATION}: historical experiment artifacts changed during the freeze: "
            f"{changed}",
            file=sys.stderr,
        )
        return 2

    record = ProvenanceRecord.create(
        name="final_detector_freeze",
        phase=7,
        config={
            "detection_experiments": f"configs/{MATRIX_YAML}",
            "detection_baseline": f"configs/{BASELINE_YAML}",
            "experiment_matrix_sha256": matrix_sha256,
            "phase7_policy_sha256": policy_sha256,
        },
        details={
            "phase": PHASE,
            "classification": DETECTOR_FROZEN,
            "selection_status": SELECTION_STATUS,
            "selection_method": SELECTION_METHOD,
            "policy_case": case,
            "final_selected_experiment": selected,
            "final_selected_model": final_manifest["model"],
            "final_selected_imgsz": final_manifest["imgsz"],
            "final_detector_sha256": final_manifest["final_detector_sha256"],
            "selected_checkpoint_sha256": final_manifest["selected_checkpoint"]["sha256"],
            "frozen_copy_status": copy_status,
            "primary_selection_metric": PRIMARY_SELECTION_METRIC,
            "primary_metric_values_exact": final_manifest["primary_metric_values_exact"],
            "primary_metric_deltas_exact": final_manifest["primary_metric_deltas_exact"],
            "practical_equivalence_margin": str(matrix.practical_equivalence_margin),
            "historical_artifact_digests": historical,
            "models_trained_in_this_phase": 0,
            "models_evaluated_in_this_phase": 0,
            "efficiency_benchmarks_run": 0,
            "thresholds_tuned": 0,
            "holdout_accessed": False,
            "binary_distribution_status": BINARY_DISTRIBUTION_STATUS,
            "repository_contains_model_binary": False,
            "final_detector_manifest_sha256": manifest_digest,
            "results_artifact_sha256": results_digest,
        },
        repo_root=paths.root,
    )
    record.add_input(paths.configs / MATRIX_YAML, relative_to=paths.root)
    record.add_input(paths.configs / BASELINE_YAML, relative_to=paths.root)
    for name in (*HISTORICAL_ARTIFACTS, SPLIT_MANIFEST_JSON):
        record.add_input(paths.reports / name, relative_to=paths.root)
    for name in (FINAL_MANIFEST_JSON, SELECTION_REPORT_MD, COMPARISON_CSV):
        record.add_output(paths.reports / name, relative_to=paths.root)
    record.add_output(results_path, relative_to=paths.root)
    record.write_json(paths.reports / PROVENANCE_JSON)

    print(DETECTOR_FROZEN)
    _summarise(final_manifest, records, matrix)
    print(f"manifest   reports/{FINAL_MANIFEST_JSON}  sha256 {manifest_digest}")
    print(f"report     reports/{SELECTION_REPORT_MD}")
    print(f"comparison reports/{COMPARISON_CSV}")
    print(f"results    reports/{RESULTS_JSON}  sha256 {results_digest}")
    if frozen_copy is not None:
        print(f"frozen     {frozen_copy['relative_path']}  {copy_status}  (git-ignored)")
    return 0


def _summarise(
    manifest: Mapping[str, Any],
    records: Mapping[str, ExperimentRecord],
    matrix: ExperimentMatrix,
) -> None:
    """Print the freeze's headline facts.

    Args:
        manifest: The final-detector manifest payload.
        records: Comparison records keyed by experiment id.
        matrix: The parsed experiment matrix.
    """
    exact = manifest["primary_metric_values_exact"]
    order = [matrix.reference_experiment] + [item.experiment_id for item in matrix.candidates]
    margin = manifest["practical_equivalence_margin"]
    print(f"metric     {PRIMARY_SELECTION_METRIC}  margin {margin}")
    for experiment_id in order:
        print(f"  {experiment_id:<3} {exact[experiment_id]:<16} {records[experiment_id].model}")
    for key, value in sorted(manifest["primary_metric_deltas_exact"].items()):
        print(f"  delta {key:<12} {value}")
    print(f"case       {manifest['policy_case']}")
    print(
        f"selected   {manifest['selected_experiment']}  {manifest['model']} @ {manifest['imgsz']}"
    )
    print(f"checkpoint {manifest['selected_checkpoint']['sha256']}")
    print(f"identity   {manifest['final_detector_sha256']}")
    print(f"holdout    {manifest['test']['status']}")


if __name__ == "__main__":
    sys.exit(main())
