"""Freeze the Phase 7 controlled detection comparison protocol.

Phase 7A. Writes down how D1 and D2 will be judged, before either exists. It
trains nothing, evaluates nothing, downloads nothing and touches no image.

What it does:

* loads and validates the committed D0 result;
* derives each class's validation support from the frozen split manifest and
  classifies it under the frozen support rule;
* computes D0's ``supported_macro_map50_95`` from its committed per-class
  metrics, with the class set produced by the rule rather than named in code;
* resolves D1's and D2's full protocols by inheriting D0's and applying their
  declared override sets, then proves each differs from D0 only where declared;
* emits the comparison policy, the D0 reference values and a provenance record.

What it deliberately does not do: train D1, train D2, fetch ``yolo11s.pt``,
re-run D0's validation, or read the holdout. The holdout has no part in Phase 7
selection and this script never names it.

Runs entirely offline. Requires:

* ``configs/detection_experiments.yaml``
* ``configs/detection_baseline.yaml``               phase 6A
* ``reports/detection_D0_manifest.json``            scripts/train_detection_baseline.py
* ``reports/split_manifest.json``                   scripts/freeze_split.py

Writes:
    reports/detection_comparison_policy.json
    reports/detection_comparison_policy.md
    reports/detection_comparison_reference.json
    reports/detection_experiments.provenance.json

Usage:
    uv run python scripts/freeze_detection_experiments.py
    uv run python scripts/freeze_detection_experiments.py --verify-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from construction_safety_vision.config import ConfigError
from construction_safety_vision.data.canonical import scan_for_sensitive
from construction_safety_vision.data.materialization import digest
from construction_safety_vision.detection_comparison import (
    CANDIDATE_ROLE,
    CASE_A,
    CASE_B,
    CASE_C,
    CASE_D,
    DESCRIPTIVE_HIGH_UNCERTAINTY,
    FORBIDDEN_SPLIT,
    OFFICIAL_ALL_CLASS_METRIC,
    POLICY_SCHEMA_VERSION,
    PRIMARY_SELECTION_METRIC,
    ComparisonConfigError,
    ComparisonError,
    ExperimentMatrix,
    build_experiment_record,
    check_protocol_compatibility,
    class_support,
    flatten_protocol,
    load_experiment_matrix,
    resolve_candidate_protocol,
)
from construction_safety_vision.detection_results import validate_result_manifest
from construction_safety_vision.experiment import load_detection_baseline_config
from construction_safety_vision.paths import ProjectPaths
from construction_safety_vision.provenance import ProvenanceRecord, git_commit
from construction_safety_vision.splits import HOLDOUT_UNLOCK_ENV_VAR, holdout_unlocked

MATRIX_YAML = "detection_experiments.yaml"
D0_MANIFEST_JSON = "detection_D0_manifest.json"
BASELINE_YAML = "detection_baseline.yaml"
SPLIT_MANIFEST_JSON = "split_manifest.json"
POLICY_JSON = "detection_comparison_policy.json"
POLICY_MD = "detection_comparison_policy.md"
REFERENCE_JSON = "detection_comparison_reference.json"
PROVENANCE_JSON = "detection_experiments.provenance.json"

PROTOCOL_FROZEN = "DETECTION_EXPERIMENT_PROTOCOL_FROZEN"
INVALID_REFERENCE = "INVALID_D0_REFERENCE"

HOLDOUT_STATUS = "PROTECTED_NOT_ACCESSED"
HOLDOUT_REASON = (
    "Phase 7 selection happens on the frozen validation split only. The holdout was not "
    "read, materialised, adapted, evaluated or counted by this phase, and no comparison "
    "artifact it produced carries a holdout number."
)

REVALIDATION_CLASSIFICATION = "NON_SELECTION_REVALIDATION"
REVALIDATION_ALTERNATIVE = "PROTOCOL_DEVIATION_WITHOUT_SELECTION_DEGREE_OF_FREEDOM"


class FreezeInputError(RuntimeError):
    """Raised when a required input artifact is missing or unusable."""


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

    Sorted keys, no timestamp, trailing newline and an explicit LF: re-running
    the script must produce byte-identical output, which is what makes "run it
    twice" a usable check that nothing here depends on when it ran - and the
    line ending has to be pinned for that to hold on more than one platform.

    Args:
        path: Destination file.
        payload: JSON-serialisable content.

    Returns:
        The SHA-256 digest of the written content.
    """
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")
    return digest(payload)


def holdout_leaks(payload: Any, *, path: str = "") -> list[str]:
    """Find every place an artifact names the protected split.

    A single shape is permitted, and only that shape: a mapping under the key
    ``test`` carrying exactly a ``status`` of :data:`HOLDOUT_STATUS` and a
    ``reason``. That is how a phase records *that it did not touch the holdout*,
    which is evidence and must stay. Anything else - a holdout count, a holdout
    id, a holdout metric, a bare mention as a value - is a leak.

    Args:
        payload: Any parsed value.
        path: Dotted location, used during recursion.

    Returns:
        One dotted path per leak, empty when the artifact is clean.
    """
    leaks: list[str] = []
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            here = f"{path}.{key}" if path else str(key)
            if isinstance(key, str) and key.strip().lower() == FORBIDDEN_SPLIT:
                if not _is_protection_notice(value):
                    leaks.append(here)
                continue
            leaks.extend(holdout_leaks(value, path=here))
    elif isinstance(payload, (list, tuple)):
        for index, item in enumerate(payload):
            leaks.extend(holdout_leaks(item, path=f"{path}[{index}]"))
    elif isinstance(payload, str) and payload.strip().lower() == FORBIDDEN_SPLIT:
        leaks.append(path)
    return leaks


def _is_protection_notice(value: Any) -> bool:
    """Report whether a ``test`` entry is only a protection notice.

    Args:
        value: The value stored under the ``test`` key.

    Returns:
        ``True`` when it carries exactly a compliant status and a reason.
    """
    return (
        isinstance(value, Mapping)
        and set(value) == {"status", "reason"}
        and value.get("status") == HOLDOUT_STATUS
    )


def revalidation_audit_note() -> dict[str, Any]:
    """Record the Phase 6B result-rebuild event as a non-selection revalidation.

    Phase 6B rebuilt D0's result artifacts to repair their provenance, and doing
    so re-executed validation of the same ``best.pt`` under the same frozen
    configuration. That is worth recording, because a second execution of a
    validation pass is exactly the shape of an event that could hide a selection
    decision - and worth classifying honestly, because this one could not: the
    checkpoint was already chosen by the predeclared rule, no threshold, image
    size or metric was varied, and no alternative was compared.

    ``evidence_basis`` is stated plainly, and it separates two different things.
    That a *separate validation execution happened* is corroborated on disk: the
    git-ignored ``artifacts/detection/D0_val/`` directory exists and six of the
    seven committed D0 figures are byte-identical to its output, so the
    committed metric figures demonstrably come from that run. That *nothing was
    varied between the two executions* is not independently verifiable here -
    Ultralytics writes no ``args.yaml`` for a validation run, so the
    configuration it used cannot be read back off disk. The first is evidence;
    the second remains the maintainer's account, and the note says so rather
    than letting the corroborated half vouch for the uncorroborated half.

    Returns:
        The audit note.
    """
    return {
        "event": "phase 6B D0 result-artifact rebuild",
        "classification": REVALIDATION_CLASSIFICATION,
        "equivalent_classification": REVALIDATION_ALTERNATIVE,
        "what_happened": (
            "The Phase 6B result-rebuild procedure re-executed validation of the same best.pt "
            "under the same frozen configuration while repairing the result provenance. The "
            "metrics were reproduced identically and the committed D0 numbers are those metrics."
        ),
        "selection_degrees_of_freedom_introduced": 0,
        "was_not": [
            "hyperparameter tuning",
            "checkpoint selection",
            "threshold selection",
            "model comparison",
            "a second independent experiment",
        ],
        "why_no_selection_freedom": (
            "The checkpoint was already fixed by the predeclared "
            "ULTRALYTICS_BEST_ON_VALIDATION_FITNESS rule before the rebuild, and the rebuild "
            "varied nothing: same weights, same validation split, same image size, same "
            "confidence and IoU thresholds, same metric set. A re-execution that varies nothing "
            "and compares nothing cannot select anything."
        ),
        "reporting_rule": (
            "D0 is one experiment with one result. The rebuild must never be presented as a "
            "second run, as a replication, or as evidence about run-to-run variance, and no D0 "
            "metric may be averaged, re-rounded or restated because of it."
        ),
        "d0_metrics_modified": False,
        "evidence_basis": "MAINTAINER_DECLARED_PARTIALLY_CORROBORATED_ON_DISK",
        "corroborated_on_disk": [
            "a separate validation run directory artifacts/detection/D0_val/ exists "
            "(git-ignored), so a second validation execution demonstrably happened",
            "six of the seven committed D0 figures under reports/figures/detection/D0/ are "
            "byte-identical to that directory's output, so the committed metric figures come "
            "from that run rather than from the training run's own validation pass",
            "the committed D0 manifest validates against the result schema and its metrics "
            "agree with the report and the provenance record",
            "exactly one D0 experiment fingerprint exists in the repository",
        ],
        "not_independently_verified_here": [
            "that nothing was varied between the two executions - Ultralytics writes no "
            "args.yaml for a validation run, so the configuration that run used cannot be read "
            "back off disk. This is the maintainer's account, and the corroborated facts above "
            "do not establish it.",
        ],
    }


def selection_logic(matrix: ExperimentMatrix) -> dict[str, Any]:
    """Describe the frozen selection cases in machine-readable form.

    Args:
        matrix: The frozen experiment matrix.

    Returns:
        The case definitions and the rules that bound them.
    """
    margin = str(matrix.practical_equivalence_margin)
    return {
        "metric": PRIMARY_SELECTION_METRIC,
        "margin": margin,
        "reference_experiment": matrix.reference_experiment,
        "cases": {
            CASE_A: (
                f"No candidate improves {PRIMARY_SELECTION_METRIC} over "
                f"{matrix.reference_experiment} by more than {margin}. Retain "
                f"{matrix.reference_experiment}: it is the lower-complexity model and the "
                "difference does not establish an ordering."
            ),
            CASE_B: (
                f"One candidate improves {PRIMARY_SELECTION_METRIC} over "
                f"{matrix.reference_experiment} by more than {margin} and separates from every "
                f"other candidate by more than {margin}. That candidate is the "
                "validation-performance leader."
            ),
            CASE_C: (
                f"A candidate improves {PRIMARY_SELECTION_METRIC} over "
                f"{matrix.reference_experiment} by more than {margin} but does not separate "
                f"from the runner-up by more than {margin}. No winner is declared; classified "
                "PERFORMANCE_TIE_REQUIRES_EFFICIENCY_COMPARISON, to be decided by a later "
                "controlled efficiency comparison that is not authorised now."
            ),
            CASE_D: (
                "An execution or protocol failure. The failure is not reinterpreted as model "
                "inferiority; the experiment returns for protocol review."
            ),
        },
        "case_c_coverage_note": (
            "Case C is defined by the leader's separation from the runner-up rather than by "
            "both candidates having cleared the reference. That covers the tie the protocol "
            "was written for and also the region where only one candidate clears the reference "
            "while the two candidates sit within the margin of each other. The reasoning is "
            "identical in both - the candidates cannot be ordered - and leaving that region "
            "undefined would mean deciding it after seeing the numbers. Recorded explicitly "
            "because it is a predeclared extension of the case wording, not a reinterpretation "
            "made later."
        ),
        "boundary": (
            f"Strictly greater than {margin} counts as an improvement; a difference of exactly "
            f"{margin} is PRACTICALLY_EQUIVALENT_ON_THIS_VALIDATION_SET. Comparisons are made "
            "on exact decimals at the manifests' six-place precision, so the boundary is "
            "reproducible rather than dependent on binary floating point."
        ),
        "not_a_significance_test": (
            "The margin is an engineering decision threshold that prevents escalating to a "
            "larger model for a trivial validation difference. It is not a hypothesis test, it "
            "carries no confidence level, and no claim of statistical significance may be "
            "derived from it."
        ),
        "overrides_forbidden": [
            "no per-class metric may override these cases",
            f"no {DESCRIPTIVE_HIGH_UNCERTAINTY} class may override these cases",
            "recall may explain a movement but never selects the winner",
            "no recall-weighted or otherwise composite metric may be introduced",
            "the holdout may not be consulted to break a tie",
        ],
    }


def post_hoc_policy() -> dict[str, Any]:
    """State what may not happen once D1 or D2 results appear.

    Returns:
        The frozen anti-fishing policy.
    """
    return {
        "authorised_experiments": ["D1", "D2"],
        "no_further_experiment_without_review": True,
        "forbidden_automatic_followups": [
            "try YOLO11m or any other capacity",
            "try imgsz 896, 960 or any other resolution",
            "tune the optimizer or the learning rate",
            "change mosaic, close_mosaic or any augmentation setting",
            "oversample or rebalance classes",
            "alter loss weights or apply class weighting",
            "relabel or filter annotations",
            "add a metric, a composite or a tie-breaker after seeing results",
            "re-run D0, D1 or D2 to obtain a better number",
        ],
        "rule": (
            "Any experiment beyond D1 and D2 requires a new, explicitly reviewed protocol "
            "frozen before it runs. There is no D3 in Phase 7A, and none may be added by "
            "reading a Phase 7B or 7C result."
        ),
    }


def build_reference(
    manifest: dict[str, Any],
    split_manifest: dict[str, Any],
    matrix: ExperimentMatrix,
) -> dict[str, Any]:
    """Build the deterministic D0 comparison reference.

    Args:
        manifest: The committed D0 result manifest.
        split_manifest: The frozen split manifest.
        matrix: The frozen experiment matrix.

    Returns:
        The reference payload.

    Raises:
        FreezeInputError: If the D0 manifest does not validate.
    """
    class_map = manifest.get("class_map")
    if not isinstance(class_map, dict) or not class_map:
        msg = "detection_D0_manifest.json carries no class_map"
        raise FreezeInputError(msg)
    class_names = tuple(sorted(class_map, key=lambda name: (class_map[name], name)))

    problems = validate_result_manifest(manifest, class_names=class_names)
    if problems:
        msg = "detection_D0_manifest.json is not a valid result manifest: " + "; ".join(problems)
        raise FreezeInputError(msg)

    support = class_support(split_manifest, class_map, rule=matrix.support_rule)
    record = build_experiment_record(manifest, support)
    payload = record.as_dict()
    payload.update(
        {
            "schema_version": POLICY_SCHEMA_VERSION,
            "role": "REFERENCE_BASELINE",
            "status": "EXECUTED",
            "phase_of_result": manifest.get("phase"),
            "result_manifest": f"reports/{D0_MANIFEST_JSON}",
            "experiment_matrix_sha256": matrix.fingerprint(),
            "primary_selection_metric": PRIMARY_SELECTION_METRIC,
            "official_all_class_metric": OFFICIAL_ALL_CLASS_METRIC,
            "support_rule": matrix.support_rule.as_dict(),
            "metrics_are_validation_only": True,
            "test": {"status": HOLDOUT_STATUS, "reason": HOLDOUT_REASON},
        }
    )
    metrics = payload["metrics"]
    payload["supported_macro_minus_all_class"] = round(
        metrics[PRIMARY_SELECTION_METRIC] - metrics[OFFICIAL_ALL_CLASS_METRIC], 6
    )
    payload["reference_freeze_rule"] = (
        "D0 was run once and is frozen. These values are the Phase 7 reference and must not be "
        "recomputed, re-rounded, re-validated for a better figure, or replaced by an average "
        "over repeated runs."
    )
    return payload


def build_policy(
    matrix: ExperimentMatrix,
    reference: dict[str, Any],
    compatibility: list[dict[str, Any]],
    protocols: dict[str, dict[str, Any]],
    *,
    matrix_sha256: str,
) -> dict[str, Any]:
    """Assemble the machine-readable comparison policy.

    Args:
        matrix: The frozen experiment matrix.
        reference: The D0 reference payload.
        compatibility: One protocol-compatibility verdict per candidate.
        protocols: Resolved flattened protocols keyed by experiment id.
        matrix_sha256: Digest of the matrix configuration content.

    Returns:
        The policy payload.
    """
    return {
        "schema_version": POLICY_SCHEMA_VERSION,
        "phase": matrix.phase,
        "task": matrix.task,
        "classification": PROTOCOL_FROZEN,
        "experiment_matrix": "configs/detection_experiments.yaml",
        "experiment_matrix_sha256": matrix_sha256,
        "reference_experiment": matrix.reference_experiment,
        "labels": {
            "PREDECLARED_PHASE7_POLICY": (
                "a rule fixed in phase 7A, before any D1 or D2 result existed"
            ),
            "FROZEN_DATASET_FACT": "a measurement of the frozen split, made before modelling",
            "D0_REFERENCE_RESULT": "a validation number produced by the single D0 run",
            "SELECTION_RULE": "a deterministic rule that decides an outcome",
            "LIMITATION": "a stated weakness of the evidence, not a caveat added afterwards",
        },
        "support_rule": {
            **matrix.support_rule.as_dict(),
            "label": "PREDECLARED_PHASE7_POLICY",
            "source": "reports/split_manifest.json",
            "applied_to": "frozen validation support, phase 5C.2",
            "rationale": (
                "A per-class AP inherits the uncertainty of the evidence behind it. Both floors "
                "are required because they fail independently: a class can hold many instances "
                "inside one or two images, leaving the instance count healthy while the estimate "
                "still rests on a single scene. Averaging such a class into the deciding metric "
                "would hand a fixed share of every decision to a handful of images."
            ),
            "derivation": "mechanical - the rule is applied to the counts, not to class names",
        },
        "class_support": reference["class_support"],
        "supported_classes": reference["selection_metric_classes"],
        "descriptive_high_uncertainty_classes": reference["descriptive_classes"],
        "metrics": {
            "primary_selection": PRIMARY_SELECTION_METRIC,
            "primary_selection_definition": (
                "unweighted arithmetic mean of per-class AP@0.50:0.95 over the classes "
                "classified COMPARISON_SUPPORTED by the frozen support rule"
            ),
            "official_all_class": OFFICIAL_ALL_CLASS_METRIC,
            "official_all_class_role": "OFFICIAL_ALL_CLASS_REPORTING_METRIC",
            "secondary": list(matrix.secondary_metrics),
            "per_class": list(matrix.per_class_metrics),
            "diagnostics": list(matrix.diagnostics),
            "label": "PREDECLARED_PHASE7_POLICY",
            "all_class_is_never_hidden": True,
            "phase_6b_protocol_unchanged": (
                "This is a phase 7 comparison policy, declared after D0 and before any "
                "comparative experiment. It does not rewrite the phase 6B protocol: D0's "
                "primary metric remains mAP@0.50:0.95 and its reported numbers are unchanged. "
                "What is new is which classes may order two models. The support structure it "
                "filters on was frozen in phase 5C.2, before any modelling, and vest_loose was "
                "already predeclared HIGH_SAMPLING_UNCERTAINTY in configs/detection_baseline.yaml "
                "before D0 trained - so the filter is not a reaction to D0's per-class results."
            ),
        },
        "rare_class_policy": rare_class_policy(reference),
        "practical_equivalence_margin": {
            "value": str(matrix.practical_equivalence_margin),
            "applies_to": [PRIMARY_SELECTION_METRIC, OFFICIAL_ALL_CLASS_METRIC],
            "kind": "absolute AP/mAP margin",
            "label": "SELECTION_RULE",
            "not_a_significance_test": True,
        },
        "selection_logic": {**selection_logic(matrix), "label": "SELECTION_RULE"},
        "one_variable_discipline": {
            "label": "PREDECLARED_PHASE7_POLICY",
            "mechanism": (
                "Candidates declare no protocol of their own. They inherit D0's protocol and "
                "declare an override set, and the parser rejects a matrix whose override set "
                "differs from its declared variable fields. So 'only one thing differs' is a "
                "structural property of the configuration, not a promise in prose."
            ),
            "frozen_fields": (
                "split, labels, class map, epochs, batch, optimizer policy, patience, seed, "
                "deterministic flag, augmentation policy, checkpoint-selection rule and "
                "validation protocol are identical across D0, D1 and D2"
            ),
            "resolved_optimizer_note": (
                "optimizer=auto is the declared policy, so the framework-resolved optimizer and "
                "learning rate may legitimately differ between experiments. Each run records "
                "what it actually used; none is forced to match another after execution, and a "
                "resolved value is never quoted as though the configuration had stated it."
            ),
            "verdicts": compatibility,
        },
        "memory_policy": {**matrix.memory_policy, "label": "PREDECLARED_PHASE7_POLICY"},
        "post_hoc_policy": {**post_hoc_policy(), "label": "PREDECLARED_PHASE7_POLICY"},
        "revalidation_audit": revalidation_audit_note(),
        "experiments": [
            {
                **declaration.as_dict(),
                "resolved_protocol": protocols.get(declaration.experiment_id, {}),
            }
            for declaration in matrix.experiments
        ],
        "reference_result": {
            "label": "D0_REFERENCE_RESULT",
            "artifact": "reports/detection_comparison_reference.json",
            "metrics": reference["metrics"],
            "experiment_sha256": reference["experiment_sha256"],
            "best_checkpoint_sha256": reference["best_checkpoint_sha256"],
            "validation_only": True,
        },
        "holdout_policy": {
            "label": "PREDECLARED_PHASE7_POLICY",
            "test": {"status": HOLDOUT_STATUS, "reason": HOLDOUT_REASON},
            "guard": {
                "env_var": HOLDOUT_UNLOCK_ENV_VAR,
                "requires_code_opt_in": True,
                "requires_environment_opt_in": True,
                "state_during_this_phase": "LOCKED",
            },
            "rule": (
                "Every phase 7 decision - model selection, capacity, resolution, thresholds, "
                "tie-breaking and error-driven iteration - uses the frozen validation split "
                "only. The holdout is read once, in phase 11, after both models are frozen."
            ),
        },
        "limitations": limitations(reference),
    }


def rare_class_policy(reference: dict[str, Any]) -> dict[str, Any]:
    """State the rare class's Phase 7 policy from its derived support.

    The class is identified by the support rule's output rather than by name, so
    the policy is a consequence of the evidence rather than a decision about
    ``vest_loose`` specifically.

    Args:
        reference: The D0 reference payload.

    Returns:
        The rare-class policy.
    """
    descriptive = reference["descriptive_classes"]
    support_by_name = {row["class_name"]: row for row in reference["class_support"]}
    return {
        "label": "PREDECLARED_PHASE7_POLICY",
        "classes": list(descriptive),
        "identified_by": "the frozen support rule applied to the frozen validation support",
        "support": [support_by_name[name] for name in descriptive],
        "still_required": True,
        "reported_for_every_experiment": list(reference["per_class_metrics"].keys()),
        "must_report": ["precision", "recall", "AP@0.50", "AP@0.50:0.95"],
        "reporting_rule": (
            "A DESCRIPTIVE_HIGH_UNCERTAINTY class remains a required project class and its "
            "full per-class metrics are reported for every experiment, wherever the framework "
            "exposes them reliably. Excluding it from the deciding metric is a statement about "
            "how much validation evidence stands behind its AP, not about whether the class "
            "matters. The academic report must show the class and explain this limitation."
        ),
        "forbidden": [
            "tuning any hyperparameter for this class",
            "rejecting a model because this class's metric decreased",
            "selecting a model because this class's metric increased",
            "ranking this class against the supported classes",
            "consulting the holdout to resolve the uncertainty",
        ],
    }


def limitations(reference: dict[str, Any]) -> list[dict[str, str]]:
    """List the stated limitations of this comparison protocol.

    Args:
        reference: The D0 reference payload.

    Returns:
        One record per limitation.
    """
    descriptive = ", ".join(reference["descriptive_classes"]) or "none"
    return [
        {
            "label": "LIMITATION",
            "id": "SMALL_VALIDATION_SPLIT",
            "statement": (
                "Selection rests on 65 validation images and 304 annotations. Every metric here "
                "is an estimate from that sample, and the support rule reduces but does not "
                "remove the problem - a supported class clears a floor on interpretability, not "
                "a bar for precise estimation."
            ),
        },
        {
            "label": "LIMITATION",
            "id": "RUN_TO_RUN_VARIANCE_NOT_MEASURED",
            "statement": (
                "deterministic: true reduces variance without eliminating it, and no experiment "
                "is repeated, so the size of the run-to-run variance on this setup is UNKNOWN. "
                "The 0.005 margin is a judgement about what difference is worth acting on, not "
                "a measurement of that variance."
            ),
        },
        {
            "label": "LIMITATION",
            "id": "DESCRIPTIVE_CLASS_EXCLUDED_FROM_SELECTION",
            "statement": (
                f"The deciding metric ignores {descriptive}. A model that improved that class "
                "while leaving the others unchanged would be recorded as practically equivalent "
                "and would not be selected. That is the deliberate cost of refusing to let one "
                "validation image decide, and it is a real cost rather than a free choice."
            ),
        },
        {
            "label": "LIMITATION",
            "id": "TWO_EXPERIMENTS_NOT_A_SEARCH",
            "statement": (
                "Two controlled experiments test two hypotheses. They do not search the "
                "hyperparameter space, and a Case A outcome means these two variations did not "
                "help by more than the margin - not that the baseline is optimal."
            ),
        },
        {
            "label": "LIMITATION",
            "id": "NO_INDEPENDENCE_CLAIM_BEYOND_DUPLICATE_SCREENING",
            "statement": (
                "The split is group-aware and class-aware but nothing establishes that two "
                "images in different splits do not share a site, a day, a camera or a worker. "
                "Any comparison made on it inherits that limitation."
            ),
        },
    ]


def policy_markdown(
    matrix: ExperimentMatrix,
    policy: dict[str, Any],
    reference: dict[str, Any],
    *,
    commit: str | None,
) -> str:
    """Render the human-readable comparison policy.

    Args:
        matrix: The frozen experiment matrix.
        policy: The machine-readable policy.
        reference: The D0 reference payload.
        commit: Repository commit, when available.

    Returns:
        The report body.
    """
    metrics = reference["metrics"]
    lines: list[str] = []
    add = lines.append

    add("# Phase 7 - Controlled Detection Comparison Policy")
    add("")
    add(
        f"Phase: {matrix.phase} · Commit: `{commit or 'unknown'}` · Classification: "
        f"**{PROTOCOL_FROZEN}**"
    )
    add("")
    add(
        "**D0 exists. D1 and D2 do not.** That ordering is the only thing that makes this "
        "document a protocol rather than a description, so it is stated first. Every threshold, "
        "metric and decision rule below was fixed before any comparative result existed."
    )
    add("")
    add(
        "Claims are labelled `PREDECLARED_PHASE7_POLICY` (fixed in this phase), "
        "`FROZEN_DATASET_FACT` (measured before modelling), `D0_REFERENCE_RESULT` (produced by "
        "the single D0 run), `SELECTION_RULE` (deterministic) and `LIMITATION`."
    )
    add("")

    add("## 1. Why a support filter is necessary")
    add("")
    add(
        "`FROZEN_DATASET_FACT` Under the split frozen in phase 5C.2, the five classes do not "
        "carry comparable validation evidence:"
    )
    add("")
    add("| Class | Validation images | Validation instances | Classification |")
    add("| --- | --- | --- | --- |")
    for row in reference["class_support"]:
        add(
            f"| `{row['class_name']}` | {row['validation_positive_images']} | "
            f"{row['validation_instances']} | `{row['classification']}` |"
        )
    add("")
    add(
        "`D0_REFERENCE_RESULT` The all-class `mAP@0.50:0.95` the framework reports is the "
        "unweighted mean over all five classes, which is checkable against D0's own numbers: "
        "the mean of its five committed per-class `AP@0.50:0.95` values is 0.464428, against a "
        "reported 0.464429, the residual being the rounding of the per-class values. So a class "
        "standing on one validation image carries a full fifth of the deciding number, and a "
        "model can be ranked above another because of what happened in a single scene."
    )
    add("")

    add(
        "`PREDECLARED_PHASE7_POLICY` The support rule removes that specific failure mode from "
        "the Phase 7 ranking. It does not make the remaining per-class estimates precise, and "
        "it does not change what the official all-class figure means."
    )
    add("")

    add("## 2. The support rule")
    add("")
    rule = policy["support_rule"]
    add(
        f"`PREDECLARED_PHASE7_POLICY` A class is `COMPARISON_SUPPORTED` when **both** "
        f"`validation_positive_source_images >= {rule['min_validation_positive_images']}` **and** "
        f"`validation_instances >= {rule['min_validation_instances']}`. Otherwise it is "
        "`DESCRIPTIVE_HIGH_UNCERTAINTY`."
    )
    add("")
    add(
        "Two floors, because they fail independently: a class can hold many instances inside one "
        "or two images, which leaves the instance count looking healthy while the estimate still "
        "rests on a single scene, camera and lighting condition."
    )
    add("")
    add(
        "The rule is written as a general threshold and applied mechanically to the frozen "
        "counts. It names no class. Excluding `vest_loose` is the rule's **output**, derived "
        "from the table above, and the code that computes the selection metric never mentions "
        "the class."
    )
    add("")
    add(f"- `COMPARISON_SUPPORTED`: {', '.join(f'`{n}`' for n in policy['supported_classes'])}")
    add(
        "- `DESCRIPTIVE_HIGH_UNCERTAINTY`: "
        f"{', '.join(f'`{n}`' for n in policy['descriptive_high_uncertainty_classes'])}"
    )
    add("")

    add("## 3. Metrics")
    add("")
    add("| Role | Metric |")
    add("| --- | --- |")
    add(f"| `PRIMARY_PHASE7_SELECTION_METRIC` | `{PRIMARY_SELECTION_METRIC}` |")
    add(f"| `OFFICIAL_ALL_CLASS_REPORTING_METRIC` | `{OFFICIAL_ALL_CLASS_METRIC}` |")
    add(f"| Secondary | {', '.join(f'`{m}`' for m in matrix.secondary_metrics)} |")
    add(f"| Per class | {', '.join(f'`{m}`' for m in matrix.per_class_metrics)} |")
    add(f"| Diagnostic | {', '.join(f'`{m}`' for m in matrix.diagnostics)} |")
    add("")
    add(
        f"`PREDECLARED_PHASE7_POLICY` `{PRIMARY_SELECTION_METRIC}` is the unweighted arithmetic "
        "mean of per-class `AP@0.50:0.95` over the `COMPARISON_SUPPORTED` classes. Unweighted "
        "deliberately: a support-weighted mean would let `person`, the most frequent class, "
        "absorb the decision, which is the opposite of what a PPE-compliance evaluation cares "
        "about."
    )
    add("")
    add(
        "`PREDECLARED_PHASE7_POLICY` The five-class `mAP@0.50:0.95` stays **mandatory** and is "
        "reported for every experiment together with `mAP@0.50`, precision and recall. It is not "
        "the selection metric and it is never hidden. This does not rewrite the phase 6B "
        "protocol: D0's primary metric is still `mAP@0.50:0.95` and its published numbers are "
        "unchanged."
    )
    add("")
    add(
        "`PREDECLARED_PHASE7_POLICY` The honest sequencing argument for adding a comparison "
        "metric after D0: the support structure it filters on was frozen in phase 5C.2 before "
        "any modelling, and `vest_loose` was recorded `HIGH_SAMPLING_UNCERTAINTY` in "
        "`configs/detection_baseline.yaml` before D0 trained. The filter is therefore not a "
        "reaction to D0's per-class results. It is still a metric declared after one result "
        "exists, which is why it decides only the Phase 7 comparison and never replaces the "
        "official figure."
    )
    add("")

    add("## 4. D0 reference values")
    add("")
    add("`D0_REFERENCE_RESULT` **Validation only. These numbers say nothing about test.**")
    add("")
    add("| Metric | Value |")
    add("| --- | --- |")
    primary = metrics[PRIMARY_SELECTION_METRIC]
    official = metrics[OFFICIAL_ALL_CLASS_METRIC]
    add(f"| `{PRIMARY_SELECTION_METRIC}` (primary selection) | {primary} |")
    add(f"| `{OFFICIAL_ALL_CLASS_METRIC}` (official, all class) | {official} |")
    for name in matrix.secondary_metrics:
        add(f"| `{name}` | {metrics[name]} |")
    add("")
    add("| Class | `AP@0.50` | `AP@0.50:0.95` | precision | recall | In selection metric |")
    add("| --- | --- | --- | --- | --- | --- |")
    supported = set(policy["supported_classes"])
    for name, table in reference["per_class_metrics"].items():
        add(
            f"| `{name}` | {table['AP@0.50']} | {table['AP@0.50:0.95']} | {table['precision']} | "
            f"{table['recall']} | {'yes' if name in supported else 'no'} |"
        )
    add("")
    add(
        f"`D0_REFERENCE_RESULT` The supported macro exceeds the official all-class figure by "
        f"{reference['supported_macro_minus_all_class']:.6f}. The gap is not a better "
        "result; it is arithmetic. Both are unweighted means over per-class AP, and the "
        "difference is exactly the effect of dropping one very low AP from the average. "
        "**The two numbers are not interchangeable and must never be compared with each other "
        "as though one improved on the other.**"
    )
    add("")
    add("| Identity | Value |")
    add("| --- | --- |")
    add(f"| Model | {reference['model']} |")
    add(f"| `imgsz` | {reference['imgsz']} |")
    parameters = reference["parameters"] or "not recorded"
    add(f"| Parameters | {parameters} |")
    add(f"| `experiment_sha256` | `{reference['experiment_sha256']}` |")
    add(f"| `best_checkpoint` SHA-256 | `{reference['best_checkpoint_sha256']}` |")
    add("")
    add(
        "`PREDECLARED_PHASE7_POLICY` D0 was run once and is frozen. It is not retuned, not "
        "re-run to improve it, and not averaged over repeated runs - a second run would measure "
        "noise, not a change."
    )
    add("")

    add("## 5. The two authorised experiments")
    add("")
    for declaration in matrix.experiments:
        if declaration.role != CANDIDATE_ROLE:
            continue
        add(f"### {declaration.experiment_id} - {declaration.intentional_variable}")
        add("")
        add(f"`PREDECLARED_PHASE7_POLICY` **Question.** {declaration.question.strip()}")
        add("")
        add(f"**Hypothesis.** {declaration.hypothesis.strip()}")
        add("")
        add("| Field | Value |")
        add("| --- | --- |")
        add(f"| Inherits | {declaration.inherits} |")
        add(f"| Intentional variable | `{declaration.intentional_variable}` |")
        for field_path, value in sorted(declaration.overrides.items()):
            kind = (
                "intentional" if field_path in declaration.intentional_fields else "consequential"
            )
            add(f"| Override `{field_path}` ({kind}) | `{value}` |")
        add(f"| Status | `{declaration.status}` |")
        add("")
        if declaration.pretrained_weights:
            weights = declaration.pretrained_weights
            add(
                f"`PREDECLARED_PHASE7_POLICY` Starting weights `{weights['identifier']}` from "
                f"`{weights['source_mechanism']}`, currently "
                f"`{weights['fingerprint_status']}`. Before training, the run must record "
                f"{', '.join(f'`{item}`' for item in weights['required_before_training'])}. "
                "Two files with the same name are not necessarily the same bytes, and the binary "
                "is never committed."
            )
            add("")

    add("## 6. One-variable discipline")
    add("")
    add(
        "`PREDECLARED_PHASE7_POLICY` The candidates carry no protocol of their own. They inherit "
        "D0's protocol from `configs/detection_baseline.yaml` and declare an override set, and "
        "the parser rejects a matrix whose override set differs from its declared variable "
        "fields. So *only one thing differs* is a structural property of the configuration "
        "rather than a promise in prose."
    )
    add("")
    add("| Comparison | Intentional difference | Verified |")
    add("| --- | --- | --- |")
    for verdict in policy["one_variable_discipline"]["verdicts"]:
        fields = ", ".join(f"`{name}`" for name in verdict["observed_differences"]) or "none"
        add(
            f"| {verdict['reference_experiment']} vs {verdict['experiment_id']} | "
            f"`{verdict['intentional_variable']}` ({fields}) | "
            f"{'yes' if verdict['compatible'] else '**NO**'} |"
        )
    add("")
    add(
        "Frozen identical across D0, D1 and D2: split, labels, class map, epochs, batch, "
        "optimizer policy, patience, seed, `deterministic`, augmentation policy, "
        "checkpoint-selection rule and validation protocol."
    )
    add("")
    add(
        "`LIMITATION` `optimizer: auto` is the declared policy, so the framework-resolved "
        "optimizer and learning rate may legitimately differ between experiments - the rule "
        "depends on class count and iteration estimate. Each run records what it actually used. "
        "None is forced to match another after execution, and a resolved value is never quoted "
        "as though the configuration had stated it."
    )
    add("")

    add("## 7. Batch and memory policy")
    add("")
    memory = matrix.memory_policy
    add(
        f"`PREDECLARED_PHASE7_POLICY` The controlled batch is **{memory['controlled_batch']}** "
        f"for D0, D1 and D2. If a candidate cannot execute at that batch because of a genuine "
        f"CUDA out-of-memory failure, the experiment **stops** and is classified "
        f"`{memory['classification']}`."
    )
    add("")
    add("Forbidden as a silent rescue:")
    add("")
    for item in memory["forbidden_mitigations"]:
        add(f"- `{item}`")
    add("")
    add(f"{memory['rationale'].strip()}")
    add("")

    add("## 8. Selection logic")
    add("")
    logic = policy["selection_logic"]
    add(f"`SELECTION_RULE` Ranked by `{PRIMARY_SELECTION_METRIC}`, margin `{logic['margin']}`.")
    add("")
    add("| Case | Condition | Outcome |")
    add("| --- | --- | --- |")
    add(
        f"| `{CASE_A}` | no candidate improves D0 by more than {logic['margin']} | retain D0, the "
        "lower-complexity baseline |"
    )
    add(
        f"| `{CASE_B}` | one candidate improves D0 by more than {logic['margin']} **and** "
        f"separates from the runner-up by more than {logic['margin']} | that candidate is the "
        "validation-performance leader |"
    )
    add(
        f"| `{CASE_C}` | a candidate improves D0 by more than {logic['margin']} but does **not** "
        "separate from the runner-up | no winner; "
        "`PERFORMANCE_TIE_REQUIRES_EFFICIENCY_COMPARISON` |"
    )
    add(
        f"| `{CASE_D}` | execution or protocol failure | return for protocol review; the failure "
        "is **not** read as model inferiority |"
    )
    add("")
    add(f"`SELECTION_RULE` {logic['boundary']}")
    add("")
    add(f"`SELECTION_RULE` {logic['case_c_coverage_note']}")
    add("")
    add(f"`LIMITATION` {logic['not_a_significance_test']}")
    add("")
    add("Nothing may override these cases:")
    add("")
    for item in logic["overrides_forbidden"]:
        add(f"- {item}")
    add("")

    add("## 9. The descriptive class")
    add("")
    rare = policy["rare_class_policy"]
    add(
        f"`PREDECLARED_PHASE7_POLICY` The rule classified "
        f"{', '.join(f'`{n}`' for n in rare['classes'])} as "
        f"`{DESCRIPTIVE_HIGH_UNCERTAINTY}` from the support in section 1 - "
        f"{rare['support'][0]['validation_positive_images']} positive validation image(s) and "
        f"{rare['support'][0]['validation_instances']} instances."
    )
    add("")
    add(rare["reporting_rule"].strip())
    add("")
    add("Forbidden:")
    add("")
    for item in rare["forbidden"]:
        add(f"- {item}")
    add("")

    add("## 10. Recall")
    add("")
    add(
        f"`D0_REFERENCE_RESULT` D0 reported precision {metrics['precision']} against recall "
        f"{metrics['recall']} - it misses far more than it mislabels."
    )
    add("")
    add(
        "`PREDECLARED_PHASE7_POLICY` Recall is recorded as an important **diagnostic** for "
        "phase 7. It is not promoted to the selection metric and no recall-weighted composite "
        "is introduced. After D1 and D2 it may help explain why "
        f"`{PRIMARY_SELECTION_METRIC}` moved; it does not independently choose a winner."
    )
    add("")

    add("## 11. No post-hoc intervention")
    add("")
    add(f"`PREDECLARED_PHASE7_POLICY` {policy['post_hoc_policy']['rule']}")
    add("")
    add("Not to be attempted automatically once a result appears:")
    add("")
    for item in policy["post_hoc_policy"]["forbidden_automatic_followups"]:
        add(f"- {item}")
    add("")

    add("## 12. D0 revalidation audit note")
    add("")
    audit = policy["revalidation_audit"]
    add(
        f"`LIMITATION` Classified `{audit['classification']}` (equivalently "
        f"`{audit['equivalent_classification']}`)."
    )
    add("")
    add(audit["what_happened"].strip())
    add("")
    add(f"**Why no selection freedom.** {audit['why_no_selection_freedom'].strip()}")
    add("")
    add(f"**Reporting rule.** {audit['reporting_rule'].strip()}")
    add("")
    add(
        f"**Evidence basis: `{audit['evidence_basis']}`.** Two different things, kept apart. "
        "Corroborated on disk: "
        + "; ".join(audit["corroborated_on_disk"])
        + ". **Not** independently verified here: "
        + "; ".join(audit["not_independently_verified_here"])
    )
    add("")

    add("## 13. Holdout")
    add("")
    holdout = policy["holdout_policy"]
    add(f"`PREDECLARED_PHASE7_POLICY` {holdout['rule']}")
    add("")
    add(
        f"During this phase: `{holdout['test']['status']}`, guard state "
        f"`{holdout['guard']['state_during_this_phase']}`, `{HOLDOUT_UNLOCK_ENV_VAR}` unset. No "
        "holdout image, label, annotation, count or adapter was read, written or produced, and "
        "no comparison artifact carries a holdout number."
    )
    add("")

    add("## 14. Limitations")
    add("")
    for item in policy["limitations"]:
        add(f"- `{item['label']}` **{item['id']}.** {item['statement']}")
    add("")

    add("## 15. What this phase did not do")
    add("")
    add("- D1 was not trained. D2 was not trained. No alternative detector was trained.")
    add("- D0 was not retrained, re-validated or modified.")
    add("- `yolo11s.pt` was not fetched or fingerprinted; that is required before D1 runs.")
    add("- No holdout data was read, materialised, adapted or measured.")
    add("- No image-level error analysis was performed.")
    add("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Freeze the Phase 7 comparison protocol.

    Args:
        argv: Command-line arguments.

    Returns:
        Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="validate the inputs and the protocol without writing any artifact",
    )
    args = parser.parse_args(argv)

    paths = ProjectPaths.from_root()
    if holdout_unlocked():
        print(
            f"REFUSED: {HOLDOUT_UNLOCK_ENV_VAR} is set. Phase 7 selection happens on validation "
            "only, and this phase declines to run in an environment where the holdout is "
            "unlocked.",
            file=sys.stderr,
        )
        return 2

    try:
        matrix = load_experiment_matrix(paths.configs / MATRIX_YAML)
        baseline = load_detection_baseline_config(paths.configs / BASELINE_YAML)
        d0_manifest = read_json(paths.reports / D0_MANIFEST_JSON)
        split_manifest = read_json(paths.reports / SPLIT_MANIFEST_JSON)
    except (ComparisonConfigError, ConfigError, FreezeInputError) as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2

    reference_declaration = matrix.reference
    if baseline.experiment_id != reference_declaration.experiment_id:
        print(
            f"BLOCKED: {BASELINE_YAML} declares {baseline.experiment_id!r} but the matrix names "
            f"{reference_declaration.experiment_id!r} as the reference",
            file=sys.stderr,
        )
        return 2
    if d0_manifest.get("baseline_config_sha256") != baseline.fingerprint():
        print(
            f"BLOCKED: the committed D0 result was produced under a different protocol than "
            f"{BASELINE_YAML} currently holds. The reference cannot be built from a protocol "
            "the result did not use.",
            file=sys.stderr,
        )
        return 2

    try:
        reference = build_reference(d0_manifest, split_manifest, matrix)
    except (FreezeInputError, ComparisonError) as exc:
        print(f"{INVALID_REFERENCE}: {exc}", file=sys.stderr)
        return 2

    reference_protocol = baseline.as_dict()
    protocols: dict[str, dict[str, Any]] = {
        reference_declaration.experiment_id: dict(
            sorted(flatten_protocol(reference_protocol).items())
        )
    }
    verdicts: list[dict[str, Any]] = []
    try:
        for declaration in matrix.candidates:
            resolved = resolve_candidate_protocol(reference_protocol, declaration)
            protocols[declaration.experiment_id] = resolved
            verdict = check_protocol_compatibility(
                reference_protocol,
                resolved,
                declaration,
                reference_experiment=reference_declaration.experiment_id,
            )
            if not verdict.compatible:
                print(
                    f"BLOCKED: {declaration.experiment_id} is not a one-variable comparison "
                    f"against {reference_declaration.experiment_id}: undeclared differences "
                    f"{list(verdict.undeclared_differences)}, unapplied declarations "
                    f"{list(verdict.unapplied_declarations)}",
                    file=sys.stderr,
                )
                return 2
            verdicts.append(verdict.as_dict())
    except ComparisonError as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2

    matrix_path = paths.configs / MATRIX_YAML
    matrix_sha256 = _file_digest(matrix_path)
    policy = build_policy(matrix, reference, verdicts, protocols, matrix_sha256=matrix_sha256)

    for name, payload in (("policy", policy), ("reference", reference)):
        leaks = holdout_leaks(payload)
        if leaks:
            print(
                f"BLOCKED: the {name} artifact names the protected split at {leaks}",
                file=sys.stderr,
            )
            return 2

    report = policy_markdown(matrix, policy, reference, commit=git_commit(paths.root))
    findings = scan_for_sensitive(report) + scan_for_sensitive(json.dumps(policy))
    if findings:
        print(f"BLOCKED: sensitive content in the emitted artifacts: {findings}", file=sys.stderr)
        return 2

    if args.verify_only:
        print("VERIFIED: the phase 7 comparison protocol is internally consistent.")
        _summarise(policy, reference)
        return 0

    policy_digest = write_json(paths.reports / POLICY_JSON, policy)
    reference_digest = write_json(paths.reports / REFERENCE_JSON, reference)
    (paths.reports / POLICY_MD).write_text(report, encoding="utf-8", newline="\n")

    record = ProvenanceRecord.create(
        name="detection_experiment_protocol_freeze",
        phase=7,
        config={
            "detection_experiments": f"configs/{MATRIX_YAML}",
            "detection_baseline": f"configs/{BASELINE_YAML}",
            "experiment_matrix_sha256": matrix_sha256,
            "matrix_content_sha256": matrix.fingerprint(),
        },
        details={
            "phase": matrix.phase,
            "classification": PROTOCOL_FROZEN,
            "reference_experiment": matrix.reference_experiment,
            "supported_classes": policy["supported_classes"],
            "descriptive_classes": policy["descriptive_high_uncertainty_classes"],
            "support_rule": matrix.support_rule.as_dict(),
            "primary_selection_metric": PRIMARY_SELECTION_METRIC,
            "d0_supported_macro_map50_95": reference["metrics"][PRIMARY_SELECTION_METRIC],
            "d0_all_class_map50_95": reference["metrics"][OFFICIAL_ALL_CLASS_METRIC],
            "practical_equivalence_margin": str(matrix.practical_equivalence_margin),
            "frozen_experiments": [item.experiment_id for item in matrix.candidates],
            "experiments_executed_in_this_phase": [],
            "models_trained_in_this_phase": 0,
            "holdout_accessed": False,
            "policy_sha256": policy_digest,
            "reference_sha256": reference_digest,
            "revalidation_audit": REVALIDATION_CLASSIFICATION,
        },
        repo_root=paths.root,
    )
    for name in (MATRIX_YAML, BASELINE_YAML):
        record.add_input(paths.configs / name, relative_to=paths.root)
    for name in (D0_MANIFEST_JSON, SPLIT_MANIFEST_JSON):
        record.add_input(paths.reports / name, relative_to=paths.root)
    for name in (POLICY_JSON, POLICY_MD, REFERENCE_JSON):
        record.add_output(paths.reports / name, relative_to=paths.root)
    record.write_json(paths.reports / PROVENANCE_JSON)

    print(f"{PROTOCOL_FROZEN}")
    _summarise(policy, reference)
    print(f"policy    reports/{POLICY_JSON}  sha256 {policy_digest}")
    print(f"reference reports/{REFERENCE_JSON}  sha256 {reference_digest}")
    return 0


def _file_digest(path: Path) -> str:
    """Digest a configuration file's exact bytes.

    Args:
        path: File to hash.

    Returns:
        Its SHA-256 hex digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summarise(policy: dict[str, Any], reference: dict[str, Any]) -> None:
    """Print the freeze summary.

    Args:
        policy: The emitted policy.
        reference: The emitted D0 reference.
    """
    metrics = reference["metrics"]
    print(f"  supported classes        {policy['supported_classes']}")
    print(f"  descriptive classes      {policy['descriptive_high_uncertainty_classes']}")
    print(f"  D0 {PRIMARY_SELECTION_METRIC}  {metrics[PRIMARY_SELECTION_METRIC]}")
    print(f"  D0 {OFFICIAL_ALL_CLASS_METRIC}           {metrics[OFFICIAL_ALL_CLASS_METRIC]}")
    print(f"  margin                   {policy['practical_equivalence_margin']['value']}")
    print(f"  declared experiments     {[item['experiment_id'] for item in policy['experiments']]}")
    print(f"  holdout                  {policy['holdout_policy']['test']['status']}")


if __name__ == "__main__":
    raise SystemExit(main())
