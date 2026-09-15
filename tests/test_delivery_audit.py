"""Phase 12A - the final repository and delivery-readiness audit.

An audit is only useful if it cannot quietly become wrong. These tests pin the
two properties that give it that: the measured half re-derives from the
repository on every run, so a fixed README is reported as fixed; and the audit
itself can never reach a model or the holdout, so running it costs the
experiment nothing.

They also pin the audit's own boundary - zero models executed, zero holdout
reads, no scientific artifact modified - and the honesty rules the validator
enforces on the declared findings.
"""

from __future__ import annotations

import ast
import csv
import json
import os

import pytest

from construction_safety_vision import delivery_audit as audit
from construction_safety_vision import delivery_audit_findings as findings
from construction_safety_vision.paths import ProjectPaths

MODULE = "src/construction_safety_vision/delivery_audit.py"
FINDINGS_MODULE = "src/construction_safety_vision/delivery_audit_findings.py"
SCRIPT = "scripts/audit_delivery_readiness.py"

FORBIDDEN_IMPORTS = ("torch", "ultralytics", "cv2", "PIL", "pycocotools", "numpy")
"""Nothing that can load a model or decode an image belongs on an audit path."""

HOLDOUT_ACCESSORS = (
    "load_frozen_splits",
    "FrozenSplits",
    "assert_split_allowed",
    "authorize_final_holdout_access",
    "allow_test",
)
"""Names that would mean the audit could reach the protected split."""


@pytest.fixture(scope="module")
def paths() -> ProjectPaths:
    return ProjectPaths.from_root()


@pytest.fixture(scope="module")
def committed(paths: ProjectPaths) -> dict:
    path = paths.reports / audit.AUDIT_JSON
    assert path.is_file(), "the repository audit must be committed"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rebuilt(paths: ProjectPaths) -> dict:
    return audit.build(paths, env=dict(os.environ))


# --- the audit re-derives ----------------------------------------------------------------


def test_the_audit_rebuilds_byte_identically(committed: dict, rebuilt: dict) -> None:
    assert rebuilt == committed
    assert rebuilt["repository_audit_sha256"] == committed["repository_audit_sha256"]


def test_the_rendered_report_rebuilds_byte_identically(
    paths: ProjectPaths, committed: dict
) -> None:
    assert audit.render(committed) == (paths.reports / audit.AUDIT_MARKDOWN).read_text(
        encoding="utf-8"
    )


def test_the_gap_register_csv_matches_the_payload(paths: ProjectPaths, committed: dict) -> None:
    with (paths.reports / audit.GAP_CSV).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows == audit.gap_rows(committed)


def test_the_compliance_csv_matches_the_payload(paths: ProjectPaths, committed: dict) -> None:
    with (paths.reports / audit.COMPLIANCE_CSV).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows == audit.compliance_rows(committed)


def test_the_fingerprint_covers_the_payload(committed: dict) -> None:
    payload = {key: value for key, value in committed.items() if key != "repository_audit_sha256"}
    assert audit.digest_payload(payload) == committed["repository_audit_sha256"]


def test_the_committed_audit_validates_and_classifies_complete(committed: dict) -> None:
    assert audit.validate(committed) == []
    assert audit.classify(committed) == audit.CLASSIFICATION_COMPLETE
    assert committed["classification"] == audit.CLASSIFICATION_COMPLETE


# --- the measured half is measured -------------------------------------------------------


def test_the_inventory_matches_the_git_index(paths: ProjectPaths, committed: dict) -> None:
    tracked = audit.tracked_files(paths.root)
    assert committed["measurements"]["inventory"] == audit.inventory(tracked)
    assert committed["measurements"]["inventory"]["total_tracked_files"] == len(tracked)


def test_the_inventory_delta_reconciles_against_git(paths: ProjectPaths, committed: dict) -> None:
    """Parent + added - deleted must equal the head count, all four read from git."""
    delta = committed["measurements"]["inventory_delta"]
    baseline = audit._tracked_at_commit(paths.root, audit.PHASE_12A_BASELINE_COMMIT)
    current = audit.tracked_files(paths.root)

    assert delta["tracked_files_at_parent"] == len(baseline)
    assert delta["tracked_files_at_phase_12a_head"] == len(current)
    assert delta["delta_tracked_files"] == len(current) - len(baseline)
    assert (
        delta["tracked_files_at_parent"] + delta["added_count"] - delta["deleted_count"]
        == delta["tracked_files_at_phase_12a_head"]
    )
    assert (
        delta["tracked_files_at_phase_12a_head"]
        == (committed["measurements"]["inventory"]["total_tracked_files"])
    )


def test_the_created_file_count_equals_the_files_listed(
    paths: ProjectPaths, committed: dict
) -> None:
    """A declared count of created files must equal git's, and equal the list's length.

    This is the inconsistency the block exists to prevent: a report that says it
    created eight files while enumerating nine.
    """
    delta = committed["measurements"]["inventory_delta"]
    baseline = set(audit._tracked_at_commit(paths.root, audit.PHASE_12A_BASELINE_COMMIT))
    added = sorted(set(audit.tracked_files(paths.root)) - baseline)

    assert delta["added"] == added
    assert delta["added_count"] == len(added)
    assert delta["added_count"] == len(delta["added"])
    assert delta["deleted_count"] == len(delta["deleted"])
    assert delta["modified_count"] == len(delta["modified"])


def test_audit_outputs_are_not_the_tracked_file_delta(committed: dict) -> None:
    """The five artifacts this phase writes are not the number of files it adds."""
    delta = committed["measurements"]["inventory_delta"]
    assert delta["audit_output_artifacts"] == sorted(audit.AUDIT_OUTPUT_ARTIFACTS)
    assert delta["audit_output_artifacts_added"] == len(audit.AUDIT_OUTPUT_ARTIFACTS)
    assert delta["implementation_support_files_added"] == len(delta["implementation_support_files"])
    assert (
        delta["audit_output_artifacts_added"] + delta["implementation_support_files_added"]
        == delta["added_count"]
    )
    assert delta["audit_outputs_are_not_the_file_delta"] is True
    assert set(delta["audit_output_artifacts"]).isdisjoint(delta["implementation_support_files"])


def test_a_broken_reconciliation_is_refused(committed: dict) -> None:
    """The validator must reject inventory arithmetic that does not close."""
    payload = json.loads(json.dumps(committed))
    payload["measurements"]["inventory_delta"]["added_count"] -= 1
    assert any("reconcile" in problem for problem in audit.validate(payload))

    payload = json.loads(json.dumps(committed))
    payload["measurements"]["inventory_delta"]["added"].pop()
    assert any("files listed" in problem for problem in audit.validate(payload))

    payload = json.loads(json.dumps(committed))
    del payload["measurements"]["inventory_delta"]
    assert any("baseline" in problem for problem in audit.validate(payload))


def test_deliverable_probes_reflect_the_index(paths: ProjectPaths, committed: dict) -> None:
    """A probe must report presence from the index, not from a stored opinion."""
    tracked = set(audit.tracked_files(paths.root))
    probes = committed["measurements"]["deliverables"]["probes"]
    for name, entry in probes.items():
        expected = next((path for path in entry["candidates_checked"] if path in tracked), None)
        assert entry["exists"] is (expected is not None), name
        assert entry["path"] == expected, name


def test_stale_probes_reflect_the_documentation(paths: ProjectPaths, committed: dict) -> None:
    """Retiring a stale string must make the next audit report it retired."""
    for probe in committed["measurements"]["stale_claims"]["probes"]:
        path = paths.root / probe["file"]
        text = path.read_text(encoding="utf-8") if path.is_file() else ""
        assert probe["occurrences"] == text.count(probe["needle"]), probe["probe_id"]
        assert probe["present"] is (probe["occurrences"] > 0), probe["probe_id"]


def test_every_stale_probe_explains_itself(committed: dict) -> None:
    for probe in committed["measurements"]["stale_claims"]["probes"]:
        assert probe["why_false"].strip(), probe["probe_id"]
        assert probe["contradicts"].strip(), probe["probe_id"]


def test_headline_values_come_from_the_committed_results(
    paths: ProjectPaths, committed: dict
) -> None:
    """Every quoted final number must still equal the artifact field it cites."""
    checks = committed["measurements"]["result_consistency"]["headlines"]
    cache: dict[str, dict] = {}
    for label, check in checks.items():
        name = check["evidence_artifact"]
        if name not in cache:
            cache[name] = json.loads((paths.root / name).read_text(encoding="utf-8"))
        value = cache[name]
        for segment in check["evidence_field"].split("."):
            value = value[segment]
        assert value == check["value"], label


def test_headline_documentation_presence_is_measured(paths: ProjectPaths, committed: dict) -> None:
    readme = (paths.root / "README.md").read_text(encoding="utf-8")
    for label, check in committed["measurements"]["result_consistency"]["headlines"].items():
        assert check["present_in_readme"] is (check["rendered"] in readme), label


# --- the scientific lock the audit must observe -------------------------------------------


def test_the_scientific_state_is_frozen(committed: dict) -> None:
    lock = committed["measurements"]["scientific_lock"]
    assert lock["final_test_state"] == "FINAL_TEST_OBSERVED"
    assert lock["holdout_reads_permitted"] == 1
    assert lock["holdout_reads_performed"] == 1
    assert lock["holdout_evaluation_attempts"] == 1
    assert lock["unique_models_executed"] == 2
    assert lock["total_model_inference_passes"] == 3
    for key in (
        "model_selection",
        "hyperparameter_tuning",
        "threshold_tuning",
        "data_cleaning_for_performance",
    ):
        assert lock[key] == "CLOSED", key


def test_the_environment_gate_is_absent_now() -> None:
    assert os.environ.get(audit.ENVIRONMENT_GATE) is None


def test_the_audit_records_the_gate_absent(committed: dict) -> None:
    lock = committed["measurements"]["scientific_lock"]
    assert lock["environment_gate_present_in_process"] is False


def test_the_audit_executed_nothing(committed: dict) -> None:
    mine = committed["this_audit"]
    assert mine["models_executed"] == 0
    assert mine["model_inference_passes"] == 0
    assert mine["holdout_accessed"] is False
    assert mine["holdout_accessor_invoked"] is False
    assert mine["holdout_content_read"] == 0
    assert mine["metrics_recomputed"] == 0
    assert mine["test_identifiers_enumerated"] == 0
    assert mine["thresholds_tuned"] == 0
    assert mine["scientific_results_modified"] is False
    assert mine["contains_new_scientific_results"] is False
    assert mine["readme_rewritten"] is False
    assert mine["report_generated"] is False
    assert mine["video_started"] is False


def test_no_holdout_identifier_reaches_the_audit_artifacts(paths: ProjectPaths) -> None:
    """The audit publishes counts and judgements, never a holdout image id."""
    with (paths.reports / "final_split_assignments.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        identifiers = {
            row["source_image_id"] for row in csv.DictReader(handle) if row["split"] == "test"
        }
    assert identifiers, "the fixture that makes this test meaningful is missing"
    for name in (audit.AUDIT_JSON, audit.AUDIT_MARKDOWN, audit.GAP_CSV, audit.COMPLIANCE_CSV):
        text = (paths.reports / name).read_text(encoding="utf-8")
        leaked = sorted(identifier for identifier in identifiers if identifier in text)
        assert leaked == [], (name, len(leaked))


# --- the audit cannot reach a model or the holdout ----------------------------------------


@pytest.mark.parametrize("relative", [MODULE, FINDINGS_MODULE, SCRIPT])
def test_no_model_or_image_library_is_imported(paths: ProjectPaths, relative: str) -> None:
    tree = ast.parse((paths.root / relative).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported.isdisjoint(FORBIDDEN_IMPORTS), f"{relative} imports {imported}"


@pytest.mark.parametrize("relative", [MODULE, FINDINGS_MODULE, SCRIPT])
def test_no_holdout_accessor_is_named(paths: ProjectPaths, relative: str) -> None:
    text = (paths.root / relative).read_text(encoding="utf-8")
    for name in HOLDOUT_ACCESSORS:
        assert name not in text, f"{relative} names the holdout accessor {name}"


def test_the_script_never_writes_the_environment_gate(paths: ProjectPaths) -> None:
    tree = ast.parse((paths.root / SCRIPT).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                assert "environ" not in ast.unparse(target), "the audit must not set the gate"
        if isinstance(node, ast.Call) and ast.unparse(node.func).endswith(
            ("environ.setdefault", "environ.update", "putenv")
        ):
            pytest.fail("the audit must never set the environment gate")


# --- the honesty rules the validator enforces ---------------------------------------------


def test_all_four_personas_are_audited(committed: dict) -> None:
    seen = [entry["persona"] for entry in committed["declared_findings"]["personas"]]
    assert sorted(seen) == sorted(audit.PERSONAS)


def test_every_persona_records_strengths_gaps_and_a_reason(committed: dict) -> None:
    for entry in committed["declared_findings"]["personas"]:
        assert entry["strengths"], entry["persona"]
        assert entry["gaps"], entry["persona"]
        assert entry["verdict_reason"].strip(), entry["persona"]


def test_the_three_readiness_verdicts_are_declared_values(committed: dict) -> None:
    readiness = committed["readiness"]
    assert readiness["professor"] in audit.PROFESSOR_VERDICTS
    assert readiness["recruiter"] in audit.RECRUITER_VERDICTS
    assert readiness["engineering"] in audit.ENGINEERING_VERDICTS


def test_an_incomplete_requirement_must_name_an_action(committed: dict) -> None:
    for row in committed["declared_findings"]["compliance_matrix"]:
        if row["state"] in {"PARTIAL", "MISSING"}:
            assert row["remaining_action"] not in ("", "none"), row["requirement_id"]


def test_the_missing_deliverables_are_reported_missing(committed: dict) -> None:
    """The audit must not mark a deliverable complete that the index lacks."""
    probes = committed["measurements"]["deliverables"]["probes"]
    states = {
        row["requirement_id"]: row["state"]
        for row in committed["declared_findings"]["compliance_matrix"]
    }
    if not probes["technical_report"]["exists"]:
        assert states["R15"] == "MISSING"
    if not probes["pitch_script"]["exists"]:
        assert states["R19"] == "MISSING"
    if not probes["video_inference_script"]["exists"]:
        assert states["R14"] == "MISSING"
    if not probes["genai_declaration"]["exists"]:
        assert states["R20"] == "MISSING"
    if not committed["measurements"]["deliverables"]["executable_notebook_exists"]:
        assert states["R18"] == "MISSING"


def test_no_production_claim_is_permitted(committed: dict) -> None:
    positioning = committed["declared_findings"]["positioning"]
    assert positioning["production_claim_permitted"] is False
    assert positioning["production_oriented_assessed"] is True


def test_every_gap_carries_evidence_effort_and_a_fix(committed: dict) -> None:
    for gap in committed["declared_findings"]["gap_register"]:
        assert gap["severity"] in audit.SEVERITIES, gap["gap_id"]
        assert gap["effort"] in audit.EFFORTS, gap["gap_id"]
        assert gap["evidence"].strip(), gap["gap_id"]
        assert gap["recommended_fix"].strip(), gap["gap_id"]
        assert gap["persona"], gap["gap_id"]


def test_gap_ids_are_unique(committed: dict) -> None:
    ids = [gap["gap_id"] for gap in committed["declared_findings"]["gap_register"]]
    assert len(ids) == len(set(ids))


def test_every_claim_says_what_the_evidence_supports(committed: dict) -> None:
    for entry in committed["declared_findings"]["claim_audit"]:
        assert entry["verdict"] in audit.CLAIM_VERDICTS, entry["claim_id"]
        assert entry["what_the_evidence_supports"].strip(), entry["claim_id"]


def test_the_bounded_findings_are_preserved(committed: dict) -> None:
    """The audit must not soften the findings the evidence actually bounds."""
    claims = {entry["claim"]: entry for entry in committed["declared_findings"]["claim_audit"]}
    localisation = claims["The segmenter is the better object localiser"]
    assert localisation["verdict"] == "UNSUPPORTED"
    association = claims["Masks improve person-PPE association"]
    assert association["verdict"] == "UNSUPPORTED"
    latency = claims["The segmenter costs roughly 30% more end-to-end latency"]
    assert latency["verdict"] == "SUPPORTED_WITH_LIMITATION"
    assert "CONTROLLED_LOCAL_HARDWARE_BENCHMARK" in latency["what_the_evidence_supports"]
    one_shot = claims["The holdout was evaluated exactly once"]
    assert one_shot["verdict"] == "SUPPORTED_WITH_LIMITATION"
    assert "three frozen-protocol inference passes" in one_shot["what_the_evidence_supports"]
    rare = claims["vest_loose performance is meaningful"]
    assert rare["verdict"] == "UNSUPPORTED"
    for word in ("production-ready", "Production-ready", "Real-time", "Robust"):
        matches = [entry for claim, entry in claims.items() if claim.startswith(word)]
        for entry in matches:
            assert entry["verdict"] == "UNSUPPORTED", entry["claim_id"]


def test_the_optional_bonus_is_not_allowed_to_block_delivery(committed: dict) -> None:
    gaps = {gap["gap_id"]: gap for gap in committed["declared_findings"]["gap_register"]}
    assert gaps["GAP-021"]["severity"] == "P3"
    assert gaps["GAP-021"]["mandatory"] is False
    video = committed["declared_findings"]["video_readiness"]
    assert "optional" in video["tracking_recommendation"].lower()


def test_the_clean_room_plan_excludes_the_holdout(committed: dict) -> None:
    plan = committed["declared_findings"]["clean_room_plan"]
    assert plan["status"] == "DESIGNED_NOT_EXECUTED"
    excluded = " ".join(plan["explicitly_excluded"]).lower()
    assert "holdout" in excluded
    assert "training" in excluded
    for stage in plan["stages"]:
        assert stage["pass_condition"].strip()


def test_the_colab_scope_forbids_training_and_the_holdout(committed: dict) -> None:
    plan = committed["declared_findings"]["colab_plan"]
    forbidden = " ".join(plan["must_not"]).lower()
    assert "train anything" in forbidden
    assert "holdout" in forbidden


# --- the declared findings are internally coherent ----------------------------------------


def test_declared_and_measured_blocks_do_not_overlap(committed: dict) -> None:
    separation = committed["evidence_separation"]
    assert set(separation["measured_blocks"]) == set(committed["measurements"])
    assert set(separation["declared_blocks"]) == set(committed["declared_findings"])
    assert not set(separation["measured_blocks"]) & set(separation["declared_blocks"])


def test_the_summary_counts_match_the_registers(committed: dict) -> None:
    summary = committed["summary"]
    states = [row["state"] for row in committed["declared_findings"]["compliance_matrix"]]
    for state, count in summary["compliance"].items():
        assert states.count(state) == count, state
    severities = [gap["severity"] for gap in committed["declared_findings"]["gap_register"]]
    for severity, count in summary["gaps"].items():
        assert severities.count(severity) == count, severity


def test_the_roadmap_closes_every_p0_gap(committed: dict) -> None:
    """A prioritised register is only useful if the plan actually addresses it."""
    p0 = {
        gap["gap_id"]
        for gap in committed["declared_findings"]["gap_register"]
        if gap["severity"] == "P0"
    }
    planned = {
        gap_id
        for phase in committed["declared_findings"]["delivery_roadmap"]
        for gap_id in phase["closes"]
    }
    assert p0 <= planned, f"P0 gaps with no phase: {sorted(p0 - planned)}"


def test_the_findings_module_is_pure_data() -> None:
    """Every declared block must be a plain call returning JSON-shaped values."""
    for name in (
        "compliance_matrix",
        "personas",
        "claim_audit",
        "portfolio_signals",
        "gap_register",
        "diagrams",
        "file_routing",
        "delivery_roadmap",
    ):
        value = getattr(findings, name)()
        assert isinstance(value, tuple) and value, name
        assert all(isinstance(entry, dict) for entry in value), name


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("this_audit", "holdout_accessed"), True),
        (("this_audit", "models_executed"), 1),
        (("this_audit", "scientific_results_modified"), True),
        (("measurements", "scientific_lock", "environment_gate_present_in_process"), True),
        (("measurements", "scientific_lock", "model_selection"), "OPEN"),
        (("measurements", "scientific_lock", "final_test_state"), "FINAL_TEST_PENDING"),
        (("declared_findings", "positioning", "production_claim_permitted"), True),
        (("readiness", "recruiter"), "SPECTACULAR"),
    ],
)
def test_validate_refuses_a_dishonest_payload(committed: dict, path: tuple, value: object) -> None:
    payload = json.loads(json.dumps(committed))
    target = payload
    for segment in path[:-1]:
        target = target[segment]
    target[path[-1]] = value
    assert audit.validate(payload), f"validate must refuse {path} = {value!r}"


def test_classify_reports_a_violation_when_the_gate_is_open(committed: dict) -> None:
    payload = json.loads(json.dumps(committed))
    payload["measurements"]["scientific_lock"]["environment_gate_present_in_process"] = True
    assert audit.classify(payload) == audit.CLASSIFICATION_VIOLATION


def test_classify_reports_a_mismatch_when_the_test_state_moves(committed: dict) -> None:
    payload = json.loads(json.dumps(committed))
    payload["measurements"]["scientific_lock"]["final_test_state"] = "FINAL_TEST_PENDING"
    assert audit.classify(payload) == audit.CLASSIFICATION_STATE_MISMATCH


def test_the_provenance_record_reports_an_audit_only_phase(paths: ProjectPaths) -> None:
    record = json.loads((paths.reports / audit.AUDIT_PROVENANCE).read_text(encoding="utf-8"))
    details = record["details"]
    assert details["classification"] == audit.CLASSIFICATION_COMPLETE
    assert details["audit_only"] is True
    assert details["models_executed"] == 0
    assert details["holdout_accessed"] is False
    assert details["test_accessed"] is False
    assert details["scientific_results_modified"] is False
    assert details["contains_new_scientific_results"] is False
    assert details["pre_existing_artifacts_unchanged"] is True
