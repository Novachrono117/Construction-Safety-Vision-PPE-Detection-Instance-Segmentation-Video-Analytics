"""Tests that the committed Phase 7D freeze describes the decision that was made.

Two things are checked here that nothing else can check.

**That the historical experiment artifacts did not move.** A freeze that quietly
rewrote D0's, D1's or D2's manifest - to refresh a status field, to restate a
number "more clearly" - would invalidate the provenance those manifests carry
for each other. Their digests are recorded in the freeze manifest, and this file
re-verifies them against the files on disk.

**That the winner was derived, not asserted.** The freeze manifest says D2. So
would a manifest someone typed by hand. These tests rebuild every experiment
record from its committed result manifest, re-derive the class-support filter
from the frozen split, recompute the selection metric, and run the frozen
selection logic - then require its output to equal what was frozen. If the
committed decision and the policy ever disagree, the tests fail rather than the
report quietly being wrong.

No test here runs a model, loads a checkpoint, or touches the holdout.
"""

from __future__ import annotations

import json
import subprocess
from decimal import Decimal

import pytest

from conftest import holdout_has_been_evaluated
from construction_safety_vision.data.canonical import scan_for_sensitive
from construction_safety_vision.detection_comparison import (
    CASE_B,
    OFFICIAL_ALL_CLASS_METRIC,
    PRACTICAL_EQUIVALENCE_MARGIN,
    PRIMARY_SELECTION_METRIC,
    build_experiment_record,
    class_support,
    compare,
    descriptive_class_names,
    load_experiment_matrix,
    supported_class_names,
)
from construction_safety_vision.detection_freeze import (
    FINAL_DETECTOR_MANIFEST,
    FROZEN,
    final_detector_fingerprint,
    holdout_leaks,
    load_final_detector,
    parse_final_detector,
)
from construction_safety_vision.paths import ProjectPaths
from construction_safety_vision.provenance import sha256_file
from construction_safety_vision.splits import HOLDOUT_UNLOCK_ENV_VAR

EXPERIMENTS = ("D0", "D1", "D2")
HISTORICAL = (
    "detection_D0_manifest.json",
    "detection_D0_report.md",
    "detection_D1_manifest.json",
    "detection_D1_report.md",
    "detection_D2_manifest.json",
    "detection_D2_report.md",
    "detection_comparison_policy.json",
    "detection_comparison_reference.json",
)


@pytest.fixture(scope="module")
def paths() -> ProjectPaths:
    return ProjectPaths.from_root()


@pytest.fixture(scope="module")
def manifest(paths: ProjectPaths) -> dict:
    path = paths.reports / FINAL_DETECTOR_MANIFEST
    if not path.is_file():
        pytest.skip(f"{FINAL_DETECTOR_MANIFEST} not present; run the Phase 7D freeze")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def report(paths: ProjectPaths) -> str:
    path = paths.reports / "detection_selection_report.md"
    if not path.is_file():
        pytest.skip("detection_selection_report.md not present")
    return path.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def comparison_csv(paths: ProjectPaths) -> list[dict[str, str]]:
    path = paths.reports / "detection_experiment_comparison.csv"
    if not path.is_file():
        pytest.skip("detection_experiment_comparison.csv not present")
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    header = lines[0].split(",")
    return [dict(zip(header, line.split(","), strict=True)) for line in lines[1:]]


@pytest.fixture(scope="module")
def results(paths: ProjectPaths) -> dict:
    path = paths.reports / "detection_experiment_results.json"
    if not path.is_file():
        pytest.skip("detection_experiment_results.json not present")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def experiment_manifests(paths: ProjectPaths) -> dict[str, dict]:
    found = {}
    for experiment_id in EXPERIMENTS:
        path = paths.reports / f"detection_{experiment_id}_manifest.json"
        if not path.is_file():
            pytest.skip(f"detection_{experiment_id}_manifest.json not present")
        found[experiment_id] = json.loads(path.read_text(encoding="utf-8"))
    return found


@pytest.fixture(scope="module")
def recomputed(paths: ProjectPaths, experiment_manifests: dict[str, dict]):
    """Rebuild the whole comparison from the committed artifacts.

    Args:
        paths: Project layout.
        experiment_manifests: The committed result manifests.

    Returns:
        The records, the comparison payload and the frozen class support.
    """
    matrix = load_experiment_matrix(paths.configs / "detection_experiments.yaml")
    split = json.loads((paths.reports / "split_manifest.json").read_text(encoding="utf-8"))
    support = class_support(
        split, experiment_manifests["D0"]["class_map"], rule=matrix.support_rule
    )
    records = {
        experiment_id: build_experiment_record(payload, support)
        for experiment_id, payload in experiment_manifests.items()
    }
    return records, compare(matrix, records), tuple(support)


# --- historical artifacts must not have moved ---------------------------------


@pytest.mark.parametrize("name", HISTORICAL)
def test_the_historical_experiment_artifacts_are_unchanged(paths, manifest, name: str):
    recorded = manifest["historical_artifact_digests"]
    assert name in recorded, f"{name} was not fingerprinted at freeze time"
    assert sha256_file(paths.reports / name) == recorded[name]


def test_the_frozen_policy_digest_is_the_one_the_experiments_ran_under(
    manifest, experiment_manifests
):
    policy = manifest["phase7_policy_sha256"]
    assert manifest["historical_artifact_digests"]["detection_comparison_policy.json"] == policy
    for experiment_id in ("D1", "D2"):
        assert experiment_manifests[experiment_id]["phase7_policy_sha256"] == policy


def test_the_experiment_matrix_digest_matches_the_committed_config(paths, manifest):
    assert (
        sha256_file(paths.configs / "detection_experiments.yaml")
        == manifest["experiment_matrix_sha256"]
    )


# --- the decision was derived, not asserted -----------------------------------


def test_the_frozen_logic_independently_produces_case_b(recomputed, manifest):
    _, comparison, _ = recomputed
    assert comparison["selection"]["case"] == CASE_B
    assert manifest["policy_case"] == CASE_B


def test_the_frozen_logic_independently_elects_the_recorded_experiment(recomputed, manifest):
    _, comparison, _ = recomputed
    selection = comparison["selection"]
    assert selection["preferred_experiment"] == manifest["selected_experiment"]
    assert selection["leader"] == manifest["selected_experiment"]


def test_the_selected_experiment_is_d2_yolo11n_at_768(manifest):
    assert manifest["selected_experiment"] == "D2"
    assert manifest["model"] == "YOLO11n"
    assert manifest["imgsz"] == 768
    assert manifest["batch"] == 16


def test_the_recorded_primary_metrics_match_the_committed_per_class_metrics(recomputed, manifest):
    records, _, _ = recomputed
    exact = manifest["primary_metric_values_exact"]
    for experiment_id, record in records.items():
        assert Decimal(exact[experiment_id]) == record.supported_macro
        assert manifest["primary_metric_values"][experiment_id] == pytest.approx(
            float(record.supported_macro), abs=5e-7
        )


def test_the_recorded_deltas_are_the_differences_of_the_recorded_values(recomputed, manifest):
    records, _, _ = recomputed
    for key, value in manifest["primary_metric_deltas_exact"].items():
        left, right = key.split("_minus_")
        assert Decimal(value) == records[left].supported_macro - records[right].supported_macro


def test_d1_is_below_the_reference_and_d2_clears_it_beyond_the_margin(recomputed):
    records, _, _ = recomputed
    margin = PRACTICAL_EQUIVALENCE_MARGIN
    assert records["D1"].supported_macro - records["D0"].supported_macro < margin
    assert records["D2"].supported_macro - records["D0"].supported_macro > margin
    assert records["D2"].supported_macro - records["D1"].supported_macro > margin


def test_the_overall_ranking_is_d2_above_d0_above_d1(recomputed, manifest):
    # The exact committed values, not the rounded ones: the reference sits
    # BETWEEN the two candidates, which is the whole reason the candidate-only
    # fields have to be qualified.
    records, _, _ = recomputed
    assert records["D2"].supported_macro > records["D0"].supported_macro
    assert records["D0"].supported_macro > records["D1"].supported_macro
    assert Decimal(manifest["primary_metric_values_exact"]["D2"]) == Decimal("0.594018000000")
    assert Decimal(manifest["primary_metric_values_exact"]["D0"]) == Decimal("0.570141750000")
    assert Decimal(manifest["primary_metric_values_exact"]["D1"]) == Decimal("0.560017000000")


def test_the_recorded_overall_ranking_matches_the_committed_metrics(recomputed, manifest):
    records, _, _ = recomputed
    ranking = manifest["overall_validation_ranking"]
    assert [entry["experiment_id"] for entry in ranking] == ["D2", "D0", "D1"]
    assert [entry["rank"] for entry in ranking] == [1, 2, 3]
    for entry in ranking:
        recorded = Decimal(entry[f"{PRIMARY_SELECTION_METRIC}_exact"])
        assert recorded == records[entry["experiment_id"]].supported_macro


def test_the_case_and_leader_are_the_frozen_ones(manifest):
    assert manifest["policy_case"] == "CASE_B_VALIDATION_PERFORMANCE_LEADER"
    assert manifest["validation_performance_leader"] == "D2"
    assert manifest["selected_experiment"] == "D2"
    assert manifest["reference_experiment"] == "D0"


def test_the_candidate_runner_up_means_the_non_reference_challenger(manifest, recomputed):
    _, comparison, _ = recomputed
    assert manifest["candidate_runner_up"] == "D1"
    # It is the frozen engine's output, and that engine ranks candidates only.
    assert comparison["selection"]["runner_up"] == manifest["candidate_runner_up"]
    semantics = manifest["policy_case_derivation"]["candidate_runner_up_semantics"]
    assert "CONTROLLED CHALLENGER" in semantics
    assert "NOT a claim" in semantics
    assert "overall_validation_ranking" in semantics


def test_the_candidate_runner_up_is_not_the_second_highest_overall(manifest):
    # The specific confusion this audit exists to prevent, asserted rather than
    # left to prose: the second-ranked candidate and the second-ranked
    # experiment are different experiments here.
    second_overall = manifest["overall_validation_ranking"][1]["experiment_id"]
    assert second_overall == "D0"
    assert manifest["candidate_runner_up"] != second_overall


def test_no_artifact_carries_an_unqualified_runner_up_field(manifest, results):
    for payload in (manifest, results):
        flat = json.dumps(payload)
        assert '"runner_up"' not in flat
        assert '"candidate_runner_up"' in flat


def test_the_leader_separation_is_between_the_two_candidates(manifest, recomputed):
    records, _, _ = recomputed
    separation = Decimal(str(manifest["policy_case_derivation"]["leader_separation"]))
    expected = records["D2"].supported_macro - records["D1"].supported_macro
    assert separation == round(expected, 6)
    assert (
        "never between a candidate and the reference"
        in (manifest["policy_case_derivation"]["leader_separation_semantics"])
    )


def test_the_report_separates_the_three_ranking_concepts(report):
    assert "Validation performance leader" in report
    assert "Second-highest experiment overall" in report
    assert "Non-reference candidate runner-up" in report
    assert "D2 > D0 > D1" in report
    assert "controlled challengers" in report
    assert "does not make its metric the second-highest overall" in report


def test_the_comparison_csv_records_the_overall_rank(comparison_csv):
    ranks = {row["experiment_id"]: int(row["overall_validation_rank"]) for row in comparison_csv}
    assert ranks == {"D2": 1, "D0": 2, "D1": 3}


def test_the_live_results_record_the_ranking_and_qualified_runner_up(results):
    selection = results["selection"]
    assert selection["reference_experiment"] == "D0"
    assert selection["validation_performance_leader"] == "D2"
    assert selection["candidate_runner_up"] == "D1"
    assert [entry["experiment_id"] for entry in selection["overall_validation_ranking"]] == [
        "D2",
        "D0",
        "D1",
    ]


def test_the_frozen_margin_is_unchanged(manifest):
    assert Decimal(manifest["practical_equivalence_margin"]) == PRACTICAL_EQUIVALENCE_MARGIN


def test_the_selection_metric_is_the_predeclared_one(manifest, recomputed):
    _, _, support = recomputed
    assert manifest["primary_selection_metric"] == PRIMARY_SELECTION_METRIC
    assert manifest["selection_metric_classes"] == list(supported_class_names(support))
    assert manifest["official_all_class_metric"] == OFFICIAL_ALL_CLASS_METRIC


# --- no efficiency tie-break, no threshold tuning -----------------------------


def test_no_efficiency_tie_break_was_run_or_required(manifest, recomputed):
    _, comparison, _ = recomputed
    assert comparison["selection"]["efficiency_comparison_required"] is False
    tie_break = manifest["efficiency_tie_break"]
    assert tie_break["benchmark_run"] is False
    assert tie_break["required_by_policy"] is False
    assert "Case C" in tie_break["reason"]


def test_no_threshold_was_tuned(manifest):
    threshold = manifest["threshold_tuning"]
    assert threshold["tuned"] is False
    assert set(threshold["parameters_untouched"]) == {"confidence", "IoU", "NMS"}


def test_no_training_or_evaluation_happened_in_this_phase(manifest):
    assert manifest["training_or_evaluation_performed_in_this_phase"] is False


# --- checkpoint identity ------------------------------------------------------


def test_the_selected_checkpoint_is_the_selected_experiments_best_checkpoint(
    manifest, experiment_manifests
):
    chosen = experiment_manifests[manifest["selected_experiment"]]["best_checkpoint"]
    recorded = manifest["selected_checkpoint"]
    assert recorded["sha256"] == chosen["sha256"]
    assert recorded["size_bytes"] == chosen["size_bytes"]
    assert recorded["relative_path"] == chosen["relative_path"]
    assert recorded["relative_path"].endswith("best.pt")


def test_the_checkpoint_digest_is_required_and_well_formed(manifest):
    sha256 = manifest["selected_checkpoint"]["sha256"]
    assert isinstance(sha256, str)
    assert len(sha256) == 64
    assert set(sha256) <= set("0123456789abcdef")


def test_no_other_experiments_checkpoint_is_named_as_the_final_detector(
    manifest, experiment_manifests
):
    selected = manifest["selected_experiment"]
    chosen = manifest["selected_checkpoint"]["sha256"]
    for experiment_id, payload in experiment_manifests.items():
        if experiment_id == selected:
            continue
        assert payload["best_checkpoint"]["sha256"] != chosen
        assert payload["last_checkpoint"]["sha256"] != chosen
    assert experiment_manifests[selected]["last_checkpoint"]["sha256"] != chosen


def test_the_frozen_copy_carries_the_selected_digest(paths, manifest):
    frozen = manifest.get("frozen_copy")
    if frozen is None:
        pytest.skip("no frozen copy was recorded on this machine")
    assert frozen["sha256"] == manifest["selected_checkpoint"]["sha256"]
    assert frozen["committed"] is False
    path = paths.root / frozen["relative_path"]
    if not path.is_file():
        pytest.skip("the git-ignored frozen copy is not present on this machine")
    assert sha256_file(path) == manifest["selected_checkpoint"]["sha256"]


def test_the_frozen_copy_lives_outside_the_training_run_directory(manifest):
    frozen = manifest.get("frozen_copy")
    if frozen is None:
        pytest.skip("no frozen copy was recorded on this machine")
    # A copy inside the run directory would be overwritten by a re-run, which is
    # exactly what freezing it is meant to survive.
    assert not frozen["relative_path"].startswith("artifacts/detection/")


# --- the semantic fingerprint -------------------------------------------------


def test_the_final_detector_fingerprint_recomputes_deterministically(manifest):
    detector = parse_final_detector(manifest)
    assert detector.recompute_fingerprint() == manifest["final_detector_sha256"]
    assert detector.recompute_fingerprint() == detector.recompute_fingerprint()


def test_the_fingerprint_changes_when_the_selected_checkpoint_changes(manifest):
    identity = {
        "selected_experiment": manifest["selected_experiment"],
        "model": manifest["model"],
        "imgsz": manifest["imgsz"],
        "selected_checkpoint_sha256": manifest["selected_checkpoint"]["sha256"],
        "experiment_sha256": manifest["experiment_sha256"],
        "phase7_policy_sha256": manifest["phase7_policy_sha256"],
        "split_assignment_sha256": manifest["split_reference"]["split_assignment_sha256"],
        "adapter_config_sha256": manifest["dataset_fingerprints"]["adapter_config_sha256"],
        "class_map_sha256": manifest["dataset_fingerprints"]["class_map_sha256"],
    }
    assert final_detector_fingerprint(identity) == manifest["final_detector_sha256"]
    swapped = dict(identity, selected_checkpoint_sha256="0" * 64)
    assert final_detector_fingerprint(swapped) != manifest["final_detector_sha256"]


def test_the_fingerprint_excludes_machine_specific_content(manifest):
    fingerprint_inputs = json.dumps(
        [
            manifest["selected_experiment"],
            manifest["model"],
            manifest["imgsz"],
            manifest["selected_checkpoint"]["sha256"],
            manifest["experiment_sha256"],
            manifest["phase7_policy_sha256"],
        ]
    )
    assert "created_at" not in fingerprint_inputs
    assert ":" not in manifest["selected_checkpoint"]["relative_path"]


# --- the manifest as a whole --------------------------------------------------


def test_the_manifest_parses_through_the_public_accessor(paths):
    detector = load_final_detector(paths.reports)
    assert detector.selected_experiment == "D2"
    assert detector.policy_case == CASE_B
    assert detector.binary_distribution_status == "LOCAL_IGNORED_FROZEN_ARTIFACT"


def test_the_manifest_declares_the_freeze_status(manifest):
    assert manifest["status"] == FROZEN
    assert manifest["selection_status"] == "FINAL_SELECTED"
    assert manifest["selection_method"] == "PREDECLARED_POLICY_PLUS_HUMAN_REVIEW"
    assert manifest["task"] == "detection"


def test_the_binary_distribution_status_is_explicit(manifest):
    assert manifest["binary_distribution_status"] == "LOCAL_IGNORED_FROZEN_ARTIFACT"
    assert manifest["repository_contains_model_binary"] is False
    assert manifest["selected_checkpoint"]["committed"] is False
    assert "cannot run inference immediately" in manifest["reproducibility_note"]


def test_no_model_binary_is_tracked_by_git(paths):
    # Ask git, not the filesystem. Untracked weights on the machine are expected
    # and fine; a *tracked* one would be the defect.
    completed = subprocess.run(
        ["git", "ls-files"],
        cwd=paths.root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        pytest.skip("git is not available here")
    tracked = [
        line
        for line in completed.stdout.splitlines()
        if line.lower().endswith((".pt", ".pth", ".onnx", ".engine"))
    ]
    assert tracked == []


def test_human_review_is_recorded_as_confirmatory_not_overriding(manifest):
    review = manifest["human_review"]
    assert review["status"] == "FINAL_SELECTED"
    assert "override the frozen policy" in review["did_not"]
    assert "consult the holdout" in review["did_not"]
    assert "Confirmatory, not corrective" in review["relationship_to_policy"]


def test_the_winner_is_recorded_as_the_engines_output_not_a_constant(manifest):
    derivation = manifest["policy_case_derivation"]
    assert derivation["hardcoded_winner"] is False
    assert derivation["validation_performance_leader"] == manifest["selected_experiment"]
    assert derivation["efficiency_comparison_required"] is False


def test_the_freeze_script_does_not_name_the_winner(paths):
    # The claim "the policy chose D2" is only worth something if the script that
    # applied the policy could not have written D2 into the answer.
    source = (paths.scripts / "freeze_final_detector.py").read_text(encoding="utf-8")
    body = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("#"))
    for banned in ('selected = "D2"', "selected = 'D2'", 'preferred = "D2"'):
        assert banned not in body


def test_the_freeze_script_contains_no_training_or_evaluation_path(paths):
    source = (paths.scripts / "freeze_final_detector.py").read_text(encoding="utf-8")
    for banned in ("YOLO(", ".train(", ".val(", ".predict(", "ultralytics"):
        assert banned not in source


# --- rare-class and interpretation limits -------------------------------------


def test_the_rare_class_is_reported_but_did_not_decide(manifest, recomputed):
    _, _, support = recomputed
    policy = manifest["rare_class_policy"]
    assert policy["classes"] == list(descriptive_class_names(support))
    assert policy["reported_in_full"] is True
    assert policy["used_in_selection"] is False
    assert policy["selection_rationale_excludes_it"] is True


def test_every_class_is_still_reported_for_every_experiment(manifest, experiment_manifests):
    class_names = set(experiment_manifests["D0"]["class_map"])
    for experiment_id in EXPERIMENTS:
        assert set(manifest["per_class_metrics"][experiment_id]) == class_names


def test_the_interpretation_limits_are_recorded(manifest):
    limits = manifest["interpretation_limits"]
    joined = " ".join(limits["does_not_establish"]).lower()
    for expected in ("statistical significance", "random seeds", "small objects", "test"):
        assert expected in joined
    assert limits["small_object_hypothesis"]["status"] == "UNSUPPORTED_BY_THE_SHAPE_OF_THE_RESULT"
    assert "not attributable to either variable" in limits["d1_d2_not_ranked"]


def test_the_metrics_are_declared_validation_only(manifest):
    assert manifest["metrics_are_validation_only"] is True


# --- the live results artifact ------------------------------------------------


def test_the_live_results_record_the_final_selection(results, manifest):
    assert results["final_selected_detector"] == manifest["selected_experiment"]
    selection = results["selection"]
    assert selection["status"] == "FINAL_SELECTED"
    assert selection["policy_case"] == CASE_B
    assert selection["preferred_experiment"] == manifest["selected_experiment"]
    assert selection["final_selected_experiment"] == manifest["selected_experiment"]
    assert selection["advisory_only"] is False


def test_the_live_results_still_record_all_three_experiments_complete(results):
    rows = {row["experiment_id"]: row for row in results["experiments"]}
    assert set(rows) == set(EXPERIMENTS)
    for experiment_id in EXPERIMENTS:
        assert rows[experiment_id]["status"] == "COMPLETE"
    assert results["pending_experiments"] == []


def test_the_live_results_metrics_were_not_touched_by_the_freeze(results, experiment_manifests):
    rows = {row["experiment_id"]: row for row in results["experiments"]}
    for experiment_id in EXPERIMENTS:
        assert (
            rows[experiment_id]["per_class_metrics"]
            == experiment_manifests[experiment_id]["per_class_metrics"]
        )


# --- the comparison table -----------------------------------------------------


def test_the_comparison_csv_covers_every_experiment(comparison_csv):
    assert [row["experiment_id"] for row in comparison_csv] == list(EXPERIMENTS)


def test_the_comparison_csv_agrees_with_the_committed_manifests(
    comparison_csv, recomputed, experiment_manifests
):
    records, _, _ = recomputed
    for row in comparison_csv:
        experiment_id = row["experiment_id"]
        record = records[experiment_id]
        payload = experiment_manifests[experiment_id]
        assert float(row["supported_macro_map50_95"]) == pytest.approx(
            float(record.supported_macro), abs=5e-7
        )
        assert float(row["all_class_map50_95"]) == pytest.approx(
            float(record.all_class_map50_95), abs=1e-9
        )
        assert row["checkpoint_sha256"] == payload["best_checkpoint"]["sha256"]
        assert row["experiment_sha256"] == record.experiment_sha256
        assert int(row["imgsz"]) == record.imgsz
        assert row["model"] == record.model


def test_the_comparison_csv_marks_the_reference_and_the_margin_statuses(comparison_csv):
    rows = {row["experiment_id"]: row for row in comparison_csv}
    assert rows["D0"]["margin_status"] == "REFERENCE"
    assert rows["D1"]["margin_status"] == "BELOW_D0"
    assert rows["D2"]["margin_status"] == "IMPROVES_D0_BEYOND_MARGIN"


# --- the holdout --------------------------------------------------------------


def test_the_freeze_records_the_holdout_as_protected(manifest):
    assert manifest["test"]["status"] == "PROTECTED_NOT_ACCESSED"
    assert "not read" in manifest["test"]["reason"]


def test_no_holdout_material_exists(paths):
    # Before phase 11B the holdout has no on-disk presence at all. Phase 11B
    # materialises it, through the phase 5D function and under both
    # authorisation gates, so from then on these paths legitimately exist.
    evaluated = holdout_has_been_evaluated(paths)
    canonical = paths.data_processed / "canonical"
    assert evaluated or not (canonical / "images" / "test").exists()
    assert evaluated or not (canonical / "annotations" / "detection_test.coco.json").exists()
    assert evaluated or not (canonical / "annotations" / "segmentation_test.coco.json").exists()
    adapter = paths.data_processed / "adapters" / "yolo_detection"
    assert not (adapter / "images" / "test").exists()
    assert not (adapter / "labels" / "test").exists()


def test_the_freeze_artifacts_name_the_protected_split_only_as_a_notice(manifest):
    # Deliberately NOT implemented by loading the holdout id list out of
    # split_manifest.json and grepping for it: reading that section to run a
    # check would be the very bypass the split guard exists to prevent. The
    # freeze module's own leak detector asks the structural question instead -
    # the only permitted mention of the protected split is a protection notice.
    assert holdout_leaks(manifest) == []


def test_the_freeze_refuses_a_manifest_that_leaks_a_holdout_number(manifest):
    # The detector above is only worth trusting if it fires on a real leak.
    leaking = dict(manifest, test={"status": "EVALUATED", "map50_95": 0.5})
    assert holdout_leaks(leaking) != []


def test_the_holdout_unlock_variable_is_not_referenced_by_the_artifacts(manifest, report):
    # Naming the variable in the report's compliance section is the point; the
    # manifest must not carry it as data.
    assert HOLDOUT_UNLOCK_ENV_VAR not in json.dumps(manifest)
    assert HOLDOUT_UNLOCK_ENV_VAR in report


# --- the report ---------------------------------------------------------------


@pytest.mark.parametrize(
    "label",
    [
        "PREDECLARED_POLICY",
        "COMPUTED_RESULT",
        "SELECTION_RULE",
        "HUMAN_REVIEW",
        "FINAL_DECISION",
        "LIMITATION",
        "HOLDOUT_POLICY",
    ],
)
def test_the_report_uses_the_declared_labels(report, label: str):
    assert f"`{label}`" in report


def test_the_report_states_the_final_decision_with_its_identity(report, manifest):
    assert manifest["final_detector_sha256"] in report
    assert manifest["selected_checkpoint"]["sha256"] in report
    assert manifest["experiment_sha256"] in report
    assert "YOLO11n at imgsz 768" in report


def test_the_report_makes_no_test_performance_claim(report):
    lowered = report.lower()
    for phrase in ("test map", "test performance of", "on the test set we", "holdout result"):
        assert phrase not in lowered
    assert "say nothing about test performance" in lowered


def test_the_report_does_not_present_the_gain_as_confirming_the_hypothesis(report):
    lowered = report.lower()
    assert "not supported by the shape of the result" in lowered
    for phrase in ("confirms the small-object", "because of small objects, as predicted"):
        assert phrase not in lowered


def test_the_report_does_not_rank_d1_against_d2_as_a_controlled_result(report):
    assert "difference, not a ranking" in report


def test_the_report_states_the_rare_class_did_not_decide(report):
    assert "`vest_loose`" in report
    assert "may not be cited as a reason for this decision" in report


def test_the_committed_artifacts_carry_no_sensitive_content(paths, manifest, report):
    findings = scan_for_sensitive(report) + scan_for_sensitive(json.dumps(manifest))
    csv_text = (paths.reports / "detection_experiment_comparison.csv").read_text(encoding="utf-8")
    findings += scan_for_sensitive(csv_text)
    assert findings == []


# --- provenance ---------------------------------------------------------------


def test_the_provenance_records_a_phase_that_trained_nothing(paths, manifest):
    path = paths.reports / "final_detector.provenance.json"
    if not path.is_file():
        pytest.skip("final_detector.provenance.json not present")
    record = json.loads(path.read_text(encoding="utf-8"))
    details = record["details"]
    assert record["phase"] == 7
    assert details["phase"] == "7D"
    assert details["classification"] == "DETECTOR_FROZEN"
    assert details["models_trained_in_this_phase"] == 0
    assert details["models_evaluated_in_this_phase"] == 0
    assert details["efficiency_benchmarks_run"] == 0
    assert details["thresholds_tuned"] == 0
    assert details["holdout_accessed"] is False
    assert details["final_selected_experiment"] == manifest["selected_experiment"]
    assert details["final_detector_sha256"] == manifest["final_detector_sha256"]


def test_the_provenance_output_digests_match_the_files_on_disk(paths):
    path = paths.reports / "final_detector.provenance.json"
    if not path.is_file():
        pytest.skip("final_detector.provenance.json not present")
    record = json.loads(path.read_text(encoding="utf-8"))
    for entry in record["outputs"]:
        target = paths.root / entry["path"]
        assert sha256_file(target) == entry["sha256"]
