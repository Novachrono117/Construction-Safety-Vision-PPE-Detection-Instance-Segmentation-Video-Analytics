"""Tests that the committed phase 8G artifacts describe the selection that happened.

Five things are checked here that the unit tests cannot.

**That the winner was derived, not asserted.** The primary metrics are read back
out of the two experiments' committed canonical evaluations, the delta is
recomputed, and the frozen margin is reapplied. If the manifest quoted a number
no artifact contains, or a verdict the margin does not produce, that shows up
here.

**That nothing was measured in this phase.** No training, no evaluation, no
inference, no benchmark, no threshold.

**That the trade-off is visible.** ``helmet_loose`` regressed while the
aggregate rose. The manifest and the report both have to say so.

**That the mechanism claim stayed honest.** ``person`` improved most, which is
consistent with the phase 8D hypothesis and is not evidence of it.

**That the holdout is absent.** Structurally, not merely unmentioned.

No test here trains, infers or reads the holdout.
"""

from __future__ import annotations

import csv
import json

import pytest

from construction_safety_vision.canonical_evaluation import (
    IMPROVES,
    PRACTICAL_EQUIVALENCE_MARGIN,
    PRIMARY_METRIC,
    classify_delta,
    load_canonical_evaluation_config,
)
from construction_safety_vision.data.canonical import scan_for_sensitive
from construction_safety_vision.paths import ProjectPaths
from construction_safety_vision.provenance import sha256_file
from construction_safety_vision.segmentation_comparison import (
    CANDIDATE,
    CHECKPOINT_POLICY,
    ONE_VARIABLE_FIELD,
    REFERENCE,
)
from construction_safety_vision.segmentation_freeze import (
    FINAL_SELECTED,
    FROZEN,
    HOLDOUT_STATUS,
    TASK,
    CheckpointMismatchError,
    holdout_leaks,
    load_checkpoint_path,
    load_final_segmenter,
)
from construction_safety_vision.segmentation_s1 import CONSISTENT, RARE_CLASS, RARE_CLASS_STATUS

MANIFEST_JSON = "final_segmenter_manifest.json"
REPORT_MD = "segmentation_selection_report.md"
COMPARISON_CSV = "segmentation_experiment_comparison.csv"
RESULTS_JSON = "segmentation_experiment_results.json"
PROVENANCE_JSON = "final_segmenter.provenance.json"

S0_CANONICAL_JSON = "segmentation_S0_canonical_evaluation.json"
S1_CANONICAL_JSON = "segmentation_S1_canonical_evaluation.json"
S0_MASK_IOU_JSON = "segmentation_S0_mask_iou.json"
S1_MASK_IOU_JSON = "segmentation_S1_mask_iou.json"
S0_RESULT_JSON = "segmentation_S0_result_manifest.json"
S1_RESULT_JSON = "segmentation_S1_result_manifest.json"

EXPECTED_S1_CHECKPOINT = "29337d671459d0f742fb713613cc41d0eafcb8a21f5846ac0da65b2ffc024f20"
EXPECTED_S0_CHECKPOINT = "d7b512b95fafdc8658fd802459d2c87d9ad3f11f922162a774dacf8ca75b87a3"

pytestmark = pytest.mark.skipif(
    not (ProjectPaths.from_root().reports / MANIFEST_JSON).is_file(),
    reason="no segmenter has been frozen in this working tree",
)


@pytest.fixture(scope="module")
def paths() -> ProjectPaths:
    """Resolve the project layout.

    Returns:
        The project paths.
    """
    return ProjectPaths.from_root()


def _read(paths: ProjectPaths, name: str) -> dict:
    """Read a committed JSON artifact.

    Args:
        paths: Project layout.
        name: File name under ``reports/``.

    Returns:
        The parsed mapping.
    """
    return json.loads((paths.reports / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def manifest(paths: ProjectPaths) -> dict:
    """Read the final-segmenter manifest.

    Args:
        paths: Project layout.

    Returns:
        The manifest.
    """
    return _read(paths, MANIFEST_JSON)


# --- the accessor, against the real artifact -------------------------------------


def test_the_committed_manifest_loads_and_recomputes(paths: ProjectPaths) -> None:
    """The accessor every later phase uses can read what this phase wrote."""
    segmenter = load_final_segmenter(paths.reports)

    assert segmenter.selected_experiment == CANDIDATE
    assert segmenter.recompute_fingerprint() == segmenter.fingerprint
    assert segmenter.model == "YOLO11n-seg"
    assert segmenter.imgsz == 768
    assert segmenter.batch == 8
    assert segmenter.mask_ratio == 4
    assert segmenter.overlap_mask is False


def test_the_frozen_checkpoint_resolves_and_verifies(paths: ProjectPaths) -> None:
    """The frozen copy exists and holds the selected bytes."""
    segmenter = load_final_segmenter(paths.reports)
    resolved = load_checkpoint_path(segmenter, paths.root)

    assert resolved.is_file()
    assert sha256_file(resolved) == EXPECTED_S1_CHECKPOINT
    assert segmenter.checkpoint_sha256 == EXPECTED_S1_CHECKPOINT


def test_the_accessor_rejects_the_reference_checkpoint(paths: ProjectPaths) -> None:
    """S0's real weights are on this machine and must not pass as the segmenter."""
    segmenter = load_final_segmenter(paths.reports)
    s0 = paths.root / "artifacts" / "segmentation" / REFERENCE / "weights" / "best.pt"
    if not s0.is_file():
        pytest.skip("S0's checkpoint is not on this machine")

    with pytest.raises(CheckpointMismatchError, match=REFERENCE):
        segmenter.verify_checkpoint(s0)


def test_the_accessor_rejects_last_pt(paths: ProjectPaths) -> None:
    """The final epoch's weights are a different model."""
    segmenter = load_final_segmenter(paths.reports)
    last = paths.root / "artifacts" / "segmentation" / CANDIDATE / "weights" / "last.pt"
    if not last.is_file():
        pytest.skip("S1's last.pt is not on this machine")

    with pytest.raises(CheckpointMismatchError, match=r"last\.pt"):
        segmenter.verify_checkpoint(last)


# --- the selection, derived rather than trusted -------------------------------------


def test_the_primary_metrics_come_from_the_committed_evaluations(
    paths: ProjectPaths, manifest: dict
) -> None:
    """Both figures must be the ones the experiments' own artifacts record."""
    s0 = _read(paths, S0_CANONICAL_JSON)
    s1 = _read(paths, S1_CANONICAL_JSON)
    comparison = manifest["comparison"]

    assert comparison["reference_supported_macro"] == s0["supported_macro"]["value"]
    assert comparison["candidate_supported_macro"] == s1["supported_macro"]["value"]
    assert manifest["primary_selection_metric"] == PRIMARY_METRIC


def test_the_delta_recomputes_and_the_margin_reapplies(manifest: dict) -> None:
    """The verdict must be the frozen rule's output, recomputed here."""
    comparison = manifest["comparison"]
    derived = classify_delta(
        comparison["reference_supported_macro"], comparison["candidate_supported_macro"]
    )

    assert comparison["delta"] == pytest.approx(derived["delta"], abs=1e-9)
    assert comparison["margin"] == PRACTICAL_EQUIVALENCE_MARGIN
    assert comparison["verdict"] == derived["verdict"] == IMPROVES
    assert manifest["margin_classification"] == derived["verdict"]
    assert comparison["margin_is_not_a_significance_test"] is True


def test_the_selection_used_no_predictions(manifest: dict) -> None:
    """This phase read artifacts; it did not recompute anything from images."""
    comparison = manifest["comparison"]

    assert comparison["derived_from"] == "COMMITTED_CANONICAL_EVALUATION_ARTIFACTS_ONLY"
    assert comparison["predictions_recomputed"] is False
    assert manifest["models_trained_in_this_phase"] == 0
    assert manifest["models_evaluated_in_this_phase"] == 0
    assert manifest["inference_run_in_this_phase"] is False
    assert manifest["benchmarks_run"] == 0
    assert manifest["thresholds_tuned"] == 0


def test_the_direct_iou_direction_recomputes(paths: ProjectPaths, manifest: dict) -> None:
    """The secondary diagnostic's direction is derived from committed values."""
    s0 = _read(paths, S0_MASK_IOU_JSON)["global"]
    s1 = _read(paths, S1_MASK_IOU_JSON)["global"]
    direct = manifest["direct_iou_comparison"]

    for key, value in direct["reference"].items():
        assert value == s0[key], key
    for key, value in direct["candidate"].items():
        assert value == s1[key], key
    assert direct["status"] == CONSISTENT
    assert direct["does_not_rank"] is True
    assert direct["composite_created"] is False


def test_the_one_variable_contract_is_verified_against_the_experiments(
    paths: ProjectPaths, manifest: dict
) -> None:
    """Only overlap_mask differs, and it is checked against each run's own config."""
    s0 = _read(paths, S0_RESULT_JSON)["effective_training_configuration"]
    s1 = _read(paths, S1_RESULT_JSON)["effective_training_configuration"]
    contract = manifest["one_variable_contract"]

    assert contract["intentional_field"] == ONE_VARIABLE_FIELD
    assert contract["one_variable"] is True
    assert s0[ONE_VARIABLE_FIELD] is True
    assert s1[ONE_VARIABLE_FIELD] is False
    for field in ("imgsz", "batch", "epochs", "seed", "mask_ratio"):
        assert s0[field] == s1[field], field
    assert contract["verified_against"] == (
        "EACH_EXPERIMENTS_OWN_COMMITTED_EFFECTIVE_CONFIGURATION"
    )


# --- the frozen identity ---------------------------------------------------------------


def test_the_selected_model_is_recorded_completely(manifest: dict) -> None:
    """Everything that makes this a specific model is in the manifest."""
    assert manifest["status"] == FROZEN
    assert manifest["selection_status"] == FINAL_SELECTED
    assert manifest["task"] == TASK
    assert manifest["selected_experiment"] == CANDIDATE
    assert manifest["selection_method"] == "PREDECLARED_CANONICAL_POLICY_PLUS_HUMAN_REVIEW"
    assert manifest["model"] == "YOLO11n-seg"
    assert manifest["imgsz"] == 768
    assert manifest["batch"] == 8
    assert manifest["mask_ratio"] == 4
    assert manifest["overlap_mask"] is False
    assert manifest["checkpoint_selection_policy"] == CHECKPOINT_POLICY


def test_the_selected_checkpoint_is_s1_best_not_s0_and_not_last(manifest: dict) -> None:
    """Both wrong checkpoints are named and rejected by identity."""
    checkpoint = manifest["selected_checkpoint"]
    rejected = manifest["rejected_checkpoints"]

    assert checkpoint["sha256"] == EXPECTED_S1_CHECKPOINT
    assert checkpoint["relative_path"].endswith("best.pt")
    assert checkpoint["is_best_not_last"] is True
    assert rejected["reference_experiment_checkpoint_sha256"] == EXPECTED_S0_CHECKPOINT
    assert rejected["last_pt_rejected"] is True
    assert checkpoint["sha256"] != rejected["reference_experiment_checkpoint_sha256"]


def test_the_frozen_copy_lives_outside_the_run_directory(
    paths: ProjectPaths, manifest: dict
) -> None:
    """A re-run must not be able to overwrite the frozen model."""
    frozen = manifest["frozen_copy"]

    assert frozen["relative_path"] == "artifacts/frozen/segmentation/S1_best.pt"
    assert "artifacts/segmentation/S1/" not in frozen["relative_path"]
    assert frozen["committed"] is False
    assert frozen["sha256"] == EXPECTED_S1_CHECKPOINT
    assert sha256_file(paths.root / frozen["relative_path"]) == EXPECTED_S1_CHECKPOINT


def test_the_frozen_copy_is_git_ignored(paths: ProjectPaths, manifest: dict) -> None:
    """The repository commits the record, not the binary."""
    import subprocess

    relative = manifest["frozen_copy"]["relative_path"]
    result = subprocess.run(
        ["git", "check-ignore", relative],
        cwd=paths.root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, f"{relative} is not git-ignored"
    assert manifest["binary_distribution_status"] == "LOCAL_IGNORED_FROZEN_ARTIFACT"
    assert manifest["repository_contains_model_binary"] is False


def test_no_model_binary_is_committed(paths: ProjectPaths) -> None:
    """Nothing under reports/ is a checkpoint."""
    assert not list(paths.reports.rglob("*.pt"))
    assert not list(paths.reports.rglob("*.pth"))


def test_the_fingerprint_is_recorded_and_deterministic(manifest: dict) -> None:
    """Recomputed by the accessor above; here its shape is checked."""
    assert len(manifest["final_segmenter_sha256"]) == 64
    assert manifest["final_segmenter_sha256"] != manifest["experiment_sha256"]


def test_the_policy_and_evaluator_are_the_frozen_ones(paths: ProjectPaths, manifest: dict) -> None:
    """The freeze names the phase 8E artifacts by digest."""
    evaluator = load_canonical_evaluation_config(
        paths.configs / "segmentation_canonical_evaluation.yaml"
    )

    assert manifest["canonical_evaluator_sha256"] == evaluator.fingerprint()
    assert manifest["comparison_policy_sha256"] == sha256_file(
        paths.reports / "segmentation_comparison_policy.json"
    )


def test_the_historical_artifacts_are_byte_identical(paths: ProjectPaths, manifest: dict) -> None:
    """Every phase 7D and 8A-8F artifact still hashes to what this phase recorded."""
    assert manifest["historical_artifacts_unchanged"] is True
    for name, expected in manifest["historical_artifact_digests"].items():
        assert sha256_file(paths.root / name) == expected, name


def test_the_detector_was_verified_and_left_alone(manifest: dict) -> None:
    """The detection block stays closed."""
    detector = manifest["frozen_detector"]

    assert detector["selected_experiment"] == "D2"
    assert detector["trained"] is False
    assert detector["validated"] is False
    assert detector["benchmarked"] is False
    assert manifest["detector_touched"] is False


# --- honesty about what the result is ------------------------------------------------------


def test_the_regression_is_recorded_not_buried(paths: ProjectPaths, manifest: dict) -> None:
    """A supported class fell while the aggregate rose, and both artifacts say so."""
    table = manifest["canonical_per_class_comparison"]
    regression = manifest["regression_note"]
    expected = sorted(
        name for name, row in table.items() if row["supported"] and row["delta_AP@0.50:0.95"] < 0
    )

    assert regression["regressed_supported_classes"] == expected
    assert expected, "this test is meaningless if nothing regressed"
    assert regression["why_it_moved"] == "UNKNOWN"

    report = (paths.reports / REPORT_MD).read_text(encoding="utf-8")
    for name in expected:
        assert name in report
        assert (
            str(regression["deltas"][name]) in report
            or f"{regression['deltas'][name]:+f}" in report
        )


def test_the_person_finding_is_not_turned_into_a_causal_claim(manifest: dict) -> None:
    """Consistency with a post-hoc hypothesis is not evidence of the mechanism."""
    person = manifest["person_diagnostic"]

    assert person["mechanism_status"] == "CONSISTENT_WITH_PHASE_8D_OVERLAP_TARGET_HYPOTHESIS"
    assert person["mechanism_is_not"] == "PROOF_OF_CAUSAL_MECHANISM"
    assert person["selected_the_model"] is False
    assert "POST_HOC_HYPOTHESIS_GENERATING" in person["note"]


def test_the_person_delta_is_the_largest_and_is_recorded(manifest: dict) -> None:
    """The claim that person improved most is checked against the table."""
    table = manifest["canonical_per_class_comparison"]
    person = manifest["person_diagnostic"]
    largest = max(table, key=lambda name: table[name]["delta_AP@0.50:0.95"])

    assert person["class"] == "person"
    assert largest == "person"
    assert person["delta_AP@0.50:0.95"] == table["person"]["delta_AP@0.50:0.95"]
    assert person["largest_canonical_improvement"] is True


def test_vest_loose_stays_descriptive_and_out_of_the_decision(manifest: dict) -> None:
    """The rare class is reported in full and decided nothing."""
    rare = manifest["rare_class"]

    assert rare["name"] == RARE_CLASS
    assert rare["status"] == RARE_CLASS_STATUS
    assert rare["in_supported_macro"] is False
    assert rare["may_decide_anything"] is False
    assert rare["reported_in_full"] is True
    assert rare["delta_AP@0.50:0.95"] is not None


def test_the_native_metric_did_not_arbitrate(manifest: dict) -> None:
    """overlap_mask changes the native target, so its AP cannot rank the two."""
    native = manifest["native_metrics_comparison"]

    assert native["used_to_arbitrate"] is False
    assert native["status"] == "NOT_CROSS_TARGET_COMPARABLE_FOR_S0_S1_SELECTION"
    assert "different targets" in native["why_not_used"]
    assert native["reference"]["mask"]["mAP@0.50:0.95"]
    assert native["candidate"]["mask"]["mAP@0.50:0.95"]


def test_human_review_confirms_rather_than_overrides(manifest: dict) -> None:
    """The reviewer accepted the policy's answer; they did not substitute one."""
    review = manifest["human_review"]

    assert review["confirms_policy_result"] is True
    assert review["overrides_policy_result"] is False


def test_the_limitations_are_stated(manifest: dict) -> None:
    """Validation-only, single-run, post-S0 policy timing, and the uneven gain."""
    text = " ".join(manifest["limitations"]).lower()

    assert "validation" in text
    assert "run-to-run variance" in text
    assert "significance test" in text
    assert "post_s0_pre_s1_protocol_freeze" in text
    assert "helmet_loose" in text


# --- the comparison table and the live state --------------------------------------------------


def test_the_comparison_csv_matches_the_manifest(paths: ProjectPaths, manifest: dict) -> None:
    """Two rows, and every figure agrees with the artifacts."""
    with (paths.reports / COMPARISON_CSV).open(encoding="utf-8", newline="") as handle:
        rows = {row["experiment_id"]: row for row in csv.DictReader(handle)}

    assert set(rows) == {REFERENCE, CANDIDATE}
    assert rows[REFERENCE]["overlap_mask"] == "True"
    assert rows[CANDIDATE]["overlap_mask"] == "False"
    assert rows[REFERENCE]["selected"] == "False"
    assert rows[CANDIDATE]["selected"] == "True"
    assert rows[REFERENCE]["margin_status"] == "REFERENCE"
    assert rows[CANDIDATE]["margin_status"] == manifest["margin_classification"]
    assert float(rows[CANDIDATE]["canonical_supported_macro_mask_map50_95"]) == pytest.approx(
        manifest["comparison"]["candidate_supported_macro"]
    )
    assert float(rows[REFERENCE]["canonical_supported_macro_mask_map50_95"]) == pytest.approx(
        manifest["comparison"]["reference_supported_macro"]
    )
    assert float(rows[CANDIDATE]["delta_vs_S0"]) == pytest.approx(manifest["comparison"]["delta"])
    assert rows[CANDIDATE]["checkpoint_sha256"] == EXPECTED_S1_CHECKPOINT
    assert rows[REFERENCE]["checkpoint_sha256"] == EXPECTED_S0_CHECKPOINT


def test_the_live_state_records_the_selection(paths: ProjectPaths, manifest: dict) -> None:
    """S0 and S1 complete, S1 selected, review no longer pending."""
    results = _read(paths, RESULTS_JSON)
    by_id = {row["experiment_id"]: row for row in results["experiments"]}

    assert results["selection_status"] == FINAL_SELECTED
    assert results["final_selected_experiment"] == CANDIDATE
    assert results["preferred_experiment"] == CANDIDATE
    assert results["final_segmenter"] == CANDIDATE
    assert results["final_segmenter_sha256"] == manifest["final_segmenter_sha256"]
    assert results["pending_human_review"] is False
    assert by_id[REFERENCE]["selected"] is False
    assert by_id[CANDIDATE]["selected"] is True
    assert by_id[REFERENCE]["status"] == "S0_COMPLETE"
    assert by_id[CANDIDATE]["status"] == "S1_CONTROLLED_EXPERIMENT_COMPLETE"


def test_the_live_state_kept_the_experiment_rows_intact(
    paths: ProjectPaths, manifest: dict
) -> None:
    """This phase measured nothing, so no experiment metric may have moved."""
    results = _read(paths, RESULTS_JSON)
    s0 = _read(paths, S0_CANONICAL_JSON)
    s1 = _read(paths, S1_CANONICAL_JSON)
    by_id = {row["experiment_id"]: row for row in results["experiments"]}

    assert by_id[REFERENCE]["canonical_supported_macro"] == s0["supported_macro"]["value"]
    assert by_id[CANDIDATE]["canonical_supported_macro"] == s1["supported_macro"]["value"]
    assert by_id[CANDIDATE]["delta_primary_vs_reference"] == manifest["comparison"]["delta"]


# --- the holdout ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", [MANIFEST_JSON, RESULTS_JSON])
def test_no_artifact_mentions_the_holdout(paths: ProjectPaths, name: str) -> None:
    """Structurally absent, not merely unmentioned."""
    payload = _read(paths, name)

    assert payload["test"]["status"] == HOLDOUT_STATUS
    assert payload["holdout_accessed"] is False
    assert holdout_leaks(payload) == []


def test_no_artifact_carries_a_test_metric(paths: ProjectPaths, manifest: dict) -> None:
    """A freeze made on validation evidence cannot contain a test number."""
    blob = json.dumps(manifest).lower()

    assert "test_map" not in blob
    assert "test_metric" not in blob
    assert "holdout_map" not in blob


def test_the_report_and_csv_carry_no_sensitive_content(paths: ProjectPaths) -> None:
    """No absolute path, signed URL or credential-shaped string."""
    for name in (REPORT_MD, COMPARISON_CSV):
        text = (paths.reports / name).read_text(encoding="utf-8")
        assert scan_for_sensitive(text) == [], name


def test_the_report_states_the_decision_and_its_limits(paths: ProjectPaths) -> None:
    """The reader is told what was chosen, on what, and what is not established."""
    text = (paths.reports / REPORT_MD).read_text(encoding="utf-8")

    for fragment in (
        "PREDECLARED_POLICY",
        "CONTROLLED_INTERVENTION",
        "CANONICAL_PRIMARY_METRIC",
        "DIRECT_IOU_DIAGNOSTIC",
        "HUMAN_REVIEW",
        "FINAL_DECISION",
        "LIMITATION",
        "HOLDOUT_POLICY",
        "not a significance test",
        "validation number",
    ):
        assert fragment in text, fragment


def test_the_provenance_records_that_nothing_ran(paths: ProjectPaths) -> None:
    """The phase's own record agrees it trained and evaluated nothing."""
    record = _read(paths, PROVENANCE_JSON)
    details = record["details"]

    assert details["classification"] == "SEGMENTER_FROZEN"
    assert details["models_trained_in_this_phase"] == 0
    assert details["models_evaluated_in_this_phase"] == 0
    assert details["inference_run_in_this_phase"] is False
    assert details["benchmarks_run"] == 0
    assert details["holdout_accessed"] is False


def test_the_freeze_script_has_no_training_or_inference_path(paths: ProjectPaths) -> None:
    """A freeze that could train is a freeze that might."""
    source = (paths.root / "scripts" / "freeze_final_segmenter.py").read_text(encoding="utf-8")

    for forbidden in (
        "model.train(",
        "model.val(",
        "model.predict(",
        "from ultralytics",
        "import ultralytics",
        "import torch",
    ):
        assert forbidden not in source, forbidden


def test_the_freeze_script_names_no_winner_as_a_constant(paths: ProjectPaths) -> None:
    """The selected experiment is derived from the policy, never hardcoded."""
    source = (paths.root / "scripts" / "freeze_final_segmenter.py").read_text(encoding="utf-8")

    assert 'selected_experiment": "S1"' not in source
    assert 'SELECTED = "S1"' not in source
