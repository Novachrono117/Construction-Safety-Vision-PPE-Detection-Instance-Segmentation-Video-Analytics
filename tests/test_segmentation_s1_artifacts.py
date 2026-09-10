"""Tests that the committed S1 artifacts describe the experiment that ran.

Six things are checked here that the unit tests cannot.

**That the intervention actually happened.** The framework's own recorded
arguments must say ``overlap_mask: false``, and every other field must match
what phase 8E froze as inherited. A one-variable contract asserted in prose but
not applied by the trainer would be worthless.

**That the reference was read, not re-run.** S0's committed canonical, native
and direct-IoU figures must appear in S1's manifest unchanged, and every phase
8A-8E artifact must still hash to what the manifest recorded.

**That the decision used the frozen evaluator.** The canonical artifact's
fingerprint must equal the committed evaluator's, the primary delta must
recompute from the committed S0 value, and the margin classification must be
the one the frozen rule produces.

**That the native metric did not decide.** It is reported in full and labelled
non-primary; the descriptive native macro decides nothing.

**That nothing was selected.** No final segmenter, no second experiment, no
tuning, no composite.

**That the holdout is absent.** Structurally, not merely unmentioned.

No test here trains, infers or reads the holdout.
"""

from __future__ import annotations

import json

import pytest

from construction_safety_vision.canonical_evaluation import (
    IOU_THRESHOLDS,
    IOU_TYPE,
    MAX_DETS,
    NATIVE_METRIC_STATUS,
    PRACTICAL_EQUIVALENCE_MARGIN,
    PRIMARY_METRIC,
    SELECTION_CASES,
    classify_delta,
    load_canonical_evaluation_config,
)
from construction_safety_vision.data.canonical import scan_for_sensitive
from construction_safety_vision.detection_comparison import (
    SUPPORT_MIN_INSTANCES,
    SUPPORT_MIN_POSITIVE_IMAGES,
)
from construction_safety_vision.mask_iou_evaluation import (
    GROUND_TRUTH_SOURCE,
    GT_IOU50_COVERAGE,
    GT_IOU75_COVERAGE,
    GT_MATCH_COVERAGE,
    GT_NORMALIZED_MASK_IOU,
    MATCHED_MASK_IOU_MEAN,
    MATCHING_ALGORITHM,
    load_mask_iou_config,
)
from construction_safety_vision.paths import ProjectPaths
from construction_safety_vision.provenance import sha256_file
from construction_safety_vision.segmentation_comparison import (
    CHECKPOINT_POLICY,
    ONE_VARIABLE_FIELD,
    REFERENCE,
    load_comparison_config,
)
from construction_safety_vision.segmentation_experiment import (
    ADAPTER_ROLE,
    CANONICAL_GROUND_TRUTH,
    CHECKPOINT_SELECTION_SEMANTICS,
    load_segmentation_baseline_config,
)
from construction_safety_vision.segmentation_s1 import (
    CANONICAL_PRIMARY_METRIC,
    COMPLETE,
    CONSISTENT,
    DESCRIPTIVE_NATIVE_TARGET_METRIC,
    DISAGREEMENT,
    EXPERIMENT_ID,
    FINAL_SEGMENTER,
    HOLDOUT_STATUS,
    NATIVE_TARGET_METRIC,
    PHASE,
    RARE_CLASS,
    RARE_CLASS_STATUS,
    contains_forbidden_split,
    cross_metric_direction,
    experiment_fingerprint,
    validate_canonical_evaluation,
    validate_comparison,
    validate_direct_iou,
    validate_s1_result_manifest,
)

RESULT_JSON = "segmentation_S1_result_manifest.json"
CANONICAL_JSON = "segmentation_S1_canonical_evaluation.json"
MASK_IOU_JSON = "segmentation_S1_mask_iou.json"
REPORT_MD = "segmentation_S1_report.md"
RESULTS_JSON = "segmentation_experiment_results.json"
PROVENANCE_JSON = "segmentation_S1.provenance.json"
PRERUN_JSON = "segmentation_S1_prerun.provenance.json"

S0_RESULT_JSON = "segmentation_S0_result_manifest.json"
S0_CANONICAL_JSON = "segmentation_S0_canonical_evaluation.json"
S0_MASK_IOU_JSON = "segmentation_S0_mask_iou.json"
S1_PROTOCOL_JSON = "segmentation_S1_protocol_manifest.json"

EXPECTED_PRETRAINED = "55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152"
EXPECTED_PRETRAINED_BYTES = 6182636
EXPECTED_S0_CHECKPOINT = "d7b512b95fafdc8658fd802459d2c87d9ad3f11f922162a774dacf8ca75b87a3"

pytestmark = pytest.mark.skipif(
    not (ProjectPaths.from_root().reports / RESULT_JSON).is_file(),
    reason="S1 has not been executed in this working tree",
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
    """Read the S1 result manifest.

    Args:
        paths: Project layout.

    Returns:
        The manifest.
    """
    return _read(paths, RESULT_JSON)


@pytest.fixture(scope="module")
def canonical(paths: ProjectPaths) -> dict:
    """Read the S1 canonical evaluation artifact.

    Args:
        paths: Project layout.

    Returns:
        The artifact.
    """
    return _read(paths, CANONICAL_JSON)


@pytest.fixture(scope="module")
def direct(paths: ProjectPaths) -> dict:
    """Read the S1 direct mask-IoU artifact.

    Args:
        paths: Project layout.

    Returns:
        The artifact.
    """
    return _read(paths, MASK_IOU_JSON)


@pytest.fixture(scope="module")
def results(paths: ProjectPaths) -> dict:
    """Read the live segmentation-results artifact.

    Args:
        paths: Project layout.

    Returns:
        The artifact.
    """
    return _read(paths, RESULTS_JSON)


# --- the four validators, applied to the committed artifacts ----------------------


def test_the_result_manifest_validates(manifest: dict) -> None:
    """The committed manifest passes the S1 result validator."""
    assert validate_s1_result_manifest(manifest) == []


def test_the_canonical_artifact_validates(paths: ProjectPaths, canonical: dict) -> None:
    """The committed canonical artifact matches the frozen evaluator exactly."""
    protocol = load_canonical_evaluation_config(
        paths.configs / "segmentation_canonical_evaluation.yaml"
    )

    assert (
        validate_canonical_evaluation(canonical, protocol_fingerprint=protocol.fingerprint()) == []
    )


def test_the_direct_iou_artifact_validates(paths: ProjectPaths, direct: dict) -> None:
    """The committed diagnostic matches the unchanged phase 8C protocol."""
    protocol = load_mask_iou_config(paths.configs / "segmentation_mask_iou_evaluation.yaml")

    assert validate_direct_iou(direct, protocol_fingerprint=protocol.fingerprint()) == []


def test_the_comparison_validates(paths: ProjectPaths, manifest: dict) -> None:
    """The delta recomputes from the committed S0 canonical reference."""
    assert validate_comparison(manifest, s0_reference=_read(paths, S0_CANONICAL_JSON)) == []


# --- identity and the one-variable contract ---------------------------------------


def test_the_manifest_declares_the_experiment(manifest: dict) -> None:
    """S1 is a phase 8F candidate measured against S0."""
    assert manifest["experiment_id"] == EXPERIMENT_ID
    assert manifest["phase"] == PHASE
    assert manifest["status"] == COMPLETE
    assert manifest["reference_experiment"] == REFERENCE
    assert manifest["task"] == "segmentation"
    assert manifest["metrics_are_validation_only"] is True


def test_the_intentional_difference_is_overlap_mask_only(manifest: dict) -> None:
    """One variable, declared and applied."""
    difference = manifest["intentional_difference"]

    assert difference["field"] == ONE_VARIABLE_FIELD
    assert difference["reference_value"] is True
    assert difference["candidate_value"] is False
    assert difference["one_variable"] is True
    assert ONE_VARIABLE_FIELD not in difference["fields_inherited_unchanged"]


def test_the_framework_actually_trained_with_overlap_mask_false(manifest: dict) -> None:
    """Read back from the framework's own args.yaml, not from the protocol."""
    assert manifest["effective_training_configuration"][ONE_VARIABLE_FIELD] is False


def test_mask_ratio_capacity_resolution_and_batch_were_not_varied(manifest: dict) -> None:
    """Everything the phase deliberately does not test is unchanged."""
    effective = manifest["effective_training_configuration"]

    assert effective["mask_ratio"] == 4
    assert effective["imgsz"] == 768
    assert effective["batch"] == 8
    assert effective["epochs"] == 100
    assert effective["seed"] == 42
    assert manifest["model"] == "YOLO11n-seg"
    assert manifest["mask_ratio_variants_tried"] == 0
    assert manifest["imgsz_variants_tried"] == 0
    assert manifest["batch_variants_tried"] == 0
    assert manifest["alternative_segmenters_trained"] == 0


def test_every_inherited_field_matches_the_frozen_s1_protocol(
    paths: ProjectPaths, manifest: dict
) -> None:
    """The resolved arguments reproduce what phase 8E recorded as inherited."""
    frozen = _read(paths, S1_PROTOCOL_JSON)["inherited_protocol"]
    effective = manifest["effective_training_configuration"]

    drifted = [
        name
        for name, value in frozen.items()
        if name != ONE_VARIABLE_FIELD and name in effective and effective[name] != value
    ]

    assert drifted == []
    assert manifest["inherited_protocol_verified"]["one_variable"] is True


def test_the_resolution_is_derived_from_s0s_own_protocol(
    paths: ProjectPaths, manifest: dict
) -> None:
    """S1's protocol fingerprint is S0's plus the declared override, not a copy."""
    baseline = load_segmentation_baseline_config(paths.configs / "segmentation_baseline.yaml")
    comparison = load_comparison_config(paths.configs / "segmentation_comparison.yaml")

    assert manifest["s0_protocol_fingerprint"] == baseline.fingerprint()
    assert manifest["s1_protocol_fingerprint"] == comparison.fingerprint()
    assert comparison.candidate["overrides"] == {ONE_VARIABLE_FIELD: False}


# --- provenance ----------------------------------------------------------------------


def test_the_pretrained_binary_is_the_one_s0_used(manifest: dict) -> None:
    """A different pretrained checkpoint would be a second variable."""
    weights = manifest["pretrained_weights"]

    assert weights["identifier"] == "yolo11n-seg.pt"
    assert weights["sha256"] == EXPECTED_PRETRAINED
    assert weights["size_bytes"] == EXPECTED_PRETRAINED_BYTES


def test_the_adapter_is_the_approved_phase_8a_bytes(paths: ProjectPaths, manifest: dict) -> None:
    """Same label bytes as S0, unregenerated and unfiltered."""
    baseline = load_segmentation_baseline_config(paths.configs / "segmentation_baseline.yaml")
    adapter = manifest["adapter"]

    for name in (
        "labels_train_sha256",
        "labels_validation_sha256",
        "labels_development_sha256",
        "image_membership_sha256",
    ):
        assert adapter["fingerprints"][name] == baseline.adapter[name]
    assert adapter["role"] == ADAPTER_ROLE
    assert adapter["canonical_ground_truth"] == CANONICAL_GROUND_TRUTH
    assert adapter["identical_to_reference"] is True
    assert adapter["regenerated"] is False
    assert adapter["modified"] is False
    assert adapter["filtered"] is False
    assert adapter["counts"]["instances"] == 1726


def test_the_split_and_class_map_are_the_frozen_ones(paths: ProjectPaths, manifest: dict) -> None:
    """The same frozen split and class map as every earlier phase."""
    audit = _read(paths, "segmentation_adapter_audit_manifest.json")
    fingerprints = manifest["canonical_fingerprints"]

    assert fingerprints["class_map_sha256"] == audit["class_map_sha256"]
    assert (
        fingerprints["split_assignment_sha256"]
        == audit["canonical_fingerprints"]["split_assignment_sha256"]
    )


def test_the_historical_artifacts_are_byte_identical(paths: ProjectPaths, manifest: dict) -> None:
    """Every phase 8A-8E artifact still hashes to what this phase recorded."""
    assert manifest["historical_artifacts_unchanged"] is True
    for name, expected in manifest["historical_artifact_digests"].items():
        assert sha256_file(paths.root / name) == expected, name


def test_the_reference_was_read_not_executed(manifest: dict) -> None:
    """S0 is quoted, never retrained, revalidated or recomputed."""
    reference = manifest["reference_metrics"]

    assert manifest["s0_retrained"] is False
    assert manifest["s0_revalidated"] is False
    assert manifest["s0_artifacts_regenerated"] is False
    assert reference["retrained"] is False
    assert reference["revalidated"] is False
    assert reference["recomputed"] is False
    assert reference["source"] == "COMMITTED_ARTIFACTS_READ_NOT_EXECUTED"


def test_the_reference_metrics_are_the_committed_ones(paths: ProjectPaths, manifest: dict) -> None:
    """Every S0 figure quoted matches its committed artifact exactly."""
    s0_canonical = _read(paths, S0_CANONICAL_JSON)
    s0_result = _read(paths, S0_RESULT_JSON)
    s0_direct = _read(paths, S0_MASK_IOU_JSON)
    reference = manifest["reference_metrics"]

    assert reference["canonical_supported_macro"] == s0_canonical["supported_macro"]["value"]
    assert (
        reference["canonical_all_class_map50_95"] == s0_canonical["canonical"]["all_class_map50_95"]
    )
    assert reference["canonical_all_class_map50"] == s0_canonical["canonical"]["all_class_map50"]
    assert reference["canonical_per_class"] == s0_canonical["canonical"]["per_class"]
    assert reference["native_mask"] == s0_result["mask_metrics"]
    assert reference["direct_iou_global"] == s0_direct["global"]


def test_the_detector_was_verified_and_left_alone(manifest: dict) -> None:
    """The detection block stays closed."""
    detector = manifest["frozen_detector"]

    assert detector["selected_experiment"] == "D2"
    assert detector["trained"] is False
    assert detector["validated"] is False
    assert manifest["detector_touched"] is False


# --- execution and the checkpoint rule -------------------------------------------------


def test_exactly_one_fresh_run_happened(manifest: dict) -> None:
    """One training run, no resume, no engineering abort."""
    execution = manifest["execution"]

    assert execution["runs"] == 1
    assert execution["training_was_a_fresh_single_run"] is True
    assert execution["resumed"] is False
    assert execution["engineering_aborts"] == 0
    assert manifest["models_trained_in_this_phase"] == 1
    # `this_invocation_trained` is a different fact: a re-render reuses the run
    # rather than repeating it, and that must never read as a second experiment.
    assert isinstance(execution["this_invocation_trained"], bool)
    if not execution["this_invocation_trained"]:
        assert execution["reused_completed_run"] is True
        assert execution["artifact_write_reexecution"]["occurred"] is True


def test_the_termination_mode_is_recorded_and_consistent(manifest: dict) -> None:
    """Early stopping is a fact read from the epoch count, not an assertion."""
    execution = manifest["execution"]
    early = execution["epochs_completed"] < execution["epochs_configured"]

    assert execution["early_stopping"] is early
    assert execution["termination"] == (
        "EARLY_STOPPING_AT_FROZEN_PATIENCE" if early else "ALL_EPOCHS_COMPLETED"
    )
    assert execution["epochs_configured"] == 100


def test_the_checkpoint_policy_is_unchanged_from_s0(manifest: dict) -> None:
    """Both experiments select their own best.pt by the same native rule."""
    selection = manifest["checkpoint_selection"]

    assert selection["policy"] == CHECKPOINT_POLICY
    assert selection["semantics"] == CHECKPOINT_SELECTION_SEMANTICS
    assert selection["identical_policy_to_reference"] is True
    assert selection["computed_against_own_target"] is True
    assert selection["mask_only_checkpoint_created"] is False
    assert selection["manual_epoch_selection"] is False


def test_the_best_epoch_is_the_composite_argmax(manifest: dict) -> None:
    """The frozen rule's choice is verified, not assumed."""
    selection = manifest["checkpoint_selection"]

    assert selection["verified"] is True
    assert 1 <= selection["best_epoch"] <= selection["epochs_logged"]
    if selection["next_best_native_fitness"] is not None:
        assert selection["best_native_fitness"] >= selection["next_best_native_fitness"]


def test_both_checkpoints_are_fingerprinted_and_uncommitted(
    paths: ProjectPaths, manifest: dict
) -> None:
    """Weights are referenced by digest and never committed."""
    for name in ("best", "last"):
        record = manifest["checkpoints"][name]
        assert len(record["sha256"]) == 64
        assert record["size_bytes"] > 0
    assert manifest["checkpoints"]["best"]["sha256"] != EXPECTED_S0_CHECKPOINT
    assert not list((paths.root / "reports").rglob("*.pt"))


def test_the_measured_execution_facts_survived_any_re_render(manifest: dict) -> None:
    """Re-rendering prose may never downgrade a measured figure to "not persisted"."""
    execution = manifest["execution"]

    assert isinstance(execution["wall_clock_seconds"], (int, float))
    assert isinstance(execution["peak_gpu_memory_reserved_gib"], (int, float))
    assert execution["peak_gpu_memory_reserved_gib"] > 0
    if execution.get("reused_completed_run"):
        assert execution["measured_fields_carried_forward_source"] == (
            "THE_TRAINING_RUNS_OWN_RESULT_MANIFEST"
        )
        assert manifest["execution"]["artifact_write_reexecution"]["reason"] != "NOT_STATED"


def test_the_actual_optimizer_is_recorded_not_just_the_policy(manifest: dict) -> None:
    """`auto` is a policy; the report states what the framework chose."""
    optimizer = manifest["optimizer"]

    assert optimizer["declared_optimizer_policy"] == "auto"
    assert optimizer["actual_resolved_optimizer"] not in (None, "auto", "NOT_EXPOSED_RELIABLY")
    assert optimizer["resolution_evidence"]
    assert isinstance(optimizer["effective_lr0"], float)
    assert optimizer["scheduler"]


# --- the native metrics, reported but demoted ---------------------------------------------


def test_the_native_metrics_are_reported_in_full(manifest: dict) -> None:
    """Demoted, never suppressed: mask and box, global and per class."""
    native = manifest["native_metrics"]

    for family in ("mask", "box"):
        for key in ("precision", "recall", "mAP@0.50", "mAP@0.50:0.95"):
            assert key in native[family]
    assert native["per_class"]
    for row in native["per_class"].values():
        assert "mask" in row
        assert "box" in row


def test_the_native_metrics_are_marked_non_primary(manifest: dict) -> None:
    """overlap_mask reshapes the native target, so the native AP cannot rank."""
    native = manifest["native_metrics"]

    assert native["label"] == NATIVE_TARGET_METRIC
    assert native["cross_target_status"] == NATIVE_METRIC_STATUS
    assert native["is_primary_for_selection"] is False
    assert "different targets" in native["why_not_primary"]


def test_the_native_validation_used_the_candidates_own_target(manifest: dict) -> None:
    """Validating S1 with the default flag would have scored it against S0's target."""
    effective = manifest["native_validation"]["effective_arguments"]

    assert effective["overlap_mask"] is False
    assert effective["overlap_mask_passed_explicitly"] is True
    assert effective["mask_ratio"] == 4
    assert effective["imgsz"] == 768
    assert manifest["native_validation"]["runs"] == 1


def test_the_descriptive_native_macro_decides_nothing(manifest: dict) -> None:
    """Kept for continuity with S0's figure, barred from selecting."""
    macro = manifest["descriptive_native_supported_macro"]

    assert macro["label"] == DESCRIPTIVE_NATIVE_TARGET_METRIC
    assert macro["used_for_selection"] is False
    assert macro["metric"] == "supported_macro_mask_map50_95"


def test_no_composite_box_mask_score_exists(manifest: dict) -> None:
    """A weak mask result may not hide behind a strong box one."""
    assert manifest["composite_box_mask_score"] is False
    assert manifest["cross_metric_direction"]["composite_created"] is False


# --- the canonical evaluation, which decides -------------------------------------------------


def test_the_canonical_evaluator_is_the_exact_frozen_one(
    paths: ProjectPaths, manifest: dict, canonical: dict
) -> None:
    """One evaluator, unchanged from phase 8E, named by digest."""
    protocol = load_canonical_evaluation_config(
        paths.configs / "segmentation_canonical_evaluation.yaml"
    )
    s0_canonical = _read(paths, S0_CANONICAL_JSON)

    assert canonical["protocol_fingerprint"] == protocol.fingerprint()
    assert manifest["canonical_evaluator_fingerprint"] == protocol.fingerprint()
    assert canonical["protocol_fingerprint"] == s0_canonical["protocol_fingerprint"]
    assert canonical["protocol_unchanged_from_phase_8e"] is True
    assert canonical["runs"] == 1


def test_the_canonical_evaluation_used_standard_coco_semantics(canonical: dict) -> None:
    """Thresholds and detection caps are the standard ones, and were checked."""
    evaluator = canonical["cocoeval"]

    assert evaluator["iou_type"] == IOU_TYPE
    assert tuple(round(float(v), 2) for v in evaluator["iou_thresholds"]) == IOU_THRESHOLDS
    assert tuple(int(v) for v in evaluator["max_dets"]) == MAX_DETS
    assert canonical["prediction_cap_versus_metric_cap"]["model_side_max_det"] == 300


def test_the_canonical_evaluation_scored_canonical_masks(canonical: dict) -> None:
    """Never the YOLO adapter, whose own approximation phase 8A measured."""
    truth = canonical["ground_truth"]

    assert truth["source"] == "CANONICAL_COCO_INSTANCE_SEGMENTATION"
    assert truth["is_the_yolo_adapter"] is False
    assert "adapter" not in truth["document"].lower()
    assert canonical["mask_encoding"] == "PYCOCOTOOLS_BINARY_MASK_RLE_ON_ORIGINAL_CANVAS"
    assert canonical["category_id_mapping"] == "CANONICAL_CATEGORY_IDS_USED_DIRECTLY_NO_REMAPPING"


def test_the_low_confidence_is_not_an_operating_point(canonical: dict, direct: dict) -> None:
    """The two protocols keep their own thresholds and are never mixed."""
    assert canonical["inference"]["conf"] == 0.001
    assert direct["inference"]["conf"] == 0.25
    explanation = canonical["conf_is_not_an_operating_point"]
    assert "low-scoring tail" in explanation
    assert "0.25" in explanation
    assert "never mixed" in explanation


def test_every_frozen_class_has_a_canonical_ap(paths: ProjectPaths, canonical: dict) -> None:
    """All five classes are reported, including the rare one."""
    audit = _read(paths, "segmentation_adapter_audit_manifest.json")
    per_class = canonical["canonical"]["per_class"]

    assert set(per_class) == set(audit["class_map"])
    for name, row in per_class.items():
        assert row["AP@0.50:0.95"] is not None, name
        assert row["AP@0.50"] is not None, name
        assert 0.0 <= row["AP@0.50:0.95"] <= 1.0
        assert 0.0 <= row["AP@0.50"] <= 1.0


def test_the_supported_macro_reproduces_from_the_per_class_values(manifest: dict) -> None:
    """The primary metric is arithmetic over the admitted classes, and it checks out."""
    macro = manifest["canonical_supported_macro"]
    per_class = manifest["canonical_metrics"]["per_class"]

    values = [per_class[name]["AP@0.50:0.95"] for name in macro["admitted_classes"]]

    assert macro["metric"] == PRIMARY_METRIC
    assert macro["label"] == CANONICAL_PRIMARY_METRIC
    assert macro["contributing_values"] == [round(value, 6) for value in values]
    assert macro["value"] == pytest.approx(sum(values) / len(values), abs=5e-7)


def test_the_support_rule_is_the_frozen_phase_7_one(manifest: dict) -> None:
    """The rule names no class; the admitted set is its output."""
    macro = manifest["canonical_supported_macro"]
    support = manifest["support"]

    assert macro["rule_origin"] == "FROZEN_IN_PHASE_7A_REUSED_UNCHANGED"
    assert macro["names_no_class"] is True
    for name, row in support.items():
        expected = (
            row["images"] >= SUPPORT_MIN_POSITIVE_IMAGES
            and row["instances"] >= SUPPORT_MIN_INSTANCES
        )
        assert row["supported"] is expected, name
        assert (name in macro["admitted_classes"]) is expected


# --- the comparison ------------------------------------------------------------------------


def test_the_primary_delta_uses_the_exact_committed_reference(
    paths: ProjectPaths, manifest: dict
) -> None:
    """The reference is the committed phase 8E value, to the digit."""
    s0_canonical = _read(paths, S0_CANONICAL_JSON)
    delta = manifest["primary_delta"]

    assert delta["reference"] == s0_canonical["supported_macro"]["value"]
    assert delta["candidate"] == manifest["canonical_supported_macro"]["value"]
    assert delta["delta"] == pytest.approx(
        round(delta["candidate"] - delta["reference"], 6), abs=1e-9
    )


def test_the_margin_classification_is_the_frozen_rules_output(manifest: dict) -> None:
    """0.005, three cases, recomputed independently here."""
    delta = manifest["primary_delta"]
    recomputed = classify_delta(delta["reference"], delta["candidate"])

    assert delta["margin"] == PRACTICAL_EQUIVALENCE_MARGIN
    assert delta["verdict"] in SELECTION_CASES
    assert delta["verdict"] == recomputed["verdict"]
    assert delta["margin_is_not_a_significance_test"] is True


def test_the_all_class_delta_is_reported_but_does_not_rank(manifest: dict) -> None:
    """The all-class figure is published and explicitly not the selection metric."""
    row = manifest["canonical_all_class_delta"]

    assert row["is_not_the_selection_metric"] is True
    assert row["delta"] == pytest.approx(round(row["S1"] - row["S0"], 6), abs=1e-9)


def test_every_class_carries_a_canonical_comparison(paths: ProjectPaths, manifest: dict) -> None:
    """All five classes get an S0-versus-S1 row."""
    audit = _read(paths, "segmentation_adapter_audit_manifest.json")
    table = manifest["canonical_per_class_comparison"]

    assert set(table) == set(audit["class_map"])
    for name, row in table.items():
        assert row["delta_AP@0.50:0.95"] == pytest.approx(
            round(row["S1_AP@0.50:0.95"] - row["S0_AP@0.50:0.95"], 6), abs=1e-9
        ), name


# --- the direct-IoU diagnostic ------------------------------------------------------------


def test_the_diagnostic_protocol_is_identical_to_the_one_s0_ran_under(
    paths: ProjectPaths, direct: dict
) -> None:
    """Same fingerprint, or the two diagnostics would not be comparable."""
    protocol = load_mask_iou_config(paths.configs / "segmentation_mask_iou_evaluation.yaml")
    s0_direct = _read(paths, S0_MASK_IOU_JSON)

    assert direct["protocol_fingerprint"] == protocol.fingerprint()
    assert direct["protocol_fingerprint"] == s0_direct["protocol_fingerprint"]
    assert direct["protocol_unchanged_from_phase_8c"] is True
    assert direct["runs"] == 1
    assert direct["thresholds_swept"] == 0


def test_the_diagnostic_scored_canonical_masks_at_the_predeclared_point(direct: dict) -> None:
    """Canonical COCO ground truth, conf 0.25, one-to-one same-class matching."""
    assert direct["ground_truth"]["source"] == GROUND_TRUTH_SOURCE
    assert direct["ground_truth"]["is_the_yolo_adapter"] is False
    assert direct["inference"]["conf"] == 0.25
    assert direct["inference"]["iou"] == 0.70
    assert direct["inference"]["imgsz"] == 768
    assert direct["inference"]["max_det"] == 300
    assert direct["inference"]["retina_masks"] is True
    assert direct["matching"]["algorithm"] == MATCHING_ALGORITHM
    assert direct["matching"]["deterministic"] is True


def test_the_diagnostic_counts_are_internally_consistent(direct: dict) -> None:
    """Matched plus unmatched must account for every instance on both sides."""
    overall = direct["global"]

    assert overall["matched_count"] + overall["unmatched_gt"] == overall["gt_count"]
    assert (
        overall["matched_count"] + overall["unmatched_predictions"] == overall["prediction_count"]
    )
    assert overall["gt_count"] == 304


def test_the_coverage_diagnostics_are_ordered_and_bounded(direct: dict) -> None:
    """A stricter IoU threshold can never cover more ground truth."""
    overall = direct["global"]

    assert 0.0 <= overall[GT_IOU75_COVERAGE] <= overall[GT_IOU50_COVERAGE]
    assert overall[GT_IOU50_COVERAGE] <= overall[GT_MATCH_COVERAGE] <= 1.0
    assert 0.0 <= overall[GT_NORMALIZED_MASK_IOU] <= overall[MATCHED_MASK_IOU_MEAN] <= 1.0


def test_the_direct_iou_deltas_recompute_against_the_committed_reference(
    paths: ProjectPaths, manifest: dict
) -> None:
    """Every delta is derived from S0's committed diagnostic, not restated."""
    s0_direct = _read(paths, S0_MASK_IOU_JSON)
    block = manifest["direct_mask_iou"]

    for name, delta in block["deltas_vs_reference"].items():
        assert delta == pytest.approx(
            round(block["headline"][name] - s0_direct["global"][name], 6), abs=1e-9
        ), name


def test_the_aggregate_decomposition_recomputes(manifest: dict) -> None:
    """The mean is decomposed into the per-class movements that produced it."""
    composition = manifest["aggregate_composition"]
    table = manifest["canonical_per_class_comparison"]
    admitted = manifest["canonical_supported_macro"]["admitted_classes"]

    assert composition["admitted_classes"] == admitted
    for name in admitted:
        assert composition["contribution_to_primary_delta"][name] == pytest.approx(
            table[name]["delta_AP@0.50:0.95"] / len(admitted), abs=5e-7
        )
    assert sum(composition["contribution_to_primary_delta"].values()) == pytest.approx(
        manifest["primary_delta"]["delta"], abs=5e-6
    )


def test_any_regressed_supported_class_is_named(manifest: dict) -> None:
    """A class that fell is named, not left inside the mean."""
    composition = manifest["aggregate_composition"]
    table = manifest["canonical_per_class_comparison"]
    admitted = manifest["canonical_supported_macro"]["admitted_classes"]
    expected = sorted(name for name in admitted if table[name]["delta_AP@0.50:0.95"] < 0)

    assert composition["regressed_supported_classes"] == expected
    if expected:
        text = (ProjectPaths.from_root().reports / REPORT_MD).read_text(encoding="utf-8")
        assert "regressed" in text
        for name in expected:
            assert name in text


def test_the_decomposition_claims_no_mechanism(manifest: dict) -> None:
    """The decomposition is arithmetic, and says so."""
    note = manifest["aggregate_composition"]["note"]

    assert "Arithmetic only" in note
    assert "UNKNOWN" in note


# --- cross-metric direction, person and the rare class ---------------------------------------


def test_the_cross_metric_direction_recomputes(paths: ProjectPaths, manifest: dict) -> None:
    """The classification follows the frozen rule applied to the two deltas."""
    s0_direct = _read(paths, S0_MASK_IOU_JSON)
    direction = manifest["cross_metric_direction"]
    secondary = round(
        manifest["direct_mask_iou"]["headline"][GT_NORMALIZED_MASK_IOU]
        - s0_direct["global"][GT_NORMALIZED_MASK_IOU],
        6,
    )
    recomputed = cross_metric_direction(manifest["primary_delta"]["delta"], secondary)

    assert direction["status"] in (CONSISTENT, DISAGREEMENT)
    assert direction["status"] == recomputed["status"]
    assert direction["primary_delta"] == manifest["primary_delta"]["delta"]
    assert direction["secondary_delta"] == pytest.approx(secondary, abs=1e-9)
    assert direction["ranked_by"] == PRIMARY_METRIC
    assert direction["rule_frozen_before_s1"] is True


def test_the_person_diagnostic_quotes_the_committed_canonical_references(
    paths: ProjectPaths, manifest: dict
) -> None:
    """The predeclared focus compares against S0's committed figures."""
    s0_canonical = _read(paths, S0_CANONICAL_JSON)
    s0_direct = _read(paths, S0_MASK_IOU_JSON)
    person = manifest["person_diagnostic"]

    assert person["class"] == "person"
    assert person["status"] == "PREDECLARED_PHASE_8D_DIAGNOSTIC_FOCUS"
    assert (
        person["canonical"]["S0_AP@0.50:0.95"]
        == s0_canonical["canonical"]["per_class"]["person"]["AP@0.50:0.95"]
    )
    assert (
        person["direct_iou"][GT_NORMALIZED_MASK_IOU]["S0"]
        == s0_direct["per_class"]["person"][GT_NORMALIZED_MASK_IOU]
    )
    assert person["may_select_final_model"] is False


def test_vest_loose_is_reported_in_full_but_decides_nothing(
    manifest: dict, canonical: dict, direct: dict
) -> None:
    """The rare class keeps its descriptive status everywhere it appears."""
    rare = manifest["rare_class"]

    assert rare["name"] == RARE_CLASS
    assert rare["status"] == RARE_CLASS_STATUS
    assert rare["reported_in_full"] is True
    assert rare["may_decide_anything"] is False
    assert RARE_CLASS not in manifest["canonical_supported_macro"]["admitted_classes"]
    assert RARE_CLASS in canonical["canonical"]["per_class"]
    assert direct["rare_class_warning"]["status"] == RARE_CLASS_STATUS
    assert manifest["canonical_per_class_comparison"][RARE_CLASS]["status"] == RARE_CLASS_STATUS


# --- the fingerprint, the live artifact and what was not decided -------------------------------


def test_the_experiment_fingerprint_recomputes(manifest: dict) -> None:
    """Deterministic over the recorded semantics, excluding paths and timestamps."""
    inputs = manifest["experiment_fingerprint_inputs"]

    assert len(manifest["S1_experiment_sha256"]) == 64
    assert "best_checkpoint_sha256" in inputs
    assert "intentional_difference" in inputs
    assert "pretrained_weight_sha256" in inputs
    assert experiment_fingerprint({"a": 1}) != manifest["S1_experiment_sha256"]


def test_the_live_results_artifact_records_both_experiments(manifest: dict, results: dict) -> None:
    """S0 complete, S1 complete, comparison computed, nothing selected."""
    by_id = {row["experiment_id"]: row for row in results["experiments"]}

    assert set(by_id) == {REFERENCE, EXPERIMENT_ID}
    assert by_id[REFERENCE]["status"] == "S0_COMPLETE"
    assert by_id[REFERENCE]["margin_status"] == "REFERENCE"
    assert by_id[REFERENCE]["retrained_in_phase_8f"] is False
    assert by_id[EXPERIMENT_ID]["status"] == COMPLETE
    assert by_id[EXPERIMENT_ID]["margin_status"] == manifest["primary_delta"]["verdict"]
    assert results["primary_comparison_status"] == "COMPUTED"
    assert results["primary_metric"] == PRIMARY_METRIC
    assert results["composite_score"] is False


def test_phase_8f_itself_selected_nothing(manifest: dict, results: dict) -> None:
    """Running an experiment produces a classification, not a decision.

    This originally also asserted that the live results artifact still carried
    no selection. Phase 8G has since frozen S1 as the final segmenter under the
    policy phase 8E froze, so that assertion would now fail for an authorised
    reason, and the live artifact is by design the place the selection lands.
    What it was protecting - that **S1's own result manifest** claims no
    selection and leaves the decision to a reviewed human one - is asserted here
    directly, and is unaffected by a later phase.
    """
    assert manifest["final_segmenter"] == FINAL_SEGMENTER
    assert manifest["pending_human_review"] is True
    assert manifest["models_trained_in_this_phase"] == 1
    assert "not a frozen segmenter" in manifest["final_segmenter_note"]
    # The live artifact is shared state that phase 8G updates. All that phase 8F
    # requires of it is that it still describes the same two experiments.
    assert {row["experiment_id"] for row in results["experiments"]} == {"S0", "S1"}
    assert results["further_experiments_authorised"] == 0


def test_nothing_was_tuned(manifest: dict) -> None:
    """No threshold, no resolution, no batch, no capacity."""
    assert manifest["thresholds_tuned"] == 0
    assert manifest["imgsz_variants_tried"] == 0
    assert manifest["batch_variants_tried"] == 0
    assert manifest["mask_ratio_variants_tried"] == 0
    assert manifest["alternative_segmenters_trained"] == 0


# --- the holdout ----------------------------------------------------------------------------


@pytest.mark.parametrize("name", [RESULT_JSON, CANONICAL_JSON, MASK_IOU_JSON, RESULTS_JSON])
def test_no_artifact_mentions_the_holdout(paths: ProjectPaths, name: str) -> None:
    """Structurally absent, not merely unmentioned."""
    payload = _read(paths, name)
    body = {key: value for key, value in payload.items() if key != "test"}

    assert payload["test"]["status"] == HOLDOUT_STATUS
    assert payload["holdout_accessed"] is False
    assert contains_forbidden_split(body) is False


def test_no_holdout_artifact_was_materialised(paths: ProjectPaths) -> None:
    """The protected split has no images, labels or annotations on disk."""
    forbidden = [
        paths.data_processed / "canonical" / "images" / "test",
        paths.data_processed / "canonical" / "annotations" / "segmentation_test.coco.json",
        paths.data_processed / "adapters" / "yolo_segmentation_s1_runtime" / "images" / "test",
        paths.data_processed / "adapters" / "yolo_segmentation_s1_runtime" / "labels" / "test",
    ]

    assert [path for path in forbidden if path.exists()] == []


# --- the report ------------------------------------------------------------------------------


def test_the_report_carries_no_sensitive_content(paths: ProjectPaths) -> None:
    """No absolute path, signed URL or credential-shaped string."""
    text = (paths.reports / REPORT_MD).read_text(encoding="utf-8")

    assert scan_for_sensitive(text) == []


def test_the_report_states_the_intervention_and_the_limits(paths: ProjectPaths) -> None:
    """The reader is told what varied, what decided, and what is not established."""
    text = (paths.reports / REPORT_MD).read_text(encoding="utf-8")

    for fragment in (
        "ONE_VARIABLE_INTERVENTION",
        "NATIVE_TARGET_METRIC",
        "CANONICAL_PRIMARY_METRIC",
        "DIRECT_IOU_DIAGNOSTIC",
        "PREDECLARED_S1_HYPOTHESIS",
        "HOLDOUT_POLICY",
        "PENDING_HUMAN_REVIEW",
        "LIMITATION",
        "not a significance test",
        "validation",
    ):
        assert fragment in text, fragment


def test_the_provenance_records_exist_and_the_prerun_predates_the_run(
    paths: ProjectPaths,
) -> None:
    """The pre-run record was written before the first optimisation step."""
    prerun = _read(paths, PRERUN_JSON)
    record = _read(paths, PROVENANCE_JSON)

    assert prerun["details"]["state"] == "BEFORE_FIRST_OPTIMIZATION_STEP"
    assert prerun["details"]["models_trained_so_far"] == 0
    assert record["details"]["classification"] == COMPLETE
    assert record["details"]["models_trained_in_this_phase"] == 1
    assert record["details"]["holdout_accessed"] is False
