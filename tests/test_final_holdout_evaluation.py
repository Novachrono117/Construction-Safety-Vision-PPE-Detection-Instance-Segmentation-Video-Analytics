"""Tests for the frozen phase 11A holdout evaluation protocol.

Every test here uses the committed protocol document and synthetic fixtures.
**No real holdout data is opened**, no model is loaded, and the dual-gate tests
pass a synthetic environment mapping rather than mutating the process
environment - so running this suite can never unlock anything.
"""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from construction_safety_vision.final_holdout_evaluation import (
    ACCESS_PURPOSE,
    AP_CONF,
    CLASSES,
    CONFUSION_MATRIX_CONF,
    CONFUSION_MATRIX_IOU,
    DETECTOR_CHECKPOINT_SHA256,
    DETECTOR_EXPERIMENT,
    ENVIRONMENT_GATE,
    EXAMPLES_PER_CATEGORY,
    FAILURE_STATES,
    FINAL_SEGMENTER_SHA256,
    FROZEN_NOT_EXECUTED,
    HOLDOUT_STATUS,
    IMGSZ,
    LEDGER_STATES,
    MAX_DET,
    MAX_DETS,
    NMS_IOU,
    OBJECT_MATCH_IOU,
    OPERATIONAL_CONF,
    PHASE,
    PRECISION,
    QUALITATIVE_CATEGORIES,
    REBUILD_FROM_PREDICTIONS_ON,
    RUNNER,
    SEGMENTER_CHECKPOINT_SHA256,
    SEGMENTER_EXPERIMENT,
    TEST_ANNOTATIONS,
    TEST_IMAGES,
    TIE_BREAKING,
    HoldoutProtocolError,
    LedgerTransitionError,
    OneShotLedger,
    QualitativeCandidate,
    authorize_final_holdout_access,
    holdout_is_locked,
    inference_rerun_permitted,
    load_protocol,
    phase_11b_steps,
    protocol_fingerprint,
    recovery_action,
    select_qualitative_examples,
    validate_protocol,
)
from construction_safety_vision.paths import ProjectPaths
from construction_safety_vision.splits import HoldoutViolationError

PATHS = ProjectPaths.from_root()
ROOT = PATHS.root
CONFIG = ROOT / "configs" / "final_holdout_evaluation.yaml"

MODULE = ROOT / "src" / "construction_safety_vision" / "final_holdout_evaluation.py"
FREEZE_SCRIPT = ROOT / "scripts" / "freeze_final_holdout_evaluation.py"
RUNNER_SCRIPT = ROOT / "scripts" / "evaluate_final_holdout.py"

UNLOCKED = {ENVIRONMENT_GATE: "1"}
"""A synthetic environment. The process environment is never mutated."""

LOCKED: dict[str, str] = {}


@pytest.fixture(scope="module")
def protocol() -> Any:
    return load_protocol(CONFIG)


@pytest.fixture(scope="module")
def raw(protocol: Any) -> dict[str, Any]:
    return protocol.raw


# --- identity ------------------------------------------------------------------------------


def test_exact_detector_identity(raw: dict[str, Any]) -> None:
    detector = raw["detector"]
    assert detector["experiment"] == DETECTOR_EXPERIMENT == "D2"
    assert detector["model"] == "YOLO11n"
    assert detector["imgsz"] == IMGSZ == 768
    assert detector["checkpoint_sha256"] == DETECTOR_CHECKPOINT_SHA256
    assert (
        detector["checkpoint_sha256"]
        == "0466f872a9de22898c70d8834cf1fcfb3d77f6c9fb27e81ee6248d3d19cbc206"
    )


def test_exact_segmenter_identity(raw: dict[str, Any]) -> None:
    segmenter = raw["segmenter"]
    assert segmenter["experiment"] == SEGMENTER_EXPERIMENT == "S1"
    assert segmenter["model"] == "YOLO11n-seg"
    assert segmenter["imgsz"] == IMGSZ == 768
    assert segmenter["mask_ratio"] == 4
    assert segmenter["overlap_mask"] is False
    assert segmenter["checkpoint_sha256"] == SEGMENTER_CHECKPOINT_SHA256
    assert (
        segmenter["checkpoint_sha256"]
        == "29337d671459d0f742fb713613cc41d0eafcb8a21f5846ac0da65b2ffc024f20"
    )
    assert segmenter["final_segmenter_sha256"] == FINAL_SEGMENTER_SHA256


def test_no_alternate_checkpoint_or_retraining(raw: dict[str, Any]) -> None:
    assert raw["alternate_checkpoint_permitted"] is False
    assert raw["retraining_to_replace_a_missing_binary_permitted"] is False


# --- status --------------------------------------------------------------------------------


def test_protocol_status_is_frozen_not_executed(raw: dict[str, Any]) -> None:
    assert raw["status"] == FROZEN_NOT_EXECUTED == "FROZEN_NOT_EXECUTED"
    assert raw["phase"] == PHASE == "11A"
    assert raw["executes_in_phase"] == "11B"
    assert raw["test_policy"] == HOLDOUT_STATUS


def test_the_committed_protocol_validates(raw: dict[str, Any]) -> None:
    assert validate_protocol(raw) == []


# --- the dual gate -------------------------------------------------------------------------


def test_dual_gate_requirement_is_declared(raw: dict[str, Any]) -> None:
    auth = raw["authorization"]
    assert auth["policy"] == "DUAL_INDEPENDENT_OPT_IN"
    assert auth["either_alone_is_refused"] is True
    assert auth["code_opt_in"] == "allow_test=true"
    assert auth["environment_opt_in"] == f"{ENVIRONMENT_GATE}=1"
    assert auth["activated_in_this_phase"] is False


def test_code_gate_alone_is_refused() -> None:
    """allow_test without the environment variable unlocks nothing."""
    with pytest.raises(HoldoutViolationError):
        authorize_final_holdout_access(
            purpose=ACCESS_PURPOSE, caller=RUNNER, allow_test=True, env=LOCKED
        )


def test_environment_gate_alone_is_refused() -> None:
    """The environment variable without allow_test unlocks nothing."""
    with pytest.raises(HoldoutViolationError):
        authorize_final_holdout_access(
            purpose=ACCESS_PURPOSE, caller=RUNNER, allow_test=False, env=UNLOCKED
        )


def test_both_gates_together_authorise_in_a_synthetic_environment() -> None:
    """Both opt-ins, against a synthetic env mapping - no real data is touched."""
    resolved = authorize_final_holdout_access(
        purpose=ACCESS_PURPOSE, caller=RUNNER, allow_test=True, env=UNLOCKED
    )
    assert resolved.value == "test"


def test_generic_development_caller_is_refused_even_with_both_gates() -> None:
    with pytest.raises(HoldoutProtocolError):
        authorize_final_holdout_access(
            purpose=ACCESS_PURPOSE,
            caller="scripts/train_detection_experiment.py",
            allow_test=True,
            env=UNLOCKED,
        )


def test_a_development_purpose_is_refused_even_with_both_gates() -> None:
    with pytest.raises(HoldoutProtocolError):
        authorize_final_holdout_access(
            purpose="quick look at the test set",
            caller=RUNNER,
            allow_test=True,
            env=UNLOCKED,
        )


def test_no_automatic_environment_mutation_anywhere() -> None:
    """No file in this phase may set, export or modify the gate variable."""
    for path in (MODULE, FREEZE_SCRIPT, RUNNER_SCRIPT):
        text = path.read_text(encoding="utf-8")
        for forbidden in (
            "os.environ[",
            "os.putenv",
            "setdefault(",
            "environ.update",
            f'"{ENVIRONMENT_GATE}"] =',
            "monkeypatch.setenv",
        ):
            assert forbidden not in text, (path.name, forbidden)


def test_the_holdout_is_locked_in_this_process() -> None:
    """Locked, unless the one-shot evaluation has been executed.

    Phase 11B is the one authorised condition under which a person deliberately
    sets the gate, and its provenance record is the committed evidence that it
    happened. Before that record exists the gate must be absent, which is what
    catches an unlock left set by accident during development.
    """
    from construction_safety_vision.paths import ProjectPaths

    executed = (
        ProjectPaths.from_root().reports / "final_test_evaluation.provenance.json"
    ).is_file()
    assert holdout_is_locked() is True or executed


# --- inference settings --------------------------------------------------------------------


@pytest.mark.parametrize("block", ["detector_inference", "segmenter_inference"])
def test_frozen_inference_settings(raw: dict[str, Any], block: str) -> None:
    settings = raw[block]
    assert settings["imgsz"] == 768
    assert settings["conf"] == AP_CONF == 0.001
    assert settings["iou"] == NMS_IOU == 0.70
    assert settings["max_det"] == MAX_DET == 300
    assert settings["precision"] == PRECISION == "FP32"
    assert settings["quantize"] == 32
    assert settings["augment"] is False
    assert settings["tta"] is False
    assert settings["half"] is False
    assert settings["threshold_sweep_permitted"] is False
    assert settings["confidence_tuning_on_test_permitted"] is False


def test_segmenter_keeps_original_canvas_masks(raw: dict[str, Any]) -> None:
    settings = raw["segmenter_inference"]
    assert settings["retina_masks"] is True
    assert settings["mask_reconstruction_change_permitted"] is False


# --- evaluators ----------------------------------------------------------------------------


def test_canonical_bbox_evaluator(raw: dict[str, Any]) -> None:
    block = raw["canonical_bbox_evaluator"]
    assert block["implementation"] == "pycocotools.cocoeval.COCOeval"
    assert block["iou_type"] == "bbox"
    assert tuple(block["max_dets"]) == MAX_DETS == (1, 10, 100)
    assert block["iou_thresholds"][0] == 0.50
    assert block["iou_thresholds"][-1] == 0.95
    assert len(block["iou_thresholds"]) == 10
    assert block["boxes_from_masks"] is False


def test_canonical_segm_evaluator(raw: dict[str, Any]) -> None:
    block = raw["canonical_segm_evaluator"]
    assert block["implementation"] == "pycocotools.cocoeval.COCOeval"
    assert block["iou_type"] == "segm"
    assert tuple(block["max_dets"]) == MAX_DETS
    assert block["mask_encoding"] == "PYCOCOTOOLS_BINARY_MASK_RLE"


def test_segmenter_boxes_go_through_the_same_bbox_evaluator(raw: dict[str, Any]) -> None:
    metrics = raw["segmenter_metrics"]
    assert metrics["box_metrics_evaluator"] == "canonical_bbox_evaluator"
    assert metrics["box_metrics_derived_from_masks"] is False
    assert "S1_CANONICAL_TEST_BOX_MAP50_95" in metrics["box_metrics"]


def test_direct_iou_protocol_identity(raw: dict[str, Any]) -> None:
    block = raw["direct_iou"]
    assert (
        block["protocol_fingerprint"]
        == "b912039ca77b36959f74fcdbaed109dbf5e3bd707709556f95a19085c7f34d80"
    )
    assert block["unchanged_from_phase_8c"] is True
    assert block["new_rule_invented"] is False
    assert block["role"] == "SECONDARY_CANONICAL_DIAGNOSTIC"
    assert block["is_primary_segmenter_metric"] is False
    assert block["inference"]["conf"] == OPERATIONAL_CONF == 0.25


def test_the_two_confidences_are_never_the_same_value() -> None:
    assert AP_CONF != OPERATIONAL_CONF


# --- confusion matrix ----------------------------------------------------------------------


def test_confusion_matrix_settings_are_frozen(raw: dict[str, Any]) -> None:
    block = raw["confusion_matrix"]
    assert block["conf"] == CONFUSION_MATRIX_CONF == 0.25
    assert block["iou_threshold"] == CONFUSION_MATRIX_IOU == 0.45
    assert block["implementation"] == "ultralytics.utils.metrics.ConfusionMatrix"
    assert block["orientation"] == "ROWS_ARE_PREDICTED_COLUMNS_ARE_GROUND_TRUTH"
    assert block["matrix_dimension"] == len(CLASSES) + 1
    assert tuple(block["class_order"]) == CLASSES
    assert block["threshold_chosen_after_seeing_test"] is False


def test_confusion_matrix_matches_the_installed_framework_defaults() -> None:
    """The frozen values are the framework's, not this project's invention."""
    import inspect

    from ultralytics.utils.metrics import ConfusionMatrix

    signature = inspect.signature(ConfusionMatrix.process_batch)
    assert signature.parameters["iou_thres"].default == CONFUSION_MATRIX_IOU
    matrix = ConfusionMatrix(names=dict(enumerate(CLASSES)))
    assert matrix.matrix.shape == (len(CLASSES) + 1, len(CLASSES) + 1)


# --- object-level FP/FN --------------------------------------------------------------------


def test_object_level_matching_semantics(raw: dict[str, Any]) -> None:
    block = raw["object_level_matching"]
    assert block["class_aware"] is True
    assert block["iou_threshold"] == OBJECT_MATCH_IOU == 0.50
    assert block["assignment"] == "ONE_TO_ONE_PER_IMAGE_PER_CLASS"
    assert block["confidence_threshold"] == OPERATIONAL_CONF
    assert block["redefinition_after_seeing_test_permitted"] is False
    assert set(block["definitions"]) == {"true_positive", "false_positive", "false_negative"}


def test_segmentation_failure_taxonomy_is_predeclared(raw: dict[str, Any]) -> None:
    taxonomy = raw["object_level_matching"]["segmentation_failure_taxonomy"]
    for name in (
        "DETECTION_MISS",
        "CLASSIFICATION_MISMATCH",
        "LOCALIZATION_FAILURE",
        "MASK_QUALITY_FAILURE",
    ):
        assert taxonomy[name]
    assert taxonomy["first_matching_category_wins"] is True


# --- deterministic qualitative selection ---------------------------------------------------


def test_qualitative_selection_is_declared_deterministic(raw: dict[str, Any]) -> None:
    block = raw["qualitative_selection"]
    assert block["deterministic"] is True
    assert block["random_sampling_permitted"] is False
    assert block["manual_cherry_picking_permitted"] is False
    assert block["images_inspected_to_design_this_rule"] == 0
    assert block["examples_per_category"] == EXAMPLES_PER_CATEGORY == 3
    assert tuple(block["categories"]) == QUALITATIVE_CATEGORIES
    assert tuple(block["tie_breaking"]) == TIE_BREAKING


def test_deterministic_ranking_is_stable_under_input_order() -> None:
    """The same candidates in any order produce the same gallery."""
    candidates = [
        QualitativeCandidate("HIGHEST_CONFIDENCE_FALSE_POSITIVE", 0.9, -1, 3, "img_c"),
        QualitativeCandidate("HIGHEST_CONFIDENCE_FALSE_POSITIVE", 0.5, -1, 1, "img_a"),
        QualitativeCandidate("HIGHEST_CONFIDENCE_FALSE_POSITIVE", 0.7, -1, 2, "img_b"),
        QualitativeCandidate("HIGHEST_CONFIDENCE_FALSE_POSITIVE", 0.4, -1, 4, "img_d"),
    ]
    descending = dict.fromkeys(QUALITATIVE_CATEGORIES, True)
    first = select_qualitative_examples(candidates, descending=descending)
    second = select_qualitative_examples(list(reversed(candidates)), descending=descending)
    assert first == second
    chosen = [c.rank_value for c in first["HIGHEST_CONFIDENCE_FALSE_POSITIVE"]]
    assert chosen == [0.9, 0.7, 0.5]


def test_tie_breaking_resolves_equal_rank_values_deterministically() -> None:
    tied = [
        QualitativeCandidate("LOWEST_IOU_MATCHED_INSTANCE", 0.3, 20, 1, "img_z"),
        QualitativeCandidate("LOWEST_IOU_MATCHED_INSTANCE", 0.3, 10, 2, "img_a"),
        QualitativeCandidate("LOWEST_IOU_MATCHED_INSTANCE", 0.3, 15, 3, "img_m"),
    ]
    descending = dict.fromkeys(QUALITATIVE_CATEGORIES, False)
    chosen = select_qualitative_examples(tied, descending=descending)
    ids = [c.canonical_annotation_id for c in chosen["LOWEST_IOU_MATCHED_INSTANCE"]]
    assert ids == [10, 15, 20]


def test_one_instance_never_appears_in_two_categories() -> None:
    shared = (7, 2)
    candidates = [
        QualitativeCandidate("HIGHEST_CONFIDENCE_FALSE_POSITIVE", 0.9, *shared, "img_a"),
        QualitativeCandidate("LOWEST_IOU_MATCHED_INSTANCE", 0.1, *shared, "img_a"),
    ]
    descending = {
        "HIGHEST_CONFIDENCE_FALSE_POSITIVE": True,
        "HIGHEST_CONFIDENCE_CLASSIFICATION_MISMATCH": True,
        "LOWEST_IOU_MATCHED_INSTANCE": False,
        "LARGEST_MISSED_INSTANCE": True,
        "SEGMENTATION_UNDER_COVERAGE": False,
        "SEGMENTATION_OVER_COVERAGE": True,
    }
    chosen = select_qualitative_examples(candidates, descending=descending)
    assert len(chosen["HIGHEST_CONFIDENCE_FALSE_POSITIVE"]) == 1
    assert chosen["LOWEST_IOU_MATCHED_INSTANCE"] == []


def test_an_undeclared_category_is_refused() -> None:
    with pytest.raises(HoldoutProtocolError):
        select_qualitative_examples(
            [QualitativeCandidate("MOST_INTERESTING_LOOKING", 1.0, 1, 1, "img")],
            descending=dict.fromkeys(QUALITATIVE_CATEGORIES, True),
        )


# --- the one-shot ledger -------------------------------------------------------------------


def test_ledger_states_are_frozen(raw: dict[str, Any]) -> None:
    assert tuple(raw["one_shot_ledger"]["states"]) == LEDGER_STATES


def test_ledger_walks_the_happy_path() -> None:
    ledger = OneShotLedger()
    for state in LEDGER_STATES[1:-2]:
        ledger.advance(state)
    ledger.advance("COMPLETE")
    assert ledger.state == "COMPLETE"
    assert ledger.attempt == 1


def test_ledger_refuses_an_illegal_transition() -> None:
    ledger = OneShotLedger()
    with pytest.raises(LedgerTransitionError):
        ledger.advance("METRICS_COMPUTED")


def test_ledger_is_terminal_once_complete() -> None:
    ledger = OneShotLedger()
    for state in LEDGER_STATES[1:-2]:
        ledger.advance(state)
    ledger.advance("COMPLETE")
    with pytest.raises(LedgerTransitionError):
        ledger.advance("ARTIFACTS_WRITTEN")


def test_attempt_counter_cannot_be_reset_or_go_below_one() -> None:
    with pytest.raises(LedgerTransitionError):
        OneShotLedger(attempt=0)


def test_a_repeat_attempt_requires_a_written_justification() -> None:
    with pytest.raises(LedgerTransitionError):
        OneShotLedger(attempt=2)
    ledger = OneShotLedger(attempt=2, justification="infrastructure failure, human-authorised")
    assert ledger.attempt == 2


def test_ledger_history_is_append_only() -> None:
    ledger = OneShotLedger()
    ledger.advance("AUTHORIZED", detail="both gates present")
    ledger.advance("MODEL_IDENTITY_VERIFIED")
    assert [entry["state"] for entry in ledger.history] == [
        "NOT_STARTED",
        "AUTHORIZED",
        "MODEL_IDENTITY_VERIFIED",
    ]
    assert ledger.as_record()["attempt_counter_resettable"] is False


# --- failure and recovery -------------------------------------------------------------------


def test_failure_states_are_frozen(raw: dict[str, Any]) -> None:
    assert tuple(raw["failure_policy"]["states"]) == FAILURE_STATES


def test_no_failure_state_authorises_rerunning_inference() -> None:
    for state in FAILURE_STATES:
        assert inference_rerun_permitted(state) is False


def test_artifact_write_failure_rebuilds_from_persisted_predictions() -> None:
    classification = "EVALUATION_ARTIFACT_WRITE_FAILURE_AFTER_VALID_PREDICTIONS"
    assert classification in REBUILD_FROM_PREDICTIONS_ON
    assert recovery_action(classification) == (
        "REBUILD_METRICS_AND_REPORTS_FROM_PERSISTED_PREDICTIONS"
    )
    assert inference_rerun_permitted(classification) is False


def test_partial_prediction_failure_stops_and_requires_review() -> None:
    classification = "PREDICTION_EXECUTION_FAILED_BEFORE_RESULTS"
    assert recovery_action(classification) == "STOP_PRESERVE_EVIDENCE_REQUIRE_HUMAN_REVIEW"
    ledger = OneShotLedger()
    ledger.advance("AUTHORIZED")
    ledger.advance("MODEL_IDENTITY_VERIFIED")
    ledger.advance("TEST_LOADED")
    ledger.advance("DETECTOR_PREDICTION_STARTED")
    ledger.fail(classification, detail="CUDA error mid-run")
    assert ledger.state == "FAILED_REQUIRES_HUMAN_REVIEW"
    assert ledger.may_rebuild_from_predictions() is False


def test_rebuild_is_possible_only_after_predictions_are_persisted() -> None:
    ledger = OneShotLedger()
    for state in (
        "AUTHORIZED",
        "MODEL_IDENTITY_VERIFIED",
        "TEST_LOADED",
        "DETECTOR_PREDICTION_STARTED",
        "DETECTOR_PREDICTION_COMPLETE",
        "SEGMENTER_PREDICTION_STARTED",
        "SEGMENTER_PREDICTION_COMPLETE",
        "PREDICTIONS_PERSISTED",
    ):
        ledger.advance(state)
    assert ledger.may_rebuild_from_predictions() is True


def test_an_undeclared_failure_classification_is_refused() -> None:
    ledger = OneShotLedger()
    with pytest.raises(LedgerTransitionError):
        ledger.fail("SOMETHING_WENT_WRONG")


def test_report_rebuild_never_reruns_predictions(raw: dict[str, Any]) -> None:
    assert raw["fingerprints"]["rebuild_policy"]["predictions_rerun_on_rebuild"] is False
    assert raw["fingerprints"]["rebuild_policy"]["metrics_regenerated_from_persisted_predictions"]


# --- fingerprints ---------------------------------------------------------------------------


def test_protocol_fingerprint_is_deterministic(raw: dict[str, Any]) -> None:
    first = protocol_fingerprint(raw)
    second = protocol_fingerprint(load_protocol(CONFIG).raw)
    assert first == second
    assert len(first) == 64


def test_protocol_fingerprint_changes_with_a_scientific_decision(raw: dict[str, Any]) -> None:
    tampered = copy.deepcopy(raw)
    tampered["detector_inference"]["conf"] = 0.25
    assert protocol_fingerprint(tampered) != protocol_fingerprint(raw)


def test_prediction_and_result_fingerprint_designs_are_declared(raw: dict[str, Any]) -> None:
    fingerprints = raw["fingerprints"]
    assert fingerprints["prediction"]["detector"] == "detector_test_prediction_sha256"
    assert fingerprints["prediction"]["segmenter"] == "segmenter_test_prediction_sha256"
    for name in ("detector", "segmenter", "direct_iou", "qualitative"):
        assert fingerprints["result"][name]
    for excluded in ("timestamp", "username", "hostname"):
        assert excluded in fingerprints["excludes"]


# --- population and scope --------------------------------------------------------------------


def test_aggregate_population_only(raw: dict[str, Any]) -> None:
    population = raw["test_population"]
    assert population["images"] == TEST_IMAGES == 65
    assert population["annotations"] == TEST_ANNOTATIONS == 305
    assert population["per_image_composition_inspected_in_phase_11a"] is False
    assert population["evaluate"] == "ALL_FROZEN_TEST_IMAGES"
    assert population["exclusions_permitted"] is False
    assert population["sampling_permitted"] is False


def test_no_latency_benchmark_path_in_phase_11b(raw: dict[str, Any]) -> None:
    cost = raw["computational_cost"]
    assert cost["rerun_on_test"] is False
    assert cost["latency_benchmark_permitted_in_phase_11b"] is False


def test_no_new_spatial_metric_on_the_holdout(raw: dict[str, Any]) -> None:
    spatial = raw["spatial_analysis_on_test"]
    assert spatial["repeat_full_phase_10b_exploration"] is False
    assert spatial["new_spatial_metrics_permitted"] is False
    assert spatial["quantities_predeclared_for_test"] == []


def test_no_threshold_tuning_path(raw: dict[str, Any]) -> None:
    for block in ("detector_inference", "segmenter_inference"):
        assert raw[block]["threshold_sweep_permitted"] is False
        assert raw[block]["confidence_tuning_on_test_permitted"] is False
    assert "tuning any threshold on the holdout" in raw["prohibited"]


def test_no_training_path_in_any_phase_11a_file() -> None:
    for path in (MODULE, FREEZE_SCRIPT, RUNNER_SCRIPT):
        text = path.read_text(encoding="utf-8")
        for forbidden in ("import torch", "from torch", "import ultralytics", "from ultralytics"):
            assert forbidden not in text, (path.name, forbidden)
        for forbidden in (".train(", "YOLO(", "model.val(", "predict("):
            assert forbidden not in text, (path.name, forbidden)


def test_no_phase_11a_file_reaches_the_split_accessor() -> None:
    """No phase 11A file may enumerate holdout membership."""
    for path in (MODULE, FREEZE_SCRIPT, RUNNER_SCRIPT):
        text = path.read_text(encoding="utf-8")
        for forbidden in (
            "load_frozen_splits(",
            "FrozenSplits(",
            ".image_ids(",
            ".group_ids(",
            "assignments_from_sections(",
        ):
            assert forbidden not in text, (path.name, forbidden)


def test_the_split_manifest_is_read_only_through_aggregate_count_fields() -> None:
    """The freeze reads counts, never the membership sections that hold ids."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("_freeze_probe", FREEZE_SCRIPT)
    assert spec is not None and spec.loader is not None
    freeze = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(freeze)

    # Every declared field is a count block, not a membership block.
    assert freeze.AGGREGATE_FIELDS == (
        "actual_image_counts",
        "actual_annotation_counts",
        "actual_group_counts",
        "actual_negative_image_counts",
    )
    for name in freeze.AGGREGATE_FIELDS:
        assert name.startswith("actual_") and name.endswith("_counts")

    # And what comes back is integers, never identifiers.
    aggregate = freeze.read_aggregate_population(ROOT)
    assert set(aggregate) == {"images", "annotations", "groups", "negative_images"}
    assert all(isinstance(value, int) for value in aggregate.values())
    assert aggregate["images"] == TEST_IMAGES
    assert aggregate["annotations"] == TEST_ANNOTATIONS


def test_support_rule_is_reused_not_reinvented(raw: dict[str, Any]) -> None:
    rule = raw["support_rule"]
    assert rule["origin"] == "PREEXISTING_PHASE_7A_SUPPORT_RULE_REUSED_UNCHANGED"
    assert rule["min_positive_images"] == 5
    assert rule["min_instances"] == 20
    assert rule["invented_after_seeing_test_results"] is False
    assert rule["thresholds_may_change_after_test"] is False


def test_validation_versus_test_comparison_is_bounded(raw: dict[str, Any]) -> None:
    comparison = raw["validation_versus_test"]
    assert comparison["permitted"] is True
    assert comparison["significance_test"] is False
    assert comparison["metrics_must_already_exist"] is True
    assert comparison["new_metric_invented_for_the_comparison"] is False


def test_final_comparison_declares_no_winner(raw: dict[str, Any]) -> None:
    final = raw["final_comparison"]
    assert final["winner_declared"] is False
    assert final["composite_score"] is False
    assert final["weighted_ranking"] is False
    assert final["model_selection_follows"] is False


# --- the phase 11B runner --------------------------------------------------------------------


def test_phase_11b_execution_order_is_frozen() -> None:
    steps = phase_11b_steps()
    assert steps[0] == "VALIDATE_AUTHORIZATION"
    assert steps[-1] == "LOCK_RESULT_PROVENANCE"
    assert len(steps) == 15
    # Persistence must precede metric computation, or a rebuild is impossible.
    assert steps.index("PERSIST_AND_FINGERPRINT_DETECTOR_PREDICTIONS") < steps.index(
        "COMPUTE_CANONICAL_METRICS"
    )
    assert steps.index("PERSIST_AND_FINGERPRINT_SEGMENTER_PREDICTIONS") < steps.index(
        "COMPUTE_CANONICAL_METRICS"
    )


def test_the_runner_is_not_authorised_to_execute() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("_runner_probe", RUNNER_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Phase 11B flipped this in its own change, after explicit human
    # authorisation. What must never change is the order: the preflight fires
    # before anything that could reach the holdout, so a call without both gates
    # refuses on the gates rather than on the phase.
    assert module.EXECUTION_AUTHORISED is True
    with pytest.raises(HoldoutViolationError):
        module.run(allow_test=False)


def test_the_runner_plan_matches_the_frozen_order() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("_runner_plan_probe", RUNNER_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.plan() == phase_11b_steps()


# --- adversarial validation --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("status",), "EXECUTED"),
        (("detector", "checkpoint_sha256"), "0" * 64),
        (("segmenter", "checkpoint_sha256"), "0" * 64),
        (("detector", "imgsz"), 640),
        (("detector_inference", "conf"), 0.25),
        (("detector_inference", "iou"), 0.5),
        (("detector_inference", "max_det"), 100),
        (("detector_inference", "precision"), "FP16"),
        (("detector_inference", "tta"), True),
        (("segmenter_inference", "retina_masks"), False),
        (("canonical_bbox_evaluator", "iou_type"), "segm"),
        (("canonical_segm_evaluator", "implementation"), "custom.evaluator"),
        (("direct_iou", "protocol_fingerprint"), "0" * 64),
        (("confusion_matrix", "conf"), 0.5),
        (("confusion_matrix", "iou_threshold"), 0.5),
        (("qualitative_selection", "deterministic"), False),
        (("qualitative_selection", "manual_cherry_picking_permitted"), True),
        (("authorization", "automatic_environment_unlock_permitted"), True),
        (("authorization", "activated_in_this_phase"), True),
        (("authorization", "either_alone_is_refused"), False),
        (("final_comparison", "winner_declared"), True),
        (("final_comparison", "composite_score"), True),
        (("computational_cost", "rerun_on_test"), True),
        (("test_population", "exclusions_permitted"), True),
        (("one_shot", "reads_permitted"), 2),
        (("test_policy",), "READ"),
    ],
)
def test_validator_rejects_a_tampered_protocol(
    raw: dict[str, Any], path: tuple[str, ...], value: Any
) -> None:
    tampered = copy.deepcopy(raw)
    target: Any = tampered
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    assert validate_protocol(tampered), f"tampering with {path} was not caught"


def test_validator_rejects_a_present_test_metric(raw: dict[str, Any]) -> None:
    tampered = copy.deepcopy(raw)
    tampered["detector_metrics"]["test_map50_95"] = 0.51
    assert validate_protocol(tampered)


def test_parser_rejects_an_unknown_key(tmp_path: Any) -> None:
    import yaml

    document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    document["a_helpful_extra_knob"] = True
    target = tmp_path / "tampered.yaml"
    target.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(HoldoutProtocolError):
        load_protocol(target)


def test_parser_rejects_a_non_frozen_status(tmp_path: Any) -> None:
    import yaml

    document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    document["status"] = "EXECUTED"
    target = tmp_path / "executed.yaml"
    target.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(HoldoutProtocolError):
        load_protocol(target)


def test_protocol_document_carries_no_holdout_identifier() -> None:
    """The frozen protocol names counts and digests, never image ids."""
    text = CONFIG.read_text(encoding="utf-8")
    document = json.dumps(yaml_safe(text))
    for marker in ("image_ids", "source_image_id:", "test_images:"):
        assert marker not in document


def yaml_safe(text: str) -> Any:
    """Parse YAML for the identifier scan.

    Args:
        text: The document text.

    Returns:
        The parsed document.
    """
    import yaml

    return yaml.safe_load(text)
