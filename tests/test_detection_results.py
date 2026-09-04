"""Tests for the detection-result bookkeeping.

Pure functions only: no torch, no ultralytics, no GPU, no training. The point of
separating this logic from the training script is that the rules protecting the
result - the fingerprint, the manifest schema, the holdout check - can be tested
without a 20-minute run.
"""

from __future__ import annotations

import pytest

from construction_safety_vision.detection_results import (
    COMPLETE,
    HIGH_SAMPLING_UNCERTAINTY,
    MANIFEST_SCHEMA_VERSION,
    NOT_EXPOSED,
    PER_CLASS_METRICS,
    PRIMARY_METRIC,
    SECONDARY_METRICS,
    TEST_PROTECTED,
    contains_forbidden_split,
    critical_arguments,
    experiment_fingerprint,
    global_metrics,
    per_class_metrics,
    ultralytics_fitness,
    validate_result_manifest,
)

CLASS_NAMES = ("helmet_loose", "helmet_on_head", "person", "vest_loose", "vest_on_body")

RESOLVED = {
    "optimizer": "AdamW",
    "lr0": 0.000714,
    "lrf": 0.01,
    "momentum": 0.9,
    "weight_decay": 0.0005,
    "epochs": 100,
    "batch": 16,
    "imgsz": 640,
    "seed": 42,
    "amp": True,
    "deterministic": True,
    "mosaic": 1.0,
    "fliplr": 0.5,
    # Not part of the experiment's identity:
    "workers": 8,
    "save_dir": "/somewhere/on/this/machine",
    "verbose": True,
}

FINGERPRINT_ARGS = {
    "baseline_config_sha256": "a" * 64,
    "pretrained_weights_sha256": "b" * 64,
    "adapter_manifest_sha256": "c" * 64,
    "split_assignment_sha256": "d" * 64,
    "resolved_arguments": RESOLVED,
    "best_checkpoint_sha256": "e" * 64,
}


def make_manifest(**overrides):
    """A structurally complete manifest, for mutation in individual tests."""
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment_id": "D0",
        "status": COMPLETE,
        "model": "YOLO11n",
        "pretrained_weights": {"identifier": "yolo11n.pt", "sha256": "b" * 64},
        "git_commit": "f" * 40,
        "baseline_config_sha256": "a" * 64,
        "adapter_manifest_sha256": "c" * 64,
        "dataset_fingerprints": {"class_map_sha256": "9" * 64},
        "split_reference": {"split_assignment_sha256": "d" * 64},
        "train_images": 303,
        "validation_images": 65,
        "test": {"status": TEST_PROTECTED, "reason": "not accessed"},
        "epochs_configured": 100,
        "epochs_completed": 100,
        "best_epoch": 71,
        "early_stopped": False,
        "resolved_optimizer": "AdamW",
        "best_checkpoint": {"sha256": "1" * 64, "size_bytes": 5_600_000, "committed": False},
        "last_checkpoint": {"sha256": "2" * 64, "size_bytes": 5_600_000, "committed": False},
        "validation_metrics": {
            PRIMARY_METRIC: 0.51,
            "mAP@0.50": 0.72,
            "precision": 0.68,
            "recall": 0.63,
        },
        "per_class_metrics": {name: dict.fromkeys(PER_CLASS_METRICS, 0.5) for name in CLASS_NAMES},
        "rare_class": {
            "name": "vest_loose",
            "warning": HIGH_SAMPLING_UNCERTAINTY,
            "validation_images": 1,
            "validation_instances": 8,
        },
        "training_duration_seconds": 1234.5,
        "d0_experiment_sha256": "3" * 64,
    }
    manifest.update(overrides)
    return manifest


# --- experiment fingerprint -------------------------------------------------


def test_fingerprint_is_deterministic():
    assert experiment_fingerprint(**FINGERPRINT_ARGS) == experiment_fingerprint(**FINGERPRINT_ARGS)


def test_fingerprint_ignores_argument_order():
    shuffled = dict(reversed(list(RESOLVED.items())))
    assert experiment_fingerprint(
        **{**FINGERPRINT_ARGS, "resolved_arguments": shuffled}
    ) == experiment_fingerprint(**FINGERPRINT_ARGS)


def test_fingerprint_ignores_timestamps_and_machine_paths():
    noisy = {
        **RESOLVED,
        "save_dir": "/a/completely/different/machine/path",
        "time": "2026-09-04T18:00:00",
        "name": "some_run_name",
        "workers": 2,
        "verbose": False,
    }
    assert experiment_fingerprint(
        **{**FINGERPRINT_ARGS, "resolved_arguments": noisy}
    ) == experiment_fingerprint(**FINGERPRINT_ARGS)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("optimizer", "SGD"),
        ("lr0", 0.01),
        ("imgsz", 960),
        ("epochs", 200),
        ("batch", 8),
        ("seed", 7),
        ("mosaic", 0.0),
        ("fliplr", 0.0),
    ],
)
def test_fingerprint_moves_when_a_critical_argument_changes(key, value):
    changed = {**RESOLVED, key: value}
    assert experiment_fingerprint(
        **{**FINGERPRINT_ARGS, "resolved_arguments": changed}
    ) != experiment_fingerprint(**FINGERPRINT_ARGS)


@pytest.mark.parametrize(
    "field",
    [
        "baseline_config_sha256",
        "pretrained_weights_sha256",
        "adapter_manifest_sha256",
        "split_assignment_sha256",
        "best_checkpoint_sha256",
    ],
)
def test_fingerprint_moves_when_an_input_changes(field):
    assert experiment_fingerprint(**{**FINGERPRINT_ARGS, field: "0" * 64}) != (
        experiment_fingerprint(**FINGERPRINT_ARGS)
    )


def test_critical_arguments_drops_run_bookkeeping():
    selected = critical_arguments(RESOLVED)
    assert "optimizer" in selected
    assert "save_dir" not in selected
    assert "workers" not in selected
    assert "verbose" not in selected


# --- metric extraction ------------------------------------------------------


def test_global_metrics_map_the_framework_keys():
    metrics = global_metrics(
        {
            "metrics/mAP50-95(B)": 0.5123456789,
            "metrics/mAP50(B)": 0.72,
            "metrics/precision(B)": 0.68,
            "metrics/recall(B)": 0.63,
            "fitness": 0.53,
        }
    )
    assert metrics[PRIMARY_METRIC] == 0.512346
    assert set(metrics) == {PRIMARY_METRIC, *SECONDARY_METRICS}


def test_missing_global_metric_is_recorded_as_missing():
    assert global_metrics({})[PRIMARY_METRIC] == NOT_EXPOSED


def test_nan_metric_is_recorded_as_missing():
    assert global_metrics({"metrics/mAP50(B)": float("nan")})["mAP@0.50"] == NOT_EXPOSED


def test_per_class_metrics_are_keyed_by_class_name():
    result = per_class_metrics(
        CLASS_NAMES,
        class_indices=[0, 2, 4],
        precision=[0.9, 0.8, 0.7],
        recall=[0.6, 0.5, 0.4],
        ap50=[0.85, 0.75, 0.65],
        ap=[0.55, 0.45, 0.35],
    )
    assert set(result) == set(CLASS_NAMES)
    assert result["helmet_loose"]["AP@0.50"] == 0.85
    assert result["person"]["precision"] == 0.8
    assert result["vest_on_body"]["AP@0.50:0.95"] == 0.35


def test_a_class_the_framework_did_not_report_is_not_given_a_zero():
    # A zero would read as "the model failed at it"; absent means "not measured".
    result = per_class_metrics(
        CLASS_NAMES,
        class_indices=[0],
        precision=[0.9],
        recall=[0.6],
        ap50=[0.85],
        ap=[0.55],
    )
    assert result["vest_loose"] == dict.fromkeys(PER_CLASS_METRICS, NOT_EXPOSED)
    assert result["vest_loose"]["AP@0.50"] != 0.0


def test_fitness_matches_the_framework_definition():
    assert ultralytics_fitness(0.8, 0.5) == pytest.approx(0.1 * 0.8 + 0.9 * 0.5)


# --- holdout protection -----------------------------------------------------


def test_forbidden_split_detected_as_a_nested_key():
    assert contains_forbidden_split({"metrics": {"test": {"mAP": 0.9}}})
    assert contains_forbidden_split([{"a": 1}, {"test": 2}])


def test_a_string_value_test_is_not_a_leak():
    # Only a keyed section carries holdout data; the word alone does not.
    assert not contains_forbidden_split({"reason": "the test split is protected"})


def test_manifest_with_holdout_metrics_is_rejected():
    manifest = make_manifest()
    manifest["extra_metrics"] = {"test": {PRIMARY_METRIC: 0.4}}
    problems = validate_result_manifest(manifest, class_names=CLASS_NAMES)
    assert any("'test' keyed section" in problem for problem in problems)


def test_manifest_test_entry_must_be_protected():
    manifest = make_manifest(test={"status": "EVALUATED", "reason": "oops"})
    problems = validate_result_manifest(manifest, class_names=CLASS_NAMES)
    assert any(TEST_PROTECTED in problem for problem in problems)


def test_manifest_test_entry_may_not_carry_extra_fields():
    manifest = make_manifest(test={"status": TEST_PROTECTED, "reason": "x", "image_count": 65})
    problems = validate_result_manifest(manifest, class_names=CLASS_NAMES)
    assert any("unexpected field" in problem for problem in problems)


# --- manifest validation ----------------------------------------------------


def test_a_well_formed_manifest_has_no_problems():
    assert validate_result_manifest(make_manifest(), class_names=CLASS_NAMES) == []


def test_missing_field_is_reported():
    manifest = make_manifest()
    del manifest["best_checkpoint"]
    assert any(
        "missing required field" in problem
        for problem in validate_result_manifest(manifest, class_names=CLASS_NAMES)
    )


def test_unknown_status_is_rejected():
    manifest = make_manifest(status="LOOKS_GOOD")
    assert any(
        "is not one of" in problem
        for problem in validate_result_manifest(manifest, class_names=CLASS_NAMES)
    )


def test_missing_primary_metric_is_rejected():
    manifest = make_manifest()
    del manifest["validation_metrics"][PRIMARY_METRIC]
    assert any(
        "primary metric" in problem
        for problem in validate_result_manifest(manifest, class_names=CLASS_NAMES)
    )


@pytest.mark.parametrize("metric", SECONDARY_METRICS)
def test_missing_secondary_metric_is_rejected(metric):
    manifest = make_manifest()
    del manifest["validation_metrics"][metric]
    assert any(
        "secondary metric" in problem
        for problem in validate_result_manifest(manifest, class_names=CLASS_NAMES)
    )


def test_per_class_table_must_cover_exactly_the_frozen_classes():
    manifest = make_manifest()
    del manifest["per_class_metrics"]["vest_loose"]
    assert any(
        "the frozen class map is" in problem
        for problem in validate_result_manifest(manifest, class_names=CLASS_NAMES)
    )


def test_per_class_table_rejects_an_invented_class():
    manifest = make_manifest()
    manifest["per_class_metrics"]["hard_hat"] = dict.fromkeys(PER_CLASS_METRICS, 0.5)
    assert any(
        "the frozen class map is" in problem
        for problem in validate_result_manifest(manifest, class_names=CLASS_NAMES)
    )


def test_per_class_entry_must_carry_every_metric():
    manifest = make_manifest()
    del manifest["per_class_metrics"]["person"]["recall"]
    assert any(
        "is missing 'recall'" in problem
        for problem in validate_result_manifest(manifest, class_names=CLASS_NAMES)
    )


def test_rare_class_warning_is_required():
    manifest = make_manifest()
    manifest["rare_class"]["warning"] = ""
    assert any(
        HIGH_SAMPLING_UNCERTAINTY in problem
        for problem in validate_result_manifest(manifest, class_names=CLASS_NAMES)
    )


def test_rare_class_must_be_a_frozen_class():
    manifest = make_manifest()
    manifest["rare_class"]["name"] = "hard_hat"
    assert any(
        "not one of the frozen classes" in problem
        for problem in validate_result_manifest(manifest, class_names=CLASS_NAMES)
    )


@pytest.mark.parametrize("name", ["best_checkpoint", "last_checkpoint"])
def test_checkpoint_hash_is_required(name):
    manifest = make_manifest()
    manifest[name]["sha256"] = ""
    assert any(
        "must be fingerprinted" in problem
        for problem in validate_result_manifest(manifest, class_names=CLASS_NAMES)
    )


@pytest.mark.parametrize("name", ["best_checkpoint", "last_checkpoint"])
def test_a_committed_checkpoint_is_rejected(name):
    manifest = make_manifest()
    manifest[name]["committed"] = True
    assert any(
        "weights are never committed" in problem
        for problem in validate_result_manifest(manifest, class_names=CLASS_NAMES)
    )


def test_resolved_optimizer_must_be_recorded():
    # `optimizer: auto` is a policy, not a setting; the run must say what it chose.
    manifest = make_manifest(resolved_optimizer=NOT_EXPOSED)
    assert any(
        "optimizer=auto must resolve" in problem
        for problem in validate_result_manifest(manifest, class_names=CLASS_NAMES)
    )


def test_experiment_fingerprint_must_not_be_empty():
    manifest = make_manifest(d0_experiment_sha256="")
    assert any(
        "d0_experiment_sha256 is empty" in problem
        for problem in validate_result_manifest(manifest, class_names=CLASS_NAMES)
    )
