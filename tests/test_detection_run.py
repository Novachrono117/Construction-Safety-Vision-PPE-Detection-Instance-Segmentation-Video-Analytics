"""Tests for the shared experiment-execution primitives.

No model is trained here. What is tested is the bookkeeping that has to behave
identically across experiments - because if D1 and D2 established their
optimizer or fingerprinted their weights differently, the comparison between
them would be measuring the bookkeeping too.

The optimizer tests are the important ones. Ultralytics colourises its log
labels, so a parser that matches the raw bytes fails on the real format and
falls back to inference without saying why. These pin the ANSI handling against
the framework's actual output, and pin the distinction between reading a value
and inferring one.
"""

from __future__ import annotations

import pytest

from construction_safety_vision.detection_comparison import (
    ComparisonError,
    flatten_protocol,
    unflatten_protocol,
)
from construction_safety_vision.detection_results import NOT_EXPOSED
from construction_safety_vision.detection_run import (
    AUTO_OPTIMIZER_ITERATION_THRESHOLD,
    CROSS_CHECK_TOLERANCE,
    DECLARED_EXPLICITLY,
    FRAMEWORK_LOG_LINE,
    METRIC_FIGURES,
    NOMINAL_BATCH_SIZE,
    REDERIVED_CORROBORATED,
    REDERIVED_UNCORROBORATED,
    WEIGHT_SOURCE_MECHANISM,
    best_epoch_from_history,
    cross_check_metrics,
    optimizer_evidence,
    optimizer_evidence_from_log,
    rederive_auto_optimizer,
    strip_ansi,
)
from construction_safety_vision.experiment import load_detection_baseline_config
from construction_safety_vision.paths import ProjectPaths


@pytest.fixture(scope="module")
def paths() -> ProjectPaths:
    return ProjectPaths.from_root()


# --- optimizer evidence -------------------------------------------------------

REAL_LOG_LINE = (
    "\x1b[34m\x1b[1moptimizer:\x1b[0m AdamW(lr=0.001111, momentum=0.9) with parameter "
    "groups 81 weight(decay=0.0), 88 weight(decay=0.0005), 87 bias(decay=0.0)"
)


def test_strip_ansi_removes_colour_codes():
    assert strip_ansi(REAL_LOG_LINE).startswith("optimizer: AdamW(lr=0.001111, momentum=0.9)")
    assert "\x1b" not in strip_ansi(REAL_LOG_LINE)
    assert strip_ansi("plain text") == "plain text"


def test_optimizer_is_read_from_the_colourised_log_line():
    # The regression this guards: without ANSI stripping the parse silently
    # fails on the framework's real output and the run loses its best evidence.
    evidence = optimizer_evidence_from_log(REAL_LOG_LINE)
    assert evidence is not None
    assert evidence["optimizer"] == "AdamW"
    assert evidence["effective_lr0"] == 0.001111
    assert evidence["effective_momentum"] == 0.9
    assert evidence["source"] == FRAMEWORK_LOG_LINE
    assert evidence["inferred"] is False
    assert "\x1b" not in evidence["evidence"]


def test_optimizer_log_line_matches_the_installed_framework_format():
    # Built from the framework's own formatting helper rather than a copy of it,
    # so a change in how it labels the line fails here instead of downstream.
    colorstr = pytest.importorskip("ultralytics.utils").colorstr
    line = f"{colorstr('optimizer:')} AdamW(lr=0.001111, momentum=0.9) with parameter groups 57"
    evidence = optimizer_evidence_from_log(line)
    assert evidence is not None
    assert evidence["optimizer"] == "AdamW"


def test_optimizer_log_line_absent_returns_none():
    assert optimizer_evidence_from_log("no optimizer line here") is None
    assert optimizer_evidence_from_log("") is None


def test_an_explicit_optimizer_needs_no_evidence(tmp_path):
    evidence = optimizer_evidence(
        tmp_path,
        requested="SGD",
        class_count=5,
        train_images=303,
        batch=16,
        epochs=100,
        history=[],
        ultralytics_version="8.4.138",
    )
    assert evidence["optimizer"] == "SGD"
    assert evidence["source"] == DECLARED_EXPLICITLY
    assert evidence["inferred"] is False


def test_direct_capture_is_preferred_over_inference(tmp_path):
    (tmp_path / "train_console.log").write_text(REAL_LOG_LINE, encoding="utf-8")
    evidence = optimizer_evidence(
        tmp_path,
        requested="auto",
        class_count=5,
        train_images=303,
        batch=16,
        epochs=100,
        history=[{"lr/pg0": "0.001078"}],
        ultralytics_version="8.4.138",
    )
    assert evidence["source"] == FRAMEWORK_LOG_LINE
    assert evidence["inferred"] is False


def test_inference_is_the_fallback_and_says_so(tmp_path):
    evidence = optimizer_evidence(
        tmp_path,
        requested="auto",
        class_count=5,
        train_images=303,
        batch=16,
        epochs=100,
        history=[{"lr/pg0": "0.001078"}],
        ultralytics_version="8.4.138",
    )
    assert evidence["inferred"] is True
    assert evidence["source"] == REDERIVED_CORROBORATED
    assert "inference, not a reading" in evidence["note"]


def test_rederivation_matches_the_framework_rule_for_this_protocol():
    # 303 training images, batch 16, 100 epochs: ceil(303 / max(16, 64)) * 100.
    evidence = rederive_auto_optimizer(
        class_count=5,
        train_images=303,
        batch=16,
        epochs=100,
        history=[{"lr/pg0": "0.001078"}],
        ultralytics_version="8.4.138",
    )
    assert evidence["iterations_estimate"] == 500
    assert evidence["iterations_estimate"] < AUTO_OPTIMIZER_ITERATION_THRESHOLD
    assert evidence["optimizer"] == "AdamW"
    assert evidence["effective_lr0"] == round(0.002 * 5 / (4 + 5), 6) == 0.001111
    assert evidence["corroborated_by_observed_lr"] is True


def test_rederivation_takes_the_high_iteration_branch_above_the_threshold():
    evidence = rederive_auto_optimizer(
        class_count=5,
        train_images=100_000,
        batch=16,
        epochs=100,
        history=[],
        ultralytics_version="8.4.138",
    )
    assert evidence["iterations_estimate"] > AUTO_OPTIMIZER_ITERATION_THRESHOLD
    assert evidence["optimizer"] == "MuSGD"
    assert evidence["effective_lr0"] == 0.01


def test_rederivation_without_an_observation_is_uncorroborated():
    evidence = rederive_auto_optimizer(
        class_count=5,
        train_images=303,
        batch=16,
        epochs=100,
        history=[],
        ultralytics_version="8.4.138",
    )
    assert evidence["observed_peak_lr"] == NOT_EXPOSED
    assert evidence["corroborated_by_observed_lr"] is False
    assert evidence["source"] == REDERIVED_UNCORROBORATED


def test_rederivation_records_the_framework_version():
    evidence = rederive_auto_optimizer(
        class_count=5,
        train_images=303,
        batch=16,
        epochs=100,
        history=[],
        ultralytics_version="9.9.9",
    )
    assert evidence["ultralytics_version"] == "9.9.9"


def test_the_nominal_batch_floors_the_iteration_estimate():
    small = rederive_auto_optimizer(
        class_count=5,
        train_images=303,
        batch=1,
        epochs=1,
        history=[],
        ultralytics_version="8.4.138",
    )
    assert small["iterations_estimate"] == -(-303 // NOMINAL_BATCH_SIZE)


# --- checkpoint and metric bookkeeping ---------------------------------------


def test_best_epoch_is_recomputed_by_ultralytics_fitness():
    history = [
        {"epoch": "1", "metrics/mAP50(B)": "0.10", "metrics/mAP50-95(B)": "0.05"},
        {"epoch": "2", "metrics/mAP50(B)": "0.90", "metrics/mAP50-95(B)": "0.60"},
        {"epoch": "3", "metrics/mAP50(B)": "0.95", "metrics/mAP50-95(B)": "0.20"},
    ]
    epoch, fitness = best_epoch_from_history(history)
    assert epoch == 2
    assert fitness == pytest.approx(0.1 * 0.90 + 0.9 * 0.60)


def test_best_epoch_of_an_empty_history_is_unknown():
    assert best_epoch_from_history([]) == (None, None)


def test_malformed_history_rows_are_skipped_not_guessed():
    history = [
        {"epoch": "oops", "metrics/mAP50(B)": "0.9", "metrics/mAP50-95(B)": "0.6"},
        {"epoch": "2", "metrics/mAP50(B)": "0.5", "metrics/mAP50-95(B)": "0.3"},
    ]
    assert best_epoch_from_history(history)[0] == 2


def test_cross_check_accepts_a_small_disagreement():
    history = [{"epoch": "7", "metrics/mAP50(B)": "0.6", "metrics/mAP50-95(B)": "0.500"}]
    record = cross_check_metrics(0.505, history, 7)
    assert record["delta"] == pytest.approx(0.005)
    assert record["tolerance"] == CROSS_CHECK_TOLERANCE


def test_cross_check_refuses_to_publish_a_mismatched_metric():
    from construction_safety_vision.detection_results import ResultError

    history = [{"epoch": "7", "metrics/mAP50(B)": "0.6", "metrics/mAP50-95(B)": "0.500"}]
    with pytest.raises(ResultError, match="may not belong to the"):
        cross_check_metrics(0.900, history, 7)


def test_metric_figures_are_an_allowlist_without_dataset_imagery():
    joined = " ".join(METRIC_FIGURES)
    assert "batch" not in joined
    assert "labels" not in joined
    assert all(name.endswith(".png") for name in METRIC_FIGURES)


def test_the_weight_source_mechanism_is_recorded_identically_for_every_run():
    # D0's phase 6A record used this exact string; a candidate that recorded a
    # different one could not be shown to have started from a comparable asset.
    assert WEIGHT_SOURCE_MECHANISM == "ULTRALYTICS_ASSET_DOWNLOAD"


# --- protocol flattening round trip ------------------------------------------


def test_unflatten_is_the_inverse_of_flatten(paths):
    protocol = load_detection_baseline_config(paths.configs / "detection_baseline.yaml").as_dict()
    flat = flatten_protocol(protocol)
    nested = unflatten_protocol(flat)
    assert nested["training"]["imgsz"] == protocol["training"]["imgsz"]
    assert nested["seed"] == protocol["seed"]
    assert flatten_protocol(nested) == flat


def test_unflatten_rebuilds_nested_branches():
    nested = unflatten_protocol({"a.b.c": 1, "a.b.d": 2, "e": 3})
    assert nested == {"a": {"b": {"c": 1, "d": 2}}, "e": 3}


def test_unflatten_refuses_a_leaf_and_branch_conflict():
    with pytest.raises(ComparisonError, match="conflicts with"):
        unflatten_protocol({"a": 1, "a.b": 2})
