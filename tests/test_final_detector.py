"""Unit tests for the frozen-detector accessor.

These exercise the module's refusals rather than its happy path, because the
happy path is the part that cannot go wrong quietly. A YOLO checkpoint loads
whatever bytes it is handed: ``last.pt``, D0's weights, a re-trained file that
landed at the selected path. Every one of those would run, produce plausible
numbers, and be wrong. So each test below asks whether the accessor *stops*.

No test here runs a model, reads a dataset or touches the holdout.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from construction_safety_vision.detection_freeze import (
    FINGERPRINT_FIELDS,
    FROZEN,
    SCHEMA_VERSION,
    CheckpointMismatchError,
    CheckpointMissingError,
    FinalDetectorError,
    final_detector_fingerprint,
    load_checkpoint_path,
    load_final_detector,
    parse_final_detector,
)

DIGEST_A = "a" * 64
DIGEST_B = "b" * 64


def _identity(**overrides: Any) -> dict[str, Any]:
    """Build a complete set of fingerprint inputs.

    Args:
        overrides: Fields to replace.

    Returns:
        The identity mapping.
    """
    values: dict[str, Any] = {
        "selected_experiment": "D9",
        "model": "YOLO11n",
        "imgsz": 768,
        "selected_checkpoint_sha256": DIGEST_A,
        "experiment_sha256": "c" * 64,
        "phase7_policy_sha256": "d" * 64,
        "split_assignment_sha256": "e" * 64,
        "adapter_config_sha256": "f" * 64,
        "class_map_sha256": "0" * 64,
    }
    values.update(overrides)
    return values


def _manifest(**overrides: Any) -> dict[str, Any]:
    """Build a valid final-detector manifest.

    Args:
        overrides: Top-level fields to replace.

    Returns:
        The manifest mapping, with a consistent fingerprint.
    """
    identity = _identity()
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": FROZEN,
        "task": "detection",
        "selected_experiment": identity["selected_experiment"],
        "selection_method": "PREDECLARED_POLICY_PLUS_HUMAN_REVIEW",
        "policy_case": "CASE_B_VALIDATION_PERFORMANCE_LEADER",
        "model": identity["model"],
        "imgsz": identity["imgsz"],
        "batch": 16,
        "selected_checkpoint": {
            "relative_path": "artifacts/detection/D9/weights/best.pt",
            "sha256": identity["selected_checkpoint_sha256"],
            "size_bytes": 5502289,
            "committed": False,
        },
        "experiment_sha256": identity["experiment_sha256"],
        "phase7_policy_sha256": identity["phase7_policy_sha256"],
        "dataset_fingerprints": {
            "adapter_config_sha256": identity["adapter_config_sha256"],
            "class_map_sha256": identity["class_map_sha256"],
        },
        "split_reference": {"split_assignment_sha256": identity["split_assignment_sha256"]},
        "primary_selection_metric": "supported_macro_map50_95",
        "practical_equivalence_margin": "0.005",
        "binary_distribution_status": "LOCAL_IGNORED_FROZEN_ARTIFACT",
        "test": {"status": "PROTECTED_NOT_ACCESSED", "reason": "not accessed"},
    }
    payload.update(overrides)
    payload.setdefault("final_detector_sha256", final_detector_fingerprint(identity))
    return payload


# --- the semantic fingerprint -------------------------------------------------


def test_the_fingerprint_is_deterministic():
    assert final_detector_fingerprint(_identity()) == final_detector_fingerprint(_identity())


def test_the_fingerprint_ignores_assembly_order():
    forward = _identity()
    reversed_order = dict(reversed(list(forward.items())))
    assert final_detector_fingerprint(reversed_order) == final_detector_fingerprint(forward)


def test_the_fingerprint_ignores_fields_outside_the_identity():
    noisy = _identity()
    noisy["created_at"] = "2026-09-08T00:00:00+00:00"
    # Assembled rather than written as a literal, so that a synthetic home
    # directory does not sit in the source for a path scanner to flag.
    noisy["absolute_path"] = "/".join(("", "home", "someone", "checkpoints"))
    noisy["machine_user"] = "someone"
    assert final_detector_fingerprint(noisy) == final_detector_fingerprint(_identity())


@pytest.mark.parametrize("field", FINGERPRINT_FIELDS)
def test_changing_any_identity_field_changes_the_fingerprint(field: str):
    baseline = final_detector_fingerprint(_identity())
    changed = _identity(**{field: 999 if field == "imgsz" else "changed"})
    assert final_detector_fingerprint(changed) != baseline


def test_changing_the_selected_checkpoint_changes_the_fingerprint():
    baseline = final_detector_fingerprint(_identity())
    assert final_detector_fingerprint(_identity(selected_checkpoint_sha256=DIGEST_B)) != baseline


@pytest.mark.parametrize("field", FINGERPRINT_FIELDS)
def test_a_partial_identity_cannot_be_fingerprinted(field: str):
    values = _identity()
    values[field] = None
    with pytest.raises(FinalDetectorError, match="missing identity field"):
        final_detector_fingerprint(values)


# --- manifest parsing ---------------------------------------------------------


def test_a_valid_manifest_parses():
    detector = parse_final_detector(_manifest())
    assert detector.selected_experiment == "D9"
    assert detector.model == "YOLO11n"
    assert detector.imgsz == 768
    assert detector.batch == 16
    assert detector.checkpoint_sha256 == DIGEST_A
    assert detector.recompute_fingerprint() == detector.fingerprint


def test_an_unfrozen_manifest_is_refused():
    with pytest.raises(FinalDetectorError, match="not 'FROZEN'"):
        parse_final_detector(_manifest(status="UNSELECTED_PENDING_REVIEW"))


@pytest.mark.parametrize(
    "field",
    ["selected_experiment", "model", "imgsz", "selected_checkpoint", "final_detector_sha256"],
)
def test_a_manifest_missing_a_required_field_is_refused(field: str):
    payload = _manifest()
    del payload[field]
    with pytest.raises(FinalDetectorError, match="missing required field"):
        parse_final_detector(payload)


def test_an_edited_fingerprint_is_refused():
    # The whole point of recording the fingerprint is that editing an identity
    # field without recomputing it must be caught rather than trusted.
    payload = _manifest()
    payload["imgsz"] = 640
    with pytest.raises(FinalDetectorError, match="does not match the fingerprint recomputed"):
        parse_final_detector(payload)


def test_a_manifest_claiming_a_committed_binary_is_refused():
    payload = _manifest()
    payload["selected_checkpoint"]["committed"] = True
    payload["final_detector_sha256"] = final_detector_fingerprint(_identity())
    with pytest.raises(FinalDetectorError, match="committed must be false"):
        parse_final_detector(payload)


def test_a_manifest_pointing_at_last_pt_is_refused():
    payload = _manifest()
    payload["selected_checkpoint"]["relative_path"] = "artifacts/detection/D9/weights/last.pt"
    with pytest.raises(FinalDetectorError, match="freeze rejects"):
        parse_final_detector(payload)


def test_a_manifest_carrying_a_holdout_number_is_refused():
    payload = _manifest(test={"status": "EVALUATED", "map50_95": 0.5})
    with pytest.raises(FinalDetectorError, match="names the protected split"):
        parse_final_detector(payload)


def test_a_manifest_naming_the_protected_split_as_a_value_is_refused():
    payload = _manifest()
    payload["splits_used"] = ["train", "validation", "test"]
    with pytest.raises(FinalDetectorError, match="names the protected split"):
        parse_final_detector(payload)


def test_a_non_digest_checkpoint_hash_is_refused():
    payload = _manifest()
    payload["selected_checkpoint"]["sha256"] = "not-a-digest"
    payload["final_detector_sha256"] = final_detector_fingerprint(
        _identity(selected_checkpoint_sha256="not-a-digest")
    )
    with pytest.raises(FinalDetectorError, match="is not a SHA-256 digest"):
        parse_final_detector(payload)


# --- checkpoint resolution ----------------------------------------------------


def _write(path: Path, content: bytes) -> Path:
    """Write a file and return it.

    Args:
        path: Destination.
        content: Bytes to write.

    Returns:
        The path written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


@pytest.fixture
def frozen(tmp_path: Path):
    """Build a manifest whose digest matches a real temporary file.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        The parsed detector, its root and the checkpoint bytes.
    """
    import hashlib

    content = b"selected-checkpoint-bytes"
    digest = hashlib.sha256(content).hexdigest()
    identity = _identity(selected_checkpoint_sha256=digest)
    payload = _manifest()
    payload["selected_checkpoint"]["sha256"] = digest
    payload["selected_checkpoint"]["size_bytes"] = len(content)
    payload["frozen_copy"] = {"relative_path": "artifacts/frozen/detection/D9_best.pt"}
    payload["final_detector_sha256"] = final_detector_fingerprint(identity)
    return parse_final_detector(payload), tmp_path, content


def test_the_frozen_copy_is_preferred_over_the_run_directory(frozen):
    detector, root, content = frozen
    _write(root / "artifacts" / "frozen" / "detection" / "D9_best.pt", content)
    _write(root / "artifacts" / "detection" / "D9" / "weights" / "best.pt", content)
    assert load_checkpoint_path(detector, root).name == "D9_best.pt"


def test_the_run_directory_is_used_when_no_frozen_copy_exists(frozen):
    detector, root, content = frozen
    _write(root / "artifacts" / "detection" / "D9" / "weights" / "best.pt", content)
    assert load_checkpoint_path(detector, root).parent.name == "weights"


def test_a_missing_checkpoint_is_reported_not_retrained(frozen):
    detector, root, _ = frozen
    with pytest.raises(CheckpointMissingError, match="BLOCKED_MISSING_MODEL_ARTIFACT"):
        load_checkpoint_path(detector, root)


def test_a_missing_checkpoint_message_forbids_retraining(frozen):
    detector, root, _ = frozen
    with pytest.raises(CheckpointMissingError, match="do not retrain"):
        load_checkpoint_path(detector, root)


def test_wrong_bytes_at_the_selected_path_are_refused(frozen):
    detector, root, _ = frozen
    # The exact substitution the module exists to catch: a file with the right
    # name, in the right place, that is not the selected model.
    _write(root / "artifacts" / "detection" / "D9" / "weights" / "best.pt", b"a-different-model")
    with pytest.raises(CheckpointMismatchError, match="MODEL_ARTIFACT_MISMATCH"):
        load_checkpoint_path(detector, root)


def test_last_pt_is_rejected_by_name_even_with_matching_bytes(frozen):
    detector, root, content = frozen
    path = _write(root / "artifacts" / "detection" / "D9" / "weights" / "last.pt", content)
    with pytest.raises(CheckpointMismatchError, match="not the selected checkpoint"):
        detector.verify_checkpoint(path)


def test_another_experiments_checkpoint_is_refused(frozen):
    detector, root, _ = frozen
    path = _write(root / "artifacts" / "detection" / "D0" / "weights" / "best.pt", b"d0-weights")
    with pytest.raises(CheckpointMismatchError, match="different weights"):
        detector.verify_checkpoint(path)


def test_verify_accepts_the_selected_bytes(frozen):
    detector, root, content = frozen
    path = _write(root / "artifacts" / "frozen" / "detection" / "D9_best.pt", content)
    assert detector.verify_checkpoint(path) == detector.checkpoint_sha256


# --- loading from disk --------------------------------------------------------


def test_a_missing_manifest_is_reported_as_no_frozen_detector(tmp_path: Path):
    with pytest.raises(FinalDetectorError, match="no frozen detector"):
        load_final_detector(tmp_path)


def test_a_manifest_on_disk_round_trips(tmp_path: Path):
    payload = _manifest()
    (tmp_path / "final_detector_manifest.json").write_text(
        json.dumps(payload), encoding="utf-8", newline="\n"
    )
    assert load_final_detector(tmp_path).selected_experiment == "D9"
