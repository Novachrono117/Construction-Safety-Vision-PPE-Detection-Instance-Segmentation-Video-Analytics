"""Tests for the frozen final-segmenter accessor.

Pure logic only: nothing here trains, loads a model, runs inference or touches
the holdout. The accessor's job is to make a quiet substitution impossible, so
most of these are refusals.

The substitution that matters here does not exist for detection. S0 and S1 are
the same architecture at the same size, trained by the same protocol, and their
checkpoints are the same number of bytes. What separates them is
``overlap_mask``, which decides what the model was trained to predict. A caller
that loads S0's weights expecting the frozen segmenter gets a model that works,
loads without complaint, and predicts against a different target - so the
accessor rejects it by digest.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from construction_safety_vision.segmentation_freeze import (
    BLOCKED_MISSING_MODEL_ARTIFACT,
    FINAL_SELECTED,
    FINGERPRINT_FIELDS,
    FROZEN,
    HOLDOUT_STATUS,
    INVALID_SELECTION_STATE,
    MODEL_ARTIFACT_MISMATCH,
    SCHEMA_VERSION,
    TASK,
    CheckpointMismatchError,
    CheckpointMissingError,
    FinalSegmenterError,
    final_segmenter_fingerprint,
    holdout_leaks,
    load_checkpoint_path,
    load_final_segmenter,
    parse_final_segmenter,
)

SELECTED_SHA = "a" * 64
REFERENCE_SHA = "b" * 64
OTHER_SHA = "c" * 64


def _manifest(**overrides: Any) -> dict[str, Any]:
    """Build a minimally valid final-segmenter manifest.

    Args:
        **overrides: Top-level fields to replace.

    Returns:
        A manifest the parser accepts, with its fingerprint filled in.
    """
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": FROZEN,
        "task": TASK,
        "selection_status": FINAL_SELECTED,
        "selected_experiment": "S1",
        "selection_method": "PREDECLARED_CANONICAL_POLICY_PLUS_HUMAN_REVIEW",
        "margin_classification": "S1_IMPROVES_S0_BEYOND_MARGIN",
        "model": "YOLO11n-seg",
        "imgsz": 768,
        "batch": 8,
        "mask_ratio": 4,
        "overlap_mask": False,
        "pretrained_weights": {"identifier": "yolo11n-seg.pt", "sha256": OTHER_SHA},
        "selected_checkpoint": {
            "relative_path": "artifacts/segmentation/S1/weights/best.pt",
            "sha256": SELECTED_SHA,
            "size_bytes": 6041685,
            "committed": False,
        },
        "frozen_copy": {"relative_path": "artifacts/frozen/segmentation/S1_best.pt"},
        "rejected_checkpoints": {"reference_experiment_checkpoint_sha256": REFERENCE_SHA},
        "experiment_sha256": "d" * 64,
        "comparison_policy_sha256": "e" * 64,
        "canonical_evaluator_sha256": "f" * 64,
        "dataset_fingerprints": {
            "class_map_sha256": "1" * 64,
            "adapter": {
                "labels_development_sha256": "2" * 64,
                "image_membership_sha256": "3" * 64,
            },
        },
        "split_reference": {"split_assignment_sha256": "4" * 64},
        "comparison": {"reference_experiment": "S0", "delta": 0.07482},
        "primary_selection_metric": "CANONICAL_SUPPORTED_MACRO_MASK_MAP50_95",
        "practical_equivalence_margin": 0.005,
        "binary_distribution_status": "LOCAL_IGNORED_FROZEN_ARTIFACT",
        "repository_contains_model_binary": False,
        "test": {"status": HOLDOUT_STATUS, "reason": "not accessed"},
    }
    manifest.update(overrides)
    manifest["final_segmenter_sha256"] = final_segmenter_fingerprint(
        {
            "selected_experiment": manifest["selected_experiment"],
            "model": manifest["model"],
            "imgsz": manifest["imgsz"],
            "batch": manifest["batch"],
            "mask_ratio": manifest["mask_ratio"],
            "overlap_mask": manifest["overlap_mask"],
            "selected_checkpoint_sha256": manifest["selected_checkpoint"]["sha256"],
            "experiment_sha256": manifest["experiment_sha256"],
            "comparison_policy_sha256": manifest["comparison_policy_sha256"],
            "canonical_evaluator_sha256": manifest["canonical_evaluator_sha256"],
            "split_assignment_sha256": manifest["split_reference"]["split_assignment_sha256"],
            "class_map_sha256": manifest["dataset_fingerprints"]["class_map_sha256"],
            "adapter_labels_development_sha256": manifest["dataset_fingerprints"]["adapter"][
                "labels_development_sha256"
            ],
            "adapter_image_membership_sha256": manifest["dataset_fingerprints"]["adapter"][
                "image_membership_sha256"
            ],
        }
    )
    return manifest


# --- the fingerprint --------------------------------------------------------------


def test_the_fingerprint_covers_the_intervention() -> None:
    """overlap_mask is identity, not a note: flipping it is a different model."""
    assert "overlap_mask" in FINGERPRINT_FIELDS

    base = _manifest()
    flipped = _manifest(overlap_mask=True)

    assert base["final_segmenter_sha256"] != flipped["final_segmenter_sha256"]


@pytest.mark.parametrize(
    "field",
    ["selected_experiment", "model", "imgsz", "batch", "mask_ratio", "overlap_mask"],
)
def test_every_identity_field_moves_the_fingerprint(field: str) -> None:
    """A change to any identity field produces a different digest."""
    values = {name: f"value-{name}" for name in FINGERPRINT_FIELDS}
    baseline = final_segmenter_fingerprint(values)

    assert final_segmenter_fingerprint({**values, field: "changed"}) != baseline


def test_the_fingerprint_ignores_prose() -> None:
    """Adding a note to the manifest must not change the model's identity."""
    base = _manifest()
    noted = _manifest(limitations=["a new note"], selection_rationale=["another"])

    assert base["final_segmenter_sha256"] == noted["final_segmenter_sha256"]


def test_the_fingerprint_refuses_a_partial_identity() -> None:
    """A digest over an incomplete identity would identify something else."""
    values = dict.fromkeys(FINGERPRINT_FIELDS, "x")
    del values["selected_checkpoint_sha256"]

    with pytest.raises(FinalSegmenterError, match="missing identity field"):
        final_segmenter_fingerprint(values)


def test_the_fingerprint_accepts_a_false_boolean() -> None:
    """overlap_mask False is a value, not an absence."""
    values = dict.fromkeys(FINGERPRINT_FIELDS, "x")
    values["overlap_mask"] = False

    assert final_segmenter_fingerprint(values)


def test_the_fingerprint_is_deterministic() -> None:
    """The digest is a property of the values, not of assembly order."""
    values = {name: f"v{index}" for index, name in enumerate(FINGERPRINT_FIELDS)}
    shuffled = dict(reversed(list(values.items())))

    assert final_segmenter_fingerprint(values) == final_segmenter_fingerprint(shuffled)


# --- parsing ------------------------------------------------------------------------


def test_a_well_formed_manifest_parses() -> None:
    """The fixture is accepted, so every refusal below is about its own change."""
    segmenter = parse_final_segmenter(_manifest())

    assert segmenter.selected_experiment == "S1"
    assert segmenter.overlap_mask is False
    assert segmenter.mask_ratio == 4
    assert segmenter.imgsz == 768
    assert segmenter.batch == 8
    assert segmenter.recompute_fingerprint() == segmenter.fingerprint


def test_an_unfrozen_manifest_is_refused() -> None:
    """A manifest that does not claim a freeze cannot identify a model."""
    with pytest.raises(FinalSegmenterError, match="not 'FROZEN'"):
        parse_final_segmenter(_manifest(status="PENDING"))


def test_an_incomplete_selection_is_refused() -> None:
    """Selection status must say the decision was actually made."""
    with pytest.raises(FinalSegmenterError, match=INVALID_SELECTION_STATE):
        parse_final_segmenter(_manifest(selection_status="UNSELECTED_PENDING_REVIEW"))


def test_a_detection_manifest_is_refused() -> None:
    """The task is checked, so a detector manifest cannot be read as a segmenter."""
    with pytest.raises(FinalSegmenterError, match="task"):
        parse_final_segmenter(_manifest(task="detection"))


def test_an_edited_fingerprint_is_refused() -> None:
    """The recorded digest must match the one recomputed from the contents."""
    manifest = _manifest()
    manifest["final_segmenter_sha256"] = "9" * 64

    with pytest.raises(FinalSegmenterError, match="does not match"):
        parse_final_segmenter(manifest)


def test_an_edited_identity_field_is_refused() -> None:
    """Changing the model after the fingerprint was written is caught."""
    manifest = _manifest()
    manifest["imgsz"] = 640

    with pytest.raises(FinalSegmenterError, match="does not match"):
        parse_final_segmenter(manifest)


def test_a_non_boolean_overlap_mask_is_refused() -> None:
    """overlap_mask decides what the model predicts; a string will not do."""
    manifest = _manifest()
    manifest["overlap_mask"] = "false"

    with pytest.raises(FinalSegmenterError, match="not a boolean"):
        parse_final_segmenter(manifest)


def test_a_committed_binary_is_refused() -> None:
    """Model binaries are deliberately not committed."""
    manifest = _manifest()
    manifest["selected_checkpoint"]["committed"] = True

    with pytest.raises(FinalSegmenterError, match="committed must be false"):
        parse_final_segmenter(manifest)

    with pytest.raises(FinalSegmenterError, match="repository_contains_model_binary"):
        parse_final_segmenter(_manifest(repository_contains_model_binary=True))


def test_a_last_pt_selection_is_refused() -> None:
    """The frozen segmenter is the predeclared best.pt, never the final epoch."""
    manifest = _manifest()
    manifest["selected_checkpoint"]["relative_path"] = "artifacts/segmentation/S1/weights/last.pt"

    with pytest.raises(FinalSegmenterError, match=r"last\.pt"):
        parse_final_segmenter(manifest)


@pytest.mark.parametrize(
    "field",
    ["selected_checkpoint", "comparison", "dataset_fingerprints", "split_reference", "test"],
)
def test_a_missing_required_block_is_refused(field: str) -> None:
    """Every block the identity rests on must be present."""
    manifest = _manifest()
    del manifest[field]

    with pytest.raises(FinalSegmenterError):
        parse_final_segmenter(manifest)


def test_a_manifest_naming_the_holdout_is_refused() -> None:
    """A results file is the easiest place for a protected id to leak."""
    manifest = _manifest()
    manifest["comparison"]["evaluated_on"] = "test"

    with pytest.raises(FinalSegmenterError, match="protected split"):
        parse_final_segmenter(manifest)


def test_the_standard_protection_notice_is_allowed() -> None:
    """Recording that the holdout was not touched is evidence, not a leak."""
    assert holdout_leaks(_manifest()) == []
    assert holdout_leaks({"test": {"status": "SOMETHING_ELSE"}}) == ["test"]
    assert holdout_leaks({"a": [{"b": "test"}]}) == ["a[0].b"]


# --- checkpoint verification -----------------------------------------------------------


def test_the_selected_checkpoint_verifies(tmp_path: Path) -> None:
    """The right bytes at any path are accepted."""
    import hashlib

    payload = b"the selected weights"
    digest = hashlib.sha256(payload).hexdigest()
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(payload)

    segmenter = parse_final_segmenter(
        _manifest(
            selected_checkpoint={
                "relative_path": "artifacts/segmentation/S1/weights/best.pt",
                "sha256": digest,
                "size_bytes": len(payload),
                "committed": False,
            }
        )
    )

    assert segmenter.verify_checkpoint(checkpoint) == digest


def test_the_reference_experiments_checkpoint_is_rejected_by_digest(tmp_path: Path) -> None:
    """S0's weights load fine and predict a different target, so digest decides."""
    import hashlib

    payload = b"the S0 weights"
    reference_sha = hashlib.sha256(payload).hexdigest()
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(payload)

    segmenter = parse_final_segmenter(
        _manifest(rejected_checkpoints={"reference_experiment_checkpoint_sha256": reference_sha})
    )

    with pytest.raises(CheckpointMismatchError, match="S0 checkpoint"):
        segmenter.verify_checkpoint(checkpoint)


def test_last_pt_is_rejected_by_name(tmp_path: Path) -> None:
    """Rejected before its bytes are even read."""
    checkpoint = tmp_path / "last.pt"
    checkpoint.write_bytes(b"anything")
    segmenter = parse_final_segmenter(_manifest())

    with pytest.raises(CheckpointMismatchError, match=r"last\.pt"):
        segmenter.verify_checkpoint(checkpoint)


def test_wrong_bytes_are_rejected(tmp_path: Path) -> None:
    """A file at the right path with the wrong content is not the frozen model."""
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"different weights")
    segmenter = parse_final_segmenter(_manifest())

    with pytest.raises(CheckpointMismatchError, match=MODEL_ARTIFACT_MISMATCH):
        segmenter.verify_checkpoint(checkpoint)


def test_a_missing_checkpoint_says_do_not_retrain(tmp_path: Path) -> None:
    """Retraining would produce different weights under the same name."""
    segmenter = parse_final_segmenter(_manifest())

    with pytest.raises(CheckpointMissingError, match="do not retrain"):
        segmenter.verify_checkpoint(tmp_path / "absent.pt")


def test_the_frozen_copy_is_preferred_over_the_run_directory(tmp_path: Path) -> None:
    """The run directory is where a re-run would land, so it is tried second."""
    segmenter = parse_final_segmenter(_manifest())
    candidates = segmenter.candidate_paths(tmp_path)

    assert candidates[0].as_posix().endswith("artifacts/frozen/segmentation/S1_best.pt")
    assert candidates[1].as_posix().endswith("artifacts/segmentation/S1/weights/best.pt")


def test_loading_with_no_checkpoint_anywhere_is_blocked(tmp_path: Path) -> None:
    """Absent is reported, never produced."""
    segmenter = parse_final_segmenter(_manifest())

    with pytest.raises(CheckpointMissingError, match=BLOCKED_MISSING_MODEL_ARTIFACT):
        load_checkpoint_path(segmenter, tmp_path)


def test_loading_resolves_and_verifies_the_frozen_copy(tmp_path: Path) -> None:
    """The accessor returns a path only after checking the bytes."""
    import hashlib

    payload = b"frozen segmenter weights"
    frozen = tmp_path / "artifacts" / "frozen" / "segmentation" / "S1_best.pt"
    frozen.parent.mkdir(parents=True)
    frozen.write_bytes(payload)

    segmenter = parse_final_segmenter(
        _manifest(
            selected_checkpoint={
                "relative_path": "artifacts/segmentation/S1/weights/best.pt",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
                "committed": False,
            }
        )
    )

    assert load_checkpoint_path(segmenter, tmp_path) == frozen


# --- loading from disk -------------------------------------------------------------------


def test_a_missing_manifest_is_reported_not_invented(tmp_path: Path) -> None:
    """No manifest means no frozen segmenter, and the message says which phase writes it."""
    with pytest.raises(FinalSegmenterError, match="Phase 8G writes"):
        load_final_segmenter(tmp_path)


def test_a_malformed_manifest_is_refused(tmp_path: Path) -> None:
    """Invalid JSON is an error, not an empty selection."""
    (tmp_path / "final_segmenter_manifest.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(FinalSegmenterError, match="not valid JSON"):
        load_final_segmenter(tmp_path)


def test_a_written_manifest_round_trips(tmp_path: Path) -> None:
    """What the freeze writes is what the accessor reads back."""
    manifest = _manifest()
    (tmp_path / "final_segmenter_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    segmenter = load_final_segmenter(tmp_path)

    assert segmenter.fingerprint == manifest["final_segmenter_sha256"]
    assert segmenter.overlap_mask is False
