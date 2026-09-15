"""Delivery identities and corruption rejection use metadata or synthetic files only."""

import hashlib
import json
from pathlib import Path

import pytest

from construction_safety_vision.checkpoint_delivery import (
    MANIFEST_PATH,
    build_manifest,
    load_manifest,
    validate_manifest,
    verify_file,
)

ROOT = Path(__file__).resolve().parents[1]


def test_public_identities_are_derived_from_the_authoritative_freezes():
    assert load_manifest(ROOT / MANIFEST_PATH) == build_manifest(ROOT)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("extra_field", "not permitted"),
        ("logical_name", "S0"),
        ("architecture", "YOLO11s"),
        ("imgsz", 640),
        ("imgsz", 768.0),
        ("sha256", "not-a-digest"),
        ("bytes", 0),
        ("bytes", True),
        ("status", "NOT_FROZEN"),
        ("distribution_status", "CHECKPOINT_REDISTRIBUTION_READY"),
        ("download_locator", "https://example.invalid/unpublished.pt"),
        ("freeze_manifest", "../untrusted.json"),
        ("license_notice", "https://example.invalid/license"),
    ],
)
def test_invalid_or_premature_distribution_metadata_is_rejected(field, value):
    payload = build_manifest(ROOT)
    payload["checkpoints"][0][field] = value
    with pytest.raises(ValueError):
        validate_manifest(payload)


def test_missing_duplicate_and_unknown_schema_entries_are_rejected():
    for mutate in (
        lambda x: x.update(schema_version=True),
        lambda x: x.update(unknown=True),
        lambda x: x["checkpoints"].pop(),
        lambda x: x["checkpoints"].__setitem__(1, x["checkpoints"][0]),
    ):
        payload = build_manifest(ROOT)
        mutate(payload)
        with pytest.raises(ValueError):
            validate_manifest(payload)


def test_synthetic_bytes_verify_without_deserialization(tmp_path):
    data = b"synthetic-checkpoint-fixture; deliberately not a serialized model"
    path = tmp_path / "fixture.pt"
    path.write_bytes(data)
    verify_file(path, sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data))


def test_same_size_wrong_checkpoint_is_rejected(tmp_path):
    path = tmp_path / "fixture.pt"
    path.write_bytes(b"synthetic-A")
    with pytest.raises(ValueError, match="SHA256_MISMATCH"):
        verify_file(path, sha256=hashlib.sha256(b"synthetic-B").hexdigest(), size_bytes=11)
    assert path.read_bytes() == b"synthetic-A"


def test_wrong_size_and_missing_file_fail_closed(tmp_path):
    path = tmp_path / "fixture.pt"
    expected = hashlib.sha256(b"synthetic").hexdigest()
    with pytest.raises(ValueError, match="SIZE_MISMATCH_OR_MISSING"):
        verify_file(path, sha256=expected, size_bytes=9)
    path.write_bytes(b"truncated")
    with pytest.raises(ValueError, match="SIZE_MISMATCH_OR_MISSING"):
        verify_file(path, sha256=expected, size_bytes=10)


def test_manifest_round_trip_is_strict(tmp_path):
    path = tmp_path / "checkpoints.json"
    payload = build_manifest(ROOT)
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_manifest(path) == payload
