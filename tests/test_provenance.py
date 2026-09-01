"""Tests for artifact hashing and provenance records."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from construction_safety_vision.provenance import (
    SCHEMA_VERSION,
    ArtifactRecord,
    ProvenanceRecord,
    git_commit,
    sha256_file,
)


def write_bytes(path: Path, payload: bytes) -> Path:
    path.write_bytes(payload)
    return path


def test_hash_matches_hashlib(tmp_path: Path) -> None:
    payload = b"annotation bytes"
    target = write_bytes(tmp_path / "sample.bin", payload)

    assert sha256_file(target) == hashlib.sha256(payload).hexdigest()


def test_hash_is_chunk_size_independent(tmp_path: Path) -> None:
    payload = b"x" * 5000
    target = write_bytes(tmp_path / "large.bin", payload)

    assert sha256_file(target, chunk_size=7) == sha256_file(target, chunk_size=4096)


def test_hashing_a_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        sha256_file(tmp_path / "absent.bin")


def test_artifact_record_stores_a_relative_posix_path(tmp_path: Path) -> None:
    nested = tmp_path / "data" / "raw"
    nested.mkdir(parents=True)
    target = write_bytes(nested / "image.jpg", b"jpeg")

    record = ArtifactRecord.from_path(target, relative_to=tmp_path)

    assert record.path == "data/raw/image.jpg"
    assert record.size_bytes == 4
    assert record.sha256 == hashlib.sha256(b"jpeg").hexdigest()


def test_artifact_record_falls_back_to_the_full_path(tmp_path: Path) -> None:
    other = tmp_path / "outside"
    other.mkdir()
    target = write_bytes(tmp_path / "file.txt", b"data")

    record = ArtifactRecord.from_path(target, relative_to=other)

    assert record.path.endswith("file.txt")


def test_record_round_trips_through_json(tmp_path: Path) -> None:
    source = write_bytes(tmp_path / "raw.json", b"{}")
    derived = write_bytes(tmp_path / "processed.json", b"[]")

    record = ProvenanceRecord.create(
        "split-freeze",
        phase=5,
        config={"seed": 42},
        repo_root=tmp_path,
        details={"note": "unit test"},
    )
    record.add_input(source, relative_to=tmp_path)
    record.add_output(derived, relative_to=tmp_path)
    destination = record.write_json(tmp_path / "records" / "split-freeze.provenance.json")

    payload = json.loads(destination.read_text(encoding="utf-8"))

    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["name"] == "split-freeze"
    assert payload["phase"] == 5
    assert payload["config"] == {"seed": 42}
    assert payload["details"] == {"note": "unit test"}
    assert [item["path"] for item in payload["inputs"]] == ["raw.json"]
    assert [item["path"] for item in payload["outputs"]] == ["processed.json"]
    assert payload["environment"]["python_version"]
    assert payload["created_at"].endswith("+00:00")


def test_record_is_written_with_lf_line_endings(tmp_path: Path) -> None:
    record = ProvenanceRecord.create("acquisition", phase=3, repo_root=tmp_path)
    destination = record.write_json(tmp_path / "acquisition.provenance.json")

    assert b"\r\n" not in destination.read_bytes()


def test_git_commit_returns_a_hash_or_none(tmp_path: Path) -> None:
    # Outside a repository (or without git installed) the call must degrade to
    # None instead of raising, so provenance writing never blocks a run.
    assert git_commit(tmp_path) is None

    commit = git_commit()
    assert commit is None or len(commit) == 40
