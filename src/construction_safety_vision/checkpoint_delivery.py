"""Public checkpoint metadata and byte verification, without model deserialization."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from construction_safety_vision.provenance import sha256_file

REVIEW_REQUIRED = "CHECKPOINT_REDISTRIBUTION_REQUIRES_LICENSE_REVIEW"
MANIFEST_PATH = "delivery/checkpoints.json"
SOURCES = (
    ("detector", "reports/detection_D2.provenance.json"),
    ("segmenter", "reports/segmentation_S1.provenance.json"),
)


def build_manifest(root: Path) -> dict[str, Any]:
    """Derive distribution metadata only from the two recorded freeze manifests."""
    entries = []
    for task, provenance in SOURCES:
        source = f"reports/final_{task}_manifest.json"
        frozen = json.loads((root / source).read_text(encoding="utf-8"))
        if frozen["status"] != "FROZEN" or frozen["selection_status"] != "FINAL_SELECTED":
            raise ValueError("SCIENTIFIC_STATE_MISMATCH: final model is not frozen")
        checkpoint = frozen["selected_checkpoint"]
        for field in ("sha256", "size_bytes"):
            if checkpoint[field] != frozen["frozen_copy"][field]:
                raise ValueError("SCIENTIFIC_STATE_MISMATCH: checkpoint identities differ")
        entries.append(
            {
                "logical_name": frozen["selected_experiment"],
                "architecture": frozen["model"],
                "imgsz": frozen["imgsz"],
                "sha256": checkpoint["sha256"],
                "bytes": checkpoint["size_bytes"],
                "status": frozen["status"],
                "distribution_status": REVIEW_REQUIRED,
                "download_locator": None,
                "freeze_manifest": source,
                "training_provenance": provenance,
                "experiment_sha256": frozen["experiment_sha256"],
                "license_notice": "delivery/LICENSING.md",
            }
        )
    payload = {"schema_version": 1, "checkpoints": entries}
    validate_manifest(payload)
    return payload


def validate_manifest(payload: Any) -> None:
    """Reject unknown fields, invalid identity data and premature download locators."""
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "checkpoints"}:
        raise ValueError("invalid checkpoint manifest fields")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError("unsupported checkpoint schema version")
    entries = payload["checkpoints"]
    if not isinstance(entries, list) or len(entries) != 2:
        raise ValueError("exactly two frozen checkpoints are required")
    expected = {
        "logical_name",
        "architecture",
        "imgsz",
        "sha256",
        "bytes",
        "status",
        "distribution_status",
        "download_locator",
        "freeze_manifest",
        "training_provenance",
        "experiment_sha256",
        "license_notice",
    }
    identities = {"D2": "YOLO11n", "S1": "YOLO11n-seg"}
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != expected:
            raise ValueError("invalid checkpoint entry fields")
        name = entry["logical_name"]
        if not isinstance(name, str) or name not in identities or name in seen:
            raise ValueError("unknown or duplicate checkpoint identity")
        seen.add(name)
        if (
            entry["architecture"] != identities[name]
            or type(entry["imgsz"]) is not int
            or entry["imgsz"] != 768
        ):
            raise ValueError("architecture or input size mismatch")
        for field in ("sha256", "experiment_sha256"):
            if not isinstance(entry[field], str) or not re.fullmatch("[0-9a-f]{64}", entry[field]):
                raise ValueError(f"invalid {field}")
        if type(entry["bytes"]) is not int or entry["bytes"] <= 0:
            raise ValueError("checkpoint byte size must be a positive integer")
        if entry["status"] != "FROZEN" or entry["distribution_status"] != REVIEW_REQUIRED:
            raise ValueError("unsupported freeze or distribution status")
        if entry["download_locator"] is not None:
            raise ValueError("no download locator is approved in this schema version")
        for field in ("freeze_manifest", "training_provenance", "license_notice"):
            value = entry[field]
            if (
                not isinstance(value, str)
                or not re.fullmatch(r"(?:reports|delivery)/[A-Za-z0-9_.-]+", value)
                or ".." in value
            ):
                raise ValueError(f"{field} must be a repository-relative reference")


def load_manifest(path: Path) -> dict[str, Any]:
    """Load the strict public metadata schema; never open a model file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_manifest(payload)
    return payload


def verify_file(path: Path, *, sha256: str, size_bytes: int) -> None:
    """Verify explicit file bytes without loading a checkpoint or substituting it.

    Raises:
        ValueError: If the expected identity or the supplied file does not match.
        OSError: If the supplied file cannot be read.
    """
    if not re.fullmatch("[0-9a-f]{64}", sha256):
        raise ValueError("invalid expected SHA-256")
    if type(size_bytes) is not int or size_bytes <= 0:
        raise ValueError("invalid expected byte size")
    if not path.is_file() or path.stat().st_size != size_bytes:
        raise ValueError("CHECKPOINT_SIZE_MISMATCH_OR_MISSING")
    if sha256_file(path) != sha256:
        raise ValueError("CHECKPOINT_SHA256_MISMATCH")
