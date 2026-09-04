"""Tests for the canonical-snapshot decision record.

The manifest is the evidence a reader checks the project's headline numbers
against, and it is committed to a public repository. So it is validated for
internal consistency and scanned for anything that must never be published.
"""

from __future__ import annotations

import json

import pytest

from construction_safety_vision.data.canonical import (
    CURRENT_COMPLETE_GEOMETRY,
    FROZEN_V4_SNAPSHOT,
    PHASE_CLASSIFICATION,
    CanonicalDecisionError,
    assert_manifest_committable,
    scan_for_sensitive,
    validate_manifest,
)


def manifest(**overrides: object) -> dict:
    """Build a minimal well-formed manifest.

    Args:
        **overrides: Fields to replace.

    Returns:
        The manifest.
    """
    document = {
        "decision": CURRENT_COMPLETE_GEOMETRY,
        "phase_classification": PHASE_CLASSIFICATION[CURRENT_COMPLETE_GEOMETRY],
        "source_project": {"workspace": "w", "project": "p"},
        "source_image_count": 436,
        "canonical_snapshot": {"state": "live"},
        "annotation_count": 2031,
        "geometry_representation_counts": {
            "POLYGON": 1007,
            "RLE": 1022,
            "BITMASK_OR_MASK_ASSET": 0,
            "OTHER_SUPPORTED": 0,
            "UNSUPPORTED": 2,
        },
        "category_map": {"person": 0},
        "acquisition_method": {"endpoint": "GET images"},
        "artifact_fingerprints": {"reports/x.json": "abc"},
        "drift_summary": {"net_delta": 70},
        "limitations": ["something honest"],
        "git_commit": "deadbeef",
    }
    document.update(overrides)
    return document


def test_a_well_formed_manifest_validates() -> None:
    assert validate_manifest(manifest()) == []


def test_a_missing_required_field_is_a_defect() -> None:
    incomplete = manifest()
    del incomplete["limitations"]
    problems = validate_manifest(incomplete)
    assert any("limitations" in problem for problem in problems)


def test_an_unknown_decision_is_rejected() -> None:
    problems = validate_manifest(manifest(decision="PROBABLY_FINE"))
    assert any("not one of" in problem for problem in problems)


def test_the_phase_classification_must_match_the_decision() -> None:
    problems = validate_manifest(
        manifest(decision=FROZEN_V4_SNAPSHOT, phase_classification="COMPLETE_CURRENT")
    )
    assert any("does not match" in problem for problem in problems)


def test_geometry_counts_that_do_not_sum_to_the_total_are_rejected() -> None:
    # This is the check that catches an annotation being silently dropped.
    problems = validate_manifest(
        manifest(geometry_representation_counts={"POLYGON": 1007, "RLE": 1022})
    )
    assert any("dropped or double counted" in problem for problem in problems)


def test_unsupported_annotations_must_be_counted_in_the_total() -> None:
    # Omitting the two geometry-less records would make the counts sum short.
    counts = {"POLYGON": 1007, "RLE": 1022, "UNSUPPORTED": 0}
    problems = validate_manifest(manifest(geometry_representation_counts=counts))
    assert problems


def test_a_signed_url_is_refused() -> None:
    problems = scan_for_sensitive(
        '{"link": "https://storage.example.com/x.zip?X-Amz-Signature=ab"}'
    )
    assert any("query string" in problem for problem in problems)


def test_a_plain_public_url_is_allowed() -> None:
    # The manifest legitimately records the dataset's public page. The scan must
    # not mistake the "s:/" inside "https://" for a Windows drive letter.
    assert scan_for_sensitive('{"public_url": "https://universe.roboflow.com/ws/project"}') == []


def test_a_windows_path_is_refused() -> None:
    problems = scan_for_sensitive('{"path": "C:\\\\Users\\\\someone\\\\project"}')
    assert any("filesystem path" in problem for problem in problems)


def test_a_posix_home_path_is_refused() -> None:
    problems = scan_for_sensitive('{"path": "/home/someone/project"}')
    assert any("filesystem path" in problem for problem in problems)


def test_a_credential_assignment_is_refused() -> None:
    problems = scan_for_sensitive('api_key = "abcdef1234567890"')
    assert any("credential" in problem for problem in problems)


def test_the_committed_manifest_round_trips_through_json() -> None:
    document = manifest()
    serialised = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True)
    assert json.loads(serialised) == document
    assert_manifest_committable(document, serialised)


def test_an_unsafe_manifest_is_refused_rather_than_written() -> None:
    document = manifest(canonical_snapshot={"path": "/Users/someone/data"})
    serialised = json.dumps(document)
    with pytest.raises(CanonicalDecisionError, match="not fit to commit"):
        assert_manifest_committable(document, serialised)


def test_every_decision_has_a_phase_classification() -> None:
    from construction_safety_vision.data.canonical import DECISIONS

    assert set(DECISIONS) == set(PHASE_CLASSIFICATION)
