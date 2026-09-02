"""Tests for the phase 4B manual-audit record.

The record carries human judgements, so nothing here can check whether a
judgement is right. What it can check is that a judgement cannot be attached to
the wrong subject, cannot be spelled in a way nobody defined, and cannot be
renumbered by re-running the recorder. Those are the failure modes that would
silently corrupt the audit trail.
"""

from __future__ import annotations

import pytest

from construction_safety_vision.data.manualaudit import (
    DUPLICATE_GROUP_PREFIX,
    FIELDNAMES,
    ManualAuditError,
    ManualDecision,
    assign_duplicate_group_ids,
    read_decisions,
    resolve_short_id,
    review_pool,
    validate_decisions,
    write_decisions,
)

POOL = {
    "9b7VLDWD": "9b7VLDWDFhy21Ygmz7ss",
    "KgOOMlBE": "KgOOMlBEXp0QxhSxTT6a",
    "KqRSROf9": "KqRSROf90W22N1qzQzN0",
    "wpbZJyUR": "wpbZJyURzZHruvXGMRtd",
}


def decision(**overrides: str) -> ManualDecision:
    """Build a valid decision, overriding selected fields."""
    fields = {
        "decision_id": "dup-001",
        "review_type": "cross_split_near_duplicate",
        "subject_scope": "pair",
        "decision": "EXACT_SEMANTIC_DUPLICATE",
        "confidence": "HIGH",
        "rationale": "same content",
        "phase5_action": "GROUP_TOGETHER",
        "group_id": "manual_dup_001",
        "subject_id_a": "aaa",
        "subject_id_b": "bbb",
        "provider_split_a": "train",
        "provider_split_b": "valid",
    }
    fields.update(overrides)
    return ManualDecision(**fields)


SOURCE_INDEX = {
    "aaa": {"split": "train"},
    "bbb": {"split": "valid"},
    "ccc": {"split": "test"},
}
REVIEW_INDEX = {
    "aaa": [{"contact_sheet": "reports/figures/review_h_near_duplicates.jpg"}],
    "bbb": [{"contact_sheet": "reports/figures/review_h_near_duplicates.jpg"}],
    "ccc": [{"contact_sheet": "reports/figures/review_f_zero_instance.jpg"}],
}


def check(decisions, root):
    """Validate decisions against the synthetic indexes."""
    return validate_decisions(
        [item.as_row() for item in decisions],
        source_index=SOURCE_INDEX,
        review_index=REVIEW_INDEX,
        root=root,
    )


# --- short-id resolution -------------------------------------------------


def test_resolve_short_id_is_case_insensitive():
    match = resolve_short_id("9b7vLDWD", POOL)
    assert match.image_id == "9b7VLDWDFhy21Ygmz7ss"
    assert match.distance == 0


def test_resolve_short_id_folds_confusable_characters():
    # A reviewer reading a caption writes zero for the letter O.
    match = resolve_short_id("KqRSR0F9", POOL)
    assert match.image_id == "KqRSROf90W22N1qzQzN0"
    assert match.distance == 0


def test_resolve_short_id_tolerates_a_small_transcription_error():
    match = resolve_short_id("KgOOOMBE", POOL)
    assert match.image_id == "KgOOMlBEXp0QxhSxTT6a"
    assert match.distance <= 2
    assert match.runner_up_distance - match.distance >= 2


def test_resolve_short_id_rejects_a_transcription_that_matches_nothing():
    with pytest.raises(ManualAuditError, match="cannot resolve"):
        resolve_short_id("ZZZZZZZZ", POOL)


def test_resolve_short_id_rejects_an_ambiguous_transcription():
    # Two candidates one edit apart: the transcription cannot decide between them.
    pool = {"abcdefgh": "image-one", "abcdefgi": "image-two"}
    with pytest.raises(ManualAuditError, match="ambiguous"):
        resolve_short_id("abcdefgx", pool)


def test_resolve_short_id_rejects_an_empty_pool():
    with pytest.raises(ManualAuditError, match="empty"):
        resolve_short_id("9b7vLDWD", {})


def test_review_pool_is_restricted_to_one_review_reason():
    index = {
        "one": [{"short_id": "AAA", "reason": "zero_instance"}],
        "two": [{"short_id": "BBB", "reason": "crowded_scene"}],
    }
    assert review_pool(index, "zero_instance") == {"AAA": "one"}


# --- duplicate groups ----------------------------------------------------


def test_group_ids_depend_only_on_sorted_members():
    forward = assign_duplicate_group_ids([("b", "a"), ("d", "c")])
    reversed_declaration = assign_duplicate_group_ids([("c", "d"), ("a", "b")])
    assert forward == reversed_declaration
    assert forward[("a", "b")] == f"{DUPLICATE_GROUP_PREFIX}001"
    assert forward[("c", "d")] == f"{DUPLICATE_GROUP_PREFIX}002"


def test_group_rejects_a_repeated_member():
    with pytest.raises(ManualAuditError, match="repeats a member"):
        assign_duplicate_group_ids([("a", "a")])


def test_group_rejects_fewer_than_two_members():
    with pytest.raises(ManualAuditError, match="at least two"):
        assign_duplicate_group_ids([("a",)])


def test_group_rejects_the_same_group_declared_twice():
    with pytest.raises(ManualAuditError, match="more than once"):
        assign_duplicate_group_ids([("a", "b"), ("b", "a")])


# --- validation ----------------------------------------------------------


def test_a_well_formed_record_validates(tmp_path):
    assert check([decision()], tmp_path) == []


def test_unknown_vocabulary_is_rejected(tmp_path):
    problems = check([decision(decision="LOOKS_FINE_TO_ME")], tmp_path)
    assert any("controlled vocabulary" in problem for problem in problems)


def test_an_unknown_subject_is_rejected(tmp_path):
    problems = check([decision(subject_id_b="zzz")], tmp_path)
    assert any("not in the source manifest" in problem for problem in problems)


def test_a_contradicted_split_is_rejected(tmp_path):
    problems = check([decision(provider_split_a="test")], tmp_path)
    assert any("contradicts the manifest" in problem for problem in problems)


def test_a_subject_absent_from_the_cited_figure_is_rejected(tmp_path):
    problems = check(
        [decision(evidence_figure="reports/figures/review_f_zero_instance.jpg")], tmp_path
    )
    assert any("does not appear on" in problem for problem in problems)


def test_a_missing_evidence_figure_is_rejected(tmp_path):
    problems = check([decision(evidence_figure="reports/figures/nope.jpg")], tmp_path)
    assert any("does not exist" in problem for problem in problems)


def test_a_set_level_row_may_not_name_a_subject(tmp_path):
    problems = check([decision(subject_scope="set")], tmp_path)
    assert any("must not name subject_id_a" in problem for problem in problems)


def test_a_pair_naming_one_image_twice_is_rejected(tmp_path):
    problems = check([decision(subject_id_b="aaa", provider_split_b="train")], tmp_path)
    assert any("names the same image twice" in problem for problem in problems)


def test_a_group_with_a_single_member_is_rejected(tmp_path):
    lone = decision(
        subject_scope="image",
        subject_id_b="",
        provider_split_b="",
    )
    problems = check([lone], tmp_path)
    assert any("fewer than two member ids" in problem for problem in problems)


def test_a_repeated_decision_id_is_rejected(tmp_path):
    problems = check([decision(), decision(group_id="manual_dup_002")], tmp_path)
    assert any("not unique" in problem for problem in problems)


def test_an_empty_rationale_is_rejected(tmp_path):
    problems = check([decision(rationale="   ")], tmp_path)
    assert any("rationale is empty" in problem for problem in problems)


def test_an_absolute_path_is_rejected(tmp_path):
    problems = check([decision(notes="see C:/Users/someone/scratch.txt")], tmp_path)
    assert any("absolute path" in problem for problem in problems)


def test_credential_material_is_rejected(tmp_path):
    problems = check([decision(notes="ROBOFLOW_API_KEY=redacted-looking-value")], tmp_path)
    assert any("credential material" in problem for problem in problems)


def test_a_signed_url_is_rejected(tmp_path):
    problems = check(
        [decision(notes="https://example.invalid/img.jpg?X-Amz-Signature=abc")], tmp_path
    )
    assert any("credential material" in problem for problem in problems)


def test_an_empty_record_is_rejected(tmp_path):
    assert validate_decisions(
        [], source_index=SOURCE_INDEX, review_index=REVIEW_INDEX, root=tmp_path
    ) == ["no decisions were recorded"]


# --- round trip ----------------------------------------------------------


def test_written_decisions_read_back_unchanged(tmp_path):
    path = tmp_path / "decisions.csv"
    original = decision(notes="dhash=1; phash=0")
    write_decisions(path, [original])
    rows = read_decisions(path)
    assert rows == [original.as_row()]
    assert tuple(rows[0]) == FIELDNAMES


def test_an_unexpected_header_is_rejected(tmp_path):
    path = tmp_path / "decisions.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ManualAuditError, match="unexpected header"):
        read_decisions(path)
