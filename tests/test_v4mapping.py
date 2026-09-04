"""Tests for mapping the version-4 export back onto the source images.

Identity resolution and pristine-representation selection are the two claims the
canonical decision rests on, so both are tested on synthetic inputs. Nothing here
reads the export or the network.
"""

from __future__ import annotations

import numpy as np
import pytest

from construction_safety_vision.data.v4mapping import (
    AMBIGUOUS,
    EXACT,
    HIGH,
    NORMALISED,
    MappingError,
    SourceMapping,
    V4Representation,
    choose_representation,
    mean_absolute_error,
    normalise_name,
    resolve_identity,
)

MOJIBAKE = "wm_0044_Å pejchar_zastÃ¡vka_linky_745.jpg"  # noqa: RUF001
"""How the source project stores a name whose UTF-8 bytes were read as latin-1.

The literal contains a no-break space, which the linter flags as an ambiguous
character. That is the point and the rule is suppressed rather than the value
"cleaned": this invisible character is precisely what makes the exact filename
match fail for this one image, so a test using an ordinary space would prove
nothing.
"""

ASCII_FOLDED = "wm_0044_Spejchar_zastavka_linky_745.jpg"
"""How the exporter writes the same name after folding it to ASCII."""


def representation(name: str, mae: float | None, *, split: str = "train") -> V4Representation:
    """Build an export record.

    Args:
        name: Original source filename.
        mae: Score against the deterministic re-render.
        split: Export split.

    Returns:
        The record.
    """
    return V4Representation(
        split=split,
        file_name=f"{name}.rf.{mae}.jpg",
        coco_image_id=int((mae or 0) * 100),
        source_name=name,
        mae=mae,
    )


def test_normalise_name_bridges_the_mojibake_and_ascii_forms() -> None:
    # One real filename differs between the two sides only by this transform.
    assert normalise_name(MOJIBAKE) == normalise_name(ASCII_FOLDED)


def test_normalise_name_keeps_genuinely_different_names_apart() -> None:
    assert normalise_name("worker_01.jpg") != normalise_name("worker_02.jpg")


def test_identity_prefers_an_exact_name_match() -> None:
    resolved, unmatched, unclaimed = resolve_identity(
        {"a": "one.jpg", "b": "two.jpg"}, {"one.jpg", "two.jpg"}
    )
    assert resolved == {"a": ("one.jpg", EXACT), "b": ("two.jpg", EXACT)}
    assert not unmatched
    assert not unclaimed


def test_identity_falls_back_to_normalisation_only_for_leftovers() -> None:
    resolved, unmatched, unclaimed = resolve_identity(
        {"a": "one.jpg", "b": MOJIBAKE}, {"one.jpg", ASCII_FOLDED}
    )
    assert resolved["a"] == ("one.jpg", EXACT)
    assert resolved["b"] == (ASCII_FOLDED, NORMALISED)
    assert not unmatched
    assert not unclaimed


def test_identity_refuses_to_guess_when_normalisation_is_ambiguous() -> None:
    with pytest.raises(MappingError, match="refusing to guess"):
        resolve_identity({"a": "a-b.jpg"}, {"a_b.jpg", "ab.jpg"})


def test_identity_reports_what_it_could_not_match() -> None:
    resolved, unmatched, unclaimed = resolve_identity({"a": "missing.jpg"}, {"other.jpg"})
    assert resolved == {}
    assert unmatched == ["a"]
    assert unclaimed == ["other.jpg"]


def test_a_single_close_representation_resolves_with_high_confidence() -> None:
    selected, confidence, _ = choose_representation([representation("v.jpg", 2.0, split="valid")])
    assert selected is not None
    assert selected.mae == 2.0
    assert confidence == HIGH


def test_the_pristine_member_of_a_pair_is_chosen() -> None:
    pair = [representation("t.jpg", 54.0), representation("t.jpg", 1.5)]
    selected, confidence, note = choose_representation(pair)
    assert selected is not None
    assert selected.mae == 1.5
    assert confidence == HIGH
    assert "augmented" in note


def test_a_pair_that_is_not_separated_is_refused_rather_than_guessed() -> None:
    # Two similar scores mean the evidence cannot tell them apart. Picking the
    # lower one anyway would silently make an augmented copy canonical.
    selected, confidence, _ = choose_representation(
        [representation("t.jpg", 2.0), representation("t.jpg", 2.4)]
    )
    assert selected is not None
    assert confidence == AMBIGUOUS


def test_a_candidate_worse_than_the_pristine_threshold_is_flagged() -> None:
    selected, confidence, note = choose_representation([representation("t.jpg", 60.0)])
    assert selected is not None
    assert confidence == AMBIGUOUS
    assert "pristine threshold" in note


def test_no_scored_candidate_selects_nothing() -> None:
    selected, confidence, _ = choose_representation([representation("t.jpg", None)])
    assert selected is None
    assert confidence == AMBIGUOUS


def test_mean_absolute_error_is_zero_for_identical_images() -> None:
    array = np.zeros((4, 4, 3), dtype=np.float32)
    assert mean_absolute_error(array, array.copy()) == 0.0


def test_mean_absolute_error_rejects_mismatched_shapes() -> None:
    with pytest.raises(MappingError, match="Cannot compare"):
        mean_absolute_error(np.zeros((4, 4, 3)), np.zeros((5, 5, 3)))


def test_a_pristine_selection_is_not_marked_augmented() -> None:
    chosen = representation("t.jpg", 1.2)
    mapping = SourceMapping(
        source_image_id="abc",
        source_name="t.jpg",
        candidates=[chosen, representation("t.jpg", 60.0)],
        selected=chosen,
        identity_method=EXACT,
        confidence=HIGH,
    )
    assert not mapping.is_augmented_selection
    assert mapping.csv_row()["is_augmented"] == "false"


def test_a_selection_above_the_threshold_is_marked_augmented() -> None:
    chosen = representation("t.jpg", 55.0)
    mapping = SourceMapping(
        source_image_id="abc",
        source_name="t.jpg",
        candidates=[chosen],
        selected=chosen,
        identity_method=EXACT,
        confidence=AMBIGUOUS,
    )
    assert mapping.is_augmented_selection
    assert mapping.csv_row()["is_augmented"] == "true"


def test_csv_row_carries_no_local_path() -> None:
    chosen = representation("t.jpg", 1.2)
    row = SourceMapping(
        source_image_id="abc",
        source_name="t.jpg",
        candidates=[chosen],
        selected=chosen,
        identity_method=EXACT,
        confidence=HIGH,
    ).csv_row()
    assert not any(":" in str(value) and "\\" in str(value) for value in row.values())
