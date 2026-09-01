"""Tests for the holdout protection guard.

These are the tests that matter most in the foundation phase: the experimental
protocol is only credible if the rule protecting the test split is enforced by
code rather than by discipline.
"""

from __future__ import annotations

import pytest

from construction_safety_vision.splits import (
    DEVELOPMENT_SPLITS,
    HOLDOUT_UNLOCK_ENV_VAR,
    PROTECTED_SPLITS,
    HoldoutViolationError,
    Split,
    assert_split_allowed,
    holdout_unlocked,
)

UNLOCKED_ENV = {HOLDOUT_UNLOCK_ENV_VAR: "1"}
LOCKED_ENV: dict[str, str] = {}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("train", Split.TRAIN),
        ("training", Split.TRAIN),
        ("val", Split.VAL),
        ("valid", Split.VAL),
        ("validation", Split.VAL),
        ("  TEST  ", Split.TEST),
        ("holdout", Split.TEST),
        (Split.VAL, Split.VAL),
    ],
)
def test_parse_accepts_known_aliases(raw: str | Split, expected: Split) -> None:
    assert Split.parse(raw) is expected


def test_parse_rejects_unknown_split() -> None:
    with pytest.raises(ValueError, match="Unknown split"):
        Split.parse("trainval")


def test_split_is_its_own_string_value() -> None:
    # Downstream code interpolates splits into paths and dataset YAML files.
    assert f"{Split.TRAIN}" == "train"
    assert Split.VAL.value == "val"


def test_protected_and_development_splits_are_disjoint_and_complete() -> None:
    assert set(DEVELOPMENT_SPLITS).isdisjoint(PROTECTED_SPLITS)
    assert set(DEVELOPMENT_SPLITS) | set(PROTECTED_SPLITS) == set(Split)


@pytest.mark.parametrize("split", DEVELOPMENT_SPLITS)
def test_development_splits_are_always_allowed(split: Split) -> None:
    assert assert_split_allowed(split, purpose="training", env=LOCKED_ENV) is split


def test_test_split_refused_without_code_opt_in() -> None:
    with pytest.raises(HoldoutViolationError, match="allow_test=True"):
        assert_split_allowed("test", purpose="training", env=UNLOCKED_ENV)


def test_test_split_refused_without_environment_unlock() -> None:
    with pytest.raises(HoldoutViolationError, match=HOLDOUT_UNLOCK_ENV_VAR):
        assert_split_allowed("test", purpose="final evaluation", allow_test=True, env=LOCKED_ENV)


def test_test_split_allowed_only_with_both_opt_ins() -> None:
    resolved = assert_split_allowed(
        "test",
        purpose="final evaluation",
        allow_test=True,
        env=UNLOCKED_ENV,
    )
    assert resolved is Split.TEST


def test_violation_message_reports_the_purpose() -> None:
    with pytest.raises(HoldoutViolationError, match="hyperparameter search"):
        assert_split_allowed("test", purpose="hyperparameter search", env=LOCKED_ENV)


@pytest.mark.parametrize("value", ["", "0", "true", "yes", "1 1"])
def test_only_the_exact_unlock_value_unlocks(value: str) -> None:
    assert holdout_unlocked({HOLDOUT_UNLOCK_ENV_VAR: value}) is False


def test_unlock_value_tolerates_surrounding_whitespace() -> None:
    assert holdout_unlocked({HOLDOUT_UNLOCK_ENV_VAR: " 1 "}) is True


def test_guard_reads_process_environment_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(HOLDOUT_UNLOCK_ENV_VAR, raising=False)
    assert holdout_unlocked() is False
    with pytest.raises(HoldoutViolationError):
        assert_split_allowed(Split.TEST, purpose="smoke test", allow_test=True)

    monkeypatch.setenv(HOLDOUT_UNLOCK_ENV_VAR, "1")
    assert holdout_unlocked() is True
    assert assert_split_allowed(Split.TEST, purpose="smoke test", allow_test=True) is Split.TEST
