"""Split identifiers and the holdout protection guard.

The experimental protocol treats ``test`` as a locked holdout: it may only be
read once, after the detection and segmentation models are frozen. This module
centralises that rule so that every future data loader, evaluation script and
notebook goes through the same check instead of re-implementing it.

The guard is a guardrail against accidental leakage during development, not a
security boundary: anyone can set the environment variable. Its value is that
touching the holdout becomes a deliberate, auditable act.
"""

from __future__ import annotations

import os
from enum import StrEnum

HOLDOUT_UNLOCK_ENV_VAR = "CSVISION_ALLOW_TEST_SPLIT"
"""Environment variable that unlocks the holdout split."""

HOLDOUT_UNLOCK_VALUE = "1"
"""The only value that unlocks the holdout split."""


class HoldoutViolationError(RuntimeError):
    """Raised when protected holdout data is requested without an explicit unlock."""


class Split(StrEnum):
    """Canonical dataset split names.

    The same identifiers are used by the detection and the segmentation
    pipelines: both tasks are derived from one canonical source and must share
    image IDs split by split.
    """

    TRAIN = "train"
    VAL = "val"
    TEST = "test"

    @classmethod
    def parse(cls, value: str | Split) -> Split:
        """Normalise a split name, accepting the usual external aliases.

        ``valid`` and ``validation`` (used by common dataset exports) map to
        :attr:`VAL`.

        Args:
            value: Split name to normalise.

        Returns:
            The canonical split.

        Raises:
            ValueError: If the name does not map to a known split.
        """
        if isinstance(value, cls):
            return value
        normalised = str(value).strip().lower()
        aliases = {
            "train": cls.TRAIN,
            "training": cls.TRAIN,
            "val": cls.VAL,
            "valid": cls.VAL,
            "validation": cls.VAL,
            "test": cls.TEST,
            "testing": cls.TEST,
            "holdout": cls.TEST,
        }
        try:
            return aliases[normalised]
        except KeyError:
            msg = f"Unknown split {value!r}; expected one of {sorted(aliases)}"
            raise ValueError(msg) from None


DEVELOPMENT_SPLITS: tuple[Split, ...] = (Split.TRAIN, Split.VAL)
"""Splits that may be freely used while developing and tuning models."""

PROTECTED_SPLITS: tuple[Split, ...] = (Split.TEST,)
"""Splits locked until the final one-shot evaluation."""


def holdout_unlocked(env: dict[str, str] | None = None) -> bool:
    """Report whether the environment currently unlocks the holdout split.

    Args:
        env: Environment mapping to read. Defaults to ``os.environ``.

    Returns:
        ``True`` when the unlock variable is set to ``"1"``.
    """
    environment = os.environ if env is None else env
    return environment.get(HOLDOUT_UNLOCK_ENV_VAR, "").strip() == HOLDOUT_UNLOCK_VALUE


def assert_split_allowed(
    split: str | Split,
    *,
    purpose: str,
    allow_test: bool = False,
    env: dict[str, str] | None = None,
) -> Split:
    """Authorise access to a split, enforcing the holdout protocol.

    Access to :attr:`Split.TEST` requires two independent opt-ins: the caller
    must pass ``allow_test=True`` in code, and the environment must set
    ``CSVISION_ALLOW_TEST_SPLIT=1``. Either one alone is refused, so a stray
    environment variable cannot silently unlock a training script and a stray
    flag cannot silently unlock a shell session.

    Args:
        split: Split being requested.
        purpose: Short human-readable reason, recorded in the error message and
            intended to be logged by callers.
        allow_test: Explicit in-code opt-in for the holdout split.
        env: Environment mapping to read. Defaults to ``os.environ``.

    Returns:
        The normalised split.

    Raises:
        HoldoutViolationError: If a protected split is requested without both
            opt-ins.
    """
    resolved = Split.parse(split)
    if resolved not in PROTECTED_SPLITS:
        return resolved

    if not allow_test:
        msg = (
            f"Split {resolved.value!r} is a locked holdout and was requested for "
            f"{purpose!r} without an explicit in-code opt-in (allow_test=True)."
        )
        raise HoldoutViolationError(msg)

    if not holdout_unlocked(env):
        msg = (
            f"Split {resolved.value!r} is a locked holdout and was requested for "
            f"{purpose!r} without {HOLDOUT_UNLOCK_ENV_VAR}={HOLDOUT_UNLOCK_VALUE}. "
            "Unlock it only for the final one-shot test evaluation."
        )
        raise HoldoutViolationError(msg)

    return resolved
