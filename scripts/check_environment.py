"""Report the runtime environment and the state of the holdout lock.

Run this first in a fresh local shell or Colab runtime to confirm that the
repository resolves, the configuration parses, and the test split is still
locked.

Usage:
    uv run python scripts/check_environment.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from construction_safety_vision import __version__
from construction_safety_vision.config import ConfigError, load_experiment_config
from construction_safety_vision.env import load_project_env
from construction_safety_vision.paths import (
    ProjectPaths,
    ProjectRootNotFoundError,
    find_project_root,
    in_colab,
)
from construction_safety_vision.provenance import git_commit, runtime_environment
from construction_safety_vision.splits import HOLDOUT_UNLOCK_ENV_VAR, holdout_unlocked

OPTIONAL_PACKAGES = ("numpy", "torch", "torchvision", "ultralytics", "cv2")
"""Heavy packages installed only by the phases that need them."""


def _package_version(name: str) -> str:
    """Return an installed package version, or a not-installed marker.

    Args:
        name: Importable module name.

    Returns:
        The version string, ``"installed"`` when no version is exposed, or
        ``"not installed"``.
    """
    try:
        module = __import__(name)
    except ImportError:
        return "not installed"
    return str(getattr(module, "__version__", "installed"))


def main(argv: list[str] | None = None) -> int:
    """Print the environment report.

    Args:
        argv: Command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code: ``0`` on success, ``1`` when the repository root or
        the configuration cannot be resolved.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Configuration file to validate (default: <root>/configs/project.yaml).",
    )
    args = parser.parse_args(argv)

    print(f"construction_safety_vision {__version__}")
    for key, value in runtime_environment().items():
        print(f"  {key}: {value}")
    print(f"  google_colab: {in_colab()}")

    try:
        root = find_project_root()
    except ProjectRootNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    paths = ProjectPaths.from_root(root)
    print(f"  project_root: {paths.root}")
    # Names only: a value must never reach stdout.
    print(f"  .env supplied: {load_project_env(paths.root) or 'nothing'}")
    print(f"  git_commit: {git_commit(paths.root) or 'unavailable'}")

    config_path = args.config or (paths.configs / "project.yaml")
    try:
        config = load_experiment_config(config_path)
    except ConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"  config: {config_path} (seed={config.seed})")
    print(f"  dataset: {config.dataset.name} (verified={config.dataset.verified})")

    unlocked = holdout_unlocked()
    state = "UNLOCKED" if unlocked else "locked"
    print(f"  holdout ({config.holdout_split}): {state}")
    if unlocked:
        print(
            f"  WARNING: {HOLDOUT_UNLOCK_ENV_VAR} is set. Unset it unless this is "
            "the final one-shot test evaluation.",
            file=sys.stderr,
        )

    print("optional packages:")
    for name in OPTIONAL_PACKAGES:
        print(f"  {name}: {_package_version(name)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
