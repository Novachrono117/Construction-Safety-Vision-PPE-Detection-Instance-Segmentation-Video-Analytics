"""Tests for repository layout resolution."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from construction_safety_vision.paths import (
    PROJECT_ROOT_ENV_VAR,
    WINDOWS_EXTENDED_PREFIX,
    WINDOWS_MAX_PATH,
    ProjectPaths,
    ProjectRootNotFoundError,
    find_project_root,
    long_path,
)


def make_fake_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "src" / "pkg").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\n", encoding="utf-8", newline="\n")
    return root


def test_finds_root_by_walking_up_from_a_nested_directory(tmp_path: Path) -> None:
    root = make_fake_repo(tmp_path)

    assert find_project_root(root / "src" / "pkg", env={}) == root.resolve()


def test_finds_root_when_starting_from_a_file(tmp_path: Path) -> None:
    root = make_fake_repo(tmp_path)
    module = root / "src" / "pkg" / "module.py"
    module.write_text("", encoding="utf-8", newline="\n")

    assert find_project_root(module, env={}) == root.resolve()


def test_environment_override_wins_over_discovery(tmp_path: Path) -> None:
    root = make_fake_repo(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    resolved = find_project_root(root / "src", env={PROJECT_ROOT_ENV_VAR: str(elsewhere)})

    assert resolved == elsewhere.resolve()


def test_environment_override_pointing_nowhere_is_an_error(tmp_path: Path) -> None:
    missing = tmp_path / "missing"

    with pytest.raises(ProjectRootNotFoundError, match=PROJECT_ROOT_ENV_VAR):
        find_project_root(env={PROJECT_ROOT_ENV_VAR: str(missing)})


def test_missing_marker_is_an_error(tmp_path: Path) -> None:
    # tmp_path has no pyproject.toml above it inside the temporary tree; the
    # search must fail loudly instead of silently returning a wrong directory.
    orphan = tmp_path / "orphan"
    orphan.mkdir()

    with pytest.raises(ProjectRootNotFoundError, match="No project root marker"):
        find_project_root(orphan, env={PROJECT_ROOT_ENV_VAR: ""})


def test_layout_is_derived_from_the_root(tmp_path: Path) -> None:
    root = make_fake_repo(tmp_path)
    paths = ProjectPaths.from_root(root)

    assert paths.root == root.resolve()
    assert paths.configs == root.resolve() / "configs"
    assert paths.data_processed == root.resolve() / "data" / "processed"
    assert paths.figures == paths.reports / "figures"


def test_ensure_data_dirs_is_idempotent(tmp_path: Path) -> None:
    paths = ProjectPaths.from_root(make_fake_repo(tmp_path))

    paths.ensure_data_dirs()
    paths.ensure_data_dirs()

    assert all(directory.is_dir() for directory in paths.data_dirs())


def test_real_repository_layout_exists() -> None:
    # Guards against a directory being renamed without updating the code.
    paths = ProjectPaths.from_root()

    for directory in (paths.configs, paths.notebooks, paths.reports, paths.scripts):
        assert directory.is_dir(), f"missing repository directory: {directory}"


def test_a_short_path_is_returned_unchanged() -> None:
    assert long_path("data/raw/thing.jpg") == "data/raw/thing.jpg"


@pytest.mark.skipif(os.name != "nt", reason="the MAX_PATH ceiling is Windows-only")
def test_a_long_windows_path_gains_the_extended_prefix() -> None:
    # 137 of the export's 742 images exceed the Win32 limit once nested under a
    # project directory with a long name; without the prefix they list fine and
    # open never. scripts/map_v4_sources.py reads every one of them, so the
    # end-to-end behaviour is exercised there; what is checked here is the rule.
    result = long_path(Path("C:/") / ("x" * 300) / "image.jpg")

    assert result.startswith(WINDOWS_EXTENDED_PREFIX)
    assert len(result) > WINDOWS_MAX_PATH


@pytest.mark.skipif(os.name != "nt", reason="the MAX_PATH ceiling is Windows-only")
def test_an_already_prefixed_path_is_not_prefixed_twice() -> None:
    already = WINDOWS_EXTENDED_PREFIX + "C:" + chr(92) + "y" * 300 + chr(92) + "image.jpg"

    assert long_path(already) == already


@pytest.mark.skipif(os.name != "nt", reason="the MAX_PATH ceiling is Windows-only")
def test_a_path_just_under_the_limit_is_left_alone() -> None:
    # Applying the prefix everywhere would also disable path normalisation, so
    # it is applied only where the limit actually bites.
    root = "C:/"
    padding = WINDOWS_MAX_PATH - len(root) - len("/f.jpg") - 1
    result = long_path(Path(root) / ("z" * padding) / "f.jpg")

    assert not result.startswith(WINDOWS_EXTENDED_PREFIX)
