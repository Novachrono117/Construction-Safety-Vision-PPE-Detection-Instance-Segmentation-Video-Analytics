"""Tests for repository layout resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from construction_safety_vision.paths import (
    PROJECT_ROOT_ENV_VAR,
    ProjectPaths,
    ProjectRootNotFoundError,
    find_project_root,
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
