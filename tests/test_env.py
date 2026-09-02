"""Tests for loading local settings from a git-ignored .env file.

The behavioural rules that matter here are that the real environment wins over
the file, and that no function ever hands a secret value back to a caller that
only asked which names were loaded.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from construction_safety_vision.env import (
    load_env_file,
    load_project_env,
    parse_env_text,
)

SECRET = "s3cr3t-value-must-not-leak"


def write_env(tmp_path: Path, body: str, name: str = ".env") -> Path:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8", newline="\n")
    return path


def test_parses_simple_assignments() -> None:
    assert parse_env_text("A=1\nB=two\n") == {"A": "1", "B": "two"}


def test_ignores_comments_and_blank_lines() -> None:
    text = "# a comment\n\n  \nA=1\n# B=2\n"
    assert parse_env_text(text) == {"A": "1"}


def test_ignores_lines_without_an_assignment() -> None:
    assert parse_env_text("just some prose\nA=1\n") == {"A": "1"}


def test_accepts_an_export_prefix() -> None:
    # The same file should be sourceable by a shell.
    assert parse_env_text("export A=1\n") == {"A": "1"}


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ('A="quoted"', "quoted"),
        ("A='quoted'", "quoted"),
        ("A=unquoted", "unquoted"),
        ("A=\"mismatched'", "\"mismatched'"),
        ("A=", ""),
    ],
)
def test_strips_one_surrounding_quote_pair(line: str, expected: str) -> None:
    assert parse_env_text(line)["A"] == expected


def test_value_may_contain_equals_signs() -> None:
    # Base64-ish credentials end in '='; splitting on the first '=' only.
    assert parse_env_text("TOKEN=abc=def==")["TOKEN"] == "abc=def=="


def test_loads_into_the_given_mapping() -> None:
    env: dict[str, str] = {}
    names = load_env_file(Path("does-not-exist"), env=env)
    assert names == []
    assert env == {}


def test_missing_file_is_not_an_error(tmp_path: Path) -> None:
    assert load_env_file(tmp_path / "absent.env", env={}) == []


def test_real_environment_wins_over_the_file(tmp_path: Path) -> None:
    # A value exported in the shell must not be silently replaced by the file.
    path = write_env(tmp_path, f"ROBOFLOW_API_KEY={SECRET}\n")
    env = {"ROBOFLOW_API_KEY": "from-the-shell"}

    names = load_env_file(path, env=env)

    assert names == []
    assert env["ROBOFLOW_API_KEY"] == "from-the-shell"


def test_override_is_opt_in(tmp_path: Path) -> None:
    path = write_env(tmp_path, f"ROBOFLOW_API_KEY={SECRET}\n")
    env = {"ROBOFLOW_API_KEY": "from-the-shell"}

    names = load_env_file(path, env=env, override=True)

    assert names == ["ROBOFLOW_API_KEY"]
    assert env["ROBOFLOW_API_KEY"] == SECRET


def test_empty_values_do_not_shadow_anything(tmp_path: Path) -> None:
    # .env.example ships empty placeholders; copying it must not define blanks.
    path = write_env(tmp_path, "ROBOFLOW_API_KEY=\nCSVISION_PROJECT_ROOT=\n")
    env: dict[str, str] = {}

    assert load_env_file(path, env=env) == []
    assert env == {}


def test_returns_names_never_values(tmp_path: Path) -> None:
    # The return value is designed to be safe to print in a log line.
    path = write_env(tmp_path, f"ROBOFLOW_API_KEY={SECRET}\nOTHER=plain\n")
    env: dict[str, str] = {}

    names = load_env_file(path, env=env)

    assert names == ["OTHER", "ROBOFLOW_API_KEY"]
    assert SECRET not in " ".join(names)
    assert env["ROBOFLOW_API_KEY"] == SECRET, "the value must still reach the environment"


def test_names_are_sorted_for_deterministic_output(tmp_path: Path) -> None:
    path = write_env(tmp_path, "Z=1\nA=2\nM=3\n")
    assert load_env_file(path, env={}) == ["A", "M", "Z"]


def test_load_project_env_reads_the_repository_dotenv(tmp_path: Path) -> None:
    write_env(tmp_path, f"ROBOFLOW_API_KEY={SECRET}\n")
    env: dict[str, str] = {}

    assert load_project_env(tmp_path, env=env) == ["ROBOFLOW_API_KEY"]
    assert env["ROBOFLOW_API_KEY"] == SECRET


def test_load_project_env_tolerates_a_repository_without_dotenv(tmp_path: Path) -> None:
    assert load_project_env(tmp_path, env={}) == []
