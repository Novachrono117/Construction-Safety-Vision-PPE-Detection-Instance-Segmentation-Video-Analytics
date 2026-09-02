"""Loading local settings from a git-ignored ``.env`` file.

``.env.example`` documents ``.env`` as the place to keep credentials, so
something has to actually read it. This is that something: a small stdlib parser
rather than a dependency, because the format the project needs is
``KEY=value`` and nothing more.

Two rules make it safe to call from any entry point:

* the real environment always wins - a value already exported in the shell is
  never overwritten by the file, so CI and one-off overrides behave predictably;
* nothing here ever returns, logs or formats a value. Callers get the *names*
  that were loaded, never the secrets themselves.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_ENV_FILENAME = ".env"
"""Name of the local settings file, relative to the repository root."""


def parse_env_text(text: str) -> dict[str, str]:
    """Parse ``KEY=value`` lines.

    Blank lines and ``#`` comments are ignored. A single surrounding pair of
    quotes is stripped from the value. An ``export`` prefix is accepted so the
    same file can be sourced by a shell.

    Args:
        text: Contents of a ``.env`` file.

    Returns:
        The parsed name-to-value mapping.
    """
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name = name.removeprefix("export ").strip()
        if not name:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        values[name] = value
    return values


def load_env_file(
    path: Path,
    *,
    env: dict[str, str] | None = None,
    override: bool = False,
) -> list[str]:
    """Load a ``.env`` file into an environment mapping.

    Args:
        path: File to read. A missing file is not an error.
        env: Mapping to populate. Defaults to ``os.environ``.
        override: Whether file values replace existing ones. Defaults to
            ``False`` so the real environment wins.

    Returns:
        The sorted names that were set, for logging. Never the values.
    """
    target = os.environ if env is None else env
    if not path.is_file():
        return []
    applied: list[str] = []
    for name, value in parse_env_text(path.read_text(encoding="utf-8")).items():
        if not value:
            continue
        if not override and target.get(name):
            continue
        target[name] = value
        applied.append(name)
    return sorted(applied)


def load_project_env(
    root: Path | None = None,
    *,
    env: dict[str, str] | None = None,
) -> list[str]:
    """Load the repository's ``.env`` file, if present.

    Args:
        root: Repository root. Discovered automatically when omitted.
        env: Mapping to populate. Defaults to ``os.environ``.

    Returns:
        The sorted names that were set. Never the values.
    """
    if root is None:
        from construction_safety_vision.paths import ProjectRootNotFoundError, find_project_root

        try:
            root = find_project_root()
        except ProjectRootNotFoundError:
            return []
    return load_env_file(Path(root) / DEFAULT_ENV_FILENAME, env=env)
