"""Repository layout resolution for local and Google Colab execution.

The project must run unchanged from a local checkout and from a Colab runtime
where the repository is cloned into an arbitrary directory. Every module that
needs a filesystem location resolves it through :class:`ProjectPaths` instead of
hardcoding relative paths, so notebooks and scripts behave identically no matter
what the current working directory is.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT_ENV_VAR = "CSVISION_PROJECT_ROOT"
"""Environment variable that overrides root discovery."""

ROOT_MARKERS: tuple[str, ...] = ("pyproject.toml",)
"""Files whose presence identifies the repository root."""

WINDOWS_EXTENDED_PREFIX = "\\\\?\\"
"""Prefix that lifts the Win32 ``MAX_PATH`` limit for a fully qualified path."""

WINDOWS_MAX_PATH = 259
"""Longest path the Win32 API accepts without the extended-length prefix."""


class ProjectRootNotFoundError(RuntimeError):
    """Raised when the repository root cannot be resolved."""


def long_path(path: str | Path) -> str:
    """Return a filesystem path that is openable regardless of its length.

    Windows rejects paths longer than 260 characters unless long-path support is
    enabled system-wide, and the relative form is no escape: the API resolves it
    against the working directory before applying the limit. This repository hits
    that ceiling for real - the provider's export uses descriptive filenames, and
    137 of its 742 images exceed it once nested under a project directory with a
    long name. Those files list fine and open never.

    The extended-length prefix lifts the limit without touching the machine's
    configuration. It is applied only where it is needed and only on Windows,
    because the prefix also disables path normalisation.

    Args:
        path: Path to open. May be relative.

    Returns:
        The path as a string, prefixed only when required.
    """
    text = os.fspath(path)
    if os.name != "nt" or text.startswith(WINDOWS_EXTENDED_PREFIX):
        return text
    # Path.absolute rather than Path.resolve: the prefix requires a normalised
    # absolute path, but resolve() walks the filesystem to expand symlinks, and
    # doing that to a path the API cannot open yet is exactly what must be avoided.
    absolute = str(Path(os.path.normpath(Path(text).absolute())))
    if len(absolute) <= WINDOWS_MAX_PATH:
        return text
    return WINDOWS_EXTENDED_PREFIX + absolute


def in_colab() -> bool:
    """Report whether the current process runs inside a Google Colab runtime.

    Returns:
        ``True`` when the Colab-specific module is importable.
    """
    try:
        import google.colab  # noqa: F401  # pragma: no cover - Colab only
    except ImportError:
        return False
    return True  # pragma: no cover - Colab only


def find_project_root(
    start: Path | None = None,
    *,
    env: dict[str, str] | None = None,
) -> Path:
    """Resolve the repository root.

    Resolution order:

    1. the ``CSVISION_PROJECT_ROOT`` environment variable, when set;
    2. the closest ancestor of ``start`` that contains a root marker.

    Args:
        start: Directory (or file) to start the upward search from. Defaults to
            the directory of this module.
        env: Environment mapping to read. Defaults to ``os.environ``.

    Returns:
        The absolute path of the repository root.

    Raises:
        ProjectRootNotFoundError: If neither strategy resolves an existing
            directory.
    """
    environment = os.environ if env is None else env
    override = environment.get(PROJECT_ROOT_ENV_VAR, "").strip()
    if override:
        root = Path(override).expanduser().resolve()
        if not root.is_dir():
            msg = f"{PROJECT_ROOT_ENV_VAR} points to a missing directory: {root}"
            raise ProjectRootNotFoundError(msg)
        return root

    origin = Path(__file__) if start is None else Path(start)
    origin = origin.expanduser().resolve()
    candidates = [origin, *origin.parents] if origin.is_dir() else list(origin.parents)
    for candidate in candidates:
        if any((candidate / marker).is_file() for marker in ROOT_MARKERS):
            return candidate

    msg = f"No project root marker {ROOT_MARKERS} found above {origin}"
    raise ProjectRootNotFoundError(msg)


@dataclass(frozen=True)
class ProjectPaths:
    """Canonical directory layout of the project.

    Attributes:
        root: Repository root.
        configs: Versioned experiment configuration files.
        data_raw: Immutable downloaded datasets. Never edited in place.
        data_interim: Intermediate artifacts of the preprocessing pipeline.
        data_processed: Task-ready datasets derived from the canonical source.
        data_external: Third-party assets that are not the primary dataset
            (e.g. inference videos).
        notebooks: Colab-executable notebooks.
        reports: Written deliverables and generated evaluation reports.
        figures: Generated figures and qualitative samples.
        scripts: Command-line entry points.
    """

    root: Path
    configs: Path
    data_raw: Path
    data_interim: Path
    data_processed: Path
    data_external: Path
    notebooks: Path
    reports: Path
    figures: Path
    scripts: Path

    @classmethod
    def from_root(cls, root: Path | None = None) -> ProjectPaths:
        """Build the layout from a repository root.

        Args:
            root: Repository root. Discovered automatically when omitted.

        Returns:
            The resolved project layout.
        """
        base = find_project_root() if root is None else Path(root).expanduser().resolve()
        data = base / "data"
        reports = base / "reports"
        return cls(
            root=base,
            configs=base / "configs",
            data_raw=data / "raw",
            data_interim=data / "interim",
            data_processed=data / "processed",
            data_external=data / "external",
            notebooks=base / "notebooks",
            reports=reports,
            figures=reports / "figures",
            scripts=base / "scripts",
        )

    def data_dirs(self) -> tuple[Path, ...]:
        """Return the data directories owned by the preprocessing pipeline.

        Returns:
            The raw, interim, processed and external data directories.
        """
        return (self.data_raw, self.data_interim, self.data_processed, self.data_external)

    def ensure_data_dirs(self) -> None:
        """Create the data directories if they do not exist."""
        for directory in self.data_dirs():
            directory.mkdir(parents=True, exist_ok=True)
