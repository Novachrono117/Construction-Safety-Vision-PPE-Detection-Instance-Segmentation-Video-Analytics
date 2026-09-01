"""Provenance records for data and model artifacts.

Reproducibility claims are only credible if every artifact can be traced back to
the code, configuration and inputs that produced it. This module provides the
minimal primitives every later phase writes with: content hashing, code version
capture, and a JSON record written next to the artifact it describes.

The record format is intentionally small and stable; phases add payload under
``details`` rather than changing the schema.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
"""Version of the provenance record schema."""

DEFAULT_CHUNK_SIZE = 1 << 20
"""Chunk size (bytes) used when hashing files."""


def sha256_file(path: str | Path, *, chunk_size: int = DEFAULT_CHUNK_SIZE) -> str:
    """Compute the SHA-256 digest of a file.

    Args:
        path: File to hash.
        chunk_size: Number of bytes read per iteration.

    Returns:
        The hexadecimal digest.

    Raises:
        FileNotFoundError: If the path is not an existing file.
    """
    file_path = Path(path)
    if not file_path.is_file():
        msg = f"Cannot hash a missing file: {file_path}"
        raise FileNotFoundError(msg)
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(root: str | Path | None = None) -> str | None:
    """Return the current git commit of the repository, if available.

    Args:
        root: Directory inside the repository. Defaults to the current working
            directory.

    Returns:
        The full commit hash, or ``None`` when git is unavailable or the
        directory is not a repository.
    """
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=None if root is None else str(root),
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def runtime_environment() -> dict[str, str]:
    """Capture the interpreter and platform the artifact was produced on.

    Returns:
        A mapping with the Python version, implementation and platform string.
    """
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "executable": sys.executable,
    }


@dataclass(frozen=True)
class ArtifactRecord:
    """Identity of a single file produced or consumed by a phase.

    Attributes:
        path: Path of the artifact, relative to the repository root when
            possible.
        sha256: SHA-256 digest of the file contents.
        size_bytes: Size of the file in bytes.
    """

    path: str
    sha256: str
    size_bytes: int

    @classmethod
    def from_path(
        cls, path: str | Path, *, relative_to: str | Path | None = None
    ) -> ArtifactRecord:
        """Build a record by hashing an existing file.

        Args:
            path: File to record.
            relative_to: Base directory used to shorten the stored path.

        Returns:
            The artifact record.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        file_path = Path(path)
        stored = file_path
        if relative_to is not None:
            try:
                stored = file_path.resolve().relative_to(Path(relative_to).resolve())
            except ValueError:
                stored = file_path
        return cls(
            path=stored.as_posix(),
            sha256=sha256_file(file_path),
            size_bytes=file_path.stat().st_size,
        )


@dataclass
class ProvenanceRecord:
    """Everything needed to explain how an artifact came to exist.

    Attributes:
        name: Short identifier of the step that produced the artifacts.
        phase: Roadmap phase number the step belongs to.
        created_at: UTC timestamp in ISO-8601 format.
        schema_version: Version of this record format.
        git_commit: Repository commit at production time, when available.
        environment: Interpreter and platform description.
        config: Snapshot of the configuration used.
        inputs: Records of the files consumed.
        outputs: Records of the files produced.
        details: Free-form, phase-specific payload.
    """

    name: str
    phase: int
    created_at: str
    schema_version: int = SCHEMA_VERSION
    git_commit: str | None = None
    environment: dict[str, str] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    inputs: list[ArtifactRecord] = field(default_factory=list)
    outputs: list[ArtifactRecord] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        name: str,
        *,
        phase: int,
        config: dict[str, Any] | None = None,
        repo_root: str | Path | None = None,
        details: dict[str, Any] | None = None,
    ) -> ProvenanceRecord:
        """Start a record with timestamp, code version and environment filled in.

        Args:
            name: Short identifier of the producing step.
            phase: Roadmap phase number.
            config: Configuration snapshot to embed.
            repo_root: Repository directory used to resolve the git commit.
            details: Free-form, phase-specific payload.

        Returns:
            A record ready to receive inputs and outputs.
        """
        return cls(
            name=name,
            phase=phase,
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
            git_commit=git_commit(repo_root),
            environment=runtime_environment(),
            config=dict(config or {}),
            details=dict(details or {}),
        )

    def add_input(self, path: str | Path, *, relative_to: str | Path | None = None) -> None:
        """Record a consumed file.

        Args:
            path: File to record.
            relative_to: Base directory used to shorten the stored path.
        """
        self.inputs.append(ArtifactRecord.from_path(path, relative_to=relative_to))

    def add_output(self, path: str | Path, *, relative_to: str | Path | None = None) -> None:
        """Record a produced file.

        Args:
            path: File to record.
            relative_to: Base directory used to shorten the stored path.
        """
        self.outputs.append(ArtifactRecord.from_path(path, relative_to=relative_to))

    def to_dict(self) -> dict[str, Any]:
        """Serialise the record.

        Returns:
            A JSON-serialisable mapping.
        """
        return asdict(self)

    def write_json(self, path: str | Path) -> Path:
        """Write the record as indented JSON with LF line endings.

        Args:
            path: Destination file.

        Returns:
            The path written.
        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), indent=2, ensure_ascii=False, sort_keys=True)
        destination.write_text(f"{payload}\n", encoding="utf-8", newline="\n")
        return destination
