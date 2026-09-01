"""Streaming download and safe extraction of an external dataset archive.

The acquired archive is treated as an immutable artifact: it is hashed while it
is downloaded, written atomically, and never overwritten in place. Re-running an
acquisition is idempotent - an archive whose digest already matches the recorded
one is left untouched and not downloaded again.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from construction_safety_vision.data.roboflow import USER_AGENT, Transport, _default_transport

DOWNLOAD_CHUNK_SIZE = 1 << 20
"""Chunk size (bytes) used while streaming a download."""

DOWNLOAD_TIMEOUT = 600.0
"""Socket timeout in seconds for the archive download."""


class AcquisitionError(RuntimeError):
    """Base class for acquisition failures."""


class DownloadError(AcquisitionError):
    """Raised when a download fails or is incomplete."""


class HashMismatchError(AcquisitionError):
    """Raised when an existing artifact does not match its recorded digest."""


class UnsafeArchiveError(AcquisitionError):
    """Raised when an archive member would be written outside the destination."""


@dataclass(frozen=True)
class DownloadResult:
    """Outcome of a streamed download.

    Attributes:
        path: Where the payload was written.
        sha256: Digest computed while streaming.
        size_bytes: Number of bytes written.
        downloaded: ``False`` when an already-valid artifact was reused.
    """

    path: Path
    sha256: str
    size_bytes: int
    downloaded: bool


def stream_download(
    url: str,
    destination: Path,
    *,
    transport: Transport | None = None,
    chunk_size: int = DOWNLOAD_CHUNK_SIZE,
    timeout: float = DOWNLOAD_TIMEOUT,
) -> DownloadResult:
    """Download a URL to a file, hashing it as it streams.

    The payload is written to a ``.part`` file and renamed only after the
    transfer completes, so an interrupted run can never leave a truncated file
    that looks valid. The whole payload is never held in memory.

    Args:
        url: Source URL. Treated as a secret and never included in messages.
        destination: Final path of the artifact.
        transport: Request performer. Injected by tests.
        chunk_size: Bytes read per iteration.
        timeout: Socket timeout in seconds.

    Returns:
        The download result.

    Raises:
        DownloadError: If the transfer fails, or if the provider announced a
            content length that does not match the bytes received.
    """
    perform = _default_transport if transport is None else transport
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(f"{destination.suffix}.part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    digest = hashlib.sha256()
    written = 0
    try:
        with perform(request, timeout) as response:
            announced = response.headers.get("Content-Length")
            with partial.open("wb") as handle:
                while chunk := response.read(chunk_size):
                    handle.write(chunk)
                    digest.update(chunk)
                    written += len(chunk)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
        partial.unlink(missing_ok=True)
        # The URL carries a short-lived credential: report the failure type only.
        msg = f"Download failed after {written} bytes: {type(exc).__name__}"
        raise DownloadError(msg) from None

    if announced is not None and announced.isdigit() and int(announced) != written:
        partial.unlink(missing_ok=True)
        msg = f"Incomplete download: provider announced {announced} bytes, received {written}"
        raise DownloadError(msg)
    if written == 0:
        partial.unlink(missing_ok=True)
        msg = "Download produced an empty file"
        raise DownloadError(msg)

    partial.replace(destination)
    return DownloadResult(
        path=destination,
        sha256=digest.hexdigest(),
        size_bytes=written,
        downloaded=True,
    )


def download_if_absent(
    url: str,
    destination: Path,
    *,
    expected_sha256: str | None = None,
    transport: Transport | None = None,
) -> DownloadResult:
    """Download an artifact unless a matching one is already present.

    Args:
        url: Source URL.
        destination: Final path of the artifact.
        expected_sha256: Digest recorded by a previous acquisition, if any.
        transport: Request performer. Injected by tests.

    Returns:
        The download result, with ``downloaded=False`` when an existing valid
        artifact was reused.

    Raises:
        HashMismatchError: If the artifact exists and its digest differs from
            ``expected_sha256``, or if it exists with no recorded digest to
            compare against. Existing artifacts are never overwritten silently.
    """
    if destination.exists():
        actual = _sha256_of(destination)
        size = destination.stat().st_size
        if expected_sha256 is None:
            msg = (
                f"{destination.name} already exists but no recorded digest was supplied to "
                f"verify it (found sha256={actual}). Refusing to overwrite; delete it "
                "explicitly to force a fresh download."
            )
            raise HashMismatchError(msg)
        if actual != expected_sha256:
            msg = (
                f"{destination.name} already exists with sha256={actual}, which differs from "
                f"the recorded {expected_sha256}. Refusing to overwrite a differing artifact."
            )
            raise HashMismatchError(msg)
        return DownloadResult(destination, actual, size, downloaded=False)
    return stream_download(url, destination, transport=transport)


def _sha256_of(path: Path, *, chunk_size: int = DOWNLOAD_CHUNK_SIZE) -> str:
    """Hash a file in chunks.

    Args:
        path: File to hash.
        chunk_size: Bytes read per iteration.

    Returns:
        The hexadecimal digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _is_within(base: Path, target: Path) -> bool:
    """Report whether a resolved path stays inside a base directory.

    Args:
        base: Directory that must contain the target.
        target: Path to test.

    Returns:
        ``True`` when the target is inside the base directory.
    """
    try:
        target.resolve().relative_to(base.resolve())
    except ValueError:
        return False
    return True


WINDOWS_EXTENDED_PREFIX = f"{chr(92) * 2}?{chr(92)}"
r"""The Windows extended-length path prefix ``\\?\``, built without escape noise."""


def extended_path(path: Path) -> Path:
    r"""Return a path that is not subject to the Windows 260-character limit.

    Dataset exports combine a long repository path with long provider-generated
    file names, which exceeds ``MAX_PATH`` on Windows hosts that do not have long
    paths enabled. The ``\\?\`` prefix lifts that limit per call, which is
    preferable to requiring a machine-wide registry change.

    Measured on a Windows 11 host with ``LongPathsEnabled=0``: a plain
    ``Path.mkdir(parents=True)`` fails at 258 characters with WinError 206, while
    the same call on the prefixed path succeeds at 262. ``pathlib`` preserves the
    prefix, so both ``mkdir`` and ``open`` can be used normally on the result.

    Args:
        path: Path to convert.

    Returns:
        The original path on POSIX, or the extended-length form on Windows.
    """
    if os.name != "nt":
        return path
    # Lexical, unlike resolve(): the target usually does not exist yet.
    absolute = os.path.abspath(str(path))  # noqa: PTH100
    if absolute.startswith(WINDOWS_EXTENDED_PREFIX):
        return Path(absolute)
    if absolute.startswith(chr(92) * 2):
        return Path(f"{WINDOWS_EXTENDED_PREFIX}UNC{chr(92)}{absolute[2:]}")
    return Path(f"{WINDOWS_EXTENDED_PREFIX}{absolute}")


def safe_extract_zip(archive: Path, destination: Path) -> int:
    """Extract a ZIP archive, refusing any member that escapes the destination.

    Guards against absolute paths, ``..`` traversal, Windows drive-letter and
    UNC paths, and symbolic links stored inside the archive. Every member is
    validated before anything is written, and members are streamed one at a time
    so a large archive is never held in memory.

    Args:
        archive: ZIP file to extract.
        destination: Directory to extract into. Created if absent.

    Returns:
        The number of members extracted.

    Raises:
        UnsafeArchiveError: If any member would be written outside the
            destination directory.
        DownloadError: If the archive is not a valid ZIP file or is corrupt.
    """
    extended_path(destination).mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(archive) as bundle:
            members = bundle.infolist()
            for member in members:
                _reject_unsafe_member(member, destination)
            for member in members:
                # Both the directory and the file go through the extended-length
                # path: either component can push the total past the limit.
                target = extended_path(destination / member.filename)
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(member) as source, target.open("wb") as sink:
                    shutil.copyfileobj(source, sink, DOWNLOAD_CHUNK_SIZE)
            return len(members)
    except zipfile.BadZipFile as exc:
        msg = f"Not a valid ZIP archive: {archive.name} ({exc})"
        raise DownloadError(msg) from None


def _reject_unsafe_member(member: zipfile.ZipInfo, destination: Path) -> None:
    """Validate a single archive member.

    Args:
        member: Archive member to validate.
        destination: Directory the member must stay inside.

    Raises:
        UnsafeArchiveError: If the member is unsafe to extract.
    """
    name = member.filename
    if name.startswith(("/", "\\")) or ".." in Path(name.replace("\\", "/")).parts:
        msg = f"Archive member escapes the destination directory: {name!r}"
        raise UnsafeArchiveError(msg)
    if len(name) > 1 and name[1] == ":":
        msg = f"Archive member uses an absolute drive path: {name!r}"
        raise UnsafeArchiveError(msg)
    # High 16 bits of external_attr hold the Unix mode; 0xA000 marks a symlink.
    if (member.external_attr >> 16) & 0xF000 == 0xA000:
        msg = f"Archive member is a symbolic link: {name!r}"
        raise UnsafeArchiveError(msg)
    if not _is_within(destination, destination / name):
        msg = f"Archive member resolves outside the destination directory: {name!r}"
        raise UnsafeArchiveError(msg)
