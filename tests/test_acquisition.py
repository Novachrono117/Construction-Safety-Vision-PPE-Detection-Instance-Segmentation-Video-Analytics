"""Tests for streaming download, idempotency and safe archive extraction.

No test touches the network: downloads run through an injected fake transport.
"""

from __future__ import annotations

import hashlib
import os
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import pytest

from construction_safety_vision.data.acquisition import (
    WINDOWS_EXTENDED_PREFIX,
    DownloadError,
    HashMismatchError,
    UnsafeArchiveError,
    download_if_absent,
    extended_path,
    safe_extract_zip,
    stream_download,
)

URL = "https://example.test/export.zip"
PAYLOAD = b"archive-bytes-" * 5000
PAYLOAD_SHA = hashlib.sha256(PAYLOAD).hexdigest()


class FakeBody:
    """Response stub that streams a byte payload in chunks."""

    def __init__(self, payload: bytes, headers: dict[str, str] | None = None) -> None:
        self._payload = payload
        self.headers = headers if headers is not None else {"Content-Length": str(len(payload))}

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            body, self._payload = self._payload, b""
            return body
        body, self._payload = self._payload[:size], self._payload[size:]
        return body

    def __enter__(self) -> FakeBody:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def serving(payload: bytes, headers: dict[str, str] | None = None):
    """Build a transport that serves a fixed payload."""

    def transport(request: urllib.request.Request, timeout: float) -> FakeBody:
        return FakeBody(payload, headers)

    return transport


def failing(error: Exception):
    """Build a transport that raises."""

    def transport(request: urllib.request.Request, timeout: float) -> Any:
        raise error

    return transport


def test_streaming_hash_matches_hashlib(tmp_path: Path) -> None:
    result = stream_download(
        "https://example/x.zip", tmp_path / "x.zip", transport=serving(PAYLOAD), chunk_size=997
    )

    assert result.sha256 == PAYLOAD_SHA
    assert result.size_bytes == len(PAYLOAD)
    assert result.downloaded is True
    assert (tmp_path / "x.zip").read_bytes() == PAYLOAD


def test_hash_is_independent_of_chunk_size(tmp_path: Path) -> None:
    a = stream_download(URL, tmp_path / "a.zip", transport=serving(PAYLOAD), chunk_size=13)
    b = stream_download(URL, tmp_path / "b.zip", transport=serving(PAYLOAD), chunk_size=1 << 16)
    assert a.sha256 == b.sha256 == PAYLOAD_SHA


def test_truncated_download_is_rejected_and_leaves_no_file(tmp_path: Path) -> None:
    # The provider announces more bytes than it sends.
    transport = serving(PAYLOAD, headers={"Content-Length": str(len(PAYLOAD) + 10)})

    with pytest.raises(DownloadError, match="Incomplete download"):
        stream_download(URL, tmp_path / "x.zip", transport=transport)

    assert not (tmp_path / "x.zip").exists()
    assert not (tmp_path / "x.zip.part").exists(), "partial file must be cleaned up"


def test_empty_download_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(DownloadError, match="empty file"):
        stream_download(URL, tmp_path / "x.zip", transport=serving(b"", headers={}))


def test_transport_failure_does_not_leave_a_partial_file(tmp_path: Path) -> None:
    with pytest.raises(DownloadError):
        stream_download(URL, tmp_path / "x.zip", transport=failing(urllib.error.URLError("boom")))
    assert list(tmp_path.iterdir()) == []


def test_download_error_never_echoes_the_signed_url(tmp_path: Path) -> None:
    # The download link is a short-lived credential; it must not reach a log.
    url = "https://download.example/x.zip?Signature=abc123SECRET&Expires=999"
    with pytest.raises(DownloadError) as info:
        stream_download(url, tmp_path / "x.zip", transport=failing(urllib.error.URLError("boom")))
    assert "SECRET" not in str(info.value)
    assert "Signature" not in str(info.value)


def test_reuses_an_existing_artifact_with_a_matching_digest(tmp_path: Path) -> None:
    destination = tmp_path / "x.zip"
    destination.write_bytes(PAYLOAD)

    def exploding(request: urllib.request.Request, timeout: float) -> Any:
        raise AssertionError("must not download when a valid artifact is present")

    result = download_if_absent(URL, destination, expected_sha256=PAYLOAD_SHA, transport=exploding)

    assert result.downloaded is False
    assert result.sha256 == PAYLOAD_SHA


def test_refuses_to_overwrite_an_artifact_with_a_different_digest(tmp_path: Path) -> None:
    destination = tmp_path / "x.zip"
    destination.write_bytes(b"different bytes")

    with pytest.raises(HashMismatchError, match="differs from the recorded"):
        download_if_absent(URL, destination, expected_sha256=PAYLOAD_SHA)

    assert destination.read_bytes() == b"different bytes", "existing artifact must be untouched"


def test_refuses_an_unrecorded_existing_artifact(tmp_path: Path) -> None:
    destination = tmp_path / "x.zip"
    destination.write_bytes(PAYLOAD)

    with pytest.raises(HashMismatchError, match="no recorded digest"):
        download_if_absent(URL, destination, expected_sha256=None)


def test_downloads_when_absent(tmp_path: Path) -> None:
    result = download_if_absent(URL, tmp_path / "x.zip", transport=serving(PAYLOAD))
    assert result.downloaded is True
    assert result.sha256 == PAYLOAD_SHA


def build_zip(path: Path, members: dict[str, bytes]) -> Path:
    """Write a ZIP archive containing the given members."""
    with zipfile.ZipFile(path, "w") as bundle:
        for name, data in members.items():
            bundle.writestr(name, data)
    return path


def test_extracts_normal_members(tmp_path: Path) -> None:
    archive = build_zip(
        tmp_path / "a.zip", {"train/img.jpg": b"jpeg", "train/_annotations.coco.json": b"{}"}
    )
    destination = tmp_path / "out"

    count = safe_extract_zip(archive, destination)

    assert count == 2
    assert (destination / "train" / "img.jpg").read_bytes() == b"jpeg"


@pytest.mark.parametrize(
    "member",
    [
        "../escape.txt",
        "train/../../escape.txt",
        "/absolute.txt",
        "..\\escape.txt",
    ],
)
def test_rejects_path_traversal(tmp_path: Path, member: str) -> None:
    archive = build_zip(tmp_path / "evil.zip", {member: b"pwned"})
    destination = tmp_path / "out"

    with pytest.raises(UnsafeArchiveError):
        safe_extract_zip(archive, destination)

    assert not (tmp_path / "escape.txt").exists()
    assert not (tmp_path.parent / "escape.txt").exists()


def test_rejects_a_drive_absolute_member(tmp_path: Path) -> None:
    archive = build_zip(tmp_path / "evil.zip", {"C:/windows/system32/evil.txt": b"pwned"})
    with pytest.raises(UnsafeArchiveError, match="absolute drive path"):
        safe_extract_zip(archive, tmp_path / "out")


def test_rejects_a_symlink_member(tmp_path: Path) -> None:
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        info = zipfile.ZipInfo("link")
        info.external_attr = (0xA1FF) << 16  # symlink mode bits
        bundle.writestr(info, "/etc/passwd")

    with pytest.raises(UnsafeArchiveError, match="symbolic link"):
        safe_extract_zip(archive, tmp_path / "out")


def test_nothing_is_written_when_a_member_is_unsafe(tmp_path: Path) -> None:
    # Validation happens before extraction: a poisoned archive must not deposit
    # its harmless members either.
    archive = build_zip(tmp_path / "evil.zip", {"ok.txt": b"fine", "../escape.txt": b"pwned"})
    destination = tmp_path / "out"

    with pytest.raises(UnsafeArchiveError):
        safe_extract_zip(archive, destination)

    assert not (destination / "ok.txt").exists()


def test_rejects_a_non_zip_file(tmp_path: Path) -> None:
    broken = tmp_path / "broken.zip"
    broken.write_bytes(b"this is not a zip archive")

    with pytest.raises(DownloadError, match="Not a valid ZIP archive"):
        safe_extract_zip(broken, tmp_path / "out")


def test_extraction_is_idempotent(tmp_path: Path) -> None:
    archive = build_zip(tmp_path / "a.zip", {"train/img.jpg": b"jpeg"})
    destination = tmp_path / "out"

    assert safe_extract_zip(archive, destination) == 1
    assert safe_extract_zip(archive, destination) == 1
    assert (destination / "train" / "img.jpg").read_bytes() == b"jpeg"


def test_extended_path_lifts_the_windows_length_limit() -> None:
    # Measured behaviour: pathlib PRESERVES the prefix, so the prefixed Path can
    # be used for mkdir and open directly.
    result = extended_path(Path("relative") / "file.txt")
    assert isinstance(result, Path)
    if os.name == "nt":
        assert str(result).startswith(WINDOWS_EXTENDED_PREFIX)
        assert str(Path(str(result))) == str(result), "pathlib must preserve the prefix"
    else:
        assert result == Path("relative") / "file.txt"


def test_extended_path_is_idempotent() -> None:
    once = extended_path(Path("a") / "b.txt")
    assert extended_path(once) == once


def test_extracts_members_past_the_windows_path_limit(tmp_path: Path) -> None:
    # Regression: the real export failed to extract because the repository path
    # plus a provider-generated file name exceeded MAX_PATH (260) on Windows.
    long_name = "premium_photo-" + "a" * 60 + "_jpg.rf." + "b" * 32 + ".jpg"
    archive = build_zip(tmp_path / "long.zip", {f"train/{long_name}": b"jpeg-bytes"})
    destination = tmp_path / ("d" * 90) / "export"

    count = safe_extract_zip(archive, destination)

    written = destination / "train" / long_name
    assert count == 1
    if os.name == "nt":
        assert len(str(written)) > 240, "the fixture must approach the limit to be meaningful"
    assert extended_path(written).read_bytes() == b"jpeg-bytes"
