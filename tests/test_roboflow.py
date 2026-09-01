"""Tests for the Roboflow client.

Every test uses an injected fake transport: the suite must never reach the
network. The credential-handling tests are the important ones - a leaked key in
a log line or an exception message is the failure mode this module exists to
prevent.
"""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request
from typing import Any

import pytest

from construction_safety_vision.data.roboflow import (
    API_KEY_ENV_VAR,
    AuthenticationError,
    DatasetCoordinates,
    ExportNotReadyError,
    MissingApiKeyError,
    RoboflowClient,
    RoboflowError,
    api_key_from_env,
    redact,
)

# Fake credential used only by these tests; it is not a real key.
SECRET = "s3cr3t-key-value-do-not-leak"


class FakeResponse:
    """Minimal stand-in for an HTTP response."""

    def __init__(self, status: int, payload: Any, headers: dict[str, str] | None = None) -> None:
        self.status = status
        self._body = json.dumps(payload).encode() if not isinstance(payload, bytes) else payload
        self.headers = headers or {}

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            body, self._body = self._body, b""
            return body
        body, self._body = self._body[:size], self._body[size:]
        return body

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def transport_returning(*specs: tuple[int, Any]):
    """Build a transport yielding the given (status, payload) pairs in order.

    A fresh response object is built per call: a response body can only be read
    once, and polling reads it repeatedly.
    """
    queue = list(specs)
    captured: list[urllib.request.Request] = []

    def transport(request: urllib.request.Request, timeout: float) -> FakeResponse:
        captured.append(request)
        status, payload = queue.pop(0) if len(queue) > 1 else queue[0]
        return FakeResponse(status, payload)

    transport.captured = captured  # type: ignore[attr-defined]
    return transport


def transport_raising(error: Exception):
    """Build a transport that always raises."""

    def transport(request: urllib.request.Request, timeout: float) -> Any:
        raise error

    return transport


COORDS = DatasetCoordinates(
    workspace="ws", project="proj", version=4, export_format="coco-segmentation"
)


def test_coordinates_build_deterministic_identifiers() -> None:
    assert COORDS.project_path == "ws/proj"
    assert COORDS.version_path == "ws/proj/4"
    assert COORDS.slug == "proj-v4-coco-segmentation"
    # Public URLs must never carry credentials: they end up in committed reports.
    assert "api_key" not in COORDS.public_url
    assert COORDS.public_version_url.endswith("/dataset/4")


def test_slug_is_stable_across_calls() -> None:
    other = DatasetCoordinates("ws", "proj", 4, "coco-segmentation")
    assert COORDS.slug == other.slug


def test_api_key_read_from_environment() -> None:
    assert api_key_from_env({API_KEY_ENV_VAR: "  abc  "}) == "abc"


@pytest.mark.parametrize("env", [{}, {API_KEY_ENV_VAR: ""}, {API_KEY_ENV_VAR: "   "}])
def test_missing_api_key_raises_without_echoing_anything(env: dict[str, str]) -> None:
    with pytest.raises(MissingApiKeyError) as info:
        api_key_from_env(env)
    assert API_KEY_ENV_VAR in str(info.value)


def test_redact_removes_the_secret() -> None:
    assert SECRET not in redact(f"url?api_key={SECRET}&x=1", SECRET)


def test_redact_is_a_noop_without_a_secret() -> None:
    assert redact("unchanged", None) == "unchanged"
    assert redact("unchanged", "") == "unchanged"


def test_repr_never_exposes_the_key() -> None:
    client = RoboflowClient(SECRET)
    assert SECRET not in repr(client)
    assert "REDACTED" in repr(client)


def test_key_travels_in_a_header_never_in_the_url() -> None:
    transport = transport_returning((200, {"project": {"images": 1}, "versions": []}))
    client = RoboflowClient(SECRET, transport=transport)
    client.project_metadata(COORDS)

    request = transport.captured[0]  # type: ignore[attr-defined]
    assert SECRET not in request.full_url, "the credential must never appear in a URL"
    assert request.get_header("Authorization") == f"Bearer {SECRET}"


def test_authentication_error_is_distinct_and_redacted() -> None:
    error = urllib.error.HTTPError(
        url=f"https://api.roboflow.com/ws/proj?api_key={SECRET}",
        code=401,
        msg="Unauthorized",
        hdrs=None,
        fp=io.BytesIO(f'{{"error":"bad key {SECRET}"}}'.encode()),
    )
    client = RoboflowClient(SECRET, transport=transport_raising(error))

    with pytest.raises(AuthenticationError) as info:
        client.project_metadata(COORDS)
    message = str(info.value)
    assert SECRET not in message, "the credential leaked into an exception message"
    assert "401" in message and API_KEY_ENV_VAR in message


def test_non_auth_http_error_is_not_an_authentication_error() -> None:
    # Phase 3 requires distinguishing auth failures from other transport failures.
    error = urllib.error.HTTPError(
        url="https://api.roboflow.com/ws/proj", code=500, msg="Server Error", hdrs=None, fp=None
    )
    client = RoboflowClient(SECRET, transport=transport_raising(error))

    with pytest.raises(RoboflowError) as info:
        client.project_metadata(COORDS)
    assert not isinstance(info.value, AuthenticationError)
    assert "500" in str(info.value)


def test_network_error_is_reported_without_the_key() -> None:
    client = RoboflowClient(
        SECRET, transport=transport_raising(urllib.error.URLError(f"host down {SECRET}"))
    )
    with pytest.raises(RoboflowError) as info:
        client.project_metadata(COORDS)
    assert SECRET not in str(info.value)


def test_empty_key_is_rejected_at_construction() -> None:
    with pytest.raises(MissingApiKeyError):
        RoboflowClient("")


def test_project_metadata_requires_a_project_key() -> None:
    client = RoboflowClient(SECRET, transport=transport_returning((200, {"nope": 1})))
    with pytest.raises(RoboflowError, match="Unexpected project metadata"):
        client.project_metadata(COORDS)


def test_non_object_payload_is_rejected() -> None:
    client = RoboflowClient(SECRET, transport=transport_returning((200, [1, 2])))
    with pytest.raises(RoboflowError, match="expected a JSON object"):
        client.project_metadata(COORDS)


def test_invalid_json_is_rejected() -> None:
    client = RoboflowClient(SECRET, transport=transport_returning((200, b"not json")))
    with pytest.raises(RoboflowError, match="not valid JSON"):
        client.project_metadata(COORDS)


def test_resolve_export_returns_the_link_when_ready() -> None:
    client = RoboflowClient(
        SECRET,
        transport=transport_returning(
            (200, {"export": {"link": "https://download.example/x", "size": 1.5}})
        ),
    )

    payload, link = client.resolve_export(COORDS)

    assert link == "https://download.example/x"
    assert payload["export"]["size"] == 1.5


def test_resolve_export_polls_while_generating() -> None:
    transport = transport_returning(
        (202, {"progress": 0.4}),
        (200, {"export": {"link": "https://download.example/x"}}),
    )
    slept: list[float] = []
    client = RoboflowClient(SECRET, transport=transport)

    _, link = client.resolve_export(COORDS, poll_seconds=0.0, sleep=slept.append)

    assert link == "https://download.example/x"
    assert slept == [0.0], "must wait between polls"


def test_resolve_export_gives_up_after_the_poll_budget() -> None:
    client = RoboflowClient(SECRET, transport=transport_returning((202, {"progress": 0.1})))
    with pytest.raises(ExportNotReadyError, match="still generating"):
        client.resolve_export(COORDS, max_attempts=3, poll_seconds=0.0, sleep=lambda _: None)


def test_ready_export_without_a_link_is_an_error() -> None:
    client = RoboflowClient(SECRET, transport=transport_returning((200, {"export": {}})))
    with pytest.raises(RoboflowError, match="no download link"):
        client.resolve_export(COORDS)
