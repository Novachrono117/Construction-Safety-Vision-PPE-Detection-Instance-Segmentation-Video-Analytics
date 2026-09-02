"""Minimal Roboflow REST client built on the standard library.

Only the endpoints this project needs are implemented: project metadata and
version export resolution. The full Roboflow SDK is deliberately not a
dependency; it would pull a large transitive tree to perform two GET requests.

Credential handling rules enforced here:

* the API key is read from the environment, never from a file in the repository;
* it is sent as an ``Authorization: Bearer`` header, never as a query parameter,
  so it cannot leak through a URL that ends up in a log, an error message or a
  provenance record;
* every message produced by this module passes through :func:`redact`;
* signed download links returned by the provider are treated as secrets too:
  they are used immediately and never persisted.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

API_BASE = "https://api.roboflow.com"
"""Root of the Roboflow REST API."""

SOURCE_BASE = "https://source.roboflow.com"
"""Root of the provider's original source-image storage."""

UNIVERSE_BASE = "https://universe.roboflow.com"
"""Root of the public Roboflow Universe site."""

API_KEY_ENV_VAR = "ROBOFLOW_API_KEY"
"""Environment variable holding the Roboflow API key."""

USER_AGENT = "construction-safety-vision/0.1 (+acquisition)"
"""User agent sent with every request."""

DEFAULT_TIMEOUT = 120.0
"""Socket timeout in seconds for metadata requests."""

RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
"""HTTP statuses worth retrying with backoff."""

MAX_RETRIES = 5
"""Attempts per request before giving up."""

BACKOFF_SECONDS = 5.0
"""Base delay for exponential backoff; the provider rate-limits rapid bursts."""

REQUEST_INTERVAL_SECONDS = 0.4
"""Minimum spacing between requests. Measured: bursts draw HTTP 429 from this
provider, and a rate-limited inventory walk is both slower and less reliable than
a paced one."""

SEARCH_PAGE_SIZE = 100
"""Records requested per search page."""

SEARCH_MAX_PAGE_SIZE = 250
"""Largest page the provider will actually return, measured; larger asks are capped."""

SEARCH_FIELDS = (
    "id",
    "name",
    "split",
    "width",
    "height",
    "created",
    "tags",
    "annotations",
    "owner",
)
"""Source-image fields requested from the search API. Embeddings are never requested."""

Transport = Callable[[urllib.request.Request, float], Any]
"""Callable that performs a request. Injected so tests never touch the network."""


class RoboflowError(RuntimeError):
    """Base class for provider communication failures."""


class MissingApiKeyError(RoboflowError):
    """Raised when the API key is absent or empty in the environment."""


class AuthenticationError(RoboflowError):
    """Raised when the provider rejects the credentials."""


class _RetryableStatusError(Exception):
    """Internal marker for a transient provider failure worth retrying."""


class ExportNotReadyError(RoboflowError):
    """Raised when an export is still being generated after the poll budget."""


def _default_transport(request: urllib.request.Request, timeout: float) -> Any:
    """Perform a request with :mod:`urllib`.

    Args:
        request: Prepared request.
        timeout: Socket timeout in seconds.

    Returns:
        The open response object.
    """
    return urllib.request.urlopen(request, timeout=timeout)


def redact(text: str, secret: str | None) -> str:
    """Remove a secret from a string.

    Args:
        text: Text that may contain the secret.
        secret: The secret to remove. Ignored when empty or ``None``.

    Returns:
        The text with every occurrence of the secret replaced.
    """
    if not secret:
        return text
    return text.replace(secret, "<REDACTED-API-KEY>")


def api_key_from_env(env: dict[str, str] | None = None) -> str:
    """Read the Roboflow API key from the environment.

    Args:
        env: Environment mapping to read. Defaults to ``os.environ``.

    Returns:
        The API key.

    Raises:
        MissingApiKeyError: If the variable is absent or empty. The message
            never contains a value.
    """
    import os

    environment = os.environ if env is None else env
    key = environment.get(API_KEY_ENV_VAR, "").strip()
    if not key:
        msg = (
            f"{API_KEY_ENV_VAR} is not set or is empty. Export it in the shell or place it "
            "in a git-ignored .env file; never commit it."
        )
        raise MissingApiKeyError(msg)
    return key


@dataclass(frozen=True)
class DatasetCoordinates:
    """Identity of one exportable dataset version.

    Attributes:
        workspace: Provider workspace slug.
        project: Provider project slug.
        version: Version number.
        export_format: Provider export format slug.
    """

    workspace: str
    project: str
    version: int
    export_format: str

    @property
    def project_path(self) -> str:
        """Provider path of the project.

        Returns:
            ``<workspace>/<project>``.
        """
        return f"{self.workspace}/{self.project}"

    @property
    def version_path(self) -> str:
        """Provider path of the version.

        Returns:
            ``<workspace>/<project>/<version>``.
        """
        return f"{self.project_path}/{self.version}"

    @property
    def public_url(self) -> str:
        """Public Universe URL of the project.

        Returns:
            A credential-free URL safe to store in reports.
        """
        return f"{UNIVERSE_BASE}/{self.project_path}"

    @property
    def public_version_url(self) -> str:
        """Public Universe URL of the dataset version.

        Returns:
            A credential-free URL safe to store in reports.
        """
        return f"{UNIVERSE_BASE}/{self.project_path}/dataset/{self.version}"

    @property
    def slug(self) -> str:
        """Deterministic filesystem-safe identifier of this export.

        Returns:
            A slug usable as a directory or file stem.
        """
        return f"{self.project}-v{self.version}-{self.export_format}"


class RoboflowClient:
    """Read-only client for the Roboflow REST API.

    The API key is kept private and never appears in :func:`repr`, in log
    output, or in any exception raised by this class.
    """

    def __init__(self, api_key: str, *, transport: Transport | None = None) -> None:
        """Initialise the client.

        Args:
            api_key: Roboflow API key.
            transport: Request performer. Defaults to :mod:`urllib`; tests
                inject a fake so no network access occurs.

        Raises:
            MissingApiKeyError: If the key is empty.
        """
        if not api_key:
            msg = "An API key is required."
            raise MissingApiKeyError(msg)
        self._api_key = api_key
        self._transport = _default_transport if transport is None else transport
        self._interval = REQUEST_INTERVAL_SECONDS if transport is None else 0.0
        self._last_request_at = 0.0

    def __repr__(self) -> str:
        """Return a representation that cannot leak the credential.

        Returns:
            A redacted representation.
        """
        return "RoboflowClient(api_key=<REDACTED-API-KEY>)"

    def redact(self, text: str) -> str:
        """Strip this client's credential from a string.

        Args:
            text: Text to sanitise.

        Returns:
            The sanitised text.
        """
        return redact(text, self._api_key)

    def _pace(self) -> None:
        """Wait, if needed, so requests stay at least one interval apart."""
        if self._interval <= 0:
            return
        elapsed = time.monotonic() - self._last_request_at
        if 0 < elapsed < self._interval:
            time.sleep(self._interval - elapsed)
        self._last_request_at = time.monotonic()

    def _get_json(
        self, path: str, *, timeout: float = DEFAULT_TIMEOUT
    ) -> tuple[int, dict[str, Any]]:
        """Perform an authenticated GET returning parsed JSON.

        Args:
            path: API path, without a leading slash.
            timeout: Socket timeout in seconds.

        Returns:
            The HTTP status code and the decoded body.
        """
        return self._request_json(path, timeout=timeout)

    def _request_json(
        self,
        path: str,
        *,
        method: str = "GET",
        body: dict[str, Any] | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        sleep: Callable[[float], None] = time.sleep,
    ) -> tuple[int, dict[str, Any]]:
        """Perform an authenticated request, retrying transient failures.

        The provider rate-limits rapid bursts, so a retryable status backs off
        exponentially instead of failing a long inventory walk outright.

        Args:
            path: API path, without a leading slash.
            method: HTTP method.
            body: JSON body for POST requests.
            timeout: Socket timeout in seconds.
            sleep: Sleep function. Injected for tests.

        Returns:
            The HTTP status code and the decoded body.

        Raises:
            RoboflowError: If every attempt failed with a retryable status.
        """
        last = ""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return self._single_request(path, method=method, body=body, timeout=timeout)
            except _RetryableStatusError as exc:
                last = str(exc)
                if attempt < MAX_RETRIES:
                    sleep(BACKOFF_SECONDS * (2 ** (attempt - 1)))
        msg = f"{method} {path!r} still failing after {MAX_RETRIES} attempts: {last}"
        raise RoboflowError(msg)

    def _single_request(
        self,
        path: str,
        *,
        method: str = "GET",
        body: dict[str, Any] | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> tuple[int, dict[str, Any]]:
        """Perform one authenticated request returning parsed JSON.

        Args:
            path: API path, without a leading slash.
            method: HTTP method.
            body: JSON body for POST requests.
            timeout: Socket timeout in seconds.

        Returns:
            The HTTP status code and the decoded body.

        Raises:
            AuthenticationError: On HTTP 401 or 403.
            _RetryableStatusError: On a transient provider failure.
            RoboflowError: On any other HTTP or decoding failure.
        """
        self._pace()
        payload_bytes = json.dumps(body).encode("utf-8") if body is not None else None
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        }
        if payload_bytes is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            f"{API_BASE}/{path}",
            data=payload_bytes,
            method=method,
            headers=headers,
        )
        try:
            with self._transport(request, timeout) as response:
                status = int(response.status)
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = self.redact(exc.read(500).decode("utf-8", "replace")) if exc.fp else ""
            if exc.code in RETRY_STATUSES:
                raise _RetryableStatusError(f"HTTP {exc.code} {exc.reason}") from None
            if exc.code in (401, 403):
                msg = (
                    f"Provider rejected the credentials for {path!r}: HTTP {exc.code} "
                    f"{exc.reason}. Check {API_KEY_ENV_VAR}. Detail: {detail}"
                )
                raise AuthenticationError(msg) from None
            msg = f"{method} {path!r} failed: HTTP {exc.code} {exc.reason}. Detail: {detail}"
            raise RoboflowError(msg) from None
        except (urllib.error.URLError, TimeoutError) as exc:
            # DNS hiccups and dropped connections are transient; a 436-image
            # walk must survive one rather than restarting from scratch.
            raise _RetryableStatusError(f"{type(exc).__name__}: {self.redact(str(exc))}") from None
        except json.JSONDecodeError as exc:
            msg = f"{method} {path!r} returned a body that is not valid JSON: {exc}"
            raise RoboflowError(msg) from None
        if not isinstance(payload, dict):
            msg = f"{method} {path!r} returned {type(payload).__name__}, expected a JSON object"
            raise RoboflowError(msg)
        return status, payload

    def project_metadata(self, coordinates: DatasetCoordinates) -> dict[str, Any]:
        """Fetch project-level metadata, including the list of versions.

        Args:
            coordinates: Dataset identity.

        Returns:
            The decoded provider response.

        Raises:
            RoboflowError: If the response is not a successful project payload.
        """
        status, payload = self._get_json(coordinates.project_path)
        if status != 200 or "project" not in payload:
            msg = f"Unexpected project metadata response: HTTP {status}, keys={sorted(payload)}"
            raise RoboflowError(msg)
        return payload

    def resolve_export(
        self,
        coordinates: DatasetCoordinates,
        *,
        max_attempts: int = 30,
        poll_seconds: float = 10.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> tuple[dict[str, Any], str]:
        """Resolve a version export to a download link, waiting if it is generating.

        The provider answers HTTP 202 while an export is still being built.

        Args:
            coordinates: Dataset identity, including the export format.
            max_attempts: Maximum number of polls before giving up.
            poll_seconds: Delay between polls.
            sleep: Sleep function. Injected for tests.

        Returns:
            The export response payload and the download link. The link is a
            short-lived credential-bearing URL: use it immediately, never store it.

        Raises:
            ExportNotReadyError: If the export is still generating after
                ``max_attempts`` polls.
            RoboflowError: If the response lacks a usable download link.
        """
        path = f"{coordinates.version_path}/{coordinates.export_format}"
        for attempt in range(1, max_attempts + 1):
            status, payload = self._get_json(path)
            if status == 200:
                link = payload.get("export", {}).get("link")
                if not isinstance(link, str) or not link:
                    msg = f"Export {path!r} reported ready but returned no download link"
                    raise RoboflowError(msg)
                return payload, link
            if attempt < max_attempts:
                sleep(poll_seconds)
        msg = (
            f"Export {path!r} was still generating after {max_attempts} polls; "
            "re-run the acquisition later."
        )
        raise ExportNotReadyError(msg)

    def search_images_page(
        self,
        coordinates: DatasetCoordinates,
        *,
        limit: int = SEARCH_PAGE_SIZE,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Fetch one page of the project's source-image inventory.

        Args:
            coordinates: Dataset identity.
            limit: Records per page.
            offset: Records to skip.

        Returns:
            The decoded page, containing ``total``, ``offset`` and ``results``.

        Raises:
            RoboflowError: If the response is not a search payload.
        """
        status, payload = self._request_json(
            f"{coordinates.project_path}/search",
            method="POST",
            body={"limit": limit, "offset": offset, "fields": list(SEARCH_FIELDS)},
        )
        if status != 200 or "results" not in payload:
            msg = f"Unexpected search response: HTTP {status}, keys={sorted(payload)}"
            raise RoboflowError(msg)
        return payload

    def image_details(self, coordinates: DatasetCoordinates, image_id: str) -> dict[str, Any]:
        """Fetch one source image's authoritative record, including annotations.

        The provider also returns a similarity ``embedding``. It is dropped here
        and never persisted: it is large, opaque, and of no use to this audit.

        Args:
            coordinates: Dataset identity.
            image_id: Provider image identifier.

        Returns:
            The ``image`` object, without its embedding.

        Raises:
            RoboflowError: If the response has no image object.
        """
        status, payload = self._request_json(f"{coordinates.project_path}/images/{image_id}")
        image = payload.get("image")
        if status != 200 or not isinstance(image, dict):
            msg = f"Unexpected image-details response for {image_id!r}: HTTP {status}"
            raise RoboflowError(msg)
        image.pop("embedding", None)
        return image


def original_image_url(owner: str, image_id: str) -> str:
    """Build the provider URL of an original source image.

    Args:
        owner: Provider owner identifier.
        image_id: Provider image identifier.

    Returns:
        The URL of the original-quality image.
    """
    return f"{SOURCE_BASE}/{owner}/{image_id}/original.jpg"
