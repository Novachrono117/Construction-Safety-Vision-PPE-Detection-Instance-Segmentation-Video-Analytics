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

UNIVERSE_BASE = "https://universe.roboflow.com"
"""Root of the public Roboflow Universe site."""

API_KEY_ENV_VAR = "ROBOFLOW_API_KEY"
"""Environment variable holding the Roboflow API key."""

USER_AGENT = "construction-safety-vision/0.1 (+acquisition)"
"""User agent sent with every request."""

DEFAULT_TIMEOUT = 120.0
"""Socket timeout in seconds for metadata requests."""

Transport = Callable[[urllib.request.Request, float], Any]
"""Callable that performs a request. Injected so tests never touch the network."""


class RoboflowError(RuntimeError):
    """Base class for provider communication failures."""


class MissingApiKeyError(RoboflowError):
    """Raised when the API key is absent or empty in the environment."""


class AuthenticationError(RoboflowError):
    """Raised when the provider rejects the credentials."""


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

    def _get_json(
        self, path: str, *, timeout: float = DEFAULT_TIMEOUT
    ) -> tuple[int, dict[str, Any]]:
        """Perform an authenticated GET returning parsed JSON.

        Args:
            path: API path, without a leading slash.
            timeout: Socket timeout in seconds.

        Returns:
            The HTTP status code and the decoded body.

        Raises:
            AuthenticationError: On HTTP 401 or 403.
            RoboflowError: On any other HTTP or decoding failure.
        """
        request = urllib.request.Request(
            f"{API_BASE}/{path}",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "User-Agent": USER_AGENT,
                "Accept": "application/json",
            },
        )
        try:
            with self._transport(request, timeout) as response:
                status = int(response.status)
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = self.redact(exc.read(500).decode("utf-8", "replace")) if exc.fp else ""
            if exc.code in (401, 403):
                msg = (
                    f"Provider rejected the credentials for {path!r}: HTTP {exc.code} "
                    f"{exc.reason}. Check {API_KEY_ENV_VAR}. Detail: {detail}"
                )
                raise AuthenticationError(msg) from None
            msg = f"GET {path!r} failed: HTTP {exc.code} {exc.reason}. Detail: {detail}"
            raise RoboflowError(msg) from None
        except (urllib.error.URLError, TimeoutError) as exc:
            msg = f"GET {path!r} failed: {self.redact(str(exc))}"
            raise RoboflowError(msg) from None
        except json.JSONDecodeError as exc:
            msg = f"GET {path!r} returned a body that is not valid JSON: {exc}"
            raise RoboflowError(msg) from None
        if not isinstance(payload, dict):
            msg = f"GET {path!r} returned {type(payload).__name__}, expected a JSON object"
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
