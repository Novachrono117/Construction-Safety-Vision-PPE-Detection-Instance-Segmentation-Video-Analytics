"""Acquire the canonical instance-segmentation export and record its provenance.

The archive received from the provider is stored untouched under
``data/external/`` and extracted, also untouched, under ``data/raw/``. Nothing is
resized, renamed, recompressed or converted: this script only obtains bytes and
proves what they are.

Re-running is idempotent. An archive whose digest matches the one recorded by a
previous run is reused instead of downloaded again, and an archive whose digest
differs is never silently overwritten.

Usage:
    uv run python scripts/download_dataset.py
    uv run python scripts/download_dataset.py --skip-extract
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from construction_safety_vision.config import ConfigError, load_experiment_config
from construction_safety_vision.data.acquisition import (
    AcquisitionError,
    download_if_absent,
    safe_extract_zip,
)
from construction_safety_vision.data.roboflow import (
    DatasetCoordinates,
    MissingApiKeyError,
    RoboflowClient,
    RoboflowError,
    api_key_from_env,
)
from construction_safety_vision.paths import ProjectPaths
from construction_safety_vision.provenance import ProvenanceRecord

PHASE = 3
"""Roadmap phase this script belongs to."""

RECORD_NAME = "dataset-acquisition"
"""Name recorded in the provenance record."""

# Provider metadata keys copied into the record. Everything else is dropped so
# that no signed link or account-specific field can reach a committed file.
PROJECT_FIELDS = (
    "id",
    "type",
    "name",
    "images",
    "unannotated",
    "annotation",
    "versions",
    "public",
    "license",
    "splits",
    "classes",
)
VERSION_FIELDS = ("id", "name", "created", "images", "splits", "preprocessing", "augmentation")


def _coordinates_from_config(dataset: Any) -> DatasetCoordinates:
    """Build provider coordinates from the dataset configuration block.

    Args:
        dataset: The ``dataset`` section of the experiment configuration.

    Returns:
        The dataset coordinates.

    Raises:
        ConfigError: If a provider identifier is missing.
    """
    missing = [
        name
        for name in ("workspace", "project_slug", "export_format")
        if not getattr(dataset, name)
    ]
    if dataset.version <= 0:
        missing.append("version")
    if missing:
        msg = f"dataset section is missing provider identifier(s): {sorted(missing)}"
        raise ConfigError(msg)
    return DatasetCoordinates(
        workspace=dataset.workspace,
        project=dataset.project_slug,
        version=dataset.version,
        export_format=dataset.export_format,
    )


def _subset(payload: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    """Copy an explicit allow-list of keys from a provider payload.

    Args:
        payload: Raw provider payload.
        fields: Keys to keep.

    Returns:
        A mapping containing only the allowed keys that were present.
    """
    return {key: payload[key] for key in fields if key in payload}


def _recorded_digest(record_path: Path) -> str | None:
    """Read the archive digest recorded by a previous acquisition.

    Args:
        record_path: Path of the provenance record.

    Returns:
        The recorded digest, or ``None`` when no usable record exists.
    """
    if not record_path.is_file():
        return None
    try:
        payload = json.loads(record_path.read_text(encoding="utf-8"))
        outputs = payload.get("outputs", [])
    except (json.JSONDecodeError, OSError):
        return None
    for entry in outputs:
        if isinstance(entry, dict) and str(entry.get("path", "")).endswith(".zip"):
            digest = entry.get("sha256")
            return str(digest) if digest else None
    return None


def main(argv: list[str] | None = None) -> int:
    """Run the acquisition.

    Args:
        argv: Command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="Acquire the canonical dataset export.")
    parser.add_argument("--config", type=Path, default=None, help="Configuration file to use.")
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Download and hash the archive without extracting it.",
    )
    args = parser.parse_args(argv)

    paths = ProjectPaths.from_root()
    config_path = args.config or (paths.configs / "project.yaml")
    try:
        config = load_experiment_config(config_path)
        coordinates = _coordinates_from_config(config.dataset)
    except ConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    try:
        api_key = api_key_from_env()
    except MissingApiKeyError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3

    client = RoboflowClient(api_key)
    print(f"project:  {coordinates.public_url}")
    print(f"version:  {coordinates.version}  format: {coordinates.export_format}")

    try:
        project_payload = client.project_metadata(coordinates)
        export_payload, link = client.resolve_export(coordinates)
    except RoboflowError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 4

    project = project_payload.get("project", {})
    versions = project_payload.get("versions", [])
    version = next(
        (v for v in versions if str(v.get("id", "")).endswith(f"/{coordinates.version}")),
        {},
    )
    print(
        f"provider reports: project images={project.get('images')} "
        f"version images={version.get('images')}"
    )

    paths.ensure_data_dirs()
    archive_path = paths.data_external / f"{coordinates.slug}.zip"
    record_path = paths.data_external / f"{coordinates.slug}.provenance.json"

    try:
        result = download_if_absent(
            link,
            archive_path,
            expected_sha256=_recorded_digest(record_path),
        )
    except AcquisitionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 5
    state = "downloaded" if result.downloaded else "reused (digest matches record)"
    print(f"archive:  {archive_path.name}  {result.size_bytes} bytes  {state}")
    print(f"sha256:   {result.sha256}")

    extracted_root = paths.data_raw / coordinates.slug
    # The archive's provenance is written before extraction: the identity of the
    # downloaded bytes does not depend on a later step succeeding, and a crash
    # during extraction must not strand an unrecorded artifact.
    _write_record(
        record_path,
        paths=paths,
        config=config,
        coordinates=coordinates,
        project=project,
        version=version,
        export_payload=export_payload,
        archive_path=archive_path,
        members=0,
    )

    members = 0
    if not args.skip_extract:
        try:
            members = safe_extract_zip(archive_path, extracted_root)
        except AcquisitionError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 6
        print(f"extracted: {members} members -> data/raw/{coordinates.slug}/")

    _write_record(
        record_path,
        paths=paths,
        config=config,
        coordinates=coordinates,
        project=project,
        version=version,
        export_payload=export_payload,
        archive_path=archive_path,
        members=members,
    )
    print(f"provenance: data/external/{record_path.name}")
    return 0


def _write_record(
    record_path: Path,
    *,
    paths: ProjectPaths,
    config: Any,
    coordinates: DatasetCoordinates,
    project: dict[str, Any],
    version: dict[str, Any],
    export_payload: dict[str, Any],
    archive_path: Path,
    members: int,
) -> None:
    """Write the acquisition provenance record.

    Args:
        record_path: Destination of the record.
        paths: Project layout.
        config: Experiment configuration to snapshot.
        coordinates: Dataset identity.
        project: Provider project payload.
        version: Provider version payload.
        export_payload: Provider export payload.
        archive_path: Archive to hash into the record.
        members: Number of archive members extracted so far.
    """
    record = ProvenanceRecord.create(
        RECORD_NAME,
        phase=PHASE,
        config=config.to_dict(),
        repo_root=paths.root,
        details={
            "provider": "roboflow",
            "acquisition_method": (
                "Roboflow REST API, stdlib urllib, Authorization: Bearer header; "
                "export link resolved at run time and never stored"
            ),
            "workspace": coordinates.workspace,
            "project_slug": coordinates.project,
            "version": coordinates.version,
            "export_format": coordinates.export_format,
            "public_project_url": coordinates.public_url,
            "public_version_url": coordinates.public_version_url,
            "provider_reported_project": _subset(project, PROJECT_FIELDS),
            "provider_reported_version": _subset(version, VERSION_FIELDS),
            "provider_reported_export": _subset(
                export_payload.get("export", {}), ("format", "size")
            ),
            "extracted_to": f"data/raw/{coordinates.slug}",
            "extracted_members": members,
        },
    )
    record.add_output(archive_path, relative_to=paths.root)
    record.write_json(record_path)


if __name__ == "__main__":
    raise SystemExit(main())
