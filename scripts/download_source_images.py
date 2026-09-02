"""Acquire the 436 ORIGINAL source images and measure them.

The version-4 export is preprocessed: every image was auto-oriented and stretched
to 640x640, and the train split was additionally augmented. Original resolution,
aspect ratio, luminance and file size therefore cannot be studied there. This
script fetches the untouched originals instead.

Images are stored in the git-ignored data layer and are never committed. The
inventory is re-walked here rather than read from the committed manifest,
because addressing an original requires the provider's account identifier, which
deliberately does not appear in any committed file.

Writes:
    data/external/source_images/<image_id>.jpg   git-ignored originals
    data/interim/source_image_stats.jsonl        git-ignored measurements

Usage:
    uv run python scripts/download_source_images.py
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from dataclasses import asdict
from pathlib import Path

from construction_safety_vision.config import ConfigError, load_experiment_config
from construction_safety_vision.data.imagestats import compute_image_stats
from construction_safety_vision.data.roboflow import (
    USER_AGENT,
    DatasetCoordinates,
    MissingApiKeyError,
    RoboflowClient,
    RoboflowError,
    api_key_from_env,
    original_image_url,
)
from construction_safety_vision.data.source import InventoryError, collect_source_images
from construction_safety_vision.env import load_project_env
from construction_safety_vision.paths import ProjectPaths

EXPECTED_SOURCE_IMAGES = 436
"""Source population established in phase 3."""

DOWNLOAD_TIMEOUT = 120.0
"""Socket timeout in seconds per image."""


def fetch_image(url: str, destination: Path, *, timeout: float = DOWNLOAD_TIMEOUT) -> str:
    """Download one image to disk, writing atomically.

    Args:
        url: Source URL. Never included in an error message.
        destination: Final path.
        timeout: Socket timeout in seconds.

    Returns:
        An empty string on success, or a short failure description.
    """
    partial = destination.with_suffix(".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read()
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
        return f"{type(exc).__name__}"
    if not payload:
        return "empty response"
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial.write_bytes(payload)
    partial.replace(destination)
    return ""


def main(argv: list[str] | None = None) -> int:
    """Acquire and measure the original source images.

    Args:
        argv: Command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code; non-zero when the population is incomplete.
    """
    parser = argparse.ArgumentParser(description="Acquire the original source images.")
    parser.add_argument("--config", type=Path, default=None, help="Configuration file to use.")
    args = parser.parse_args(argv)

    paths = ProjectPaths.from_root()
    load_project_env(paths.root)
    try:
        config = load_experiment_config(args.config or (paths.configs / "project.yaml"))
        api_key = api_key_from_env()
    except (ConfigError, MissingApiKeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 2

    dataset = config.dataset
    coordinates = DatasetCoordinates(
        workspace=dataset.workspace,
        project=dataset.project_slug,
        version=dataset.version,
        export_format=dataset.export_format,
    )

    print("walking source inventory ...", flush=True)
    try:
        walk = collect_source_images(RoboflowClient(api_key), coordinates)
    except (RoboflowError, InventoryError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 3
    records = walk.records
    print(f"inventory: {len(records)} records, {walk.pages_fetched} pages", flush=True)

    target_dir = paths.data_external / "source_images"
    target_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    reused = 0
    failures: list[dict[str, str]] = []
    rows: list[dict] = []

    for index, record in enumerate(records, start=1):
        destination = target_dir / f"{record.image_id}.jpg"
        if destination.exists() and destination.stat().st_size > 0:
            reused += 1
        else:
            error = fetch_image(original_image_url(record.owner, record.image_id), destination)
            if error:
                failures.append({"image_id": record.image_id, "stage": "download", "error": error})
                continue
            downloaded += 1

        stats = compute_image_stats(destination)
        if not stats.decoded:
            failures.append({"image_id": record.image_id, "stage": "decode", "error": stats.error})
        rows.append(
            {
                "image_id": record.image_id,
                "relative_path": f"data/external/source_images/{record.image_id}.jpg",
                "name": record.name,
                "split": record.split,
                "provider_width": record.width,
                "provider_height": record.height,
                **{k: v for k, v in asdict(stats).items() if k != "error"},
                "decode_error": stats.error,
                "aspect_ratio": stats.aspect_ratio,
                "pixels": stats.pixels,
                "dynamic_range": stats.dynamic_range,
            }
        )
        if index % 50 == 0 or index == len(records):
            print(f"  {index}/{len(records)}  downloaded={downloaded} reused={reused}", flush=True)

    stats_path = paths.data_interim / "source_image_stats.jsonl"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )

    decoded = sum(1 for r in rows if r["decoded"])
    mismatched = [
        r["image_id"]
        for r in rows
        if r["decoded"] and (r["width"], r["height"]) != (r["provider_width"], r["provider_height"])
    ]

    print(flush=True)
    print(f"inventory records   : {len(records)} (expected {EXPECTED_SOURCE_IMAGES})", flush=True)
    print(f"downloaded now      : {downloaded}", flush=True)
    print(f"reused from disk    : {reused}", flush=True)
    print(f"decoded successfully: {decoded}", flush=True)
    print(f"failures            : {len(failures)}", flush=True)
    for failure in failures[:10]:
        print(f"  {failure}", flush=True)
    print(f"provider/decoded size mismatches: {len(mismatched)}", flush=True)
    print(f"wrote data/interim/{stats_path.name}", flush=True)

    if len(records) != EXPECTED_SOURCE_IMAGES or decoded != EXPECTED_SOURCE_IMAGES:
        print(
            f"\nINCOMPLETE: {decoded} of {EXPECTED_SOURCE_IMAGES} originals are usable.",
            file=sys.stderr,
            flush=True,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
