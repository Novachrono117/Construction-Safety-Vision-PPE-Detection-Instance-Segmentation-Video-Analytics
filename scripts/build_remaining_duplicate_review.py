"""Contact sheet for near-duplicate candidates that no human has dispositioned.

Phase 4A raised 11 perceptual near-duplicate candidates. Phase 4B reviewed the
six that crossed a provider split boundary. The phase 4A review package showed
the eight strongest candidates by perceptual distance, which happened to include
four of those six plus four same-split pairs - so a candidate could be both
same-split *and* outside the eight strongest, and then appear on no sheet at all.

This script draws exactly the candidates that still carry no human decision, so
the gap is closed by looking rather than by assuming. It merges nothing and
decides nothing.

Runs entirely offline. Requires:

* ``reports/unconfirmed_group_candidates.csv``  scripts/build_modeling_population.py
* ``reports/near_duplicate_candidates.csv``     scripts/audit_source_dataset.py
* ``data/external/source_images/``              scripts/download_source_images.py

Writes:
    reports/figures/review_o_remaining_near_duplicates.jpg
    appends its rows to reports/manual_review_manifest.csv

Usage:
    uv run python scripts/build_remaining_duplicate_review.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from construction_safety_vision.paths import ProjectPaths, long_path

FIGURE_DPI = 92
"""Resolution for review sheets; modest because they are committed."""

JPEG_QUALITY = 80
"""Matches the other review packages."""

REVIEW_FIGURE = "review_o_remaining_near_duplicates.jpg"
"""Committed sheet. Letters a-n are taken by the phase 4 and 5A packages."""

UNCONFIRMED_CSV = "unconfirmed_group_candidates.csv"
"""Candidates with no human decision, under ``reports``."""

DISTANCES_CSV = "near_duplicate_candidates.csv"
"""Perceptual distances measured in phase 4A, under ``reports``."""

SOURCE_IMAGE_DIRNAME = "source_images"
"""Directory of downloaded source originals, under ``data/external``."""

REVIEW_MANIFEST = "manual_review_manifest.csv"
"""Which sheet showed which image, under ``reports``."""

MANIFEST_COLUMNS = ("short_id", "image_id", "name", "split", "reason", "contact_sheet", "note")
"""Column order of the manual review manifest."""


class ReviewInputError(RuntimeError):
    """Raised when a required input artifact is missing."""


def read_csv(path: Path) -> list[dict]:
    """Read a committed CSV artifact.

    Args:
        path: File to read.

    Returns:
        The parsed rows.

    Raises:
        ReviewInputError: If the file is absent.
    """
    if not path.is_file():
        msg = f"{path.name} not found; run the earlier phase scripts first"
        raise ReviewInputError(msg)
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def distance_index(rows: list[dict]) -> dict[frozenset[str], dict]:
    """Index the measured perceptual distances by the pair they describe.

    Args:
        rows: Rows of the phase 4A candidate table.

    Returns:
        Distance records keyed by the unordered image pair.
    """
    return {frozenset((row["image_a_id"], row["image_b_id"])): row for row in rows}


def draw_sheet(
    candidates: list[dict],
    distances: dict[frozenset[str], dict],
    source_dir: Path,
    destination: Path,
) -> str:
    """Draw one row per unresolved candidate, both members side by side.

    Args:
        candidates: Unresolved candidate rows.
        distances: Measured distances keyed by image pair.
        source_dir: Directory of downloaded source originals.
        destination: Output file.

    Returns:
        The repository-relative figure path.

    Raises:
        ReviewInputError: If a member image is not on disk.
    """
    rows = len(candidates)
    fig, axes = plt.subplots(rows, 2, figsize=(9.0, 4.6 * rows), squeeze=False)

    for row, candidate in enumerate(candidates):
        members = [value for value in candidate["source_image_ids"].split(";") if value]
        measured: dict[str, Any] = distances.get(frozenset(members), {})
        for column, image_id in enumerate(members[:2]):
            path = source_dir / f"{image_id}.jpg"
            if not path.is_file():
                msg = f"Source original for {image_id!r} is not downloaded"
                raise ReviewInputError(msg)
            with Image.open(long_path(path)) as handle:
                axes[row][column].imshow(handle.convert("RGB"))
            split = measured.get(f"provider_split_{'ab'[column]}", "?")
            axes[row][column].set_title(
                f"{image_id[:8]}   (provider split: {split}, provenance only)\n{image_id}",
                fontsize=7,
            )
            axes[row][column].set_xticks([])
            axes[row][column].set_yticks([])
        axes[row][0].set_ylabel(
            f"{candidate['candidate_id']}\n"
            f"dHash {measured.get('dhash_distance', '?')}   "
            f"pHash {measured.get('phash_distance', '?')}",
            fontsize=8,
        )

    fig.suptitle(
        "O. Near-duplicate candidates still awaiting a human decision\n"
        "Same content, or merely similar? A confirmed duplicate must stay inside one split.",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=FIGURE_DPI, pil_kwargs={"quality": JPEG_QUALITY, "optimize": True})
    plt.close(fig)
    return f"reports/figures/{destination.name}"


def record_provenance(
    path: Path,
    candidates: list[dict],
    distances: dict[frozenset[str], dict],
    figure: str,
) -> int:
    """Record which images this sheet displayed, so a verdict can cite it.

    A decision may only reference a sheet the reviewers were actually shown, and
    that is checked against this manifest. Rows for this sheet are rewritten
    rather than appended, so re-running the script does not duplicate them.

    Args:
        path: The manual review manifest.
        candidates: The candidates drawn on the sheet.
        distances: Measured distances keyed by image pair.
        figure: Repository-relative path of the sheet.

    Returns:
        The total number of rows in the rewritten manifest.
    """
    rows: list[dict] = []
    for candidate in candidates:
        members = [value for value in candidate["source_image_ids"].split(";") if value]
        measured = distances.get(frozenset(members), {})
        for index, image_id in enumerate(members):
            rows.append(
                {
                    "short_id": image_id[:8],
                    "image_id": image_id,
                    "name": "",
                    "split": measured.get(f"provider_split_{'ab'[index]}", ""),
                    "reason": "unresolved_near_duplicate_candidate",
                    "contact_sheet": figure,
                    "note": (
                        f"{candidate['candidate_id']} "
                        f"dhash={measured.get('dhash_distance', '?')} "
                        f"phash={measured.get('phash_distance', '?')}"
                    ),
                }
            )

    existing: list[dict] = []
    if path.is_file():
        with path.open(encoding="utf-8", newline="") as handle:
            existing = [r for r in csv.DictReader(handle) if r["contact_sheet"] != figure]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MANIFEST_COLUMNS), lineterminator="\n")
        writer.writeheader()
        writer.writerows(existing)
        writer.writerows(rows)
    return len(existing) + len(rows)


def main(argv: list[str] | None = None) -> int:
    """Draw the outstanding near-duplicate candidates.

    Args:
        argv: Command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)

    paths = ProjectPaths.from_root()
    try:
        candidates = read_csv(paths.reports / UNCONFIRMED_CSV)
        distances = distance_index(read_csv(paths.reports / DISTANCES_CSV))
    except ReviewInputError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if not candidates:
        print("no unresolved near-duplicate candidates; nothing to draw")
        return 0

    print(f"unresolved candidates awaiting a decision: {len(candidates)}")
    for candidate in candidates:
        members = candidate["source_image_ids"].split(";")
        measured = distances.get(frozenset(members), {})
        print(
            f"  {candidate['candidate_id']}  {' ~ '.join(m[:8] for m in members)}  "
            f"dHash={measured.get('dhash_distance', '?')} "
            f"pHash={measured.get('phash_distance', '?')}"
        )

    try:
        figure = draw_sheet(
            candidates,
            distances,
            paths.data_external / SOURCE_IMAGE_DIRNAME,
            paths.figures / REVIEW_FIGURE,
        )
    except ReviewInputError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3

    recorded = record_provenance(paths.reports / REVIEW_MANIFEST, candidates, distances, figure)
    size = (paths.figures / REVIEW_FIGURE).stat().st_size
    print(f"\nwrote {figure} ({size / 1024:.0f} KiB)")
    print(f"manual review manifest now holds {recorded} rows")
    print("This sheet decides nothing. No candidate was merged.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
