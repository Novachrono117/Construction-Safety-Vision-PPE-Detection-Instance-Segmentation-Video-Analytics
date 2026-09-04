"""Map the version-4 export back onto the 436 independent source images.

Answers two questions the canonical-snapshot decision depends on:

* can every source image be tied to an export record by a derivation rather than
  by a filename guess?
* does a *non-augmented* representation of every source image actually exist in
  the export, given that the export README claims augmentation produced two
  versions of each source image?

Both are settled by measurement, not by assertion. Identity comes from the
export's own ``extra.name``; the pristine representation is found by re-rendering
each source original through the export's declared preprocessing and comparing
pixels.

Runs entirely offline against already downloaded data.

Writes:
    reports/v4_source_mapping.csv   committed; identities and evidence, no paths

Usage:
    uv run python scripts/map_v4_sources.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

from construction_safety_vision.data.v4mapping import (
    EXACT,
    HIGH,
    NORMALISED,
    PRISTINE_MAE_MAX,
    MappingError,
    SourceMapping,
    V4Representation,
    choose_representation,
    mean_absolute_error,
    preprocessed_render,
    resolve_identity,
)
from construction_safety_vision.paths import ProjectPaths

EXPORT_DIRNAME = "construction-ppe-compliance-detection-v4-coco-segmentation"
"""Directory the version-4 export was extracted into, under ``data/raw``."""

EXPORT_SPLITS = ("train", "valid", "test")
"""Split directories inside the export."""

ANNOTATION_FILENAME = "_annotations.coco.json"
"""COCO document inside each split directory."""

SOURCE_IMAGE_DIRNAME = "source_images"
"""Directory of downloaded source originals, under ``data/external``."""

MANIFEST_FILENAME = "source_image_manifest.jsonl"
"""Phase 4A source manifest, under ``reports``."""

MAPPING_FILENAME = "v4_source_mapping.csv"
"""Committed mapping table."""

CSV_COLUMNS = (
    "source_image_id",
    "source_name",
    "v4_split",
    "v4_image_id_or_filename",
    "v4_coco_image_id",
    "mapping_method",
    "mapping_confidence",
    "is_augmented",
    "candidates",
    "selected_mae",
    "rejected_mae",
    "notes",
)
"""Column order of the committed mapping table."""


def load_source_manifest(path: Path) -> dict[str, dict]:
    """Load the phase 4A source manifest.

    Args:
        path: Manifest file.

    Returns:
        Records keyed by source image id.

    Raises:
        MappingError: If the manifest is missing.
    """
    if not path.is_file():
        msg = f"Source manifest not found at {path.name}; run scripts/fetch_source_inventory.py"
        raise MappingError(msg)
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return {record["image_id"]: record for record in records}


def load_export_records(export_root: Path) -> dict[str, list[V4Representation]]:
    """Read every export image record, grouped by its original source name.

    Args:
        export_root: Directory the export was extracted into.

    Returns:
        Export records keyed by ``extra.name``.

    Raises:
        MappingError: If the export or one of its annotation documents is absent.
    """
    if not export_root.is_dir():
        msg = f"Version-4 export not found at {export_root.name}; run scripts/download_dataset.py"
        raise MappingError(msg)
    grouped: dict[str, list[V4Representation]] = {}
    for split in EXPORT_SPLITS:
        document = export_root / split / ANNOTATION_FILENAME
        if not document.is_file():
            msg = f"Export split {split!r} has no {ANNOTATION_FILENAME}"
            raise MappingError(msg)
        payload = json.loads(document.read_text(encoding="utf-8"))
        for image in payload.get("images", []):
            source_name = str((image.get("extra") or {}).get("name", "")).strip()
            if not source_name:
                msg = f"Export record {image.get('file_name')!r} carries no extra.name"
                raise MappingError(msg)
            grouped.setdefault(source_name, []).append(
                V4Representation(
                    split=split,
                    file_name=str(image["file_name"]),
                    coco_image_id=int(image["id"]),
                    source_name=source_name,
                )
            )
    return grouped


def build_mappings(
    manifest: dict[str, dict],
    export_records: dict[str, list[V4Representation]],
    source_image_dir: Path,
    export_root: Path,
) -> list[SourceMapping]:
    """Resolve identity and choose the pristine representation for every source.

    Args:
        manifest: Source records keyed by image id.
        export_records: Export records keyed by original source name.
        source_image_dir: Directory holding the downloaded source originals.
        export_root: Directory the version-4 export was extracted into.

    Returns:
        One mapping per source image, ordered by source image id.

    Raises:
        MappingError: If identity resolution is ambiguous.
    """
    source_names = {image_id: record["name"] for image_id, record in manifest.items()}
    resolved, unmatched, unclaimed = resolve_identity(source_names, set(export_records))
    if unmatched or unclaimed:
        print(
            f"  identity: {len(unmatched)} source image(s) unmatched, "
            f"{len(unclaimed)} export name(s) unclaimed",
            file=sys.stderr,
        )

    mappings: list[SourceMapping] = []
    for index, image_id in enumerate(sorted(manifest), start=1):
        name = source_names[image_id]
        if image_id not in resolved:
            mappings.append(
                SourceMapping(
                    source_image_id=image_id,
                    source_name=name,
                    candidates=[],
                    notes="no export record carries this source name",
                )
            )
            continue

        export_name, method = resolved[image_id]
        candidates = export_records[export_name]
        original = source_image_dir / f"{image_id}.jpg"
        if original.is_file():
            reference = preprocessed_render(original)
            scored = [
                V4Representation(
                    split=c.split,
                    file_name=c.file_name,
                    coco_image_id=c.coco_image_id,
                    source_name=c.source_name,
                    mae=mean_absolute_error(
                        reference, preprocessed_render(export_root / c.split / c.file_name)
                    ),
                )
                for c in candidates
            ]
            selected, confidence, note = choose_representation(scored)
        else:
            scored = list(candidates)
            selected, confidence, note = None, "ambiguous", "source original not downloaded"

        mappings.append(
            SourceMapping(
                source_image_id=image_id,
                source_name=name,
                candidates=scored,
                selected=selected,
                identity_method=method,
                confidence=confidence,
                notes=note,
            )
        )
        if index % 50 == 0 or index == len(manifest):
            print(f"  {index}/{len(manifest)}", flush=True)
    return mappings


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write the mapping table with LF endings.

    Args:
        path: Destination file.
        rows: Rows to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    """Build and write the version-4 source mapping.

    Args:
        argv: Command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)

    paths = ProjectPaths.from_root()
    export_root = paths.data_raw / EXPORT_DIRNAME
    try:
        manifest = load_source_manifest(paths.reports / MANIFEST_FILENAME)
        export_records = load_export_records(export_root)
    except MappingError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"source images: {len(manifest)}   export source names: {len(export_records)}")
    print("scoring export representations against deterministic re-renders ...", flush=True)
    try:
        mappings = build_mappings(
            manifest, export_records, paths.data_external / SOURCE_IMAGE_DIRNAME, export_root
        )
    except MappingError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3

    write_csv(paths.reports / MAPPING_FILENAME, [m.csv_row() for m in mappings])

    methods = Counter(m.identity_method for m in mappings)
    confidences = Counter(m.confidence for m in mappings)
    selected = [m for m in mappings if m.selected is not None]
    pristine = [m for m in selected if not m.is_augmented_selection]
    per_split = Counter(m.selected.split for m in selected if m.selected)

    print("\n--- mapping summary ---")
    print(f"source images mapped:            {len(selected)}/{len(mappings)}")
    print(f"identity methods:                {dict(sorted(methods.items()))}")
    print(f"selection confidence:            {dict(sorted(confidences.items()))}")
    print(f"non-augmented representations:   {len(pristine)}")
    print(f"selected per export split:       {dict(sorted(per_split.items()))}")
    print(f"pristine threshold (MAE):        {PRISTINE_MAE_MAX}")

    scored = [m.selected.mae for m in selected if m.selected and m.selected.mae is not None]
    rejected = [
        c.mae for m in mappings for c in m.candidates if c is not m.selected and c.mae is not None
    ]
    if scored:
        print(f"selected MAE   min/max:          {min(scored):.3f} / {max(scored):.3f}")
    if rejected:
        print(f"rejected MAE   min/max:          {min(rejected):.3f} / {max(rejected):.3f}")

    ok = (
        len(pristine) == len(mappings)
        and confidences.get(HIGH, 0) == len(mappings)
        and methods.get(EXACT, 0) + methods.get(NORMALISED, 0) == len(mappings)
    )
    print(f"\nevery source image has exactly one non-augmented v4 representation: {ok}")
    print(f"wrote {paths.reports / MAPPING_FILENAME}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
