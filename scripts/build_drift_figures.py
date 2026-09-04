"""Visual evidence for the phase 5A annotation-drift and microannotation findings.

Two questions in phase 5A are decided partly by looking:

* the drift analysis says 76 annotations appeared and 6 disappeared. Are the
  additions real objects that were previously unlabelled, or noise?
* two live-source records carry no geometry at all. Is there anything at those
  coordinates, or are they degenerate leftovers?

Both sheets are drawn only from already downloaded source originals, in original
image coordinates, and are deliberately small: they are committed to the
repository as review evidence, not as deliverables.

Requires:
    data/interim/source_geometry.jsonl     scripts/recover_source_geometry.py
    reports/annotation_drift.csv           scripts/analyze_annotation_drift.py
    data/external/source_images/           scripts/download_source_images.py

Writes:
    reports/figures/review_l_annotation_drift.jpg
    reports/figures/review_m_microannotations.jpg
    appends rows to reports/manual_review_manifest.csv

Usage:
    uv run python scripts/build_drift_figures.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from PIL import Image

from construction_safety_vision.data.drift import DriftInstance, match_instances, normalise_bbox
from construction_safety_vision.data.geometry import bbox_from_segmentation
from construction_safety_vision.paths import ProjectPaths, long_path

FIGURE_DPI = 92
"""Resolution for review sheets; modest because they are committed."""

JPEG_QUALITY = 78
"""Matches the phase 4B review package."""

ADDED_COLOUR = "#00C853"
"""Colour of an annotation present now and absent from the snapshot."""

REMOVED_COLOUR = "#FF3B30"
"""Colour of an annotation present in the snapshot and absent now."""

KEPT_COLOUR = "#FFB000"
"""Colour of an annotation present in both states."""

MICRO_COLOUR = "#FF3B30"
"""Colour of a geometry-less microannotation."""

MAX_DRIFT_PANELS = 12
"""Panels on the drift sheet."""

DRIFT_COLUMNS = 4
"""Panels per row on the drift sheet."""

ZOOM_PAD_PX = 60
"""Context, in original pixels, around a microannotation crop."""

EXPORT_DIRNAME = "construction-ppe-compliance-detection-v4-coco-segmentation"
"""Directory the version-4 export was extracted into, under ``data/raw``."""

MICRO_IMAGE_ID = "OQJwjQoYsf1KUgr9G0V8"
"""Source image holding the two geometry-less records."""

DRIFT_FIGURE = "review_l_annotation_drift.jpg"
"""Committed drift sheet."""

MICRO_FIGURE = "review_m_microannotations.jpg"
"""Committed microannotation sheet."""

ADDED_FIGURE = "review_n_added_annotations.jpg"
"""Committed sheet showing every annotation added since the snapshot."""

ADDED_COLUMNS = 8
"""Panels per row on the additions sheet."""

MIN_CROP_PIXELS = 150
"""Smallest crop side, in output pixels, before a panel is upscaled to be legible."""

MANIFEST_COLUMNS = ("short_id", "image_id", "name", "split", "reason", "contact_sheet", "note")
"""Column order of the manual review manifest."""


def _write(fig: Any, destination: Path) -> str:
    """Save a sheet as JPEG and close the figure.

    Args:
        fig: Matplotlib figure.
        destination: Output file.

    Returns:
        The repository-relative path.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=FIGURE_DPI, pil_kwargs={"quality": JPEG_QUALITY, "optimize": True})
    plt.close(fig)
    return f"reports/figures/{destination.name}"


def load_geometry(path: Path) -> dict[str, dict]:
    """Load the recovered live geometry.

    Args:
        path: The interim geometry file.

    Returns:
        Records keyed by source image id.
    """
    return {
        json.loads(line)["image_id"]: json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    }


def load_snapshot_annotations(export_root: Path) -> dict[tuple[str, int], dict]:
    """Read the export's image records with their annotations attached.

    Args:
        export_root: Directory the export was extracted into.

    Returns:
        Records keyed by ``(split, coco image id)``.
    """
    images: dict[tuple[str, int], dict] = {}
    for split in ("train", "valid", "test"):
        payload = json.loads(
            (export_root / split / "_annotations.coco.json").read_text(encoding="utf-8")
        )
        categories = {int(c["id"]): str(c["name"]) for c in payload["categories"]}
        for image in payload["images"]:
            images[(split, int(image["id"]))] = {
                "width": int(image["width"]),
                "height": int(image["height"]),
                "annotations": [],
            }
        for annotation in payload["annotations"]:
            key = (split, int(annotation["image_id"]))
            if key in images:
                images[key]["annotations"].append(
                    {
                        "id": int(annotation["id"]),
                        "label": categories[int(annotation["category_id"])],
                        "segmentation": annotation.get("segmentation"),
                    }
                )
    return images


def _draw_box(
    axis: Any, box: tuple[float, float, float, float], width: int, height: int, colour: str
) -> None:
    """Draw one normalised box on an axis showing a full image.

    Args:
        axis: Matplotlib axis.
        box: Box in the unit square.
        width: Image width in pixels.
        height: Image height in pixels.
        colour: Edge colour.
    """
    x, y, w, h = box
    axis.add_patch(
        mpatches.Rectangle(
            (x * width, y * height),
            w * width,
            h * height,
            fill=False,
            edgecolor=colour,
            linewidth=1.4,
        )
    )


def drift_sheet(
    rows: list[dict],
    geometry: dict[str, dict],
    snapshot: dict[tuple[str, int], dict],
    mapping: dict[str, dict],
    source_dir: Path,
    destination: Path,
) -> tuple[str, list[dict]]:
    """Draw the most-edited images with added, removed and retained annotations.

    Args:
        rows: Drift table rows, already ordered by how much changed.
        geometry: Recovered live geometry keyed by source image id.
        snapshot: Export records keyed by ``(split, coco image id)``.
        mapping: Source-to-export mapping keyed by source image id.
        source_dir: Directory of downloaded source originals.
        destination: Output file.

    Returns:
        The repository-relative figure path and the manifest rows it justifies.
    """
    chosen = rows[:MAX_DRIFT_PANELS]
    columns = DRIFT_COLUMNS
    figure_rows = (len(chosen) + columns - 1) // columns
    fig, axes = plt.subplots(
        figure_rows, columns, figsize=(3.1 * columns, 3.1 * figure_rows), constrained_layout=True
    )
    flat = axes.ravel() if hasattr(axes, "ravel") else [axes]
    manifest: list[dict] = []

    for axis, row in zip(flat, chosen, strict=False):
        image_id = row["source_image_id"]
        record = geometry[image_id]
        map_row = mapping[image_id]
        snap = snapshot[(map_row["v4_split"], int(map_row["v4_coco_image_id"]))]

        current = [
            DriftInstance(
                str(a["annotation_id"]),
                str(a["label"]),
                normalise_bbox(a["bbox"], width=record["width"], height=record["height"]),
            )
            for a in record["annotations"]
        ]
        old = [
            DriftInstance(
                str(a["id"]),
                a["label"],
                normalise_bbox(
                    bbox_from_segmentation(a["segmentation"]).as_list(),
                    width=snap["width"],
                    height=snap["height"],
                ),
            )
            for a in snap["annotations"]
        ]
        matches, added, removed = match_instances(current, old)

        with Image.open(long_path(source_dir / f"{image_id}.jpg")) as handle:
            picture = handle.convert("RGB")
        axis.imshow(picture)
        width, height = picture.size
        for instance, _, _ in matches:
            _draw_box(axis, instance.bbox, width, height, KEPT_COLOUR)
        for instance in removed:
            _draw_box(axis, instance.bbox, width, height, REMOVED_COLOUR)
        for instance in added:
            _draw_box(axis, instance.bbox, width, height, ADDED_COLOUR)

        axis.set_title(
            f"{image_id[:8]}  {row['provider_split']}\n"
            f"v4 {row['v4_count']} -> now {row['current_count']}  "
            f"(+{len(added)} / -{len(removed)})",
            fontsize=7,
        )
        axis.set_xticks([])
        axis.set_yticks([])
        manifest.append(
            {
                "short_id": image_id[:8],
                "image_id": image_id,
                "name": record["name"],
                "split": row["provider_split"],
                "reason": "annotation_drift",
                "contact_sheet": f"reports/figures/{destination.name}",
                "note": f"added={len(added)} removed={len(removed)}",
            }
        )

    for axis in flat[len(chosen) :]:
        axis.axis("off")

    fig.suptitle(
        "L. Annotation drift: green = added since v4, red = removed, amber = retained",
        fontsize=10,
    )
    return _write(fig, destination), manifest


def collect_additions(
    geometry: dict[str, dict],
    snapshot: dict[tuple[str, int], dict],
    mapping: dict[str, dict],
) -> list[tuple[str, str, str, tuple[float, float, float, float]]]:
    """Find every annotation the live state has and the snapshot does not.

    Args:
        geometry: Recovered live geometry keyed by source image id.
        snapshot: Export records keyed by ``(split, coco image id)``.
        mapping: Source-to-export mapping keyed by source image id.

    Returns:
        One ``(source image id, annotation id, label, pixel box)`` per addition,
        ordered deterministically.
    """
    found: list[tuple[str, str, str, tuple[float, float, float, float]]] = []
    for image_id in sorted(geometry):
        record = geometry[image_id]
        row = mapping.get(image_id)
        if not row or not row["v4_coco_image_id"]:
            continue
        snap = snapshot[(row["v4_split"], int(row["v4_coco_image_id"]))]
        current = [
            DriftInstance(
                str(a["annotation_id"]),
                str(a["label"]),
                normalise_bbox(a["bbox"], width=record["width"], height=record["height"]),
            )
            for a in record["annotations"]
        ]
        old = [
            DriftInstance(
                str(a["id"]),
                a["label"],
                normalise_bbox(
                    bbox_from_segmentation(a["segmentation"]).as_list(),
                    width=snap["width"],
                    height=snap["height"],
                ),
            )
            for a in snap["annotations"]
        ]
        _, added, _ = match_instances(current, old)
        for instance in added:
            x, y, w, h = instance.bbox
            found.append(
                (
                    image_id,
                    instance.key,
                    instance.label,
                    (
                        x * record["width"],
                        y * record["height"],
                        w * record["width"],
                        h * record["height"],
                    ),
                )
            )
    return found


def added_sheet(
    additions: list[tuple[str, str, str, tuple[float, float, float, float]]],
    source_dir: Path,
    destination: Path,
) -> str:
    """Draw every annotation added since the snapshot, zoomed to be judgeable.

    The drift measurement establishes that no addition covers a previously
    unlabelled object. What it cannot say is what the additions are drawn on;
    that needs looking, so every one gets a panel rather than a sample.

    Args:
        additions: One ``(source image id, annotation id, label, pixel box)``
            per addition, in a deterministic order.
        source_dir: Directory of downloaded source originals.
        destination: Output file.

    Returns:
        The repository-relative figure path.
    """
    columns = ADDED_COLUMNS
    rows = (len(additions) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(1.55 * columns, 1.75 * rows))
    flat = axes.ravel() if hasattr(axes, "ravel") else [axes]

    for axis, (image_id, key, label, (x, y, w, h)) in zip(flat, additions, strict=False):
        with Image.open(long_path(source_dir / f"{image_id}.jpg")) as handle:
            picture = handle.convert("RGB")
        pad = max(25, int(max(w, h) * 2.5))
        left, top = max(0, int(x - pad)), max(0, int(y - pad))
        right, bottom = min(picture.width, int(x + w + pad)), min(picture.height, int(y + h + pad))
        crop = picture.crop((left, top, right, bottom))
        scale = max(1, int(MIN_CROP_PIXELS / max(1, crop.width)))
        if scale > 1:
            crop = crop.resize((crop.width * scale, crop.height * scale), Image.LANCZOS)
        axis.imshow(crop)
        axis.add_patch(
            mpatches.Rectangle(
                ((x - left) * scale, (y - top) * scale),
                w * scale,
                h * scale,
                fill=False,
                edgecolor=ADDED_COLOUR,
                linewidth=1.3,
            )
        )
        axis.set_title(f"{image_id[:8]} {key}\n{label} {w:.0f}x{h:.0f}px", fontsize=5.5)
        axis.set_xticks([])
        axis.set_yticks([])

    for axis in flat[len(additions) :]:
        axis.axis("off")
    fig.suptitle(
        f"N. All {len(additions)} annotations added since v4, zoomed (green = the addition)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    return _write(fig, destination)


def micro_sheet(record: dict, source_dir: Path, destination: Path) -> tuple[str, list[dict]]:
    """Draw the geometry-less microannotations in context and zoomed.

    Args:
        record: The recovered live record for the affected image.
        source_dir: Directory of downloaded source originals.
        destination: Output file.

    Returns:
        The repository-relative figure path and the manifest rows it justifies.
    """
    micro = [a for a in record["annotations"] if a["geometry_kind"] == "UNSUPPORTED"]
    with Image.open(long_path(source_dir / f"{record['image_id']}.jpg")) as handle:
        picture = handle.convert("RGB")

    fig, axes = plt.subplots(1, 1 + len(micro), figsize=(4.2 * (1 + len(micro)), 4.0))
    flat = list(axes.ravel()) if hasattr(axes, "ravel") else [axes]

    flat[0].imshow(picture)
    flat[0].set_xticks([])
    flat[0].set_yticks([])
    flat[0].set_title(
        f"{record['image_id'][:8]}  {record['width']}x{record['height']}  "
        f"split={record['split']}\nall {len(record['annotations'])} records",
        fontsize=8,
    )
    for annotation in record["annotations"]:
        x, y, w, h = annotation["bbox"]
        colour = MICRO_COLOUR if annotation["geometry_kind"] == "UNSUPPORTED" else KEPT_COLOUR
        flat[0].add_patch(
            mpatches.Rectangle((x, y), w, h, fill=False, edgecolor=colour, linewidth=1.2)
        )

    for axis, annotation in zip(flat[1:], micro, strict=False):
        x, y, w, h = annotation["bbox"]
        left = max(0, int(x - ZOOM_PAD_PX))
        top = max(0, int(y - ZOOM_PAD_PX))
        right = min(record["width"], int(x + w + ZOOM_PAD_PX))
        bottom = min(record["height"], int(y + h + ZOOM_PAD_PX))
        axis.imshow(picture.crop((left, top, right, bottom)))
        axis.add_patch(
            mpatches.Rectangle(
                (x - left, y - top), w, h, fill=False, edgecolor=MICRO_COLOUR, linewidth=1.6
            )
        )
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title(
            f"id={annotation['annotation_id']}  {annotation['label']}\n"
            f"bbox {w:.1f}x{h:.1f} px at ({x:.1f}, {y:.1f})\n"
            f"provider type: none, no geometry",
            fontsize=8,
        )

    fig.suptitle("M. Geometry-less source records (red) shown in context and zoomed", fontsize=10)
    fig.tight_layout()
    manifest = [
        {
            "short_id": f"{record['image_id'][:8]}-{a['annotation_id']}",
            "image_id": record["image_id"],
            "name": record["name"],
            "split": record["split"],
            "reason": "geometry_less_microannotation",
            "contact_sheet": f"reports/figures/{destination.name}",
            "note": f"annotation {a['annotation_id']} label={a['label']} no segmentation",
        }
        for a in micro
    ]
    return _write(fig, destination), manifest


def append_manifest(path: Path, rows: list[dict]) -> int:
    """Append review rows, replacing any earlier rows for the same sheets.

    Re-running the script must not duplicate rows, so entries pointing at the
    sheets this script owns are rewritten rather than added to.

    Args:
        path: The manual review manifest.
        rows: Rows to record.

    Returns:
        The total number of rows in the rewritten manifest.
    """
    owned = {row["contact_sheet"] for row in rows}
    existing: list[dict] = []
    if path.is_file():
        with path.open(encoding="utf-8", newline="") as handle:
            existing = [r for r in csv.DictReader(handle) if r["contact_sheet"] not in owned]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MANIFEST_COLUMNS), lineterminator="\n")
        writer.writeheader()
        writer.writerows(existing)
        writer.writerows(rows)
    return len(existing) + len(rows)


def main(argv: list[str] | None = None) -> int:
    """Build the phase 5A review sheets.

    Args:
        argv: Command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)

    paths = ProjectPaths.from_root()
    source_dir = paths.data_external / "source_images"
    geometry_path = paths.data_interim / "source_geometry.jsonl"
    drift_path = paths.reports / "annotation_drift.csv"
    mapping_path = paths.reports / "v4_source_mapping.csv"

    for required in (geometry_path, drift_path, mapping_path):
        if not required.is_file():
            print(
                f"ERROR: {required.name} not found; run the phase 5A scripts first",
                file=sys.stderr,
            )
            return 2
    if not source_dir.is_dir():
        print("ERROR: source originals not downloaded", file=sys.stderr)
        return 2

    geometry = load_geometry(geometry_path)
    with drift_path.open(encoding="utf-8", newline="") as handle:
        drift_rows = list(csv.DictReader(handle))
    with mapping_path.open(encoding="utf-8", newline="") as handle:
        mapping = {r["source_image_id"]: r for r in csv.DictReader(handle)}
    snapshot = load_snapshot_annotations(paths.data_raw / EXPORT_DIRNAME)

    edited = sorted(
        (r for r in drift_rows if r["added_class_counts"] or r["removed_class_counts"]),
        key=lambda r: (-abs(int(r["delta"])), r["source_image_id"]),
    )
    drift_figure, drift_manifest = drift_sheet(
        edited, geometry, snapshot, mapping, source_dir, paths.figures / DRIFT_FIGURE
    )
    print(f"wrote {drift_figure} ({len(edited[:MAX_DRIFT_PANELS])} panels)")

    additions = collect_additions(geometry, snapshot, mapping)
    added_figure = added_sheet(additions, source_dir, paths.figures / ADDED_FIGURE)
    print(f"wrote {added_figure} ({len(additions)} panels)")

    micro_figure, micro_manifest = micro_sheet(
        geometry[MICRO_IMAGE_ID], source_dir, paths.figures / MICRO_FIGURE
    )
    print(f"wrote {micro_figure}")

    total = append_manifest(
        paths.reports / "manual_review_manifest.csv", drift_manifest + micro_manifest
    )
    print(f"manual review manifest now holds {total} rows")

    for name in (DRIFT_FIGURE, MICRO_FIGURE):
        size = (paths.figures / name).stat().st_size
        print(f"  {name}: {size / 1024:.0f} KiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
