"""Build the human visual-review package for phase 4B.

Automated statistics cannot answer semantic questions: whether a zero-instance
image is a deliberate negative or a missed label, whether two perceptually
similar frames are really the same scene, or whether a segmentation mask or the
provider's bounding box better describes an object. This script assembles the
evidence a person needs to answer them, and deliberately answers none of them.

Every panel carries a short stable identifier that resolves through
``reports/manual_review_manifest.csv``.

Writes:
    reports/figures/review_*.png
    reports/manual_review_manifest.csv

Usage:
    uv run python scripts/build_review_package.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask

from construction_safety_vision.data.coco import classify_segmentation, load_coco_document
from construction_safety_vision.data.geometry import GeometryError, compare_to_segmentation
from construction_safety_vision.paths import ProjectPaths

FIGURE_DPI = 92
"""Resolution for contact sheets; modest because they are committed."""

JPEG_QUALITY = 78
"""Contact sheets are photographs, so they are stored as JPEG. As PNG the review
package came to ~9.8 MB, which is not a reasonable thing to put in a repository
for figures whose purpose is a visual once-over."""

THUMB_COLUMNS = 6
"""Panels per row on a contact sheet."""

MAX_OVERVIEW = 24
"""Panels on the representative overview sheet."""

MAX_PER_CLASS = 8
"""Panels per class on the per-class sheet."""

MAX_PAIRS = 8
"""Near-duplicate pairs shown per sheet."""

SUPPLIED_COLOUR = "#FFB000"
"""Colour of the provider-supplied COCO box."""

DERIVED_COLOUR = "#00B0F0"
"""Colour of the box derived from the segmentation."""

MASK_COLOUR = "#FF3B30"
"""Colour of the segmentation outline/overlay."""


def _write(fig: Any, destination: Path) -> str:
    """Save a sheet as JPEG and close the figure.

    Args:
        fig: Matplotlib figure.
        destination: Output file; the suffix is forced to ``.jpg``.

    Returns:
        The repository-relative path.
    """
    target = destination.with_suffix(".jpg")
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=FIGURE_DPI, pil_kwargs={"quality": JPEG_QUALITY, "optimize": True})
    plt.close(fig)
    return f"reports/figures/{target.name}"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read newline-delimited JSON.

    Args:
        path: File to read.

    Returns:
        The parsed rows, or an empty list when the file is absent.
    """
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def short_id(image_id: str) -> str:
    """Shorten a provider identifier for on-panel display.

    Args:
        image_id: Full provider identifier.

    Returns:
        The first eight characters, which resolve through the manifest.
    """
    return image_id[:8]


def contact_sheet(
    panels: list[tuple[Path, str]],
    destination: Path,
    title: str,
    *,
    columns: int = THUMB_COLUMNS,
) -> str | None:
    """Render a grid of thumbnails with captions.

    Args:
        panels: ``(image path, caption)`` pairs.
        destination: Output file.
        title: Sheet title.
        columns: Panels per row.

    Returns:
        The repository-relative path, or ``None`` when there was nothing to show.
    """
    if not panels:
        return None
    rows = (len(panels) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(2.3 * columns, 2.6 * rows))
    axes = np.atleast_1d(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for ax, (path, caption) in zip(axes, panels, strict=False):
        try:
            with Image.open(path) as image:
                image.thumbnail((420, 420))
                ax.imshow(image)
        except OSError:
            ax.text(0.5, 0.5, "unreadable", ha="center", va="center", fontsize=7)
        ax.set_title(caption, fontsize=6.5, pad=2)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _write(fig, destination)


def pair_sheet(
    pairs: list[tuple[Path, str, Path, str, str]], destination: Path, title: str
) -> str | None:
    """Render near-duplicate candidate pairs side by side.

    Args:
        pairs: ``(path a, caption a, path b, caption b, verdict caption)`` tuples.
        destination: Output file.
        title: Sheet title.

    Returns:
        The repository-relative path, or ``None`` when there was nothing to show.
    """
    if not pairs:
        return None
    fig, axes = plt.subplots(len(pairs), 2, figsize=(6.4, 3.1 * len(pairs)))
    axes = np.atleast_2d(axes)
    for row, (path_a, caption_a, path_b, caption_b, verdict) in enumerate(pairs):
        for column, (path, caption) in enumerate(((path_a, caption_a), (path_b, caption_b))):
            ax = axes[row, column]
            ax.axis("off")
            try:
                with Image.open(path) as image:
                    image.thumbnail((520, 520))
                    ax.imshow(image)
            except OSError:
                ax.text(0.5, 0.5, "unreadable", ha="center", va="center", fontsize=7)
            ax.set_title(caption, fontsize=7, pad=2)
        axes[row, 0].set_ylabel(verdict, fontsize=7)
        axes[row, 0].text(
            0.0, -0.06, verdict, transform=axes[row, 0].transAxes, fontsize=7, color="#444444"
        )
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _write(fig, destination)


def draw_geometry_panel(ax: Any, image_path: Path, annotation: dict[str, Any]) -> None:
    """Draw one annotation with its mask, derived box and supplied box.

    The figure deliberately does not say which box is correct. That is the
    question the reviewer is being asked.

    Args:
        ax: Matplotlib axis.
        image_path: Image the annotation belongs to.
        annotation: COCO annotation.
    """
    ax.axis("off")
    try:
        with Image.open(image_path) as image:
            ax.imshow(image.convert("RGB"))
    except OSError:
        ax.text(0.5, 0.5, "unreadable", ha="center", va="center", fontsize=7)
        return

    segmentation = annotation.get("segmentation")
    kind = classify_segmentation(segmentation)
    if kind == "rle":
        payload = dict(segmentation)
        if isinstance(payload["counts"], str):
            payload["counts"] = payload["counts"].encode("utf-8")
        decoded = coco_mask.decode(payload)
        overlay = np.zeros((*decoded.shape, 4), dtype=float)
        overlay[decoded > 0] = (1.0, 0.23, 0.19, 0.35)
        ax.imshow(overlay)
    elif kind == "polygon":
        for ring in segmentation:
            points = np.asarray(ring, dtype=float).reshape(-1, 2)
            ax.plot(
                np.append(points[:, 0], points[0, 0]),
                np.append(points[:, 1], points[0, 1]),
                color=MASK_COLOUR,
                linewidth=1.2,
            )

    supplied = annotation.get("bbox")
    if supplied:
        ax.add_patch(
            mpatches.Rectangle(
                (supplied[0], supplied[1]),
                supplied[2],
                supplied[3],
                fill=False,
                edgecolor=SUPPLIED_COLOUR,
                linewidth=1.6,
                linestyle="-",
            )
        )
    try:
        comparison = compare_to_segmentation(supplied, segmentation, representation=kind)
        derived = comparison.derived
        ax.add_patch(
            mpatches.Rectangle(
                (derived.x, derived.y),
                derived.width,
                derived.height,
                fill=False,
                edgecolor=DERIVED_COLOUR,
                linewidth=1.6,
                linestyle="--",
            )
        )
    except GeometryError:
        pass


def geometry_sheet(
    entries: list[tuple[Path, dict[str, Any], str]], destination: Path, title: str
) -> str | None:
    """Render a sheet of geometry-comparison panels.

    Args:
        entries: ``(image path, annotation, caption)`` tuples.
        destination: Output file.
        title: Sheet title.

    Returns:
        The repository-relative path, or ``None`` when there was nothing to show.
    """
    if not entries:
        return None
    columns = 3
    rows = (len(entries) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(3.4 * columns, 3.6 * rows))
    axes = np.atleast_1d(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for ax, (path, annotation, caption) in zip(axes, entries, strict=False):
        draw_geometry_panel(ax, path, annotation)
        ax.set_title(caption, fontsize=6.5, pad=2)
    handles = [
        mpatches.Patch(color=MASK_COLOUR, label="segmentation"),
        mpatches.Patch(color=SUPPLIED_COLOUR, label="supplied COCO bbox"),
        mpatches.Patch(color=DERIVED_COLOUR, label="bbox derived from segmentation"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    return _write(fig, destination)


def select_geometry_examples(
    export_root: Path, audit: dict[str, Any]
) -> list[tuple[Path, dict[str, Any], str, str]]:
    """Pick deterministic bbox-discrepancy examples across the discrepancy range.

    Args:
        export_root: Root of the extracted v4 export.
        audit: The bbox consistency audit payload.

    Returns:
        ``(image path, annotation, caption, reason)`` tuples.
    """
    wanted: list[tuple[str, dict[str, Any], str]] = []
    for split in audit.get("splits", []):
        for rank, item in enumerate(split.get("worst_examples", [])[:3], start=1):
            wanted.append((split["split"], item, f"largest_{split['split']}_{rank}"))

    selected: list[tuple[Path, dict[str, Any], str, str]] = []
    for split_name, item, reason in wanted:
        split_dir = export_root / split_name
        try:
            document = load_coco_document(split_dir / "_annotations.coco.json")
        except Exception:
            continue
        annotation = next(
            (a for a in document["annotations"] if a.get("id") == item["annotation_id"]), None
        )
        image = next((i for i in document["images"] if i.get("id") == item["image_id"]), None)
        if annotation is None or image is None:
            continue
        caption = (
            f"{split_name} ann={item['annotation_id']} {item['category']}\n"
            f"{item['representation']} delta={item['max_delta_px']}px"
        )
        selected.append((split_dir / image["file_name"], annotation, caption, reason))

    # Representative low-discrepancy examples, one polygon and one RLE per split.
    for split_name in ("valid", "test"):
        split_dir = export_root / split_name
        try:
            document = load_coco_document(split_dir / "_annotations.coco.json")
        except Exception:
            continue
        images = {i["id"]: i for i in document["images"]}
        seen: set[str] = set()
        for annotation in document["annotations"]:
            kind = classify_segmentation(annotation.get("segmentation"))
            if kind not in {"polygon", "rle"} or kind in seen or not annotation.get("bbox"):
                continue
            try:
                comparison = compare_to_segmentation(
                    annotation["bbox"], annotation["segmentation"], representation=kind
                )
            except GeometryError:
                continue
            image = images.get(annotation["image_id"])
            if image is None:
                continue
            seen.add(kind)
            selected.append(
                (
                    split_dir / image["file_name"],
                    annotation,
                    f"{split_name} ann={annotation['id']} {kind}\n"
                    f"delta={comparison.max_delta:.2f}px (representative)",
                    f"representative_{kind}",
                )
            )
            if len(seen) == 2:
                break
    return selected


def main(argv: list[str] | None = None) -> int:
    """Build the review package.

    Args:
        argv: Command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="Build the manual-review package.")
    parser.parse_args(argv)

    paths = ProjectPaths.from_root()
    manifest = read_jsonl(paths.reports / "source_image_manifest.jsonl")
    stats = read_jsonl(paths.data_interim / "source_image_stats.jsonl")
    if not manifest or not stats:
        print("ERROR: run the inventory and download scripts first.", file=sys.stderr)
        return 2

    bbox_path = paths.reports / "bbox_consistency_audit.json"
    bbox_audit = json.loads(bbox_path.read_text(encoding="utf-8")) if bbox_path.is_file() else {}

    image_dir = paths.data_external / "source_images"
    by_id = {m["image_id"]: m for m in manifest}
    path_of = {m["image_id"]: image_dir / f"{m['image_id']}.jpg" for m in manifest}

    sheets: list[str] = []
    review_rows: list[dict[str, Any]] = []

    def record(image_id: str, reason: str, sheet: str | None, note: str = "") -> None:
        if sheet is None:
            return
        entry = by_id.get(image_id, {})
        review_rows.append(
            {
                "short_id": short_id(image_id),
                "image_id": image_id,
                "name": entry.get("name", ""),
                "split": entry.get("split", ""),
                "reason": reason,
                "contact_sheet": sheet,
                "note": note,
            }
        )

    # A - representative overview, deterministic: evenly spaced by sorted id per split.
    overview: list[tuple[Path, str]] = []
    overview_ids: list[str] = []
    for split in ("train", "valid", "test"):
        members = sorted(m["image_id"] for m in manifest if m["split"] == split)
        take = max(1, MAX_OVERVIEW // 3)
        step = max(1, len(members) // take)
        for image_id in members[::step][:take]:
            overview.append(
                (
                    path_of[image_id],
                    f"{short_id(image_id)} · {split}\n{by_id[image_id]['annotation_count']} inst",
                )
            )
            overview_ids.append(image_id)
    sheet = contact_sheet(
        overview,
        paths.figures / "review_a_overview.png",
        "A. Source dataset overview (deterministic sample across provider splits)",
    )
    if sheet:
        sheets.append(sheet)
        for image_id in overview_ids:
            record(image_id, "overview_sample", sheet)

    # B/C - per-class examples, vest_loose included by construction.
    per_class: dict[str, list[str]] = defaultdict(list)
    for entry in sorted(manifest, key=lambda m: m["image_id"]):
        for label in entry["class_counts"]:
            if len(per_class[label]) < MAX_PER_CLASS:
                per_class[label].append(entry["image_id"])
    for label, ids in sorted(per_class.items()):
        panels = [
            (
                path_of[i],
                f"{short_id(i)} · {by_id[i]['split']}\n{label} x{by_id[i]['class_counts'][label]}",
            )
            for i in ids
        ]
        name = f"review_b_class_{label}.png"
        sheet = contact_sheet(
            panels, paths.figures / name, f"B. Source examples containing '{label}'", columns=4
        )
        if sheet:
            sheets.append(sheet)
            for image_id in ids:
                record(image_id, f"class_example:{label}", sheet)

    # D - crowded scenes.
    crowded = sorted(manifest, key=lambda m: -m["annotation_count"])[:12]
    sheet = contact_sheet(
        [
            (
                path_of[m["image_id"]],
                f"{short_id(m['image_id'])} · {m['split']}\n{m['annotation_count']} inst",
            )
            for m in crowded
        ],
        paths.figures / "review_d_crowded.png",
        "D. Highest instance-count source images",
        columns=4,
    )
    if sheet:
        sheets.append(sheet)
        for entry in crowded:
            record(
                entry["image_id"], "crowded_scene", sheet, f"{entry['annotation_count']} instances"
            )

    # E - lower-luminance subset (neutral wording, not a quality judgement).
    lit = sorted(
        (s for s in stats if s.get("decoded") and s.get("mean_luminance") is not None),
        key=lambda s: s["mean_luminance"],
    )[:12]
    sheet = contact_sheet(
        [
            (
                path_of[s["image_id"]],
                f"{short_id(s['image_id'])} · {s['split']}\nlum={s['mean_luminance']:.3f}",
            )
            for s in lit
        ],
        paths.figures / "review_e_lower_luminance.png",
        "E. Lower-luminance subset (measured proxy, not a quality verdict)",
        columns=4,
    )
    if sheet:
        sheets.append(sheet)
        for entry in lit:
            record(
                entry["image_id"],
                "lower_luminance",
                sheet,
                f"mean_luminance={entry['mean_luminance']:.3f}",
            )

    # F - every zero-instance image, not a sample.
    empty = sorted((m for m in manifest if m["annotation_count"] == 0), key=lambda m: m["image_id"])
    sheet = contact_sheet(
        [
            (path_of[m["image_id"]], f"{short_id(m['image_id'])} · {m['split']}\nzero instances")
            for m in empty
        ],
        paths.figures / "review_f_zero_instance.png",
        f"F. ALL {len(empty)} zero-instance source images - deliberate negative or missing label?",
        columns=5,
    )
    if sheet:
        sheets.append(sheet)
        for entry in empty:
            record(entry["image_id"], "zero_instance", sheet, "requires semantic judgement")

    # G/H - near-duplicate candidate pairs.
    candidates_path = paths.reports / "near_duplicate_candidates.csv"
    if candidates_path.is_file():
        with candidates_path.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        rows.sort(key=lambda r: int(r["min_distance"]))
        for tag, subset, title in (
            ("g", list(rows), "G. Strongest near-duplicate candidate pairs"),
            (
                "h",
                [r for r in rows if r["cross_split"] == "True"],
                "H. Strongest CROSS-SPLIT near-duplicate candidates (leakage risk)",
            ),
        ):
            pairs = [
                (
                    path_of.get(r["image_a_id"], Path()),
                    f"{short_id(r['image_a_id'])} · {r['provider_split_a']}",
                    path_of.get(r["image_b_id"], Path()),
                    f"{short_id(r['image_b_id'])} · {r['provider_split_b']}",
                    f"dhash={r['dhash_distance']} phash={r['phash_distance']} "
                    f"({r['reason']}) - CANDIDATE ONLY",
                )
                for r in subset[:MAX_PAIRS]
            ]
            sheet = pair_sheet(pairs, paths.figures / f"review_{tag}_near_duplicates.png", title)
            if sheet:
                sheets.append(sheet)
                for r in subset[:MAX_PAIRS]:
                    note = f"dhash={r['dhash_distance']} phash={r['phash_distance']}"
                    reason = "near_duplicate_candidate" + ("_cross_split" if tag == "h" else "")
                    record(r["image_a_id"], reason, sheet, note)
                    record(r["image_b_id"], reason, sheet, note)

    # I/J/K - geometry: polygon, RLE and the bbox discrepancy question.
    export_root = paths.data_raw / "construction-ppe-compliance-detection-v4-coco-segmentation"
    geometry_examples = select_geometry_examples(export_root, bbox_audit) if bbox_audit else []
    sheet = geometry_sheet(
        [(p, a, c) for p, a, c, _ in geometry_examples],
        paths.figures / "review_k_bbox_vs_segmentation.png",
        "I/J/K. Segmentation vs supplied bbox - polygon and RLE, largest and typical discrepancies",
    )
    bbox_example_count = 0
    if sheet:
        sheets.append(sheet)
        bbox_example_count = len(geometry_examples)
        for _, annotation, caption, reason in geometry_examples:
            review_rows.append(
                {
                    "short_id": f"ann{annotation.get('id')}",
                    "image_id": f"v4-image-{annotation.get('image_id')}",
                    "name": caption.replace("\n", " "),
                    "split": reason.split("_")[1] if "_" in reason else "",
                    "reason": "bbox_segmentation_discrepancy",
                    "contact_sheet": sheet,
                    "note": reason,
                }
            )

    manifest_path = paths.reports / "manual_review_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["short_id", "image_id", "name", "split", "reason", "contact_sheet", "note"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(review_rows)

    print(f"contact sheets : {len(sheets)}", flush=True)
    for name in sheets:
        print(f"  {name}", flush=True)
    print(f"review entries : {len(review_rows)}", flush=True)
    print(f"bbox discrepancy examples: {bbox_example_count}", flush=True)
    print(f"wrote reports/{manifest_path.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
