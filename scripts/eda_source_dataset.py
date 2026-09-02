"""Exploratory analysis of the 436 ORIGINAL source images and their annotations.

Every figure and every number here describes the source population. The
742-image version-4 export is never used: its images were stretched to 640x640
and its training half was augmented, so its resolution, aspect-ratio, luminance
and file-size distributions describe the provider's pipeline rather than the
data.

Language note: luminance and contrast are reported as measured proxies. A
low-luminance image is described as belonging to a "lower-luminance subset",
never as "badly lit" - a dark frame may be correctly exposed for a dark scene.

Writes:
    reports/figures/*.png
    reports/eda_source.json

Usage:
    uv run python scripts/eda_source_dataset.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from construction_safety_vision.data.imagestats import summarise
from construction_safety_vision.paths import ProjectPaths
from construction_safety_vision.provenance import SCHEMA_VERSION, git_commit

PHASE = 4
"""Roadmap phase this script belongs to."""

SMALL_OBJECT_AREA_FRACTION = 0.01
"""Relative box area below which an instance is counted as a small object.

1% of the image area is the conventional dividing line used when reporting
small-object prevalence; it is stated here so the figure can be reproduced or
disagreed with, not tuned to a desired outcome.
"""

FIGURE_DPI = 110
"""Resolution for saved figures; kept modest because they are committed."""

PALETTE = {
    "train": "#4C72B0",
    "valid": "#DD8452",
    "test": "#55A868",
    "bar": "#4C72B0",
    "accent": "#C44E52",
}


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


def style(ax: Any, title: str, xlabel: str, ylabel: str) -> None:
    """Apply a consistent, restrained style to an axis.

    Args:
        ax: Matplotlib axis.
        title: Axis title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
    """
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)


def save(fig: Any, path: Path) -> str:
    """Save and close a figure.

    Args:
        fig: Matplotlib figure.
        path: Destination file.

    Returns:
        The path relative to the repository, as a POSIX string.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    return f"reports/figures/{path.name}"


def figure_dimensions(stats: list[dict[str, Any]], out: Path) -> str:
    """Plot original width against height, coloured by split."""
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for split in ("train", "valid", "test"):
        rows = [r for r in stats if r["split"] == split and r["decoded"]]
        ax.scatter(
            [r["width"] for r in rows],
            [r["height"] for r in rows],
            s=14,
            alpha=0.65,
            label=f"{split} (n={len(rows)})",
            color=PALETTE[split],
            edgecolors="none",
        )
    style(ax, "Original image dimensions (436 source images)", "width (px)", "height (px)")
    ax.legend(fontsize=8, frameon=False)
    return save(fig, out / "source_dimensions.png")


def figure_histogram(
    values: list[float], out: Path, name: str, title: str, xlabel: str, bins: int = 30
) -> str:
    """Plot a single distribution."""
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.hist(values, bins=bins, color=PALETTE["bar"], alpha=0.85)
    style(ax, title, xlabel, "images")
    return save(fig, out / name)


def figure_class_counts(
    images_with: dict[str, int], instances: dict[str, int], total: int, out: Path
) -> str:
    """Plot images-containing-class beside instances-per-class."""
    labels = sorted(instances, key=lambda k: -instances[k])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    axes[0].barh(labels, [images_with.get(k, 0) for k in labels], color=PALETTE["bar"])
    style(axes[0], f"Source images containing each class (of {total})", "images", "")
    axes[1].barh(labels, [instances[k] for k in labels], color=PALETTE["accent"])
    style(axes[1], "Annotated instances per class", "instances", "")
    for ax in axes:
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    return save(fig, out / "source_class_distribution.png")


def figure_class_by_split(per_split: dict[str, dict[str, int]], out: Path) -> str:
    """Plot instances per class grouped by provider split."""
    labels = sorted({c for counts in per_split.values() for c in counts})
    splits = ["train", "valid", "test"]
    width = 0.26
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    for index, split in enumerate(splits):
        offsets = [i + (index - 1) * width for i in range(len(labels))]
        ax.bar(
            offsets,
            [per_split.get(split, {}).get(label, 0) for label in labels],
            width=width,
            label=split,
            color=PALETTE[split],
        )
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    style(ax, "Instances per class by provider split (source images)", "", "instances")
    ax.legend(fontsize=8, frameon=False)
    return save(fig, out / "source_class_by_split.png")


def main(argv: list[str] | None = None) -> int:
    """Run the source EDA.

    Args:
        argv: Command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="Exploratory analysis of the source images.")
    parser.parse_args(argv)

    paths = ProjectPaths.from_root()
    manifest = read_jsonl(paths.reports / "source_image_manifest.jsonl")
    stats = read_jsonl(paths.data_interim / "source_image_stats.jsonl")
    annotations = read_jsonl(paths.data_interim / "source_annotations.jsonl")
    if not manifest or not stats:
        print("ERROR: run the inventory and download scripts first.", file=sys.stderr)
        return 2

    decoded = [r for r in stats if r["decoded"]]
    figures: list[str] = []
    out = paths.figures

    # ---- image-level -----------------------------------------------------
    image_level = {
        "images": len(stats),
        "decoded": len(decoded),
        "decode_failures": len(stats) - len(decoded),
        "modes": dict(Counter(r["mode"] for r in decoded)),
        "width": summarise([r["width"] for r in decoded]),
        "height": summarise([r["height"] for r in decoded]),
        "pixels": summarise([float(r["pixels"]) for r in decoded]),
        "aspect_ratio": summarise([r["aspect_ratio"] for r in decoded if r["aspect_ratio"]]),
        "file_size_bytes": summarise([float(r["size_bytes"]) for r in decoded]),
        "mean_luminance": summarise([r["mean_luminance"] for r in decoded if r["mean_luminance"]]),
        "std_luminance_contrast_proxy": summarise(
            [r["std_luminance"] for r in decoded if r["std_luminance"]]
        ),
        "dynamic_range_p05_p95": summarise(
            [r["dynamic_range"] for r in decoded if r["dynamic_range"]]
        ),
    }
    figures.append(figure_dimensions(stats, out))
    figures.append(
        figure_histogram(
            [r["aspect_ratio"] for r in decoded if r["aspect_ratio"]],
            out,
            "source_aspect_ratio.png",
            "Aspect ratio of original source images",
            "width / height",
        )
    )
    figures.append(
        figure_histogram(
            [r["mean_luminance"] for r in decoded if r["mean_luminance"] is not None],
            out,
            "source_luminance.png",
            "Mean luminance proxy (0 = black, 1 = white)",
            "mean luminance",
        )
    )
    figures.append(
        figure_histogram(
            [r["size_bytes"] / 1024 for r in decoded],
            out,
            "source_file_size.png",
            "Original file size",
            "kilobytes",
        )
    )

    # ---- class / instance level -----------------------------------------
    images_with: Counter[str] = Counter()
    instances: Counter[str] = Counter()
    per_split_instances: dict[str, Counter[str]] = defaultdict(Counter)
    per_split_images: dict[str, Counter[str]] = defaultdict(Counter)
    co_occurrence: Counter[tuple[str, str]] = Counter()
    for record in manifest:
        classes = sorted(record["class_counts"])
        for label, count in record["class_counts"].items():
            images_with[label] += 1
            instances[label] += count
            per_split_instances[record["split"]][label] += count
            per_split_images[record["split"]][label] += 1
        for pair in combinations(classes, 2):
            co_occurrence[pair] += 1

    total_images = len(manifest)
    class_level = {
        "images_containing_class": dict(images_with.most_common()),
        "percent_images_containing_class": {
            k: round(100.0 * v / total_images, 2) for k, v in images_with.most_common()
        },
        "instances_per_class": dict(instances.most_common()),
        "instances_per_image": summarise([float(m["annotation_count"]) for m in manifest]),
        "zero_instance_images": sum(1 for m in manifest if m["annotation_count"] == 0),
        "class_cooccurrence_image_counts": {
            f"{a}+{b}": n for (a, b), n in co_occurrence.most_common()
        },
        "instances_per_class_by_split": {
            s: dict(sorted(c.items())) for s, c in sorted(per_split_instances.items())
        },
        "images_with_class_by_split": {
            s: dict(sorted(c.items())) for s, c in sorted(per_split_images.items())
        },
    }
    figures.append(figure_class_counts(images_with, instances, total_images, out))
    figures.append(figure_class_by_split({s: dict(c) for s, c in per_split_instances.items()}, out))
    figures.append(
        figure_histogram(
            [float(m["annotation_count"]) for m in manifest],
            out,
            "source_instances_per_image.png",
            "Annotated instances per source image",
            "instances",
            bins=max(1, max(m["annotation_count"] for m in manifest) + 1),
        )
    )

    # ---- object geometry -------------------------------------------------
    size_of = {r["image_id"]: (r["width"], r["height"]) for r in decoded}
    relative_area: list[float] = []
    per_class_small: Counter[str] = Counter()
    per_class_total: Counter[str] = Counter()
    relative_width: list[float] = []
    relative_height: list[float] = []
    geometry_types = Counter(a["geometry_type"] for a in annotations)
    for annotation in annotations:
        width, height = size_of.get(annotation["image_id"], (0, 0))
        if not width or not height:
            continue
        rel_w = annotation["width"] / width
        rel_h = annotation["height"] / height
        relative_width.append(rel_w)
        relative_height.append(rel_h)
        area = rel_w * rel_h
        relative_area.append(area)
        per_class_total[annotation["label"]] += 1
        if area < SMALL_OBJECT_AREA_FRACTION:
            per_class_small[annotation["label"]] += 1

    geometry_level = {
        "annotations": len(annotations),
        "geometry_types": dict(sorted(geometry_types.items())),
        "mask_geometry_vertices_available": False,
        "note": (
            "The provider exposes polygon vertices only for 'polygon' instances. "
            "'mask' instances carry a bounding box but no vertex list, so source "
            "mask-area statistics are NOT available and are reported as PARTIAL."
        ),
        "relative_box_width": summarise(relative_width),
        "relative_box_height": summarise(relative_height),
        "relative_box_area": summarise(relative_area),
        "small_object_threshold_area_fraction": SMALL_OBJECT_AREA_FRACTION,
        "small_objects_by_class": {
            label: {
                "small": per_class_small.get(label, 0),
                "total": total,
                "percent": round(100.0 * per_class_small.get(label, 0) / total, 2)
                if total
                else 0.0,
            }
            for label, total in sorted(per_class_total.items())
        },
    }
    if relative_area:
        figures.append(
            figure_histogram(
                [a for a in relative_area if a <= 1.0],
                out,
                "source_object_relative_area.png",
                "Annotated box area as a fraction of image area",
                "box area / image area",
            )
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "report": "source_eda",
        "phase": PHASE,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(paths.root),
        "population": {
            "source_images": total_images,
            "basis": "436 independent source images; the 742-image v4 export is not used here",
        },
        "image_level": image_level,
        "class_level": class_level,
        "geometry_level": geometry_level,
        "figures": figures,
    }
    (paths.reports / "eda_source.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )

    print(f"images analysed: {len(decoded)} of {len(stats)}", flush=True)
    print(f"figures written : {len(figures)}", flush=True)
    for name in figures:
        print(f"  {name}", flush=True)
    print("wrote reports/eda_source.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
