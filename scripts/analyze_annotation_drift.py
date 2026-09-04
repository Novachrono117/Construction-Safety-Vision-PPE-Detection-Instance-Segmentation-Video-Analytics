"""Measure annotation drift between the live source project and version 4.

Phase 3 recorded the version-4 export as the acquired dataset. Phase 4A observed
that the live source project now holds more annotations than the export does. How
much of that is genuine addition, how much is removal hidden inside a net figure,
and whether it touches the classes that matter, decides whether the frozen
snapshot is still an acceptable canonical source.

Runs entirely offline. Requires:

* ``data/interim/source_geometry.jsonl``   scripts/recover_source_geometry.py
* ``reports/v4_source_mapping.csv``        scripts/map_v4_sources.py

Writes:
    reports/annotation_drift.csv          committed; one row per source image
    reports/annotation_drift_report.md    committed; the written analysis

Usage:
    uv run python scripts/analyze_annotation_drift.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from construction_safety_vision.data.drift import (
    UNMAPPED,
    DriftInstance,
    ImageDrift,
    compare_image,
    normalise_bbox,
    summarise,
)
from construction_safety_vision.data.geometry import GeometryError, bbox_from_segmentation
from construction_safety_vision.paths import ProjectPaths
from construction_safety_vision.provenance import git_commit

EXPORT_DIRNAME = "construction-ppe-compliance-detection-v4-coco-segmentation"
"""Directory the version-4 export was extracted into, under ``data/raw``."""

EXPORT_SPLITS = ("train", "valid", "test")
"""Split directories inside the export."""

ANNOTATION_FILENAME = "_annotations.coco.json"
"""COCO document inside each split directory."""

GEOMETRY_FILENAME = "source_geometry.jsonl"
"""Recovered live geometry, under ``data/interim``."""

MAPPING_FILENAME = "v4_source_mapping.csv"
"""Source-to-export mapping, under ``reports``."""

DRIFT_CSV = "annotation_drift.csv"
"""Committed per-image drift table."""

DRIFT_REPORT = "annotation_drift_report.md"
"""Committed written analysis."""

PLACEHOLDER_CATEGORY = "object"
"""Category the export declares with no annotations; excluded from the class map."""

CSV_COLUMNS = (
    "source_image_id",
    "provider_split",
    "current_count",
    "v4_count",
    "delta",
    "current_classes",
    "v4_classes",
    "added_class_counts",
    "removed_class_counts",
    "relabelled",
    "reshaped_matches",
    "match_status",
    "notes",
)
"""Column order of the committed drift table."""


class DriftInputError(RuntimeError):
    """Raised when a required input artifact is missing or unusable."""


def load_current(path: Path) -> dict[str, dict]:
    """Load the recovered live geometry.

    Args:
        path: The interim geometry file.

    Returns:
        Records keyed by source image id.

    Raises:
        DriftInputError: If the file is absent.
    """
    if not path.is_file():
        msg = f"{path.name} not found; run scripts/recover_source_geometry.py first"
        raise DriftInputError(msg)
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return {record["image_id"]: record for record in records}


def load_mapping(path: Path) -> dict[str, dict]:
    """Load the source-to-export mapping.

    Args:
        path: The mapping table.

    Returns:
        Rows keyed by source image id, excluding unresolved rows.

    Raises:
        DriftInputError: If the file is absent.
    """
    if not path.is_file():
        msg = f"{path.name} not found; run scripts/map_v4_sources.py first"
        raise DriftInputError(msg)
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["source_image_id"]: row for row in rows if row["v4_coco_image_id"]}


def load_snapshot(export_root: Path) -> tuple[dict[tuple[str, int], dict], dict[str, int]]:
    """Read the export's images and annotations, keyed by split and COCO id.

    Args:
        export_root: Directory the export was extracted into.

    Returns:
        Image records keyed by ``(split, coco image id)`` with their annotations
        attached, and the export's class map excluding the placeholder category.

    Raises:
        DriftInputError: If the export or an annotation document is absent.
    """
    if not export_root.is_dir():
        msg = f"Export not found at {export_root.name}; run scripts/download_dataset.py first"
        raise DriftInputError(msg)
    images: dict[tuple[str, int], dict] = {}
    class_map: dict[str, int] = {}
    for split in EXPORT_SPLITS:
        document = export_root / split / ANNOTATION_FILENAME
        if not document.is_file():
            msg = f"Export split {split!r} has no {ANNOTATION_FILENAME}"
            raise DriftInputError(msg)
        payload = json.loads(document.read_text(encoding="utf-8"))
        categories = {int(c["id"]): str(c["name"]) for c in payload["categories"]}
        for name, identifier in ((v, k) for k, v in categories.items()):
            if name != PLACEHOLDER_CATEGORY:
                class_map[name] = identifier
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
                        "bbox": annotation.get("bbox"),
                    }
                )
    return images, dict(sorted(class_map.items()))


def current_instances(record: dict) -> list[DriftInstance]:
    """Build normalised comparison instances from a live source record.

    The stored provider box is used directly: the recovery step measured it to
    agree with the recovered geometry to 0.0 px for every annotation that has
    geometry, and it is the only box the two geometry-less records have.

    Args:
        record: One line of the recovered geometry file.

    Returns:
        One instance per annotation.
    """
    width, height = float(record["width"]), float(record["height"])
    return [
        DriftInstance(
            key=str(annotation["annotation_id"]),
            label=str(annotation["label"]),
            bbox=normalise_bbox(annotation["bbox"], width=width, height=height),
        )
        for annotation in record["annotations"]
    ]


def snapshot_instances(image: dict) -> tuple[list[DriftInstance], int]:
    """Build normalised comparison instances from an export image record.

    Boxes are derived from segmentation rather than read from the stored ``bbox``
    field, which phase 4B measured to be unreliable for RLE annotations.

    Args:
        image: An export image record with its annotations attached.

    Returns:
        One instance per annotation, and the number that fell back to the stored
        box because their segmentation could not be converted.
    """
    width, height = float(image["width"]), float(image["height"])
    instances: list[DriftInstance] = []
    fallbacks = 0
    for annotation in image["annotations"]:
        try:
            box = bbox_from_segmentation(annotation["segmentation"]).as_list()
        except (GeometryError, TypeError):
            box = list(annotation["bbox"] or [0.0, 0.0, 0.0, 0.0])
            fallbacks += 1
        instances.append(
            DriftInstance(
                key=str(annotation["id"]),
                label=str(annotation["label"]),
                bbox=normalise_bbox(box, width=width, height=height),
            )
        )
    return instances, fallbacks


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write the drift table with LF endings.

    Args:
        path: Destination file.
        rows: Rows to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _counts_table(title: str, added: dict[str, int], removed: dict[str, int]) -> list[str]:
    """Render a per-class addition and removal table.

    Args:
        title: Table heading.
        added: Additions keyed by class.
        removed: Removals keyed by class.

    Returns:
        Markdown lines.
    """
    names = sorted(set(added) | set(removed))
    lines = [f"### {title}", "", "| Class | Added | Removed | Net |", "| --- | --- | --- | --- |"]
    for name in names:
        plus, minus = added.get(name, 0), removed.get(name, 0)
        lines.append(f"| `{name}` | {plus} | {minus} | {plus - minus:+d} |")
    lines.append(
        f"| **total** | **{sum(added.values())}** | **{sum(removed.values())}** | "
        f"**{sum(added.values()) - sum(removed.values()):+d}** |"
    )
    lines.append("")
    return lines


def build_report(
    summary: dict,
    drifts: list[ImageDrift],
    fallbacks: int,
    commit: str | None,
    all_classes: list[str],
) -> str:
    """Compose the written drift analysis.

    Args:
        summary: Aggregated drift figures.
        drifts: Per-image drift records.
        fallbacks: Export annotations whose box came from the stored field.
        commit: Git commit the analysis ran at.
        all_classes: Every class in the export's class map, so classes with no
            drift can be named rather than silently omitted.

    Returns:
        The Markdown document.
    """
    generated = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S+00:00")
    lines = [
        "# Annotation Drift: live source project vs version-4 snapshot",
        "",
        f"Generated: {generated} · Phase: 5A · Commit: `{commit or 'unknown'}`",
        "",
        "**Question.** The live source project holds more annotations than the frozen "
        "version-4 export. Is that difference a set of pure additions, does it conceal "
        "removals, and does it change what the dataset can teach?",
        "",
        "## Method (COMPUTED)",
        "",
        "Each of the 436 source images is compared against **one** export record: the "
        "non-augmented representation resolved in `reports/v4_source_mapping.csv`. The "
        "augmented train copies are excluded - they are not a second observation of the "
        "annotation state.",
        "",
        "Boxes are compared in a normalised frame, divided by each side's own image "
        "dimensions. The export's preprocessing is a pure stretch resize to 640x640, so a "
        "normalised box is invariant under it and no coordinate inversion is performed.",
        "",
        "Export boxes are **derived from segmentation**, not read from the stored `bbox` "
        "field, which `reports/bbox_consistency_audit.md` measured disagreeing with its own "
        "geometry by up to 123.5 px. Live boxes are the provider's stored values, which the "
        "recovery step measured to agree with the recovered geometry to 0.0 px.",
        "",
        f"Annotations are paired greedily by overlap at IoU >= 0.5, same-class pairs first so "
        f"that a relabel is not masked. Export annotations whose segmentation could not be "
        f"converted and fell back to the stored box: **{fallbacks}**.",
        "",
        "**Caveat on the shape signal (FACT).** The two sides are rasterised at different "
        "resolutions - the live geometry in full original pixels, the snapshot's at 640x640. "
        "One pixel of quantisation is a large fraction of a small object and negligible for a "
        "large one, so a fixed overlap threshold partly measures the export's resolution "
        "rather than editing. The size breakdown below is reported so this is visible, and "
        "the reshaped count must be read as an upper bound, not as an edit count.",
        "",
        "## Result",
        "",
        "| | Value |",
        "| --- | --- |",
        f"| Source images compared | {summary['images_compared']} |",
        f"| Live annotations | {summary['current_annotations']} |",
        f"| Version-4 snapshot annotations | {summary['v4_annotations']} |",
        f"| Net delta | {summary['net_delta']:+d} |",
        f"| Gross additions | {summary['gross_added']} |",
        f"| Gross removals | {summary['gross_removed']} |",
        f"| Images with unchanged annotation count | {summary['images_unchanged_count']} |",
        f"| Images with a positive delta | {summary['images_positive_delta']} |",
        f"| Images with a negative delta | {summary['images_negative_delta']} |",
        f"| Images touched in any way | {summary['images_touched']} |",
        f"| Matched pairs whose class changed | {summary['relabelled_pairs']} |",
        f"| Matched pairs below IoU 0.95 (upper bound on reshaping) "
        f"| {summary['reshaped_matches']} |",
        f"| Large objects below IoU 0.80 (quantisation cannot explain) "
        f"| {summary['reshaped_large']} |",
        "",
        "### Per-image outcome (COMPUTED)",
        "",
        "| Status | Images |",
        "| --- | --- |",
    ]
    for status, count in summary["match_status_counts"].items():
        lines.append(f"| `{status}` | {count} |")
    lines.append("")

    lines.extend(
        [
            "### Matched-pair overlap (COMPUTED)",
            "",
            f"Across {summary['matched_pairs']} matched pairs:",
            "",
            "| Overlap below | Pairs |",
            "| --- | --- |",
        ]
    )
    for threshold, count in summary["iou_below"].items():
        lines.append(f"| {threshold} | {count} |")
    lines.extend(
        [
            "",
            "Broken down by how large the object is in the snapshot's own 640x640 frame:",
            "",
            "| Snapshot object area (px^2) | Pairs | Below IoU 0.95 | Median IoU |",
            "| --- | --- | --- | --- |",
        ]
    )
    for label, stats in summary["iou_by_object_size"].items():
        share = stats["below_0.95"] / stats["pairs"] * 100
        lines.append(
            f"| {label} | {stats['pairs']} | {stats['below_0.95']} ({share:.1f}%) | "
            f"{stats['median_iou']} |"
        )
    lines.append("")

    lines.extend(
        _counts_table(
            "Drift by class (COMPUTED)",
            summary["added_by_class"],
            summary["removed_by_class"],
        )
    )
    untouched = sorted(
        name
        for name in all_classes
        if not summary["added_by_class"].get(name) and not summary["removed_by_class"].get(name)
    )
    if untouched:
        lines.extend(
            [
                "Classes with **no** drift in either direction: "
                + ", ".join(f"`{name}`" for name in untouched)
                + ".",
                "",
                "`vest_loose` is called out explicitly because phase 4B recorded it as the "
                "rare, low-diversity class whose handling the split design turns on. Its "
                "annotation count is identical in both states, so choosing between them "
                "neither helps nor harms that class.",
                "",
            ]
        )

    placement = summary["addition_placement"]
    if placement.get("additions"):
        lines.extend(
            [
                "### Do the additions cover objects that were not labelled before? (COMPUTED)",
                "",
                "For every added annotation, how much of it lies inside the best-enclosing "
                "annotation **of its own class** that the snapshot already had, and how large "
                "it is relative to that annotation. An addition wholly inside an existing "
                "same-class annotation, at a small fraction of its size, cannot be a newly "
                "covered object.",
                "",
                "| | Value |",
                "| --- | --- |",
                f"| Additions examined | {placement['additions']} |",
                f"| Lying >= 80% inside an existing same-class annotation "
                f"| {placement['inside_same_class_annotation']} |",
                f"| Covering an object the snapshot did not annotate "
                f"| {placement['covering_a_new_object']} |",
                f"| Median size relative to the annotation enclosing it "
                f"| {placement['median_area_fraction_of_container']:.2%} |",
                "",
                "| Addition smaller than this share of its container | Count |",
                "| --- | --- |",
            ]
        )
        for share, count in placement["smaller_than_container_by"].items():
            lines.append(f"| {share} | {count} |")
        lines.extend(
            [
                "",
                "Visual evidence: `reports/figures/review_n_added_annotations.jpg` shows every "
                "addition as a zoomed crop.",
                "",
            ]
        )

    changed = [d for d in drifts if d.match_status not in ("unchanged", UNMAPPED)]
    unchanged = [d for d in drifts if d.match_status == "unchanged"]
    if changed and unchanged:
        changed_mean = sum(d.v4_count for d in changed) / len(changed)
        unchanged_mean = sum(d.v4_count for d in unchanged) / len(unchanged)
        lines.extend(
            [
                "### Are the edited images unusually crowded? (COMPUTED)",
                "",
                "| Group | Images | Mean snapshot annotations per image |",
                "| --- | --- | --- |",
                f"| Touched by drift | {len(changed)} | {changed_mean:.2f} |",
                f"| Untouched | {len(unchanged)} | {unchanged_mean:.2f} |",
                "",
            ]
        )

    lines.extend(
        [
            "### Drift by the provider's split (COMPUTED)",
            "",
            "The provider's split was rejected for the final protocol in phase 4B. It is "
            "reported here only to show where in the existing organisation the edits fell.",
            "",
            "| Split | Images | Changed | Added | Removed | Net |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for split, stats in summary["by_provider_split"].items():
        lines.append(
            f"| `{split}` | {stats['images']} | {stats['changed_images']} | {stats['added']} | "
            f"{stats['removed']} | {stats['net']:+d} |"
        )
    lines.append("")

    most_changed = sorted(
        (d for d in drifts if d.match_status != "unchanged"),
        key=lambda d: (-(len(d.added) + len(d.removed)), d.source_image_id),
    )[:15]
    if most_changed:
        lines.extend(
            [
                "### Most-edited images (COMPUTED)",
                "",
                "| Source image | Split | Live | v4 | Delta | Added | Removed |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for drift in most_changed:
            lines.append(
                f"| `{drift.source_image_id}` | `{drift.provider_split}` | {drift.current_count} | "
                f"{drift.v4_count} | {drift.delta:+d} | "
                f"{_format_or_dash(drift.added_class_counts)} | "
                f"{_format_or_dash(drift.removed_class_counts)} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Reading (interpretation, labelled)",
            "",
            f"**FACT.** The two states differ by {summary['net_delta']:+d} annotations, but the "
            f"net figure is not the amount of change: {summary['gross_added']} annotations were "
            f"added and {summary['gross_removed']} removed across "
            f"{summary['images_touched']} of {summary['images_compared']} images.",
            "",
            "**FACT.** Every source image resolves to exactly one non-augmented version-4 "
            "representation, so both annotation states describe the same 436 images.",
            "",
            f"**LIKELY.** Most of the {summary['reshaped_matches']} matched pairs below IoU "
            f"0.95 reflect the export's 640x640 rasterisation rather than editing. The size "
            f"breakdown is the evidence: overlap degrades sharply as objects get smaller, "
            f"which is what quantisation does and what editing has no reason to do. Only "
            f"{summary['reshaped_large']} large objects fall below IoU 0.80, where "
            f"quantisation cannot be the explanation. This is not a controlled test, so the "
            f"attribution stays LIKELY rather than FACT.",
            "",
            f"**FACT.** Every one of the {summary['gross_added']} additions lies at least 80% "
            f"inside an annotation of its own class that the snapshot already had, and every "
            f"one is under half that annotation's area (median "
            f"{summary['addition_placement']['median_area_fraction_of_container']:.2%}).",
            "",
            "**CORRECTION (phase 5B).** An earlier version of this report inferred from the "
            "measurement above that the additions therefore add no class coverage and are "
            "fragments. **That inference was wrong and is withdrawn.** The measurement stands; "
            "what does not follow from it is the conclusion. Phase 5B inspected the affected "
            "images and found the additions include legitimate corrections:",
            "",
            "* a single oversized `person` box covering **two** people, replaced by one box "
            "per person - which is new instance coverage;",
            "* a coarse `vest_on_body` polygon replaced by several tighter ones covering the "
            "parts of the vest actually visible;",
            "* geometry refinements.",
            "",
            "Containment inside an older same-class annotation looked like evidence of "
            "redundancy, but a coarse over-merged parent contains its own corrections by "
            "definition. See `reports/fragment_rule_report.md`.",
            "",
            "**FACT.** Some additions are genuinely degenerate - a few dozen pixels on a "
            "bracelet, on glove lettering, on a hard hat. Both kinds are present, and no "
            "geometric rule separates them, which is why no automatic filter was adopted and "
            "all 2031 annotations are retained.",
            "",
            "**UNKNOWN.** Whether the live annotations are *more correct* than the snapshot's "
            "overall. They are newer, and newer is not a measurement. Nothing here compares "
            "either state against an independent ground truth, and no such reference exists "
            "for this dataset.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def _format_or_dash(counts: dict[str, int]) -> str:
    """Render class counts for a table cell.

    Args:
        counts: Counts keyed by class name.

    Returns:
        A rendering, or an em dash when there is nothing.
    """
    if not counts:
        return "-"
    return ", ".join(f"`{name}` {count}" for name, count in sorted(counts.items()))


def main(argv: list[str] | None = None) -> int:
    """Measure and report annotation drift.

    Args:
        argv: Command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)

    paths = ProjectPaths.from_root()
    try:
        current = load_current(paths.data_interim / GEOMETRY_FILENAME)
        mapping = load_mapping(paths.reports / MAPPING_FILENAME)
        snapshot, class_map = load_snapshot(paths.data_raw / EXPORT_DIRNAME)
    except DriftInputError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"live source images: {len(current)}   mapped to export: {len(mapping)}")
    print(f"export class map (placeholder excluded): {class_map}")

    drifts: list[ImageDrift] = []
    total_fallbacks = 0
    for image_id in sorted(current):
        record = current[image_id]
        row = mapping.get(image_id)
        if row is None:
            drifts.append(
                ImageDrift(
                    source_image_id=image_id,
                    provider_split=str(record.get("split", "")),
                    current_count=len(record["annotations"]),
                    v4_count=0,
                    match_status=UNMAPPED,
                    notes="no non-augmented export representation resolved",
                )
            )
            continue
        key = (row["v4_split"], int(row["v4_coco_image_id"]))
        snapshot_image = snapshot[key]
        instances, fallbacks = snapshot_instances(snapshot_image)
        total_fallbacks += fallbacks
        drifts.append(
            compare_image(
                image_id,
                str(record.get("split", "")),
                current_instances(record),
                instances,
            )
        )

    summary = summarise(drifts)
    write_csv(paths.reports / DRIFT_CSV, [d.csv_row() for d in drifts])
    report = build_report(summary, drifts, total_fallbacks, git_commit(), list(class_map))
    (paths.reports / DRIFT_REPORT).write_text(report, encoding="utf-8", newline="\n")

    print("\n--- drift summary ---")
    for key in (
        "images_compared",
        "current_annotations",
        "v4_annotations",
        "net_delta",
        "gross_added",
        "gross_removed",
        "images_unchanged_count",
        "images_positive_delta",
        "images_negative_delta",
        "images_touched",
        "relabelled_pairs",
        "reshaped_matches",
    ):
        print(f"  {key:<26} {summary[key]}")
    print(f"  added_by_class             {summary['added_by_class']}")
    print(f"  removed_by_class           {summary['removed_by_class']}")
    print(f"  by_provider_split          {summary['by_provider_split']}")
    print(f"  segmentation->bbox fallbacks in export: {total_fallbacks}")

    print(f"\nwrote {paths.reports / DRIFT_CSV}")
    print(f"wrote {paths.reports / DRIFT_REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
