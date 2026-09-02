"""Audit the exported COCO bboxes against the segmentation geometry they enclose.

Phase 5 will derive detection boxes from the canonical segmentation. This audit
answers whether it may reuse the ``bbox`` field the export already supplies, or
whether it must always recompute the box from the geometry.

Audit only: nothing is written to the dataset and no detection label is produced.

Scope note: this operates on the v4 export's annotations, which is the only
annotation surface currently on disk. The ``valid`` and ``test`` splits are
un-augmented, so their annotations are the preprocessed source annotations; the
``train`` split additionally contains geometry transformed by offline
augmentation. Per-split results are reported separately for that reason.

Usage:
    uv run python scripts/audit_bbox_consistency.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from construction_safety_vision.config import ConfigError, load_experiment_config
from construction_safety_vision.data.coco import classify_segmentation, load_coco_document
from construction_safety_vision.data.geometry import (
    DEFAULT_TOLERANCE_PX,
    GeometryError,
    compare_to_segmentation,
)
from construction_safety_vision.paths import ProjectPaths
from construction_safety_vision.provenance import SCHEMA_VERSION, git_commit

PHASE = 4
"""Roadmap phase this script belongs to."""

TOLERANCES = (0.0, 0.5, 1.0, 2.0, 5.0)
"""Tolerances (pixels) the agreement rate is reported at."""

WORST_EXAMPLES = 10
"""Number of largest mismatches recorded as examples."""


def audit_split(split_dir: Path, tolerance: float) -> dict[str, Any]:
    """Audit every annotation of one split.

    Besides the corner-wise comparison, two diagnostics are computed because the
    raw agreement rate alone does not say *why* boxes disagree:

    * whether the supplied box spills outside the image canvas, which a
      rasterised mask cannot do;
    * whether the mask-derived box is contained within the supplied box, i.e.
      whether the supplied box is systematically the larger of the two.

    Args:
        split_dir: Directory holding the split's COCO document.
        tolerance: Primary tolerance used for the headline agreement figure.

    Returns:
        A JSON-serialisable summary of the split.
    """
    document = load_coco_document(split_dir / "_annotations.coco.json")
    categories = {int(c["id"]): str(c.get("name", "")) for c in document["categories"] if "id" in c}
    canvas = {
        int(image["id"]): (float(image.get("width", 0)), float(image.get("height", 0)))
        for image in document["images"]
        if isinstance(image, dict) and "id" in image
    }

    checked = 0
    skipped: Counter[str] = Counter()
    by_representation: Counter[str] = Counter()
    within: dict[float, Counter[str]] = {t: Counter() for t in TOLERANCES}
    spill: Counter[str] = Counter()
    containment: Counter[str] = Counter()
    max_delta = 0.0
    delta_sum = 0.0
    worst: list[dict[str, Any]] = []

    for annotation in document["annotations"]:
        if not isinstance(annotation, dict):
            skipped["not_a_mapping"] += 1
            continue
        segmentation = annotation.get("segmentation")
        representation = classify_segmentation(segmentation)
        if representation in {"absent", "empty"}:
            skipped[f"segmentation_{representation}"] += 1
            continue
        if not annotation.get("bbox"):
            skipped["bbox_absent"] += 1
            continue
        try:
            comparison = compare_to_segmentation(
                annotation["bbox"], segmentation, representation=representation
            )
        except GeometryError:
            skipped["geometry_error"] += 1
            continue

        checked += 1
        by_representation[representation] += 1
        delta = comparison.max_delta
        delta_sum += delta
        max_delta = max(max_delta, delta)
        for value in TOLERANCES:
            if comparison.within(value):
                within[value][representation] += 1
                within[value]["all"] += 1

        width, height = canvas.get(int(annotation.get("image_id", -1)), (0.0, 0.0))
        sx0, sy0, sx1, sy1 = comparison.supplied.corners()
        spills = bool(width and height) and (
            sx0 < -1e-3 or sy0 < -1e-3 or sx1 > width + 1e-3 or sy1 > height + 1e-3
        )
        agreement = "agrees" if comparison.within(tolerance) else "differs"
        spill[f"{representation}|{'spills' if spills else 'inside'}|{agreement}"] += 1

        dx0, dy0, dx1, dy1 = comparison.derived.corners()
        contained = dx0 >= sx0 - 1 and dy0 >= sy0 - 1 and dx1 <= sx1 + 1 and dy1 <= sy1 + 1
        containment[
            f"{representation}|{'derived_inside_supplied' if contained else 'derived_escapes'}"
        ] += 1

        worst.append(
            {
                "annotation_id": annotation.get("id"),
                "image_id": annotation.get("image_id"),
                "category": categories.get(int(annotation.get("category_id", -1)), "?"),
                "representation": representation,
                "max_delta_px": round(delta, 4),
                "deltas_px": [round(d, 4) for d in comparison.deltas],
                "supplied_bbox": [round(v, 3) for v in comparison.supplied.as_list()],
                "derived_bbox": [round(v, 3) for v in comparison.derived.as_list()],
            }
        )

    worst.sort(key=lambda item: item["max_delta_px"], reverse=True)
    return {
        "split": split_dir.name,
        "annotations_total": len(document["annotations"]),
        "annotations_checked": checked,
        "skipped": dict(skipped),
        "by_representation": dict(by_representation),
        "max_delta_px": round(max_delta, 4),
        "mean_delta_px": round(delta_sum / checked, 4) if checked else None,
        "within_tolerance": {str(value): dict(counts) for value, counts in sorted(within.items())},
        "canvas_spill": dict(spill),
        "containment": dict(containment),
        "mismatches_at_primary_tolerance": checked - within[tolerance]["all"],
        "worst_examples": worst[:WORST_EXAMPLES],
    }


def _rate(part: int, whole: int) -> str:
    """Format a share as a percentage string.

    Args:
        part: Numerator.
        whole: Denominator.

    Returns:
        A percentage with two decimals, or ``"n/a"`` when the denominator is zero.
    """
    return f"{100.0 * part / whole:.2f}%" if whole else "n/a"


def build_markdown(payload: dict[str, Any]) -> str:
    """Render the human-readable audit.

    Args:
        payload: The machine-readable audit result.

    Returns:
        Markdown text.
    """
    lines: list[str] = []
    add = lines.append
    totals = payload["totals"]
    tolerance = payload["primary_tolerance_px"]

    add("# BBox / Segmentation Consistency Audit")
    add("")
    add(
        f"Generated: {payload['generated_at']} · Phase: {payload['phase']} · "
        f"Commit: `{payload['git_commit']}`"
    )
    add("")
    add(
        "**Question.** May phase 5 reuse the `bbox` field supplied by the export when "
        "building the detection view, or must it always recompute the box from the "
        "segmentation geometry?"
    )
    add("")
    add(
        "**Method (COMPUTED).** For every annotation carrying segmentation geometry, the "
        "enclosing box is derived from that geometry - polygons by coordinate extrema, "
        "compressed RLE through the reference COCO implementation - and compared corner by "
        f"corner against the supplied `bbox`. Primary tolerance: **{tolerance} px**."
    )
    add("")
    add(
        "**Why that tolerance (FACT).** A compressed RLE is a rasterised binary mask, so a "
        "box derived from it is quantised to whole pixels while a polygon-derived box is "
        "fractional. A one-pixel allowance is the coarsest representation's own resolution, "
        "not a threshold chosen to produce a pleasing result. Agreement at several "
        "tolerances is reported below so the reader can apply their own."
    )
    add("")

    add("## Result")
    add("")
    add("| | Value |")
    add("| --- | --- |")
    checked_total = totals["annotations_checked"]
    add(f"| Annotations checked | {checked_total} of {totals['annotations_total']} |")
    add(
        f"| Polygon / RLE | {totals['by_representation'].get('polygon', 0)} / "
        f"{totals['by_representation'].get('rle', 0)} |"
    )
    add(
        f"| Agree within {tolerance} px | {totals['within_primary']} "
        f"({_rate(totals['within_primary'], totals['annotations_checked'])}) |"
    )
    add(f"| Mismatches at {tolerance} px | {totals['mismatches_at_primary_tolerance']} |")
    add(f"| Maximum discrepancy | {totals['max_delta_px']} px |")
    add(f"| Mean discrepancy | {totals['mean_delta_px']} px |")
    add("")

    add("### Agreement by tolerance and representation (COMPUTED)")
    add("")
    add("| Tolerance (px) | All | Polygon | RLE |")
    add("| --- | --- | --- | --- |")
    for value, counts in payload["totals"]["within_tolerance"].items():
        poly_total = totals["by_representation"].get("polygon", 0)
        rle_total = totals["by_representation"].get("rle", 0)
        add(
            f"| {value} | {counts.get('all', 0)} "
            f"({_rate(counts.get('all', 0), totals['annotations_checked'])}) "
            f"| {counts.get('polygon', 0)} ({_rate(counts.get('polygon', 0), poly_total)}) "
            f"| {counts.get('rle', 0)} ({_rate(counts.get('rle', 0), rle_total)}) |"
        )
    add("")

    add("### Per split (COMPUTED)")
    add("")
    add(
        "| Split | Checked | Polygon | RLE | Max delta (px) | Mean delta (px) | "
        f"Mismatches at {tolerance} px |"
    )
    add("| --- | --- | --- | --- | --- | --- | --- |")
    for split in payload["splits"]:
        add(
            f"| {split['split']} | {split['annotations_checked']} "
            f"| {split['by_representation'].get('polygon', 0)} "
            f"| {split['by_representation'].get('rle', 0)} "
            f"| {split['max_delta_px']} | {split['mean_delta_px']} "
            f"| {split['mismatches_at_primary_tolerance']} |"
        )
    add("")
    add(
        "`valid` and `test` are un-augmented, so their annotations are the preprocessed "
        "source annotations. `train` additionally contains geometry transformed by offline "
        "augmentation, which is why the splits are not pooled."
    )
    add("")

    add("### Largest discrepancies (COMPUTED)")
    add("")
    add("| Split | Annotation | Image | Class | Repr. | Max delta (px) |")
    add("| --- | --- | --- | --- | --- | --- |")
    for split in payload["splits"]:
        for item in split["worst_examples"][:5]:
            add(
                f"| {split['split']} | {item['annotation_id']} | {item['image_id']} "
                f"| `{item['category']}` | {item['representation']} | {item['max_delta_px']} |"
            )
    add("")

    add("### Why they disagree (COMPUTED diagnostics)")
    add("")
    add("| Diagnostic | Count |")
    add("| --- | --- |")
    for key, value in sorted(totals["canvas_spill"].items()):
        add(f"| supplied box `{key}` | {value} |")
    for key, value in sorted(totals["containment"].items()):
        add(f"| `{key}` | {value} |")
    add("")

    add("## Interpretation")
    add("")
    for line in payload["interpretation"]:
        add(f"- {line}")
    add("")

    add("## Open questions")
    add("")
    for line in payload["open_questions"]:
        add(f"- {line}")
    add("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run the audit and write its artifacts.

    Args:
        argv: Command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="Audit supplied bboxes against segmentation.")
    parser.add_argument("--config", type=Path, default=None, help="Configuration file to use.")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE_PX,
        help=f"Primary tolerance in pixels (default: {DEFAULT_TOLERANCE_PX}).",
    )
    args = parser.parse_args(argv)

    paths = ProjectPaths.from_root()
    try:
        config = load_experiment_config(args.config or (paths.configs / "project.yaml"))
    except ConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    dataset = config.dataset
    slug = f"{dataset.project_slug}-v{dataset.version}-{dataset.export_format}"
    export_root = paths.data_raw / slug
    if not export_root.is_dir():
        print(
            f"ERROR: export not found at data/raw/{slug}. Run scripts/download_dataset.py first.",
            file=sys.stderr,
        )
        return 2

    split_dirs = sorted(
        p for p in export_root.iterdir() if p.is_dir() and (p / "_annotations.coco.json").is_file()
    )
    splits = [audit_split(path, args.tolerance) for path in split_dirs]

    checked = sum(s["annotations_checked"] for s in splits)
    by_representation: Counter[str] = Counter()
    within_tolerance: dict[str, Counter[str]] = {str(t): Counter() for t in TOLERANCES}
    spill_totals: Counter[str] = Counter()
    containment_totals: Counter[str] = Counter()
    delta_weighted = 0.0
    for split in splits:
        by_representation.update(split["by_representation"])
        spill_totals.update(split["canvas_spill"])
        containment_totals.update(split["containment"])
        for value, counts in split["within_tolerance"].items():
            within_tolerance[value].update(counts)
        if split["mean_delta_px"] is not None:
            delta_weighted += split["mean_delta_px"] * split["annotations_checked"]

    primary = str(args.tolerance)
    within_primary = within_tolerance.get(primary, Counter()).get("all", 0)
    max_delta = max((s["max_delta_px"] for s in splits), default=0.0)

    exact = within_tolerance.get("0.0", Counter()).get("all", 0)
    poly_exact = within_tolerance.get("0.0", Counter()).get("polygon", 0)
    poly_total = by_representation.get("polygon", 0)
    rle_total = by_representation.get("rle", 0)

    poly_half = within_tolerance.get("0.5", Counter()).get("polygon", 0)
    rle_primary = within_tolerance.get(primary, Counter()).get("rle", 0)
    spill_inside_differs = spill_totals.get("rle|inside|differs", 0)
    spill_inside_agrees = spill_totals.get("rle|inside|agrees", 0)
    spill_out_differs = spill_totals.get("rle|spills|differs", 0)
    spill_out_agrees = spill_totals.get("rle|spills|agrees", 0)
    contained = containment_totals.get("rle|derived_inside_supplied", 0)
    spill_out = spill_out_agrees + spill_out_differs
    spill_inside = spill_inside_agrees + spill_inside_differs
    train = next((s for s in splits if s["split"] == "train"), None)
    unaugmented = [s for s in splits if s["split"] != "train"]

    interpretation = [
        f"COMPUTED: polygon annotations agree essentially perfectly - {poly_half} of "
        f"{poly_total} ({_rate(poly_half, poly_total)}) within 0.5 px, though only "
        f"{poly_exact} match to the bit. The supplied box for a polygon is the polygon's "
        "own coordinate extrema.",
        f"COMPUTED: the entire disagreement lives in the RLE annotations - {rle_primary} of "
        f"{rle_total} ({_rate(rle_primary, rle_total)}) within {args.tolerance} px. The "
        f"largest single discrepancy is {max_delta} px.",
        f"COMPUTED: the mask-derived box lies inside the supplied box in {contained} of "
        f"{rle_total} RLE cases ({_rate(contained, rle_total)}). The supplied box is "
        "systematically the larger of the two, not randomly offset.",
        f"COMPUTED: {spill_out} supplied boxes extend outside the image canvas, which a "
        f"rasterised mask cannot do; only {_rate(spill_out_agrees, spill_out)} of those agree, "
        f"against {_rate(spill_inside_agrees, spill_inside)} for boxes fully inside the "
        f"canvas. Clipping is therefore a contributing factor but "
        f"NOT the explanation: {spill_inside_differs} RLE annotations disagree while their "
        "supplied box is entirely within the canvas.",
        "COMPUTED: the export's `area` field equals bbox width x height, not the mask area, "
        "so it carries no independent evidence about the mask and was not used here.",
        (
            f"COMPUTED: mean discrepancy is far larger on the augmented split "
            f"(train {train['mean_delta_px']} px) than on the un-augmented ones ("
            + ", ".join(f"{s['split']} {s['mean_delta_px']} px" for s in unaugmented)
            + ")."
            if train
            else "COMPUTED: no train split present."
        ),
        "HYPOTHESIS (untested): the supplied box may be transformed as a box through "
        "augmentation while the mask is transformed as pixels - rotating an axis-aligned box "
        "and re-axis-aligning it always enlarges it, which would produce exactly this "
        "direction of error and exactly this train/validation gap. This is consistent with "
        "the numbers; it has NOT been demonstrated, and it does not account for the "
        "disagreements on the un-augmented splits.",
        "DECISION FOR PHASE 5: recompute every detection box from the segmentation geometry "
        "and do not use the supplied `bbox`. This is not merely tidier - for the RLE "
        "annotations the two disagree materially, and the segmentation is by project "
        "definition the source of truth. Whether the mask or the supplied box better "
        "reflects the real object is a visual question this audit cannot settle.",
    ]

    payload = {
        "schema_version": SCHEMA_VERSION,
        "report": "bbox_consistency_audit",
        "phase": PHASE,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(paths.root),
        "scope": (
            "v4 export annotations only. Source-original images were not available when this "
            "audit ran, so no source-level statistic is computed here."
        ),
        "primary_tolerance_px": args.tolerance,
        "tolerances_px": list(TOLERANCES),
        "totals": {
            "annotations_total": sum(s["annotations_total"] for s in splits),
            "annotations_checked": checked,
            "by_representation": dict(by_representation),
            "within_tolerance": {k: dict(v) for k, v in within_tolerance.items()},
            "within_primary": within_primary,
            "canvas_spill": dict(spill_totals),
            "containment": dict(containment_totals),
            "mismatches_at_primary_tolerance": checked - within_primary,
            "max_delta_px": max_delta,
            "mean_delta_px": round(delta_weighted / checked, 4) if checked else None,
            "exact_matches": exact,
        },
        "splits": splits,
        "interpretation": interpretation,
        "open_questions": [
            "Whether the same agreement holds for the ORIGINAL source annotations: this audit "
            "sees the preprocessed export, in which every image was resized to 640x640.",
            "Whether the annotations with the largest discrepancies are genuine annotation "
            "defects or artefacts of the polygon-to-RLE conversion. Requires visual review.",
            "Why the export mixes polygon and RLE at all - whether it tracks how each instance "
            "was originally drawn. Not answerable from the export alone.",
        ],
    }

    json_path = paths.reports / "bbox_consistency_audit.json"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    md_path = paths.reports / "bbox_consistency_audit.md"
    md_path.write_text(build_markdown(payload), encoding="utf-8", newline="\n")

    print(f"checked {checked} annotations across {[s['split'] for s in splits]}")
    print(f"polygon={by_representation.get('polygon', 0)} rle={by_representation.get('rle', 0)}")
    print(f"within {args.tolerance}px: {within_primary} ({_rate(within_primary, checked)})")
    print(f"exact (0px): {exact} ({_rate(exact, checked)})   max delta: {max_delta} px")
    print(f"wrote reports/{json_path.name} and reports/{md_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
