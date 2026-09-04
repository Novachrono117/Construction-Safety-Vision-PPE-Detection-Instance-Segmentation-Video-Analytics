"""Test whether a geometry rule can identify the annotations added since v4.

Phase 5A measured that the 76 annotations added since the version-4 snapshot each
lie inside an annotation of their own class. Phase 5B needs that observation as a
*rule* the pipeline can apply from the canonical current state alone - excluding
annotations by a memorised list of identifiers would not be reproducible and
would not generalise to the next provider edit.

So this script computes same-class enclosure, relative size and absolute size for
every current annotation, sweeps candidate thresholds, and scores each candidate
against the version-4 diff. Version 4 is the yardstick here and nothing more: no
candidate rule reads it, the provider split, or any identifier list.

The honest outcome may be that no rule works. That is a result, not a failure to
report: a rule that deletes real annotations to reach a target count would be
worse than no rule.

Runs entirely offline. Requires:

* ``data/interim/source_geometry.jsonl``   scripts/recover_source_geometry.py
* ``reports/v4_source_mapping.csv``        scripts/map_v4_sources.py

Writes:
    reports/fragment_rule_analysis.csv           committed; per-annotation features
    reports/fragment_rule_report.md              committed; the written analysis
    reports/figures/fragment_rule_distribution.png

Usage:
    uv run python scripts/analyze_fragment_rule.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from construction_safety_vision.data.drift import DriftInstance, match_instances, normalise_bbox
from construction_safety_vision.data.fragments import (
    AnnotationShape,
    FragmentFeatures,
    box_containment,
    describe_image,
    encode_mask,
    evaluate_rule,
)
from construction_safety_vision.data.geometry import bbox_from_segmentation
from construction_safety_vision.paths import ProjectPaths
from construction_safety_vision.provenance import git_commit

EXPORT_DIRNAME = "construction-ppe-compliance-detection-v4-coco-segmentation"
"""Directory the version-4 export was extracted into, under ``data/raw``."""

GEOMETRY_FILENAME = "source_geometry.jsonl"
"""Recovered canonical geometry, under ``data/interim``."""

ANALYSIS_CSV = "fragment_rule_analysis.csv"
"""Committed per-annotation feature table."""

ANALYSIS_REPORT = "fragment_rule_report.md"
"""Committed written analysis."""

DISTRIBUTION_FIGURE = "fragment_rule_distribution.png"
"""Committed distribution plot."""

ENCLOSURE_GRID = (0.80, 0.90, 0.95, 0.99)
"""Same-class enclosure thresholds swept."""

RATIO_GRID = (0.01, 0.05, 0.10, 0.25, 0.50, 1.01)
"""Relative-size thresholds swept."""

ACCEPTABLE_PRECISION = 0.95
"""Precision a rule must reach before it may be applied without review."""

ACCEPTABLE_RECALL = 0.95
"""Recall a rule must reach before it may be applied without review."""

DIAGNOSTIC_ENCLOSURE = 0.95
"""Enclosure at which the diagnosis counts an addition as still nested."""

CSV_COLUMNS = (
    "image_id",
    "annotation_id",
    "label",
    "geometry_kind",
    "enclosed_bbox",
    "enclosed_mask",
    "area_ratio",
    "relative_image_area",
    "absolute_area_px",
    "container_id",
    "is_v4_addition",
)
"""Column order of the committed feature table."""


class AnalysisInputError(RuntimeError):
    """Raised when a required input artifact is missing."""


def load_geometry(path: Path) -> dict[str, dict]:
    """Load the recovered canonical geometry.

    Args:
        path: The interim geometry file.

    Returns:
        Records keyed by source image id.

    Raises:
        AnalysisInputError: If the file is absent.
    """
    if not path.is_file():
        msg = f"{path.name} not found; run scripts/recover_source_geometry.py first"
        raise AnalysisInputError(msg)
    return {
        json.loads(line)["image_id"]: json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    }


def load_reference(paths: ProjectPaths, geometry: dict[str, dict]) -> set[tuple[str, str]]:
    """Recompute which annotations the version-4 diff calls additions.

    This is the evaluation yardstick. It is deliberately recomputed here rather
    than read from a stored list, so the report cannot drift away from the diff
    it claims to be scored against.

    Args:
        paths: Project layout.
        geometry: Recovered canonical geometry keyed by source image id.

    Returns:
        ``(image id, annotation id)`` pairs with no version-4 counterpart.

    Raises:
        AnalysisInputError: If the mapping or the export is absent.
    """
    mapping_path = paths.reports / "v4_source_mapping.csv"
    if not mapping_path.is_file():
        msg = "v4_source_mapping.csv not found; run scripts/map_v4_sources.py first"
        raise AnalysisInputError(msg)
    with mapping_path.open(encoding="utf-8", newline="") as handle:
        mapping = {row["source_image_id"]: row for row in csv.DictReader(handle)}

    export_root = paths.data_raw / EXPORT_DIRNAME
    snapshot: dict[tuple[str, int], dict] = {}
    for split in ("train", "valid", "test"):
        document = export_root / split / "_annotations.coco.json"
        if not document.is_file():
            msg = f"Export split {split!r} not found; run scripts/download_dataset.py first"
            raise AnalysisInputError(msg)
        payload = json.loads(document.read_text(encoding="utf-8"))
        categories = {int(c["id"]): str(c["name"]) for c in payload["categories"]}
        for image in payload["images"]:
            snapshot[(split, int(image["id"]))] = {
                "width": int(image["width"]),
                "height": int(image["height"]),
                "annotations": [],
            }
        for annotation in payload["annotations"]:
            key = (split, int(annotation["image_id"]))
            if key in snapshot:
                snapshot[key]["annotations"].append(
                    (categories[int(annotation["category_id"])], annotation["segmentation"])
                )

    additions: set[tuple[str, str]] = set()
    for image_id, record in geometry.items():
        row = mapping.get(image_id)
        if not row or not row["v4_coco_image_id"]:
            continue
        image = snapshot[(row["v4_split"], int(row["v4_coco_image_id"]))]
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
                f"v4-{index}",
                label,
                normalise_bbox(
                    bbox_from_segmentation(segmentation).as_list(),
                    width=image["width"],
                    height=image["height"],
                ),
            )
            for index, (label, segmentation) in enumerate(image["annotations"])
        ]
        _, added, _ = match_instances(current, old)
        additions.update((image_id, instance.key) for instance in added)
    return additions


def diagnose_additions(paths: ProjectPaths, geometry: dict[str, dict]) -> dict[str, Any]:
    """Explain why an addition is or is not reachable by a same-class enclosure rule.

    An addition can only be found by enclosure if the annotation that encloses it
    still exists in the current state. Where the version-4 parent was a single
    coarse annotation that the current state replaced with several precise ones,
    the parent is gone and nothing encloses the additions any more.

    Args:
        paths: Project layout.
        geometry: Recovered canonical geometry keyed by source image id.

    Returns:
        Counts of additions by the fate of their version-4 parent, plus the
        images where a coarse parent was replaced.
    """
    with (paths.reports / "v4_source_mapping.csv").open(encoding="utf-8", newline="") as handle:
        mapping = {row["source_image_id"]: row for row in csv.DictReader(handle)}
    export_root = paths.data_raw / EXPORT_DIRNAME

    parent_removed = 0
    parent_survived = 0
    parent_fates: dict[str, int] = {}
    replaced_images: list[dict[str, Any]] = []

    for image_id in sorted(geometry):
        row = mapping.get(image_id)
        if not row or not row["v4_coco_image_id"]:
            continue
        payload = json.loads(
            (export_root / row["v4_split"] / "_annotations.coco.json").read_text(encoding="utf-8")
        )
        categories = {int(c["id"]): str(c["name"]) for c in payload["categories"]}
        target = int(row["v4_coco_image_id"])
        image = next(im for im in payload["images"] if int(im["id"]) == target)
        annotations = [a for a in payload["annotations"] if int(a["image_id"]) == target]
        record = geometry[image_id]
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
                f"v4-{index}",
                categories[int(a["category_id"])],
                normalise_bbox(
                    bbox_from_segmentation(a["segmentation"]).as_list(),
                    width=image["width"],
                    height=image["height"],
                ),
            )
            for index, a in enumerate(annotations)
        ]
        matches, added, removed = match_instances(current, old)
        if not added:
            continue
        surviving = {snapshot.key for _, snapshot, _ in matches}
        local_orphaned = 0
        for instance in added:
            # Is the addition still enclosed by a same-class annotation that exists now?
            enclosed_now = max(
                (
                    box_containment(instance.bbox, other.bbox)
                    for other in current
                    if other.key != instance.key and other.label == instance.label
                ),
                default=0.0,
            )
            if enclosed_now >= DIAGNOSTIC_ENCLOSURE:
                parent_survived += 1
                continue
            parent_removed += 1
            local_orphaned += 1
            enclosing = [
                (box_containment(instance.bbox, o.bbox), o)
                for o in old
                if o.label == instance.label
            ]
            if enclosing:
                _, parent = max(enclosing, key=lambda entry: entry[0])
                fate = "reshaped" if parent.key in surviving else "removed"
            else:
                fate = "none"
            parent_fates[fate] = parent_fates.get(fate, 0) + 1
        if local_orphaned:
            replaced_images.append(
                {
                    "image_id": image_id,
                    "additions": len(added),
                    "orphaned": local_orphaned,
                    "v4_annotations": len(old),
                    "current_annotations": len(current),
                    "removed": len(removed),
                }
            )

    return {
        "enclosed_now": parent_survived,
        "not_enclosed_now": parent_removed,
        "parent_fates": dict(sorted(parent_fates.items())),
        "replaced_images": sorted(replaced_images, key=lambda e: -e["orphaned"]),
    }


def compute_features(geometry: dict[str, dict]) -> list[FragmentFeatures]:
    """Compute fragment features for every canonical annotation.

    Args:
        geometry: Recovered canonical geometry keyed by source image id.

    Returns:
        One feature record per annotation, ordered by image then annotation id.
    """
    features: list[FragmentFeatures] = []
    for image_id in sorted(geometry):
        record = geometry[image_id]
        height, width = int(record["height"]), int(record["width"])
        shapes = [
            AnnotationShape(
                image_id=image_id,
                annotation_id=str(a["annotation_id"]),
                label=str(a["label"]),
                geometry_kind=str(a["geometry_kind"]),
                bbox=tuple(float(v) for v in a["bbox"]),  # type: ignore[arg-type]
                area=a.get("area"),
                rle=(
                    encode_mask(a["segmentation"], height=height, width=width)
                    if a.get("segmentation") is not None
                    else None
                ),
            )
            for a in record["annotations"]
        ]
        features.extend(describe_image(shapes, height=height, width=width))
    return features


def candidate_rules() -> list[tuple[str, Any]]:
    """Build the candidate rules to sweep.

    Returns:
        ``(description, predicate)`` pairs.
    """
    rules: list[tuple[str, Any]] = []
    for enclosure in ENCLOSURE_GRID:
        for ratio in RATIO_GRID:
            rules.append(
                (
                    f"enclosed_bbox >= {enclosure} AND area_ratio < {ratio}",
                    lambda f, e=enclosure, r=ratio: f.enclosed_bbox >= e and f.area_ratio < r,
                )
            )
            rules.append(
                (
                    f"enclosed_mask >= {enclosure} AND area_ratio < {ratio}",
                    lambda f, e=enclosure, r=ratio: f.enclosed_mask >= e and f.area_ratio < r,
                )
            )
    return rules


def quantiles(values: list[float]) -> list[float]:
    """Return a fixed quantile summary.

    Args:
        values: Sample values.

    Returns:
        Quantiles at 0, 25, 50, 75 and 100 percent.
    """
    if not values:
        return [0.0] * 5
    ordered = sorted(values)
    return [
        round(ordered[min(int(q * (len(ordered) - 1)), len(ordered) - 1)], 6)
        for q in (0.0, 0.25, 0.5, 0.75, 1.0)
    ]


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write the feature table with LF endings.

    Args:
        path: Destination file.
        rows: Rows to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def draw_distribution(
    features: list[FragmentFeatures], reference: set[tuple[str, str]], destination: Path
) -> None:
    """Plot the candidate features for additions against every other annotation.

    Args:
        features: Feature records.
        reference: Version-4 additions.
        destination: Output file.
    """
    additions = [f for f in features if (f.image_id, f.annotation_id) in reference]
    others = [f for f in features if (f.image_id, f.annotation_id) not in reference]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    panels = (
        ("same-class enclosure (bbox)", lambda f: f.enclosed_bbox, None),
        ("same-class enclosure (mask)", lambda f: f.enclosed_mask, None),
        ("area relative to enclosing annotation", lambda f: min(f.area_ratio, 1.0), None),
    )
    for axis, (title, getter, _) in zip(axes, panels, strict=True):
        axis.hist(
            [getter(f) for f in others],
            bins=25,
            alpha=0.65,
            label=f"other annotations (n={len(others)})",
            color="#4C78A8",
            density=True,
        )
        axis.hist(
            [getter(f) for f in additions],
            bins=25,
            alpha=0.65,
            label=f"v4 additions (n={len(additions)})",
            color="#E45756",
            density=True,
        )
        axis.set_title(title, fontsize=9)
        axis.set_ylabel("density", fontsize=8)
        axis.tick_params(labelsize=8)
        axis.legend(fontsize=7)
    fig.suptitle(
        "Fragment-rule candidate features: annotations added since v4 vs the rest", fontsize=11
    )
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=110)
    plt.close(fig)


def build_report(
    features: list[FragmentFeatures],
    reference: set[tuple[str, str]],
    outcomes: list[Any],
    best: Any,
    safest: Any,
    accepted: bool,
    diagnosis: dict[str, Any],
    commit: str | None,
) -> str:
    """Compose the written rule analysis.

    Args:
        features: Feature records.
        reference: Version-4 additions.
        outcomes: Every scored candidate rule.
        best: The best-scoring candidate by precision plus recall.
        safest: The highest-recall candidate that selects no annotation outside
            the reference set.
        accepted: Whether the best candidate meets the acceptance bar.
        diagnosis: Why unreachable additions are unreachable.
        commit: Git commit the analysis ran at.

    Returns:
        The Markdown document.
    """
    additions = [f for f in features if (f.image_id, f.annotation_id) in reference]
    others = [f for f in features if (f.image_id, f.annotation_id) not in reference]
    generated = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S+00:00")

    lines = [
        "# Fragment-Rule Analysis",
        "",
        f"Generated: {generated} · Phase: 5B · Commit: `{commit or 'unknown'}`",
        "",
        "**Question.** Phase 5A found that the 76 annotations added since the version-4 "
        "snapshot each sit inside an annotation of their own class. Can that be turned into a "
        "deterministic rule, computed from the canonical current state alone, that identifies "
        "them without deleting legitimate annotations?",
        "",
        "**Why a rule and not a list.** Excluding 76 memorised identifiers would not be "
        "reproducible, would not survive the next provider edit, and would encode a conclusion "
        "rather than a criterion. Version 4 is used here only to score candidates.",
        "",
        "> **What the scores below mean, and what they do not.**",
        ">",
        "> The 76 additions are a **historical annotation-drift reference set**: the "
        "annotations that changed between two snapshots. They are **not** ground truth for bad "
        "annotations, and nobody has established them to be wrong.",
        ">",
        "> So precision here measures **agreement with historical drift**, not detection of "
        "annotation error. A rule at precision 1.00 selects annotations that all changed since "
        "version 4; it does not select annotations that are all incorrect. Reading it the "
        "second way would be invalid.",
        "",
        "## Feature distributions (COMPUTED RESULT)",
        "",
        f"Across {len(features)} canonical annotations, of which {len(additions)} are version-4 "
        f"additions and {len(others)} are not:",
        "",
        "| Feature | Group | min | q25 | median | q75 | max |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, getter in (
        ("same-class enclosure (bbox)", lambda f: f.enclosed_bbox),
        ("same-class enclosure (mask)", lambda f: f.enclosed_mask),
        ("area / enclosing annotation", lambda f: f.area_ratio),
        ("area / image", lambda f: f.relative_image_area),
        ("absolute area (px)", lambda f: f.absolute_area_px),
    ):
        for group, sample in (("additions", additions), ("others", others)):
            values = quantiles([getter(f) for f in sample])
            cells = " | ".join(str(v) for v in values)
            lines.append(f"| {name} | {group} | {cells} |")
    lines.append("")

    lines.extend(
        [
            "See `reports/figures/fragment_rule_distribution.png`.",
            "",
            "## Candidate rules scored against the version-4 diff (COMPUTED RESULT)",
            "",
            "| Rule | Selected | TP | FP | FN | Precision | Recall |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for outcome in sorted(outcomes, key=lambda o: (-(o.precision + o.recall), o.name))[:16]:
        lines.append(
            f"| `{outcome.name}` | {outcome.selected} | {outcome.true_positives} | "
            f"{outcome.false_positives} | {outcome.false_negatives} | "
            f"{outcome.precision:.4f} | {outcome.recall:.4f} |"
        )
    lines.extend(
        [
            "",
            f"Best candidate by precision plus recall: `{best.name}`, with precision "
            f"{best.precision:.4f} and recall {best.recall:.4f}.",
            "",
            f"Acceptance bar for applying a rule without review: precision >= "
            f"{ACCEPTABLE_PRECISION} and recall >= {ACCEPTABLE_RECALL}.",
            "",
            f"**Result: the bar is {'met' if accepted else 'NOT met'}.**",
            "",
            "## Verdict: NO AUTOMATIC FRAGMENT FILTER ADOPTED",
            "",
            "`fragment_rule_status = REJECTED_FOR_AUTOMATIC_FILTERING`. All 2031 canonical "
            "annotations are retained and no annotation is excluded on geometric grounds.",
            "",
            "The experiment is kept here because a failed rule is useful negative evidence: it "
            "records what was tried, on what features, at what thresholds, and why it does not "
            "work. It should not be re-run in the hope of a better threshold - the reason it "
            "fails is not the threshold, as the next two sections show.",
            "",
        ]
    )

    survived = diagnosis["enclosed_now"]
    orphaned = diagnosis["not_enclosed_now"]
    lines.extend(
        [
            "## Why recall is capped (COMPUTED RESULT)",
            "",
            "An enclosure rule can only find an addition if the annotation that encloses it "
            "still exists. Splitting the additions by what happened to their version-4 parent:",
            "",
            "| Is the addition still enclosed by a same-class annotation? | Additions |",
            "| --- | --- |",
            f"| Yes, at >= {DIAGNOSTIC_ENCLOSURE} | {survived} |",
            f"| No - nothing encloses it any more | {orphaned} |",
            f"| **total** | **{survived + orphaned}** |",
            "",
            "For the ones nothing encloses, what became of the version-4 annotation that did:",
            "",
            "| Version-4 parent | Additions |",
            "| --- | --- |",
            *(f"| {fate} | {count} |" for fate, count in diagnosis["parent_fates"].items()),
            "",
            f"These {orphaned} additions are unreachable by construction, not by a poor choice "
            "of threshold: no same-class annotation encloses them today. Either the coarse "
            "version-4 annotation that did was deleted, or it survived in a much smaller "
            "reshaped form that no longer covers them. In both cases the current state has "
            "replaced one loose annotation with several tighter ones.",
            "",
            "### What replaced them (COMPUTED RESULT)",
            "",
            "| Source image | v4 annotations | current | additions | orphaned | v4 removed |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for entry in diagnosis["replaced_images"][:12]:
        lines.append(
            f"| `{entry['image_id'][:8]}` | {entry['v4_annotations']} | "
            f"{entry['current_annotations']} | {entry['additions']} | {entry['orphaned']} | "
            f"{entry['removed']} |"
        )
    lines.extend(
        [
            "",
            "## Reading (interpretation, labelled)",
            "",
            "**FACT.** No candidate rule reaches the acceptance bar. The best one is precise "
            f"({best.precision:.2f}) but recovers under half the additions "
            f"({best.recall:.2f}).",
            "",
            "**FACT, and a correction to phase 5A.** The phase 5A report described the 76 "
            "additions as fragments that 'add no coverage'. The containment measurement behind "
            "that statement is correct, but the inference was not. Inspecting the affected "
            "images shows two different things mixed together:",
            "",
            "* genuinely degenerate slivers - a few dozen pixels drawn on a bracelet, on glove "
            "  lettering, on a hard hat - which the precise rule above does find;",
            "* **re-annotations that improve the labels**: a single coarse `vest_on_body` "
            "  polygon replaced by several precise ones covering the parts of the vest that are "
            "  actually visible, and in at least one image a single oversized `person` box "
            "  covering two people replaced by one box per person.",
            "",
            "The second kind *does* add coverage. It looked like it did not because the "
            "containment test asked whether an addition sits inside a version-4 annotation, and "
            "a coarse over-merged parent contains its own corrections by definition.",
            "",
            "**MANUAL DECISION.** The project owner reviewed this analysis and rejected "
            "automatic filtering. Containment inside an older same-class annotation is not "
            "evidence that an annotation is wrong, and the evaluated rules cannot separate",
            "",
            "* annotation fragments, from",
            "* legitimate instance splits, from",
            "* geometry refinements, from",
            "* corrections of previously merged objects.",
            "",
            "Automatic exclusion would therefore carry an unacceptable risk of deleting real "
            "ground truth. All 2031 annotations are retained.",
            f"The {safest.selected} annotations the strictest zero-disagreement candidate "
            f"(`{safest.name}`) would have selected are **not** excluded either, and no second "
            "action row was created for them.",
            "",
            "**What the flag means now.** Annotations the broader candidate rule selects carry "
            "`action = KEEP` and `review_flag = NESTED_SAME_CLASS_CANDIDATE` in "
            "`canonical_annotation_actions.csv`. The wording is deliberate: the flag records a "
            "geometric relationship, not a defect. Calling them fragments would assert "
            "something no one has established.",
            "",
            "**PHASE 5C INPUT.** The modelling annotation population is the full canonical "
            "2031. If a subset is ever to be removed, it needs a manual disposition per "
            "annotation, not a threshold.",
            "",
        ]
    )
    # A trailing empty entry would end the file on a blank line, which git
    # reports as whitespace damage.
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Analyse candidate fragment rules.

    Args:
        argv: Command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code. Always zero: the script reports evidence, and whether a
        rule is adopted is a recorded decision rather than something this script
        gates. The verdict is printed and written into the report.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)

    paths = ProjectPaths.from_root()
    try:
        geometry = load_geometry(paths.data_interim / GEOMETRY_FILENAME)
        reference = load_reference(paths, geometry)
    except AnalysisInputError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"canonical annotations: {sum(len(r['annotations']) for r in geometry.values())}")
    print(f"version-4 additions (evaluation reference only): {len(reference)}")
    print("computing same-class enclosure features ...", flush=True)
    features = compute_features(geometry)

    outcomes = [
        evaluate_rule(name, features, predicate, reference) for name, predicate in candidate_rules()
    ]
    best = max(outcomes, key=lambda o: (o.precision + o.recall, o.recall))
    accepted = best.precision >= ACCEPTABLE_PRECISION and best.recall >= ACCEPTABLE_RECALL

    write_csv(
        paths.reports / ANALYSIS_CSV,
        [
            {
                **f.as_row(),
                "is_v4_addition": (f.image_id, f.annotation_id) in reference,
            }
            for f in features
        ],
    )
    draw_distribution(features, reference, paths.figures / DISTRIBUTION_FIGURE)
    diagnosis = diagnose_additions(paths, geometry)
    clean = [o for o in outcomes if o.false_positives == 0 and o.selected]
    safest = max(clean, key=lambda o: (o.recall, -o.selected)) if clean else best
    report = build_report(
        features, reference, outcomes, best, safest, accepted, diagnosis, git_commit()
    )
    (paths.reports / ANALYSIS_REPORT).write_text(report, encoding="utf-8", newline="\n")

    print("\n--- best candidate ---")
    print(f"  rule:      {best.name}")
    print(f"  selected:  {best.selected}")
    print(f"  TP/FP/FN:  {best.true_positives}/{best.false_positives}/{best.false_negatives}")
    print(f"  precision: {best.precision:.4f}")
    print(f"  recall:    {best.recall:.4f}")
    print(f"  accepted:  {accepted}  (bar: P>={ACCEPTABLE_PRECISION}, R>={ACCEPTABLE_RECALL})")
    print("\n--- safest candidate (no false positives) ---")
    print(f"  rule:      {safest.name}")
    print(
        f"  selected:  {safest.selected}  TP {safest.true_positives}  FP {safest.false_positives}"
    )
    print(f"  precision: {safest.precision:.4f}   recall: {safest.recall:.4f}")
    print(f"\nwrote {paths.reports / ANALYSIS_CSV}")
    print(f"wrote {paths.reports / ANALYSIS_REPORT}")
    print(f"wrote {paths.figures / DISTRIBUTION_FIGURE}")
    if not accepted:
        print(
            "\nVERDICT: no candidate reaches the acceptance bar. Automatic filtering "
            "was rejected by owner decision; all annotations are retained."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
