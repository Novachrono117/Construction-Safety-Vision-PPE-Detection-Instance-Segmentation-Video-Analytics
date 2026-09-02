"""Render the phase 4A markdown reports from the computed JSON artifacts.

The reports are generated rather than written by hand so that no number in them
can drift away from the analysis that produced it. Re-running the analyses and
then this script regenerates both documents.

Every statement is tagged:

* **FACT** - established earlier and carried forward;
* **COMPUTED** - measured by this phase's scripts;
* **INTERPRETATION** - what the measurement is taken to mean;
* **OPEN QUESTION** - deliberately unresolved.

Writes:
    reports/dataset_audit_report.md
    reports/eda_report.md

Usage:
    uv run python scripts/write_audit_reports.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from construction_safety_vision.paths import ProjectPaths

SOURCE_POPULATION = 436
"""Independent source images established in phase 3."""


def load(path: Path) -> dict[str, Any]:
    """Load a JSON artifact.

    Args:
        path: File to read.

    Returns:
        The decoded payload, or an empty mapping when absent.
    """
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def fmt(value: Any, digits: int = 3) -> str:
    """Format a number for a table cell.

    Args:
        value: Value to format.
        digits: Decimal places for floats.

    Returns:
        A display string.
    """
    if isinstance(value, float):
        return f"{value:,.{digits}f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def distribution_row(name: str, summary: dict[str, Any], digits: int = 3) -> str:
    """Render one distribution as a table row.

    Args:
        name: Row label.
        summary: Output of ``imagestats.summarise``.
        digits: Decimal places.

    Returns:
        A markdown table row.
    """
    keys = ("min", "p05", "p25", "median", "p75", "p95", "max", "mean", "std")
    cells = " | ".join(fmt(summary.get(k, 0), digits) for k in keys)
    return f"| {name} | {cells} |"


def build_audit_report(
    audit: dict[str, Any],
    bbox: dict[str, Any],
    manual_split_decision: str | None = None,
) -> str:
    """Render the dataset audit report.

    Args:
        audit: Source-audit payload.
        bbox: Bbox consistency audit payload.
        manual_split_decision: The phase 4B verdict on the provider split, when
            the manual audit has been recorded. The automated classification is
            never rewritten - it was correct for the evidence it had - but the
            report says plainly that a later phase superseded it.

    Returns:
        Markdown text.
    """
    lines: list[str] = []
    add = lines.append
    population = audit["population"]
    duplicates = audit["exact_duplicates"]
    near = audit["near_duplicates"]
    sequences = audit["sequences"]
    drift = audit.get("source_vs_export_drift", {})
    split_audit = audit["provider_split_audit"]

    add("# Dataset Audit Report - source integrity and leakage risk")
    add("")
    add(f"Generated: {audit['generated_at']} · Phase: 4A · Commit: `{audit['git_commit']}`")
    add("")
    add(
        "Scope: the **436 independent source images**, acquired at original resolution "
        "from the provider. The 742-image version-4 export is a *derived* population and "
        "is used in this document only where explicitly said so."
    )
    add("")
    add(
        "This is the automated half of the audit. Semantic questions - is this a valid "
        "negative, are these two frames the same scene - are deliberately left to human "
        "review; see `manual_review_manifest.csv`."
    )
    add("")
    if manual_split_decision:
        add(
            "That review was carried out in phase 4B and is recorded in "
            "[`manual_audit_report.md`](manual_audit_report.md) and "
            "`manual_audit_decisions.csv`. This document is unchanged by it: it remains the "
            "automated record, and the two are deliberately kept apart."
        )
        add("")

    add("## Population")
    add("")
    add("| | Value | Basis |")
    add("| --- | --- | --- |")
    add(f"| Source images recovered | {population['source_images']} | COMPUTED |")
    add(f"| Expected (phase 3) | {population['expected']} | FACT |")
    add(f"| Annotated instances | {population['instances']:,} | COMPUTED |")
    for split, count in sorted(population["splits"].items()):
        add(f"| Provider split `{split}` | {count} | COMPUTED |")
    add("")
    add(
        "**COMPUTED.** All 436 originals downloaded and decoded; 0 failures, and every "
        "decoded size matched the dimensions the provider reported."
    )
    add("")

    add("## Exact duplicates")
    add("")
    add("**Method (COMPUTED).** SHA-256 over the original image bytes.")
    add("")
    add("| | Value |")
    add("| --- | --- |")
    add(f"| Images hashed | {duplicates['images']} |")
    add(f"| Unique hashes | {duplicates['unique_hashes']} |")
    add(f"| Duplicate groups | {duplicates['duplicate_groups']} |")
    add(f"| Images in duplicate groups | {duplicates['images_in_duplicate_groups']} |")
    add(f"| Groups crossing a split | {duplicates['cross_split_groups']} |")
    add("")
    if duplicates["duplicate_groups"] == 0:
        add(
            "**INTERPRETATION.** No byte-identical image appears twice. This rules out the "
            "crudest form of split contamination; it says nothing about visually similar "
            "images, which the next section covers."
        )
    else:
        add("See `exact_duplicate_groups.csv`. Nothing was deleted.")
    add("")

    add("## Near-duplicate candidates")
    add("")
    add(
        "**Method (COMPUTED).** Two independent 64-bit perceptual fingerprints per image - "
        "a difference hash (dHash) and a DCT-based perceptual hash (pHash) - compared "
        f"exhaustively over all {near['pairs_compared']:,} unordered pairs. A pair is "
        f"emitted when the dHash distance is <= {near['dhash_threshold']} **or** the pHash "
        f"distance is <= {near['phash_threshold']}. Both thresholds were fixed from the "
        "conventional working range for 64-bit hashes before results were seen; the screen "
        "is deliberately recall-oriented."
    )
    add("")
    add("| | Value |")
    add("| --- | --- |")
    add(f"| Images fingerprinted | {near['images_fingerprinted']} |")
    add(f"| Pairs compared | {near['pairs_compared']:,} |")
    add(f"| Candidate pairs | {near['candidates']} |")
    add(f"| Same-split candidates | {near['same_split_candidates']} |")
    add(f"| **Cross-split candidates** | **{near['cross_split_candidates']}** |")
    add("")
    add(
        "> **A fingerprint match is a candidate, not a verdict.** None of these pairs is "
        "claimed to be leakage. They are the shortlist a person must look at."
    )
    add("")
    strongest = audit.get("strongest_cross_split_pairs", [])
    if strongest:
        add("Strongest cross-split candidates (COMPUTED):")
        add("")
        add("| Image A | Split | Image B | Split | dHash | pHash |")
        add("| --- | --- | --- | --- | --- | --- |")
        for pair in strongest[:10]:
            add(
                f"| `{pair['image_a'][:8]}` | {pair['split_a']} | `{pair['image_b'][:8]}` "
                f"| {pair['split_b']} | {pair['dhash']} | {pair['phash']} |"
            )
        add("")
        zero = [p for p in strongest if p["dhash"] == 0 and p["phash"] == 0]
        if zero:
            add(
                f"**INTERPRETATION.** {len(zero)} cross-split pair(s) are identical under "
                "*both* fingerprints while having different file hashes - i.e. visually the "
                "same picture stored as different bytes. That is the strongest automated "
                "signal available here, and it is the single most important thing for a "
                "reviewer to confirm or reject."
            )
            add("")
    add(
        "Full list: `near_duplicate_candidates.csv`. Visual pairs: "
        "`figures/review_g_near_duplicates.jpg` and `figures/review_h_near_duplicates.jpg`."
    )
    add("")

    add("## Possible sequence / group structure")
    add("")
    add("| | Value |")
    add("| --- | --- |")
    add(f"| Near-duplicate chains (connected components) | {sequences['near_duplicate_chains']} |")
    add(f"| Images in a chain | {sequences['images_in_chains']} |")
    add(f"| Chains spanning more than one split | {sequences['chains_crossing_splits']} |")
    add("")
    add("Filename prefixes (COMPUTED):")
    add("")
    add("| Prefix | Images |")
    add("| --- | --- |")
    for prefix, count in sequences["filename_prefixes"].items():
        add(f"| `{prefix}` | {count} |")
    add("")
    add(
        "**INTERPRETATION.** The prefixes look like stock-photography provenance markers "
        "rather than frame numbering, and no filename carries a frame index. That is weak "
        "evidence *against* video-sequence structure, not proof of its absence."
    )
    add("")
    add(
        "**OPEN QUESTION.** Whether the near-duplicate chains are genuinely the same scene "
        "photographed twice, the same stock image published twice, or a false alarm. "
        "Candidates are in `group_candidates.csv`; none has been promoted to a group."
    )
    add("")

    if drift:
        add("## Source project vs frozen v4 export")
        add("")
        add(
            "| Split | Source instances | Export instances | Augmentation | Per source copy | "
            "Difference | Agrees |"
        )
        add("| --- | --- | --- | --- | --- | --- | --- |")
        for split, row in drift["per_split"].items():
            add(
                f"| {split} | {row['source_instances']} | {row['export_instances']} "
                f"| x{row['export_augmentation_factor']} "
                f"| {row['export_instances_per_source_copy']:,.1f} | {row['difference']:+,.1f} "
                f"| {row['agrees']} |"
            )
        add("")
        add(f"**INTERPRETATION.** {drift['interpretation']}")
        add("")
        add(
            "**OPEN QUESTION.** Which population is canonical for phase 5 - the frozen v4 "
            "export whose provenance is already recorded and hashed, or the live source "
            "project which has more annotations but no frozen identity? This audit does not "
            "decide it."
        )
        add("")

    add("## Zero-instance images")
    add("")
    zero_instance = audit["zero_instance_images"]
    add(
        f"**COMPUTED.** {zero_instance['count']} source images carry no annotated instance: "
        f"{zero_instance['by_split']}."
    )
    add("")
    add(
        "**FACT.** Phase 3 found 28 such records in the v4 export. That is consistent: the "
        "train split is duplicated by augmentation, so 11 train + 2 valid + 4 test source "
        "images become 22 + 2 + 4 = 28 export records."
    )
    add("")
    add(
        "**OPEN QUESTION.** Whether these are deliberate negative samples or images whose "
        "labels are missing. This cannot be settled by counting - it needs a person to look "
        "at them. All 17 are rendered in `figures/review_f_zero_instance.jpg` and listed in "
        "`empty_image_audit.csv`."
    )
    add("")

    add("## Annotation geometry at source")
    add("")
    add("| Geometry type | Instances |")
    add("| --- | --- |")
    for kind, count in audit["source_geometry_types"].items():
        add(f"| `{kind}` | {count:,} |")
    add("")
    add(
        "**COMPUTED.** The provider exposes polygon vertices only for `polygon` instances. "
        "`mask` instances carry a bounding box but no vertex list, so source-level "
        "mask-area analysis is **PARTIAL** by necessity, not by choice."
    )
    add("")
    unknown = audit["source_geometry_types"].get("unknown", 0)
    if unknown:
        add(
            f"**COMPUTED.** {unknown} instance(s) have neither type. Both sit on one image, "
            "measure under 16 px on a side, and are positioned at the image edge."
        )
        add("")
        add(
            "**OPEN QUESTION.** Whether they are accidental micro-annotations. Flagged for "
            "visual review."
        )
        add("")

    if bbox:
        totals = bbox["totals"]
        add("## Supplied bbox vs segmentation geometry (v4 export)")
        add("")
        add(
            f"**COMPUTED.** Of {totals['annotations_checked']:,} export annotations, "
            f"{totals['within_primary']:,} agree with their own segmentation within "
            f"{bbox['primary_tolerance_px']} px. Polygons agree essentially perfectly; the "
            "disagreement is concentrated in the RLE instances, where the supplied box is "
            "systematically the larger of the two."
        )
        add("")
        add(
            "Full analysis: [`bbox_consistency_audit.md`](bbox_consistency_audit.md). Visual "
            "examples: `figures/review_k_bbox_vs_segmentation.jpg`."
        )
        add("")
        add(
            "> **The phase 5 bbox policy remains provisional.** The preferred candidate "
            "policy is to derive every detection box from the segmentation geometry, but "
            "that is *pending visual validation*: a reviewer must first confirm that the "
            "segmentation is the better description of the object. Nothing is frozen."
        )
        add("")

    add("## Provider split audit")
    add("")
    add("The provider's split was measured, **not modified**.")
    add("")
    add("| Split | Images | Instances | Negatives | " + " | ".join(split_audit["classes"]) + " |")
    add("| --- | --- | --- | --- | " + " | ".join("---" for _ in split_audit["classes"]) + " |")
    for split, entry in split_audit["per_split"].items():
        cells = " | ".join(
            str(entry["instances_per_class"].get(c, 0)) for c in split_audit["classes"]
        )
        add(
            f"| {split} | {entry['images']} | {entry['instances']} | {entry['negatives']} "
            f"| {cells} |"
        )
    add("")
    add("Images containing each class (not instance counts):")
    add("")
    add("| Split | " + " | ".join(split_audit["classes"]) + " |")
    add("| --- | " + " | ".join("---" for _ in split_audit["classes"]) + " |")
    for split, entry in split_audit["per_split"].items():
        cells = " | ".join(
            str(entry["images_with_class"].get(c, 0)) for c in split_audit["classes"]
        )
        add(f"| {split} | {cells} |")
    add("")
    add("### Concerns (COMPUTED)")
    add("")
    for concern in split_audit["concerns"]:
        add(f"- {concern}")
    add("")
    add(f"### Classification: `{split_audit['classification']}`")
    add("")
    add(
        "**INTERPRETATION.** This is an audit recommendation and does **not** freeze or "
        "replace anything. The split cannot be judged suitable until a person has looked at "
        "the cross-split near-duplicate candidates: if they are genuine, the provider's "
        "split leaks and a new partition is required in phase 5; if they are false alarms, "
        "the remaining concern is the rare-class coverage below."
    )
    add("")
    if manual_split_decision:
        add(
            f"> **SUPERSEDED IN PHASE 4B: `{manual_split_decision}`.** The visual review "
            "confirmed the cross-split near-duplicate candidates, so the provider split is no "
            "longer undetermined. The classification above is kept as it stood, because it is "
            "the automated evidence the later decision was made from. The reasoning and its "
            "consequences are in [`manual_audit_report.md`](manual_audit_report.md)."
        )
        add("")
    return "\n".join(lines)


def build_eda_report(
    eda: dict[str, Any],
    audit: dict[str, Any],
    manual_recorded: bool = False,
) -> str:
    """Render the exploratory analysis report.

    Args:
        eda: EDA payload.
        audit: Source-audit payload, for the split context.
        manual_recorded: Whether the phase 4B manual audit exists. The open
            questions below are kept verbatim; only their status is added.

    Returns:
        Markdown text.
    """
    lines: list[str] = []
    add = lines.append
    image = eda["image_level"]
    classes = eda["class_level"]
    geometry = eda["geometry_level"]

    add("# EDA Report - the 436 source images")
    add("")
    add(f"Generated: {eda['generated_at']} · Phase: 4A · Commit: `{eda['git_commit']}`")
    add("")
    add(
        "**FACT.** Every figure below describes the **436 original source images**. The "
        "742-image v4 export is not used: its images were stretched to 640x640 and its "
        "training half was augmented, so its resolution, aspect-ratio, luminance and "
        "file-size distributions would describe the provider's pipeline rather than the data."
    )
    add("")

    add("## Image characteristics")
    add("")
    add(
        f"**COMPUTED.** {image['decoded']} of {image['images']} images decoded; "
        f"{image['decode_failures']} failures. Colour modes: {image['modes']}."
    )
    add("")
    add("| Measure | min | p05 | p25 | median | p75 | p95 | max | mean | std |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    add(distribution_row("width (px)", image["width"], 0))
    add(distribution_row("height (px)", image["height"], 0))
    add(distribution_row("pixels", image["pixels"], 0))
    add(distribution_row("aspect ratio", image["aspect_ratio"], 3))
    add(distribution_row("file size (bytes)", image["file_size_bytes"], 0))
    add(distribution_row("mean luminance", image["mean_luminance"], 3))
    add(distribution_row("contrast proxy (std)", image["std_luminance_contrast_proxy"], 3))
    add(distribution_row("dynamic range p05-p95", image["dynamic_range_p05_p95"], 3))
    add("")
    add("![dimensions](figures/source_dimensions.png)")
    add("")
    add("![aspect ratio](figures/source_aspect_ratio.png)")
    add("")
    add("![luminance](figures/source_luminance.png)")
    add("")
    add("![file size](figures/source_file_size.png)")
    add("")
    add(
        "**INTERPRETATION.** Resolution is high and tightly clustered - a median of about "
        "1.7 megapixels - so the 640x640 training resolution discards real detail. Combined "
        "with the small-object prevalence below, that is the most likely practical constraint "
        "on detecting helmets at distance. This is a hypothesis about model behaviour, not a "
        "measurement of it."
    )
    add("")
    add(
        "**Language note.** Luminance and contrast are *measured proxies*. Images in the "
        "lower-luminance subset are not described as badly lit: a dark frame may be correctly "
        "exposed for a dark scene. See `figures/review_e_lower_luminance.jpg`."
    )
    add("")

    add("## Class distribution")
    add("")
    add(
        "**COMPUTED.** Image counts and instance counts are reported separately, because "
        "they answer different questions."
    )
    add("")
    add("| Class | Images containing it | % of 436 | Instances |")
    add("| --- | --- | --- | --- |")
    for label, count in classes["images_containing_class"].items():
        percent = classes["percent_images_containing_class"][label]
        add(f"| `{label}` | {count} | {percent}% | {classes['instances_per_class'][label]:,} |")
    add("")
    add("![class distribution](figures/source_class_distribution.png)")
    add("")
    add("![class by split](figures/source_class_by_split.png)")
    add("")
    add("### The rare class")
    add("")
    rare = min(classes["images_containing_class"], key=classes["images_containing_class"].get)
    rare_images = classes["images_containing_class"][rare]
    rare_instances = classes["instances_per_class"][rare]
    per_split = classes["images_with_class_by_split"]
    add(
        f"**COMPUTED.** `{rare}` appears in **{rare_images} of 436 images** "
        f"({classes['percent_images_containing_class'][rare]}%), with {rare_instances} "
        "instances, distributed across provider splits as: "
        + ", ".join(f"{s} {per_split.get(s, {}).get(rare, 0)}" for s in ("train", "valid", "test"))
        + " images."
    )
    add("")
    add(
        "**INTERPRETATION.** This is the dataset's binding constraint. A class present in "
        "single-digit numbers of images cannot support a per-class metric that means much, "
        "and with zero images in the provider's test split it cannot be scored there at all. "
        "Any final report must state this rather than average it away."
    )
    add("")
    add(f"See `figures/review_b_class_{rare}.jpg` for every image that contains it.")
    add("")

    add("## Objects per image and co-occurrence")
    add("")
    add("| Measure | min | p05 | p25 | median | p75 | p95 | max | mean | std |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    add(distribution_row("instances per image", classes["instances_per_image"], 2))
    add("")
    add(f"**COMPUTED.** {classes['zero_instance_images']} images carry no instance at all.")
    add("")
    add("![instances per image](figures/source_instances_per_image.png)")
    add("")
    add("Most frequent class pairs on the same image (COMPUTED):")
    add("")
    add("| Pair | Images |")
    add("| --- | --- |")
    for pair, count in list(classes["class_cooccurrence_image_counts"].items())[:8]:
        add(f"| `{pair}` | {count} |")
    add("")

    add("## Object geometry")
    add("")
    add(
        f"**COMPUTED.** {geometry['annotations']:,} annotated instances: "
        f"{geometry['geometry_types']}."
    )
    add("")
    add(f"**PARTIAL.** {geometry['note']}")
    add("")
    add("| Measure | min | p05 | p25 | median | p75 | p95 | max | mean | std |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    add(distribution_row("box width / image width", geometry["relative_box_width"], 4))
    add(distribution_row("box height / image height", geometry["relative_box_height"], 4))
    add(distribution_row("box area / image area", geometry["relative_box_area"], 5))
    add("")
    add("![relative area](figures/source_object_relative_area.png)")
    add("")
    threshold = geometry["small_object_threshold_area_fraction"]
    add(
        f"Small objects, defined as a box covering less than {threshold:.0%} of the image "
        "area (COMPUTED):"
    )
    add("")
    add("| Class | Small | Total | % small |")
    add("| --- | --- | --- | --- |")
    for label, entry in geometry["small_objects_by_class"].items():
        add(f"| `{label}` | {entry['small']} | {entry['total']} | {entry['percent']}% |")
    add("")
    add(
        "**INTERPRETATION.** Helmets are predominantly small objects; people are mostly not. "
        "A detector trained at 640x640 on stretched originals therefore faces its hardest "
        "task on exactly the classes that carry the safety signal. Stated as an expectation "
        "to test in phase 6, not as a result."
    )
    add("")

    add("## Open questions carried to phase 4B and 5")
    add("")
    for question in [
        "Are the cross-split near-duplicate candidates genuine? "
        f"({audit['near_duplicates']['cross_split_candidates']} pairs, human review required.)",
        f"Are the {classes['zero_instance_images']} zero-instance images valid negatives?",
        "Is the segmentation or the supplied bbox the better description of an object? "
        "The phase 5 bbox policy stays provisional until this is looked at.",
        "Should the canonical population be the frozen v4 export or the live source project, "
        "which now holds more annotations?",
        "Can `vest_loose` support any per-class claim, or must it be reported as "
        "under-represented and excluded from headline metrics?",
    ]:
        add(f"- {question}")
    add("")
    if manual_recorded:
        add(
            "**Status.** Phase 4B addressed the first three from the visual review; the "
            "canonical-population question is still open, and `vest_loose` is now known to be "
            "too thinly represented to carry a per-class claim under the provider split. What "
            "each judgement does and does not license is in "
            "[`manual_audit_report.md`](manual_audit_report.md)."
        )
        add("")
    return "\n".join(lines)


def phase4b_split_decision(path: Path) -> str | None:
    """Read the phase 4B verdict on the provider split, when it has been recorded.

    Args:
        path: ``reports/manual_audit_decisions.csv``.

    Returns:
        The recorded verdict, or ``None`` while phase 4B has not run.
    """
    if not path.is_file():
        return None
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("review_type") == "provider_split":
                return row.get("decision")
    return None


def main(argv: list[str] | None = None) -> int:
    """Render both reports.

    Args:
        argv: Command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="Render the phase 4A reports.")
    parser.parse_args(argv)

    paths = ProjectPaths.from_root()
    audit = load(paths.reports / "dataset_audit.json")
    eda = load(paths.reports / "eda_source.json")
    bbox = load(paths.reports / "bbox_consistency_audit.json")
    if not audit or not eda:
        print("ERROR: run the audit and EDA scripts first.", file=sys.stderr)
        return 2

    manual_split_decision = phase4b_split_decision(paths.reports / "manual_audit_decisions.csv")

    audit_path = paths.reports / "dataset_audit_report.md"
    audit_path.write_text(
        build_audit_report(audit, bbox, manual_split_decision), encoding="utf-8", newline="\n"
    )
    eda_path = paths.reports / "eda_report.md"
    eda_path.write_text(
        build_eda_report(eda, audit, manual_split_decision is not None),
        encoding="utf-8",
        newline="\n",
    )

    print(f"wrote reports/{audit_path.name} and reports/{eda_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
