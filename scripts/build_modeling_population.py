"""Construct the canonical modelling population and its indivisible split units.

Answers the phase 5B question: which images and annotations may enter a future
split, and which images must stay together when one is designed. It does not
design the split, and nothing here knows the words train, validation or test.

Every decision is derived from a committed artifact, never from a list typed into
this file: the human judgements come from `reports/manual_audit_decisions.csv`,
the annotations from the phase 5A canonical geometry, and the grouping from the
phase 4B duplicate confirmations.

Runs entirely offline. Requires:

* ``data/interim/source_geometry.jsonl``    scripts/recover_source_geometry.py
* ``reports/manual_audit_decisions.csv``    scripts/record_manual_audit.py
* ``reports/group_candidates.csv``          scripts/audit_source_dataset.py
* ``reports/fragment_rule_analysis.csv``    scripts/analyze_fragment_rule.py

Writes:
    reports/canonical_modeling_population.csv
    reports/canonical_annotation_actions.csv
    reports/group_manifest.csv
    reports/group_split_features.csv
    reports/unconfirmed_group_candidates.csv
    reports/canonical_modeling_manifest.json
    reports/canonical_modeling_population_report.md
    data/interim/modeling_annotations.jsonl   git-ignored; geometry

Usage:
    uv run python scripts/build_modeling_population.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from construction_safety_vision.data.canonical import scan_for_sensitive
from construction_safety_vision.data.population import (
    CLASS_ORDER,
    ELIGIBLE,
    EXCLUDE_IMAGE,
    EXCLUDE_WITH_IMAGE,
    EXCLUDED,
    KEEP_ANNOTATION,
    KEEP_IMAGE,
    MATERIALISE_RECTANGLE,
    NESTED_SAME_CLASS_CANDIDATE,
    PROVIDER_GEOMETRY,
    SEMANTIC_DUPLICATE,
    SINGLETON,
    SYNTHETIC_GEOMETRY,
    AnnotationRecord,
    Group,
    ImageRecord,
    PopulationError,
    build_class_map,
    build_groups,
    fingerprint_population,
    group_features,
    rectangle_polygon,
    validate_population,
)
from construction_safety_vision.data.sourcegeometry import UNSUPPORTED
from construction_safety_vision.paths import ProjectPaths
from construction_safety_vision.provenance import git_commit

GEOMETRY_FILENAME = "source_geometry.jsonl"
"""Canonical geometry from phase 5A, under ``data/interim``."""

DECISIONS_FILENAME = "manual_audit_decisions.csv"
"""Recorded human judgements from phase 4B, under ``reports``."""

CANDIDATES_FILENAME = "group_candidates.csv"
"""Perceptual near-duplicate chains from phase 4A, under ``reports``."""

FRAGMENT_FILENAME = "fragment_rule_analysis.csv"
"""Per-annotation fragment features, under ``reports``."""

SOURCE_POPULATION = 436
"""The independent source population every phase must preserve."""

OUT_OF_DOMAIN = "OUT_OF_DOMAIN"
"""Phase 4B decision marking an image outside the project's target domain."""

NESTED_ENCLOSURE = 0.95
"""Enclosure at which an annotation is described as nested in its own class."""

NESTED_AREA_RATIO = 0.01
"""Relative size below which a nested annotation is flagged for the record."""

POPULATION_CSV = "canonical_modeling_population.csv"
ACTIONS_CSV = "canonical_annotation_actions.csv"
GROUPS_CSV = "group_manifest.csv"
FEATURES_CSV = "group_split_features.csv"
UNCONFIRMED_CSV = "unconfirmed_group_candidates.csv"
MANIFEST_JSON = "canonical_modeling_manifest.json"
REPORT_MD = "canonical_modeling_population_report.md"
INTERIM_ANNOTATIONS = "modeling_annotations.jsonl"


class PopulationInputError(RuntimeError):
    """Raised when a required input artifact is missing."""


def read_csv(path: Path) -> list[dict]:
    """Read a committed CSV artifact.

    Args:
        path: File to read.

    Returns:
        The parsed rows.

    Raises:
        PopulationInputError: If the file is absent.
    """
    if not path.is_file():
        msg = f"{path.name} not found; run the earlier phase scripts first"
        raise PopulationInputError(msg)
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    """Write a CSV artifact with LF endings.

    Args:
        path: Destination file.
        columns: Column order.
        rows: Rows to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_geometry(path: Path) -> dict[str, dict]:
    """Load the canonical annotation geometry.

    Args:
        path: The interim geometry file.

    Returns:
        Records keyed by source image id.

    Raises:
        PopulationInputError: If the file is absent.
    """
    if not path.is_file():
        msg = f"{path.name} not found; run scripts/recover_source_geometry.py first"
        raise PopulationInputError(msg)
    return {
        json.loads(line)["image_id"]: json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    }


def read_decisions(rows: list[dict]) -> tuple[dict[str, dict], dict[str, list[str]]]:
    """Extract the phase 4B judgements this phase acts on.

    Args:
        rows: Rows of the manual decision record.

    Returns:
        Per-image zero-instance judgements keyed by image id, and the confirmed
        semantic duplicate groups keyed by their phase 4B group id.

    Raises:
        PopulationError: If a confirmed duplicate group is malformed.
    """
    zero_instance: dict[str, dict] = {}
    duplicates: dict[str, list[str]] = {}
    for row in rows:
        if row["review_type"] == "zero_instance_image":
            zero_instance[row["subject_id_a"]] = row
        elif row["review_type"] == "cross_split_near_duplicate":
            group_id = row["group_id"].strip()
            members = [row["subject_id_a"].strip(), row["subject_id_b"].strip()]
            if not group_id or not all(members):
                msg = f"Duplicate decision {row['decision_id']!r} is missing a group or a member"
                raise PopulationError(msg)
            duplicates.setdefault(group_id, []).extend(members)
    return zero_instance, duplicates


def build_images(geometry: dict[str, dict], zero_instance: dict[str, dict]) -> list[ImageRecord]:
    """Decide each source image's status in the modelling population.

    Args:
        geometry: Canonical geometry keyed by source image id.
        zero_instance: Phase 4B zero-instance judgements keyed by image id.

    Returns:
        One record per source image, ordered by identifier.
    """
    records: list[ImageRecord] = []
    for image_id in sorted(geometry):
        record = geometry[image_id]
        judgement = zero_instance.get(image_id)
        out_of_domain = judgement is not None and judgement["decision"] == OUT_OF_DOMAIN
        records.append(
            ImageRecord(
                source_image_id=image_id,
                name=str(record["name"]),
                width=int(record["width"]),
                height=int(record["height"]),
                status=EXCLUDED if out_of_domain else ELIGIBLE,
                action=EXCLUDE_IMAGE if out_of_domain else KEEP_IMAGE,
                reason=(
                    "OUT_OF_DOMAIN_MANUAL_REVIEW"
                    if out_of_domain
                    else (
                        "NO_OBVIOUS_MISSING_TARGET_LABEL"
                        if judgement is not None
                        else "IN_DOMAIN_ANNOTATED_SOURCE_IMAGE"
                    )
                ),
                decision_source=(
                    "PHASE_4B_HUMAN_AUDIT"
                    if judgement is not None
                    else "PHASE_5A_CANONICAL_SNAPSHOT"
                ),
                annotation_count=len(record["annotations"]),
            )
        )
    return records


def build_annotations(
    geometry: dict[str, dict],
    images: dict[str, ImageRecord],
    flagged: set[tuple[str, str]],
) -> list[AnnotationRecord]:
    """Decide what the modelling layer does with each canonical annotation.

    Args:
        geometry: Canonical geometry keyed by source image id.
        images: Image records keyed by source image id.
        flagged: Annotations a candidate fragment rule would select.

    Returns:
        One record per canonical annotation.

    Raises:
        PopulationError: If a geometry-less annotation cannot be materialised.
    """
    records: list[AnnotationRecord] = []
    for image_id in sorted(geometry):
        source = geometry[image_id]
        image = images[image_id]
        for annotation in source["annotations"]:
            annotation_id = str(annotation["annotation_id"])
            bbox = [float(v) for v in annotation["bbox"]]
            excluded_image = image.status == EXCLUDED
            unsupported = annotation["geometry_kind"] == UNSUPPORTED

            if excluded_image:
                action, reason = EXCLUDE_WITH_IMAGE, "IMAGE_EXCLUDED_FROM_MODELING_POPULATION"
                segmentation, origin, area = annotation.get("segmentation"), PROVIDER_GEOMETRY, None
            elif unsupported:
                segmentation = rectangle_polygon(bbox, width=image.width, height=image.height)
                action = MATERIALISE_RECTANGLE
                reason = "SOURCE_GEOMETRY_UNAVAILABLE"
                origin = SYNTHETIC_GEOMETRY
                area = (segmentation[0][2] - segmentation[0][0]) * (
                    segmentation[0][5] - segmentation[0][1]
                )
            else:
                action, reason = KEEP_ANNOTATION, "CANONICAL_PROVIDER_SEGMENTATION"
                segmentation = annotation.get("segmentation")
                origin, area = PROVIDER_GEOMETRY, annotation.get("area")

            records.append(
                AnnotationRecord(
                    source_image_id=image_id,
                    annotation_id=annotation_id,
                    label=str(annotation["label"]),
                    geometry_kind=str(annotation["geometry_kind"]),
                    action=action,
                    reason=reason,
                    decision_source=(
                        "PHASE_4B_HUMAN_AUDIT" if excluded_image else "PHASE_5B_CANONICALIZATION"
                    ),
                    geometry_origin=origin,
                    review_flag=(
                        NESTED_SAME_CLASS_CANDIDATE
                        if (image_id, annotation_id) in flagged and not excluded_image
                        else ""
                    ),
                    segmentation=segmentation,
                    bbox=bbox,
                    area=float(area) if area is not None else None,
                )
            )
    return records


def read_unconfirmed(rows: list[dict], confirmed: dict[str, list[str]]) -> list[dict]:
    """List perceptual near-duplicate chains no person has confirmed.

    Args:
        rows: Rows of the phase 4A candidate chains.
        confirmed: Confirmed duplicate groups keyed by phase 4B group id.

    Returns:
        One row per unconfirmed chain, carrying no split assignment.
    """
    confirmed_pairs = {frozenset(members) for members in confirmed.values()}
    unconfirmed: list[dict] = []
    for row in rows:
        members = [value for value in row["image_ids"].split("|") if value]
        if frozenset(members) in confirmed_pairs:
            continue
        unconfirmed.append(
            {
                "candidate_id": row["group_id"],
                "source_image_ids": ";".join(sorted(members)),
                "member_count": len(members),
                "evidence": row["evidence"],
                "status": "UNCONFIRMED_GROUP_CANDIDATE",
                "action": "NOT_MERGED",
                "notes": (
                    "perceptual-hash relation only; no person reviewed it, so its members "
                    "remain independent split units"
                ),
            }
        )
    return sorted(unconfirmed, key=lambda entry: entry["candidate_id"])


def summarise(
    images: list[ImageRecord],
    annotations: list[AnnotationRecord],
    groups: list[Group],
) -> dict[str, Any]:
    """Aggregate the population into the counts the report and manifest carry.

    Args:
        images: Every source image record.
        annotations: Every annotation record.
        groups: Every group.

    Returns:
        A JSON-serialisable summary.
    """
    eligible_images = [record for record in images if record.status == ELIGIBLE]
    eligible_annotations = [record for record in annotations if record.is_eligible]
    by_class: Counter[str] = Counter()
    images_with_class: dict[str, set[str]] = {name: set() for name in CLASS_ORDER}
    for record in eligible_annotations:
        by_class[record.label] += 1
        images_with_class[record.label].add(record.source_image_id)

    return {
        "source_provenance_population": len(images),
        "excluded_images": sum(1 for r in images if r.status == EXCLUDED),
        "modeling_image_population": len(eligible_images),
        "zero_instance_modeling_images": sum(1 for r in eligible_images if r.is_zero_instance),
        "canonical_annotations": len(annotations),
        "annotations_excluded_with_image": sum(
            1 for r in annotations if r.action == EXCLUDE_WITH_IMAGE
        ),
        "annotations_materialised_from_bbox": sum(
            1 for r in annotations if r.action == MATERIALISE_RECTANGLE
        ),
        "annotations_flagged_nested_same_class_candidate": sum(
            1 for r in annotations if r.review_flag == NESTED_SAME_CLASS_CANDIDATE
        ),
        "annotations_excluded_as_fragments": 0,
        "modeling_annotation_population": len(eligible_annotations),
        "annotations_by_class": {name: by_class.get(name, 0) for name in CLASS_ORDER},
        "modeling_images_by_class": {name: len(images_with_class[name]) for name in CLASS_ORDER},
        "groups_total": len(groups),
        "groups_singleton": sum(1 for g in groups if g.group_type == SINGLETON),
        "groups_semantic_duplicate": sum(1 for g in groups if g.group_type == SEMANTIC_DUPLICATE),
    }


def build_report(
    summary: dict[str, Any],
    manifest: dict[str, Any],
    images: list[ImageRecord],
    unconfirmed: list[dict],
    problems: list[str],
) -> str:
    """Compose the written population report.

    Args:
        summary: Aggregated counts.
        manifest: The machine-readable manifest.
        images: Every source image record.
        unconfirmed: Unconfirmed near-duplicate chains.
        problems: Invariant violations, empty when the population is sound.

    Returns:
        The Markdown document.
    """
    excluded = [record for record in images if record.status == EXCLUDED]
    lines = [
        "# Canonical Modelling Population",
        "",
        f"Generated: {manifest['generated']} · Phase: 5B · "
        f"Commit: `{manifest['git_commit'] or 'unknown'}`",
        "",
        "**Question.** Which images and annotations are eligible to enter a future split, and "
        "which images must stay together when one is designed?",
        "",
        "No split exists after this phase. Nothing here assigns an image to train, validation "
        "or test, and the provider's rejected split appears in no artifact a split designer "
        "will read.",
        "",
        "## Population (COMPUTED RESULT)",
        "",
        "| | Count |",
        "| --- | --- |",
        f"| Source provenance population | {summary['source_provenance_population']} |",
        f"| Excluded as out of domain | {summary['excluded_images']} |",
        f"| **Modelling image population** | **{summary['modeling_image_population']}** |",
        f"| Of those, carrying no annotation | {summary['zero_instance_modeling_images']} |",
        f"| Canonical annotations | {summary['canonical_annotations']} |",
        f"| Excluded with their image | {summary['annotations_excluded_with_image']} |",
        f"| Excluded as fragments | {summary['annotations_excluded_as_fragments']} |",
        f"| Geometry materialised from a bounding box | "
        f"{summary['annotations_materialised_from_bbox']} |",
        f"| **Modelling annotation population** | "
        f"**{summary['modeling_annotation_population']}** |",
        f"| Flagged NESTED_SAME_CLASS_CANDIDATE (not excluded) | "
        f"{summary['annotations_flagged_nested_same_class_candidate']} |",
        "",
        "### Per class (COMPUTED RESULT)",
        "",
        "| Class | Index | Retained annotations | Modelling images containing it |",
        "| --- | --- | --- | --- |",
    ]
    for name, index in manifest["class_map"].items():
        lines.append(
            f"| `{name}` | {index} | {summary['annotations_by_class'][name]} | "
            f"{summary['modeling_images_by_class'][name]} |"
        )
    lines.append("")

    lines.extend(
        [
            "## Image exclusions (MANUAL DECISION)",
            "",
            "Excluded logically. The files stay on disk, the images stay in the source "
            "provenance population of 436, and nothing was sent to the provider.",
            "",
            "| Source image | Reason | Decision source | Annotations lost |",
            "| --- | --- | --- | --- |",
        ]
    )
    for record in excluded:
        lines.append(
            f"| `{record.source_image_id}` | `{record.reason}` | `{record.decision_source}` | "
            f"{record.annotation_count} |"
        )
    lines.extend(
        [
            "",
            "The other zero-instance images phase 4B reviewed are **retained**: a person found "
            "no missing target label on them, so they are usable as negatives rather than "
            "defects. Their split allocation is a phase 5C question.",
            "",
            "## Geometry canonicalisation (CANONICALIZATION RULE)",
            "",
            f"{summary['annotations_materialised_from_bbox']} annotation(s) reach this phase "
            "with a class and a bounding box but no segmentation. Each is materialised as a "
            "four-corner rectangle clipped to the image canvas, in original source coordinates, "
            f"and recorded with geometry_origin = `{SYNTHETIC_GEOMETRY}`.",
            "",
            "This is not a claim that anyone drew that outline. Phase 5A established that the "
            "version-4 export contains the same rectangle for these records, generated by the "
            "provider's exporter from the same box, so nothing is lost and nothing is invented "
            "beyond what the export already did - the difference is that here it is labelled.",
            "",
            "## Nested same-class annotations (MANUAL DECISION)",
            "",
            "**Decision: `REJECTED_FOR_AUTOMATIC_FILTERING`. All 2031 canonical annotations "
            "are retained and no automatic filter is applied.**",
            "",
            "Phase 5B was asked to turn the phase 5A observation about the 76 annotations "
            "added since version 4 into a deterministic geometric rule. The experiment is "
            "preserved in [`fragment_rule_report.md`](fragment_rule_report.md) because a "
            "failed rule is useful evidence. It failed for a substantive reason, not a "
            "tuning one: containment inside an older same-class annotation is not evidence "
            "that an annotation is wrong. Annotations that appear as additions relative to "
            "version 4 include",
            "",
            "* legitimate **instance splits** - one oversized `person` box covering two "
            "people replaced by one box per person;",
            "* **coarse annotations replaced by several precise ones** - a single loose "
            "`vest_on_body` polygon replaced by the parts of the vest actually visible;",
            "* **geometry refinements**;",
            "* and genuinely degenerate slivers.",
            "",
            "No evaluated rule separates the first three from the fourth, so automatic "
            "exclusion would risk deleting real ground truth. The project owner therefore "
            "rejected automatic filtering.",
            "",
            f"The {summary['annotations_flagged_nested_same_class_candidate']} annotations the "
            "evaluated rule would have selected carry `action = KEEP` and "
            "`review_flag = NESTED_SAME_CLASS_CANDIDATE`. That flag is **descriptive**: it "
            "records that an annotation lies inside another of its own class. It does not "
            "assert that the annotation is wrong, and none has been established as a fragment "
            "by manual review.",
            "",
            "## Split units (COMPUTED RESULT)",
            "",
            "| | Count |",
            "| --- | --- |",
            f"| Semantic duplicate groups | {summary['groups_semantic_duplicate']} |",
            f"| Singleton groups | {summary['groups_singleton']} |",
            f"| **Total split units** | **{summary['groups_total']}** |",
            "",
            "Every eligible image belongs to exactly one group. The six semantic duplicate "
            "groups are the pairs phase 4B visually confirmed; they must not be split apart.",
            "",
            f"{len(unconfirmed)} further near-duplicate chains were raised by perceptual hashing "
            "in phase 4A and never reviewed by a person. They are **not** merged: a hash "
            "collision is not a confirmed duplicate, and merging on it would shrink the pool a "
            "split can draw from on evidence nobody checked. They are recorded in "
            "`unconfirmed_group_candidates.csv` for phase 5C to consider explicitly.",
            "",
            "## Rare class (PHASE 5C INPUT)",
            "",
            f"`vest_loose` retains {summary['annotations_by_class']['vest_loose']} instances "
            f"across {summary['modeling_images_by_class']['vest_loose']} modelling images. "
            "Whether any of those images sit in a semantic duplicate group is recorded in the "
            "manifest, because it constrains how the class can be distributed.",
            "",
            "## Fingerprint (FACT)",
            "",
            "| Component | SHA-256 |",
            "| --- | --- |",
        ]
    )
    for name, digest in manifest["fingerprints"].items():
        lines.append(f"| `{name}` | `{digest}` |")
    lines.extend(
        [
            "",
            "Computed over the semantic content only - image statuses, annotation actions, "
            "geometry digests, the class map and group membership. No timestamp or file path "
            "enters it, so re-running the pipeline on unchanged inputs reproduces it exactly.",
            "",
            "`review_flag` is deliberately **outside** the fingerprint. It describes where an "
            "annotation sits relative to its own class; it does not decide whether the "
            "annotation is in the population. These hashes were recomputed after the decision "
            "to retain all annotations and are unchanged, because that decision confirmed the "
            "population rather than altering it.",
            "",
            "## Phase 5C entry gate (PHASE 5C INPUT)",
            "",
            "`MANUAL_DISPOSITION_OF_REMAINING_NEAR_DUPLICATE_CANDIDATES` - **open**.",
            "",
            f"The {len(unconfirmed)} remaining phase 4A near-duplicate candidates have not "
            "received a complete human visual disposition. Phase 5B does not need them to "
            "complete, and they are deliberately not merged, but they **must be dispositioned "
            "before the final split is frozen**: if any turns out to be a real duplicate, a "
            "split frozen without it would leak content across a boundary.",
            "",
            "## Validation (COMPUTED RESULT)",
            "",
        ]
    )
    if problems:
        lines.append("**Invariant violations found:**")
        lines.append("")
        lines.extend(f"* {problem}" for problem in problems)
    else:
        lines.append(
            "All invariants hold: every eligible image belongs to exactly one group, no image "
            "belongs to two, every semantic duplicate group has at least two members, every "
            "retained annotation has segmentation geometry, a canonical class and a positive "
            "area, and no retained annotation is UNSUPPORTED."
        )
    lines.append("")
    # A trailing empty entry would end the file on a blank line, which git
    # reports as whitespace damage.
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Build the canonical modelling population.

    Args:
        argv: Command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)

    paths = ProjectPaths.from_root()
    try:
        geometry = load_geometry(paths.data_interim / GEOMETRY_FILENAME)
        decisions = read_csv(paths.reports / DECISIONS_FILENAME)
        candidates = read_csv(paths.reports / CANDIDATES_FILENAME)
        fragment_rows = read_csv(paths.reports / FRAGMENT_FILENAME)
    except PopulationInputError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if len(geometry) != SOURCE_POPULATION:
        print(
            f"WARNING: canonical snapshot holds {len(geometry)} source images, "
            f"expected {SOURCE_POPULATION}",
            file=sys.stderr,
        )

    flagged = {
        (row["image_id"], row["annotation_id"])
        for row in fragment_rows
        if float(row["enclosed_bbox"]) >= NESTED_ENCLOSURE
        and float(row["area_ratio"]) < NESTED_AREA_RATIO
    }

    try:
        zero_instance, confirmed = read_decisions(decisions)
        images = build_images(geometry, zero_instance)
        by_id = {record.source_image_id: record for record in images}
        annotations = build_annotations(geometry, by_id, flagged)
        eligible_ids = [r.source_image_id for r in images if r.status == ELIGIBLE]
        groups = build_groups(eligible_ids, confirmed)
    except PopulationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3

    eligible_by_image: dict[str, list[AnnotationRecord]] = {}
    for record in annotations:
        if record.is_eligible:
            eligible_by_image.setdefault(record.source_image_id, []).append(record)
    for record in images:
        record.eligible_annotation_count = len(eligible_by_image.get(record.source_image_id, []))
    for group in groups:
        for member in group.members:
            by_id[member].group_id = group.group_id

    problems = validate_population(images, annotations, groups)
    class_map = build_class_map()
    fingerprints = fingerprint_population(images, annotations, groups, class_map)
    summary = summarise(images, annotations, groups)
    unconfirmed = read_unconfirmed(candidates, confirmed)

    duplicate_members = {m for g in groups if g.group_type == SEMANTIC_DUPLICATE for m in g.members}
    vest_loose_images = sorted(
        {r.source_image_id for r in annotations if r.is_eligible and r.label == "vest_loose"}
    )
    # Measured, not asserted: compare what the canonical snapshot held for the
    # class against what survived every action this phase took.
    vest_loose_canonical = sum(1 for r in annotations if r.label == "vest_loose")
    vest_loose_retained = summary["annotations_by_class"]["vest_loose"]
    vest_loose_lost = vest_loose_canonical - vest_loose_retained
    manifest: dict[str, Any] = {
        "phase": "5B",
        "generated": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        "canonical_annotation_snapshot": "CURRENT_COMPLETE_GEOMETRY",
        "class_map": class_map,
        "class_order_is_explicit": True,
        "excluded_provider_category": "object",
        "population": summary,
        "image_exclusions": [
            {
                "source_image_id": r.source_image_id,
                "reason": r.reason,
                "decision_source": r.decision_source,
                "annotations_lost": r.annotation_count,
            }
            for r in images
            if r.status == EXCLUDED
        ],
        "synthetic_geometry": [
            {
                "source_image_id": r.source_image_id,
                "annotation_id": r.annotation_id,
                "label": r.label,
                "action": r.action,
                "reason": r.reason,
                "geometry_origin": r.geometry_origin,
            }
            for r in annotations
            if r.action == MATERIALISE_RECTANGLE
        ],
        "nested_annotation_handling": {
            "fragment_rule_status": "REJECTED_FOR_AUTOMATIC_FILTERING",
            "decision_source": "PROJECT_OWNER_REVIEW",
            "applied": False,
            "annotations_excluded": 0,
            "annotations_flagged": summary["annotations_flagged_nested_same_class_candidate"],
            "flag": NESTED_SAME_CLASS_CANDIDATE,
            "evaluated_rule": (
                f"enclosed_bbox >= {NESTED_ENCLOSURE} AND area_ratio < {NESTED_AREA_RATIO}"
            ),
            "evaluated_rule_thresholds": {
                "enclosed_bbox_min": NESTED_ENCLOSURE,
                "area_ratio_max": NESTED_AREA_RATIO,
            },
            "why_rejected": (
                "containment inside an older same-class annotation is not evidence of "
                "annotation error. Phase 5B established that annotations appearing as "
                "additions relative to version 4 include legitimate instance splits, "
                "replacements of coarse annotations by several precise ones, and geometry "
                "refinements. No evaluated rule separates those from genuine slivers, so "
                "automatic exclusion would risk deleting real ground truth. See "
                "reports/fragment_rule_report.md"
            ),
            "flag_is_descriptive_not_a_verdict": (
                "the flag records that an annotation lies inside another of its own class. It "
                "does not assert that the annotation is wrong; nothing has been established "
                "as a fragment by manual review"
            ),
        },
        "rare_class": {
            "class": "vest_loose",
            "retained_instances": summary["annotations_by_class"]["vest_loose"],
            "modeling_images": len(vest_loose_images),
            "images_in_a_semantic_duplicate_group": sorted(
                set(vest_loose_images) & duplicate_members
            ),
            "canonical_instances": vest_loose_canonical,
            "instances_lost_to_any_exclusion": vest_loose_lost,
            "affected_by_any_exclusion": vest_loose_lost != 0,
        },
        "groups": {
            "total": summary["groups_total"],
            "singleton": summary["groups_singleton"],
            "semantic_duplicate": summary["groups_semantic_duplicate"],
            "semantic_duplicate_ids": sorted(
                g.group_id for g in groups if g.group_type == SEMANTIC_DUPLICATE
            ),
            "unconfirmed_candidates": len(unconfirmed),
        },
        "phase_5b_classification": "READY_FOR_SPLIT_DESIGN",
        "phase_5c_entry_gates": [
            {
                "gate": "MANUAL_DISPOSITION_OF_REMAINING_NEAR_DUPLICATE_CANDIDATES",
                "status": "OPEN",
                "candidates": len(unconfirmed),
                "requirement": (
                    "the remaining phase 4A near-duplicate candidates have not received a "
                    "complete human visual disposition. They must be dispositioned before the "
                    "final split is frozen; they are not merged automatically"
                ),
            }
        ],
        "split_assignment_created": False,
        "holdout_frozen": False,
        "fingerprints": fingerprints,
        "fingerprint_scope": {
            "covers": [
                "eligible and excluded source image ids, with their action and reason",
                "every annotation's action, geometry origin, geometry digest and box",
                "group membership",
                "the canonical class map",
            ],
            "excludes": [
                "timestamps, file paths and run metadata",
                "review_flag, which is descriptive and asserts nothing about eligibility",
            ],
            "note": (
                "recomputed after the owner's decision to retain all annotations. It is "
                "unchanged from the previous run because that decision confirmed the "
                "population rather than altering it: no image, action, geometry, group or "
                "class index moved"
            ),
        },
        "validation_problems": problems,
        "git_commit": git_commit(),
    }

    write_csv(
        paths.reports / POPULATION_CSV,
        list(images[0].csv_row()) if images else [],
        [record.csv_row() for record in images],
    )
    write_csv(
        paths.reports / ACTIONS_CSV,
        list(annotations[0].csv_row()) if annotations else [],
        [record.csv_row() for record in annotations],
    )
    group_rows = [row for group in groups for row in group.csv_rows()]
    write_csv(paths.reports / GROUPS_CSV, list(group_rows[0]) if group_rows else [], group_rows)
    feature_rows = [group_features(group, eligible_by_image) for group in groups]
    write_csv(
        paths.reports / FEATURES_CSV,
        list(feature_rows[0]) if feature_rows else [],
        feature_rows,
    )
    write_csv(
        paths.reports / UNCONFIRMED_CSV,
        list(unconfirmed[0]) if unconfirmed else [],
        unconfirmed,
    )

    interim = paths.data_interim / INTERIM_ANNOTATIONS
    interim.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(
        json.dumps(
            {
                "source_image_id": r.source_image_id,
                "annotation_id": r.annotation_id,
                "label": r.label,
                "class_index": class_map[r.label],
                "geometry_origin": r.geometry_origin,
                "segmentation": r.segmentation,
                "bbox": r.bbox,
                "area": r.area,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        for r in annotations
        if r.is_eligible
    )
    interim.write_text(f"{body}\n" if body else "", encoding="utf-8", newline="\n")

    serialised = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
    unsafe = scan_for_sensitive(serialised)
    if unsafe:
        print(f"ERROR: manifest is not fit to commit: {'; '.join(unsafe)}", file=sys.stderr)
        return 4
    (paths.reports / MANIFEST_JSON).write_text(f"{serialised}\n", encoding="utf-8", newline="\n")
    (paths.reports / REPORT_MD).write_text(
        build_report(summary, manifest, images, unconfirmed, problems),
        encoding="utf-8",
        newline="\n",
    )

    print("--- canonical modelling population ---")
    for key in (
        "source_provenance_population",
        "excluded_images",
        "modeling_image_population",
        "zero_instance_modeling_images",
        "canonical_annotations",
        "annotations_excluded_with_image",
        "annotations_materialised_from_bbox",
        "annotations_excluded_as_fragments",
        "annotations_flagged_nested_same_class_candidate",
        "modeling_annotation_population",
        "groups_total",
        "groups_singleton",
        "groups_semantic_duplicate",
    ):
        print(f"  {key:<46} {summary[key]}")
    print(f"  annotations_by_class                           {summary['annotations_by_class']}")
    print(f"  modeling_images_by_class                       {summary['modeling_images_by_class']}")
    print(f"  unconfirmed_group_candidates                   {len(unconfirmed)}")
    print(f"\n  modeling_population_sha256 = {fingerprints['modeling_population_sha256']}")
    if problems:
        print("\nINVARIANT VIOLATIONS:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 5
    print("\n  all invariants hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
