"""Record the phase 4B human visual audit as reviewable artifacts.

Phase 4A built the evidence package and answered no semantic question. A person
then looked at the committed contact sheets and answered them. This script is
where those answers enter the repository.

The judgements are declared once, as data, at the top of this file. Everything
else - identifier resolution, duplicate-group numbering, the CSV and the report
- is derived from them and from the phase 4A manifests, so the record cannot
drift away from the evidence it refers to, and re-running the script reproduces
both outputs byte for byte.

What this script does **not** do: it makes no visual judgement of its own, reads
no dataset image, modifies no annotation, and creates no split. A recorded
``phase5_action`` is a note for a later phase, not an action taken here.

Writes:
    reports/manual_audit_decisions.csv
    reports/manual_audit_report.md

Usage:
    uv run python scripts/record_manual_audit.py
    uv run python scripts/record_manual_audit.py --check
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from construction_safety_vision.data.manualaudit import (
    ManualAuditError,
    ManualDecision,
    assign_duplicate_group_ids,
    load_review_index,
    load_source_image_index,
    read_decisions,
    resolve_short_id,
    review_pool,
    validate_decisions,
    write_decisions,
)
from construction_safety_vision.paths import ProjectPaths

RECORDED_ON = "2026-09-02"
"""Date the human judgements were transcribed into the repository. This is the
recording date, not a claim about when the reviewers looked at the sheets."""

REVIEWERS = "project owner with a technical reviewer"
"""Who made the judgements. The review took place outside this repository."""

FIGURE_DUPLICATES = "reports/figures/review_h_near_duplicates.jpg"
FIGURE_ZERO_INSTANCE = "reports/figures/review_f_zero_instance.jpg"
FIGURE_GEOMETRY = "reports/figures/review_k_bbox_vs_segmentation.jpg"
FIGURE_VEST_LOOSE = "reports/figures/review_b_class_vest_loose.jpg"
FIGURE_OVERVIEW = "reports/figures/review_a_overview.jpg"

DUPLICATE_RATIONALE = (
    "Reviewers judged the pair to depict the same source content rather than two "
    "similar independent construction scenes. Byte-level identity is not claimed: "
    "phase 4A established distinct SHA-256 hashes, so this is a semantic duplicate, "
    "not a byte duplicate."
)

OUT_OF_DOMAIN_TRANSCRIPTIONS: tuple[tuple[str, str], ...] = (
    ("9b7vLDWD", "cartoon/illustration of a worker"),
    ("KgOOOMBE", "office/laboratory-like interior"),
    ("KqRSR0F9", "yellow taxi/street scene"),
)
"""Short ids as the reviewers transcribed them, with the scene each describes.

The transcriptions are approximate by the reviewers' own statement, so they are
resolved against the seventeen images that were actually on the zero-instance
sheet and are rejected if the mapping is not unambiguous.
"""

OUT_OF_DOMAIN_RATIONALE = (
    "Reviewers judged the image to fall outside the project's target domain of real "
    "construction and PPE imagery. This is a domain-curation judgement, not a claim "
    "that an annotation is missing."
)

CARTOON_NOTE = (
    "Contains a human-like worker/PPE illustration. Deliberately NOT recorded as a "
    "valid negative: its zero annotations are not evidence that a real worker was "
    "missed, because the project targets real imagery."
)

ZERO_INSTANCE_RETAINED_RATIONALE = (
    "Reviewers found no evidence of a missing target label. Most of the seventeen "
    "zero-instance images read as genuine zero-target or context images. This is a "
    "set-level conclusion applied to each remaining member; the reviewers stated no "
    "per-image confidence and gave no finer domain classification."
)

GEOMETRY_PREFERRED_RATIONALE = (
    "On the readable high-discrepancy RLE panels, the box derived from the "
    "segmentation appeared more tightly aligned with the visible object than the "
    "supplied COCO box, which appeared systematically inflated."
)

GEOMETRY_ACCEPTABLE_RATIONALE = (
    "On the low-discrepancy and representative panels, both the supplied box and "
    "the segmentation-derived box appeared operationally acceptable."
)

GEOMETRY_UNCERTAIN_RATIONALE = (
    "Some panels on the sheet were not readable or did not render legibly. The "
    "reviewers did not identify which, so no per-annotation judgement is recorded "
    "for them, and the membership of the two set-level judgements above is stated "
    "as the panel groups the sheet was built from rather than as a per-panel "
    "certification."
)

VEST_SEMANTICS_RATIONALE = (
    "The reviewed examples generally support the reading that vest_loose is a "
    "safety vest not being worn on a person - hanging, stored or otherwise loose "
    "in the scene."
)

VEST_DIVERSITY_RATIONALE = (
    "The class is carried by a handful of source images, several instances are "
    "concentrated inside individual multi-object images, and some examples with "
    "light or white hanging garments are semantically more ambiguous than the "
    "clear high-visibility-vest examples."
)

VEST_DIVERSITY_FLAGS = (
    "RARE_CLASS|LOW_SOURCE_IMAGE_DIVERSITY|CONTEXT_SHORTCUT_RISK|SOME_SEMANTIC_AMBIGUITY"
)

DOMAIN_RATIONALE = (
    "The overview sheet shows real construction-site imagery mixed with "
    "occupational-safety imagery, posed worker portraits, stock photography, "
    "product-like PPE images, isolated PPE objects, contextual construction scenes "
    "and background scenes. The dataset is usable, but it is not a purely "
    "field-captured construction-site dataset."
)

PROVIDER_SPLIT_RATIONALE = (
    "Rejected for three reasons: (1) six semantic duplicate pairs were visually "
    "confirmed to cross split boundaries; (2) the provider test split contains zero "
    "vest_loose instances; (3) the rare class is too thinly represented for a "
    "defensible per-class final evaluation under this partition."
)

DRIFT_RATIONALE = (
    "The live source project and the frozen v4 export describe different annotation "
    "counts, and the source API exposes complete mask geometry for only part of the "
    "current annotations. Which population is canonical is unresolved and is not "
    "decided by the visual audit."
)

DRIFT_NOTE = (
    "Alternatives: (A) use v4 as the canonical frozen snapshot; (B) use the current "
    "source state despite incomplete mask geometry through the inspected API; "
    "(C) find a reproducible provider mechanism to obtain the current source state "
    "with complete instance-segmentation geometry. Preferred investigation: C before "
    "accepting A. Priority: HIGH phase 5 entry condition."
)

MICROANNOTATION_RATIONALE = (
    "Two source annotations carry no recognised representation. They must not "
    "silently enter or disappear from the model-ready dataset, so they are held open "
    "rather than deleted."
)

PHASE5_CONSTRAINTS: tuple[tuple[str, str], ...] = (
    ("P5-01", "Provider split must not be reused as final protocol."),
    (
        "P5-02",
        "Visually confirmed semantic duplicates must be grouped and cannot cross split boundaries.",
    ),
    ("P5-03", "Split design must be group-aware."),
    (
        "P5-04",
        "Split design must explicitly consider rare-class coverage, especially vest_loose.",
    ),
    (
        "P5-05",
        "The three clearly out-of-domain zero-instance images are deterministic exclusion "
        "candidates, pending canonical manifest implementation.",
    ),
    ("P5-06", "Other zero-instance images are not automatically excluded."),
    (
        "P5-07",
        "Segmentation-derived bounding boxes are the preferred detection-label policy, "
        "supported by quantitative and visual evidence, but conversion has not yet occurred.",
    ),
    ("P5-08", "Polygon and compressed-RLE segmentation must both be supported."),
    (
        "P5-09",
        "The current-source-vs-v4 annotation drift must be resolved before freezing the "
        "canonical modeling dataset.",
    ),
    ("P5-10", "The two unknown microannotations require explicit disposition."),
    (
        "P5-11",
        "Final test must contain defensible class coverage subject to grouping constraints.",
    ),
    ("P5-12", "The final holdout must remain protected after freeze."),
)

LIMITATIONS: tuple[str, ...] = (
    "The judgements are human readings of downsampled JPEG contact sheets, not "
    "measurements. They carry no error bar and are not reproducible in the sense a "
    "computed statistic is.",
    "Two reviewers looked at the same sheets together, so there is no independent "
    "second opinion and no inter-rater agreement to report.",
    "Only the sheets listed under 'evidence reviewed' produced an explicit decision. "
    "The remaining sheets in the phase 4A package were available but no judgement was "
    "supplied for them, so none is recorded.",
    "Near-duplicate detection was a perceptual-hash screen with a recall-oriented "
    "threshold. It bounds what the reviewers could confirm: a duplicate pair that both "
    "fingerprints missed was never shown to anyone.",
    "Some panels on the geometry sheet were not readable, and the reviewers did not "
    "record which, so the geometry judgements are set-level rather than per-annotation.",
    "The domain judgement is a qualitative characterisation of the image population; "
    "no domain taxonomy was defined and no image was assigned to a domain category "
    "beyond the three marked out-of-domain.",
)


def load_json(path: Path) -> dict[str, Any]:
    """Load a phase 4A JSON artifact.

    Args:
        path: File to read.

    Returns:
        The parsed payload.

    Raises:
        ManualAuditError: If the artifact is missing.
    """
    if not path.is_file():
        msg = f"required phase 4A artifact not found: {path}"
        raise ManualAuditError(msg)
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a phase 4A CSV artifact.

    Args:
        path: File to read.

    Returns:
        The rows, in file order.

    Raises:
        ManualAuditError: If the artifact is missing.
    """
    if not path.is_file():
        msg = f"required phase 4A artifact not found: {path}"
        raise ManualAuditError(msg)
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def cross_split_pairs(candidates: list[dict[str, str]]) -> list[dict[str, str]]:
    """Select the cross-split near-duplicate candidate pairs.

    Args:
        candidates: Rows of ``reports/near_duplicate_candidates.csv``.

    Returns:
        The rows whose pair crosses a provider split boundary.
    """
    return [row for row in candidates if row["cross_split"] == "True"]


def build_duplicate_decisions(
    pairs: list[dict[str, str]], review_index: dict[str, list[dict[str, str]]]
) -> list[ManualDecision]:
    """Record the confirmed cross-split semantic duplicates.

    Args:
        pairs: Cross-split candidate rows from phase 4A.
        review_index: Output of ``load_review_index``.

    Returns:
        One ``pair``-scoped decision per confirmed group, ordered by group id.

    Raises:
        ManualAuditError: If a pair member was not on the cross-split sheet, so
            the reviewers cannot have judged it there.
    """
    shown = {
        image_id
        for image_id, rows in review_index.items()
        if any(row.get("contact_sheet") == FIGURE_DUPLICATES for row in rows)
    }
    for row in pairs:
        for image_id in (row["image_a_id"], row["image_b_id"]):
            if image_id not in shown:
                msg = f"{image_id} is not on {FIGURE_DUPLICATES}; it cannot carry a verdict from it"
                raise ManualAuditError(msg)

    group_ids = assign_duplicate_group_ids(
        [(row["image_a_id"], row["image_b_id"]) for row in pairs]
    )
    by_group: dict[str, ManualDecision] = {}
    for row in pairs:
        members = tuple(sorted((row["image_a_id"], row["image_b_id"])))
        group_id = group_ids[members]
        splits = {
            row["image_a_id"]: row["provider_split_a"],
            row["image_b_id"]: row["provider_split_b"],
        }
        by_group[group_id] = ManualDecision(
            decision_id=f"dup-{group_id.removeprefix('manual_dup_')}",
            review_type="cross_split_near_duplicate",
            subject_scope="pair",
            group_id=group_id,
            subject_id_a=members[0],
            subject_id_b=members[1],
            provider_split_a=splits[members[0]],
            provider_split_b=splits[members[1]],
            decision="EXACT_SEMANTIC_DUPLICATE",
            confidence="HIGH",
            rationale=DUPLICATE_RATIONALE,
            evidence_figure=FIGURE_DUPLICATES,
            phase5_action="GROUP_TOGETHER",
            notes=(
                f"byte_duplicate=NO; semantic_duplicate=YES; "
                f"dhash={row['dhash_distance']}; phash={row['phash_distance']}"
            ),
        )
    return [by_group[group_id] for group_id in sorted(by_group)]


def build_zero_instance_decisions(
    empty_rows: list[dict[str, str]], review_index: dict[str, list[dict[str, str]]]
) -> tuple[list[ManualDecision], list[dict[str, Any]]]:
    """Record the zero-instance review.

    Args:
        empty_rows: Rows of ``reports/empty_image_audit.csv``.
        review_index: Output of ``load_review_index``.

    Returns:
        The decisions, and the resolution record of each transcribed short id.

    Raises:
        ManualAuditError: If a transcription does not resolve unambiguously, or
            resolves to an image outside the zero-instance set.
    """
    pool = review_pool(review_index, "zero_instance")
    splits = {row["image_id"]: row["split"] for row in empty_rows}
    if set(pool.values()) != set(splits):
        msg = "the zero-instance contact sheet and the zero-instance audit list disagree"
        raise ManualAuditError(msg)

    resolutions: list[dict[str, Any]] = []
    out_of_domain: dict[str, str] = {}
    for transcription, description in OUT_OF_DOMAIN_TRANSCRIPTIONS:
        match = resolve_short_id(transcription, pool)
        if match.image_id in out_of_domain:
            msg = f"two transcriptions resolved to the same image {match.image_id}"
            raise ManualAuditError(msg)
        out_of_domain[match.image_id] = description
        resolutions.append(
            {
                "transcription": match.transcription,
                "image_id": match.image_id,
                "split": splits[match.image_id],
                "description": description,
                "distance": match.distance,
                "runner_up_distance": match.runner_up_distance,
            }
        )

    decisions: list[ManualDecision] = []
    for index, image_id in enumerate(sorted(out_of_domain), start=1):
        description = out_of_domain[image_id]
        note = f"reviewer description: {description}"
        if "cartoon" in description:
            note = f"{note}. {CARTOON_NOTE}"
        decisions.append(
            ManualDecision(
                decision_id=f"zero-ood-{index:03d}",
                review_type="zero_instance_image",
                subject_scope="image",
                subject_id_a=image_id,
                provider_split_a=splits[image_id],
                decision="OUT_OF_DOMAIN",
                confidence="HIGH",
                rationale=OUT_OF_DOMAIN_RATIONALE,
                evidence_figure=FIGURE_ZERO_INSTANCE,
                phase5_action="EXCLUDE_CANDIDATE",
                notes=note,
            )
        )
    retained = sorted(set(splits) - set(out_of_domain))
    for index, image_id in enumerate(retained, start=1):
        decisions.append(
            ManualDecision(
                decision_id=f"zero-keep-{index:03d}",
                review_type="zero_instance_image",
                subject_scope="image",
                subject_id_a=image_id,
                provider_split_a=splits[image_id],
                decision="NO_OBVIOUS_MISSING_TARGET_LABEL",
                confidence="NOT_STATED",
                rationale=ZERO_INSTANCE_RETAINED_RATIONALE,
                evidence_figure=FIGURE_ZERO_INSTANCE,
                phase5_action="RETAIN_CANDIDATE",
                notes="domain relevance ranges from construction context to ambiguous "
                "industrial/safety context; no finer classification was supplied",
            )
        )
    return decisions, resolutions


def geometry_panels(review_index: dict[str, list[dict[str, str]]]) -> dict[str, list[str]]:
    """Group the geometry sheet's panels by how they were selected.

    Args:
        review_index: Output of ``load_review_index``.

    Returns:
        Mapping of ``high_discrepancy``/``representative`` to panel labels.
    """
    panels: dict[str, list[str]] = {"high_discrepancy": [], "representative": []}
    for rows in review_index.values():
        for row in rows:
            if row.get("contact_sheet") != FIGURE_GEOMETRY:
                continue
            bucket = "high_discrepancy" if row["note"].startswith("largest_") else "representative"
            panels[bucket].append(f"{row['note']}:{row['short_id']}")
    return {key: sorted(value) for key, value in panels.items()}


def build_geometry_decisions(panels: dict[str, list[str]]) -> list[ManualDecision]:
    """Record the bbox-versus-segmentation review.

    The reviewers reported that some panels were not readable but did not say
    which, so the record stays at set level: assigning a verdict to a specific
    annotation would claim a judgement nobody made.

    Args:
        panels: Output of :func:`geometry_panels`.

    Returns:
        Three set-level decisions.
    """
    return [
        ManualDecision(
            decision_id="geom-001",
            review_type="bbox_vs_segmentation",
            subject_scope="set",
            decision="SEGMENTATION_GEOMETRY_PREFERRED",
            confidence="NOT_STATED",
            rationale=GEOMETRY_PREFERRED_RATIONALE,
            evidence_figure=FIGURE_GEOMETRY,
            phase5_action="PREFER_SEGMENTATION_DERIVED_BBOX",
            notes="panel group: " + ", ".join(panels["high_discrepancy"]),
        ),
        ManualDecision(
            decision_id="geom-002",
            review_type="bbox_vs_segmentation",
            subject_scope="set",
            decision="BOTH_ACCEPTABLE",
            confidence="NOT_STATED",
            rationale=GEOMETRY_ACCEPTABLE_RATIONALE,
            evidence_figure=FIGURE_GEOMETRY,
            phase5_action="NONE",
            notes="panel group: " + ", ".join(panels["representative"]),
        ),
        ManualDecision(
            decision_id="geom-003",
            review_type="bbox_vs_segmentation",
            subject_scope="set",
            decision="UNCERTAIN",
            confidence="NOT_STATED",
            rationale=GEOMETRY_UNCERTAIN_RATIONALE,
            evidence_figure=FIGURE_GEOMETRY,
            phase5_action="NONE",
            notes="unreadable panels were not identified by the reviewers; count unknown",
        ),
    ]


def build_remaining_decisions(
    audit: dict[str, Any], microannotation_image: str, microannotation_split: str
) -> list[ManualDecision]:
    """Record the class, domain, split, drift and microannotation decisions.

    Args:
        audit: Payload of ``reports/dataset_audit.json``.
        microannotation_image: Source image carrying the unknown annotations.
        microannotation_split: Provider split of that image.

    Returns:
        The remaining decisions, in report order.
    """
    per_split = audit["provider_split_audit"]["per_split"]
    vest_images = sum(
        split.get("images_with_class", {}).get("vest_loose", 0) for split in per_split.values()
    )
    vest_instances = sum(
        split.get("instances_per_class", {}).get("vest_loose", 0) for split in per_split.values()
    )
    population = audit["population"]["source_images"]
    return [
        ManualDecision(
            decision_id="class-001",
            review_type="class_semantics",
            subject_scope="set",
            decision="VALID_OVERALL",
            confidence="NOT_STATED",
            rationale=VEST_SEMANTICS_RATIONALE,
            evidence_figure=FIGURE_VEST_LOOSE,
            phase5_action="NONE",
            notes="class: vest_loose",
        ),
        ManualDecision(
            decision_id="class-002",
            review_type="class_data_diversity",
            subject_scope="set",
            decision="VERY_LOW",
            confidence="NOT_STATED",
            rationale=VEST_DIVERSITY_RATIONALE,
            evidence_figure=FIGURE_VEST_LOOSE,
            phase5_action="ENFORCE_RARE_CLASS_COVERAGE",
            notes=(
                f"class: vest_loose; {vest_images}/{population} source images; "
                f"{vest_instances} instances; provider test coverage 0 images; "
                f"flags: {VEST_DIVERSITY_FLAGS}"
            ),
        ),
        ManualDecision(
            decision_id="domain-001",
            review_type="dataset_domain",
            subject_scope="set",
            decision="ACCEPTABLE_WITH_DOMAIN_HETEROGENEITY",
            confidence="NOT_STATED",
            rationale=DOMAIN_RATIONALE,
            evidence_figure=FIGURE_OVERVIEW,
            phase5_action="CONSTRAIN_EXTERNAL_VALIDITY_CLAIMS",
            notes=(
                "describe the dataset as heterogeneous construction/PPE imagery containing "
                "site, stock, portrait and product-style images; do not imply that a result "
                "here proves deployment performance on arbitrary construction-site video"
            ),
        ),
        ManualDecision(
            decision_id="split-001",
            review_type="provider_split",
            subject_scope="set",
            decision="UNSUITABLE_FOR_FINAL_PROTOCOL",
            confidence="HIGH",
            rationale=PROVIDER_SPLIT_RATIONALE,
            evidence_figure=f"{FIGURE_DUPLICATES}|{FIGURE_VEST_LOOSE}",
            phase5_action="REDESIGN_SPLIT",
            notes=(
                "supersedes the phase 4A classification UNDETERMINED_PENDING_VISUAL_REVIEW; "
                "the SPLIT is rejected, the DATASET is not"
            ),
        ),
        ManualDecision(
            decision_id="drift-001",
            review_type="annotation_snapshot_drift",
            subject_scope="set",
            decision="UNRESOLVED_CANONICALIZATION",
            confidence="NOT_STATED",
            rationale=DRIFT_RATIONALE,
            phase5_action="RESOLVE_BEFORE_FREEZE",
            notes=DRIFT_NOTE,
        ),
        ManualDecision(
            decision_id="geomsrc-001",
            review_type="source_annotation_geometry",
            subject_scope="image",
            subject_id_a=microannotation_image,
            provider_split_a=microannotation_split,
            decision="MANUAL_REVIEW_STILL_REQUIRED",
            confidence="NOT_STATED",
            rationale=MICROANNOTATION_RATIONALE,
            phase5_action="PHASE5_CANONICALIZATION_ITEM",
            notes=(
                "2 annotations of unknown representation on this image; both under 16 px on a "
                "side and positioned at the image edge; not shown on any contact sheet, so no "
                "visual judgement exists"
            ),
        ),
    ]


def find_microannotation_image(source_index: dict[str, dict[str, Any]]) -> tuple[str, str, int]:
    """Locate the source image carrying annotations of unknown representation.

    Args:
        source_index: Output of ``load_source_image_index``.

    Returns:
        The image id, its provider split and the number of unknown annotations.

    Raises:
        ManualAuditError: If the unknown annotations are not confined to a
            single image, which the phase 4A audit reported.
    """
    carriers = [
        (image_id, record)
        for image_id, record in source_index.items()
        if (record.get("geometry_types") or {}).get("unknown")
    ]
    if len(carriers) != 1:
        msg = f"expected exactly one image with unknown-geometry annotations, found {len(carriers)}"
        raise ManualAuditError(msg)
    image_id, record = carriers[0]
    return image_id, str(record["split"]), int(record["geometry_types"]["unknown"])


def _n(value: int | float) -> str:
    """Format a count with thousands separators, matching the phase 4A reports.

    Args:
        value: The count.

    Returns:
        The formatted number.
    """
    return f"{int(value):,}"


def _table(header: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    """Render a markdown table.

    Args:
        header: Column titles.
        rows: Row values, already stringified.

    Returns:
        The table as markdown.
    """
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def render_report(
    *,
    decisions: list[ManualDecision],
    audit: dict[str, Any],
    bbox: dict[str, Any],
    resolutions: list[dict[str, Any]],
    panels: dict[str, list[str]],
    microannotations: tuple[str, str, int],
) -> str:
    """Render ``reports/manual_audit_report.md``.

    Args:
        decisions: Every recorded decision.
        audit: Payload of ``reports/dataset_audit.json``.
        bbox: Payload of ``reports/bbox_consistency_audit.json``.
        resolutions: Short-id resolutions for the out-of-domain images.
        panels: Output of :func:`geometry_panels`.
        microannotations: Image id, split and count of unknown annotations.

    Returns:
        The complete markdown document.
    """
    population = audit["population"]
    per_split = audit["provider_split_audit"]["per_split"]
    totals = bbox["totals"]
    drift = audit["source_vs_export_drift"]["per_split"]
    drift_total = sum(item["difference"] for item in drift.values())
    vest_images = sum(
        split.get("images_with_class", {}).get("vest_loose", 0) for split in per_split.values()
    )
    vest_instances = sum(
        split.get("instances_per_class", {}).get("vest_loose", 0) for split in per_split.values()
    )
    duplicates = [item for item in decisions if item.review_type == "cross_split_near_duplicate"]
    out_of_domain = [item for item in decisions if item.decision == "OUT_OF_DOMAIN"]
    retained = [item for item in decisions if item.decision == "NO_OBVIOUS_MISSING_TARGET_LABEL"]
    micro_image, micro_split, micro_count = microannotations
    zero_total = len(out_of_domain) + len(retained)
    zero_by_split = ", ".join(
        f"{split} {count}"
        for split, count in sorted(audit["zero_instance_images"]["by_split"].items())
    )
    geometry_panel_count = len(panels["high_discrepancy"]) + len(panels["representative"])

    comparison_table = _table(
        ("", "Phase 4A", "Phase 4B"),
        [
            ("Produced by", "executed code", "people looking at figures"),
            ("Reproducible by re-running", "yes", "no"),
            (
                "Answers",
                "how many, how different, how consistent",
                "is this the same scene, is this in domain, which geometry is better",
            ),
            (
                "Recorded in",
                "`dataset_audit.json`, `bbox_consistency_audit.json`, `eda_source.json`",
                "`manual_audit_decisions.csv`",
            ),
            ("Can be wrong by", "a bug", "a misreading"),
        ],
    )
    evidence_table = _table(
        ("Contact sheet", "Question put to the reviewers", "Decision recorded"),
        [
            (
                "`review_h_near_duplicates.jpg`",
                "are these cross-split pairs the same content?",
                f"yes, {len(duplicates)} pairs",
            ),
            (
                "`review_f_zero_instance.jpg`",
                "negative sample or missing label?",
                f"yes, {zero_total} images",
            ),
            (
                "`review_k_bbox_vs_segmentation.jpg`",
                "which geometry describes the object better?",
                "yes, set level",
            ),
            (
                "`review_b_class_vest_loose.jpg`",
                "is the class semantics coherent?",
                "yes, set level",
            ),
            ("`review_a_overview.jpg`", "what kind of imagery is this?", "yes, set level"),
        ],
    )
    duplicate_table = _table(
        ("Group", "Image A", "Split A", "Image B", "Split B", "Fingerprints"),
        [
            (
                f"`{item.group_id}`",
                f"`{item.subject_id_a}`",
                item.provider_split_a,
                f"`{item.subject_id_b}`",
                item.provider_split_b,
                item.notes.split("; ", 2)[2],
            )
            for item in duplicates
        ],
    )
    resolution_table = _table(
        ("Transcribed", "Resolved image id", "Split", "Scene", "Edit distance", "Runner-up"),
        [
            (
                f"`{item['transcription']}`",
                f"`{item['image_id']}`",
                str(item["split"]),
                str(item["description"]),
                str(item["distance"]),
                str(item["runner_up_distance"]),
            )
            for item in resolutions
        ],
    )
    panel_table = _table(
        ("Panel group", "Panels", "Selected as"),
        [
            (
                "high discrepancy",
                str(len(panels["high_discrepancy"])),
                "three largest per-split deltas",
            ),
            (
                "representative",
                str(len(panels["representative"])),
                "one polygon and one RLE example per sheet split",
            ),
        ],
    )
    alternatives_table = _table(
        ("", "Option", "Cost"),
        [
            ("A", "use v4 as the canonical frozen snapshot", "accepts known-stale annotations"),
            (
                "B",
                "use the current source state",
                "incomplete mask geometry through the inspected API",
            ),
            (
                "C",
                "obtain the current source state with complete geometry, reproducibly",
                "unknown; not yet investigated",
            ),
        ],
    )

    parts: list[str] = []
    parts.append(
        f"""# Manual Visual Audit - phase 4B

Recorded: {RECORDED_ON} · Reviewers: {REVIEWERS} · Evidence commit: `{audit["git_commit"]}`

Phase 4A measured the dataset and deliberately answered no semantic question.
This document records the answers a person gave after looking at the contact
sheets phase 4A produced. Machine-readable form:
[`manual_audit_decisions.csv`](manual_audit_decisions.csv); {len(decisions)} decisions.

Every statement below is tagged:

* **AUTOMATED RESULT** - measured in phase 4A and carried forward unchanged;
* **HUMAN VISUAL JUDGEMENT** - what a person concluded from looking at a figure;
* **INTERPRETATION** - what the two together are taken to mean;
* **PHASE 5 REQUIREMENT** - a constraint handed forward, not an action taken here;
* **OPEN QUESTION** - deliberately unresolved.

> A human judgement is not a measurement. Nothing in this document has an error
> bar, a threshold or a p-value, and none of it should be quoted as if it did.

## 1. Scope and methodology

The reviewers were shown the committed phase 4A contact sheets and asked the
semantic questions the automated audit could not answer. The judgements were
made outside this repository and transcribed here; the transcription resolves
every subject to a stable identifier in the phase 4A manifests, and refuses any
identifier that does not resolve unambiguously.

This phase changed no dataset image, no annotation and no split. It created no
partition and froze no holdout. A recorded `phase5_action` is a note for a later
phase; nothing was acted on.

Regenerate and re-validate:

```bash
uv run python scripts/record_manual_audit.py
uv run python scripts/record_manual_audit.py --check
```

## 2. Phase 4A and phase 4B are different kinds of evidence

{comparison_table}

**INTERPRETATION.** The two are kept in separate files on purpose. A judgement
that migrates into a metrics file becomes indistinguishable from a measurement,
and the project's evidence is only worth what that distinction is worth.

## 3. Evidence reviewed

{evidence_table}

The package also contains `review_b` sheets for the other four classes,
`review_d_crowded.jpg`, `review_e_lower_luminance.jpg` and
`review_g_near_duplicates.jpg` (same-split pairs). The reviewers had them
available but supplied no explicit judgement for them, so **none is recorded**.
In particular, the same-split near-duplicate candidates remain unconfirmed."""
    )

    parts.append(
        f"""## 4. Semantic duplicates across split boundaries

**AUTOMATED RESULT.** Phase 4A hashed all {_n(population["source_images"])} source images
with SHA-256 and found no byte-identical pair. A separate perceptual screen
(dHash and pHash, both 64-bit) emitted {audit["near_duplicates"]["candidates"]} candidate
pairs, of which {audit["near_duplicates"]["cross_split_candidates"]} cross a provider
split boundary. A fingerprint match is a candidate, not a verdict.

**HUMAN VISUAL JUDGEMENT.** All {len(duplicates)} cross-split pairs depict effectively
the same source content rather than similar independent scenes. Confidence: HIGH.

**Byte duplicate: NO. Semantic duplicate: YES.** The pairs differ in stored
bytes - JPEG encoding, metadata, small crop or position differences, colour and
compression - while showing the same picture. Nothing here contradicts the
SHA-256 result; the two questions are different questions.

{duplicate_table}

Group ids are a function of the sorted member ids alone, so re-running the
recorder cannot renumber them.

**PHASE 5 REQUIREMENT.** Every group is `GROUP_TOGETHER`: members of one group
must land in the same split. No group is assigned to train, validation or test
here."""
    )

    parts.append(
        f"""## 5. Zero-instance images

**AUTOMATED RESULT.** {zero_total} source images carry no annotated instance
({zero_by_split}). All of them are on the sheet; none was excluded from review.

**HUMAN VISUAL JUDGEMENT.** There is **no evidence of widespread missing
annotations**. Most of the images read as genuine zero-target images, or as
domain/context images with no clearly applicable target instance. The review did
surface domain-curation noise: three images are clearly outside the project's
target domain.

{resolution_table}

The reviewers stated their short ids were approximate. Each was resolved against the
{zero_total} images actually on the zero-instance sheet, and accepted only because one
candidate fitted within two edits while the next-best fitted no closer than the
required margin. The resolution is recorded so it can be checked rather than trusted.

**Decision for the three: `OUT_OF_DOMAIN`, confidence HIGH,
`phase5_action=EXCLUDE_CANDIDATE`.** They are candidates for deterministic exclusion
from the canonical modelling population. Nothing was removed: `data/raw/` is untouched
and the source population is still {_n(population["source_images"])} images.

**Decision for the remaining {len(retained)}: `NO_OBVIOUS_MISSING_TARGET_LABEL`,
`phase5_action=RETAIN_CANDIDATE`.** Their domain relevance ranges from strong
construction context to ambiguous industrial/safety context. The reviewers gave no
finer classification, so none is invented, and no per-image confidence was stated.

## 6. The cartoon image is out of domain, not a valid negative

**HUMAN VISUAL JUDGEMENT.** The cartoon/illustration image does contain a
human-like worker and PPE. It is recorded as `OUT_OF_DOMAIN` and **not** as
`VALID_NEGATIVE`.

**INTERPRETATION.** The distinction matters and is easy to lose. Calling it a
valid negative would assert that a real worker was present and correctly left
unannotated, which would be false; the image has no real worker at all. Its zero
annotations are therefore not evidence about annotation quality. It is excluded,
if phase 5 excludes it, because the project targets real construction and PPE
imagery - not because anything was mislabelled."""
    )

    parts.append(
        f"""## 7. Supplied bounding box versus segmentation geometry

**AUTOMATED RESULT.** Of {_n(totals["annotations_total"])} v4 export annotations,
{_n(totals["by_representation"]["polygon"])} are polygons and
{_n(totals["by_representation"]["rle"])} are compressed RLE, and
{_n(totals["within_primary"])} agree with their own segmentation within 1.0 px.
Polygon agreement is essentially perfect: all
{_n(totals["within_tolerance"]["0.5"]["polygon"])} of them fall within 0.5 px. The
disagreement is concentrated in the RLE instances, where
{_n(totals["containment"]["rle|derived_inside_supplied"])} of
{_n(totals["by_representation"]["rle"])} segmentation-derived boxes sit **inside** the
supplied box - the supplied box is systematically the larger of the two.

**HUMAN VISUAL JUDGEMENT.**

* On the readable high-discrepancy RLE panels, the segmentation-derived box
  appeared more tightly aligned with the visible object than the supplied COCO
  box: `SEGMENTATION_GEOMETRY_PREFERRED`.
* On the low-discrepancy and representative panels, both representations
  appeared operationally acceptable: `BOTH_ACCEPTABLE`.
* Some panels were not readable or did not render legibly: `UNCERTAIN`.

**The record stays at set level.** The reviewers did not identify which panels were
unreadable, so no per-annotation verdict is recorded for any of the
{geometry_panel_count} panels. The groups below are how the sheet was built, not a
certification that every panel in a group was legible.

{panel_table}

## 8. Phase 5 bounding-box policy status

**Status: `PREFERRED_AND_VISUALLY_SUPPORTED`. Not `ALREADY_APPLIED`.**

The candidate policy - canonical segmentation geometry, detection boxes derived
from it - is now supported by both a quantitative and a visual argument:

1. instance segmentation is the project's canonical annotation concept;
2. polygon cases show essentially perfect geometric agreement;
3. on readable high-discrepancy RLE cases the derived box appeared tighter and
   more representative;
4. the supplied COCO box appears systematically inflated in many RLE cases.

**No annotation was modified in phase 4B, and no conversion was executed.**
Implementing and validating the conversion is phase 5's responsibility, and the
policy is not frozen until phase 5 does it."""
    )

    parts.append(
        f"""## 9. `vest_loose`

**HUMAN VISUAL JUDGEMENT - class semantics: `VALID_OVERALL`.** The examples
support the reading that `vest_loose` is a safety vest not being worn on a
person: hanging, stored, or otherwise loose in the scene.

**HUMAN VISUAL JUDGEMENT - data diversity: `VERY_LOW`.**

**AUTOMATED RESULT.** {vest_images} of {_n(population["source_images"])} source images
contain the class, carrying {vest_instances} instances:
{per_split["train"]["images_with_class"].get("vest_loose", 0)} train images
({per_split["train"]["instances_per_class"].get("vest_loose", 0)} instances),
{per_split["valid"]["images_with_class"].get("vest_loose", 0)} validation image
({per_split["valid"]["instances_per_class"].get("vest_loose", 0)} instances), and
{per_split["test"]["images_with_class"].get("vest_loose", 0)} images in the provider
test split. Several instances are concentrated inside individual multi-object images.

**HUMAN VISUAL JUDGEMENT.** Some examples - particularly scenes with light or
white hanging garments and PPE - are semantically more ambiguous than the clear
high-visibility-vest examples. They are **not** removed on that basis; the
observation is recorded, not acted on.

Recorded flags: `RARE_CLASS`, `LOW_SOURCE_IMAGE_DIVERSITY`,
`CONTEXT_SHORTCUT_RISK`, `SOME_SEMANTIC_AMBIGUITY`.

**HYPOTHESIS, not a finding.** A model trained on this may associate
`vest_loose` with racks, walls and stored-PPE context rather than learning a
robust object concept. No model exists, so this is an analysis risk to test in
phase 12, not an observed behaviour.

**PHASE 5 REQUIREMENT.** The split design must explicitly consider `vest_loose`
source-image coverage in validation and test wherever the grouping constraints
make it feasible. No numerical allocation is specified here.

## 10. What kind of dataset this is

**HUMAN VISUAL JUDGEMENT: `ACCEPTABLE_WITH_DOMAIN_HETEROGENEITY`.** The overview
sheet shows a mixture of real construction-site imagery, occupational-safety
imagery, posed worker portraits, stock photography, product-like PPE images,
isolated PPE objects, contextual construction scenes and background scenes.

**INTERPRETATION.** The dataset is usable, and the heterogeneity is not a defect
to be hidden - but it constrains the language the project may use. The correct
description is *mixed construction and PPE imagery*, or *a heterogeneous
construction/PPE dataset containing site, stock, portrait and product-style
imagery*. It is **not** a field-captured construction-site dataset.

**This limits external validity.** The final report must not imply that
performance measured on this dataset demonstrates robust deployment performance
on arbitrary real-world construction-site video. That claim would need a
different evaluation set."""
    )

    parts.append(
        f"""## 11. Provider split: `UNSUITABLE_FOR_FINAL_PROTOCOL`

Phase 4A classified the provider split `UNDETERMINED_PENDING_VISUAL_REVIEW`,
because the automated evidence could not settle whether the cross-split
candidates were genuine. The visual review settles it.

**Classification: `UNSUITABLE_FOR_FINAL_PROTOCOL`** - a human-audited
methodological decision, on three grounds:

1. **{len(duplicates)} visually confirmed semantic duplicate pairs cross split
   boundaries.** Training on one member and evaluating on the other measures
   memorisation, not generalisation.
2. **The provider test split contains zero `vest_loose` instances.** A class
   that is absent cannot be scored, so no per-class test metric exists for it
   under this partition.
3. **The rare class is too thinly represented for a defensible per-class final
   evaluation**: {vest_images} source images in total, one of them in validation.

> **The SPLIT is rejected. The DATASET is not.** The dataset remains suitable
> for the project, subject to phase 5 canonicalisation and split redesign. This
> distinction is the whole point of the section.

## 12. Current source project versus frozen v4: `UNRESOLVED_CANONICALIZATION`

**AUTOMATED RESULT.** The live source project holds {_n(drift_total)} more annotations
than the v4 source-equivalent snapshot (train +{int(drift["train"]["difference"])},
valid +{int(drift["valid"]["difference"])}, test +{int(drift["test"]["difference"])}),
so the source project was edited after version 4 was generated.

**AUTOMATED RESULT.** The two populations also differ in what geometry they expose.
The current source API provides polygon vertices for
{_n(audit["source_geometry_types"]["polygon"])} annotations, while
{_n(audit["source_geometry_types"]["mask"])} mask-type records provide a bounding box
but no complete mask geometry, and {audit["source_geometry_types"]["unknown"]} records
carry no recognised representation. The v4 COCO export has complete segmentation
geometry for both polygon and RLE, but describes the older annotation snapshot.

**OPEN QUESTION.** Which is canonical. The alternatives:

{alternatives_table}

**PHASE 5 REQUIREMENT (HIGH).** Investigate C before accepting A. This is an
entry condition for phase 5, not a phase 4B deliverable, and phase 4B does not
resolve it.

## 13. Two annotations of unknown representation

**AUTOMATED RESULT.** {micro_count} source annotations carry no recognised
representation. Both sit on the same image (`{micro_image}`, provider split
`{micro_split}`), both measure under 16 px on a side, and both are positioned at the
image edge.

**Status: `MANUAL_REVIEW_STILL_REQUIRED` and `PHASE5_CANONICALIZATION_ITEM`.**
They were not on any contact sheet, so no visual judgement exists for them. They
are **not** deleted. Phase 5 must give them an explicit disposition: they must
not silently enter the model-ready dataset, and they must not silently vanish
from it."""
    )

    parts.append(
        """## 14. Phase 5 entry constraints

Recorded, not implemented. No split ratio is specified, and nothing below has
been acted on.

"""
        + _table(("Id", "Constraint"), [(f"**{key}**", text) for key, text in PHASE5_CONSTRAINTS])
    )

    parts.append(
        """## 15. Limitations of this review

"""
        + "\n".join(f"* {item}" for item in LIMITATIONS)
        + """

**INTERPRETATION.** These limitations do not undermine the decisions recorded
here, but they do fix what the decisions can carry. The duplicate groups are
strong enough to constrain a split; the geometry judgement is strong enough to
support a policy phase 5 must still validate; the domain judgement is strong
enough to constrain the report's language and nothing more.

## 16. Phase 4B outcome

| | Status |
| --- | --- |
| Phase 4B | `COMPLETE` |
| Dataset | `ACCEPTED_WITH_DOCUMENTED_LIMITATIONS` |
| Provider split | `UNSUITABLE_FOR_FINAL_PROTOCOL` |
| Canonical split | not created |
| Holdout | not frozen; still protected |
| Models | none trained; no metric exists |
| Next | phase 5 - canonical dataset, group-aware split and freeze |
"""
    )
    return "\n\n".join(parts).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    """Record or validate the phase 4B manual audit.

    Args:
        argv: Command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit status.
    """
    parser = argparse.ArgumentParser(description="Record the phase 4B manual visual audit.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate the committed artifacts instead of rewriting them",
    )
    args = parser.parse_args(argv)

    paths = ProjectPaths.from_root()
    reports = paths.reports
    decisions_path = reports / "manual_audit_decisions.csv"
    report_path = reports / "manual_audit_report.md"

    try:
        source_index = load_source_image_index(reports / "source_image_manifest.jsonl")
        review_index = load_review_index(reports / "manual_review_manifest.csv")
        audit = load_json(reports / "dataset_audit.json")
        bbox = load_json(reports / "bbox_consistency_audit.json")
        pairs = cross_split_pairs(read_csv_rows(reports / "near_duplicate_candidates.csv"))
        empty_rows = read_csv_rows(reports / "empty_image_audit.csv")

        expected_pairs = audit["near_duplicates"]["cross_split_candidates"]
        if len(pairs) != expected_pairs:
            msg = f"expected {expected_pairs} cross-split pairs, found {len(pairs)}"
            raise ManualAuditError(msg)

        microannotations = find_microannotation_image(source_index)
        panels = geometry_panels(review_index)
        zero_decisions, resolutions = build_zero_instance_decisions(empty_rows, review_index)
        decisions = [
            *build_duplicate_decisions(pairs, review_index),
            *zero_decisions,
            *build_geometry_decisions(panels),
            *build_remaining_decisions(audit, microannotations[0], microannotations[1]),
        ]
    except ManualAuditError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    if not args.check:
        write_decisions(decisions_path, decisions)
        report_path.write_text(
            render_report(
                decisions=decisions,
                audit=audit,
                bbox=bbox,
                resolutions=resolutions,
                panels=panels,
                microannotations=microannotations,
            ),
            encoding="utf-8",
            newline="\n",
        )
        print(f"wrote {decisions_path.relative_to(paths.root)} ({len(decisions)} decisions)")
        print(f"wrote {report_path.relative_to(paths.root)}")

    try:
        rows = read_decisions(decisions_path)
    except ManualAuditError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    problems = validate_decisions(
        rows, source_index=source_index, review_index=review_index, root=paths.root
    )
    if not report_path.is_file():
        problems.append(f"{report_path.name} is missing")
    if problems:
        print(f"FAILED: {len(problems)} problem(s) in the manual audit record", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print(f"OK: {len(rows)} decisions validated against the phase 4A manifests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
