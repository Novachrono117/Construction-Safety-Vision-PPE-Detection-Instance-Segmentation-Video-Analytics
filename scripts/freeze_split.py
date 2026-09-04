"""Freeze the human-selected split candidate as the project's authoritative split.

Phase 5C.2. This is the step after which the train/validation/test membership
stops being a suggestion and becomes the protocol. It does exactly three things:
it re-verifies that the selected candidate still reproduces, it writes the
canonical membership with fingerprints that move when membership moves, and it
locks the holdout.

It is deliberately narrow. It selects nothing - the selection is a human decision
recorded in ``configs/split_freeze.yaml`` - and it re-runs no search. It copies
no image, writes no label, resizes nothing and reads no pixel: phase 5D
materialises the detection and segmentation views from this membership. Nothing
here reads the provider's rejected split.

Verification comes before writing, and every check is a stop rather than a
warning. If ``candidate_001`` no longer reproduces its recorded fingerprint, or
the modelling population underneath it has moved, the freeze fails and writes
nothing rather than freezing a different assignment that happens to be feasible.

Runs entirely offline. Requires:

* ``configs/split_freeze.yaml``
* ``configs/split_search.yaml``                        phase 5C.1 protocol
* ``reports/split_candidates/candidate_001.csv``       scripts/optimize_split_candidates.py
* ``reports/split_candidates/summary.csv``             scripts/optimize_split_candidates.py
* ``reports/group_manifest.csv``                       scripts/build_modeling_population.py
* ``reports/group_split_features.csv``                 scripts/build_modeling_population.py
* ``reports/canonical_modeling_population.csv``        scripts/build_modeling_population.py
* ``reports/canonical_annotation_actions.csv``         scripts/build_modeling_population.py
* ``reports/canonical_modeling_manifest.json``         scripts/build_modeling_population.py

Writes:
    reports/split_manifest.json
    reports/final_split_assignments.csv
    reports/split_freeze_report.md
    reports/split_candidates/selection.csv
    reports/split_freeze.provenance.json
    reports/split_candidate_report.md  (selection status only)

Usage:
    uv run python scripts/freeze_split.py
    uv run python scripts/freeze_split.py --verify-only
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from construction_safety_vision.config import ConfigError
from construction_safety_vision.data.canonical import scan_for_sensitive
from construction_safety_vision.data.population import ELIGIBLE, KEEP_ANNOTATION
from construction_safety_vision.data.population import (
    MATERIALISE_RECTANGLE as MATERIALISE,
)
from construction_safety_vision.data.split_freeze import (
    FROZEN,
    MANIFEST_SPLITS,
    NON_SELECTED,
    SELECTED,
    TEST,
    SplitAssignment,
    SplitFreezeConfig,
    SplitManifestError,
    assignment_rows,
    fingerprint_holdout,
    fingerprint_split_assignment,
    load_split_freeze_config,
    normalise_assignments,
    split_sections,
    stamp_selection_status,
    validate_manifest,
)
from construction_safety_vision.data.split_optimization import (
    fingerprint_assignment,
    load_split_search_config,
)
from construction_safety_vision.paths import ProjectPaths
from construction_safety_vision.provenance import ProvenanceRecord, git_commit
from construction_safety_vision.splits import (
    HOLDOUT_UNLOCK_ENV_VAR,
    HOLDOUT_UNLOCK_VALUE,
)

CANDIDATE_DIR = "split_candidates"
MANIFEST_JSON = "split_manifest.json"
ASSIGNMENTS_CSV = "final_split_assignments.csv"
SELECTION_CSV = "selection.csv"
FREEZE_REPORT_MD = "split_freeze_report.md"
CANDIDATE_REPORT_MD = "split_candidate_report.md"
POPULATION_CSV = "canonical_modeling_population.csv"
ACTIONS_CSV = "canonical_annotation_actions.csv"
GROUPS_CSV = "group_manifest.csv"
FEATURES_CSV = "group_split_features.csv"
POPULATION_MANIFEST_JSON = "canonical_modeling_manifest.json"
PROVENANCE_JSON = "split_freeze.provenance.json"

ELIGIBLE_ANNOTATION_ACTIONS = frozenset({KEEP_ANNOTATION, MATERIALISE})
"""Annotation actions that put an annotation in the modelling population."""


class FreezeInputError(RuntimeError):
    """Raised when a required input artifact is missing or unusable."""


class FreezeVerificationError(RuntimeError):
    """Raised when the selected candidate does not reproduce as recorded."""


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a committed CSV artifact.

    Args:
        path: File to read.

    Returns:
        The parsed rows.

    Raises:
        FreezeInputError: If the file is absent.
    """
    if not path.is_file():
        msg = f"{path.name} not found; run the phase that produces it first"
        raise FreezeInputError(msg)
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
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


def read_json(path: Path) -> dict[str, Any]:
    """Read a committed JSON artifact.

    Args:
        path: File to read.

    Returns:
        The parsed mapping.

    Raises:
        FreezeInputError: If the file is absent or unparsable.
    """
    if not path.is_file():
        msg = f"{path.name} not found; run the phase that produces it first"
        raise FreezeInputError(msg)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"{path.name} is not valid JSON ({exc})"
        raise FreezeInputError(msg) from exc


class Inputs:
    """Every committed artifact the freeze reads, parsed once.

    Attributes:
        candidate: Split name keyed by group id, from the selected candidate.
        summary: The phase 5C.1 candidate summary rows.
        members: Member image ids keyed by group id.
        eligible: Eligible source image ids.
        excluded: Excluded source image ids, with the reason.
        annotations: Class instance counts keyed by source image id.
        annotation_total: Eligible annotations across the whole population.
        population: The phase 5B population manifest.
        classes: Canonical class order.
        feature_group_ids: Group ids in the optimiser's feature table.
    """

    def __init__(
        self, paths: ProjectPaths, config: SplitFreezeConfig, classes: tuple[str, ...]
    ) -> None:
        """Load and parse every input.

        Args:
            paths: Project layout.
            config: The freeze protocol.
            classes: Canonical class order, from the search protocol.

        Raises:
            FreezeInputError: If an artifact is missing or malformed.
        """
        reports = paths.reports
        candidate_path = reports / CANDIDATE_DIR / f"{config.selected_candidate}.csv"
        candidate_rows = read_csv(candidate_path)
        if not candidate_rows:
            msg = f"{candidate_path.name} is empty"
            raise FreezeInputError(msg)
        if "provisional_split" not in candidate_rows[0]:
            msg = (
                f"{candidate_path.name} has no 'provisional_split' column; it is not a "
                "phase 5C.1 candidate file"
            )
            raise FreezeInputError(msg)
        self.candidate: dict[str, str] = {
            row["group_id"]: row["provisional_split"] for row in candidate_rows
        }
        if len(self.candidate) != len(candidate_rows):
            msg = f"{candidate_path.name} assigns a group more than once"
            raise FreezeInputError(msg)

        self.summary = read_csv(reports / CANDIDATE_DIR / "summary.csv")

        self.members: dict[str, list[str]] = {}
        for row in read_csv(reports / GROUPS_CSV):
            self.members.setdefault(row["group_id"], []).append(row["source_image_id"])

        self.eligible: set[str] = set()
        self.excluded: dict[str, str] = {}
        for row in read_csv(reports / POPULATION_CSV):
            if row["status"] == ELIGIBLE:
                self.eligible.add(row["source_image_id"])
            else:
                self.excluded[row["source_image_id"]] = row["reason"]

        self.annotations: dict[str, dict[str, int]] = {}
        self.annotation_total = 0
        for row in read_csv(reports / ACTIONS_CSV):
            if row["action"] not in ELIGIBLE_ANNOTATION_ACTIONS:
                continue
            self.annotation_total += 1
            counts = self.annotations.setdefault(row["source_image_id"], {})
            counts[row["label"]] = counts.get(row["label"], 0) + 1

        self.population = read_json(reports / POPULATION_MANIFEST_JSON)
        self.classes = classes
        self.feature_group_ids = {row["group_id"] for row in read_csv(reports / FEATURES_CSV)}


def build_assignments(inputs: Inputs) -> tuple[SplitAssignment, ...]:
    """Expand the group-level candidate to the canonical image-level assignment.

    Args:
        inputs: The parsed inputs.

    Returns:
        One assignment per modelling image, deterministically ordered.

    Raises:
        FreezeVerificationError: If a group has no recorded membership.
    """
    rows: list[SplitAssignment] = []
    for group_id, split in inputs.candidate.items():
        members = inputs.members.get(group_id)
        if not members:
            msg = f"group {group_id!r} is assigned a split but has no members in {GROUPS_CSV}"
            raise FreezeVerificationError(msg)
        rows.extend(
            SplitAssignment(group_id=group_id, source_image_id=member, split=split)
            for member in members
        )
    return normalise_assignments(rows)


def totals(assignments: tuple[SplitAssignment, ...], inputs: Inputs) -> dict[str, dict[str, Any]]:
    """Aggregate the frozen assignment into the statistics the report quotes.

    Args:
        assignments: The frozen assignment.
        inputs: The parsed inputs.

    Returns:
        Per-split totals keyed by split name.
    """
    result: dict[str, dict[str, Any]] = {}
    for split in MANIFEST_SPLITS:
        rows = [row for row in assignments if row.split == split]
        images = [row.source_image_id for row in rows]
        group_sizes: dict[str, int] = {}
        for row in rows:
            group_sizes[row.group_id] = group_sizes.get(row.group_id, 0) + 1
        images_with = dict.fromkeys(inputs.classes, 0)
        instances = dict.fromkeys(inputs.classes, 0)
        negatives = 0
        annotations = 0
        for image_id in images:
            counts = inputs.annotations.get(image_id, {})
            if not counts:
                negatives += 1
            for name, value in counts.items():
                if name in instances:
                    instances[name] += value
                    images_with[name] += 1
                annotations += value
        result[split] = {
            "images": len(images),
            "groups": len(group_sizes),
            "non_singleton_groups": sum(1 for size in group_sizes.values() if size > 1),
            "negative_images": negatives,
            "annotations": annotations,
            "images_with": images_with,
            "instances": instances,
        }
    return result


def verify(
    assignments: tuple[SplitAssignment, ...],
    inputs: Inputs,
    config: SplitFreezeConfig,
    stats: dict[str, dict[str, Any]],
) -> list[str]:
    """Re-verify the selected candidate against every declared expectation.

    Args:
        assignments: The frozen assignment.
        inputs: The parsed inputs.
        config: The freeze protocol.
        stats: Per-split totals.

    Returns:
        One description per failed check, empty when the candidate reproduces.
    """
    problems: list[str] = []
    problems.extend(_verify_candidate_identity(inputs, config))
    problems.extend(_verify_coverage(assignments, inputs))
    problems.extend(_verify_group_integrity(assignments, inputs))
    problems.extend(_verify_counts(config, stats))
    problems.extend(_verify_classes(inputs, config, stats))
    return problems


def _verify_candidate_identity(inputs: Inputs, config: SplitFreezeConfig) -> list[str]:
    """Check that the candidate on disk is the one that was reviewed.

    Args:
        inputs: The parsed inputs.
        config: The freeze protocol.

    Returns:
        One description per failed check.
    """
    problems: list[str] = []
    recomputed = fingerprint_assignment(inputs.candidate)
    if recomputed != config.expected_candidate_assignment_sha256:
        problems.append(
            f"{config.selected_candidate} no longer reproduces: its assignment digest is "
            f"{recomputed}, the reviewed candidate was "
            f"{config.expected_candidate_assignment_sha256}"
        )
    rows = [row for row in inputs.summary if row["candidate_id"] == config.selected_candidate]
    if not rows:
        problems.append(f"{config.selected_candidate} does not appear in the phase 5C.1 summary")
    elif rows[0]["candidate_assignment_sha256"] != recomputed:
        problems.append(
            f"the phase 5C.1 summary records {rows[0]['candidate_assignment_sha256']} for "
            f"{config.selected_candidate} but the candidate file digests to {recomputed}"
        )

    fingerprints = inputs.population.get("fingerprints", {})
    for name, expected in (
        ("modeling_population_sha256", config.expected_modeling_population_sha256),
        ("groups_sha256", config.expected_groups_sha256),
    ):
        found = fingerprints.get(name)
        if found != expected:
            problems.append(
                f"{POPULATION_MANIFEST_JSON} records {name} {found!r}, the freeze protocol "
                f"expects {expected!r}"
            )
    return problems


def _verify_coverage(assignments: tuple[SplitAssignment, ...], inputs: Inputs) -> list[str]:
    """Check that every modelling image and annotation is placed exactly once.

    Args:
        assignments: The frozen assignment.
        inputs: The parsed inputs.

    Returns:
        One description per failed check.
    """
    problems: list[str] = []
    assigned: dict[str, str] = {}
    for row in assignments:
        if row.source_image_id in assigned:
            problems.append(
                f"image {row.source_image_id!r} is assigned to "
                f"{assigned[row.source_image_id]!r} and {row.split!r}"
            )
        assigned[row.source_image_id] = row.split

    missing = sorted(inputs.eligible - set(assigned))
    if missing:
        problems.append(f"{len(missing)} eligible image(s) unassigned, e.g. {missing[:3]}")
    unknown = sorted(set(assigned) - inputs.eligible)
    if unknown:
        problems.append(f"{len(unknown)} assigned image(s) are not eligible, e.g. {unknown[:3]}")
    leaked = sorted(set(assigned) & set(inputs.excluded))
    if leaked:
        problems.append(f"{len(leaked)} excluded out-of-domain image(s) appear in the split")

    placed = sum(sum(inputs.annotations.get(image_id, {}).values()) for image_id in assigned)
    if placed != inputs.annotation_total:
        problems.append(
            f"{placed} annotation(s) land in a split but the modelling population holds "
            f"{inputs.annotation_total}"
        )
    orphaned = sorted(set(inputs.annotations) - set(assigned))
    if orphaned:
        problems.append(
            f"{len(orphaned)} image(s) carry annotations but belong to no split, "
            f"e.g. {orphaned[:3]}"
        )
    return problems


def _verify_group_integrity(assignments: tuple[SplitAssignment, ...], inputs: Inputs) -> list[str]:
    """Check that no indivisible group was separated.

    Args:
        assignments: The frozen assignment.
        inputs: The parsed inputs.

    Returns:
        One description per failed check.
    """
    problems: list[str] = []
    splits_of: dict[str, set[str]] = {}
    seen_members: dict[str, set[str]] = {}
    for row in assignments:
        splits_of.setdefault(row.group_id, set()).add(row.split)
        seen_members.setdefault(row.group_id, set()).add(row.source_image_id)
    for group_id, splits in sorted(splits_of.items()):
        if len(splits) > 1:
            problems.append(f"group {group_id!r} is split across {sorted(splits)}")
    for group_id, members in sorted(seen_members.items()):
        declared = set(inputs.members.get(group_id, []))
        if members != declared:
            problems.append(
                f"group {group_id!r} carries {sorted(members)} but {GROUPS_CSV} declares "
                f"{sorted(declared)}"
            )
    if set(splits_of) != inputs.feature_group_ids:
        problems.append(
            f"the frozen split covers {len(splits_of)} group(s) but {FEATURES_CSV} declares "
            f"{len(inputs.feature_group_ids)}"
        )
    if set(splits_of) != set(inputs.candidate):
        problems.append("the expanded assignment does not cover the candidate's groups exactly")
    return problems


def _verify_counts(config: SplitFreezeConfig, stats: dict[str, dict[str, Any]]) -> list[str]:
    """Check the realised counts against the declared expectations.

    Args:
        config: The freeze protocol.
        stats: Per-split totals.

    Returns:
        One description per failed check.
    """
    problems: list[str] = []
    expectations = (
        ("images", config.expected_image_counts, "images"),
        ("groups", config.expected_group_counts, "groups"),
        ("negative images", config.expected_negative_images, "negative_images"),
    )
    for label, expected, key in expectations:
        for split in MANIFEST_SPLITS:
            found = stats[split][key]
            if found != expected[split]:
                problems.append(
                    f"{split} holds {found} {label}, the freeze protocol expects {expected[split]}"
                )
    return problems


def _verify_classes(
    inputs: Inputs, config: SplitFreezeConfig, stats: dict[str, dict[str, Any]]
) -> list[str]:
    """Check class coverage and the rare class's declared allocation.

    Args:
        inputs: The parsed inputs.
        config: The freeze protocol.
        stats: Per-split totals.

    Returns:
        One description per failed check.
    """
    problems: list[str] = []
    for split in MANIFEST_SPLITS:
        for name in inputs.classes:
            if stats[split]["images_with"][name] <= 0:
                problems.append(f"{split} holds no image carrying {name}")
            if stats[split]["instances"][name] <= 0:
                problems.append(f"{split} holds no {name} instance")
    rare = config.rare_class
    for split in MANIFEST_SPLITS:
        found_images = stats[split]["images_with"][rare]
        if found_images != config.expected_rare_class_images[split]:
            problems.append(
                f"{split} holds {found_images} {rare} image(s), the freeze protocol expects "
                f"{config.expected_rare_class_images[split]}"
            )
        found_instances = stats[split]["instances"][rare]
        if found_instances != config.expected_rare_class_instances[split]:
            problems.append(
                f"{split} holds {found_instances} {rare} instance(s), the freeze protocol "
                f"expects {config.expected_rare_class_instances[split]}"
            )
    return problems


def build_manifest(
    assignments: tuple[SplitAssignment, ...],
    inputs: Inputs,
    config: SplitFreezeConfig,
    stats: dict[str, dict[str, Any]],
    target_ratios: dict[str, float],
    seed: int,
) -> dict[str, Any]:
    """Assemble the authoritative split manifest.

    No timestamp enters the file. Every field is a function of the frozen
    membership and the committed protocol, so two runs over the same inputs
    produce a byte-identical manifest and any difference is a real difference.

    Args:
        assignments: The frozen assignment.
        inputs: The parsed inputs.
        config: The freeze protocol.
        stats: Per-split totals.
        target_ratios: Target share of images per split, from the search protocol.
        seed: The project seed the candidate search ran under.

    Returns:
        The manifest mapping.
    """
    sections = split_sections(assignments)
    split_sha = fingerprint_split_assignment(assignments)
    holdout_sha = fingerprint_holdout(
        assignments, modeling_population_sha256=config.expected_modeling_population_sha256
    )
    manifest: dict[str, Any] = {
        "schema_version": config.manifest_schema_version,
        "phase": "5C.2",
        "status": FROZEN,
        "created_from_candidate": config.selected_candidate,
        "selection_status": SELECTED,
        "selection_method": config.selection_method,
        "selection_decision_source": config.selection_decision_source,
        "algorithmic_best_candidate": config.algorithmic_best_candidate,
        "final_selected_candidate": config.selected_candidate,
        "non_selected_candidates": sorted(
            row["candidate_id"]
            for row in inputs.summary
            if row["candidate_id"] != config.selected_candidate
        ),
        "seed": seed,
        "target_ratios": target_ratios,
        "actual_image_counts": {split: stats[split]["images"] for split in MANIFEST_SPLITS},
        "actual_group_counts": {split: stats[split]["groups"] for split in MANIFEST_SPLITS},
        "actual_non_singleton_group_counts": {
            split: stats[split]["non_singleton_groups"] for split in MANIFEST_SPLITS
        },
        "actual_negative_image_counts": {
            split: stats[split]["negative_images"] for split in MANIFEST_SPLITS
        },
        "actual_annotation_counts": {
            split: stats[split]["annotations"] for split in MANIFEST_SPLITS
        },
        "images_with_class": {
            split: dict(stats[split]["images_with"]) for split in MANIFEST_SPLITS
        },
        "instances_by_class": {split: dict(stats[split]["instances"]) for split in MANIFEST_SPLITS},
        "modeling_population_sha256": config.expected_modeling_population_sha256,
        "groups_sha256": config.expected_groups_sha256,
        "candidate_assignment_sha256": config.expected_candidate_assignment_sha256,
        "split_assignment_sha256": split_sha,
        "holdout_sha256": holdout_sha,
        "fingerprint_scope": {
            "candidate_assignment_sha256": (
                "phase 5C.1 digest over the sorted (group_id, split) pairs. It is the identity "
                "of the reviewed candidate and is preserved unchanged"
            ),
            "split_assignment_sha256": (
                "phase 5C.2 digest over the sorted (group_id, source_image_id, split) triples. "
                "It differs from candidate_assignment_sha256 because the canonical final "
                "representation names images as well as groups, so it also moves when a "
                "group's membership changes while its split does not"
            ),
            "holdout_sha256": (
                "digest over the holdout's group ids, its source image ids and "
                "modeling_population_sha256. It detects membership drift in the protected "
                "split; it covers no label, no metric and no model output"
            ),
            "excludes": [
                "timestamps",
                "machine-specific filesystem paths",
                "the provider's rejected split",
                "labels, geometry and any model result",
            ],
        },
        "split_semantics": {
            "train": "may be used to fit model parameters",
            "validation": (
                "may be used for model selection, hyperparameter selection, threshold and "
                "configuration decisions, and early stopping where a later protocol permits it"
            ),
            "test": (
                "final untouched holdout. After this freeze it must not be used for model "
                "selection, architecture selection, hyperparameter tuning, augmentation "
                "tuning, image-size tuning, threshold tuning, qualitative model debugging or "
                "error-driven iteration. It may be evaluated only in the final-evaluation "
                "phase, once, after the models are frozen"
            ),
        },
        "holdout_guard": config.holdout_guard.as_dict(),
        "holdout_frozen": True,
        "physical_materialisation": {
            "performed": False,
            "note": (
                "this phase freezes membership only. No image was copied, moved, resized or "
                "converted, and no label file was written. Phase 5D materialises the "
                "detection and segmentation views from this membership"
            ),
        },
        "provider_split_used": False,
        "excluded_images": sorted(inputs.excluded),
        "git_commit": git_commit(),
    }
    manifest.update({split: sections[split] for split in MANIFEST_SPLITS})
    return manifest


def _percentage(part: int, whole: int) -> str:
    """Format a share as a percentage string.

    Args:
        part: Numerator.
        whole: Denominator.

    Returns:
        The formatted share, or ``"-"`` when the denominator is zero.
    """
    if whole <= 0:
        return "-"
    return f"{100.0 * part / whole:.1f}%"


def build_report(
    manifest: dict[str, Any],
    inputs: Inputs,
    config: SplitFreezeConfig,
    stats: dict[str, dict[str, Any]],
    candidate_row: dict[str, str],
) -> str:
    """Write the split freeze report.

    Args:
        manifest: The manifest that was written.
        inputs: The parsed inputs.
        config: The freeze protocol.
        stats: Per-split totals.
        candidate_row: The phase 5C.1 summary row of the selected candidate.

    Returns:
        The report as Markdown.
    """
    lines: list[str] = []
    add = lines.append
    commit = manifest["git_commit"] or "unavailable"
    total_images = sum(stats[s]["images"] for s in MANIFEST_SPLITS)
    total_annotations = sum(stats[s]["annotations"] for s in MANIFEST_SPLITS)

    add("# Split Freeze - Phase 5C.2")
    add("")
    add(f"Phase: 5C.2 · Commit: `{commit}` · Status: **{manifest['status']}**")
    add("")
    add(
        "**The split is frozen.** `candidate_001` is now the authoritative "
        "train/validation/test membership of this project. The `test` split is a locked "
        "holdout from this point: it has not been evaluated, inspected, or used to make any "
        "decision, and it may be read only in the final-evaluation phase after both models "
        "are frozen."
    )
    add("")
    add(
        "Claims below are labelled `FACT` (measured from a committed artifact), "
        "`COMPUTED RESULT` (produced by this run), `HUMAN PROTOCOL DECISION` (a choice a "
        "person made) and `LIMITATION` (something this split cannot support)."
    )
    add("")
    add("## 1. Source and modelling population")
    add("")
    add(
        "`FACT` The dataset is Roboflow Universe `agis-workspace-8gs52/"
        "construction-ppe-compliance-detection` v4, CC BY 4.0, with the canonical annotation "
        "state recovered from the live source project in phase 5A "
        "(`CURRENT_COMPLETE_GEOMETRY`)."
    )
    add("")
    add(
        f"`FACT` The provenance population is **436 source images**. The modelling population "
        f"is **{total_images} images** - 436 minus the 3 out-of-domain exclusions confirmed by "
        "the phase 4B human audit. The two populations are different and must not be "
        "conflated. Excluded images stay on disk and stay in the provenance population; "
        "exclusion is logical, never physical."
    )
    add("")
    add(
        f"`FACT` The modelling population carries **{total_annotations} annotations**, all "
        "retained: 0 excluded as fragments, 2 materialised as four-corner rectangles from the "
        "provider's box and labelled `SYNTHETIC_FROM_PROVIDER_BBOX`."
    )
    add("")
    add(f"Modelling population SHA-256: `{manifest['modeling_population_sha256']}`")
    add("")
    add("## 2. Group construction")
    add("")
    add(
        "`FACT` The unit a split assigns is the **group**, never the image. Phase 4B confirmed "
        "6 semantic-duplicate relations and phase 5B.1 confirmed 5 more, giving **11 confirmed "
        "groups over 22 images** plus **411 singletons** = **422 indivisible split units**."
    )
    add("")
    add(
        "`FACT` Two distinct findings both make a group indivisible, and they are recorded "
        "separately in `group_manifest.csv`: `EXACT_SEMANTIC_DUPLICATE` (10 groups - the same "
        "frame stored twice) and `NEAR_DUPLICATE_SAME_SCENE` (1 group, `manual_dup_008` - the "
        "same worker and scene at a different moment). The second is not an exact duplicate "
        "and is not separable either."
    )
    add("")
    add(
        "`FACT` Groups are connected components of the confirmed relations, so a chain "
        "A~B, B~C forms one group rather than two overlapping pairs. All 11 phase 4A "
        "near-duplicate candidates carry a human disposition; none was merged on perceptual "
        "distance alone."
    )
    add("")
    add(f"Groups SHA-256: `{manifest['groups_sha256']}`")
    add("")
    add("## 3. Why the provider's split was not reused")
    add("")
    add(
        "`HUMAN PROTOCOL DECISION` The provider's split was classified "
        "`UNSUITABLE_FOR_FINAL_PROTOCOL` in phase 4B. It is not an input, an initialisation "
        "or a target anywhere in the search or the freeze. `group_split_features.csv` omits "
        "it deliberately and `optimize_split_candidates.py` refuses to run if such a column "
        "is present."
    )
    add("")
    add("## 4. Optimisation protocol (phase 5C.1)")
    add("")
    add(
        f"`FACT` Deterministic multi-start local search over the 422 groups, seeded from the "
        f"project seed **{manifest['seed']}**. Hard constraints are rejections, not penalties. "
        "The soft objective is normalised per class and averaged, so a 914-instance class "
        "cannot outweigh a 45-instance one, and each component is reported separately. The "
        "protocol is `configs/split_search.yaml` and is unchanged by this phase."
    )
    add("")
    add("## 5. Candidates generated")
    add("")
    add("| Candidate | Images | `vest_loose` images | `vest_loose` instances | Total objective |")
    add("| --- | --- | --- | --- | --- |")
    for row in inputs.summary:
        sizes = "/".join(row[f"{s}_images"] for s in MANIFEST_SPLITS)
        vest_images = "/".join(row[f"{s}_vest_loose_images"] for s in MANIFEST_SPLITS)
        vest_instances = "/".join(row[f"{s}_vest_loose_instances"] for s in MANIFEST_SPLITS)
        marker = " **(selected)**" if row["candidate_id"] == config.selected_candidate else ""
        add(
            f"| `{row['candidate_id']}`{marker} | {sizes} | {vest_images} | "
            f"{vest_instances} | {row['total_objective']} |"
        )
    add("")
    add(
        f"`HUMAN PROTOCOL DECISION` The candidates other than "
        f"`{config.selected_candidate}` are preserved unchanged and marked `{NON_SELECTED}` in "
        "`reports/split_candidates/selection.csv`. Their scores are historical measurements "
        "and were not rewritten."
    )
    add("")
    add("## 6. Selection")
    add("")
    add(f"`HUMAN PROTOCOL DECISION` `final_selected_candidate` = **`{config.selected_candidate}`**")
    add("")
    add(f"* `selection_status`: `{manifest['selection_status']}`")
    add(f"* `selection_method`: `{config.selection_method}`")
    add(f"* `decision_source`: `{config.selection_decision_source}`")
    add(f"* `algorithmic_best_candidate`: `{config.algorithmic_best_candidate}`")
    add("")
    add(
        "The human selection coincides with the algorithmic best candidate. That does **not** "
        "make it an automatic selection: the candidates were predeclared and deterministically "
        "generated, and the choice among them was made after reviewing the rare-class and "
        "evaluation trade-offs described below. A coincidence of outcome is not a substitute "
        "for the review step, and the review step is recorded here because it happened."
    )
    add("")
    add("Rationale, as reviewed:")
    add("")
    add(
        f"* preserves 5 of the 8 `{config.rare_class}` source images in training, keeping "
        "training-time diversity of the rare class as high as any feasible candidate does;"
    )
    add(
        f"* reserves 2 independent `{config.rare_class}` source images for the final holdout, "
        "the minimum at which a per-class claim about the holdout is meaningful at all;"
    )
    add("* satisfies every hard class, group and negative constraint;")
    add(f"* matches the 303/65/65 target exactly (`size_error` = {candidate_row['size_error']});")
    add(
        f"* has the best predeclared deterministic objective among feasible candidates "
        f"(total {candidate_row['total_objective']});"
    )
    alternative = next((r for r in inputs.summary if r["candidate_id"] == "candidate_006"), None)
    if alternative is not None:
        add(
            "* materially outperforms the 4/2/2 family on both global image-level and "
            "instance-level balance - `candidate_006` scores "
            f"`E_img` {alternative['image_class_balance_error']} and `E_inst` "
            f"{alternative['instance_class_balance_error']}, against "
            f"{candidate_row['image_class_balance_error']} and "
            f"{candidate_row['instance_class_balance_error']};"
        )
    add(
        f"* avoids reducing `{config.rare_class}` training-image diversity from 5 images to 4 "
        "solely to obtain a second validation image."
    )
    add("")
    add("## 7. Selected candidate metrics (phase 5C.1 objective)")
    add("")
    add("| Component | Value |")
    add("| --- | --- |")
    add(f"| `E_img` image class balance | {candidate_row['image_class_balance_error']} |")
    add(f"| `E_inst` instance class balance | {candidate_row['instance_class_balance_error']} |")
    add(f"| `E_neg` negative balance | {candidate_row['negative_balance_error']} |")
    add(f"| `E_size` size | {candidate_row['size_error']} |")
    add(f"| **total** | **{candidate_row['total_objective']}** |")
    add("")
    add(
        "`COMPUTED RESULT` These are the values measured in phase 5C.1 and carried over "
        "unchanged; this phase re-verified the assignment they describe rather than "
        "re-scoring it."
    )
    add("")
    add("## 8. Final image and group counts")
    add("")
    add("| Split | Images | Share | Groups | Non-singleton groups | Negatives | Annotations |")
    add("| --- | --- | --- | --- | --- | --- | --- |")
    for split in MANIFEST_SPLITS:
        row = stats[split]
        add(
            f"| {split} | {row['images']} | {_percentage(row['images'], total_images)} | "
            f"{row['groups']} | {row['non_singleton_groups']} | {row['negative_images']} | "
            f"{row['annotations']} |"
        )
    add(
        f"| **total** | **{total_images}** | 100.0% | "
        f"**{sum(stats[s]['groups'] for s in MANIFEST_SPLITS)}** | "
        f"**{sum(stats[s]['non_singleton_groups'] for s in MANIFEST_SPLITS)}** | "
        f"**{sum(stats[s]['negative_images'] for s in MANIFEST_SPLITS)}** | "
        f"**{total_annotations}** |"
    )
    add("")
    add(
        f"`COMPUTED RESULT` {total_images} images and {total_annotations} annotations, matching "
        "the phase 5B modelling population exactly. Every eligible image is assigned once; no "
        "excluded out-of-domain image appears."
    )
    add("")
    add("## 9. Per-class image distribution")
    add("")
    add("| Class | train | validation | test | total |")
    add("| --- | --- | --- | --- | --- |")
    for name in inputs.classes:
        values = [stats[s]["images_with"][name] for s in MANIFEST_SPLITS]
        add(f"| `{name}` | {values[0]} | {values[1]} | {values[2]} | {sum(values)} |")
    add("")
    add("## 10. Per-class instance distribution")
    add("")
    add("| Class | train | validation | test | total |")
    add("| --- | --- | --- | --- | --- |")
    for name in inputs.classes:
        values = [stats[s]["instances"][name] for s in MANIFEST_SPLITS]
        add(f"| `{name}` | {values[0]} | {values[1]} | {values[2]} | {sum(values)} |")
    add("")
    add(
        "`COMPUTED RESULT` All five classes appear in all three splits, at image level and at "
        "instance level."
    )
    add("")
    lines.extend(_report_tail(manifest, inputs, config, stats))
    return "\n".join(lines) + "\n"


def _report_tail(
    manifest: dict[str, Any],
    inputs: Inputs,
    config: SplitFreezeConfig,
    stats: dict[str, dict[str, Any]],
) -> list[str]:
    """Write the report sections from the rare class onward.

    Args:
        manifest: The manifest that was written.
        inputs: The parsed inputs.
        config: The freeze protocol.
        stats: Per-split totals.

    Returns:
        The remaining report lines.
    """
    lines: list[str] = []
    add = lines.append
    rare = config.rare_class
    rare_images = [stats[s]["images_with"][rare] for s in MANIFEST_SPLITS]
    rare_instances = [stats[s]["instances"][rare] for s in MANIFEST_SPLITS]
    negatives = [stats[s]["negative_images"] for s in MANIFEST_SPLITS]

    add(f"## 11. The `{rare}` limitation")
    add("")
    add(
        f"`FACT` `{rare}` occurs in only **{sum(rare_images)} modelling source images**, "
        "spanning **7 indivisible groups** - two of the eight images sit in `manual_dup_010`, "
        "so the class moves in chunks and cannot be freely rebalanced."
    )
    add("")
    add(
        f"`COMPUTED RESULT` The frozen split allocates `{rare}` "
        f"{rare_images[0]}/{rare_images[1]}/{rare_images[2]} images and "
        f"{rare_instances[0]}/{rare_instances[1]}/{rare_instances[2]} instances."
    )
    add("")
    add(
        f"`LIMITATION` **Validation holds exactly one `{rare}` source image.** Validation "
        f"metrics specific to `{rare}` therefore carry high sampling uncertainty and **must "
        "not be used in isolation** for model or hyperparameter selection. Model selection "
        "relies primarily on predeclared global and macro validation criteria; a `"
        f"{rare}`-specific validation result is interpreted cautiously and supported by "
        "qualitative analysis."
    )
    add("")
    add(
        f"`LIMITATION` **The holdout holds two independent `{rare}` source images.** Even the "
        f"final per-class metrics for `{rare}` must be reported with an explicit small-sample "
        "limitation attached."
    )
    add("")
    add(
        "`LIMITATION` This split is **not perfectly stratified** and must not be described as "
        "such. It is a **group-aware and class-aware constrained split**: group indivisibility "
        "and the rare-class floors are hard constraints, and proportionality is a scored "
        "preference that the group structure sometimes makes unreachable."
    )
    add("")
    add("## 12. Negative images")
    add("")
    add(
        f"`COMPUTED RESULT` The 14 deliberate negatives - images the phase 4B audit confirmed "
        "as annotation-free rather than unlabelled - are allocated "
        f"{negatives[0]}/{negatives[1]}/{negatives[2]}, so none of the three splits is free of "
        "them."
    )
    add("")
    add("## 13. Group-integrity checks")
    add("")
    add("`COMPUTED RESULT` Every check below was executed by this run and passed:")
    add("")
    add(f"* `{config.selected_candidate}` reproduces its recorded assignment digest;")
    add("* the phase 5C.1 summary agrees with the candidate file;")
    add("* all 422 canonical groups are assigned exactly once;")
    add(f"* all {sum(stats[s]['images'] for s in MANIFEST_SPLITS)} images are assigned once;")
    add("* every group's membership matches `group_manifest.csv` exactly;")
    add("* no group appears in more than one split, so no confirmed duplicate or same-scene")
    add("  pair crosses a boundary;")
    add("* all five classes appear in every split at image and instance level;")
    add(f"* `{rare}` is allocated {rare_images[0]}/{rare_images[1]}/{rare_images[2]} images and")
    add(f"  {rare_instances[0]}/{rare_instances[1]}/{rare_instances[2]} instances;")
    add(f"* negatives are allocated {negatives[0]}/{negatives[1]}/{negatives[2]};")
    add("* no excluded out-of-domain image appears in any split;")
    add(
        f"* all {sum(stats[s]['annotations'] for s in MANIFEST_SPLITS)} annotations belong to "
        "an eligible image in exactly one split."
    )
    add("")
    add("## 14. Fingerprints")
    add("")
    add("| Fingerprint | Value |")
    add("| --- | --- |")
    add(f"| `modeling_population_sha256` | `{manifest['modeling_population_sha256']}` |")
    add(f"| `groups_sha256` | `{manifest['groups_sha256']}` |")
    add(f"| `candidate_assignment_sha256` | `{manifest['candidate_assignment_sha256']}` |")
    add(f"| `split_assignment_sha256` | `{manifest['split_assignment_sha256']}` |")
    add(f"| `holdout_sha256` | `{manifest['holdout_sha256']}` |")
    add("")
    add(
        "`FACT` `candidate_assignment_sha256` and `split_assignment_sha256` differ, and are "
        "expected to. The first is the phase 5C.1 digest over sorted `(group_id, split)` "
        "pairs - the identity of the reviewed candidate, preserved unchanged. The second is "
        "the phase 5C.2 digest over sorted `(group_id, source_image_id, split)` triples, "
        "because the canonical final representation names images as well as groups. The "
        "consequence is that the freeze fingerprint also moves if a group's membership "
        "changes while its split assignment does not - which is exactly the drift the "
        "candidate digest cannot see."
    )
    add("")
    add(
        "`FACT` `holdout_sha256` covers the holdout's group ids, its source image ids and "
        "`modeling_population_sha256`. It carries no label, no metric and no model output: it "
        "exists to detect membership drift in the protected split, not to record results."
    )
    add("")
    add("## 15. Holdout access policy")
    add("")
    add(
        "`HUMAN PROTOCOL DECISION` `test` is a locked holdout. Reading it requires **two "
        "independent opt-ins**, and neither alone is sufficient:"
    )
    add("")
    add("1. an in-code `allow_test=True` passed to `assert_split_allowed`; **and**")
    add(f"2. `{HOLDOUT_UNLOCK_ENV_VAR}={HOLDOUT_UNLOCK_VALUE}` in the environment.")
    add("")
    add(
        "`FACT` The default state is `LOCKED`. The environment variable is **not** set in this "
        "repository, and `construction_safety_vision.data.split_freeze.FrozenSplits` routes "
        "every `test` request through the same guard, so the frozen holdout ids cannot be "
        "obtained through the data access layer without both opt-ins. `train` and `validation` "
        "need no override."
    )
    add("")
    add("`HUMAN PROTOCOL DECISION` From this freeze onward, `test` must not be used for:")
    add("")
    add(
        "model selection, architecture selection, hyperparameter tuning, augmentation tuning, "
        "image-size tuning, threshold tuning, qualitative model debugging, or error-driven "
        "iteration. It may be evaluated once, in the final-evaluation phase, after both models "
        "are frozen."
    )
    add("")
    add(
        "`FACT` As of this report the holdout has never been evaluated, inspected or plotted, "
        "and no model exists to evaluate on it."
    )
    add("")
    add("## 16. Limitations")
    add("")
    add(
        "* `LIMITATION` **Independence is screened, not proven.** The groups rest on two "
        "perceptual fingerprints plus human visual review of every candidate they raised. "
        "That screens for duplicated frames and one same-scene pair; it does not prove that "
        "no two images in different splits share a site, a day, a camera or a worker. No such "
        "metadata exists in this dataset, so no stronger claim is made."
    )
    add(
        f"* `LIMITATION` **`{rare}` is thin whatever the split.** 8 images and 45 instances "
        "cannot support a confident per-class conclusion. This is a property of the dataset, "
        "not of the split."
    )
    add(
        "* `LIMITATION` **Balanced counts are not balanced difficulty.** The objective counts "
        "images and instances; nothing here measures how hard those images are, so comparable "
        "counts do not guarantee comparable difficulty across splits."
    )
    add(
        "* `LIMITATION` **The search was reproducible, not exhaustive.** A better assignment "
        "may exist; the space was not brute-forced."
    )
    add(
        "* `LIMITATION` **Instance-level proportionality is partly unreachable.** Objects "
        "co-occur inside images and images are bound into groups, so some deviation is "
        "structural rather than a search failure."
    )
    add(
        "* `LIMITATION` **No model exists.** No metric, no evaluation and no inference has "
        "been produced, on any split."
    )
    add("")
    add("## 17. What this phase did not do")
    add("")
    add(
        "No model was trained. No YOLO or COCO dataset was generated. No image was copied, "
        "moved, resized or preprocessed, and no label file was written - `data/processed/` is "
        f"untouched. No inference was run. The holdout was not evaluated or inspected. "
        f"{HOLDOUT_UNLOCK_ENV_VAR} was not set. Phase 5D materialises the detection and "
        "segmentation views from this membership."
    )
    return lines


def _write_outputs(
    paths: ProjectPaths,
    manifest: dict[str, Any],
    assignments: tuple[SplitAssignment, ...],
    inputs: Inputs,
    config: SplitFreezeConfig,
    stats: dict[str, dict[str, Any]],
    candidate_row: dict[str, str],
) -> list[Path]:
    """Write every artifact the freeze produces.

    Args:
        paths: Project layout.
        manifest: The manifest to write.
        assignments: The frozen assignment.
        inputs: The parsed inputs.
        config: The freeze protocol.
        stats: Per-split totals.
        candidate_row: The phase 5C.1 summary row of the selected candidate.

    Returns:
        The files written, in a deterministic order.

    Raises:
        FreezeVerificationError: If an artifact is not fit to commit.
    """
    reports = paths.reports
    serialised = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
    report = build_report(manifest, inputs, config, stats, candidate_row)
    for name, text in ((MANIFEST_JSON, serialised), (FREEZE_REPORT_MD, report)):
        unsafe = scan_for_sensitive(text)
        if unsafe:
            msg = f"{name} is not fit to commit: {'; '.join(unsafe)}"
            raise FreezeVerificationError(msg)

    manifest_path = reports / MANIFEST_JSON
    manifest_path.write_text(f"{serialised}\n", encoding="utf-8", newline="\n")

    assignments_path = reports / ASSIGNMENTS_CSV
    write_csv(
        assignments_path,
        ["group_id", "source_image_id", "split"],
        assignment_rows(assignments),
    )

    selection_path = reports / CANDIDATE_DIR / SELECTION_CSV
    write_csv(
        selection_path,
        ["candidate_id", "selection_status", "selected_in"],
        [
            {
                "candidate_id": row["candidate_id"],
                "selection_status": (
                    SELECTED if row["candidate_id"] == config.selected_candidate else NON_SELECTED
                ),
                "selected_in": ("5C.2" if row["candidate_id"] == config.selected_candidate else ""),
            }
            for row in sorted(inputs.summary, key=lambda r: r["candidate_id"])
        ],
    )

    report_path = reports / FREEZE_REPORT_MD
    report_path.write_text(report, encoding="utf-8", newline="\n")

    candidate_report_path = reports / CANDIDATE_REPORT_MD
    stamped = stamp_selection_status(
        candidate_report_path.read_text(encoding="utf-8"),
        selected=config.selected_candidate,
        selected_in="phase 5C.2",
    )
    candidate_report_path.write_text(stamped, encoding="utf-8", newline="\n")

    return [
        manifest_path,
        assignments_path,
        report_path,
        selection_path,
        candidate_report_path,
    ]


def _write_provenance(
    paths: ProjectPaths,
    manifest: dict[str, Any],
    config: SplitFreezeConfig,
    outputs: list[Path],
) -> Path:
    """Record how the freeze was produced.

    Args:
        paths: Project layout.
        manifest: The manifest that was written.
        config: The freeze protocol.
        outputs: The files the freeze produced.

    Returns:
        The provenance record path.
    """
    record = ProvenanceRecord.create(
        "split_freeze",
        phase=5,
        repo_root=paths.root,
        config={
            "split_freeze": "configs/split_freeze.yaml",
            "selected_candidate": config.selected_candidate,
            "selection_method": config.selection_method,
            "decision_source": config.selection_decision_source,
        },
        details={
            "phase": "5C.2",
            "status": manifest["status"],
            "split_assignment_sha256": manifest["split_assignment_sha256"],
            "holdout_sha256": manifest["holdout_sha256"],
            "candidate_assignment_sha256": manifest["candidate_assignment_sha256"],
            "modeling_population_sha256": manifest["modeling_population_sha256"],
            "groups_sha256": manifest["groups_sha256"],
            "actual_image_counts": manifest["actual_image_counts"],
            "holdout_guard": manifest["holdout_guard"],
            "holdout_evaluated": False,
            "images_materialised": False,
            "models_trained": 0,
        },
    )
    for name in (
        f"{CANDIDATE_DIR}/{config.selected_candidate}.csv",
        f"{CANDIDATE_DIR}/summary.csv",
        GROUPS_CSV,
        FEATURES_CSV,
        POPULATION_CSV,
        ACTIONS_CSV,
        POPULATION_MANIFEST_JSON,
    ):
        record.add_input(paths.reports / name, relative_to=paths.root)
    record.add_input(paths.configs / "split_freeze.yaml", relative_to=paths.root)
    record.add_input(paths.configs / "split_search.yaml", relative_to=paths.root)
    for path in outputs:
        record.add_output(path, relative_to=paths.root)
    destination = paths.reports / PROVENANCE_JSON
    record.write_json(destination)
    return destination


def main(argv: list[str] | None = None) -> int:
    """Verify and freeze the selected split candidate.

    Args:
        argv: Command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="Split-freeze configuration.")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Run every verification and write nothing.",
    )
    args = parser.parse_args(argv)

    paths = ProjectPaths.from_root()
    try:
        config = load_split_freeze_config(args.config or (paths.configs / "split_freeze.yaml"))
        search = load_split_search_config(paths.configs / "split_search.yaml")
        inputs = Inputs(paths, config, search.classes)
    except (ConfigError, FreezeInputError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    try:
        assignments = build_assignments(inputs)
    except (FreezeVerificationError, SplitManifestError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3

    stats = totals(assignments, inputs)
    problems = verify(assignments, inputs, config, stats)
    if problems:
        print(
            f"INVALID_CANDIDATE: {config.selected_candidate} does not reproduce. "
            "Nothing was written.",
            file=sys.stderr,
        )
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 4

    candidate_rows = [
        row for row in inputs.summary if row["candidate_id"] == config.selected_candidate
    ]
    manifest = build_manifest(
        assignments, inputs, config, stats, dict(search.target_ratios), search.seed
    )
    manifest_problems = validate_manifest(manifest)
    if manifest_problems:
        print("ERROR: the assembled manifest is not internally consistent:", file=sys.stderr)
        for problem in manifest_problems:
            print(f"  - {problem}", file=sys.stderr)
        return 5

    print(f"selected candidate:          {config.selected_candidate}")
    print(f"candidate_assignment_sha256: {manifest['candidate_assignment_sha256']}")
    print(f"split_assignment_sha256:     {manifest['split_assignment_sha256']}")
    print(f"holdout_sha256:              {manifest['holdout_sha256']}")
    for split in MANIFEST_SPLITS:
        row = stats[split]
        print(
            f"  {split:<11} images {row['images']:>3}  groups {row['groups']:>3}  "
            f"negatives {row['negative_images']:>2}  annotations {row['annotations']:>4}  "
            f"{config.rare_class} {row['images_with'][config.rare_class]}"
            f"/{row['instances'][config.rare_class]}"
        )
    print("all verifications passed")

    if args.verify_only:
        print("--verify-only: nothing written")
        return 0

    try:
        outputs = _write_outputs(
            paths, manifest, assignments, inputs, config, stats, candidate_rows[0]
        )
    except (FreezeVerificationError, SplitManifestError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 6
    provenance_path = _write_provenance(paths, manifest, config, outputs)

    for path in [*outputs, provenance_path]:
        print(f"wrote {path.relative_to(paths.root).as_posix()}")
    print(f"\nSPLIT_FROZEN   holdout ({TEST}): LOCKED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
