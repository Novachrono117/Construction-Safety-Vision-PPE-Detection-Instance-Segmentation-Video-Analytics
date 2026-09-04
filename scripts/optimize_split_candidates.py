"""Search for candidate train/validation/test splits over the canonical groups.

Phase 5C.1 generates and compares **provisional** split assignments. It selects
nothing, freezes nothing, and creates no holdout. The split that later becomes
authoritative will be chosen by a person from the candidates this produces.

The unit assigned is the group, never the image, so no confirmed duplicate pair
can be separated. The provider's split is not read: it was rejected in phase 4B
and appears nowhere in the objective, the initialisation or the ranking.

Runs entirely offline. Requires:

* ``reports/group_split_features.csv``            scripts/build_modeling_population.py
* ``reports/canonical_modeling_population.csv``   scripts/build_modeling_population.py
* ``configs/split_search.yaml``

Writes:
    reports/split_candidates/summary.csv
    reports/split_candidates/candidate_XXX.csv
    reports/split_candidate_report.md

Usage:
    uv run python scripts/optimize_split_candidates.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from construction_safety_vision.config import ConfigError
from construction_safety_vision.data.split_optimization import (
    SPLITS,
    GroupFeature,
    InfeasibleSplitError,
    SplitCandidate,
    SplitSearchConfig,
    hard_violations,
    load_split_search_config,
    search_candidates,
)
from construction_safety_vision.paths import ProjectPaths
from construction_safety_vision.provenance import git_commit

FEATURES_CSV = "group_split_features.csv"
"""Canonical group metadata, the only optimisation input."""

POPULATION_CSV = "canonical_modeling_population.csv"
"""Per-image population table, used for post-hoc diagnostics only."""

MANIFEST_CSV = "group_manifest.csv"
"""Group membership, used to expand a candidate to image level for diagnostics."""

STATS_JSONL = "source_image_stats.jsonl"
"""Phase 4A image measurements, under ``data/interim``. Diagnostics only."""

CANDIDATE_DIR = "split_candidates"
"""Directory holding the candidate assignments, under ``reports``."""

REPORT_MD = "split_candidate_report.md"
"""Committed candidate comparison."""

SUMMARY_COLUMNS = (
    "candidate_id",
    "rank",
    "train_images",
    "validation_images",
    "test_images",
    "train_groups",
    "validation_groups",
    "test_groups",
    "train_negatives",
    "validation_negatives",
    "test_negatives",
    "train_vest_loose_images",
    "validation_vest_loose_images",
    "test_vest_loose_images",
    "train_vest_loose_instances",
    "validation_vest_loose_instances",
    "test_vest_loose_instances",
    "image_class_balance_error",
    "instance_class_balance_error",
    "negative_balance_error",
    "size_error",
    "total_objective",
    "candidate_assignment_sha256",
    "search_origin_start",
    "vest_loose_family",
    "family_best",
    "distribution_warning",
)
"""Column order of the candidate summary."""

DIAGNOSTIC_FIELDS = ("width", "height", "aspect_ratio", "luminance_mean", "contrast")
"""Source characteristics checked post hoc for a catastrophic shift."""

WARNING_RATIO = 0.25
"""Relative gap between split means that raises a distribution warning."""


class SplitInputError(RuntimeError):
    """Raised when a required input artifact is missing."""


def read_csv(path: Path) -> list[dict]:
    """Read a committed CSV artifact.

    Args:
        path: File to read.

    Returns:
        The parsed rows.

    Raises:
        SplitInputError: If the file is absent.
    """
    if not path.is_file():
        msg = f"{path.name} not found; run scripts/build_modeling_population.py first"
        raise SplitInputError(msg)
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


def load_groups(rows: list[dict], config: SplitSearchConfig) -> list[GroupFeature]:
    """Build the optimisation input from the canonical group features.

    The provider's split is not among the columns read, and would be refused if
    it were: this function names every field it consumes.

    Args:
        rows: Rows of the group feature table.
        config: The search protocol.

    Returns:
        One feature record per group, ordered by group id.

    Raises:
        SplitInputError: If the table carries a provider split column.
    """
    forbidden = {"split", "provider_split", "original_split"} & set(rows[0])
    if forbidden:
        msg = (
            f"{FEATURES_CSV} carries provider split column(s) {sorted(forbidden)}. The provider "
            "split is UNSUITABLE_FOR_FINAL_PROTOCOL and must not reach the optimiser."
        )
        raise SplitInputError(msg)
    return sorted(
        (
            GroupFeature(
                group_id=row["group_id"],
                image_count=int(row["image_count"]),
                negative_images=int(row["zero_instance_image_count"]),
                images_with={name: int(row[f"images_with_{name}"]) for name in config.classes},
                instances={name: int(row[f"instances_{name}"]) for name in config.classes},
                group_type=row["group_type"],
            )
            for row in rows
        ),
        key=lambda group: group.group_id,
    )


def load_image_stats(paths: ProjectPaths) -> dict[str, dict[str, float]]:
    """Load the phase 4A per-image measurements for post-hoc diagnostics.

    Args:
        paths: Project layout.

    Returns:
        Measurements keyed by source image id, empty when unavailable.
    """
    path = paths.data_interim / STATS_JSONL
    if not path.is_file():
        return {}
    stats: dict[str, dict[str, float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        record = json.loads(line)
        stats[record["image_id"]] = record
    return stats


def diagnose(
    candidate: SplitCandidate,
    members: dict[str, list[str]],
    stats: dict[str, dict[str, float]],
) -> tuple[dict[str, dict[str, float]], list[str]]:
    """Summarise source characteristics per split, after the ranking is fixed.

    This runs **after** candidates are scored and never feeds back into the
    objective. Its purpose is to catch a catastrophic accidental shift, not to
    make the optimiser cleverer; changing the objective in response to what is
    seen here would be tuning on the answer.

    Args:
        candidate: The candidate to describe.
        members: Source image ids keyed by group id.
        stats: Phase 4A per-image measurements.

    Returns:
        Mean of each characteristic per split, and any warnings raised.
    """
    if not stats:
        return {}, []

    per_split: dict[str, list[dict[str, float]]] = {split: [] for split in SPLITS}
    for group_id, split in candidate.assignment.items():
        for image_id in members.get(group_id, []):
            record = stats.get(image_id)
            if record is not None:
                per_split[split].append(record)

    summary: dict[str, dict[str, float]] = {}
    for split, records in per_split.items():
        summary[split] = {}
        for field in DIAGNOSTIC_FIELDS:
            values = [float(r[field]) for r in records if r.get(field) is not None]
            summary[split][field] = round(sum(values) / len(values), 4) if values else 0.0

    warnings: list[str] = []
    for field in DIAGNOSTIC_FIELDS:
        values = [summary[split][field] for split in SPLITS if summary[split][field]]
        if len(values) < len(SPLITS):
            continue
        spread = (max(values) - min(values)) / max(abs(sum(values) / len(values)), 1e-9)
        if spread > WARNING_RATIO:
            warnings.append(f"{field}: split means differ by {spread:.0%}")
    return summary, warnings


def summary_row(
    candidate: SplitCandidate,
    rank: int,
    groups: list[GroupFeature],
    config: SplitSearchConfig,
    warnings: list[str],
) -> dict[str, Any]:
    """Build one row of the candidate summary.

    Args:
        candidate: The candidate to describe.
        rank: Its position in the ranking, one-based.
        groups: Every group.
        config: The search protocol.
        warnings: Post-hoc distribution warnings.

    Returns:
        A mapping of column name to value.
    """
    totals = candidate.totals(groups, config.classes)
    row: dict[str, Any] = {"candidate_id": candidate.candidate_id, "rank": rank}
    for split in SPLITS:
        row[f"{split}_images"] = totals[split].images
        row[f"{split}_groups"] = totals[split].groups
        row[f"{split}_negatives"] = totals[split].negative_images
        row[f"{split}_vest_loose_images"] = totals[split].images_with["vest_loose"]
        row[f"{split}_vest_loose_instances"] = totals[split].instances["vest_loose"]
    row.update(candidate.objective.as_row())
    row["candidate_assignment_sha256"] = candidate.fingerprint
    row["search_origin_start"] = candidate.origin
    row["vest_loose_family"] = candidate.family
    row["family_best"] = "true" if candidate.family_best else "false"
    row["distribution_warning"] = "DISTRIBUTION_WARNING" if warnings else ""
    return row


def _distribution_table(
    candidate: SplitCandidate,
    groups: list[GroupFeature],
    config: SplitSearchConfig,
    *,
    level: str,
) -> list[str]:
    """Render a class distribution table for one candidate.

    Args:
        candidate: The candidate to describe.
        groups: Every group.
        config: The search protocol.
        level: ``images`` or ``instances``.

    Returns:
        Markdown lines.
    """
    totals = candidate.totals(groups, config.classes)
    lines = [
        "| Class | train | % | validation | % | test | % | total |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name in config.classes:
        values = {
            split: (
                totals[split].images_with[name]
                if level == "images"
                else totals[split].instances[name]
            )
            for split in SPLITS
        }
        total = sum(values.values())
        cells = []
        for split in SPLITS:
            share = values[split] / total * 100 if total else 0.0
            cells.append(f"{values[split]} | {share:.1f}%")
        lines.append(f"| `{name}` | " + " | ".join(cells) + f" | {total} |")
    return lines


def build_report(
    candidates: list[SplitCandidate],
    groups: list[GroupFeature],
    config: SplitSearchConfig,
    counters: dict[str, int],
    diagnostics: dict[str, tuple[dict[str, dict[str, float]], list[str]]],
    commit: str | None,
) -> str:
    """Compose the candidate comparison report.

    Args:
        candidates: The retained candidates, best first.
        groups: Every group.
        config: The search protocol.
        counters: Search statistics.
        diagnostics: Post-hoc summaries and warnings, keyed by candidate id.
        commit: Git commit the search ran at.

    Returns:
        The Markdown document.
    """
    generated = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S+00:00")
    lines = [
        "# Provisional Split Candidates",
        "",
        f"Generated: {generated} · Phase: 5C.1 · Commit: `{commit or 'unknown'}`",
        "",
        "**No split is frozen and no candidate is selected.** Every assignment below is "
        "provisional. `final_selected_candidate` is `UNSELECTED_PENDING_REVIEW`; the holdout "
        "remains locked and no candidate test set has been evaluated.",
        "",
        "## 1. Objective and hard constraints",
        "",
        "The unit assigned is the **group**, never the image. Phase 5B.1 established 422 "
        "indivisible groups over 433 modelling images, so a confirmed duplicate pair cannot "
        "be separated by construction rather than by penalty.",
        "",
        "**Hard constraints.** An assignment breaking any of these is rejected outright, not "
        "ranked low:",
        "",
        "1. every group assigned to exactly one split;",
        "2. no group split across boundaries (structural: the group is the unit);",
        "3. all 433 modelling images assigned;",
        "4. all five classes present in all three splits, at image **and** instance level;",
        f"5. `vest_loose` images at least {config.min_rare_class_images['train']} / "
        f"{config.min_rare_class_images['validation']} / "
        f"{config.min_rare_class_images['test']} (train / validation / test);",
        f"6. negative images at least {config.min_negative_images['train']} / "
        f"{config.min_negative_images['validation']} / "
        f"{config.min_negative_images['test']};",
        f"7. split sizes within {config.max_size_deviation} image(s) of "
        f"{config.target_image_counts['train']} / "
        f"{config.target_image_counts['validation']} / "
        f"{config.target_image_counts['test']}.",
        "",
        "**Soft objective.** With `r(s)` the target ratio of split `s`:",
        "",
        "```",
        "E_img  = mean over (class, split) of",
        "         |images_with(c,s) - r(s) * total_images_with(c)| / max(target, 1)",
        "E_inst = mean over (class, split) of",
        "         |instances(c,s)   - r(s) * total_instances(c)|   / max(target, 1)",
        "E_neg  = mean over split of",
        "         |negatives(s)     - r(s) * total_negatives|      / max(target, 1)",
        "E_size = mean over split of",
        "         |images(s)        - target_images(s)|            / max(target, 1)",
        "",
        "total  = w_img*E_img + w_inst*E_inst + w_neg*E_neg + w_size*E_size",
        "```",
        "",
        "Each class contributes one normalised term per split and the terms are **averaged**, "
        "not summed. That is what stops `person` (914 instances) drowning out `vest_loose` "
        "(45): being ten images short of 640 is a small normalised error, being ten short of "
        "12 is not.",
        "",
        "Weights: "
        + ", ".join(f"`{k}` {v}" for k, v in sorted(config.weights.items()))
        + ". All equal, and deliberately so - each component is already normalised, so no "
        "rescaling is needed and equal weights cannot be mistaken for tuning toward a "
        "preferred answer.",
        "",
        "## 2. Algorithm",
        "",
        "Deterministic multi-start local search, no new dependency:",
        "",
        "1. **Rarity-aware initialisation.** Scarce requirements are placed first, rarest "
        "class first and smallest split first. Filling by size and hoping the rare class "
        "lands well fails reliably - `vest_loose` occupies 7 groups out of 422, so a "
        "size-first pass puts all 7 in train and every restart then has to dig out.",
        "2. **Size repair.** Groups move from the most over-target split to the most "
        "under-target one until the size tolerance is met.",
        "3. **Local search.** Equal-size swaps between splits (which cannot break the size "
        "constraint) and single-group moves. Only strictly improving steps are taken, and a "
        "step breaking a hard constraint is discarded rather than penalised.",
        f"4. **{config.starts} restarts**, each seeded `seed + index` from the project seed "
        f"{config.seed}, {config.iterations} local-search steps each.",
        "",
        "## 3. Reproducibility",
        "",
        "The search is a pure function of the group features and "
        "`configs/split_search.yaml`. No unseeded randomness, no filesystem ordering, no "
        "provider split, no image pixels. Candidates are deduplicated by assignment "
        "fingerprint and ranked by total objective with the fingerprint as tie-break, so the "
        "ranking cannot depend on discovery order.",
        "",
        f"Split-search configuration SHA-256: `{config.fingerprint()}`",
        "",
        "## 4. Target split sizes",
        "",
        "| Split | Target images | Target ratio |",
        "| --- | --- | --- |",
    ]
    for split in SPLITS:
        lines.append(
            f"| {split} | {config.target_image_counts[split]} | {config.target_ratios[split]:.0%} |"
        )
    lines.extend(
        [
            f"| **total** | **{sum(config.target_image_counts.values())}** | **100%** |",
            "",
            "## 5. Candidate ranking",
            "",
            f"Restarts: {counters['starts']}. Feasible: {counters['feasible']}. "
            f"Unique after deduplication: {counters['unique']}. Retained: {len(candidates)}.",
            "",
            "| Rank | Candidate | Images (t/v/te) | Negatives | `vest_loose` images | "
            "E_img | E_inst | E_neg | E_size | **Total** |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for rank, candidate in enumerate(candidates, start=1):
        totals = candidate.totals(groups, config.classes)
        sizes = "/".join(str(totals[s].images) for s in SPLITS)
        negatives = "/".join(str(totals[s].negative_images) for s in SPLITS)
        vest = "/".join(str(totals[s].images_with["vest_loose"]) for s in SPLITS)
        objective = candidate.objective
        lines.append(
            f"| {rank} | `{candidate.candidate_id}`"
            + (" *(family best)*" if candidate.family_best else "")
            + f" | {sizes} | {negatives} | {vest} | "
            f"{objective.image_class_balance:.4f} | {objective.instance_class_balance:.4f} | "
            f"{objective.negative_balance:.4f} | {objective.size:.4f} | "
            f"**{objective.total:.4f}** |"
        )
    lines.extend(
        [
            "",
            "Full assignments are in `reports/split_candidates/`, one file per candidate, "
            "keyed by `group_id` with a `provisional_split` column. The column is named "
            "`provisional_split` rather than `split` so no downstream reader can mistake it "
            "for a frozen assignment.",
            "",
            "| Candidate | Assignment SHA-256 |",
            "| --- | --- |",
        ]
    )
    for candidate in candidates:
        lines.append(f"| `{candidate.candidate_id}` | `{candidate.fingerprint}` |")
    return lines


def build_report_tail(
    lines: list[str],
    candidates: list[SplitCandidate],
    groups: list[GroupFeature],
    config: SplitSearchConfig,
    diagnostics: dict[str, tuple[dict[str, dict[str, float]], list[str]]],
    vest_detail: dict[str, list[dict[str, Any]]],
) -> str:
    """Append the per-candidate detail, diagnostics and limitations.

    Args:
        lines: The report built so far.
        candidates: The retained candidates, best first.
        groups: Every group.
        config: The search protocol.
        diagnostics: Post-hoc summaries and warnings, keyed by candidate id.
        vest_detail: Rare-class image detail, keyed by candidate id.

    Returns:
        The finished Markdown document.
    """
    best = candidates[0]

    lines.extend(["", "## 6. Class distribution by images", ""])
    for candidate in candidates:
        lines.append(f"**`{candidate.candidate_id}`** - images carrying each class:")
        lines.append("")
        lines.extend(_distribution_table(candidate, groups, config, level="images"))
        lines.append("")

    lines.extend(["## 7. Class distribution by instances", ""])
    for candidate in candidates:
        lines.append(f"**`{candidate.candidate_id}`** - annotation counts:")
        lines.append("")
        lines.extend(_distribution_table(candidate, groups, config, level="instances"))
        lines.append("")

    lines.extend(
        [
            "## 8. `vest_loose` allocation",
            "",
            "45 instances across 8 images held in **7** indivisible groups: one group "
            "(`manual_dup_010`) contains two of the images, so they cannot be separated and "
            "the class moves in chunks. This corrects the phase 5C.1 brief, which stated that "
            "no `vest_loose` image belongs to a duplicate group - that was true before phase "
            "5B.1 confirmed that pair.",
            "",
            "| Candidate | train | validation | test | instances (t/v/te) | family |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for candidate in candidates:
        totals = candidate.totals(groups, config.classes)
        images = [totals[s].images_with["vest_loose"] for s in SPLITS]
        instances = [totals[s].instances["vest_loose"] for s in SPLITS]
        family = candidate.family + (" (family best)" if candidate.family_best else "")
        lines.append(
            f"| `{candidate.candidate_id}` | {images[0]} | {images[1]} | {images[2]} | "
            f"{'/'.join(str(v) for v in instances)} | {family} |"
        )
    lines.extend(
        [
            "",
            "Source images carrying the class, per split, for the best-scoring candidate "
            f"`{best.candidate_id}`:",
            "",
            "| Split | Source image | Group | `vest_loose` instances | Co-occurring classes |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for entry in vest_detail[best.candidate_id]:
        lines.append(
            f"| {entry['split']} | `{entry['image_id']}` | `{entry['group_id']}` | "
            f"{entry['instances']} | {entry['co_occurring'] or '-'} |"
        )

    lines.extend(
        [
            "",
            "## 9. Negative images",
            "",
            "| Candidate | train | validation | test |",
            "| --- | --- | --- | --- |",
        ]
    )
    for candidate in candidates:
        totals = candidate.totals(groups, config.classes)
        lines.append(
            f"| `{candidate.candidate_id}` | "
            + " | ".join(str(totals[s].negative_images) for s in SPLITS)
            + " |"
        )
    lines.extend(
        [
            "",
            "14 negatives in total. The phase 4B audit found no missing target label on them, "
            "so they are deliberate background examples rather than defects, and none of them "
            "may be hoarded in train.",
            "",
            "## 10. Group integrity",
            "",
            "| Check | Result |",
            "| --- | --- |",
        ]
    )
    for candidate in candidates:
        violations = hard_violations(candidate.assignment, groups, config)
        assigned = len(candidate.assignment)
        images = sum(
            group.image_count
            for group in groups
            if candidate.assignment.get(group.group_id) is not None
        )
        lines.append(
            f"| `{candidate.candidate_id}` | {assigned} groups assigned exactly once, "
            f"{images} images, hard violations: {len(violations)} |"
        )
    lines.extend(
        [
            "",
            "Group indivisibility is structural rather than checked: the assignment maps group "
            "identifiers, so an image cannot be assigned independently of its group.",
            "",
            "## 11. Post-hoc source-distribution diagnostics",
            "",
            "Computed **after** ranking and never fed back into the objective. The purpose is "
            "to catch a catastrophic accidental shift, not to make the optimiser cleverer; "
            "changing the objective in response to what is seen here would be tuning on the "
            "answer, and would require regenerating every candidate under a new protocol.",
            "",
        ]
    )
    summary, _ = diagnostics[best.candidate_id]
    if summary:
        lines.extend(
            [
                f"Split means for `{best.candidate_id}`:",
                "",
                "| Characteristic | train | validation | test |",
                "| --- | --- | --- | --- |",
            ]
        )
        for field in DIAGNOSTIC_FIELDS:
            lines.append(
                f"| {field} | " + " | ".join(f"{summary[s][field]:g}" for s in SPLITS) + " |"
            )
        lines.append("")
    flagged = [cid for cid, (_, warnings) in diagnostics.items() if warnings]
    if flagged:
        lines.append("**DISTRIBUTION_WARNING** raised for: " + ", ".join(f"`{c}`" for c in flagged))
        for cid in flagged:
            for warning in diagnostics[cid][1]:
                lines.append(f"* `{cid}`: {warning}")
    else:
        lines.append(
            "No candidate raised a `DISTRIBUTION_WARNING`: no source characteristic differs "
            f"between splits by more than {WARNING_RATIO:.0%} of its mean."
        )

    lines.extend(
        [
            "",
            "## 12. Trade-offs between the top candidates",
            "",
            "The scalar total is a ranking device, not a verdict. What separates these "
            "candidates in practice:",
            "",
        ]
    )
    for candidate in candidates:
        totals = candidate.totals(groups, config.classes)
        vest = "/".join(str(totals[s].images_with["vest_loose"]) for s in SPLITS)
        vest_inst = "/".join(str(totals[s].instances["vest_loose"]) for s in SPLITS)
        negatives = "/".join(str(totals[s].negative_images) for s in SPLITS)
        lines.append(
            f"* **`{candidate.candidate_id}`** (total {candidate.objective.total:.4f}): "
            f"`vest_loose` {vest} images and {vest_inst} instances, negatives {negatives}. "
            f"Image balance {candidate.objective.image_class_balance:.4f}, instance balance "
            f"{candidate.objective.instance_class_balance:.4f}."
        )
    lines.extend(
        [
            "",
            "The rare class is where these differ most and where the scalar helps least. "
            "`vest_loose` instances are very unevenly distributed across its 8 images - one "
            "image alone carries 13 - so two candidates with the same image split can differ "
            "sharply in instances. A reviewer should decide whether image coverage or "
            "instance coverage matters more for the per-class claims this project intends to "
            "make about the holdout.",
            "",
            "## 13. Limitations",
            "",
            "* **No candidate is selected.** `final_selected_candidate` is "
            "`UNSELECTED_PENDING_REVIEW`. `algorithmic_best_candidate` names only the lowest "
            f"scorer under the predeclared objective, which is `{best.candidate_id}`.",
            "* **The objective is a proxy.** Balanced counts do not guarantee comparable "
            "difficulty. Nothing here measures how hard the images are.",
            "* **Local search is not exhaustive.** A better assignment may exist. The search "
            "is reproducible, not optimal, and the space was not brute-forced.",
            "* **Instance-level proportionality is not always reachable.** Objects co-occur "
            "inside images and images are bound into groups, so some deviation is structural "
            "rather than a search failure.",
            "* **`vest_loose` remains thin.** Whatever the split, per-class conclusions about "
            "it will rest on a handful of images. That is a property of the dataset, not of "
            "the split.",
            "* **No model exists**, so nothing here has been validated against downstream "
            "behaviour.",
            "",
        ]
    )
    return "\n".join(lines).rstrip("\n") + "\n"


def rare_class_detail(
    candidate: SplitCandidate,
    members: dict[str, list[str]],
    annotations: dict[str, dict[str, int]],
) -> list[dict[str, Any]]:
    """Describe every image carrying the rare class under one candidate.

    Args:
        candidate: The candidate to describe.
        members: Source image ids keyed by group id.
        annotations: Class counts keyed by source image id.

    Returns:
        One record per rare-class image, ordered by split then image id.
    """
    records: list[dict[str, Any]] = []
    for group_id, split in candidate.assignment.items():
        for image_id in members.get(group_id, []):
            counts = annotations.get(image_id, {})
            if not counts.get("vest_loose"):
                continue
            others = sorted(
                name for name, count in counts.items() if count and name != "vest_loose"
            )
            records.append(
                {
                    "split": split,
                    "image_id": image_id,
                    "group_id": group_id,
                    "instances": counts["vest_loose"],
                    "co_occurring": ", ".join(f"`{n}`" for n in others),
                }
            )
    return sorted(records, key=lambda r: (SPLITS.index(r["split"]), r["image_id"]))


def main(argv: list[str] | None = None) -> int:
    """Search for and report provisional split candidates.

    Args:
        argv: Command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="Split-search configuration.")
    args = parser.parse_args(argv)

    paths = ProjectPaths.from_root()
    try:
        config = load_split_search_config(args.config or (paths.configs / "split_search.yaml"))
        feature_rows = read_csv(paths.reports / FEATURES_CSV)
        manifest_rows = read_csv(paths.reports / MANIFEST_CSV)
        groups = load_groups(feature_rows, config)
    except (ConfigError, SplitInputError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    members: dict[str, list[str]] = {}
    for row in manifest_rows:
        members.setdefault(row["group_id"], []).append(row["source_image_id"])

    annotations: dict[str, dict[str, int]] = {}
    for row in read_csv(paths.reports / "canonical_annotation_actions.csv"):
        if row["action"] not in ("KEEP", "MATERIALIZE_RECTANGLE_FROM_BBOX"):
            continue
        counts = annotations.setdefault(row["source_image_id"], {})
        counts[row["label"]] = counts.get(row["label"], 0) + 1

    total_images = sum(group.image_count for group in groups)
    print(f"groups: {len(groups)}   images: {total_images}   seed: {config.seed}")
    print(f"target: {config.target_image_counts}   starts: {config.starts}")
    print("searching ...", flush=True)

    try:
        candidates, counters = search_candidates(groups, config)
    except InfeasibleSplitError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3

    stats = load_image_stats(paths)
    diagnostics = {
        candidate.candidate_id: diagnose(candidate, members, stats) for candidate in candidates
    }
    vest_detail = {
        candidate.candidate_id: rare_class_detail(candidate, members, annotations)
        for candidate in candidates
    }

    candidate_dir = paths.reports / CANDIDATE_DIR
    candidate_dir.mkdir(parents=True, exist_ok=True)
    for existing in sorted(candidate_dir.glob("candidate_*.csv")):
        existing.unlink()
    for candidate in candidates:
        write_csv(
            candidate_dir / f"{candidate.candidate_id}.csv",
            ["group_id", "provisional_split"],
            [
                {"group_id": group_id, "provisional_split": candidate.assignment[group_id]}
                for group_id in sorted(candidate.assignment)
            ],
        )
    write_csv(
        candidate_dir / "summary.csv",
        list(SUMMARY_COLUMNS),
        [
            summary_row(candidate, rank, groups, config, diagnostics[candidate.candidate_id][1])
            for rank, candidate in enumerate(candidates, start=1)
        ],
    )

    lines = build_report(candidates, groups, config, counters, diagnostics, git_commit())
    report = build_report_tail(lines, candidates, groups, config, diagnostics, vest_detail)
    (paths.reports / REPORT_MD).write_text(report, encoding="utf-8", newline="\n")

    print(f"\nfeasible: {counters['feasible']}/{counters['starts']}   unique: {counters['unique']}")
    print(f"retained: {len(candidates)}")
    for rank, candidate in enumerate(candidates, start=1):
        totals = candidate.totals(groups, config.classes)
        sizes = "/".join(str(totals[s].images) for s in SPLITS)
        vest = "/".join(str(totals[s].images_with["vest_loose"]) for s in SPLITS)
        negatives = "/".join(str(totals[s].negative_images) for s in SPLITS)
        print(
            f"  {rank}. {candidate.candidate_id}  images {sizes}  vest_loose {vest}  "
            f"neg {negatives}  total {candidate.objective.total:.6f}  "
            f"{candidate.fingerprint[:16]}"
        )
    print(f"\nalgorithmic_best_candidate: {candidates[0].candidate_id}")
    print("final_selected_candidate:   UNSELECTED_PENDING_REVIEW")
    print(f"split_search config sha256: {config.fingerprint()}")
    print(f"\nwrote {candidate_dir}")
    print(f"wrote {paths.reports / REPORT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
