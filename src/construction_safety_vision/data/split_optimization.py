"""Deterministic group-aware, class-aware search for candidate dataset splits.

The unit a split assigns is a **group**, never an image. Phase 5B.1 established
422 indivisible groups over 433 modelling images: 411 singletons and 11 groups of
two images that a person confirmed show the same scene. Assigning images directly
would let two views of one scene land on opposite sides of a split boundary,
which is precisely the leakage the grouping exists to prevent.

Three properties are deliberate.

**Nothing here reads the provider's split.** It was rejected in phase 4B, so it
is not an input, not an initialisation and not a target. The search sees only
canonical group metadata: sizes, class presence, class instance counts and
negative-image counts.

**Hard constraints are rejections, not penalties.** A candidate that leaves a
class out of a split is not a poor solution to be ranked low; it is not a
solution. Only the remaining preferences are scored.

**The score is normalised per class and averaged.** Summing raw deviations would
let ``person`` (914 instances) drown out ``vest_loose`` (45), and the rare class
is the one a split can most easily ruin.

Nothing in this module freezes a split, writes a holdout manifest, or reads an
image.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from construction_safety_vision.config import ConfigError, check_keys

TRAIN = "train"
VALIDATION = "validation"
TEST = "test"

SPLITS: tuple[str, ...] = (TRAIN, VALIDATION, TEST)
"""The three provisional splits, in a fixed order for deterministic output."""

CONFIG_SCHEMA_VERSION = 1
"""Schema version of ``configs/split_search.yaml``."""


class SplitSearchError(RuntimeError):
    """Raised when a split search cannot be run as configured."""


class InfeasibleSplitError(SplitSearchError):
    """Raised when no assignment can satisfy the hard constraints."""


@dataclass(frozen=True)
class GroupFeature:
    """One indivisible split unit, described only by canonical metadata.

    Attributes:
        group_id: Deterministic group identifier.
        image_count: Images the group holds.
        negative_images: Images in it carrying no annotation.
        images_with: Images carrying each class, keyed by class name.
        instances: Annotation counts, keyed by class name.
        group_type: ``SINGLETON`` or ``SEMANTIC_DUPLICATE``.
    """

    group_id: str
    image_count: int
    negative_images: int
    images_with: dict[str, int]
    instances: dict[str, int]
    group_type: str

    @property
    def total_instances(self) -> int:
        """Annotations across every class.

        Returns:
            The summed instance count.
        """
        return sum(self.instances.values())


@dataclass(frozen=True)
class SplitSearchConfig:
    """The split-search protocol, loaded from a versioned file.

    Attributes:
        schema_version: Version of the configuration schema.
        seed: Master seed. Every random draw derives from it.
        classes: Canonical class order.
        target_ratios: Desired share of images per split.
        target_image_counts: Desired integer image count per split.
        max_size_deviation: Images a split may deviate from its target before the
            assignment is rejected.
        min_class_images: Images carrying each class that every split must hold.
        rare_class: The class whose scarcity needs protecting beyond the generic
            floor. Named here rather than hardcoded: which class is rare is a
            property of the dataset, not of the algorithm.
        min_rare_class_images: Per-split minimum for that class.
        min_negative_images: Per-split minimum for images carrying no annotation.
        rare_class_families: Declared rare-class allocations to search, each an
            exact per-split image count. Searching them separately is what keeps
            the alternatives visible instead of letting the score pick one.
        exact_rare_class_images: When set, the rare-class image count per split
            must match exactly. Set per family during the search, never in the
            configuration file.
        weights: Weight of each soft objective component.
        starts: Independent search restarts.
        iterations: Local-search steps per restart.
        top_candidates: Unique candidates retained.
    """

    schema_version: int
    seed: int
    classes: tuple[str, ...]
    target_ratios: dict[str, float]
    target_image_counts: dict[str, int]
    max_size_deviation: int
    min_class_images: dict[str, int]
    rare_class: str
    min_rare_class_images: dict[str, int]
    min_negative_images: dict[str, int]
    rare_class_families: tuple[tuple[int, int, int], ...]
    weights: dict[str, float]
    starts: int
    iterations: int
    top_candidates: int
    exact_rare_class_images: dict[str, int] | None = None

    def minimum_images_for(self, class_name: str, split: str) -> int:
        """Return the floor on images carrying a class in a split.

        Args:
            class_name: Canonical class name.
            split: Split name.

        Returns:
            The larger of the generic class floor and any class-specific floor.
        """
        generic = self.min_class_images.get(split, 1)
        if class_name == self.rare_class:
            return max(generic, self.min_rare_class_images.get(split, 0))
        return generic

    def fingerprint(self) -> str:
        """Hash the protocol, so a candidate can name the rules that produced it.

        Returns:
            A SHA-256 hex digest over the configuration content only.
        """
        payload = {
            "schema_version": self.schema_version,
            "seed": self.seed,
            "classes": list(self.classes),
            "target_ratios": self.target_ratios,
            "target_image_counts": self.target_image_counts,
            "max_size_deviation": self.max_size_deviation,
            "min_class_images": self.min_class_images,
            "rare_class": self.rare_class,
            "min_rare_class_images": self.min_rare_class_images,
            "min_negative_images": self.min_negative_images,
            "rare_class_families": [list(family) for family in self.rare_class_families],
            "weights": self.weights,
            "starts": self.starts,
            "iterations": self.iterations,
            "top_candidates": self.top_candidates,
        }
        return _digest(payload)


def _digest(payload: Any) -> str:
    """Hash a JSON-serialisable payload deterministically.

    Args:
        payload: Content to hash.

    Returns:
        Its SHA-256 hex digest.
    """
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_split_search_config(path: str | Path) -> SplitSearchConfig:
    """Load and validate the split-search protocol.

    Parsing is strict: an unknown key is an error rather than a silently ignored
    setting, so a typo cannot quietly change the protocol.

    Args:
        path: Configuration file.

    Returns:
        The parsed protocol.

    Raises:
        ConfigError: If the file is missing, malformed or has unexpected keys.
    """
    config_path = Path(path)
    if not config_path.is_file():
        msg = f"Split-search configuration not found: {config_path.name}"
        raise ConfigError(msg)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        msg = f"{config_path.name} is not valid YAML: {exc}"
        raise ConfigError(msg) from None
    if not isinstance(raw, dict):
        msg = f"{config_path.name} must contain a mapping at the top level"
        raise ConfigError(msg)

    check_keys(
        raw,
        required=[
            "schema_version",
            "seed",
            "classes",
            "target_ratios",
            "target_image_counts",
            "max_size_deviation",
            "min_class_images",
            "rare_class",
            "min_rare_class_images",
            "min_negative_images",
            "rare_class_allocation_families",
            "weights",
            "search",
        ],
        context="split_search",
    )
    search = raw["search"]
    if not isinstance(search, dict):
        msg = "split_search.search must be a mapping"
        raise ConfigError(msg)
    check_keys(
        search,
        required=["starts", "iterations", "top_candidates"],
        context="split_search.search",
    )
    for name in ("target_ratios", "target_image_counts", "min_class_images"):
        check_keys(raw[name], required=list(SPLITS), context=f"split_search.{name}")
    check_keys(
        raw["min_rare_class_images"],
        required=list(SPLITS),
        context="split_search.min_rare_class_images",
    )
    check_keys(
        raw["min_negative_images"],
        required=list(SPLITS),
        context="split_search.min_negative_images",
    )
    check_keys(
        raw["weights"],
        required=["image_class_balance", "instance_class_balance", "negative_balance", "size"],
        context="split_search.weights",
    )

    if int(raw["schema_version"]) != CONFIG_SCHEMA_VERSION:
        msg = (
            f"split_search schema_version {raw['schema_version']} is not the supported "
            f"version {CONFIG_SCHEMA_VERSION}"
        )
        raise ConfigError(msg)

    ratios = {name: float(raw["target_ratios"][name]) for name in SPLITS}
    if abs(sum(ratios.values()) - 1.0) > 1e-9:
        msg = f"split_search.target_ratios must sum to 1.0, got {sum(ratios.values())}"
        raise ConfigError(msg)

    return SplitSearchConfig(
        schema_version=int(raw["schema_version"]),
        seed=int(raw["seed"]),
        classes=tuple(str(name) for name in raw["classes"]),
        target_ratios=ratios,
        target_image_counts={name: int(raw["target_image_counts"][name]) for name in SPLITS},
        max_size_deviation=int(raw["max_size_deviation"]),
        min_class_images={name: int(raw["min_class_images"][name]) for name in SPLITS},
        rare_class=str(raw["rare_class"]),
        min_rare_class_images={name: int(raw["min_rare_class_images"][name]) for name in SPLITS},
        min_negative_images={name: int(raw["min_negative_images"][name]) for name in SPLITS},
        rare_class_families=tuple(
            tuple(int(family[name]) for name in SPLITS)
            for family in raw["rare_class_allocation_families"]
        ),
        weights={key: float(value) for key, value in raw["weights"].items()},
        starts=int(search["starts"]),
        iterations=int(search["iterations"]),
        top_candidates=int(search["top_candidates"]),
    )


@dataclass(frozen=True)
class SplitTotals:
    """What one split holds under a given assignment.

    Attributes:
        images: Images in the split.
        groups: Groups in the split.
        negative_images: Images carrying no annotation.
        images_with: Images carrying each class.
        instances: Annotations of each class.
    """

    images: int
    groups: int
    negative_images: int
    images_with: dict[str, int]
    instances: dict[str, int]


def totals_for(
    assignment: dict[str, str], groups: list[GroupFeature], classes: tuple[str, ...]
) -> dict[str, SplitTotals]:
    """Aggregate an assignment into per-split totals.

    Args:
        assignment: Split name keyed by group id.
        groups: Every group.
        classes: Canonical class order.

    Returns:
        Totals keyed by split name.
    """
    images = dict.fromkeys(SPLITS, 0)
    group_counts = dict.fromkeys(SPLITS, 0)
    negatives = dict.fromkeys(SPLITS, 0)
    with_class = {split: dict.fromkeys(classes, 0) for split in SPLITS}
    instances = {split: dict.fromkeys(classes, 0) for split in SPLITS}

    for group in groups:
        split = assignment[group.group_id]
        images[split] += group.image_count
        group_counts[split] += 1
        negatives[split] += group.negative_images
        for name in classes:
            with_class[split][name] += group.images_with.get(name, 0)
            instances[split][name] += group.instances.get(name, 0)

    return {
        split: SplitTotals(
            images=images[split],
            groups=group_counts[split],
            negative_images=negatives[split],
            images_with=with_class[split],
            instances=instances[split],
        )
        for split in SPLITS
    }


@dataclass(frozen=True)
class ObjectiveBreakdown:
    """The soft objective, reported component by component.

    A single scalar hides which trade-off a candidate actually made, so each
    normalised component is kept alongside the total.

    Attributes:
        image_class_balance: Mean normalised error in images carrying each class.
        instance_class_balance: Mean normalised error in class instance counts.
        negative_balance: Normalised error in negative-image placement.
        size: Normalised error in split sizes.
        total: Weighted sum of the components.
    """

    image_class_balance: float
    instance_class_balance: float
    negative_balance: float
    size: float
    total: float

    def as_row(self) -> dict[str, Any]:
        """Serialise for the committed summary table.

        Returns:
            A mapping of column name to rounded value.
        """
        return {
            "image_class_balance_error": round(self.image_class_balance, 6),
            "instance_class_balance_error": round(self.instance_class_balance, 6),
            "negative_balance_error": round(self.negative_balance, 6),
            "size_error": round(self.size, 6),
            "total_objective": round(self.total, 6),
        }


def _normalised_error(observed: float, target: float) -> float:
    """Normalised absolute deviation from a target.

    Dividing by the target is what stops a large class dominating: being ten
    images short of 640 is a small error, being ten short of 12 is not.

    Args:
        observed: Measured value.
        target: Desired value.

    Returns:
        ``|observed - target| / max(target, 1)``.
    """
    return abs(observed - target) / max(target, 1.0)


def evaluate(
    assignment: dict[str, str], groups: list[GroupFeature], config: SplitSearchConfig
) -> ObjectiveBreakdown:
    """Score an assignment against the documented soft objective.

    The formula, with ``r(s)`` the target ratio of split ``s``:

    * ``E_img``: mean over (class, split) of
      ``|images_with(c,s) - r(s) * total_images_with(c)| / max(target, 1)``
    * ``E_inst``: mean over (class, split) of
      ``|instances(c,s) - r(s) * total_instances(c)| / max(target, 1)``
    * ``E_neg``: mean over split of
      ``|negatives(s) - r(s) * total_negatives| / max(target, 1)``
    * ``E_size``: mean over split of
      ``|images(s) - target_images(s)| / max(target, 1)``
    * ``total = w_img*E_img + w_inst*E_inst + w_neg*E_neg + w_size*E_size``

    Each class contributes one normalised term per split and the terms are
    averaged, so every class carries the same weight regardless of frequency.

    Args:
        assignment: Split name keyed by group id.
        groups: Every group.
        config: The search protocol.

    Returns:
        The scored breakdown.
    """
    totals = totals_for(assignment, groups, config.classes)
    global_images_with = {
        name: sum(group.images_with.get(name, 0) for group in groups) for name in config.classes
    }
    global_instances = {
        name: sum(group.instances.get(name, 0) for group in groups) for name in config.classes
    }
    global_negatives = sum(group.negative_images for group in groups)

    image_errors: list[float] = []
    instance_errors: list[float] = []
    for name in config.classes:
        for split in SPLITS:
            ratio = config.target_ratios[split]
            image_errors.append(
                _normalised_error(totals[split].images_with[name], ratio * global_images_with[name])
            )
            instance_errors.append(
                _normalised_error(totals[split].instances[name], ratio * global_instances[name])
            )

    negative_errors = [
        _normalised_error(
            totals[split].negative_images, config.target_ratios[split] * global_negatives
        )
        for split in SPLITS
    ]
    size_errors = [
        _normalised_error(totals[split].images, config.target_image_counts[split])
        for split in SPLITS
    ]

    image_error = sum(image_errors) / len(image_errors)
    instance_error = sum(instance_errors) / len(instance_errors)
    negative_error = sum(negative_errors) / len(negative_errors)
    size_error = sum(size_errors) / len(size_errors)

    total = (
        config.weights["image_class_balance"] * image_error
        + config.weights["instance_class_balance"] * instance_error
        + config.weights["negative_balance"] * negative_error
        + config.weights["size"] * size_error
    )
    return ObjectiveBreakdown(
        image_class_balance=image_error,
        instance_class_balance=instance_error,
        negative_balance=negative_error,
        size=size_error,
        total=total,
    )


def hard_violations(
    assignment: dict[str, str], groups: list[GroupFeature], config: SplitSearchConfig
) -> list[str]:
    """List every hard constraint an assignment breaks.

    Args:
        assignment: Split name keyed by group id.
        groups: Every group.
        config: The search protocol.

    Returns:
        One description per violation, empty when the assignment is feasible.
    """
    problems: list[str] = []
    known = {group.group_id for group in groups}
    assigned = set(assignment)
    if assigned != known:
        missing = sorted(known - assigned)
        extra = sorted(assigned - known)
        if missing:
            problems.append(f"{len(missing)} group(s) unassigned, e.g. {missing[:3]}")
        if extra:
            problems.append(f"{len(extra)} unknown group(s) assigned, e.g. {extra[:3]}")
        return problems
    unknown_splits = sorted(set(assignment.values()) - set(SPLITS))
    if unknown_splits:
        problems.append(f"unknown split name(s): {unknown_splits}")
        return problems

    totals = totals_for(assignment, groups, config.classes)
    for split in SPLITS:
        deviation = abs(totals[split].images - config.target_image_counts[split])
        if deviation > config.max_size_deviation:
            problems.append(
                f"{split} holds {totals[split].images} images, "
                f"target {config.target_image_counts[split]} "
                f"(deviation {deviation} > {config.max_size_deviation})"
            )
        if totals[split].negative_images < config.min_negative_images[split]:
            problems.append(
                f"{split} holds {totals[split].negative_images} negative image(s), "
                f"minimum {config.min_negative_images[split]}"
            )
        if config.exact_rare_class_images is not None:
            required = config.exact_rare_class_images[split]
            held = totals[split].images_with.get(config.rare_class, 0)
            if held != required:
                problems.append(
                    f"{split} holds {held} {config.rare_class} image(s), "
                    f"family requires exactly {required}"
                )
        for name in config.classes:
            floor = config.minimum_images_for(name, split)
            if totals[split].images_with[name] < floor:
                problems.append(
                    f"{split} holds {totals[split].images_with[name]} image(s) with {name}, "
                    f"minimum {floor}"
                )
            if totals[split].instances[name] <= 0:
                problems.append(f"{split} holds no {name} instance")
    return problems


def is_feasible(
    assignment: dict[str, str], groups: list[GroupFeature], config: SplitSearchConfig
) -> bool:
    """Report whether an assignment satisfies every hard constraint.

    Args:
        assignment: Split name keyed by group id.
        groups: Every group.
        config: The search protocol.

    Returns:
        ``True`` when nothing is violated.
    """
    return not hard_violations(assignment, groups, config)


def fingerprint_assignment(assignment: dict[str, str]) -> str:
    """Fingerprint an assignment by its content alone.

    Args:
        assignment: Split name keyed by group id.

    Returns:
        A SHA-256 hex digest over the sorted ``(group id, split)`` pairs. No
        timestamp, path or provider metadata enters it.
    """
    return _digest([[group_id, assignment[group_id]] for group_id in sorted(assignment)])


@dataclass
class SplitCandidate:
    """One provisional assignment and how it scored.

    Attributes:
        candidate_id: Stable identifier assigned after ranking.
        assignment: Split name keyed by group id.
        objective: The scored breakdown.
        fingerprint: Content hash of the assignment.
        origin: Which restart produced it, for traceability.
        family: The rare-class allocation it realises, as ``train/val/test``
            images carrying ``vest_loose``.
        family_best: Whether it is the best-scoring candidate of that family.
    """

    candidate_id: str
    assignment: dict[str, str]
    objective: ObjectiveBreakdown
    fingerprint: str
    origin: int
    family: str = ""
    family_best: bool = False

    def totals(
        self, groups: list[GroupFeature], classes: tuple[str, ...]
    ) -> dict[str, SplitTotals]:
        """Aggregate this candidate's assignment.

        Args:
            groups: Every group.
            classes: Canonical class order.

        Returns:
            Totals keyed by split name.
        """
        return totals_for(self.assignment, groups, classes)


def _class_rarity_order(groups: list[GroupFeature], classes: tuple[str, ...]) -> list[str]:
    """Order classes from rarest to commonest by the images that carry them.

    Args:
        groups: Every group.
        classes: Canonical class order.

    Returns:
        Class names, rarest first, ties broken by name so the order is stable.
    """
    totals = {name: sum(group.images_with.get(name, 0) for group in groups) for name in classes}
    return sorted(classes, key=lambda name: (totals[name], name))


def _seed_assignment(
    groups: list[GroupFeature], config: SplitSearchConfig, rng: random.Random
) -> dict[str, str]:
    """Build a starting assignment that respects the scarce constraints first.

    Filling by size and hoping the rare class lands somewhere useful fails
    reliably: ``vest_loose`` occupies seven groups out of 422, so a size-first
    pass puts all seven in train and every restart then has to dig its way out.
    Scarce requirements are therefore placed first, in order of scarcity, and the
    bulk is filled around them.

    Args:
        groups: Every group.
        config: The search protocol.
        rng: Seeded generator; the only source of randomness.

    Returns:
        A complete assignment. It is a starting point, not necessarily feasible.
    """
    assignment: dict[str, str] = {}
    remaining = {group.group_id: group for group in groups}
    images = dict.fromkeys(SPLITS, 0)

    def place(group: GroupFeature, split: str) -> None:
        assignment[group.group_id] = split
        images[split] += group.image_count
        remaining.pop(group.group_id, None)

    # Scarce classes first, rarest to commonest, smallest splits first so the
    # tight validation and test quotas are satisfied while choice still exists.
    for name in _class_rarity_order(groups, config.classes):
        for split in sorted(SPLITS, key=lambda s: config.target_image_counts[s]):
            floor = config.minimum_images_for(name, split)
            carried = sum(
                group.images_with.get(name, 0)
                for group in groups
                if assignment.get(group.group_id) == split
            )
            candidates = [
                group for group in remaining.values() if group.images_with.get(name, 0) > 0
            ]
            rng.shuffle(candidates)
            while carried < floor and candidates:
                chosen = candidates.pop()
                if chosen.group_id not in remaining:
                    continue
                place(chosen, split)
                carried += chosen.images_with.get(name, 0)

    # Negative images next: they are also scarce and also constrained per split.
    for split in sorted(SPLITS, key=lambda s: config.target_image_counts[s]):
        held = sum(
            group.negative_images for group in groups if assignment.get(group.group_id) == split
        )
        candidates = [group for group in remaining.values() if group.negative_images > 0]
        rng.shuffle(candidates)
        while held < config.min_negative_images[split] and candidates:
            chosen = candidates.pop()
            if chosen.group_id not in remaining:
                continue
            place(chosen, split)
            held += chosen.negative_images

    # Everything else fills whichever split is furthest below its target.
    rest = sorted(remaining.values(), key=lambda group: (-group.image_count, group.group_id))
    rng.shuffle(rest)
    for group in rest:
        split = min(SPLITS, key=lambda s: (images[s] - config.target_image_counts[s], s))
        place(group, split)
    return assignment


def _repair_sizes(
    assignment: dict[str, str], groups: list[GroupFeature], config: SplitSearchConfig
) -> dict[str, str]:
    """Move groups until every split is within its size tolerance, if possible.

    Args:
        assignment: Split name keyed by group id.
        groups: Every group.
        config: The search protocol.

    Returns:
        A possibly improved assignment. Moves that break a hard constraint are
        never applied, so this may return without reaching the tolerance.
    """
    by_id = {group.group_id: group for group in groups}
    working = dict(assignment)
    for _ in range(len(groups)):
        totals = totals_for(working, groups, config.classes)
        over = max(SPLITS, key=lambda s: totals[s].images - config.target_image_counts[s])
        under = min(SPLITS, key=lambda s: totals[s].images - config.target_image_counts[s])
        surplus = totals[over].images - config.target_image_counts[over]
        if surplus <= config.max_size_deviation and over != under:
            deficit = config.target_image_counts[under] - totals[under].images
            if deficit <= config.max_size_deviation:
                break
        moved = False
        for group_id in sorted(gid for gid, split in working.items() if split == over):
            if by_id[group_id].image_count > max(surplus, 1):
                continue
            trial = dict(working)
            trial[group_id] = under
            improves = _size_gap(trial, groups, config) < _size_gap(working, groups, config)
            if improves or not hard_violations(trial, groups, config):
                working = trial
                moved = True
                break
        if not moved:
            break
    return working


def _size_gap(
    assignment: dict[str, str], groups: list[GroupFeature], config: SplitSearchConfig
) -> int:
    """Total absolute deviation from the target split sizes.

    Args:
        assignment: Split name keyed by group id.
        groups: Every group.
        config: The search protocol.

    Returns:
        The summed absolute image-count deviation.
    """
    totals = totals_for(assignment, groups, config.classes)
    return sum(abs(totals[s].images - config.target_image_counts[s]) for s in SPLITS)


def _local_search(
    assignment: dict[str, str],
    groups: list[GroupFeature],
    config: SplitSearchConfig,
    rng: random.Random,
) -> dict[str, str]:
    """Improve a feasible assignment by single moves and pairwise swaps.

    Only strictly improving steps are taken, and any step that would break a hard
    constraint is discarded rather than penalised.

    Args:
        assignment: A feasible starting assignment.
        groups: Every group.
        config: The search protocol.
        rng: Seeded generator.

    Returns:
        The improved assignment.
    """
    by_id = {group.group_id: group for group in groups}
    order = sorted(by_id)
    working = dict(assignment)
    best = evaluate(working, groups, config).total

    for _ in range(config.iterations):
        improved = False

        # Swap two groups of equal image count between different splits: this
        # preserves every split size, so it can never break the size constraint.
        first, second = rng.sample(order, 2)
        same_size = by_id[first].image_count == by_id[second].image_count
        if working[first] != working[second] and same_size:
            trial = dict(working)
            trial[first], trial[second] = working[second], working[first]
            score = evaluate(trial, groups, config).total
            if score < best - 1e-12 and is_feasible(trial, groups, config):
                working, best, improved = trial, score, True

        if not improved:
            group_id = rng.choice(order)
            for split in SPLITS:
                if split == working[group_id]:
                    continue
                trial = dict(working)
                trial[group_id] = split
                score = evaluate(trial, groups, config).total
                if score < best - 1e-12 and is_feasible(trial, groups, config):
                    working, best = trial, score
                    break
    return working


def search_candidates(
    groups: list[GroupFeature], config: SplitSearchConfig
) -> tuple[list[SplitCandidate], dict[str, int]]:
    """Search for feasible split assignments and rank them.

    Every restart derives its generator from the master seed, so the whole search
    is a pure function of ``(groups, config)``. Candidates are deduplicated by
    assignment fingerprint before ranking, and ranked by total objective with the
    fingerprint as a tie-break so the order cannot depend on discovery order.

    Args:
        groups: Every group.
        config: The search protocol.

    Returns:
        The retained candidates, best first, and counters describing the search.

    Raises:
        InfeasibleSplitError: If no restart produced a feasible assignment.
    """
    ordered = sorted(groups, key=lambda group: group.group_id)
    feasible: list[tuple[ObjectiveBreakdown, dict[str, str], int]] = []
    families = config.rare_class_families or ((0, 0, 0),)
    counters = {
        "starts": config.starts * len(families),
        "families": len(families),
        "feasible": 0,
        "infeasible": 0,
    }
    per_family: dict[str, int] = {}

    for index, family in enumerate(families):
        pinned = replace(
            config,
            exact_rare_class_images=(
                dict(zip(SPLITS, family, strict=True)) if config.rare_class_families else None
            ),
        )
        found = 0
        for start in range(config.starts):
            # Offsetting by family keeps every restart's seed distinct while the
            # whole search stays a pure function of the master seed.
            rng = random.Random(config.seed + start + index * 10_000)
            assignment = _seed_assignment(ordered, pinned, rng)
            assignment = _repair_sizes(assignment, ordered, pinned)
            if not is_feasible(assignment, ordered, pinned):
                counters["infeasible"] += 1
                continue
            assignment = _local_search(assignment, ordered, pinned, rng)
            if not is_feasible(assignment, ordered, pinned):
                counters["infeasible"] += 1
                continue
            counters["feasible"] += 1
            found += 1
            # Scored against the unpinned objective so families compare fairly.
            feasible.append((evaluate(assignment, ordered, config), assignment, start))
        per_family["/".join(str(v) for v in family)] = found
    counters["per_family"] = per_family  # type: ignore[assignment]

    if not feasible:
        msg = (
            f"No feasible assignment after {counters['starts']} restarts across "
            f"{len(families)} allocation family/families. Relax a hard constraint or widen "
            "max_size_deviation, and record the change as a protocol iteration."
        )
        raise InfeasibleSplitError(msg)

    unique: dict[str, tuple[ObjectiveBreakdown, dict[str, str], int]] = {}
    for objective, assignment, start in feasible:
        key = fingerprint_assignment(assignment)
        if key not in unique or objective.total < unique[key][0].total:
            unique[key] = (objective, assignment, start)
    counters["unique"] = len(unique)

    ranked = sorted(unique.items(), key=lambda item: (item[1][0].total, item[0]))

    def family_of(assignment: dict[str, str]) -> str:
        totals = totals_for(assignment, ordered, config.classes)
        return "/".join(
            str(totals[split].images_with.get(config.rare_class, 0)) for split in SPLITS
        )

    # The best candidate of each family, so a family that never wins the global
    # ranking is still represented rather than silently dropped.
    best_of_family: dict[str, str] = {}
    for key, (_, assignment, _start) in ranked:
        best_of_family.setdefault(family_of(assignment), key)

    selected = [key for key, _ in ranked[: config.top_candidates]]
    for key in best_of_family.values():
        if key not in selected:
            selected.append(key)
    selected.sort(key=lambda k: (unique[k][0].total, k))

    candidates: list[SplitCandidate] = []
    for index, key in enumerate(selected, start=1):
        objective, assignment, start = unique[key]
        family = family_of(assignment)
        candidates.append(
            SplitCandidate(
                candidate_id=f"candidate_{index:03d}",
                assignment=assignment,
                objective=objective,
                fingerprint=key,
                origin=start,
                family=family,
                family_best=best_of_family.get(family) == key,
            )
        )
    counters["retained"] = len(candidates)
    return candidates, counters
