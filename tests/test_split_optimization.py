"""Tests for the provisional split search.

Two properties matter most and neither is about score quality. The search must be
a pure function of its inputs, so a candidate can be reproduced from the
configuration alone. And a hard constraint must be a rejection, not a penalty: an
assignment that empties a class out of the holdout is not a bad solution to be
ranked low, it is not a solution.

Fixtures are small and synthetic. Nothing here reads the real dataset or the
network.
"""

from __future__ import annotations

import dataclasses

import pytest

from construction_safety_vision.config import ConfigError
from construction_safety_vision.data.split_optimization import (
    SPLITS,
    GroupFeature,
    InfeasibleSplitError,
    SplitSearchConfig,
    evaluate,
    fingerprint_assignment,
    hard_violations,
    is_feasible,
    load_split_search_config,
    search_candidates,
    totals_for,
)

CLASSES = ("rare", "common")
"""A two-class world: one class scarce enough to be lost by a careless split."""


def group(
    group_id: str,
    *,
    images: int = 1,
    negatives: int = 0,
    rare_images: int = 0,
    common_images: int = 0,
    rare: int = 0,
    common: int = 0,
) -> GroupFeature:
    """Build a synthetic group.

    Args:
        group_id: Identifier.
        images: Images the group holds.
        negatives: Images carrying no annotation.
        rare_images: Images carrying the rare class.
        common_images: Images carrying the common class.
        rare: Rare-class instances.
        common: Common-class instances.

    Returns:
        The group feature record.
    """
    return GroupFeature(
        group_id=group_id,
        image_count=images,
        negative_images=negatives,
        images_with={"rare": rare_images, "common": common_images},
        instances={"rare": rare, "common": common},
        group_type="SEMANTIC_DUPLICATE" if images > 1 else "SINGLETON",
    )


def config(**overrides: object) -> SplitSearchConfig:
    """Build a small search protocol.

    Args:
        **overrides: Fields to replace.

    Returns:
        The protocol.
    """
    base = SplitSearchConfig(
        schema_version=1,
        seed=42,
        classes=CLASSES,
        target_ratios={"train": 0.70, "validation": 0.15, "test": 0.15},
        target_image_counts={"train": 14, "validation": 3, "test": 3},
        max_size_deviation=0,
        min_class_images={"train": 1, "validation": 1, "test": 1},
        rare_class="rare",
        min_rare_class_images={"train": 0, "validation": 0, "test": 0},
        min_negative_images={"train": 1, "validation": 1, "test": 1},
        rare_class_families=(),
        weights={
            "image_class_balance": 1.0,
            "instance_class_balance": 1.0,
            "negative_balance": 1.0,
            "size": 1.0,
        },
        starts=12,
        iterations=200,
        top_candidates=5,
    )
    return dataclasses.replace(base, **overrides)  # type: ignore[arg-type]


def population() -> list[GroupFeature]:
    """Build a 20-image world that admits a feasible 14/3/3 split.

    Returns:
        Twenty groups: three negatives, six rare-class carriers, the rest common.
    """
    groups = [group(f"neg-{i}", negatives=1) for i in range(3)]
    groups += [
        group(f"rare-{i}", rare_images=1, rare=2, common_images=1, common=1) for i in range(6)
    ]
    groups += [group(f"common-{i}", common_images=1, common=3) for i in range(11)]
    return sorted(groups, key=lambda g: g.group_id)


# --- totals and objective -----------------------------------------------------


def test_totals_add_up_to_the_population() -> None:
    groups = population()
    assignment = {g.group_id: SPLITS[index % 3] for index, g in enumerate(groups)}
    totals = totals_for(assignment, groups, CLASSES)
    assert sum(totals[s].images for s in SPLITS) == sum(g.image_count for g in groups)
    assert sum(totals[s].groups for s in SPLITS) == len(groups)
    assert sum(totals[s].instances["rare"] for s in SPLITS) == sum(
        g.instances["rare"] for g in groups
    )


def test_the_objective_is_normalised_so_a_rare_class_is_not_drowned_out() -> None:
    # Ten images missing from a target of 12 must score worse than ten missing
    # from a target of 640, or the frequent class decides everything.
    from construction_safety_vision.data.split_optimization import _normalised_error

    assert _normalised_error(2, 12) > _normalised_error(630, 640)


def test_a_perfectly_proportional_split_scores_near_zero() -> None:
    groups = [group(f"g-{i}", common_images=1, common=1, negatives=0) for i in range(20)]
    assignment = {}
    for index, g in enumerate(groups):
        assignment[g.group_id] = "train" if index < 14 else ("validation" if index < 17 else "test")
    scored = evaluate(assignment, groups, config(min_negative_images=dict.fromkeys(SPLITS, 0)))
    assert scored.size == pytest.approx(0.0)
    assert scored.total < 0.1


def test_objective_components_are_reported_separately() -> None:
    groups = population()
    assignment = {g.group_id: "train" for g in groups}
    scored = evaluate(assignment, groups, config())
    row = scored.as_row()
    assert set(row) == {
        "image_class_balance_error",
        "instance_class_balance_error",
        "negative_balance_error",
        "size_error",
        "total_objective",
    }
    assert row["total_objective"] > 0


# --- hard constraints ---------------------------------------------------------


def test_putting_everything_in_train_is_rejected() -> None:
    groups = population()
    assignment = {g.group_id: "train" for g in groups}
    problems = hard_violations(assignment, groups, config())
    assert problems
    assert any("no rare instance" in p or "image(s) with rare" in p for p in problems)


def test_a_split_missing_a_class_is_rejected() -> None:
    groups = population()
    assignment = {}
    for g in groups:
        if g.group_id.startswith("rare"):
            assignment[g.group_id] = "train"
        else:
            assignment[g.group_id] = "validation" if g.group_id.endswith("0") else "test"
    problems = hard_violations(assignment, groups, config())
    assert any("rare" in p for p in problems)


def test_an_unassigned_group_is_rejected() -> None:
    groups = population()
    assignment = {g.group_id: "train" for g in groups[:-1]}
    problems = hard_violations(assignment, groups, config())
    assert any("unassigned" in p for p in problems)


def test_an_unknown_split_name_is_rejected() -> None:
    groups = population()
    assignment = {g.group_id: "holdout" for g in groups}
    assert any("unknown split" in p for p in hard_violations(assignment, groups, config()))


def test_a_size_deviation_beyond_tolerance_is_rejected() -> None:
    groups = population()
    assignment = {g.group_id: SPLITS[index % 3] for index, g in enumerate(groups)}
    problems = hard_violations(assignment, groups, config())
    assert any("target" in p and "deviation" in p for p in problems)


def test_missing_negatives_are_rejected() -> None:
    groups = population()
    assignment = {}
    for index, g in enumerate(groups):
        if g.group_id.startswith("neg"):
            assignment[g.group_id] = "train"
        else:
            assignment[g.group_id] = SPLITS[index % 3]
    assert any("negative image" in p for p in hard_violations(assignment, groups, config()))


# --- search -------------------------------------------------------------------


def test_the_search_finds_a_feasible_assignment() -> None:
    groups = population()
    settings = config()
    candidates, counters = search_candidates(groups, settings)
    assert candidates
    assert counters["feasible"] >= 1
    for candidate in candidates:
        assert is_feasible(candidate.assignment, groups, settings)


def test_every_group_is_assigned_exactly_once() -> None:
    groups = population()
    candidates, _ = search_candidates(groups, config())
    for candidate in candidates:
        assert sorted(candidate.assignment) == sorted(g.group_id for g in groups)
        assert len(candidate.assignment) == len(groups)


def test_group_members_cannot_be_separated() -> None:
    # Structural: the assignment maps groups, so a two-image group has one split
    # by construction and its images cannot be placed independently.
    groups = [*population(), group("pair", images=2, common_images=2, common=2)]
    groups = sorted(groups, key=lambda g: g.group_id)
    settings = config(target_image_counts={"train": 15, "validation": 3, "test": 4})
    candidates, _ = search_candidates(groups, settings)
    for candidate in candidates:
        assert candidate.assignment["pair"] in SPLITS


def test_the_same_seed_reproduces_the_same_ranking() -> None:
    groups = population()
    first, _ = search_candidates(groups, config())
    second, _ = search_candidates(groups, config())
    assert [c.candidate_id for c in first] == [c.candidate_id for c in second]
    assert [c.fingerprint for c in first] == [c.fingerprint for c in second]
    assert [c.objective.total for c in first] == [c.objective.total for c in second]


def test_a_different_seed_may_reproduce_itself_too() -> None:
    groups = population()
    first, _ = search_candidates(groups, config(seed=7))
    second, _ = search_candidates(groups, config(seed=7))
    assert [c.fingerprint for c in first] == [c.fingerprint for c in second]


def test_candidates_are_unique() -> None:
    groups = population()
    candidates, _ = search_candidates(groups, config())
    fingerprints = [c.fingerprint for c in candidates]
    assert len(fingerprints) == len(set(fingerprints))


def test_candidates_are_ranked_by_total_objective() -> None:
    groups = population()
    candidates, _ = search_candidates(groups, config())
    totals = [c.objective.total for c in candidates]
    assert totals == sorted(totals)


def test_impossible_constraints_fail_loudly() -> None:
    # A holdout demanding more rare-class images than exist cannot be satisfied,
    # and must raise rather than return a quietly invalid assignment.
    groups = population()
    settings = config(min_class_images={"train": 1, "validation": 1, "test": 99})
    with pytest.raises(InfeasibleSplitError, match="No feasible assignment"):
        search_candidates(groups, settings)


def test_group_order_does_not_change_the_result() -> None:
    groups = population()
    forward, _ = search_candidates(groups, config())
    backward, _ = search_candidates(list(reversed(groups)), config())
    assert [c.fingerprint for c in forward] == [c.fingerprint for c in backward]


# --- fingerprints -------------------------------------------------------------


def test_the_assignment_fingerprint_ignores_key_order() -> None:
    first = {"a": "train", "b": "test"}
    second = {"b": "test", "a": "train"}
    assert fingerprint_assignment(first) == fingerprint_assignment(second)


def test_moving_one_group_changes_the_fingerprint() -> None:
    baseline = {"a": "train", "b": "test"}
    moved = {"a": "validation", "b": "test"}
    assert fingerprint_assignment(baseline) != fingerprint_assignment(moved)


def test_the_config_fingerprint_reacts_to_a_protocol_change() -> None:
    assert config().fingerprint() != config(seed=7).fingerprint()
    assert config().fingerprint() == config().fingerprint()


# --- configuration ------------------------------------------------------------


def test_the_real_configuration_loads_and_targets_433_images() -> None:
    from construction_safety_vision.paths import ProjectPaths

    loaded = load_split_search_config(ProjectPaths.from_root().configs / "split_search.yaml")
    assert loaded.seed == 42
    assert sum(loaded.target_image_counts.values()) == 433
    assert loaded.target_image_counts == {"train": 303, "validation": 65, "test": 65}
    assert abs(sum(loaded.target_ratios.values()) - 1.0) < 1e-9
    assert loaded.rare_class_families
    assert loaded.rare_class == "vest_loose"


def test_an_unknown_configuration_key_is_refused(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from construction_safety_vision.paths import ProjectPaths

    source = (ProjectPaths.from_root().configs / "split_search.yaml").read_text(encoding="utf-8")
    broken = tmp_path / "split_search.yaml"
    broken.write_text(source + "\nunexpected_key: 1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown key"):
        load_split_search_config(broken)


def test_ratios_that_do_not_sum_to_one_are_refused(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from construction_safety_vision.paths import ProjectPaths

    source = (ProjectPaths.from_root().configs / "split_search.yaml").read_text(encoding="utf-8")
    broken = tmp_path / "split_search.yaml"
    broken.write_text(source.replace("train: 0.70", "train: 0.80"), encoding="utf-8")
    with pytest.raises(ConfigError, match=r"must sum to 1\.0"):
        load_split_search_config(broken)


def test_the_protocol_carries_no_provider_split_setting() -> None:
    # The provider split is UNSUITABLE_FOR_FINAL_PROTOCOL and must not be
    # reachable from the search configuration at all.
    from construction_safety_vision.paths import ProjectPaths

    text = (ProjectPaths.from_root().configs / "split_search.yaml").read_text(encoding="utf-8")
    settings = [line.split(":")[0].strip() for line in text.splitlines() if ":" in line]
    assert not [name for name in settings if name in {"provider_split", "original_split"}]
