"""Tests for the direct instance-mask IoU diagnostic.

Every fixture here is synthetic. The evaluator is exercised on hand-built
rectangles whose IoU can be computed on paper, so a failure points at the
evaluator rather than at a model, and so these tests could run before S0 had
ever been trained - which is when they were written.

The cases worth naming:

**Greedy is not the same rule.** One test builds a matrix where taking the
single best pair first forces a worse total, and asserts the assignment returns
the optimal one. That is the whole reason the protocol names
``linear_sum_assignment`` rather than "match the best pair, then the next".

**A zero-overlap assignment is not a match.** The solver will pair two instances
that share no pixel when the alternative is leaving both unassigned; the
evaluator must throw that pair away.

**A missed instance is a zero, not an absence.** ``gt_normalized_mask_iou``
divides by every canonical instance, so dropping unmatched ground truth instead
of counting it would turn a model that finds one object into a near-perfect one.

No test here trains, infers, reads a real image, or touches the holdout.
"""

from __future__ import annotations

import itertools
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from construction_safety_vision.config import ConfigError
from construction_safety_vision.mask_iou_evaluation import (
    COVERAGE_THRESHOLDS,
    GROUND_TRUTH_SOURCE,
    GT_IOU50_COVERAGE,
    GT_IOU75_COVERAGE,
    GT_MATCH_COVERAGE,
    GT_NORMALIZED_MASK_IOU,
    HEADLINE_DIAGNOSTICS,
    MATCHED_MASK_IOU_MEAN,
    MATCHING_ALGORITHM,
    PROTOCOL_NAME,
    MaskInstance,
    MaskIoUEvaluationError,
    accumulate,
    assign_maximum_total_iou,
    evaluate,
    iou_matrix,
    mask_iou,
    match_image,
)
from construction_safety_vision.paths import ProjectPaths

CANVAS = (40, 40)
CONFIG_NAME = "segmentation_mask_iou_evaluation.yaml"


def rectangle(top: int, left: int, height: int, width: int, *, canvas=CANVAS) -> np.ndarray:
    """Build a boolean mask holding one filled rectangle."""
    mask = np.zeros(canvas, dtype=bool)
    mask[top : top + height, left : left + width] = True
    return mask


def instance(class_id: int, top: int, left: int, height: int, width: int) -> MaskInstance:
    """Build one rectangular instance of a class."""
    return MaskInstance(class_id=class_id, mask=rectangle(top, left, height, width))


def brute_force_best_total(matrix: np.ndarray) -> float:
    """Optimal assignment total, found exhaustively. Only for tiny matrices."""
    rows, columns = matrix.shape
    size = min(rows, columns)
    best = 0.0
    for chosen_rows in itertools.combinations(range(rows), size):
        for chosen_columns in itertools.permutations(range(columns), size):
            total = sum(
                matrix[row, column] for row, column in zip(chosen_rows, chosen_columns, strict=True)
            )
            best = max(best, total)
    return best


# --- pairwise IoU -------------------------------------------------------------


def test_a_perfect_match_scores_one():
    mask = rectangle(5, 5, 10, 10)
    assert mask_iou(mask, mask.copy()) == 1.0


def test_disjoint_masks_score_zero():
    assert mask_iou(rectangle(0, 0, 5, 5), rectangle(20, 20, 5, 5)) == 0.0


def test_partial_overlap_is_exact_arithmetic():
    # 10x10 and 10x10 offset by 5 in one axis: intersection 5x10 = 50,
    # union = 100 + 100 - 50 = 150.
    value = mask_iou(rectangle(0, 0, 10, 10), rectangle(0, 5, 10, 10))
    assert value == pytest.approx(50 / 150)


def test_containment_is_the_area_ratio():
    # 4x4 fully inside 8x8: intersection 16, union 64.
    assert mask_iou(rectangle(0, 0, 8, 8), rectangle(2, 2, 4, 4)) == pytest.approx(16 / 64)


def test_two_empty_masks_score_zero_rather_than_one():
    empty = np.zeros(CANVAS, dtype=bool)
    assert mask_iou(empty, empty.copy()) == 0.0


def test_mismatched_canvases_are_refused():
    with pytest.raises(MaskIoUEvaluationError, match="original image canvas"):
        mask_iou(np.zeros((4, 4), dtype=bool), np.zeros((5, 5), dtype=bool))


# --- assignment ---------------------------------------------------------------


def test_an_empty_matrix_assigns_nothing():
    assert assign_maximum_total_iou(np.zeros((0, 0))) == []
    assert assign_maximum_total_iou(np.zeros((0, 3))) == []
    assert assign_maximum_total_iou(np.zeros((3, 0))) == []


def test_the_assignment_is_one_to_one():
    matrix = np.array([[0.9, 0.8], [0.7, 0.6]])
    pairs = assign_maximum_total_iou(matrix)
    assert len(pairs) == 2
    assert len({row for row, _ in pairs}) == 2
    assert len({column for _, column in pairs}) == 2


def test_the_assignment_maximises_the_total_not_the_first_pair():
    # Greedy takes (0, 0) at 0.90 and is then stuck with (1, 1) at 0.10, for
    # 1.00. The optimal assignment is (0, 1) + (1, 0) = 0.85 + 0.80 = 1.65.
    matrix = np.array([[0.90, 0.85], [0.80, 0.10]])
    pairs = assign_maximum_total_iou(matrix)
    total = sum(matrix[row, column] for row, column in pairs)
    assert pairs == [(0, 1), (1, 0)]
    assert total == pytest.approx(1.65)
    assert total > 0.90 + 0.10


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5, 6, 7])
def test_the_assignment_matches_an_exhaustive_optimum(seed: int):
    generator = np.random.default_rng(seed)
    rows = int(generator.integers(1, 6))
    columns = int(generator.integers(1, 6))
    matrix = np.round(generator.random((rows, columns)), 3)
    pairs = assign_maximum_total_iou(matrix)
    total = sum(matrix[row, column] for row, column in pairs)
    assert total == pytest.approx(brute_force_best_total(matrix))


def test_the_assignment_is_deterministic():
    matrix = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
    first = assign_maximum_total_iou(matrix)
    for _ in range(5):
        assert assign_maximum_total_iou(matrix) == first


def test_the_assignment_is_returned_in_ground_truth_order():
    matrix = np.array([[0.1, 0.9], [0.9, 0.1]])
    assert assign_maximum_total_iou(matrix) == [(0, 1), (1, 0)]


# --- per-image, per-class matching --------------------------------------------


def test_matching_never_crosses_classes():
    truth = [instance(0, 0, 0, 10, 10)]
    # Same pixels, different class: it must not match.
    predictions = [instance(1, 0, 0, 10, 10)]
    matching = match_image("img", truth, predictions)
    assert matching.pairs == ()
    assert matching.gt_counts == {0: 1}
    assert matching.prediction_counts == {1: 1}


def test_a_same_class_overlap_matches():
    truth = [instance(2, 0, 0, 10, 10)]
    predictions = [instance(2, 0, 5, 10, 10)]
    matching = match_image("img", truth, predictions)
    assert len(matching.pairs) == 1
    assert matching.pairs[0].class_id == 2
    assert matching.pairs[0].iou == pytest.approx(50 / 150)


def test_a_zero_overlap_assignment_is_not_reported_as_a_match():
    truth = [instance(0, 0, 0, 5, 5)]
    predictions = [instance(0, 30, 30, 5, 5)]
    matching = match_image("img", truth, predictions)
    assert matching.pairs == ()
    result = accumulate([matching])
    summary = result.overall.summary()
    assert summary["matched_count"] == 0
    assert summary["unmatched_gt"] == 1
    assert summary["unmatched_predictions"] == 1
    assert summary[MATCHED_MASK_IOU_MEAN] is None


def test_two_ground_truth_one_prediction_leaves_one_unmatched():
    truth = [instance(0, 0, 0, 10, 10), instance(0, 20, 20, 10, 10)]
    predictions = [instance(0, 0, 0, 10, 10)]
    matching = match_image("img", truth, predictions)
    assert len(matching.pairs) == 1
    summary = accumulate([matching]).overall.summary()
    assert summary["gt_count"] == 2
    assert summary["prediction_count"] == 1
    assert summary["matched_count"] == 1
    assert summary["unmatched_gt"] == 1
    assert summary["unmatched_predictions"] == 0


def test_one_ground_truth_two_predictions_leaves_one_unmatched():
    truth = [instance(0, 0, 0, 10, 10)]
    predictions = [instance(0, 0, 0, 10, 10), instance(0, 0, 2, 10, 10)]
    matching = match_image("img", truth, predictions)
    assert len(matching.pairs) == 1
    # The one-to-one rule must keep the better of the two, not both.
    assert matching.pairs[0].iou == 1.0
    summary = accumulate([matching]).overall.summary()
    assert summary["matched_count"] == 1
    assert summary["unmatched_predictions"] == 1


def test_classes_are_matched_independently():
    truth = [instance(0, 0, 0, 10, 10), instance(3, 20, 20, 10, 10)]
    predictions = [instance(3, 20, 20, 10, 10), instance(0, 0, 0, 10, 10)]
    matching = match_image("img", truth, predictions)
    assert {pair.class_id for pair in matching.pairs} == {0, 3}
    assert all(pair.iou == 1.0 for pair in matching.pairs)


# --- aggregation --------------------------------------------------------------


def test_unmatched_ground_truth_contributes_zero_to_the_normalized_iou():
    # One perfect match and one entirely missed instance: the mean over matched
    # pairs is 1.0, but the ground-truth-normalised figure is 0.5.
    truth = [instance(0, 0, 0, 10, 10), instance(0, 20, 20, 10, 10)]
    predictions = [instance(0, 0, 0, 10, 10)]
    summary = accumulate([match_image("img", truth, predictions)]).overall.summary()
    assert summary[MATCHED_MASK_IOU_MEAN] == pytest.approx(1.0)
    assert summary[GT_NORMALIZED_MASK_IOU] == pytest.approx(0.5)
    assert summary[GT_MATCH_COVERAGE] == pytest.approx(0.5)


def test_the_coverage_thresholds_are_inclusive_at_the_boundary():
    # 10x10 against 10x10 offset so that IoU is exactly 100/175... build one at
    # exactly 0.5 instead: intersection 50, union 100 -> use nested rectangles.
    truth = [MaskInstance(0, rectangle(0, 0, 10, 10))]
    predictions = [MaskInstance(0, rectangle(0, 0, 5, 10))]  # 50 / 100 = 0.5
    summary = accumulate([match_image("img", truth, predictions)]).overall.summary()
    assert summary[MATCHED_MASK_IOU_MEAN] == pytest.approx(0.5)
    assert summary[GT_IOU50_COVERAGE] == pytest.approx(1.0)
    assert summary[GT_IOU75_COVERAGE] == pytest.approx(0.0)


def test_per_class_aggregation_keeps_classes_apart():
    truth = [instance(0, 0, 0, 10, 10), instance(4, 20, 20, 10, 10)]
    predictions = [instance(0, 0, 0, 10, 10)]
    result = accumulate([match_image("img", truth, predictions)])
    rendered = result.as_dict({0: "helmet_loose", 4: "vest_on_body"})
    assert rendered["per_class"]["helmet_loose"][GT_NORMALIZED_MASK_IOU] == pytest.approx(1.0)
    assert rendered["per_class"]["vest_on_body"][GT_NORMALIZED_MASK_IOU] == pytest.approx(0.0)
    assert rendered["per_class"]["vest_on_body"][MATCHED_MASK_IOU_MEAN] is None
    assert rendered["global"]["gt_count"] == 2


def test_diagnostics_accumulate_across_images():
    first = ("a", [instance(0, 0, 0, 10, 10)], [instance(0, 0, 0, 10, 10)])
    second = ("b", [instance(0, 0, 0, 10, 10)], [])
    result = evaluate([first, second])
    summary = result.overall.summary()
    assert result.images == 2
    assert summary["gt_count"] == 2
    assert summary["matched_count"] == 1
    assert summary[GT_NORMALIZED_MASK_IOU] == pytest.approx(0.5)


def test_an_image_with_no_ground_truth_still_counts_its_predictions():
    result = evaluate([("negative", [], [instance(0, 0, 0, 10, 10)])])
    summary = result.overall.summary()
    assert summary["gt_count"] == 0
    assert summary["prediction_count"] == 1
    assert summary["unmatched_predictions"] == 1
    assert summary[GT_NORMALIZED_MASK_IOU] is None


def test_the_iou_matrix_has_one_entry_per_pair():
    truth = [instance(0, 0, 0, 10, 10), instance(0, 20, 20, 10, 10)]
    predictions = [instance(0, 0, 0, 10, 10)]
    matrix = iou_matrix(truth, predictions)
    assert matrix.shape == (2, 1)
    assert matrix[0, 0] == 1.0
    assert matrix[1, 0] == 0.0


# --- the frozen protocol -------------------------------------------------------


@pytest.fixture(scope="module")
def paths() -> ProjectPaths:
    return ProjectPaths.from_root()


@pytest.fixture(scope="module")
def raw(paths: ProjectPaths) -> dict[str, Any]:
    return yaml.safe_load((paths.configs / CONFIG_NAME).read_text(encoding="utf-8"))


def write(tmp_path: Path, document: dict[str, Any]) -> Path:
    destination = tmp_path / CONFIG_NAME
    destination.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return destination


def test_the_committed_protocol_parses(paths: ProjectPaths):
    from construction_safety_vision.mask_iou_evaluation import load_mask_iou_config

    config = load_mask_iou_config(paths.configs / CONFIG_NAME)
    assert config["protocol"] == PROTOCOL_NAME
    assert config["experiment"] == "S0"
    assert config["split"] == "validation"


def test_the_frozen_operating_point(paths: ProjectPaths):
    from construction_safety_vision.mask_iou_evaluation import load_mask_iou_config

    inference = load_mask_iou_config(paths.configs / CONFIG_NAME).inference
    assert inference["imgsz"] == 768
    assert inference["conf"] == 0.25
    assert inference["iou"] == 0.70
    assert inference["max_det"] == 300
    assert inference["retina_masks"] is True
    assert inference["augment"] is False


def test_ground_truth_is_the_canonical_coco_document(paths: ProjectPaths):
    from construction_safety_vision.mask_iou_evaluation import load_mask_iou_config

    config = load_mask_iou_config(paths.configs / CONFIG_NAME)
    assert config["ground_truth_source"] == GROUND_TRUTH_SOURCE
    assert "adapter" not in config["ground_truth_document"]
    assert config["ground_truth_document"].endswith("segmentation_validation.coco.json")


def test_the_matching_rule_is_the_frozen_one(paths: ProjectPaths):
    from construction_safety_vision.mask_iou_evaluation import load_mask_iou_config

    matching = load_mask_iou_config(paths.configs / CONFIG_NAME)["matching"]
    assert matching["algorithm"] == MATCHING_ALGORITHM
    assert matching["implementation"] == "scipy.optimize.linear_sum_assignment"
    assert matching["scope"] == "PER_IMAGE_PER_CLASS_ONE_TO_ONE"
    assert matching["deterministic"] is True


def test_the_headline_diagnostics_are_frozen(paths: ProjectPaths):
    from construction_safety_vision.mask_iou_evaluation import load_mask_iou_config

    diagnostics = load_mask_iou_config(paths.configs / CONFIG_NAME)["diagnostics"]
    assert tuple(diagnostics["headline"]) == HEADLINE_DIAGNOSTICS
    assert tuple(diagnostics["coverage_thresholds"]) == COVERAGE_THRESHOLDS
    assert diagnostics["combined_score"] is False


def test_the_protocol_fingerprint_is_stable(paths: ProjectPaths):
    from construction_safety_vision.mask_iou_evaluation import load_mask_iou_config

    first = load_mask_iou_config(paths.configs / CONFIG_NAME)
    second = load_mask_iou_config(paths.configs / CONFIG_NAME)
    assert first.fingerprint() == second.fingerprint()


def test_naming_the_holdout_is_refused(tmp_path: Path, raw: dict[str, Any]):
    from construction_safety_vision.mask_iou_evaluation import load_mask_iou_config

    document = dict(raw)
    document["split"] = "test"
    with pytest.raises(ConfigError, match="split must be"):
        load_mask_iou_config(write(tmp_path, document))


def test_a_holdout_reference_anywhere_is_refused(tmp_path: Path, raw: dict[str, Any]):
    from construction_safety_vision.mask_iou_evaluation import load_mask_iou_config

    document = dict(raw)
    document["test_policy"] = "test"
    with pytest.raises(ConfigError, match="protected split"):
        load_mask_iou_config(write(tmp_path, document))


def test_scoring_against_the_adapter_is_refused(tmp_path: Path, raw: dict[str, Any]):
    from construction_safety_vision.mask_iou_evaluation import load_mask_iou_config

    document = dict(raw)
    document["ground_truth_document"] = "data/processed/adapters/yolo_segmentation_audit/labels/val"
    with pytest.raises(ConfigError, match="points at an adapter"):
        load_mask_iou_config(write(tmp_path, document))


def test_a_different_ground_truth_source_is_refused(tmp_path: Path, raw: dict[str, Any]):
    from construction_safety_vision.mask_iou_evaluation import load_mask_iou_config

    document = dict(raw)
    document["ground_truth_source"] = "YOLO_SEGMENTATION_ADAPTER"
    with pytest.raises(ConfigError, match="ground_truth_source must be"):
        load_mask_iou_config(write(tmp_path, document))


def test_a_swept_threshold_is_refused_by_the_schema(tmp_path: Path, raw: dict[str, Any]):
    from construction_safety_vision.mask_iou_evaluation import load_mask_iou_config

    document = dict(raw)
    document["inference"] = dict(raw["inference"])
    document["inference"]["conf_sweep"] = [0.1, 0.25, 0.5]
    with pytest.raises(ConfigError, match="unknown key"):
        load_mask_iou_config(write(tmp_path, document))


def test_test_time_augmentation_is_refused(tmp_path: Path, raw: dict[str, Any]):
    from construction_safety_vision.mask_iou_evaluation import load_mask_iou_config

    document = dict(raw)
    document["inference"] = dict(raw["inference"])
    document["inference"]["augment"] = True
    with pytest.raises(ConfigError, match="augment must be false"):
        load_mask_iou_config(write(tmp_path, document))


def test_a_combined_score_is_refused(tmp_path: Path, raw: dict[str, Any]):
    from construction_safety_vision.mask_iou_evaluation import load_mask_iou_config

    document = dict(raw)
    document["diagnostics"] = dict(raw["diagnostics"])
    document["diagnostics"]["combined_score"] = True
    with pytest.raises(ConfigError, match="combined_score must be false"):
        load_mask_iou_config(write(tmp_path, document))


def test_a_changed_matching_rule_is_refused(tmp_path: Path, raw: dict[str, Any]):
    from construction_safety_vision.mask_iou_evaluation import load_mask_iou_config

    document = dict(raw)
    document["matching"] = dict(raw["matching"])
    document["matching"]["algorithm"] = "GREEDY_DESCENDING_IOU"
    with pytest.raises(ConfigError, match=re.escape("matching.algorithm must be")):
        load_mask_iou_config(write(tmp_path, document))
