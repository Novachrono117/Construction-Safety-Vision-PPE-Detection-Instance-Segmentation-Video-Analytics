"""Tests for the canonical COCO mask-AP evaluator.

Every fixture is synthetic, and the evaluator is exercised on hand-built
rectangles whose outcome can be reasoned about on paper. These ran before the
evaluator was ever pointed at S0, which is the point: an evaluator validated
after producing a number it was built to produce validates nothing.

The cases worth naming:

**A perfect prediction must score 1.0.** If it does not, the category ids, the
RLE encoding or the canvas convention is wrong, and every later comparison is
meaningless.

**The RLE round-trip must be pixel-exact.** The evaluator serialises predictions
and pycocotools decodes them; any loss there would be charged to the model.

**A wrong-class or disjoint prediction must score 0.0.** Otherwise the metric is
rewarding something other than agreement.

No test here trains, infers on real data, or touches the holdout.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from construction_safety_vision.canonical_evaluation import (
    ALL_CLASS_METRIC,
    BELOW,
    EQUIVALENT,
    GROUND_TRUTH_SOURCE,
    IMPROVES,
    IOU_THRESHOLDS,
    IOU_TYPE,
    MAX_DETS,
    NATIVE_METRIC_STATUS,
    PRACTICAL_EQUIVALENCE_MARGIN,
    PRIMARY_METRIC,
    PROTOCOL_NAME,
    CanonicalEvaluationError,
    classify_delta,
    decode_mask,
    encode_mask,
    evaluate_canonical,
    load_canonical_evaluation_config,
    prediction_record,
    supported_macro,
    validation_support,
    verify_prediction_geometry,
)
from construction_safety_vision.config import ConfigError
from construction_safety_vision.paths import ProjectPaths

CONFIG_NAME = "segmentation_canonical_evaluation.yaml"
CLASS_MAP = {
    "helmet_loose": 0,
    "helmet_on_head": 1,
    "person": 2,
    "vest_loose": 3,
    "vest_on_body": 4,
}
CANVAS = (60, 80)


def rectangle(top: int, left: int, height: int, width: int, canvas=CANVAS) -> np.ndarray:
    """Build a boolean mask holding one filled rectangle."""
    mask = np.zeros(canvas, dtype=bool)
    mask[top : top + height, left : left + width] = True
    return mask


def ground_truth(entries: list[tuple[int, int, np.ndarray]]) -> dict[str, Any]:
    """Build a canonical COCO document from (image_id, category_id, mask) triples."""
    from pycocotools import mask as mask_utils

    images = sorted({image_id for image_id, _, _ in entries})
    annotations = []
    for index, (image_id, category_id, mask) in enumerate(entries, start=1):
        rle = encode_mask(mask)
        payload = {"size": rle["size"], "counts": rle["counts"].encode("utf-8")}
        annotations.append(
            {
                "id": index,
                "image_id": image_id,
                "category_id": category_id,
                "segmentation": rle,
                "area": int(mask_utils.area(payload)),
                "bbox": [float(value) for value in mask_utils.toBbox(payload)],
                "iscrowd": 0,
            }
        )
    return {
        "info": {},
        "licenses": [],
        "images": [
            {
                "id": image_id,
                "file_name": f"{image_id}.jpg",
                "height": CANVAS[0],
                "width": CANVAS[1],
            }
            for image_id in images
        ],
        "categories": [
            {"id": index, "name": name, "supercategory": "ppe"}
            for name, index in sorted(CLASS_MAP.items(), key=lambda item: item[1])
        ],
        "annotations": annotations,
    }


# --- RLE serialisation --------------------------------------------------------


def test_the_rle_round_trip_is_pixel_exact():
    mask = rectangle(5, 7, 20, 26)
    mask[40:50, 3:11] = True  # a second, disjoint component
    assert np.array_equal(decode_mask(encode_mask(mask)), mask)


def test_an_empty_mask_round_trips():
    mask = np.zeros(CANVAS, dtype=bool)
    assert np.array_equal(decode_mask(encode_mask(mask)), mask)


def test_a_full_mask_round_trips():
    mask = np.ones(CANVAS, dtype=bool)
    assert np.array_equal(decode_mask(encode_mask(mask)), mask)


def test_a_mask_with_a_hole_round_trips():
    mask = rectangle(10, 10, 30, 40)
    mask[20:30, 20:30] = False
    assert np.array_equal(decode_mask(encode_mask(mask)), mask)


def test_the_encoded_size_is_the_original_canvas():
    encoded = encode_mask(rectangle(0, 0, 5, 5))
    assert encoded["size"] == [CANVAS[0], CANVAS[1]]
    assert isinstance(encoded["counts"], str)


def test_the_framework_and_reference_encoders_agree_on_orientation():
    # Ultralytics flattens with transpose(2, 1), i.e. column-major, and records
    # size as [h, w] - the same convention pycocotools uses. If that ever
    # diverged, masks would silently transpose.
    mask = rectangle(3, 11, 9, 4)
    decoded = decode_mask(encode_mask(mask))
    assert decoded.shape == mask.shape
    assert np.array_equal(decoded, mask)


# --- geometry verification ----------------------------------------------------


def test_a_prediction_on_the_wrong_canvas_is_refused():
    detection = prediction_record(
        image_id=1, category_id=2, mask=np.zeros((10, 10), dtype=bool), score=0.9
    )
    with pytest.raises(CanonicalEvaluationError, match="original canvas"):
        verify_prediction_geometry([detection], {1: CANVAS})


def test_a_prediction_for_an_unknown_image_is_refused():
    detection = prediction_record(
        image_id=99, category_id=2, mask=np.zeros(CANVAS, dtype=bool), score=0.9
    )
    with pytest.raises(CanonicalEvaluationError, match="outside the canonical document"):
        verify_prediction_geometry([detection], {1: CANVAS})


def test_a_correct_canvas_passes():
    detection = prediction_record(image_id=1, category_id=2, mask=rectangle(0, 0, 5, 5), score=0.9)
    verify_prediction_geometry([detection], {1: CANVAS})


# --- COCOeval behaviour -------------------------------------------------------


def test_a_perfect_prediction_scores_one():
    mask = rectangle(10, 10, 20, 20)
    truth = ground_truth([(1, 2, mask)])
    detections = [prediction_record(image_id=1, category_id=2, mask=mask, score=0.9)]
    result = evaluate_canonical(truth, detections, class_map=CLASS_MAP)
    assert result["all_class_map50_95"] == pytest.approx(1.0)
    assert result["all_class_map50"] == pytest.approx(1.0)
    assert result["per_class"]["person"]["AP@0.50:0.95"] == pytest.approx(1.0)


def test_a_disjoint_prediction_scores_zero():
    truth = ground_truth([(1, 2, rectangle(0, 0, 10, 10))])
    detections = [
        prediction_record(image_id=1, category_id=2, mask=rectangle(40, 60, 10, 10), score=0.9)
    ]
    result = evaluate_canonical(truth, detections, class_map=CLASS_MAP)
    assert result["all_class_map50_95"] == pytest.approx(0.0)


def test_a_wrong_class_prediction_scores_zero_for_the_true_class():
    truth = ground_truth([(1, 2, rectangle(10, 10, 20, 20))])
    detections = [
        prediction_record(image_id=1, category_id=0, mask=rectangle(10, 10, 20, 20), score=0.9)
    ]
    result = evaluate_canonical(truth, detections, class_map=CLASS_MAP)
    assert result["per_class"]["person"]["AP@0.50:0.95"] == pytest.approx(0.0)


def test_a_missing_prediction_scores_zero():
    truth = ground_truth([(1, 2, rectangle(10, 10, 20, 20))])
    result = evaluate_canonical(truth, [], class_map=CLASS_MAP)
    assert result["no_detections"] is True
    assert result["all_class_map50_95"] == pytest.approx(0.0)


def test_a_partial_overlap_scores_between_zero_and_one():
    truth = ground_truth([(1, 2, rectangle(10, 10, 20, 20))])
    # 20x20 against 20x20 offset by 6: IoU = 280 / 520 ~ 0.538, so it counts at
    # IoU 0.50 but not above.
    detections = [
        prediction_record(image_id=1, category_id=2, mask=rectangle(10, 16, 20, 20), score=0.9)
    ]
    result = evaluate_canonical(truth, detections, class_map=CLASS_MAP)
    assert 0.0 < result["all_class_map50"] <= 1.0
    assert result["all_class_map50_95"] < result["all_class_map50"]


def test_a_duplicate_prediction_does_not_double_count():
    mask = rectangle(10, 10, 20, 20)
    truth = ground_truth([(1, 2, mask)])
    single = evaluate_canonical(
        truth,
        [prediction_record(image_id=1, category_id=2, mask=mask, score=0.9)],
        class_map=CLASS_MAP,
    )
    doubled = evaluate_canonical(
        truth,
        [
            prediction_record(image_id=1, category_id=2, mask=mask, score=0.9),
            prediction_record(image_id=1, category_id=2, mask=mask, score=0.8),
        ],
        class_map=CLASS_MAP,
    )
    # The duplicate becomes a false positive, so the score cannot rise.
    assert doubled["all_class_map50_95"] <= single["all_class_map50_95"]


def test_score_ordering_matters():
    truth = ground_truth([(1, 2, rectangle(10, 10, 20, 20))])
    good = rectangle(10, 10, 20, 20)
    bad = rectangle(45, 60, 8, 8)
    correct_order = evaluate_canonical(
        truth,
        [
            prediction_record(image_id=1, category_id=2, mask=good, score=0.9),
            prediction_record(image_id=1, category_id=2, mask=bad, score=0.1),
        ],
        class_map=CLASS_MAP,
    )
    inverted = evaluate_canonical(
        truth,
        [
            prediction_record(image_id=1, category_id=2, mask=good, score=0.1),
            prediction_record(image_id=1, category_id=2, mask=bad, score=0.9),
        ],
        class_map=CLASS_MAP,
    )
    assert correct_order["all_class_map50_95"] > inverted["all_class_map50_95"]


def test_multiple_images_and_classes_are_scored_together():
    truth = ground_truth(
        [
            (1, 2, rectangle(10, 10, 20, 20)),
            (2, 0, rectangle(5, 5, 10, 10)),
            (2, 4, rectangle(30, 40, 15, 15)),
        ]
    )
    detections = [
        prediction_record(image_id=1, category_id=2, mask=rectangle(10, 10, 20, 20), score=0.9),
        prediction_record(image_id=2, category_id=0, mask=rectangle(5, 5, 10, 10), score=0.8),
        prediction_record(image_id=2, category_id=4, mask=rectangle(30, 40, 15, 15), score=0.7),
    ]
    result = evaluate_canonical(truth, detections, class_map=CLASS_MAP)
    assert result["all_class_map50_95"] == pytest.approx(1.0)
    for name in ("person", "helmet_loose", "vest_on_body"):
        assert result["per_class"][name]["AP@0.50:0.95"] == pytest.approx(1.0)
    # Classes with no ground truth report None rather than a misleading zero.
    assert result["per_class"]["vest_loose"]["AP@0.50:0.95"] is None


def test_the_evaluator_uses_the_frozen_semantics():
    truth = ground_truth([(1, 2, rectangle(10, 10, 20, 20))])
    detections = [
        prediction_record(image_id=1, category_id=2, mask=rectangle(10, 10, 20, 20), score=0.9)
    ]
    result = evaluate_canonical(truth, detections, class_map=CLASS_MAP)
    assert tuple(result["cocoeval"]["iou_thresholds"]) == IOU_THRESHOLDS
    assert tuple(result["cocoeval"]["max_dets"]) == MAX_DETS
    assert result["cocoeval"]["iou_type"] == IOU_TYPE


def test_canonical_category_ids_are_used_directly():
    # Category id 0 is legal in COCO and pycocotools handles it; the project
    # therefore applies no evaluator-only remapping, and this proves it works.
    truth = ground_truth([(1, 0, rectangle(10, 10, 20, 20))])
    detections = [
        prediction_record(image_id=1, category_id=0, mask=rectangle(10, 10, 20, 20), score=0.9)
    ]
    result = evaluate_canonical(truth, detections, class_map=CLASS_MAP)
    assert result["per_class"]["helmet_loose"]["AP@0.50:0.95"] == pytest.approx(1.0)


# --- support rule and macro ---------------------------------------------------


def test_the_support_rule_is_mechanical_and_names_no_class():
    entries = [(image, 2, rectangle(0, 0, 5, 5)) for image in range(1, 8)]
    entries += [(image, 2, rectangle(10, 10, 5, 5)) for image in range(1, 8)]
    entries += [(image, 2, rectangle(20, 20, 5, 5)) for image in range(1, 8)]
    entries += [(1, 3, rectangle(30, 30, 5, 5))]
    support = validation_support(ground_truth(entries), CLASS_MAP)
    assert support["person"]["images"] == 7
    assert support["person"]["instances"] == 21
    assert support["person"]["supported"] is True
    assert support["vest_loose"]["supported"] is False
    assert support["helmet_loose"]["supported"] is False


def test_the_macro_averages_only_admitted_classes():
    per_class = {
        "helmet_loose": {"AP@0.50:0.95": 0.8},
        "person": {"AP@0.50:0.95": 0.4},
        "vest_loose": {"AP@0.50:0.95": 0.0},
    }
    support = {
        "helmet_loose": {"supported": True},
        "person": {"supported": True},
        "vest_loose": {"supported": False},
    }
    macro = supported_macro(per_class, support)
    assert macro["admitted_classes"] == ["helmet_loose", "person"]
    assert macro["value"] == pytest.approx(0.6)
    assert macro["metric"] == PRIMARY_METRIC
    assert macro["names_no_class"] is True


def test_the_macro_ignores_a_class_with_no_score():
    per_class = {"person": {"AP@0.50:0.95": 0.5}, "helmet_loose": {"AP@0.50:0.95": None}}
    support = {"person": {"supported": True}, "helmet_loose": {"supported": True}}
    assert supported_macro(per_class, support)["value"] == pytest.approx(0.5)


# --- the selection policy -----------------------------------------------------


def test_a_gain_beyond_the_margin_improves():
    verdict = classify_delta(0.500, 0.500 + PRACTICAL_EQUIVALENCE_MARGIN + 0.001)
    assert verdict["verdict"] == IMPROVES


def test_a_loss_beyond_the_margin_is_below():
    verdict = classify_delta(0.500, 0.500 - PRACTICAL_EQUIVALENCE_MARGIN - 0.001)
    assert verdict["verdict"] == BELOW


@pytest.mark.parametrize("delta", [0.0, 0.005, -0.005, 0.004, -0.004])
def test_a_move_inside_the_margin_is_practically_equivalent(delta):
    verdict = classify_delta(0.500, 0.500 + delta)
    assert verdict["verdict"] == EQUIVALENT
    assert verdict["practically_equivalent_prefers_reference"] is True


def test_the_margin_is_not_a_significance_test():
    assert classify_delta(0.5, 0.6)["margin_is_not_a_significance_test"] is True


# --- the frozen protocol ------------------------------------------------------


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
    config = load_canonical_evaluation_config(paths.configs / CONFIG_NAME)
    assert config["protocol"] == PROTOCOL_NAME
    assert config["split"] == "validation"


def test_the_protocol_scores_against_canonical_masks(paths: ProjectPaths):
    config = load_canonical_evaluation_config(paths.configs / CONFIG_NAME)
    assert config["ground_truth_source"] == GROUND_TRUTH_SOURCE
    assert "adapter" not in config["ground_truth_document"]
    assert config["ground_truth_document"].endswith("segmentation_validation.coco.json")


def test_the_frozen_inference_settings(paths: ProjectPaths):
    inference = load_canonical_evaluation_config(paths.configs / CONFIG_NAME).inference
    assert inference["conf"] == 0.001
    assert inference["iou"] == 0.70
    assert inference["imgsz"] == 768
    assert inference["max_det"] == 300
    assert inference["augment"] is False
    assert inference["retina_masks"] is True


def test_the_frozen_cocoeval_settings(paths: ProjectPaths):
    evaluator = load_canonical_evaluation_config(paths.configs / CONFIG_NAME).cocoeval
    assert evaluator["iou_type"] == IOU_TYPE
    assert tuple(round(float(v), 2) for v in evaluator["iou_thresholds"]) == IOU_THRESHOLDS
    assert tuple(evaluator["max_dets"]) == MAX_DETS
    assert evaluator["category_id_mapping"] == "NONE_CANONICAL_IDS_USED_DIRECTLY"
    assert evaluator["mask_encoding"] == "PYCOCOTOOLS_BINARY_MASK_RLE"


def test_the_prediction_cap_and_the_metric_cap_are_recorded_separately(paths: ProjectPaths):
    config = load_canonical_evaluation_config(paths.configs / CONFIG_NAME)
    assert config.inference["max_det"] == 300
    assert tuple(config.cocoeval["max_dets"]) == MAX_DETS
    assert config.inference["max_det"] not in config.cocoeval["max_dets"]


def test_the_metric_hierarchy_is_frozen(paths: ProjectPaths):
    metrics = load_canonical_evaluation_config(paths.configs / CONFIG_NAME)["metrics"]
    assert metrics["primary"] == PRIMARY_METRIC
    assert ALL_CLASS_METRIC in metrics["all_class"]
    assert metrics["native_framework_status"] == NATIVE_METRIC_STATUS
    assert metrics["composite_score"] is False
    assert "GT_NORMALIZED_MASK_IOU" in metrics["secondary_direct_iou"]


def test_a_changed_iou_sweep_is_refused(tmp_path: Path, raw: dict[str, Any]):
    document = dict(raw)
    document["cocoeval"] = dict(raw["cocoeval"])
    document["cocoeval"]["iou_thresholds"] = [0.5, 0.75]
    with pytest.raises(ConfigError, match=re.escape("iou_thresholds must be exactly")):
        load_canonical_evaluation_config(write(tmp_path, document))


def test_changed_max_dets_are_refused(tmp_path: Path, raw: dict[str, Any]):
    document = dict(raw)
    document["cocoeval"] = dict(raw["cocoeval"])
    document["cocoeval"]["max_dets"] = [1, 10, 300]
    with pytest.raises(ConfigError, match=re.escape("max_dets must be exactly")):
        load_canonical_evaluation_config(write(tmp_path, document))


def test_an_operational_confidence_is_refused(tmp_path: Path, raw: dict[str, Any]):
    document = dict(raw)
    document["inference"] = dict(raw["inference"])
    document["inference"]["conf"] = 0.25
    with pytest.raises(ConfigError, match=re.escape("conf must be 0.001")):
        load_canonical_evaluation_config(write(tmp_path, document))


def test_adapter_ground_truth_is_refused(tmp_path: Path, raw: dict[str, Any]):
    document = dict(raw)
    document["ground_truth_document"] = "data/processed/adapters/yolo_segmentation_audit/x.json"
    with pytest.raises(ConfigError, match="points at an adapter"):
        load_canonical_evaluation_config(write(tmp_path, document))


def test_a_composite_score_is_refused(tmp_path: Path, raw: dict[str, Any]):
    document = dict(raw)
    document["metrics"] = dict(raw["metrics"])
    document["metrics"]["composite_score"] = True
    with pytest.raises(ConfigError, match="composite_score must be false"):
        load_canonical_evaluation_config(write(tmp_path, document))


def test_promoting_the_native_metric_is_refused(tmp_path: Path, raw: dict[str, Any]):
    document = dict(raw)
    document["metrics"] = dict(raw["metrics"])
    document["metrics"]["native_framework_status"] = "PRIMARY"
    with pytest.raises(ConfigError, match="native_framework_status"):
        load_canonical_evaluation_config(write(tmp_path, document))


def test_naming_the_holdout_is_refused(tmp_path: Path, raw: dict[str, Any]):
    document = dict(raw)
    document["test_policy"] = "test"
    with pytest.raises(ConfigError, match="protected split"):
        load_canonical_evaluation_config(write(tmp_path, document))


def test_the_protocol_fingerprint_is_stable(paths: ProjectPaths):
    first = load_canonical_evaluation_config(paths.configs / CONFIG_NAME)
    second = load_canonical_evaluation_config(paths.configs / CONFIG_NAME)
    assert first.fingerprint() == second.fingerprint()
    assert re.fullmatch(r"[0-9a-f]{64}", first.fingerprint())


def test_the_protocol_does_not_claim_to_predate_s0(paths: ProjectPaths):
    text = " ".join((paths.configs / CONFIG_NAME).read_text(encoding="utf-8").split())
    assert "frozen **after S0 ran and before # S1 exists**" in text or (
        "after S0 ran" in text and "before # S1 exists" in text
    )
    assert "0.407942" in text  # S0's native result is named as still valid
    assert "0.556977" in text  # so is its direct-IoU result
