"""Synthetic protocol tests and committed validation-only delivery evidence checks."""

import ast
import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from construction_safety_vision.final_holdout_execution import (
    GroundTruthInstance,
    PredictedInstance,
    object_level_outcomes,
)
from construction_safety_vision.qualitative_gallery import (
    CHECKPOINTS,
    CLASSES,
    CONFIG,
    NO_EXAMPLE,
    REPORT,
    choose_examples,
    choose_hero,
    choose_masks,
    load_policy,
    mask_category,
    match_objects,
    validate_rows,
)
from construction_safety_vision.qualitative_gallery_report import validate_gallery

ROOT = Path(__file__).resolve().parents[1]


def test_framework_directory_source_respects_batch_one(tmp_path):
    from PIL import Image
    from ultralytics.data.build import load_inference_source

    for index in range(3):
        Image.new("RGB", (16, 16)).save(tmp_path / f"synthetic_{index}.png")
    dataset = load_inference_source(str(tmp_path), batch=1)
    batches = list(dataset)
    assert len(batches) == 3
    assert all(len(batch[0]) == 1 for batch in batches)


def gt(identifier, *, image="synthetic-a", label="person", box=(0, 0, 10, 10), area=100):
    return dict(
        annotation_id=identifier,
        image_id=image,
        split="validation",
        box=list(box),
        area=area,
        **{"class": label},
    )


def pred(index, *, image="synthetic-a", label="person", box=(0, 0, 10, 10), score=0.9):
    return dict(
        prediction_index=index,
        image_id=image,
        split="validation",
        box=list(box),
        score=score,
        **{"class": label},
    )


@pytest.mark.parametrize(
    "mutation",
    [
        {"split": "test"},
        {"image_id": "not-validation"},
        {"class": "object"},
        {"score": 0.249},
        {"box": [0, 0, 0, 10]},
    ],
)
def test_only_valid_validation_predictions_are_admitted(mutation):
    row = pred(0)
    row.update(mutation)
    with pytest.raises(ValueError):
        validate_rows([row], {"synthetic-a"})


def test_matching_is_class_aware_and_one_to_one_and_includes_empty_images():
    truth = [gt(1), gt(2, label="vest_loose", box=(20, 0, 30, 10))]
    predictions = [
        pred(0),
        pred(1, score=0.8),
        pred(2, label="person", box=(20, 0, 30, 10)),
        pred(0, image="empty-scene"),
    ]
    rows = match_objects(truth, predictions, {"synthetic-a", "empty-scene"})
    assert sum(r["category"] == "TRUE_POSITIVE" for r in rows) == 1
    assert sum(r["category"] == "FALSE_POSITIVE" for r in rows) == 3
    fn = next(r for r in rows if r["category"] == "FALSE_NEGATIVE")
    assert fn["annotation_id"] == 2 and fn["classification_mismatch"] is True


def test_ties_choose_lowest_prediction_index_then_canonical_gt_id():
    rows = match_objects([gt(9), gt(3)], [pred(8), pred(2)], {"synthetic-a"})
    pairs = {r["prediction_index"]: r["annotation_id"] for r in rows}
    assert pairs == {2: 3, 8: 9}


@pytest.mark.parametrize("seed", range(12))
def test_delivery_matcher_matches_authoritative_loop_on_synthetic_inputs(seed):
    rng = random.Random(seed)
    truth = [gt(i, label=rng.choice(CLASSES), box=(i % 3, 0, i % 3 + 10, 10)) for i in range(12)]
    predictions = [
        pred(
            i,
            label=rng.choice(CLASSES),
            score=rng.choice([0.25, 0.5, 0.9]),
            box=(i % 4, 0, i % 4 + 10, 10),
        )
        for i in range(16)
    ]
    actual = match_objects(truth, predictions, {"synthetic-a"})
    reference = object_level_outcomes(
        {
            "synthetic-a": [
                PredictedInstance(p["image_id"], p["class"], p["score"], tuple(p["box"]))
                for p in predictions
            ]
        },
        {
            "synthetic-a": [
                GroundTruthInstance(g["annotation_id"], g["image_id"], g["class"], tuple(g["box"]))
                for g in truth
            ]
        },
        SimpleNamespace(image_order=["synthetic-a"]),
        with_predicted_masks=False,
    )
    assert {
        (r["annotation_id"], r["prediction_index"])
        for r in actual
        if r["category"] == "TRUE_POSITIVE"
    } == {
        (r.instance.annotation_id, r.prediction_index)
        for r in reference["_outcomes"]
        if r.prediction_index >= 0
    }
    assert {r["prediction_index"] for r in actual if r["category"] == "FALSE_POSITIVE"} == {
        r["prediction_index"] for r in reference["_false_positives"]
    }


def test_selection_is_deterministic_and_never_fabricates_missing_class_errors():
    rows = match_objects(
        [gt(4, area=20), gt(2, image="synthetic-b", area=20)],
        [pred(5, box=(50, 50, 60, 60)), pred(1, image="synthetic-b", box=(50, 50, 60, 60))],
        {"synthetic-a", "synthetic-b"},
    )
    a = choose_examples(rows)
    assert a == choose_examples(list(reversed(rows)))
    person = [r for r in a if r["class"] == "person"]
    assert person[0]["example"]["prediction_index"] == 1
    assert person[1]["example"]["annotation_id"] == 2
    assert all(r["status"] == NO_EXAMPLE for r in a if r["class"] != "person")


@pytest.mark.parametrize(
    ("iou", "area", "expected"),
    [
        (0.75, 0, "GOOD_MASK_MATCH"),
        (0.5, -0.26, "MASK_UNDER_COVERAGE"),
        (0.5, 0.26, "MASK_OVER_COVERAGE"),
        (0.5, 0.25, None),
        (0, 1, None),
    ],
)
def test_mask_display_labels_use_existing_bands(iou, area, expected):
    assert mask_category(iou, area) == expected


def test_mask_and_hero_ties_are_stable_and_hero_requires_both_models():
    good = dict(
        pred(0, label="helmet_on_head"),
        category="TRUE_POSITIVE",
        annotation_id=5,
        gt_area=10000,
        mask_iou=0.9,
        mask_category="GOOD_MASK_MATCH",
        mask_box_fill=0.6,
    )
    other = dict(good, annotation_id=4, prediction_index=1)
    assert choose_masks([good, other])[0]["example"] == other
    policy = load_policy(ROOT / CONFIG)["hero"]
    images = {"synthetic-a": {"width": 1000, "height": 1000}}
    records = {"D2": [dict(good)], "S1": [good, other]}
    assert choose_hero(records, images, policy)["example"]["annotation_id"] == 5
    assert choose_hero({"D2": [], "S1": [good]}, images, policy)["status"] == NO_EXAMPLE


@pytest.mark.parametrize(
    "section", ["root", "inference", "matching", "hero", "selection", "render"]
)
def test_policy_rejects_unknown_keys(tmp_path, section):
    policy = load_policy(ROOT / CONFIG)
    if section == "root":
        policy["unexpected"] = True
    else:
        policy[section]["unexpected"] = True
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump(policy), encoding="utf-8")
    with pytest.raises(ValueError):
        load_policy(path)


def test_gallery_production_code_has_no_holdout_execution_import_or_access():
    for relative in (
        "scripts/build_qualitative_gallery.py",
        "scripts/validate_qualitative_gallery.py",
        "src/construction_safety_vision/qualitative_gallery.py",
        "src/construction_safety_vision/qualitative_gallery_report.py",
    ):
        tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert "final_holdout" not in (node.module or "")
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "image_ids"
            ):
                assert isinstance(node.args[0], ast.Constant) and node.args[0].value == "validation"


@pytest.mark.skipif(not (ROOT / REPORT).exists(), reason="Gallery has not been built yet")
def test_committed_gallery_provenance_membership_attribution_and_science_integrity():
    assert validate_gallery(ROOT) == []
    payload = json.loads((ROOT / REPORT).read_text(encoding="utf-8"))
    assert payload["population_images"] == 65
    assert payload["population_annotations"] == 304
    assert payload["holdout_content_accessed"] is False
    assert payload["headline_metrics_computed"] is False
    assert payload["scientific_results_modified"] is False
    assert payload["attribution"]["license"] == "CC BY 4.0"
    for model, sha in CHECKPOINTS.items():
        assert payload["execution"]["models"][model]["checkpoint_sha256"] == sha


def test_selection_excludes_higher_confidence_true_positives():
    records = match_objects(
        [gt(1)],
        [pred(0, score=0.95), pred(1, score=0.8), pred(2, score=0.7)],
        {"synthetic-a"},
    )
    selected = next(
        s
        for s in choose_examples(records)
        if s["class"] == "person" and s["category"] == "FALSE_POSITIVE"
    )
    assert selected["available"] == 2
    assert selected["example"]["prediction_index"] == 1
