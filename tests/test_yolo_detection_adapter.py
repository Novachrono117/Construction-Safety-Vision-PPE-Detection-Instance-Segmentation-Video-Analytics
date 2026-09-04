"""Tests for the YOLO detection adapter and the D0 baseline protocol.

Offline and CPU-only by construction: nothing here imports torch or ultralytics,
loads a model or touches a GPU. The runtime and smoke-test checks live in
`scripts/detection_runtime_check.py`, which is a preflight rather than a unit
test, because a test suite that silently passes when no GPU is present would be
worse than no check at all.
"""

from __future__ import annotations

import json

import pytest

from construction_safety_vision.config import ConfigError
from construction_safety_vision.data.yolo_detection_adapter import (
    ADAPTER_ROOT_PLACEHOLDER,
    ADAPTER_TYPE,
    NOT_MATERIALIZED,
    TEST_DISABLED,
    YoloAdapterError,
    YoloBox,
    coco_to_yolo,
    dataset_yaml,
    format_label_line,
    label_fingerprint,
    label_text,
    load_adapter_config,
    parse_label_line,
    round_trip_boxes,
    validate_normalised,
    yolo_to_coco,
)
from construction_safety_vision.experiment import (
    ALLOWED_PRIMARY_METRICS,
    DEVELOPMENT_SPLITS,
    ExperimentConfigError,
    load_detection_baseline_config,
)
from construction_safety_vision.paths import ProjectPaths

CLASS_MAP = {
    "helmet_loose": 0,
    "helmet_on_head": 1,
    "person": 2,
    "vest_loose": 3,
    "vest_on_body": 4,
}
FROZEN_CLASS_MAP_SHA256 = "596dab5b5756e3d4253924f84bf46d37a87d80b6ec830da781a08a93cd823753"
EPSILON = 1e-9


# --- bbox conversion -----------------------------------------------------------


def test_conversion_matches_the_documented_formula():
    box = coco_to_yolo([10.0, 20.0, 40.0, 60.0], class_id=2, image_width=100, image_height=200)
    assert box.class_id == 2
    assert box.centre_x == pytest.approx((10 + 40 / 2) / 100)
    assert box.centre_y == pytest.approx((20 + 60 / 2) / 200)
    assert box.width == pytest.approx(40 / 100)
    assert box.height == pytest.approx(60 / 200)


def test_conversion_round_trips_exactly_for_representable_values():
    original = [10.0, 20.0, 40.0, 60.0]
    box = coco_to_yolo(original, class_id=0, image_width=100, image_height=200)
    assert yolo_to_coco(box, image_width=100, image_height=200) == pytest.approx(original)


@pytest.mark.parametrize(
    ("bbox", "width", "height"),
    [
        ([0.0, 0.0, 1880.0, 1253.0], 1880, 1253),
        ([1422.0, 826.0, 112.0, 423.0], 1880, 1253),
        ([0.5, 0.25, 3.125, 7.0625], 640, 640),
        ([1879.0, 1252.0, 1.0, 1.0], 1880, 1253),
    ],
)
def test_round_trip_is_within_a_micropixel(bbox, width, height):
    box = coco_to_yolo(bbox, class_id=1, image_width=width, image_height=height)
    decoded = yolo_to_coco(box, image_width=width, image_height=height)
    assert max(abs(a - b) for a, b in zip(decoded, bbox, strict=True)) < 1e-6


def test_conversion_rejects_a_degenerate_box():
    with pytest.raises(YoloAdapterError, match="non-positive extent"):
        coco_to_yolo([10.0, 10.0, 0.0, 5.0], class_id=0, image_width=100, image_height=100)


def test_conversion_rejects_a_degenerate_canvas():
    with pytest.raises(YoloAdapterError, match="canvas"):
        coco_to_yolo([1.0, 1.0, 2.0, 2.0], class_id=0, image_width=0, image_height=100)


def test_conversion_rejects_a_malformed_box():
    with pytest.raises(YoloAdapterError, match="four numeric"):
        coco_to_yolo([1.0, 2.0, 3.0], class_id=0, image_width=10, image_height=10)


# --- normalised bounds ----------------------------------------------------------


def test_a_box_inside_the_canvas_is_valid():
    box = coco_to_yolo([10.0, 10.0, 20.0, 20.0], class_id=0, image_width=100, image_height=100)
    assert validate_normalised(box, epsilon=EPSILON) == []


def test_a_full_canvas_box_is_valid():
    box = coco_to_yolo([0.0, 0.0, 100.0, 100.0], class_id=0, image_width=100, image_height=100)
    assert validate_normalised(box, epsilon=EPSILON) == []


def test_a_box_leaving_the_canvas_is_reported():
    box = YoloBox(class_id=0, centre_x=0.95, centre_y=0.5, width=0.2, height=0.2)
    problems = validate_normalised(box, epsilon=EPSILON)
    assert any("leaves the unit square" in problem for problem in problems)


def test_a_negative_coordinate_is_reported():
    box = YoloBox(class_id=0, centre_x=-0.1, centre_y=0.5, width=0.2, height=0.2)
    assert any("outside [0, 1]" in problem for problem in validate_normalised(box, epsilon=EPSILON))


def test_a_zero_extent_box_is_reported():
    box = YoloBox(class_id=0, centre_x=0.5, centre_y=0.5, width=0.0, height=0.2)
    assert any("non-positive" in problem for problem in validate_normalised(box, epsilon=EPSILON))


# --- serialisation ---------------------------------------------------------------


def test_label_line_is_fixed_point_and_deterministic():
    box = coco_to_yolo([10.0, 20.0, 40.0, 60.0], class_id=3, image_width=100, image_height=200)
    line = format_label_line(box, precision=9)
    assert line == "3 0.300000000 0.250000000 0.400000000 0.300000000"
    assert format_label_line(box, precision=9) == line


def test_label_line_parses_back():
    box = coco_to_yolo([10.0, 20.0, 40.0, 60.0], class_id=3, image_width=100, image_height=200)
    parsed = parse_label_line(format_label_line(box, precision=9))
    assert parsed.class_id == 3
    assert parsed.centre_x == pytest.approx(box.centre_x, abs=1e-9)


def test_parse_rejects_a_wrong_field_count():
    with pytest.raises(YoloAdapterError, match="5 fields"):
        parse_label_line("2 0.1 0.2 0.3")


def test_parse_rejects_non_numeric_content():
    with pytest.raises(YoloAdapterError, match="not numeric"):
        parse_label_line("2 0.1 0.2 0.3 banana")


def test_label_order_follows_the_order_given():
    boxes = [
        YoloBox(2, 0.1, 0.1, 0.1, 0.1),
        YoloBox(0, 0.2, 0.2, 0.2, 0.2),
        YoloBox(4, 0.3, 0.3, 0.3, 0.3),
    ]
    text = label_text(boxes, precision=9)
    assert [int(line.split()[0]) for line in text.splitlines()] == [2, 0, 4]


def test_label_text_is_byte_stable():
    boxes = [YoloBox(2, 0.1, 0.1, 0.1, 0.1)]
    assert label_text(boxes, precision=9) == label_text(boxes, precision=9)
    assert label_text(boxes, precision=9).endswith("\n")


def test_zero_instance_image_yields_an_empty_label_file():
    assert label_text([], precision=9) == ""


def test_label_fingerprint_moves_with_content():
    first = {"a": "2 0.1 0.1 0.1 0.1\n", "b": ""}
    second = {"a": "2 0.1 0.1 0.1 0.2\n", "b": ""}
    assert label_fingerprint(first) == label_fingerprint(dict(reversed(list(first.items()))))
    assert label_fingerprint(first) != label_fingerprint(second)


def test_label_fingerprint_notices_a_lost_negative():
    with_negative = {"a": "2 0.1 0.1 0.1 0.1\n", "b": ""}
    without = {"a": "2 0.1 0.1 0.1 0.1\n"}
    assert label_fingerprint(with_negative) != label_fingerprint(without)


# --- round-trip audit -------------------------------------------------------------


def test_round_trip_audit_passes_on_a_clean_conversion():
    bbox = [10.0, 20.0, 40.0, 60.0]
    box = coco_to_yolo(bbox, class_id=0, image_width=100, image_height=200)
    result = round_trip_boxes([("img#1", bbox, box, 100, 200)], tolerance_px=1e-4)
    assert result.checked == 1
    assert result.within_tolerance == 1
    assert result.mismatches == []


def test_round_trip_audit_catches_a_corrupted_label():
    bbox = [10.0, 20.0, 40.0, 60.0]
    corrupted = YoloBox(class_id=0, centre_x=0.9, centre_y=0.25, width=0.4, height=0.3)
    result = round_trip_boxes([("img#1", bbox, corrupted, 100, 200)], tolerance_px=1e-4)
    assert result.within_tolerance == 0
    assert result.mismatches


# --- dataset descriptor ------------------------------------------------------------


def test_dataset_yaml_has_no_test_key():
    text = dataset_yaml(
        path_value="/tmp/adapter",
        train_directory="images/train",
        validation_directory="images/val",
        class_map=CLASS_MAP,
    )
    assert "\ntest:" not in text
    assert "\ntrain: images/train\n" in text
    assert "\nval: images/val\n" in text


def test_dataset_yaml_uses_the_frozen_class_order():
    text = dataset_yaml(
        path_value=".",
        train_directory="images/train",
        validation_directory="images/val",
        class_map=CLASS_MAP,
    )
    names = [line.strip() for line in text.splitlines() if line.startswith("  ")]
    assert names == [
        "0: helmet_loose",
        "1: helmet_on_head",
        "2: person",
        "3: vest_loose",
        "4: vest_on_body",
    ]
    assert "object" not in text


def test_committed_template_matches_the_generator():
    paths = ProjectPaths.from_root()
    template = (paths.configs / "detection_dataset.template.yaml").read_text(encoding="utf-8")
    population = json.loads(
        (paths.reports / "canonical_modeling_manifest.json").read_text(encoding="utf-8")
    )
    assert template == dataset_yaml(
        path_value=ADAPTER_ROOT_PLACEHOLDER,
        train_directory="images/train",
        validation_directory="images/val",
        class_map=population["class_map"],
    )


# --- adapter configuration ----------------------------------------------------------


def test_committed_adapter_config_parses():
    config = load_adapter_config(ProjectPaths.from_root().configs / "detection_adapter.yaml")
    assert config.adapter_type == ADAPTER_TYPE
    assert config.test_adapter == TEST_DISABLED
    assert config.bbox_source == "CANONICAL_COCO_DETECTION_BBOX"
    assert config.image_copy_mode == "BINARY_IDENTICAL"
    assert config.split_directories == {"train": "train", "validation": "val"}
    assert "test" not in config.split_directories


def test_adapter_config_fingerprint_is_stable():
    path = ProjectPaths.from_root().configs / "detection_adapter.yaml"
    assert load_adapter_config(path).fingerprint() == load_adapter_config(path).fingerprint()


def test_adapter_config_rejects_a_test_split_directory(tmp_path):
    path = ProjectPaths.from_root().configs / "detection_adapter.yaml"
    edited = tmp_path / "detection_adapter.yaml"
    edited.write_text(
        path.read_text(encoding="utf-8").replace(
            "split_directories:\n  train: train\n  validation: val",
            "split_directories:\n  train: train\n  validation: val\n  test: test",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="protected split"):
        load_adapter_config(edited)


def test_adapter_config_rejects_enabling_the_test_adapter(tmp_path):
    path = ProjectPaths.from_root().configs / "detection_adapter.yaml"
    edited = tmp_path / "detection_adapter.yaml"
    edited.write_text(
        path.read_text(encoding="utf-8").replace(
            f"test_adapter: {TEST_DISABLED}", "test_adapter: enabled"
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="test_adapter"):
        load_adapter_config(edited)


def test_adapter_config_rejects_an_unknown_key(tmp_path):
    path = ProjectPaths.from_root().configs / "detection_adapter.yaml"
    edited = tmp_path / "detection_adapter.yaml"
    edited.write_text(path.read_text(encoding="utf-8") + "\nsurprise: true\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown key"):
        load_adapter_config(edited)


# --- D0 baseline protocol ------------------------------------------------------------


def test_committed_baseline_protocol_parses():
    config = load_detection_baseline_config(
        ProjectPaths.from_root().configs / "detection_baseline.yaml"
    )
    assert config.experiment_id == "D0"
    assert config.task == "detection"
    assert config.model == "YOLO11n"
    assert config.weight_identifier == "yolo11n.pt"
    assert config.weights == "pretrained"
    assert config.seed == 42
    assert config.splits == DEVELOPMENT_SPLITS
    assert config.training["imgsz"] == 640
    assert config.training["batch"] == 16
    assert config.device_requirement == "CUDA_GPU_REQUIRED"


def test_baseline_protocol_declares_the_metric_hierarchy():
    config = load_detection_baseline_config(
        ProjectPaths.from_root().configs / "detection_baseline.yaml"
    )
    assert config.metrics.primary == "mAP@0.50:0.95"
    assert config.metrics.primary in ALLOWED_PRIMARY_METRICS
    assert set(config.metrics.secondary) == {"mAP@0.50", "precision", "recall"}
    assert "AP@0.50" in config.metrics.per_class
    assert "confusion_matrix" in config.metrics.artifacts


def test_baseline_protocol_names_a_checkpoint_rule_before_training():
    config = load_detection_baseline_config(
        ProjectPaths.from_root().configs / "detection_baseline.yaml"
    )
    assert "ULTRALYTICS_BEST_ON_VALIDATION_FITNESS" in config.checkpoint_selection


def test_baseline_protocol_records_the_rare_class_limitation():
    config = load_detection_baseline_config(
        ProjectPaths.from_root().configs / "detection_baseline.yaml"
    )
    assert config.rare_class == "vest_loose"
    limitation = config.rare_class_limitation
    assert "1 validation source image" in limitation
    assert "8 instances" in limitation
    assert "do not tune" in limitation.lower()


def test_baseline_protocol_fingerprint_is_stable():
    path = ProjectPaths.from_root().configs / "detection_baseline.yaml"
    first = load_detection_baseline_config(path)
    assert first.fingerprint() == load_detection_baseline_config(path).fingerprint()


def test_baseline_protocol_never_names_the_holdout(tmp_path):
    path = ProjectPaths.from_root().configs / "detection_baseline.yaml"
    edited = tmp_path / "detection_baseline.yaml"
    edited.write_text(
        path.read_text(encoding="utf-8").replace(
            "splits:\n  - train\n  - validation",
            "splits:\n  - train\n  - validation\n  - test",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ExperimentConfigError, match="never names the holdout"):
        load_detection_baseline_config(edited)


def test_baseline_protocol_rejects_a_holdout_reference_anywhere(tmp_path):
    path = ProjectPaths.from_root().configs / "detection_baseline.yaml"
    edited = tmp_path / "detection_baseline.yaml"
    edited.write_text(
        path.read_text(encoding="utf-8").replace(
            "  val: true", "  val: true\n  extra_eval_split: test"
        ),
        encoding="utf-8",
    )
    with pytest.raises(ExperimentConfigError):
        load_detection_baseline_config(edited)


@pytest.mark.parametrize("identifier", ["baseline", "0D", "d-zero", "", "D"])
def test_baseline_protocol_rejects_a_malformed_experiment_id(tmp_path, identifier):
    path = ProjectPaths.from_root().configs / "detection_baseline.yaml"
    edited = tmp_path / "detection_baseline.yaml"
    edited.write_text(
        path.read_text(encoding="utf-8").replace(
            "experiment_id: D0", f"experiment_id: {identifier!r}"
        ),
        encoding="utf-8",
    )
    with pytest.raises(ExperimentConfigError, match="not a valid identifier"):
        load_detection_baseline_config(edited)


def test_baseline_protocol_rejects_an_invented_primary_metric(tmp_path):
    path = ProjectPaths.from_root().configs / "detection_baseline.yaml"
    edited = tmp_path / "detection_baseline.yaml"
    edited.write_text(
        path.read_text(encoding="utf-8").replace(
            "  primary: mAP@0.50:0.95", "  primary: ppe_compliance_score"
        ),
        encoding="utf-8",
    )
    with pytest.raises(ExperimentConfigError, match="not one of"):
        load_detection_baseline_config(edited)


def test_baseline_protocol_rejects_a_missing_hyperparameter(tmp_path):
    path = ProjectPaths.from_root().configs / "detection_baseline.yaml"
    edited = tmp_path / "detection_baseline.yaml"
    edited.write_text(
        path.read_text(encoding="utf-8").replace("  weight_decay: 0.0005\n", ""),
        encoding="utf-8",
    )
    with pytest.raises(ExperimentConfigError, match="missing required key"):
        load_detection_baseline_config(edited)


def test_baseline_protocol_rejects_a_duplicated_primary_metric(tmp_path):
    path = ProjectPaths.from_root().configs / "detection_baseline.yaml"
    edited = tmp_path / "detection_baseline.yaml"
    edited.write_text(
        path.read_text(encoding="utf-8").replace(
            "  secondary:\n    - mAP@0.50", "  secondary:\n    - mAP@0.50:0.95\n    - mAP@0.50"
        ),
        encoding="utf-8",
    )
    with pytest.raises(ExperimentConfigError, match="must not repeat"):
        load_detection_baseline_config(edited)


def test_frozen_class_map_sha256_is_the_one_the_adapter_targets():
    paths = ProjectPaths.from_root()
    population = json.loads(
        (paths.reports / "canonical_modeling_manifest.json").read_text(encoding="utf-8")
    )
    assert population["fingerprints"]["class_map_sha256"] == FROZEN_CLASS_MAP_SHA256
    assert population["class_map"] == CLASS_MAP


def test_not_materialized_marker_is_shared_with_the_earlier_phases():
    assert NOT_MATERIALIZED == "NOT_MATERIALIZED_PROTECTED_HOLDOUT"
