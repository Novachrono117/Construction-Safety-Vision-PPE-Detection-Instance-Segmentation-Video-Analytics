"""Tests for the task-dataset materialisation primitives.

Everything here runs on synthetic fixtures built in ``tmp_path``. The tests that
exercise a fully unlocked holdout materialisation use a fixture split, never the
project's real one, so proving the guard opens when both opt-ins are present
never touches protected data.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from construction_safety_vision.data.coco_materialization import (
    DETECTION,
    POLYGON,
    RLE,
    SEGMENTATION,
    SYNTHETIC_RECTANGLE,
    CanonicalAnnotation,
    CanonicalImage,
    alignment_report,
    bbox_agreement,
    build_detection_coco,
    build_segmentation_coco,
    categories,
    decode_mask,
    derive_bbox,
    document_fingerprint,
    round_trip_geometry,
    validate_coco,
)
from construction_safety_vision.data.materialization import (
    ALREADY_PRESENT,
    COPIED,
    DEVELOPMENT_SPLITS,
    TEST_DISABLED,
    MaterializationError,
    Tolerances,
    assert_development_split,
    build_canonical_ids,
    copy_image,
    image_content_fingerprint,
    load_materialization_config,
    load_population,
    materialized_name,
    resolve_split_images,
    sha256_bytes,
)
from construction_safety_vision.data.population import PROVIDER_GEOMETRY, SYNTHETIC_GEOMETRY
from construction_safety_vision.data.split_freeze import (
    FrozenSplits,
    SplitAssignment,
    fingerprint_holdout,
    fingerprint_split_assignment,
    normalise_assignments,
    split_sections,
)
from construction_safety_vision.paths import ProjectPaths
from construction_safety_vision.splits import HOLDOUT_UNLOCK_ENV_VAR, HoldoutViolationError

CLASS_MAP = {
    "helmet_loose": 0,
    "helmet_on_head": 1,
    "person": 2,
    "vest_loose": 3,
    "vest_on_body": 4,
}
TOLERANCES = Tolerances(bbox_canvas_px=1.0, polygon_coordinate_px=0.0, bbox_agreement_px=1.0)

LOCKED_ENV: dict[str, str] = {}
UNLOCKED_ENV = {HOLDOUT_UNLOCK_ENV_VAR: "1"}


def rle_for(box: tuple[int, int, int, int], *, height: int, width: int) -> dict:
    """Build a compressed COCO RLE for one filled rectangle."""
    from pycocotools import mask as coco_mask

    x, y, w, h = box
    array = np.zeros((height, width), dtype=np.uint8, order="F")
    array[y : y + h, x : x + w] = 1
    encoded = coco_mask.encode(array)
    return {"size": [height, width], "counts": encoded["counts"].decode("ascii")}


def polygon_annotation(image_id: str, annotation_id: str, label: str = "person"):
    return CanonicalAnnotation(
        source_image_id=image_id,
        annotation_id=annotation_id,
        label=label,
        class_index=CLASS_MAP[label],
        segmentation=[[10.0, 10.0, 30.0, 10.0, 30.0, 40.0, 10.0, 40.0]],
        area=600.0,
        canonical_bbox=[10.0, 10.0, 20.0, 30.0],
        geometry_origin=PROVIDER_GEOMETRY,
    )


def rle_annotation(image_id: str, annotation_id: str, label: str = "helmet_loose"):
    segmentation = rle_for((5, 5, 10, 20), height=100, width=100)
    return CanonicalAnnotation(
        source_image_id=image_id,
        annotation_id=annotation_id,
        label=label,
        class_index=CLASS_MAP[label],
        segmentation=segmentation,
        area=200.0,
        canonical_bbox=[5.0, 5.0, 10.0, 20.0],
        geometry_origin=PROVIDER_GEOMETRY,
    )


def synthetic_annotation(image_id: str, annotation_id: str, label: str = "vest_on_body"):
    return CanonicalAnnotation(
        source_image_id=image_id,
        annotation_id=annotation_id,
        label=label,
        class_index=CLASS_MAP[label],
        segmentation=[[1.0, 2.0, 21.0, 2.0, 21.0, 12.0, 1.0, 12.0]],
        area=200.0,
        canonical_bbox=[1.0, 2.0, 20.0, 10.0],
        geometry_origin=SYNTHETIC_GEOMETRY,
    )


def fixture_images() -> list[CanonicalImage]:
    return [
        CanonicalImage("img_a", "img_a.jpg", 100, 100, "singleton-img_a"),
        CanonicalImage("img_b", "img_b.jpg", 100, 100, "singleton-img_b"),
        CanonicalImage("img_empty", "img_empty.jpg", 100, 100, "singleton-img_empty"),
    ]


def fixture_annotations() -> list[CanonicalAnnotation]:
    # img_empty deliberately carries none: it is a negative and must survive.
    return [
        polygon_annotation("img_a", "1"),
        rle_annotation("img_a", "2"),
        synthetic_annotation("img_b", "1"),
    ]


def fixture_ids():
    images = [image.source_image_id for image in fixture_images()]
    keys = [(a.source_image_id, a.annotation_id) for a in fixture_annotations()]
    return build_canonical_ids(images, keys)


def build_views(split: str = "train"):
    images, annotations, ids = fixture_images(), fixture_annotations(), fixture_ids()
    common = {
        "ids": ids,
        "class_map": CLASS_MAP,
        "split": split,
        "config_fingerprint": "cfg",
        "split_sha256": "split",
    }
    return (
        build_detection_coco(images, annotations, **common),
        build_segmentation_coco(images, annotations, **common),
        ids,
    )


# --- configuration ----------------------------------------------------------


def test_committed_config_parses_and_keeps_the_holdout_disabled():
    config = load_materialization_config(
        ProjectPaths.from_root().configs / "task_materialization.yaml"
    )
    assert config.development_splits == DEVELOPMENT_SPLITS
    assert config.test_materialization == TEST_DISABLED
    assert config.detection_bbox_source == "DERIVED_FROM_CANONICAL_SEGMENTATION"
    assert config.segmentation_geometry_policy == "PRESERVE_CANONICAL_REPRESENTATION"
    assert config.image_copy_mode == "BINARY_IDENTICAL"
    assert config.output_format == "COCO"


def test_config_fingerprint_is_stable_and_content_derived(tmp_path):
    path = ProjectPaths.from_root().configs / "task_materialization.yaml"
    first = load_materialization_config(path)
    second = load_materialization_config(path)
    assert first.fingerprint() == second.fingerprint()

    edited = tmp_path / "task_materialization.yaml"
    edited.write_text(
        path.read_text(encoding="utf-8").replace("bbox_canvas_px: 1.0", "bbox_canvas_px: 2.0"),
        encoding="utf-8",
    )
    assert load_materialization_config(edited).fingerprint() != first.fingerprint()


def test_config_rejects_test_in_the_development_splits(tmp_path):
    from construction_safety_vision.config import ConfigError

    path = ProjectPaths.from_root().configs / "task_materialization.yaml"
    edited = tmp_path / "task_materialization.yaml"
    edited.write_text(
        path.read_text(encoding="utf-8").replace(
            "development_splits:\n  - train\n  - validation",
            "development_splits:\n  - train\n  - validation\n  - test",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="development_splits"):
        load_materialization_config(edited)


def test_config_rejects_enabling_test_materialization(tmp_path):
    from construction_safety_vision.config import ConfigError

    path = ProjectPaths.from_root().configs / "task_materialization.yaml"
    edited = tmp_path / "task_materialization.yaml"
    edited.write_text(
        path.read_text(encoding="utf-8").replace(
            f"test_materialization: {TEST_DISABLED}", "test_materialization: enabled"
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="test_materialization"):
        load_materialization_config(edited)


# --- image naming and copying ------------------------------------------------


def test_image_naming_is_deterministic_and_id_based(tmp_path):
    source = tmp_path / "a-very-long-provider-filename.JPG"
    assert materialized_name("abc123", source) == "abc123.jpg"
    assert materialized_name("abc123", source) == materialized_name("abc123", source)


def test_image_naming_rejects_an_extensionless_source(tmp_path):
    with pytest.raises(MaterializationError, match="no extension"):
        materialized_name("abc123", tmp_path / "noextension")


def test_copy_is_byte_preserving(tmp_path):
    source = tmp_path / "src.jpg"
    source.write_bytes(b"\xff\xd8\xff\xe0 not really a jpeg \x00\x01\x02")
    destination = tmp_path / "out" / "img.jpg"
    result = copy_image(source, destination, source_image_id="img")
    assert result.action == COPIED
    assert result.byte_identical
    assert result.source_sha256 == sha256_bytes(destination)
    assert destination.read_bytes() == source.read_bytes()


def test_copy_is_idempotent(tmp_path):
    source = tmp_path / "src.jpg"
    source.write_bytes(b"payload")
    destination = tmp_path / "out" / "img.jpg"
    copy_image(source, destination, source_image_id="img")
    second = copy_image(source, destination, source_image_id="img")
    assert second.action == ALREADY_PRESENT
    assert second.byte_identical


def test_copy_refuses_a_destination_holding_different_bytes(tmp_path):
    source = tmp_path / "src.jpg"
    source.write_bytes(b"payload")
    destination = tmp_path / "out" / "img.jpg"
    destination.parent.mkdir()
    destination.write_bytes(b"something else entirely")
    with pytest.raises(MaterializationError, match="different bytes"):
        copy_image(source, destination, source_image_id="img")


def test_copy_reports_a_missing_source(tmp_path):
    with pytest.raises(MaterializationError, match="not found"):
        copy_image(tmp_path / "absent.jpg", tmp_path / "out.jpg", source_image_id="img")


def test_image_content_fingerprint_moves_with_content(tmp_path):
    source = tmp_path / "src.jpg"
    source.write_bytes(b"one")
    first = copy_image(source, tmp_path / "a" / "img.jpg", source_image_id="img")
    source.write_bytes(b"two")
    second = copy_image(source, tmp_path / "b" / "img.jpg", source_image_id="img")
    assert image_content_fingerprint([first]) != image_content_fingerprint([second])
    assert image_content_fingerprint([first]) == image_content_fingerprint([first])


# --- population and identifiers ----------------------------------------------


def population_rows(**extra: str) -> list[dict[str, str]]:
    base = [
        {
            "source_image_id": "img_a",
            "status": "ELIGIBLE",
            "width": "100",
            "height": "80",
            "group_id": "singleton-img_a",
        },
        {
            "source_image_id": "img_b",
            "status": "ELIGIBLE",
            "width": "60",
            "height": "40",
            "group_id": "pair_001",
        },
        {
            "source_image_id": "img_out",
            "status": "EXCLUDED",
            "width": "10",
            "height": "10",
            "group_id": "",
        },
    ]
    for row in base:
        row.update(extra)
    return base


def test_population_view_keeps_only_eligible_images():
    view = load_population(population_rows(), eligible_status="ELIGIBLE")
    assert view.eligible == ("img_a", "img_b")
    assert view.dimensions["img_a"] == (100, 80)
    assert view.groups["img_b"] == "pair_001"


def test_population_view_ignores_a_provider_split_column():
    # The provider split was rejected in phase 4B. Even if its provenance were
    # carried in the population table, it must not reach the materialiser.
    clean = load_population(population_rows(), eligible_status="ELIGIBLE")
    contaminated = load_population(
        population_rows(provider_split="test", original_split="test", split="test"),
        eligible_status="ELIGIBLE",
    )
    assert contaminated == clean


def test_population_view_rejects_a_repeated_image():
    rows = population_rows() + population_rows()[:1]
    with pytest.raises(MaterializationError, match="more than once"):
        load_population(rows, eligible_status="ELIGIBLE")


def test_canonical_ids_are_deterministic_and_order_independent():
    images = ["img_c", "img_a", "img_b"]
    keys = [("img_c", "2"), ("img_a", "1"), ("img_b", "1")]
    first = build_canonical_ids(images, keys)
    second = build_canonical_ids(list(reversed(images)), list(reversed(keys)))
    assert first == second
    assert first.fingerprint() == second.fingerprint()
    # Sorted order, ids starting at 1.
    assert first.image_ids == {"img_a": 1, "img_b": 2, "img_c": 3}


def test_canonical_ids_are_global_so_splits_never_collide():
    ids = build_canonical_ids(["img_a", "img_b", "img_c"], [])
    assert len(set(ids.image_ids.values())) == 3


def test_canonical_ids_reject_duplicates():
    with pytest.raises(MaterializationError, match="more than once"):
        build_canonical_ids(["img_a", "img_a"], [])


# --- COCO construction --------------------------------------------------------


def test_categories_follow_the_frozen_class_map():
    assert categories(CLASS_MAP) == [
        {"id": 0, "name": "helmet_loose", "supercategory": "ppe"},
        {"id": 1, "name": "helmet_on_head", "supercategory": "ppe"},
        {"id": 2, "name": "person", "supercategory": "ppe"},
        {"id": 3, "name": "vest_loose", "supercategory": "ppe"},
        {"id": 4, "name": "vest_on_body", "supercategory": "ppe"},
    ]


def test_placeholder_object_category_is_absent():
    detection, segmentation, _ = build_views()
    for document in (detection, segmentation):
        assert "object" not in {record["name"] for record in document["categories"]}


def test_zero_instance_images_are_retained_with_no_annotations():
    detection, segmentation, ids = build_views()
    empty_id = ids.image_ids["img_empty"]
    for document in (detection, segmentation):
        assert empty_id in {record["id"] for record in document["images"]}
        assert not [r for r in document["annotations"] if r["image_id"] == empty_id]


def test_detection_and_segmentation_hold_identical_image_identity():
    detection, segmentation, _ = build_views()
    assert [
        (r["id"], r["source_image_id"], r["file_name"], r["width"], r["height"])
        for r in detection["images"]
    ] == [
        (r["id"], r["source_image_id"], r["file_name"], r["width"], r["height"])
        for r in segmentation["images"]
    ]


def test_detection_and_segmentation_hold_identical_annotation_identity():
    detection, segmentation, _ = build_views()
    keys = ("id", "image_id", "category_id", "source_annotation_id", "bbox")
    assert [tuple(r[k] for k in keys) for r in detection["annotations"]] == [
        tuple(r[k] for k in keys) for r in segmentation["annotations"]
    ]


def test_alignment_report_passes_for_the_two_views():
    detection, segmentation, _ = build_views()
    report = alignment_report(detection, segmentation, "train")
    assert report["aligned"]
    assert report["problems"] == []


def test_alignment_report_catches_a_dropped_annotation():
    detection, segmentation, _ = build_views()
    detection["annotations"] = detection["annotations"][:-1]
    report = alignment_report(detection, segmentation, "train")
    assert not report["aligned"]
    assert any("annotation ids" in problem for problem in report["problems"])


def test_detection_view_carries_no_mask():
    detection, _, _ = build_views()
    assert all("segmentation" not in record for record in detection["annotations"])


def test_segmentation_view_carries_the_canonical_geometry():
    _, segmentation, _ = build_views()
    by_source = {
        r["source_annotation_id"] + r["source_image_id"]: r for r in segmentation["annotations"]
    }
    assert isinstance(by_source["1img_a"]["segmentation"], list)
    assert isinstance(by_source["2img_a"]["segmentation"], dict)


def test_documents_carry_no_timestamp():
    detection, segmentation, _ = build_views()
    for document in (detection, segmentation):
        assert "date_created" not in document["info"]
        assert "generated" not in document["info"]


def test_document_fingerprint_ignores_info_and_moves_with_data():
    detection, _, _ = build_views()
    baseline = document_fingerprint(detection)
    detection["info"]["description"] = "reworded"
    assert document_fingerprint(detection) == baseline
    detection["annotations"][0]["category_id"] = 4
    assert document_fingerprint(detection) != baseline


# --- boxes derived from segmentation ------------------------------------------


def test_bbox_is_derived_from_polygon_geometry():
    assert derive_bbox(polygon_annotation("img_a", "1")) == [10.0, 10.0, 20.0, 30.0]


def test_bbox_is_derived_from_rle_geometry():
    assert derive_bbox(rle_annotation("img_a", "2")) == [5.0, 5.0, 10.0, 20.0]


def test_bbox_is_derived_from_a_synthetic_rectangle():
    assert derive_bbox(synthetic_annotation("img_b", "1")) == [1.0, 2.0, 20.0, 10.0]


def test_provider_bbox_is_not_used_as_the_detection_box():
    # A canonical record whose stored bbox disagrees with its geometry: the
    # emitted box must follow the geometry, which is the project's ground truth.
    annotation = CanonicalAnnotation(
        source_image_id="img_a",
        annotation_id="1",
        label="person",
        class_index=2,
        segmentation=[[10.0, 10.0, 30.0, 10.0, 30.0, 40.0, 10.0, 40.0]],
        area=600.0,
        canonical_bbox=[0.0, 0.0, 99.0, 99.0],
        geometry_origin=PROVIDER_GEOMETRY,
    )
    assert derive_bbox(annotation) == [10.0, 10.0, 20.0, 30.0]

    detection = build_detection_coco(
        [CanonicalImage("img_a", "img_a.jpg", 100, 100, "singleton-img_a")],
        [annotation],
        ids=build_canonical_ids(["img_a"], [("img_a", "1")]),
        class_map=CLASS_MAP,
        split="train",
        config_fingerprint="cfg",
        split_sha256="split",
    )
    assert detection["annotations"][0]["bbox"] == [10.0, 10.0, 20.0, 30.0]
    assert detection["annotations"][0]["bbox"] != [0.0, 0.0, 99.0, 99.0]


def test_bbox_agreement_reports_a_disagreement():
    annotation = CanonicalAnnotation(
        source_image_id="img_a",
        annotation_id="1",
        label="person",
        class_index=2,
        segmentation=[[10.0, 10.0, 30.0, 10.0, 30.0, 40.0, 10.0, 40.0]],
        area=600.0,
        canonical_bbox=[0.0, 0.0, 99.0, 99.0],
        geometry_origin=PROVIDER_GEOMETRY,
    )
    report = bbox_agreement([annotation], 1.0)
    assert not report["within_tolerance"]
    assert report["problems"]


def test_bbox_agreement_passes_on_consistent_records():
    report = bbox_agreement(fixture_annotations(), 1.0)
    assert report["within_tolerance"]
    assert report["max_delta_px"] == 0.0


def test_geometry_kinds_are_classified():
    assert polygon_annotation("i", "1").geometry_kind == POLYGON
    assert rle_annotation("i", "2").geometry_kind == RLE
    assert synthetic_annotation("i", "3").geometry_kind == SYNTHETIC_RECTANGLE


# --- validation ----------------------------------------------------------------


def written_images(tmp_path):
    directory = tmp_path / "images"
    directory.mkdir()
    for image in fixture_images():
        (directory / image.file_name).write_bytes(b"stand-in")
    return directory


def test_valid_documents_report_no_problems(tmp_path):
    directory = written_images(tmp_path)
    detection, segmentation, _ = build_views()
    for task, document in ((DETECTION, detection), (SEGMENTATION, segmentation)):
        assert (
            validate_coco(
                document,
                task=task,
                split="train",
                image_directory=directory,
                expected_images=3,
                expected_annotations=3,
                class_map=CLASS_MAP,
                tolerances=TOLERANCES,
            )
            == []
        )


def test_validation_catches_a_missing_image_file(tmp_path):
    directory = written_images(tmp_path)
    (directory / "img_b.jpg").unlink()
    detection, _, _ = build_views()
    problems = validate_coco(
        detection,
        task=DETECTION,
        split="train",
        image_directory=directory,
        expected_images=3,
        expected_annotations=3,
        class_map=CLASS_MAP,
        tolerances=TOLERANCES,
    )
    assert any("missing from disk" in problem for problem in problems)


def test_validation_catches_a_dangling_annotation(tmp_path):
    directory = written_images(tmp_path)
    detection, _, _ = build_views()
    detection["annotations"][0]["image_id"] = 999
    problems = validate_coco(
        detection,
        task=DETECTION,
        split="train",
        image_directory=directory,
        expected_images=3,
        expected_annotations=3,
        class_map=CLASS_MAP,
        tolerances=TOLERANCES,
    )
    assert any("not in this split" in problem for problem in problems)


def test_validation_catches_a_wrong_annotation_count(tmp_path):
    directory = written_images(tmp_path)
    detection, _, _ = build_views()
    problems = validate_coco(
        detection,
        task=DETECTION,
        split="train",
        image_directory=directory,
        expected_images=3,
        expected_annotations=99,
        class_map=CLASS_MAP,
        tolerances=TOLERANCES,
    )
    assert any("the canonical population requires" in problem for problem in problems)


def test_validation_catches_a_foreign_category(tmp_path):
    directory = written_images(tmp_path)
    detection, _, _ = build_views()
    detection["categories"].append({"id": 9, "name": "object", "supercategory": "ppe"})
    problems = validate_coco(
        detection,
        task=DETECTION,
        split="train",
        image_directory=directory,
        expected_images=3,
        expected_annotations=3,
        class_map=CLASS_MAP,
        tolerances=TOLERANCES,
    )
    assert any("placeholder category" in problem for problem in problems)


def test_validation_catches_a_box_outside_the_canvas(tmp_path):
    directory = written_images(tmp_path)
    detection, _, _ = build_views()
    detection["annotations"][0]["bbox"] = [90.0, 90.0, 500.0, 500.0]
    problems = validate_coco(
        detection,
        task=DETECTION,
        split="train",
        image_directory=directory,
        expected_images=3,
        expected_annotations=3,
        class_map=CLASS_MAP,
        tolerances=TOLERANCES,
    )
    assert any("outside its" in problem for problem in problems)


# --- geometry round trip --------------------------------------------------------


def round_trip_through_json(tmp_path):
    _, segmentation, ids = build_views()
    path = tmp_path / "segmentation.json"
    path.write_text(json.dumps(segmentation, sort_keys=True), encoding="utf-8")
    emitted = json.loads(path.read_text(encoding="utf-8"))
    return round_trip_geometry(
        fixture_annotations(),
        {record["id"]: record for record in emitted["annotations"]},
        ids,
        {image.source_image_id: (image.width, image.height) for image in fixture_images()},
        TOLERANCES,
    )


def test_geometry_survives_a_json_round_trip(tmp_path):
    result = round_trip_through_json(tmp_path)
    assert result.mismatches == []
    assert result.matches == 3
    assert result.polygons_checked == 1
    assert result.rle_checked == 1
    assert result.synthetic_checked == 1
    assert result.max_polygon_delta_px == 0.0


def test_round_trip_catches_a_perturbed_polygon(tmp_path):
    _, segmentation, ids = build_views()
    for record in segmentation["annotations"]:
        if isinstance(record["segmentation"], list) and record["source_annotation_id"] == "1":
            record["segmentation"][0][0] += 0.5
    result = round_trip_geometry(
        fixture_annotations(),
        {record["id"]: record for record in segmentation["annotations"]},
        ids,
        {image.source_image_id: (image.width, image.height) for image in fixture_images()},
        TOLERANCES,
    )
    assert result.mismatches


def test_round_trip_catches_a_changed_rle_mask(tmp_path):
    _, segmentation, ids = build_views()
    for record in segmentation["annotations"]:
        if isinstance(record["segmentation"], dict):
            record["segmentation"] = rle_for((0, 0, 3, 3), height=100, width=100)
    result = round_trip_geometry(
        fixture_annotations(),
        {record["id"]: record for record in segmentation["annotations"]},
        ids,
        {image.source_image_id: (image.width, image.height) for image in fixture_images()},
        TOLERANCES,
    )
    assert any("decoded masks differ" in problem for problem in result.mismatches)


def test_round_trip_catches_a_representation_change(tmp_path):
    _, segmentation, ids = build_views()
    for record in segmentation["annotations"]:
        if isinstance(record["segmentation"], dict):
            record["segmentation"] = [[0.0, 0.0, 5.0, 0.0, 5.0, 5.0]]
    result = round_trip_geometry(
        fixture_annotations(),
        {record["id"]: record for record in segmentation["annotations"]},
        ids,
        {image.source_image_id: (image.width, image.height) for image in fixture_images()},
        TOLERANCES,
    )
    assert any("changed representation" in problem for problem in result.mismatches)


def test_rle_decodes_to_the_expected_mask():
    segmentation = rle_for((5, 5, 10, 20), height=100, width=100)
    mask = decode_mask(segmentation, height=100, width=100)
    assert mask.sum() == 200
    assert mask[5, 5] == 1
    assert mask[0, 0] == 0


# --- holdout guard on the materialisation path ---------------------------------


def synthetic_frozen_splits() -> FrozenSplits:
    assignments = normalise_assignments(
        [
            SplitAssignment("g1", "img_a", "train"),
            SplitAssignment("g2", "img_b", "train"),
            SplitAssignment("g3", "img_c", "validation"),
            SplitAssignment("g4", "img_d", "test"),
        ]
    )
    sections = split_sections(assignments)
    manifest = {
        "split_assignment_sha256": fingerprint_split_assignment(assignments),
        "holdout_sha256": fingerprint_holdout(assignments, modeling_population_sha256="pop"),
        **{split: sections[split] for split in sections},
    }
    return FrozenSplits(manifest=manifest, assignments=assignments)


def test_train_is_resolvable_without_any_override():
    splits = synthetic_frozen_splits()
    assert resolve_split_images(splits, "train", purpose="materialising", env=LOCKED_ENV) == (
        "img_a",
        "img_b",
    )


def test_validation_is_resolvable_without_any_override():
    splits = synthetic_frozen_splits()
    assert resolve_split_images(splits, "validation", purpose="materialising", env=LOCKED_ENV) == (
        "img_c",
    )


def test_test_materialization_is_denied_by_default():
    splits = synthetic_frozen_splits()
    with pytest.raises(HoldoutViolationError, match="allow_test=True"):
        resolve_split_images(splits, "test", purpose="materialising", env=LOCKED_ENV)


def test_code_only_override_is_insufficient_for_materialization():
    splits = synthetic_frozen_splits()
    with pytest.raises(HoldoutViolationError, match=HOLDOUT_UNLOCK_ENV_VAR):
        resolve_split_images(
            splits, "test", purpose="materialising", allow_test=True, env=LOCKED_ENV
        )


def test_environment_only_override_is_insufficient_for_materialization():
    splits = synthetic_frozen_splits()
    with pytest.raises(HoldoutViolationError, match="allow_test=True"):
        resolve_split_images(splits, "test", purpose="materialising", env=UNLOCKED_ENV)


def test_both_overrides_materialize_a_synthetic_holdout():
    # A fixture holdout, never the project's. This proves the future
    # final-evaluation path works through the same guard.
    splits = synthetic_frozen_splits()
    assert resolve_split_images(
        splits, "test", purpose="final materialisation", allow_test=True, env=UNLOCKED_ENV
    ) == ("img_d",)


def test_development_run_refuses_the_protected_split():
    with pytest.raises(MaterializationError, match="protected holdout"):
        assert_development_split("test")


def test_development_run_refuses_an_unknown_split():
    with pytest.raises(MaterializationError, match="unknown split"):
        assert_development_split("trainval")


@pytest.mark.parametrize("split", DEVELOPMENT_SPLITS)
def test_development_splits_are_accepted(split):
    assert assert_development_split(split) == split
