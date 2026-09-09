"""Unit tests for the YOLO segmentation adapter and its fidelity measurement.

Every topology case here is synthetic. A hand-built mask with two components, or
one with a known hole of a known area, is the only way to check that the audit
detects and charges for what it claims to - a real annotation would confound the
measurement with whatever else its geometry happens to do.

Nothing here trains a model, reads the dataset, or touches the holdout.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from construction_safety_vision.config import ConfigError
from construction_safety_vision.data.segmentation_adapter import (
    ADAPTER_TYPE,
    CANONICAL_POLYGON,
    CANONICAL_RLE,
    COMPONENT_JOIN_APPROXIMATION,
    HOLE_FILL_APPROXIMATION,
    MULTI_COMPONENT_APPROXIMATION,
    RASTER_BOUNDARY_DIFFERENCE,
    REASON_FLAGS,
    SERIALIZATION_ONLY,
    SMALL_MASK_SENSITIVITY,
    SYNTHETIC_RECTANGLE,
    SegmentationAdapterError,
    analyse_topology,
    bounds_violations,
    canonical_mask,
    classify_geometry,
    denormalise_ring,
    format_label_line,
    label_fingerprint,
    label_text,
    load_audit_config,
    merge_rings,
    normalise_ring,
    parse_label_line,
    rings_for_annotation,
)
from construction_safety_vision.data.segmentation_fidelity import (
    area_error_band_counts,
    attribute_reasons,
    band_counts,
    compare_masks,
    mask_bbox,
    quartile_edges,
    rasterise_ring,
)
from construction_safety_vision.paths import ProjectPaths

SQUARE = [[10.0, 10.0, 30.0, 10.0, 30.0, 30.0, 10.0, 30.0]]


def _rle(mask: np.ndarray) -> dict:
    """Encode a binary mask as compressed COCO RLE.

    Args:
        mask: A binary mask.

    Returns:
        The RLE mapping, with `counts` as a str as the canonical files store it.
    """
    from pycocotools import mask as mask_utils

    encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    return {"counts": encoded["counts"].decode("utf-8"), "size": list(encoded["size"])}


# --- configuration ------------------------------------------------------------


@pytest.fixture(scope="module")
def config():
    return load_audit_config(ProjectPaths.from_root().configs / "segmentation_adapter_audit.yaml")


def test_the_committed_protocol_loads(config):
    assert config["adapter_type"] == ADAPTER_TYPE
    assert config.precision >= 1
    assert config.connectivity in (4, 8)


def test_the_protocol_fingerprint_is_deterministic(config):
    assert config.fingerprint() == config.fingerprint()
    assert len(config.fingerprint()) == 64


def test_the_protocol_forbids_the_holdout(config):
    assert "test" not in config["split_directories"]
    assert "test" not in config["allowed_splits"]
    assert config["test_policy"] == "PROTECTED_NOT_ACCESSED"


def test_the_protocol_pins_the_instance_invariant(config):
    assert config["instance_split_allowed"] is False
    assert config["instance_merge_allowed"] is False


def test_an_unknown_key_is_rejected(tmp_path: Path, config):
    import yaml

    payload = dict(config.raw)
    payload["surprise"] = 1
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown key"):
        load_audit_config(path)


def test_a_holdout_split_directory_is_rejected(tmp_path: Path, config):
    import yaml

    payload = json.loads(json.dumps(config.raw))
    payload["split_directories"]["test"] = "test"
    payload["allowed_splits"].append("test")
    path = tmp_path / "leak.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ConfigError, match="holdout gets no adapter"):
        load_audit_config(path)


def test_relaxing_the_instance_invariant_is_rejected(tmp_path: Path, config):
    import yaml

    payload = json.loads(json.dumps(config.raw))
    payload["instance_split_allowed"] = True
    path = tmp_path / "split.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ConfigError, match="not optional"):
        load_audit_config(path)


# --- geometry classification --------------------------------------------------


def test_a_polygon_annotation_is_classified_as_a_polygon():
    assert classify_geometry({"id": "a", "segmentation": SQUARE}) == CANONICAL_POLYGON


def test_an_rle_annotation_is_classified_as_rle():
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[10:30, 10:30] = 1
    assert classify_geometry({"id": "a", "segmentation": _rle(mask)}) == CANONICAL_RLE


def test_a_synthetic_rectangle_is_its_own_stratum():
    # Stored as a polygon, but not human-drawn segmentation. Reporting it inside
    # the polygon stratum would let it vouch for real outlines.
    annotation = {
        "id": "a",
        "segmentation": SQUARE,
        "geometry_origin": "SYNTHETIC_FROM_PROVIDER_BBOX",
    }
    assert classify_geometry(annotation) == SYNTHETIC_RECTANGLE


def test_an_unrecognised_geometry_is_refused():
    with pytest.raises(SegmentationAdapterError, match="neither a polygon"):
        classify_geometry({"id": "a", "segmentation": 7})


# --- decoding -----------------------------------------------------------------


def test_a_polygon_decodes_to_the_expected_area():
    mask = canonical_mask({"id": "a", "segmentation": SQUARE}, height=40, width=40)
    assert mask.shape == (40, 40)
    assert 300 <= int(mask.sum()) <= 441


def test_rle_decoding_is_deterministic():
    source = np.zeros((40, 40), dtype=np.uint8)
    source[10:30, 10:30] = 1
    annotation = {"id": "a", "segmentation": _rle(source)}
    first = canonical_mask(annotation, height=40, width=40)
    second = canonical_mask(annotation, height=40, width=40)
    assert np.array_equal(first, second)
    assert np.array_equal(first, source)


def test_a_mask_on_the_wrong_canvas_is_refused():
    source = np.zeros((40, 40), dtype=np.uint8)
    source[10:30, 10:30] = 1
    with pytest.raises(SegmentationAdapterError, match="does not belong to this canvas"):
        canonical_mask({"id": "a", "segmentation": _rle(source)}, height=50, width=50)


# --- topology -----------------------------------------------------------------


def test_a_single_blob_has_one_component_and_no_holes():
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[10:30, 10:30] = 1
    topology = analyse_topology(mask, connectivity=8)
    assert topology.component_count == 1
    assert topology.hole_count == 0
    assert topology.area_px == 400


def test_two_separated_blobs_are_two_components():
    mask = np.zeros((40, 60), dtype=np.uint8)
    mask[10:20, 5:15] = 1
    mask[10:20, 40:50] = 1
    topology = analyse_topology(mask, connectivity=8)
    assert topology.component_count == 2
    assert topology.component_areas == (100, 100)


def test_connectivity_changes_the_component_count():
    # Two blocks touching only at a corner: one component at 8, two at 4.
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[2:6, 2:6] = 1
    mask[6:10, 6:10] = 1
    assert analyse_topology(mask, connectivity=8).component_count == 1
    assert analyse_topology(mask, connectivity=4).component_count == 2


def test_a_ring_has_one_hole_of_the_expected_area():
    mask = np.zeros((60, 60), dtype=np.uint8)
    mask[10:50, 10:50] = 1
    mask[20:40, 20:40] = 0
    topology = analyse_topology(mask, connectivity=8)
    assert topology.component_count == 1
    assert topology.hole_count == 1
    # Counted in pixels, so a 20x20 hole is exactly 400 - not the 441 that
    # cv2.contourArea would report for the path around it.
    assert topology.hole_pixels == 400
    assert topology.hole_area_fraction > 0.0


def test_two_holes_are_counted_separately():
    mask = np.zeros((60, 80), dtype=np.uint8)
    mask[10:50, 10:70] = 1
    mask[20:30, 20:30] = 0
    mask[20:30, 50:60] = 0
    topology = analyse_topology(mask, connectivity=8)
    assert topology.hole_count == 2
    assert topology.hole_pixels == 200


def test_an_empty_mask_reports_no_topology():
    topology = analyse_topology(np.zeros((10, 10), dtype=np.uint8), connectivity=8)
    assert topology.component_count == 0
    assert topology.area_px == 0
    assert topology.thinness == 0.0


def test_thinness_is_larger_for_an_elongated_shape():
    square = np.zeros((60, 60), dtype=np.uint8)
    square[10:50, 10:50] = 1
    strip = np.zeros((60, 60), dtype=np.uint8)
    strip[10:12, 2:58] = 1
    assert (
        analyse_topology(strip, connectivity=8).thinness
        > analyse_topology(square, connectivity=8).thinness
    )


# --- ring extraction and merging ---------------------------------------------


def test_a_polygon_keeps_its_own_coordinates():
    annotation = {"id": "a", "segmentation": SQUARE}
    mask = canonical_mask(annotation, height=40, width=40)
    rings = rings_for_annotation(annotation, mask, geometry_type=CANONICAL_POLYGON)
    assert len(rings) == 1
    assert np.allclose(rings[0], np.array(SQUARE[0]).reshape(-1, 2))


def test_an_rle_mask_is_traced_into_one_ring_per_component():
    mask = np.zeros((40, 60), dtype=np.uint8)
    mask[10:20, 5:15] = 1
    mask[10:20, 40:50] = 1
    annotation = {"id": "a", "segmentation": _rle(mask)}
    rings = rings_for_annotation(annotation, mask, geometry_type=CANONICAL_RLE)
    assert len(rings) == 2


def test_contour_extraction_is_deterministic():
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[8:32, 8:32] = 1
    annotation = {"id": "a", "segmentation": _rle(mask)}
    first = rings_for_annotation(annotation, mask, geometry_type=CANONICAL_RLE)
    second = rings_for_annotation(annotation, mask, geometry_type=CANONICAL_RLE)
    assert len(first) == len(second)
    assert all(np.array_equal(a, b) for a, b in zip(first, second, strict=True))


def test_a_geometry_too_small_to_express_is_refused():
    with pytest.raises(SegmentationAdapterError, match="not expressible"):
        rings_for_annotation(
            {"id": "a", "segmentation": [[1.0, 1.0, 2.0, 2.0]]},
            np.zeros((10, 10), dtype=np.uint8),
            geometry_type=CANONICAL_POLYGON,
        )


def test_one_ring_is_returned_unmerged():
    ring = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    merged, joined = merge_rings([ring])
    assert joined is False
    assert np.array_equal(merged, ring)


def test_several_rings_merge_into_one_path():
    left = np.array([[0.0, 0.0], [5.0, 0.0], [5.0, 5.0], [0.0, 5.0]])
    right = np.array([[20.0, 0.0], [25.0, 0.0], [25.0, 5.0], [20.0, 5.0]])
    merged, joined = merge_rings([left, right])
    assert joined is True
    assert merged.ndim == 2
    assert merged.shape[1] == 2
    # One path, so at least every original vertex plus the bridging duplicates.
    assert len(merged) >= len(left) + len(right)


def test_merging_nothing_is_refused():
    with pytest.raises(SegmentationAdapterError, match="empty ring list"):
        merge_rings([])


# --- serialisation ------------------------------------------------------------


def test_normalisation_round_trips_within_serialisation_precision():
    ring = np.array([[100.0, 200.0], [300.0, 400.0], [500.0, 600.0]])
    normalised = normalise_ring(ring, image_width=1000, image_height=800)
    restored = denormalise_ring(normalised, image_width=1000, image_height=800)
    assert np.allclose(ring, restored)


def test_normalisation_rejects_a_degenerate_canvas():
    with pytest.raises(SegmentationAdapterError, match="degenerate canvas"):
        normalise_ring(np.zeros((3, 2)), image_width=0, image_height=10)


def test_a_row_round_trips_through_text():
    ring = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    line = format_label_line(2, ring, precision=6)
    class_index, parsed = parse_label_line(line)
    assert class_index == 2
    assert parsed.shape == (3, 2)
    assert np.allclose(parsed, ring, atol=1e-6)


def test_a_row_is_written_with_fixed_precision():
    ring = np.array([[0.5, 0.5], [0.25, 0.25], [0.125, 0.125]])
    line = format_label_line(0, ring, precision=6)
    assert line.split()[1] == "0.500000"
    assert line == format_label_line(0, ring, precision=6)


def test_a_two_point_row_cannot_be_written():
    with pytest.raises(SegmentationAdapterError, match="at least 3 points"):
        format_label_line(0, np.array([[0.1, 0.1], [0.2, 0.2]]), precision=6)


def test_a_detection_shaped_row_is_not_parsed_as_a_segment():
    # Exactly 5 fields is a detection box in Ultralytics' parser, not a segment.
    with pytest.raises(SegmentationAdapterError, match="not parse it as a segment"):
        parse_label_line("0 0.1 0.2 0.3 0.4")


def test_a_row_with_an_odd_coordinate_count_is_refused():
    with pytest.raises(SegmentationAdapterError, match="not an even number"):
        parse_label_line("0 0.1 0.2 0.3 0.4 0.5 0.6 0.7")


def test_parsing_uses_float32_like_the_framework():
    ring = np.array([[0.123456789, 0.2], [0.3, 0.4], [0.5, 0.6]])
    _, parsed = parse_label_line(format_label_line(0, ring, precision=9))
    assert parsed.dtype == np.float32


def test_an_image_with_no_instance_gets_an_empty_file_not_a_missing_one():
    # Ultralytics counts an empty file as a background image and a missing file
    # as a missing label, so the negative would be lost if this returned None.
    assert label_text([]) == ""


def test_label_text_ends_with_a_newline():
    assert label_text(["0 0.1 0.1 0.2 0.2 0.3 0.3"]).endswith("\n")


def test_the_label_fingerprint_ignores_iteration_order():
    forward = {"a": "1\n", "b": "2\n"}
    backward = {"b": "2\n", "a": "1\n"}
    assert label_fingerprint(forward) == label_fingerprint(backward)


def test_out_of_canvas_coordinates_are_counted():
    assert bounds_violations(np.array([[0.5, 0.5], [1.5, 0.2]])) == 1
    assert bounds_violations(np.array([[0.5, 0.5], [0.9, 0.2]])) == 0


# --- rasterisation and metrics ------------------------------------------------


def test_rasterising_a_square_recovers_its_area():
    ring = np.array([[10.0, 10.0], [30.0, 10.0], [30.0, 30.0], [10.0, 30.0]])
    mask = rasterise_ring(ring, height=40, width=40)
    assert mask.shape == (40, 40)
    assert 380 <= int(mask.sum()) <= 441


def test_identical_masks_score_one():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 5:15] = 1
    comparison = compare_masks(mask, mask)
    assert comparison.iou == 1.0
    assert comparison.dice == 1.0
    assert comparison.false_positive == 0
    assert comparison.false_negative == 0
    assert comparison.relative_area_error == 0.0


def test_disjoint_masks_score_zero():
    left = np.zeros((20, 20), dtype=np.uint8)
    left[0:5, 0:5] = 1
    right = np.zeros((20, 20), dtype=np.uint8)
    right[10:15, 10:15] = 1
    assert compare_masks(left, right).iou == 0.0


def test_iou_and_dice_match_a_hand_computed_case():
    left = np.zeros((10, 10), dtype=np.uint8)
    left[0:4, 0:4] = 1  # 16 px
    right = np.zeros((10, 10), dtype=np.uint8)
    right[2:6, 0:4] = 1  # 16 px, 8 px overlap
    comparison = compare_masks(left, right)
    assert comparison.intersection == 8
    assert comparison.union == 24
    assert comparison.iou == pytest.approx(8 / 24)
    assert comparison.dice == pytest.approx(2 * 8 / 32)


def test_false_positive_and_negative_pixels_are_accounted():
    canonical = np.zeros((10, 10), dtype=np.uint8)
    canonical[0:4, 0:4] = 1
    reconstructed = np.zeros((10, 10), dtype=np.uint8)
    reconstructed[0:4, 0:5] = 1
    comparison = compare_masks(canonical, reconstructed)
    assert comparison.false_positive == 4
    assert comparison.false_negative == 0
    assert comparison.absolute_area_error == 4
    assert comparison.signed_relative_area_error == pytest.approx(4 / 16)


def test_comparing_across_canvases_is_refused():
    with pytest.raises(ValueError, match="different shapes"):
        compare_masks(np.zeros((10, 10), np.uint8), np.zeros((12, 12), np.uint8))


def test_the_bbox_diagnostic_tracks_a_shifted_mask():
    canonical = np.zeros((20, 20), dtype=np.uint8)
    canonical[5:10, 5:10] = 1
    shifted = np.zeros((20, 20), dtype=np.uint8)
    shifted[7:12, 5:10] = 1
    assert compare_masks(canonical, shifted).bbox_max_delta == pytest.approx(2.0)


def test_an_empty_mask_has_no_bbox():
    assert mask_bbox(np.zeros((5, 5), dtype=np.uint8)) is None


# --- topology approximations are actually charged -----------------------------


def test_filling_a_hole_adds_exactly_the_hole_area():
    canonical = np.zeros((60, 60), dtype=np.uint8)
    canonical[10:50, 10:50] = 1
    canonical[20:40, 20:40] = 0
    filled = np.zeros((60, 60), dtype=np.uint8)
    filled[10:50, 10:50] = 1
    comparison = compare_masks(canonical, filled)
    assert comparison.false_positive == 400
    assert comparison.false_negative == 0
    assert comparison.iou < 1.0


def test_bridging_two_components_costs_measurable_area():
    mask = np.zeros((40, 80), dtype=np.uint8)
    mask[10:30, 5:25] = 1
    mask[10:30, 55:75] = 1
    annotation = {"id": "a", "segmentation": _rle(mask)}
    rings = rings_for_annotation(annotation, mask, geometry_type=CANONICAL_RLE)
    assert len(rings) == 2
    merged, joined = merge_rings(rings)
    assert joined is True
    reconstructed = rasterise_ring(merged, height=40, width=80)
    comparison = compare_masks(mask, reconstructed)
    # One ring cannot describe two separated blocks without covering the gap.
    assert comparison.iou < 1.0
    assert comparison.false_positive > 0


# --- bands, percentiles and attribution ---------------------------------------


def test_iou_bands_partition_the_sample():
    values = [1.0, 0.995, 0.97, 0.92, 0.5]
    bands = band_counts(values)
    assert sum(entry["count"] for entry in bands.values()) == len(values)
    assert bands["iou_exact"]["count"] == 1
    assert bands["iou_ge_0.99"]["count"] == 1
    assert bands["iou_0.95_to_0.99"]["count"] == 1
    assert bands["iou_0.90_to_0.95"]["count"] == 1
    assert bands["iou_lt_0.90"]["count"] == 1


def test_area_error_bands_count_exceedances():
    bands = area_error_band_counts([0.0, 0.015, 0.03, 0.2])
    assert bands["above_0.01"]["count"] == 3
    assert bands["above_0.02"]["count"] == 2
    assert bands["above_0.05"]["count"] == 1
    assert bands["above_0.1"]["count"] == 1


def test_quartile_edges_are_ordered():
    edges = quartile_edges([float(value) for value in range(1, 101)])
    assert edges[0] < edges[1] < edges[2]


def test_a_near_exact_instance_gets_no_reason_flags():
    flags = attribute_reasons(
        {
            "mask_iou": 0.9999,
            "merged_iou": 1.0,
            "control_iou": 1.0,
            "hole_count": 0,
            "connected_components": 1,
            "component_rings": 1,
            "joined": False,
            "canonical_area_px": 100000,
        },
        material_iou=0.999,
        small_area_px=1024,
    )
    assert flags == ()


def test_serialization_only_is_claimed_only_when_it_is_the_sole_cause():
    flags = attribute_reasons(
        {
            "mask_iou": 0.95,
            "merged_iou": 1.0,
            "control_iou": 1.0,
            "hole_count": 0,
            "connected_components": 1,
            "component_rings": 1,
            "joined": False,
            "canonical_area_px": 100000,
        },
        material_iou=0.999,
        small_area_px=1024,
    )
    assert flags == (SERIALIZATION_ONLY,)


def test_serialization_only_is_not_claimed_alongside_another_cause():
    flags = attribute_reasons(
        {
            "mask_iou": 0.90,
            "merged_iou": 0.95,
            "control_iou": 0.99,
            "hole_count": 0,
            "connected_components": 1,
            "component_rings": 2,
            "joined": True,
            "canonical_area_px": 100000,
        },
        material_iou=0.999,
        small_area_px=1024,
    )
    assert SERIALIZATION_ONLY not in flags
    assert COMPONENT_JOIN_APPROXIMATION in flags
    assert RASTER_BOUNDARY_DIFFERENCE in flags


def test_a_hole_bearing_instance_is_flagged():
    flags = attribute_reasons(
        {
            "mask_iou": 0.90,
            "merged_iou": 0.90,
            "control_iou": 0.90,
            "hole_count": 3,
            "connected_components": 1,
            "component_rings": 1,
            "joined": False,
            "canonical_area_px": 100000,
        },
        material_iou=0.999,
        small_area_px=1024,
    )
    assert HOLE_FILL_APPROXIMATION in flags


def test_a_multi_component_instance_is_flagged():
    flags = attribute_reasons(
        {
            "mask_iou": 0.90,
            "merged_iou": 0.92,
            "control_iou": 1.0,
            "hole_count": 0,
            "connected_components": 4,
            "component_rings": 4,
            "joined": True,
            "canonical_area_px": 100000,
        },
        material_iou=0.999,
        small_area_px=1024,
    )
    assert MULTI_COMPONENT_APPROXIMATION in flags
    assert COMPONENT_JOIN_APPROXIMATION in flags


def test_a_tiny_mask_is_flagged_as_size_sensitive():
    flags = attribute_reasons(
        {
            "mask_iou": 0.5,
            "merged_iou": 0.5,
            "control_iou": 0.5,
            "hole_count": 0,
            "connected_components": 1,
            "component_rings": 1,
            "joined": False,
            "canonical_area_px": 9,
        },
        material_iou=0.999,
        small_area_px=1024,
    )
    assert SMALL_MASK_SENSITIVITY in flags


def test_flags_are_emitted_in_the_declared_order():
    flags = attribute_reasons(
        {
            "mask_iou": 0.5,
            "merged_iou": 0.7,
            "control_iou": 0.9,
            "hole_count": 1,
            "connected_components": 3,
            "component_rings": 3,
            "joined": True,
            "canonical_area_px": 10,
        },
        material_iou=0.999,
        small_area_px=1024,
    )
    assert list(flags) == [flag for flag in REASON_FLAGS if flag in set(flags)]
