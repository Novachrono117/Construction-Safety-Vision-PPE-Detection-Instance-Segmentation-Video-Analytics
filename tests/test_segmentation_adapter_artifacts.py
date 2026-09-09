"""Tests that the committed phase 8A audit describes the conversion that was run.

Three things are checked here that the unit tests cannot.

**That no annotation was lost.** The audit's central claim is that 1726 canonical
annotations became 1726 YOLO rows, none split, merged or dropped. That is checked
against the canonical COCO documents themselves rather than against the audit's
own bookkeeping.

**That the audit decided nothing.** Phase 8A produces evidence for a human. A
report that quietly concluded "YOLO segmentation is fine" would be the failure
mode, so the artifacts are checked for exactly that.

**That the holdout is absent.** Not merely unmentioned - structurally absent from
the adapter, the descriptor, the table and the manifest.

No test here runs a model, and none reads the holdout.
"""

from __future__ import annotations

import csv
import json

import pytest

from construction_safety_vision.data.canonical import scan_for_sensitive
from construction_safety_vision.data.segmentation_adapter import (
    ADAPTER_TYPE,
    CANONICAL_POLYGON,
    CANONICAL_RLE,
    REASON_FLAGS,
    SYNTHETIC_RECTANGLE,
    load_audit_config,
)
from construction_safety_vision.paths import ProjectPaths
from construction_safety_vision.provenance import sha256_file
from construction_safety_vision.splits import HOLDOUT_UNLOCK_ENV_VAR

DEVELOPMENT_SPLITS = ("train", "validation")
EXPECTED_IMAGES = 368
EXPECTED_ANNOTATIONS = 1726


@pytest.fixture(scope="module")
def paths() -> ProjectPaths:
    return ProjectPaths.from_root()


@pytest.fixture(scope="module")
def manifest(paths: ProjectPaths) -> dict:
    path = paths.reports / "segmentation_adapter_audit_manifest.json"
    if not path.is_file():
        pytest.skip("segmentation_adapter_audit_manifest.json not present; run the phase 8A audit")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def report(paths: ProjectPaths) -> str:
    path = paths.reports / "segmentation_adapter_fidelity_report.md"
    if not path.is_file():
        pytest.skip("segmentation_adapter_fidelity_report.md not present")
    return path.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def rows(paths: ProjectPaths) -> list[dict[str, str]]:
    path = paths.reports / "segmentation_adapter_fidelity.csv"
    if not path.is_file():
        pytest.skip("segmentation_adapter_fidelity.csv not present")
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def canonical(paths: ProjectPaths) -> dict[str, dict]:
    documents = {}
    for split in DEVELOPMENT_SPLITS:
        path = (
            paths.data_processed / "canonical" / "annotations" / f"segmentation_{split}.coco.json"
        )
        if not path.is_file():
            pytest.skip(f"canonical segmentation_{split}.coco.json not materialised")
        documents[split] = json.loads(path.read_text(encoding="utf-8"))
    return documents


# --- the population is the canonical one --------------------------------------


def test_the_development_population_matches_the_canonical_documents(manifest, canonical):
    images = sum(len(canonical[split]["images"]) for split in DEVELOPMENT_SPLITS)
    annotations = sum(len(canonical[split]["annotations"]) for split in DEVELOPMENT_SPLITS)
    assert manifest["development_counts"]["images"] == images == EXPECTED_IMAGES
    assert manifest["development_counts"]["annotations"] == annotations == EXPECTED_ANNOTATIONS


def test_the_per_split_counts_match_the_canonical_documents(manifest, canonical):
    per_split = manifest["development_counts"]["per_split"]
    for split in DEVELOPMENT_SPLITS:
        assert per_split[split]["images"] == len(canonical[split]["images"])
        assert per_split[split]["annotations"] == len(canonical[split]["annotations"])


def test_the_frozen_split_shape_is_unchanged(manifest):
    per_split = manifest["development_counts"]["per_split"]
    assert per_split["train"]["images"] == 303
    assert per_split["train"]["annotations"] == 1422
    assert per_split["validation"]["images"] == 65
    assert per_split["validation"]["annotations"] == 304


def test_the_geometry_census_matches_the_canonical_documents(manifest, canonical):
    polygons = rle = synthetic = 0
    for split in DEVELOPMENT_SPLITS:
        for annotation in canonical[split]["annotations"]:
            if annotation.get("geometry_origin") == "SYNTHETIC_FROM_PROVIDER_BBOX":
                synthetic += 1
            elif isinstance(annotation["segmentation"], dict):
                rle += 1
            else:
                polygons += 1
    counts = manifest["canonical_geometry_counts"]
    assert counts[CANONICAL_POLYGON] == polygons
    assert counts[CANONICAL_RLE] == rle
    assert counts[SYNTHETIC_RECTANGLE] == synthetic
    assert sum(counts.values()) == EXPECTED_ANNOTATIONS


def test_zero_instance_images_are_retained(manifest):
    per_split = manifest["development_counts"]["per_split"]
    assert per_split["train"]["negative_images"] == 10
    assert per_split["validation"]["negative_images"] == 2


def test_the_class_map_is_the_frozen_one(manifest):
    assert manifest["class_map"] == {
        "helmet_loose": 0,
        "helmet_on_head": 1,
        "person": 2,
        "vest_loose": 3,
        "vest_on_body": 4,
    }
    assert (
        manifest["class_map_sha256"]
        == "596dab5b5756e3d4253924f84bf46d37a87d80b6ec830da781a08a93cd823753"
    )


def test_the_placeholder_class_is_absent(manifest):
    assert manifest["placeholder_category_excluded"] == "object"
    assert "object" not in manifest["class_map"]


def test_the_provider_split_is_not_referenced(manifest):
    serialised = json.dumps(manifest)
    for phrase in ("provider_split", "PROVIDER_SPLIT", "provider bbox"):
        assert phrase not in serialised


# --- one annotation, one row --------------------------------------------------


def test_instance_cardinality_is_preserved(manifest):
    cardinality = manifest["instance_cardinality"]
    assert cardinality["canonical_annotations"] == EXPECTED_ANNOTATIONS
    assert cardinality["model_instance_rows"] == EXPECTED_ANNOTATIONS
    assert cardinality["preserved"] is True
    assert cardinality["instances_split"] == 0
    assert cardinality["instances_merged"] == 0
    assert cardinality["instances_dropped"] == 0


def test_the_table_carries_one_row_per_canonical_annotation(rows, canonical):
    assert len(rows) == EXPECTED_ANNOTATIONS
    expected = {
        (split, str(annotation["id"]))
        for split in DEVELOPMENT_SPLITS
        for annotation in canonical[split]["annotations"]
    }
    found = {(row["split"], row["annotation_id"]) for row in rows}
    assert found == expected


def test_no_annotation_was_silently_lost(rows, canonical):
    for split in DEVELOPMENT_SPLITS:
        in_document = len(canonical[split]["annotations"])
        in_table = sum(1 for row in rows if row["split"] == split)
        assert in_table == in_document


def test_no_duplicate_bounding_box_collision_occurred(manifest):
    # Ultralytics silently drops duplicate (class, box) rows, which would change
    # instance cardinality without raising.
    assert manifest["duplicate_box_collisions"] == []


def test_no_coordinate_left_the_canvas(manifest):
    assert manifest["bounds_violations"] == 0


# --- the framework accepted it ------------------------------------------------


def test_the_framework_parser_discovered_every_development_image(manifest):
    splits = manifest["parser_validation"]["splits"]
    assert splits["train"]["images_discovered"] == 303
    assert splits["validation"]["images_discovered"] == 65


def test_the_framework_parser_read_back_every_instance(manifest):
    splits = manifest["parser_validation"]["splits"]
    assert splits["train"]["instances"] == 1422
    assert splits["validation"]["instances"] == 304
    total = sum(splits[split]["instances"] for split in splits)
    assert total == EXPECTED_ANNOTATIONS


def test_the_framework_read_one_segment_per_instance(manifest):
    for split, found in manifest["parser_validation"]["splits"].items():
        assert found["segment_rows"] == found["instances"], split


def test_the_framework_kept_the_negative_images(manifest):
    splits = manifest["parser_validation"]["splits"]
    assert splits["train"]["negative_images"] == 10
    assert splits["validation"]["negative_images"] == 2


def test_the_framework_found_no_corrupt_label_and_five_classes(manifest):
    assert manifest["parser_validation"]["corrupt_labels"] == 0
    assert manifest["parser_validation"]["classes"] == 5


def test_the_descriptor_exposes_no_holdout_split(manifest):
    assert manifest["parser_validation"]["test_split_present"] is False


# --- the format contract ------------------------------------------------------


def test_the_format_note_comes_from_the_installed_package(manifest):
    fmt = manifest["framework_format"]
    assert fmt["package"] == "ultralytics"
    assert fmt["evidence"] == "INSTALLED_PACKAGE_SOURCE_INSPECTION"
    assert fmt["one_row_per_instance"] is True
    assert fmt["row_is_single_ring"] is True


def test_the_format_note_records_that_holes_are_not_representable(manifest):
    fmt = manifest["framework_format"]
    assert fmt["holes_representable"] is False
    assert "RETR_EXTERNAL" in fmt["hole_evidence"]


def test_the_format_note_records_the_multi_segment_strategy(manifest):
    assert "merge_multi_segment" in manifest["framework_format"]["multi_segment_handling"]
    assert manifest["conversion_policy"]["multi_component_policy"] == (
        "ULTRALYTICS_MERGE_MULTI_SEGMENT"
    )


def test_the_format_note_records_resampling_and_the_cardinality_hazard(manifest):
    fmt = manifest["framework_format"]
    assert "resample_segments" in fmt["internal_resampling"]
    assert "np.unique" in fmt["cardinality_hazard"]


# --- topology is quantified, not assumed away ---------------------------------


def test_the_topology_census_is_recorded(manifest):
    topology = manifest["topology"]
    assert topology["connectivity"] in (4, 8)
    assert topology["instances_with_multiple_components"] > 0
    assert topology["maximum_component_count"] > 1
    distribution = topology["connected_component_distribution"]
    assert sum(distribution.values()) == EXPECTED_ANNOTATIONS
    # The census and the multi-component count must describe the same instances.
    assert (
        sum(count for key, count in distribution.items() if int(key) > 1)
        == (topology["instances_with_multiple_components"])
    )
    assert max(int(key) for key in distribution) == topology["maximum_component_count"]


def test_hole_statistics_are_recorded(manifest):
    topology = manifest["topology"]
    assert topology["instances_with_holes"] > 0
    assert topology["total_holes"] >= topology["instances_with_holes"]
    assert topology["total_hole_pixels"] > 0
    assert 0.0 < topology["largest_hole_area_fraction"] < 1.0


def test_the_hole_and_component_approximations_are_classified(manifest):
    assert manifest["conversion_policy"]["hole_policy"] == "NOT_REPRESENTABLE_FILLED"
    counts = manifest["fidelity"]["reason_flag_counts"]
    assert counts.get("HOLE_FILL_APPROXIMATION", 0) == manifest["topology"]["instances_with_holes"]
    assert (
        counts.get("MULTI_COMPONENT_APPROXIMATION", 0)
        == manifest["topology"]["instances_with_multiple_components"]
    )


def test_every_reason_flag_used_is_a_declared_one(manifest):
    for flag in manifest["fidelity"]["reason_flag_counts"]:
        assert flag in REASON_FLAGS


# --- fidelity is stratified, not aggregated away ------------------------------


def test_fidelity_is_reported_per_canonical_representation(manifest):
    by_type = manifest["fidelity"]["by_geometry_type"]
    assert set(by_type) == {CANONICAL_POLYGON, CANONICAL_RLE, SYNTHETIC_RECTANGLE}
    for summary in by_type.values():
        assert summary["instances"] > 0
        assert 0.0 <= summary["mask_iou"]["mean"] <= 1.0


def test_the_geometry_strata_sum_to_the_population(manifest):
    by_type = manifest["fidelity"]["by_geometry_type"]
    assert sum(summary["instances"] for summary in by_type.values()) == EXPECTED_ANNOTATIONS


def test_fidelity_is_reported_per_class(manifest):
    by_class = manifest["fidelity"]["by_class"]
    assert set(by_class) == set(manifest["class_map"])
    assert sum(summary["instances"] for summary in by_class.values()) == EXPECTED_ANNOTATIONS


def test_fidelity_is_reported_by_topology_and_size(manifest):
    fidelity = manifest["fidelity"]
    for key in ("by_component_count", "by_hole_presence", "by_mask_area_quartile"):
        assert fidelity[key], key
        assert sum(entry["instances"] for entry in fidelity[key].values()) == EXPECTED_ANNOTATIONS


def test_the_global_distribution_is_complete(manifest):
    iou = manifest["fidelity"]["global"]["mask_iou"]
    for key in ("min", "p01", "p05", "p25", "median", "mean", "p75", "p95", "p99", "max"):
        assert key in iou
    assert iou["count"] == EXPECTED_ANNOTATIONS
    assert iou["min"] <= iou["median"] <= iou["max"]


def test_the_iou_bands_partition_the_population(manifest):
    bands = manifest["fidelity"]["global"]["iou_bands"]
    assert sum(entry["count"] for entry in bands.values()) == EXPECTED_ANNOTATIONS


def test_the_three_level_decomposition_is_ordered(manifest):
    # Each stage can only lose fidelity, never recover it, so the means must be
    # ordered. If they were not, the decomposition would not be measuring stages.
    overall = manifest["fidelity"]["global"]
    assert overall["control_iou"]["mean"] >= overall["merged_iou"]["mean"]
    assert overall["join_loss"]["mean"] >= 0.0
    assert overall["serialization_loss"]["mean"] >= 0.0


def test_area_error_bands_are_recorded(manifest):
    bands = manifest["fidelity"]["global"]["area_error_bands"]
    assert set(bands) == {"above_0.01", "above_0.02", "above_0.05", "above_0.1"}
    counts = [
        bands[key]["count"] for key in ("above_0.01", "above_0.02", "above_0.05", "above_0.1")
    ]
    assert counts == sorted(counts, reverse=True)


def test_the_worst_cases_are_recorded(manifest):
    cases = manifest["worst_cases"]
    assert len(cases["worst_mask_iou"]) == 20
    assert len(cases["worst_relative_area_error"]) == 20
    ious = [case["mask_iou"] for case in cases["worst_mask_iou"]]
    assert ious == sorted(ious)


def test_every_table_row_carries_its_metrics(rows):
    for row in rows:
        assert 0.0 <= float(row["mask_iou"]) <= 1.0
        assert 0.0 <= float(row["dice"]) <= 1.0
        assert float(row["relative_area_error"]) >= 0.0
        assert int(row["fp_pixels"]) >= 0
        assert int(row["fn_pixels"]) >= 0
        assert int(row["canonical_area_px"]) >= 0
        assert int(row["adapter_point_count"]) >= 3


def test_the_table_metrics_agree_with_the_manifest_summary(rows, manifest):
    values = sorted(float(row["mask_iou"]) for row in rows)
    assert values[0] == pytest.approx(manifest["fidelity"]["global"]["mask_iou"]["min"], abs=1e-9)
    assert values[-1] == pytest.approx(manifest["fidelity"]["global"]["mask_iou"]["max"], abs=1e-9)
    mean = sum(values) / len(values)
    assert mean == pytest.approx(manifest["fidelity"]["global"]["mask_iou"]["mean"], abs=1e-9)


# --- fingerprints -------------------------------------------------------------


def test_the_audit_config_fingerprint_matches_the_committed_protocol(paths, manifest):
    config = load_audit_config(paths.configs / "segmentation_adapter_audit.yaml")
    assert config.fingerprint() == manifest["audit_config_sha256"]
    assert manifest["adapter_fingerprints"]["audit_config_sha256"] == config.fingerprint()


def test_every_required_fingerprint_is_recorded(manifest):
    fingerprints = manifest["adapter_fingerprints"]
    for key in (
        "audit_config_sha256",
        "labels_train_sha256",
        "labels_validation_sha256",
        "labels_development_sha256",
        "image_membership_sha256",
        "fidelity_rows_sha256",
        "fidelity_table_sha256",
    ):
        assert key in fingerprints, key
        assert len(fingerprints[key]) == 64


def test_the_fidelity_table_digest_matches_the_file(paths, manifest):
    path = paths.reports / "segmentation_adapter_fidelity.csv"
    assert sha256_file(path) == manifest["adapter_fingerprints"]["fidelity_table_sha256"]


def test_the_fingerprints_exclude_machine_specific_content(manifest):
    serialised = json.dumps(manifest["adapter_fingerprints"])
    assert "created_at" not in serialised
    assert ":" not in serialised.replace('":', "").replace('",', "")


def test_the_canonical_fingerprints_are_carried_through(manifest):
    fingerprints = manifest["canonical_fingerprints"]
    assert fingerprints["split_assignment_sha256"] == manifest["split_assignment_sha256"]
    assert fingerprints["class_map_sha256"] == manifest["class_map_sha256"]


# --- nothing was decided, trained or touched ----------------------------------


def test_no_architecture_was_selected(manifest):
    assert manifest["segmentation_architecture_selection"] == "UNSELECTED_PENDING_FIDELITY_REVIEW"
    assert manifest["architecture_decision"]["decided_here"] is False


def test_the_segmentation_baseline_is_unfrozen_and_s0_undefined(manifest):
    assert manifest["segmentation_baseline"] == "UNFROZEN"
    assert manifest["S0"] == "NOT_DEFINED"


def test_no_model_was_trained_or_evaluated(manifest):
    assert manifest["models_trained_in_this_phase"] == 0
    assert manifest["models_evaluated_in_this_phase"] == 0
    assert manifest["detector_touched"] is False


def test_the_fallback_is_recorded_but_not_selected(manifest):
    fallback = manifest["architecture_decision"]["fallback_recorded_not_selected"]
    assert fallback["status"] == "FUTURE_ALTERNATIVE_NOT_IMPLEMENTED_NOT_BENCHMARKED"
    assert "Mask R-CNN" in fallback["example"]
    assert manifest["segmentation_architecture_selection"] != "Mask R-CNN"


def test_the_adapter_declares_itself_audit_only(manifest):
    status = manifest["adapter_status"]
    assert status["status"] == "AUDIT_ONLY"
    assert status["canonical_status"] == "NOT_CANONICAL"
    assert status["training_status"] == "NOT_YET_APPROVED_FOR_TRAINING"
    assert manifest["adapter_type"] == ADAPTER_TYPE


def test_the_report_reaches_no_architecture_verdict(report):
    lowered = report.lower()
    for phrase in (
        "yolo segmentation is approved",
        "yolo segmentation approved",
        "yolo segmentation rejected",
        "we therefore select",
        "we recommend yolo",
        "s0 will be",
    ):
        assert phrase not in lowered
    assert "does not approve yolo segmentation and it does not reject it" in lowered


def test_the_report_defers_the_decision_to_human_review(report):
    assert "`PENDING_HUMAN_DECISION`" in report
    assert "phase 8B" in report
    assert "UNSELECTED_PENDING_FIDELITY_REVIEW" in report


def test_the_report_does_not_compare_the_detector_against_segmentation(report):
    lowered = report.lower()
    for phrase in ("latency benchmark", "detector outperforms", "mask ap versus box ap"):
        assert phrase not in lowered


def test_the_report_states_the_representation_limitation(report):
    assert "representation" in report.lower()
    assert "not model performance" in report.lower() or "not performance" in report.lower()


# --- the holdout --------------------------------------------------------------


def test_the_manifest_records_the_holdout_as_protected(manifest):
    assert manifest["test"]["status"] == "PROTECTED_NOT_ACCESSED"
    assert "not read" in manifest["test"]["reason"]


def test_no_holdout_directory_exists_in_the_audit_adapter(paths, manifest):
    root = paths.root / manifest["adapter"]["root"]
    if not root.exists():
        pytest.skip("the git-ignored audit adapter is not present on this machine")
    assert not (root / "images" / "test").exists()
    assert not (root / "labels" / "test").exists()
    descriptor = (root / "dataset.yaml").read_text(encoding="utf-8")
    assert "test:" not in descriptor


def test_no_holdout_split_appears_in_the_table(rows):
    assert {row["split"] for row in rows} == set(DEVELOPMENT_SPLITS)


def test_the_manifest_names_the_protected_split_only_as_a_notice(manifest):
    # Deliberately not implemented by loading the holdout id list and grepping
    # for it: reading that section to run a check would be the very bypass the
    # split guard exists to prevent.
    from construction_safety_vision.detection_freeze import holdout_leaks

    assert holdout_leaks(manifest) == []


def test_the_unlock_variable_is_not_carried_as_data(manifest):
    assert HOLDOUT_UNLOCK_ENV_VAR not in json.dumps(manifest)


def test_no_segmentation_model_weight_or_training_output_exists(paths, manifest):
    root = paths.root / manifest["adapter"]["root"]
    if not root.exists():
        pytest.skip("the git-ignored audit adapter is not present on this machine")
    assert list(root.rglob("*.pt")) == []
    assert list(root.rglob("*.cache")) == []
    assert not (paths.root / "artifacts" / "segmentation").exists()


# --- provenance and hygiene ---------------------------------------------------


def test_the_committed_artifacts_carry_no_sensitive_content(paths, manifest, report):
    csv_text = (paths.reports / "segmentation_adapter_fidelity.csv").read_text(encoding="utf-8")
    findings = (
        scan_for_sensitive(report)
        + scan_for_sensitive(json.dumps(manifest))
        + scan_for_sensitive(csv_text)
    )
    assert findings == []


def test_the_provenance_records_a_phase_that_trained_nothing(paths, manifest):
    path = paths.reports / "segmentation_adapter_audit.provenance.json"
    if not path.is_file():
        pytest.skip("segmentation_adapter_audit.provenance.json not present")
    record = json.loads(path.read_text(encoding="utf-8"))
    details = record["details"]
    assert record["phase"] == 8
    assert details["phase"] == "8A"
    assert details["classification"] == "SEGMENTATION_ADAPTER_AUDIT_COMPLETE"
    assert details["models_trained_in_this_phase"] == 0
    assert details["models_evaluated_in_this_phase"] == 0
    assert details["detector_touched"] is False
    assert details["holdout_accessed"] is False
    assert details["instance_cardinality_preserved"] is True
    assert details["segmentation_architecture_selection"] == "UNSELECTED_PENDING_FIDELITY_REVIEW"
    assert details["S0"] == "NOT_DEFINED"


def test_the_frozen_detector_is_referenced_but_untouched(paths, manifest):
    detector_path = paths.reports / "final_detector_manifest.json"
    if not detector_path.is_file():
        pytest.skip("final_detector_manifest.json not present")
    detector = json.loads(detector_path.read_text(encoding="utf-8"))
    assert manifest["frozen_detector_sha256"] == detector["final_detector_sha256"]
    assert detector["status"] == "FROZEN"


def test_no_bulk_adapter_data_is_tracked_by_git(paths, manifest):
    import subprocess

    completed = subprocess.run(
        ["git", "ls-files", manifest["adapter"]["root"]],
        cwd=paths.root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        pytest.skip("git is not available here")
    assert completed.stdout.strip() == ""
