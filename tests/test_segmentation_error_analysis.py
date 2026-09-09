"""Tests for the S0 validation error analysis.

The algorithmic tests use synthetic fixtures: bands, flags and selection are
arithmetic over a table, and a failure should point at the logic rather than at
a model. The artifact tests check that the committed analysis describes the
frozen S0 run, reuses the phase 8C protocol rather than a new one, and claims
nothing this phase could not establish.

No test here trains, infers, or reads the holdout.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pytest

from construction_safety_vision.data.canonical import scan_for_sensitive
from construction_safety_vision.mask_iou_evaluation import (
    GROUND_TRUTH_SOURCE,
    MATCHING_ALGORITHM,
    load_mask_iou_config,
)
from construction_safety_vision.paths import ProjectPaths
from construction_safety_vision.provenance import sha256_file
from construction_safety_vision.segmentation_error_analysis import (
    ADAPTER_FIDELITY_RISK,
    ADAPTER_FIDELITY_RISK_IOU,
    AUTOMATIC_FLAGS,
    BOUNDARY_ERROR,
    DETECTION_MISS,
    FORBIDDEN_FLAGS,
    HIGH_QUALITY_MASK,
    IOU_HIGH,
    IOU_LOW,
    LOW_OVERLAP_MASK,
    MANUAL_ONLY_FLAGS,
    MECHANISM_FLAGS,
    MODERATE_MASK,
    OUTCOME_BANDS,
    OVERSEGMENTATION,
    TINY_MASK_AREA_PX,
    TINY_MASK_ASSOCIATED,
    UNATTRIBUTED,
    UNDERSEGMENTATION,
    ErrorAnalysisError,
    InstanceRow,
    automatic_flags,
    outcome_band,
    overlap_resolved_targets,
    review_identifiers,
    select_review_set,
    summarise,
    validate_manual_attributions,
)
from construction_safety_vision.splits import HOLDOUT_UNLOCK_ENV_VAR

ANALYSIS_JSON = "segmentation_S0_error_analysis.json"
ANALYSIS_MD = "segmentation_S0_error_analysis.md"
INSTANCES_CSV = "segmentation_S0_error_instances.csv"

EXPECTED_CHECKPOINT = "d7b512b95fafdc8658fd802459d2c87d9ad3f11f922162a774dacf8ca75b87a3"
EXPECTED_GT = 304
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def row(
    *,
    image: str = "img",
    annotation: int = 1,
    class_name: str = "person",
    area: int = 50_000,
    adapter: float | None = 0.99,
    matched: bool = True,
    iou: float = 0.8,
    predicted: int | None = 50_000,
) -> InstanceRow:
    """Build a synthetic instance row."""
    band = outcome_band(matched=matched, mask_iou=iou)
    relative = None if predicted is None or not area else (predicted - area) / area
    return InstanceRow(
        image_id=image,
        annotation_id=annotation,
        class_name=class_name,
        canonical_area_px=area,
        adapter_mask_iou=adapter,
        adapter_geometry_type="CANONICAL_POLYGON",
        connected_components=1,
        hole_count=0,
        matched=matched,
        mask_iou=iou,
        confidence=0.9 if matched else None,
        predicted_area_px=predicted if matched else None,
        false_positive_px=0 if matched else None,
        false_negative_px=0 if matched else None,
        band=band,
        automatic_flags=automatic_flags(
            band=band,
            canonical_area_px=area,
            adapter_mask_iou=adapter,
            relative_area_error=relative if matched else None,
        ),
        contested_fraction=0.0,
        overlap_resolved_iou=iou,
        false_negative_contested_share=None,
    )


# --- band arithmetic ----------------------------------------------------------


@pytest.mark.parametrize(
    ("matched", "iou", "expected"),
    [
        (False, 0.0, DETECTION_MISS),
        (True, 0.0, DETECTION_MISS),
        (True, 0.01, LOW_OVERLAP_MASK),
        (True, 0.4999, LOW_OVERLAP_MASK),
        (True, IOU_LOW, MODERATE_MASK),
        (True, 0.7499, MODERATE_MASK),
        (True, IOU_HIGH, HIGH_QUALITY_MASK),
        (True, 1.0, HIGH_QUALITY_MASK),
    ],
)
def test_the_bands_are_exhaustive_and_boundary_inclusive(matched, iou, expected):
    assert outcome_band(matched=matched, mask_iou=iou) == expected


def test_a_zero_iou_match_is_a_miss_not_a_low_overlap():
    # The matcher already discards zero-overlap pairs; if one ever arrived, it is
    # a miss rather than the worst possible match.
    assert outcome_band(matched=True, mask_iou=0.0) == DETECTION_MISS


def test_the_bands_partition_a_population():
    rows = [row(iou=value, annotation=index) for index, value in enumerate([0.9, 0.6, 0.3])]
    rows.append(row(annotation=99, matched=False, iou=0.0, predicted=None))
    counts = {band: sum(1 for item in rows if item.band == band) for band in OUTCOME_BANDS}
    assert sum(counts.values()) == len(rows)


# --- automatic flags ----------------------------------------------------------


def test_a_tiny_mask_is_flagged():
    assert TINY_MASK_ASSOCIATED in row(area=TINY_MASK_AREA_PX - 1).automatic_flags
    assert TINY_MASK_ASSOCIATED not in row(area=TINY_MASK_AREA_PX).automatic_flags


def test_a_low_fidelity_adapter_instance_is_flagged():
    assert ADAPTER_FIDELITY_RISK in row(adapter=ADAPTER_FIDELITY_RISK_IOU - 0.01).automatic_flags
    assert ADAPTER_FIDELITY_RISK not in row(adapter=ADAPTER_FIDELITY_RISK_IOU).automatic_flags


def test_a_much_smaller_prediction_is_undersegmentation():
    flags = row(iou=0.3, area=1000, predicted=400).automatic_flags
    assert UNDERSEGMENTATION in flags
    assert OVERSEGMENTATION not in flags


def test_a_much_larger_prediction_is_oversegmentation():
    flags = row(iou=0.3, area=1000, predicted=5000).automatic_flags
    assert OVERSEGMENTATION in flags
    assert UNDERSEGMENTATION not in flags


def test_a_similar_area_but_poor_overlap_is_a_boundary_error():
    flags = row(iou=0.4, area=10_000, predicted=10_500).automatic_flags
    assert BOUNDARY_ERROR in flags


def test_a_good_mask_gets_no_support_flag():
    flags = row(iou=0.95, area=10_000, predicted=10_000).automatic_flags
    assert UNDERSEGMENTATION not in flags
    assert OVERSEGMENTATION not in flags
    assert BOUNDARY_ERROR not in flags
    assert flags == (UNATTRIBUTED,)


def test_an_unexplained_instance_is_unattributed_not_guessed():
    assert row(matched=False, iou=0.0, predicted=None).automatic_flags == (UNATTRIBUTED,)


def test_manual_only_mechanisms_are_never_assigned_automatically():
    for flag in MANUAL_ONLY_FLAGS:
        assert flag not in AUTOMATIC_FLAGS
        for candidate in (
            row(iou=0.1, area=500, predicted=50),
            row(iou=0.6, area=100_000, predicted=99_000),
            row(matched=False, iou=0.0, predicted=None),
        ):
            assert flag not in candidate.automatic_flags


# --- overlap-resolved targets -------------------------------------------------


def test_the_smaller_instance_wins_contested_pixels():
    # This is the framework's rule, and the whole overlap analysis rests on it.
    big = np.zeros((10, 10), dtype=bool)
    big[0:10, 0:10] = True
    small = np.zeros((10, 10), dtype=bool)
    small[2:5, 2:5] = True
    resolved = overlap_resolved_targets([big, small])
    assert resolved[1].sum() == 9
    assert resolved[0].sum() == 100 - 9
    assert not (resolved[0] & resolved[1]).any()


def test_disjoint_instances_are_unchanged_by_resolution():
    first = np.zeros((10, 10), dtype=bool)
    first[0:3, 0:3] = True
    second = np.zeros((10, 10), dtype=bool)
    second[7:10, 7:10] = True
    resolved = overlap_resolved_targets([first, second])
    assert (resolved[0] == first).all()
    assert (resolved[1] == second).all()


def test_no_masks_resolve_to_nothing():
    assert overlap_resolved_targets([]) == {}


# --- aggregation --------------------------------------------------------------


def test_group_summaries_separate_the_two_iou_readings():
    rows = [row(iou=1.0, annotation=1), row(matched=False, iou=0.0, predicted=None, annotation=2)]
    summary = summarise(rows, lambda item: "all")["all"]
    assert summary["instances"] == 2
    assert summary["matched"] == 1
    assert summary["misses"] == 1
    assert summary["matched_mask_iou_mean"] == pytest.approx(1.0)
    assert summary["gt_normalized_mask_iou"] == pytest.approx(0.5)
    assert summary["gt_match_coverage"] == pytest.approx(0.5)


def test_a_group_with_no_match_reports_none_rather_than_zero():
    rows = [row(matched=False, iou=0.0, predicted=None)]
    summary = summarise(rows, lambda item: "all")["all"]
    assert summary["matched_mask_iou_mean"] is None
    assert summary["gt_normalized_mask_iou"] == pytest.approx(0.0)


# --- deterministic selection --------------------------------------------------


def test_selection_is_deterministic_and_order_independent():
    rows = [row(annotation=index, iou=value) for index, value in enumerate([0.1, 0.9, 0.4, 0.7])]
    first = select_review_set(rows, focus_class="person", supported_classes=["person"])
    second = select_review_set(
        list(reversed(rows)), focus_class="person", supported_classes=["person"]
    )
    assert first == second


def test_selection_takes_the_worst_and_the_best(monkeypatch):
    rows = [row(annotation=index, iou=value) for index, value in enumerate([0.1, 0.9, 0.4, 0.7])]
    selection = select_review_set(
        rows, focus_class="person", supported_classes=["person"], worst_focus=2, best_focus=2
    )
    worst = [entry["mask_iou"] for entry in selection["worst_matched_person"]]
    best = [entry["mask_iou"] for entry in selection["best_matched_person"]]
    assert worst == [0.1, 0.4]
    assert best == [0.9, 0.7]


def test_selection_separates_misses_from_poor_masks():
    rows = [
        row(annotation=1, iou=0.2),
        row(annotation=2, matched=False, iou=0.0, predicted=None),
    ]
    selection = select_review_set(rows, focus_class="person", supported_classes=["person"])
    assert [entry["annotation_id"] for entry in selection["misses_person"]] == [2]
    assert [entry["annotation_id"] for entry in selection["worst_matched_person"]] == [1]


def test_ties_resolve_by_identity():
    rows = [
        row(image="b", annotation=1, iou=0.5),
        row(image="a", annotation=2, iou=0.5),
        row(image="a", annotation=1, iou=0.5),
    ]
    selection = select_review_set(
        rows, focus_class="person", supported_classes=["person"], worst_focus=3
    )
    picked = [
        (entry["image_id"], entry["annotation_id"]) for entry in selection["worst_matched_person"]
    ]
    assert picked == [("a", 1), ("a", 2), ("b", 1)]


def test_an_instance_selected_twice_is_inspected_once():
    rows = [row(annotation=1, iou=0.5)]
    selection = select_review_set(rows, focus_class="person", supported_classes=["person"])
    assert review_identifiers(selection) == ["img#1"]


# --- manual attribution validation --------------------------------------------


def _selection():
    return select_review_set(
        [row(annotation=1, iou=0.2)], focus_class="person", supported_classes=["person"]
    )


def test_a_judgement_outside_the_review_set_is_refused():
    with pytest.raises(ErrorAnalysisError, match="not in the deterministic review set"):
        validate_manual_attributions(
            {"other#9": {"flags": (UNDERSEGMENTATION,), "evidence": "x"}}, _selection()
        )


def test_a_judgement_without_evidence_is_refused():
    with pytest.raises(ErrorAnalysisError, match="records no evidence"):
        validate_manual_attributions(
            {"img#1": {"flags": (UNDERSEGMENTATION,), "evidence": "  "}}, _selection()
        )


def test_a_judgement_with_no_flag_is_refused():
    with pytest.raises(ErrorAnalysisError, match="carries no flag"):
        validate_manual_attributions({"img#1": {"flags": (), "evidence": "x"}}, _selection())


@pytest.mark.parametrize("flag", FORBIDDEN_FLAGS)
def test_a_causal_label_this_phase_cannot_establish_is_refused(flag):
    with pytest.raises(ErrorAnalysisError, match="controlled experiment"):
        validate_manual_attributions({"img#1": {"flags": (flag,), "evidence": "x"}}, _selection())


def test_an_unknown_flag_is_refused():
    with pytest.raises(ErrorAnalysisError, match="outside the taxonomy"):
        validate_manual_attributions(
            {"img#1": {"flags": ("SOMETHING_ELSE",), "evidence": "x"}}, _selection()
        )


def test_unattributed_cannot_coexist_with_a_mechanism():
    with pytest.raises(ErrorAnalysisError, match="both UNATTRIBUTED"):
        validate_manual_attributions(
            {"img#1": {"flags": (UNATTRIBUTED, UNDERSEGMENTATION), "evidence": "x"}}, _selection()
        )


def test_a_valid_judgement_is_accepted():
    validate_manual_attributions(
        {"img#1": {"flags": (UNDERSEGMENTATION,), "evidence": "the jacket is missing"}},
        _selection(),
    )


# --- the committed artifacts --------------------------------------------------


@pytest.fixture(scope="module")
def paths() -> ProjectPaths:
    return ProjectPaths.from_root()


@pytest.fixture(scope="module")
def analysis(paths: ProjectPaths) -> dict:
    path = paths.reports / ANALYSIS_JSON
    if not path.is_file():
        pytest.skip(f"{ANALYSIS_JSON} not present; run the phase 8D analysis")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def report(paths: ProjectPaths) -> str:
    path = paths.reports / ANALYSIS_MD
    if not path.is_file():
        pytest.skip(f"{ANALYSIS_MD} not present; run the phase 8D analysis")
    return path.read_text(encoding="utf-8")


def test_the_analysis_used_the_frozen_s0_checkpoint(analysis: dict):
    checkpoint = analysis["checkpoint"]
    assert checkpoint["sha256"] == EXPECTED_CHECKPOINT
    assert checkpoint["name"] == "best.pt"
    assert checkpoint["last_pt_used"] is False
    assert checkpoint["checkpoints_compared"] == 0


def test_the_analysis_reused_the_phase_8c_protocol(paths: ProjectPaths, analysis: dict):
    protocol = load_mask_iou_config(paths.configs / "segmentation_mask_iou_evaluation.yaml")
    assert analysis["protocol_fingerprint"] == protocol.fingerprint()
    assert analysis["protocol_reused_unchanged"] is True
    assert analysis["matching"]["algorithm"] == MATCHING_ALGORITHM
    assert analysis["matching"]["reused_from_phase_8c"] is True
    assert analysis["matching"]["redefined"] is False


def test_the_operational_settings_are_the_frozen_ones(analysis: dict):
    inference = analysis["inference"]
    assert inference["conf"] == 0.25
    assert inference["iou"] == 0.70
    assert inference["imgsz"] == 768
    assert inference["max_det"] == 300
    assert inference["augment"] is False
    assert analysis["thresholds_changed"] == 0


def test_ground_truth_is_the_canonical_coco(analysis: dict):
    assert analysis["ground_truth_source"] == GROUND_TRUTH_SOURCE


def test_nothing_was_trained_or_revalidated(analysis: dict):
    assert analysis["models_trained"] == 0
    assert analysis["s0_revalidated"] is False
    assert analysis["s0_modified"] is False
    assert analysis["alternative_models_tested"] == 0


def test_the_analysis_covers_every_canonical_instance(analysis: dict):
    assert analysis["gt_instances"] == EXPECTED_GT
    assert sum(analysis["band_census"].values()) == EXPECTED_GT
    assert set(analysis["band_census"]) == set(OUTCOME_BANDS)


def test_the_analysis_reproduces_the_committed_phase_8c_aggregates(analysis: dict):
    check = analysis["cross_check"]
    assert check["agree"] is True
    for name, value in check["recomputed"].items():
        assert check["committed"][name] == pytest.approx(value, abs=1e-6)


def test_the_person_focus_is_derived_from_the_committed_gap(paths: ProjectPaths, analysis: dict):
    manifest = json.loads(
        (paths.reports / "segmentation_S0_result_manifest.json").read_text(encoding="utf-8")
    )
    person = analysis["person_diagnostic"]
    committed = manifest["per_class_metrics"]["person"]
    assert person["box_ap50_95"] == pytest.approx(committed["box"]["AP@0.50:0.95"])
    assert person["mask_ap50_95"] == pytest.approx(committed["mask"]["AP@0.50:0.95"])
    assert person["box_minus_mask"] == pytest.approx(
        person["box_ap50_95"] - person["mask_ap50_95"], abs=1e-6
    )
    # It really is the largest gap among supported classes, not merely asserted.
    gaps = {
        name: entry["box_minus_mask"]
        for name, entry in analysis["box_vs_mask"].items()
        if entry["supported"]
    }
    assert max(gaps, key=gaps.get) == "person"


def test_every_class_appears_in_the_cross_class_context(analysis: dict):
    expected = {"helmet_loose", "helmet_on_head", "person", "vest_loose", "vest_on_body"}
    assert set(analysis["per_class"]) == expected
    assert set(analysis["box_vs_mask"]) == expected
    assert sum(entry["instances"] for entry in analysis["per_class"].values()) == EXPECTED_GT


def test_vest_loose_keeps_its_uncertainty(analysis: dict):
    assert analysis["rare_class"]["name"] == "vest_loose"
    assert analysis["rare_class"]["status"] == "DESCRIPTIVE_HIGH_UNCERTAINTY"
    assert analysis["support_classification"]["vest_loose"]["supported"] is False
    assert "vest_loose" not in analysis["supported_classes"]


def test_the_taxonomy_was_frozen_before_review(analysis: dict):
    taxonomy = analysis["taxonomy"]
    assert taxonomy["frozen_before_visual_review"] is True
    assert set(taxonomy["mechanism_flags"]) == set(MECHANISM_FLAGS)
    for flag in MANUAL_ONLY_FLAGS:
        assert flag not in taxonomy["automatic_flags"]


def test_the_review_set_was_selected_before_inspection(analysis: dict):
    review = analysis["qualitative_review"]
    assert "before any image was opened" in review["selection_method"]
    assert review["instances_selected"] > 0
    assert review["figures_committed"] is False


def test_the_inspected_count_is_not_inflated_by_the_selected_count(analysis: dict):
    review = analysis["qualitative_review"]
    # The two are different numbers and the artifact must not conflate them: the
    # rules select more than a person looks at in detail.
    assert review["instances_visually_inspected"] == len(review["manual_attributions"])
    assert review["instances_visually_inspected"] <= review["instances_selected"]


def test_every_manual_judgement_is_in_the_review_set(analysis: dict):
    review = analysis["qualitative_review"]
    allowed = {
        f"{entry['image_id']}#{entry['annotation_id']}"
        for entries in review["selection_rules"].values()
        for entry in entries
    }
    for key, entry in review["manual_attributions"].items():
        assert key in allowed, key
        assert entry["evidence"].strip()
        for flag in entry["flags"]:
            assert flag in MECHANISM_FLAGS
            assert flag not in FORBIDDEN_FLAGS


def test_no_forbidden_causal_label_appears_anywhere(analysis: dict, report: str):
    serialised = json.dumps(analysis)
    for flag in FORBIDDEN_FLAGS:
        assert flag not in serialised
        assert flag not in report


def test_the_overlap_analysis_is_labelled_post_hoc(analysis: dict):
    overlap = analysis["overlap_target_analysis"]
    assert overlap["label"] == "POST_HOC_HYPOTHESIS_GENERATING"
    assert overlap["canonical_gt_does_not_resolve_overlap"] is True
    assert "not a finding" in json.dumps(overlap).lower() or overlap["not_a_finding_because"]
    # The signature the mechanism predicts: person moves, compact classes do not.
    both = overlap["gt_normalized_iou_scored_both_ways"]
    person_delta = both["person"]["overlap_resolved"] - both["person"]["canonical"]
    for name in ("helmet_loose", "helmet_on_head"):
        assert abs(both[name]["overlap_resolved"] - both[name]["canonical"]) < 0.01
    assert person_delta > 0.01


def test_no_s1_result_exists(analysis: dict, paths: ProjectPaths):
    assert analysis["final_segmenter"] == "UNSELECTED_PENDING_REVIEW"
    assert not (paths.reports / "segmentation_S1_result_manifest.json").exists()
    for entry in analysis["candidates"].values():
        assert "verdict" in entry
        assert entry["verdict"] != "SELECTED"


def test_hypotheses_are_labelled_as_hypotheses(analysis: dict):
    assert analysis["hypotheses"]
    for name, entry in analysis["hypotheses"].items():
        assert entry["status"].startswith("HYPOTHESIS"), name
        assert entry["supported_by"].strip()
        assert entry["weakened_by"].strip()


def test_the_classification_is_one_of_the_four(analysis: dict):
    assert analysis["classification"] in {
        "S1_HYPOTHESIS_READY_FOR_PROTOCOL_FREEZE",
        "MORE_DIAGNOSTIC_EVIDENCE_REQUIRED",
        "NO_FURTHER_SEGMENTATION_EXPERIMENT_JUSTIFIED",
        "BLOCKED",
    }


def test_the_historical_artifacts_are_unchanged(paths: ProjectPaths, analysis: dict):
    for name, expected in analysis["historical_artifact_digests"].items():
        assert sha256_file(paths.root / name) == expected, f"{name} changed since phase 8D"


def test_the_instance_table_has_one_row_per_canonical_instance(paths: ProjectPaths, analysis: dict):
    import csv

    path = paths.reports / INSTANCES_CSV
    if not path.is_file():
        pytest.skip("instance table not present")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == EXPECTED_GT
    assert analysis["instance_table"]["sha256"] == sha256_file(path)
    assert {row["class"] for row in rows} == set(analysis["per_class"])


def test_the_holdout_is_untouched(analysis: dict):
    assert analysis["test"]["status"] == "PROTECTED_NOT_ACCESSED"
    assert analysis["holdout_accessed"] is False
    serialised = json.dumps(analysis)
    assert "images/test" not in serialised
    assert "labels/test" not in serialised
    assert HOLDOUT_UNLOCK_ENV_VAR not in serialised


def test_no_test_path_is_reachable_from_the_analysis_command(paths: ProjectPaths):
    source = (paths.scripts / "analyze_segmentation_errors.py").read_text(encoding="utf-8")
    assert "split='test'" not in source
    assert '"test"' not in source.replace('"test": {', "").replace('"test"]', "")
    assert "allow_test" not in source


def test_no_training_path_exists_in_the_analysis_command(paths: ProjectPaths):
    source = (paths.scripts / "analyze_segmentation_errors.py").read_text(encoding="utf-8")
    assert ".train(" not in source
    assert "model.val(" not in source


def test_the_artifacts_carry_no_sensitive_content(analysis: dict, report: str):
    assert scan_for_sensitive(report) == []
    assert scan_for_sensitive(json.dumps(analysis)) == []


def test_the_report_declares_its_sections(report: str):
    for heading in (
        "## 1. What this analysis is, and what it is not",
        "## 2. Outcome census",
        "## 3. The `person` diagnostic",
        "## 4. Cross-class context",
        "## 5. Stratified diagnostics",
        "## 6. Adapter fidelity versus model error",
        "## 6b. The overlap-resolved training target",
        "## 7. Deterministic qualitative review",
        "## 8. Box versus mask, per class",
        "## 9. Hypotheses for a future experiment",
        "## 10. Candidate interventions, assessed not chosen",
        "## 11. Limitations",
        "## 12. Holdout compliance",
        "## 13. Next decision",
    ):
        assert heading in report, heading
