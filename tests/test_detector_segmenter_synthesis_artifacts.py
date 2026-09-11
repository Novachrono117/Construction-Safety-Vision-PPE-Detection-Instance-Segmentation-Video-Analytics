"""Tests for the committed phase 10D synthesis artifacts.

These read what phase 10D actually wrote: the JSON synthesis, the human-readable
report and the trade-off table. They re-derive the synthesis from the committed
phase 10B and 10C evidence and check it reproduces byte for byte, so a hand edit
to any artifact fails here rather than travelling into the academic report.

No model is loaded, no inference is run and no holdout data is read.
"""

from __future__ import annotations

import csv
import json
from typing import Any

import pytest

from construction_safety_vision.detector_segmenter_comparison import (
    DETECTOR_EXPERIMENT,
    END_TO_END_LATENCY,
    MODEL_INFERENCE_LATENCY,
    SEGMENTER_EXPERIMENT,
)
from construction_safety_vision.detector_segmenter_synthesis import (
    AXES,
    CLAIM_FIELDS,
    CLAIM_IDS,
    HISTORICAL,
    HOLDOUT_STATUS,
    LATENCY_SCOPE,
    PHASE,
    SOURCES,
    SYNTHESIS_COMPLETE,
    TRADEOFF_FIELDS,
    build_synthesis,
    load_sources,
    render_markdown,
    tradeoff_rows,
    validate_synthesis,
)
from construction_safety_vision.paths import ProjectPaths
from construction_safety_vision.provenance import sha256_file

PATHS = ProjectPaths.from_root()
ROOT = PATHS.root

SYNTHESIS_JSON = PATHS.reports / "detector_segmenter_scientific_synthesis.json"
SYNTHESIS_MD = PATHS.reports / "detector_segmenter_scientific_synthesis.md"
TRADEOFF_CSV = PATHS.reports / "detector_segmenter_tradeoff.csv"
PROVENANCE = PATHS.reports / "detector_segmenter_synthesis.provenance.json"


@pytest.fixture(scope="module")
def committed() -> dict[str, Any]:
    return json.loads(SYNTHESIS_JSON.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def sources() -> dict[str, dict[str, Any]]:
    return load_sources(ROOT)


def test_every_artifact_exists() -> None:
    for path in (SYNTHESIS_JSON, SYNTHESIS_MD, TRADEOFF_CSV, PROVENANCE):
        assert path.is_file(), path.name


def test_the_committed_synthesis_validates(
    committed: dict[str, Any], sources: dict[str, dict[str, Any]]
) -> None:
    assert validate_synthesis(committed, sources) == []


def test_the_committed_synthesis_rederives_from_committed_evidence(
    committed: dict[str, Any], sources: dict[str, dict[str, Any]]
) -> None:
    """A hand edit to the synthesis fails here rather than reaching the report."""
    rebuilt = build_synthesis(ROOT, sources)
    assert rebuilt == committed
    assert rebuilt["synthesis_sha256"] == committed["synthesis_sha256"]


def test_the_committed_report_rederives(committed: dict[str, Any]) -> None:
    assert SYNTHESIS_MD.read_text(encoding="utf-8") == render_markdown(committed)


def test_the_committed_tradeoff_table_rederives(committed: dict[str, Any]) -> None:
    with TRADEOFF_CSV.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [dict(row) for row in rows] == tradeoff_rows(committed)


def test_source_digests_match_the_artifacts_on_disk(committed: dict[str, Any]) -> None:
    for role, relative in SOURCES.items():
        recorded = committed["source_artifacts"][role]
        assert recorded["path"] == relative
        assert recorded["file_sha256"] == sha256_file(ROOT / relative)


def test_historical_artifacts_are_still_present_and_readable() -> None:
    for relative in HISTORICAL:
        assert (ROOT / relative).is_file(), relative


def test_status_phase_and_axes(committed: dict[str, Any]) -> None:
    assert committed["status"] == SYNTHESIS_COMPLETE
    assert committed["phase"] == PHASE == "10D"
    assert committed["axes"] == list(AXES)
    assert len(committed["axes"]) == 4


def test_headline_numbers_in_the_committed_artifact(committed: dict[str, Any]) -> None:
    recognition = committed["recognition"]
    assert recognition["all_class"]["CANONICAL_BOX_MAP50_95"]["delta"] == 0.020292
    assert recognition["supported_class_sensitivity"]["delta"] == -0.00804
    assert committed["spatial_representation"]["MASK_TO_BOX_FILL_RATIO"]["median"] == 0.664433
    assert committed["association"]["mask_only_associations"] == 0
    latency = committed["cost"]["latency"]["boundaries"]
    assert latency[MODEL_INFERENCE_LATENCY]["absolute_latency_delta_ms"] == 1.721747
    assert latency[END_TO_END_LATENCY]["absolute_latency_delta_ms"] == 2.756991
    assert committed["cost"]["memory"]["peak_reserved"]["ratio"] == 2.375


def test_no_composite_score_anywhere_in_the_committed_json(committed: dict[str, Any]) -> None:
    text = json.dumps(committed).lower()
    for forbidden in ('cost_benefit_index": 0.', 'overall_benefit_score": 0.'):
        assert forbidden not in text
    assert committed["aggregate_benefit_score"] is False
    assert committed["weighted_score"] is False
    assert committed["winner_declared"] is False


def test_claim_register_is_committed_in_full(committed: dict[str, Any]) -> None:
    register = committed["claim_register"]
    assert tuple(entry["claim_id"] for entry in register) == CLAIM_IDS
    for entry in register:
        assert set(CLAIM_FIELDS) <= set(entry)
        for field in CLAIM_FIELDS:
            assert entry[field]


def test_holdout_is_protected_in_every_artifact(committed: dict[str, Any]) -> None:
    assert committed["holdout_policy"]["status"] == HOLDOUT_STATUS
    assert committed["test"]["status"] == HOLDOUT_STATUS
    assert committed["execution"]["test_accessed"] is False
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    assert provenance["details"]["holdout"] == HOLDOUT_STATUS
    assert provenance["details"]["execution"]["models_executed"] == 0


def test_no_holdout_derived_value_in_any_artifact(committed: dict[str, Any]) -> None:
    """No artifact may carry a holdout-derived number.

    This is structural rather than a substring hunt: the artifacts *do* discuss
    the holdout, to say it was never read, and a phrase like "no figure exists
    for it" is the disclosure working rather than a leak. What must not exist is
    a value.
    """
    test_block = committed["test"]
    assert test_block["images_read"] == 0
    assert test_block["predictions"] == 0
    assert test_block["statistics"] == 0

    def numeric_under(node: Any, key_path: str = "") -> list[str]:
        found: list[str] = []
        if isinstance(node, dict):
            for key, value in node.items():
                where = f"{key_path}.{key}" if key_path else str(key)
                lowered = str(key).lower()
                if (
                    ("test" in lowered or "holdout" in lowered)
                    and isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and value != 0
                ):
                    found.append(where)
                found.extend(numeric_under(value, where))
        elif isinstance(node, list):
            for index, value in enumerate(node):
                found.extend(numeric_under(value, f"{key_path}[{index}]"))
        return found

    assert numeric_under(committed) == []


def test_provenance_records_zero_execution() -> None:
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    details = provenance["details"]
    assert details["phase"] == PHASE
    assert details["classification"] == SYNTHESIS_COMPLETE
    assert details["historical_artifacts_unchanged"] is True
    execution = details["execution"]
    assert execution["models_executed"] == 0
    assert execution["latency_measurements_taken"] == 0
    assert execution["AP_recomputed"] is False
    assert execution["spatial_analysis_rerun"] is False
    assert details["next_phase"] == "PHASE_11A_FINAL_HOLDOUT_EVALUATION_PROTOCOL_FREEZE"


def test_report_discloses_the_confidence_protocol_gap() -> None:
    text = SYNTHESIS_MD.read_text(encoding="utf-8")
    assert "PRE_BENCHMARK_PROTOCOL_GAP_RESOLUTION" in text
    assert LATENCY_SCOPE in text
    assert "no confidence threshold" in text
    assert "No latency claim is made for conf 0.001" in text


def test_report_preserves_the_dvfs_evidentiary_status() -> None:
    text = SYNTHESIS_MD.read_text(encoding="utf-8")
    assert "causal_attribution: UNKNOWN" in text
    assert "UNTESTED_HYPOTHESIS" in text
    assert "NO_SYNCHRONIZED_PER_OBSERVATION_POWER_STATE_TELEMETRY" in text
    assert "not stated that DVFS caused" in text


def test_report_preserves_the_rare_class_caveat() -> None:
    text = SYNTHESIS_MD.read_text(encoding="utf-8")
    assert "DESCRIPTIVE_HIGH_UNCERTAINTY" in text
    assert "Never quote the all-class delta on its own." in text


def test_report_preserves_the_taxonomy_exception() -> None:
    text = SYNTHESIS_MD.read_text(encoding="utf-8")
    assert "UNCLASSIFIED_BY_FROZEN_FOUR_CATEGORY_TAXONOMY" in text
    assert "fifth_peer_category_added: false" in text


def test_report_names_the_next_phase_without_starting_it() -> None:
    text = SYNTHESIS_MD.read_text(encoding="utf-8")
    assert "Phase 11A" in text
    assert "LOCKED" in text
    assert "CSVISION_ALLOW_TEST_SPLIT=1" in text


def test_tradeoff_table_has_no_score_or_ranking_column() -> None:
    with TRADEOFF_CSV.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == list(TRADEOFF_FIELDS)
        rows = list(reader)
    assert len(rows) >= 10
    joined = json.dumps(rows).lower()
    for forbidden in ("overall score", "★", "rank:", "weighted"):
        assert forbidden not in joined


def test_tradeoff_table_distinguishes_both_models(committed: dict[str, Any]) -> None:
    with TRADEOFF_CSV.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_dimension = {row["dimension"]: row for row in rows}
    latency_row = by_dimension[f"Mean {END_TO_END_LATENCY}"]
    boundary = committed["cost"]["latency"]["boundaries"][END_TO_END_LATENCY]
    assert latency_row["D2"] == str(boundary[DETECTOR_EXPERIMENT]["mean"])
    assert latency_row["S1"] == str(boundary[SEGMENTER_EXPERIMENT]["mean"])
    assert "relative" in latency_row["comparison"]
