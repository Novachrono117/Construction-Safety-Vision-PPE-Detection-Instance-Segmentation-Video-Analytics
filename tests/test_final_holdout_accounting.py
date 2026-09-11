"""Phase 11B execution-accounting clarification.

The clarification exists because "one-shot" and ``models_executed: 2`` can both
be misread as "one prediction call per model". These tests pin the reading that
is true - one authorised evaluation attempt, two distinct models, three real
inference passes, the segmenter twice - and pin the honesty properties that make
the artifact worth having: the mismatch it discloses stays disclosed, the
scientific numbers stay identical to the committed results, and neither the
module nor the script can reach a model or the holdout.
"""

from __future__ import annotations

import ast
import csv
import json
import os
from pathlib import Path

import pytest

from construction_safety_vision import final_holdout_accounting as accounting
from construction_safety_vision.paths import ProjectPaths

MODULE = "src/construction_safety_vision/final_holdout_accounting.py"
SCRIPT = "scripts/clarify_holdout_execution_accounting.py"

ACCOUNTING_JSON = "final_test_execution_accounting.json"
ACCOUNTING_MARKDOWN = "final_test_execution_accounting.md"
ACCOUNTING_PROVENANCE = "final_test_execution_accounting.provenance.json"

FORBIDDEN_IMPORTS = ("torch", "ultralytics", "cv2", "PIL", "pycocotools")
"""Nothing that can load a model or decode an image belongs on this path."""

HOLDOUT_ACCESSORS = (
    "load_frozen_splits",
    "FrozenSplits",
    "assert_split_allowed",
    "authorize_final_holdout_access",
    "allow_test",
)
"""Names that would mean this clarification could reach the protected split."""


@pytest.fixture(scope="module")
def paths() -> ProjectPaths:
    return ProjectPaths.from_root()


@pytest.fixture(scope="module")
def committed(paths: ProjectPaths) -> dict:
    path = paths.reports / ACCOUNTING_JSON
    assert path.is_file(), "the execution-accounting clarification must be committed"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def results(paths: ProjectPaths) -> dict:
    return {
        name: json.loads((paths.reports / f"final_test_{name}.json").read_text(encoding="utf-8"))
        for name in ("detector", "segmenter", "direct_iou")
    }


@pytest.fixture(scope="module")
def evaluation_provenance(paths: ProjectPaths) -> dict:
    path = paths.reports / "final_test_evaluation.provenance.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def frozen_protocol(paths: ProjectPaths) -> dict:
    path = paths.reports / "final_holdout_evaluation_protocol.json"
    return json.loads(path.read_text(encoding="utf-8"))


# --- inference accounting ---------------------------------------------------------------


def test_one_evaluation_attempt(committed: dict) -> None:
    assert committed["inference_accounting"]["holdout_evaluation_attempts"] == 1
    assert committed["one_shot_semantics"]["evaluation_attempt_count"] == 1


def test_two_unique_models_three_passes(committed: dict) -> None:
    block = committed["inference_accounting"]
    assert block["unique_models_executed"] == 2
    assert block["total_model_inference_passes"] == 3


def test_per_model_invocation_counts(committed: dict) -> None:
    block = committed["inference_accounting"]
    assert block["d2_inference_invocations"] == 1
    assert block["s1_inference_invocations"] == 2
    assert (
        block["d2_inference_invocations"] + block["s1_inference_invocations"]
        == block["total_model_inference_passes"]
    )


def test_the_three_passes_are_named_in_order(committed: dict) -> None:
    names = [entry["pass"] for entry in committed["inference_accounting"]["passes"]]
    assert names == [
        "DETECTOR_AP_PASS",
        "SEGMENTER_AP_PASS",
        "SEGMENTER_OPERATIONAL_PASS_FOR_DIRECT_IOU",
    ]


def test_pass_confidences_come_from_the_frozen_protocol(
    committed: dict, frozen_protocol: dict
) -> None:
    by_name = {entry["pass"]: entry for entry in committed["inference_accounting"]["passes"]}
    assert (
        by_name["DETECTOR_AP_PASS"]["confidence"] == frozen_protocol["detector_inference"]["conf"]
    )
    assert (
        by_name["SEGMENTER_AP_PASS"]["confidence"] == frozen_protocol["segmenter_inference"]["conf"]
    )
    assert (
        by_name["SEGMENTER_OPERATIONAL_PASS_FOR_DIRECT_IOU"]["confidence"]
        == frozen_protocol["direct_iou"]["inference"]["conf"]
    )


def test_second_s1_pass_is_represented_as_real_inference(committed: dict) -> None:
    block = committed["inference_accounting"]
    operational = next(
        entry
        for entry in block["passes"]
        if entry["pass"] == "SEGMENTER_OPERATIONAL_PASS_FOR_DIRECT_IOU"
    )
    assert operational["real_model_prediction_execution"] is True
    assert operational["derived_view_of_another_pass"] is False
    representation = block["second_s1_pass_representation"]
    assert representation["was_a_real_model_predict_execution"] is True
    assert representation["described_as_derived_view"] is False


def test_second_s1_pass_really_was_a_second_predict_call(paths: ProjectPaths) -> None:
    """The claim is checked against the code that ran, not merely asserted."""
    source = (
        paths.root / "src" / "construction_safety_vision" / "final_holdout_execution.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    execute = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "execute"
    )
    run_pass_calls = [
        node
        for node in ast.walk(execute)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_pass"
    ]
    assert len(run_pass_calls) == 3, "phase 11B ran three inference passes"
    segmenter_calls = [
        call
        for call in run_pass_calls
        if ast.unparse(call.args[0]).replace('"', "'") == "models['S1']['path']"
    ]
    assert len(segmenter_calls) == 2, "the segmenter checkpoint was passed to run_pass twice"


def test_prediction_fingerprints_match_the_committed_provenance(
    committed: dict, evaluation_provenance: dict
) -> None:
    recorded = evaluation_provenance["details"]["fingerprints"]
    by_name = {entry["pass"]: entry for entry in committed["inference_accounting"]["passes"]}
    assert (
        by_name["DETECTOR_AP_PASS"]["prediction_fingerprint"]
        == recorded["detector_test_prediction_sha256"]
    )
    assert (
        by_name["SEGMENTER_AP_PASS"]["prediction_fingerprint"]
        == recorded["segmenter_test_prediction_sha256"]
    )
    assert (
        by_name["SEGMENTER_OPERATIONAL_PASS_FOR_DIRECT_IOU"]["prediction_fingerprint"]
        == recorded["segmenter_operational_test_prediction_sha256"]
    )


def test_historical_models_executed_is_disambiguated_not_reinterpreted(
    committed: dict, evaluation_provenance: dict
) -> None:
    historical = committed["inference_accounting"]["historical_field_semantics"]
    assert historical["recorded_value"] == evaluation_provenance["details"]["models_executed"]
    assert historical["means"] == "NUMBER_OF_DISTINCT_MODEL_IDENTITIES_EXECUTED"
    assert historical["does_not_mean"] == "NUMBER_OF_MODEL_INFERENCE_PASSES"
    assert historical["silently_reinterpreted"] is False
    assert historical["historical_artifact_edited"] is False
    assert set(historical["sibling_fields_added_here"]) == {
        "unique_models_executed",
        "total_model_inference_passes",
        "d2_inference_invocations",
        "s1_inference_invocations",
    }


# --- one-shot semantics -----------------------------------------------------------------


def test_one_shot_label_is_scoped_to_the_evaluation(committed: dict) -> None:
    semantics = committed["one_shot_semantics"]
    assert semantics["label"] == "ONE_SHOT_FINAL_HOLDOUT_EVALUATION_VALID"
    assert semantics["rejected_label"] == "ONE_SHOT_MODEL_PREDICTION_EXECUTION_VALID"


def test_the_per_invocation_reading_is_claimed_nowhere(paths: ProjectPaths) -> None:
    """The rejected label may appear only as the thing being rejected.

    Scanning every committed report is the point: the label is false, so a file
    that carries it without also disowning it would be asserting it.
    """
    seen = 0
    for path in sorted(paths.reports.rglob("*.json")) + sorted(paths.reports.rglob("*.md")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "ONE_SHOT_MODEL_PREDICTION_EXECUTION_VALID" not in text:
            continue
        seen += 1
        assert "reject" in text.lower(), f"{path.name} claims the rejected one-shot label"
    assert seen, "the clarification must name the label it rejects"


def test_no_adaptive_rerun_and_no_post_metric_invocation(committed: dict) -> None:
    semantics = committed["one_shot_semantics"]
    assert semantics["adaptive_prediction_rerun_count"] == 0
    assert semantics["post_metric_model_invocation_count"] == 0
    assert semantics["prediction_regeneration_after_immutability_barrier"] is False


def test_phase_11b_state_is_unchanged(committed: dict, evaluation_provenance: dict) -> None:
    assert committed["one_shot_semantics"]["phase_11b_classification"] == "TEST_EVALUATION_COMPLETE"
    assert evaluation_provenance["details"]["classification"] == "TEST_EVALUATION_COMPLETE"
    assert committed["one_shot_semantics"]["frozen_schema_prohibits_this_clarification"] is False


# --- frozen protocol context ------------------------------------------------------------


def test_phase_11a_declared_the_direct_iou_inference_block(frozen_protocol: dict) -> None:
    assert "detector_inference" in frozen_protocol
    assert "segmenter_inference" in frozen_protocol
    assert "inference" in frozen_protocol["direct_iou"]
    assert frozen_protocol["direct_iou"]["inference"]["conf"] == 0.25
    assert frozen_protocol["direct_iou"]["unchanged_from_phase_8c"] is True


def test_three_passes_were_consistent_with_the_frozen_protocol(committed: dict) -> None:
    context = committed["frozen_protocol_context"]
    assert context["FROZEN_PROTOCOL_SATISFIED"] is True
    assert context["declared_inference_block_count"] == 3
    assert context["declared_inference_blocks"] == [
        "detector_inference",
        "segmenter_inference",
        "direct_iou.inference",
    ]


def test_the_later_single_s1_invocation_instruction_was_not_satisfied(committed: dict) -> None:
    context = committed["frozen_protocol_context"]
    assert context["LATER_EXECUTION_INSTRUCTION_SINGLE_S1_INVOCATION_SATISFIED"] is False


def test_the_mismatch_is_disclosed_and_not_outcome_driven(committed: dict) -> None:
    context = committed["frozen_protocol_context"]
    assert context["mismatch_disclosed"] is True
    assert context["mismatch_caused_by_observing_holdout_outcomes"] is False
    assert context["mismatch_description"].strip()


def test_the_mismatch_reaches_the_rendered_report(paths: ProjectPaths) -> None:
    text = (paths.reports / ACCOUNTING_MARKDOWN).read_text(encoding="utf-8")
    assert "LATER_EXECUTION_INSTRUCTION_SINGLE_S1_INVOCATION_SATISFIED" in text
    assert "**false**" in text


# --- second-pass equivalence audit ------------------------------------------------------


def test_equivalence_evidence_is_65_of_65_with_zero_divergence(committed: dict) -> None:
    audit = committed["second_s1_pass_audit"]
    assert audit["SECOND_S1_PASS_EQUIVALENCE_TO_AP_FILTER"] == "VERIFIED"
    assert audit["images_compared"]["value"] == 65
    assert audit["images_total"]["value"] == 65
    assert audit["count_divergences"]["value"] == 0
    assert audit["content_divergences"]["value"] == 0
    assert audit["compared_fields"] == ["class", "score", "box", "mask_rle"]


def test_equivalence_counts_agree_with_the_committed_results(
    committed: dict, results: dict
) -> None:
    audit = committed["second_s1_pass_audit"]
    assert (
        audit["s1_ap_predictions_total"]["value"]
        == results["segmenter"]["canonical_mask"]["detections_scored"]
    )
    assert (
        audit["s1_operational_pass_predictions"]["value"]
        == results["direct_iou"]["diagnostic"]["global"]["prediction_count"]
    )
    assert audit["images_total"]["value"] == results["direct_iou"]["diagnostic"]["images"]


def test_every_audited_quantity_declares_its_evidence_status(committed: dict) -> None:
    audit = committed["second_s1_pass_audit"]
    evidence = [value for value in audit.values() if isinstance(value, dict) and "value" in value]
    assert evidence
    for entry in evidence:
        assert entry["evidence_status"] in accounting.EVIDENCE_STATUSES
        assert entry["source"]


def test_the_second_pass_is_never_rewritten_as_a_derived_view(committed: dict) -> None:
    audit = committed["second_s1_pass_audit"]
    assert audit["second_pass_was_real_inference"] is True
    assert audit["second_pass_rewritten_as_derived_view"] is False
    assert audit["required_in_hindsight"] is False
    assert (
        audit["reported_metric_dependency_on_second_execution"]
        == "NONE_BEYOND_IDENTICAL_REPRODUCTION"
    )


def test_the_audit_was_not_recomputed_here(committed: dict) -> None:
    assert committed["second_s1_pass_audit"]["recomputed_in_this_clarification"] is False


# --- nothing scientific moved -----------------------------------------------------------


def test_every_restated_metric_matches_the_committed_result(committed: dict, results: dict) -> None:
    unchanged = committed["results_unchanged"]
    detector = results["detector"]
    segmenter = results["segmenter"]
    direct = results["direct_iou"]["diagnostic"]["global"]

    assert (
        unchanged["detector"]["canonical_box_map50_95"]
        == detector["canonical_box"]["CANONICAL_TEST_BOX_MAP50_95"]
    )
    assert (
        unchanged["detector"]["canonical_box_map50"]
        == detector["canonical_box"]["CANONICAL_TEST_BOX_MAP50"]
    )
    assert (
        unchanged["segmenter"]["canonical_mask_map50_95"]
        == segmenter["canonical_mask"]["CANONICAL_TEST_MASK_MAP50_95"]
    )
    assert (
        unchanged["segmenter"]["canonical_box_map50_95"]
        == segmenter["canonical_box"]["S1_CANONICAL_TEST_BOX_MAP50_95"]
    )
    assert unchanged["direct_iou"]["matched_mask_iou_mean"] == direct["matched_mask_iou_mean"]
    assert unchanged["direct_iou"]["gt_normalized_mask_iou"] == direct["gt_normalized_mask_iou"]

    for name, payload in (("detector", detector), ("segmenter", segmenter)):
        for key in ("true_positives", "false_positives", "false_negatives", "precision", "recall"):
            assert unchanged[name][key] == payload["object_level"][key]


def test_result_and_prediction_fingerprints_are_unchanged(committed: dict, results: dict) -> None:
    unchanged = committed["results_unchanged"]
    for name in ("detector", "segmenter", "direct_iou"):
        assert unchanged[name]["result_sha256"] == results[name]["result_sha256"]
        assert unchanged[name]["prediction_fingerprint"] == results[name]["prediction_fingerprint"]
    assert (
        unchanged["segmenter"]["operational_prediction_fingerprint"]
        == results["segmenter"]["operational_prediction_fingerprint"]
    )


def test_the_protected_artifacts_are_byte_identical(committed: dict, paths: ProjectPaths) -> None:
    recorded = committed["historical_artifacts_unchanged"]
    assert set(recorded) == set(accounting.PROTECTED)
    for name, digest in recorded.items():
        assert accounting.sha256_bytes(paths.root / name) == digest, f"{name} changed"


def test_no_result_correction_is_claimed(committed: dict) -> None:
    assert committed["is_a_test_result_correction"] is False
    assert committed["scientific_results_changed"] is False
    assert committed["clarification_type"] == "PROVENANCE_METADATA_ONLY"


# --- chronology, disclosure and self-accounting -----------------------------------------


def test_chronology_does_not_overclaim(committed: dict) -> None:
    chronology = committed["chronology"]
    assert (
        chronology["RUNTIME_FILESYSTEM_EVIDENCE"]
        == "CONSISTENT_WITH_RESOLVED_BEFORE_HOLDOUT_ACCESS"
    )
    assert chronology["VCS_PRE_EXECUTION_CHECKPOINT"] == "ABSENT"
    assert (
        chronology["strongest_unambiguous_scientific_claim"]
        == "RESOLVED_BEFORE_OUTCOME_METRICS_WERE_OBSERVED"
    )
    assert chronology["mtime_treated_as_cryptographic_provenance"] is False
    assert chronology["code_and_results_committed_together"] is True


def test_post_observation_metadata_inspection_is_disclosed(committed: dict) -> None:
    access = committed["post_observation_access"]
    assert access["post_observation_test_metadata_inspection"] is True
    assert access["zero_filesystem_contact_claimed"] is False
    assert access["post_observation_test_content_access"] is False
    assert access["post_observation_model_execution"] is False
    assert access["post_observation_prediction_generation"] is False
    assert access["role_in_model_selection_tuning_or_reported_metrics"] == "NONE"


def test_this_clarification_touched_no_test_content(committed: dict) -> None:
    mine = committed["this_clarification"]
    assert mine["models_executed"] == 0
    assert mine["model_inference_passes"] == 0
    assert mine["holdout_reads"] == 0
    assert mine["test_images_read"] == 0
    assert mine["test_annotations_read"] == 0
    assert mine["test_identifiers_enumerated"] == 0
    assert mine["predictions_regenerated"] == 0
    assert mine["metrics_recomputed_from_test_data"] == 0
    assert mine["prediction_bytes_altered"] == 0
    assert mine["metric_values_altered"] == 0
    assert mine["qualitative_selection_rerun"] is False
    assert mine["holdout_accessor_invoked"] is False
    assert mine["test_image_directory_listed"] is False
    assert mine["frozen_historical_protocol_altered"] is False


def test_the_environment_gate_is_recorded_absent(committed: dict) -> None:
    authorization = committed["authorization"]
    assert authorization["environment_test_gate_present"] is False
    assert authorization["effective_holdout_access_authorized"] is False
    assert authorization["final_test_observed"] is True
    assert authorization["gate_written_by_this_clarification"] is False
    assert set(authorization["gate_scopes_verified_absent"]) == {"PROCESS", "USER", "MACHINE"}


def test_the_environment_gate_really_is_absent_now() -> None:
    assert os.environ.get(accounting.ENVIRONMENT_GATE) is None


# --- the clarification cannot reach a model or the holdout -------------------------------


@pytest.mark.parametrize("relative", [MODULE, SCRIPT])
def test_no_model_or_image_library_is_imported(paths: ProjectPaths, relative: str) -> None:
    tree = ast.parse((paths.root / relative).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported.isdisjoint(FORBIDDEN_IMPORTS), f"{relative} imports {imported}"


@pytest.mark.parametrize("relative", [MODULE, SCRIPT])
def test_no_holdout_accessor_is_named(paths: ProjectPaths, relative: str) -> None:
    text = (paths.root / relative).read_text(encoding="utf-8")
    for name in HOLDOUT_ACCESSORS:
        assert name not in text, f"{relative} names the holdout accessor {name}"


def test_the_script_never_writes_the_environment_gate(paths: ProjectPaths) -> None:
    tree = ast.parse((paths.root / SCRIPT).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and "environ" in ast.unparse(node.value):
            parents = [
                parent
                for parent in ast.walk(tree)
                if isinstance(parent, ast.Assign) and node in parent.targets
            ]
            assert not parents, "the clarification must never set the environment gate"
        if isinstance(node, ast.Call) and ast.unparse(node.func).endswith("environ.setdefault"):
            pytest.fail("the clarification must never set the environment gate")


def test_no_holdout_identifier_reaches_the_clarification(paths: ProjectPaths) -> None:
    """Whatever the artifact says, it must carry no holdout image id.

    The ids are read the way every other leak proof in this repository reads
    them - from the committed assignment table, never through the guarded
    accessor - they never leave this process, and they are never printed.
    Proving the absence of a leak requires knowing what a leak would look like.
    """
    with (paths.reports / "final_split_assignments.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        identifiers = {
            row["source_image_id"] for row in csv.DictReader(handle) if row["split"] == "test"
        }
    assert identifiers, "the fixture that makes this test meaningful is missing"
    for name in (ACCOUNTING_JSON, ACCOUNTING_MARKDOWN, ACCOUNTING_PROVENANCE):
        text = (paths.reports / name).read_text(encoding="utf-8")
        leaked = sorted(identifier for identifier in identifiers if identifier in text)
        assert leaked == [], (name, len(leaked))


# --- rebuild, validation and the digest --------------------------------------------------


def test_the_clarification_rebuilds_byte_identically(paths: ProjectPaths, committed: dict) -> None:
    rebuilt = accounting.build(paths, protected_digests=committed["historical_artifacts_unchanged"])
    assert rebuilt == committed
    assert rebuilt["execution_accounting_sha256"] == committed["execution_accounting_sha256"]


def test_the_rendered_report_rebuilds_byte_identically(
    paths: ProjectPaths, committed: dict
) -> None:
    assert accounting.render(committed) == (paths.reports / ACCOUNTING_MARKDOWN).read_text(
        encoding="utf-8"
    )


def test_the_committed_clarification_validates(committed: dict) -> None:
    assert accounting.validate(committed) == []


def test_the_fingerprint_covers_the_payload(committed: dict) -> None:
    payload = {
        key: value for key, value in committed.items() if key != "execution_accounting_sha256"
    }
    assert accounting.digest_payload(payload) == committed["execution_accounting_sha256"]


def test_the_local_digest_agrees_with_the_execution_module() -> None:
    from construction_safety_vision.final_holdout_execution import digest_payload

    sample = {"b": [1, 2, {"c": None}], "a": "x", "d": 0.5}
    assert accounting.digest_payload(sample) == digest_payload(sample)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("inference_accounting", "s1_inference_invocations"), 1),
        (("inference_accounting", "total_model_inference_passes"), 2),
        (("one_shot_semantics", "label"), "ONE_SHOT_MODEL_PREDICTION_EXECUTION_VALID"),
        (("frozen_protocol_context", "mismatch_disclosed"), False),
        (
            (
                "frozen_protocol_context",
                "LATER_EXECUTION_INSTRUCTION_SINGLE_S1_INVOCATION_SATISFIED",
            ),
            True,
        ),
        (("post_observation_access", "zero_filesystem_contact_claimed"), True),
        (("chronology", "VCS_PRE_EXECUTION_CHECKPOINT"), "PRESENT"),
        (("scientific_results_changed",), True),
    ],
)
def test_validate_refuses_a_dishonest_payload(committed: dict, path: tuple, value: object) -> None:
    payload = json.loads(json.dumps(committed))
    target = payload
    for segment in path[:-1]:
        target = target[segment]
    target[path[-1]] = value
    assert accounting.validate(payload), f"validate must refuse {path} = {value!r}"


def test_validate_refuses_a_composite_score(committed: dict) -> None:
    payload = json.loads(json.dumps(committed))
    payload["inference_accounting"]["composite_score"] = 0.5
    assert any("composite_score" in problem for problem in accounting.validate(payload))


def test_a_derived_view_representation_is_refused(committed: dict) -> None:
    payload = json.loads(json.dumps(committed))
    for entry in payload["inference_accounting"]["passes"]:
        if entry["pass"] == "SEGMENTER_OPERATIONAL_PASS_FOR_DIRECT_IOU":
            entry["derived_view_of_another_pass"] = True
            entry["real_model_prediction_execution"] = False
    assert accounting.validate(payload)


def test_field_raises_on_a_missing_path() -> None:
    with pytest.raises(accounting.AccountingError):
        accounting.field({"a": {"b": 1}}, "a.c")


def test_the_provenance_record_reports_zero_execution(paths: ProjectPaths) -> None:
    record = json.loads((paths.reports / ACCOUNTING_PROVENANCE).read_text(encoding="utf-8"))
    details = record["details"]
    assert details["classification"] == accounting.CLASSIFICATION
    assert details["models_executed"] == 0
    assert details["models_loaded"] == 0
    assert details["model_inference_passes"] == 0
    assert details["holdout_accessed"] is False
    assert details["holdout_accessor_invoked"] is False
    assert details["scientific_results_changed"] is False
    assert details["environment_test_gate_present"] is False
    assert details["clarified"]["s1_inference_invocations"] == 2
    assert details["clarified"]["later_single_s1_invocation_instruction_satisfied"] is False


def test_the_module_and_script_exist_where_documented(paths: ProjectPaths) -> None:
    for relative in (MODULE, SCRIPT):
        assert (paths.root / Path(relative)).is_file()
