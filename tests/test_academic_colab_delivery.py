"""Metadata-only checks for cloud-evidence attribution and immutable execution bytes."""

import copy
import json
import shutil
from pathlib import Path

import pytest

from construction_safety_vision import academic_colab_delivery as colab

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def provenance():
    return json.loads((ROOT / colab.PROVENANCE).read_text(encoding="utf-8"))


@pytest.fixture
def portable_reports(tmp_path, monkeypatch, provenance):
    paths = [row["path"] for row in provenance["execution_surface"]]
    for relative in [*paths, colab.REPORT, colab.MARKDOWN, colab.PROVENANCE]:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    original = colab.git_bytes
    monkeypatch.setattr(
        colab,
        "git_bytes",
        lambda root, relative, revision=colab.VALIDATED: original(ROOT, relative, revision),
    )
    return tmp_path


def test_real_candidate_matches_human_evidence_and_validated_revision():
    assert colab.validate(ROOT) == []


def test_validation_needs_no_local_media_checkpoints_or_outputs(portable_reports):
    assert not (portable_reports / "outputs").exists()
    assert not (portable_reports / "artifacts").exists()
    assert not (portable_reports / "data").exists()
    assert colab.validate(portable_reports) == []


@pytest.mark.parametrize(
    "route,value",
    [
        (("classification",), "CLOUD_NOT_VERIFIED"),
        (("gap_004_status",), "OPEN"),
        (("protocol", "models_executed_during_finalization"), 1),
        (("cloud_validation_scope", "final_candidate_commit_executed_in_cloud"), True),
        (
            ("validation_evidence", "REAL_COLAB_MODE_B_VALIDATION", "publication_status"),
            "READY_FOR_PUBLICATION",
        ),
        (
            ("validation_evidence", "REAL_COLAB_MODE_A_VALIDATION", "execution_kind"),
            "AGENT_EXECUTED",
        ),
    ],
)
def test_status_labels_cannot_override_preserved_evidence(portable_reports, route, value):
    path = portable_reports / colab.REPORT
    report = json.loads(path.read_text(encoding="utf-8"))
    target = report
    for key in route[:-1]:
        target = target[key]
    target[route[-1]] = value
    path.write_text(json.dumps(report), encoding="utf-8")
    assert "Report claims disagree with preserved validation evidence" in colab.validate(
        portable_reports
    )


def test_warning_cannot_be_erased_even_if_attacker_rehashes_source(provenance):
    source = provenance["source_records"]["mode_b_execution"]
    source["record"]["warnings"].remove(colab.WARNING)
    source["record"]["source_attribution"]["license"] = "CC BY-SA 3.0"
    text = json.dumps(source["record"], indent=2, sort_keys=True) + "\n"
    source["original_utf8_text"] = text
    source["identity"] = colab.identity(text.encode())
    problems = colab.provenance_problems(provenance)
    assert "Original source bytes changed: mode_b_execution" in problems
    assert "Original source-license warning/unspecified attribution changed" in problems


def test_parsed_evidence_cannot_diverge_from_original_bytes(provenance):
    provenance["source_records"]["mode_b_execution"]["record"]["processed_frames"] = 1
    assert "Embedded parsed source differs from original: mode_b_execution" in (
        colab.provenance_problems(provenance)
    )


def test_original_text_round_trips_lf_and_crlf_without_normalization(provenance):
    records = provenance["source_records"]
    run = records["mode_b_execution"]["original_utf8_text"].encode()
    context = records["external_clip_preparation"]["original_utf8_text"].encode()
    assert b"\r\n" not in run
    assert b"\r\n" in context
    assert colab.identity(run) == records["mode_b_execution"]["identity"]
    assert colab.identity(context) == records["external_clip_preparation"]["identity"]


def test_local_verification_cannot_be_relabelled_as_cloud_execution(provenance):
    provenance["human_observation"]["agent_cloud_execution"] = True
    assert "Human cloud observation missing or misrepresented" in colab.provenance_problems(
        provenance
    )


def test_failed_local_quality_gate_cannot_support_pass(provenance):
    provenance["quality_gates"]["pytest"]["failed"] = 1
    assert "Local quality gates do not support PASS" in colab.provenance_problems(provenance)


@pytest.mark.parametrize(
    "relative",
    [
        colab.NOTEBOOK,
        "src/construction_safety_vision/delivery_demo.py",
        "src/construction_safety_vision/video_runtime.py",
        "configs/detector_segmenter_comparison.yaml",
    ],
)
def test_critical_byte_change_stops_finalization(tmp_path, provenance, relative):
    row = next(row for row in provenance["execution_surface"] if row["path"] == relative)
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    original = (ROOT / relative).read_bytes()
    path.write_bytes(original)
    assert colab.candidate_surface_problems(tmp_path, [row]) == []
    path.write_bytes(original + b"\n")
    assert colab.candidate_surface_problems(tmp_path, [row]) == [
        f"STOP: execution-critical bytes changed: {relative}"
    ]


def test_critical_crlf_conversion_is_also_rejected(tmp_path, provenance):
    relative = "src/construction_safety_vision/delivery_demo.py"
    row = next(row for row in provenance["execution_surface"] if row["path"] == relative)
    target = tmp_path / relative
    target.parent.mkdir(parents=True)
    target.write_bytes((ROOT / relative).read_bytes().replace(b"\n", b"\r\n"))
    assert colab.candidate_surface_problems(tmp_path, [row])


def test_omitted_surface_entry_cannot_weaken_the_manifest(portable_reports):
    path = portable_reports / colab.PROVENANCE
    record = json.loads(path.read_text(encoding="utf-8"))
    record["execution_surface"].pop()
    path.write_text(json.dumps(record), encoding="utf-8")
    assert "Execution-surface inventory/validated identities mismatch" in colab.validate(
        portable_reports
    )


@pytest.mark.parametrize(
    "flag", ["RUN_INFERENCE", "UPLOAD_RECORDED_VIDEO", "CONFIRM_EXTERNAL_VIDEO"]
)
def test_notebook_rejects_automatic_execution_or_upload(flag):
    raw = (ROOT / colab.NOTEBOOK).read_bytes()
    changed = raw.replace(f"{flag} = False".encode(), f"{flag} = True".encode())
    assert f"Unsafe notebook default: {flag}" in colab.notebook_problems(changed)


def test_notebook_rejects_temporary_branch_and_saved_binary():
    notebook = json.loads((ROOT / colab.NOTEBOOK).read_bytes())
    code = next(cell for cell in notebook["cells"] if cell["cell_type"] == "code")
    code["outputs"] = [{"output_type": "display_data", "data": {"image/png": "aGVsbG8="}}]
    raw = json.dumps(notebook).encode()
    assert "Embedded notebook output or attachment" in colab.notebook_problems(raw)
    notebook = json.loads((ROOT / colab.NOTEBOOK).read_bytes())
    for cell in notebook["cells"]:
        cell["source"] = [
            line.replace('"--branch", "main"', '"--branch", "phase14a-colab-validation"')
            for line in cell["source"]
        ]
    raw = json.dumps(notebook).encode()
    assert "Canonical notebook must clone main exactly once" in colab.notebook_problems(raw)


@pytest.mark.parametrize(
    "unsafe",
    [
        "https://drive." + "google.com/file/d/example",
        "https://colab.research.google.com/" + "drive/example",
        chr(92).join(("C:", "Users", "example", "secret.txt")),
        "data:" + "video/mp4;base64,aGVsbG8=",
    ],
)
def test_public_scan_rejects_private_locators_and_embedded_media(unsafe):
    assert colab.public_text_problems(unsafe)


def test_canonical_public_badge_is_not_a_private_drive_url():
    assert colab.public_text_problems(colab.COLAB_URL) == []
    assert "/blob/main/notebooks/" in colab.COLAB_URL
    assert "phase14a-colab-validation" not in colab.COLAB_URL


def test_holdout_opt_in_refuses_validation_before_any_file_read(monkeypatch, tmp_path):
    monkeypatch.setenv("CSVISION_ALLOW_TEST_SPLIT", "1")
    assert colab.validate(tmp_path) == ["Holdout authorization must be absent"]


def test_payload_construction_does_not_mutate_original_source(provenance):
    original = copy.deepcopy(provenance)
    report = colab.build_report(provenance)
    assert provenance == original
    assert report["source_license_distinction"]["original_cloud_attribution"]["license"] is None
    assert report["source_license_distinction"]["independent_source_context"]["license"] == (
        "CC BY-SA 3.0"
    )
