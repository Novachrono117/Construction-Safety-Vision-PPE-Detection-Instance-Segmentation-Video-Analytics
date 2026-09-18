"""The pitch package presents committed evidence; it never becomes its own source."""

import ast
import json
from pathlib import Path

import pytest

from construction_safety_vision.delivery_status import (
    PITCH_MANIFEST_PATH,
    PITCH_READY,
    STATUS_PATH,
    build_status,
)
from construction_safety_vision.pitch_delivery import (
    DOCUMENTS,
    SPOKEN_WORD_RANGE,
    load_manifest,
    narration,
    spoken_words,
    validate,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = load_manifest(ROOT)
SCRIPT = (ROOT / "delivery/pitch/full_script.md").read_text(encoding="utf-8")


def _field(payload, dotted):
    for key in dotted.split("."):
        payload = payload[key]
    return payload


def test_the_package_validates_against_its_evidence():
    assert validate(ROOT) == []


def test_every_declared_document_exists_and_is_linked_from_the_entry_point():
    entry = (ROOT / "delivery/pitch/README.md").read_text(encoding="utf-8")
    for relative in DOCUMENTS:
        assert (ROOT / relative).is_file(), relative
        if relative != "delivery/pitch/README.md":
            assert Path(relative).name in entry, relative
    assert Path(PITCH_MANIFEST_PATH).name in entry


@pytest.mark.parametrize("claim", MANIFEST["spoken_claims"], ids=lambda c: c["id"])
def test_each_spoken_number_matches_its_committed_artifact(claim):
    payload = json.loads((ROOT / claim["artifact"]).read_text(encoding="utf-8"))
    assert _field(payload, claim["field"]) == claim["value"]
    assert claim["spoken"] in SCRIPT


def test_the_script_length_keeps_the_pitch_inside_the_assignment_window():
    words = spoken_words(ROOT)
    assert words == MANIFEST["duration"]["spoken_words"]
    assert SPOKEN_WORD_RANGE[0] <= words <= SPOKEN_WORD_RANGE[1]
    # The assignment hard-limits 5-8 minutes; check the whole plausible rate band.
    for rate in (120, 160):
        assert 300 <= words / rate * 60 <= 480, rate


def test_the_narration_never_claims_a_winner_or_real_time():
    spoken = narration(SCRIPT)
    assert "não vou dizer que o s1 venceu" in spoken
    assert "três das cinco classes pioraram" in spoken
    for forbidden in ("a segmentação foi melhor", "o d2 perdeu", "estado da arte"):
        assert forbidden not in spoken, forbidden
    index = spoken.find("tempo real")
    assert index != -1, "the real-time refusal must be spoken"
    assert "nada aqui é tempo real" in spoken[max(0, index - 60) : index + 60]


def test_the_shown_video_segment_is_unedited_and_anchored_on_frozen_instants():
    segment = MANIFEST["video_segment"]
    anchors = json.loads((ROOT / segment["anchor_evidence"]).read_text(encoding="utf-8"))
    documented = {shot["timestamp_seconds"] for shot in anchors["screenshots"]}
    assert set(segment["anchor_instants_seconds"]) <= documented
    assert set(segment["alternative_anchor_instants_seconds"]) <= documented
    assert set(segment["rejected_segment_seconds"]) <= documented
    assert segment["predictions_edited"] is False
    assert segment["highlight_reel"] is False
    assert segment["shows_visible_failure"] is True
    assert segment["source_output_committed"] is False
    demo = json.loads((ROOT / "reports/final_real_video_demo.json").read_text(encoding="utf-8"))
    duration = demo["execution"]["output_video"]["duration_seconds"]
    for bound in ("selected_end_seconds", "alternative_end_seconds"):
        assert segment[bound] <= duration, bound


def test_the_pitch_is_ready_to_record_and_never_claims_to_be_delivered():
    assert MANIFEST["classification"] == PITCH_READY
    assert MANIFEST["recorded_video_exists"] is False
    assert MANIFEST["published_link_exists"] is False
    assert MANIFEST["publication"]["link"] is None
    assert MANIFEST["publication"]["uploaded_by_this_repository"] is False
    tracker = json.loads((ROOT / STATUS_PATH).read_text(encoding="utf-8"))
    assert tracker == build_status(ROOT)
    assert tracker["delivery_state"]["pitch"] == PITCH_READY
    gap = next(item for item in tracker["gaps"] if item["gap_id"] == MANIFEST["gap_id"])
    assert gap["current_status"] == "READY_TO_RECORD"
    assert gap["current_status"] != "RESOLVED"
    assert tracker["phase_15b"]["models_executed"] == 0
    assert tracker["phase_15b"]["holdout_accessed"] is False


def test_presenters_come_from_documented_authorship_not_from_commit_history():
    names = [person["name"] for person in MANIFEST["presenters"]]
    assert names, "at least one presenter must be documented"
    report = (ROOT / "academic/final_report.md").read_text(encoding="utf-8")
    for person in MANIFEST["presenters"]:
        assert (ROOT / person["evidence"]).is_file()
        assert person["name"] in report


def test_the_pitch_tooling_runs_no_model_and_reads_no_dataset():
    files = (
        ROOT / "src/construction_safety_vision/pitch_delivery.py",
        ROOT / "scripts/prepare_video_pitch.py",
    )
    forbidden = {"torch", "ultralytics", "cv2", "numpy", "pycocotools"}
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                names = {(node.module or "").split(".")[0]}
            else:
                continue
            assert not names & forbidden, (path.name, names & forbidden)
        assert "os.environ[" not in path.read_text(encoding="utf-8")


def test_no_holdout_identifier_reaches_the_pitch_package():
    import csv

    with (ROOT / "reports/final_split_assignments.csv").open(encoding="utf-8", newline="") as fh:
        test_ids = {row["source_image_id"] for row in csv.DictReader(fh) if row["split"] == "test"}
    assert test_ids, "the fixture that makes this test meaningful is missing"
    for relative in (*DOCUMENTS, PITCH_MANIFEST_PATH):
        body = (ROOT / relative).read_text(encoding="utf-8")
        assert sorted(image_id for image_id in test_ids if image_id in body) == [], relative
