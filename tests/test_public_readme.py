"""The public README is a presentation of committed evidence, never its own source."""

import csv
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
NOTEBOOK = "notebooks/construction_safety_vision_demo.ipynb"
COLAB_PREFIX = (
    "https://colab.research.google.com/github/Novachrono117/"
    "Construction-Safety-Vision-PPE-Detection-Instance-Segmentation-Video-Analytics/blob/"
)


def _artifact(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def _field(payload: dict, dotted: str):
    for key in dotted.split("."):
        payload = payload[key]
    return payload


def test_open_in_colab_targets_main_and_the_canonical_notebook():
    targets = re.findall(r"https://colab\.research\.google\.com/[^\s)\"]+", README)
    notebook_targets = [url for url in targets if url.endswith(".ipynb")]
    assert notebook_targets, "the README must offer an Open in Colab link"
    for url in notebook_targets:
        assert url == f"{COLAB_PREFIX}main/{NOTEBOOK}"
    assert (ROOT / NOTEBOOK).is_file()


def test_no_temporary_validation_branch_or_private_location_is_published():
    for branch in ("phase14a-colab-validation", "phase14c-colab-validation"):
        assert branch not in README
    # A Colab or raw link may only ever resolve against the canonical branch.
    hosts = "colab.research.google.com|raw.githubusercontent.com"
    for url in re.findall(rf"https://(?:{hosts})/[^\s)\"]+", README):
        assert "/blob/main/" in url or url.endswith(".svg"), url
    assert "localhost" not in README
    assert not re.search(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]", README)
    assert not re.search(r"(?:^|[\s\"'])(?:/home/|/Users/|/root/|\\\\)", README, re.MULTILINE)
    assert "ROBOFLOW_API_KEY" in README and "ROBOFLOW_API_KEY=" not in README


@pytest.mark.parametrize(
    ("relative", "dotted"),
    [
        ("reports/final_test_detector.json", "canonical_box.CANONICAL_TEST_BOX_MAP50_95"),
        ("reports/final_test_detector.json", "canonical_box.CANONICAL_TEST_BOX_MAP50"),
        ("reports/final_test_detector.json", "canonical_precision_recall.precision"),
        ("reports/final_test_detector.json", "canonical_precision_recall.recall"),
        ("reports/final_test_segmenter.json", "canonical_box.S1_CANONICAL_TEST_BOX_MAP50_95"),
        ("reports/final_test_segmenter.json", "canonical_box.S1_CANONICAL_TEST_BOX_MAP50"),
        ("reports/final_test_segmenter.json", "canonical_mask.CANONICAL_TEST_MASK_MAP50_95"),
        ("reports/final_test_segmenter.json", "canonical_mask.CANONICAL_TEST_MASK_MAP50"),
        ("reports/final_test_segmenter.json", "canonical_mask_precision_recall.precision"),
        ("reports/final_test_segmenter.json", "canonical_mask_precision_recall.recall"),
        ("reports/final_test_direct_iou.json", "diagnostic.global.matched_mask_iou_mean"),
        ("reports/final_test_direct_iou.json", "diagnostic.global.gt_normalized_mask_iou"),
        (
            "reports/detector_segmenter_latency_comparison.json",
            "latency.D2.END_TO_END_MODEL_OUTPUT_LATENCY_MS.mean",
        ),
        (
            "reports/detector_segmenter_latency_comparison.json",
            "latency.S1.END_TO_END_MODEL_OUTPUT_LATENCY_MS.mean",
        ),
    ],
)
def test_every_headline_number_is_quoted_from_its_committed_artifact(relative, dotted):
    value = _field(_artifact(relative), dotted)
    assert f"{value:f}".rstrip("0").rstrip(".") in README or f"{value}" in README, dotted


def test_population_and_split_facts_come_from_the_frozen_manifest():
    manifest = _artifact("reports/split_manifest.json")
    for split in ("train", "validation", "test"):
        images = manifest["actual_image_counts"][split]
        annotations = manifest["actual_annotation_counts"][split]
        assert f"| {images} | {annotations} |" in README, split
    detector = _artifact("reports/final_test_detector.json")
    assert str(detector["population"]["images"]) in README
    assert str(detector["population"]["annotations"]) in README


def test_object_level_outcome_counts_match_both_holdout_artifacts():
    for relative in ("reports/final_test_detector.json", "reports/final_test_segmenter.json"):
        level = _artifact(relative)["object_level"]
        row = "| {} | {} | {} |".format(
            level["true_positives"], level["false_positives"], level["false_negatives"]
        )
        assert row in README, relative


def test_no_unsupported_capability_or_ranking_is_claimed():
    lowered = README.lower()
    for phrase in (
        "state-of-the-art",
        "state of the art",
        "beats d2",
        "beats s1",
        "outperform",
        "best model",
        "superior model",
        "universally superior",
    ):
        assert phrase not in lowered, phrase
    # These words may appear only inside an explicit refusal of the claim. The
    # refusal is looked for in a window, because prose wraps across lines.
    flat = " ".join(lowered.split())
    refusals = (
        "not claimed",
        "no real-time",
        "unsupported",
        "no winner",
        "no robust",
        "refuse",
        "never",
    )
    for word in ("real-time", "production-ready", "production ready", "winner"):
        start = 0
        while (index := flat.find(word, start)) != -1:
            window = flat[max(0, index - 160) : index + 160]
            assert any(mark in window for mark in refusals), (word, window)
            start = index + len(word)


def test_the_required_academic_deliverables_are_linked_and_present():
    for relative in (
        "academic/final_report.md",
        "academic/final_report.pdf",
        NOTEBOOK,
        "reports/final_test_evaluation.md",
        "reports/qualitative_validation_gallery.md",
        "reports/final_real_video_demo.md",
        "delivery/REPRODUCTION.md",
        "delivery/AI_USAGE.md",
        "delivery/LICENSING.md",
        "LICENSE",
    ):
        assert (ROOT / relative).exists(), relative
        assert relative in README, relative


def test_the_holdout_is_described_as_spent_and_never_as_untouched():
    assert "**OBSERVED**" in README and "**CLOSED**" in README
    lowered = README.lower()
    for denial in ("never been read", "never evaluated", "has not been evaluated"):
        assert denial not in lowered, denial
    # The established leak-check route: read the frozen assignment only to prove
    # that none of its holdout identifiers is published. Nothing is evaluated.
    with (ROOT / "reports/final_split_assignments.csv").open(encoding="utf-8", newline="") as fh:
        test_ids = {row["source_image_id"] for row in csv.DictReader(fh) if row["split"] == "test"}
    assert test_ids, "the fixture that makes this test meaningful is missing"
    assert sorted(image_id for image_id in test_ids if image_id in README) == []
