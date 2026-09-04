"""Tests that the committed D0 result describes the run that actually happened.

These read the committed manifest and, where the ignored run directory is still
present, check the report's checkpoint hashes against the files on disk. They
assert structure and internal consistency rather than particular metric values:
hard-coding an expected mAP would turn a test into a claim about how well the
model *should* do, which is exactly the kind of assertion this project refuses to
make.
"""

from __future__ import annotations

import json

import pytest

from construction_safety_vision.data.canonical import scan_for_sensitive
from construction_safety_vision.data.materialization import sha256_bytes
from construction_safety_vision.detection_results import (
    COMPLETE,
    HIGH_SAMPLING_UNCERTAINTY,
    NOT_EXPOSED,
    PER_CLASS_METRICS,
    PRIMARY_METRIC,
    SECONDARY_METRICS,
    TEST_PROTECTED,
    experiment_fingerprint,
    validate_result_manifest,
)
from construction_safety_vision.experiment import load_detection_baseline_config
from construction_safety_vision.paths import ProjectPaths

CLASS_NAMES = ("helmet_loose", "helmet_on_head", "person", "vest_loose", "vest_on_body")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


@pytest.fixture(scope="module")
def paths() -> ProjectPaths:
    return ProjectPaths.from_root()


@pytest.fixture(scope="module")
def manifest(paths: ProjectPaths) -> dict:
    path = paths.reports / "detection_D0_manifest.json"
    if not path.is_file():
        pytest.skip("detection_D0_manifest.json not present; run train_detection_baseline.py")
    return json.loads(path.read_text(encoding="utf-8"))


# --- manifest -----------------------------------------------------------------


def test_manifest_is_valid(manifest):
    assert validate_result_manifest(manifest, class_names=CLASS_NAMES) == []


def test_manifest_declares_the_predeclared_experiment(manifest):
    assert manifest["experiment_id"] == "D0"
    assert manifest["task"] == "detection"
    assert manifest["model"] == "YOLO11n"
    assert manifest["status"] == COMPLETE
    assert manifest["weights_committed"] is False


def test_manifest_matches_the_committed_protocol(paths, manifest):
    config = load_detection_baseline_config(paths.configs / "detection_baseline.yaml")
    assert manifest["baseline_config_sha256"] == config.fingerprint()
    assert manifest["epochs_configured"] == config.training["epochs"]
    assert manifest["declared_optimizer_policy"] == config.training["optimizer"]
    assert manifest["checkpoint_selection"] == config.checkpoint_selection
    assert manifest["metric_hierarchy"] == config.metrics.as_dict()


def test_manifest_binds_to_the_frozen_data(paths, manifest):
    adapter = json.loads(
        (paths.reports / "detection_adapter_manifest.json").read_text(encoding="utf-8")
    )
    split = json.loads((paths.reports / "split_manifest.json").read_text(encoding="utf-8"))
    assert (
        manifest["split_reference"]["split_assignment_sha256"] == (split["split_assignment_sha256"])
    )
    assert manifest["dataset_fingerprints"]["class_map_sha256"] == adapter["class_map_sha256"]
    assert manifest["train_images"] == adapter["train"]["image_count"]
    assert manifest["train_annotations"] == adapter["train"]["annotation_count"]
    assert manifest["validation_images"] == adapter["validation"]["image_count"]
    assert manifest["validation_annotations"] == adapter["validation"]["annotation_count"]


def test_training_used_only_the_development_splits(manifest):
    assert manifest["train_images"] == 303
    assert manifest["validation_images"] == 65
    assert manifest["test"]["status"] == TEST_PROTECTED


def test_optimizer_auto_resolved_to_something_concrete(manifest):
    # The protocol declares a policy; the record must say what it became.
    assert manifest["declared_optimizer_policy"] == "auto"
    assert manifest["resolved_optimizer"] not in ("", "auto", NOT_EXPOSED, None)


def test_resolved_arguments_carry_no_machine_paths(manifest):
    resolved = manifest["resolved_training_arguments"]
    for key in ("save_dir", "project", "name", "data", "model"):
        assert key not in resolved
    assert scan_for_sensitive(json.dumps(resolved)) == []


def test_epochs_completed_are_consistent(manifest):
    completed = manifest["epochs_completed"]
    configured = manifest["epochs_configured"]
    assert 0 < completed <= configured
    assert manifest["early_stopped"] == (completed < configured)
    assert 1 <= manifest["best_epoch"] <= completed


def test_exactly_one_training_run_was_recorded(paths):
    provenance = json.loads(
        (paths.reports / "detection_D0.provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["details"]["runs_performed"] == 1
    assert provenance["details"]["holdout_accessed"] is False
    assert provenance["details"]["weights_committed"] is False


def test_the_protocol_was_recorded_before_the_result(paths, manifest):
    prerun = json.loads(
        (paths.reports / "detection_D0_prerun.provenance.json").read_text(encoding="utf-8")
    )
    assert prerun["details"]["stage"] == "PRE_RUN"
    assert prerun["config"]["baseline_config_sha256"] == manifest["baseline_config_sha256"]
    assert (
        prerun["details"]["pretrained_weights_sha256"] == (manifest["pretrained_weights"]["sha256"])
    )
    assert (
        prerun["created_at"]
        <= json.loads((paths.reports / "detection_D0.provenance.json").read_text(encoding="utf-8"))[
            "created_at"
        ]
    )


# --- metrics -------------------------------------------------------------------


def test_metric_hierarchy_is_the_predeclared_one(manifest):
    metrics = manifest["validation_metrics"]
    assert set(metrics) == {PRIMARY_METRIC, *SECONDARY_METRICS}
    assert manifest["metric_hierarchy"]["primary"] == PRIMARY_METRIC


def test_reported_metrics_are_in_range(manifest):
    for name, value in manifest["validation_metrics"].items():
        if value == NOT_EXPOSED:
            continue
        assert 0.0 <= value <= 1.0, name


def test_per_class_names_are_exactly_the_frozen_class_map(paths, manifest):
    population = json.loads(
        (paths.reports / "canonical_modeling_manifest.json").read_text(encoding="utf-8")
    )
    assert set(manifest["per_class_metrics"]) == set(population["class_map"])
    assert set(manifest["per_class_metrics"]) == set(CLASS_NAMES)
    for entry in manifest["per_class_metrics"].values():
        assert set(entry) == set(PER_CLASS_METRICS)


def test_rare_class_warning_is_present_and_correct(manifest):
    rare = manifest["rare_class"]
    assert rare["name"] == "vest_loose"
    assert rare["warning"] == HIGH_SAMPLING_UNCERTAINTY
    assert rare["validation_images"] == 1
    assert rare["validation_instances"] == 8


def test_report_carries_the_rare_class_warning(paths):
    text = (paths.reports / "detection_D0_report.md").read_text(encoding="utf-8")
    assert HIGH_SAMPLING_UNCERTAINTY in text
    assert "vest_loose" in text


# --- fingerprint ----------------------------------------------------------------


def test_experiment_fingerprint_reproduces_from_the_manifest(manifest):
    recomputed = experiment_fingerprint(
        baseline_config_sha256=manifest["baseline_config_sha256"],
        pretrained_weights_sha256=manifest["pretrained_weights"]["sha256"],
        adapter_manifest_sha256=manifest["adapter_manifest_sha256"],
        split_assignment_sha256=manifest["split_reference"]["split_assignment_sha256"],
        resolved_arguments=manifest["resolved_training_arguments"],
        best_checkpoint_sha256=manifest["best_checkpoint"]["sha256"],
    )
    assert recomputed == manifest["d0_experiment_sha256"]


def test_checkpoint_hashes_match_the_files_on_disk(paths, manifest):
    for key in ("best_checkpoint", "last_checkpoint"):
        entry = manifest[key]
        path = paths.root / entry["relative_path"]
        if not path.is_file():
            pytest.skip("run artifacts not present; they are git-ignored")
        assert sha256_bytes(path) == entry["sha256"], key
        assert path.stat().st_size == entry["size_bytes"], key


# --- holdout and committed content -------------------------------------------------


def test_no_holdout_identifier_appears_in_any_result_artifact(paths, manifest):
    split = json.loads((paths.reports / "split_manifest.json").read_text(encoding="utf-8"))
    holdout = {
        image_id for group in split["test"]["groups"] for image_id in group["source_image_ids"]
    }
    for name in (
        "detection_D0_manifest.json",
        "detection_D0_report.md",
        "detection_D0.provenance.json",
        "detection_D0_prerun.provenance.json",
    ):
        text = (paths.reports / name).read_text(encoding="utf-8")
        assert not [image_id for image_id in holdout if image_id in text], name


def test_result_artifacts_carry_no_sensitive_content(paths):
    for name in (
        "detection_D0_manifest.json",
        "detection_D0_report.md",
        "detection_D0.provenance.json",
        "detection_D0_prerun.provenance.json",
    ):
        assert scan_for_sensitive((paths.reports / name).read_text(encoding="utf-8")) == [], name


def test_no_checkpoint_was_committed(paths):
    assert not list((paths.root / "reports").rglob("*.pt"))


def test_committed_figures_are_metric_plots_only(paths, manifest):
    directory = paths.reports / "figures" / "detection" / "D0"
    if not directory.is_dir():
        pytest.skip("D0 figures not present")
    for path in directory.iterdir():
        assert path.suffix.lower() not in IMAGE_SUFFIXES, (
            f"{path.name} is a photographic format; dataset imagery must not be committed"
        )
        assert not path.name.startswith(("train_batch", "val_batch", "labels"))
    assert manifest["committed_figures"]


def test_no_dataset_or_prediction_image_reached_the_reports_tree(paths):
    directory = paths.reports / "figures" / "detection"
    if not directory.is_dir():
        pytest.skip("D0 figures not present")
    offenders = [
        path.name
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    assert offenders == []
