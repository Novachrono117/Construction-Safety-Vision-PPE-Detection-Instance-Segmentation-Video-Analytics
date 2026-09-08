"""Run one frozen Phase 7 controlled detection experiment.

Phase 7B onward. Takes an experiment id declared in
``configs/detection_experiments.yaml`` and runs exactly that experiment - it
does not accept hyperparameters, because a Phase 7 candidate has none of its
own. The protocol is resolved through the committed inheritance mechanism:
D0's protocol is read from ``configs/detection_baseline.yaml`` and the
candidate's declared override set is applied over it. Nothing here reconstructs
a parameter by hand, so the run cannot drift from the frozen comparison.

The order is deliberate and is the point of the script. Before a single weight
is downloaded, let alone trained:

1. the holdout guard is checked and the run refuses to proceed if it is unlocked;
2. the committed Phase 7 policy is verified against the configuration it names;
3. the D0 reference is validated from its committed manifest, never re-run;
4. the candidate's protocol is resolved and the one-variable contract is
   **proven** - if the candidate differs from D0 anywhere it did not declare,
   the script exits and trains nothing.

Only then are the pretrained weights fetched and fingerprinted, a pre-run
provenance record written, and one training run started.

What it will not do: train more than one run, resume with changed settings,
retry after an OOM or a NaN, reduce batch to make a run fit, sweep a threshold,
evaluate ``last.pt`` against ``best.pt``, or touch the holdout. Each of those
either creates a selection degree of freedom or breaks the comparison, so each
is a stop rather than a fallback.

Requires:

* ``configs/detection_experiments.yaml``            phase 7A
* ``configs/detection_baseline.yaml``               phase 6A
* ``reports/detection_comparison_policy.json``      phase 7A
* ``reports/detection_comparison_reference.json``   phase 7A
* ``reports/detection_D0_manifest.json``            phase 6B
* ``reports/split_manifest.json``                   phase 5C.2
* ``reports/detection_adapter_manifest.json``       phase 6A

Writes (run directory and weights git-ignored, evidence committed):
    artifacts/detection/<ID>/                       training run
    artifacts/detection/<ID>_val/                   authoritative validation
    reports/detection_<ID>_manifest.json
    reports/detection_<ID>_report.md
    reports/detection_<ID>_prerun.provenance.json
    reports/detection_<ID>.provenance.json
    reports/detection_experiment_results.json
    reports/figures/detection/<ID>/                 metric-only plots

Usage:
    uv run python scripts/train_detection_experiment.py --experiment D1 --verify-only
    uv run python scripts/train_detection_experiment.py --experiment D1 --memory-preflight
    uv run python scripts/train_detection_experiment.py --experiment D1
    uv run python scripts/train_detection_experiment.py --experiment D1 --resume
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from construction_safety_vision.config import ConfigError
from construction_safety_vision.data.canonical import scan_for_sensitive
from construction_safety_vision.data.materialization import sha256_bytes
from construction_safety_vision.detection_comparison import (
    CANDIDATE_ROLE,
    DESCRIPTIVE_HIGH_UNCERTAINTY,
    OFFICIAL_ALL_CLASS_METRIC,
    PRIMARY_SELECTION_METRIC,
    ComparisonConfigError,
    ComparisonError,
    ExperimentDeclaration,
    ExperimentMatrix,
    build_experiment_record,
    check_protocol_compatibility,
    class_support,
    classify_delta,
    descriptive_class_names,
    load_experiment_matrix,
    resolve_candidate_protocol,
    supported_class_names,
    supported_macro,
    unflatten_protocol,
)
from construction_safety_vision.detection_results import (
    EXPERIMENT_COMPLETE as EXPERIMENT_STATUS_COMPLETE,
)
from construction_safety_vision.detection_results import (
    HIGH_SAMPLING_UNCERTAINTY,
    NOT_EXPOSED,
    PRIMARY_METRIC,
    TEST_PROTECTED,
    ResultError,
    critical_arguments,
    digest,
    global_metrics,
    per_class_metrics,
    validate_result_manifest,
)
from construction_safety_vision.detection_run import (
    RUN_ROOT,
    TRAIN_CONSOLE_LOG,
    RunError,
    best_epoch_from_history,
    capture_framework_log,
    checkpoint_record,
    copy_metric_figures,
    cross_check_metrics,
    ensure_pretrained_weights,
    model_complexity,
    optimizer_evidence,
    read_epoch_history,
    release_framework_log,
    resolved_training_arguments,
    runtime_facts,
)
from construction_safety_vision.experiment import load_detection_baseline_config
from construction_safety_vision.paths import ProjectPaths
from construction_safety_vision.provenance import ProvenanceRecord, git_commit
from construction_safety_vision.splits import HOLDOUT_UNLOCK_ENV_VAR, holdout_unlocked

MATRIX_YAML = "detection_experiments.yaml"
BASELINE_YAML = "detection_baseline.yaml"
POLICY_JSON = "detection_comparison_policy.json"
REFERENCE_JSON = "detection_comparison_reference.json"
SPLIT_MANIFEST_JSON = "split_manifest.json"
ADAPTER_MANIFEST_JSON = "detection_adapter_manifest.json"
POPULATION_MANIFEST_JSON = "canonical_modeling_manifest.json"
TASK_MANIFEST_JSON = "task_dataset_manifest.json"
RESULTS_JSON = "detection_experiment_results.json"

DATASET_DESCRIPTOR = "data/processed/adapters/yolo_detection/dataset.yaml"

MEMORY_PREFLIGHT_NAME = "_memory_preflight"

EXPERIMENT_COMPLETE = "D1_CAPACITY_EXPERIMENT_COMPLETE"
MEMORY_CONSTRAINT = "MEMORY_CONSTRAINT_REVIEW_REQUIRED"
TRAINING_FAILED = "TRAINING_FAILED"
PROTOCOL_MISMATCH = "COMPARISON_PROTOCOL_MISMATCH"
PROTOCOL_VIOLATION = "PROTOCOL_VIOLATION"
BLOCKED = "BLOCKED"

IMPROVES = "IMPROVES_D0_BEYOND_MARGIN"
EQUIVALENT = "PRACTICALLY_EQUIVALENT_TO_D0"
BELOW = "BELOW_D0"

HOLDOUT_REASON = (
    "the holdout was not adapted, loaded, evaluated or measured in this experiment. It has "
    "no labels, no dataset entry and no result."
)


class ExperimentRunError(RuntimeError):
    """Raised when the experiment cannot run as specified."""


def read_json(path: Path) -> dict[str, Any]:
    """Read a committed JSON artifact.

    Args:
        path: File to read.

    Returns:
        The parsed mapping.

    Raises:
        ExperimentRunError: If it is missing or is not a JSON object.
    """
    if not path.is_file():
        msg = f"required input not found: {path.name}"
        raise ExperimentRunError(msg)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"{path.name} is not valid JSON ({exc})"
        raise ExperimentRunError(msg) from exc
    if not isinstance(data, dict):
        msg = f"{path.name} must contain a JSON object"
        raise ExperimentRunError(msg)
    return data


def write_json(path: Path, payload: Any) -> str:
    """Write a JSON artifact deterministically and return its digest.

    Args:
        path: Destination file.
        payload: JSON-serialisable content.

    Returns:
        The SHA-256 digest of the payload.

    Raises:
        ExperimentRunError: If the serialised content is unfit to commit.
    """
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    unsafe = scan_for_sensitive(text)
    if unsafe:
        msg = f"{path.name} is not fit to commit: {'; '.join(unsafe)}"
        raise ExperimentRunError(msg)
    path.write_text(f"{text}\n", encoding="utf-8", newline="\n")
    return digest(payload)


# --- verification -------------------------------------------------------------


def verify_policy(paths: ProjectPaths, matrix: ExperimentMatrix) -> dict[str, Any]:
    """Verify the committed Phase 7 policy against the configuration it names.

    Args:
        paths: Project layout.
        matrix: The parsed experiment matrix.

    Returns:
        The parsed policy and reference artifacts plus the matrix digest.

    Raises:
        ExperimentRunError: If a policy artifact disagrees with the frozen
            configuration. A candidate judged under a policy that has drifted
            from the one that was frozen is not a controlled comparison.
    """
    policy = read_json(paths.reports / POLICY_JSON)
    reference = read_json(paths.reports / REFERENCE_JSON)
    matrix_sha = sha256_bytes(paths.configs / MATRIX_YAML)

    problems: list[str] = []
    if policy.get("experiment_matrix_sha256") != matrix_sha:
        problems.append(
            f"{POLICY_JSON} was generated from a different {MATRIX_YAML} "
            f"(policy records {policy.get('experiment_matrix_sha256')}, file digests {matrix_sha})"
        )
    if reference.get("experiment_matrix_sha256") != matrix.fingerprint():
        problems.append(f"{REFERENCE_JSON} was generated from a different matrix content")
    if policy.get("classification") != "DETECTION_EXPERIMENT_PROTOCOL_FROZEN":
        problems.append(f"{POLICY_JSON} is not classified as frozen")
    if policy.get("reference_experiment") != matrix.reference_experiment:
        problems.append("the policy and the matrix disagree about the reference experiment")
    if policy["metrics"]["primary_selection"] != PRIMARY_SELECTION_METRIC:
        problems.append("the policy declares a different primary selection metric")
    rule = policy["support_rule"]
    if (
        rule["min_validation_positive_images"] != matrix.support_rule.min_positive_images
        or rule["min_validation_instances"] != matrix.support_rule.min_instances
    ):
        problems.append("the policy and the matrix disagree about the support thresholds")
    if policy["practical_equivalence_margin"]["value"] != str(matrix.practical_equivalence_margin):
        problems.append("the policy and the matrix disagree about the equivalence margin")
    if problems:
        raise ExperimentRunError("; ".join(problems))
    return {"policy": policy, "reference": reference, "matrix_sha256": matrix_sha}


def verify_reference(
    paths: ProjectPaths, matrix: ExperimentMatrix, reference_artifact: dict[str, Any]
) -> dict[str, Any]:
    """Validate the D0 reference from its committed manifest.

    D0 is read, never re-run. Its manifest must still validate, still describe
    the protocol the candidate is about to inherit, and still reproduce the
    selection metric recorded in the phase 7A reference artifact.

    Args:
        paths: Project layout.
        matrix: The parsed experiment matrix.
        reference_artifact: The parsed phase 7A reference artifact.

    Returns:
        The reference facts the comparison needs.

    Raises:
        ExperimentRunError: If the reference cannot be trusted.
    """
    manifest = read_json(paths.reports / f"detection_{matrix.reference_experiment}_manifest.json")
    baseline = load_detection_baseline_config(paths.configs / BASELINE_YAML)
    class_map = manifest["class_map"]
    names = tuple(sorted(class_map, key=lambda name: (class_map[name], name)))

    problems = validate_result_manifest(manifest, class_names=names)
    if manifest.get("baseline_config_sha256") != baseline.fingerprint():
        problems.append(
            "the committed D0 result was produced under a different protocol than "
            f"{BASELINE_YAML} now holds; the candidate would inherit a protocol D0 never used"
        )
    if problems:
        raise ExperimentRunError("; ".join(problems))

    split = read_json(paths.reports / SPLIT_MANIFEST_JSON)
    support = class_support(split, class_map, rule=matrix.support_rule)
    record = build_experiment_record(manifest, support)

    recorded = reference_artifact["metrics"][PRIMARY_SELECTION_METRIC]
    if round(float(record.supported_macro), 6) != recorded:
        msg = (
            f"the D0 reference recomputes to {round(float(record.supported_macro), 6)} but the "
            f"committed reference artifact records {recorded}"
        )
        raise ExperimentRunError(msg)

    return {
        "manifest": manifest,
        "baseline": baseline,
        "class_map": class_map,
        "class_names": names,
        "support": support,
        "record": record,
        "split": split,
    }


def verify_frozen_inputs(paths: ProjectPaths, split: dict[str, Any]) -> dict[str, Any]:
    """Re-check every frozen data input the experiment depends on.

    Counts are read from the frozen manifests rather than hardcoded, so this
    check verifies agreement between the artifacts instead of agreement with a
    number typed into a script.

    Args:
        paths: Project layout.
        split: The parsed frozen split manifest.

    Returns:
        The verified fingerprints and dataset facts.

    Raises:
        ExperimentRunError: If any fingerprint or count no longer matches.
    """
    problems: list[str] = []
    population = read_json(paths.reports / POPULATION_MANIFEST_JSON)
    task = read_json(paths.reports / TASK_MANIFEST_JSON)
    adapter = read_json(paths.reports / ADAPTER_MANIFEST_JSON)

    class_map_sha = population["fingerprints"]["class_map_sha256"]
    for label, value in (
        ("task manifest class_map_sha256", task["class_map_sha256"]),
        ("adapter manifest class_map_sha256", adapter["class_map_sha256"]),
    ):
        if value != class_map_sha:
            problems.append(f"{label} disagrees with the frozen population")
    if adapter["split_assignment_sha256"] != split["split_assignment_sha256"]:
        problems.append("the adapter was built over a different frozen split")
    if (
        adapter["modeling_population_sha256"]
        != (population["fingerprints"]["modeling_population_sha256"])
    ):
        problems.append("the adapter was built over a different modelling population")
    if task["class_map"] != population["class_map"]:
        problems.append("the task manifest class map disagrees with the frozen one")

    for name in ("train", "validation"):
        expected_images = split["actual_image_counts"][name]
        expected_annotations = split["actual_annotation_counts"][name]
        expected_negatives = split["actual_negative_image_counts"][name]
        found = adapter[name]
        for field, expected in (
            ("image_count", expected_images),
            ("annotation_count", expected_annotations),
            ("negative_images", expected_negatives),
        ):
            if found[field] != expected:
                problems.append(
                    f"adapter {name}.{field} is {found[field]}, the frozen split records {expected}"
                )
    if adapter["test"]["status"] != "NOT_MATERIALIZED_PROTECTED_HOLDOUT":
        problems.append("the adapter does not record the holdout as unmaterialised")

    dataset = paths.root / DATASET_DESCRIPTOR
    if not dataset.is_file():
        problems.append(f"the adapter dataset descriptor is missing: {DATASET_DESCRIPTOR}")
    else:
        # Checked on the parsed keys, not on the raw text: the file's comments
        # explain that it deliberately has no holdout key, and a substring match
        # would read that explanation as the violation it rules out.
        descriptor = yaml.safe_load(dataset.read_text(encoding="utf-8")) or {}
        keys = {str(key).strip().lower() for key in descriptor}
        if "test" in keys:
            problems.append("the adapter dataset descriptor declares a holdout split key")
        if not {"train", "val"} <= keys:
            problems.append("the adapter dataset descriptor does not declare train and val")
        if descriptor.get("names") != {
            index: name
            for name, index in sorted(population["class_map"].items(), key=lambda kv: kv[1])
        }:
            problems.append("the adapter dataset descriptor class names are not the frozen map")

    if problems:
        raise ExperimentRunError("; ".join(problems))

    return {
        "class_map": population["class_map"],
        "class_map_sha256": class_map_sha,
        "modeling_population_sha256": population["fingerprints"]["modeling_population_sha256"],
        "split_assignment_sha256": split["split_assignment_sha256"],
        "task_materialization_config_sha256": task["materialization_config_sha256"],
        "adapter_config_sha256": adapter["adapter_config_sha256"],
        "adapter_manifest_sha256": sha256_bytes(paths.reports / ADAPTER_MANIFEST_JSON),
        "yolo_label_sha256": {
            name: adapter[name]["yolo_label_sha256"] for name in ("train", "validation")
        },
        "train_images": adapter["train"]["image_count"],
        "train_annotations": adapter["train"]["annotation_count"],
        "validation_images": adapter["validation"]["image_count"],
        "validation_annotations": adapter["validation"]["annotation_count"],
        "dataset_descriptor": DATASET_DESCRIPTOR,
    }


def resolve_candidate(
    matrix: ExperimentMatrix, baseline: Any, declaration: ExperimentDeclaration
) -> dict[str, Any]:
    """Resolve a candidate's protocol and prove its one-variable contract.

    Args:
        matrix: The parsed experiment matrix.
        baseline: The reference protocol.
        declaration: The candidate declaration.

    Returns:
        The flat protocol, the nested protocol and the compatibility verdict.

    Raises:
        ExperimentRunError: If the candidate differs from the reference anywhere
            it did not declare, or fails to apply a variable it did declare.
    """
    reference_protocol = baseline.as_dict()
    flat = resolve_candidate_protocol(reference_protocol, declaration)
    verdict = check_protocol_compatibility(
        reference_protocol,
        flat,
        declaration,
        reference_experiment=matrix.reference_experiment,
    )
    if not verdict.compatible:
        msg = (
            f"{declaration.experiment_id} is not a one-variable comparison against "
            f"{matrix.reference_experiment}: undeclared differences "
            f"{list(verdict.undeclared_differences)}, unapplied declarations "
            f"{list(verdict.unapplied_declarations)}. Refusing to train an invalid comparison."
        )
        raise ExperimentRunError(msg)
    nested = unflatten_protocol(flat)
    return {"flat": flat, "nested": nested, "verdict": verdict}


# --- execution ----------------------------------------------------------------


def write_prerun_record(
    paths: ProjectPaths,
    declaration: ExperimentDeclaration,
    *,
    protocol: dict[str, Any],
    flat: dict[str, Any],
    inputs: dict[str, Any],
    weights: dict[str, Any],
    runtime: dict[str, Any],
    policy_sha256: str,
    matrix_sha256: str,
) -> Path:
    """Record what is about to run, before the first optimisation step.

    Evidential rather than decorative: a protocol that only ever appears next to
    a result cannot be shown to have preceded it.

    Args:
        paths: Project layout.
        declaration: The candidate declaration.
        protocol: The resolved nested protocol.
        flat: The resolved flat protocol.
        inputs: The verified frozen input fingerprints.
        weights: The pretrained weight provenance.
        runtime: The runtime facts.
        policy_sha256: Digest of the committed phase 7 policy.
        matrix_sha256: Digest of the experiment matrix configuration.

    Returns:
        The record path.

    Raises:
        ExperimentRunError: If the record is unfit to commit.
    """
    experiment_id = declaration.experiment_id
    training = protocol["training"]
    record = ProvenanceRecord.create(
        "detection_experiment_prerun",
        phase=7,
        repo_root=paths.root,
        config={
            "detection_experiments": f"configs/{MATRIX_YAML}",
            "detection_baseline": f"configs/{BASELINE_YAML}",
            "experiment_id": experiment_id,
            "experiment_matrix_sha256": matrix_sha256,
            "phase7_policy_sha256": policy_sha256,
            "resolved_config_sha256": digest(flat),
        },
        details={
            "phase": "7B",
            "stage": "PRE_RUN",
            "experiment_id": experiment_id,
            "intentional_variable": declaration.intentional_variable,
            "inherits": declaration.inherits,
            "overrides": dict(declaration.overrides),
            "model": protocol["model"],
            "declared_optimizer_policy": training["optimizer"],
            "seed": protocol["seed"],
            "device": "cuda:0",
            "imgsz": training["imgsz"],
            "batch": training["batch"],
            "epochs_configured": training["epochs"],
            "patience": training["patience"],
            "deterministic": training["deterministic"],
            "amp": training["amp"],
            "augmentation": training["augmentation"],
            "checkpoint_selection": protocol["checkpoint_selection"],
            "dataset_fingerprints": {
                key: inputs[key]
                for key in (
                    "class_map_sha256",
                    "modeling_population_sha256",
                    "split_assignment_sha256",
                    "adapter_config_sha256",
                    "adapter_manifest_sha256",
                    "yolo_label_sha256",
                )
            },
            "pretrained_weights": weights,
            "runtime": runtime,
            "test_status": TEST_PROTECTED,
            "started_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        },
    )
    record.add_input(paths.configs / MATRIX_YAML, relative_to=paths.root)
    record.add_input(paths.configs / BASELINE_YAML, relative_to=paths.root)
    record.add_input(paths.reports / POLICY_JSON, relative_to=paths.root)
    record.add_input(paths.reports / ADAPTER_MANIFEST_JSON, relative_to=paths.root)
    destination = paths.reports / f"detection_{experiment_id}_prerun.provenance.json"
    write_json(destination, record.to_dict())
    return destination


def memory_preflight(
    paths: ProjectPaths, protocol: dict[str, Any], weights_path: Path
) -> dict[str, Any]:
    """Prove the frozen batch and image size fit in memory, without measuring skill.

    One optimisation step in a throwaway directory. It is engineering, not an
    experiment: no metric it produces is recorded, and it never shares the
    experimental run directory.

    Args:
        paths: Project layout.
        protocol: The resolved nested protocol.
        weights_path: The pretrained checkpoint.

    Returns:
        The feasibility record.

    Raises:
        ExperimentRunError: If the frozen batch does not fit. The caller must
            classify that as a memory constraint rather than reduce the batch.
    """
    import torch
    from ultralytics import YOLO

    training = protocol["training"]
    scratch = paths.root / RUN_ROOT / MEMORY_PREFLIGHT_NAME
    if scratch.exists():
        shutil.rmtree(scratch)
    torch.cuda.reset_peak_memory_stats()
    try:
        model = YOLO(str(weights_path))
        model.train(
            data=str(paths.root / DATASET_DESCRIPTOR),
            epochs=1,
            imgsz=training["imgsz"],
            batch=training["batch"],
            seed=protocol["seed"],
            deterministic=training["deterministic"],
            workers=training["workers"],
            amp=training["amp"],
            optimizer=training["optimizer"],
            patience=training["patience"],
            pretrained=training["pretrained"],
            val=False,
            plots=False,
            device=0,
            project=str(paths.root / RUN_ROOT),
            name=MEMORY_PREFLIGHT_NAME,
            exist_ok=True,
            verbose=False,
        )
    except torch.cuda.OutOfMemoryError as exc:
        msg = (
            f"CUDA out of memory at the frozen batch {training['batch']} and imgsz "
            f"{training['imgsz']}: {exc}"
        )
        raise ExperimentRunError(msg) from exc
    finally:
        peak = int(torch.cuda.max_memory_allocated())
        reserved = int(torch.cuda.max_memory_reserved())
        if scratch.exists():
            shutil.rmtree(scratch)
        torch.cuda.empty_cache()
    return {
        "status": "FEASIBLE_AT_FROZEN_BATCH",
        "classification": "NON_EXPERIMENTAL_ENGINEERING_PREFLIGHT",
        "batch": training["batch"],
        "imgsz": training["imgsz"],
        "epochs": 1,
        "peak_allocated_bytes": peak,
        "peak_reserved_bytes": reserved,
        "note": (
            "One optimisation step in a throwaway directory, to establish that the frozen "
            "batch fits. DO_NOT_REPORT_AS_MODEL_RESULT: validation was disabled, no metric "
            "was computed and the directory was deleted."
        ),
    }


def train(
    paths: ProjectPaths,
    declaration: ExperimentDeclaration,
    protocol: dict[str, Any],
    weights_path: Path,
    *,
    resume: bool,
    resume_reason: str | None,
) -> dict[str, Any]:
    """Run the single training execution for this experiment.

    Args:
        paths: Project layout.
        declaration: The candidate declaration.
        protocol: The resolved nested protocol.
        weights_path: The pretrained checkpoint.
        resume: Whether this is an identical operational resume.
        resume_reason: Why the run was interrupted, required when resuming.

    Returns:
        The execution record.

    Raises:
        ExperimentRunError: If a completed run already exists, or a resume is
            requested with nothing to resume from.
    """
    import torch
    from ultralytics import YOLO

    experiment_id = declaration.experiment_id
    run_directory = paths.root / RUN_ROOT / experiment_id
    training = protocol["training"]

    if resume:
        last = run_directory / "weights" / "last.pt"
        if not last.is_file():
            msg = f"--resume was requested but no {experiment_id} checkpoint exists to resume from"
            raise ExperimentRunError(msg)
        before = sha256_bytes(last)
        model = YOLO(str(last))
        handler = capture_framework_log(run_directory / TRAIN_CONSOLE_LOG)
        started = time.perf_counter()
        try:
            model.train(resume=True)
        finally:
            release_framework_log(handler)
        return {
            "resumed": True,
            "resume_reason": resume_reason,
            "resumed_from_sha256": before,
            "seconds": round(time.perf_counter() - started, 1),
            "run_directory": f"{RUN_ROOT}/{experiment_id}",
        }

    if run_directory.exists():
        msg = (
            f"{RUN_ROOT}/{experiment_id} already exists. {experiment_id} is one experiment; "
            "refusing to overwrite or mix runs. Move it aside deliberately if it must be re-run."
        )
        raise ExperimentRunError(msg)

    model = YOLO(str(weights_path))
    complexity = model_complexity(model)
    # Attaching the log handler creates the run directory, so `exist_ok` must be
    # True here or the framework treats its own output directory as taken and
    # silently diverts the run to `<id>-2`. The guard against re-running an
    # experiment is the explicit `run_directory.exists()` check above, not this
    # flag; the post-condition below re-checks that the run landed where the
    # result will be read from.
    handler = capture_framework_log(run_directory / TRAIN_CONSOLE_LOG)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        model.train(
            data=str(paths.root / DATASET_DESCRIPTOR),
            epochs=training["epochs"],
            imgsz=training["imgsz"],
            batch=training["batch"],
            seed=protocol["seed"],
            deterministic=training["deterministic"],
            workers=training["workers"],
            amp=training["amp"],
            optimizer=training["optimizer"],
            lr0=training["lr0"],
            lrf=training["lrf"],
            momentum=training["momentum"],
            weight_decay=training["weight_decay"],
            warmup_epochs=training["warmup_epochs"],
            cos_lr=training["cos_lr"],
            close_mosaic=training["close_mosaic"],
            patience=training["patience"],
            pretrained=training["pretrained"],
            val=training["val"],
            device=0,
            project=str(paths.root / RUN_ROOT),
            name=experiment_id,
            exist_ok=True,
            plots=True,
            verbose=True,
        )
    finally:
        release_framework_log(handler)

    # Post-condition: the run must have written into the directory the result is
    # about to be read from. A diverted `<id>-N` directory would otherwise be
    # reported under this experiment's name while its metrics came from nowhere.
    if not (run_directory / "results.csv").is_file():
        msg = (
            f"training produced no results.csv in {RUN_ROOT}/{experiment_id}; the framework may "
            "have written elsewhere and the run cannot be attributed to this experiment"
        )
        raise ExperimentRunError(msg)
    diverted = sorted(
        path.name
        for path in (paths.root / RUN_ROOT).iterdir()
        if path.is_dir() and path.name.startswith(f"{experiment_id}-")
    )
    if diverted:
        msg = (
            f"the framework created {diverted} alongside {RUN_ROOT}/{experiment_id}; the run "
            "was diverted and its output cannot be attributed to this experiment"
        )
        raise ExperimentRunError(msg)
    return {
        "resumed": False,
        "resume_reason": None,
        "seconds": round(time.perf_counter() - started, 1),
        "run_directory": f"{RUN_ROOT}/{experiment_id}",
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "model_complexity": complexity,
    }


def validate_best(
    paths: ProjectPaths, experiment_id: str, protocol: dict[str, Any]
) -> dict[str, Any]:
    """Run the single authoritative validation of the selected checkpoint.

    One evaluation, of ``best.pt``, on the validation split, at the frozen image
    size and batch - the same validation protocol D0 used. No threshold sweep,
    no IoU sweep, no alternative image size, no ``last.pt`` comparison: each of
    those would turn a measurement into a selection.

    Args:
        paths: Project layout.
        experiment_id: The experiment id.
        protocol: The resolved nested protocol.

    Returns:
        The validation record.
    """
    from ultralytics import YOLO

    training = protocol["training"]
    best = paths.root / RUN_ROOT / experiment_id / "weights" / "best.pt"
    model = YOLO(str(best))
    metrics = model.val(
        data=str(paths.root / DATASET_DESCRIPTOR),
        split="val",
        imgsz=training["imgsz"],
        batch=training["batch"],
        device=0,
        project=str(paths.root / RUN_ROOT),
        name=f"{experiment_id}_val",
        exist_ok=True,
        plots=True,
        verbose=False,
    )
    box = metrics.box
    names = [name for _, name in sorted(model.names.items())]
    matrix = getattr(getattr(metrics, "confusion_matrix", None), "matrix", None)
    return {
        "confusion_matrix": (
            [[round(float(value)) for value in row] for row in matrix]
            if matrix is not None
            else None
        ),
        "confusion_matrix_axes": {
            "rows": "predicted",
            "columns": "ground truth",
            "labels": [*names, "background"],
        },
        "global": global_metrics(metrics.results_dict),
        "per_class": per_class_metrics(
            names,
            class_indices=list(box.ap_class_index),
            precision=list(box.p),
            recall=list(box.r),
            ap50=list(box.ap50),
            ap=list(box.ap),
        ),
        "speed": {key: round(float(value), 3) for key, value in metrics.speed.items()},
        "fitness": round(float(metrics.fitness), 6),
        "effective_arguments": {
            "split": "val",
            "imgsz": training["imgsz"],
            "batch": training["batch"],
            "conf": float(getattr(metrics, "conf", 0.001) or 0.001),
            "iou": 0.7,
            "max_det": 300,
            "device": "cuda:0",
            "note": (
                "Ultralytics validation defaults for the pinned version, identical to the "
                "reference experiment's; no threshold, IoU or image-size variant was explored"
            ),
        },
        "output_directory": f"{RUN_ROOT}/{experiment_id}_val",
        "model_complexity": model_complexity(model),
    }


def main(argv: list[str] | None = None) -> int:
    """Run and record one frozen Phase 7 detection experiment.

    Args:
        argv: Command-line arguments.

    Returns:
        Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, help="experiment id declared in the matrix")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="run every pre-flight check and stop before fetching weights or training",
    )
    parser.add_argument(
        "--memory-preflight",
        action="store_true",
        help="prove the frozen batch fits before the full run; non-experimental",
    )
    parser.add_argument(
        "--rebuild-report",
        action="store_true",
        help=(
            "re-render the report and the results table from the committed manifest; "
            "trains nothing, validates nothing and changes no metric"
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue an operationally interrupted run with identical settings",
    )
    parser.add_argument(
        "--resume-reason",
        default=None,
        help="why the run was interrupted; required with --resume",
    )
    args = parser.parse_args(argv)

    paths = ProjectPaths.from_root()
    if holdout_unlocked():
        print(
            f"{BLOCKED}: {HOLDOUT_UNLOCK_ENV_VAR} is set. Phase 7 develops on train and "
            "validation only and this phase declines to run with the holdout unlocked.",
            file=sys.stderr,
        )
        return 2
    if args.resume and not args.resume_reason:
        print(
            f"{BLOCKED}: --resume requires --resume-reason. A resume is an operational event "
            "and must say what interrupted the run.",
            file=sys.stderr,
        )
        return 2

    # --- pre-flight, before any weight is fetched or any step is taken --------
    try:
        matrix = load_experiment_matrix(paths.configs / MATRIX_YAML)
        declaration = matrix.declaration(args.experiment)
        if declaration.role != CANDIDATE_ROLE:
            msg = (
                f"{args.experiment} carries role {declaration.role!r}; only a "
                f"{CANDIDATE_ROLE} may be executed by this script. The reference experiment is "
                "frozen and must not be re-run."
            )
            raise ExperimentRunError(msg)
        verified = verify_policy(paths, matrix)
    except (ComparisonConfigError, ComparisonError, ConfigError, ExperimentRunError) as exc:
        print(f"{PROTOCOL_MISMATCH}: {exc}", file=sys.stderr)
        return 2

    try:
        reference = verify_reference(paths, matrix, verified["reference"])
    except (ExperimentRunError, ComparisonError, ConfigError) as exc:
        print(f"INVALID_D0_REFERENCE: {exc}", file=sys.stderr)
        return 2

    try:
        inputs = verify_frozen_inputs(paths, reference["split"])
        resolved = resolve_candidate(matrix, reference["baseline"], declaration)
    except (ExperimentRunError, ComparisonError) as exc:
        print(f"{PROTOCOL_VIOLATION}: {exc}", file=sys.stderr)
        return 2

    protocol = resolved["nested"]
    verdict = resolved["verdict"]
    experiment_id = declaration.experiment_id
    policy_sha256 = sha256_bytes(paths.reports / POLICY_JSON)

    if args.rebuild_report:
        try:
            rebuilt = rebuild_report(paths, matrix, declaration, reference)
        except ExperimentRunError as exc:
            print(f"{BLOCKED}: {exc}", file=sys.stderr)
            return 2
        print(f"REPORT_REBUILT for {experiment_id} from its committed manifest")
        for line in rebuilt:
            print(f"  {line}")
        return 0

    print(f"pre-flight OK for {experiment_id}")
    print(f"  intentional variable   {declaration.intentional_variable}")
    print(f"  observed differences   {list(verdict.observed_differences)}")
    print(f"  model                  {protocol['model']}  weights {protocol['weight_identifier']}")
    print(
        f"  imgsz {protocol['training']['imgsz']}  batch {protocol['training']['batch']}  "
        f"epochs {protocol['training']['epochs']}  seed {protocol['seed']}"
    )
    print(f"  D0 {PRIMARY_SELECTION_METRIC}  {reference['record'].supported_macro}")
    print(f"  holdout                {TEST_PROTECTED}")

    if args.verify_only:
        print("VERIFIED: pre-flight complete. Nothing was fetched and nothing was trained.")
        return 0

    run_directory = paths.root / RUN_ROOT / experiment_id
    if run_directory.exists() and not args.resume:
        print(
            f"{BLOCKED}: {RUN_ROOT}/{experiment_id} already exists. One experiment means one "
            "run; refusing to overwrite it.",
            file=sys.stderr,
        )
        return 2

    try:
        runtime = runtime_facts()
        weights = ensure_pretrained_weights(paths, protocol["weight_identifier"])
    except RunError as exc:
        print(f"{BLOCKED}: {exc}", file=sys.stderr)
        return 2
    weights_path = paths.root / weights["relative_path"]
    print(
        f"  weights {weights['identifier']}  {weights['sha256']}  {weights['size_bytes']} B  "
        f"via {weights['source_mechanism']}"
    )

    preflight: dict[str, Any] | None = None
    if args.memory_preflight:
        try:
            preflight = memory_preflight(paths, protocol, weights_path)
        except ExperimentRunError as exc:
            print(f"{MEMORY_CONSTRAINT}: {exc}", file=sys.stderr)
            return 3
        print(
            f"  memory preflight       {preflight['status']} peak "
            f"{preflight['peak_allocated_bytes']} B allocated"
        )

    prerun = write_prerun_record(
        paths,
        declaration,
        protocol=protocol,
        flat=resolved["flat"],
        inputs=inputs,
        weights=weights,
        runtime=runtime,
        policy_sha256=policy_sha256,
        matrix_sha256=verified["matrix_sha256"],
    )
    print(f"  pre-run record         {prerun.relative_to(paths.root).as_posix()}")

    # --- the single training run ----------------------------------------------
    import torch

    try:
        execution = train(
            paths,
            declaration,
            protocol,
            weights_path,
            resume=args.resume,
            resume_reason=args.resume_reason,
        )
    except torch.cuda.OutOfMemoryError as exc:
        print(
            f"{MEMORY_CONSTRAINT}: CUDA out of memory at the frozen batch "
            f"{protocol['training']['batch']}: {exc}. The batch is a controlled variable and is "
            "not reduced; this experiment stops for review.",
            file=sys.stderr,
        )
        return 3
    except (ExperimentRunError, RunError) as exc:
        print(f"{TRAINING_FAILED}: {exc}", file=sys.stderr)
        return 4
    except Exception as exc:  # a real failure stops for review, never auto-repairs
        print(f"{TRAINING_FAILED}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 4

    try:
        outcome = record_results(
            paths,
            matrix,
            declaration,
            protocol=protocol,
            flat=resolved["flat"],
            verdict=verdict,
            inputs=inputs,
            weights=weights,
            runtime=runtime,
            reference=reference,
            execution=execution,
            preflight=preflight,
            policy_sha256=policy_sha256,
            matrix_sha256=verified["matrix_sha256"],
        )
    except (ExperimentRunError, ResultError, ComparisonError) as exc:
        print(f"{TRAINING_FAILED}: {exc}", file=sys.stderr)
        return 4

    print(EXPERIMENT_COMPLETE)
    for line in outcome:
        print(f"  {line}")
    return 0


def build_report(
    manifest: dict[str, Any],
    matrix: ExperimentMatrix,
    declaration: ExperimentDeclaration,
    reference: dict[str, Any],
) -> str:
    """Render the human-readable experiment report.

    Args:
        manifest: The emitted result manifest.
        matrix: The parsed experiment matrix.
        declaration: The candidate declaration.
        reference: The verified reference facts.

    Returns:
        The report body.
    """
    experiment_id = manifest["experiment_id"]
    comparison = manifest["phase7_comparison"]
    metrics = manifest["validation_metrics"]
    reference_metrics = reference["manifest"]["validation_metrics"]
    reference_per_class = reference["manifest"]["per_class_metrics"]
    deltas = comparison["deltas_vs_reference"]
    supported = set(comparison["selection_metric_classes"])
    optimizer = manifest["resolved_optimizer_determination"]
    reference_id = matrix.reference_experiment

    lines: list[str] = []
    add = lines.append

    add(f"# {experiment_id} - YOLO11s Detection Capacity Experiment - Phase 7B")
    add("")
    add(
        f"Phase: {manifest['phase']} · Commit: `{manifest['git_commit']}` · Status: "
        f"**{manifest['status']}**"
    )
    add("")
    add(
        f"**{experiment_id} is one controlled experiment, not a verdict.** It answers a single "
        f"question against {reference_id}, and the Phase 7 winner cannot be declared here "
        "because D2 has not been executed. Every number below is a **validation** number."
    )
    add("")
    add(
        "Claims are labelled `PREDECLARED_PROTOCOL` (fixed before the run), `COMPUTED_RESULT` "
        "(measured by this run), `CONTROLLED_COMPARISON` (a difference against the reference "
        "under the frozen contract), `OBSERVATION` (a reading of those numbers), `LIMITATION` "
        "and `PENDING_EXPERIMENT`."
    )
    add("")

    add("## 1. Experimental question")
    add("")
    add(f"`PREDECLARED_PROTOCOL` {manifest['question']}")
    add("")
    add(f"**Hypothesis, recorded before the run.** {manifest['hypothesis']}")
    add("")

    add("## 2. Controlled-variable contract")
    add("")
    verdict = manifest["protocol_compatibility"]
    add(
        f"`PREDECLARED_PROTOCOL` Intentional variable `{manifest['intentional_variable']}`. "
        f"{experiment_id} carries no protocol of its own: it inherits "
        f"`configs/{BASELINE_YAML}` and applies a declared override set, and the run refuses "
        "to start if the resolved protocol differs from the reference anywhere it did not "
        "declare."
    )
    add("")
    add("| Field | Kind | Value |")
    add("| --- | --- | --- |")
    for field_path, value in sorted(manifest["overrides"].items()):
        kind = "intentional" if field_path in manifest["intentional_fields"] else "consequential"
        add(f"| `{field_path}` | {kind} | `{value}` |")
    add("")
    add(
        f"`CONTROLLED_COMPARISON` Observed differences against {reference_id}: "
        + ", ".join(f"`{name}`" for name in verdict["observed_differences"])
        + f". Contract satisfied: **{'yes' if verdict['compatible'] else 'NO'}**."
    )
    add("")
    add(
        "Identical to the reference by inheritance: split, labels, class map, epochs, batch, "
        "image size, optimizer policy, patience, seed, `deterministic`, augmentation policy, "
        "checkpoint-selection rule and validation protocol."
    )
    add("")

    add("## 3. Frozen Phase 7 comparison policy")
    add("")
    add("| Item | Value |")
    add("| --- | --- |")
    add(f"| Primary selection metric | `{PRIMARY_SELECTION_METRIC}` |")
    add(f"| Official all-class metric | `{OFFICIAL_ALL_CLASS_METRIC}` |")
    add(
        f"| Support rule | >= {matrix.support_rule.min_positive_images} positive validation "
        f"images **and** >= {matrix.support_rule.min_instances} instances |"
    )
    add(
        "| Selection classes | "
        + ", ".join(f"`{name}`" for name in comparison["selection_metric_classes"])
        + " |"
    )
    add(
        "| Descriptive classes | "
        + ", ".join(f"`{name}`" for name in comparison["descriptive_classes"])
        + " |"
    )
    add(f"| Practical-equivalence margin | {comparison['practical_equivalence_margin']} |")
    add(f"| Policy SHA-256 | `{manifest['phase7_policy_sha256']}` |")
    add(f"| Experiment matrix SHA-256 | `{manifest['experiment_matrix_sha256']}` |")
    add("")
    add(
        "`PREDECLARED_PROTOCOL` The support rule and the margin were frozen in phase 7A, before "
        f"{experiment_id} existed. Neither was touched by this phase."
    )
    add("")

    add("## 4. Input and runtime provenance")
    add("")
    add("| Fingerprint | Value |")
    add("| --- | --- |")
    split_sha = manifest["split_reference"]["split_assignment_sha256"]
    add(f"| `split_assignment_sha256` | `{split_sha}` |")
    add(f"| `class_map_sha256` | `{manifest['dataset_fingerprints']['class_map_sha256']}` |")
    add(f"| `adapter_manifest_sha256` | `{manifest['adapter_manifest_sha256']}` |")
    add(f"| `resolved_config_sha256` | `{manifest['resolved_config_sha256']}` |")
    add(f"| `{experiment_id}_experiment_sha256` | `{manifest['experiment_sha256']}` |")
    add("")
    weights = manifest["pretrained_weights"]
    add("| Pretrained weights | Value |")
    add("| --- | --- |")
    add(f"| Identifier | `{weights['identifier']}` |")
    add(f"| SHA-256 | `{weights['sha256']}` |")
    add(f"| Size | {weights['size_bytes']} bytes |")
    add(
        f"| Source | {weights['source_mechanism']} (ultralytics {weights['ultralytics_version']}) |"
    )
    add(f"| Committed | {weights['committed']} |")
    add("")
    add(
        "`PREDECLARED_PROTOCOL` Fingerprinted rather than trusted by name: two files called "
        f"`{weights['identifier']}` are not necessarily the same bytes. The binary is not "
        "committed."
    )
    add("")
    runtime = manifest["runtime"]
    add("| Component | Version |")
    add("| --- | --- |")
    for label, key in (
        ("Python", "python"),
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("ultralytics", "ultralytics"),
        ("CUDA runtime", "cuda_runtime"),
    ):
        add(f"| {label} | {runtime[key]} |")
    add(f"| GPU | {runtime['gpu_name']} ({runtime['gpu_arch']}) |")
    add("")

    add("## 5. Effective training configuration")
    add("")
    add(
        "`COMPUTED RESULT` The protocol declares `optimizer: auto`, so the declared policy and "
        "the settings that actually ran are **not the same thing**. What follows is what the "
        "run used."
    )
    add("")
    add("| Setting | Declared | Effective |")
    add("| --- | --- | --- |")
    resolved = manifest["resolved_training_arguments"]
    declared = manifest["resolved_protocol"]
    add(f"| Optimizer | `{manifest['declared_optimizer_policy']}` | **{optimizer['optimizer']}** |")
    add(f"| `lr0` | {declared['training.lr0']} | {optimizer.get('effective_lr0')} |")
    add(
        f"| Momentum | {declared['training.momentum']} | "
        f"{optimizer.get('effective_momentum', NOT_EXPOSED)} |"
    )
    for label, path in (
        ("`imgsz`", "training.imgsz"),
        ("Batch", "training.batch"),
        ("Epochs", "training.epochs"),
        ("Patience", "training.patience"),
        ("Seed", "seed"),
        ("`deterministic`", "training.deterministic"),
        ("AMP", "training.amp"),
        ("`weight_decay`", "training.weight_decay"),
        ("`warmup_epochs`", "training.warmup_epochs"),
        ("`close_mosaic`", "training.close_mosaic"),
        ("`cos_lr`", "training.cos_lr"),
        ("Workers", "training.workers"),
    ):
        key = path.rsplit(".", 1)[-1]
        add(f"| {label} | {declared[path]} | {resolved.get(key, NOT_EXPOSED)} |")
    add("")
    add(
        f"`COMPUTED_RESULT` **Optimizer evidence: `{optimizer['source']}`.** "
        + (
            "Read directly from the framework's own log line at the moment it built the "
            f"optimizer: `{optimizer.get('evidence')}`. Nothing was inferred."
            if not optimizer.get("inferred")
            else "This is an **inference** from the framework's documented rule, not a reading. "
            f"Iteration estimate {optimizer.get('iterations_estimate')}, observed peak "
            f"learning rate {optimizer.get('observed_peak_lr')}, corroborated: "
            f"{optimizer.get('corroborated_by_observed_lr')}."
        )
    )
    add("")
    add("`PREDECLARED_PROTOCOL` Effective augmentation values, as resolved by the framework:")
    add("")
    add("| Augmentation | Value |")
    add("| --- | --- |")
    for key in (
        "hsv_h",
        "hsv_s",
        "hsv_v",
        "degrees",
        "translate",
        "scale",
        "shear",
        "perspective",
        "flipud",
        "fliplr",
        "mosaic",
        "mixup",
        "cutmix",
        "copy_paste",
        "erasing",
        "auto_augment",
    ):
        if key in resolved:
            add(f"| `{key}` | {resolved[key]} |")
    add("")
    add(
        "`CONTROLLED_COMPARISON` These are the frozen `ULTRALYTICS_DEFAULT` policy for the "
        "pinned version, inherited unchanged from the reference. No augmentation was tuned."
    )
    add("")

    add("## 6. Training execution")
    add("")
    add("| | |")
    add("| --- | --- |")
    add(f"| Termination | `{manifest['termination_mode']}` |")
    add(f"| Epochs configured | {manifest['epochs_configured']} |")
    add(f"| Epochs completed | {manifest['epochs_completed']} |")
    add(f"| Early stopped | {manifest['early_stopped']} |")
    add(f"| Resumed | {manifest['resumed']} |")
    add(f"| Duration | {manifest['training_duration_seconds']} s |")
    memory = manifest["peak_gpu_memory"]
    add(f"| Peak GPU allocated | {memory['allocated_bytes']} bytes |")
    add(f"| Peak GPU reserved | {memory['reserved_bytes']} bytes |")
    add("")
    complexity = manifest.get("model_complexity") or {}
    if complexity:
        add("| Model complexity | Value |")
        add("| --- | --- |")
        add(f"| Parameters | {complexity.get('parameters', NOT_EXPOSED)} |")
        add(f"| GFLOPs | {complexity.get('gflops', NOT_EXPOSED)} |")
        add(f"| Source | {complexity.get('source', NOT_EXPOSED)} |")
        add("")
        add(
            "`OBSERVATION` Parameter count and FLOPs describe what the intentional variable "
            "actually changed. They are descriptive facts at this stage and are **not** used to "
            "choose between experiments; that would be an efficiency comparison, which is not "
            "authorised here."
        )
        add("")

    add("## 7. Checkpoint selection")
    add("")
    add(f"`PREDECLARED_PROTOCOL` {manifest['checkpoint_selection']}")
    add("")
    add("| | |")
    add("| --- | --- |")
    add(f"| Best epoch | {manifest['best_epoch']} |")
    add(f"| Best validation fitness | {manifest['best_validation_fitness']} |")
    add(f"| `best.pt` SHA-256 | `{manifest['best_checkpoint']['sha256']}` |")
    add(f"| `best.pt` size | {manifest['best_checkpoint']['size_bytes']} bytes |")
    add(f"| `last.pt` SHA-256 | `{manifest['last_checkpoint']['sha256']}` |")
    add(f"| `last.pt` size | {manifest['last_checkpoint']['size_bytes']} bytes |")
    add("")
    cross = manifest["headline_metric_cross_check"]
    add(
        f"`COMPUTED_RESULT` The published headline metric was cross-checked against the epoch "
        f"history recomputed independently from `results.csv`: authoritative "
        f"{cross['authoritative']}, history at the selected epoch {cross['history']}, delta "
        f"{cross['delta']} against a tolerance of {cross['tolerance']}. So the reported number "
        "provably belongs to `best.pt` rather than to the last epoch."
    )
    add("")
    add("`LIMITATION` Neither checkpoint is committed. Both are referenced by digest.")
    add("")

    add("## 8. Validation metrics")
    add("")
    configuration = manifest["validation_configuration"]
    add(
        f"`PREDECLARED_PROTOCOL` One evaluation of `best.pt` on the frozen validation split at "
        f"imgsz {configuration['imgsz']}, batch {configuration['batch']}, conf "
        f"{configuration['conf']}, IoU {configuration['iou']} - the same validation protocol "
        f"the reference used. No threshold sweep, no IoU sweep, no alternative image size, no "
        "`last.pt` comparison."
    )
    add("")
    add(f"| Global metric | {experiment_id} | {reference_id} | Delta |")
    add("| --- | --- | --- | --- |")
    for name in (OFFICIAL_ALL_CLASS_METRIC, "mAP@0.50", "precision", "recall"):
        current = metrics.get(name)
        prior = reference_metrics.get(name)
        add(f"| `{name}` | {current} | {prior} | {deltas[name]:+f} |")
    add("")

    add("## 9. Supported-class primary metric")
    add("")
    add(
        f"`COMPUTED_RESULT` `{PRIMARY_SELECTION_METRIC}` is the unweighted mean of per-class "
        "`AP@0.50:0.95` over the classes the frozen support rule admits: "
        + ", ".join(f"`{name}`" for name in comparison["selection_metric_classes"])
        + "."
    )
    add("")
    add("| | Value |")
    add("| --- | --- |")
    add(f"| {experiment_id} | **{comparison['supported_macro_map50_95']}** |")
    add(f"| {experiment_id} exact | {comparison['supported_macro_exact']} |")
    add(f"| {reference_id} | {comparison['reference_supported_macro_map50_95']} |")
    add(f"| Delta | **{comparison['delta_supported_macro_vs_reference']:+f}** |")
    add("")

    add("## 10. Comparison with the reference")
    add("")
    add(
        f"`CONTROLLED_COMPARISON` Margin status: **`{comparison['margin_status']}`** at a frozen "
        f"margin of {comparison['practical_equivalence_margin']}."
    )
    add("")
    add(
        "`LIMITATION` The margin is an engineering decision threshold, not a significance test. "
        "`deterministic: true` reduces run-to-run variance without eliminating it, and no "
        "experiment is repeated, so the size of that variance on this setup is **UNKNOWN**. A "
        "difference near the margin is not an ordering."
    )
    add("")
    add(
        "`PENDING_EXPERIMENT` The frozen Phase 7 selection logic is **not applied here**. It "
        "runs once every declared candidate has a result, and D2 has not been executed. "
        f"{experiment_id} is therefore not the Phase 7 winner, and no final detector is "
        "declared by this phase."
    )
    add("")

    add("## 11. Official all-class reporting metric")
    add("")
    add(
        f"`COMPUTED_RESULT` The five-class `{OFFICIAL_ALL_CLASS_METRIC}` is "
        f"{metrics.get(OFFICIAL_ALL_CLASS_METRIC)} against the reference's "
        f"{reference_metrics.get(OFFICIAL_ALL_CLASS_METRIC)}, a delta of "
        f"{deltas[OFFICIAL_ALL_CLASS_METRIC]:+f}. It remains "
        "`OFFICIAL_ALL_CLASS_REPORTING_METRIC` and is reported whichever direction it moved."
    )
    add("")
    add(
        "`LIMITATION` The all-class figure includes the descriptive class, so part of any "
        "movement in it can come from a class standing on very little validation evidence. That "
        "is exactly why it is not the selection metric - and exactly why it is not suppressed "
        "either."
    )
    add("")

    add("## 11b. When the two metrics disagree")
    add("")
    per_class = manifest["per_class_metrics"]
    selection_delta = comparison["delta_supported_macro_vs_reference"]
    all_class_delta = deltas[OFFICIAL_ALL_CLASS_METRIC]
    if not (
        isinstance(all_class_delta, (int, float))
        and (selection_delta > 0) != (all_class_delta > 0)
        and selection_delta != 0
        and all_class_delta != 0
    ):
        add(
            "`OBSERVATION` The selection metric and the all-class metric moved in the same "
            "direction here, so nothing turns on which of the two is read. The section is kept "
            "because that agreement is a fact about this run, not a property of the metrics."
        )
        add("")
    else:
        add(
            f"`COMPUTED_RESULT` **The two metrics moved in opposite directions.** "
            f"`{PRIMARY_SELECTION_METRIC}` fell by {abs(selection_delta):f} while the "
            f"all-class `{OFFICIAL_ALL_CLASS_METRIC}` rose by {abs(all_class_delta):f}. That is "
            "not a contradiction and neither number is wrong; they average different class sets, "
            "and this run happens to sit where the choice of set flips the sign. Since it would "
            "be easy to quote whichever one flatters the experiment, the arithmetic is set out "
            "in full."
        )
        add("")
        add("| Class | `AP@0.50:0.95` delta | In selection metric |")
        add("| --- | --- | --- |")
        for name in sorted(per_class):
            prior = reference_per_class.get(name, {}).get("AP@0.50:0.95")
            current = per_class[name].get("AP@0.50:0.95")
            if not isinstance(prior, (int, float)) or not isinstance(current, (int, float)):
                continue
            add(f"| `{name}` | {current - prior:+f} | {'yes' if name in supported else 'no'} |")
        add("")
        descriptive_names = [
            name for name in comparison["descriptive_classes"] if name in per_class
        ]
        contribution = 0.0
        for name in descriptive_names:
            prior = reference_per_class.get(name, {}).get("AP@0.50:0.95")
            current = per_class[name].get("AP@0.50:0.95")
            if isinstance(prior, (int, float)) and isinstance(current, (int, float)):
                contribution += (current - prior) / len(per_class)
        add(
            "`COMPUTED_RESULT` Both metrics are unweighted means over the same per-class "
            "values, so each class's contribution is its delta divided by the number of classes "
            "averaged. The descriptive class "
            + ", ".join(f"`{name}`" for name in descriptive_names)
            + f" contributes {contribution:+f} to the all-class mean. Remove that single "
            f"contribution and the all-class delta becomes "
            f"{all_class_delta - contribution:+f} - the same direction as the selection metric. "
            "**The entire sign change in the official metric comes from the class the support "
            "rule set aside.**"
        )
        add("")
        add(
            f"`CONTROLLED_COMPARISON` The frozen policy governs: "
            f"`{PRIMARY_SELECTION_METRIC}` decides the ranking, and its verdict is "
            f"`{comparison['margin_status']}`. The all-class improvement **must not** be quoted "
            "as evidence that this experiment beat the reference. That is precisely the "
            "metric-shopping the phase 7A policy was written to prevent, and the policy was "
            "frozen before this result existed - which is the only reason this paragraph is a "
            "rule being applied rather than an excuse being made."
        )
        add("")
        add(
            "`LIMITATION` Equally, the all-class figure is not suppressed and is not being "
            "called wrong. It is the official reporting metric and it improved. What it cannot "
            "do is order two models when its movement is driven by a class standing on one "
            "validation image."
        )
        add("")

    add("## 12. Per-class metrics")
    add("")
    add(
        f"| Class | `AP@0.50` | `AP@0.50:0.95` | precision | recall | {reference_id} "
        "`AP@0.50:0.95` | In selection metric |"
    )
    add("| --- | --- | --- | --- | --- | --- | --- |")
    for name in sorted(manifest["per_class_metrics"]):
        table = manifest["per_class_metrics"][name]
        prior = reference_per_class.get(name, {}).get("AP@0.50:0.95", NOT_EXPOSED)
        add(
            f"| `{name}` | {table['AP@0.50']} | {table['AP@0.50:0.95']} | {table['precision']} "
            f"| {table['recall']} | {prior} | {'yes' if name in supported else 'no'} |"
        )
    add("")

    add("## 13. The descriptive class")
    add("")
    rare = manifest["rare_class"]
    support_rows = {row["class_name"]: row for row in manifest["class_support"]}
    for name in rare["classes"]:
        row = support_rows[name]
        add(
            f"`LIMITATION` `{name}` holds {row['validation_positive_images']} positive "
            f"validation image(s) and {row['validation_instances']} instances, so the frozen "
            f"rule classified it `{DESCRIPTIVE_HIGH_UNCERTAINTY}`."
        )
        add("")
    add(rare["limitation"])
    add("")
    add(
        "It is reported in full in section 12 and was **not** used in the selection metric, not "
        "tuned for, and not allowed to move the comparison in either direction."
    )
    add("")

    add("## 14. Recall and precision diagnostic")
    add("")
    add(
        f"`COMPUTED_RESULT` Precision {metrics.get('precision')} against recall "
        f"{metrics.get('recall')}; the reference reported {reference_metrics.get('precision')} "
        f"against {reference_metrics.get('recall')}. Recall delta {deltas['recall']:+f}, "
        f"precision delta {deltas['precision']:+f}."
    )
    add("")
    add(
        "`PREDECLARED_PROTOCOL` Recall is a **diagnostic** here. It helps explain why the "
        "selection metric moved; it is not promoted to a selection objective and no "
        "recall-weighted composite is introduced."
    )
    add("")

    add("## 15. Training dynamics")
    add("")
    add(
        f"`OBSERVATION` The run terminated as `{manifest['termination_mode']}` after "
        f"{manifest['epochs_completed']} of {manifest['epochs_configured']} configured epochs, "
        f"with the predeclared rule selecting epoch {manifest['best_epoch']}."
    )
    add("")
    add(
        "`LIMITATION` Curves are read at the level of aggregate metrics only. No individual "
        "validation image was opened, and no qualitative source-image analysis was performed - "
        "that is a deliberate later phase."
    )
    add("")

    add("## 16. Confusion matrix")
    add("")
    matrix_values = manifest["confusion_matrix"]
    if matrix_values:
        labels = manifest["confusion_matrix_axes"]["labels"]
        add("`COMPUTED_RESULT` Rows predicted, columns ground truth.")
        add("")
        add("| predicted \\ truth | " + " | ".join(f"`{name}`" for name in labels) + " |")
        add("| --- " * (len(labels) + 1) + "|")
        for index, row in enumerate(matrix_values):
            add(f"| `{labels[index]}` | " + " | ".join(str(value) for value in row) + " |")
        add("")
        # Rows are predictions, columns ground truth, so the background row and
        # the background column mean opposite things: the row is "predicted
        # something, nothing was there", the column is "something was there,
        # predicted nothing".
        background = len(labels) - 1
        unmatched_predictions = sum(row[background] for row in matrix_values)
        undetected_objects = sum(matrix_values[background][column] for column in range(background))
        confusions = sum(
            matrix_values[predicted][truth]
            for predicted in range(background)
            for truth in range(background)
            if predicted != truth
        )
        add(
            f"`OBSERVATION` Structural reading only: {unmatched_predictions} predictions "
            f"matched no ground-truth object, {undetected_objects} ground-truth objects went "
            f"undetected, and {confusions} objects were detected but given the wrong class."
        )
        add("")
        add(
            "`LIMITATION` These are counts. Explaining any individual error would require "
            "opening validation images, which this phase does not do."
        )
        add("")

    add("## 17. Runtime and resource observations")
    add("")
    speed = manifest["framework_validation_speed_ms_per_image"]
    add("| Validation stage | ms/image |")
    add("| --- | --- |")
    for key in sorted(speed):
        add(f"| {key} | {speed[key]} |")
    add("")
    add(
        "`OBSERVATION` `FRAMEWORK_VALIDATION_SPEED`, as the framework reported it during the "
        "authoritative validation. No separate benchmark was run, the numbers include this "
        "machine's incidental load, and they are **descriptive only** - runtime efficiency is "
        "not used to choose between experiments at this stage."
    )
    add("")

    add("## 18. Holdout compliance")
    add("")
    add(
        f"`PREDECLARED_PROTOCOL` `{manifest['test']['status']}` - {manifest['test']['reason']} "
        f"`{HOLDOUT_UNLOCK_ENV_VAR}` was not set at any point, the adapter dataset descriptor "
        "carries no holdout key, and no holdout image, label, count, prediction or metric was "
        "produced. Every number in this report comes from the "
        f"{manifest['validation_images']}-image validation split."
    )
    add("")

    add("## 19. Interpretation")
    add("")
    add(
        f"`CONTROLLED_COMPARISON` Increasing capacity from YOLO11n to YOLO11s, with the data, "
        f"image size, batch and training protocol held fixed by inheritance, moved "
        f"`{PRIMARY_SELECTION_METRIC}` by "
        f"{comparison['delta_supported_macro_vs_reference']:+f}, which the frozen margin "
        f"classifies as `{comparison['margin_status']}`."
    )
    add("")
    # The aggregate hides the shape of the change, and the shape is the useful
    # part: a metric that moved because one class collapsed is a different
    # finding from a metric that moved because everything drifted.
    movements: list[tuple[str, float]] = []
    for name in supported:
        prior = reference_per_class.get(name, {}).get("AP@0.50:0.95")
        current = per_class.get(name, {}).get("AP@0.50:0.95")
        if isinstance(prior, (int, float)) and isinstance(current, (int, float)):
            movements.append((name, current - prior))
    if movements:
        gained = sorted(
            ((name, delta) for name, delta in movements if delta > 0),
            key=lambda item: -item[1],
        )
        lost = sorted((name, delta) for name, delta in movements if delta <= 0)
        add(
            f"`OBSERVATION` The aggregate hides the shape of the change. Of the "
            f"{len(movements)} classes in the selection metric, {len(gained)} improved and "
            f"{len(lost)} did not."
        )
        add("")
        if gained:
            add(
                "- improved: "
                + ", ".join(f"`{name}` {delta:+f}" for name, delta in gained)
                + f" (total {sum(delta for _, delta in gained):+f})"
            )
        if lost:
            add(
                "- declined: "
                + ", ".join(f"`{name}` {delta:+f}" for name, delta in lost)
                + f" (total {sum(delta for _, delta in lost):+f})"
            )
        add("")
        if gained and lost and abs(sum(d for _, d in lost)) > sum(d for _, d in gained):
            worst = min(movements, key=lambda item: item[1])
            worst_recall = per_class.get(worst[0], {}).get("recall")
            prior_recall = reference_per_class.get(worst[0], {}).get("recall")
            recall_note = ""
            if isinstance(worst_recall, (int, float)) and isinstance(prior_recall, (int, float)):
                recall_note = (
                    f" Its recall moved {worst_recall - prior_recall:+f}, from "
                    f"{prior_recall} to {worst_recall}, so the class lost detections rather "
                    "than localisation quality."
                )
            add(
                f"`OBSERVATION` So this is not a uniformly worse model. A single class, "
                f"`{worst[0]}`, fell by {abs(worst[1]):f} - more than the other improvements "
                f"combined - and that one class is what carries the selection metric below the "
                f"reference.{recall_note}"
            )
            add("")
            add(
                f"`LIMITATION` Why `{worst[0]}` behaved that way is **UNKNOWN**. A plausible "
                "story is easy to construct and none is tested here: this is one run, the "
                "movement could be run-to-run variance, and diagnosing it would need either a "
                "repeated run or the image-level error analysis this phase deliberately does "
                "not perform. It is recorded as an open question, not explained."
            )
            add("")
    add(
        "`LIMITATION` This is one run of each configuration on a 65-image validation split. It "
        "establishes what these two runs scored; it does not establish that capacity causes the "
        "difference in general, and it cannot separate a real effect from run-to-run variance, "
        "because neither experiment was repeated."
    )
    add("")
    add(
        "`LIMITATION` Nothing here was tuned, and nothing may be tuned in response to it. The "
        "phase 7A policy forbids trying another capacity, another resolution, another optimizer "
        "or another augmentation setting because of what this result shows."
    )
    add("")

    add("## 20. D2 remains pending")
    add("")
    add(
        "`PENDING_EXPERIMENT` D2 (`INPUT_RESOLUTION`, imgsz 768) is frozen and "
        "**not executed**. Until it has a result:"
    )
    add("")
    add("- the Phase 7 A/B/C selection logic is not applied;")
    add(f"- no experiment is called the Phase 7 winner, {experiment_id} included;")
    add("- the reference remains the preferred detector by default, not by comparison;")
    add("- no efficiency or latency benchmark is run to break any tie.")
    add("")
    add(f"Committed metric-only figures: `reports/figures/detection/{experiment_id}/`.")
    add("")
    return "\n".join(lines)


def rebuild_report(
    paths: ProjectPaths,
    matrix: ExperimentMatrix,
    declaration: ExperimentDeclaration,
    reference: dict[str, Any],
) -> list[str]:
    """Re-render the report and results table from the committed manifest.

    For correcting or extending the prose around a result without touching the
    result. It reads the committed manifest and writes only the report and the
    results table: it does not train, does not validate, does not load a
    checkpoint and does not recompute a metric. The manifest is left byte-identical,
    which is what makes this safe to run on a published experiment.

    Args:
        paths: Project layout.
        matrix: The parsed experiment matrix.
        declaration: The candidate declaration.
        reference: The verified reference facts.

    Returns:
        Summary lines for the console.

    Raises:
        ExperimentRunError: If no committed manifest exists, or rewriting it
            would change it.
    """
    experiment_id = declaration.experiment_id
    manifest_path = paths.reports / f"detection_{experiment_id}_manifest.json"
    if not manifest_path.is_file():
        msg = f"no committed manifest for {experiment_id}; there is no result to re-render"
        raise ExperimentRunError(msg)
    before = manifest_path.read_bytes()
    manifest = read_json(manifest_path)

    problems = validate_result_manifest(manifest, class_names=reference["class_names"])
    if problems:
        msg = f"the committed {experiment_id} manifest is not valid: {'; '.join(problems)}"
        raise ExperimentRunError(msg)

    report = build_report(manifest, matrix, declaration, reference)
    unsafe = scan_for_sensitive(report)
    if unsafe:
        msg = f"the re-rendered {experiment_id} report is not fit to commit: {'; '.join(unsafe)}"
        raise ExperimentRunError(msg)
    (paths.reports / f"detection_{experiment_id}_report.md").write_text(
        report, encoding="utf-8", newline="\n"
    )
    update_results_artifact(paths, matrix, manifest, reference)

    if manifest_path.read_bytes() != before:
        msg = (
            f"re-rendering changed detection_{experiment_id}_manifest.json; a report rebuild "
            "must never touch a published metric"
        )
        raise ExperimentRunError(msg)
    comparison = manifest["phase7_comparison"]
    return [
        f"manifest unchanged           {manifest_path.name}",
        f"{PRIMARY_SELECTION_METRIC}  {comparison['supported_macro_map50_95']} "
        f"(delta {comparison['delta_supported_macro_vs_reference']:+f})",
        f"margin status                {comparison['margin_status']}",
        "no training, no validation, no metric recomputed",
    ]


def record_results(
    paths: ProjectPaths,
    matrix: ExperimentMatrix,
    declaration: ExperimentDeclaration,
    *,
    protocol: dict[str, Any],
    flat: dict[str, Any],
    verdict: Any,
    inputs: dict[str, Any],
    weights: dict[str, Any],
    runtime: dict[str, Any],
    reference: dict[str, Any],
    execution: dict[str, Any],
    preflight: dict[str, Any] | None,
    policy_sha256: str,
    matrix_sha256: str,
) -> list[str]:
    """Validate the run, compare it against the reference and record everything.

    Args:
        paths: Project layout.
        matrix: The parsed experiment matrix.
        declaration: The candidate declaration.
        protocol: The resolved nested protocol.
        flat: The resolved flat protocol.
        verdict: The protocol-compatibility verdict.
        inputs: The verified frozen input fingerprints.
        weights: The pretrained weight provenance.
        runtime: The runtime facts.
        reference: The verified reference facts.
        execution: The training execution record.
        preflight: The memory feasibility record, when one was run.
        policy_sha256: Digest of the committed phase 7 policy.
        matrix_sha256: Digest of the experiment matrix configuration.

    Returns:
        Summary lines for the console.

    Raises:
        ExperimentRunError: If the result cannot be recorded honestly.
    """
    experiment_id = declaration.experiment_id
    run_directory = paths.root / RUN_ROOT / experiment_id
    validation_directory = paths.root / RUN_ROOT / f"{experiment_id}_val"

    history = read_epoch_history(run_directory)
    if not history:
        msg = f"{experiment_id} produced no results.csv; there is no run to report"
        raise ExperimentRunError(msg)
    best_epoch, best_fitness = best_epoch_from_history(history)
    epochs_completed = int(float(history[-1]["epoch"]))
    epochs_configured = int(protocol["training"]["epochs"])
    early_stopped = epochs_completed < epochs_configured

    best = checkpoint_record(run_directory / "weights" / "best.pt", run_name=experiment_id)
    last = checkpoint_record(run_directory / "weights" / "last.pt", run_name=experiment_id)

    validation = validate_best(paths, experiment_id, protocol)
    cross_check = cross_check_metrics(validation["global"].get(PRIMARY_METRIC), history, best_epoch)

    resolved_arguments = resolved_training_arguments(run_directory)
    optimizer = optimizer_evidence(
        run_directory,
        requested=protocol["training"]["optimizer"],
        class_count=len(inputs["class_map"]),
        train_images=inputs["train_images"],
        batch=int(protocol["training"]["batch"]),
        epochs=epochs_configured,
        history=history,
        ultralytics_version=runtime["ultralytics"],
    )

    support = reference["support"]
    supported = supported_class_names(support)
    descriptive = descriptive_class_names(support)
    macro = supported_macro(validation["per_class"], supported)
    reference_macro = reference["record"].supported_macro
    delta_macro = macro - reference_macro
    margin = matrix.practical_equivalence_margin
    classification = classify_delta(delta_macro, margin=margin)
    margin_status = {
        "IMPROVED_BEYOND_MARGIN": IMPROVES,
        "PRACTICALLY_EQUIVALENT_ON_THIS_VALIDATION_SET": EQUIVALENT,
        "REGRESSED_BEYOND_MARGIN": BELOW,
    }[classification]

    reference_global = reference["manifest"]["validation_metrics"]
    deltas: dict[str, Any] = {}
    for name in (OFFICIAL_ALL_CLASS_METRIC, "mAP@0.50", "precision", "recall"):
        current = validation["global"].get(name)
        prior = reference_global.get(name)
        deltas[name] = (
            round(float(current) - float(prior), 6)
            if isinstance(current, (int, float)) and isinstance(prior, (int, float))
            else NOT_EXPOSED
        )

    complexity = execution.get("model_complexity") or validation.get("model_complexity") or {}

    fingerprint = digest(
        {
            "phase7_policy_sha256": policy_sha256,
            "experiment_matrix_sha256": matrix_sha256,
            "resolved_config_sha256": digest(flat),
            "pretrained_weights_sha256": weights["sha256"],
            "adapter_manifest_sha256": inputs["adapter_manifest_sha256"],
            "split_assignment_sha256": inputs["split_assignment_sha256"],
            "class_map_sha256": inputs["class_map_sha256"],
            "critical_arguments": critical_arguments(resolved_arguments),
            "best_checkpoint_sha256": best["sha256"],
        }
    )

    figures = copy_metric_figures(
        paths, [run_directory, validation_directory], run_name=experiment_id
    )

    manifest = {
        "schema_version": 1,
        "phase": "7B",
        "task": "detection",
        "experiment_id": experiment_id,
        "status": EXPERIMENT_STATUS_COMPLETE,
        "role": declaration.role,
        "reference_experiment": matrix.reference_experiment,
        "intentional_variable": declaration.intentional_variable,
        "intentional_fields": list(declaration.intentional_fields),
        "consequential_fields": list(declaration.consequential_fields),
        "overrides": dict(declaration.overrides),
        "inherits": declaration.inherits,
        "question": declaration.question.strip(),
        "hypothesis": declaration.hypothesis.strip(),
        "model": protocol["model"],
        "git_commit": git_commit(paths.root),
        "phase7_policy_sha256": policy_sha256,
        "experiment_matrix_sha256": matrix_sha256,
        "resolved_config_sha256": digest(flat),
        "resolved_protocol": dict(sorted(flat.items())),
        "protocol_compatibility": verdict.as_dict(),
        "baseline_config_sha256": reference["baseline"].fingerprint(),
        "adapter_manifest_sha256": inputs["adapter_manifest_sha256"],
        "dataset_fingerprints": {
            key: inputs[key]
            for key in (
                "adapter_config_sha256",
                "class_map_sha256",
                "modeling_population_sha256",
                "task_materialization_config_sha256",
                "yolo_label_sha256",
            )
        },
        "split_reference": {
            "manifest": f"reports/{SPLIT_MANIFEST_JSON}",
            "split_assignment_sha256": inputs["split_assignment_sha256"],
        },
        "class_map": inputs["class_map"],
        "pretrained_weights": weights,
        "train_images": inputs["train_images"],
        "train_annotations": inputs["train_annotations"],
        "validation_images": inputs["validation_images"],
        "validation_annotations": inputs["validation_annotations"],
        "test": {"status": TEST_PROTECTED, "reason": HOLDOUT_REASON},
        "epochs_configured": epochs_configured,
        "epochs_completed": epochs_completed,
        "best_epoch": best_epoch,
        "best_validation_fitness": (
            round(best_fitness, 6) if isinstance(best_fitness, float) else NOT_EXPOSED
        ),
        "early_stopped": early_stopped,
        "termination_mode": (
            "EARLY_STOPPED_ON_PATIENCE" if early_stopped else "COMPLETED_ALL_EPOCHS"
        ),
        "resumed": execution["resumed"],
        "resume_reason": execution.get("resume_reason"),
        "checkpoint_selection": protocol["checkpoint_selection"],
        "best_checkpoint": best,
        "last_checkpoint": last,
        "weights_committed": False,
        "declared_optimizer_policy": protocol["training"]["optimizer"],
        "resolved_optimizer": optimizer["optimizer"],
        "resolved_optimizer_determination": optimizer,
        "resolved_training_arguments": resolved_arguments,
        "validation_configuration": validation["effective_arguments"],
        "validation_metrics": validation["global"],
        "per_class_metrics": validation["per_class"],
        "metric_hierarchy": reference["baseline"].metrics.as_dict(),
        "phase7_comparison": {
            "primary_selection_metric": PRIMARY_SELECTION_METRIC,
            "supported_macro_map50_95": round(float(macro), 6),
            "supported_macro_exact": str(macro.normalize()),
            "selection_metric_classes": list(supported),
            "descriptive_classes": list(descriptive),
            "reference_supported_macro_map50_95": round(float(reference_macro), 6),
            "delta_supported_macro_vs_reference": round(float(delta_macro), 6),
            "practical_equivalence_margin": str(margin),
            "margin_status": margin_status,
            "delta_classification": classification,
            "official_all_class_metric": OFFICIAL_ALL_CLASS_METRIC,
            "deltas_vs_reference": deltas,
            "selection_pending": True,
            "selection_pending_reason": (
                "D2 has not been executed, so the frozen phase 7 A/B/C selection logic is not "
                "applied and no final detector is declared."
            ),
        },
        "class_support": [record.as_dict() for record in support],
        "rare_class": {
            "name": descriptive[0] if descriptive else None,
            "classes": list(descriptive),
            "warning": HIGH_SAMPLING_UNCERTAINTY,
            "classification": DESCRIPTIVE_HIGH_UNCERTAINTY,
            "limitation": (
                "Every class listed here failed the frozen support rule on the validation "
                "split, so its AP is reported in full but excluded from the selection metric. "
                "It must not be tuned for, must not decide between models, and must be quoted "
                "with an explicit small-sample caveat. The holdout remains protected and "
                "cannot be consulted to resolve the uncertainty."
            ),
        },
        "confusion_matrix": validation["confusion_matrix"],
        "confusion_matrix_axes": validation["confusion_matrix_axes"],
        "framework_validation_speed_ms_per_image": validation["speed"],
        "headline_metric_cross_check": cross_check,
        "model_complexity": complexity,
        "training_duration_seconds": execution["seconds"],
        "peak_gpu_memory": {
            "allocated_bytes": execution.get("peak_allocated_bytes", NOT_EXPOSED),
            "reserved_bytes": execution.get("peak_reserved_bytes", NOT_EXPOSED),
            "source": "TORCH_CUDA_MAX_MEMORY_ALLOCATED",
        },
        "memory_preflight": preflight,
        "runtime": runtime,
        "committed_figures": figures,
        "experiment_sha256": fingerprint,
    }

    problems = validate_result_manifest(
        {**manifest, "d0_experiment_sha256": fingerprint},
        class_names=reference["class_names"],
    )
    if problems:
        msg = f"the {experiment_id} manifest is not fit to publish: {'; '.join(problems)}"
        raise ExperimentRunError(msg)

    write_json(paths.reports / f"detection_{experiment_id}_manifest.json", manifest)

    report = build_report(manifest, matrix, declaration, reference)
    unsafe = scan_for_sensitive(report)
    if unsafe:
        msg = f"the {experiment_id} report is not fit to commit: {'; '.join(unsafe)}"
        raise ExperimentRunError(msg)
    (paths.reports / f"detection_{experiment_id}_report.md").write_text(
        report, encoding="utf-8", newline="\n"
    )

    update_results_artifact(paths, matrix, manifest, reference)
    write_run_provenance(paths, declaration, manifest, policy_sha256=policy_sha256)

    official = validation["global"].get(OFFICIAL_ALL_CLASS_METRIC)
    return [
        f"{PRIMARY_SELECTION_METRIC}  {round(float(macro), 6)}  "
        f"(reference {round(float(reference_macro), 6)}, delta {round(float(delta_macro), 6):+f})",
        f"margin status                {margin_status}",
        f"{OFFICIAL_ALL_CLASS_METRIC} official          {official}  "
        f"(delta {deltas[OFFICIAL_ALL_CLASS_METRIC]:+f})",
        f"precision / recall           {validation['global'].get('precision')} / "
        f"{validation['global'].get('recall')}",
        f"best epoch                   {best_epoch} of {epochs_completed} completed",
        f"optimizer                    {optimizer['optimizer']} via {optimizer['source']}",
        f"{experiment_id} experiment_sha256      {fingerprint}",
        "selection                    PENDING - D2 has not been executed",
        f"holdout                      {TEST_PROTECTED}",
    ]


def write_run_provenance(
    paths: ProjectPaths,
    declaration: ExperimentDeclaration,
    manifest: dict[str, Any],
    *,
    policy_sha256: str,
) -> Path:
    """Write the post-run provenance record.

    Args:
        paths: Project layout.
        declaration: The candidate declaration.
        manifest: The emitted result manifest.
        policy_sha256: Digest of the committed phase 7 policy.

    Returns:
        The record path.
    """
    experiment_id = declaration.experiment_id
    comparison = manifest["phase7_comparison"]
    record = ProvenanceRecord.create(
        "detection_experiment",
        phase=7,
        repo_root=paths.root,
        config={
            "detection_experiments": f"configs/{MATRIX_YAML}",
            "detection_baseline": f"configs/{BASELINE_YAML}",
            "experiment_id": experiment_id,
            "phase7_policy_sha256": policy_sha256,
            "resolved_config_sha256": manifest["resolved_config_sha256"],
        },
        details={
            "phase": "7B",
            "experiment_id": experiment_id,
            "intentional_variable": declaration.intentional_variable,
            "model": manifest["model"],
            "experiment_sha256": manifest["experiment_sha256"],
            "best_checkpoint_sha256": manifest["best_checkpoint"]["sha256"],
            "last_checkpoint_sha256": manifest["last_checkpoint"]["sha256"],
            "pretrained_weights_sha256": manifest["pretrained_weights"]["sha256"],
            "epochs_completed": manifest["epochs_completed"],
            "best_epoch": manifest["best_epoch"],
            "resolved_optimizer": manifest["resolved_optimizer"],
            "optimizer_evidence_source": manifest["resolved_optimizer_determination"]["source"],
            "validation_metrics": manifest["validation_metrics"],
            "supported_macro_map50_95": comparison["supported_macro_map50_95"],
            "delta_supported_macro_vs_reference": comparison["delta_supported_macro_vs_reference"],
            "margin_status": comparison["margin_status"],
            "selection_pending": True,
            "models_trained_in_this_phase": 1,
            "holdout_accessed": False,
            "runtime": manifest["runtime"],
        },
    )
    record.add_input(paths.configs / MATRIX_YAML, relative_to=paths.root)
    record.add_input(paths.reports / POLICY_JSON, relative_to=paths.root)
    record.add_input(paths.reports / ADAPTER_MANIFEST_JSON, relative_to=paths.root)
    record.add_output(
        paths.reports / f"detection_{experiment_id}_manifest.json", relative_to=paths.root
    )
    record.add_output(
        paths.reports / f"detection_{experiment_id}_report.md", relative_to=paths.root
    )
    destination = paths.reports / f"detection_{experiment_id}.provenance.json"
    write_json(destination, record.to_dict())
    return destination


def update_results_artifact(
    paths: ProjectPaths,
    matrix: ExperimentMatrix,
    manifest: dict[str, Any],
    reference: dict[str, Any],
) -> Path:
    """Assemble the Phase 7 experiment-results table across every declaration.

    A candidate with no result is recorded as declared with a ``null`` metrics
    block. No placeholder number is invented for an experiment that has not run,
    and the frozen selection logic is deliberately **not** applied while any
    declared candidate is still missing - choosing a winner from an incomplete
    matrix is exactly what the phase 7A policy forbids.

    Args:
        paths: Project layout.
        matrix: The parsed experiment matrix.
        manifest: The experiment manifest just written.
        reference: The verified reference facts.

    Returns:
        The artifact path.
    """
    comparison = manifest["phase7_comparison"]
    rows: list[dict[str, Any]] = [
        {
            "experiment_id": matrix.reference_experiment,
            "role": "REFERENCE_BASELINE",
            "status": "COMPLETE",
            "intentional_variable": "NONE_REFERENCE",
            "model": reference["manifest"]["model"],
            "imgsz": reference["manifest"]["resolved_training_arguments"]["imgsz"],
            "metrics": {
                PRIMARY_SELECTION_METRIC: round(float(reference["record"].supported_macro), 6),
                **dict(reference["manifest"]["validation_metrics"]),
            },
            "per_class_metrics": reference["manifest"]["per_class_metrics"],
            "delta_supported_macro_vs_reference": 0.0,
            "margin_status": "REFERENCE",
            "experiment_sha256": reference["record"].experiment_sha256,
        }
    ]
    for declaration in matrix.candidates:
        if declaration.experiment_id == manifest["experiment_id"]:
            rows.append(
                {
                    "experiment_id": declaration.experiment_id,
                    "role": declaration.role,
                    "status": "COMPLETE",
                    "intentional_variable": declaration.intentional_variable,
                    "model": manifest["model"],
                    "imgsz": manifest["resolved_training_arguments"]["imgsz"],
                    "metrics": {
                        PRIMARY_SELECTION_METRIC: comparison["supported_macro_map50_95"],
                        **dict(manifest["validation_metrics"]),
                    },
                    "per_class_metrics": manifest["per_class_metrics"],
                    "delta_supported_macro_vs_reference": comparison[
                        "delta_supported_macro_vs_reference"
                    ],
                    "margin_status": comparison["margin_status"],
                    "experiment_sha256": manifest["experiment_sha256"],
                }
            )
            continue
        existing = paths.reports / f"detection_{declaration.experiment_id}_manifest.json"
        if existing.is_file():
            other = read_json(existing)
            other_comparison = other.get("phase7_comparison", {})
            rows.append(
                {
                    "experiment_id": declaration.experiment_id,
                    "role": declaration.role,
                    "status": "COMPLETE",
                    "intentional_variable": declaration.intentional_variable,
                    "model": other.get("model"),
                    "imgsz": other.get("resolved_training_arguments", {}).get("imgsz"),
                    "metrics": {
                        PRIMARY_SELECTION_METRIC: other_comparison.get("supported_macro_map50_95"),
                        **dict(other.get("validation_metrics", {})),
                    },
                    "per_class_metrics": other.get("per_class_metrics"),
                    "delta_supported_macro_vs_reference": other_comparison.get(
                        "delta_supported_macro_vs_reference"
                    ),
                    "margin_status": other_comparison.get("margin_status"),
                    "experiment_sha256": other.get("experiment_sha256"),
                }
            )
            continue
        rows.append(
            {
                "experiment_id": declaration.experiment_id,
                "role": declaration.role,
                "status": declaration.status,
                "intentional_variable": declaration.intentional_variable,
                "model": None,
                "imgsz": None,
                "metrics": None,
                "per_class_metrics": None,
                "delta_supported_macro_vs_reference": None,
                "margin_status": None,
                "experiment_sha256": None,
            }
        )

    pending = [row["experiment_id"] for row in rows if row["metrics"] is None]
    payload = {
        "schema_version": 1,
        "phase": "7B",
        "task": "detection",
        "reference_experiment": matrix.reference_experiment,
        "primary_selection_metric": PRIMARY_SELECTION_METRIC,
        "official_all_class_metric": OFFICIAL_ALL_CLASS_METRIC,
        "practical_equivalence_margin": str(matrix.practical_equivalence_margin),
        "support_rule": matrix.support_rule.as_dict(),
        "selection_metric_classes": comparison["selection_metric_classes"],
        "descriptive_classes": comparison["descriptive_classes"],
        "experiments": rows,
        "pending_experiments": pending,
        "selection": {
            "status": "PENDING_INCOMPLETE_MATRIX",
            "case": None,
            "preferred_experiment": None,
            "reason": (
                "the frozen selection logic is applied only once every declared candidate has "
                f"a result. Still outstanding: {pending}. No case is assigned and no final "
                "detector is declared."
            ),
        },
        "metrics_are_validation_only": True,
        "test": {"status": TEST_PROTECTED, "reason": HOLDOUT_REASON},
    }
    destination = paths.reports / RESULTS_JSON
    write_json(destination, payload)
    return destination


if __name__ == "__main__":
    sys.exit(main())
