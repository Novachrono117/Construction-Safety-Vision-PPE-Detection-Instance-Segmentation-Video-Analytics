"""Freeze the detector-versus-segmenter comparison protocol.

Phase 10A. It runs no model, produces no prediction, measures no latency and
reads no image pixel. What it does is fix, in advance and in writing, what the
later comparison will measure and what it will not be allowed to claim.

Four things here are deliberate.

**Both models are verified by digest, never executed.** Their freeze accessors
resolve the checkpoints, confirm the bytes and confirm both take the same input
size. Nothing is loaded into a network.

**The benchmark subset is chosen before any timing exists, and without looking
at an image.** Validation ids are ranked by their own SHA-256 and truncated.
That is the whole rule, so the subset cannot have been picked for being easy,
crowded or visually interesting - and the ordering is fingerprinted, because
the order is part of the protocol.

**Membership is read from the canonical document, not restated.** The 65
validation images and their fingerprint come from the phase 5D COCO file that
the later comparison will score against.

**Nothing is decided.** The protocol carries no result, and the parser refuses
one.

Requires:

* ``configs/detector_segmenter_comparison.yaml``               phase 10A
* ``reports/final_detector_manifest.json``                     phase 7D
* ``reports/final_segmenter_manifest.json``                    phase 8G
* ``data/processed/canonical/annotations/detection_validation.coco.json``  5D

Writes:
    reports/detector_segmenter_comparison_protocol.json
    reports/detector_segmenter_comparison_protocol.md
    reports/detector_segmenter_comparison_membership.csv
    reports/detector_segmenter_latency_membership.csv
    reports/detector_segmenter_comparison.provenance.json

Usage:
    uv run python scripts/freeze_detector_segmenter_comparison.py
    uv run python scripts/freeze_detector_segmenter_comparison.py --verify-only
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from construction_safety_vision.config import ConfigError
from construction_safety_vision.data.canonical import scan_for_sensitive
from construction_safety_vision.detection_freeze import (
    FinalDetectorError,
    load_final_detector,
)
from construction_safety_vision.detection_freeze import (
    load_checkpoint_path as detector_checkpoint,
)
from construction_safety_vision.detector_segmenter_comparison import (
    ASSOCIATION_CATEGORIES,
    BENCHMARK_IMAGE_COUNT,
    CANONICAL_BOX_METRIC,
    CANONICAL_BOX_METRIC_50,
    COMPARISON_IMGSZ,
    DETECTOR_EXPERIMENT,
    END_TO_END_LATENCY,
    EVALUATION_SPLIT,
    HOLDOUT_STATUS,
    MODEL_INFERENCE_LATENCY,
    PROTOCOL_NAME,
    SEGMENTER_EXPERIMENT,
    SELECTION_RULE,
    SPATIAL_FEATURES,
    STATUS_FROZEN_NOT_EXECUTED,
    ComparisonProtocolError,
    load_comparison_protocol,
    membership_fingerprint,
    ordered_fingerprint,
    select_benchmark_images,
    stable_rank,
    validate_protocol_manifest,
)
from construction_safety_vision.paths import ProjectPaths, long_path
from construction_safety_vision.provenance import ProvenanceRecord, git_commit, sha256_file
from construction_safety_vision.segmentation_freeze import (
    FinalSegmenterError,
    load_final_segmenter,
)
from construction_safety_vision.segmentation_freeze import (
    load_checkpoint_path as segmenter_checkpoint,
)
from construction_safety_vision.splits import HOLDOUT_UNLOCK_ENV_VAR, holdout_unlocked

CONFIG_YAML = "detector_segmenter_comparison.yaml"

PROTOCOL_JSON = "detector_segmenter_comparison_protocol.json"
PROTOCOL_MD = "detector_segmenter_comparison_protocol.md"
MEMBERSHIP_CSV = "detector_segmenter_comparison_membership.csv"
LATENCY_CSV = "detector_segmenter_latency_membership.csv"
PROVENANCE_JSON = "detector_segmenter_comparison.provenance.json"

SCHEMA_VERSION = 1
PHASE = "10A"

FROZEN = "DETECTOR_SEGMENTER_COMPARISON_PROTOCOL_FROZEN"
MODEL_IDENTITY_MISMATCH = "MODEL_IDENTITY_MISMATCH"
INVALID_COMPARISON_PROTOCOL = "INVALID_COMPARISON_PROTOCOL"
BLOCKED = "BLOCKED"

HOLDOUT_REASON = (
    "Phase 10A froze a comparison protocol. It ran no model, produced no prediction, measured "
    "no latency and read no image pixel. The holdout was not read, materialised, adapted, "
    "counted, predicted on or inspected; no holdout identifier, image, prediction or statistic "
    "exists in any artifact this phase wrote."
)

HISTORICAL: tuple[str, ...] = (
    "reports/final_detector_manifest.json",
    "reports/detection_experiment_results.json",
    "reports/final_segmenter_manifest.json",
    "reports/segmentation_experiment_results.json",
    "reports/segmentation_selection_report.md",
    "reports/segmentation_experiment_comparison.csv",
    "reports/segmentation_S0_result_manifest.json",
    "reports/segmentation_S0_canonical_evaluation.json",
    "reports/segmentation_S0_mask_iou.json",
    "reports/segmentation_S1_result_manifest.json",
    "reports/segmentation_S1_canonical_evaluation.json",
    "reports/segmentation_S1_mask_iou.json",
    "reports/segmentation_comparison_policy.json",
    "configs/segmentation_baseline.yaml",
    "configs/segmentation_comparison.yaml",
    "configs/segmentation_canonical_evaluation.yaml",
    "configs/segmentation_mask_iou_evaluation.yaml",
    "reports/split_manifest.json",
    "reports/task_dataset_manifest.json",
)
"""Every artifact this phase must leave byte-identical."""


class FreezeError(RuntimeError):
    """Raised when a required input is missing or an invariant fails."""


class ModelIdentityError(FreezeError):
    """Raised when a frozen model is not the one the protocol names."""


# --- helpers ------------------------------------------------------------------


def read_json(path: Path) -> dict[str, Any]:
    """Read a committed JSON artifact.

    Args:
        path: File to read.

    Returns:
        The parsed mapping.

    Raises:
        FreezeError: If the file is missing or is not a JSON object.
    """
    if not path.is_file():
        msg = f"required input not found: {path.name}"
        raise FreezeError(msg)
    try:
        data = json.loads(Path(long_path(path)).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"{path.name} is not valid JSON ({exc})"
        raise FreezeError(msg) from exc
    if not isinstance(data, dict):
        msg = f"{path.name} must contain a JSON object"
        raise FreezeError(msg)
    return data


def write_json(path: Path, payload: Any) -> str:
    """Write a JSON artifact deterministically and return its digest.

    Args:
        path: Destination file.
        payload: JSON-serialisable content.

    Returns:
        The SHA-256 digest of the bytes written.
    """
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return sha256_file(path)


def historical_digests(paths: ProjectPaths) -> dict[str, str]:
    """Digest every artifact this phase must not change.

    Args:
        paths: Project layout.

    Returns:
        Digests keyed by repository-relative name.

    Raises:
        FreezeError: If one is absent.
    """
    digests: dict[str, str] = {}
    for name in HISTORICAL:
        path = paths.root / name
        if not path.is_file():
            msg = f"historical artifact missing: {name}"
            raise FreezeError(msg)
        digests[name] = sha256_file(path)
    return digests


# --- the two frozen models ------------------------------------------------------


def verify_models(paths: ProjectPaths, protocol: Any) -> dict[str, Any]:
    """Verify both frozen models by digest, without executing either.

    Args:
        paths: Project layout.
        protocol: The parsed comparison protocol.

    Returns:
        Both models' verified identities.

    Raises:
        ModelIdentityError: If either model is not the frozen one, or the two
            do not share the comparison input size.
        FreezeError: If either freeze manifest is unreadable.
    """
    try:
        detector = load_final_detector(paths.reports)
        segmenter = load_final_segmenter(paths.reports)
    except (FinalDetectorError, FinalSegmenterError) as exc:
        raise FreezeError(str(exc)) from exc

    for name, model in (("detector", detector), ("segmenter", segmenter)):
        if model.recompute_fingerprint() != model.fingerprint:
            msg = f"the frozen {name} manifest does not recompute"
            raise ModelIdentityError(msg)

    if detector.selected_experiment != DETECTOR_EXPERIMENT:
        msg = f"the frozen detector is {detector.selected_experiment}, not {DETECTOR_EXPERIMENT}"
        raise ModelIdentityError(msg)
    if segmenter.selected_experiment != SEGMENTER_EXPERIMENT:
        msg = f"the frozen segmenter is {segmenter.selected_experiment}, not {SEGMENTER_EXPERIMENT}"
        raise ModelIdentityError(msg)

    declared_detector = protocol["detector"]["checkpoint_sha256"]
    declared_segmenter = protocol["segmenter"]["checkpoint_sha256"]
    if detector.checkpoint_sha256 != declared_detector:
        msg = (
            f"the protocol names detector checkpoint {declared_detector}, the freeze records "
            f"{detector.checkpoint_sha256}"
        )
        raise ModelIdentityError(msg)
    if segmenter.checkpoint_sha256 != declared_segmenter:
        msg = (
            f"the protocol names segmenter checkpoint {declared_segmenter}, the freeze records "
            f"{segmenter.checkpoint_sha256}"
        )
        raise ModelIdentityError(msg)

    if detector.imgsz != segmenter.imgsz != COMPARISON_IMGSZ:
        msg = (
            f"the two frozen models do not share the comparison input size: detector "
            f"{detector.imgsz}, segmenter {segmenter.imgsz}, protocol {COMPARISON_IMGSZ}"
        )
        raise ModelIdentityError(msg)

    # Resolve the checkpoints so a later phase is not surprised by a missing
    # binary. This verifies bytes; it loads nothing into a model.
    resolved: dict[str, str] = {}
    for name, model, resolver in (
        ("detector", detector, detector_checkpoint),
        ("segmenter", segmenter, segmenter_checkpoint),
    ):
        try:
            path = resolver(model, paths.root)
        except Exception as exc:  # missing or mismatched bytes stop the phase
            msg = f"the frozen {name} checkpoint could not be resolved: {exc}"
            raise ModelIdentityError(msg) from exc
        resolved[name] = path.relative_to(paths.root).as_posix()

    return {
        "detector": {
            "experiment": detector.selected_experiment,
            "role": protocol["detector"]["role"],
            "model": detector.model,
            "imgsz": detector.imgsz,
            "checkpoint_sha256": detector.checkpoint_sha256,
            "checkpoint_size_bytes": detector.checkpoint_size_bytes,
            "resolved_path": resolved["detector"],
            "manifest": "reports/final_detector_manifest.json",
            "manifest_sha256": sha256_file(paths.reports / "final_detector_manifest.json"),
            "identity_fingerprint": detector.fingerprint,
            "outputs": list(protocol["detector"]["outputs"]),
            "executed_in_this_phase": False,
        },
        "segmenter": {
            "experiment": segmenter.selected_experiment,
            "role": protocol["segmenter"]["role"],
            "model": segmenter.model,
            "imgsz": segmenter.imgsz,
            "batch_trained_at": segmenter.batch,
            "mask_ratio": segmenter.mask_ratio,
            "overlap_mask": segmenter.overlap_mask,
            "checkpoint_sha256": segmenter.checkpoint_sha256,
            "checkpoint_size_bytes": segmenter.checkpoint_size_bytes,
            "resolved_path": resolved["segmenter"],
            "manifest": "reports/final_segmenter_manifest.json",
            "manifest_sha256": sha256_file(paths.reports / "final_segmenter_manifest.json"),
            "identity_fingerprint": segmenter.fingerprint,
            "final_segmenter_sha256": segmenter.fingerprint,
            "outputs": list(protocol["segmenter"]["outputs"]),
            "executed_in_this_phase": False,
        },
    }


# --- the comparison population ----------------------------------------------------


def load_population(paths: ProjectPaths, protocol: Any) -> dict[str, Any]:
    """Read the validation membership from the canonical document.

    Args:
        paths: Project layout.
        protocol: The parsed comparison protocol.

    Returns:
        The membership, its fingerprints and the benchmark subset.

    Raises:
        FreezeError: If the document is missing, its digest has moved, or the
            declared membership fingerprint disagrees with the file.
    """
    declared = protocol["population"]
    document = paths.root / str(declared["source"])
    if not document.is_file():
        msg = f"the canonical validation document is not on this machine: {declared['source']}"
        raise FreezeError(msg)
    digest = sha256_file(document)
    if digest != declared["source_sha256"]:
        msg = (
            f"the canonical validation document has digest {digest}, but the protocol names "
            f"{declared['source_sha256']}"
        )
        raise FreezeError(msg)

    payload = json.loads(Path(long_path(document)).read_text(encoding="utf-8"))
    image_ids = sorted(Path(str(entry["file_name"])).stem for entry in payload["images"])
    if len(image_ids) != int(declared["images"]):
        msg = (
            f"the canonical validation document holds {len(image_ids)} images, the protocol "
            f"declares {declared['images']}"
        )
        raise FreezeError(msg)
    if len(payload["annotations"]) != int(declared["annotations"]):
        msg = (
            f"the canonical validation document holds {len(payload['annotations'])} "
            f"annotations, the protocol declares {declared['annotations']}"
        )
        raise FreezeError(msg)

    fingerprint = membership_fingerprint(image_ids)
    if fingerprint != declared["membership_sha256"]:
        msg = (
            f"the validation membership fingerprints to {fingerprint}, but the protocol names "
            f"{declared['membership_sha256']}"
        )
        raise FreezeError(msg)

    benchmark = select_benchmark_images(image_ids, BENCHMARK_IMAGE_COUNT)
    benchmark_digest = ordered_fingerprint(benchmark)
    declared_benchmark = protocol.latency["benchmark_membership_sha256"]
    if benchmark_digest != declared_benchmark:
        msg = (
            f"the benchmark subset fingerprints to {benchmark_digest}, but the protocol names "
            f"{declared_benchmark}"
        )
        raise FreezeError(msg)

    return {
        "image_ids": image_ids,
        "membership_sha256": fingerprint,
        "images": len(image_ids),
        "annotations": len(payload["annotations"]),
        "document": str(declared["source"]),
        "document_sha256": digest,
        "benchmark_ids": list(benchmark),
        "benchmark_membership_sha256": benchmark_digest,
        "benchmark_set_sha256": membership_fingerprint(benchmark),
    }


def build_membership_csv(image_ids: Sequence[str]) -> str:
    """Render the full validation membership, with its rank key.

    Args:
        image_ids: The validation image ids, sorted.

    Returns:
        CSV text with a trailing newline.
    """
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer, fieldnames=["source_image_id", "split", "stable_rank_sha256"], lineterminator="\n"
    )
    writer.writeheader()
    for image_id in image_ids:
        writer.writerow(
            {
                "source_image_id": image_id,
                "split": EVALUATION_SPLIT,
                "stable_rank_sha256": stable_rank(image_id),
            }
        )
    return buffer.getvalue()


def build_latency_csv(benchmark_ids: Sequence[str]) -> str:
    """Render the latency benchmark subset in its frozen order.

    Args:
        benchmark_ids: The selected ids, in benchmark order.

    Returns:
        CSV text with a trailing newline.
    """
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=["benchmark_order", "source_image_id", "split", "stable_rank_sha256"],
        lineterminator="\n",
    )
    writer.writeheader()
    for position, image_id in enumerate(benchmark_ids):
        writer.writerow(
            {
                "benchmark_order": position,
                "source_image_id": image_id,
                "split": EVALUATION_SPLIT,
                "stable_rank_sha256": stable_rank(image_id),
            }
        )
    return buffer.getvalue()


# --- artifacts ------------------------------------------------------------------


def build_manifest(
    *,
    protocol: Any,
    models: Mapping[str, Any],
    population: Mapping[str, Any],
    historical: Mapping[str, str],
    config_sha256: str,
) -> dict[str, Any]:
    """Assemble the frozen comparison-protocol manifest.

    Args:
        protocol: The parsed comparison protocol.
        models: Both models' verified identities.
        population: The verified comparison population.
        historical: Digests of every artifact this phase must not change.
        config_sha256: Digest of the committed configuration file.

    Returns:
        The manifest. It carries no result, by construction.
    """
    latency = protocol.latency
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "protocol": PROTOCOL_NAME,
        "status": STATUS_FROZEN_NOT_EXECUTED,
        "protocol_config": f"configs/{CONFIG_YAML}",
        "protocol_config_sha256": config_sha256,
        "protocol_fingerprint": protocol.fingerprint(),
        "scientific_question": protocol["scientific_question"].strip(),
        "comparison_axes": dict(protocol["comparison_axes"]),
        "detector": dict(models["detector"]),
        "segmenter": dict(models["segmenter"]),
        "comparison_input_imgsz": COMPARISON_IMGSZ,
        "models_are_not_solving_the_same_task": (
            "The detector emits a class, a confidence and a box. The segmenter emits those plus "
            "an instance mask. The comparison asks what the mask adds and what it costs; it "
            "does not rank the two, and no aggregate score is defined."
        ),
        "population": {
            "split": EVALUATION_SPLIT,
            "images": population["images"],
            "annotations": population["annotations"],
            "membership_sha256": population["membership_sha256"],
            "membership_artifact": f"reports/{MEMBERSHIP_CSV}",
            "document": population["document"],
            "document_sha256": population["document_sha256"],
            "identical_images_for_both_models": True,
            "identical_order_for_both_models": True,
            "resplit": False,
            "split_assignment_sha256": protocol["population"]["split_assignment_sha256"],
            "class_map_sha256": protocol["population"]["class_map_sha256"],
        },
        "ap_inference": protocol.ap_inference,
        "operational_inference": protocol.operational_inference,
        "threshold_policy": protocol["threshold_policy"].strip(),
        "canonical_box_evaluator": dict(protocol["canonical_box_evaluator"]),
        "box_comparability": dict(protocol["box_comparability"]),
        "spatial_features": dict(protocol["spatial_features"]),
        "spatial_feature_status": protocol["spatial_feature_status"].strip(),
        "box_proxies": dict(protocol["box_proxies"]),
        "box_proxy_note": protocol["box_proxy_note"].strip(),
        "spatial_value_metrics": list(protocol["spatial_value_metrics"]),
        "association": dict(protocol["association"]),
        "latency_protocol": {
            **latency,
            "benchmark_membership_artifact": f"reports/{LATENCY_CSV}",
            "benchmark_membership_sha256": population["benchmark_membership_sha256"],
            "benchmark_set_sha256": population["benchmark_set_sha256"],
            "benchmark_image_count": len(population["benchmark_ids"]),
        },
        "memory_protocol": dict(protocol["memory_protocol"]),
        "model_complexity": dict(protocol["model_complexity"]),
        "reporting": dict(protocol["reporting"]),
        "interpretation": dict(protocol["interpretation"]),
        "limitations": list(protocol["limitations"]),
        "historical_artifact_digests": dict(historical),
        "historical_artifacts_unchanged": True,
        "models_executed_in_this_phase": 0,
        "predictions_produced": 0,
        "latency_measurements_taken": 0,
        "memory_measurements_taken": 0,
        "images_read": 0,
        "thresholds_tuned": 0,
        "models_trained_in_this_phase": 0,
        "detector_touched": False,
        "segmenter_touched": False,
        "next_phase": "PHASE_10B_CONTROLLED_VALIDATION_RECOGNITION_AND_SPATIAL_COMPARISON",
        "test": {"status": HOLDOUT_STATUS, "reason": HOLDOUT_REASON},
        "holdout_accessed": False,
    }


def build_report(manifest: Mapping[str, Any], *, commit: str | None) -> str:
    """Render the protocol report.

    Args:
        manifest: The frozen protocol manifest.
        commit: Repository commit, when available.

    Returns:
        The report text.
    """
    detector = manifest["detector"]
    segmenter = manifest["segmenter"]
    population = manifest["population"]
    ap = manifest["ap_inference"]
    operational = manifest["operational_inference"]
    evaluator = manifest["canonical_box_evaluator"]
    comparability = manifest["box_comparability"]
    association = manifest["association"]
    latency = manifest["latency_protocol"]
    memory = manifest["memory_protocol"]

    lines: list[str] = []
    add = lines.append

    add("# Detector-versus-segmenter comparison protocol")
    add("")
    add(
        f"Phase {manifest['phase']} · status `{manifest['status']}` · protocol fingerprint "
        f"`{manifest['protocol_fingerprint']}`"
    )
    add("")
    add(
        "**This document contains no results.** It is written before any comparison runs, "
        "which is the only thing that makes it a protocol. Both models were frozen first - the "
        "detector in phase 7D, the segmenter in phase 8G - so nothing here could have been "
        "chosen to flatter either of them."
    )
    add("")
    if commit:
        add(f"Repository commit at production time: `{commit}`.")
        add("")

    add("## 1. Scientific question")
    add("")
    add(f"> {manifest['scientific_question']}")
    add("")
    add(f"`PREDECLARED_PROTOCOL`. {manifest['models_are_not_solving_the_same_task']}")
    add("")

    add("## 2. Frozen model identities")
    add("")
    add("| | Detector | Segmenter |")
    add("| --- | --- | --- |")
    add(f"| Experiment | **{detector['experiment']}** | **{segmenter['experiment']}** |")
    add(f"| Role | `{detector['role']}` | `{segmenter['role']}` |")
    add(f"| Model | {detector['model']} | {segmenter['model']} |")
    add(f"| imgsz | {detector['imgsz']} | {segmenter['imgsz']} |")
    add(
        f"| Checkpoint SHA-256 | `{detector['checkpoint_sha256']}` | "
        f"`{segmenter['checkpoint_sha256']}` |"
    )
    add(f"| Bytes | {detector['checkpoint_size_bytes']} | {segmenter['checkpoint_size_bytes']} |")
    add(f"| Outputs | {', '.join(detector['outputs'])} | {', '.join(segmenter['outputs'])} |")
    add("")
    add(
        f"`FROZEN_MODEL`. Both were verified by digest through their freeze accessors and "
        f"**neither was executed** (`executed_in_this_phase: "
        f"{detector['executed_in_this_phase']}`). The segmenter additionally carries "
        f"`overlap_mask: {segmenter['overlap_mask']}` and `mask_ratio: "
        f"{segmenter['mask_ratio']}`, which are part of its identity."
    )
    add("")

    add("## 3. Comparison scope")
    add("")
    for axis, description in sorted(manifest["comparison_axes"].items()):
        add(f"* **{axis}** - {description.strip()}")
    add("")

    add("## 4. Validation population")
    add("")
    add(
        f"`{population['split']}` only: **{population['images']} images**, "
        f"{population['annotations']} annotations, from the frozen phase 5C.2 split "
        f"(`{population['split_assignment_sha256']}`). Membership fingerprint "
        f"`{population['membership_sha256']}`, committed in "
        f"`{population['membership_artifact']}`."
    )
    add("")
    add(
        f"Both models see the same images in the same order "
        f"(`identical_images_for_both_models: {population['identical_images_for_both_models']}`) "
        f"and nothing is re-split (`resplit: {population['resplit']}`)."
    )
    add("")

    add("## 5. Holdout policy")
    add("")
    add(f"`HOLDOUT_POLICY` `{manifest['test']['status']}`. {manifest['test']['reason']}")
    add("")

    add("## 6-8. Recognition comparison and the AP protocol")
    add("")
    add(
        "`CANONICAL_EVALUATION`. Both models' predicted boxes are scored against the **same** "
        "canonical ground truth by the **same** external evaluator."
    )
    add("")
    add(f"* Implementation: `{evaluator['implementation']}`, `iouType='{evaluator['iou_type']}'`")
    add(f"* IoU thresholds: {evaluator['iou_thresholds']}")
    add(f"* `maxDets`: {evaluator['max_dets']}")
    add(f"* Ground truth: `{evaluator['ground_truth']}` - `{evaluator['ground_truth_document']}`")
    add(f"* Category ids: `{evaluator['category_id_mapping']}`")
    add("")
    add(f"{evaluator['why_not_native']}")
    add("")
    add(
        f"AP inference, for both models: imgsz {ap['imgsz']}, conf **{ap['conf']}**, NMS IoU "
        f"{ap['iou']}, max_det {ap['max_det']}, augment {ap['augment']}, TTA {ap['tta']}, "
        f"precision **{ap['precision']}** (`{ap['precision_argument']}: "
        f"{ap['precision_value']}`)."
    )
    add("")
    add(f"`LIMITATION`. {evaluator['prediction_cap_versus_metric_cap']}")
    add("")
    add("**Box comparability.** " + comparability["reason"].strip())
    add("")

    add("## 9. Operational inference protocol")
    add("")
    add(
        f"`OPERATIONAL_ANALYSIS`. imgsz {operational['imgsz']}, conf "
        f"**{operational['conf']}**, NMS IoU {operational['iou']}, max_det "
        f"{operational['max_det']}, augment {operational['augment']}, precision "
        f"**{operational['precision']}**."
    )
    add("")
    add(f"{manifest['threshold_policy']}")
    add("")

    add("## 10-11. Why masks add representation, and what they measure")
    add("")
    add(
        "`SPATIAL_INFORMATION`. A box asserts an axis-aligned rectangle; a mask asserts which "
        "pixels belong to the instance. The quantities below are what that difference makes "
        "computable."
    )
    add("")
    add("| Quantity | Definition |")
    add("| --- | --- |")
    for name in SPATIAL_FEATURES:
        add(f"| `{name}` | {manifest['spatial_features'][name].strip()} |")
    add("")
    add(f"`LIMITATION` `{manifest['spatial_feature_status']}`")
    add("")

    add("## 12. Box-only proxies")
    add("")
    add(
        "`BOX_PROXY`. Each mask quantity is paired with what a box-only pipeline could compute "
        "instead, or declared to have no equivalent. That pairing is the answer to the "
        "question this comparison asks."
    )
    add("")
    add("| Mask measurement | Box proxy |")
    add("| --- | --- |")
    for name in SPATIAL_FEATURES:
        add(f"| `{name}` | `{manifest['box_proxies'][name]}` |")
    add("")
    add(f"{manifest['box_proxy_note']}")
    add("")

    add("## 13. Person-PPE association")
    add("")
    add(
        f"`{association['label']}` · `{association['status']}`. Relationships: "
        + ", ".join(f"`{name}`" for name in association["relationships"])
        + f", evaluated at the operational threshold with a containment floor of "
        f"{association['containment_floor']}."
    )
    add("")
    add(f"* Mask rule: {association['mask_rule'].strip()}")
    add(f"* Box rule: {association['box_rule'].strip()}")
    add(
        f"* Tie-breaking: `{association['tie_breaking']}`, `deterministic: "
        f"{association['deterministic']}`"
    )
    add("")
    add(
        f"No appearance embeddings (`{association['appearance_embeddings']}`), no tracking "
        f"(`{association['tracking']}`), no learned association "
        f"(`{association['learned_association']}`), and the five frozen classes are not "
        f"collapsed (`classes_collapsed: {association['classes_collapsed']}`)."
    )
    add("")
    add(f"`LIMITATION`. {association['semantics'].strip()}")
    add("")

    add("## 14. Association disagreement categories")
    add("")
    for category in ASSOCIATION_CATEGORIES:
        add(f"* `{category}`")
    add("")
    add(
        "Each deterministic person-PPE candidate pair falls in exactly one. This is a "
        "descriptive comparison of two geometric rules, **not** a model accuracy metric: there "
        "is no association ground truth to be accurate against."
    )
    add("")

    add("## 15-18. Latency benchmark")
    add("")
    add(
        f"`CONTROLLED_HARDWARE_BENCHMARK` `{latency['label']}`. batch {latency['batch']}, imgsz "
        f"{latency['imgsz']}, precision **{latency['precision']}** "
        f"(`{latency['precision_argument']}: {latency['precision_value']}`), warmup "
        f"**{latency['warmup_iterations']}** iterations discarded, "
        f"**{latency['timed_iterations_per_image']}** timed repetitions per image over "
        f"**{latency['benchmark_image_count']}** images."
    )
    add("")
    add("### Timing boundaries")
    add("")
    for name in (MODEL_INFERENCE_LATENCY, END_TO_END_LATENCY):
        boundary = latency["timing_boundaries"][name]
        add(f"**`{name}`** - {boundary['note'].strip()}")
        add("")
        add(f"* includes: {', '.join(boundary['includes'])}")
        if "excludes" in boundary:
            add(f"* excludes: {', '.join(boundary['excludes'])}")
        add("")
    included = latency["timing_boundaries"][END_TO_END_LATENCY][
        "segmenter_mask_reconstruction_included"
    ]
    add(
        "The distinction matters because segmentation adds postprocessing, not just a heavier "
        "forward pass. Mask reconstruction is **inside** the segmenter's end-to-end boundary "
        f"(`segmenter_mask_reconstruction_included: {included}`); "
        "putting it outside would hide the cost this comparison exists to quantify."
    )
    add("")
    add("### GPU synchronization")
    add("")
    add(
        f"{latency['synchronization']['note'].strip()} Timing primitive: "
        f"`{latency['timing_primitive']}`, the same for both models."
    )
    add("")
    add("### Statistics")
    add("")
    add("Reported for each boundary and each model: " + ", ".join(latency["statistics"]) + ".")
    add("")
    for name, definition in sorted(latency["derived_quantities"].items()):
        add(f"* `{name}` - {definition}")
    add("")

    add("## 19. Deterministic benchmark membership")
    add("")
    add(
        f"`{latency['benchmark_selection_rule']}`. {latency['benchmark_selection_note'].strip()} "
        f"Ordered fingerprint `{latency['benchmark_membership_sha256']}`, committed in "
        f"`{latency['benchmark_membership_artifact']}`."
    )
    add("")

    add("## 20. Execution-order control")
    add("")
    add(f"{latency['execution_order']['design'].strip()}")
    add("")
    add(
        f"`interleaved: {latency['execution_order']['interleaved']}`, `symmetric: "
        f"{latency['execution_order']['symmetric']}`, `randomized: "
        f"{latency['execution_order']['randomized']}`. The runner aborts if either model "
        f"deviates from the frozen invariants (`abort_on_deviation: "
        f"{latency['abort_on_deviation']}`):"
    )
    add("")
    for invariant in latency["fairness_invariants"]:
        add(f"* {invariant}")
    add("")

    add("## 21. Memory measurement")
    add("")
    add(
        f"batch {memory['batch']}, imgsz {memory['imgsz']}, peak stats reset **after** warmup "
        f"(`reset_peak_stats_after_warmup: {memory['reset_peak_stats_after_warmup']}`). "
        f"Recorded: {', '.join(memory['measured'])}. "
        f"{memory['note'].strip()}"
    )
    add("")

    add("## 22. Model complexity")
    add("")
    add(
        f"`{manifest['model_complexity']['status']}`. Fields: "
        + ", ".join(manifest["model_complexity"]["fields"])
        + f". {manifest['model_complexity']['note'].strip()}"
    )
    add("")

    add("## 23. Reporting plan")
    add("")
    add(f"* Box metrics: `{CANONICAL_BOX_METRIC}`, `{CANONICAL_BOX_METRIC_50}`")
    add(f"* Per-class box AP: {manifest['reporting']['per_class_box_ap']}")
    add(f"* Deltas reported: {manifest['reporting']['report_deltas']}")
    add(f"* Aggregate benefit score: **{manifest['reporting']['aggregate_benefit_score']}**")
    add(f"* Winner declared: **{manifest['reporting']['winner_declared']}**")
    add("")
    add(f"{manifest['reporting']['note'].strip()}")
    add("")
    add("Spatial-value comparisons frozen in advance:")
    add("")
    for metric in manifest["spatial_value_metrics"]:
        add(f"* `{metric}`")
    add("")

    add("## 24. Interpretation boundaries")
    add("")
    for key in ("BENEFIT", "COST", "RECOGNITION_TRADEOFF", "OPERATIONAL_VALUE"):
        add(f"* **{key}** - {str(manifest['interpretation'][key]).strip()}")
    add("")
    add(
        f"`single_aggregate_score: {manifest['interpretation']['single_aggregate_score']}`, "
        f"`weighted_cost_benefit_index: "
        f"{manifest['interpretation']['weighted_cost_benefit_index']}`. Benefit and cost are "
        "reported side by side and never collapsed."
    )
    add("")

    add("## 25. Limitations")
    add("")
    add("`LIMITATION`.")
    add("")
    for item in manifest["limitations"]:
        add(f"* {item.strip()}")
    add("")

    add("## 26. Next phase")
    add("")
    add(
        f"`{manifest['next_phase']}`. Nothing has been executed here: "
        f"`models_executed_in_this_phase: {manifest['models_executed_in_this_phase']}`, "
        f"`predictions_produced: {manifest['predictions_produced']}`, "
        f"`latency_measurements_taken: {manifest['latency_measurements_taken']}`, "
        f"`images_read: {manifest['images_read']}`."
    )
    add("")

    return "\n".join(lines) + "\n"


# --- entry point ----------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Freeze the detector-versus-segmenter comparison protocol.

    Args:
        argv: Command-line arguments.

    Returns:
        Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="run every precondition check without writing",
    )
    args = parser.parse_args(argv)

    paths = ProjectPaths.from_root()
    if holdout_unlocked():
        print(
            f"REFUSED: {HOLDOUT_UNLOCK_ENV_VAR} is set. This phase reads committed development "
            "artifacts only and declines to run in an environment where the holdout is "
            "unlocked.",
            file=sys.stderr,
        )
        return 2

    try:
        historical = historical_digests(paths)
        protocol = load_comparison_protocol(paths.configs / CONFIG_YAML)
        config_sha256 = sha256_file(paths.configs / CONFIG_YAML)
        models = verify_models(paths, protocol)
        population = load_population(paths, protocol)
    except (ConfigError, ComparisonProtocolError, FreezeError) as exc:
        if isinstance(exc, ModelIdentityError):
            classification = MODEL_IDENTITY_MISMATCH
        elif isinstance(exc, ComparisonProtocolError):
            classification = INVALID_COMPARISON_PROTOCOL
        else:
            classification = BLOCKED
        print(f"{classification}: {exc}", file=sys.stderr)
        return 2

    print(f"protocol   {protocol.fingerprint()}  {STATUS_FROZEN_NOT_EXECUTED}")
    print(
        f"detector   {models['detector']['experiment']}  {models['detector']['model']}  imgsz "
        f"{models['detector']['imgsz']}  VERIFIED (not executed)"
    )
    print(
        f"segmenter  {models['segmenter']['experiment']}  {models['segmenter']['model']}  imgsz "
        f"{models['segmenter']['imgsz']}  overlap_mask "
        f"{models['segmenter']['overlap_mask']}  VERIFIED (not executed)"
    )
    print(
        f"population {population['images']} validation images  "
        f"{population['annotations']} annotations  {population['membership_sha256'][:16]}..."
    )
    print(
        f"benchmark  {len(population['benchmark_ids'])} images  "
        f"{population['benchmark_membership_sha256'][:16]}...  {SELECTION_RULE}"
    )
    print(
        f"thresholds AP conf {protocol.ap_inference['conf']}  operational conf "
        f"{protocol.operational_inference['conf']}  (never mixed)"
    )

    if args.verify_only:
        print("VERIFIED: preconditions hold. Nothing executed or written.")
        print(f"holdout    {HOLDOUT_STATUS}")
        return 0

    manifest = build_manifest(
        protocol=protocol,
        models=models,
        population=population,
        historical=historical,
        config_sha256=config_sha256,
    )

    problems = validate_protocol_manifest(
        manifest,
        protocol_fingerprint=protocol.fingerprint(),
        detector_sha256=models["detector"]["checkpoint_sha256"],
        segmenter_sha256=models["segmenter"]["checkpoint_sha256"],
        membership_sha256=population["membership_sha256"],
        benchmark_sha256=population["benchmark_membership_sha256"],
    )
    if problems:
        print(f"{INVALID_COMPARISON_PROTOCOL}: the manifest does not validate:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 2

    report = build_report(manifest, commit=git_commit(paths.root))
    membership_csv = build_membership_csv(population["image_ids"])
    latency_csv = build_latency_csv(population["benchmark_ids"])

    findings = (
        scan_for_sensitive(report)
        + scan_for_sensitive(membership_csv)
        + scan_for_sensitive(latency_csv)
        + scan_for_sensitive(json.dumps(manifest))
    )
    if findings:
        print(f"{BLOCKED}: sensitive content in the emitted artifacts: {findings}", file=sys.stderr)
        return 2

    manifest_sha256 = write_json(paths.reports / PROTOCOL_JSON, manifest)
    (paths.reports / PROTOCOL_MD).write_text(report, encoding="utf-8", newline="\n")
    (paths.reports / MEMBERSHIP_CSV).write_text(membership_csv, encoding="utf-8", newline="\n")
    (paths.reports / LATENCY_CSV).write_text(latency_csv, encoding="utf-8", newline="\n")

    after = historical_digests(paths)
    changed = sorted(name for name in historical if historical[name] != after.get(name))
    if changed:
        print(f"{BLOCKED}: historical artifacts changed: {changed}", file=sys.stderr)
        return 2

    record = ProvenanceRecord.create(
        name="detector_segmenter_comparison_protocol",
        phase=10,
        config={
            "detector_segmenter_comparison": f"configs/{CONFIG_YAML}",
            "protocol_fingerprint": protocol.fingerprint(),
        },
        details={
            "phase": PHASE,
            "classification": FROZEN,
            "status": STATUS_FROZEN_NOT_EXECUTED,
            "detector_experiment": models["detector"]["experiment"],
            "detector_checkpoint_sha256": models["detector"]["checkpoint_sha256"],
            "segmenter_experiment": models["segmenter"]["experiment"],
            "segmenter_checkpoint_sha256": models["segmenter"]["checkpoint_sha256"],
            "final_segmenter_sha256": models["segmenter"]["final_segmenter_sha256"],
            "comparison_input_imgsz": COMPARISON_IMGSZ,
            "population_split": EVALUATION_SPLIT,
            "population_images": population["images"],
            "membership_sha256": population["membership_sha256"],
            "benchmark_membership_sha256": population["benchmark_membership_sha256"],
            "ap_confidence": protocol.ap_inference["conf"],
            "operational_confidence": protocol.operational_inference["conf"],
            "models_executed_in_this_phase": 0,
            "predictions_produced": 0,
            "latency_measurements_taken": 0,
            "thresholds_tuned": 0,
            "detector_touched": False,
            "segmenter_touched": False,
            "holdout_accessed": False,
            "historical_artifacts_unchanged": True,
        },
        repo_root=paths.root,
    )
    record.add_input(paths.configs / CONFIG_YAML, relative_to=paths.root)
    for name in ("final_detector_manifest.json", "final_segmenter_manifest.json"):
        record.add_input(paths.reports / name, relative_to=paths.root)
    for name in (PROTOCOL_JSON, PROTOCOL_MD, MEMBERSHIP_CSV, LATENCY_CSV):
        record.add_output(paths.reports / name, relative_to=paths.root)
    record.write_json(paths.reports / PROVENANCE_JSON)

    print(FROZEN)
    print(f"manifest   reports/{PROTOCOL_JSON}  sha256 {manifest_sha256}")
    print(f"report     reports/{PROTOCOL_MD}")
    print(f"membership reports/{MEMBERSHIP_CSV}  ({population['images']} validation images)")
    print(f"benchmark  reports/{LATENCY_CSV}  ({len(population['benchmark_ids'])} images)")
    print("executed   0 models, 0 predictions, 0 latency measurements")
    print(f"holdout    {HOLDOUT_STATUS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
