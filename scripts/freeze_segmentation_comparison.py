"""Freeze the canonical S0-versus-S1 comparison protocol and evaluate S0 once under it.

Phase 8E. It trains nothing, retrains nothing, tunes nothing and selects no
segmenter. It does four things, in this order and for a reason:

* freezes a common canonical evaluator, so two experiments trained against
  different targets can be compared on ground truth neither of them can move;
* evaluates the frozen S0 checkpoint under it **exactly once**, producing the
  reference the future candidate will be measured against;
* freezes S1 as a one-variable ``overlap_mask`` intervention, before it exists;
* proves the frozen batch still fits with ``overlap_mask: false``, without
  training.

The timing is the awkward part and is recorded rather than smoothed over. This
protocol was frozen **after S0 ran** - the need for it was discovered by phase
8D, not anticipated before phase 8C - and **before S1 exists**, which is what
still makes it a protocol rather than a description. S0's published native and
direct-IoU results are not withdrawn, revised or regenerated; a third number is
added beside them.

Requires:

* ``configs/segmentation_canonical_evaluation.yaml``     phase 8E
* ``configs/segmentation_comparison.yaml``               phase 8E
* ``configs/segmentation_baseline.yaml``                 phase 8B
* ``reports/segmentation_S0_result_manifest.json``       phase 8C
* ``reports/segmentation_S0_mask_iou.json``              phase 8C
* ``data/processed/canonical/annotations/``              phase 5D

Writes:
    reports/segmentation_S0_canonical_evaluation.json
    reports/segmentation_canonical_comparison_reference.md
    reports/segmentation_comparison_policy.json
    reports/segmentation_comparison_policy.md
    reports/segmentation_S1_protocol.md
    reports/segmentation_S1_protocol_manifest.json
    reports/segmentation_comparison.provenance.json

Usage:
    uv run python scripts/freeze_segmentation_comparison.py --verify-only
    uv run python scripts/freeze_segmentation_comparison.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from construction_safety_vision.canonical_evaluation import (
    ALL_CLASS_METRIC,
    ALL_CLASS_METRIC_50,
    DISAGREEMENT,
    GROUND_TRUTH_SOURCE,
    MAX_DETS,
    NATIVE_METRIC_STATUS,
    PRACTICAL_EQUIVALENCE_MARGIN,
    PRIMARY_METRIC,
    PROTOCOL_NAME,
    SELECTION_CASES,
    CanonicalEvaluationError,
    evaluate_canonical,
    load_canonical_evaluation_config,
    prediction_record,
    supported_macro,
    validation_support,
)
from construction_safety_vision.config import ConfigError
from construction_safety_vision.data.canonical import scan_for_sensitive
from construction_safety_vision.paths import ProjectPaths, long_path
from construction_safety_vision.provenance import ProvenanceRecord, git_commit, sha256_file
from construction_safety_vision.segmentation_comparison import (
    CANDIDATE,
    ONE_VARIABLE_FIELD,
    REFERENCE,
    S1_STATUS,
    ComparisonConfigError,
    load_comparison_config,
    verify_one_variable_contract,
)
from construction_safety_vision.segmentation_experiment import load_segmentation_baseline_config
from construction_safety_vision.segmentation_run import RUNTIME_VIEW_ROOT
from construction_safety_vision.splits import HOLDOUT_UNLOCK_ENV_VAR, holdout_unlocked

CANONICAL_YAML = "segmentation_canonical_evaluation.yaml"
COMPARISON_YAML = "segmentation_comparison.yaml"
BASELINE_YAML = "segmentation_baseline.yaml"

S0_RESULT_JSON = "segmentation_S0_result_manifest.json"
S0_MASK_IOU_JSON = "segmentation_S0_mask_iou.json"

S0_CANONICAL_JSON = "segmentation_S0_canonical_evaluation.json"
REFERENCE_MD = "segmentation_canonical_comparison_reference.md"
POLICY_JSON = "segmentation_comparison_policy.json"
POLICY_MD = "segmentation_comparison_policy.md"
S1_PROTOCOL_MD = "segmentation_S1_protocol.md"
S1_MANIFEST_JSON = "segmentation_S1_protocol_manifest.json"
PROVENANCE_JSON = "segmentation_comparison.provenance.json"

SCHEMA_VERSION = 1
PHASE = "8E"

FROZEN = "S1_PROTOCOL_FROZEN"
MEMORY_CONSTRAINT_REVIEW_REQUIRED = "MEMORY_CONSTRAINT_REVIEW_REQUIRED"
GEOMETRY_BLOCKED = "CANONICAL_EVALUATOR_GEOMETRY_BLOCKED"
EVALUATION_FAILED = "CANONICAL_EVALUATION_FAILED"
PROTOCOL_INVALID = "S1_PROTOCOL_INVALID"
BLOCKED = "BLOCKED"

TIMING = "POST_S0_PRE_S1_CANONICAL_COMPARISON_REFERENCE_EVALUATION"
FINAL_SEGMENTER = "UNSELECTED"

EXPECTED_S0_CHECKPOINT = "d7b512b95fafdc8658fd802459d2c87d9ad3f11f922162a774dacf8ca75b87a3"
EXPECTED_PRETRAINED = "55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152"
EXPECTED_PRETRAINED_BYTES = 6182636

RARE_CLASS = "vest_loose"
RARE_CLASS_STATUS = "DESCRIPTIVE_HIGH_UNCERTAINTY"

FEASIBILITY_NON_EXPERIMENTAL = "NON_EXPERIMENTAL"
FEASIBILITY_BAN = "DO_NOT_REPORT_AS_MODEL_RESULT"

HOLDOUT_STATUS = "PROTECTED_NOT_ACCESSED"
HOLDOUT_REASON = (
    "Phase 8E evaluated the frozen S0 checkpoint on the frozen validation split under a newly "
    "frozen canonical evaluator, and ran one non-experimental memory feasibility check. The "
    "holdout was not read, materialised, adapted, counted, predicted on or inspected; no "
    "holdout identifier, image, mask or statistic exists in any artifact this phase wrote."
)

HISTORICAL: tuple[str, ...] = (
    "reports/final_detector_manifest.json",
    "reports/segmentation_adapter_audit_manifest.json",
    "reports/segmentation_adapter_fidelity_report.md",
    "reports/segmentation_adapter_fidelity.csv",
    "configs/segmentation_adapter_audit.yaml",
    "reports/segmentation_adapter_approval.json",
    "reports/segmentation_S0_manifest.json",
    "reports/segmentation_S0_protocol.md",
    "configs/segmentation_baseline.yaml",
    "configs/segmentation_mask_iou_evaluation.yaml",
    "reports/segmentation_S0_result_manifest.json",
    "reports/segmentation_S0_mask_iou.json",
    "reports/segmentation_S0_report.md",
    "reports/segmentation_S0_error_analysis.json",
    "reports/segmentation_S0_error_analysis.md",
    "reports/segmentation_S0_error_instances.csv",
)


class FreezeError(RuntimeError):
    """Raised when a required input is missing or an invariant fails."""


class MemoryConstraintError(FreezeError):
    """Raised when the frozen batch exhausts device memory."""


# --- helpers -------------------------------------------------------------------


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
    data = json.loads(path.read_text(encoding="utf-8"))
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
    """Digest every artifact this phase must leave untouched.

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


# --- the canonical evaluation --------------------------------------------------


def predict_canonical(
    paths: ProjectPaths,
    protocol: Any,
    checkpoint: Path,
    images: Mapping[str, int],
) -> list[dict[str, Any]]:
    """Run inference under the frozen canonical settings and serialise the masks.

    Masks come back on the original image canvas through the framework's own
    native mask path, then go through the reference RLE encoder - so no
    resampling of this project's own devising sits between the model and the
    metric.

    Args:
        paths: Project layout.
        protocol: The frozen canonical evaluation protocol.
        checkpoint: The experiment's checkpoint.
        images: Image stem to canonical image id.

    Returns:
        COCO-format detections.

    Raises:
        FreezeError: If prediction fails or an image is not in the canonical
            document.
    """
    from ultralytics import YOLO

    inference = protocol.inference
    images_root = paths.root / RUNTIME_VIEW_ROOT / "images" / "val"
    if not images_root.is_dir():
        msg = f"the validation image view is not on this machine: {RUNTIME_VIEW_ROOT}/images/val"
        raise FreezeError(msg)

    model = YOLO(str(checkpoint))
    detections: list[dict[str, Any]] = []
    for image_path in sorted(path for path in images_root.iterdir() if path.suffix != ".txt"):
        stem = image_path.stem
        if stem not in images:
            msg = f"predicted image {stem} has no canonical validation entry"
            raise FreezeError(msg)
        try:
            outputs = model.predict(
                source=str(image_path),
                imgsz=inference["imgsz"],
                conf=inference["conf"],
                iou=inference["iou"],
                max_det=inference["max_det"],
                retina_masks=inference["retina_masks"],
                augment=inference["augment"],
                agnostic_nms=inference["agnostic_nms"],
                half=inference["half"],
                verbose=False,
                save=False,
                stream=False,
            )
        except Exception as exc:
            msg = f"{EVALUATION_FAILED}: prediction failed on one validation image ({exc})"
            raise FreezeError(msg) from exc

        result = outputs[0]
        if result.masks is None:
            continue
        masks = result.masks.data.cpu().numpy().astype(bool)
        classes = result.boxes.cls.cpu().numpy().astype(int).tolist()
        scores = result.boxes.conf.cpu().numpy().astype(float).tolist()
        for mask, class_index, score in zip(masks, classes, scores, strict=True):
            detections.append(
                prediction_record(
                    image_id=images[stem],
                    category_id=int(class_index),
                    mask=mask,
                    score=float(score),
                )
            )
    return detections


# --- the S1 feasibility preflight ----------------------------------------------


def memory_feasibility(paths: ProjectPaths, baseline: Any) -> dict[str, Any]:
    """Prove the frozen batch fits with ``overlap_mask: false``, without training.

    One real batch: build the dataset with the candidate's target construction,
    run a forward pass, build the loss, and back-propagate. **No optimizer step
    is taken and no checkpoint is written**, so nothing here is a partially
    trained model and no metric of any kind is produced.

    The reason this check exists is specific: with overlap resolution off, every
    instance gets its own mask plane instead of sharing one indexed map, so the
    target tensor grows with the number of instances in an image. Whether the
    frozen batch of 8 still fits is an engineering question that must be settled
    before the experiment is authorised, not discovered during it.

    Args:
        paths: Project layout.
        baseline: The frozen S0 protocol S1 inherits.

    Returns:
        What happened, with no accuracy figure.

    Raises:
        MemoryConstraintError: If the frozen batch exhausts device memory. It is
            never reduced to rescue the check.
        FreezeError: If the preflight cannot execute.
    """
    import torch
    from ultralytics.cfg import get_cfg
    from ultralytics.data import build_dataloader, build_yolo_dataset
    from ultralytics.nn.tasks import SegmentationModel
    from ultralytics.utils import DEFAULT_CFG

    batch = int(baseline.training["batch"])
    imgsz = int(baseline.training["imgsz"])
    descriptor = paths.root / RUNTIME_VIEW_ROOT / "dataset.yaml"

    overrides = {
        **{key: value for key, value in baseline.training_arguments().items() if key != "batch"},
        "task": "segment",
        "mode": "train",
        "data": str(descriptor.resolve()),
        "imgsz": imgsz,
        "batch": batch,
        # The single intentional difference the candidate declares.
        "overlap_mask": False,
    }
    args = get_cfg(DEFAULT_CFG, overrides=overrides)

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        from construction_safety_vision.canonical_evaluation import decode_mask  # noqa: F401

        data = {
            "train": str((paths.root / RUNTIME_VIEW_ROOT / "images" / "train").resolve()),
            "val": str((paths.root / RUNTIME_VIEW_ROOT / "images" / "val").resolve()),
            "names": {index: name for name, index in _class_map(paths).items()},
            "nc": len(_class_map(paths)),
            "channels": 3,
        }
        dataset = build_yolo_dataset(
            args, data["train"], batch, data, mode="train", rect=False, stride=32
        )
        loader = build_dataloader(dataset, batch, workers=0, shuffle=False, rank=-1)
        sample = next(iter(loader))

        model = SegmentationModel(cfg="yolo11n-seg.yaml", nc=data["nc"], ch=3, verbose=False)
        model.args = args
        model = model.to("cuda").train()

        images = sample["img"].to("cuda", non_blocking=True).float() / 255
        moved = {
            key: (value.to("cuda") if hasattr(value, "to") else value)
            for key, value in sample.items()
            if key != "img"
        }
        moved["img"] = images

        loss, items = model.loss(moved)
        loss.sum().backward()
        torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError as exc:
        msg = (
            f"{MEMORY_CONSTRAINT_REVIEW_REQUIRED}: the frozen batch of {batch} exhausted device "
            f"memory with overlap_mask disabled ({exc}). The batch is not reduced, auto-batch is "
            "not enabled and imgsz is not lowered: the protocol stops for review."
        )
        raise MemoryConstraintError(msg) from exc
    except Exception as exc:
        msg = f"the memory feasibility preflight could not execute ({type(exc).__name__}: {exc})"
        raise FreezeError(msg) from exc

    elapsed = time.perf_counter() - started
    peak_reserved = int(torch.cuda.max_memory_reserved())
    peak_allocated = int(torch.cuda.max_memory_allocated())
    mask_shape = list(sample["masks"].shape) if "masks" in sample else None
    del model, loss, items, images, moved, sample
    torch.cuda.empty_cache()

    return {
        "status": "SUCCESS",
        "classification": FEASIBILITY_NON_EXPERIMENTAL,
        "reporting_ban": FEASIBILITY_BAN,
        "implementation": (
            "One real training batch built with the candidate's own target construction, then a "
            "forward pass, the loss, and a backward pass."
        ),
        "optimizer_step_taken": False,
        "validation_run": False,
        "checkpoint_written": False,
        "metrics_recorded": False,
        "epochs": 0,
        "batch": batch,
        "imgsz": imgsz,
        "overlap_mask": False,
        "mask_target_shape": mask_shape,
        "mask_target_note": (
            "With overlap resolution off the loader returns one mask plane per instance rather "
            "than a single indexed map, which is the growth this check exists to size."
        ),
        "peak_gpu_memory_reserved_bytes": peak_reserved,
        "peak_gpu_memory_reserved_gib": round(peak_reserved / 1024**3, 3),
        "peak_gpu_memory_allocated_bytes": peak_allocated,
        "out_of_memory": False,
        "runtime_seconds": round(elapsed, 3),
        "what_it_establishes": (
            "That the frozen batch of 8 at imgsz 768 completes a forward and backward pass with "
            "overlap_mask disabled on this device. Nothing about accuracy, convergence or the "
            "eventual S1 result."
        ),
    }


def _class_map(paths: ProjectPaths) -> dict[str, int]:
    """Read the frozen class map from the phase 8A record.

    Args:
        paths: Project layout.

    Returns:
        Class name to canonical index.
    """
    audit = read_json(paths.reports / "segmentation_adapter_audit_manifest.json")
    return {name: int(index) for name, index in audit["class_map"].items()}


# --- artifacts ------------------------------------------------------------------


def build_reference_report(payload: Mapping[str, Any], *, commit: str | None) -> str:
    """Render the canonical comparison reference report.

    Args:
        payload: The S0 canonical evaluation record.
        commit: Repository commit, when available.

    Returns:
        The report text.
    """
    lines: list[str] = []
    add = lines.append
    canonical = payload["canonical"]
    macro = payload["supported_macro"]
    direct = payload["direct_iou_reference"]
    native = payload["native_reference"]

    add("# Canonical segmentation comparison reference")
    add("")
    add(
        f"Phase {payload['phase']} · timing `{payload['timing']}` · S1 "
        f"`{payload['s1_status']}` · final segmenter `{payload['final_segmenter']}`"
    )
    add("")
    add(
        "**This is an addendum, not a correction.** Nothing in the phase 8C S0 report is "
        "withdrawn, revised or regenerated. A third measurement is added beside the two S0 "
        "already publishes, because a comparison that phase 8C did not anticipate now needs one."
    )
    add("")
    if commit:
        add(f"Repository commit at production time: `{commit}`.")
        add("")

    add("## 1. Why a common evaluator was introduced")
    add("")
    add(payload["why"])
    add("")
    add(
        "The timing is recorded rather than smoothed over: the need was discovered by the phase "
        "8D error analysis, **after S0 ran**. The protocol is nonetheless frozen **before S1 "
        "exists**, which is what keeps it a protocol rather than a description of whichever "
        "number turns out to be convenient."
    )
    add("")

    add("## 2. Why native mask AP will not arbitrate S0 versus S1")
    add("")
    add(payload["native_incomparability"])
    add("")
    add(
        f"Native mask AP is therefore labelled `{NATIVE_METRIC_STATUS}`. It is **not** "
        "suppressed: both experiments report it in full, and for each it remains a valid "
        "statement about that model against its own target."
    )
    add("")

    add("## 3. What remains valid from phase 8C")
    add("")
    add("| S0 result | Value | Status |")
    add("| --- | --- | --- |")
    add(
        f"| Native framework mask mAP@0.50:0.95 | {native['mask_map50_95']} | "
        "`NATIVE_TARGET_EVALUATION` - still valid |"
    )
    add(
        f"| Direct GT-normalised mask IoU | {direct['gt_normalized_mask_iou']} | "
        "`CANONICAL_GT_RECOVERY_DIAGNOSTIC` - still valid |"
    )
    add(
        f"| Direct matched mask IoU mean | {direct['matched_mask_iou_mean']} | "
        "`CANONICAL_GT_RECOVERY_DIAGNOSTIC` - still valid |"
    )
    add("")
    add(
        "The phase 8D re-score against effective overlap targets stays "
        "`HYPOTHESIS_GENERATING_DIAGNOSTIC_ONLY` and replaces no published S0 result."
    )
    add("")

    add("## 4. The canonical evaluator")
    add("")
    add(
        f"`PREDECLARED_PROTOCOL`, fingerprint `{payload['protocol_fingerprint']}`, frozen in "
        f"`configs/{CANONICAL_YAML}` and validated against synthetic fixtures before it was "
        "pointed at any checkpoint."
    )
    add("")
    add("| Setting | Value |")
    add("| --- | --- |")
    for name in sorted(payload["inference"]):
        add(f"| `{name}` | {payload['inference'][name]} |")
    thresholds = payload["cocoeval"]["iou_thresholds"]
    add(
        f"| COCOeval IoU thresholds | {thresholds[0]}:0.05:{thresholds[-1]} "
        f"({len(thresholds)} steps) |"
    )
    add(f"| COCOeval maxDets | {payload['cocoeval']['max_dets']} |")
    add(f"| Mask encoding | `{payload['mask_encoding']}` |")
    add(f"| Category mapping | `{payload['category_id_mapping']}` |")
    add("")
    add(
        "The confidence of 0.001 is **not an operating point**. Average precision integrates "
        "over the score curve and needs the low-scoring tail; the phase 8C direct-IoU diagnostic "
        "keeps its own operational 0.25, and the two protocols are never mixed. Note also that "
        f"the model may propose up to {payload['inference']['max_det']} candidates per image "
        "while COCOeval applies its conventional cap of 100 when scoring - two different numbers, "
        "both recorded, neither constraining the other."
    )
    add("")

    add("## 5. S0 canonical reference result")
    add("")
    add(
        f"`COMPUTED_RESULT`, classified `{payload['timing']}` and executed exactly once. "
        "Validation only."
    )
    add("")
    add("| Metric | Value |")
    add("| --- | --- |")
    add(f"| **{PRIMARY_METRIC}** | **{macro['value']}** |")
    add(f"| {ALL_CLASS_METRIC} | {canonical['all_class_map50_95']} |")
    add(f"| {ALL_CLASS_METRIC_50} | {canonical['all_class_map50']} |")
    add("")
    add("| Class | canonical AP@0.50:0.95 | canonical AP@0.50 | admitted by the support rule |")
    add("| --- | --- | --- | --- |")
    for name in sorted(canonical["per_class"]):
        entry = canonical["per_class"][name]
        admitted = payload["support"][name]["supported"]
        add(f"| `{name}` | {entry['AP@0.50:0.95']} | {entry['AP@0.50']} | {admitted} |")
    add("")
    add(
        f"The primary metric averages the {len(macro['admitted_classes'])} admitted classes "
        f"({', '.join('`' + name + '`' for name in macro['admitted_classes'])}) under the "
        f"unchanged phase 7A rule: {macro['rule']}. The rule names no class; the admitted set is "
        f"its output. `{RARE_CLASS}` stays `{RARE_CLASS_STATUS}`, reported in full and deciding "
        "nothing."
    )
    add("")
    add(
        "**These canonical numbers are not comparable with the native mask AP above.** They are "
        "computed against different ground truth, by a different implementation, at a different "
        "confidence. Reading a difference between them as a change in the model would be a "
        "mistake; each answers its own question about the same checkpoint."
    )
    add("")

    add("### 5.1 Where the two evaluators disagree, and why that is interesting")
    add("")
    add(
        "The two measurements are **not comparable in absolute terms** - different ground "
        "truth, a different implementation and a different confidence - so no figure below is a "
        "difference anyone should quote as a change in the model. What *is* legible is the "
        "**shape** of the disagreement across classes:"
    )
    add("")
    add("| Class | canonical AP@0.50:0.95 | native AP@0.50:0.95 |")
    add("| --- | --- | --- |")
    for name in sorted(canonical["per_class"]):
        native_value = payload["native_per_class"].get(name)
        add(f"| `{name}` | {canonical['per_class'][name]['AP@0.50:0.95']} | {native_value} |")
    add("")
    add(
        "The compact classes land close together under the two evaluators. `person` does not: it "
        "is by a wide margin the class where the canonical evaluator and the native one most "
        "disagree. That is exactly the class phase 8D identified, and exactly the direction the "
        "overlap-target mechanism predicts - the canonical ground truth includes the "
        "vest-and-helmet pixels the training target assigns away from the person, and the native "
        "evaluator does not."
    )
    add("")
    add(
        "**This corroborates the phase 8D mechanism; it does not prove it, and it certainly does "
        "not show that S1 will be better.** It is one more observation consistent with the same "
        "explanation, produced by an evaluator built for a different purpose - which is worth "
        "more than a second look at the same numbers, and still less than an experiment."
    )
    add("")

    add("## 6. The frozen S0-versus-S1 comparison")
    add("")
    add(
        f"`PREDECLARED_PROTOCOL`, fingerprint `{payload['comparison_fingerprint']}`. The primary "
        f"metric is `{PRIMARY_METRIC}`, the margin is {PRACTICAL_EQUIVALENCE_MARGIN}, and the "
        "three cases are fixed before the candidate exists:"
    )
    add("")
    add("| Case | Condition | Outcome |")
    add("| --- | --- | --- |")
    add(f"| `{SELECTION_CASES[0]}` | delta > +{PRACTICAL_EQUIVALENCE_MARGIN} | S1 leads |")
    add(
        f"| `{SELECTION_CASES[1]}` | within +/-{PRACTICAL_EQUIVALENCE_MARGIN} | S0 preferred, "
        "because the baseline exists and no material canonical advantage was shown |"
    )
    add(f"| `{SELECTION_CASES[2]}` | delta < -{PRACTICAL_EQUIVALENCE_MARGIN} | S0 retained |")
    add("")
    add(
        f"The margin is an engineering threshold, **not a significance test**: nothing is "
        f"repeated, so run-to-run variance remains UNKNOWN. If the canonical AP and the direct "
        f"IoU diagnostic move in opposite directions the outcome is `{DISAGREEMENT}` - recorded, "
        "ranked by the canonical metric, and never resolved by inventing a weighted score."
    )
    add("")

    add("## 7. The overlap-target hypothesis this tests")
    add("")
    add(payload["hypothesis"])
    add("")

    add("## 8. S1 status")
    add("")
    add(
        f"`{payload['s1_status']}`. S1 has not been trained, and no S1 number exists anywhere in "
        f"this repository. The final segmenter remains `{payload['final_segmenter']}`."
    )
    add("")
    add("---")
    add("")
    add(
        f"S0 checkpoint `{payload['checkpoint']['sha256']}` · canonical evaluation "
        f"`reports/{S0_CANONICAL_JSON}`."
    )
    add("")
    return "\n".join(lines)


def build_policy_report(policy: Mapping[str, Any], *, commit: str | None) -> str:
    """Render the comparison policy report.

    Args:
        policy: The assembled policy record.
        commit: Repository commit, when available.

    Returns:
        The report text.
    """
    lines: list[str] = []
    add = lines.append
    candidate = policy["candidate"]
    contract = policy["one_variable_contract"]

    add("# S0-versus-S1 segmentation comparison policy")
    add("")
    add(
        f"`{policy['protocol_timing']}` · reference `{policy['reference_experiment']}` · "
        f"candidate `{candidate['experiment_id']}` `{candidate['status']}` · final segmenter "
        f"`{policy['final_segmenter']}`"
    )
    add("")
    if commit:
        add(f"Repository commit at production time: `{commit}`.")
        add("")
    add(
        "**Frozen after S0 ran and before S1 exists.** That is stated first because it is the "
        "protocol's main limitation: phase 7A's detection policy was written before either of "
        "its candidates existed, and this one could not be. What it still guarantees is that no "
        "S1 number influenced any rule below."
    )
    add("")

    add("## 1. `CANONICAL_GROUND_TRUTH`")
    add("")
    add(
        f"Every comparison is scored against `{GROUND_TRUTH_SOURCE}` - the canonical phase 5D "
        "COCO instance segmentation, which no training flag can reshape. Never the YOLO adapter."
    )
    add("")

    add("## 2. `NATIVE_TARGET_METRIC`")
    add("")
    add(policy["native_incomparability"])
    add("")
    add(
        f"Status `{NATIVE_METRIC_STATUS}`. Both experiments still report native mask and box "
        "metrics in full; they are demoted, not suppressed."
    )
    add("")

    add("## 3. `PRIMARY_SELECTION_METRIC`")
    add("")
    add(f"`{policy['metrics']['primary']}`, with the margin {policy['margin']}.")
    add("")
    for case, text in sorted(policy["selection_logic"].items()):
        add(f"- **`{case}`** - {text}")
    add("")
    add(
        "All-class canonical metrics are required reporting and never override the selection "
        "policy - the same discipline phase 7A applied when D1's all-class figure moved opposite "
        "to its selection metric."
    )
    add("")

    add("## 4. `PREDECLARED_S1_HYPOTHESIS`")
    add("")
    add(f"> {candidate['question']}")
    add("")
    add("| Field | Value |")
    add("| --- | --- |")
    add(f"| Inherits | `{candidate['inherits']}` |")
    add(f"| Intentional field | `{candidate['intentional_field']}` |")
    add(
        f"| Reference value | `{contract['reference_value']}` -> candidate "
        f"`{contract['candidate_value']}` |"
    )
    add(f"| Fields inherited unchanged | {contract['inherited_count']} |")
    add(f"| Checkpoint policy | `{candidate['checkpoint_policy']}` |")
    add(f"| Adapter | `{candidate['adapter_identity']}` |")
    add("")
    add(
        "The one-variable contract is enforced by the parser and re-verified against S0's own "
        "protocol at freeze time, so it is a structural property rather than a promise."
    )
    add("")

    add("## 5. `LIMITATION`")
    add("")
    for limitation in policy["limitations"]:
        add(f"- {limitation}")
    add("")

    add("## 6. `HOLDOUT_POLICY`")
    add("")
    add(f"`{policy['test']['status']}`. {policy['test']['reason']}")
    add("")
    add("---")
    add("")
    add(f"Comparison protocol fingerprint `{policy['comparison_fingerprint']}`.")
    add("")
    return "\n".join(lines)


def build_s1_report(manifest: Mapping[str, Any], *, commit: str | None) -> str:
    """Render the S1 protocol report.

    Args:
        manifest: The assembled S1 protocol manifest.
        commit: Repository commit, when available.

    Returns:
        The report text.
    """
    lines: list[str] = []
    add = lines.append
    contract = manifest["one_variable_contract"]
    feasibility = manifest["memory_feasibility"]

    add("# S1 - overlap-mask segmentation experiment protocol")
    add("")
    add(
        f"Phase {manifest['phase']} · status `{manifest['status']}` · final segmenter "
        f"`{manifest['final_segmenter']}`"
    )
    add("")
    add(
        "**S1 has not been trained.** This document freezes what it will be, before it exists. "
        "No S1 number appears anywhere in this repository."
    )
    add("")
    if commit:
        add(f"Repository commit at production time: `{commit}`.")
        add("")

    add("## 1. Experiment identity and hypothesis")
    add("")
    add(f"> {manifest['question']}")
    add("")
    add(manifest["motivation"])
    add("")
    add(
        "**This is a hypothesis, not a prediction.** Phase 8D measured a mechanism aligned with "
        "S0's dominant person mask error; it did not establish that removing overlap resolution "
        "improves anything. The experiment exists to find out."
    )
    add("")

    add("## 2. The one intentional difference")
    add("")
    add(
        f"`{contract['intentional_field']}`: `{contract['reference_value']}` in S0, "
        f"`{contract['candidate_value']}` in S1. **Nothing else.** "
        f"{contract['inherited_count']} other fields are inherited unchanged, verified against "
        "S0's own frozen protocol rather than against a restatement of it."
    )
    add("")

    add("## 3. Inherited protocol, in full")
    add("")
    add("| Field | Value | Source |")
    add("| --- | --- | --- |")
    for name in sorted(manifest["inherited_protocol"]):
        add(f"| `{name}` | {manifest['inherited_protocol'][name]} | inherited from S0 |")
    add(f"| `overlap_mask` | {contract['candidate_value']} | **the intervention** |")
    add("")

    add("## 4. Checkpoint semantics")
    add("")
    add(manifest["checkpoint_semantics"])
    add("")

    add("## 5. What S1 must report")
    add("")
    add("`PREDECLARED_PROTOCOL`. All three families, none of them optional:")
    add("")
    for requirement in manifest["required_reporting"]:
        add(f"- {requirement}")
    add("")

    add("## 6. Adapter identity")
    add("")
    add(manifest["adapter"]["statement"])
    add("")
    add("| Fingerprint | Value |")
    add("| --- | --- |")
    for name in sorted(manifest["adapter"]["fingerprints"]):
        add(f"| `{name}` | `{manifest['adapter']['fingerprints'][name]}` |")
    add("")

    add("## 7. Memory feasibility")
    add("")
    add(
        f"`{feasibility['classification']}` · `{feasibility['reporting_ban']}` · status "
        f"`{feasibility['status']}`."
    )
    add("")
    add(feasibility["implementation"])
    add("")
    add("| Field | Value |")
    add("| --- | --- |")
    add(f"| Batch | {feasibility['batch']} |")
    add(f"| imgsz | {feasibility['imgsz']} |")
    add(f"| `overlap_mask` | {feasibility['overlap_mask']} |")
    add(f"| Mask target shape | {feasibility['mask_target_shape']} |")
    add(
        f"| Peak GPU memory reserved | {feasibility['peak_gpu_memory_reserved_gib']} GiB "
        f"({feasibility['peak_gpu_memory_reserved_bytes']} bytes) |"
    )
    add(f"| Out of memory | {feasibility['out_of_memory']} |")
    add(f"| Optimizer step taken | {feasibility['optimizer_step_taken']} |")
    add(f"| Validation run | {feasibility['validation_run']} |")
    add(f"| Checkpoint written | {feasibility['checkpoint_written']} |")
    add(f"| Accuracy metrics recorded | {feasibility['metrics_recorded']} |")
    add(f"| Runtime | {feasibility['runtime_seconds']} s |")
    add("")
    add(f"**What it establishes.** {feasibility['what_it_establishes']}")
    add("")
    add(f"{feasibility['mask_target_note']}")
    add("")

    add("## 8. Holdout")
    add("")
    add(f"`HOLDOUT_POLICY` · `{manifest['test']['status']}`. {manifest['test']['reason']}")
    add("")

    add("## 9. Next phase")
    add("")
    add(
        f"Phase 8F: run S1 exactly once under this frozen protocol, then apply the frozen "
        f"comparison policy. S1 is `{manifest['status']}` and the final segmenter is "
        f"`{manifest['final_segmenter']}`. Do not begin without an explicit instruction."
    )
    add("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Freeze the canonical comparison protocol and evaluate S0 once under it.

    Args:
        argv: Command-line arguments.

    Returns:
        Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="run every precondition check without evaluating, probing memory or writing",
    )
    args = parser.parse_args(argv)

    paths = ProjectPaths.from_root()
    if holdout_unlocked():
        print(
            f"REFUSED: {HOLDOUT_UNLOCK_ENV_VAR} is set. This phase reads development data only "
            "and declines to run in an environment where the holdout is unlocked.",
            file=sys.stderr,
        )
        return 2

    try:
        historical = historical_digests(paths)
        protocol = load_canonical_evaluation_config(paths.configs / CANONICAL_YAML)
        comparison = load_comparison_config(paths.configs / COMPARISON_YAML)
        baseline = load_segmentation_baseline_config(paths.configs / BASELINE_YAML)

        if comparison["canonical_evaluation_sha256"] != protocol.fingerprint():
            msg = (
                "the comparison protocol references a different canonical evaluator: it records "
                f"{comparison['canonical_evaluation_sha256']}, the committed evaluator hashes to "
                f"{protocol.fingerprint()}"
            )
            raise FreezeError(msg)

        s0_manifest = read_json(paths.reports / S0_RESULT_JSON)
        s0_direct = read_json(paths.reports / S0_MASK_IOU_JSON)

        checkpoint = paths.root / s0_manifest["execution"]["run_directory"] / "weights" / "best.pt"
        if not checkpoint.is_file():
            msg = (
                f"the frozen S0 checkpoint is not on this machine: {checkpoint.name}. It is "
                "git-ignored; obtain the artifact rather than retraining."
            )
            raise FreezeError(msg)
        checkpoint_sha256 = sha256_file(checkpoint)
        if (
            checkpoint_sha256 != EXPECTED_S0_CHECKPOINT
            or checkpoint_sha256 != s0_manifest["checkpoints"]["best"]["sha256"]
        ):
            msg = f"the S0 checkpoint bytes are not the frozen ones: observed {checkpoint_sha256}"
            raise FreezeError(msg)

        pretrained = paths.root / "artifacts" / "weights" / "yolo11n-seg.pt"
        if not pretrained.is_file():
            msg = "the pretrained yolo11n-seg.pt is not on this machine"
            raise FreezeError(msg)
        pretrained_sha256 = sha256_file(pretrained)
        pretrained_bytes = pretrained.stat().st_size
        if (
            pretrained_sha256 != EXPECTED_PRETRAINED
            or pretrained_bytes != EXPECTED_PRETRAINED_BYTES
        ):
            msg = (
                "S1 would not start from the same pretrained binary S0 used: observed "
                f"{pretrained_sha256} ({pretrained_bytes} bytes)"
            )
            raise FreezeError(msg)

        # The one-variable contract, checked against S0's own protocol rather
        # than against a second copy of it.
        reference_arguments = baseline.training_arguments()
        contract = verify_one_variable_contract(
            reference_arguments, comparison.candidate["overrides"]
        )

        class_map = _class_map(paths)
        approval = read_json(paths.reports / "segmentation_adapter_approval.json")
        adapter_fingerprints = dict(approval["adapter_fingerprints"])

        document = paths.root / str(protocol["ground_truth_document"])
        ground_truth = json.loads(Path(long_path(document)).read_text(encoding="utf-8"))
        images = {
            Path(str(entry["file_name"])).stem: int(entry["id"]) for entry in ground_truth["images"]
        }
        support = validation_support(ground_truth, class_map)
    except (
        ConfigError,
        ComparisonConfigError,
        CanonicalEvaluationError,
        FreezeError,
    ) as exc:
        classification = PROTOCOL_INVALID if isinstance(exc, ComparisonConfigError) else BLOCKED
        print(f"{classification}: {exc}", file=sys.stderr)
        return 2

    print(
        f"evaluator  {protocol.fingerprint()}  conf {protocol.inference['conf']}  "
        f"imgsz {protocol.inference['imgsz']}  IoU {protocol.inference['iou']}"
    )
    print(
        f"comparison {comparison.fingerprint()}  {REFERENCE} vs {CANDIDATE}  "
        f"one variable: {ONE_VARIABLE_FIELD} "
        f"{contract['reference_value']} -> {contract['candidate_value']}"
    )
    print(f"S0 ckpt    {checkpoint_sha256}  VERIFIED")
    print(f"pretrained {pretrained_sha256}  {pretrained_bytes} bytes  VERIFIED")
    print(
        f"canonical  {len(images)} validation images  "
        f"{len(ground_truth['annotations'])} GT instances"
    )
    print(f"support    admitted {[n for n in sorted(support) if support[n]['supported']]}")

    if args.verify_only:
        print("VERIFIED: preconditions hold. Nothing evaluated, probed or written.")
        print(f"holdout    {HOLDOUT_STATUS}")
        return 0

    # --- the single canonical evaluation of S0 ------------------------------------
    try:
        started = time.perf_counter()
        detections = predict_canonical(paths, protocol, checkpoint, images)
        canonical = evaluate_canonical(ground_truth, detections, class_map=class_map)
        elapsed = time.perf_counter() - started
    except CanonicalEvaluationError as exc:
        print(f"{GEOMETRY_BLOCKED}: {exc}", file=sys.stderr)
        return 2
    except FreezeError as exc:
        print(f"{EVALUATION_FAILED}: {exc}", file=sys.stderr)
        return 2

    macro = supported_macro(canonical["per_class"], support)
    print(
        f"S0 canonical  mask mAP@0.50:0.95 {canonical['all_class_map50_95']}  "
        f"mAP@0.50 {canonical['all_class_map50']}  supported macro {macro['value']}"
    )

    # --- the S1 feasibility preflight ---------------------------------------------
    try:
        feasibility = memory_feasibility(paths, baseline)
    except MemoryConstraintError as exc:
        print(f"{MEMORY_CONSTRAINT_REVIEW_REQUIRED}: {exc}", file=sys.stderr)
        return 2
    except FreezeError as exc:
        print(f"{BLOCKED}: {exc}", file=sys.stderr)
        return 2
    print(
        f"feasibility {feasibility['status']}  peak "
        f"{feasibility['peak_gpu_memory_reserved_gib']} GiB  "
        f"optimizer step {feasibility['optimizer_step_taken']}"
    )

    # --- artifacts -----------------------------------------------------------------
    native_reference = {
        "mask_map50_95": s0_manifest["mask_metrics"]["mAP@0.50:0.95"],
        "mask_map50": s0_manifest["mask_metrics"]["mAP@0.50"],
        "status": "NATIVE_TARGET_EVALUATION",
        "still_valid": True,
        "not_invalidated_by_this_phase": True,
    }
    direct_reference = {
        "gt_normalized_mask_iou": s0_direct["global"]["gt_normalized_mask_iou"],
        "matched_mask_iou_mean": s0_direct["global"]["matched_mask_iou_mean"],
        "gt_match_coverage": s0_direct["global"]["gt_match_coverage"],
        "gt_iou50_coverage": s0_direct["global"]["gt_iou50_coverage"],
        "gt_iou75_coverage": s0_direct["global"]["gt_iou75_coverage"],
        "status": "CANONICAL_GT_RECOVERY_DIAGNOSTIC",
        "still_valid": True,
        "protocol_fingerprint": s0_direct["protocol_fingerprint"],
        "rerun_in_this_phase": False,
    }

    why = (
        "Phase 8D established from the installed source that the framework builds both its mask "
        "target and its validation ground truth according to overlap_mask: "
        "polygons2masks_overlap gives contested pixels to the smaller instance, and "
        "SegmentationValidator._prepare_batch scores against that same resolved target. S0 "
        "trained with the flag set; the S1 candidate clears it. Their native mask AP would "
        "therefore be measured against different ground truth, so a common reference that no "
        "training flag can move was needed before the two could be compared at all."
    )
    native_incomparability = (
        "Native mask AP is not wrong and is not withdrawn - it is a valid statement about each "
        "model against its own target. It simply cannot rank two models whose targets differ, "
        "because a difference between the two numbers would confound the model with the target "
        "it was scored on. The canonical COCO evaluator decides instead."
    )
    hypothesis = (
        "Phase 8D measured that 27.3% of canonical person pixels are contested by another "
        "class, while 71.0% of the pixels S0 misses on person lie in that contested region; "
        "contested fraction versus person mask IoU has Spearman -0.5466; and re-scoring the same "
        "predictions against the overlap-resolved target recovers +0.0700 GT-normalised IoU for "
        "person and under 0.002 for every other class. S1 tests whether removing the overlap "
        "resolution from training changes canonical performance. It is a hypothesis: the "
        "measurement showed a target mismatch, not that training differently helps, and even "
        "against its own target S0's person mask IoU remained far below the compact classes."
    )

    canonical_payload = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "experiment_id": REFERENCE,
        "timing": TIMING,
        "timing_note": (
            "The canonical evaluator was frozen after S0 ran, because the need for it was "
            "discovered by the phase 8D error analysis, and before S1 exists. This evaluation "
            "is the reference the future candidate will be compared against."
        ),
        "why": why,
        "native_incomparability": native_incomparability,
        "hypothesis": hypothesis,
        "protocol": PROTOCOL_NAME,
        "protocol_fingerprint": protocol.fingerprint(),
        "protocol_config": f"configs/{CANONICAL_YAML}",
        "comparison_fingerprint": comparison.fingerprint(),
        "comparison_config": f"configs/{COMPARISON_YAML}",
        "checkpoint": {
            "experiment": REFERENCE,
            "name": "best.pt",
            "sha256": checkpoint_sha256,
            "selected_by": s0_manifest["checkpoint_selection"]["policy"],
            "last_pt_used": False,
        },
        "ground_truth": {
            "source": GROUND_TRUTH_SOURCE,
            "document": str(protocol["ground_truth_document"]),
            "sha256": sha256_file(paths.root / str(protocol["ground_truth_document"])),
            "images": len(images),
            "annotations": len(ground_truth["annotations"]),
            "is_the_yolo_adapter": False,
        },
        "split": "validation",
        "class_map_sha256": str(protocol["class_map_sha256"]),
        "split_assignment_sha256": str(protocol["split_assignment_sha256"]),
        "inference": {
            key: value
            for key, value in protocol.inference.items()
            if key not in ("threshold_policy", "checkpoint")
        },
        "cocoeval": canonical["cocoeval"],
        "mask_encoding": protocol.cocoeval["mask_encoding"],
        "category_id_mapping": protocol.cocoeval["category_id_mapping"],
        "prediction_cap_versus_metric_cap": (
            f"The model may propose up to {protocol.inference['max_det']} candidates per image; "
            f"COCOeval applies its conventional cap of {MAX_DETS[-1]} when scoring. Both are "
            "recorded so neither is read as constraining the other."
        ),
        "native_per_class": {
            name: s0_manifest["per_class_metrics"][name]["mask"]["AP@0.50:0.95"]
            for name in sorted(s0_manifest["per_class_metrics"])
        },
        "native_per_class_note": (
            "Reproduced from the committed phase 8C manifest for shape comparison only. The two "
            "evaluators use different ground truth, implementations and confidences, so their "
            "absolute values are not comparable and no difference between them measures the "
            "model."
        ),
        "canonical": {
            "all_class_map50_95": canonical["all_class_map50_95"],
            "all_class_map50": canonical["all_class_map50"],
            "per_class": canonical["per_class"],
            "detections_scored": canonical["detections"],
        },
        "supported_macro": macro,
        "support": support,
        "native_reference": native_reference,
        "direct_iou_reference": direct_reference,
        "phase_8d_rescore_status": "HYPOTHESIS_GENERATING_DIAGNOSTIC_ONLY",
        "phase_8d_rescore_note": (
            "The phase 8D re-score against effective overlap targets is a diagnostic. It does "
            "not replace any published S0 performance result and is not used by this protocol."
        ),
        "runs": 1,
        "runtime_seconds": round(elapsed, 3),
        "s1_status": S1_STATUS,
        "final_segmenter": FINAL_SEGMENTER,
        "s0_retrained": False,
        "s0_native_metrics_regenerated": False,
        "test": {"status": HOLDOUT_STATUS, "reason": HOLDOUT_REASON},
    }

    policy = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "protocol_timing": comparison["protocol_timing"],
        "frozen_before_s1": True,
        "frozen_before_s0": False,
        "reference_experiment": comparison["reference_experiment"],
        "candidate": comparison.candidate,
        "one_variable_contract": contract,
        "canonical_evaluation_config": comparison["canonical_evaluation_config"],
        "canonical_evaluation_fingerprint": protocol.fingerprint(),
        "comparison_fingerprint": comparison.fingerprint(),
        "metrics": comparison["metrics"],
        "margin": comparison["margin"],
        "margin_is_not_a_significance_test": True,
        "selection_logic": comparison["selection_logic"],
        "disagreement_policy": comparison["disagreement_policy"],
        "native_incomparability": native_incomparability,
        "rare_class": {"name": RARE_CLASS, "status": RARE_CLASS_STATUS},
        "reference_values": {
            "canonical_supported_macro_mask_map50_95": macro["value"],
            "canonical_all_class_mask_map50_95": canonical["all_class_map50_95"],
            "canonical_all_class_mask_map50": canonical["all_class_map50"],
            "direct_gt_normalized_mask_iou": direct_reference["gt_normalized_mask_iou"],
            "direct_matched_mask_iou_mean": direct_reference["matched_mask_iou_mean"],
            "native_mask_map50_95": native_reference["mask_map50_95"],
        },
        "candidate_values": None,
        "candidate_values_reason": "S1 has not been executed. No number exists to record.",
        "limitations": [
            "**Frozen after S0 ran.** Phase 7A's detection policy predated both its candidates; "
            "this one could not, because the need for a common evaluator was discovered by phase "
            "8D. No S1 number influenced any rule here, but the asymmetry is real and recorded.",
            "**The margin is an engineering threshold.** It carries no confidence level, nothing "
            "is repeated, and run-to-run variance on this setup remains UNKNOWN.",
            "**One dataset, one split, one seed.** 65 validation images and 304 instances; a "
            "result under this policy generalises no further than that.",
            "**The canonical evaluator is not the framework's.** Its absolute values are not "
            "comparable with native mask AP, and no reader should difference the two.",
            "**overlap_mask changes the treatment environment, not just a hyperparameter.** The "
            "native checkpoint fitness that selects each experiment's best.pt is itself computed "
            "against the flag's own target, so the two experiments select their checkpoints "
            "under different fitness definitions. That is why the comparison is external, and it "
            "is a genuine limitation rather than a detail.",
        ],
        "final_segmenter": FINAL_SEGMENTER,
        "test": {"status": HOLDOUT_STATUS, "reason": HOLDOUT_REASON},
    }

    inherited = {
        key: value
        for key, value in sorted(reference_arguments.items())
        if key != ONE_VARIABLE_FIELD
    }
    s1_manifest = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "experiment_id": CANDIDATE,
        "status": S1_STATUS,
        "task": "segmentation",
        "question": comparison.candidate["question"],
        "motivation": hypothesis,
        "inherits": REFERENCE,
        "inherited_protocol_fingerprint": baseline.fingerprint(),
        "inherited_protocol": inherited,
        "one_variable_contract": contract,
        "model": baseline.model,
        "architecture": baseline.architecture,
        "pretrained_weights": {
            "identifier": baseline.weight_identifier,
            "sha256": pretrained_sha256,
            "size_bytes": pretrained_bytes,
            "identical_to_s0": True,
            "committed": False,
        },
        "checkpoint_policy": comparison.candidate["checkpoint_policy"],
        "checkpoint_semantics": (
            "S1 keeps S0's checkpoint rule, ULTRALYTICS_NATIVE_SEGMENTATION_FITNESS, unchanged. "
            "Because overlap_mask alters the native target semantics, that fitness is computed "
            "against the candidate's own target rather than against S0's - so the checkpoint "
            "rule is part of the treatment environment, not a constant held across the two "
            "runs. It is kept identical in policy precisely so that no second variable is "
            "introduced, and the S0-versus-S1 decision is then made externally, against "
            "canonical COCO masks that neither flag can move. Creating a mask-only checkpoint "
            "selector for either experiment is not authorised."
        ),
        "adapter": {
            "statement": (
                "S1 reads exactly the same approved phase 8A adapter label bytes as S0. The "
                "intervention happens inside the framework's target construction, not in the "
                "data: no conversion change, no mask filtering, no relabelling, and no change "
                "to the canonical modelling population."
            ),
            "identity": comparison.candidate["adapter_identity"],
            "fingerprints": adapter_fingerprints,
            "regenerated": False,
        },
        "required_reporting": list(comparison.candidate["required_reporting"]),
        "comparison_protocol_fingerprint": comparison.fingerprint(),
        "canonical_evaluation_fingerprint": protocol.fingerprint(),
        "memory_feasibility": feasibility,
        "models_trained_in_this_phase": 0,
        "final_segmenter": FINAL_SEGMENTER,
        "test": {"status": HOLDOUT_STATUS, "reason": HOLDOUT_REASON},
    }

    commit = git_commit(paths.root)
    reference_report = build_reference_report(canonical_payload, commit=commit)
    policy_report = build_policy_report(policy, commit=commit)
    s1_report = build_s1_report(s1_manifest, commit=commit)

    findings: list[str] = []
    for document in (canonical_payload, policy, s1_manifest):
        findings += scan_for_sensitive(json.dumps(document))
    for text in (reference_report, policy_report, s1_report):
        findings += scan_for_sensitive(text)
    if findings:
        print(f"{BLOCKED}: sensitive content in the emitted artifacts: {findings}", file=sys.stderr)
        return 2

    canonical_sha256 = write_json(paths.reports / S0_CANONICAL_JSON, canonical_payload)
    policy_sha256 = write_json(paths.reports / POLICY_JSON, policy)
    s1_sha256 = write_json(paths.reports / S1_MANIFEST_JSON, s1_manifest)
    (paths.reports / REFERENCE_MD).write_text(reference_report, encoding="utf-8", newline="\n")
    (paths.reports / POLICY_MD).write_text(policy_report, encoding="utf-8", newline="\n")
    (paths.reports / S1_PROTOCOL_MD).write_text(s1_report, encoding="utf-8", newline="\n")

    after = historical_digests(paths)
    changed = sorted(name for name in historical if historical[name] != after.get(name))
    if changed:
        print(f"{BLOCKED}: historical artifacts changed: {changed}", file=sys.stderr)
        return 2

    record = ProvenanceRecord.create(
        name="segmentation_canonical_comparison_freeze",
        phase=8,
        config={
            "canonical_evaluation": f"configs/{CANONICAL_YAML}",
            "comparison": f"configs/{COMPARISON_YAML}",
            "canonical_evaluation_fingerprint": protocol.fingerprint(),
            "comparison_fingerprint": comparison.fingerprint(),
        },
        details={
            "phase": PHASE,
            "classification": FROZEN,
            "timing": TIMING,
            "reference_experiment": REFERENCE,
            "candidate": CANDIDATE,
            "candidate_status": S1_STATUS,
            "one_variable_field": ONE_VARIABLE_FIELD,
            "s0_checkpoint_sha256": checkpoint_sha256,
            "s0_canonical_supported_macro": macro["value"],
            "s0_canonical_all_class_map50_95": canonical["all_class_map50_95"],
            "s0_retrained": False,
            "s0_native_metrics_regenerated": False,
            "models_trained_in_this_phase": 0,
            "memory_feasibility": feasibility["status"],
            "optimizer_step_taken": feasibility["optimizer_step_taken"],
            "final_segmenter": FINAL_SEGMENTER,
            "historical_artifacts_unchanged": True,
            "holdout_accessed": False,
        },
        repo_root=paths.root,
    )
    for name in (CANONICAL_YAML, COMPARISON_YAML, BASELINE_YAML):
        record.add_input(paths.configs / name, relative_to=paths.root)
    for name in (S0_RESULT_JSON, S0_MASK_IOU_JSON):
        record.add_input(paths.reports / name, relative_to=paths.root)
    for name in (
        S0_CANONICAL_JSON,
        POLICY_JSON,
        S1_MANIFEST_JSON,
        REFERENCE_MD,
        POLICY_MD,
        S1_PROTOCOL_MD,
    ):
        record.add_output(paths.reports / name, relative_to=paths.root)
    record.write_json(paths.reports / PROVENANCE_JSON)

    print(FROZEN)
    print(
        f"S0 canonical  supported macro {macro['value']}  "
        f"all-class {canonical['all_class_map50_95']} / {canonical['all_class_map50']}"
    )
    print(f"reference  reports/{S0_CANONICAL_JSON}  sha256 {canonical_sha256}")
    print(f"policy     reports/{POLICY_JSON}  sha256 {policy_sha256}")
    print(f"S1         reports/{S1_MANIFEST_JSON}  sha256 {s1_sha256}  {S1_STATUS}")
    print(f"segmenter  {FINAL_SEGMENTER}")
    print(f"holdout    {HOLDOUT_STATUS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
