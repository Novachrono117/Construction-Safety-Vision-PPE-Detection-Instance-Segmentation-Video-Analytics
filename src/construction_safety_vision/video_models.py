"""Frozen D2/S1 frame inference; no evaluation, data, training or tracking route."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from construction_safety_vision.checkpoint_delivery import (
    build_manifest,
    load_manifest,
    verify_file,
)
from construction_safety_vision.detection_freeze import load_final_detector
from construction_safety_vision.provenance import sha256_file
from construction_safety_vision.segmentation_freeze import load_final_segmenter

CLASSES = ("helmet_loose", "helmet_on_head", "person", "vest_loose", "vest_on_body")
MODES = {"detector": ("D2",), "segmenter": ("S1",), "compare": ("D2", "S1")}
IDENTITIES = {
    "D2": ("0466f872a9de22898c70d8834cf1fcfb3d77f6c9fb27e81ee6248d3d19cbc206", 5502289),
    "S1": ("29337d671459d0f742fb713613cc41d0eafcb8a21f5846ac0da65b2ffc024f20", 6041685),
}
POLICY_PATH = "configs/detector_segmenter_comparison.yaml"


class VideoRuntimeError(RuntimeError):
    """An actionable error whose text contains no private source path."""


def frozen_settings(root: Path) -> dict[str, Any]:
    """Read the authoritative operating point and reject scientific drift."""
    raw = yaml.safe_load((root / POLICY_PATH).read_text(encoding="utf-8"))
    actual = dict(raw["operational_inference"])
    actual.pop("purpose")
    expected = {
        "imgsz": 768,
        "conf": 0.25,
        "iou": 0.70,
        "max_det": 300,
        "augment": False,
        "tta": False,
        "precision": "FP32",
        "precision_argument": "quantize",
        "precision_value": 32,
    }
    if actual != expected:
        raise VideoRuntimeError("SCIENTIFIC_STATE_MISMATCH: frozen operating point changed")
    return {
        **actual,
        "batch": 1,
        "retina_masks_s1": True,
        "source": POLICY_PATH,
        "source_sha256": sha256_file(root / POLICY_PATH),
    }


def require_device(device: str) -> dict[str, Any]:
    """Resolve an explicit device; a CUDA request never falls back to CPU."""
    import torch

    if device not in ("cuda", "cpu"):
        raise VideoRuntimeError("Device must be cuda or cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise VideoRuntimeError("CUDA_UNAVAILABLE: requested CUDA; no CPU fallback")
    return {
        "requested": device,
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "name": torch.cuda.get_device_name(0) if device == "cuda" else "CPU",
        "index": 0 if device == "cuda" else None,
    }


def verified_checkpoints(root: Path, mode: str) -> dict[str, dict[str, Any]]:
    """Verify public metadata, freeze identities and only the local frozen copies."""
    if mode not in MODES:
        raise VideoRuntimeError("Unknown mode: use detector, segmenter or compare")
    public = load_manifest(root / "delivery/checkpoints.json")
    if public != build_manifest(root):
        raise VideoRuntimeError("SCIENTIFIC_STATE_MISMATCH: public checkpoint metadata")
    result = {}
    for name in MODES[mode]:
        frozen = (load_final_detector if name == "D2" else load_final_segmenter)(root / "reports")
        entry = next(e for e in public["checkpoints"] if e["logical_name"] == name)
        if (entry["sha256"], entry["bytes"]) != IDENTITIES[name]:
            raise VideoRuntimeError("SCIENTIFIC_STATE_MISMATCH: checkpoint identity")
        if name == "S1" and (frozen.mask_ratio != 4 or frozen.overlap_mask is not False):
            raise VideoRuntimeError("SCIENTIFIC_STATE_MISMATCH: S1 identity")
        relative = frozen.frozen_copy_relative_path
        if not relative or not (root / relative).resolve().is_relative_to(root.resolve()):
            raise VideoRuntimeError("Invalid frozen checkpoint path")
        path = root / relative
        if not path.is_file():
            raise VideoRuntimeError("BLOCKED_MISSING_CHECKPOINT: restore the local frozen copy")
        try:
            verify_file(path, sha256=entry["sha256"], size_bytes=entry["bytes"])
        except ValueError as exc:
            raise VideoRuntimeError("SCIENTIFIC_STATE_MISMATCH: checkpoint bytes") from exc
        result[name] = {**entry, "local_path": relative}
    return result


class FrozenFramePredictor:
    """Run one BGR frame per predict call, with runtime FP32 and canvas checks."""

    def __init__(self, root: Path, mode: str, device: str) -> None:
        import torch
        import ultralytics
        from ultralytics import YOLO

        if "CSVISION_ALLOW_TEST_SPLIT" in os.environ:
            raise VideoRuntimeError("Holdout opt-in must be unset")
        self.settings = frozen_settings(root)
        self.device = require_device(device)
        self.identities = verified_checkpoints(root, mode)
        self.models: dict[str, Any] = {}
        self.states: dict[str, Any] = {}
        self.handles: list[Any] = []
        self.framework = ultralytics.__version__
        self.torch = torch
        for name, identity in self.identities.items():
            model = YOLO(str(root / identity["local_path"]))
            if tuple(model.names[i] for i in range(len(model.names))) != CLASSES:
                raise VideoRuntimeError("Checkpoint canonical class order mismatch")
            state = {
                "predict_invocations": 0,
                "frames_completed": 0,
                "forward_calls_including_framework_warmup": 0,
                "effective_dtypes": [],
                "effective_batch_sizes": [],
                "detections": 0,
            }
            self.states[name] = state
            self.models[name] = model

            def attach(predictor: Any, state: dict[str, Any] = state) -> None:
                backend = predictor.model
                if backend.fp16 or {str(p.dtype) for p in backend.model.parameters()} != {
                    "torch.float32"
                }:
                    raise VideoRuntimeError("FP32 parameter contract violated")
                if state.get("hook_attached"):
                    return

                def hook(_module: Any, args: Any) -> None:
                    tensor = args[0]
                    if (
                        tensor.dtype != torch.float32
                        or tensor.shape[0] != 1
                        or tensor.device.type != device
                        or torch.is_autocast_enabled(device)
                    ):
                        raise VideoRuntimeError("FP32/batch/device contract violated")
                    state["forward_calls_including_framework_warmup"] += 1
                    state["effective_dtypes"] = [str(tensor.dtype)]
                    state["effective_batch_sizes"] = [int(tensor.shape[0])]

                self.handles.append(backend.model.register_forward_pre_hook(hook))
                state["hook_attached"] = True

            model.add_callback("on_predict_start", attach)

    def predict(self, frame: np.ndarray) -> dict[str, list[dict[str, Any]]]:
        """Infer the same original frame independently with each requested frozen model."""
        outputs = {}
        for name, model in self.models.items():
            state = self.states[name]
            state["predict_invocations"] += 1
            params = {k: self.settings[k] for k in ("imgsz", "conf", "iou", "max_det", "augment")}
            results = model.predict(
                source=frame.copy(),
                batch=1,
                quantize=32,
                device=0 if self.device["requested"] == "cuda" else "cpu",
                retina_masks=name == "S1",
                save=False,
                save_txt=False,
                verbose=False,
                **params,
            )
            if len(results) != 1 or tuple(results[0].orig_shape) != frame.shape[:2]:
                raise VideoRuntimeError("Prediction canvas/frame count mismatch")
            result = results[0]
            masks = result.masks.data.cpu().numpy() if result.masks is not None else None
            rows = []
            for i, box in enumerate(result.boxes):
                class_id = int(box.cls.item())
                if class_id not in range(len(CLASSES)):
                    raise VideoRuntimeError("Invalid class ID")
                mask = None
                if name == "S1":
                    if masks is None or masks[i].shape != frame.shape[:2]:
                        raise VideoRuntimeError("S1 mask does not match original canvas")
                    mask = masks[i].astype(bool)
                rows.append(
                    {
                        "class_id": class_id,
                        "confidence": float(box.conf.item()),
                        "box": box.xyxy[0].cpu().tolist(),
                        "mask": mask,
                    }
                )
            outputs[name] = rows
            state["frames_completed"] += 1
            state["detections"] += len(rows)
        return outputs

    def evidence(self) -> dict[str, Any]:
        """Return small invocation/identity records, never bulk predictions."""
        return {
            "device": self.device,
            "ultralytics": self.framework,
            "settings": self.settings,
            "models": {n: {"identity": self.identities[n], **s} for n, s in self.states.items()},
        }

    def close(self) -> None:
        """Remove forward instrumentation deterministically."""
        for handle in self.handles:
            handle.remove()
