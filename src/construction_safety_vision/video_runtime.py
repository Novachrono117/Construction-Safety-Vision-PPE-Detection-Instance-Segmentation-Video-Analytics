"""Sequential constant-frame-rate video processing with verified, atomic delivery."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import cv2
import numpy as np

from construction_safety_vision.data.canonical import scan_for_sensitive
from construction_safety_vision.provenance import git_commit, runtime_environment, sha256_file
from construction_safety_vision.video_models import MODES, FrozenFramePredictor, VideoRuntimeError
from construction_safety_vision.video_render import render_frame

CODEC = "mp4v"
INPUT_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
SOURCE_FIELDS = {
    "source_type",
    "title",
    "creator",
    "url",
    "license",
    "retrieval_date",
    "modifications",
}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write portable small metadata without NaN values."""
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def source_metadata(value: dict[str, Any] | None) -> dict[str, Any]:
    """Validate optional source attribution; never persist signed URLs or private paths."""
    if value is None:
        return {"source_type": "UNSPECIFIED", **dict.fromkeys(SOURCE_FIELDS - {"source_type"})}
    if set(value) != SOURCE_FIELDS or value["source_type"] not in {
        "SYNTHETIC_ENGINEERING_FIXTURE",
        "VALIDATION_ENGINEERING_FIXTURE",
        "EXTERNAL_REAL_VIDEO",
    }:
        raise VideoRuntimeError("Source metadata requires the documented seven fields and type")
    if any(v is not None and (not isinstance(v, str) or len(v) > 2000) for v in value.values()):
        raise VideoRuntimeError("Source metadata values must be bounded text or null")
    if scan_for_sensitive(json.dumps(value)):
        raise VideoRuntimeError("Source metadata contains sensitive information")
    if value["url"]:
        url = urlsplit(value["url"])
        if (
            url.scheme != "https"
            or not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise VideoRuntimeError("Source URL must be public HTTPS without query/credentials")
    return dict(value)


def validate_paths(
    root: Path,
    source: Path,
    output: Path,
    mode: str,
    device: str,
    overwrite: bool,
    max_frames: int | None,
) -> None:
    """Reject unsafe input/output contracts before opening any media."""
    if "CSVISION_ALLOW_TEST_SPLIT" in os.environ:
        raise VideoRuntimeError("Holdout opt-in must be unset")
    if mode not in MODES or device not in {"cuda", "cpu"}:
        raise VideoRuntimeError("Unknown mode or device")
    if max_frames is not None and (type(max_frames) is not int or max_frames <= 0):
        raise VideoRuntimeError("max-frames must be positive; engineering smoke tests only")
    if source.resolve().is_relative_to((root / "data").resolve()):
        raise VideoRuntimeError(
            "Dataset paths are not video inputs; use a separately attributed fixture"
        )
    if not source.is_file() or source.suffix.lower() not in INPUT_SUFFIXES:
        raise VideoRuntimeError("Input must be an existing ordinary video file")
    if source.resolve() == output.resolve() or output.suffix.lower() != ".mp4":
        raise VideoRuntimeError("Output must be a distinct MP4 file")
    if output.resolve().is_relative_to(root.resolve()) and not output.resolve().is_relative_to(
        (root / "outputs").resolve()
    ):
        raise VideoRuntimeError("In-repository video outputs must be under outputs/")
    sidecar = Path(str(output) + ".provenance.json")
    if (output.exists() or sidecar.exists()) and not overwrite:
        raise VideoRuntimeError("Output/provenance already exists; use --overwrite explicitly")
    if output.is_dir() or sidecar.is_dir():
        raise VideoRuntimeError("Output/provenance must be files")


def open_capture(path: Path) -> Any:
    """Open the FFmpeg backend explicitly, rejecting unavailable codec/container input."""
    capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    if not capture.isOpened():
        capture.release()
        raise VideoRuntimeError("INPUT_OPEN_FAILED: FFmpeg could not open video")
    # Keep the encoded raster, not backend-specific orientation transforms.
    capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    return capture


def video_metadata(capture: Any) -> dict[str, Any]:
    """Read and validate metadata before allocating an encoder or executing a model."""
    width, height, fps, count = (
        capture.get(k)
        for k in (
            cv2.CAP_PROP_FRAME_WIDTH,
            cv2.CAP_PROP_FRAME_HEIGHT,
            cv2.CAP_PROP_FPS,
            cv2.CAP_PROP_FRAME_COUNT,
        )
    )
    if (
        not all(math.isfinite(v) for v in (width, height, fps))
        or not 0 < fps <= 1000
        or width < 2
        or height < 2
        or not width.is_integer()
        or not height.is_integer()
    ):
        raise VideoRuntimeError("INVALID_VIDEO_METADATA: dimensions/FPS")
    if int(width) % 2 or int(height) % 2:
        raise VideoRuntimeError("ODD_DIMENSIONS_UNSUPPORTED: mp4v must not silently crop a pixel")
    frames = round(count) if math.isfinite(count) and count > 0 else None
    fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
    return {
        "width": int(width),
        "height": int(height),
        "fps": fps,
        "frame_count": frames,
        "duration_seconds": frames / fps if frames else None,
        "codec_fourcc": "".join(chr((fourcc >> (8 * i)) & 255) for i in range(4)),
        "backend": capture.getBackendName(),
        "orientation_auto": False,
    }


def open_writer(path: Path, fps: float, dimensions: tuple[int, int]) -> Any:
    """Open the tested MP4/mp4v writer; no untested codec fallback."""
    writer = cv2.VideoWriter(
        str(path), cv2.CAP_FFMPEG, cv2.VideoWriter_fourcc(*CODEC), fps, dimensions
    )
    if not writer.isOpened():
        writer.release()
        raise VideoRuntimeError("WRITER_OPEN_FAILED: MP4/mp4v encoder unavailable")
    return writer


def verify_output(
    path: Path, *, frames: int, fps: float, dimensions: tuple[int, int]
) -> dict[str, Any]:
    """Reopen and decode the whole output; verify count, dimensions, FPS and duration."""
    if not path.is_file() or path.stat().st_size == 0:
        raise VideoRuntimeError("OUTPUT_EMPTY_OR_MISSING")
    cap = open_capture(path)
    try:
        meta = video_metadata(cap)
        if (meta["width"], meta["height"]) != dimensions or not math.isclose(
            meta["fps"], fps, rel_tol=1e-4, abs_tol=1e-3
        ):
            raise VideoRuntimeError("OUTPUT_METADATA_MISMATCH")
        count = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame is None or frame.shape != (dimensions[1], dimensions[0], 3):
                raise VideoRuntimeError("OUTPUT_FRAME_SHAPE_MISMATCH")
            count += 1
        if count != frames or meta["frame_count"] != frames:
            raise VideoRuntimeError("OUTPUT_FRAME_COUNT_MISMATCH")
        if not math.isclose(meta["duration_seconds"], frames / fps, abs_tol=1 / fps):
            raise VideoRuntimeError("OUTPUT_DURATION_MISMATCH")
        return {**meta, "verified_decoded_frames": count, "requested_codec": CODEC}
    finally:
        cap.release()


def file_identity(path: Path) -> dict[str, Any]:
    """Return portable filename, byte size and content digest only."""
    return {"filename": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size}


def run_video(
    source: Path,
    output: Path,
    *,
    mode: str,
    device: str,
    root: Path,
    overwrite: bool = False,
    max_frames: int | None = None,
    fps_overlay: bool = True,
    attribution: dict[str, Any] | None = None,
    predictor_factory: Callable[..., Any] = FrozenFramePredictor,
) -> dict[str, Any]:
    """Process every frame once and publish only a fully verified output.

    A capped engineering run is explicitly ENGINEERING_PREFIX_COMPLETE, never a
    complete source-video deliverable. The test injection point does not exist
    in the public CLI. Failures retain a FAILED sidecar and remove partial media.
    """
    validate_paths(root, source, output, mode, device, overwrite, max_frames)
    attribution = source_metadata(attribution)
    output.parent.mkdir(parents=True, exist_ok=True)
    running = Path(str(output) + ".incomplete.provenance.json")
    partial = output.with_name(output.stem + ".incomplete.mp4")
    if partial.exists():
        raise VideoRuntimeError("Incomplete output exists; inspect the previous attempt first")
    record: dict[str, Any] = {
        "schema_version": 1,
        "status": "STARTED",
        "purpose": "DEMO_RUNTIME_MEASUREMENT",
        "mode": mode,
        "requested_device": device,
        "source_attribution": attribution,
        "input": {"filename": source.name},
        "output": {"filename": output.name},
        "git_commit": git_commit(root),
        "environment": runtime_environment(),
        "opencv": cv2.__version__,
        "implementation_sha256": {
            name: sha256_file(Path(__file__).with_name(name))
            for name in ("video_runtime.py", "video_models.py", "video_render.py")
        },
        "decoded_frames": 0,
        "processed_frames": 0,
        "encoded_frames": 0,
        "warnings": ["NO_AUDIO_OR_TIMESTAMP_TRACK_COPIED"],
        "max_frames": max_frames,
        "holdout_accessed": False,
        "training": False,
        "threshold_changed": False,
        "tracking": False,
        "fps_overlay": fps_overlay,
        "timing_contract": "Decode + predict/preprocess/postprocess + render + encode + writer "
        "flush; includes first-call framework warmup; excludes setup/model loading, hashes, "
        "output verification and sidecar writes. Not comparable to Phase 10C.",
    }
    with running.open("x", encoding="utf-8") as handle:
        json.dump(record, handle)
    cap = writer = predictor = None
    published = False
    try:
        record["input"] = file_identity(source)
        cap = open_capture(source)
        meta = video_metadata(cap)
        record["source"] = meta
        record["source"]["container_extension"] = source.suffix.lower()
        dims = (meta["width"] * (2 if mode == "compare" else 1), meta["height"])
        record["layout"] = "D2_LEFT_S1_RIGHT_UNSCALED" if mode == "compare" else "ORIGINAL_CANVAS"
        if meta["frame_count"] is None:
            record["warnings"].append("UNKNOWN_SOURCE_COUNT: early EOF cannot be distinguished")
        if attribution["source_type"] == "UNSPECIFIED":
            record["warnings"].append("SOURCE_LICENSE_UNSPECIFIED: not ready for publication")
        writer = open_writer(partial, meta["fps"], dims)
        record["encoder_backend"] = writer.getBackendName()
        predictor = predictor_factory(root, mode, device)
        record["inference"] = predictor.evidence()
        write_json(running, record)
        source_digest = hashlib.sha256()
        previous_timestamp = None
        positive_timestamps = 0
        elapsed = 0.0
        started = time.perf_counter()
        while max_frames is None or record["processed_frames"] < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            record["decoded_frames"] += 1
            if frame is None or frame.shape != (meta["height"], meta["width"], 3):
                raise VideoRuntimeError("DECODE_SHAPE_CHANGED_OR_EMPTY")
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
            if math.isfinite(timestamp) and timestamp > 0:
                positive_timestamps += 1
                if previous_timestamp is not None and not math.isclose(
                    timestamp - previous_timestamp,
                    1000 / meta["fps"],
                    rel_tol=0.05,
                    abs_tol=2.0,
                ):
                    raise VideoRuntimeError("VARIABLE_OR_DISCONTINUOUS_TIMESTAMPS_UNSUPPORTED")
            previous_timestamp = timestamp if math.isfinite(timestamp) else None
            source_digest.update(frame.tobytes())
            predictions = predictor.predict(frame)
            rate = record["processed_frames"] / elapsed if elapsed > 0 else None
            rendered = render_frame(frame, predictions, mode, meta["fps"], rate, fps_overlay)
            if rendered.shape != (dims[1], dims[0], 3) or rendered.dtype != np.uint8:
                raise VideoRuntimeError("RENDERED_FRAME_CONTRACT_MISMATCH")
            record["processed_frames"] += 1
            writer.write(rendered)
            record["encoded_frames"] += 1  # Verified independently after finalization.
            elapsed = time.perf_counter() - started
        writer.release()
        writer = None
        elapsed = time.perf_counter() - started
        cap.release()
        cap = None
        count = record["processed_frames"]
        if count == 0:
            raise VideoRuntimeError("EMPTY_VIDEO: no decoded frame")
        expected = meta["frame_count"]
        if expected is not None and count != (
            min(expected, max_frames) if max_frames else expected
        ):
            raise VideoRuntimeError("PREMATURE_OR_INCONSISTENT_DECODE: frame count differs")
        if positive_timestamps == 0:
            record["warnings"].append("SOURCE_TIMESTAMPS_UNAVAILABLE: CFR metadata assumed")
        record["output_video"] = verify_output(
            partial, frames=count, fps=meta["fps"], dimensions=dims
        )
        record["processing_elapsed_seconds"] = elapsed
        record["processing_throughput_fps"] = count / elapsed
        record["source_decoded_sequence_sha256"] = source_digest.hexdigest()
        record["input_unchanged"] = file_identity(source) == record["input"]
        if not record["input_unchanged"]:
            raise VideoRuntimeError("INPUT_CHANGED_DURING_EXECUTION")
        record["inference"] = predictor.evidence()
        record["output"] = {**file_identity(partial), "filename": output.name}
        record["status"] = "ENGINEERING_PREFIX_COMPLETE" if max_frames else "COMPLETE"
        record["full_source_processed"] = max_frames is None
        write_json(running, record)
        partial.replace(output)
        published = True
        running.replace(Path(str(output) + ".provenance.json"))
        return record
    except BaseException as exc:
        if writer is not None:
            writer.release()
        if cap is not None:
            cap.release()
        if predictor is not None:
            record["inference"] = predictor.evidence()
        partial.unlink(missing_ok=True)
        if published:
            output.unlink(missing_ok=True)
        record["status"] = "FAILED"
        record["failure_type"] = type(exc).__name__
        record["failure"] = (
            str(exc)
            if isinstance(exc, VideoRuntimeError)
            else "Execution interrupted or failed; see exception type"
        )
        write_json(running, record)
        running.replace(Path(str(output) + ".failed.provenance.json"))
        raise VideoRuntimeError(record["failure"]) from exc
    finally:
        if predictor is not None:
            predictor.close()
