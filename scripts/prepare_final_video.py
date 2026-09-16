"""Acquire and inspect the single Phase 13B source; freeze before model execution."""

import argparse
import hashlib
import math
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from construction_safety_vision.video_models import frozen_settings
from construction_safety_vision.video_runtime import (
    file_identity,
    open_capture,
    source_metadata,
    video_metadata,
    write_json,
)

ROOT = Path(__file__).resolve().parents[1]
NAME = (
    "Malta_-_Mdina_-_Lorenzo_Calleja_ditch_-_Il-Foss_tal-Imdina_"
    "%28construction%29_01_%281%29_ies.webm"
)
PAGE = "https://commons.wikimedia.org/wiki/File:" + NAME
ASSET = "https://upload.wikimedia.org/wikipedia/commons/3/35/" + NAME
SOURCE = "outputs/phase13b_source/mdina_construction.webm"
NORMALIZED = "outputs/phase13b_source/mdina_construction_lossless.avi"
ACQUISITION = "reports/final_real_video_source.json"
PLAN = "reports/final_real_video_execution_plan.json"
ATTRIBUTION = "reports/final_real_video_attribution.json"


def prepare() -> None:
    """Full sequential decode and source preview, without a model or dataset read."""
    import json

    if (ROOT / ACQUISITION).exists():
        raise ValueError("Acquisition already recorded; do not replace the selected source")
    path = ROOT / SOURCE
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        request = urllib.request.Request(ASSET, headers={"User-Agent": "CSVisionAcademicDemo/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response, path.open("xb") as out:
            while chunk := response.read(1 << 20):
                out.write(chunk)
    cap = open_capture(path)
    meta = video_metadata(cap)
    count, previous, differences, previews = 0, None, [], []
    decoded_hash = hashlib.sha256()
    normalized_path = ROOT / NORMALIZED
    if normalized_path.exists():
        raise ValueError("Lossless input already exists; inspect acquisition state")
    writer = cv2.VideoWriter(
        str(normalized_path),
        cv2.CAP_FFMPEG,
        cv2.VideoWriter_fourcc(*"FFV1"),
        meta["fps"],
        (meta["width"], meta["height"]),
    )
    if not writer.isOpened():
        raise ValueError("Lossless FFV1 writer unavailable")
    indices = [round(meta["frame_count"] * f) for f in (1 / 6, 1 / 2, 5 / 6)]
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape != (meta["height"], meta["width"], 3):
                raise ValueError("Source frame dimensions changed")
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
            if not math.isfinite(timestamp):
                raise ValueError("Source timestamp unavailable")
            if previous is not None:
                delta = timestamp - previous
                if not math.isclose(delta, 1000 / meta["fps"], rel_tol=0.05, abs_tol=2.0):
                    raise ValueError("Source fails approved runtime CFR contract")
                differences.append(delta)
            previous = timestamp
            decoded_hash.update(frame.tobytes())
            writer.write(frame)
            if count in indices:
                preview = cv2.resize(frame, (640, 360))
                cv2.putText(
                    preview,
                    f"Source only: {count / meta['fps']:.3f}s",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
                previews.append(preview)
            count += 1
    finally:
        cap.release()
        writer.release()
    if count / meta["fps"] < 30:
        raise ValueError("Too short source")
    normalized_capture = open_capture(normalized_path)
    normalized_meta = video_metadata(normalized_capture)
    normalized_hash = hashlib.sha256()
    verified = 0
    try:
        while True:
            ok, frame = normalized_capture.read()
            if not ok:
                break
            normalized_hash.update(frame.tobytes())
            verified += 1
    finally:
        normalized_capture.release()
    if (
        verified != count
        or normalized_meta["frame_count"] != count
        or normalized_hash.digest() != decoded_hash.digest()
    ):
        raise ValueError("Lossless input changed pixels or frame count")
    attribution = source_metadata(
        {
            "source_type": "EXTERNAL_REAL_VIDEO",
            "title": "Malta - Mdina - Lorenzo Calleja ditch - "
            "Il-Foss tal-Imdina (construction) 01 (1) ies",
            "creator": "Frank Vincentz",
            "url": PAGE,
            "license": "CC BY-SA 3.0",
            "retrieval_date": datetime.now(UTC).date().isoformat(),
            "modifications": "Full source decoded to lossless FFV1/AVI without audio; "
            "D2 box and S1 mask overlays; side-by-side unscaled panels; "
            "audio omitted; timestamp/credit footer added only to extracted screenshots.",
        }
    )
    record = {
        **attribution,
        "direct_asset_url": ASSET,
        "license_url": "https://creativecommons.org/licenses/by-sa/3.0/",
        "license_evidence": "Commons creator own-work CC BY-SA 3.0 grant; "
        "this option selected from the explicit dual-license grant.",
        "original_filename": urllib.parse.unquote(NAME).replace("_", " "),
        "local_path": SOURCE,
        "identity": file_identity(path),
        "container": "WebM (VP8/Vorbis)",
        "video": meta,
        "actual_video_frame_count": count,
        "actual_video_duration_seconds": count / meta["fps"],
        "decode": {
            "status": "COMPLETE",
            "frames": count,
            "cfr_contract": "PASS",
            "timestamp_delta_min_ms": min(differences),
            "timestamp_delta_max_ms": max(differences),
        },
        "audio_not_preserved": True,
        "acquired_at_utc": datetime.now(UTC).isoformat(),
        "normalization": {
            "reason": "WebM CAP_PROP_FRAME_COUNT estimates 2616 from container duration; "
            "2613 regularly timestamped video frames decode. Use exact-count lossless AVI "
            "to satisfy the unchanged approved runtime contract without frame insertion/deletion.",
            "codec": "FFV1",
            "local_path": NORMALIZED,
            "identity": file_identity(normalized_path),
            "video": normalized_meta,
            "frames": verified,
            "pixel_sequence_sha256": decoded_hash.hexdigest(),
            "pixel_sequence_identical": True,
            "spatial_transform": "NONE",
            "frame_insertion_or_deletion": False,
        },
    }
    write_json(ROOT / ACQUISITION, record)
    write_json(ROOT / ATTRIBUTION, attribution)
    ok, encoded = cv2.imencode(".jpg", np.vstack(previews))
    if not ok:
        raise ValueError("Source preview encode failed")
    (path.parent / "source_preview.jpg").write_bytes(encoded.tobytes())
    print(json.dumps(record, indent=2))


def freeze() -> None:
    """Freeze the full clip and deterministic temporal review after source-only review."""
    import json

    if (ROOT / PLAN).exists():
        raise ValueError("Execution plan already frozen")
    source = json.loads((ROOT / ACQUISITION).read_text())
    if file_identity(ROOT / SOURCE) != source["identity"]:
        raise ValueError("Source identity changed")
    normalized = source["normalization"]
    if file_identity(ROOT / NORMALIZED) != normalized["identity"]:
        raise ValueError("Normalized input identity changed")
    meta = normalized["video"]
    starts = [round(meta["frame_count"] * f) for f in (1 / 6, 1 / 2, 5 / 6)]
    indices = [index for start in starts for index in (start, start + round(meta["fps"] * 0.5))]
    plan = {
        "phase": "13B",
        "status": "FROZEN_BEFORE_INFERENCE",
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "approved_runtime_commit": "57e6aa8d95eded75ba7416d1001d24dde274ecfd",
        "source_record": ACQUISITION,
        "source_identity": source["identity"],
        "input_identity": normalized["identity"],
        "source_metadata": ATTRIBUTION,
        "input": NORMALIZED,
        "selected_range": {
            "start_frame_inclusive": 0,
            "end_frame_exclusive": meta["frame_count"],
            "duration_seconds": meta["duration_seconds"],
            "selection": "FULL_CLIP_NO_SAMPLING_LOSSLESS_CONTAINER_NORMALIZATION",
        },
        "source_visual_review": "Real construction site with visible workers; "
        "three fixed source-only previews inspected before any inference.",
        "mode": "compare",
        "device": "cuda",
        "output": "outputs/final_construction_ppe_compare.mp4",
        "runtime_settings": frozen_settings(ROOT),
        "max_frames": None,
        "runtime_sha256": {
            name: file_identity(ROOT / "src/construction_safety_vision" / name)["sha256"]
            for name in ("video_runtime.py", "video_models.py", "video_render.py")
        },
        "temporal_review": {
            "method": "Inspect contact sheets at every whole second across the complete output; "
            "inspect fixed pairs at each third midpoint and 0.5 seconds later at full resolution. "
            "Describe one observation per pair including genuine visible failures. "
            "No source replacement, model rerun or screenshot reselection for prediction quality.",
            "contact_sheet_step_seconds": 1,
            "screenshot_frame_indices_zero_based": indices,
            "timestamp_semantics": "zero-based frame index / source FPS",
            "scope": "Systematic sampled qualitative review, not exhaustive error annotation",
        },
        "execution_policy": "ONE_CLI_RUN; retries only after documented engineering failure",
        "holdout_accessed": False,
        "training": False,
        "model_selection_changed": False,
        "threshold_changed": False,
        "tracking": False,
    }
    write_json(ROOT / PLAN, plan)
    print(json.dumps(file_identity(ROOT / PLAN)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "freeze"))
    args = parser.parse_args()
    (prepare if args.action == "prepare" else freeze)()
