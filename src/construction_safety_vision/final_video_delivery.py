"""Phase 13B delivery evidence; these helpers never load or execute a model."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import cv2
import numpy as np

from construction_safety_vision.data.canonical import scan_for_sensitive
from construction_safety_vision.delivery_status import build_status, historical_changes
from construction_safety_vision.provenance import ProvenanceRecord, sha256_file
from construction_safety_vision.video_delivery import APPROVED_RUNTIME_COMMIT
from construction_safety_vision.video_models import IDENTITIES, frozen_settings
from construction_safety_vision.video_runtime import (
    file_identity,
    open_capture,
    verify_output,
    write_json,
)

REPORT = "reports/final_real_video_demo.json"
MARKDOWN = "reports/final_real_video_demo.md"
PROVENANCE = "reports/final_real_video_demo.provenance.json"
SOURCE = "reports/final_real_video_source.json"
PLAN = "reports/final_real_video_execution_plan.json"
REVIEW = "reports/final_real_video_temporal_review.json"
SCREENSHOTS = "reports/final_real_video_screenshots.json"
FIGURES = "reports/figures/final_video"
OUTPUT = "outputs/final_construction_ppe_compare.mp4"
COMPLETE = "FINAL_REAL_VIDEO_DEMO_COMPLETE"
BOUNDARIES = (
    "holdout_accessed",
    "training",
    "model_selection_changed",
    "threshold_changed",
    "tracking",
)


def read(root: Path, relative: str) -> dict[str, Any]:
    """Read an explicitly named delivery text record."""
    return json.loads((root / relative).read_text(encoding="utf-8"))


def write_jpeg(path: Path, pixels: np.ndarray) -> None:
    """Use Python filesystem I/O because OpenCV imwrite rejects Windows Unicode paths."""
    ok, encoded = cv2.imencode(".jpg", pixels, [cv2.IMWRITE_JPEG_QUALITY, 94])
    if not ok:
        raise ValueError("Screenshot encode failed")
    path.write_bytes(encoded.tobytes())


def extract(root: Path) -> None:
    """Extract the six frozen frame indices and complete one-second contact coverage."""
    plan = read(root, PLAN)
    run = read(root, OUTPUT + ".provenance.json")
    if run["status"] != "COMPLETE" or file_identity(root / OUTPUT) != run["output"]:
        raise ValueError("Final complete output required")
    chosen = plan["temporal_review"]["screenshot_frame_indices_zero_based"]
    fps = run["source"]["fps"]
    contacts = {round(t * fps) for t in range(math.ceil(run["output_video"]["duration_seconds"]))}
    directory = root / FIGURES
    directory.mkdir(parents=True, exist_ok=True)
    scratch = root / "outputs/phase13b_review"
    scratch.mkdir(parents=True, exist_ok=True)
    cap = open_capture(root / OUTPUT)
    index, small, screenshots = 0, [], []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index in chosen:
                footer = np.full((110, frame.shape[1], 3), 245, dtype=np.uint8)
                lines = [
                    f"t={index / fps:.3f}s | frame={index} (zero-based) | D2 left / S1 right",
                    "Source: Frank Vincentz, Mdina construction 01 (1) ies | CC BY-SA 3.0",
                    "Model overlays; audio omitted. Source/license URLs: final_real_video_demo.md",
                ]
                for row, text in enumerate(lines):
                    cv2.putText(
                        footer,
                        text,
                        (20, 30 + 32 * row),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.85,
                        (30, 30, 30),
                        2,
                        cv2.LINE_AA,
                    )
                path = directory / f"frame_{index:06d}.jpg"
                write_jpeg(path, np.vstack((frame, footer)))
                screenshots.append(
                    {
                        "path": path.relative_to(root).as_posix(),
                        "frame_zero_based": index,
                        "timestamp_seconds": index / fps,
                        "identity": file_identity(path),
                        "decoded_output_frame_sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
                        "modification": "Original output raster plus external "
                        "credit/timestamp footer; JPEG 94",
                        "license": "CC BY-SA 3.0",
                    }
                )
            if index in contacts:
                tile = cv2.resize(frame, (768, 216))
                label = np.full((28, 768, 3), 245, dtype=np.uint8)
                cv2.putText(
                    label,
                    f"t={index / fps:.2f}s",
                    (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    1,
                )
                small.append(np.vstack((tile, label)))
            index += 1
    finally:
        cap.release()
    for start in range(0, len(small), 10):
        write_jpeg(scratch / f"timeline_{start:03d}.jpg", np.vstack(small[start : start + 10]))
    if [item["frame_zero_based"] for item in screenshots] != chosen:
        raise ValueError("Frozen screenshots missing")
    write_json(
        root / SCREENSHOTS,
        {
            "output_sha256": run["output"]["sha256"],
            "contact_samples": len(small),
            "screenshots": screenshots,
        },
    )


def validate_payload(p: dict[str, Any], settings: dict[str, Any]) -> list[str]:
    """Reject delivery claims that contradict their recorded execution evidence."""
    problems = []
    run, source, plan = p["execution"], p["source"], p["execution_plan"]
    meta, out = run["source"], run["output_video"]
    count, fps = source["actual_video_frame_count"], meta["fps"]
    if p["classification"] != COMPLETE or any(p[k] is not False for k in BOUNDARIES):
        problems.append("Phase boundary mismatch")
    if any(run[k] is not False for k in BOUNDARIES if k != "model_selection_changed"):
        problems.append("Runtime phase boundary mismatch")
    ledger = p["execution_ledger"]
    if (
        ledger["attempt"] != 1
        or ledger["engineering_retries"] != 0
        or p["engineering_retry_count"] != 0
        or ledger["plan_sha256"] != p["execution_plan_sha256"]
        or plan["status"] != "FROZEN_BEFORE_INFERENCE"
        or plan["frozen_at_utc"] >= ledger["started_at_utc"]
        or plan["selected_range"]["start_frame_inclusive"] != 0
        or plan["selected_range"]["end_frame_exclusive"] != count
        or plan["max_frames"] is not None
        or plan["mode"] != "compare"
        or plan["device"] != "cuda"
        or plan["output"] != OUTPUT
    ):
        problems.append("Execution plan/accounting mismatch")
    if (
        p["media_license"] != source["license"]
        or p["media_license"] != "CC BY-SA 3.0"
        or p["code_license"] != "AGPL-3.0"
        or p["audio_not_preserved"] is not True
    ):
        problems.append("Rights/audio mismatch")
    if source["source_type"] != "EXTERNAL_REAL_VIDEO" or any(
        not source.get(k)
        for k in (
            "title",
            "creator",
            "license",
            "license_url",
            "url",
            "retrieval_date",
            "original_filename",
        )
    ):
        problems.append("Real source/license metadata incomplete")
    for key in ("url", "license_url", "direct_asset_url"):
        url = urlsplit(source[key])
        if url.scheme != "https" or not url.hostname or url.query or url.fragment or url.username:
            problems.append("Non-public source URL")
    if source["decode"]["cfr_contract"] != "PASS" or source["decode"]["frames"] != count:
        problems.append("Source integrity mismatch")
    normalized = source["normalization"]
    if (
        normalized["pixel_sequence_identical"] is not True
        or normalized["frame_insertion_or_deletion"] is not False
        or normalized["pixel_sequence_sha256"] != run["source_decoded_sequence_sha256"]
    ):
        problems.append("Lossless source identity mismatch")
    if (
        any(meta[key] != source["video"][key] for key in ("width", "height", "fps"))
        or any(run["source_attribution"][key] != source[key] for key in run["source_attribution"])
        or p["screenshots"]["output_sha256"] != run["output"]["sha256"]
    ):
        problems.append("Source/attribution correspondence mismatch")
    if (
        run["status"] != "COMPLETE"
        or run["full_source_processed"] is not True
        or run["max_frames"] is not None
        or run["input_unchanged"] is not True
    ):
        problems.append("Incomplete final execution")
    if (
        run["mode"] != "compare"
        or run["requested_device"] != "cuda"
        or run["layout"] != "D2_LEFT_S1_RIGHT_UNSCALED"
        or run["inference"]["settings"] != settings
        or plan["runtime_settings"] != settings
    ):
        problems.append("Frozen runtime contract mismatch")
    if (
        not math.isfinite(fps)
        or fps <= 0
        or count / fps < 30
        or any(run[k] != count for k in ("decoded_frames", "processed_frames", "encoded_frames"))
        or out["verified_decoded_frames"] != count
        or out["frame_count"] != count
        or meta["frame_count"] != count
        or normalized["frames"] != count
    ):
        problems.append("Frame accounting/duration mismatch")
    if (
        (out["width"], out["height"]) != (2 * meta["width"], meta["height"])
        or out["fps"] != fps
        or not math.isclose(out["duration_seconds"], count / fps)
    ):
        problems.append("Output metadata mismatch")
    if (
        run["purpose"] != "DEMO_RUNTIME_MEASUREMENT"
        or run["processing_elapsed_seconds"] <= 0
        or not math.isclose(
            run["processing_throughput_fps"], count / run["processing_elapsed_seconds"]
        )
    ):
        problems.append("Demo timing mismatch")
    for name, identity in IDENTITIES.items():
        model = run["inference"]["models"][name]
        if (
            (model["identity"]["sha256"], model["identity"]["bytes"]) != identity
            or model["predict_invocations"] != count
            or model["frames_completed"] != count
            or model["effective_dtypes"] != ["torch.float32"]
            or model["effective_batch_sizes"] != [1]
        ):
            problems.append("Frozen model identity/invocation mismatch")
    if (
        run["input"] != normalized["identity"]
        or run["input"] != plan["input_identity"]
        or source["identity"] != plan["source_identity"]
    ):
        problems.append("Plan/input mismatch")
    for identity in (source["identity"], normalized["identity"], run["output"]):
        if (
            len(identity["sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in identity["sha256"])
            or identity["bytes"] <= 0
        ):
            problems.append("Invalid artifact identity")
    shots = p["screenshots"]["screenshots"]
    chosen = plan["temporal_review"]["screenshot_frame_indices_zero_based"]
    if [s["frame_zero_based"] for s in shots] != chosen or len(shots) != 6:
        problems.append("Screenshot selection mismatch")
    for shot in shots:
        if not 0 <= shot["frame_zero_based"] < count or not math.isclose(
            shot["timestamp_seconds"], shot["frame_zero_based"] / fps
        ):
            problems.append("Screenshot timestamp mismatch")
    observations = p["temporal_review"]["observations"]
    if len(observations) != 3:
        problems.append("Temporal coverage mismatch")
    else:
        for i, observation in enumerate(observations):
            if (
                observation["frames_zero_based"] != chosen[2 * i : 2 * i + 2]
                or observation["timestamps_seconds"] != [n / fps for n in chosen[2 * i : 2 * i + 2]]
                or not observation["visible_behavior"]
                or not observation["systems_classes"]
            ):
                problems.append("Temporal evidence mismatch")
    if (
        p["gap_002_status"] != "RESOLVED"
        or p["gap_010_status"] != "PARTIALLY_RESOLVED"
        or p["assignment_video_requirement"] != "COMPLETE"
        or p["storage"]["classification"] != "EXTERNAL_DELIVERY_ARTIFACT"
        or p["storage"]["public_url"] is not None
    ):
        problems.append("Delivery status mismatch")
    return problems


def validate(root: Path, *, with_video: bool = False) -> list[str]:
    """Check committed metadata offline; optionally hash/decode only the named output."""
    p = read(root, REPORT)
    problems = validate_payload(p, frozen_settings(root))
    if "CSVISION_ALLOW_TEST_SPLIT" in os.environ or historical_changes(root):
        problems.append("Scientific lock mismatch")
    for key, path in (
        ("execution_plan", PLAN),
        ("source", SOURCE),
        ("temporal_review", REVIEW),
        ("screenshots", SCREENSHOTS),
    ):
        if p[key] != read(root, path):
            problems.append(f"Evidence snapshot mismatch: {path}")
    if p["execution_plan_sha256"] != sha256_file(root / PLAN):
        problems.append("Execution plan digest mismatch")
    if (root / MARKDOWN).read_text(encoding="utf-8") != report_text(p):
        problems.append("Rendered report differs from evidence")
    for shot in p["screenshots"]["screenshots"]:
        if file_identity(root / shot["path"]) != shot["identity"]:
            problems.append("Committed screenshot identity mismatch")
    for name, digest in p["execution"]["implementation_sha256"].items():
        path = "src/construction_safety_vision/" + name
        approved = subprocess.check_output(
            ["git", "show", f"{APPROVED_RUNTIME_COMMIT}:{path}"], cwd=root
        )
        if digest != sha256_file(root / path) or digest != hashlib.sha256(approved).hexdigest():
            problems.append("Approved runtime changed")
    if read(root, "reports/delivery_gap_resolution_status.json") != build_status(root):
        problems.append("Live delivery tracker mismatch")
    provenance = read(root, PROVENANCE)
    for entry in provenance["inputs"] + provenance["outputs"]:
        if entry["path"].startswith("outputs/"):
            continue
        if sha256_file(root / entry["path"]) != entry["sha256"]:
            problems.append(f"Provenance mismatch: {entry['path']}")
    if with_video:
        run = p["execution"]
        if file_identity(root / OUTPUT) != run["output"]:
            problems.append("Final MP4 identity mismatch")
        verify_output(
            root / OUTPUT,
            frames=run["processed_frames"],
            fps=run["source"]["fps"],
            dimensions=(run["output_video"]["width"], run["output_video"]["height"]),
        )
        capture = open_capture(root / OUTPUT)
        try:
            for shot in p["screenshots"]["screenshots"]:
                capture.set(cv2.CAP_PROP_POS_FRAMES, shot["frame_zero_based"])
                ok, frame = capture.read()
                if (
                    not ok
                    or hashlib.sha256(frame.tobytes()).hexdigest()
                    != shot["decoded_output_frame_sha256"]
                ):
                    problems.append("Screenshot/output timeline mismatch")
        finally:
            capture.release()
    for path in (REPORT, MARKDOWN, PROVENANCE, PLAN, SOURCE, REVIEW, SCREENSHOTS):
        problems.extend(scan_for_sensitive((root / path).read_text(encoding="utf-8")))
    return problems


def report_text(p: dict[str, Any]) -> str:
    """Render the delivery report from measured evidence and explicit review notes."""
    run, source = p["execution"], p["source"]
    out, device = run["output_video"], run["inference"]["device"]
    lines = [
        "# Phase 13B - final real-video PPE demonstration",
        "",
        f"**{COMPLETE}**. Assignment real-video requirement: **COMPLETE**.",
        "",
        "**NO TRACKING. NO MODEL/TUNING CHANGES.** Final test remains observed and locked. "
        "This is application delivery, with no new scientific evaluation or benchmark.",
        "",
        "## Source and attribution",
        "",
        f"[{source['title']}]({source['url']}) by **{source['creator']}**, "
        f"[{source['license']}]({source['license_url']}); retrieved {source['retrieval_date']}.",
        "",
        "Original footage, derived MP4 and six screenshots retain **CC BY-SA 3.0**. "
        "The project applies that same license to its contributions to these media. "
        "Code remains **AGPL-3.0**. No endorsement by the creator is implied. "
        "Include this attribution and license link beside any delivered video or screenshot.",
        "",
        f"Modifications: {source['modifications']}",
        "",
        "[Full acquisition record](final_real_video_source.json) includes the public asset URL, "
        "original filename, hashes, container metadata and full decode evidence. "
        "[License terms](https://creativecommons.org/licenses/by-sa/3.0/) permit adaptation "
        "with attribution and ShareAlike; attribution is adjacent to the media.",
        "",
        "## Input integrity and pre-inference plan",
        "",
        "The original WebM declares an estimated 2616 frames (104.64 s), while full sequential "
        "decode yields **2613 frames / 104.52 s**, spaced 40 ms apart. The estimate is retained "
        "as container metadata, not substituted for the measured video count. "
        "Before inference, all decoded frames were encoded as lossless FFV1/AVI without audio. "
        "Re-decoding verified an identical SHA-256 over the entire BGR pixel sequence. "
        "No frame was inserted/deleted, no image resized/cropped, and the approved runtime "
        "was unchanged. The silent input has an exact count. No excerpt was selected.",
        "",
        f"Original: {source['identity']['bytes']} bytes, SHA-256 `{source['identity']['sha256']}`. "
        f"Lossless input: {run['input']['bytes']} bytes, SHA-256 `{run['input']['sha256']}`.",
        "",
        "An earlier Welsh Government candidate was rejected before model execution for "
        "timestamp discontinuities and count mismatch; "
        "[rejection evidence](final_real_video_source_rejection.json). This was source eligibility "
        "screening, with zero model calls, before the final source/plan was frozen.",
        "",
        f"[Frozen plan](final_real_video_execution_plan.json): SHA-256 "
        f"`{p['execution_plan_sha256']}`. Full source range [0, 2613), compare, CUDA, "
        "one CLI execution, no frame cap, **zero engineering retries**.",
        "",
        "## Frozen runtime and actual execution",
        "",
        "D2: YOLO11n bounding boxes (left). S1: YOLO11n-seg instance masks (right). "
        "Both consume the same original raster, imgsz 768, conf 0.25, NMS IoU 0.70, "
        "max_det 300, batch 1, FP32, augment/TTA false; S1 retina_masks true. "
        "Checkpoint hashes and actual float32/device/batch evidence are in "
        "[the JSON report](final_real_video_demo.json).",
        "",
        f"Device: **{device['name']}**; torch {device['torch']}, CUDA build "
        f"{device['cuda_build']}, OpenCV {run['opencv']}. "
        "See JSON for the exact framework/environment identity.",
        "",
        "| Model | predict invocations | Completed frames | SHA-256 |",
        "| --- | ---: | ---: | --- |",
    ]
    for name, model in run["inference"]["models"].items():
        lines.append(
            f"| {name} | {model['predict_invocations']} | {model['frames_completed']} | "
            f"`{model['identity']['sha256']}` |"
        )
    lines += [
        "",
        "## Output and demo throughput",
        "",
        f"Local deliverable: `{OUTPUT}`. MP4/mp4v, **{out['width']} x {out['height']}**, "
        f"**{out['frame_count']} frames / {out['fps']:.0f} FPS / "
        f"{out['duration_seconds']:.2f} s**. "
        "D2 left / S1 right, both unscaled; full decode verification and COMPLETE sidecar. "
        "Audio was **not preserved**.",
        "",
        f"Output: **{run['output']['bytes']} bytes**, SHA-256 `{run['output']['sha256']}`.",
        "",
        f"**DEMO_RUNTIME_MEASUREMENT**: {run['processing_elapsed_seconds']:.6f} seconds; "
        f"{run['processing_throughput_fps']:.6f} processing FPS. "
        f"Source playback: {out['fps']:.0f} FPS. "
        + (
            "Processing was slower than playback."
            if run["processing_throughput_fps"] < out["fps"]
            else "Processing was faster than playback."
        ),
        "",
        run["timing_contract"],
        "",
        "This number includes full-resolution lossless input decoding, pixel hashing, both "
        "models and side-by-side output encoding. It does not isolate inference latency "
        "and is not comparable to Phase 10C. No controlled repetitions occurred.",
        "",
        "Predict counts exclude the one native framework warmup forward per model. "
        "Each model records 2614 forward calls including that warmup, and 2613 predict calls.",
        "",
        "## Temporal qualitative review",
        "",
        p["execution_plan"]["temporal_review"]["method"],
        "",
        "Six frames were frozen before inference, at each third midpoint and the nearest "
        "integer-frame offset to 0.5 s (12 frames = 0.48 s at 25 FPS). The overview contains "
        f"{p['screenshots']['contact_samples']} samples at one-second intervals. "
        "These are descriptive visual observations, not annotated precision/recall. "
        "No cause, statistical model ordering or object identity across frames is inferred.",
        "",
    ]
    for index, obs in enumerate(p["temporal_review"]["observations"]):
        times = " / ".join(f"{t:.2f}s" for t in obs["timestamps_seconds"])
        lines += [
            f"### {obs['third']} third - {times}",
            "",
            f"**{obs['evidence_level']}** - {obs['systems_classes']}. " + obs["visible_behavior"],
            "",
            obs["interpretation"],
            "",
        ]
        for shot in p["screenshots"]["screenshots"][2 * index : 2 * index + 2]:
            relative = shot["path"].removeprefix("reports/")
            lines += [f"![D2/S1 at {shot['timestamp_seconds']:.2f}s]({relative})", ""]
    lines += [
        "## Delivery and limitations",
        "",
        "**EXTERNAL_DELIVERY_ARTIFACT**: the MP4 and 3.67 GB lossless intermediate remain "
        "git-ignored under outputs. The MP4 is a locally available academic deliverable; "
        "no external upload or public playback URL exists. Deliver the MP4 with this attribution "
        "report and its provenance sidecar; verify its recorded SHA-256 before submission. "
        "Committed screenshots are the explicitly approved small visual evidence exception.",
        "",
        "**GAP-002: OPEN -> RESOLVED. GAP-010: PARTIALLY_RESOLVED -> PARTIALLY_RESOLVED.** "
        "Public MP4 distribution and public checkpoint acquisition remain unavailable. "
        "No unrelated gap changes. This demonstrates one real clip on one local GPU; "
        "no deployment safety, generalization, calibrated compliance, tracking, real-time "
        "capability or codec portability is established. The footage is not newly annotated. "
        "Labels can overlap in dense regions; translucent masks are model predictions, "
        "not ground truth.",
        "",
        "A portable [attribution notice](../delivery/FINAL_VIDEO_ATTRIBUTION.md) is also copied "
        "beside the local MP4 as `final_construction_ppe_compare.ATTRIBUTION.md`. "
        "Include it with any copy. No public upload was performed.",
        "",
        "## Verification and delivery tooling",
        "",
        "```text",
        "uv run python scripts/final_video_delivery.py validate",
        "uv run python scripts/final_video_delivery.py validate --with-video",
        "uv run pytest tests/test_final_video_delivery.py --metadata-only",
        "```",
        "",
        "Validation never invokes a model. The optional media check fully decodes the named MP4 "
        "and checks each screenshot's source-frame hash. `extract` consumes only the existing "
        "final output; `build` consumes recorded evidence and review notes. "
        "The executed CLI command is recorded in the JSON execution ledger; do not rerun it "
        "as a validation step. Scientific and earlier phase reports remain unchanged.",
        "",
    ]
    return "\n".join(lines)


def build(root: Path) -> None:
    """Assemble measured runtime evidence and reviewed observations, without inference."""
    run = read(root, OUTPUT + ".provenance.json")
    ledger = read(root, "outputs/phase13b_execution_ledger.json")
    p = {
        "schema_version": 1,
        "phase": "13B",
        "classification": COMPLETE,
        "source": read(root, SOURCE),
        "execution_plan": read(root, PLAN),
        "execution_plan_sha256": sha256_file(root / PLAN),
        "execution": run,
        "execution_ledger": ledger,
        "engineering_retry_count": 0,
        "temporal_review": read(root, REVIEW),
        "screenshots": read(root, SCREENSHOTS),
        "gap_002_status": "RESOLVED",
        "gap_010_status": "PARTIALLY_RESOLVED",
        "assignment_video_requirement": "COMPLETE",
        "audio_not_preserved": True,
        "storage": {
            "classification": "EXTERNAL_DELIVERY_ARTIFACT",
            "local_path": OUTPUT,
            "public_url": None,
            "distribution": "LOCAL_ONLY; no upload authorized",
            "delivery_instructions": "Deliver MP4 together with final_real_video_demo.md "
            "and its provenance/attribution. Verify recorded SHA-256 before submission. "
            "Public URL and checkpoint retrieval remain pending; GAP-010 stays partial.",
        },
        "media_license": "CC BY-SA 3.0",
        "code_license": "AGPL-3.0",
        "delivery_tooling_note": "First screenshot extraction failed at OpenCV imwrite on "
        "a Unicode Windows directory. Replaced only JPEG file I/O with imencode/Path.write_bytes; "
        "extracted the same frozen frame indices from the existing MP4. No inference retry.",
        **dict.fromkeys(BOUNDARIES, False),
    }
    problems = validate_payload(p, frozen_settings(root))
    if problems or ledger["plan_sha256"] != p["execution_plan_sha256"]:
        raise ValueError(problems or "Plan changed after execution start")
    write_json(root / REPORT, p)
    (root / MARKDOWN).write_text(report_text(p), encoding="utf-8", newline="\n")
    (root / "outputs/final_construction_ppe_compare.ATTRIBUTION.md").write_bytes(
        (root / "delivery/FINAL_VIDEO_ATTRIBUTION.md").read_bytes()
    )
    write_json(root / "reports/delivery_gap_resolution_status.json", build_status(root))
    provenance = ProvenanceRecord.create(
        "final_real_video_demonstration",
        phase=13,
        repo_root=root,
        config=p["execution_plan"],
        details={
            "classification": COMPLETE,
            "approved_runtime_commit": APPROVED_RUNTIME_COMMIT,
            "model_invocations": {
                name: run["inference"]["models"][name]["predict_invocations"] for name in IDENTITIES
            },
            **dict.fromkeys(BOUNDARIES, False),
        },
    )
    inputs = [
        SOURCE,
        PLAN,
        REVIEW,
        SCREENSHOTS,
        "reports/final_real_video_attribution.json",
        "reports/final_real_video_source_rejection.json",
        "scripts/prepare_final_video.py",
        "scripts/final_video_delivery.py",
        "src/construction_safety_vision/final_video_delivery.py",
        "src/construction_safety_vision/delivery_status.py",
        "tests/test_final_video_delivery.py",
        "src/construction_safety_vision/video_delivery.py",
        "tests/test_video_runtime_artifacts.py",
        "README.md",
        "reports/README.md",
        "reports/roadmap.md",
        "delivery/README.md",
        "delivery/REPRODUCTION.md",
        "delivery/VIDEO.md",
        "delivery/LICENSING.md",
        "delivery/FINAL_VIDEO_ATTRIBUTION.md",
        "reports/figures/final_video/README.md",
        "outputs/phase13b_execution_ledger.json",
        OUTPUT + ".provenance.json",
    ]
    for path in inputs:
        provenance.add_input(root / path, relative_to=root)
    for path in [
        REPORT,
        MARKDOWN,
        "reports/delivery_gap_resolution_status.json",
        *[shot["path"] for shot in p["screenshots"]["screenshots"]],
    ]:
        provenance.add_output(root / path, relative_to=root)
    provenance.write_json(root / PROVENANCE)
