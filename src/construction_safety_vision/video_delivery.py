"""Small auditable Phase 13A engineering report, independent of model execution."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from construction_safety_vision.data.canonical import scan_for_sensitive
from construction_safety_vision.delivery_status import build_status, historical_changes
from construction_safety_vision.provenance import ProvenanceRecord, sha256_file
from construction_safety_vision.video_models import IDENTITIES, frozen_settings
from construction_safety_vision.video_runtime import file_identity, verify_output, write_json

COMPLETE = "VIDEO_RUNTIME_FOUNDATION_COMPLETE"
REPORT = "reports/video_runtime_foundation.json"
MARKDOWN = "reports/video_runtime_foundation.md"
PROVENANCE = "reports/video_runtime_foundation.provenance.json"
BASELINE = "fd8c13581e31bbadd4369ae5b0a56f227a03d934"
SOURCES = [
    "src/construction_safety_vision/video_models.py",
    "src/construction_safety_vision/video_runtime.py",
    "src/construction_safety_vision/video_render.py",
    "src/construction_safety_vision/video_fixture.py",
    "src/construction_safety_vision/video_delivery.py",
    "src/construction_safety_vision/delivery_status.py",
    "scripts/run_video_demo.py",
    "scripts/smoke_video_runtime.py",
    "scripts/build_video_runtime_report.py",
    "scripts/validate_video_runtime.py",
    "tests/test_video_runtime.py",
    "tests/test_video_runtime_artifacts.py",
    "delivery/VIDEO.md",
    "configs/video_runtime_smoke.yaml",
    "configs/detector_segmenter_comparison.yaml",
    "delivery/checkpoints.json",
    "reports/final_detector_manifest.json",
    "reports/final_segmenter_manifest.json",
]


def report_text(payload: dict[str, Any]) -> str:
    """Render bounded engineering findings from executed evidence only."""
    tests = payload["synthetic_tests"]
    lines = [
        "# Phase 13A — video runtime foundation",
        "",
        f"**{COMPLETE}**. The final real >=30-second assignment video remains **OPEN**.",
        "",
        "## Architecture and public interface",
        "",
        "[CLI and source attribution](../delivery/VIDEO.md) · "
        "[Runtime](../src/construction_safety_vision/video_runtime.py) · "
        "[Frozen frame adapter](../src/construction_safety_vision/video_models.py)",
        "",
        "The CLI decodes sequential BGR frames with OpenCV/FFmpeg, calls each requested "
        "frozen model once per frame, renders on the original canvas and writes MP4/mp4v. "
        "No new dependency was installed: OpenCV and NumPy were already pinned through "
        "the existing environment. The runtime does not import a dataset or holdout accessor.",
        "",
        "`detector`: D2 boxes/class/confidence. `segmenter`: S1 boxes/class/confidence plus "
        "translucent original-canvas masks. `compare`: D2 left and S1 right, both from the "
        "same frame; no resize, output width = 2 * source width, height unchanged. "
        "Five canonical classes stay separate. No winner or compliance overlay exists.",
        "",
        "## Frozen execution contract",
        "",
        "Public checkpoint metadata must equal the two authoritative freeze manifests. "
        "Only the local frozen copies are loaded after size and SHA-256 checks; missing "
        "or changed bytes fail closed. D2 is YOLO11n; S1 is YOLO11n-seg with training "
        "mask_ratio 4 and overlap_mask false.",
        "",
        "Both use imgsz 768, conf 0.25, NMS IoU 0.70, max_det 300, FP32, batch 1, "
        "augment/TTA false. S1 uses retina_masks true. Settings come from the frozen "
        "Phase 10 operational configuration. Forward hooks verify actual float32, "
        "batch size and device; no threshold or checkpoint override is exposed by the CLI.",
        "",
        "Explicit `cuda` and `cpu` were smoke-tested. Requested CUDA without availability "
        "fails; CPU is never a fallback. CPU support is functional evidence from a one-frame "
        "synthetic smoke, not a throughput equivalence claim.",
        "",
        "## Decode, encode and failure contract",
        "",
        "Every source frame is decoded and processed once in order. Source FPS is copied "
        "to the encoder; inference throughput never determines playback FPS. MP4/mp4v "
        "was exercised on this Windows OpenCV/FFmpeg build; H.264 and cross-platform codec "
        "portability are not claimed. Output is reopened and fully decoded to verify "
        "frame count, dimensions, FPS, approximate duration and nonempty bytes.",
        "",
        "Incomplete media uses an explicit `.incomplete.mp4` name. Normal exceptions and "
        "KeyboardInterrupt remove partial media and leave FAILED provenance. Verified "
        "output is renamed into place only at completion. An explicit engineering frame "
        "cap is ENGINEERING_PREFIX_COMPLETE, never a complete source-video claim. "
        "A hard process kill can leave an incomplete file/STARTED record with stale counts; "
        "a later run refuses to overwrite that attempt silently. Two-file output/sidecar "
        "publication is not a filesystem transaction.",
        "",
        "## Synthetic unit/integration evidence",
        "",
        f"Executed focused suite: **{tests['passed']} passed**, {tests['failures']} failures, "
        f"{tests['errors']} errors, {tests['skipped']} skipped. Real codec round-trips "
        "exercise all three modes with model stubs: six 320x192 frames at 10 FPS. "
        "Tests verify order/content within lossy-codec tolerance, no canvas scaling, "
        "source/output counts and FPS, mask coordinates, strict CLI/settings/identity "
        "checks and injected decode/inference/writer/verification/interruption failures.",
        "",
        "## Minimal real-model smoke",
        "",
        "Predeclared in [video_runtime_smoke.yaml](../configs/video_runtime_smoke.yaml): "
        "one generated three-frame 320x192 MP4 at 10 FPS (0.3 seconds); compare mode "
        "processes all three on CUDA and an explicit one-frame engineering prefix on CPU. "
        "No validation image, external source or holdout content was used. "
        "The fixture is SYNTHETIC_ENGINEERING_FIXTURE and cannot satisfy the assignment video.",
        "",
        "| Device | Frames | D2 calls | S1 calls | Output | Loop seconds | Demo loop FPS |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for run in payload["smoke"]["executions"]:
        models = run["inference"]["models"]
        out = run["output_video"]
        lines.append(
            f"| {run['requested_device']} | {run['processed_frames']} | "
            f"{models['D2']['predict_invocations']} | {models['S1']['predict_invocations']} | "
            f"{out['width']}x{out['height']} / {out['fps']:.1f} FPS | "
            f"{run['processing_elapsed_seconds']:.6f} | "
            f"{run['processing_throughput_fps']:.6f} |"
        )
    lines += [
        "",
        "**Total predict() invocations: D2=4, S1=4.** CUDA records four forward calls "
        "per model including normal framework warmup; CPU records one per model. "
        "Both models emitted predictions on the synthetic shapes, so the real box "
        "and S1 mask render paths executed. These detections have no semantic accuracy "
        "interpretation. Reopened outputs contain exactly three and one frames. "
        "The CPU one-frame prefix cannot establish timestamp cadence and records "
        "that limitation explicitly.",
        "",
        "Hardware: " + payload["smoke"]["executions"][0]["inference"]["device"]["name"] + ". "
        "See JSON for exact torch/CUDA/Ultralytics/OpenCV versions and per-run hashes.",
        "",
        "## FPS semantics",
        "",
        "DEMO_RUNTIME_MEASUREMENT = processed frames / end-to-end loop wall time. "
        "Includes decode, the decoded-frame sequence checksum, preprocessing, inference, "
        "postprocessing, render, encode "
        "and writer flush, including first-call framework setup warmup. Excludes "
        "model loading, initial metadata/identity checks, source/output file hashing, "
        "output verification and sidecar writes. The overlay displays the rate of "
        "previously completed frames; the sidecar contains final elapsed time. "
        "These very short, different-length runs are not a CPU/GPU comparison or "
        "a steady-state speed estimate. They are not comparable to Phase 10C. "
        "No controlled repetitions or memory benchmark occurred.",
        "",
        "## Provenance and known limitations",
        "",
        "Each execution records portable input/output filenames, SHA-256 and bytes, "
        "source attribution, mode, model identity/settings, device/runtime, video "
        "metadata, frame counters, decoded-sequence digest, timing, warnings and "
        "completion status. No username, hostname or personal absolute path is stored. "
        "The local videos and execution sidecars remain ignored; this report commits "
        "their small evidence records and hashes.",
        "",
        "Supported contract: local constant-frame-rate video with even dimensions. "
        "Observed variable/discontinuous timestamps fail; unavailable timestamps "
        "produce a CFR-assumption warning. A known frame-count mismatch fails; "
        "when the decoder has no count, early EOF cannot be distinguished and is "
        "disclosed. Audio, metadata rotation, source timestamp tracks and HDR/color "
        "metadata are not preserved. Headers/FPS text occupy thin overlay strips. "
        "Long-running, high-resolution and real construction footage behavior has "
        "not yet been validated. Checkpoint public retrieval remains pending license review.",
        "",
        "GAP-002 **OPEN**. GAP-010 **PARTIALLY_RESOLVED**: a functional CLI exists; "
        "the final real video/demo and publication do not. Tracking, temporal IDs, "
        "business compliance rules, training, tuning and holdout access: **false**. "
        "Scientific artifacts and Phase 12C evidence remain historical and unchanged.",
        "",
        "## Verification",
        "",
        "```text",
        "uv run pytest tests/test_video_runtime.py --metadata-only",
        "uv run python scripts/validate_video_runtime.py",
        "uv run python scripts/validate_video_runtime.py --with-videos",
        "```",
        "",
        "The validator never executes a model. The optional video check reads only "
        "the named synthetic outputs. The one-time smoke script refuses an existing "
        "ledger; rebuilding this report consumes persisted evidence only.",
        "",
        "API references: [OpenCV VideoCapture]"
        "(https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html) "
        "and [VideoWriter](https://docs.opencv.org/4.x/dd/d9e/classcv_1_1VideoWriter.html). "
        "Runtime behavior was verified against the installed build, not assumed from codec names.",
        "",
    ]
    return "\n".join(lines)


def build_report(root: Path) -> None:
    """Generate evidence from already executed tests/smoke; never run models."""
    smoke = json.loads((root / "outputs/phase13a_smoke/execution.json").read_text())
    suite = ElementTree.parse(root / "outputs/phase13a_unit_tests.xml").getroot().find("testsuite")
    counts = {k: int(suite.attrib[k]) for k in ("tests", "failures", "errors", "skipped")}
    counts["passed"] = counts["tests"] - counts["failures"] - counts["errors"] - counts["skipped"]
    if smoke["status"] != "COMPLETE" or counts["failures"] or counts["errors"]:
        raise ValueError("Cannot claim completion from failed checks")
    payload = {
        "schema_version": 1,
        "phase": "13A",
        "classification": COMPLETE,
        "baseline_commit": BASELINE,
        "synthetic_tests": counts,
        "smoke": smoke,
        "modes": ["detector", "segmenter", "compare"],
        "devices": ["cuda", "cpu"],
        "gap_002_status": "OPEN",
        "gap_010_status": "PARTIALLY_RESOLVED",
        "holdout_accessed": False,
        "training": False,
        "threshold_changed": False,
        "tracking": False,
        "new_scientific_metric": False,
    }
    write_json(root / REPORT, payload)
    (root / MARKDOWN).write_text(report_text(payload), encoding="utf-8", newline="\n")
    write_json(root / "reports/delivery_gap_resolution_status.json", build_status(root))
    provenance = ProvenanceRecord.create(
        "video_runtime_foundation",
        phase=13,
        repo_root=root,
        config=smoke["config"],
        details={
            "classification": COMPLETE,
            "baseline_commit": BASELINE,
            "model_invocations": {"D2": 4, "S1": 4},
            "synthetic_tests": counts,
            "holdout_accessed": False,
        },
    )
    for path in [
        *SOURCES,
        "outputs/phase13a_smoke/execution.json",
        "outputs/phase13a_unit_tests.xml",
    ]:
        provenance.add_input(root / path, relative_to=root)
    for path in [REPORT, MARKDOWN, "reports/delivery_gap_resolution_status.json"]:
        provenance.add_output(root / path, relative_to=root)
    provenance.write_json(root / PROVENANCE)


def validate_report(root: Path, *, with_videos: bool = False) -> list[str]:
    """Verify committed identity/count evidence; optionally recheck only synthetic outputs."""
    problems = []
    p = json.loads((root / REPORT).read_text(encoding="utf-8"))
    expected_settings = frozen_settings(root)
    if (
        p["classification"] != COMPLETE
        or p["gap_002_status"] != "OPEN"
        or p["gap_010_status"] != "PARTIALLY_RESOLVED"
    ):
        problems.append("Completion/gap boundary mismatch")
    if historical_changes(root):
        problems.append("Scientific artifacts changed")
    for field in (
        "holdout_accessed",
        "training",
        "threshold_changed",
        "tracking",
        "new_scientific_metric",
    ):
        if p[field] is not False:
            problems.append(f"Forbidden phase behavior: {field}")
    totals = dict.fromkeys(IDENTITIES, 0)
    if len(p["smoke"]["executions"]) != 2:
        problems.append("Unexpected smoke execution population")
        return problems
    for run, (device, frames) in zip(
        p["smoke"]["executions"], [("cuda", 3), ("cpu", 1)], strict=True
    ):
        if run["requested_device"] != device or run["inference"]["settings"] != expected_settings:
            problems.append("Device/settings mismatch")
        expected_status = "COMPLETE" if device == "cuda" else "ENGINEERING_PREFIX_COMPLETE"
        if run["status"] != expected_status or run["full_source_processed"] != (device == "cuda"):
            problems.append("Smoke completion status mismatch")
        if any(
            run[k] is not False
            for k in ("holdout_accessed", "training", "threshold_changed", "tracking")
        ):
            problems.append("Smoke violated phase boundary")
        if (
            run["source"]["width"],
            run["source"]["height"],
            run["source"]["fps"],
            run["source"]["frame_count"],
        ) != (320, 192, 10.0, 3):
            problems.append("Synthetic source metadata mismatch")
        if run["processing_elapsed_seconds"] <= 0 or not math.isclose(
            run["processing_throughput_fps"], frames / run["processing_elapsed_seconds"]
        ):
            problems.append("Demo timing arithmetic mismatch")
        if any(run[k] != frames for k in ("decoded_frames", "processed_frames", "encoded_frames")):
            problems.append("Frame accounting mismatch")
        out = run["output_video"]
        if (out["width"], out["height"], out["fps"], out["verified_decoded_frames"]) != (
            640,
            192,
            10.0,
            frames,
        ):
            problems.append("Output contract mismatch")
        if set(run["inference"]["models"]) != set(IDENTITIES):
            problems.append("Smoke model set mismatch")
            continue
        for name, state in run["inference"]["models"].items():
            identity = state["identity"]
            if (identity["sha256"], identity["bytes"]) != IDENTITIES[name]:
                problems.append("Frozen checkpoint mismatch")
            if (
                state["effective_dtypes"] != ["torch.float32"]
                or state["effective_batch_sizes"] != [1]
                or state["frames_completed"] != frames
                or state["predict_invocations"] != frames
            ):
                problems.append("Invocation/precision mismatch")
            totals[name] += state["predict_invocations"]
        for name, digest in run["implementation_sha256"].items():
            if sha256_file(root / "src/construction_safety_vision" / name) != digest:
                problems.append("Runtime changed after real-model smoke")
        if with_videos:
            if run["output"]["filename"] != f"compare_{device}.mp4":
                problems.append("Unexpected smoke video path")
                continue
            path = root / "outputs/phase13a_smoke" / f"compare_{device}.mp4"
            if file_identity(path) != run["output"]:
                problems.append("Smoke output identity mismatch")
            verify_output(path, frames=frames, fps=10.0, dimensions=(640, 192))
            if file_identity(root / "outputs/phase13a_smoke/synthetic.mp4") != run["input"]:
                problems.append("Synthetic input identity mismatch")
    if totals != {"D2": 4, "S1": 4}:
        problems.append("Total invocation mismatch")
    if (root / MARKDOWN).read_text(encoding="utf-8") != report_text(p):
        problems.append("Rendered report differs from evidence")
    provenance = json.loads((root / PROVENANCE).read_text(encoding="utf-8"))
    for entry in provenance["inputs"] + provenance["outputs"]:
        if entry["path"].startswith("outputs/") and not with_videos:
            continue
        if sha256_file(root / entry["path"]) != entry["sha256"]:
            problems.append(f"Provenance digest mismatch: {entry['path']}")
    for path in [REPORT, MARKDOWN, PROVENANCE, "delivery/VIDEO.md", *SOURCES]:
        problems.extend(scan_for_sensitive((root / path).read_text(encoding="utf-8")))
    return problems
