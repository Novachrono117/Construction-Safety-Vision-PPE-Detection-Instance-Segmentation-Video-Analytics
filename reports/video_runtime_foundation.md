# Phase 13A — video runtime foundation

**VIDEO_RUNTIME_FOUNDATION_COMPLETE**. The final real >=30-second assignment video remains **OPEN**.

## Architecture and public interface

[CLI and source attribution](../delivery/VIDEO.md) · [Runtime](../src/construction_safety_vision/video_runtime.py) · [Frozen frame adapter](../src/construction_safety_vision/video_models.py)

The CLI decodes sequential BGR frames with OpenCV/FFmpeg, calls each requested frozen model once per frame, renders on the original canvas and writes MP4/mp4v. No new dependency was installed: OpenCV and NumPy were already pinned through the existing environment. The runtime does not import a dataset or holdout accessor.

`detector`: D2 boxes/class/confidence. `segmenter`: S1 boxes/class/confidence plus translucent original-canvas masks. `compare`: D2 left and S1 right, both from the same frame; no resize, output width = 2 * source width, height unchanged. Five canonical classes stay separate. No winner or compliance overlay exists.

## Frozen execution contract

Public checkpoint metadata must equal the two authoritative freeze manifests. Only the local frozen copies are loaded after size and SHA-256 checks; missing or changed bytes fail closed. D2 is YOLO11n; S1 is YOLO11n-seg with training mask_ratio 4 and overlap_mask false.

Both use imgsz 768, conf 0.25, NMS IoU 0.70, max_det 300, FP32, batch 1, augment/TTA false. S1 uses retina_masks true. Settings come from the frozen Phase 10 operational configuration. Forward hooks verify actual float32, batch size and device; no threshold or checkpoint override is exposed by the CLI.

Explicit `cuda` and `cpu` were smoke-tested. Requested CUDA without availability fails; CPU is never a fallback. CPU support is functional evidence from a one-frame synthetic smoke, not a throughput equivalence claim.

## Decode, encode and failure contract

Every source frame is decoded and processed once in order. Source FPS is copied to the encoder; inference throughput never determines playback FPS. MP4/mp4v was exercised on this Windows OpenCV/FFmpeg build; H.264 and cross-platform codec portability are not claimed. Output is reopened and fully decoded to verify frame count, dimensions, FPS, approximate duration and nonempty bytes.

Incomplete media uses an explicit `.incomplete.mp4` name. Normal exceptions and KeyboardInterrupt remove partial media and leave FAILED provenance. Verified output is renamed into place only at completion. An explicit engineering frame cap is ENGINEERING_PREFIX_COMPLETE, never a complete source-video claim. A hard process kill can leave an incomplete file/STARTED record with stale counts; a later run refuses to overwrite that attempt silently. Two-file output/sidecar publication is not a filesystem transaction.

## Synthetic unit/integration evidence

Executed focused suite: **33 passed**, 0 failures, 0 errors, 0 skipped. Real codec round-trips exercise all three modes with model stubs: six 320x192 frames at 10 FPS. Tests verify order/content within lossy-codec tolerance, no canvas scaling, source/output counts and FPS, mask coordinates, strict CLI/settings/identity checks and injected decode/inference/writer/verification/interruption failures.

## Minimal real-model smoke

Predeclared in [video_runtime_smoke.yaml](../configs/video_runtime_smoke.yaml): one generated three-frame 320x192 MP4 at 10 FPS (0.3 seconds); compare mode processes all three on CUDA and an explicit one-frame engineering prefix on CPU. No validation image, external source or holdout content was used. The fixture is SYNTHETIC_ENGINEERING_FIXTURE and cannot satisfy the assignment video.

| Device | Frames | D2 calls | S1 calls | Output | Loop seconds | Demo loop FPS |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| cuda | 3 | 3 | 3 | 640x192 / 10.0 FPS | 5.621177 | 0.533696 |
| cpu | 1 | 1 | 1 | 640x192 / 10.0 FPS | 0.150850 | 6.629111 |

**Total predict() invocations: D2=4, S1=4.** CUDA records four forward calls per model including normal framework warmup; CPU records one per model. Both models emitted predictions on the synthetic shapes, so the real box and S1 mask render paths executed. These detections have no semantic accuracy interpretation. Reopened outputs contain exactly three and one frames. The CPU one-frame prefix cannot establish timestamp cadence and records that limitation explicitly.

Hardware: NVIDIA GeForce RTX 5070 Laptop GPU. See JSON for exact torch/CUDA/Ultralytics/OpenCV versions and per-run hashes.

## FPS semantics

DEMO_RUNTIME_MEASUREMENT = processed frames / end-to-end loop wall time. Includes decode, the decoded-frame sequence checksum, preprocessing, inference, postprocessing, render, encode and writer flush, including first-call framework setup warmup. Excludes model loading, initial metadata/identity checks, source/output file hashing, output verification and sidecar writes. The overlay displays the rate of previously completed frames; the sidecar contains final elapsed time. These very short, different-length runs are not a CPU/GPU comparison or a steady-state speed estimate. They are not comparable to Phase 10C. No controlled repetitions or memory benchmark occurred.

## Provenance and known limitations

Each execution records portable input/output filenames, SHA-256 and bytes, source attribution, mode, model identity/settings, device/runtime, video metadata, frame counters, decoded-sequence digest, timing, warnings and completion status. No username, hostname or personal absolute path is stored. The local videos and execution sidecars remain ignored; this report commits their small evidence records and hashes.

Supported contract: local constant-frame-rate video with even dimensions. Observed variable/discontinuous timestamps fail; unavailable timestamps produce a CFR-assumption warning. A known frame-count mismatch fails; when the decoder has no count, early EOF cannot be distinguished and is disclosed. Audio, metadata rotation, source timestamp tracks and HDR/color metadata are not preserved. Headers/FPS text occupy thin overlay strips. Long-running, high-resolution and real construction footage behavior has not yet been validated. Checkpoint public retrieval remains pending license review.

GAP-002 **OPEN**. GAP-010 **PARTIALLY_RESOLVED**: a functional CLI exists; the final real video/demo and publication do not. Tracking, temporal IDs, business compliance rules, training, tuning and holdout access: **false**. Scientific artifacts and Phase 12C evidence remain historical and unchanged.

## Verification

```text
uv run pytest tests/test_video_runtime.py --metadata-only
uv run python scripts/validate_video_runtime.py
uv run python scripts/validate_video_runtime.py --with-videos
```

The validator never executes a model. The optional video check reads only the named synthetic outputs. The one-time smoke script refuses an existing ledger; rebuilding this report consumes persisted evidence only.

API references: [OpenCV VideoCapture](https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html) and [VideoWriter](https://docs.opencv.org/4.x/dd/d9e/classcv_1_1VideoWriter.html). Runtime behavior was verified against the installed build, not assumed from codec names.
