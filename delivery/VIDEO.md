# Frozen-model video runtime

**Phase 13A runtime complete. Final real >=30-second assignment video: OPEN.**
This CLI processes ordinary local videos with the frozen final D2 and S1.
Engineering evidence: [runtime report](../reports/video_runtime_foundation.md).

## Prerequisites and invocation

Use the locked environment (`uv sync --locked`, Python 3.12 tested) and the exact
local frozen checkpoints. Public download locations remain unavailable pending
[license review](LICENSING.md); the runtime never downloads or retrains a model.
Identity metadata: [checkpoints.json](checkpoints.json). Required local copies:

| Model | Path | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| D2 | `artifacts/frozen/detection/D2_best.pt` | 5502289 | `0466f872a9de22898c70d8834cf1fcfb3d77f6c9fb27e81ee6248d3d19cbc206` |
| S1 | `artifacts/frozen/segmentation/S1_best.pt` | 6041685 | `29337d671459d0f742fb713613cc41d0eafcb8a21f5846ac0da65b2ffc024f20` |

The selected mode requires only its model(s). Missing or changed bytes fail
closed. Keep `CSVISION_ALLOW_TEST_SPLIT` unset.

PowerShell or Bash, repository root, with a user-supplied licensed video:

```text
uv run python scripts/run_video_demo.py --input input.mp4 --output outputs/compare.mp4 --mode compare --device cuda --source-metadata source.json
```

`input.mp4` and `source.json` are user-supplied filenames, not bundled media.
This command is the runtime interface; Phase 13A did not acquire or execute a
real assignment video. Its integration check used generated geometry only.

| Option | Contract |
| --- | --- |
| `--input`, `--output` | Existing local video; distinct MP4 output. Within this repository, outputs must be under `outputs/`. |
| `--mode detector` | D2 bounding boxes, canonical class and confidence. |
| `--mode segmenter` | S1 boxes, canonical class, confidence and translucent masks. |
| `--mode compare` | D2 left, S1 right; exact same source frame, unscaled. Output is `2W x H` for source `W x H`. |
| `--device cuda` | Explicit NVIDIA CUDA device 0; unavailable CUDA is an error, never CPU fallback. |
| `--device cpu` | Functional CPU support verified on a one-frame synthetic integration check; no equivalent-performance claim. |
| `--overwrite` | Permit replacing an existing output after successful verification. Failed processing preserves the previous successful output. |
| `--max-frames N` | Engineering prefix only, explicitly marked `ENGINEERING_PREFIX_COMPLETE`; not a completed assignment video. |
| `--no-fps-overlay` | Omit timing overlay; provenance still records timing. |
| `--source-metadata` | Optional source-attribution JSON below. Omitting the file produces a warning. Incomplete source/rights fields are not publication-ready. |

Thresholds are not CLI knobs: both models use the committed operational settings
in [the comparison configuration](../configs/detector_segmenter_comparison.yaml):
imgsz 768, conf 0.25, NMS IoU 0.70, max_det 300, FP32, no augment/TTA, batch 1.
S1 uses `retina_masks=true`. Its frozen training identity retains mask_ratio 4
and overlap_mask false. Actual tensor dtype, batch and device are checked.

## Frame and playback contract

OpenCV/FFmpeg sequentially decodes every frame without sampling. The MP4/mp4v
writer uses **source FPS**, preserving approximate duration when the whole
constant-frame-rate source is processed. Each panel preserves the entire raster
and aspect ratio; thin title/timing strips overlay it. Colors are fixed per
canonical class across models and frames. No tracking IDs, compliance rule or
winner overlay is added.

The runtime closes the writer, reopens and fully decodes the output, then checks
count, dimensions, FPS, duration and nonempty bytes before publishing it.
An `.incomplete.mp4` is removed on handled failure/KeyboardInterrupt; a
`.failed.provenance.json` records FAILED. Hard termination may leave explicitly
incomplete media and a STARTED record; inspect it before a later attempt. Output
and sidecar publication involves two renames, not an atomic two-file transaction.

Supported scope is local constant-frame-rate input with even dimensions.
Odd dimensions are rejected rather than silently truncated by the encoder.
Detected irregular timestamps are rejected. Unavailable timestamp/count metadata
is disclosed; with unknown frame count an early EOF cannot be distinguished from
normal completion. Rotation metadata is not applied; encoded dimensions are used.
Audio, HDR/color metadata and original timestamp tracks are not copied.
Only MP4/mp4v on the recorded local OpenCV/FFmpeg build was tested; H.264 and
cross-platform codec availability are not promised.

## Source licensing metadata

The JSON object has exactly these seven fields (values shown as schema guidance,
not as evidence of a real acquisition):

```json
{
  "source_type": "EXTERNAL_REAL_VIDEO",
  "title": null,
  "creator": null,
  "url": null,
  "license": null,
  "retrieval_date": null,
  "modifications": "Model overlays and side-by-side layout; no audio"
}
```

Fill the source/rights fields before publishing any real footage. Use public
HTTPS URLs without query parameters or embedded credentials; never signed URLs,
secrets or personal paths. Other source types are
`SYNTHETIC_ENGINEERING_FIXTURE` and `VALIDATION_ENGINEERING_FIXTURE`.
A validation-derived fixture requires its CC BY 4.0 attribution and remains an
engineering fixture. Phase 13A needed no dataset imagery.

## Timing and provenance

`<output>.provenance.json` stores filenames, hashes, sizes, source attribution,
model identities, frozen settings, device/runtime, source/output metadata,
decoded/processed/encoded counts, warnings and completion status. It stores no
personal absolute paths, username or hostname. The runtime hashes the decoded
frame sequence and checks that the input bytes did not change during execution.

**SOURCE_VIDEO_FPS** controls playback. **PROCESSING_THROUGHPUT_FPS** is processed
frames divided by loop wall time, including decode, decoded-frame sequence
checksum, preprocessing, inference,
postprocessing, render, encode and writer flush. The first framework setup warmup
is included. Model loading, metadata/identity checks, source/output file hashing, output verification
and sidecar writes are excluded. Overlay timing describes previously completed
frames; the sidecar stores final timing. Label: `DEMO_RUNTIME_MEASUREMENT`.
It is not Phase 10C's controlled benchmark and must not be compared as one.

## Verification and current delivery boundary

```text
uv run pytest tests/test_video_runtime.py --metadata-only
uv run python scripts/validate_video_runtime.py
uv run python scripts/validate_video_runtime.py --with-videos
```

The last command needs the ignored local synthetic smoke outputs. Unit tests
generate small video fixtures in temporary directories and require no real model.
The committed smoke configuration/script documents the executed integration
procedure; it refuses an existing execution ledger instead of repeating it.

GAP-010 is partially resolved by this functional runtime. GAP-002 stays OPEN:
no real >=30-second construction video, final annotated deliverable or temporal
failure analysis exists yet. Long/high-resolution real footage, sustained memory
behavior and codec portability are not validated. Tracking remains deferred.
