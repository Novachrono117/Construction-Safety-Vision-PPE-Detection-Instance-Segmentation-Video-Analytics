# Colab delivery: evidence first, optional inference

Notebook: [construction_safety_vision_demo.ipynb](../notebooks/construction_safety_vision_demo.ipynb).
Support: [delivery_demo.py](../src/construction_safety_vision/delivery_demo.py).

## Mode A — no checkpoints

1. Open/import the `.ipynb` in Colab.
2. Leave `RUN_INFERENCE = False` and run the notebook from top to bottom.
3. The bootstrap clones the repository, prints its exact Git revision and checks
   that it contains the approved Phase 13B evidence and the delivery support module.
4. Inspect the architecture, committed final-test and validation metrics, validation
   FP/FN/mask gallery and the three recorded video frame pairs. Change a display
   selector and rerun that cell to inspect another existing view.

Python's standard library handles metadata. IPython display is provided by the
notebook environment. Mode A installs no project packages, reads no dataset or
checkpoint, and requires no GPU or dataset credentials. Loading the standalone
support file through `runpy` intentionally avoids the scientific package initializer.

Metrics are read from committed JSON, with source links to the approved evidence
revision. The original reports, figures, numbers and selection rules are preserved.
FP/FN imagery is validation-only. Final-test aggregate reports may be displayed;
holdout membership, images, annotations and prediction caches are never accessed.
The notebook refuses a present `CSVISION_ALLOW_TEST_SPLIT` environment variable.

### Full video availability

Six approved screenshots are in Git. The final MP4 is an external artifact with
recorded size **581898040 bytes** and SHA-256
`03c1e2ba2894862db7225a17ceefc1c6eb6a54534f2f94eef6230d0135ba4a69`.
No public download URL exists yet. Mode A can optionally accept a copy of this
exact recorded MP4 and verify it before opening the Colab file viewer; this never
executes a model. The upload is disabled by default.

Browser playback depends on MP4/mp4v support. Figures remain available when the
video is absent or unsupported. A future GitHub Release asset can distribute the
MP4 without adding it to Git history; no release or upload is performed by this
notebook. Include [attribution](FINAL_VIDEO_ATTRIBUTION.md) and provenance with it.

## Mode B — optional real inference

Enable `RUN_INFERENCE`, select `detector`, `segmenter` or `compare`, and choose an
explicit `cpu` or `cuda` device. For CUDA, first select a GPU runtime in Colab.

The install cell uses the existing package manager (`uv==0.12.13`, pinned bootstrap)
and `uv sync --locked --python 3.12 --no-dev`. It creates an isolated `.venv` and
keeps `pyproject.toml` and `uv.lock` unchanged. CUDA wheel download/storage costs
apply even if CPU is selected. The selected device never silently falls back.

Upload the exact required checkpoint(s). Identities are cross-checked against
[`checkpoints.json`](checkpoints.json) and both frozen model manifests. Size and
SHA-256 are verified before bytes are published to the expected ignored path;
existing files are never replaced. The approved runtime verifies them again
before loading. Wrong or missing weights stop the workflow. No automatic
checkpoint download, substitution or training route exists.

Confirm that you can use your **external MP4**, outside the research dataset,
then upload it. The Colab upload API buffers bytes in memory; short clips are
more practical. The helper accepts up to 2 GiB after upload. It invokes only
`scripts/run_video_demo.py` with the selected mode/device and a new output path.
There are no threshold or model tuning controls. The original frozen runtime
and configuration are unchanged, including codec/timestamp failure checks.

The output and provenance are downloaded separately. MP4/mp4v playback may require
a compatible local player. Audio is not retained. User-video source/license is
recorded as unspecified, so attribution must be supplied before publication.
Uploaded inputs and generated outputs live under ignored `outputs/colab_demo/`;
Colab storage is temporary. **Clear notebook outputs before saving to Git.**

## Readiness and publication boundary

- The notebook is a new delivery surface, not a research reproduction or a new
  experimental phase. Historical experiments and the holdout remain closed.
- Mode A metadata and display dispatch are tested locally in a copy containing
  only public evidence, without data, checkpoints or project dependencies.
- Mode B tests use synthetic bytes and a mocked CLI process to verify identity
  rejection, existing-file preservation and dispatch. They do not execute D2/S1.
- A fresh Colab execution, cloud dependency installation, real browser uploads
  and GPU/CPU inference in Colab have **not** been demonstrated yet.
- The notebook, support module and local Phase 13B commit must first be published
  to GitHub. The bootstrap stops with `PUBLICATION_REQUIRED` if they are absent.
  No commit, push, tag, visibility change or release publication is implied.
- GAP-010 (public artifact delivery), GAP-009 (reproduction) and GAP-014 (Colab)
  keep their existing tracked states. A prepared notebook does not prove cloud
  execution or public availability.

The upload/download integration follows the official
[Google Colab file helpers](https://github.com/googlecolab/colabtools/blob/main/google/colab/files.py).
No extra project dependency or package-policy change was introduced.
