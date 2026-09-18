# Colab delivery: evidence first, optional inference

Notebook: [construction_safety_vision_demo.ipynb](../notebooks/construction_safety_vision_demo.ipynb).
Support: [delivery_demo.py](../src/construction_safety_vision/delivery_demo.py) and
[delivery_academic.py](../src/construction_safety_vision/delivery_academic.py).

**Phase 14C: `EXECUTABLE_ACADEMIC_COLAB_COMPLETE` / `COLAB_EXACT_WORDING_SATISFIED`.**
Human-executed Mode A cloud validation passed on 2026-09-18 on a fresh **CPU**
runtime at revision `2e355d8012b6f1f49923c5cb3ead06760490f024`, after the notebook
gained explicit TREINO and AVALIACAO sections. Mode B passed on 2026-09-17 at
revision `8ce5d0375903e3e3760873e0a75ddce37cc1149b` and was **not** re-executed:
all 25 Mode B execution inputs are byte-identical between the two revisions, which
the report derives by comparing Git blobs rather than asserting.
[Validation report and execution identities](../reports/academic_colab_delivery.md).

## Notebook structure

1. Overview - clone and load both support modules; architecture and headline metrics.
2. Dataset and classes - modelling population, class inventory, frozen split sizes.
3. **TREINO / TRAINING** - `RECORDED_TRAINING_EVIDENCE` / `NO_NEW_TRAINING_EXECUTED`.
4. **AVALIACAO / EVALUATION** - committed one-shot holdout results.
5. Qualitative evidence - holdout selection accounting plus the committed validation gallery.
6. Video evidence - the three reviewed frame pairs.
7. **INFERENCIA / INFERENCE** - the optional Mode B path, unchanged.
8. Limitations and licences.

### What the training and evaluation sections do, and do not, execute

Both sections are executable and produce visible output in artifact-only Mode A.
Neither runs a model. The training cells read each frozen model's committed
experiment manifest and print the complete recipe - architecture, image size,
epochs, batch, seed, the optimizer that `optimizer: auto` actually resolved to,
every augmentation argument, the checkpoint-selection rule, the selected epoch and
the frozen checkpoint identity - followed by the training and validation curves
recorded at the time. The reproducible entry points
(`scripts/train_detection_experiment.py --experiment D2` and
`scripts/train_segmentation_comparison.py`) are printed with each recipe; running
them needs a CUDA GPU and the materialised dataset, which the notebook does not
provide.

**The default notebook does not retrain the models and does not rerun the spent
holdout evaluation.** Each model was trained once under a protocol frozen before
the run, so a re-run would produce different bytes under the same name and
invalidate the published results. The holdout was read exactly once and is spent.
The evaluation cells therefore read the committed result artifacts by field; no
executable final-test path exists, and no executable validation-evaluation path is
offered either, because that would need the dataset and the frozen checkpoints.
The final-test confusion matrices are rendered from their recorded counts, never
from the figures under `reports/figures/final_test/`.

The repository is public. The canonical notebook clones `main`, and the
[production Colab URL](https://colab.research.google.com/github/Novachrono117/Construction-Safety-Vision-PPE-Detection-Instance-Segmentation-Video-Analytics/blob/main/notebooks/construction_safety_vision_demo.ipynb)
targets that branch, and the notebook is published there.

## Mode A — no checkpoints

1. Open/import the `.ipynb` in Colab.
2. Leave `RUN_INFERENCE = False` and run the notebook from top to bottom.
3. The bootstrap clones the repository, prints its exact Git revision and checks
   that it contains the approved Phase 13B evidence and both delivery support modules.
4. Inspect the dataset and class inventory, the recorded training recipes and curves,
   the committed final-test and validation metrics, the validation FP/FN/mask gallery
   and the three recorded video frame pairs. Change a display selector and rerun that
   cell to inspect another existing view.

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

## Validation evidence and publication boundary

- The notebook is a new delivery surface, not a research reproduction or a new
  experimental phase. Historical experiments and the holdout remain closed.
- **LOCAL_IMPLEMENTATION_VERIFICATION:** Mode A metadata and display dispatch
  are tested locally in a copy containing
  only public evidence, without data, checkpoints or project dependencies.
  Mode B tests use synthetic bytes and a mocked CLI process to verify identity
  rejection, existing-file preservation and dispatch. They do not execute D2/S1.
- **REAL_COLAB_MODE_A_VALIDATION: PASS (Phase 14C, 2026-09-18).** The human used a
  fresh **CPU** Colab runtime, observed revision
  `2e355d8012b6f1f49923c5cb3ead06760490f024`, ran the notebook top to bottom with
  `RUN_INFERENCE=False`, saw no traceback, and reached `Mode A complete. No
  inference output was created.` The dataset, TREINO, AVALIACAO, qualitative and
  video sections all rendered, and the human confirmed every table, curve, matrix
  and figure was readable. No GPU, checkpoints, Roboflow key, holdout
  authorization or model inference were used; no training ran and no final-test
  evaluation was rerun.
- **REAL_COLAB_MODE_B_VALIDATION: PASS (Phase 14A, 2026-09-17, not re-executed in
  Phase 14C).** `MODE_B_REVALIDATION_NOT_REQUIRED`: every Mode B execution input is
  byte-identical between the two validated revisions. The human installed the locked
  environment, verified CUDA on a Tesla T4, uploaded the exact D2/S1 bytes,
  observed verification before deserialization, and ran `compare` on an external
  five-second, 125-frame clip. Output completion, identity and full decoding were
  checked without repeating inference. The observed runtime was Python 3.12.3,
  PyTorch 2.11.0+cu128 / CUDA 12.8, Ultralytics 8.4.138 and OpenCV 5.0.0.
  This is one recorded CUDA configuration; cloud CPU inference was not validated.
- Cloud evidence belongs to the temporary revisions above. In each session the
  human changed the clone branch only inside Colab - to
  `phase14a-colab-validation`, then to `phase14c-colab-validation` - and enabled
  the optional Mode B controls for the Phase 14A validation. The canonical
  notebook still clones `main`, defaults to `RUN_INFERENCE = False`, and retains
  its validated bytes. Finalization preserves the execution-critical files.
- The Phase 14A session added a separate GPU diagnostic cell that was **never
  committed** and is absent from every canonical notebook. A stale saved copy of
  that session will fail on a CPU runtime; that is a session artifact, not a
  delivery defect. Always open the notebook from GitHub rather than from a saved
  Drive copy.
- The notebook's prepublication notice is historical wording retained to preserve
  those validated bytes. Current availability is stated here and in the report.
  The bootstrap still stops with `PUBLICATION_REQUIRED` if the cloned checkout
  lacks the required evidence or support module. No final main push or video
  release is part of this step.

### Academic scope and remaining gaps

GAP-004 is **RESOLVED** and assignment Colab requirement R18 is **COMPLETE**
under the human-approved two-mode delivery scope. Phase 14C additionally satisfies
the literal wording the assignment uses for the notebook - executable training,
evaluation and inference cells with visible outputs - on the stated basis that the
training and evaluation sections present recorded evidence rather than
re-executing the closed study. The immutable Phase 12A audit proposed reproducing
validation metrics; the approved scope instead displays the committed results and
demonstrates optional external-video inference. No validation metric was
recomputed. GAP-014 is the separate **clean-room research
reproduction audit**, and remains OPEN. The Phase 12A snapshot is unchanged.

GAP-008 (public checkpoint redistribution) remains OPEN: users supply the frozen
weights manually. GAP-010 (public demo delivery) remains PARTIALLY_RESOLVED while
distribution constraints remain; GAP-009 retains its documented status.

### Five-second validation output

The Mode B output is `COLAB_TECHNICAL_VALIDATION_ARTIFACT` /
`NOT_PUBLICATION_READY` and is not published. Its original generated provenance
retains `SOURCE_LICENSE_UNSPECIFIED`, because the upload interface did not receive
attribution metadata. The source lineage is independently documented as Frank
Vincentz / CC BY-SA 3.0 from the approved Phase 13B external source. That separate
knowledge does not rewrite the generated warning. The report records identities;
neither the input, generated MP4 nor uploaded weights is committed.

The upload/download integration follows the official
[Google Colab file helpers](https://github.com/googlecolab/colabtools/blob/main/google/colab/files.py).
No extra project dependency or package-policy change was introduced.
