# Rubric Contract

Version: 1.0 · Status: **active** · Phase: 1-2 (foundation)

This document turns the grading rubric into a verifiable contract. Every
criterion below states what must exist, which artifact proves it, which roadmap
phase produces it, and the objective condition under which it is considered
done. Nothing here may be marked done on the basis of intention: an item is done
when the named artifact exists in the repository and satisfies its condition.

**Nothing in this document is a claim about results.** No metric value, no
dataset statistic and no qualitative conclusion appears here — only the
obligations that must later be met with real evidence.

## Weight summary

| # | Criterion | Weight | Producing phases |
| --- | --- | --- | --- |
| C1 | Problem and dataset | 10% | 3, 4, 5 |
| C2 | Object detection | 25% | 6, 7 |
| C3 | Segmentation | 20% | 8, 9 |
| C4 | Evaluation and critical analysis | 20% | 10, 11, 12 |
| C5 | Video application | 10% | 13 |
| C6 | Report and repository | 10% | 2, 14 |
| C7 | Video pitch | 5% | 14 |
| B1 | Bonus: tracking or interactive demo | +0.5 | 13 (optional) |

Total: 100% + up to 0.5 bonus points.

---

## C1 - Problem and dataset (10%)

| Field | Content |
| --- | --- |
| **Requirement** | A well-defined construction-safety problem, and a dataset of at least 300 annotated images with a train/validation/test split. The dataset must be described, audited and its provenance recorded. |
| **Planned evidence** | Dataset provenance record (source URL, version, license, download date, file hashes); audit report with image counts per split, class distribution, annotation-integrity checks, exact- and near-duplicate analysis, and a frame-leakage assessment; a frozen split manifest listing every image ID and its split. |
| **Expected artifacts** | `data/raw/*.provenance.json`, `reports/dataset_audit.md`, `notebooks/01_dataset_audit.ipynb`, `data/processed/splits/*.manifest.csv`, `reports/figures/class_distribution.*` |
| **Responsible phases** | 3 (acquisition and provenance), 4 (audit and EDA), 5 (split freeze) |
| **Definition of done** | (a) total annotated image count is measured and >= 300; (b) the three splits are non-empty, disjoint by image ID, and their union equals the audited dataset; (c) the class distribution per split is reported as measured counts; (d) duplicate and leakage checks have been run and their outcome documented, including what was removed and why; (e) the split manifest is committed and hashed; (f) the license and terms of the source dataset are recorded and compatible with public release. |
| **Failure condition** | Any count, class or split property stated in the report that was not produced by an executed audit script. |

## C2 - Object detection (25%)

| Field | Content |
| --- | --- |
| **Requirement** | Fine-tuning of a modern detector on the project dataset, with fully documented hyperparameters and a reproducible training procedure. |
| **Planned evidence** | Training configuration files; training logs and loss/metric curves; a baseline run plus at least one deliberate variation; the frozen final model with a recorded selection rationale based only on validation results. |
| **Expected artifacts** | `configs/detection_*.yaml`, `reports/detection_experiments.md`, `reports/figures/detection_curves.*`, `runs/detection/*/` (untracked, described by `*.provenance.json`), `notebooks/02_detection_training.ipynb` |
| **Responsible phases** | 6 (baseline), 7 (experiments and model freeze) |
| **Definition of done** | (a) the detector, its pretrained weights and their source are named exactly; (b) every hyperparameter that was set is documented, including image size, epochs, batch size, optimizer, learning-rate schedule, augmentations and seed; (c) at least two runs exist and are comparable because only documented factors differ; (d) model selection used validation data only, and the rationale is written down; (e) the selected checkpoint is identified by hash in a provenance record; (f) the training procedure can be re-executed from the repository by following the documented commands. |
| **Failure condition** | Any hyperparameter mentioned in the report that does not appear in a committed configuration file; any selection decision that used test data. |

## C3 - Segmentation (20%)

| Field | Content |
| --- | --- |
| **Requirement** | Instance segmentation in the same domain, on the same images and the same splits as detection. |
| **Planned evidence** | Segmentation training configurations and runs; a demonstration that detection and segmentation share image IDs split by split; mask-based qualitative outputs. |
| **Expected artifacts** | `configs/segmentation_*.yaml`, `reports/segmentation_experiments.md`, `reports/figures/segmentation_qualitative.*`, `notebooks/03_segmentation_training.ipynb` |
| **Responsible phases** | 8 (baseline), 9 (experiments and model freeze) |
| **Definition of done** | (a) segmentation uses the frozen split manifest, verified by an automated ID-alignment check that passes; (b) the derivation of detection boxes from the segmentation polygons is implemented in code and documented, so both tasks provably describe the same objects; (c) hyperparameters are documented to the same standard as C2; (d) the selected checkpoint is frozen and hashed; (e) mask quality is shown on validation images, including failure cases. |
| **Failure condition** | Detection and segmentation trained on differently split data, or boxes annotated independently of the masks. |

## C4 - Evaluation and critical analysis (20%)

| Field | Content |
| --- | --- |
| **Requirement** | mAP@0.5, mAP@0.5:0.95, IoU, precision, recall, a confusion matrix, and a qualitative analysis of false positives and false negatives. |
| **Planned evidence** | A metrics table for both tasks on validation and on the final holdout; a confusion matrix figure; a curated gallery of FP/FN cases with per-case commentary; a written discussion of error patterns and their likely causes, separating observation from hypothesis. |
| **Expected artifacts** | `reports/evaluation.md`, `reports/metrics/*.json`, `reports/figures/confusion_matrix_*.png`, `reports/error_analysis.md`, `reports/figures/errors/*`, `notebooks/04_evaluation.ipynb` |
| **Responsible phases** | 10 (controlled validation comparison), 11 (one-shot test evaluation), 12 (error analysis) |
| **Definition of done** | (a) every required metric is reported for detection and for segmentation, with the IoU threshold, matching rule and averaging scheme stated; (b) the evaluation protocol names the tool or implementation used and its version; (c) the confusion matrix states how background/unmatched predictions are handled; (d) at least one documented false positive and one documented false negative per class are analysed, each with the image, the prediction, the ground truth and a stated hypothesis; (e) the test split was evaluated exactly once, after the model freeze, and the run is recorded; (f) every causal statement is labelled as hypothesis unless a controlled experiment supports it. |
| **Failure condition** | Any metric value in the report that is not reproducible from a committed metrics file; test-set numbers used to iterate on the models. |

## C5 - Video application (10%)

| Field | Content |
| --- | --- |
| **Requirement** | Inference on a real construction-scene video of at least 30 seconds. |
| **Planned evidence** | The source video's provenance and license; an inference script; the annotated output video; measured throughput; a written discussion of temporal failure modes. |
| **Expected artifacts** | `scripts/run_video_inference.py`, `data/external/*.provenance.json`, `reports/video_analysis.md`, output video (untracked, linked in the report) |
| **Responsible phases** | 13 |
| **Definition of done** | (a) the video is >= 30 s, real construction footage, and its source and license are recorded; (b) inference runs end to end from the repository with a documented command; (c) the output shows class labels and confidences; (d) processing speed (FPS) and the hardware it was measured on are reported as measurements; (e) at least three temporal failure modes (e.g. flicker, identity instability, occlusion loss) are described with timestamps. |
| **Failure condition** | A demo video assembled from still images, or a video whose source cannot be attributed. |

## C6 - Report and repository (10%)

| Field | Content |
| --- | --- |
| **Requirement** | A technical report and a reproducible public GitHub repository. |
| **Planned evidence** | The technical report; a README that matches the actual repository state; a documented environment; an executable Colab notebook; a reproducibility audit performed from a clean clone. |
| **Expected artifacts** | `reports/technical_report.md` (plus PDF), `README.md`, `CLAUDE.md`, `pyproject.toml`, `uv.lock`, `notebooks/*.ipynb`, `reports/reproducibility_audit.md` |
| **Responsible phases** | 2 (foundation), 14 (submission package and audit) |
| **Definition of done** | (a) the report covers problem, dataset, method, hyperparameters, results, error analysis, limitations and future work; (b) the README describes only what exists, with the project status stated honestly; (c) the environment is pinned by a lockfile and the setup commands are verified from a clean clone; (d) at least one notebook runs top to bottom on a fresh Colab runtime; (e) no dataset, checkpoint or large binary is committed; (f) every result in the report is traceable to a committed configuration and a provenance record. |
| **Failure condition** | Documentation describing features, scripts or results that do not exist in the repository. |

## C7 - Video pitch (5%)

| Field | Content |
| --- | --- |
| **Requirement** | A 5-8 minute presentation video. |
| **Planned evidence** | A script/outline with timings, the recording, and the slides. |
| **Expected artifacts** | `reports/pitch_script.md`, slide deck, recording link |
| **Responsible phases** | 14 |
| **Definition of done** | (a) duration is between 5 and 8 minutes; (b) it covers problem, dataset, method, results, error analysis, video demo and limitations; (c) every number spoken matches the report; (d) the demo shown is the real output of the pipeline. |
| **Failure condition** | Claims in the pitch that exceed what the evaluation supports. |

## B1 - Bonus: tracking or interactive demo (up to +0.5)

| Field | Content |
| --- | --- |
| **Requirement** | Multi-object tracking (ByteTrack) or an interactive demo, beyond the required scope. |
| **Planned evidence** | Tracking integrated into the video pipeline with stable identities, or a deployed interactive demo. |
| **Expected artifacts** | `scripts/run_video_tracking.py`, `reports/tracking_notes.md`, or a demo app plus its link |
| **Responsible phases** | 13 (optional, only after C1-C7 are done) |
| **Definition of done** | (a) identities persist across frames and the behaviour is demonstrated on the reference video; (b) the tracker configuration is documented; (c) the qualitative effect on the failure modes listed in C5 is discussed; (d) no required deliverable regressed to make room for it. |
| **Failure condition** | Bonus work started before the mandatory criteria are complete. |

---

## Global constraints that override every criterion

1. **Holdout integrity.** The test split is read exactly once, after both models
   are frozen (see `CLAUDE.md`). A violation invalidates C4 regardless of the
   numbers obtained.
2. **No fabricated evidence.** Every number, count and figure originates from an
   executed run whose provenance record exists. Placeholders, if ever needed
   during drafting, are marked `TBD` and never look like results.
3. **Task alignment.** Detection and segmentation share image IDs split by
   split. This is enforced by an automated check, not by assertion.
4. **Reproducibility.** Every preprocessing step is code in the repository. No
   manual dataset edits.
5. **Traceability.** Any claim in any report maps to a configuration file and a
   provenance record.

## Contract change log

| Date | Version | Change |
| --- | --- | --- |
| 2026-09-01 | 1.0 | Initial contract derived from the rubric during the foundation phase. |
