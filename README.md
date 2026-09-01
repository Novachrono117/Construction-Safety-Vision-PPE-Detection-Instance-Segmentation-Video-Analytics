# Construction Safety Vision - PPE Detection, Instance Segmentation & Video Analytics

> **Status: Foundation Phase (phase 2 of 14).** No dataset has been downloaded,
> no model has been trained, and **no results exist yet**. This README documents
> the problem, the plan and the current repository state only. Every metric
> section is intentionally empty until a real, recorded run produces it.

A reproducible computer-vision system for detecting and segmenting people and
personal protective equipment (PPE) in construction scenes, with a controlled
quantitative evaluation, an explicit error analysis, and inference on real
video footage.

Built as a graduate assignment in Computer Vision and Pattern Recognition, and
as a public technical portfolio project.

---

## Problem statement

Construction sites are among the most hazardous work environments, and a large
share of severe incidents involve missing or incorrectly worn protective
equipment. Manual supervision of PPE compliance does not scale: it is
intermittent, subjective, and cannot review footage after the fact.

The task addressed here is: **given an image or a video frame of a construction
scene, locate every person and every piece of protective equipment, and
distinguish equipment that is actually being worn from equipment that is merely
present in the scene.**

That distinction is what makes the problem interesting rather than routine. A
helmet lying on a bench and a helmet on a worker's head are visually similar
objects with opposite safety meanings, so a system that only detects "helmet" is
useless for compliance. The planned class set therefore separates worn from
loose equipment.

Secondary difficulties expected in this domain, to be confirmed empirically
during the dataset audit: small objects at distance, heavy occlusion in crowded
scenes, strong outdoor lighting variation, and class imbalance between people
and equipment.

## Planned architecture

```text
canonical source: instance segmentation annotations (polygons)
        |
        |  boxes derived mathematically from polygons
        v
one frozen split (train / val / test) shared by both tasks
        |
        +--> detection view  ----> fine-tuned modern detector ----+
        |                                                          |
        +--> segmentation view ---> fine-tuned instance segmenter -+
                                                                   |
                                                                   v
                                         single evaluation protocol
                                    (mAP@0.5, mAP@0.5:0.95, IoU,
                                     precision, recall, confusion matrix)
                                                                   |
                                                                   v
                                    error analysis  +  video inference
```

Two design decisions define this architecture:

1. **Instance segmentation is the single source of truth.** Bounding boxes are
   computed from the polygons rather than annotated separately, so the detector
   and the segmenter describe exactly the same objects. Any difference in their
   results is attributable to the models, not to differing labels.
2. **One split, frozen once.** Both tasks use identical image IDs per split, and
   the `test` split is a locked holdout, read exactly once after both models are
   frozen. Selection and tuning use validation data only.

## Current status

| Area | State |
| --- | --- |
| Repository foundation | In progress (this phase) |
| Dataset | **Not acquired.** A candidate is declared in `configs/project.yaml` with `verified: false`. |
| Splits | Not defined, not frozen. |
| Detection model | Not trained. |
| Segmentation model | Not trained. |
| Metrics | **None.** No evaluation has been run. |
| Video inference | Not implemented. |
| Tracking (bonus) | Not started; deliberately deferred. |

What exists today: the project layout, a pinned environment, strict typed
configuration loading, the holdout protection guard, provenance primitives,
their tests, an environment-check script, and the planning documents
(`reports/rubric_contract.md`, `reports/roadmap.md`, `CLAUDE.md`).

The candidate dataset is *Construction PPE Compliance Detection* (Roboflow
Universe), with a hypothesised class set of `person`, `helmet_loose`,
`helmet_on_head`, `vest_loose`, `vest_on_body`. Its image count, split
composition, class balance and annotation quality are **unverified** and are
treated as hypotheses until the audit phase measures them.

## Academic requirements

The assignment requires all of the following. Each is mapped to a verifiable
definition of done in [`reports/rubric_contract.md`](reports/rubric_contract.md).

- At least 300 annotated images, with a train/validation/test split
- Fine-tuning of a modern object detector
- Instance segmentation in the same domain
- mAP@0.5 and mAP@0.5:0.95
- IoU, precision and recall
- Confusion matrix
- Qualitative analysis of false positives and false negatives
- Inference on real video of at least 30 seconds
- Complete hyperparameter documentation
- An executable Colab notebook
- A technical report
- A reproducible GitHub repository
- A 5-8 minute video pitch
- Bonus (up to +0.5): ByteTrack tracking or an interactive demo

## Planned methodology

Work proceeds through 14 gated phases (see
[`reports/roadmap.md`](reports/roadmap.md)). In outline:

1. **Scope and contract.** Turn the rubric into falsifiable acceptance criteria
   and fix the experimental protocol before seeing any data.
2. **Foundation.** Reproducible skeleton, tooling, protocol primitives.
3. **Acquisition and provenance.** Record source, version, license and file
   hashes for everything downloaded.
4. **Audit and EDA.** Measure class distribution and annotation integrity;
   detect exact duplicates, near duplicates and video-frame leakage.
5. **Split freeze.** Freeze one partition, keeping duplicates and same-sequence
   frames inside a single split; derive the detection and segmentation views.
6-7. **Detection.** Baseline first, then controlled variations, then freeze one
   model selected on validation data.
8-9. **Segmentation.** The same procedure on the same splits.
10. **Controlled validation comparison.** One evaluation protocol for all
    models, with matching rules and thresholds stated.
11. **One-shot test evaluation.** The holdout is unlocked and read exactly once.
12. **Error analysis.** Per-class FP/FN review, with observations separated from
    hypotheses.
13. **Video inference.** Real footage, measured throughput, temporal failure
    modes. Tracking only as a bonus, after the required work is complete.
14. **Submission and reproducibility audit.** Verify the whole thing from a
    clean clone.

## Repository structure

```text
.
├── CLAUDE.md                  # Operating constitution: protocol rules for AI-assisted sessions
├── README.md
├── pyproject.toml             # Project metadata, dependencies, ruff and pytest configuration
├── .env.example               # Documented environment variables (no secrets)
├── .gitattributes             # LF normalisation, binary declarations
├── configs/                   # Versioned experiment configuration (single source of settings)
│   └── project.yaml
├── data/                      # Never committed; see data/README.md
│   ├── raw/                   # Immutable downloads
│   ├── interim/               # Regenerable intermediates
│   ├── processed/             # Task-ready datasets derived from raw/
│   └── external/              # Non-dataset assets (e.g. inference video)
├── notebooks/                 # Colab-executable notebooks (added by the phase that needs them)
├── reports/
│   ├── rubric_contract.md     # Rubric as a verifiable contract
│   ├── roadmap.md             # 14 phases with validation gates
│   └── figures/
├── scripts/
│   └── check_environment.py   # Environment, configuration and holdout-lock report
├── src/construction_safety_vision/
│   ├── config.py              # Strict typed configuration loading
│   ├── paths.py               # Repository layout for local and Colab runs
│   ├── provenance.py          # Hashing and run provenance records
│   └── splits.py              # Split identifiers and the holdout guard
└── tests/
```

## Development principles

- **Correctness > Simplicity > Maintainability > Performance > Cleverness.**
- **The holdout is sacred.** The test split is read once, after the models are
  frozen. Enforced in code by `splits.assert_split_allowed`, which requires both
  an in-code opt-in and an environment unlock.
- **No fabricated evidence.** Every number comes from an executed run with a
  provenance record. Unknown values are written `TBD`, never estimated.
- **Causation requires an experiment.** Claims that a change caused an
  improvement require a controlled comparison; otherwise they are labelled as
  hypotheses.
- **Raw data is immutable.** Corrections are code in the pipeline, never manual
  edits.
- **Configuration is a file, not a memory.** Every hyperparameter lives in
  `configs/` and is snapshotted into each run's provenance record.
- **Dependencies must be justified.** Heavy frameworks arrive with the phase
  that needs them.
- **Documentation describes reality.** Planned work is labelled as planned.

## Reproducibility goals

The target is that a third party, starting from a clean clone, can reproduce
every reported number without contacting the author.

- Environment pinned by `pyproject.toml` + `uv.lock`, with a single documented
  setup command.
- One global seed, recorded in every run.
- Frozen split manifests, listing every image ID and its content hash.
- Provenance records tying each artifact to its inputs, configuration, code
  commit and environment.
- Datasets and checkpoints re-obtainable from recorded sources and hashes rather
  than committed.
- Notebooks that run top to bottom on a fresh Colab runtime.
- A final reproducibility audit performed from a clean clone (phase 14).

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11+.

```bash
git clone https://github.com/Novachrono117/Construction-Safety-Vision-PPE-Detection-Instance-Segmentation-Video-Analytics.git
cd Construction-Safety-Vision-PPE-Detection-Instance-Segmentation-Video-Analytics
uv sync
cp .env.example .env          # optional; nothing is required in the foundation phase
uv run python scripts/check_environment.py
```

Checks:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

## Results

Not available. No model has been trained and no evaluation has been run. This
section will be filled by phases 10-12, from committed metrics files.

## License

Not yet chosen. The dataset's license (recorded in phase 3) constrains what may
be redistributed here.
