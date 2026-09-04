# Construction Safety Vision - PPE Detection, Instance Segmentation & Video Analytics

> **Status: dataset audit complete (phase 4 of 14).** The dataset is acquired,
> hashed, structurally verified, and its 436 original source images have been
> audited automatically (4A) and reviewed visually by people (4B). The provider's
> split was **rejected for the final protocol**; the canonical split has not been
> created. **No split has been frozen, no model has been trained, and no results
> exist yet.** Every metric section is intentionally empty until a real, recorded
> run produces it.

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
| Repository foundation | Done (phase 2). |
| Dataset acquisition and provenance | Done (phase 3). Archive hashed, export structurally verified. |
| Automated audit + source EDA | Done (phase 4A). 436 originals acquired, measured and screened. |
| Manual visual audit | Done (phase 4B). 32 human decisions recorded and validated against the phase 4A manifests. |
| Canonical annotation snapshot | Done (phase 5A). The live source state, recovered read-only with complete geometry in original coordinates. |
| Splits | **Provider split rejected for the final protocol. No canonical split created, none frozen.** |
| Detection model | Not trained. |
| Segmentation model | Not trained. |
| Metrics | **None.** No evaluation has been run. |
| Video inference | Not implemented. |
| Tracking (bonus) | Not started; deliberately deferred. |

What exists today: the project layout, a pinned environment, strict typed
configuration, the holdout protection guard, provenance primitives, a
dependency-free Roboflow acquisition client, COCO structural inspection, the
source audit / EDA / visual-review tooling, the phase 4B decision recorder, the
phase 5A geometry recovery and canonical-snapshot resolution, the test suite, and
the planning documents (`reports/rubric_contract.md`, `reports/roadmap.md`,
`CLAUDE.md`).

### The dataset

*Construction PPE Compliance Detection* (Roboflow Universe,
`agis-workspace-8gs52`, version 4), acquired as a **COCO instance-segmentation**
export under **CC BY 4.0**. Classes verified in the annotations: `person`,
`helmet_loose`, `helmet_on_head`, `vest_loose`, `vest_on_body`.

> **742 exported images are not 742 independent samples.** The project holds
> **436 independent source images**; version 4 augments the train split x2
> offline (306 -> 612) and leaves validation (87) and test (43) unchanged. This
> was established from provider metadata, exact arithmetic and the provider's own
> export documentation - see
> [`reports/dataset_provenance.md`](reports/dataset_provenance.md).

Structural inspection found the export internally consistent: 742 image records
matching 742 files on disk, 3373 annotations, no dangling image or category
references, no duplicate ids. Segmentation geometry is a **mix of polygon (1570)
and RLE (1803)**.

### What the audit found (phase 4A)

All 436 originals were acquired at full resolution, measured and screened. See
[`reports/dataset_audit_report.md`](reports/dataset_audit_report.md) and
[`reports/eda_report.md`](reports/eda_report.md).

- **No exact duplicates.** 436 images, 436 unique content hashes.
- **11 near-duplicate candidates, 6 of them crossing a split boundary**, two of
  which are identical under both perceptual fingerprints. Phase 4A raised these
  as *candidates*, not established leakage; phase 4B confirmed them (below).
- **`vest_loose` appears in only 8 of 436 images** (45 instances): 7 train,
  1 valid, **0 test**. The rare class cannot be scored on the provider's test split.
- **17 source images carry no annotation.** Whether they are deliberate negatives
  or missing labels needed a person to look at them; phase 4B did.
- **The live source project now holds 76 more annotations than the frozen v4
  export**, so the two are not interchangeable and phase 5 must choose one.
- **The supplied COCO bbox disagrees materially with RLE segmentation geometry.**
  The preferred policy is to derive boxes from the geometry, but that decision is
  **provisional pending visual validation**.

### What the visual review decided (phase 4B)

The project owner and a technical reviewer looked at the phase 4A contact sheets
and answered the semantic questions counting cannot. The judgements are recorded
in [`reports/manual_audit_report.md`](reports/manual_audit_report.md) and, in
machine-readable form, in `reports/manual_audit_decisions.csv`. They are kept in
separate files from the computed artifacts on purpose: a human judgement is not a
measurement, and the two must never be quoted as if they were the same evidence.

- **All 6 cross-split near-duplicate pairs are semantic duplicates.** Same
  content, different bytes - the SHA-256 result stands, byte identity is *not*
  claimed. They form six groups that phase 5 must keep inside one split.
- **The provider split is `UNSUITABLE_FOR_FINAL_PROTOCOL`**: confirmed duplicates
  cross its boundaries, its test split has no `vest_loose` at all, and the rare
  class is too thin for a defensible per-class evaluation. **The split is
  rejected; the dataset is not.**
- **No widespread missing labels among the 17 zero-instance images.** Three are
  out of domain (a cartoon illustration, an office-like interior, a street scene)
  and become exclusion *candidates* for phase 5. Nothing was removed.
- **Segmentation-derived boxes are preferred, and now visually supported** - but
  the policy is `PREFERRED_AND_VISUALLY_SUPPORTED`, not applied. No annotation
  was converted.
- **The dataset is heterogeneous.** Site photography mixed with stock, posed
  portraits and product-style PPE images. It must be described as *mixed
  construction and PPE imagery*, and results on it do not demonstrate deployment
  performance on arbitrary construction-site video.

### Which annotations are canonical (phase 5A)

Phase 5A answered only one question: **which reproducible annotation state is the
source of truth**. It created no split, excluded no image and converted no
geometry. The decision and its evidence are in
[`reports/canonical_annotation_decision.md`](reports/canonical_annotation_decision.md)
and `reports/canonical_annotation_manifest.json`.

- **Decision: the live source project**, recovered read-only. 436 source images,
  2031 annotations, of which **2029 carry complete instance-segmentation
  geometry in original image coordinates** (1007 polygon, 1022 RLE).
- **Phase 4A's "1022 annotations have no geometry" was a consumption gap, not a
  provider limitation.** Those annotations carry their geometry inline as
  base64-wrapped, zlib-compressed COCO run-length encoding in a field the earlier
  walk did not read. No mutating call was made to recover it.
- **The decode is verified, not assumed.** Every recovered instance was
  re-measured and checked against the provider's own declared area and box; all
  2029 agree to **0.0 px**.
- **Version 4 was viable and was not chosen for coordinate fidelity reasons.**
  All 436 source images map to exactly one non-augmented v4 representation, so
  option A was available. It was rejected because v4 geometry is expressed after
  a stretch resize to 640x640, which would have to be inverted for every
  annotation and cannot recover what rasterisation discarded.
- **The drift adds no coverage.** The live state has 76 more annotations and 6
  fewer than v4 (net +70), but **every one of the 76 additions lies at least 80%
  inside an existing annotation of its own class**, at a median 0.44% of its
  area. None labels a previously unlabelled object; visual review indicates they
  are fragments drawn on details such as a bracelet and glove lettering. Phase 5B
  must decide their disposition as an explicit, recorded pipeline step.
  `vest_loose` is identical in both states.
- **The two unrecognised annotations are resolved** as
  `VALID_BUT_UNSUPPORTED_GEOMETRY`: both fall on a real distant worker, and their
  v4 counterparts are rectangles the exporter derived from the same boxes, not
  masks anyone drew. Nothing is lost by adopting the live state.

Newer is not treated as more correct. Neither state was compared against an
independent ground truth, because none exists for this dataset.

Open for phase 5B: building the modelling population - applying the three
out-of-domain exclusion candidates, keeping the six semantic duplicate groups
inside one split, deciding the 76 fragment additions, and designing the split
itself.

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
│   ├── external/              # Provider archive, source originals, provenance record
│   ├── raw/                   # Extracted canonical export, untouched
│   ├── interim/               # Derived representations (source geometry, image stats)
│   └── processed/             # Reserved: model-ready datasets (phase 5B onward)
├── notebooks/                 # Colab-executable notebooks (added by the phase that needs them)
├── reports/                   # The committed evidence; every number traces to a file here
│   ├── rubric_contract.md     # Rubric as a verifiable contract
│   ├── roadmap.md             # 14 phases with validation gates
│   ├── dataset_provenance.md  # What the dataset is, and how we know (phase 3)
│   ├── dataset_audit_report.md    # Automated audit (phase 4A)
│   ├── eda_report.md              # Source EDA (phase 4A)
│   ├── bbox_consistency_audit.md  # Supplied bbox vs segmentation geometry (phase 4A)
│   ├── manual_audit_report.md     # Human visual decisions (phase 4B)
│   ├── annotation_drift_report.md # Live source vs v4 snapshot (phase 5A)
│   ├── canonical_annotation_decision.md   # Which annotations are canonical (phase 5A)
│   ├── canonical_annotation_manifest.json # The same decision, machine-readable
│   └── figures/               # Contact sheets and analytical plots
├── scripts/                   # Command-line entry points, one job each
│   ├── check_environment.py       # Environment, configuration and holdout-lock report
│   ├── download_dataset.py        # Acquire + hash + extract the canonical export
│   ├── inspect_dataset.py         # Structural inspection -> provenance report
│   ├── fetch_source_inventory.py  # Recover the 436-image source inventory
│   ├── download_source_images.py  # Acquire the source originals at full resolution
│   ├── audit_source_dataset.py    # Duplicates, near duplicates, zero-instance images
│   ├── audit_bbox_consistency.py  # Supplied bbox vs the geometry it claims to enclose
│   ├── eda_source_dataset.py      # Source-population statistics and plots
│   ├── write_audit_reports.py     # Audit + EDA measurements -> written reports
│   ├── build_review_package.py    # Contact sheets for the human visual review
│   ├── record_manual_audit.py     # Validate and record the phase 4B decisions
│   ├── recover_source_geometry.py # Complete live geometry, read-only and verified
│   ├── map_v4_sources.py          # Source images -> non-augmented v4 representations
│   ├── analyze_annotation_drift.py # Live source vs v4 snapshot, per image
│   ├── build_drift_figures.py     # Review sheets for the drift and the odd records
│   └── resolve_canonical_snapshot.py # The canonical-snapshot decision + manifest
├── src/construction_safety_vision/
│   ├── config.py              # Strict typed configuration loading
│   ├── paths.py               # Repository layout, Colab support, long-path handling
│   ├── provenance.py          # Hashing and run provenance records
│   ├── splits.py              # Split identifiers and the holdout guard
│   └── data/                  # Acquisition, COCO inspection, geometry, drift, decision
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
cp .env.example .env          # then set ROBOFLOW_API_KEY to acquire the dataset
uv run python scripts/check_environment.py
```

Acquire the dataset. The key is read from the environment only, sent as an
`Authorization` header rather than a URL parameter, and never written to any
file, log or provenance record:

```bash
uv run python scripts/download_dataset.py
uv run python scripts/inspect_dataset.py
```

Reproduce the audit (phase 4). The first two steps contact the provider
read-only; the rest run offline:

```bash
uv run python scripts/fetch_source_inventory.py
uv run python scripts/download_source_images.py
uv run python scripts/audit_source_dataset.py
uv run python scripts/audit_bbox_consistency.py
uv run python scripts/eda_source_dataset.py
uv run python scripts/write_audit_reports.py
uv run python scripts/build_review_package.py
uv run python scripts/record_manual_audit.py
```

Reproduce the canonical-snapshot decision (phase 5A). Only the first step
contacts the provider, and only to read:

```bash
uv run python scripts/recover_source_geometry.py
uv run python scripts/map_v4_sources.py
uv run python scripts/analyze_annotation_drift.py
uv run python scripts/build_drift_figures.py
uv run python scripts/resolve_canonical_snapshot.py
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

## License and attribution

The **software** license of this repository is not yet chosen.

The **dataset** is a separate matter and is licensed **CC BY 4.0** by AGIs
Workspace. It is not redistributed here; `scripts/download_dataset.py` obtains it
from the original source. Attribution:

```text
Construction PPE Compliance Detection [dataset], version 4.
AGIs Workspace, Roboflow Universe. Licensed CC BY 4.0.
https://universe.roboflow.com/agis-workspace-8gs52/construction-ppe-compliance-detection/dataset/4
```
