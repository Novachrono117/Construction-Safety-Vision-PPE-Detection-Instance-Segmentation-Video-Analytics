# scripts/

Command-line entry points. Anything that produces a reported number runs from
here (or from a notebook that only calls into `src/`), never from an
undocumented one-off shell invocation.

| Script | Phase | Purpose |
| --- | --- | --- |
| `check_environment.py` | 2 | Report interpreter, platform, repository root, configuration validity and holdout lock state. |
| `download_dataset.py` | 3 | Acquire the canonical COCO instance-segmentation export, hash it while streaming, extract it safely, and write its provenance record. Idempotent. |
| `inspect_dataset.py` | 3 | Structurally inspect the extracted export and write `reports/dataset_provenance.{md,json}`. Read-only; no EDA. |
| `fetch_source_inventory.py` | 4A | Walk the provider's source inventory and recover the 436 independent images with their per-class annotation counts. |
| `download_source_images.py` | 4A | Acquire the 436 source originals at full resolution into the git-ignored external layer. |
| `audit_source_dataset.py` | 4A | Exact duplicates, perceptual near-duplicate candidates, zero-instance images, split-crossing checks. |
| `audit_bbox_consistency.py` | 4A | Compare each supplied COCO `bbox` against the box its own segmentation implies. |
| `eda_source_dataset.py` | 4A | Source-population statistics and the plots under `reports/figures/`. |
| `write_audit_reports.py` | 4A | Turn the audit and EDA measurements into the written reports. Re-run after phase 4B. |
| `build_review_package.py` | 4A | Contact sheets so a person can answer the questions counting cannot. |
| `record_manual_audit.py` | 4B | Validate the declared human judgements against the manifests and record them. |
| `recover_source_geometry.py` | 5A | Recover complete instance-segmentation geometry for the live source state, read-only, verifying every instance against the provider's own area and box. |
| `map_v4_sources.py` | 5A | Map each source image to its one non-augmented version-4 representation, by filename identity plus a pixel comparison against a deterministic re-render. |
| `analyze_annotation_drift.py` | 5A | Per-image comparison of the live annotations against the version-4 snapshot, counting additions and removals separately. |
| `build_drift_figures.py` | 5A | Review sheets L, M and N: the drift, the geometry-less records, and every added annotation. |
| `resolve_canonical_snapshot.py` | 5A | Apply the decision rule and write `reports/canonical_annotation_{decision.md,manifest.json}`. |
| `analyze_fragment_rule.py` | 5B | Test whether a geometry rule computed from the current state alone can identify the annotations added since version 4. Scores candidates against the v4 diff; exits non-zero when none is acceptable. |
| `build_modeling_population.py` | 5B | Decide which images and annotations may be modelled, materialise geometry-less records, and build the indivisible split units. Creates no split. |
| `build_remaining_duplicate_review.py` | 5B.1 | Draw the near-duplicate candidates that still carry no human decision, so the gap is closed by looking. Merges nothing. |
| `optimize_split_candidates.py` | 5C.1 | Search for provisional train/validation/test assignments over the canonical groups and compare them. Selects nothing and freezes nothing. |
| `freeze_split.py` | 5C.2 | Re-verify the human-selected candidate, freeze it as the authoritative split, and lock the holdout. Freezes membership only. |
| `materialize_task_datasets.py` | 5D | Build the canonical COCO detection and instance-segmentation views of `train` and `validation`. Never materialises the holdout. |
| `build_detection_adapter.py` | 6A | Derive YOLO detection labels from the canonical COCO detection dataset, with a per-box fidelity audit. Detection only. |
| `detection_runtime_check.py` | 6A | Verify the CUDA runtime by executing real kernels, fingerprint the pretrained weights, and optionally run a minimal smoke test. |

## Rules

- Every script takes its settings from a file in `configs/` and writes a
  provenance record for what it produced.
- Scripts that touch the `test` split must go through
  `construction_safety_vision.splits.assert_split_allowed` and require the
  documented double opt-in.
- Prefer `uv run python scripts/<name>.py` so the pinned environment is used.

`download_dataset.py` requires `ROBOFLOW_API_KEY` in the environment. The key is
read from the environment only, sent as an `Authorization` header rather than a
URL parameter, and never written to any file, log or provenance record.

Phase 4A scripts run in this order: `fetch_source_inventory` -> `download_source_images`
-> `audit_source_dataset` -> `audit_bbox_consistency` -> `eda_source_dataset` ->
`build_review_package` -> `write_audit_reports`. The first two need
`ROBOFLOW_API_KEY`; the rest are offline.

`record_manual_audit.py` (phase 4B) turns the human visual review of that package
into `reports/manual_audit_decisions.csv` and `reports/manual_audit_report.md`. It
reads no dataset image and makes no judgement of its own: the judgements are
declared as data at the top of the file, and the script resolves them against the
phase 4A manifests, refusing anything that does not resolve unambiguously. Run it
with `--check` to re-validate the committed record without rewriting it.
`write_audit_reports.py` then cross-references the outcome; re-run it after
recording. Both are offline.

Phase 5A scripts run in this order: `recover_source_geometry` -> `map_v4_sources`
-> `analyze_annotation_drift` -> `build_drift_figures` ->
`resolve_canonical_snapshot`. Only the first needs `ROBOFLOW_API_KEY`, and it only
reads: it issues one documented GET per source image and makes no mutating call.
It writes the bulk geometry to the git-ignored interim layer and a counts-only
summary to `reports/`.

`resolve_canonical_snapshot.py` refuses to write a manifest that is internally
inconsistent or that contains a signed URL, a credential or an absolute local
path. That check is a gate, not a formality: the manifest is the evidence a
reader checks the project's headline numbers against.

Phase 5B scripts run in this order: `analyze_fragment_rule` ->
`build_modeling_population`. Both are offline and read only committed artifacts,
so every decision they encode is traceable to a file rather than to a list typed
into the script. `analyze_fragment_rule.py` uses the version-4 export **only** to
score candidate rules; no rule it evaluates may read the export, the provider
split, or a list of identifiers.

`build_modeling_population.py` excludes logically and never deletes: an excluded
image keeps its file, keeps its place in the source provenance population of 436,
and gains a reason and a decision source. It refuses to write a manifest whose
invariants are broken.

`optimize_split_candidates.py` assigns **groups**, never images, so a confirmed
duplicate pair cannot be separated. It never reads the provider split - rejected
in phase 4B - and refuses to run if the group feature table carries one. Its
output column is `provisional_split` rather than `split`, so nothing downstream
can mistake a candidate for a frozen assignment. The protocol lives in
`configs/split_search.yaml`, parsed strictly, and the whole search is a pure
function of that file and the group features.

`freeze_split.py` (phase 5C.2) promotes the human-selected candidate to the
project's authoritative split and locks the holdout. It **selects nothing**: the
selection is a human decision recorded in `configs/split_freeze.yaml`, which the
script only enforces. Every expectation is declared in that file in advance -
the candidate's assignment digest, the phase 5B population and group
fingerprints, and the exact image, group, negative and rare-class counts - and
verification runs to completion **before anything is written**. If the candidate
no longer reproduces, the script writes nothing and exits non-zero rather than
freezing a different assignment that happens to be feasible.

It is idempotent by construction: no timestamp enters `split_manifest.json`,
`final_split_assignments.csv`, `split_freeze_report.md` or
`split_candidates/selection.csv`, so re-running over the same inputs reproduces
them byte for byte. Only `split_freeze.provenance.json` changes, in its
`created_at`, because that is what a run record is for. Use
`--verify-only` to run every check and write nothing.

It froze **membership only**. It copies no image, writes no label, resizes
nothing and creates nothing under `data/processed/`; phase 5D does that. It never
reads the provider's rejected split, and it touches the phase 5C.1 candidate
files not at all - the one thing it rewrites in that phase's output is the
status sentence in `split_candidate_report.md` that would otherwise still claim
no candidate had been selected.

`materialize_task_datasets.py` (phase 5D) turns the frozen membership into two
COCO views of the same data. Its whole claim is that it **adds nothing**: images
are transferred with binary copy semantics and never routed through an image
library, so no decoder can quietly re-encode them, and both sides are hashed
after the copy so byte-identity is measured rather than asserted. Canonical
geometry is carried through unchanged - a polygon stays a polygon, a compressed
RLE stays a compressed RLE - and the emitted file is read back from disk and
compared against the canonical state, RLE by decoded mask rather than by
comparing `counts` strings.

It writes **`train` and `validation` only**. The holdout is materialised by the
same function with the same configuration, which is the point: it cannot receive
different preprocessing than the data the models were developed on. Reaching it
requires `allow_test=True` and `CSVISION_ALLOW_TEST_SPLIT=1`, and the code path
is exercised only against synthetic fixtures until the final-evaluation phase.

Two things it deliberately does not do. It writes **no YOLO labels**: only COCO
carries both polygon and RLE natively, so a conversion now would approximate the
ground truth before a model has been chosen, and a future adapter must first pass
a documented mask-IoU fidelity audit. And it never reads the provider's rejected
split - membership comes from `reports/split_manifest.json` alone, and the
population loader names every column it consumes so a provenance column could not
influence a destination even if one were added.

The protocol lives in `configs/task_materialization.yaml`, parsed strictly. Use
`--verify-only` to run every validator and write nothing. Re-running is
byte-stable: no timestamp enters an emitted file and COCO ids come from a sorted
canonical ordering rather than filesystem traversal.

`build_detection_adapter.py` (phase 6A) produces a **derived** representation.
Canonical COCO remains the ground truth, and the script exists together with the
audit that proves it changed nothing: every box is converted, written, read back
from disk and decoded, because comparing against the in-memory boxes would prove
nothing about the files a trainer loads. It writes detection labels only - RLE
masks cannot become YOLO polygons without loss - and it never reads the
provider's bbox or the holdout.

`detection_runtime_check.py` (phase 6A) refuses to trust
`torch.cuda.is_available()`. A torch build can report a device it has no kernels
for, which is a live risk on a Blackwell GPU, so the preflight runs a matmul
checked against the CPU, a convolution backward pass and an AMP step. If a GPU is
visible through the driver but unusable, it reports `BLOCKED_FOR_GPU` rather than
falling back to CPU, because a CPU baseline is not the same experiment. Its
`--smoke-test` runs one epoch purely to prove the stack executes; those metrics
are marked `NON_EXPERIMENTAL` and never recorded as results.

## Planned scripts

Training, evaluation, error analysis and video inference scripts are added by
their respective roadmap phases. None are stubbed in advance.
