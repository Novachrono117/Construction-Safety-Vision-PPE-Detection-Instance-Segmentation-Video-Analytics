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
| `train_detection_baseline.py` | 6B | Verify every frozen input, record the protocol, run the single D0 training, select the checkpoint by the predeclared rule and validate it once. |
| `freeze_detection_experiments.py` | 7A | Freeze how D1 and D2 will be judged, before either exists: derive the class-support rule's verdict, compute D0's selection metric, resolve both candidate protocols and prove each is a one-variable comparison. Trains nothing. |
| `train_detection_experiment.py` | 7B+ | Run one frozen Phase 7 candidate. Takes an experiment id, never hyperparameters: the protocol is resolved by inheriting D0's and applying the declared override set. Proves the one-variable contract before fetching weights. |
| `freeze_final_detector.py` | 7D | Apply the frozen Phase 7 policy to the committed results, record the human-reviewed decision and freeze the selected checkpoint's identity. Trains, validates, benchmarks and tunes nothing. |
| `audit_segmentation_adapter.py` | 8A | Measure how much canonical COCO instance-mask geometry survives the Ultralytics YOLO segmentation label format. Development splits only. Trains nothing, downloads nothing, selects no architecture. |

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

`train_detection_baseline.py` (phase 6B) runs one predeclared experiment and is
ordered so that the protocol demonstrably precedes the result: every frozen
fingerprint is re-checked first (a mismatch is `PROTOCOL_INPUT_MISMATCH` and
nothing runs), a pre-run provenance record is written **before the first
optimisation step**, and only then does training start. It refuses to reuse or
overwrite an existing `D0` directory, because one experiment means one run.

Two habits in it are worth knowing about. The protocol declares
`optimizer: auto`, so the script reads the framework's *resolved* arguments back
off disk and reports those - quoting the configuration file's `lr0` would
describe a learning rate the run never used. And the headline metric is
cross-checked against `results.csv` at the independently recomputed best epoch
before anything is published, so a number belonging to `last.pt` or to an
earlier run cannot be reported as D0's.

`--resume` exists for operational interruptions only: it continues the same run
from its checkpoint with identical hyperparameters and records that it did. It
is not a way to restart a failed experiment with different settings.

`freeze_detection_experiments.py` (phase 7A) writes rules rather than results,
and it is ordered so that the rules cannot have been fitted to a number that
does not exist: it refuses to run at all if `CSVISION_ALLOW_TEST_SPLIT` is set,
it refuses to build the D0 reference if the committed result was produced under a
different protocol fingerprint than `detection_baseline.yaml` currently holds,
and it exits non-zero if either candidate turns out to differ from D0 in an
undeclared field. It re-reads the committed D0 manifest and never re-runs
training or validation.

Its output is deterministic: sorted keys, no timestamp in the policy or the
reference, so running it twice produces byte-identical artifacts. The class set
that enters the selection metric is *derived* from the frozen split manifest by
applying the support thresholds - no class is named in the code - which is what
makes the exclusion of the rare class a consequence of its evidence rather than
a decision about the class.

`train_detection_experiment.py` (phase 7B onward) is the general runner for a
Phase 7 candidate, and its ordering is the point. Before a single weight is
downloaded: the holdout guard is checked, the committed policy is verified
against the configuration it names, the D0 reference is validated from its
committed manifest, and the candidate's protocol is resolved and **proven** to
differ from D0 only where it declared. A candidate that differs anywhere else
exits non-zero and trains nothing.

It accepts no hyperparameter flags, because a Phase 7 candidate has none of its
own - `--experiment D1` is the whole specification. `--verify-only` runs every
pre-flight check and stops; `--memory-preflight` proves the frozen batch fits by
taking one throwaway optimisation step in a separate directory that is deleted
afterwards (marked `NON_EXPERIMENTAL`, validation disabled, no metric recorded).

Three refusals worth knowing about. A genuine CUDA OOM at the frozen batch stops
the experiment as `MEMORY_CONSTRAINT_REVIEW_REQUIRED` rather than reducing the
batch, because batch is a controlled variable. `--resume` requires
`--resume-reason` and continues the same run with identical settings; it is for
operational interruptions, not for restarting a failed experiment differently.
And after training it re-checks that the framework wrote into the directory the
result is read from - Ultralytics silently diverts to `<id>-2` if the directory
already exists, which would otherwise publish metrics under an experiment id
they did not come from.

The optimizer is established rather than assumed. `optimizer: auto` is the
frozen policy, so the run tees the framework's log to `train_console.log` and
reads the line where it names the optimizer it built. That line is colourised,
so the escape codes are stripped before parsing; without that step the parse
fails on the real format and the run falls back to inference for no reason. The
manifest records which of the two happened, and an inferred value is labelled
`inferred`.

Two more guards. A candidate that does **not** override `weight_identifier` is
asserting it starts from the same *bytes* as the reference, so the runner
compares digests against the reference's committed record and refuses to run if
the asset changed - checking the file name alone would let a silently replaced
download add an undeclared variable. Conversely, a candidate that *does* declare
new weights and then loads the reference's bytes is refused too: the variable was
never applied. The verdict is recorded as `pretrained_weight_identity`.

`--rebuild-report` re-renders the report and the live results table from a
committed manifest. It trains nothing, validates nothing, loads no checkpoint and
recomputes no metric, and it verifies afterwards that the manifest is still
byte-identical - so prose can be corrected or extended without touching a
published result.

Selection is deliberately **not** made by this script. An experiment phase
reports the numbers the frozen rule needs and stops; while it is the last
candidate to run, the live results artifact records the computed case as an
`advisory_only` `policy_case_candidate` with `preferred_experiment: null`.
Freezing the project's detector is a reviewed step of its own, and it belongs to
`freeze_final_detector.py`.

`freeze_final_detector.py` (phase 7D) is that step. It trains nothing, validates
nothing, runs no inference, benchmarks no latency, tunes no threshold and opens
no image; every number it emits was read from a committed result manifest, and
the only computation is arithmetic over those numbers plus SHA-256 over files
already on disk.

The winning experiment's id is never written into it. The script rebuilds each
record from its committed manifest, re-derives the class-support filter from the
frozen split, recomputes the selection metric and asks
`detection_comparison.compare` which case holds - then refuses unless the answer
is the one Phase 7D is authorised for, a single validation-performance leader. A
test asserts the id appears nowhere as a constant, because "the policy chose it"
is only worth something if the script could not have said so on its own.

Two independent checks stand behind the one-variable claim: the contract is
re-derived from the committed configuration *and* required to agree with the
verdict each experiment recorded when it ran, so configuration drift after a run
cannot pass. The six historical D0/D1/D2 artifacts and the frozen 7A policy are
hashed before and after and must be byte-identical; the freeze never regenerates
them.

The checkpoint is verified by digest against the committed manifest and copied
byte-identically to `artifacts/frozen/detection/<id>_best.pt` - deliberately
outside the training run directory, which a re-run would overwrite. A missing
binary is `BLOCKED_MISSING_MODEL_ARTIFACT` and the script says to obtain it
rather than retrain, because a re-run produces different weights under the same
experiment name. A destination that already exists with a different digest fails
loudly and is never overwritten.

`--verify-only` runs every check and derives the selection without writing
anything.

`construction_safety_vision.detection_freeze` is the accessor future phases
should use. It resolves the frozen checkpoint by digest rather than by path,
prefers the immutable copy over the run directory, and refuses `last.pt` by name
as well as by digest - the failure mode being that a YOLO checkpoint loads
whatever bytes it is handed and produces plausible numbers from the wrong model.

`audit_segmentation_adapter.py` (phase 8A) answers one question - how much of a
canonical COCO instance mask is still there after a round trip through the YOLO
segmentation label format - and refuses to answer any other. It trains nothing,
downloads no weights, and selects no architecture; a good IoU distribution is
not an approval and a poor one is not a rejection.

Three design decisions carry the audit's credibility.

**It measures the label as written to disk.** Comparing against an in-memory
object would skip serialisation, float32 parsing and the integer snap - three of
the places fidelity is actually lost. The row is written, re-read with the
framework's own parsing semantics, denormalised and rasterised by Ultralytics'
own `polygon2mask`.

**It separates the rasteriser from the format.** pycocotools and OpenCV disagree
on boundary-pixel inclusion, so even an unchanged polygon will not reproduce the
canonical mask. Charging that to the YOLO format would be wrong, so every
instance is measured at three levels - before conversion, after component
joining, after serialisation - and each stage is charged only for what it
consumed. That decomposition changed the finding: joining costs 0.000843 mean
IoU and serialisation costs 0.012458, so the loss a naive audit would have
blamed on multi-component masks is mostly coordinate quantisation.

**One canonical annotation becomes exactly one row, always.** Splitting a
disconnected mask into several rows would raise every fidelity number and would
quietly redefine what an instance is. The invariant is enforced in code, and it
is re-verified by handing the result to the framework's own dataset scanner -
which also catches the latent hazard that Ultralytics drops duplicate
`(class, box)` rows with `np.unique`, so two same-class instances sharing a box
would silently collapse into one.

Conversion uses the framework's primitives wherever they exist
(`merge_multi_segment`, the same contour settings its own mask converter uses),
so the audit measures Ultralytics rather than a private variant. Every
methodological choice that could move a number lives in
`configs/segmentation_adapter_audit.yaml`, not in Python.

`--verify-only` checks the canonical inputs, the class map and the frozen
detector, then stops without writing anything. The adapter it generates is
git-ignored and marked `AUDIT_ONLY` / `NOT_CANONICAL` /
`NOT_YET_APPROVED_FOR_TRAINING`: it is a measurement instrument, not the
project's segmentation dataset.

## Planned scripts

Evaluation, error analysis and video inference scripts are added by their
respective roadmap phases. None are stubbed in advance.
