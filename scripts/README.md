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
| `freeze_segmentation_baseline.py` | 8B | Record the architecture decision, approve the audited adapter by digest, freeze the S0 protocol and run one non-experimental smoke test. Does not run S0 and reports no model performance. |
| `train_segmentation_baseline.py` | 8C | Run the S0 baseline exactly once, validate the natively selected checkpoint once, and execute the predeclared direct mask-IoU diagnostic once. Selects no final segmenter and tunes nothing. |
| `analyze_segmentation_errors.py` | 8D | Attribute S0's validation errors one canonical instance at a time, from a frozen taxonomy and a deterministic review set. Trains nothing, re-validates nothing and selects nothing. |
| `freeze_segmentation_comparison.py` | 8E | Freeze the canonical COCOeval protocol, evaluate S0 under it exactly once, freeze S1 as a one-variable `overlap_mask` change, and size its batch. Trains nothing. |
| `train_segmentation_comparison.py` | 8F | Run S1 exactly once under the frozen phase 8E protocol, validate it natively once, evaluate it under the canonical evaluator once, and run the phase 8C direct mask-IoU diagnostic once. Reads S0, never re-runs it, and selects no final segmenter. |
| `freeze_final_segmenter.py` | 8G | Apply the frozen phase 8E policy to the committed S0 and S1 results, record the human-reviewed selection, and freeze the selected checkpoint's identity. Trains nothing, evaluates nothing and runs no inference. |

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
git-ignored and was marked `AUDIT_ONLY` / `NOT_CANONICAL` /
`NOT_YET_APPROVED_FOR_TRAINING` when phase 8A wrote it. Phase 8B approved those
exact bytes for controlled training; the approval lives in
`reports/segmentation_adapter_approval.json` rather than in the generated
directory, because that directory's own README is a phase 8A output and editing
generated evidence by hand is worse than a stale line inside a git-ignored file.

`freeze_segmentation_baseline.py` (phase 8B) records the human architecture
decision, approves the audited label bytes, freezes the S0 protocol and proves
the runtime executes. **It does not run S0**, and no accuracy figure it observes
reaches an artifact.

The order of its checks is the point.

**Phase 8A and phase 7D are digested first and digested again last.** "The
historical evidence is unchanged" is a measurement here, not an intention: ten
files are hashed at entry, hashed at exit, and compared. The frozen detector's
manifest is additionally recomputed from its own contents and its checkpoint
bytes verified - and then left alone, because the detection block is closed.

**The adapter is verified before a single weight is fetched.** All four phase 8A
label fingerprints are recomputed from the files on disk, and the instance and
image counts with them. A mismatch is `ADAPTER_FINGERPRINT_MISMATCH` and stops
the phase; it never triggers a rebuild, because a rebuild would replace measured
bytes with unmeasured ones. The same check runs again after the smoke test.

**The protocol's framework arguments are checked against the installed library.**
Every segmentation-specific and augmentation value declared in
`configs/segmentation_baseline.yaml` must equal the installed effective
configuration, so the file cannot claim a default the library does not have, and
each general hyperparameter is classified `FRAMEWORK_DEFAULT` or
`PROJECT_OVERRIDE` in the manifest rather than left for a reader to work out.

**The checkpoint rule's behaviour is read before training, not after.** For a
segmentation model Ultralytics' validation fitness is the unweighted sum of box
and mask mAP@0.50:0.95, so `best.pt` is not selected on the mask metric alone.
That is established by inspecting the installed source and checked structurally,
so a future version that changes the definition fails the phase instead of
quietly invalidating the report. The behaviour was returned for methodological
review before S0 and the native composite was **accepted** rather than replaced
by a custom mask-only selector; the config records the policy, its semantics,
both component weights and - explicitly - that the selector is **not** the
project's reported metric.

The smoke test is one epoch at the frozen batch and resolution, marked
`NON_EXPERIMENTAL` / `DO_NOT_REPORT_AS_MODEL_RESULT`. It records success,
runtime, peak GPU memory, the resolved optimizer and whether a checkpoint was
written - and no accuracy number whatever. Plots are disabled so it renders no
dataset or prediction imagery, and the framework label caches it leaves inside
the audited directory are deleted afterwards, so those bytes stay exactly as
phase 8A wrote them. A genuine CUDA out-of-memory event is
`MEMORY_CONSTRAINT_REVIEW_REQUIRED`: the batch is never reduced to rescue it.

`--verify-only` runs every precondition, fetches no weights, trains nothing and
writes nothing.

`--rebuild-protocol` re-renders the artifacts from the committed manifest's own
smoke record, for a protocol clarification that changes no runtime behaviour.
Every verification still runs, but nothing is trained: re-running the smoke test
would replace a recorded runtime and peak-memory figure with a different one for
no reason. The reused record is checked rather than trusted - it must report
success, must carry no metrics, and the pretrained checkpoint on disk must hash
to the bytes it ran on, so one run's evidence cannot be attributed to another's.

`train_segmentation_baseline.py` (phase 8C) runs the S0 baseline. One training
run, one validation, one diagnostic - and no decision.

Four design decisions carry the experiment's credibility.

**Training reads a hard-linked runtime view, not the audited adapter.**
Ultralytics writes `.cache` files beside the labels it scans, and the audited
adapter is phase 8A evidence. So the run gets a disposable view whose label bytes
are verified identical to the approved digests first, before and after training.
The audited directory ends the phase exactly as phase 8A wrote it.

**The selected checkpoint is verified against the rule, not assumed.** The
framework writes `best.pt` by its own composite fitness and records the terms but
not the fitness, so the composite is recomputed from `results.csv` as
`metrics/mAP50-95(B) + metrics/mAP50-95(M)` and the recorded best epoch is
confirmed to be its argmax. Without that step, "the frozen rule chose this
checkpoint" would be an article of faith.

**The direct IoU diagnostic scores against canonical COCO masks.** Never the
YOLO adapter: its own round-trip error was measured in phase 8A, and scoring
against it would fold that approximation into the model's result by an amount
nobody could separate afterwards. Predictions are produced with
`retina_masks=True` so they land on the original image canvas the canonical masks
live on, rather than this project resampling them and measuring its own
interpolation.

**An out-of-memory event stops the phase.** `MEMORY_CONSTRAINT_REVIEW_REQUIRED`;
the batch is never reduced, auto-batch is never enabled and imgsz is never
lowered to rescue a run.

The script refuses to start if a completed S0 run already exists, because exactly
one valid full run is authorised and overwriting it would destroy the experiment
being reported. `--verify-only` runs every precondition and writes nothing;
`--diagnostic-only` re-renders the artifacts from a completed run without
training, for recovering from a failure after training rather than for repeating
the experiment.

`analyze_segmentation_errors.py` (phase 8D) explains what the S0 aggregates do
not. It trains nothing, re-validates nothing, and selects no segmenter.

Four properties keep it an analysis rather than a search.

**It reuses the phase 8C protocol instead of restating it.** The inference
settings and the matching rule are read from the committed diagnostic
configuration and asserted, so the phase cannot quietly become a threshold
sweep. It then re-derives the committed aggregates from its own per-instance
table and refuses to continue if they disagree - the published result stands and
the analysis is what would be wrong.

**The taxonomy is frozen before any image is opened**, and mechanism flags an
automated pass may assign are separated from the ones only a person can.
`INSTANCE_SEPARATION_ERROR` and `OCCLUSION_ASSOCIATED` are deliberately not
derivable from an IoU and an area ratio; assigning them automatically would
manufacture a census out of arithmetic. An instance nothing explains stays
`UNATTRIBUTED`.

**The review set is chosen from the table before inspection.** Deterministic
slices of a totally ordered list, identifiers written out first, and no example
swapped afterwards for a more photogenic one. Human judgements are then declared
as data in the script - the convention `record_manual_audit.py` established in
phase 4B - and validated against that set: a judgement about an unselected
instance is refused, as is any label naming a cause a controlled experiment would
be needed to establish.

**Review figures stay git-ignored.** They overlay canonical and predicted masks
on source images, and this repository does not publish dataset imagery.

`--verify-only` checks every precondition and writes nothing; `--build-review`
computes the table, records the selection and renders its figures, then stops
before any human judgement is recorded.

`freeze_segmentation_comparison.py` (phase 8E) does four things in one pass, in
an order that matters: it freezes a common canonical evaluator, evaluates the
frozen S0 checkpoint under it **exactly once**, freezes S1 as a one-variable
`overlap_mask` intervention, and proves the frozen batch still fits with the flag
off. It trains nothing and selects no segmenter.

**The evaluator is validated before it is used.** Synthetic fixtures check that a
perfect prediction scores 1.0, a disjoint or wrong-class one scores 0.0,
duplicates cannot raise the score, ordering matters, and RLE round-trips
pixel-exactly - all before any real checkpoint is loaded. An evaluator validated
after producing the number it was built to produce validates nothing.

**Predictions reach the metric without this project resampling them.** Masks come
back on the original image canvas through the framework's own native mask path,
then go through pycocotools' reference encoder - the same library that decodes
the canonical ground truth, so both sides of every IoU come from one
implementation. A prediction whose canvas does not match its image's original
size is refused rather than resized.

**The one-variable contract is verified against S0's own protocol**, not against
a second copy of it, and a variable declared but never applied is refused as
firmly as an undeclared one.

**The feasibility check takes no optimizer step.** One real batch, a forward
pass, the loss, a backward pass - enough to size the target tensor that grows
when each instance gets its own mask plane, and nothing more. No validation, no
checkpoint, no metric. A genuine OOM is `MEMORY_CONSTRAINT_REVIEW_REQUIRED`; the
batch is never reduced to rescue it.

`--verify-only` runs every precondition and writes nothing.

`train_segmentation_comparison.py` (phase 8F) executes S1. Exactly one training
run, one native validation, one canonical evaluation, one direct mask-IoU
diagnostic. It selects no final segmenter, tunes nothing and never touches the
holdout.

**S1 has no protocol of its own.** Its framework arguments are resolved in code
from S0's, with the single override the frozen comparison protocol declares, so
a drift in S0's protocol surfaces as a changed S1 argument rather than being
masked by a second copy of the same values. The resolved set is then checked
against the `inherited_protocol` block phase 8E recorded, and the run's own
`args.yaml` is checked afterwards - the intervention has to have actually
reached the trainer, not merely been requested.

**The native validation is given `overlap_mask: false` explicitly.** Read from
the installed source: `Model._reset_ckpt_args` keeps only `imgsz`, `data`,
`task` and `single_cls` from a checkpoint, and the framework default is `True`.
Omitting the flag would have scored S1's predictions against S0's
overlap-resolved target, which is the precise confusion the phase exists to
avoid. The runner refuses to continue if the resolved value is not `False`.

**S0 is read, never executed.** Its checkpoint bytes are verified by digest and
its committed canonical, native and direct-IoU figures are quoted. Every phase
8A-8E artifact is digested before and after and required to be byte-identical.

**Four validators run before anything is published**, not after: the result
manifest, the canonical evaluation, the direct-IoU diagnostic and the
S0-versus-S1 comparison, whose delta is recomputed from the committed S0 value
rather than trusted. `--validate-only` re-runs all four against the committed
artifacts.

`--verify-only` runs every precondition and writes nothing.

`freeze_final_segmenter.py` (phase 8G) selects and freezes the project's
segmenter. It **trains nothing, evaluates nothing and runs no inference** - a
test asserts the script contains no such call path, because a freeze that could
train is a freeze that might.

**The winner is derived, not asserted.** Both primary figures are read out of
each experiment's own committed canonical evaluation, the delta is recomputed,
and the frozen margin is reapplied by the same function the policy names. No
experiment id appears in the script as the answer, and a test asserts that too.
Human review **confirms** the derived result; if the arithmetic said anything
else, the script stops as `INVALID_SELECTION_STATE` rather than record a
selection the evidence does not support.

**The checkpoint is frozen by digest.** The runtime file is verified against the
committed manifest before anything is written, and a byte-identical copy is
placed outside the run directory a re-run would overwrite. An existing frozen
copy holding different bytes stops the phase rather than being replaced.

**Two wrong checkpoints are rejected by identity.** S0's `best.pt` and any
`last.pt`. S0 and S1 are the same architecture, the same size and the same
number of bytes, differing only in what they were trained to predict, so a
digest check is the only thing that separates them.

`--verify-only` derives the comparison and writes nothing.

## Planned scripts

Evaluation, error analysis and video inference scripts are added by their
respective roadmap phases. None are stubbed in advance.
