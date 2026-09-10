# configs/

Versioned experiment configuration. Everything that changes an experimental
outcome lives here, never in notebook cells or ad-hoc command-line flags.

| File | Purpose |
| --- | --- |
| `project.yaml` | Project-level settings: seed, split ratios, declared dataset. |
| `split_search.yaml` | Phase 5C.1 split-search protocol (see below). |
| `split_freeze.yaml` | Phase 5C.2 split-freeze protocol. |
| `task_materialization.yaml` | Phase 5D canonical COCO task-dataset materialisation. |
| `detection_adapter.yaml` | Phase 6A YOLO detection adapter. |
| `detection_baseline.yaml` | Phase 6A D0 protocol: model, every hyperparameter, metric hierarchy. |
| `detection_experiments.yaml` | Phase 7A comparison protocol: D0/D1/D2, support rule, margin (see below). |
| `segmentation_adapter_audit.yaml` | Phase 8A YOLO segmentation-adapter fidelity audit: a measurement, not a dataset. |
| `segmentation_baseline.yaml` | Phase 8B S0 protocol: architecture, every hyperparameter, adapter digests, mask metric hierarchy (see below). |
| `segmentation_mask_iou_evaluation.yaml` | Phase 8C direct instance-mask IoU diagnostic: ground truth, operating point, matching rule (see below). |
| `segmentation_canonical_evaluation.yaml` | Phase 8E common COCOeval protocol: the one yardstick every segmentation experiment is scored by (see below). |
| `segmentation_comparison.yaml` | Phase 8E S0-vs-S1 policy: primary metric, margin, the one-variable contract (see below). |
| `detector_segmenter_comparison.yaml` | Phase 10A detector-versus-segmenter comparison protocol: the two frozen models, two inference protocols, the spatial features and their box proxies, the association rule, and the latency benchmark (see below). |

## Rules

- One file per experiment. Configuration files are append-only in spirit: to run
  a variant, add a new file instead of editing a file whose results are already
  reported.
- Parsing is strict (`construction_safety_vision.config`). An unknown or
  misspelled key raises instead of being silently ignored.
- Every run embeds its configuration snapshot in a provenance record, so a
  reported number can always be traced back to the exact settings that produced it.
- Hyperparameter files for detection and segmentation are added by phases 6-9.

## `split_search.yaml`

The phase 5C.1 split-search protocol: target ratios and integer image counts,
the hard constraints a candidate must satisfy, the rare class to protect and its
per-split floors, the allocation families to search, the objective weights and
the search parameters. Parsed strictly - an unknown key raises rather than being
ignored, so a typo cannot silently change the protocol.

It defines how candidates are *searched for and scored*. It does not select one
and it freezes nothing. The provider's split is absent by design.

## `detection_experiments.yaml`

The phase 7A controlled-comparison protocol, frozen while D1 and D2 did not yet
exist: the class-support rule that decides which classes may order two models,
the primary selection metric, the mandatory all-class metric, the
practical-equivalence margin, the batch/memory policy and the two authorised
experiments.

Its shape is the point. D1 and D2 carry **no protocol of their own**: they name
`inherits: D0` plus a set of dotted-path `overrides`, and D0's protocol is read
from `detection_baseline.yaml`. So every shared hyperparameter is stated once
and cannot drift between experiments, and "only one thing differs" is checkable
rather than asserted - the parser rejects a candidate whose `overrides` do not
exactly match its declared `intentional_fields` plus `consequential_fields`.

Parsing is strict in two extra ways beyond unknown keys: the support thresholds
and the margin must equal the frozen constants in
`construction_safety_vision.detection_comparison`, and the metric names must be
the frozen ones. Relaxing a threshold or renaming the deciding metric after a
result exists is exactly the failure the file is meant to prevent, so it raises
rather than loads.

## `segmentation_baseline.yaml`

The phase 8B S0 protocol, frozen before S0 was trained and before any
segmentation performance number existed. Parsed by
`construction_safety_vision.segmentation_experiment`, deliberately a **sibling**
of the detection schema rather than an extension of it: `detection_baseline.yaml`
is inherited by D0, D1 and D2 and its digest is recorded in three committed
manifests, so widening its parser would put an untested change underneath a
frozen comparison.

Parsing is strict in five ways beyond unknown keys:

- **the holdout may not appear anywhere**, keys included, checked recursively;
- **the primary metric must be `mask_mAP@0.50:0.95`**, and mask sections may
  contain only `mask_*` metrics while box sections may contain only `box_*` -
  merging the families is how a weak mask result hides behind a strong box one;
- **a composite box-plus-mask score is refused outright** rather than left
  available for a later phase to reach for;
- **the adapter is named by digest**, and the declared instance counts must add
  up, with `all_instances_retained: true` and `fidelity_based_filtering: NONE`
  enforced - excluding instances after observing adapter fidelity would change
  the modelling population to suit the model format;
- **each framework argument is declared exactly once** across the `training`,
  `segmentation_arguments` and `augmentation_arguments` blocks, so no precedence
  rule is needed to know what ran.

It also carries the reviewed checkpoint-selection decision as its own block, and
the parser enforces every part of it: the policy is
`ULTRALYTICS_NATIVE_SEGMENTATION_FITNESS`, its semantics are
`BOX_MAP50_95_PLUS_MASK_MAP50_95` at component weights 1.0 and 1.0, the review
status is `HUMAN_REVIEWED_AND_ACCEPTED_BEFORE_S0`, the reported metric stays
`MASK_MAP50_95`, and `selection_metric_equals_primary_reporting_metric` must be
`false`. That last one is the point: the checkpoint is chosen on a composite
while the project reports the mask metric, and the parser refuses both ways of
hiding that - relabelling the composite as the headline metric, and swapping in a
mask-only selector once results are visible. A rationale entry claiming the
composite is *superior* is also refused, because nothing in this project compares
the two.

The segmentation and augmentation blocks record the installed framework's own
effective defaults, and `scripts/freeze_segmentation_baseline.py` **verifies
them against the installed configuration** at freeze time rather than trusting
the file. The rare-class support thresholds and the macro metric name are
derived from `construction_safety_vision.detection_comparison`, so the
segmentation phase cannot quietly acquire a friendlier rule.

## `segmentation_mask_iou_evaluation.yaml`

The direct instance-mask IoU diagnostic, frozen **before the first optimisation
step of S0**. It exists because the academic deliverable requires an explicit
IoU result and mask average precision does not supply one: AP is an averaged,
ranking-sensitive summary over IoU thresholds, so a reader cannot recover from it
how similar a predicted mask actually was to the object it covered.

Four decisions carry the diagnostic's meaning, and the parser enforces all four:

- **ground truth is the canonical COCO instance segmentation**, never the YOLO
  adapter - a `ground_truth_document` whose path contains `adapter` is refused,
  because scoring against a representation with its own measured approximation
  would fold that error into the model's result;
- **one predeclared operating point** - confidence 0.25, NMS IoU 0.70, imgsz 768,
  `max_det` 300, no test-time augmentation, no sweep, no second threshold;
- **one-to-one matching per image and per class, maximising total IoU**, solved
  with `scipy.optimize.linear_sum_assignment`; a greedy rule is a different rule
  and is refused by name;
- **no combined score** - the five diagnostics answer different questions, and
  blending them would let a coverage failure hide behind good mask quality on the
  instances that happened to be found.

Two policies decide what the numbers mean. An assigned pair sharing no pixel is
**not** a match: the assignment problem pairs those when the alternative is
leaving both unassigned, and counting them would inflate coverage with unrelated
objects. And an unmatched ground-truth instance contributes **zero** rather than
disappearing, which is the whole difference between `matched_mask_iou_mean` (mask
quality where the model found something) and `gt_normalized_mask_iou` (the same
IoU sum divided by every canonical instance, so misses lower it).

It is a diagnostic. It selects no checkpoint, tunes nothing, and does not replace
mask mAP@0.50:0.95 as the primary scientific result.

## `segmentation_canonical_evaluation.yaml` and `segmentation_comparison.yaml`

The phase 8E pair, frozen **after S0 ran and before S1 exists** - stated in the
files themselves, and the parser refuses a `protocol_timing` claiming otherwise.

They exist because of a problem phase 8D uncovered. `overlap_mask` decides both
the framework's training target and its validation ground truth, so S0 and an
`overlap_mask: false` S1 would have their native mask AP measured against
*different* ground truth. Differencing those two numbers would compare the
targets as much as the models. The canonical evaluator supplies one yardstick
neither flag can move: pycocotools `COCOeval` at `iouType=segm` against the
canonical phase 5D COCO masks.

Parsing is strict in ways that matter:

- **the standard COCO semantics are pinned** - IoU 0.50:0.05:0.95 and `maxDets`
  [1, 10, 100] - and re-checked against what `COCOeval` actually used, because
  changing the sweep after a candidate exists is the classic way to move a
  result;
- **conf must be 0.001**, which is not an operating point: AP needs the
  low-scoring tail, and the phase 8C direct-IoU diagnostic keeps its own
  operational 0.25;
- **ground truth may not be an adapter**, and the native framework metric may not
  be promoted to primary;
- **the candidate may override exactly one field**, `overlap_mask`, and a second
  override, a changed checkpoint policy, a relaxed margin or a recorded result
  for the unexecuted candidate each raise;
- **no composite score**, in either file. If the canonical AP and the direct IoU
  move in opposite directions the outcome is
  `CROSS_METRIC_DIRECTION_DISAGREEMENT` - recorded and ranked by the canonical
  metric, never resolved by a weighting invented once the numbers are visible.

The one-variable contract is additionally verified at run time against S0's own
frozen protocol rather than a restatement of it, and a variable that is declared
but never applied is refused just as firmly as an undeclared one.


## `detector_segmenter_comparison.yaml`

Frozen in phase 10A, **before** any comparison ran and **after** both models
were frozen, so nothing in it could have been chosen to flatter either. Parsed
by `construction_safety_vision.detector_segmenter_comparison`.

The question it encodes is not "which model is better". The detector emits a
class, a confidence and a box; the segmenter emits those plus an instance mask.
They do not produce the same output, so the protocol asks what the mask adds
and what it costs, and refuses to collapse that into one number.

What the parser refuses:

- **swapping the two confidences.** AP needs the low-scoring tail (0.001);
  operational analysis needs a working point (0.25). Each belongs to one
  protocol and a value from one may not be reported under the other.
- **deriving the segmenter's boxes from its masks.** The comparison uses each
  model's actual predicted boxes; re-deriving them would measure a
  post-processing choice this project invented.
- **a spatial feature with no declared box proxy.** Every mask quantity is
  paired with what a box-only pipeline could compute instead, or explicitly
  marked `NO_BOX_ONLY_EQUIVALENT`. That pairing is the answer to the question.
- **an accuracy framing for the association analysis.** There is no canonical
  compliance ground truth in this project, so there is nothing to be accurate
  against.
- **a benchmark that is not symmetric.** Interleaved execution in both
  directions, one precision for both models, a fixed warmup and iteration
  count, and CUDA synchronisation on both edges of every timed region.
- **hiding mask reconstruction.** It must sit inside the segmenter's end-to-end
  timing boundary, because that is the cost being measured.
- **an aggregate benefit score**, a declared winner, or any result at all.
