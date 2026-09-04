# Manual Visual Audit - phase 4B

Recorded: 2026-09-02 · Reviewers: project owner with a technical reviewer · Evidence commit: `ed9f59aed5b54335a344ec12e82b8bf1e9cd92d2`

Phase 4A measured the dataset and deliberately answered no semantic question.
This document records the answers a person gave after looking at the contact
sheets phase 4A produced. Machine-readable form:
[`manual_audit_decisions.csv`](manual_audit_decisions.csv); 37 decisions.

Every statement below is tagged:

* **AUTOMATED RESULT** - measured in phase 4A and carried forward unchanged;
* **HUMAN VISUAL JUDGEMENT** - what a person concluded from looking at a figure;
* **INTERPRETATION** - what the two together are taken to mean;
* **PHASE 5 REQUIREMENT** - a constraint handed forward, not an action taken here;
* **OPEN QUESTION** - deliberately unresolved.

> A human judgement is not a measurement. Nothing in this document has an error
> bar, a threshold or a p-value, and none of it should be quoted as if it did.

## 1. Scope and methodology

The reviewers were shown the committed phase 4A contact sheets and asked the
semantic questions the automated audit could not answer. The judgements were
made outside this repository and transcribed here; the transcription resolves
every subject to a stable identifier in the phase 4A manifests, and refuses any
identifier that does not resolve unambiguously.

This phase changed no dataset image, no annotation and no split. It created no
partition and froze no holdout. A recorded `phase5_action` is a note for a later
phase; nothing was acted on.

Regenerate and re-validate:

```bash
uv run python scripts/record_manual_audit.py
uv run python scripts/record_manual_audit.py --check
```

## 2. Phase 4A and phase 4B are different kinds of evidence

|  | Phase 4A | Phase 4B |
| --- | --- | --- |
| Produced by | executed code | people looking at figures |
| Reproducible by re-running | yes | no |
| Answers | how many, how different, how consistent | is this the same scene, is this in domain, which geometry is better |
| Recorded in | `dataset_audit.json`, `bbox_consistency_audit.json`, `eda_source.json` | `manual_audit_decisions.csv` |
| Can be wrong by | a bug | a misreading |

**INTERPRETATION.** The two are kept in separate files on purpose. A judgement
that migrates into a metrics file becomes indistinguishable from a measurement,
and the project's evidence is only worth what that distinction is worth.

## 3. Evidence reviewed

| Contact sheet | Question put to the reviewers | Decision recorded |
| --- | --- | --- |
| `review_h_near_duplicates.jpg` | are these cross-split pairs the same content? | yes, 6 pairs |
| `review_f_zero_instance.jpg` | negative sample or missing label? | yes, 17 images |
| `review_k_bbox_vs_segmentation.jpg` | which geometry describes the object better? | yes, set level |
| `review_b_class_vest_loose.jpg` | is the class semantics coherent? | yes, set level |
| `review_a_overview.jpg` | what kind of imagery is this? | yes, set level |

The package also contains `review_b` sheets for the other four classes,
`review_d_crowded.jpg`, `review_e_lower_luminance.jpg` and
`review_g_near_duplicates.jpg` (same-split pairs). The reviewers had them
available but supplied no explicit judgement for them, so **none is recorded**.
In particular, the same-split near-duplicate candidates remain unconfirmed.

## 4. Semantic duplicates across split boundaries

**AUTOMATED RESULT.** Phase 4A hashed all 436 source images
with SHA-256 and found no byte-identical pair. A separate perceptual screen
(dHash and pHash, both 64-bit) emitted 11 candidate
pairs, of which 6 cross a provider
split boundary. A fingerprint match is a candidate, not a verdict.

**HUMAN VISUAL JUDGEMENT.** All 6 cross-split pairs depict effectively
the same source content rather than similar independent scenes. Confidence: HIGH.

**Byte duplicate: NO. Semantic duplicate: YES.** The pairs differ in stored
bytes - JPEG encoding, metadata, small crop or position differences, colour and
compression - while showing the same picture. Nothing here contradicts the
SHA-256 result; the two questions are different questions.

| Group | Image A | Split A | Image B | Split B | Fingerprints |
| --- | --- | --- | --- | --- | --- |
| `manual_dup_001` | `4UEcuYsspsOsSV0XYgSR` | valid | `6hpR44qukoBLy0iFwzip` | train | dhash=1; phash=0 |
| `manual_dup_002` | `FBzxttXYsNSZKcVNofUI` | valid | `Um5Oft7wh92nWZzcd33C` | train | dhash=0; phash=0 |
| `manual_dup_003` | `L51HLb9pTcXWg0hlY2eI` | train | `WRFgtFooB6kjdViPXWHk` | valid | dhash=1; phash=8 |
| `manual_dup_004` | `OgbeepjjlVIRghzAMurI` | valid | `Ud5BlYIB2FSpmLDEEq9z` | train | dhash=0; phash=0 |
| `manual_dup_005` | `THaelQmgDKdUwfSFIAx5` | train | `ymICdUCkVq8IfApx2TvK` | valid | dhash=2; phash=0 |
| `manual_dup_006` | `kvlkvPKhlcIlNoa2lJKq` | valid | `tjHCOqwRIYw66vZcCHJp` | test | dhash=5; phash=5 |

Group ids are a function of the sorted member ids alone, so re-running the
recorder cannot renumber them.

**PHASE 5 REQUIREMENT.** Every group is `GROUP_TOGETHER`: members of one group
must land in the same split. No group is assigned to train, validation or test
here.

## 5. Zero-instance images

**AUTOMATED RESULT.** 17 source images carry no annotated instance
(test 4, train 11, valid 2). All of them are on the sheet; none was excluded from review.

**HUMAN VISUAL JUDGEMENT.** There is **no evidence of widespread missing
annotations**. Most of the images read as genuine zero-target images, or as
domain/context images with no clearly applicable target instance. The review did
surface domain-curation noise: three images are clearly outside the project's
target domain.

| Transcribed | Resolved image id | Split | Scene | Edit distance | Runner-up |
| --- | --- | --- | --- | --- | --- |
| `9b7vLDWD` | `9b7VLDWDFhy21Ygmz7ss` | train | cartoon/illustration of a worker | 0 | 7 |
| `KgOOOMBE` | `KgOOMlBEXp0QxhSxTT6a` | train | office/laboratory-like interior | 2 | 7 |
| `KqRSR0F9` | `KqRSROf90W22N1qzQzN0` | valid | yellow taxi/street scene | 0 | 7 |

The reviewers stated their short ids were approximate. Each was resolved against the
17 images actually on the zero-instance sheet, and accepted only because one
candidate fitted within two edits while the next-best fitted no closer than the
required margin. The resolution is recorded so it can be checked rather than trusted.

**Decision for the three: `OUT_OF_DOMAIN`, confidence HIGH,
`phase5_action=EXCLUDE_CANDIDATE`.** They are candidates for deterministic exclusion
from the canonical modelling population. Nothing was removed: `data/raw/` is untouched
and the source population is still 436 images.

**Decision for the remaining 14: `NO_OBVIOUS_MISSING_TARGET_LABEL`,
`phase5_action=RETAIN_CANDIDATE`.** Their domain relevance ranges from strong
construction context to ambiguous industrial/safety context. The reviewers gave no
finer classification, so none is invented, and no per-image confidence was stated.

## 6. The cartoon image is out of domain, not a valid negative

**HUMAN VISUAL JUDGEMENT.** The cartoon/illustration image does contain a
human-like worker and PPE. It is recorded as `OUT_OF_DOMAIN` and **not** as
`VALID_NEGATIVE`.

**INTERPRETATION.** The distinction matters and is easy to lose. Calling it a
valid negative would assert that a real worker was present and correctly left
unannotated, which would be false; the image has no real worker at all. Its zero
annotations are therefore not evidence about annotation quality. It is excluded,
if phase 5 excludes it, because the project targets real construction and PPE
imagery - not because anything was mislabelled.

## 7. Supplied bounding box versus segmentation geometry

**AUTOMATED RESULT.** Of 3,373 v4 export annotations,
1,570 are polygons and
1,803 are compressed RLE, and
2,589 agree with their own segmentation within 1.0 px.
Polygon agreement is essentially perfect: all
1,570 of them fall within 0.5 px. The
disagreement is concentrated in the RLE instances, where
1,777 of
1,803 segmentation-derived boxes sit **inside** the
supplied box - the supplied box is systematically the larger of the two.

**HUMAN VISUAL JUDGEMENT.**

* On the readable high-discrepancy RLE panels, the segmentation-derived box
  appeared more tightly aligned with the visible object than the supplied COCO
  box: `SEGMENTATION_GEOMETRY_PREFERRED`.
* On the low-discrepancy and representative panels, both representations
  appeared operationally acceptable: `BOTH_ACCEPTABLE`.
* Some panels were not readable or did not render legibly: `UNCERTAIN`.

**The record stays at set level.** The reviewers did not identify which panels were
unreadable, so no per-annotation verdict is recorded for any of the
13 panels. The groups below are how the sheet was built, not a
certification that every panel in a group was legible.

| Panel group | Panels | Selected as |
| --- | --- | --- |
| high discrepancy | 9 | three largest per-split deltas |
| representative | 4 | one polygon and one RLE example per sheet split |

## 8. Phase 5 bounding-box policy status

**Status: `PREFERRED_AND_VISUALLY_SUPPORTED`. Not `ALREADY_APPLIED`.**

The candidate policy - canonical segmentation geometry, detection boxes derived
from it - is now supported by both a quantitative and a visual argument:

1. instance segmentation is the project's canonical annotation concept;
2. polygon cases show essentially perfect geometric agreement;
3. on readable high-discrepancy RLE cases the derived box appeared tighter and
   more representative;
4. the supplied COCO box appears systematically inflated in many RLE cases.

**No annotation was modified in phase 4B, and no conversion was executed.**
Implementing and validating the conversion is phase 5's responsibility, and the
policy is not frozen until phase 5 does it.

## 9. `vest_loose`

**HUMAN VISUAL JUDGEMENT - class semantics: `VALID_OVERALL`.** The examples
support the reading that `vest_loose` is a safety vest not being worn on a
person: hanging, stored, or otherwise loose in the scene.

**HUMAN VISUAL JUDGEMENT - data diversity: `VERY_LOW`.**

**AUTOMATED RESULT.** 8 of 436 source images
contain the class, carrying 45 instances:
7 train images
(32 instances),
1 validation image
(13 instances), and
0 images in the provider
test split. Several instances are concentrated inside individual multi-object images.

**HUMAN VISUAL JUDGEMENT.** Some examples - particularly scenes with light or
white hanging garments and PPE - are semantically more ambiguous than the clear
high-visibility-vest examples. They are **not** removed on that basis; the
observation is recorded, not acted on.

Recorded flags: `RARE_CLASS`, `LOW_SOURCE_IMAGE_DIVERSITY`,
`CONTEXT_SHORTCUT_RISK`, `SOME_SEMANTIC_AMBIGUITY`.

**HYPOTHESIS, not a finding.** A model trained on this may associate
`vest_loose` with racks, walls and stored-PPE context rather than learning a
robust object concept. No model exists, so this is an analysis risk to test in
phase 12, not an observed behaviour.

**PHASE 5 REQUIREMENT.** The split design must explicitly consider `vest_loose`
source-image coverage in validation and test wherever the grouping constraints
make it feasible. No numerical allocation is specified here.

## 10. What kind of dataset this is

**HUMAN VISUAL JUDGEMENT: `ACCEPTABLE_WITH_DOMAIN_HETEROGENEITY`.** The overview
sheet shows a mixture of real construction-site imagery, occupational-safety
imagery, posed worker portraits, stock photography, product-like PPE images,
isolated PPE objects, contextual construction scenes and background scenes.

**INTERPRETATION.** The dataset is usable, and the heterogeneity is not a defect
to be hidden - but it constrains the language the project may use. The correct
description is *mixed construction and PPE imagery*, or *a heterogeneous
construction/PPE dataset containing site, stock, portrait and product-style
imagery*. It is **not** a field-captured construction-site dataset.

**This limits external validity.** The final report must not imply that
performance measured on this dataset demonstrates robust deployment performance
on arbitrary real-world construction-site video. That claim would need a
different evaluation set.

## 11. Provider split: `UNSUITABLE_FOR_FINAL_PROTOCOL`

Phase 4A classified the provider split `UNDETERMINED_PENDING_VISUAL_REVIEW`,
because the automated evidence could not settle whether the cross-split
candidates were genuine. The visual review settles it.

**Classification: `UNSUITABLE_FOR_FINAL_PROTOCOL`** - a human-audited
methodological decision, on three grounds:

1. **6 visually confirmed semantic duplicate pairs cross split
   boundaries.** Training on one member and evaluating on the other measures
   memorisation, not generalisation.
2. **The provider test split contains zero `vest_loose` instances.** A class
   that is absent cannot be scored, so no per-class test metric exists for it
   under this partition.
3. **The rare class is too thinly represented for a defensible per-class final
   evaluation**: 8 source images in total, one of them in validation.

> **The SPLIT is rejected. The DATASET is not.** The dataset remains suitable
> for the project, subject to phase 5 canonicalisation and split redesign. This
> distinction is the whole point of the section.

## 12. Current source project versus frozen v4: `UNRESOLVED_CANONICALIZATION`

**AUTOMATED RESULT.** The live source project holds 76 more annotations
than the v4 source-equivalent snapshot (train +61,
valid +2, test +13),
so the source project was edited after version 4 was generated.

**AUTOMATED RESULT.** The two populations also differ in what geometry they expose.
The current source API provides polygon vertices for
1,007 annotations, while
1,022 mask-type records provide a bounding box
but no complete mask geometry, and 2 records
carry no recognised representation. The v4 COCO export has complete segmentation
geometry for both polygon and RLE, but describes the older annotation snapshot.

**OPEN QUESTION.** Which is canonical. The alternatives:

|  | Option | Cost |
| --- | --- | --- |
| A | use v4 as the canonical frozen snapshot | accepts known-stale annotations |
| B | use the current source state | incomplete mask geometry through the inspected API |
| C | obtain the current source state with complete geometry, reproducibly | unknown; not yet investigated |

**PHASE 5 REQUIREMENT (HIGH).** Investigate C before accepting A. This is an
entry condition for phase 5, not a phase 4B deliverable, and phase 4B does not
resolve it.

## 13. Two annotations of unknown representation

**AUTOMATED RESULT.** 2 source annotations carry no recognised
representation. Both sit on the same image (`OQJwjQoYsf1KUgr9G0V8`, provider split
`test`), both measure under 16 px on a side, and both are positioned at the
image edge.

**Status: `MANUAL_REVIEW_STILL_REQUIRED` and `PHASE5_CANONICALIZATION_ITEM`.**
They were not on any contact sheet, so no visual judgement exists for them. They
are **not** deleted. Phase 5 must give them an explicit disposition: they must
not silently enter the model-ready dataset, and they must not silently vanish
from it.

## 14. Phase 5 entry constraints

Recorded, not implemented. No split ratio is specified, and nothing below has
been acted on.

| Id | Constraint |
| --- | --- |
| **P5-01** | Provider split must not be reused as final protocol. |
| **P5-02** | Visually confirmed semantic duplicates must be grouped and cannot cross split boundaries. |
| **P5-03** | Split design must be group-aware. |
| **P5-04** | Split design must explicitly consider rare-class coverage, especially vest_loose. |
| **P5-05** | The three clearly out-of-domain zero-instance images are deterministic exclusion candidates, pending canonical manifest implementation. |
| **P5-06** | Other zero-instance images are not automatically excluded. |
| **P5-07** | Segmentation-derived bounding boxes are the preferred detection-label policy, supported by quantitative and visual evidence, but conversion has not yet occurred. |
| **P5-08** | Polygon and compressed-RLE segmentation must both be supported. |
| **P5-09** | The current-source-vs-v4 annotation drift must be resolved before freezing the canonical modeling dataset. |
| **P5-10** | The two unknown microannotations require explicit disposition. |
| **P5-11** | Final test must contain defensible class coverage subject to grouping constraints. |
| **P5-12** | The final holdout must remain protected after freeze. |

## 14b. Addendum - phase 5B.1: same-split semantic duplicates

**This section records a later review. It adds to the phase 4B decisions above
and rewrites none of them.**

Phase 4B reviewed only the near-duplicate candidates that crossed a *provider*
split boundary, because that was where leakage could occur under the provider's
own split. Phase 4B then rejected that split. Once the split is rebuilt from
scratch, a duplicate pair that happened to sit inside one of the provider's
splits constrains the new split exactly as much as one that crossed a boundary -
nothing stops an optimiser putting the two halves of such a pair on opposite
sides. The reviewers therefore looked at the same-split candidates as well.

| Group | Image A | Image B | Provider splits | Scene | Decision |
| --- | --- | --- | --- | --- | --- |
| `manual_dup_007` | `3LUOWEww` | `LVQG21gd` | train / train | orange forklift in front of stored material racks | `EXACT_SEMANTIC_DUPLICATE` (HIGH) |
| `manual_dup_008` | `66p9gzaQ` | `pbOZlges` | train / train | same worker shelving stock, same scene and framing, different moment | `NEAR_DUPLICATE_SAME_SCENE` (MEDIUM) |
| `manual_dup_009` | `IyVJr4Dc` | `ghzZPWPu` | train / train | single yellow hard hat resting on dark weathered beams | `EXACT_SEMANTIC_DUPLICATE` (HIGH) |
| `manual_dup_010` | `TwF1Rahc` | `pd0Syehn` | train / train | hanging white PPE garments with white helmets | `EXACT_SEMANTIC_DUPLICATE` (HIGH) |
| `manual_dup_011` | `p5lxtMTK` | `xwPbKAjD` | train / train | grid shelf holding many hard hats | `EXACT_SEMANTIC_DUPLICATE` (HIGH) |

**FACT.** These are semantic duplicates, not byte duplicates. Every source image
has a distinct SHA-256; the phase 4A result that there are no exact duplicates
stands unchanged.

**FACT.** Group numbering continues from the phase 4B sequence rather than
restarting, so the identifiers already published for groups 001-006 keep pointing
at the same images.

**Two findings, both indivisible.** `EXACT_SEMANTIC_DUPLICATE` means the same
frame stored twice. `NEAR_DUPLICATE_SAME_SCENE` means the same subject and scene
at a different moment - here a changed arm and body position - which is
correlated but not identical. Both make a pair indivisible for splitting, because
grouping serves statistical independence across splits rather than image
identity, and they are recorded distinctly so the evidence behind each group
stays visible.

**Reconciliation of the candidate count.** Phase 4A raised 11 near-duplicate
candidates: 6 crossing a provider split and 5 inside one. `review_h` showed all 6
cross-split pairs, which phase 4B decided. `review_g` showed the 8 candidates
with the smallest perceptual distance, which happened to be 4 of those 6 plus 4
of the same-split pairs - so it displayed 8 pairs of which only 4 were new. The
11th candidate had the largest distance of all and fell outside that cap, so it
appeared on neither sheet; it was drawn on its own sheet and decided here. **All
11 candidates now carry an explicit human disposition.**



## 15. Limitations of this review

* The judgements are human readings of downsampled JPEG contact sheets, not measurements. They carry no error bar and are not reproducible in the sense a computed statistic is.
* Two reviewers looked at the same sheets together, so there is no independent second opinion and no inter-rater agreement to report.
* Only the sheets listed under 'evidence reviewed' produced an explicit decision. The remaining sheets in the phase 4A package were available but no judgement was supplied for them, so none is recorded.
* Near-duplicate detection was a perceptual-hash screen with a recall-oriented threshold. It bounds what the reviewers could confirm: a duplicate pair that both fingerprints missed was never shown to anyone.
* Some panels on the geometry sheet were not readable, and the reviewers did not record which, so the geometry judgements are set-level rather than per-annotation.
* The domain judgement is a qualitative characterisation of the image population; no domain taxonomy was defined and no image was assigned to a domain category beyond the three marked out-of-domain.

**INTERPRETATION.** These limitations do not undermine the decisions recorded
here, but they do fix what the decisions can carry. The duplicate groups are
strong enough to constrain a split; the geometry judgement is strong enough to
support a policy phase 5 must still validate; the domain judgement is strong
enough to constrain the report's language and nothing more.

## 16. Phase 4B outcome

| | Status |
| --- | --- |
| Phase 4B | `COMPLETE` |
| Dataset | `ACCEPTED_WITH_DOCUMENTED_LIMITATIONS` |
| Provider split | `UNSUITABLE_FOR_FINAL_PROTOCOL` |
| Canonical split | not created |
| Holdout | not frozen; still protected |
| Models | none trained; no metric exists |
| Next | phase 5 - canonical dataset, group-aware split and freeze |
