# Fragment-Rule Analysis

Generated: 2026-09-04T12:39:31+00:00 · Phase: 5B · Commit: `10ff78972f857c97a1c56fe10249d95578f447e6`

**Question.** Phase 5A found that the 76 annotations added since the version-4 snapshot each sit inside an annotation of their own class. Can that be turned into a deterministic rule, computed from the canonical current state alone, that identifies them without deleting legitimate annotations?

**Why a rule and not a list.** Excluding 76 memorised identifiers would not be reproducible, would not survive the next provider edit, and would encode a conclusion rather than a criterion. Version 4 is used here only to score candidates.

> **What the scores below mean, and what they do not.**
>
> The 76 additions are a **historical annotation-drift reference set**: the annotations that changed between two snapshots. They are **not** ground truth for bad annotations, and nobody has established them to be wrong.
>
> So precision here measures **agreement with historical drift**, not detection of annotation error. A rule at precision 1.00 selects annotations that all changed since version 4; it does not select annotations that are all incorrect. Reading it the second way would be invalid.

## Feature distributions (COMPUTED RESULT)

Across 2031 canonical annotations, of which 76 are version-4 additions and 1955 are not:

| Feature | Group | min | q25 | median | q75 | max |
| --- | --- | --- | --- | --- | --- | --- |
| same-class enclosure (bbox) | additions | 0.0 | 0.0 | 0.76178 | 1.0 | 1.0 |
| same-class enclosure (bbox) | others | 0.0 | 0.0 | 0.0 | 0.16886 | 1.0 |
| same-class enclosure (mask) | additions | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 |
| same-class enclosure (mask) | others | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 |
| area / enclosing annotation | additions | 4e-06 | 0.000267 | 0.041991 | 1.0 | 393.559524 |
| area / enclosing annotation | others | 0.002692 | 1.0 | 1.0 | 1.0 | 18204.357143 |
| area / image | additions | 3e-06 | 9.2e-05 | 0.000492 | 0.01124 | 0.109566 |
| area / image | others | 0.0 | 0.00427 | 0.01794 | 0.075778 | 0.857915 |
| absolute area (px) | additions | 4.0 | 116.0 | 651.0 | 18700.0 | 146444.0 |
| absolute area (px) | others | 0.0 | 6990.0 | 34721.0 | 139741.0 | 3294392.0 |

See `reports/figures/fragment_rule_distribution.png`.

## Candidate rules scored against the version-4 diff (COMPUTED RESULT)

| Rule | Selected | TP | FP | FN | Precision | Recall |
| --- | --- | --- | --- | --- | --- | --- |
| `enclosed_bbox >= 0.95 AND area_ratio < 0.01` | 34 | 33 | 1 | 43 | 0.9706 | 0.4342 |
| `enclosed_bbox >= 0.99 AND area_ratio < 0.01` | 34 | 33 | 1 | 43 | 0.9706 | 0.4342 |
| `enclosed_mask >= 0.8 AND area_ratio < 0.05` | 29 | 29 | 0 | 47 | 1.0000 | 0.3816 |
| `enclosed_mask >= 0.8 AND area_ratio < 0.1` | 29 | 29 | 0 | 47 | 1.0000 | 0.3816 |
| `enclosed_mask >= 0.9 AND area_ratio < 0.05` | 29 | 29 | 0 | 47 | 1.0000 | 0.3816 |
| `enclosed_mask >= 0.9 AND area_ratio < 0.1` | 29 | 29 | 0 | 47 | 1.0000 | 0.3816 |
| `enclosed_mask >= 0.95 AND area_ratio < 0.05` | 29 | 29 | 0 | 47 | 1.0000 | 0.3816 |
| `enclosed_mask >= 0.95 AND area_ratio < 0.1` | 29 | 29 | 0 | 47 | 1.0000 | 0.3816 |
| `enclosed_mask >= 0.99 AND area_ratio < 0.05` | 29 | 29 | 0 | 47 | 1.0000 | 0.3816 |
| `enclosed_mask >= 0.99 AND area_ratio < 0.1` | 29 | 29 | 0 | 47 | 1.0000 | 0.3816 |
| `enclosed_bbox >= 0.8 AND area_ratio < 0.01` | 35 | 33 | 2 | 43 | 0.9429 | 0.4342 |
| `enclosed_bbox >= 0.9 AND area_ratio < 0.01` | 35 | 33 | 2 | 43 | 0.9429 | 0.4342 |
| `enclosed_mask >= 0.8 AND area_ratio < 0.01` | 26 | 26 | 0 | 50 | 1.0000 | 0.3421 |
| `enclosed_mask >= 0.9 AND area_ratio < 0.01` | 26 | 26 | 0 | 50 | 1.0000 | 0.3421 |
| `enclosed_mask >= 0.95 AND area_ratio < 0.01` | 26 | 26 | 0 | 50 | 1.0000 | 0.3421 |
| `enclosed_mask >= 0.99 AND area_ratio < 0.01` | 26 | 26 | 0 | 50 | 1.0000 | 0.3421 |

Best candidate by precision plus recall: `enclosed_bbox >= 0.95 AND area_ratio < 0.01`, with precision 0.9706 and recall 0.4342.

Acceptance bar for applying a rule without review: precision >= 0.95 and recall >= 0.95.

**Result: the bar is NOT met.**

## Verdict: NO AUTOMATIC FRAGMENT FILTER ADOPTED

`fragment_rule_status = REJECTED_FOR_AUTOMATIC_FILTERING`. All 2031 canonical annotations are retained and no annotation is excluded on geometric grounds.

The experiment is kept here because a failed rule is useful negative evidence: it records what was tried, on what features, at what thresholds, and why it does not work. It should not be re-run in the hope of a better threshold - the reason it fails is not the threshold, as the next two sections show.

## Why recall is capped (COMPUTED RESULT)

An enclosure rule can only find an addition if the annotation that encloses it still exists. Splitting the additions by what happened to their version-4 parent:

| Is the addition still enclosed by a same-class annotation? | Additions |
| --- | --- |
| Yes, at >= 0.95 | 38 |
| No - nothing encloses it any more | 38 |
| **total** | **76** |

For the ones nothing encloses, what became of the version-4 annotation that did:

| Version-4 parent | Additions |
| --- | --- |
| removed | 18 |
| reshaped | 20 |

These 38 additions are unreachable by construction, not by a poor choice of threshold: no same-class annotation encloses them today. Either the coarse version-4 annotation that did was deleted, or it survived in a much smaller reshaped form that no longer covers them. In both cases the current state has replaced one loose annotation with several tighter ones.

### What replaced them (COMPUTED RESULT)

| Source image | v4 annotations | current | additions | orphaned | v4 removed |
| --- | --- | --- | --- | --- | --- |
| `CDbWzCea` | 3 | 6 | 4 | 4 | 1 |
| `Y1meTyz9` | 3 | 6 | 4 | 4 | 1 |
| `EMcfZIuO` | 3 | 5 | 3 | 3 | 1 |
| `MkxXQ8wm` | 3 | 6 | 3 | 3 | 0 |
| `RRlPrUJX` | 3 | 5 | 3 | 3 | 1 |
| `WvGmeXrU` | 3 | 5 | 3 | 3 | 1 |
| `lHkj3xNm` | 3 | 8 | 5 | 3 | 0 |
| `zIjMWx3E` | 3 | 6 | 3 | 3 | 0 |
| `5p4hDwmB` | 3 | 4 | 2 | 2 | 1 |
| `q2wyXKnA` | 3 | 6 | 3 | 2 | 0 |
| `ywH2LFvR` | 3 | 5 | 2 | 2 | 0 |
| `0ajF0d1e` | 3 | 4 | 1 | 1 | 0 |

## Reading (interpretation, labelled)

**FACT.** No candidate rule reaches the acceptance bar. The best one is precise (0.97) but recovers under half the additions (0.43).

**FACT, and a correction to phase 5A.** The phase 5A report described the 76 additions as fragments that 'add no coverage'. The containment measurement behind that statement is correct, but the inference was not. Inspecting the affected images shows two different things mixed together:

* genuinely degenerate slivers - a few dozen pixels drawn on a bracelet, on glove   lettering, on a hard hat - which the precise rule above does find;
* **re-annotations that improve the labels**: a single coarse `vest_on_body`   polygon replaced by several precise ones covering the parts of the vest that are   actually visible, and in at least one image a single oversized `person` box   covering two people replaced by one box per person.

The second kind *does* add coverage. It looked like it did not because the containment test asked whether an addition sits inside a version-4 annotation, and a coarse over-merged parent contains its own corrections by definition.

**MANUAL DECISION.** The project owner reviewed this analysis and rejected automatic filtering. Containment inside an older same-class annotation is not evidence that an annotation is wrong, and the evaluated rules cannot separate

* annotation fragments, from
* legitimate instance splits, from
* geometry refinements, from
* corrections of previously merged objects.

Automatic exclusion would therefore carry an unacceptable risk of deleting real ground truth. All 2031 annotations are retained.
The 29 annotations the strictest zero-disagreement candidate (`enclosed_mask >= 0.8 AND area_ratio < 0.05`) would have selected are **not** excluded either, and no second action row was created for them.

**What the flag means now.** Annotations the broader candidate rule selects carry `action = KEEP` and `review_flag = NESTED_SAME_CLASS_CANDIDATE` in `canonical_annotation_actions.csv`. The wording is deliberate: the flag records a geometric relationship, not a defect. Calling them fragments would assert something no one has established.

**PHASE 5C INPUT.** The modelling annotation population is the full canonical 2031. If a subset is ever to be removed, it needs a manual disposition per annotation, not a threshold.
