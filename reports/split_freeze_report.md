# Split Freeze - Phase 5C.2

Phase: 5C.2 · Commit: `93e364b647f0f755d16d94a76cdd487c66c4f79f` · Status: **FROZEN**

**The split is frozen.** `candidate_001` is now the authoritative train/validation/test membership of this project. The `test` split is a locked holdout from this point: it has not been evaluated, inspected, or used to make any decision, and it may be read only in the final-evaluation phase after both models are frozen.

Claims below are labelled `FACT` (measured from a committed artifact), `COMPUTED RESULT` (produced by this run), `HUMAN PROTOCOL DECISION` (a choice a person made) and `LIMITATION` (something this split cannot support).

## 1. Source and modelling population

`FACT` The dataset is Roboflow Universe `agis-workspace-8gs52/construction-ppe-compliance-detection` v4, CC BY 4.0, with the canonical annotation state recovered from the live source project in phase 5A (`CURRENT_COMPLETE_GEOMETRY`).

`FACT` The provenance population is **436 source images**. The modelling population is **433 images** - 436 minus the 3 out-of-domain exclusions confirmed by the phase 4B human audit. The two populations are different and must not be conflated. Excluded images stay on disk and stay in the provenance population; exclusion is logical, never physical.

`FACT` The modelling population carries **2031 annotations**, all retained: 0 excluded as fragments, 2 materialised as four-corner rectangles from the provider's box and labelled `SYNTHETIC_FROM_PROVIDER_BBOX`.

Modelling population SHA-256: `afe8b73752f2dffda51e5cf9b67c2169966e94a73c0806c4c96405420e2c86d7`

## 2. Group construction

`FACT` The unit a split assigns is the **group**, never the image. Phase 4B confirmed 6 semantic-duplicate relations and phase 5B.1 confirmed 5 more, giving **11 confirmed groups over 22 images** plus **411 singletons** = **422 indivisible split units**.

`FACT` Two distinct findings both make a group indivisible, and they are recorded separately in `group_manifest.csv`: `EXACT_SEMANTIC_DUPLICATE` (10 groups - the same frame stored twice) and `NEAR_DUPLICATE_SAME_SCENE` (1 group, `manual_dup_008` - the same worker and scene at a different moment). The second is not an exact duplicate and is not separable either.

`FACT` Groups are connected components of the confirmed relations, so a chain A~B, B~C forms one group rather than two overlapping pairs. All 11 phase 4A near-duplicate candidates carry a human disposition; none was merged on perceptual distance alone.

Groups SHA-256: `d6065c4e2adc56b5bcc6e027458807ef2f3281ed530915d690d55921a289faaf`

## 3. Why the provider's split was not reused

`HUMAN PROTOCOL DECISION` The provider's split was classified `UNSUITABLE_FOR_FINAL_PROTOCOL` in phase 4B. It is not an input, an initialisation or a target anywhere in the search or the freeze. `group_split_features.csv` omits it deliberately and `optimize_split_candidates.py` refuses to run if such a column is present.

## 4. Optimisation protocol (phase 5C.1)

`FACT` Deterministic multi-start local search over the 422 groups, seeded from the project seed **42**. Hard constraints are rejections, not penalties. The soft objective is normalised per class and averaged, so a 914-instance class cannot outweigh a 45-instance one, and each component is reported separately. The protocol is `configs/split_search.yaml` and is unchanged by this phase.

## 5. Candidates generated

| Candidate | Images | `vest_loose` images | `vest_loose` instances | Total objective |
| --- | --- | --- | --- | --- |
| `candidate_001` **(selected)** | 303/65/65 | 5/1/2 | 30/8/7 | 0.132203 |
| `candidate_002` | 303/65/65 | 5/1/2 | 33/8/4 | 0.16101 |
| `candidate_003` | 303/65/65 | 5/1/2 | 33/3/9 | 0.177606 |
| `candidate_004` | 303/65/65 | 5/1/2 | 33/3/9 | 0.182085 |
| `candidate_005` | 303/65/65 | 5/1/2 | 30/4/11 | 0.186418 |
| `candidate_006` | 303/65/65 | 4/2/2 | 27/7/11 | 0.213679 |

`HUMAN PROTOCOL DECISION` The candidates other than `candidate_001` are preserved unchanged and marked `NON_SELECTED_PROVISIONAL_CANDIDATE` in `reports/split_candidates/selection.csv`. Their scores are historical measurements and were not rewritten.

## 6. Selection

`HUMAN PROTOCOL DECISION` `final_selected_candidate` = **`candidate_001`**

* `selection_status`: `FINAL_SELECTED`
* `selection_method`: `HUMAN_REVIEW_OF_PREDECLARED_DETERMINISTIC_CANDIDATES`
* `decision_source`: `PROJECT_OWNER_REVIEW`
* `algorithmic_best_candidate`: `candidate_001`

The human selection coincides with the algorithmic best candidate. That does **not** make it an automatic selection: the candidates were predeclared and deterministically generated, and the choice among them was made after reviewing the rare-class and evaluation trade-offs described below. A coincidence of outcome is not a substitute for the review step, and the review step is recorded here because it happened.

Rationale, as reviewed:

* preserves 5 of the 8 `vest_loose` source images in training, keeping training-time diversity of the rare class as high as any feasible candidate does;
* reserves 2 independent `vest_loose` source images for the final holdout, the minimum at which a per-class claim about the holdout is meaningful at all;
* satisfies every hard class, group and negative constraint;
* matches the 303/65/65 target exactly (`size_error` = 0.0);
* has the best predeclared deterministic objective among feasible candidates (total 0.132203);
* materially outperforms the 4/2/2 family on both global image-level and instance-level balance - `candidate_006` scores `E_img` 0.117674 and `E_inst` 0.057455, against 0.070263 and 0.023392;
* avoids reducing `vest_loose` training-image diversity from 5 images to 4 solely to obtain a second validation image.

## 7. Selected candidate metrics (phase 5C.1 objective)

| Component | Value |
| --- | --- |
| `E_img` image class balance | 0.070263 |
| `E_inst` instance class balance | 0.023392 |
| `E_neg` negative balance | 0.038549 |
| `E_size` size | 0.0 |
| **total** | **0.132203** |

`COMPUTED RESULT` These are the values measured in phase 5C.1 and carried over unchanged; this phase re-verified the assignment they describe rather than re-scoring it.

## 8. Final image and group counts

| Split | Images | Share | Groups | Non-singleton groups | Negatives | Annotations |
| --- | --- | --- | --- | --- | --- | --- |
| train | 303 | 70.0% | 294 | 9 | 10 | 1422 |
| validation | 65 | 15.0% | 63 | 2 | 2 | 304 |
| test | 65 | 15.0% | 65 | 0 | 2 | 305 |
| **total** | **433** | 100.0% | **422** | **11** | **14** | **2031** |

`COMPUTED RESULT` 433 images and 2031 annotations, matching the phase 5B modelling population exactly. Every eligible image is assigned once; no excluded out-of-domain image appears.

## 9. Per-class image distribution

| Class | train | validation | test | total |
| --- | --- | --- | --- | --- |
| `helmet_loose` | 53 | 11 | 11 | 75 |
| `helmet_on_head` | 123 | 26 | 27 | 176 |
| `person` | 256 | 55 | 55 | 366 |
| `vest_loose` | 5 | 1 | 2 | 8 |
| `vest_on_body` | 146 | 31 | 31 | 208 |

## 10. Per-class instance distribution

| Class | train | validation | test | total |
| --- | --- | --- | --- | --- |
| `helmet_loose` | 276 | 57 | 60 | 393 |
| `helmet_on_head` | 220 | 47 | 47 | 314 |
| `person` | 641 | 137 | 136 | 914 |
| `vest_loose` | 30 | 8 | 7 | 45 |
| `vest_on_body` | 255 | 55 | 55 | 365 |

`COMPUTED RESULT` All five classes appear in all three splits, at image level and at instance level.

## 11. The `vest_loose` limitation

`FACT` `vest_loose` occurs in only **8 modelling source images**, spanning **7 indivisible groups** - two of the eight images sit in `manual_dup_010`, so the class moves in chunks and cannot be freely rebalanced.

`COMPUTED RESULT` The frozen split allocates `vest_loose` 5/1/2 images and 30/8/7 instances.

`LIMITATION` **Validation holds exactly one `vest_loose` source image.** Validation metrics specific to `vest_loose` therefore carry high sampling uncertainty and **must not be used in isolation** for model or hyperparameter selection. Model selection relies primarily on predeclared global and macro validation criteria; a `vest_loose`-specific validation result is interpreted cautiously and supported by qualitative analysis.

`LIMITATION` **The holdout holds two independent `vest_loose` source images.** Even the final per-class metrics for `vest_loose` must be reported with an explicit small-sample limitation attached.

`LIMITATION` This split is **not perfectly stratified** and must not be described as such. It is a **group-aware and class-aware constrained split**: group indivisibility and the rare-class floors are hard constraints, and proportionality is a scored preference that the group structure sometimes makes unreachable.

## 12. Negative images

`COMPUTED RESULT` The 14 deliberate negatives - images the phase 4B audit confirmed as annotation-free rather than unlabelled - are allocated 10/2/2, so none of the three splits is free of them.

## 13. Group-integrity checks

`COMPUTED RESULT` Every check below was executed by this run and passed:

* `candidate_001` reproduces its recorded assignment digest;
* the phase 5C.1 summary agrees with the candidate file;
* all 422 canonical groups are assigned exactly once;
* all 433 images are assigned once;
* every group's membership matches `group_manifest.csv` exactly;
* no group appears in more than one split, so no confirmed duplicate or same-scene
  pair crosses a boundary;
* all five classes appear in every split at image and instance level;
* `vest_loose` is allocated 5/1/2 images and
  30/8/7 instances;
* negatives are allocated 10/2/2;
* no excluded out-of-domain image appears in any split;
* all 2031 annotations belong to an eligible image in exactly one split.

## 14. Fingerprints

| Fingerprint | Value |
| --- | --- |
| `modeling_population_sha256` | `afe8b73752f2dffda51e5cf9b67c2169966e94a73c0806c4c96405420e2c86d7` |
| `groups_sha256` | `d6065c4e2adc56b5bcc6e027458807ef2f3281ed530915d690d55921a289faaf` |
| `candidate_assignment_sha256` | `081c39e729697f081302fc809eb4b78fff6450e910df254b88f311d623aa1fbe` |
| `split_assignment_sha256` | `a230869ff4cb45f53654d67357a27f2a0def6d8ba79f9fa4880f0fdda2f046cc` |
| `holdout_sha256` | `bb7ed43b20a84644d5a3917c6d0ead688132f82a30052b06ae7ad121e4851a00` |

`FACT` `candidate_assignment_sha256` and `split_assignment_sha256` differ, and are expected to. The first is the phase 5C.1 digest over sorted `(group_id, split)` pairs - the identity of the reviewed candidate, preserved unchanged. The second is the phase 5C.2 digest over sorted `(group_id, source_image_id, split)` triples, because the canonical final representation names images as well as groups. The consequence is that the freeze fingerprint also moves if a group's membership changes while its split assignment does not - which is exactly the drift the candidate digest cannot see.

`FACT` `holdout_sha256` covers the holdout's group ids, its source image ids and `modeling_population_sha256`. It carries no label, no metric and no model output: it exists to detect membership drift in the protected split, not to record results.

## 15. Holdout access policy

`HUMAN PROTOCOL DECISION` `test` is a locked holdout. Reading it requires **two independent opt-ins**, and neither alone is sufficient:

1. an in-code `allow_test=True` passed to `assert_split_allowed`; **and**
2. `CSVISION_ALLOW_TEST_SPLIT=1` in the environment.

`FACT` The default state is `LOCKED`. The environment variable is **not** set in this repository, and `construction_safety_vision.data.split_freeze.FrozenSplits` routes every `test` request through the same guard, so the frozen holdout ids cannot be obtained through the data access layer without both opt-ins. `train` and `validation` need no override.

`HUMAN PROTOCOL DECISION` From this freeze onward, `test` must not be used for:

model selection, architecture selection, hyperparameter tuning, augmentation tuning, image-size tuning, threshold tuning, qualitative model debugging, or error-driven iteration. It may be evaluated once, in the final-evaluation phase, after both models are frozen.

`FACT` As of this report the holdout has never been evaluated, inspected or plotted, and no model exists to evaluate on it.

## 16. Limitations

* `LIMITATION` **Independence is screened, not proven.** The groups rest on two perceptual fingerprints plus human visual review of every candidate they raised. That screens for duplicated frames and one same-scene pair; it does not prove that no two images in different splits share a site, a day, a camera or a worker. No such metadata exists in this dataset, so no stronger claim is made.
* `LIMITATION` **`vest_loose` is thin whatever the split.** 8 images and 45 instances cannot support a confident per-class conclusion. This is a property of the dataset, not of the split.
* `LIMITATION` **Balanced counts are not balanced difficulty.** The objective counts images and instances; nothing here measures how hard those images are, so comparable counts do not guarantee comparable difficulty across splits.
* `LIMITATION` **The search was reproducible, not exhaustive.** A better assignment may exist; the space was not brute-forced.
* `LIMITATION` **Instance-level proportionality is partly unreachable.** Objects co-occur inside images and images are bound into groups, so some deviation is structural rather than a search failure.
* `LIMITATION` **No model exists.** No metric, no evaluation and no inference has been produced, on any split.

## 17. What this phase did not do

No model was trained. No YOLO or COCO dataset was generated. No image was copied, moved, resized or preprocessed, and no label file was written - `data/processed/` is untouched. No inference was run. The holdout was not evaluated or inspected. CSVISION_ALLOW_TEST_SPLIT was not set. Phase 5D materialises the detection and segmentation views from this membership.
