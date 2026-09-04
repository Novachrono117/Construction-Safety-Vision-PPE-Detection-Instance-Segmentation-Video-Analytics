# Provisional Split Candidates

Generated: 2026-09-04T15:37:06+00:00 · Phase: 5C.1 · Commit: `c6a1eb44d80edd73cffd24480de4e5cdf6fb21f8`

**No split is frozen and no candidate is selected.** Every assignment below is provisional. `final_selected_candidate` is `UNSELECTED_PENDING_REVIEW`; the holdout remains locked and no candidate test set has been evaluated.

## 1. Objective and hard constraints

The unit assigned is the **group**, never the image. Phase 5B.1 established 422 indivisible groups over 433 modelling images, so a confirmed duplicate pair cannot be separated by construction rather than by penalty.

**Hard constraints.** An assignment breaking any of these is rejected outright, not ranked low:

1. every group assigned to exactly one split;
2. no group split across boundaries (structural: the group is the unit);
3. all 433 modelling images assigned;
4. all five classes present in all three splits, at image **and** instance level;
5. `vest_loose` images at least 4 / 1 / 2 (train / validation / test);
6. negative images at least 1 / 1 / 1;
7. split sizes within 0 image(s) of 303 / 65 / 65.

**Soft objective.** With `r(s)` the target ratio of split `s`:

```
E_img  = mean over (class, split) of
         |images_with(c,s) - r(s) * total_images_with(c)| / max(target, 1)
E_inst = mean over (class, split) of
         |instances(c,s)   - r(s) * total_instances(c)|   / max(target, 1)
E_neg  = mean over split of
         |negatives(s)     - r(s) * total_negatives|      / max(target, 1)
E_size = mean over split of
         |images(s)        - target_images(s)|            / max(target, 1)

total  = w_img*E_img + w_inst*E_inst + w_neg*E_neg + w_size*E_size
```

Each class contributes one normalised term per split and the terms are **averaged**, not summed. That is what stops `person` (914 instances) drowning out `vest_loose` (45): being ten images short of 640 is a small normalised error, being ten short of 12 is not.

Weights: `image_class_balance` 1.0, `instance_class_balance` 1.0, `negative_balance` 1.0, `size` 1.0. All equal, and deliberately so - each component is already normalised, so no rescaling is needed and equal weights cannot be mistaken for tuning toward a preferred answer.

## 2. Algorithm

Deterministic multi-start local search, no new dependency:

1. **Rarity-aware initialisation.** Scarce requirements are placed first, rarest class first and smallest split first. Filling by size and hoping the rare class lands well fails reliably - `vest_loose` occupies 7 groups out of 422, so a size-first pass puts all 7 in train and every restart then has to dig out.
2. **Size repair.** Groups move from the most over-target split to the most under-target one until the size tolerance is met.
3. **Local search.** Equal-size swaps between splits (which cannot break the size constraint) and single-group moves. Only strictly improving steps are taken, and a step breaking a hard constraint is discarded rather than penalised.
4. **64 restarts**, each seeded `seed + index` from the project seed 42, 4000 local-search steps each.

## 3. Reproducibility

The search is a pure function of the group features and `configs/split_search.yaml`. No unseeded randomness, no filesystem ordering, no provider split, no image pixels. Candidates are deduplicated by assignment fingerprint and ranked by total objective with the fingerprint as tie-break, so the ranking cannot depend on discovery order.

Split-search configuration SHA-256: `467d9fd74553221f969fd793bef90db14e9f0fabbd6262a13321400f429e6a6c`

## 4. Target split sizes

| Split | Target images | Target ratio |
| --- | --- | --- |
| train | 303 | 70% |
| validation | 65 | 15% |
| test | 65 | 15% |
| **total** | **433** | **100%** |

## 5. Candidate ranking

Restarts: 192. Feasible: 49. Unique after deduplication: 49. Retained: 6.

| Rank | Candidate | Images (t/v/te) | Negatives | `vest_loose` images | E_img | E_inst | E_neg | E_size | **Total** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `candidate_001` *(family best)* | 303/65/65 | 10/2/2 | 5/1/2 | 0.0703 | 0.0234 | 0.0385 | 0.0000 | **0.1322** |
| 2 | `candidate_002` | 303/65/65 | 10/2/2 | 5/1/2 | 0.0717 | 0.0507 | 0.0385 | 0.0000 | **0.1610** |
| 3 | `candidate_003` | 303/65/65 | 10/2/2 | 5/1/2 | 0.0727 | 0.0664 | 0.0385 | 0.0000 | **0.1776** |
| 4 | `candidate_004` | 303/65/65 | 10/2/2 | 5/1/2 | 0.0729 | 0.0707 | 0.0385 | 0.0000 | **0.1821** |
| 5 | `candidate_005` | 303/65/65 | 10/2/2 | 5/1/2 | 0.0703 | 0.0776 | 0.0385 | 0.0000 | **0.1864** |
| 6 | `candidate_006` *(family best)* | 303/65/65 | 10/2/2 | 4/2/2 | 0.1177 | 0.0575 | 0.0385 | 0.0000 | **0.2137** |

Full assignments are in `reports/split_candidates/`, one file per candidate, keyed by `group_id` with a `provisional_split` column. The column is named `provisional_split` rather than `split` so no downstream reader can mistake it for a frozen assignment.

| Candidate | Assignment SHA-256 |
| --- | --- |
| `candidate_001` | `081c39e729697f081302fc809eb4b78fff6450e910df254b88f311d623aa1fbe` |
| `candidate_002` | `aa357d6f7bd7740ef28063ccd2aee47d6be72b92f3b5917759f442499b89b55c` |
| `candidate_003` | `0415850581e7412805acd4c2a9c3a1f469ff86053276f168feb76008b79d8b65` |
| `candidate_004` | `6c7d33a49bfa226cbbd30b43798285e33f66fc5e4c3a5dadb9eaa998b9f8afe3` |
| `candidate_005` | `b026fef61ab619667ab6e5bd490cfd43862433a2ccab824ebc38500197fca9de` |
| `candidate_006` | `fd051d520670be521bae3b7100e035615df12659745ad1ca2cbbd75577756e4c` |

## 6. Class distribution by images

**`candidate_001`** - images carrying each class:

| Class | train | % | validation | % | test | % | total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `helmet_loose` | 53 | 70.7% | 11 | 14.7% | 11 | 14.7% | 75 |
| `helmet_on_head` | 123 | 69.9% | 26 | 14.8% | 27 | 15.3% | 176 |
| `person` | 256 | 69.9% | 55 | 15.0% | 55 | 15.0% | 366 |
| `vest_loose` | 5 | 62.5% | 1 | 12.5% | 2 | 25.0% | 8 |
| `vest_on_body` | 146 | 70.2% | 31 | 14.9% | 31 | 14.9% | 208 |

**`candidate_002`** - images carrying each class:

| Class | train | % | validation | % | test | % | total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `helmet_loose` | 53 | 70.7% | 11 | 14.7% | 11 | 14.7% | 75 |
| `helmet_on_head` | 123 | 69.9% | 27 | 15.3% | 26 | 14.8% | 176 |
| `person` | 255 | 69.7% | 56 | 15.3% | 55 | 15.0% | 366 |
| `vest_loose` | 5 | 62.5% | 1 | 12.5% | 2 | 25.0% | 8 |
| `vest_on_body` | 146 | 70.2% | 31 | 14.9% | 31 | 14.9% | 208 |

**`candidate_003`** - images carrying each class:

| Class | train | % | validation | % | test | % | total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `helmet_loose` | 53 | 70.7% | 11 | 14.7% | 11 | 14.7% | 75 |
| `helmet_on_head` | 124 | 70.5% | 26 | 14.8% | 26 | 14.8% | 176 |
| `person` | 258 | 70.5% | 55 | 15.0% | 53 | 14.5% | 366 |
| `vest_loose` | 5 | 62.5% | 1 | 12.5% | 2 | 25.0% | 8 |
| `vest_on_body` | 146 | 70.2% | 31 | 14.9% | 31 | 14.9% | 208 |

**`candidate_004`** - images carrying each class:

| Class | train | % | validation | % | test | % | total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `helmet_loose` | 53 | 70.7% | 11 | 14.7% | 11 | 14.7% | 75 |
| `helmet_on_head` | 123 | 69.9% | 27 | 15.3% | 26 | 14.8% | 176 |
| `person` | 258 | 70.5% | 53 | 14.5% | 55 | 15.0% | 366 |
| `vest_loose` | 5 | 62.5% | 1 | 12.5% | 2 | 25.0% | 8 |
| `vest_on_body` | 146 | 70.2% | 31 | 14.9% | 31 | 14.9% | 208 |

**`candidate_005`** - images carrying each class:

| Class | train | % | validation | % | test | % | total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `helmet_loose` | 53 | 70.7% | 11 | 14.7% | 11 | 14.7% | 75 |
| `helmet_on_head` | 123 | 69.9% | 27 | 15.3% | 26 | 14.8% | 176 |
| `person` | 256 | 69.9% | 55 | 15.0% | 55 | 15.0% | 366 |
| `vest_loose` | 5 | 62.5% | 1 | 12.5% | 2 | 25.0% | 8 |
| `vest_on_body` | 146 | 70.2% | 31 | 14.9% | 31 | 14.9% | 208 |

**`candidate_006`** - images carrying each class:

| Class | train | % | validation | % | test | % | total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `helmet_loose` | 53 | 70.7% | 11 | 14.7% | 11 | 14.7% | 75 |
| `helmet_on_head` | 122 | 69.3% | 27 | 15.3% | 27 | 15.3% | 176 |
| `person` | 257 | 70.2% | 54 | 14.8% | 55 | 15.0% | 366 |
| `vest_loose` | 4 | 50.0% | 2 | 25.0% | 2 | 25.0% | 8 |
| `vest_on_body` | 146 | 70.2% | 31 | 14.9% | 31 | 14.9% | 208 |

## 7. Class distribution by instances

**`candidate_001`** - annotation counts:

| Class | train | % | validation | % | test | % | total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `helmet_loose` | 276 | 70.2% | 57 | 14.5% | 60 | 15.3% | 393 |
| `helmet_on_head` | 220 | 70.1% | 47 | 15.0% | 47 | 15.0% | 314 |
| `person` | 641 | 70.1% | 137 | 15.0% | 136 | 14.9% | 914 |
| `vest_loose` | 30 | 66.7% | 8 | 17.8% | 7 | 15.6% | 45 |
| `vest_on_body` | 255 | 69.9% | 55 | 15.1% | 55 | 15.1% | 365 |

**`candidate_002`** - annotation counts:

| Class | train | % | validation | % | test | % | total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `helmet_loose` | 280 | 71.2% | 59 | 15.0% | 54 | 13.7% | 393 |
| `helmet_on_head` | 220 | 70.1% | 47 | 15.0% | 47 | 15.0% | 314 |
| `person` | 640 | 70.0% | 137 | 15.0% | 137 | 15.0% | 914 |
| `vest_loose` | 33 | 73.3% | 8 | 17.8% | 4 | 8.9% | 45 |
| `vest_on_body` | 255 | 69.9% | 55 | 15.1% | 55 | 15.1% | 365 |

**`candidate_003`** - annotation counts:

| Class | train | % | validation | % | test | % | total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `helmet_loose` | 275 | 70.0% | 60 | 15.3% | 58 | 14.8% | 393 |
| `helmet_on_head` | 220 | 70.1% | 47 | 15.0% | 47 | 15.0% | 314 |
| `person` | 639 | 69.9% | 137 | 15.0% | 138 | 15.1% | 914 |
| `vest_loose` | 33 | 73.3% | 3 | 6.7% | 9 | 20.0% | 45 |
| `vest_on_body` | 255 | 69.9% | 55 | 15.1% | 55 | 15.1% | 365 |

**`candidate_004`** - annotation counts:

| Class | train | % | validation | % | test | % | total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `helmet_loose` | 272 | 69.2% | 61 | 15.5% | 60 | 15.3% | 393 |
| `helmet_on_head` | 221 | 70.4% | 46 | 14.6% | 47 | 15.0% | 314 |
| `person` | 639 | 69.9% | 138 | 15.1% | 137 | 15.0% | 914 |
| `vest_loose` | 33 | 73.3% | 3 | 6.7% | 9 | 20.0% | 45 |
| `vest_on_body` | 256 | 70.1% | 55 | 15.1% | 54 | 14.8% | 365 |

**`candidate_005`** - annotation counts:

| Class | train | % | validation | % | test | % | total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `helmet_loose` | 278 | 70.7% | 56 | 14.2% | 59 | 15.0% | 393 |
| `helmet_on_head` | 220 | 70.1% | 47 | 15.0% | 47 | 15.0% | 314 |
| `person` | 640 | 70.0% | 137 | 15.0% | 137 | 15.0% | 914 |
| `vest_loose` | 30 | 66.7% | 4 | 8.9% | 11 | 24.4% | 45 |
| `vest_on_body` | 255 | 69.9% | 55 | 15.1% | 55 | 15.1% | 365 |

**`candidate_006`** - annotation counts:

| Class | train | % | validation | % | test | % | total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `helmet_loose` | 275 | 70.0% | 58 | 14.8% | 60 | 15.3% | 393 |
| `helmet_on_head` | 220 | 70.1% | 47 | 15.0% | 47 | 15.0% | 314 |
| `person` | 640 | 70.0% | 137 | 15.0% | 137 | 15.0% | 914 |
| `vest_loose` | 27 | 60.0% | 7 | 15.6% | 11 | 24.4% | 45 |
| `vest_on_body` | 255 | 69.9% | 55 | 15.1% | 55 | 15.1% | 365 |

## 8. `vest_loose` allocation

45 instances across 8 images held in **7** indivisible groups: one group (`manual_dup_010`) contains two of the images, so they cannot be separated and the class moves in chunks. This corrects the phase 5C.1 brief, which stated that no `vest_loose` image belongs to a duplicate group - that was true before phase 5B.1 confirmed that pair.

| Candidate | train | validation | test | instances (t/v/te) | family |
| --- | --- | --- | --- | --- | --- |
| `candidate_001` | 5 | 1 | 2 | 30/8/7 | 5/1/2 (family best) |
| `candidate_002` | 5 | 1 | 2 | 33/8/4 | 5/1/2 |
| `candidate_003` | 5 | 1 | 2 | 33/3/9 | 5/1/2 |
| `candidate_004` | 5 | 1 | 2 | 33/3/9 | 5/1/2 |
| `candidate_005` | 5 | 1 | 2 | 30/4/11 | 5/1/2 |
| `candidate_006` | 4 | 2 | 2 | 27/7/11 | 4/2/2 (family best) |

Source images carrying the class, per split, for the best-scoring candidate `candidate_001`:

| Split | Source image | Group | `vest_loose` instances | Co-occurring classes |
| --- | --- | --- | --- | --- |
| train | `TwF1Rahc7CIP2gpmlbLS` | `manual_dup_010` | 6 | `helmet_loose` |
| train | `Vtt35PA66TdcCFU5LqWG` | `singleton-Vtt35PA66TdcCFU5LqWG` | 13 | `helmet_loose` |
| train | `dOdZyD1R1kVyKKu9TD1D` | `singleton-dOdZyD1R1kVyKKu9TD1D` | 3 | - |
| train | `pd0SyehnE6l5xoGZqJCt` | `manual_dup_010` | 7 | `helmet_loose` |
| train | `xCsoN8tQQsB259iyGJJD` | `singleton-xCsoN8tQQsB259iyGJJD` | 1 | - |
| validation | `Mbgd4Kw22vX5vdWxqfRn` | `singleton-Mbgd4Kw22vX5vdWxqfRn` | 8 | - |
| test | `1dnS0NtMgQN35Eb4GNTe` | `singleton-1dnS0NtMgQN35Eb4GNTe` | 3 | - |
| test | `HwlufkrcXA9g8e3aNCm5` | `singleton-HwlufkrcXA9g8e3aNCm5` | 4 | `helmet_loose` |

## 9. Negative images

| Candidate | train | validation | test |
| --- | --- | --- | --- |
| `candidate_001` | 10 | 2 | 2 |
| `candidate_002` | 10 | 2 | 2 |
| `candidate_003` | 10 | 2 | 2 |
| `candidate_004` | 10 | 2 | 2 |
| `candidate_005` | 10 | 2 | 2 |
| `candidate_006` | 10 | 2 | 2 |

14 negatives in total. The phase 4B audit found no missing target label on them, so they are deliberate background examples rather than defects, and none of them may be hoarded in train.

## 10. Group integrity

| Check | Result |
| --- | --- |
| `candidate_001` | 422 groups assigned exactly once, 433 images, hard violations: 0 |
| `candidate_002` | 422 groups assigned exactly once, 433 images, hard violations: 0 |
| `candidate_003` | 422 groups assigned exactly once, 433 images, hard violations: 0 |
| `candidate_004` | 422 groups assigned exactly once, 433 images, hard violations: 0 |
| `candidate_005` | 422 groups assigned exactly once, 433 images, hard violations: 0 |
| `candidate_006` | 422 groups assigned exactly once, 433 images, hard violations: 0 |

Group indivisibility is structural rather than checked: the assignment maps group identifiers, so an image cannot be assigned independently of its group.

## 11. Post-hoc source-distribution diagnostics

Computed **after** ranking and never fed back into the objective. The purpose is to catch a catastrophic accidental shift, not to make the optimiser cleverer; changing the objective in response to what is seen here would be tuning on the answer, and would require regenerating every candidate under a new protocol.

Split means for `candidate_001`:

| Characteristic | train | validation | test |
| --- | --- | --- | --- |
| width | 1503.09 | 1505.48 | 1437.09 |
| height | 1355.36 | 1335.75 | 1358.17 |
| aspect_ratio | 1.2285 | 1.2361 | 1.1773 |
| luminance_mean | 0 | 0 | 0 |
| contrast | 0 | 0 | 0 |

No candidate raised a `DISTRIBUTION_WARNING`: no source characteristic differs between splits by more than 25% of its mean.

## 12. Trade-offs between the top candidates

The scalar total is a ranking device, not a verdict. What separates these candidates in practice:

* **`candidate_001`** (total 0.1322): `vest_loose` 5/1/2 images and 30/8/7 instances, negatives 10/2/2. Image balance 0.0703, instance balance 0.0234.
* **`candidate_002`** (total 0.1610): `vest_loose` 5/1/2 images and 33/8/4 instances, negatives 10/2/2. Image balance 0.0717, instance balance 0.0507.
* **`candidate_003`** (total 0.1776): `vest_loose` 5/1/2 images and 33/3/9 instances, negatives 10/2/2. Image balance 0.0727, instance balance 0.0664.
* **`candidate_004`** (total 0.1821): `vest_loose` 5/1/2 images and 33/3/9 instances, negatives 10/2/2. Image balance 0.0729, instance balance 0.0707.
* **`candidate_005`** (total 0.1864): `vest_loose` 5/1/2 images and 30/4/11 instances, negatives 10/2/2. Image balance 0.0703, instance balance 0.0776.
* **`candidate_006`** (total 0.2137): `vest_loose` 4/2/2 images and 27/7/11 instances, negatives 10/2/2. Image balance 0.1177, instance balance 0.0575.

The rare class is where these differ most and where the scalar helps least. `vest_loose` instances are very unevenly distributed across its 8 images - one image alone carries 13 - so two candidates with the same image split can differ sharply in instances. A reviewer should decide whether image coverage or instance coverage matters more for the per-class claims this project intends to make about the holdout.

## 13. Limitations

* **No candidate is selected.** `final_selected_candidate` is `UNSELECTED_PENDING_REVIEW`. `algorithmic_best_candidate` names only the lowest scorer under the predeclared objective, which is `candidate_001`.
* **The objective is a proxy.** Balanced counts do not guarantee comparable difficulty. Nothing here measures how hard the images are.
* **Local search is not exhaustive.** A better assignment may exist. The search is reproducible, not optimal, and the space was not brute-forced.
* **Instance-level proportionality is not always reachable.** Objects co-occur inside images and images are bound into groups, so some deviation is structural rather than a search failure.
* **`vest_loose` remains thin.** Whatever the split, per-class conclusions about it will rest on a handful of images. That is a property of the dataset, not of the split.
* **No model exists**, so nothing here has been validated against downstream behaviour.
