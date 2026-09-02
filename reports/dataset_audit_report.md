# Dataset Audit Report - source integrity and leakage risk

Generated: 2026-09-02T12:24:45+00:00 · Phase: 4A · Commit: `ed9f59aed5b54335a344ec12e82b8bf1e9cd92d2`

Scope: the **436 independent source images**, acquired at original resolution from the provider. The 742-image version-4 export is a *derived* population and is used in this document only where explicitly said so.

This is the automated half of the audit. Semantic questions - is this a valid negative, are these two frames the same scene - are deliberately left to human review; see `manual_review_manifest.csv`.

## Population

| | Value | Basis |
| --- | --- | --- |
| Source images recovered | 436 | COMPUTED |
| Expected (phase 3) | 436 | FACT |
| Annotated instances | 2,031 | COMPUTED |
| Provider split `test` | 43 | COMPUTED |
| Provider split `train` | 306 | COMPUTED |
| Provider split `valid` | 87 | COMPUTED |

**COMPUTED.** All 436 originals downloaded and decoded; 0 failures, and every decoded size matched the dimensions the provider reported.

## Exact duplicates

**Method (COMPUTED).** SHA-256 over the original image bytes.

| | Value |
| --- | --- |
| Images hashed | 436 |
| Unique hashes | 436 |
| Duplicate groups | 0 |
| Images in duplicate groups | 0 |
| Groups crossing a split | 0 |

**INTERPRETATION.** No byte-identical image appears twice. This rules out the crudest form of split contamination; it says nothing about visually similar images, which the next section covers.

## Near-duplicate candidates

**Method (COMPUTED).** Two independent 64-bit perceptual fingerprints per image - a difference hash (dHash) and a DCT-based perceptual hash (pHash) - compared exhaustively over all 94,830 unordered pairs. A pair is emitted when the dHash distance is <= 10 **or** the pHash distance is <= 10. Both thresholds were fixed from the conventional working range for 64-bit hashes before results were seen; the screen is deliberately recall-oriented.

| | Value |
| --- | --- |
| Images fingerprinted | 436 |
| Pairs compared | 94,830 |
| Candidate pairs | 11 |
| Same-split candidates | 5 |
| **Cross-split candidates** | **6** |

> **A fingerprint match is a candidate, not a verdict.** None of these pairs is claimed to be leakage. They are the shortlist a person must look at.

Strongest cross-split candidates (COMPUTED):

| Image A | Split | Image B | Split | dHash | pHash |
| --- | --- | --- | --- | --- | --- |
| `4UEcuYss` | valid | `6hpR44qu` | train | 1 | 0 |
| `FBzxttXY` | valid | `Um5Oft7w` | train | 0 | 0 |
| `Ogbeepjj` | valid | `Ud5BlYIB` | train | 0 | 0 |
| `THaelQmg` | train | `ymICdUCk` | valid | 2 | 0 |
| `L51HLb9p` | train | `WRFgtFoo` | valid | 1 | 8 |
| `kvlkvPKh` | valid | `tjHCOqwR` | test | 5 | 5 |

**INTERPRETATION.** 2 cross-split pair(s) are identical under *both* fingerprints while having different file hashes - i.e. visually the same picture stored as different bytes. That is the strongest automated signal available here, and it is the single most important thing for a reviewer to confirm or reject.

Full list: `near_duplicate_candidates.csv`. Visual pairs: `figures/review_g_near_duplicates.jpg` and `figures/review_h_near_duplicates.jpg`.

## Possible sequence / group structure

| | Value |
| --- | --- |
| Near-duplicate chains (connected components) | 11 |
| Images in a chain | 22 |
| Chains spanning more than one split | 6 |

Filename prefixes (COMPUTED):

| Prefix | Images |
| --- | --- |
| `pb_` | 12 |
| `photo-` | 213 |
| `premium_` | 75 |
| `px_` | 73 |
| `tu_` | 35 |
| `un_` | 21 |
| `wm_` | 7 |

**INTERPRETATION.** The prefixes look like stock-photography provenance markers rather than frame numbering, and no filename carries a frame index. That is weak evidence *against* video-sequence structure, not proof of its absence.

**OPEN QUESTION.** Whether the near-duplicate chains are genuinely the same scene photographed twice, the same stock image published twice, or a false alarm. Candidates are in `group_candidates.csv`; none has been promoted to a group.

## Source project vs frozen v4 export

| Split | Source instances | Export instances | Augmentation | Per source copy | Difference | Agrees |
| --- | --- | --- | --- | --- | --- | --- |
| train | 1479 | 2836 | x2 | 1,418.0 | +61.0 | False |
| valid | 373 | 371 | x1 | 371.0 | +2.0 | False |
| test | 179 | 166 | x1 | 166.0 | +13.0 | False |

**INTERPRETATION.** The live source project and the frozen v4 export describe different annotation counts. The export is a snapshot taken before later edits, so the two are not interchangeable. Phase 5 must state which is canonical; this audit does not decide it.

**OPEN QUESTION.** Which population is canonical for phase 5 - the frozen v4 export whose provenance is already recorded and hashed, or the live source project which has more annotations but no frozen identity? This audit does not decide it.

## Zero-instance images

**COMPUTED.** 17 source images carry no annotated instance: {'test': 4, 'train': 11, 'valid': 2}.

**FACT.** Phase 3 found 28 such records in the v4 export. That is consistent: the train split is duplicated by augmentation, so 11 train + 2 valid + 4 test source images become 22 + 2 + 4 = 28 export records.

**OPEN QUESTION.** Whether these are deliberate negative samples or images whose labels are missing. This cannot be settled by counting - it needs a person to look at them. All 17 are rendered in `figures/review_f_zero_instance.jpg` and listed in `empty_image_audit.csv`.

## Annotation geometry at source

| Geometry type | Instances |
| --- | --- |
| `mask` | 1,022 |
| `polygon` | 1,007 |
| `unknown` | 2 |

**COMPUTED.** The provider exposes polygon vertices only for `polygon` instances. `mask` instances carry a bounding box but no vertex list, so source-level mask-area analysis is **PARTIAL** by necessity, not by choice.

**COMPUTED.** 2 instance(s) have neither type. Both sit on one image, measure under 16 px on a side, and are positioned at the image edge.

**OPEN QUESTION.** Whether they are accidental micro-annotations. Flagged for visual review.

## Supplied bbox vs segmentation geometry (v4 export)

**COMPUTED.** Of 3,373 export annotations, 2,589 agree with their own segmentation within 1.0 px. Polygons agree essentially perfectly; the disagreement is concentrated in the RLE instances, where the supplied box is systematically the larger of the two.

Full analysis: [`bbox_consistency_audit.md`](bbox_consistency_audit.md). Visual examples: `figures/review_k_bbox_vs_segmentation.jpg`.

> **The phase 5 bbox policy remains provisional.** The preferred candidate policy is to derive every detection box from the segmentation geometry, but that is *pending visual validation*: a reviewer must first confirm that the segmentation is the better description of the object. Nothing is frozen.

## Provider split audit

The provider's split was measured, **not modified**.

| Split | Images | Instances | Negatives | helmet_loose | helmet_on_head | person | vest_loose | vest_on_body |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | 43 | 179 | 4 | 8 | 39 | 85 | 0 | 47 |
| train | 306 | 1479 | 11 | 337 | 194 | 657 | 32 | 259 |
| valid | 87 | 373 | 2 | 48 | 81 | 172 | 13 | 59 |

Images containing each class (not instance counts):

| Split | helmet_loose | helmet_on_head | person | vest_loose | vest_on_body |
| --- | --- | --- | --- | --- | --- |
| test | 5 | 20 | 35 | 0 | 23 |
| train | 55 | 116 | 254 | 7 | 147 |
| valid | 15 | 40 | 77 | 1 | 38 |

### Concerns (COMPUTED)

- split 'test' has zero instances of ['vest_loose']
- 6 near-duplicate candidate pair(s) cross a split boundary (candidates, not confirmed leakage)
- 6 near-duplicate chain(s) span more than one split

### Classification: `UNDETERMINED_PENDING_VISUAL_REVIEW`

**INTERPRETATION.** This is an audit recommendation and does **not** freeze or replace anything. The split cannot be judged suitable until a person has looked at the cross-split near-duplicate candidates: if they are genuine, the provider's split leaks and a new partition is required in phase 5; if they are false alarms, the remaining concern is the rare-class coverage below.
