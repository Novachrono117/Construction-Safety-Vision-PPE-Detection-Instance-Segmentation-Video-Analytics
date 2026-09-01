# Dataset Provenance

Generated: 2026-09-01T14:03:09+00:00 · Phase: 3 · Schema: 1

Every fact below is labelled **provider-reported** (copied from the provider's API or its exported documentation) or **computed** (derived here from the downloaded files). No number in this document was estimated.

## Identity

| Field | Value | Basis |
| --- | --- | --- |
| Provider | roboflow | given |
| Workspace | `agis-workspace-8gs52` | given |
| Project slug | `construction-ppe-compliance-detection` | given |
| Project name | Construction PPE Compliance Detection | provider-reported |
| Task type | `instance-segmentation` | provider-reported |
| Project URL | <https://universe.roboflow.com/agis-workspace-8gs52/construction-ppe-compliance-detection> | given |
| Version URL | <https://universe.roboflow.com/agis-workspace-8gs52/construction-ppe-compliance-detection/dataset/4> | given |
| Version | 4 ("2026-05-09 7:18pm") | provider-reported |
| Export format | `coco-segmentation` | given |
| Acquired at (UTC) | 2026-09-01T14:03:09+00:00 | computed |
| Acquired at commit | `985a4c816846c5ecfcb05f915ed03209b13ec260` | computed |
| Method | Roboflow REST API, stdlib urllib, Authorization: Bearer header; export link resolved at run time and never stored | given |

## Project level vs version level

These are **different populations** and are never collapsed into one number.

| | Project (source) | Version 4 |
| --- | --- | --- |
| Total images | 436 | 742 |
| test | 43 | 43 |
| train | 306 | 612 |
| valid | 87 | 87 |
| Unannotated | 0 | - |

Both rows are provider-reported. The version-level split counts were independently **confirmed** against the extracted files (see Structure below).

### Why the counts differ

**Conclusion: `B_source_plus_generated`**

Version contains source images plus offline-generated variants. Augmentation was applied to ['train'] only (2 versions per source image); ['test', 'valid'] are unchanged. The 742 version images are therefore NOT 742 independent samples: the independent source population is 436.

| Split | Source | Version | Status |
| --- | --- | --- | --- |
| test | 43 | 43 | unchanged |
| train | 306 | 612 | generated (x2) |
| valid | 87 | 87 | unchanged |

Corroborating evidence, three independent sources:

1. Provider version metadata declares `augmentation.image.versions = 2`.
2. The arithmetic is exact: the augmented split equals source x that multiplier, and the other splits are unchanged.
3. The provider's own exported `README.roboflow.txt` states the augmentation was applied "to create 2 versions of each source image".

> **Consequence for phases 4 and 5.** The 742 version images are not 742 independent samples. The independent population is 436 source images. Any metric, split or duplicate analysis that treats generated variants as independent would be wrong.

## Acquired artifact

| Field | Value |
| --- | --- |
| Path | `data/external/construction-ppe-compliance-detection-v4-coco-segmentation.zip` |
| SHA-256 | `5c0c35f79be251af349f289ab300a8c706f260f9a5b52f66facfbd23466538f6` |
| Size | 51867677 bytes |
| Extracted to | `data/raw/construction-ppe-compliance-detection-v4-coco-segmentation` |
| Members | 747 |

Annotation file digests (computed):

| File | SHA-256 | Bytes |
| --- | --- | --- |
| `test/_annotations.coco.json` | `6eae9600dc34bed12ef6b20f6f2772e792f6ff188048774e00f226d848875d56` | 147798 |
| `train/_annotations.coco.json` | `ff45f7fbefff1c125e13b1d212cab2ed1dede6acad909a86a333380872de4f5a` | 2195560 |
| `valid/_annotations.coco.json` | `cf30a026be7e2916942227231ddd9771b0f34ea11cce8b82f8403557b71bd822` | 274458 |

The archive is **not** committed. It is reproducible from the recorded provider coordinates; the digest above proves any future download is the same bytes.

## Structure (computed from the export)

| Split | Image records | Image files | Annotations | Sound |
| --- | --- | --- | --- | --- |
| test | 43 | 43 | 166 | yes |
| train | 612 | 612 | 2836 | yes |
| valid | 87 | 87 | 371 | yes |
| **total** | **742** | **742** | **3373** | |

Annotations per category (computed, all splits):

| Category | Annotations |
| --- | --- |
| `helmet_loose` | 718 |
| `helmet_on_head` | 508 |
| `person` | 1505 |
| `vest_loose` | 77 |
| `vest_on_body` | 565 |

Segmentation representation (computed, all annotations):

| Representation | Count |
| --- | --- |
| polygon | 1570 |
| rle | 1803 |

## Integrity checks

| Check | Result |
| --- | --- |
| splits_structurally_sound | `{'test': True, 'train': True, 'valid': True}` |
| all_splits_sound | `True` |
| image_records_match_files_on_disk | `True` |
| annotations_all_have_segmentation | `True` |
| expected_classes_present_as_categories | `True` |
| expected_classes_with_annotations | `True` |
| undeclared_categories | `['object']` |

## Notable findings

### `mixed_segmentation_representation` (HIGH)

Computed: `{'polygon': 1570, 'rle': 1803}`

The export mixes polygon and RLE segmentation in the same files. Phase 5 must decode both representations to derive bounding boxes; code that assumes polygons only would silently drop the RLE annotations.

### `class_absent_from_split` (HIGH)

Computed: `{'test': ['vest_loose']}`

At least one expected class has zero annotations in a split. A class absent from the evaluation split cannot be scored there, which constrains what the final metrics can claim.

### `images_without_annotations` (MEDIUM)

Computed: `{'per_split': {'test': 4, 'train': 22, 'valid': 2}, 'total': 28, 'provider_reported_unannotated': 0}`

Image records carry no annotation, while the provider reports unannotated=0. These may be deliberate background/negative samples rather than missed labels; phase 4 must determine which, since the distinction changes how they are used and scored.

### `undeclared_categories_present` (LOW)

Computed: `{'categories': ['object']}`

The export declares categories that the project configuration does not expect and that carry no annotations. This is the provider's conventional root placeholder; phase 5 must exclude it from the class map rather than shifting every class index.

Expected classes with zero annotations, per split (computed):

| Split | Classes absent |
| --- | --- |
| test | `['vest_loose']` |
| train | `[]` |
| valid | `[]` |

## Classes

- Expected from configuration: `['helmet_loose', 'helmet_on_head', 'person', 'vest_loose', 'vest_on_body']`
- Declared as COCO categories (computed): `['helmet_loose', 'helmet_on_head', 'object', 'person', 'vest_loose', 'vest_on_body']`
- Categories carrying annotations (computed): `['helmet_loose', 'helmet_on_head', 'person', 'vest_loose', 'vest_on_body']`
- Declared but never annotated (computed): `['object']`

## Academic requirement

- Minimum required annotated images: **300**
- Independent source images: **436**
- Currently supported: **True**
- Basis: Independent source images are provider-reported (project-level count); the version-level count includes offline-generated variants and must not be used for this requirement.

## License and attribution

- Dataset license: **CC BY 4.0** (<https://creativecommons.org/licenses/by/4.0/>)
- Creator: AGIs Workspace
- Attribution required: True
- Applies to the dataset only. The software license of this repository is a separate, still undecided matter. The dataset is not redistributed here.

Suggested attribution:

```text
Construction PPE Compliance Detection [dataset], version 4. AGIs Workspace, Roboflow Universe. Licensed CC BY 4.0.
https://universe.roboflow.com/agis-workspace-8gs52/construction-ppe-compliance-detection/dataset/4
```

## Unresolved questions (deferred to phases 4 and 5)

- Which of the 436 source images survive de-duplication (phase 4).
- Whether near-duplicate or same-scene frames exist across the provider's splits (phase 4); the provider's split assignment has not been audited.
- Whether augmented train variants leak information about validation or test images; this cannot be ruled out until source-image identity is recovered (phase 4/5).
- Whether the provider's train/valid/test assignment is adopted or the data is re-split (phase 5).
- Annotation quality: polygon geometry has been counted, not validated against the imagery (phase 4).
