# Canonical Task Dataset Materialisation - Phase 5D

Phase: 5D · Commit: `c952b43a00168e198c190cd5798a8d2d1de2c488` · Status: **DEVELOPMENT_DATASETS_MATERIALIZED**

This phase turned the membership frozen in phase 5C.2 into two task views of the same data. The property it claims is that it **added nothing**: the same source bytes, the same annotations, the same class order. Anything it changed about the data would be a defect, so every such property below is measured rather than asserted.

**Only `train` and `validation` were materialised.** The holdout was not read, not measured, not copied and not counted from its labels; the counts quoted for it anywhere in this repository are the phase 5C.2 protocol facts, not new findings.

Claims are labelled `FACT` (measured from a committed artifact), `COMPUTED RESULT` (produced by this run), `CANONICALIZATION POLICY` (a decision about the ground truth), `HOLDOUT POLICY` and `FUTURE MODEL ADAPTER REQUIREMENT`.

## 1. Canonical input state

`FACT` The annotation source is the phase 5A canonical snapshot (`CURRENT_COMPLETE_GEOMETRY`) as filtered by the phase 5B modelling population: 433 eligible images and 2031 retained annotations in **original image coordinates**. The v4 export is not used; its geometry is expressed after a stretch resize to 640x640.

| Fingerprint | Value |
| --- | --- |
| `modeling_population_sha256` | `afe8b73752f2dffda51e5cf9b67c2169966e94a73c0806c4c96405420e2c86d7` |
| `groups_sha256` | `d6065c4e2adc56b5bcc6e027458807ef2f3281ed530915d690d55921a289faaf` |
| `class_map_sha256` | `596dab5b5756e3d4253924f84bf46d37a87d80b6ec830da781a08a93cd823753` |
| `split_assignment_sha256` | `a230869ff4cb45f53654d67357a27f2a0def6d8ba79f9fa4880f0fdda2f046cc` |
| `materialization_config_sha256` | `352efcc105c295c17599fe73dda1506c61aa17de0f90bd3caa1612e40c44a541` |
| `canonical_id_assignment_sha256` | `04101ab45cc7e40c7b2a5cd7d43124644cb5fe0c067b30abe2fb5233dd0854cb` |

## 2. Frozen split reference

`FACT` Membership comes from `reports/split_manifest.json` and from nothing else. The provider's split is read nowhere - not as an input, an initialisation or a destination - and it is not present in any artifact this phase consults.

## 3. Why only train and validation

`HOLDOUT POLICY` The holdout is protected data, and materialising it now would create an opportunity to look at it that the protocol does not permit. The code path that will materialise it exists and is exercised by tests against synthetic fixtures, but it requires `allow_test=True` **and** `CSVISION_ALLOW_TEST_SPLIT=1`, and it was not invoked. The configuration records this as `test_materialization: disabled_during_development`.

`HOLDOUT POLICY` The final-evaluation phase runs the **same function with the same configuration**. There is no split-specific branch, so the holdout cannot receive different preprocessing than the data the models were developed on - which is the point of materialising it late rather than differently.

## 4. Canonical task formats

* `canonical_detection_format`: **COCO**
* `canonical_segmentation_format`: **COCO_INSTANCE_SEGMENTATION**
* `model_specific_adapter`: **NOT_YET_SELECTED**

`CANONICALIZATION POLICY` COCO for both, and **no YOLO labels yet**. The canonical annotation state holds polygon geometry *and* compressed RLE; COCO carries both natively, while YOLO's segmentation format carries polygons only. Writing YOLO now would mean rasterising and re-polygonising every RLE mask - an approximation applied to the ground truth, before a model has even been chosen. A model-specific representation must not redefine ground truth.

## 5. Image copy policy

`CANONICALIZATION POLICY` `image_copy_mode: BINARY_IDENTICAL`. No resize, no crop, no re-encode, no EXIF-driven rotation, no colour-space change, no augmentation. Images are transferred with binary copy semantics and never routed through an image library, so no decoder can quietly re-encode them.

`CANONICALIZATION POLICY` `image_naming: SOURCE_IMAGE_ID_PLUS_ORIGINAL_EXTENSION`. The provider's filenames are descriptive, long enough to hit the Windows path limit on this machine, and not guaranteed unique; the canonical id is short, unique and already the join key.

## 6. Class map

| Class | COCO `category_id` |
| --- | --- |
| `helmet_loose` | 0 |
| `helmet_on_head` | 1 |
| `person` | 2 |
| `vest_loose` | 3 |
| `vest_on_body` | 4 |

`CANONICALIZATION POLICY` `category_id_policy: CANONICAL_CLASS_INDEX` - the COCO category id **is** the frozen class index. A second numbering would create two orderings to keep in step, and `class_map_sha256` would stop describing what the datasets contain. The order was not re-derived from dictionary iteration, provider category order or runtime sorting.

`FACT` The provider's placeholder category `object` carries no annotation and is absent from both task views; its exclusion does not shift any other class index.

## 7. Segmentation geometry preservation

`CANONICALIZATION POLICY` `segmentation_geometry_policy: PRESERVE_CANONICAL_REPRESENTATION`. A polygon stays a polygon; a compressed RLE stays a compressed RLE, emitted as `{"size": [h, w], "counts": ...}`. No RLE is converted to a polygon anywhere in this phase.

`FACT` Two annotations on image `OQJwjQoYsf1KUgr9G0V8` carry `geometry_origin = SYNTHETIC_FROM_PROVIDER_BBOX`: they had no provider segmentation and phase 5B materialised them as four-corner rectangles clipped to the canvas. They are emitted as those exact rectangles and flagged in the emitted records. **They are not human-drawn masks** and must never be reported as such.

## 8. Detection bbox derivation

`CANONICALIZATION POLICY` `detection_bbox_source: DERIVED_FROM_CANONICAL_SEGMENTATION`. Every detection box is computed from the canonical segmentation by the geometry implementation audited in phase 4A, which handles polygons, compressed RLE and the synthetic rectangles. **The provider's supplied bbox is not used as ground truth anywhere**: phase 4A measured it disagreeing with its own geometry by up to 123.5 px inside the v4 export.

`COMPUTED RESULT` As a corroboration, each derived box was compared against the box phase 5A measured independently from the same geometry:

| Split | Annotations compared | Max delta (px) | Tolerance (px) | Within tolerance |
| --- | --- | --- | --- | --- |
| train | 1422 | 0.0 | 1.0 | True |
| validation | 304 | 0.0 | 1.0 | True |

`CANONICALIZATION POLICY` The detection view keeps the **segmentation-derived `area`** rather than recomputing it from the box. That is COCO's own convention - the reference dataset's detection annotations carry mask area - and it is what keeps the small/medium/large size buckets identical between the two tasks. A `bbox_area` field is emitted alongside for anything that wants the rectangle's area explicitly.

## 9. Output counts

| Split | Images | Annotations | Zero-instance images | Status |
| --- | --- | --- | --- | --- |
| train | 303 | 1422 | 10 | MATERIALIZED |
| validation | 65 | 304 | 2 | MATERIALIZED |
| test | - | - | - | **NOT_MATERIALIZED_PROTECTED_HOLDOUT** |
| **development total** | **368** | **1726** | **12** | |

`COMPUTED RESULT` 368 images and 1726 annotations, derived from the frozen manifest and verified against the canonical population - not copied from a brief.

`COMPUTED RESULT` Instances per class in the materialised development data:

| Class | train | validation | development total |
| --- | --- | --- | --- |
| `helmet_loose` | 276 | 57 | 333 |
| `helmet_on_head` | 220 | 47 | 267 |
| `person` | 641 | 137 | 778 |
| `vest_loose` | 30 | 8 | 38 |
| `vest_on_body` | 255 | 55 | 310 |

`FACT` Zero-instance images are **retained as image records with no annotations**. The phase 4B audit confirmed them as deliberate negatives rather than unlabelled images, and a task view that silently dropped them would change the dataset.

## 10. Cross-task alignment

| Split | Image ids match | Annotation ids match | Categories match | Boxes match | Aligned |
| --- | --- | --- | --- | --- | --- |
| train | True | True | True | True | True |
| validation | True | True | True | True | True |

`COMPUTED RESULT` For each development split the two views hold the same COCO image ids, the same source image ids, the same annotation ids, the same categories, the same class per annotation and the same boxes. No image or annotation exists in one view and not the other. The only intended difference is that the segmentation view carries mask geometry and the detection view does not.

## 11. Geometry round-trip validation

`COMPUTED RESULT` Preservation is claimed, so it is measured. The emitted segmentation file is **read back from disk** and compared against the canonical state, which tests the serialisation as well as the construction. RLE is compared by **decoded mask**, not by comparing `counts` strings: identical strings would prove only that a copy happened, whereas decoding both sides proves the file describes the same pixels.

| Split | Polygons | RLE | Synthetic rectangles | Matches | Mismatches | Max polygon delta (px) |
| --- | --- | --- | --- | --- | --- | --- |
| train | 691 | 729 | 2 | 1422 | 0 | 0.0 |
| validation | 152 | 152 | 0 | 304 | 0 | 0.0 |

## 12. Fingerprints

| Split | Task | Document SHA-256 | Image content SHA-256 |
| --- | --- | --- | --- |
| train | detection | `e17556634fe76304766ea143856777e1e1e3cf7c7ba2331a95eb47732f16ec1e` | `78b392958070ab9ecb229cc9699bc54c92bf6a389511dadf582950963c07cb5f` |
| train | segmentation | `25fee62e589f84839b63d6dd5d8c3f6596dc155e0a5b86cc7b575cfb037955c8` | `` |
| validation | detection | `3e95f7b39e86234e879eb63d5875c171fba4799086d2b0235eb4c593d52782f9` | `15b3d63745bcf06bd7a9ebbde490f91493e0d60256a16370ff26fefd3408f9ab` |
| validation | segmentation | `3be5d29ac973bcd81e57307759175d2e857ed10ad05e4795675ce1e037839068` | `` |

`FACT` The document fingerprint covers the categories, the image table and the annotations - the ground truth - and excludes the `info` block, which describes the protocol rather than the data. The image-content fingerprint covers the bytes of every materialised image, so it moves when an image changes and not when a directory is re-listed.

## 13. Reproducibility

`FACT` Materialisation is a pure function of the canonical population, the frozen split manifest, the frozen class map, `configs/task_materialization.yaml` and this code. No timestamp is written into any emitted file, COCO ids come from a sorted canonical ordering rather than filesystem traversal, and JSON is emitted with sorted keys, so two runs produce byte-identical output. Only the provenance record's `created_at` differs between runs, which is what a run record is for.

`FACT` COCO numeric ids are assigned over the **whole modelling population**, not per split. That is what lets the holdout be materialised later, by this same code, without renumbering anything that already exists, and it is why an image carries one id in both task views.

## 14. Protected-test policy

`HOLDOUT POLICY` `test` is `NOT_MATERIALIZED_PROTECTED_HOLDOUT`. This phase created no test image directory, no test COCO file, no test contact sheet, no test class diagnostic and no test geometry statistic. It computed nothing new about the holdout at all: the manifest records only a status and a reason. `CSVISION_ALLOW_TEST_SPLIT` was not set.

`HOLDOUT POLICY` Reading the holdout still requires both opt-ins, and the materialisation path routes through the same guard as every other consumer. Tests exercise that path against synthetic fixtures only.

## 15. Future model-adapter requirement

`FUTURE MODEL ADAPTER REQUIREMENT` `model_specific_adapter: NOT_YET_SELECTED`. If a YOLO segmentation stack is chosen later, then **before any training run**, the adapter must:

1. convert the canonical masks to the model-required polygon format;
2. rasterise the converted polygons;
3. compare the rasterised result against the canonical masks;
4. report per-instance mask IoU and area error;
5. identify disconnected-component and hole cases, which polygons cannot express;
6. reject or reconsider the adapter if the geometry loss is material.

That audit is a phase 8 / model-adapter concern and **was not performed here**. No conversion was attempted, so this phase makes no claim about how lossy it would be.

## 16. Limitations

* `LIMITATION` **Preservation is verified, fitness is not.** This phase proves the emitted datasets carry the canonical geometry unchanged. It says nothing about whether that geometry is correct, or whether the classes are separable.
* `LIMITATION` **Nothing is validated against the holdout.** No property measured here has been checked on the protected split, by design. If the holdout contains a geometry case the development data does not, it will first be seen at final materialisation.
* `LIMITATION` **Round-trip equality is not accuracy.** Decoded masks matching proves the emitted file describes the same pixels as the canonical state; it does not prove those pixels outline the object correctly.
* `LIMITATION` **The category id starts at 0.** COCO permits it and the frozen class map requires it, but some tooling reserves 0 for background. A model adapter may need to re-index, and if it does, that re-indexing must be recorded rather than done silently.
* `LIMITATION` **Bulk data is not committed.** `data/processed/canonical` is git-ignored. What is committed is the manifest and this report, which together make the datasets re-derivable; the data itself is reproduced by re-running the command.
* `LIMITATION` **No model exists.** No training, no inference and no metric has been produced, on any split.

## 17. What this phase did not do

No model was trained and no framework for training one was installed. No inference was run. No YOLO detection or segmentation label was written. No image was resized, cropped, re-encoded or augmented. No real holdout image or annotation was materialised, read or measured, and no new statistic about the holdout was computed. Phase 6 has not been started.
