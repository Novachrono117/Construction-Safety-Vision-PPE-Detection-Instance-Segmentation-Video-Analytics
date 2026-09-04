# YOLO Detection Adapter - Phase 6A

Phase: 6A · Commit: `33430b10835fd3383e14de8a37a5f6052035cc35` · Status: **DEVELOPMENT_ADAPTER_BUILT**

## 1. Purpose

`MODEL-ADAPTER POLICY` Ultralytics reads labels in its own format, so a derived representation is unavoidable. What is avoidable is a **lossy** one, and a lossy adapter is invisible: training would learn slightly wrong boxes and every downstream metric would quietly measure the wrong thing. This adapter therefore exists together with an audit that proves it changed nothing.

## 2. Canonical source

`FACT` The input is the canonical COCO detection dataset frozen in phase 5D (`reports/task_dataset_manifest.json`), whose boxes are themselves derived from the canonical segmentation geometry.

| Fingerprint | Value |
| --- | --- |
| `split_assignment_sha256` | `a230869ff4cb45f53654d67357a27f2a0def6d8ba79f9fa4880f0fdda2f046cc` |
| `modeling_population_sha256` | `afe8b73752f2dffda51e5cf9b67c2169966e94a73c0806c4c96405420e2c86d7` |
| `class_map_sha256` | `596dab5b5756e3d4253924f84bf46d37a87d80b6ec830da781a08a93cd823753` |
| `adapter_config_sha256` | `5f70fe2441526ef00d215e3a247bbd78eb116cf30094ead83f2821d21a065927` |

## 3. Why the adapter is derived, not canonical

`MODEL-ADAPTER POLICY` A model-specific representation must never redefine ground truth. If this adapter and the COCO file ever disagree, **the COCO file is right and the adapter is broken**. Concretely: the adapter is regenerated from COCO, never edited; it is git-ignored rather than committed; and no downstream phase may treat a YOLO label as the authority for what an object is.

`MODEL-ADAPTER POLICY` **Detection only.** No YOLO segmentation labels were generated. The canonical segmentation state holds compressed RLE, which YOLO's polygon-only format cannot express without rasterising and re-polygonising; that conversion must be quantitatively fidelity-audited before any model sees it, and it is a later model-adapter concern. Boxes have no such problem - a box is a box - which is exactly why the detection adapter can be proven lossless and a segmentation one cannot be assumed to be.

## 4. Class mapping

| YOLO index | Class |
| --- | --- |
| 0 | `helmet_loose` |
| 1 | `helmet_on_head` |
| 2 | `person` |
| 3 | `vest_loose` |
| 4 | `vest_on_body` |

`COMPUTED RESULT` The mapping is the frozen canonical class map, verified against `class_map_sha256` = `596dab5b5756e3d4253924f84bf46d37a87d80b6ec830da781a08a93cd823753`. No second ordering was introduced, and no index was re-derived from dictionary iteration or runtime sorting.

`FACT` The provider's placeholder category `object` carries no annotation and does not appear in the adapter, the dataset descriptor or any label file.

## 5. Membership validation

| Split | Adapter directory | Images | Annotations | Negatives |
| --- | --- | --- | --- | --- |
| train | `train` | 303 | 1422 | 10 |
| validation | `val` | 65 | 304 | 2 |
| test | - | - | - | **NOT_MATERIALIZED_PROTECTED_HOLDOUT** |
| **total** | | **368** | **1726** | **12** |

`COMPUTED RESULT` Every adapter image id was checked against the frozen split membership, resolved through the holdout guard rather than by listing a directory. Adapter images are byte-identical copies of the canonical materialised images, which are themselves byte-identical copies of the source originals - verified by hashing both sides of every copy.

## 6. Negative images

`MODEL-ADAPTER POLICY` `zero_instance_policy: EMPTY_LABEL_FILE`. The 12 development images carrying no annotation get an **empty label file**, which is Ultralytics' own representation of a background image. An empty label file is a valid negative, not a missing label: the dataset scan counts these as backgrounds rather than as corrupt entries. They were confirmed as deliberate negatives by the phase 4B audit and are never dropped.

## 7. Bbox conversion

```text
COCO   [x, y, w, h]        top-left corner plus extent, in source pixels
YOLO   [cx, cy, nw, nh]    centre plus extent, normalised by W and H

cx = (x + w / 2) / W       nw = w / W
cy = (y + h / 2) / H       nh = h / H
```

`MODEL-ADAPTER POLICY` `bbox_source: CANONICAL_COCO_DETECTION_BBOX`. The input is the segmentation-derived canonical box from phase 5D. **The provider's supplied bbox is not read here and never was ground truth**; phase 4A measured it disagreeing with its own geometry by up to 123.5 px.

`FACT` Coordinates are written fixed-point at 9 decimal places. Nine rather than the conventional six for a measured reason: the largest source image is 1880 px wide, so one unit in the ninth decimal is about 2e-6 px. Fixed-point rather than a shortest-repr float, so the label files are byte-identical between runs and machines.

`FACT` **Nothing was clamped.** Every canonical development box was measured to lie strictly inside its own canvas - zero boxes overshoot by any amount - so the `normalised_bound_epsilon` of 1e-09 absorbs floating-point error in the division and nothing else. A box outside the unit square is a hard error here, not a value to be quietly corrected.

## 8. Round-trip fidelity audit

`COMPUTED RESULT` Every development box was converted, written, **read back from the label file on disk**, decoded to source pixels and compared with the canonical box. Reading back from disk is the point: comparing against the in-memory boxes would prove nothing about the files a trainer actually loads.

| Split | Checked | Within tolerance | Exact | Max delta (px) | Mean delta (px) | Mismatches |
| --- | --- | --- | --- | --- | --- | --- |
| train | 1422 | 1422 | 24 | 1.474e-06 | 2.099e-07 | 0 |
| validation | 304 | 304 | 8 | 1.360e-06 | 2.110e-07 | 0 |
| **total** | **1726** | **1726** | **32** | **1.474e-06** | | **0** |

`COMPUTED RESULT` Tolerance 0.0001 px. 1726/1726 boxes within tolerance, 0 mismatches. The conversion is lossless within serialisation precision.

## 9. Fingerprints

| Split | YOLO labels | Adapter images | Canonical images |
| --- | --- | --- | --- |
| train | `6237624f585f8e9f7ac2565be1eacf4140f6d55ad2de5d3c85f2972efa649e5f` | `78b392958070ab9ecb229cc9699bc54c92bf6a389511dadf582950963c07cb5f` | `78b392958070ab9ecb229cc9699bc54c92bf6a389511dadf582950963c07cb5f` |
| validation | `d9ac55c70918b7ad7621ad6f7849d70c15f66585ec6b66f97b07fedfe6102c4b` | `15b3d63745bcf06bd7a9ebbde490f91493e0d60256a16370ff26fefd3408f9ab` | `15b3d63745bcf06bd7a9ebbde490f91493e0d60256a16370ff26fefd3408f9ab` |

`COMPUTED RESULT` The adapter and canonical image fingerprints are equal per split, which is what byte-identical means when stated as a measurement rather than an intention.

## 10. Test protection

`HOLDOUT POLICY` `test` is `NOT_MATERIALIZED_PROTECTED_HOLDOUT`. There is no test image directory, no test label directory, no test label file, and **no `test` key in the Ultralytics dataset descriptor** - a dataset file being exactly the kind of place a protected split gets reached by accident. `CSVISION_ALLOW_TEST_SPLIT` was not set and no new statistic about the holdout was computed.

`HOLDOUT POLICY` Split membership is resolved through the same guard as every other consumer, so building a holdout adapter needs both an in-code opt-in and the environment opt-in; a development run refuses it outright before either is consulted.

## 11. Limitations

* `LIMITATION` **Losslessness is about geometry, not correctness.** The audit proves the adapter reproduces the canonical boxes; it says nothing about whether those boxes are right.
* `LIMITATION` **Only boxes are proven.** No claim is made here about segmentation conversion, which is a different problem and has not been attempted.
* `LIMITATION` **Adapter images are a second copy.** They are verified byte-identical, but they are storage that must be regenerated rather than trusted after any change to the canonical datasets.
* `LIMITATION` **Nothing here has been validated against the holdout**, by design. If the holdout contains a box case the development data does not, it will first be seen at final materialisation.
* `LIMITATION` **No model exists.** No training, inference or metric has been produced from this adapter.
