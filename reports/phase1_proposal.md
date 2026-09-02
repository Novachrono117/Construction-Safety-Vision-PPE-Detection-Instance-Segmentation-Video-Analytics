# Phase 1 Proposal - Construction Safety Vision

**Author:** Vinicius Gomes · **Course:** Computer Vision and Pattern Recognition
**Repository:** <https://github.com/Novachrono117/Construction-Safety-Vision-PPE-Detection-Instance-Segmentation-Video-Analytics>

## Problem

Construction sites are among the most hazardous work environments, and a large
share of severe incidents involves protective equipment that is missing or not
actually being worn. Manual supervision of PPE compliance does not scale: it is
intermittent, subjective, and cannot review footage after the fact.

The task is: **given an image or video frame of a construction scene, locate
every person and every piece of protective equipment, and distinguish equipment
that is being worn from equipment that is merely present in the scene.**

That distinction is what makes the problem non-trivial. A helmet on a bench and a
helmet on a worker's head are visually similar objects with opposite safety
meanings, so a system that only detects "helmet" is useless for compliance
monitoring. The class set is therefore built around the *worn / not worn* state
rather than around the object alone.

## Target classes (verified in the annotations)

| Class | Meaning |
| --- | --- |
| `person` | A person in the scene. |
| `helmet_on_head` | A helmet being worn. |
| `helmet_loose` | A helmet present but not worn. |
| `vest_on_body` | A safety vest being worn. |
| `vest_loose` | A safety vest present but not worn. |

All five were confirmed to exist as COCO categories and to carry annotations in
the acquired export. The export additionally declares a placeholder category
`object` that carries no annotations and is excluded from the class map.

## Data source

**Construction PPE Compliance Detection** - AGIs Workspace, published on Roboflow
Universe, licensed **CC BY 4.0**.

- Project: <https://universe.roboflow.com/agis-workspace-8gs52/construction-ppe-compliance-detection>
- Version used: 4 · Acquired as a **COCO instance-segmentation** export
- Archive SHA-256: `5c0c35f79be251af349f289ab300a8c706f260f9a5b52f66facfbd23466538f6`

The dataset is not redistributed in this repository. It is re-obtainable from the
recorded coordinates, and the digest above proves any future download is the same
bytes. Full provenance: [`dataset_provenance.md`](dataset_provenance.md).

## Population: 436 source images, not 742

The project holds **436 independent annotated source images**, split by the
provider into 306 train / 87 validation / 43 test.

Version 4 exports **742** images. The difference is not additional data:
augmentation generates **two versions of each of the 306 source training
images** (306 x 2 = 612), while validation (87) and test (43) are left unchanged
(612 + 87 + 43 = 742). Three independent sources agree on this - the provider's
augmentation metadata (`image.versions: 2`), the exact arithmetic, and the
provider's own exported `README.roboflow.txt`.

**The 436 figure is therefore the one that matters** for the >= 300 annotated
image requirement, for all dataset statistics, and for split design. Treating the
742 as independent samples would inflate every count and would risk placing
augmented variants of the same source image on both sides of a split boundary.

## Annotation source and tool

Annotations were produced by the dataset's authors in Roboflow's annotation
tool and are consumed here as exported COCO instance segmentation. No new manual
annotation is performed in this project; the existing labels are audited rather
than re-drawn. Corrections, if any prove necessary, will be implemented as
reproducible pipeline code rather than as manual edits to the source files.

Instance segmentation is treated as the **single source of truth**: detection
boxes are derived mathematically from the segmentation geometry, so the detector
and the segmenter describe exactly the same objects and any difference in their
results is attributable to the models rather than to differing labels.

## Known data-quality items carried into the audit

Established, not speculated:

- Segmentation geometry mixes two representations - **1 570 polygon** and
  **1 803 compressed RLE** annotations. Both must be decoded.
- `vest_loose` is the rare class (77 annotations overall) and has **zero
  annotations in the provider's test split**, which constrains what a final
  per-class metric on that split could claim.
- 28 exported image records carry no annotations, while the provider reports
  zero unannotated images. Whether these are deliberate negatives or missing
  labels is an open question.
- The supplied `bbox` field disagrees materially with the RLE geometry it claims
  to enclose (see [`bbox_consistency_audit.md`](bbox_consistency_audit.md)), so
  boxes will be recomputed from the geometry rather than reused.

## Status

No model has been trained and no results exist. This proposal states the problem,
the classes, the data and its verified population only.
