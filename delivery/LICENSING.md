# Licensing and checkpoint distribution

## Repository license and decision

**PROJECT_REPOSITORY_LICENSE: AGPL-3.0**

The project source and repository are licensed under [GNU AGPL-3.0](../LICENSE)
(GNU Affero General Public License, version 3). This is an open-source academic
and public portfolio project built around Ultralytics YOLO11 code, architectures,
training/inference tooling, pretrained lineage and fine-tuned models.

The vendor's published open-source path expects the complete project to be
released under AGPL-3.0, or an applicable Enterprise license to be obtained.
No Enterprise license is recorded here. Adopting AGPL-3.0 at repository level
aligns the root license signal with that path. This is a repository compliance
and documentation decision based on published vendor terms, not legal advice
or a determination that every distribution obligation is already satisfied.
[Ultralytics licensing](https://www.ultralytics.com/license).

### Copyright notice

Copyright (c) 2026 Vinicius Gomes for original project contributions. Vinicius
retains copyright ownership; publishing under AGPL-3.0 does not transfer it.
Authorship is recorded in [project metadata](../pyproject.toml) and Git history.
Third-party copyrights and notices remain with their respective holders.

### Scope of the licensing evidence

| Subject | Evidence and public status |
| --- | --- |
| A. Ultralytics package/code | Installed package metadata declares AGPL-3.0; the vendor's current licensing page describes the AGPL-3.0/Enterprise paths for its code and training/inference pipelines. |
| B. YOLO architecture and pretrained models | The same vendor guidance explicitly includes architectures and models. The project's YOLO11 pretrained lineage remains subject to Ultralytics AGPL-3.0 terms and applicable Ultralytics terms. |
| C. Trained/fine-tuned D2 and S1 | The vendor explicitly includes trained/fine-tuned models. Both are Ultralytics YOLO models; repository licensing alone does not finalize their redistribution packaging or notices. |
| D. Original project code | Released as part of this GNU AGPL-3.0 repository; original authors retain copyright. |
| E. Dataset images and annotations | CC BY 4.0 remains separate. The repository code license does not relicense this material. |

The distinction above follows the [vendor's licensing page](https://www.ultralytics.com/license)
and [official YOLO licensing overview](https://docs.ultralytics.com/#yolo-licenses-how-is-ultralytics-yolo-licensed),
checked on 2026-09-15. Package metadata alone is not the basis for the repository
decision. Other dependencies retain their respective licenses.

### Canonical license integrity

The root `LICENSE` is the unmodified
[canonical GNU AGPL v3.0 plain text](https://www.gnu.org/licenses/agpl-3.0.txt),
downloaded from GNU on 2026-09-15 and verified byte-for-byte after copying:
34523 bytes, SHA-256
`0d96a4ff68ad6d4b6f1f30f713b18d5184912ba8dd389f86aa7710db079abcb0`.
The project copyright notice above is separate from the canonical license body.

## Dataset attribution

Construction PPE Compliance Detection [instance-segmentation dataset], version 4.
AGIs Workspace, Roboflow Universe. Workspace `agis-workspace-8gs52`, project
`construction-ppe-compliance-detection`.

- [Original source and version](https://universe.roboflow.com/agis-workspace-8gs52/construction-ppe-compliance-detection/dataset/4)
- [CC BY 4.0 terms](https://creativecommons.org/licenses/by/4.0/)
- [Acquisition provenance and limitations](../reports/dataset_provenance.md)
- [Canonical annotation decision](../reports/canonical_annotation_decision.md)

Canonical annotations use the recorded live source snapshot in original image
coordinates. Detection boxes were derived from segmentation; documented
synthetic geometry and preprocessing decisions retain their provenance. These
are project transformations, not an endorsement by the dataset creator.
Attribution, license links and identification of changes must accompany any
permitted downstream reuse. Phase 12B redistributes no dataset images. Older
provenance reports describe the code-license decision as pending at their date;
this document records the subsequent decision without rewriting those reports.
The repository's AGPL-3.0 grant does not replace CC BY 4.0 for dataset images
or annotations.

## Dependency and pretrained lineage evidence

Installed metadata inspected without importing a model runtime:

| Package | Inspected version | Declared license |
| --- | --- | --- |
| Ultralytics | 8.4.138 | AGPL-3.0 |
| PyTorch | 2.11.0+cu128 | BSD-3-Clause |
| torchvision | 0.26.0+cu128 | BSD |
| NumPy | 2.5.2 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| pycocotools | 2.0.11 | FreeBSD |
| SciPy | 1.18.1 | BSD-3-Clause; bundled components have separate notices |
| PyYAML | 6.0.3 | MIT |

This is a direct-dependency review, not a complete transitive software bill of
materials. Third-party dependencies retain their own licenses and notices.
The research lockfile is unchanged by this delivery phase.

The detector descends from `yolo11n.pt`; the segmenter from `yolo11n-seg.pt`.
Their exact lineage and training provenance are in the
[D2 training manifest](../reports/detection_D2_manifest.json) and
[S1 result manifest](../reports/segmentation_S1_result_manifest.json).

Official references (Ultralytics and CC BY terms checked on 2026-09-15):

- [Ultralytics licensing](https://www.ultralytics.com/license) describes AGPL-3.0
  or Enterprise licensing and explicitly includes trained/fine-tuned models.
- [Official YOLO licensing overview](https://docs.ultralytics.com/#yolo-licenses-how-is-ultralytics-yolo-licensed)
  corroborates the AGPL-3.0 and Enterprise options. The general FAQ is not relied
  on for the repository-level conclusion.
- [GNU AGPL v3.0 text](https://www.gnu.org/licenses/agpl-3.0.txt) is the canonical
  source of the root license.

## Distribution decision

**CHECKPOINT_REDISTRIBUTION_REQUIRES_LICENSE_REVIEW** for D2 and S1.

FACT: their recorded identities are frozen, but no public retrieval location is
recorded in the authoritative repository. UNKNOWN: whether a proposed release's
complete source package, notices and dataset-derived attribution satisfy all
applicable obligations. CC BY 4.0 on the dataset is not by itself a weight
redistribution clearance. No determination of prohibited redistribution is made.

Before publishing, review the Ultralytics pretrained/trained-model terms, the
scope of the corresponding source offering, dataset attribution and any rights
not granted by the dataset license. Record the resulting license basis and
notices. Until then, no weights are uploaded and GAP-008 remains OPEN.

## Conditional release plan

The proposed release name is `frozen-models-v1`. If cleared and subsequently
authorized for publication, assets would be:

- `construction-safety-d2-yolo11n-768.pt`
- `construction-safety-s1-yolo11n-seg-768.pt`
- `checkpoints.json`, `SHA256SUMS`, and a weight-specific license/notice file.

Use the exact bytes, sizes and provenance references in
[checkpoints.json](checkpoints.json), generated from the authoritative freezes.
Attach a public stable locator only after the asset actually exists. Do not
retrain, silently substitute base pretrained weights, or commit binaries to Git.

## Verifier now; fetcher later

`uv run python scripts/verify_checkpoint.py --model D2 --path <checkpoint-file>`
checks a manually supplied file's size and SHA-256 without deserializing it.
Use `--model S1` for the segmenter. Store lawfully obtained files under ignored
`artifacts/`; no publicly usable acquisition method is claimed today.

The current command does not download, load a model, or run inference. Phase 12B
tests it only with synthetic bytes and does not read the real checkpoints.
A later fetcher must accept only the explicitly documented source, write to
ignored storage, verify size and digest before an atomic promotion, and fail
closed on mismatch. Network tests must use synthetic fixtures.

## Phase 13B real-video media

The original Mdina construction footage is by **Frank Vincentz**, under
**CC BY-SA 3.0**. The derived demonstration MP4 and its six screenshots retain
that license; project contributions to these media use the same license.
They are not relicensed as repository code. Preserve the full title, creator,
source URL, license link and modification notice in the
[video attribution report](../reports/final_real_video_demo.md) beside any copy.
No endorsement by the creator is implied. Project code remains GNU AGPL-3.0.
