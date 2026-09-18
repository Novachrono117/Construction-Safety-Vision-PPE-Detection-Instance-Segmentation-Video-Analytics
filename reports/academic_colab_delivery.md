# Phase 15C - executable academic Colab with pedagogical training and evaluation

**Classification:** `EXECUTABLE_ACADEMIC_COLAB_COMPLETE`. **Assignment wording:** `COLAB_EXACT_WORDING_SATISFIED`.

Human cloud validation and local metadata verification are separate evidence categories.
Approved scientific baseline: `fa0bea2b072b124d65f4e9bd1b932a22d98149d2`.
Mode A validated revision: `958eefaa58b785ff2f9eb4dd2751efc26dd9b09e` (phase 15C, fresh CPU runtime).
Mode B validated revision: `8ce5d0375903e3e3760873e0a75ddce37cc1149b` (phase 14A, not re-executed).

## Human cloud validation events

Each event is evidence for the revision it names. A later event does not restate an earlier one as though it had been observed again, and none is removed when superseded.

| Phase | Mode | Revision | Runtime | RUN_INFERENCE | Date | Result | Standing |
|---|---|---|---|---|---|---|---|
| 14A | B | `8ce5d0375903e3e3760873e0a75ddce37cc1149b` | GPU | true | 2026-09-17 | PASS | `CURRENT_MODE_B_EVIDENCE` |
| 14C | A | `2e355d8012b6f1f49923c5cb3ead06760490f024` | CPU | false | 2026-09-18 | PASS | `SUPERSEDED_AS_CURRENT_MODE_A_EVIDENCE_BY_PHASE_15C` |
| 15C | A | `958eefaa58b785ff2f9eb4dd2751efc26dd9b09e` | CPU | false | 2026-09-18 | PASS | `CURRENT_MODE_A_EVIDENCE` |

## Notebook structure

- 1 Overview
- 2 Dataset and classes
- 3 TREINO / TRAINING
- 4 AVALIACAO / EVALUATION
- 5 Qualitative evidence
- 6 Video evidence
- 7 INFERENCIA / INFERENCE
- 8 Limitations and licences

### TREINO / TRAINING

`RECORDED_TRAINING_EVIDENCE` / `NO_NEW_TRAINING_EXECUTED`. Executable cells read each frozen model's committed experiment manifest and render its architecture, image size, epochs, batch, seed, the optimizer that `optimizer: auto` resolved to, the full augmentation set, the checkpoint selection rule, the selected epoch, the frozen checkpoint identity and the committed training and validation curves.

The default notebook does **not** retrain the models (`default_notebook_retrains: false`). Each model was trained exactly once under a protocol frozen before the run, and re-running it would produce a different checkpoint under the same name. The reproducible entry points are printed with each recipe:

- `uv run python scripts/train_detection_experiment.py --experiment D2`
- `uv run python scripts/train_segmentation_comparison.py`

### AVALIACAO / EVALUATION

Banner: `FINAL TEST RESULTS - READ FROM THE COMMITTED ONE-SHOT EVALUATION`. Executable cells read the committed one-shot holdout result artifacts by field and render canonical COCO average precision, operating-point precision and recall, per-class AP, the direct instance-mask IoU diagnostic, object-level TP/FP/FN, both final-test confusion matrices from their recorded counts, and the bounded validation-versus-test comparison.

The default notebook does **not** rerun the final-test evaluation (`default_notebook_reruns_final_test: false`, `models_invoked: 0`, `metrics_recomputed: 0`, `holdout_content_accessed: false`). The holdout was read once and is spent, so no executable evaluation path is offered for it; none is offered for validation either (`executable_validation_evaluation_path: false`), because that would need the materialised dataset and the frozen checkpoints.

## Mode B carry-forward

**MODE_B_REVALIDATION_NOT_REQUIRED.** Phase 15C re-validated Mode A only. Every one of the 25 Mode B execution inputs is byte-identical between `8ce5d0375903e3e3760873e0a75ddce37cc1149b` and `958eefaa58b785ff2f9eb4dd2751efc26dd9b09e`, so the phase 14A inference evidence still describes the code that would run. This is derived by comparing Git blobs, not asserted. Changed paths: none. The notebook is the one file phase 15C changed, so it is compared separately by notebook_executable_identity, which measures whether any executable cell moved..

The notebook itself is measured separately: **NOTEBOOK_EXECUTABLE_SURFACE_UNCHANGED**. Its 15 code cells are compared as whole objects (`FULL_CODE_CELL_OBJECTS_INCLUDING_METADATA`) against the phase 14C validated notebook `2e355d8012b6f1f49923c5cb3ead06760490f024`; changed code cells: none. Phase 15C is therefore `MARKDOWN_ONLY`, growing the Markdown from 14 to 20 cells while executing identically. This matters because a Mode A run never enters the Mode B cells, so a Mode A pass alone could not validate them. This proof is relative to the phase 14C validated notebook. The notebook's code cells did change between the phase 14A Mode B revision and phase 14C; that pre-existing, phase 14C-disclosed distance is unchanged by phase 15C, not closed.

## LOCAL_IMPLEMENTATION_VERIFICATION

PASS: notebook safety, source provenance, publication scan and execution-surface identity checks. No models run and no holdout access during finalization.

```json
{
  "focused_tests": {
    "failed": 0,
    "files": [
      "tests/test_delivery_demo.py",
      "tests/test_delivery_academic.py",
      "tests/test_academic_colab_delivery.py",
      "tests/test_public_delivery.py",
      "tests/test_academic_report.py"
    ],
    "passed": 98
  },
  "note": "Executed on the phase 15C finalization candidate. The metadata-only I/O guard blocked real local datasets and checkpoints; selected integration checks were skipped. The records were first generated from a provisional quality record, the gates were then measured against the resulting tree, and the records were regenerated with those measured numbers.",
  "pytest": {
    "failed": 0,
    "mode": "metadata-only",
    "passed": 2837,
    "skipped": 47,
    "warnings": 27
  },
  "ruff_check": "PASS",
  "ruff_format": "PASS",
  "ruff_formatted_files": 273,
  "scope": "FINALIZATION_CANDIDATE"
}
```

## REAL_COLAB_MODE_A_VALIDATION

**ARTIFACT_ONLY_COLAB_CLOUD_EXECUTION: PASS** - executed by the human maintainer on 2026-09-18, fresh runtime, accelerator `CPU`, Run all, no traceback.
A fresh Colab run required no GPU, checkpoints, Roboflow key, holdout authorization or inference. It rendered:

- every new pedagogical Markdown section;
- TREINO / TRAINING with the recorded D2 and S1 evidence;
- AVALIACAO / EVALUATION with the committed final-test evidence;
- INFERENCIA / INFERENCE present, with the checkpoint and video path inactive;

No training was executed and no final-test evaluation was rerun (`training_executed: false`, `final_test_evaluation_executed: false`).

The human visually confirmed every new pedagogical section:

- 3.1 models and selection;
- 3.2 training hyperparameters;
- 3.3 augmentation;
- 4.1 evaluation algorithms;
- 4.2 operating points;
- 7.1 inference pipeline;

| Human readability review (`PHASE_15C_ATTESTED_ITEMS_ONLY`) | Result |
|---|---|
| pedagogical markdown | PASS |
| training section | PASS |
| evaluation section | PASS |

The wider phase 14C readability review - tables, training curves, evaluation tables, confusion matrices, qualitative figures and video evidence - is preserved in the structured report and is not restated here as a phase 15C observation.

Verbatim lines attested for phase 15C (`PHASE_15C_VERBATIM_ATTESTED_LINES_ONLY`):

```text
Repository revision: 958eefaa58b785ff2f9eb4dd2751efc26dd9b09e
Mode B installation skipped.
Video upload and inference skipped.
Mode A complete. No inference output was created.
```

## REAL_COLAB_MODE_B_VALIDATION

**MODE_B_COLAB_OPTIONAL_INFERENCE_VALIDATION: PASS** - executed by the human maintainer on 2026-09-17 at `8ce5d0375903e3e3760873e0a75ddce37cc1149b`. It was **not** re-executed for phase 14C or phase 15C; see the carry-forward proofs above (`MODE_B_EXECUTION_SURFACE_BYTE_IDENTICAL`).

| Runtime | Observed value |
|---|---|
| gpu | Tesla T4 |
| cuda_available | True |
| pytorch_cuda | 12.8 |
| pytorch | 2.11.0+cu128 |
| python | 3.12.3 |
| ultralytics | 8.4.138 |
| opencv | 5.0.0 |
| mode | compare |
| requested_device | cuda |

Both checkpoints were VERIFIED_BEFORE_DESERIALIZATION after manual upload.

| Checkpoint | Bytes | SHA-256 |
|---|---:|---|
| D2 | 5502289 | `0466f872a9de22898c70d8834cf1fcfb3d77f6c9fb27e81ee6248d3d19cbc206` |
| S1 | 6041685 | `29337d671459d0f742fb713613cc41d0eafcb8a21f5846ac0da65b2ffc024f20` |

The external input is a five-second, 640x360, 25 FPS excerpt (125 frames) from the approved Phase 13B real-world construction source. Output is 1280x360, with D2 left and S1 right; both models completed 125 frames.

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| input | 2068541 | `b00bbd908a2e7a0e4b40073200bab555ad9f87be4e8c3ddcbd0f994fbd32532b` |
| output | 8901739 | `12df710d54828195e7e79d2579de84865654a853141219dac40548b001448c69` |

Downloaded output size/hash, full decoding and layout were verified locally; no model inference was repeated. Media remains ignored and unpublished.

## Source and licensing

**COLAB_TECHNICAL_VALIDATION_ARTIFACT / NOT_PUBLICATION_READY.**

The original generated provenance retains `SOURCE_LICENSE_UNSPECIFIED: not ready for publication` and null attribution. Its exact UTF-8 text, byte identity and parsed content are archived in [the provenance record](academic_colab_delivery.provenance.json). It has not been rewritten.
Independent preparation evidence identifies Frank Vincentz / CC BY-SA 3.0. That context does not change the original warning or authorize publication of this five-second output.
Code licensing, checkpoint redistribution, dataset licensing and external-video licensing remain separate. See [licensing](../delivery/LICENSING.md).

## Canonical notebook and validation scope

Notebook: `notebooks/construction_safety_vision_demo.ipynb`; 34771 bytes; SHA-256 `5c92ff84289cf2a47cc5f5164826a099a5c43bf665ff8003f05bf254dc625dea`.
The canonical file still clones `main`, defaults to `RUN_INFERENCE=False`, and contains no saved outputs. It was not changed to the temporary branch locally.
In each Colab session the human changed the clone branch, so the canonical notebook was not executed verbatim, and the final candidate commit itself was not cloud executed. The phase 14A session additionally added a separate GPU diagnostic cell; that cell was never committed and is absent from every canonical notebook, so a stale saved copy of it is a session artifact rather than a delivery defect.
Finalization relies on that disclosed cloud execution and unchanged execution-critical bytes.

## Execution-surface identities

Each identity is the exact Git blob content at the validated revision. All candidate files must match those raw bytes; Windows CRLF conversion also fails this check. Any execution-critical change stops finalization.

| Path | Git blob OID | Bytes | SHA-256 |
|---|---|---:|---|
| `.python-version` | `e4fba2183587225f216eeada4c78dfab6b2e65f5` | 5 | `7b55f8e67b5623c4bef3fa691288da9437d79d3aba156de48d481db32ac7d16d` |
| `configs/detector_segmenter_comparison.yaml` | `5f10336c22e9ae6f3ab7f1a7f4e3fd0f07bf0a00` | 17494 | `50c113208402986881223eec068f8bb52655212f68f593c37457ed618b539c25` |
| `delivery/checkpoints.json` | `da88c6bc85ac7e063efe99ee2fa6f209a5004ffe` | 1295 | `8595a00d618fc57ee79b1342c31b872fe3892572dfccea9ab07a1fafc7c66ce1` |
| `notebooks/construction_safety_vision_demo.ipynb` | `703b6effe3cb3688fb523cb93edf93a6f0d09dac` | 34771 | `5c92ff84289cf2a47cc5f5164826a099a5c43bf665ff8003f05bf254dc625dea` |
| `pyproject.toml` | `0a9dbefcda73bb4827b8be74626ff919ee1c02d0` | 5405 | `4d45c1d8d20839311e0be589f1b60bda92ab1902bc942955e79fd660ae3f2cd8` |
| `reports/canonical_modeling_manifest.json` | `4760c62ff980d3b7643729d4d2b7797a746aa69d` | 6531 | `26ffea257c28630d8d5fb4ca8c310b5eb11f22ad85a1dbb4acea4090b5b24f8f` |
| `reports/detection_D2_manifest.json` | `cd0e27574f8f43590bde3b55239e31f048dd6f24` | 17485 | `c5777819a405c7ac0133c5c2bd30dfeac77eca68080b647254dd52a6204ee6ef` |
| `reports/detector_segmenter_box_comparison.json` | `9f74ea027d27f84656fe901032255234c948c30f` | 9217 | `ebeabe3f6bf87e2059446d21bba7ae61cbd717a2b38b4b6eba90fbe90ec76704` |
| `reports/figures/detection/D2/results.png` | `14dd774aa3b8ac1595a2a0a5827c470743461c1c` | 322982 | `71e1eb86df79f8d3f1accbe26a7f105002aa0b26add4bce24cb368b99f8c3f5c` |
| `reports/figures/final_video/frame_000436.jpg` | `46cfdaf1fbd2a3652ef149603f9c585fb8c6dbb7` | 1315142 | `369673ec53dd1c9102cfb98fc4152cafac22ce7ed19dafeede80fe647101117a` |
| `reports/figures/final_video/frame_000448.jpg` | `d93d626b09d3ed6869c126ebac19b1a3690f5bd2` | 1320946 | `c5dcd63245d231eed2204ffd6ba1066153542daa9df656a865ef2b446b17ab5a` |
| `reports/figures/final_video/frame_001306.jpg` | `f717b9267605ee36bff82b241a1a2d5d05c37b78` | 1304521 | `5fc7588cfe0c4f1c8912888cfbc5a6d914935467bd41b44af93d3c89d9671c49` |
| `reports/figures/final_video/frame_001318.jpg` | `bebaaaabc6b560c615ea92af0956578f8ffcd7b1` | 1335474 | `8fda3c1055b1961a219086b3b1855d07ea61682492963982f1fcacd90789a781` |
| `reports/figures/final_video/frame_002178.jpg` | `2ee8b7b9b5e79db5e9bb3a4c321275ac6a8ee7da` | 1295065 | `bb3f0782c8af6b0569483a57078f2ed668da94e49c9c92a3c2a813b283aecd83` |
| `reports/figures/final_video/frame_002190.jpg` | `8a41567dceb469078e3625c6ac41f6b33e720aa1` | 1301210 | `c7033acf7aba5756f36d507e1fca1ee8d7f401989e18a8df7931ac90aff84aae` |
| `reports/figures/qualitative/box_vs_mask_hero_candidate.png` | `5bd64252d093dad1cbe8e8c753b1fd8c8fc778f9` | 1673971 | `0cd046500e88b339e72625f3f4dc058f6f601e0001d8e95ecc57d1c8142c825f` |
| `reports/figures/qualitative/mask_quality_gallery.png` | `399aa972942b98b0a0740a74113539c24d24e26e` | 4026207 | `98a38207402202f1d01b5a22e1477d01022c57cf5278b8781e0a58f4b818c35e` |
| `reports/figures/qualitative/validation_fp_fn_d2.png` | `1f1b7907ddbf8765c7252bcaa9edc780668bf8ad` | 3885900 | `6ce6dc3e2efca38693a6a9aca19b84641a45bb67b0eb1b03e3806e0d9e28288d` |
| `reports/figures/qualitative/validation_fp_fn_s1.png` | `53615ff7b2cd9fa6be21643c4d2c0febbe4bd787` | 4569947 | `0cf28e35e88ea450e6ddf179c15ecd718cb5226041ad41187fe09f4991a3f362` |
| `reports/figures/segmentation_S1/results.png` | `adfbe111d9b78e8388c30e6f96172177abb12133` | 498789 | `d182686b8adf68b837e03980c8dbc30c26204e6f4f5d263b9cffd5010091d8e3` |
| `reports/final_detector_manifest.json` | `255fe5dbe650a159b8be09a853e46559a48b183a` | 17220 | `97c1ecaf6fdd8c7437cd063a233a3959fc7144e9f69a2c966d5b452d4bbfc205` |
| `reports/final_real_video_demo.json` | `12650879c9396b7823a8f5bbfbe35464c09d4053` | 21944 | `30330b7b01c0ba21de5b41c26d8bf4bfb0ffc228ac35990b360c60251e030e80` |
| `reports/final_segmenter_manifest.json` | `d18a11471656ec636471a42d795289c450b721d3` | 17536 | `dbdc63211c73ceb0f6ba78e493665fb8fc5432697164466e61b2c086d66acbeb` |
| `reports/final_test_detector.json` | `62e36c96cc7838e0a781bf5b9abf0be4f8a80f44` | 11248 | `2169a5317324a299123cec5109b5d346ab44a2398e57b29d0aee19d172967bd9` |
| `reports/final_test_direct_iou.json` | `ab16bf12dbd8702e6051369a3e555944675b7511` | 4656 | `0de72ae9769e0e6d3be9eb23ff52f4a489f85171d2af9d1dfacf96b3282b289c` |
| `reports/final_test_segmenter.json` | `e21130bf848644a521a556459cbcb55f1a01fa9e` | 15697 | `c77ae3ce17cff8c17a8d8de87ccc04b08d276b98c0b3096dfcfc403f2c7c7093` |
| `reports/qualitative_validation_gallery.json` | `b813783acaf911f85298ce7f86857020e76e3e24` | 487555 | `de457afb47baba86ecef5a486294a8db6891d47c143ec98ea9d547eab8826b9c` |
| `reports/qualitative_validation_gallery.provenance.json` | `9afd8196251887d68c5b46e2e4f4dd5af8013191` | 25668 | `af208eb4d7f813ef0470b0ef0bc835a23379fe2a114bf6c5fab915b4654579cb` |
| `reports/segmentation_S1_canonical_evaluation.json` | `ec9083b9e635bbb48f65849373d2ec0bd068e525` | 5102 | `edafc31a0cec9a8a8019ea18156e398d1c4c619fa9de485ccb84163e44b818a1` |
| `reports/segmentation_S1_mask_iou.json` | `37869acfd7993e48564545b600efe46a0927c9c3` | 5517 | `be3b893b602ddfe3d589c46676e599f33bc9abd813f76b1c2d3855d0d8dd149d` |
| `reports/segmentation_S1_result_manifest.json` | `94da29b9c17e31e8ecd956f54ed796728ce0207c` | 36791 | `7fe12ea2515fb2821fc7e5a6e8914ec4929393937c5234b22b817e50b3521723` |
| `scripts/run_video_demo.py` | `225b0b448d82a69b337b05fd1cfec1b04b38c7b3` | 2319 | `9bf52de2f8d6d28370411599bb90d94a37246e7fdd1f2450f90c15371b51dcee` |
| `src/construction_safety_vision/__init__.py` | `5032ca1b27e43b37d9335b1ff83aa7710fb338ef` | 1730 | `b9d8d0fe5e371fa90fd70c7e3888574670084b5ec11a85c16c73a7cb99005325` |
| `src/construction_safety_vision/checkpoint_delivery.py` | `ae5662d9f8fef6b1a58d55d69fd41788c4bc3352` | 5662 | `4394e10c10bc4503fef1de55d5b13e1952cf56962fcac8e304f7db84b36c4aa1` |
| `src/construction_safety_vision/config.py` | `c088b8ad247a45191a9c2b9d5fab9457992b35f5` | 12531 | `f4cfde636f30a7c87408ce00e8f5399db7f679e0f94fecaa89292f34c3026eeb` |
| `src/construction_safety_vision/data/__init__.py` | `2f46cc106334aa5c4e6a83947a35d3a0063208da` | 2433 | `ce7bfd4521636a75d001a6f5ef5fdf8f8c8d85a67cdc08d73cda3d1d56fbac78` |
| `src/construction_safety_vision/data/acquisition.py` | `7f7942825ef76b958d4875cdc8f1823c6b368e01` | 11211 | `893b0e80c2e2d75342e220e4fcadb26f52660ad4113d07855a05851b3a006ebc` |
| `src/construction_safety_vision/data/canonical.py` | `23817c3f1cb1076cd837b0098e59da0b21136b1d` | 5430 | `2f730ebed543fc388b0b8fa530fba0aff711357591106dc2d953397d9fb38464` |
| `src/construction_safety_vision/data/coco.py` | `e9cddceff80b87fb722ccca761afa8a0f0086b15` | 10752 | `1c3f351ff170de3a68430e7b8f6eb8d00e67099a886af270a0287401f8695ec6` |
| `src/construction_safety_vision/data/roboflow.py` | `89134b7e67a883b80ae68952a14414347c5fd054` | 17564 | `c245bd1fb8b4452049ab038c22a9d18ff30c0ce89b8c3ca45987355480fad52f` |
| `src/construction_safety_vision/data/versioning.py` | `ba13f4e4886353ed7b4d1549ecabfe8d969870f6` | 3710 | `607e3def348aaecdbbe77a6e1ed9e4e82d528832209ebc851de577aee35cf60b` |
| `src/construction_safety_vision/delivery_academic.py` | `2bd188a79a80269f3bc9cc1765e4d1945daf1457` | 32232 | `03a35ed044e6abcae350b15a5afb9bc1e130293ed8c690dcff63a82043240738` |
| `src/construction_safety_vision/delivery_demo.py` | `e98f611377a04831cbdf62e4aff27136a6f29b0e` | 19970 | `be983e8adb4e3a4f6b1b6ec683534e8a20311da9ea561bb0ef4591d16650e094` |
| `src/construction_safety_vision/detection_freeze.py` | `2c604ef0e778b5cc825b82bff9df147912d8d3f0` | 21987 | `9a54dab456e013e27d10bd7a5d6af5a62723a31e17ac824d71ee88fc47bf34ec` |
| `src/construction_safety_vision/env.py` | `33ca4367f00785443d070cd4886ca8923c9e229a` | 3382 | `13436ec33e1125abd879187f945254aa911baed2dfd16245b6081a3e178daa21` |
| `src/construction_safety_vision/paths.py` | `5dfc522001f30e9fe5f366cdff8217d62bab1540` | 6775 | `e55ce08bed78643798707a61bc90cafd7434d71fb3df844b25689e2325dc952e` |
| `src/construction_safety_vision/provenance.py` | `265b264a19c41e9072180cd0db2eaaf6293b3e0f` | 7794 | `f627dea50f470226954d87f970d9e5c9d98dc3da2aa390f7fb450ad44fb55412` |
| `src/construction_safety_vision/segmentation_freeze.py` | `0983afe39fafe5cc64a89c3dc1115f9d67208a1f` | 26829 | `dc3e439916b7eb0286434bf68012f7dd878662947a98e5e447ed082c2f81006f` |
| `src/construction_safety_vision/splits.py` | `4d8ff4d1a591903477578556234363499b4d9d7b` | 4789 | `54d173c439d1d440735e13cde0dd064e8197715ec35429856a7857170a2eb65f` |
| `src/construction_safety_vision/video_models.py` | `194d26a769833958532be6632534836f3c9730e7` | 9492 | `57c215d737993ccb0decd3b7f6869088254f06dfddcf84a4e43eca8da15e90ed` |
| `src/construction_safety_vision/video_render.py` | `45b878ee10001c9c775c6c027fd1295568185cb9` | 3963 | `0646971beb4df488e56fb316ec7cc22edec15ab2b87ee914428e0c6d47632187` |
| `src/construction_safety_vision/video_runtime.py` | `9cb31f9ab9fa04e5af2471528cb987b345c751d5` | 15516 | `77e7f21e48630eeeb0e76f4e7fe453c5f73b0042c4930a4c89b2ee667b2c3e21` |
| `uv.lock` | `055f68624372f3f762b827a40a5677a5f772b5fb` | 377798 | `b50926982203f5cc5fb37e971ada6efebf78ce812fffaad9aabf49b5582353e9` |

## Delivery status and publication boundary

**GAP-004: OPEN -> RESOLVED. R18 executable Colab: COMPLETE.**

The missing executable-notebook requirement (R18/C6) is fulfilled by artifact-only Run all and optional frozen-model external-video inference. The historical GAP-004 recommended fix mentioned metric reproduction; the approved closed-science delivery scope displays committed metrics and does not recompute them. The Phase 12A snapshot is unchanged. Broader clean-room reproduction remains GAP-014 OPEN.

GAP-008 remains OPEN; GAP-010 remains PARTIALLY_RESOLVED; GAP-014 remains OPEN.
The [canonical Open in Colab target](https://colab.research.google.com/github/Novachrono117/Construction-Safety-Vision-PPE-Detection-Instance-Segmentation-Video-Analytics/blob/main/notebooks/construction_safety_vision_demo.ipynb) points to `main`. Its URL is structurally valid and the repository is public; final notebook availability on main requires a separately authorized push. This phase creates a local final commit only.
No GitHub Release or full-video publication is part of this phase. The temporary validation branch remains evidence and is not merged or deleted.

## Known limitations

- The final candidate commit itself has not been run in Colab; surface identities are preserved.
- Mode B was validated once, at the phase 14A revision, and was not re-executed for phase 14C or phase 15C; the carry-forward rests on byte identity, not a second run.
- Phase 15C adds no distance from that Mode B evidence, because it changed no code cell. The distance the phase 14C notebook change introduced is disclosed and unchanged here, not closed.
- The training and evaluation sections present recorded evidence. The notebook does not retrain the models and does not rerun the spent holdout evaluation, and it offers no executable path that would.
- Reproducing a training run needs a CUDA GPU and the materialised dataset, neither of which the notebook provides.
- Mode B still requires manual uploads and compatible GPU capacity; future Colab changes may matter.
- The five-second output is technical evidence only and must not be published in this phase.
- MP4/mp4v browser playback is not guaranteed; no audio/timestamp track is copied.
- Inference timing is a demo measurement, not a controlled scientific benchmark.
- Checkpoint redistribution remains unresolved; licensing differs across code, data and media.
- The complete Phase 13B MP4 has no public URL; six approved frames keep Mode A useful.
- No Release, technical report, tracking, pitch or broader README professionalization is included.
