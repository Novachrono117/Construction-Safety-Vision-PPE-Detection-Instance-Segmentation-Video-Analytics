# Delivery entrypoint

**Phase 14A: executable academic Colab complete.** Mode A and optional Mode B
passed human-executed cloud validation; final publication on `main` is pending.
Scientific training and final evaluation are complete. D2 and S1 are frozen;
selection and tuning are closed. The overall submission package remains incomplete.

## Current evidence

The [two-mode Colab notebook](../notebooks/construction_safety_vision_demo.ipynb)
provides evidence-only Mode A and optional checkpoint-upload inference Mode B.
Both modes passed at revision `8ce5d0375903e3e3760873e0a75ddce37cc1149b`;
Mode B used a Tesla T4. See [setup and boundaries](COLAB.md) and the
[cloud validation record](../reports/academic_colab_delivery.md).

| Read this | Role |
| --- | --- |
| [Scientific synthesis](../reports/detector_segmenter_scientific_synthesis.md) | Current answer to the research question; validation and local benchmark evidence |
| [Final detector](../reports/final_detector_manifest.json) | Authoritative D2 freeze and identity |
| [Final segmenter](../reports/final_segmenter_manifest.json) | Authoritative S1 freeze and identity |
| [Final test report](../reports/final_test_evaluation.md) | Published results of the completed final evaluation |
| [Execution accounting](../reports/final_test_execution_accounting.md) | One attempt, one holdout read, two models, three protocol-defined inference passes |
| [Latency report](../reports/detector_segmenter_latency_report.md) | Controlled local hardware benchmark; not video FPS |
| [Repository audit](../reports/final_repository_audit.md) | Immutable Phase 12A snapshot |
| [Assignment matrix](../reports/assignment_compliance_matrix.csv) | Immutable Phase 12A compliance snapshot |
| [Live delivery tracker](../reports/delivery_gap_resolution_status.json) | Current delivery resolutions and requirement updates |
| [Phase 12B provenance](../reports/public_delivery.provenance.json) | Historical source, code and generated metadata hashes |
| [Validation gallery](../reports/qualitative_validation_gallery.md) | Completed Phase 12C FP/FN and mask evidence |
| [Real-video demonstration](../reports/final_real_video_demo.md) | 104.52-second output, rights, temporal observations and screenshots; public delivery pending |
| [Video runtime report](../reports/video_runtime_foundation.md) | Phase 13A engineering checks and limitations |
| [Academic Colab delivery](../reports/academic_colab_delivery.md) | Separate local checks, real Mode A and real Mode B validation; execution identities and publication limits |
| [Reports index](../reports/README.md) | Protocols, results, provenance and historical records |

## Delivery paths

- [Quickstart and research reproduction](REPRODUCTION.md): actual commands and
  explicit missing prerequisites. Public checkpoint retrieval remains pending.
- [Video runtime](VIDEO.md): detector, segmenter and compare CLI modes, tested on
  synthetic fixtures with explicit CUDA/CPU and a full real clip on CUDA. Local
  delivery instructions and attribution accompany the final-video report.
- [Licensing and checkpoint distribution](LICENSING.md): GNU AGPL-3.0 for the
  project source/repository; CC BY 4.0 for dataset material. Dependencies retain
  their own licenses; D2/S1 distribution still requires review.
- [Public checkpoint metadata](checkpoints.json): generated from the two final
  freeze manifests, without a download locator until one exists.
- [Academic GenAI disclosure](AI_USAGE.md).
- [Publication recommendations](PUBLICATION.md): a plan, no settings changed.

The full recruiter README, academic technical report, pitch and public video
distribution are pending. The academic Colab, real-video deliverable and
validation gallery are complete. Checkpoints still require manual upload
(GAP-008 OPEN); public demo distribution remains partial (GAP-010
PARTIALLY_RESOLVED). No full clean-room research reproduction has been
demonstrated (GAP-014 OPEN).
