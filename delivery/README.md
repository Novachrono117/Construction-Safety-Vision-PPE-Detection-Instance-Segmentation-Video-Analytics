# Delivery entrypoint

**Phase 13A: video runtime foundation complete.** Scientific training and
final evaluation are complete. D2 and S1 are frozen; selection and tuning are
closed. Presentation deliverables remain incomplete.

## Current evidence

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
| [Video runtime report](../reports/video_runtime_foundation.md) | Phase 13A engineering checks and limitations |
| [Reports index](../reports/README.md) | Protocols, results, provenance and historical records |

## Delivery paths

- [Quickstart and research reproduction](REPRODUCTION.md): actual commands and
  explicit missing prerequisites. Public checkpoint retrieval remains pending.
- [Video runtime](VIDEO.md): detector, segmenter and compare CLI modes, tested on
  synthetic fixtures with explicit CUDA/CPU; the final real video is pending.
- [Licensing and checkpoint distribution](LICENSING.md): GNU AGPL-3.0 for the
  project source/repository; CC BY 4.0 for dataset material. Dependencies retain
  their own licenses; D2/S1 distribution still requires review.
- [Public checkpoint metadata](checkpoints.json): generated from the two final
  freeze manifests, without a download locator until one exists.
- [Academic GenAI disclosure](AI_USAGE.md).
- [Publication recommendations](PUBLICATION.md): a plan, no settings changed.

The full recruiter README, real video, academic report, Colab and pitch are
pending. The validation gallery is complete. No clean-room reproduction has been
demonstrated.
