# Reports index

Start with the final evidence below. Files remain at their original paths.
**Current** means authoritative for the named subject; a historical protocol
can still govern its frozen result. A protocol's pre-run wording is not current
project status. Scientific records and the Phase 12A snapshot are immutable.

## START HERE / FINAL RESULTS

- **Current synthesis:** [Scientific answer and bounded claims](detector_segmenter_scientific_synthesis.md).
- **Current result:** [Final test evaluation](final_test_evaluation.md).
- **Current result:** [Validation comparison](detector_segmenter_validation_comparison.md).
- **Current result:** [Controlled latency and memory benchmark](detector_segmenter_latency_report.md).
- **Current identities:** [D2 freeze](final_detector_manifest.json), [S1 freeze](final_segmenter_manifest.json).
- **Current delivery status:** [Live gap tracker](delivery_gap_resolution_status.json), [delivery entrypoint](../delivery/README.md).

## DATASET & PROVENANCE

- **Provenance:** [Source, license and acquisition](dataset_provenance.md).
- **Historical audit:** [Dataset audit](dataset_audit_report.md), [EDA](eda_report.md), [manual review](manual_audit_report.md).
- **Current canonical decision:** [Annotation decision](canonical_annotation_decision.md), [manifest](canonical_annotation_manifest.json).
- **Current population:** [Modeling population](canonical_modeling_population_report.md).
- **Historical/superseded interpretation:** [Annotation drift](annotation_drift_report.md) and [fragment-rule analysis](fragment_rule_report.md). The later canonical population decision governs retention; changed annotations are not error ground truth.

## SPLIT & TASK MATERIALIZATION

- **Current frozen protocol/result:** [Split freeze report](split_freeze_report.md), [provenance](split_freeze.provenance.json).
- **Current derived development views:** [Task materialization](task_materialization_report.md), [manifest](task_dataset_manifest.json).
- **Historical search:** [Candidate report](split_candidate_report.md). Provisional candidates are not alternative current splits.

Frozen membership is committed for auditability, but must be accessed through
the guarded API. Phase 11 result/report artifacts omit test IDs. No holdout
imagery or bulk predictions are published; the spent holdout stays locked.

## DETECTION

- **Current frozen result:** [Selection report](detection_selection_report.md), [D2 manifest](final_detector_manifest.json), [freeze provenance](final_detector.provenance.json).
- **Protocol:** [Comparison policy](detection_comparison_policy.md).
- **Historical results:** [D0](detection_D0_report.md), [D1](detection_D1_report.md), [D2](detection_D2_report.md).
- **Adapter/runtime provenance:** [Detection adapter](detection_adapter_report.md), [runtime report](detection_runtime_report.md).

## SEGMENTATION

- **Current frozen result:** [Selection report](segmentation_selection_report.md), [S1 manifest](final_segmenter_manifest.json), [freeze provenance](final_segmenter.provenance.json).
- **Protocols:** [S0](segmentation_S0_protocol.md), [S1](segmentation_S1_protocol.md), [canonical comparison](segmentation_comparison_policy.md).
- **Historical results:** [S0](segmentation_S0_report.md), [S1](segmentation_S1_report.md), [S0 error analysis](segmentation_S0_error_analysis.md).
- **Historical fidelity result:** [Adapter fidelity](segmentation_adapter_fidelity_report.md); **subsequent decision:** [adapter approval](segmentation_adapter_approval.json). The audit bytes are immutable; the later approval establishes training eligibility.

## DETECTOR-vs-SEGMENTER COMPARISON

- **Frozen protocol:** [Phase 10A](detector_segmenter_comparison_protocol.md).
- **Current result:** [Recognition/spatial comparison](detector_segmenter_validation_comparison.md), [corrected spatial artifact](detector_segmenter_spatial_comparison.json).
- **Correction provenance:** [Association taxonomy correction](association_taxonomy_correction.provenance.json). The pre-correction spatial fingerprint is superseded.
- **Current cost result:** [Latency report](detector_segmenter_latency_report.md), [memory artifact](detector_segmenter_memory_comparison.json).
- **Current synthesis/result:** [Scientific synthesis](detector_segmenter_scientific_synthesis.md), [claim register](detector_segmenter_scientific_synthesis.json), [provenance](detector_segmenter_synthesis.provenance.json).

## FINAL HOLDOUT

- **Historical frozen protocol:** [Phase 11A](final_holdout_evaluation_protocol.md).
- **Current published results:** [Final test](final_test_evaluation.md), [detector](final_test_detector.json), [segmenter](final_test_segmenter.json), [direct IoU](final_test_direct_iou.json), [per-class table](final_test_per_class.csv).
- **Provenance/accounting:** [Evaluation provenance](final_test_evaluation.provenance.json), [execution clarification](final_test_execution_accounting.md).
- The evaluation is complete: one attempt, one read and three protocol-defined
  inference passes over two frozen models. These links are reports, not rerun instructions.

## DELIVERY / AUDITS

- **Historical Phase 12A snapshot:** [Audit](final_repository_audit.md), [machine-readable audit](final_repository_audit.json), [gap register](final_delivery_gap_register.csv), [assignment matrix](assignment_compliance_matrix.csv).
- **Current delivery follow-up:** [Gap resolution status](delivery_gap_resolution_status.json).
- **Delivery provenance:** [Generation record](public_delivery.provenance.json).
- **Current navigation:** [Delivery entrypoint](../delivery/README.md), [reproduction scaffold](../delivery/REPRODUCTION.md), [GenAI disclosure](../delivery/AI_USAGE.md), [licensing](../delivery/LICENSING.md).

## HISTORICAL SCIENTIFIC RECORD

- [Original proposal](phase1_proposal.md) and [rubric contract](rubric_contract.md)
  distinguish planned deliverables from delivered evidence.
- [Roadmap](roadmap.md) is live status plus an explicitly dated historical log.
- Older reports' statements about pending training, evaluation or licensing
  describe their publication date. They do not override final results or the
  delivery tracker. Superseded findings remain visible for auditability.
- `*.provenance.json` records sources, configurations and hashes. Experiment
  manifests identify runs; freeze manifests identify final models. No new metric
  is inferred from a file name or a phase number.
