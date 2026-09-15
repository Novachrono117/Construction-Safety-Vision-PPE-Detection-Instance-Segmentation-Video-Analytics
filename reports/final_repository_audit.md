# Final repository, academic and portfolio audit

**Phase 12A. `FINAL_REPOSITORY_AUDIT_COMPLETE`. Audit only.**

This phase executed no model, read no holdout content, recomputed no metric and changed no scientific result. It measures the repository and judges its delivery readiness; the two are kept apart, and the measurements re-derive on every run so the audit cannot itself go stale.

## 1. Scientific state

|  | Value |
| --- | --- |
| Final test | `FINAL_TEST_OBSERVED` |
| Holdout | `OBSERVED_ONCE_FINAL` |
| Holdout reads | 1 of 1 permitted |
| Evaluation attempts | 1 |
| Models executed / inference passes | 2 / 3 |
| `CSVISION_ALLOW_TEST_SPLIT` in process | absent |
| Model selection | `CLOSED` |
| Hyperparameter tuning | `CLOSED` |
| Threshold tuning | `CLOSED` |
| Data cleaning for performance | `CLOSED` |
| Final detector | D2 / YOLO11n / imgsz 768 |
| Final segmenter | S1 / YOLO11n-seg / imgsz 768 / overlap_mask false |

## 2. Repository inventory

**416 tracked files.**

| Area | Files |
| --- | --- |
| `academic/` | 0 |
| `configs/` | 16 |
| `data/` | 6 |
| `docs/` | 0 |
| `notebooks/` | 1 |
| `portfolio/` | 0 |
| `reports/` | 221 |
| `scripts/` | 45 |
| `src/` | 54 |
| `tests/` | 65 |
| root files | 8 |

`reports/` breaks down as 140 flat documents, 73 figures and 8 split-candidate tables, including 30 provenance records. Absent areas: `academic/`, `docs/`, `portfolio/`.

### 2.1 Reconciliation against the commit this phase started from

The audit counts its own outputs. The five artifacts it writes are **not** the number of files the phase adds, so both quantities are measured and named apart. Baseline `f1c78f25b805`.

| Quantity | Value |
| --- | --- |
| Tracked files at parent | 407 |
| Tracked files at phase 12A head | 416 |
| Delta | +9 |
| Files added | 9 |
| Files deleted | 0 |
| Live documents modified | 3 |
| of the additions: audit outputs | 5 |
| of the additions: implementation support | 4 |

407 + 9 - 0 = 416. Added and deleted are a difference of path sets, so a rename would appear as one addition plus one deletion; the deleted set is empty, so none occurred.

Added (9):

- `reports/assignment_compliance_matrix.csv`
- `reports/final_delivery_gap_register.csv`
- `reports/final_repository_audit.json`
- `reports/final_repository_audit.md`
- `reports/final_repository_audit.provenance.json`
- `scripts/audit_delivery_readiness.py`
- `src/construction_safety_vision/delivery_audit.py`
- `src/construction_safety_vision/delivery_audit_findings.py`
- `tests/test_delivery_audit.py`

Modified (3):

- `CLAUDE.md`
- `README.md`
- `reports/roadmap.md`

Summary: 12 requirements COMPLETE, 3 PARTIAL, 5 MISSING; 7 P0 gaps, 7 P1, 5 P2, 3 P3; 8 stale-claim probes currently firing.

## 3. Assignment compliance

| ID | Requirement | Criterion | State | Evidence | Remaining action |
| --- | --- | --- | --- | --- | --- |
| R01 | Dataset of at least 300 annotated images | C1 | **COMPLETE** | `reports/canonical_annotation_manifest.json` | none |
| R02 | Public-source citation and license | C1 | **COMPLETE** | `reports/dataset_provenance.md` | none |
| R03 | Train / validation / test split | C1 | **COMPLETE** | `reports/split_manifest.json` | none |
| R04 | Object detection fine-tuning | C2 | **COMPLETE** | `reports/final_detector_manifest.json` | none |
| R05 | Instance segmentation in the same domain and splits | C3 | **COMPLETE** | `reports/final_segmenter_manifest.json` | none |
| R06 | Complete hyperparameter documentation | C2/C3 | **COMPLETE** | `configs/` | none |
| R07 | mAP@0.50 | C4 | **COMPLETE** | `reports/final_test_detector.json` | none |
| R08 | mAP@0.50:0.95 | C4 | **COMPLETE** | `reports/final_test_segmenter.json` | none |
| R09 | IoU | C4 | **COMPLETE** | `reports/final_test_direct_iou.json` | none |
| R10 | Precision | C4 | **COMPLETE** | `reports/final_test_detector.json` | none |
| R11 | Recall | C4 | **COMPLETE** | `reports/final_test_segmenter.json` | none |
| R12 | Confusion matrix | C4 | **COMPLETE** | `reports/figures/final_test/` | none |
| R13 | Qualitative false-positive / false-negative analysis per class | C4 | **PARTIAL** | `reports/segmentation_S0_error_analysis.md` | build a per-class FP/FN gallery from validation imagery, one documented false positive and one false negative per class; holdout imagery stays unpublished |
| R14 | Inference on real video of at least 30 seconds | C5 | **MISSING** | `none` | acquire a licensed video and build the inference runtime |
| R15 | Technical report | C6 | **MISSING** | `none` | assemble a 6-10 page report from committed evidence |
| R16 | README that describes the repository honestly | C6 | **PARTIAL** | `README.md` | restructure and retire every stale claim |
| R17 | Reproducible public GitHub repository | C6 | **PARTIAL** | `README.md, pyproject.toml, uv.lock` | add a LICENSE and document the phase 5C.2-11 command path |
| R18 | Executable Colab notebook | C6 | **MISSING** | `none` | one notebook covering dataset, metrics and a runnable demo |
| R19 | 5-8 minute video pitch | C7 | **MISSING** | `none` | script from the phase 10D claim register, then record |
| R20 | GenAI usage declaration | C6 | **MISSING** | `none` | declare the tools, the scope of use and the review performed |
| R21 | Bonus: tracking or interactive demo | B1 | **NOT_APPLICABLE** | `none` | optional, only after every mandatory deliverable exists |

## 4. Result consistency

Every headline is read by field from a committed artifact and then searched for verbatim in the live documentation.

| Headline | Value | Source field | Documented in |
| --- | --- | --- | --- |
| d2_test_box_map50_95 | 0.427031 | `canonical_box.CANONICAL_TEST_BOX_MAP50_95` | README.md, reports/final_test_evaluation.md, reports/roadmap.md |
| d2_test_box_map50 | 0.565260 | `canonical_box.CANONICAL_TEST_BOX_MAP50` | README.md, reports/final_test_evaluation.md, reports/roadmap.md |
| s1_test_mask_map50_95 | 0.410143 | `canonical_mask.CANONICAL_TEST_MASK_MAP50_95` | README.md, reports/final_test_evaluation.md, reports/roadmap.md |
| s1_test_box_map50_95 | 0.433764 | `canonical_box.S1_CANONICAL_TEST_BOX_MAP50_95` | README.md, reports/final_test_evaluation.md, reports/roadmap.md |
| test_matched_mask_iou_mean | 0.834548 | `diagnostic.global.matched_mask_iou_mean` | README.md, reports/final_test_evaluation.md, reports/roadmap.md |
| test_gt_normalized_mask_iou | 0.585551 | `diagnostic.global.gt_normalized_mask_iou` | README.md, reports/final_test_evaluation.md, reports/roadmap.md |
| d2_test_true_positives | 189 | `object_level.true_positives` | README.md, reports/final_test_evaluation.md, reports/roadmap.md |
| s1_test_true_positives | 205 | `object_level.true_positives` | reports/final_test_evaluation.md, reports/roadmap.md |

## 5. Stale claims

| Probe | File | String | Why it is now false |
| --- | --- | --- | --- |
| STALE-01 | `README.md` | `No model has been trained and no evaluation has been run` | Five models were trained (D0, D1, D2, S0, S1), two are frozen, and both the validation comparison and the one-shot holdout evaluation are complete. |
| STALE-02 | `README.md` | `No test metric exists.` | Phase 11B measured the holdout once; the test metrics are committed in reports/final_test_detector.json and reports/final_test_segmenter.json. |
| STALE-03 | `README.md` | `no test number exists` | True when the detector and segmenter were frozen, but stated in the present tense in a status table read as current. |
| STALE-04 | `reports/roadmap.md` | `spatial `9877b88d...`` | The association-taxonomy correction rewrote the section that digest covers. The authoritative spatial fingerprint is 3988bcf6..., recorded in reports/association_taxonomy_correction.provenance.json; CLAUDE.md states the pre-correction value is superseded and must not be quoted. |
| STALE-05 | `README.md` | `**locked holdout**: it has never been` | The holdout was read once in phase 11B; it is spent, not pending, and the same blockquote goes on to quote the resulting test metrics. The sentence also asserts that every number below it is a validation number, which is no longer true. |
| STALE-06 | `README.md` | `## Planned architecture` | The architecture was built, frozen and evaluated. Presenting it as planned understates the work to every reader. |
| STALE-07 | `README.md` | `## Planned methodology` | Eleven of the fourteen planned phases are complete. |
| STALE-08 | `.env.example` | `phase 3 - not used yet` | Phase 3 acquired the dataset; the key is used by four committed scripts. |

## 6. Four perspectives

### Professor - **PROFESSOR_READY_WITH_GAPS**

The scientific core is unusually strong for a graduate assignment: a frozen group-aware split, a canonical annotation source with boxes derived from polygons, predeclared selection policies with decision margins, one external evaluator for both models, and a one-shot holdout with an append-only ledger. Methodology is traceable end to end and limitations are stated rather than smoothed. What is missing is not rigour but three whole deliverables - the video application, the Colab notebook and the pitch - plus a report the evaluator can read without reconstructing phase history.

**Strengths**

- every reported metric traces to a committed artifact and a provenance record
- holdout discipline is enforced in code by a dual gate, not by convention
- detection boxes are derived from segmentation polygons, so both tasks provably describe the same objects
- selection policies were frozen before the candidate experiments ran, with the margin declared in advance and applied mechanically
- limitations are explicit: vest_loose small-sample, UNKNOWN causes, validation-only selection, local-hardware-only latency

**Gaps**

- no technical report exists; the evidence is spread over 135 flat documents
- no video application, which is criterion C5 and 10% of the grade
- no executable Colab notebook, required by C6's definition of done
- no per-class qualitative FP/FN gallery pairing image, prediction and ground truth, required by C4's definition of done
- no GenAI declaration
- the README states 'No model has been trained and no evaluation has been run', which a reader can reasonably take as the project's own summary

*Verdict:* C1 to C4 are substantially satisfied and defensible under scrutiny. C5 and C7 are absent entirely and C6 is partial, so the submission would lose whole criteria rather than marks within them.

### Recruiter - **NOT_RECRUITER_READY**

Thirty seconds is spent on a status blockquote of roughly fifty lines dense with phase numbers, fingerprints and six-decimal metrics. Nothing states what the system does, what it achieved, or what the candidate built, before the scrolling starts. The strongest signals here - holdout discipline, a controlled one-variable experiment, a correct GPU cost benchmark - are genuine senior-engineer signals, and not one of them is visible above the fold. The Results section sits at line 2417 and says 'Not available'.

**Strengths**

- the problem framing is genuinely good: worn versus merely present PPE is a distinction a naive detector cannot make
- the engineering discipline on display is rare in portfolio projects
- there is a real cost/benefit answer - masks add representation and cost about 30% more end-to-end latency - which reads as product thinking
- 2547 automated tests signal a working engineer, not a notebook author

**Gaps**

- no hero visual: no architecture diagram, no annotated result image, no demo
- no result table above the fold, and the Results section says 'Not available'
- the first screen is a phase log, so technical depth is invisible at a glance
- no stack summary, no one-line elevator pitch, no 'what I built' section
- no software license, which reads as unfinished on a public repository
- no runnable demo a reader can try in under a minute

*Verdict:* Judged as a portfolio artifact today, the repository actively hides its own achievements. A reader spending thirty seconds would conclude nothing had been trained, because the Results section says exactly that.

### Ml Engineer - **ENGINEERING_REVIEW_READY_WITH_GAPS**

This reads like a research codebase written by someone who has been burned by leakage before. Splits are frozen and fingerprinted, duplicates are resolved into connected components rather than pairs, the holdout is gated twice, adapters are audited for fidelity rather than trusted, and two differently-targeted segmenters are adjudicated by an external canonical evaluator instead of the framework's own metric. Reporting is separated from execution. The main risk to another engineer is navigational, not scientific: knowing which of 44 scripts and 135 reports to read.

**Strengths**

- reproducibility primitives are real: pinned lockfile, one seed, provenance records with input and output hashes, configuration in files parsed strictly
- leakage control is concrete: group-aware split, duplicate connected components, identical image ids across both task views, verified by test
- the adapter is not trusted: 1726 instances audited with loss decomposed into rasterisation, component joining and quantisation levels
- checkpoints resolve by digest, never by path, and the accessor refuses the wrong experiment's weights even at identical byte length
- training, evaluation and reporting live in separate modules, so a prose fix never re-runs a model
- 2547 tests, many asserting protocol properties rather than behaviour

**Gaps**

- the documented command path stops at phase 5C.1: nothing tells an engineer how to run the split freeze, materialisation, training, comparison or evaluation, though all of those scripts exist
- frozen checkpoints are git-ignored and no documented way to obtain them exists, so a fresh clone cannot reproduce any model result
- scripts/train_detection_baseline.py keeps a known-broken optimizer regex, recorded as LOW and deliberately unfixed inside an experiment phase
- no CI: the quality gates exist but nothing runs them on push
- no clean-room reproduction has ever been executed
- 135 flat report documents with no index; discoverability is the weak point

*Verdict:* Code quality, testability, configuration discipline and scientific traceability are all strong. Reproducibility is asserted rather than demonstrated: the command path is incomplete, the weights are unobtainable and no clean-room run has been performed.

### First Time User - **NOT_READY_FOR_A_NEWCOMER**

Installation is genuinely good - uv, a lockfile, and an environment check script that prints a useful report. Then the path stops. The Setup section walks through the audit and split-search phases and ends there, so a newcomer can reproduce the dataset analysis and nothing else. There is no demo: no notebook, no inference script, no sample image, no way to see the models do anything without training them, and no documented way to get the weights.

**Strengths**

- uv sync plus a single check_environment.py command works and explains itself
- .env.example documents every variable and contains no secrets
- the API key is read from the environment only and never written to a file
- the commands shown are copy-pasteable and correctly flag which ones contact the provider read-only

**Gaps**

- no demo path at all: clone, install, see something working is impossible
- the README says 'Python 3.11+' while .python-version pins 3.12
- no instructions for obtaining the frozen checkpoints, which are git-ignored
- CUDA is a hard requirement with no CPU fallback (BLOCKED_FOR_GPU), and that is documented in CLAUDE.md rather than in the README setup section
- no expected output is described for any command
- no guidance on which of the 135 report documents to read first

*Verdict:* Install works; everything after install is undocumented or impossible without retraining. The blocking gap is the absence of any demo path.

## 7. Claim audit

| ID | Claim | Verdict | What the evidence supports |
| --- | --- | --- | --- |
| CLAIM-01 | No model has been trained and no evaluation has been run | **STALE** | Five training runs and three evaluation protocols are committed. The sentence is false and is the single most damaging line in the repository. |
| CLAIM-02 | The segmenter is the better object localiser | **UNSUPPORTED** | On validation the canonical box delta is +0.020292, carried entirely by vest_loose; excluding that one-image class the delta is -0.008040. On the holdout the delta is +0.006733 with three of five classes declining. There is no robust universal localisation winner, and the repository correctly declares none. |
| CLAIM-03 | Instance segmentation provides a richer spatial representation than boxes | **SUPPORTED_WITH_LIMITATION** | MASK_TO_BOX_FILL_RATIO median 0.664433 and SHAPE_EXTENT median 0.672 were frozen NO_BOX_ONLY_EQUIVALENT before measurement, so they are a gain in what is computable at all. Limitation: this is not an accuracy claim - no ground-truth geometry entered the comparison - and the fill ratio is not a background error rate. |
| CLAIM-04 | Masks improve person-PPE association | **UNSUPPORTED** | At the frozen 0.50 containment floor, holding the model constant: 103 agree, 3 box-only, 0 mask-only, 66 neither, over 173 relationships. Masks demonstrated no substantial association gain at this rule on this population. The finding is bounded by the rule and must not be generalised. |
| CLAIM-05 | The segmenter costs roughly 30% more end-to-end latency | **SUPPORTED_WITH_LIMITATION** | End-to-end mean 9.157766 ms for D2 against 11.914757 ms for S1, +30.11%, over 4800 timed readings. Three limitations must travel with it: it is a CONTROLLED_LOCAL_HARDWARE_BENCHMARK on one laptop GPU at batch 1 and conf 0.25; the distribution is wide (median 7.30 against 9.96 ms, block means spanning 4.32 to 11.41 ms) so the mean alone misleads; and the difference is ADDITIONAL_SEGMENTATION_PIPELINE_COST, not isolated mask-reconstruction cost. |
| CLAIM-06 | The holdout was evaluated exactly once | **SUPPORTED_WITH_LIMITATION** | One human-authorised evaluation attempt, ONE_SHOT_FINAL_HOLDOUT_EVALUATION_VALID, with zero adaptive reruns and zero post-metric model invocations. The limitation that must be stated alongside it: that attempt comprised three frozen-protocol inference passes - the detector once, the segmenter twice - and 'one-shot' never meant one invocation per model. |
| CLAIM-07 | vest_loose performance is meaningful | **UNSUPPORTED** | One validation image with 8 instances and two holdout images with 7. D2 scored AP 0.000000 on the holdout and recalled none of the 7; S1 scored 0.003850. DESCRIPTIVE_HIGH_UNCERTAINTY: reported in full, deciding nothing. Support remains weak and no threshold was relaxed after seeing it. |
| CLAIM-08 | Reproducible | **OVERSTATED** | The primitives are real - pinned lockfile, one seed, fingerprinted manifests, provenance records. But the documented command path stops at phase 5C.1, the frozen checkpoints are git-ignored with no documented way to obtain them, and no clean-room reproduction has ever been run. 'Reproducible by design, not yet demonstrated from a clean clone' is the honest wording. |
| CLAIM-09 | Production-ready | **UNSUPPORTED** | No deployment, no serving path, no video runtime, no monitoring, no throughput-under-load measurement, a single-machine latency benchmark and a 433-image dataset. Nothing supports a production-readiness claim. |
| CLAIM-10 | Real-time | **UNSUPPORTED** | Batch-1 single-image latency on one laptop GPU is not a video frame rate. images_per_second_from_mean is MEAN_DERIVED_BATCH1_THROUGHPUT and is explicitly not application or video FPS. No video has been processed, so no real-time claim is available until a video run measures one. |
| CLAIM-11 | Improves PPE compliance, or measures compliance accuracy | **UNSUPPORTED** | There is no compliance ground truth in this project. The spatial work is SPATIAL_ASSOCIATION_ANALYSIS and VISIBLE_PPE_COVERAGE_PROXY stays INTERPRETIVE_OPERATIONAL_PROXY. No compliance accuracy may be claimed. |
| CLAIM-12 | Robust | **UNSUPPORTED** | One training run per configuration, run-to-run variance UNKNOWN, no significance test anywhere by design, no cross-site or cross-camera validation, and nothing establishing that two images in different splits do not share a site, a day, a camera or a worker. Robustness is untested. |

## 8. Portfolio signals, ranked

**1. Holdout discipline enforced in code**

*What was done.* The test split was frozen in phase 5C.2 and read exactly once in phase 11B, behind a dual gate requiring both an in-code opt-in and an environment variable, granted only to one named runner for one named purpose, with no code able to satisfy its own precondition and an append-only 13-state ledger whose attempt counter cannot be reset.

*Why it matters.* Test-set leakage is the most common way ML results become fiction, and almost every portfolio project has it somewhere. Making the guard structural rather than procedural is what a senior reviewer looks for.

*Skill demonstrated.* experimental integrity as an engineering property

**2. A controlled one-variable segmentation-target experiment**

*What was done.* S1 differs from S0 in exactly one declared field, overlap_mask true to false; 43 other framework arguments are resolved from S0's own protocol in code and verified against the run's args.yaml afterwards. Because that flag reshapes the framework's validation target as well as the training target, an external canonical evaluator was frozen first so the two runs could be compared at all.

*Why it matters.* It shows the candidate noticed that a hyperparameter change had made the obvious metric incomparable, and built the yardstick before running the experiment rather than rationalising afterwards.

*Skill demonstrated.* experimental design under a confounded metric

**3. Dataset provenance recovery and canonical annotation resolution**

*What was done.* The export declared 742 images and 3373 annotations; only 436 are independent source images. Half the annotations carried geometry as base64 zlib COCO RLE that a naive reader silently drops. The live source project was recovered read-only, verified against the provider's own areas and boxes at 0.0 px agreement, and declared canonical over the export.

*Why it matters.* Most candidates take an export at face value. Finding that half the labels would have been silently lost, and that the sample count was inflated by 70%, is exactly the failure mode that invalidates real projects.

*Skill demonstrated.* data forensics and refusal to trust an input

**4. Group-aware, class-aware constrained split with duplicate reconciliation**

*What was done.* Eleven near-duplicate groups were resolved by two perceptual fingerprints plus human review of every candidate they raised, treated as connected components rather than pairs, and kept inside a single split. The rare class occupies 7 indivisible units over 8 images, so it moves in chunks. 422 units, frozen and fingerprinted; the provider's own split was rejected and is never read by the search.

*Why it matters.* Random splitting on a dataset with duplicated frames inflates every metric. Handling it correctly, and then naming what the split still does not guarantee - shared site, day, camera or worker - is honest and rare.

*Skill demonstrated.* leakage control and calibrated claims about it

**5. A cost/benefit answer instead of a leaderboard**

*What was done.* The detector-versus-segmenter question was answered on four separately interpretable axes with no composite score and no declared winner: recognition broadly similar, a real representation gain in mask-only geometry, no measured association advantage at the frozen rule, and a measured ~30% end-to-end latency premium with 2.375x peak reserved memory. The recommendation is USE_CASE_CONDITIONAL.

*Why it matters.* It is product thinking expressed as measurement. Refusing to declare a winner when the evidence does not support one is the judgement an interviewer most wants to hear a candidate defend.

*Skill demonstrated.* engineering trade-off analysis, and restraint

**6. Adapter fidelity audited rather than assumed**

*What was done.* Before training a segmenter, all 1726 development instances were round-tripped through the YOLO segmentation label format. None round-trips exactly. Loss was decomposed into three levels - rasterisation alone 0.986368, plus component joining 0.985525, plus serialisation and quantisation 0.973066 - and the format's structural inability to express a hole or a disconnected mask was read from the installed framework source.

*Why it matters.* It is the difference between using a tool and knowing what the tool destroys. The three-level decomposition also prevents the common error of blaming the format for loss that exists before the format is involved.

*Skill demonstrated.* reading framework internals; quantifying silent data loss

**7. A standardised latency and inference-memory benchmark**

*What was done.* Batch 1, imgsz 768, FP32 parity proved at runtime rather than assumed from config, 20 warmup iterations discarded, 30 timed repetitions, symmetric interleaved execution in both directions, explicit CUDA synchronisation on both edges of every timed region, and each model's memory measured in a dedicated process because a released model still leaves a 33 MiB cuBLAS workspace behind.

*Why it matters.* Most GPU timings in portfolios measure dispatch, thermal state or the previous model's allocator. Getting all three right, and publishing the wide distribution rather than only the mean, is a strong systems signal.

*Skill demonstrated.* GPU performance measurement done correctly

**8. A claim register and protocol-enforcing tests**

*What was done.* Seven headline claims each carry an evidence artifact, an evidence field, a scope and a limitation, validated by a parser that refuses an incomplete entry. Of 2547 tests, many assert protocol properties rather than behaviour: that audit paths import no model library, that no file assigns to the holdout environment variable, that artifact rebuilds are idempotent, that no committed artifact contains a holdout identifier.

*Why it matters.* It turns scientific honesty into something CI can check. The claim register in particular is a reusable interface between evidence and presentation.

*Skill demonstrated.* testing invariants, not just functions

## 9. Positioning

|  | Recommendation |
| --- | --- |
| H1 | **Construction Safety Vision** |
| Subtitle | PPE Detection & Instance Segmentation with a Locked Holdout |
| Tagline | Detecting worn versus merely present protective equipment on construction sites - and measuring, honestly, what instance masks add over boxes and what they cost. |

*Elevator pitch.* A construction-safety computer-vision system that distinguishes protective equipment being worn from equipment merely lying in the scene, built with a frozen group-aware split and a test set opened exactly once, and evaluated to answer a question rather than to win a leaderboard: instance masks add real spatial information a box cannot express, add no measurable person-PPE association advantage at the rule tested, and cost about 30% more end-to-end latency on the benchmarked hardware.

**Alternatives assessed**

- *Construction Safety Vision - Production-Oriented PPE Detection & Instance Segmentation* - **REJECTED**. 'Production-oriented' is defensible for the engineering discipline - frozen configs, digest-addressed checkpoints, provenance records, a real cost benchmark - but it is not defensible for the system. There is no serving path, no video runtime, no throughput-under-load figure and no monitoring, and the dataset is 433 images. A reader who opens the repository expecting production orientation finds a research codebase, and the gap costs more credibility than the phrase buys. Revisit only after the phase 13 video runtime exists.
- *Construction Safety Vision - Real-Time PPE Compliance Monitoring* - **REJECTED**. Two unsupported claims in one line. No video has been processed so no frame rate exists, and there is no compliance ground truth in the project, so nothing can be called compliance monitoring.
- *Construction Safety Vision - PPE Detection & Instance Segmentation with a Locked Holdout* - **RECOMMENDED**. It names the domain, both tasks and the one methodological property that actually differentiates this repository from every other YOLO portfolio project. It is checkable, and it is the thing a reviewer would be impressed by if they noticed it - so it should not be left to be noticed.

## 10. README structure

The current README is 2434 lines shaped as a `PHASE_LOG`. Recommended structure:

1. H1 + one-line tagline + badges (python, license, tests)
2. hero image: box versus mask on one validation image
3. 60-second summary: what it does, what was built, what was found
4. headline results table (validation and one-shot holdout, both models)
5. architecture diagram (DIA-01)
6. how it works: canonical annotations, frozen split, two tracks, one evaluator
7. what masks add and what they cost - the four axes, no composite
8. quickstart: install, obtain weights, run the demo
9. full reproduction pipeline, marked for GPU and provider access
10. repository map, routing by role
11. limitations, stated plainly
12. scientific record and phase history, linked out
13. license, dataset attribution, GenAI declaration

*Rule.* Move the phase narrative to a linked history document. Delete nothing: the phase record is the provenance that makes the results credible.

## 11. Diagrams

| ID | Diagram | Audience | Placement | Priority |
| --- | --- | --- | --- | --- |
| DIA-01 | Scientific pipeline | professor, ML engineer | README above the fold, and the technical report's method section | P1 |
| DIA-02 | Runtime inference path | recruiter, first-time user | README, next to the demo section | P2 |
| DIA-03 | Box versus mask, one annotated image | recruiter | README hero image | P1 |

## 12. Delivery sequence

| Phase | Name | Mandatory | Closes |
| --- | --- | --- | --- |
| 12B | Truth pass and delivery scaffolding | yes | GAP-001, GAP-006, GAP-008, GAP-011, GAP-013 |
| 12C | Qualitative FP/FN gallery | yes | GAP-005 |
| 13A | Video application | yes | GAP-002 |
| 12D | README professionalisation and visual assets | optional | GAP-010, GAP-012, GAP-015, GAP-016 |
| 12E | Colab notebook | yes | GAP-004 |
| 12F | Reproducibility hardening and clean-room run | optional | GAP-009, GAP-014, GAP-017, GAP-018 |
| 12G | Technical report | yes | GAP-003 |
| 14 | Delivery audit | yes | - |
| 15 | Pitch script and recording | yes | GAP-007 |
| 16 | Tracking bonus | optional | GAP-021 |

## 13. Delivery gap register

### P0

| ID | Gap | Persona | Effort | Dependency | Mandatory |
| --- | --- | --- | --- | --- | --- |
| GAP-001 | The README states that no model has been trained | PROFESSOR, RECRUITER | SMALL | none | yes |
| GAP-002 | No video application (criterion C5, 10% of the grade) | PROFESSOR | LARGE | frozen checkpoints must be locally available | yes |
| GAP-003 | No technical report | PROFESSOR | LARGE | GAP-005 (FP/FN gallery) for the error-analysis section | yes |
| GAP-004 | No executable Colab notebook | PROFESSOR | MEDIUM | GAP-008 (a documented way to obtain the weights) | yes |
| GAP-005 | No per-class qualitative FP/FN gallery | PROFESSOR | MEDIUM | frozen checkpoints available locally; validation imagery only | yes |
| GAP-006 | No GenAI usage declaration | PROFESSOR | SMALL | none | yes |
| GAP-007 | No 5-8 minute pitch | PROFESSOR, RECRUITER | LARGE | GAP-002 (the demo shown must be real pipeline output) | yes |

### P1

| ID | Gap | Persona | Effort | Dependency | Mandatory |
| --- | --- | --- | --- | --- | --- |
| GAP-008 | Frozen checkpoints cannot be obtained from a clean clone | ML_ENGINEER, FIRST_TIME_USER | SMALL | none | no |
| GAP-009 | The documented command path stops at phase 5C.1 | ML_ENGINEER, FIRST_TIME_USER | MEDIUM | none | no |
| GAP-010 | No demo path: clone, install, see something working is impossible | RECRUITER, FIRST_TIME_USER | MEDIUM | GAP-008 | no |
| GAP-011 | No software license | RECRUITER, PROFESSOR | SMALL | none | no |
| GAP-012 | The README is a 2434-line phase log, not a portfolio entry point | RECRUITER | LARGE | GAP-001, DIA-01, DIA-03 | no |
| GAP-013 | Stale claims presented as current in live documentation | PROFESSOR, ML_ENGINEER | SMALL | none | no |
| GAP-014 | No clean-room reproduction has ever been executed | ML_ENGINEER | MEDIUM | GAP-008, GAP-009 | no |

### P2

| ID | Gap | Persona | Effort | Dependency | Mandatory |
| --- | --- | --- | --- | --- | --- |
| GAP-015 | No architecture diagram and no hero image | RECRUITER, PROFESSOR | MEDIUM | GAP-008 | no |
| GAP-016 | 135 flat report documents with no index | ML_ENGINEER, PROFESSOR | SMALL | none | no |
| GAP-017 | No continuous integration | ML_ENGINEER | SMALL | none | no |
| GAP-018 | Python version and CUDA requirement are under-documented | FIRST_TIME_USER | SMALL | none | no |
| GAP-019 | A known-broken optimizer regex is retained in a phase 6 script | ML_ENGINEER | SMALL | none | no |

### P3

| ID | Gap | Persona | Effort | Dependency | Mandatory |
| --- | --- | --- | --- | --- | --- |
| GAP-020 | No GitHub social preview, topics or description | RECRUITER | SMALL | GAP-015 | no |
| GAP-021 | Tracking bonus not started | PROFESSOR | MEDIUM | every mandatory deliverable | no |
| GAP-022 | The repository-wide holdout-identifier statement is narrower than it reads | ML_ENGINEER | SMALL | none | no |

## 14. What this audit did

|  | Value |
| --- | --- |
| Models executed | 0 |
| Model inference passes | 0 |
| Holdout accessed | no |
| Holdout content read | 0 |
| Metrics recomputed | 0 |
| Scientific results modified | no |
| README rewritten | no |
| Report generated | no |
| Video started | no |

Audit fingerprint `619a303e5576eff3f3fa695e3f666b75c987af96287bf4b3536cbd7e0fd1e0f7`.
