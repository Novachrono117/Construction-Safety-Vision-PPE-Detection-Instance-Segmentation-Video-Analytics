# Phase 14B — final academic technical report

**Classification: FINAL_ACADEMIC_REPORT_COMPLETE.** This closes the technical
report, not the whole assignment. The newly supplied exact Colab wording remains
partial; the pitch is open. No scientific work was reopened.

## Delivery

- [Portuguese technical report — PDF](../academic/final_report.pdf): **10 pages**.
- [Authoritative Markdown source](../academic/final_report.md): **4,967 words**,
  including tables, captions and references; URLs and metadata comments excluded.
- [Headline claim map](../academic/report_claims.json): 76 registered fields,
  84 occurrences, read from 12 committed JSON sources.
- [Build and validation instructions](../academic/README.md).
- [Machine-readable delivery audit](final_academic_report_delivery.json) and
  [provenance](final_academic_report_delivery.provenance.json).

Title: **Detecção e segmentação de instâncias na análise visual de EPIs:
informação espacial, desempenho e custo computacional**.

Author: **Vinicius Pereira Gomes**. Course: **Pós-graduação · Visão Computacional
e Reconhecimento de Padrões**. Professor: **Romes Heriberto**, supplied in the
assignment screenshots. Institution was not supplied reliably and is omitted;
no institution, registration number or semester was invented.

The report includes abstract, problem/scenario, dataset and explicit EDA,
detector/segmenter methodology and hyperparameters, evaluation protocol, final
results, final-test error analysis, separate validation gallery, spatial
information, compute costs, real-video demonstration, Colab delivery limits,
discussion, limitations, explicit next steps, conclusion, GenAI disclosure,
licensing and eight references. Methodology/results receive the largest share
of space, consistent with the combined 65% rubric weight.

## Evidence and scientific interpretation

The source revision is `a788d5303e70ddb12f5dc5ae35dd9a4b56fc6736`.
The report uses 436 source images, 433 modeling images, 2,031 canonical
annotations and 422 indivisible split units. The frozen train/validation/test
counts are 303/65/65 images and 1,422/304/305 annotations. EDA intensity and
resolution statistics describe the 436-image provenance population; historical
provider splits are not reused as the final split.

| Final-test output | mAP50 | mAP50:95 | Precision | Recall |
| --- | --- | --- | --- | --- |
| D2 boxes | 0.565260 | 0.427031 | 0.787500 | 0.619672 |
| S1 boxes | 0.583500 | 0.433764 | 0.785441 | 0.672131 |
| S1 masks | 0.579074 | 0.410143 | 0.773946 | 0.662295 |

P/R uses confidence 0.25 and IoU 0.50. The final box delta +0.006733 is
descriptive; three classes decreased and no significance test or universal
winner was established. Direct mask IoU: matched mean 0.834548, GT-normalized
0.585551, match coverage 0.701639, IoU50 coverage 0.662295, IoU75 coverage
0.560656. Higher conditional quality does not imply adequate coverage.

The final-test subsection transcribes D2's recorded confusion matrix and
comments on deterministic FP/FN selection metadata. High-confidence unmatched
predictions and large-area missed instances are described without guessing
their scenes or classes. **No test identifier, individual prediction file or
test image was read or embedded.** The original Phase 11B report records
individual selections/figures in ignored local artifacts; these were not opened.
Any later request to include those visuals requires explicit human approval.

The visual gallery is **validation**, clearly labelled. It includes person and
vest_loose FP/FN panels and three mask-disagreement examples. Rare-class behavior
is not hidden: vest_loose has one validation image, two test images and no mask
true positives at the final operational point. Final AP falls relative to
validation are reported; their causes remain unknown.

The controlled RTX 5070 Laptop FP32/batch-1/imgsz-768 benchmark uses D2/S1 mean
E2E latencies 9.157766/11.914757 ms, medians 7.304750/9.962450 ms, P95
14.145295/17.240505 ms; allocated memory 0.073403/0.231621 GiB and reserved
0.125000/0.296875 GiB. The distribution caveat and timing boundaries remain
explicit. These values are not the complete video-runtime throughput.

The real-video demonstration is 104.52 seconds, 2,613 decoded frames at 25 FPS,
3840×1080 side-by-side output, with 5.950455 processing FPS labelled
DEMO_RUNTIME_MEASUREMENT. One previously published frame is included. The
five-second Colab validation clip does not replace the mandatory real video.

Central conclusion: masks add foreground support, shape and spatial
measurements at higher measured computational cost. They did not establish a
large person-PPE association advantage under the fixed rule or a universal
localization winner. These visual predictions do not certify safety compliance.

## ASSIGNMENT_EXACT_WORDING_AUDIT

The human-supplied current assignment and explicit addendum are the authority
for this audit. Earlier approved phase reports retain their original scope.

| Requirement | Status | Evidence / remaining work |
| --- | --- | --- |
| Report: PDF or MD, 6–10 pages | **PASS** | Ten-page PDF and Portuguese Markdown; explicit EDA and próximos passos |
| Test-set error analysis | **PASS — committed metadata scope** | Final metrics, recorded confusion matrix and commented deterministic FP/FN examples; no scene-level visual interpretation of test imagery |
| Exact Colab wording | **PARTIAL** | `COLAB_EXACT_WORDING_PARTIAL`; training/evaluation sections with visible outputs are not delivered literally |
| Real video >=30 seconds | **PASS** | Approved full 104.52-second inference; public MP4 distribution remains pending |
| GitHub repository | **PARTIAL** | Public main includes 14A; this report is local; final README and full reproduction audit remain open |
| Pitch | **OPEN** | 5–8 minutes, all members, working system including inference video, accessible unlisted YouTube or Drive link |
| GenAI declaration | **PASS** | Report section grounded in delivery/AI_USAGE.md |
| Bonus | **NOT_REQUIRED / NOT_STARTED** | Tracking via ByteTrack/DeepSORT or published Gradio/HF demo, after mandatory work only |

The source notebook has 20 cells and no saved outputs. Its aggregate display
cell (index 5, zero-based) and optional inference cell (index 17) are present.
There is no identifiable training section or executable training/evaluation
cell. Human cloud evidence establishes Mode A and Mode B execution under the
approved two-mode scope; it does not establish literal training/evaluation
compliance. The notebook and its 44-file validated execution surface are unchanged.

Minimum safe **proposed**, unimplemented Phase 14C correction:

1. TRAINING: exact frozen commands/configs and previously recorded training
   outputs/curves, identified as recipes and historical results.
2. EVALUATION: canonical evaluation code paths and committed validation/final-test
   outputs, explicitly not a fresh final-test execution.
3. INFERENCE: retain the validated optional frozen-model runtime.
4. If literal executable training/evaluation is required, the human must decide
   on a separately authorized optional train/validation reproduction path.
   No retraining for presentation and no final-test rerun occurred in 14B.

Planning target: complete mandatory delivery by **19 September 2026**, before
the platform deadline **20 September 2026, 23:55**. No later phase, pitch,
release, bonus or automatic follow-up was started.

## Attribution and references

GenAI assistance is disclosed: planning, code assistance/review, debugging,
test refinement and documentation; human experimental decisions and executed
project artifacts remain the basis of metrics. Private prompts are omitted.

The eight report references cover the Roboflow source, official YOLO11
documentation, COCO API/pycocotools, CC BY 4.0, Frank Vincentz's Wikimedia video,
CC BY-SA 3.0, GNU AGPL-3.0 and the pinned project record. Source attribution is
also attached to each media figure. Code is AGPL-3.0; dataset media/annotations
retain CC BY 4.0; the video frame retains CC BY-SA 3.0. Checkpoint redistribution
still requires its separate review. No checkpoint, dataset or complete MP4 is
part of this commit.

## Review and verification

The agent rendered and inspected all ten PDF pages. Body: 11 pt; tables: 9.5 pt;
captions/references: 9 pt. Pagination, accents, table alignment, complete scenes,
figure captions and page boundaries were inspected. Markdown remains the source.
No manual visual approval by the author is claimed.

**Author checklist — pending human review:** page count; title/author/course/
professor; institution if required; Portuguese; figure readability; table values;
citations/references; dataset attribution; video attribution; GenAI disclosure;
limitations; PDF opens; no clipping/overflow/tiny text; no unintended placeholders.
This checklist is a review aid, not a claim that the human reviewed this report.

Executed gates: Ruff check PASS; Ruff format PASS (261 files); metadata-only
pytest **2,764 passed, 47 skipped, 27 warnings in 44.89 s**, including all 19
report-specific tests. The warnings are the existing pycocotools/NumPy copy-keyword
deprecation warnings from synthetic tests. Sensitive/private-path scan: zero
findings. All 44 validated Colab execution-surface files are byte-identical.
The PDF is explicitly binary in .gitattributes; staged whitespace and PDF text
checks pass. Nineteen explicitly selected files exclude datasets, checkpoints,
MP4s, local session outputs and unrelated instructions.
Byte hashes and details are recorded in the accompanying JSON and provenance. The claim validator and regression tests reject altered headline
values and changed committed source records. Test execution uses metadata-only
mode; it does not invoke real checkpoints or access the holdout.

## Git and accounting

Initial 14B baseline: main at `a788d5303e70ddb12f5dc5ae35dd9a4b56fc6736`, with
only the unrelated untracked AGENTS.md. The separately authorized Phase 14A push
succeeded and the live remote main ref was verified at that revision.

GAP-003 moves OPEN to RESOLVED and R15 to COMPLETE. Every other gap row and
historical scientific accounting remain identical to the 14A snapshot. The new
exact-wording audit qualifies the old R18 scope without rewriting its historical
report. Scientific results/configs, the canonical notebook and runtime remain
unchanged. Models executed: **0**; holdout access: **false**; metrics recomputed:
**0**; selection/training/tuning: **closed**; authorization variable: **unset** at
Process, User and Machine scope.

The approved final commit subject is `docs: add final academic technical report`,
without attribution trailers. Its actual hash is returned after commit; this
record does not attempt to contain its own commit hash. **Phase 14B is not
pushed.** The temporary Colab validation branch is preserved, and AGENTS.md
remains untracked and untouched.
