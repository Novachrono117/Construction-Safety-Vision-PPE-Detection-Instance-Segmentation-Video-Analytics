"""Phase 12A declared findings - the editorial half of the final audit.

Kept apart from :mod:`construction_safety_vision.delivery_audit` on purpose.
That module *measures* the repository: file counts, artifact existence, stale
strings, whether the documentation carries the committed numbers. This one
*judges* it: persona verdicts, compliance states, claim verdicts, portfolio
ranking, positioning and the gap register.

The separation is the honesty mechanism. A measurement re-derives on every run
and cannot go stale; a judgement is one person's reading and is labelled as
such. Every judgement below names the measurement or the committed artifact it
rests on, so a reader can reject the judgement and keep the evidence.

Nothing here is a scientific result. No number in this module was produced by
running a model; the metric values quoted are read from committed artifacts by
the audit builder and appear here only as the subject of a claim verdict.
"""

from __future__ import annotations

from typing import Any


def compliance_matrix() -> tuple[dict[str, Any], ...]:
    """Map every assignment requirement to repository evidence.

    Returns:
        One row per requirement, in reporting order.
    """
    return (
        {
            "requirement_id": "R01",
            "requirement": "Dataset of at least 300 annotated images",
            "criterion": "C1",
            "state": "COMPLETE",
            "evidence_artifact": "reports/canonical_annotation_manifest.json",
            "evidence_location": "436 source images, 2031 annotations; 433 modelling images",
            "remaining_action": "none",
        },
        {
            "requirement_id": "R02",
            "requirement": "Public-source citation and license",
            "criterion": "C1",
            "state": "COMPLETE",
            "evidence_artifact": "reports/dataset_provenance.md",
            "evidence_location": "Roboflow Universe v4, CC BY 4.0, archive SHA-256 recorded",
            "remaining_action": "none",
        },
        {
            "requirement_id": "R03",
            "requirement": "Train / validation / test split",
            "criterion": "C1",
            "state": "COMPLETE",
            "evidence_artifact": "reports/split_manifest.json",
            "evidence_location": "303 / 65 / 65 images over 422 indivisible units, frozen",
            "remaining_action": "none",
        },
        {
            "requirement_id": "R04",
            "requirement": "Object detection fine-tuning",
            "criterion": "C2",
            "state": "COMPLETE",
            "evidence_artifact": "reports/final_detector_manifest.json",
            "evidence_location": "D0/D1/D2 manifests; D2 frozen, YOLO11n at imgsz 768",
            "remaining_action": "none",
        },
        {
            "requirement_id": "R05",
            "requirement": "Instance segmentation in the same domain and splits",
            "criterion": "C3",
            "state": "COMPLETE",
            "evidence_artifact": "reports/final_segmenter_manifest.json",
            "evidence_location": "S0/S1 manifests; S1 frozen, YOLO11n-seg, overlap_mask false",
            "remaining_action": "none",
        },
        {
            "requirement_id": "R06",
            "requirement": "Complete hyperparameter documentation",
            "criterion": "C2/C3",
            "state": "COMPLETE",
            "evidence_artifact": "configs/",
            "evidence_location": "16 committed configs; resolved optimizer captured per run",
            "remaining_action": "none",
        },
        {
            "requirement_id": "R07",
            "requirement": "mAP@0.50",
            "criterion": "C4",
            "state": "COMPLETE",
            "evidence_artifact": "reports/final_test_detector.json",
            "evidence_location": "canonical box and mask mAP@0.50, validation and holdout",
            "remaining_action": "none",
        },
        {
            "requirement_id": "R08",
            "requirement": "mAP@0.50:0.95",
            "criterion": "C4",
            "state": "COMPLETE",
            "evidence_artifact": "reports/final_test_segmenter.json",
            "evidence_location": "canonical box and mask mAP@0.50:0.95, validation and holdout",
            "remaining_action": "none",
        },
        {
            "requirement_id": "R09",
            "requirement": "IoU",
            "criterion": "C4",
            "state": "COMPLETE",
            "evidence_artifact": "reports/final_test_direct_iou.json",
            "evidence_location": "direct instance-mask IoU, phase 8C protocol reused unchanged",
            "remaining_action": "none",
        },
        {
            "requirement_id": "R10",
            "requirement": "Precision",
            "criterion": "C4",
            "state": "COMPLETE",
            "evidence_artifact": "reports/final_test_detector.json",
            "evidence_location": "object_level.precision and canonical_precision_recall",
            "remaining_action": "none",
        },
        {
            "requirement_id": "R11",
            "requirement": "Recall",
            "criterion": "C4",
            "state": "COMPLETE",
            "evidence_artifact": "reports/final_test_segmenter.json",
            "evidence_location": "object_level.recall and canonical_mask_precision_recall",
            "remaining_action": "none",
        },
        {
            "requirement_id": "R12",
            "requirement": "Confusion matrix",
            "criterion": "C4",
            "state": "COMPLETE",
            "evidence_artifact": "reports/figures/final_test/",
            "evidence_location": "D2 and S1, raw and normalised; semantics frozen in phase 11A",
            "remaining_action": "none",
        },
        {
            "requirement_id": "R13",
            "requirement": "Qualitative false-positive / false-negative analysis per class",
            "criterion": "C4",
            "state": "PARTIAL",
            "evidence_artifact": "reports/segmentation_S0_error_analysis.md",
            "evidence_location": (
                "a quantitative outcome census and a failure taxonomy exist for S0 on "
                "validation and for both models on the holdout, but no committed gallery "
                "pairs an image with its prediction, its ground truth and a hypothesis; the "
                "phase 8D report embeds zero images and the phase 11B qualitative selection "
                "is git-ignored under the holdout-publication policy"
            ),
            "remaining_action": (
                "build a per-class FP/FN gallery from validation imagery, one documented "
                "false positive and one false negative per class; holdout imagery stays "
                "unpublished"
            ),
        },
        {
            "requirement_id": "R14",
            "requirement": "Inference on real video of at least 30 seconds",
            "criterion": "C5",
            "state": "MISSING",
            "evidence_artifact": "none",
            "evidence_location": (
                "no video script, no video provenance record, no output video and no FPS "
                "measurement; the README records 'Video inference | Not implemented.'"
            ),
            "remaining_action": "acquire a licensed video and build the inference runtime",
        },
        {
            "requirement_id": "R15",
            "requirement": "Technical report",
            "criterion": "C6",
            "state": "MISSING",
            "evidence_artifact": "none",
            "evidence_location": (
                "reports/technical_report.md does not exist; the material exists but is "
                "distributed across 135 flat report documents with no index"
            ),
            "remaining_action": "assemble a 6-10 page report from committed evidence",
        },
        {
            "requirement_id": "R16",
            "requirement": "README that describes the repository honestly",
            "criterion": "C6",
            "state": "PARTIAL",
            "evidence_artifact": "README.md",
            "evidence_location": (
                "2434 lines organised as a phase log, carrying stale claims including "
                "'No model has been trained and no evaluation has been run'"
            ),
            "remaining_action": "restructure and retire every stale claim",
        },
        {
            "requirement_id": "R17",
            "requirement": "Reproducible public GitHub repository",
            "criterion": "C6",
            "state": "PARTIAL",
            "evidence_artifact": "README.md, pyproject.toml, uv.lock",
            "evidence_location": (
                "the repository is public and pushed and the environment is pinned, but no "
                "software license file exists and the documented command path stops at "
                "phase 5C.1 of eleven completed phases"
            ),
            "remaining_action": "add a LICENSE and document the phase 5C.2-11 command path",
        },
        {
            "requirement_id": "R18",
            "requirement": "Executable Colab notebook",
            "criterion": "C6",
            "state": "MISSING",
            "evidence_artifact": "none",
            "evidence_location": (
                "no .ipynb is tracked; notebooks/README.md lists five as planned and states "
                "that none exists"
            ),
            "remaining_action": "one notebook covering dataset, metrics and a runnable demo",
        },
        {
            "requirement_id": "R19",
            "requirement": "5-8 minute video pitch",
            "criterion": "C7",
            "state": "MISSING",
            "evidence_artifact": "none",
            "evidence_location": "no pitch script, no slide deck and no recording",
            "remaining_action": "script from the phase 10D claim register, then record",
        },
        {
            "requirement_id": "R20",
            "requirement": "GenAI usage declaration",
            "criterion": "C6",
            "state": "MISSING",
            "evidence_artifact": "none",
            "evidence_location": (
                "CLAUDE.md governs AI-assisted sessions but is an operating constitution, "
                "not a declaration of how generative AI was used in producing the work"
            ),
            "remaining_action": "declare the tools, the scope of use and the review performed",
        },
        {
            "requirement_id": "R21",
            "requirement": "Bonus: tracking or interactive demo",
            "criterion": "B1",
            "state": "NOT_APPLICABLE",
            "evidence_artifact": "none",
            "evidence_location": (
                "optional; the rubric's own failure condition is bonus work started before "
                "the mandatory criteria are complete"
            ),
            "remaining_action": "optional, only after every mandatory deliverable exists",
        },
    )


def personas() -> tuple[dict[str, Any], ...]:
    """Audit the repository from four independent perspectives.

    Returns:
        One assessment per persona, each judged on its own terms.
    """
    return (
        {
            "persona": "PROFESSOR",
            "assessment": (
                "The scientific core is unusually strong for a graduate assignment: a frozen "
                "group-aware split, a canonical annotation source with boxes derived from "
                "polygons, predeclared selection policies with decision margins, one external "
                "evaluator for both models, and a one-shot holdout with an append-only "
                "ledger. Methodology is traceable end to end and limitations are stated "
                "rather than smoothed. What is missing is not rigour but three whole "
                "deliverables - the video application, the Colab notebook and the pitch - "
                "plus a report the evaluator can read without reconstructing phase history."
            ),
            "strengths": [
                "every reported metric traces to a committed artifact and a provenance record",
                "holdout discipline is enforced in code by a dual gate, not by convention",
                "detection boxes are derived from segmentation polygons, so both tasks "
                "provably describe the same objects",
                "selection policies were frozen before the candidate experiments ran, with "
                "the margin declared in advance and applied mechanically",
                "limitations are explicit: vest_loose small-sample, UNKNOWN causes, "
                "validation-only selection, local-hardware-only latency",
            ],
            "gaps": [
                "no technical report exists; the evidence is spread over 135 flat documents",
                "no video application, which is criterion C5 and 10% of the grade",
                "no executable Colab notebook, required by C6's definition of done",
                "no per-class qualitative FP/FN gallery pairing image, prediction and ground "
                "truth, required by C4's definition of done",
                "no GenAI declaration",
                "the README states 'No model has been trained and no evaluation has been "
                "run', which a reader can reasonably take as the project's own summary",
            ],
            "verdict": "PROFESSOR_READY_WITH_GAPS",
            "verdict_reason": (
                "C1 to C4 are substantially satisfied and defensible under scrutiny. C5 and "
                "C7 are absent entirely and C6 is partial, so the submission would lose whole "
                "criteria rather than marks within them."
            ),
        },
        {
            "persona": "RECRUITER",
            "assessment": (
                "Thirty seconds is spent on a status blockquote of roughly fifty lines dense "
                "with phase numbers, fingerprints and six-decimal metrics. Nothing states "
                "what the system does, what it achieved, or what the candidate built, before "
                "the scrolling starts. The strongest signals here - holdout discipline, a "
                "controlled one-variable experiment, a correct GPU cost benchmark - are "
                "genuine senior-engineer signals, and not one of them is visible above the "
                "fold. The Results section sits at line 2417 and says 'Not available'."
            ),
            "strengths": [
                "the problem framing is genuinely good: worn versus merely present PPE is a "
                "distinction a naive detector cannot make",
                "the engineering discipline on display is rare in portfolio projects",
                "there is a real cost/benefit answer - masks add representation and cost "
                "about 30% more end-to-end latency - which reads as product thinking",
                "2547 automated tests signal a working engineer, not a notebook author",
            ],
            "gaps": [
                "no hero visual: no architecture diagram, no annotated result image, no demo",
                "no result table above the fold, and the Results section says 'Not available'",
                "the first screen is a phase log, so technical depth is invisible at a glance",
                "no stack summary, no one-line elevator pitch, no 'what I built' section",
                "no software license, which reads as unfinished on a public repository",
                "no runnable demo a reader can try in under a minute",
            ],
            "verdict": "NOT_RECRUITER_READY",
            "verdict_reason": (
                "Judged as a portfolio artifact today, the repository actively hides its own "
                "achievements. A reader spending thirty seconds would conclude nothing had "
                "been trained, because the Results section says exactly that."
            ),
        },
        {
            "persona": "ML_ENGINEER",
            "assessment": (
                "This reads like a research codebase written by someone who has been burned "
                "by leakage before. Splits are frozen and fingerprinted, duplicates are "
                "resolved into connected components rather than pairs, the holdout is gated "
                "twice, adapters are audited for fidelity rather than trusted, and two "
                "differently-targeted segmenters are adjudicated by an external canonical "
                "evaluator instead of the framework's own metric. Reporting is separated from "
                "execution. The main risk to another engineer is navigational, not "
                "scientific: knowing which of 44 scripts and 135 reports to read."
            ),
            "strengths": [
                "reproducibility primitives are real: pinned lockfile, one seed, provenance "
                "records with input and output hashes, configuration in files parsed strictly",
                "leakage control is concrete: group-aware split, duplicate connected "
                "components, identical image ids across both task views, verified by test",
                "the adapter is not trusted: 1726 instances audited with loss decomposed into "
                "rasterisation, component joining and quantisation levels",
                "checkpoints resolve by digest, never by path, and the accessor refuses the "
                "wrong experiment's weights even at identical byte length",
                "training, evaluation and reporting live in separate modules, so a prose fix "
                "never re-runs a model",
                "2547 tests, many asserting protocol properties rather than behaviour",
            ],
            "gaps": [
                "the documented command path stops at phase 5C.1: nothing tells an engineer "
                "how to run the split freeze, materialisation, training, comparison or "
                "evaluation, though all of those scripts exist",
                "frozen checkpoints are git-ignored and no documented way to obtain them "
                "exists, so a fresh clone cannot reproduce any model result",
                "scripts/train_detection_baseline.py keeps a known-broken optimizer regex, "
                "recorded as LOW and deliberately unfixed inside an experiment phase",
                "no CI: the quality gates exist but nothing runs them on push",
                "no clean-room reproduction has ever been executed",
                "135 flat report documents with no index; discoverability is the weak point",
            ],
            "verdict": "ENGINEERING_REVIEW_READY_WITH_GAPS",
            "verdict_reason": (
                "Code quality, testability, configuration discipline and scientific "
                "traceability are all strong. Reproducibility is asserted rather than "
                "demonstrated: the command path is incomplete, the weights are unobtainable "
                "and no clean-room run has been performed."
            ),
        },
        {
            "persona": "FIRST_TIME_USER",
            "assessment": (
                "Installation is genuinely good - uv, a lockfile, and an environment check "
                "script that prints a useful report. Then the path stops. The Setup section "
                "walks through the audit and split-search phases and ends there, so a "
                "newcomer can reproduce the dataset analysis and nothing else. There is no "
                "demo: no notebook, no inference script, no sample image, no way to see the "
                "models do anything without training them, and no documented way to get the "
                "weights."
            ),
            "strengths": [
                "uv sync plus a single check_environment.py command works and explains itself",
                ".env.example documents every variable and contains no secrets",
                "the API key is read from the environment only and never written to a file",
                "the commands shown are copy-pasteable and correctly flag which ones contact "
                "the provider read-only",
            ],
            "gaps": [
                "no demo path at all: clone, install, see something working is impossible",
                "the README says 'Python 3.11+' while .python-version pins 3.12",
                "no instructions for obtaining the frozen checkpoints, which are git-ignored",
                "CUDA is a hard requirement with no CPU fallback (BLOCKED_FOR_GPU), and that "
                "is documented in CLAUDE.md rather than in the README setup section",
                "no expected output is described for any command",
                "no guidance on which of the 135 report documents to read first",
            ],
            "verdict": "NOT_READY_FOR_A_NEWCOMER",
            "verdict_reason": (
                "Install works; everything after install is undocumented or impossible "
                "without retraining. The blocking gap is the absence of any demo path."
            ),
        },
    )


def claim_audit() -> tuple[dict[str, Any], ...]:
    """Audit the repository's public-facing claims against its evidence.

    Returns:
        One verdict per claim, with the limitation that must travel with it.
    """
    return (
        {
            "claim_id": "CLAIM-01",
            "claim": "No model has been trained and no evaluation has been run",
            "where": "README.md, Results section",
            "verdict": "STALE",
            "what_the_evidence_supports": (
                "Five training runs and three evaluation protocols are committed. The "
                "sentence is false and is the single most damaging line in the repository."
            ),
        },
        {
            "claim_id": "CLAIM-02",
            "claim": "The segmenter is the better object localiser",
            "where": "not claimed anywhere; audited because the aggregate invites it",
            "verdict": "UNSUPPORTED",
            "what_the_evidence_supports": (
                "On validation the canonical box delta is +0.020292, carried entirely by "
                "vest_loose; excluding that one-image class the delta is -0.008040. On the "
                "holdout the delta is +0.006733 with three of five classes declining. There "
                "is no robust universal localisation winner, and the repository correctly "
                "declares none."
            ),
        },
        {
            "claim_id": "CLAIM-03",
            "claim": "Instance segmentation provides a richer spatial representation than boxes",
            "where": "README phase 10B and 10D sections; the scientific synthesis",
            "verdict": "SUPPORTED_WITH_LIMITATION",
            "what_the_evidence_supports": (
                "MASK_TO_BOX_FILL_RATIO median 0.664433 and SHAPE_EXTENT median 0.672 were "
                "frozen NO_BOX_ONLY_EQUIVALENT before measurement, so they are a gain in what "
                "is computable at all. Limitation: this is not an accuracy claim - no "
                "ground-truth geometry entered the comparison - and the fill ratio is not a "
                "background error rate."
            ),
        },
        {
            "claim_id": "CLAIM-04",
            "claim": "Masks improve person-PPE association",
            "where": "not claimed; the repository states the opposite",
            "verdict": "UNSUPPORTED",
            "what_the_evidence_supports": (
                "At the frozen 0.50 containment floor, holding the model constant: 103 agree, "
                "3 box-only, 0 mask-only, 66 neither, over 173 relationships. Masks "
                "demonstrated no substantial association gain at this rule on this "
                "population. The finding is bounded by the rule and must not be generalised."
            ),
        },
        {
            "claim_id": "CLAIM-05",
            "claim": "The segmenter costs roughly 30% more end-to-end latency",
            "where": "README phase 10C section",
            "verdict": "SUPPORTED_WITH_LIMITATION",
            "what_the_evidence_supports": (
                "End-to-end mean 9.157766 ms for D2 against 11.914757 ms for S1, +30.11%, "
                "over 4800 timed readings. Three limitations must travel with it: it is a "
                "CONTROLLED_LOCAL_HARDWARE_BENCHMARK on one laptop GPU at batch 1 and conf "
                "0.25; the distribution is wide (median 7.30 against 9.96 ms, block means "
                "spanning 4.32 to 11.41 ms) so the mean alone misleads; and the difference is "
                "ADDITIONAL_SEGMENTATION_PIPELINE_COST, not isolated mask-reconstruction cost."
            ),
        },
        {
            "claim_id": "CLAIM-06",
            "claim": "The holdout was evaluated exactly once",
            "where": "README phase 11B section; final_test_execution_accounting",
            "verdict": "SUPPORTED_WITH_LIMITATION",
            "what_the_evidence_supports": (
                "One human-authorised evaluation attempt, "
                "ONE_SHOT_FINAL_HOLDOUT_EVALUATION_VALID, with zero adaptive reruns and zero "
                "post-metric model invocations. The limitation that must be stated alongside "
                "it: that attempt comprised three frozen-protocol inference passes - the "
                "detector once, the segmenter twice - and 'one-shot' never meant one "
                "invocation per model."
            ),
        },
        {
            "claim_id": "CLAIM-07",
            "claim": "vest_loose performance is meaningful",
            "where": "not claimed; guarded everywhere it appears",
            "verdict": "UNSUPPORTED",
            "what_the_evidence_supports": (
                "One validation image with 8 instances and two holdout images with 7. D2 "
                "scored AP 0.000000 on the holdout and recalled none of the 7; S1 scored "
                "0.003850. DESCRIPTIVE_HIGH_UNCERTAINTY: reported in full, deciding nothing. "
                "Support remains weak and no threshold was relaxed after seeing it."
            ),
        },
        {
            "claim_id": "CLAIM-08",
            "claim": "Reproducible",
            "where": "README title line and Reproducibility goals section",
            "verdict": "OVERSTATED",
            "what_the_evidence_supports": (
                "The primitives are real - pinned lockfile, one seed, fingerprinted "
                "manifests, provenance records. But the documented command path stops at "
                "phase 5C.1, the frozen checkpoints are git-ignored with no documented way to "
                "obtain them, and no clean-room reproduction has ever been run. 'Reproducible "
                "by design, not yet demonstrated from a clean clone' is the honest wording."
            ),
        },
        {
            "claim_id": "CLAIM-09",
            "claim": "Production-ready",
            "where": "not claimed; audited for the positioning decision",
            "verdict": "UNSUPPORTED",
            "what_the_evidence_supports": (
                "No deployment, no serving path, no video runtime, no monitoring, no "
                "throughput-under-load measurement, a single-machine latency benchmark and a "
                "433-image dataset. Nothing supports a production-readiness claim."
            ),
        },
        {
            "claim_id": "CLAIM-10",
            "claim": "Real-time",
            "where": "not claimed; audited because the latency numbers invite it",
            "verdict": "UNSUPPORTED",
            "what_the_evidence_supports": (
                "Batch-1 single-image latency on one laptop GPU is not a video frame rate. "
                "images_per_second_from_mean is MEAN_DERIVED_BATCH1_THROUGHPUT and is "
                "explicitly not application or video FPS. No video has been processed, so no "
                "real-time claim is available until a video run measures one."
            ),
        },
        {
            "claim_id": "CLAIM-11",
            "claim": "Improves PPE compliance, or measures compliance accuracy",
            "where": "not claimed; the repository's parsers refuse it",
            "verdict": "UNSUPPORTED",
            "what_the_evidence_supports": (
                "There is no compliance ground truth in this project. The spatial work is "
                "SPATIAL_ASSOCIATION_ANALYSIS and VISIBLE_PPE_COVERAGE_PROXY stays "
                "INTERPRETIVE_OPERATIONAL_PROXY. No compliance accuracy may be claimed."
            ),
        },
        {
            "claim_id": "CLAIM-12",
            "claim": "Robust",
            "where": "not claimed; audited as a tempting portfolio word",
            "verdict": "UNSUPPORTED",
            "what_the_evidence_supports": (
                "One training run per configuration, run-to-run variance UNKNOWN, no "
                "significance test anywhere by design, no cross-site or cross-camera "
                "validation, and nothing establishing that two images in different splits do "
                "not share a site, a day, a camera or a worker. Robustness is untested."
            ),
        },
    )


def portfolio_signals() -> tuple[dict[str, Any], ...]:
    """Rank the repository's strongest portfolio signals.

    Ranked by what a technical interviewer can probe for thirty minutes, not by
    implementation effort.

    Returns:
        Signals in descending recruiter and interviewer value.
    """
    return (
        {
            "rank": 1,
            "signal": "Holdout discipline enforced in code",
            "what_was_done": (
                "The test split was frozen in phase 5C.2 and read exactly once in phase 11B, "
                "behind a dual gate requiring both an in-code opt-in and an environment "
                "variable, granted only to one named runner for one named purpose, with no "
                "code able to satisfy its own precondition and an append-only 13-state "
                "ledger whose attempt counter cannot be reset."
            ),
            "why_it_matters": (
                "Test-set leakage is the most common way ML results become fiction, and "
                "almost every portfolio project has it somewhere. Making the guard "
                "structural rather than procedural is what a senior reviewer looks for."
            ),
            "skill_demonstrated": "experimental integrity as an engineering property",
        },
        {
            "rank": 2,
            "signal": "A controlled one-variable segmentation-target experiment",
            "what_was_done": (
                "S1 differs from S0 in exactly one declared field, overlap_mask true to "
                "false; 43 other framework arguments are resolved from S0's own protocol in "
                "code and verified against the run's args.yaml afterwards. Because that flag "
                "reshapes the framework's validation target as well as the training target, "
                "an external canonical evaluator was frozen first so the two runs could be "
                "compared at all."
            ),
            "why_it_matters": (
                "It shows the candidate noticed that a hyperparameter change had made the "
                "obvious metric incomparable, and built the yardstick before running the "
                "experiment rather than rationalising afterwards."
            ),
            "skill_demonstrated": "experimental design under a confounded metric",
        },
        {
            "rank": 3,
            "signal": "Dataset provenance recovery and canonical annotation resolution",
            "what_was_done": (
                "The export declared 742 images and 3373 annotations; only 436 are "
                "independent source images. Half the annotations carried geometry as "
                "base64 zlib COCO RLE that a naive reader silently drops. The live source "
                "project was recovered read-only, verified against the provider's own areas "
                "and boxes at 0.0 px agreement, and declared canonical over the export."
            ),
            "why_it_matters": (
                "Most candidates take an export at face value. Finding that half the labels "
                "would have been silently lost, and that the sample count was inflated by "
                "70%, is exactly the failure mode that invalidates real projects."
            ),
            "skill_demonstrated": "data forensics and refusal to trust an input",
        },
        {
            "rank": 4,
            "signal": "Group-aware, class-aware constrained split with duplicate reconciliation",
            "what_was_done": (
                "Eleven near-duplicate groups were resolved by two perceptual fingerprints "
                "plus human review of every candidate they raised, treated as connected "
                "components rather than pairs, and kept inside a single split. The rare "
                "class occupies 7 indivisible units over 8 images, so it moves in chunks. "
                "422 units, frozen and fingerprinted; the provider's own split was rejected "
                "and is never read by the search."
            ),
            "why_it_matters": (
                "Random splitting on a dataset with duplicated frames inflates every metric. "
                "Handling it correctly, and then naming what the split still does not "
                "guarantee - shared site, day, camera or worker - is honest and rare."
            ),
            "skill_demonstrated": "leakage control and calibrated claims about it",
        },
        {
            "rank": 5,
            "signal": "A cost/benefit answer instead of a leaderboard",
            "what_was_done": (
                "The detector-versus-segmenter question was answered on four separately "
                "interpretable axes with no composite score and no declared winner: "
                "recognition broadly similar, a real representation gain in mask-only "
                "geometry, no measured association advantage at the frozen rule, and a "
                "measured ~30% end-to-end latency premium with 2.375x peak reserved memory. "
                "The recommendation is USE_CASE_CONDITIONAL."
            ),
            "why_it_matters": (
                "It is product thinking expressed as measurement. Refusing to declare a "
                "winner when the evidence does not support one is the judgement an "
                "interviewer most wants to hear a candidate defend."
            ),
            "skill_demonstrated": "engineering trade-off analysis, and restraint",
        },
        {
            "rank": 6,
            "signal": "Adapter fidelity audited rather than assumed",
            "what_was_done": (
                "Before training a segmenter, all 1726 development instances were "
                "round-tripped through the YOLO segmentation label format. None round-trips "
                "exactly. Loss was decomposed into three levels - rasterisation alone "
                "0.986368, plus component joining 0.985525, plus serialisation and "
                "quantisation 0.973066 - and the format's structural inability to express a "
                "hole or a disconnected mask was read from the installed framework source."
            ),
            "why_it_matters": (
                "It is the difference between using a tool and knowing what the tool "
                "destroys. The three-level decomposition also prevents the common error of "
                "blaming the format for loss that exists before the format is involved."
            ),
            "skill_demonstrated": "reading framework internals; quantifying silent data loss",
        },
        {
            "rank": 7,
            "signal": "A standardised latency and inference-memory benchmark",
            "what_was_done": (
                "Batch 1, imgsz 768, FP32 parity proved at runtime rather than assumed from "
                "config, 20 warmup iterations discarded, 30 timed repetitions, symmetric "
                "interleaved execution in both directions, explicit CUDA synchronisation on "
                "both edges of every timed region, and each model's memory measured in a "
                "dedicated process because a released model still leaves a 33 MiB cuBLAS "
                "workspace behind."
            ),
            "why_it_matters": (
                "Most GPU timings in portfolios measure dispatch, thermal state or the "
                "previous model's allocator. Getting all three right, and publishing the "
                "wide distribution rather than only the mean, is a strong systems signal."
            ),
            "skill_demonstrated": "GPU performance measurement done correctly",
        },
        {
            "rank": 8,
            "signal": "A claim register and protocol-enforcing tests",
            "what_was_done": (
                "Seven headline claims each carry an evidence artifact, an evidence field, a "
                "scope and a limitation, validated by a parser that refuses an incomplete "
                "entry. Of 2547 tests, many assert protocol properties rather than "
                "behaviour: that audit paths import no model library, that no file assigns "
                "to the holdout environment variable, that artifact rebuilds are idempotent, "
                "that no committed artifact contains a holdout identifier."
            ),
            "why_it_matters": (
                "It turns scientific honesty into something CI can check. The claim register "
                "in particular is a reusable interface between evidence and presentation."
            ),
            "skill_demonstrated": "testing invariants, not just functions",
        },
    )


def positioning() -> dict[str, Any]:
    """Recommend how the project should present itself.

    Returns:
        The recommended H1, tagline and elevator pitch, with the rejected
        alternatives and the reason each was rejected.
    """
    return {
        "recommended_h1": "Construction Safety Vision",
        "recommended_subtitle": "PPE Detection & Instance Segmentation with a Locked Holdout",
        "recommended_tagline": (
            "Detecting worn versus merely present protective equipment on construction "
            "sites - and measuring, honestly, what instance masks add over boxes and what "
            "they cost."
        ),
        "recommended_elevator_pitch": (
            "A construction-safety computer-vision system that distinguishes protective "
            "equipment being worn from equipment merely lying in the scene, built with a "
            "frozen group-aware split and a test set opened exactly once, and evaluated to "
            "answer a question rather than to win a leaderboard: instance masks add real "
            "spatial information a box cannot express, add no measurable person-PPE "
            "association advantage at the rule tested, and cost about 30% more end-to-end "
            "latency on the benchmarked hardware."
        ),
        "assessed_alternatives": [
            {
                "candidate": "Construction Safety Vision - Production-Oriented PPE Detection "
                "& Instance Segmentation",
                "verdict": "REJECTED",
                "reason": (
                    "'Production-oriented' is defensible for the engineering discipline - "
                    "frozen configs, digest-addressed checkpoints, provenance records, a "
                    "real cost benchmark - but it is not defensible for the system. There "
                    "is no serving path, no video runtime, no throughput-under-load figure "
                    "and no monitoring, and the dataset is 433 images. A reader who opens "
                    "the repository expecting production orientation finds a research "
                    "codebase, and the gap costs more credibility than the phrase buys. "
                    "Revisit only after the phase 13 video runtime exists."
                ),
            },
            {
                "candidate": "Construction Safety Vision - Real-Time PPE Compliance Monitoring",
                "verdict": "REJECTED",
                "reason": (
                    "Two unsupported claims in one line. No video has been processed so no "
                    "frame rate exists, and there is no compliance ground truth in the "
                    "project, so nothing can be called compliance monitoring."
                ),
            },
            {
                "candidate": "Construction Safety Vision - PPE Detection & Instance "
                "Segmentation with a Locked Holdout",
                "verdict": "RECOMMENDED",
                "reason": (
                    "It names the domain, both tasks and the one methodological property "
                    "that actually differentiates this repository from every other YOLO "
                    "portfolio project. It is checkable, and it is the thing a reviewer "
                    "would be impressed by if they noticed it - so it should not be left "
                    "to be noticed."
                ),
            },
        ],
        "production_oriented_assessed": True,
        "production_claim_permitted": False,
    }


def diagrams() -> tuple[dict[str, Any], ...]:
    """Recommend the architecture diagrams that should exist.

    Returns:
        One recommendation per diagram, in build order.
    """
    return (
        {
            "diagram_id": "DIA-01",
            "name": "Scientific pipeline",
            "audience": "professor, ML engineer",
            "placement": "README above the fold, and the technical report's method section",
            "content": (
                "Roboflow v4 export -> canonical annotation recovery (live source, RLE "
                "decoded, boxes derived from polygons) -> group-aware frozen split "
                "303/65/65 -> two task views sharing image ids -> detector track "
                "(D0/D1/D2, D2 frozen) and segmenter track (S0/S1, S1 frozen) -> canonical "
                "COCOeval -> controlled validation comparison -> one-shot holdout"
            ),
            "must_show": [
                "that boxes are derived from polygons, not annotated separately",
                "that the split is frozen once and shared by both tasks",
                "that the holdout branch is opened only at the end, once",
            ],
            "priority": "P1",
        },
        {
            "diagram_id": "DIA-02",
            "name": "Runtime inference path",
            "audience": "recruiter, first-time user",
            "placement": "README, next to the demo section",
            "content": (
                "image or video frame -> letterbox to 768 -> D2 or S1 -> NMS -> boxes, or "
                "boxes plus masks reconstructed on the original canvas -> person-PPE spatial "
                "association -> overlay and per-frame summary"
            ),
            "must_show": [
                "the two selectable model paths and what each emits",
                "that mask reconstruction happens inside the measured latency boundary",
                "that association is spatial, not a compliance verdict",
            ],
            "priority": "P2",
        },
        {
            "diagram_id": "DIA-03",
            "name": "Box versus mask, one annotated image",
            "audience": "recruiter",
            "placement": "README hero image",
            "content": (
                "one validation image shown twice - detector boxes on the left, segmenter "
                "masks on the right - with the median fill ratio 0.664 called out"
            ),
            "must_show": [
                "the visual answer to 'what does the mask add', in one glance",
                "validation imagery only; no holdout image may be published",
            ],
            "priority": "P1",
        },
    )


def visual_assets() -> dict[str, Any]:
    """Inventory the committed visual assets and name what is missing.

    Returns:
        What exists and is reusable, and what still has to be produced.
    """
    return {
        "existing_useful": [
            {
                "asset": "reports/figures/final_test/{d2,s1}/confusion_matrix*.png",
                "use": "academic report, pitch; satisfies the confusion-matrix requirement",
            },
            {
                "asset": "reports/figures/detection/D2/Box{PR,F1,P,R}_curve.png, results.png",
                "use": "report appendix; training curves and operating-point behaviour",
            },
            {
                "asset": "reports/figures/segmentation_S1/Mask*_curve.png, results.png",
                "use": "report appendix; the segmenter's mask-branch behaviour",
            },
            {
                "asset": "reports/figures/source_*.png",
                "use": "report dataset section; class distribution, size, luminance",
            },
            {
                "asset": "reports/figures/review_*.jpg",
                "use": (
                    "dataset-audit evidence; check licence and holdout status of each before "
                    "publishing, since these are source images"
                ),
            },
        ],
        "missing": [
            {
                "asset": "hero result image (box versus mask on one validation image)",
                "for": "README, social preview",
                "priority": "P1",
            },
            {"asset": "scientific pipeline diagram", "for": "README, report", "priority": "P1"},
            {"asset": "runtime inference diagram", "for": "README, pitch", "priority": "P2"},
            {"asset": "per-class FP/FN gallery", "for": "report, assignment C4", "priority": "P0"},
            {
                "asset": "metrics comparison chart (D2 vs S1, validation vs holdout)",
                "for": "README, pitch",
                "priority": "P2",
            },
            {
                "asset": "latency/memory trade-off chart with the distribution shown",
                "for": "pitch",
                "priority": "P2",
            },
            {
                "asset": "demo GIF or video thumbnail",
                "for": "README, social preview",
                "priority": "P1",
            },
            {
                "asset": "GitHub social preview image (1280x640)",
                "for": "link sharing",
                "priority": "P3",
            },
        ],
        "holdout_imagery_policy": (
            "No holdout image, holdout image id or holdout prediction may appear in any "
            "published asset. Every qualitative figure must be built from validation imagery."
        ),
    }


def gap_register() -> tuple[dict[str, Any], ...]:
    """The prioritised delivery gap register.

    P0 blocks the assignment or invalidates delivery. P1 is a major quality or
    reproducibility issue. P2 is portfolio polish. P3 is optional.

    Returns:
        One entry per gap, most severe first.
    """
    return (
        {
            "gap_id": "GAP-001",
            "severity": "P0",
            "persona": ["PROFESSOR", "RECRUITER"],
            "title": "The README states that no model has been trained",
            "evidence": (
                "README.md Results section: 'Not available. No model has been trained and no "
                "evaluation has been run.' Five training runs and three evaluation protocols "
                "are committed."
            ),
            "recommended_fix": (
                "Replace with the committed final-test and validation tables, read by field "
                "from reports/final_test_*.json."
            ),
            "effort": "SMALL",
            "dependency": "none",
            "mandatory": True,
        },
        {
            "gap_id": "GAP-002",
            "severity": "P0",
            "persona": ["PROFESSOR"],
            "title": "No video application (criterion C5, 10% of the grade)",
            "evidence": (
                "No scripts/run_video_inference.py, no video provenance record, no output "
                "video, no FPS measurement. No code path in the repository opens a video."
            ),
            "recommended_fix": (
                "Acquire a licensed construction video of at least 30 s, record its "
                "provenance, build the inference runtime, measure FPS on named hardware and "
                "describe at least three temporal failure modes with timestamps."
            ),
            "effort": "LARGE",
            "dependency": "frozen checkpoints must be locally available",
            "mandatory": True,
        },
        {
            "gap_id": "GAP-003",
            "severity": "P0",
            "persona": ["PROFESSOR"],
            "title": "No technical report",
            "evidence": (
                "reports/technical_report.md does not exist. The evidence exists but is "
                "spread across 135 flat report documents written phase by phase."
            ),
            "recommended_fix": (
                "Assemble a 6-10 page report from committed artifacts, quoting the phase 10D "
                "claim register entries with their scope and limitation."
            ),
            "effort": "LARGE",
            "dependency": "GAP-005 (FP/FN gallery) for the error-analysis section",
            "mandatory": True,
        },
        {
            "gap_id": "GAP-004",
            "severity": "P0",
            "persona": ["PROFESSOR"],
            "title": "No executable Colab notebook",
            "evidence": (
                "No .ipynb is tracked. notebooks/README.md lists five planned notebooks and "
                "states that none exists. C6's definition of done requires one that runs top "
                "to bottom on a fresh Colab runtime."
            ),
            "recommended_fix": (
                "One notebook: clone, install, load the frozen checkpoints by digest, "
                "reproduce the committed validation metrics, and run a visual demo. It must "
                "not train and must never touch the holdout."
            ),
            "effort": "MEDIUM",
            "dependency": "GAP-008 (a documented way to obtain the weights)",
            "mandatory": True,
        },
        {
            "gap_id": "GAP-005",
            "severity": "P0",
            "persona": ["PROFESSOR"],
            "title": "No per-class qualitative FP/FN gallery",
            "evidence": (
                "C4's definition of done requires at least one documented false positive and "
                "one false negative per class, each with image, prediction, ground truth and "
                "a stated hypothesis. reports/segmentation_S0_error_analysis.md embeds zero "
                "images and covers S0, which is not the frozen segmenter; the phase 11B "
                "qualitative selection is git-ignored under the holdout policy."
            ),
            "recommended_fix": (
                "Render the gallery from validation imagery for both frozen models, five "
                "classes, one FP and one FN each, with hypotheses labelled as hypotheses."
            ),
            "effort": "MEDIUM",
            "dependency": "frozen checkpoints available locally; validation imagery only",
            "mandatory": True,
        },
        {
            "gap_id": "GAP-006",
            "severity": "P0",
            "persona": ["PROFESSOR"],
            "title": "No GenAI usage declaration",
            "evidence": (
                "No declaration exists. CLAUDE.md is an operating constitution for "
                "AI-assisted sessions, not a statement of how generative AI was used in "
                "producing the submitted work."
            ),
            "recommended_fix": (
                "A short declaration naming the tools, what they were used for, what was "
                "human-reviewed, and what was authored without assistance."
            ),
            "effort": "SMALL",
            "dependency": "none",
            "mandatory": True,
        },
        {
            "gap_id": "GAP-007",
            "severity": "P0",
            "persona": ["PROFESSOR", "RECRUITER"],
            "title": "No 5-8 minute pitch",
            "evidence": "No pitch script, no slide deck, no recording. Criterion C7, 5%.",
            "recommended_fix": (
                "Script from the claim register so every spoken number matches a committed "
                "artifact, then record with the real pipeline output as the demo."
            ),
            "effort": "LARGE",
            "dependency": "GAP-002 (the demo shown must be real pipeline output)",
            "mandatory": True,
        },
        {
            "gap_id": "GAP-008",
            "severity": "P1",
            "persona": ["ML_ENGINEER", "FIRST_TIME_USER"],
            "title": "Frozen checkpoints cannot be obtained from a clean clone",
            "evidence": (
                "Both checkpoints are git-ignored by policy and identified only by SHA-256. "
                "No documented download, release asset or retraining command exists, so no "
                "third party can reproduce any model result."
            ),
            "recommended_fix": (
                "Publish both checkpoints as GitHub release assets addressed by their "
                "recorded digests, and document the fetch command. Retraining is the "
                "fallback, not the primary path, because a re-run produces different bytes."
            ),
            "effort": "SMALL",
            "dependency": "none",
            "mandatory": False,
        },
        {
            "gap_id": "GAP-009",
            "severity": "P1",
            "persona": ["ML_ENGINEER", "FIRST_TIME_USER"],
            "title": "The documented command path stops at phase 5C.1",
            "evidence": (
                "The README Setup section ends at optimize_split_candidates.py. No command is "
                "documented for the split freeze, task materialisation, the detection or "
                "segmentation adapters, any training run, the comparison, the benchmark or "
                "the holdout evaluation, though all those scripts exist and ran."
            ),
            "recommended_fix": (
                "Document the full ordered pipeline, marking which steps need a GPU, which "
                "contact the provider, and which are deliberately not re-runnable."
            ),
            "effort": "MEDIUM",
            "dependency": "none",
            "mandatory": False,
        },
        {
            "gap_id": "GAP-010",
            "severity": "P1",
            "persona": ["RECRUITER", "FIRST_TIME_USER"],
            "title": "No demo path: clone, install, see something working is impossible",
            "evidence": (
                "No notebook, no inference script, no sample image, no CLI entry point that "
                "runs a frozen model on an image."
            ),
            "recommended_fix": (
                "A single predict command taking an image path and writing an annotated "
                "output, plus one committed sample image licensed for redistribution."
            ),
            "effort": "MEDIUM",
            "dependency": "GAP-008",
            "mandatory": False,
        },
        {
            "gap_id": "GAP-011",
            "severity": "P1",
            "persona": ["RECRUITER", "PROFESSOR"],
            "title": "No software license",
            "evidence": (
                "No LICENSE file. README states 'The software license of this repository is "
                "not yet chosen.' A public portfolio repository without a license is legally "
                "all-rights-reserved and reads as unfinished."
            ),
            "recommended_fix": (
                "Choose a permissive license (MIT or Apache-2.0), add the file, and keep the "
                "dataset's CC BY 4.0 attribution separate and explicit."
            ),
            "effort": "SMALL",
            "dependency": "none",
            "mandatory": False,
        },
        {
            "gap_id": "GAP-012",
            "severity": "P1",
            "persona": ["RECRUITER"],
            "title": "The README is a 2434-line phase log, not a portfolio entry point",
            "evidence": (
                "The first screen is a fifty-line status blockquote of phase numbers and "
                "fingerprints. Setup is at line 2348 and Results at line 2417. Phase-history "
                "detail dominates the document."
            ),
            "recommended_fix": (
                "Restructure to hero, results, architecture, how it works, reproduction, "
                "demo, limitations; move the phase narrative to a linked history document. "
                "Delete nothing - relocate it."
            ),
            "effort": "LARGE",
            "dependency": "GAP-001, DIA-01, DIA-03",
            "mandatory": False,
        },
        {
            "gap_id": "GAP-013",
            "severity": "P1",
            "persona": ["PROFESSOR", "ML_ENGINEER"],
            "title": "Stale claims presented as current in live documentation",
            "evidence": (
                "Eight probes, computed on each audit run. Includes 'No test metric exists.' "
                "and 'no test number exists' in the status table beside the row reporting the "
                "test metrics, and the superseded pre-correction spatial fingerprint "
                "9877b88d... quoted as the phase 10B result in reports/roadmap.md, which "
                "CLAUDE.md states must not be quoted."
            ),
            "recommended_fix": (
                "Retire each probe: correct the roadmap fingerprint to 3988bcf6..., and "
                "either re-tense or remove the superseded status rows."
            ),
            "effort": "SMALL",
            "dependency": "none",
            "mandatory": False,
        },
        {
            "gap_id": "GAP-014",
            "severity": "P1",
            "persona": ["ML_ENGINEER"],
            "title": "No clean-room reproduction has ever been executed",
            "evidence": (
                "The README promises that a third party can reproduce every number from a "
                "clean clone. Nothing has tested that claim, and reports/reproducibility_"
                "audit.md does not exist."
            ),
            "recommended_fix": (
                "Run the clean-room plan recorded in this audit and publish the result, "
                "including what could not be reproduced and why."
            ),
            "effort": "MEDIUM",
            "dependency": "GAP-008, GAP-009",
            "mandatory": False,
        },
        {
            "gap_id": "GAP-015",
            "severity": "P2",
            "persona": ["RECRUITER", "PROFESSOR"],
            "title": "No architecture diagram and no hero image",
            "evidence": (
                "The only architecture rendering is an ASCII block under the heading 'Planned "
                "architecture'. No committed figure shows a prediction on an image."
            ),
            "recommended_fix": "Produce DIA-01 and DIA-03 from validation imagery.",
            "effort": "MEDIUM",
            "dependency": "GAP-008",
            "mandatory": False,
        },
        {
            "gap_id": "GAP-016",
            "severity": "P2",
            "persona": ["ML_ENGINEER", "PROFESSOR"],
            "title": "135 flat report documents with no index",
            "evidence": (
                "reports/ holds 135 documents at one level plus 73 figures. A reader cannot "
                "tell which are the current results and which are the historical record."
            ),
            "recommended_fix": (
                "Add reports/README.md routing by role - public entrypoint, scientific "
                "record, reproducibility support, historical, delivery. Move nothing."
            ),
            "effort": "SMALL",
            "dependency": "none",
            "mandatory": False,
        },
        {
            "gap_id": "GAP-017",
            "severity": "P2",
            "persona": ["ML_ENGINEER"],
            "title": "No continuous integration",
            "evidence": (
                "ruff, ruff format and 2547 tests exist and pass, but no workflow runs them "
                "on push, so a reader has to take the quality gates on trust."
            ),
            "recommended_fix": "One GitHub Actions workflow running the three gates on push.",
            "effort": "SMALL",
            "dependency": "none",
            "mandatory": False,
        },
        {
            "gap_id": "GAP-018",
            "severity": "P2",
            "persona": ["FIRST_TIME_USER"],
            "title": "Python version and CUDA requirement are under-documented",
            "evidence": (
                "README says 'Python 3.11+' while .python-version pins 3.12. The CUDA 12.8 / "
                "sm_120 requirement and the explicit no-CPU-fallback rule live in CLAUDE.md, "
                "not in the setup section a newcomer reads."
            ),
            "recommended_fix": "State the pinned version and the GPU requirement in Setup.",
            "effort": "SMALL",
            "dependency": "none",
            "mandatory": False,
        },
        {
            "gap_id": "GAP-019",
            "severity": "P2",
            "persona": ["ML_ENGINEER"],
            "title": "A known-broken optimizer regex is retained in a phase 6 script",
            "evidence": (
                "scripts/train_detection_baseline.py carries an optimizer-capture regex that "
                "cannot match, because it does not strip the ANSI codes the framework wraps "
                "its log label in. Classified LOW and deliberately left, since D0 is frozen "
                "and its published numbers are unaffected."
            ),
            "recommended_fix": (
                "Converge it onto detection_run.py in a cleanup phase, outside any experiment "
                "phase, and note in the code why it was left."
            ),
            "effort": "SMALL",
            "dependency": "none",
            "mandatory": False,
        },
        {
            "gap_id": "GAP-020",
            "severity": "P3",
            "persona": ["RECRUITER"],
            "title": "No GitHub social preview, topics or description",
            "evidence": "Repository metadata not assessed as configured; no preview image exists.",
            "recommended_fix": "Set description and topics, upload a 1280x640 preview.",
            "effort": "SMALL",
            "dependency": "GAP-015",
            "mandatory": False,
        },
        {
            "gap_id": "GAP-021",
            "severity": "P3",
            "persona": ["PROFESSOR"],
            "title": "Tracking bonus not started",
            "evidence": (
                "B1 is optional and its own failure condition is bonus work started before "
                "the mandatory criteria are complete."
            ),
            "recommended_fix": "Attempt only after C1-C7 all exist.",
            "effort": "MEDIUM",
            "dependency": "every mandatory deliverable",
            "mandatory": False,
        },
        {
            "gap_id": "GAP-022",
            "severity": "P3",
            "persona": ["ML_ENGINEER"],
            "title": "The repository-wide holdout-identifier statement is narrower than it reads",
            "evidence": (
                "Phase 11 artifacts contain no holdout identifier and a test proves it. But "
                "the frozen split membership is deliberately committed for auditability in "
                "reports/final_split_assignments.csv, split_manifest.json and the phase 5C.1 "
                "candidate tables. That is by design and documented, yet a reader can take "
                "'no holdout identifier is committed' as repository-wide."
            ),
            "recommended_fix": (
                "Scope the sentence explicitly: no holdout imagery or prediction is "
                "published, and membership is recorded so the protocol can be audited."
            ),
            "effort": "SMALL",
            "dependency": "none",
            "mandatory": False,
        },
    )


def clean_room_plan() -> dict[str, Any]:
    """Design, without executing, the clean-room reproduction test.

    Returns:
        Scope, staged checks, the environment recommendation and exclusions.
    """
    return {
        "status": "DESIGNED_NOT_EXECUTED",
        "principle": (
            "A fresh clone, no cached project artifacts, no holdout authorisation, and only "
            "the documented dependencies. What cannot be reproduced is recorded as a "
            "limitation rather than patched by hand."
        ),
        "stages": [
            {
                "stage": 1,
                "name": "INSTALL",
                "check": "uv sync from a fresh clone on the pinned interpreter",
                "pass_condition": "resolves from uv.lock with no manual intervention",
            },
            {
                "stage": 2,
                "name": "QUALITY_GATES",
                "check": "ruff check, ruff format --check, pytest",
                "pass_condition": (
                    "all pass, with any test that legitimately skips on a clean clone reporting why"
                ),
            },
            {
                "stage": 3,
                "name": "ENVIRONMENT",
                "check": "scripts/check_environment.py",
                "pass_condition": "reports the frozen split and holdout: locked",
            },
            {
                "stage": 4,
                "name": "DATASET_ACQUISITION",
                "check": "download_dataset.py and inspect_dataset.py with a provider key",
                "pass_condition": "archive SHA-256 matches the committed provenance record",
            },
            {
                "stage": 5,
                "name": "CANONICAL_MATERIALIZATION",
                "check": "materialize_task_datasets.py for train and validation only",
                "pass_condition": (
                    "368 images / 1726 annotations, byte-identical copies, geometry "
                    "round-trip clean, and the holdout not materialised"
                ),
            },
            {
                "stage": 6,
                "name": "VALIDATION_DEMO",
                "check": (
                    "load both frozen checkpoints by digest and reproduce the committed "
                    "canonical validation metrics"
                ),
                "pass_condition": "values match the committed artifacts",
                "blocked_by": "GAP-008: the checkpoints are not obtainable today",
            },
            {
                "stage": 7,
                "name": "REPORT_REBUILD",
                "check": (
                    "re-run the idempotent artifact builders (synthesis, taxonomy correction, "
                    "execution accounting, this audit)"
                ),
                "pass_condition": "every regenerated artifact is byte-identical",
            },
        ],
        "explicitly_excluded": [
            "the final holdout evaluation - it was a one-shot and may never be repeated",
            "any training run - a re-run produces different bytes under the same name",
            "the latency benchmark - it is valid for one machine only",
        ],
        "recommended_environment": (
            "Two surfaces, for different questions. A GitHub Actions job on a clean runner "
            "for stages 1-3 and 7, because those must hold for anyone and CI proves it "
            "continuously. A local temporary directory outside the working tree for stages "
            "4-6, because they need a provider key and a CUDA GPU that CI does not have. A "
            "container adds isolation the lockfile already provides and is not worth the "
            "maintenance here; Colab is the delivery surface for the notebook, not the "
            "reproduction harness."
        ),
    }


def colab_plan() -> dict[str, Any]:
    """Assess the notebook state and recommend the final Colab scope.

    Returns:
        Current state, gaps and the recommended scope.
    """
    return {
        "current_state": "NONE_EXISTS",
        "evidence": (
            "no .ipynb is tracked; notebooks/README.md states that none exists and lists "
            "five as planned"
        ),
        "gaps": [
            "the assignment requires one that runs top to bottom on a fresh Colab runtime",
            "no way to obtain the frozen weights, so a notebook cannot load them today",
            "no committed sample image licensed for redistribution",
            "the dataset needs a provider key, which cannot be embedded in a public notebook",
        ],
        "recommended_scope": [
            "clone the repository and install from the lockfile",
            "fetch both frozen checkpoints by digest and verify the SHA-256 before use",
            "acquire the dataset with a user-supplied key, or fall back to a bundled sample",
            "materialise the canonical validation view only",
            "reproduce the committed canonical validation metrics and print the comparison",
            "render a box-versus-mask visual demo on validation images",
            "display the committed final-test table read from the JSON, without recomputing it",
        ],
        "must_not": [
            "train anything",
            "touch the holdout, materialise it, or set the environment gate",
            "recompute any test metric",
            "embed a secret",
            "depend on a local absolute path",
        ],
        "rationale": (
            "The notebook is a demonstration surface, not a re-execution of the research "
            "history. It should satisfy the assignment by showing the models working and the "
            "metrics reproducing, in minutes rather than GPU-hours."
        ),
    }


def report_plan() -> dict[str, Any]:
    """Assess report readiness and recommend the final structure.

    Returns:
        Current state, gaps and a 6-10 page structure.
    """
    return {
        "current_state": "NO_TECHNICAL_REPORT_EXISTS",
        "material_available": (
            "essentially all of it - 135 committed report documents, a claim register with "
            "scope and limitation per claim, per-class tables, confusion matrices and "
            "training curves"
        ),
        "gaps": [
            "no single document; a reader must reconstruct the story from phase artifacts",
            "no FP/FN gallery for the error-analysis section",
            "no video results section, because there is no video",
            "no GenAI declaration",
            "citations are recorded for the dataset but no reference list exists",
            "phase-numbered narrative would have to be rewritten as a scientific narrative",
        ],
        "recommended_structure": [
            {"section": "1. Problem and motivation", "pages": 0.5},
            {
                "section": "2. Dataset: source, license, provenance recovery, canonical "
                "annotations, audit findings",
                "pages": 1.5,
            },
            {
                "section": "3. Split design: grouping, duplicates, class constraints, the "
                "frozen 303/65/65 and what it does not guarantee",
                "pages": 1.0,
            },
            {
                "section": "4. Method: detection and segmentation, the shared canonical "
                "ground truth, adapters and their audited fidelity, full hyperparameters",
                "pages": 1.5,
            },
            {
                "section": "5. Model selection: the predeclared policies, the candidates, "
                "the margins, and why D2 and S1 were frozen",
                "pages": 1.0,
            },
            {
                "section": "6. Results: validation and one-shot holdout, mAP@0.50, "
                "mAP@0.50:0.95, precision, recall, IoU, confusion matrices",
                "pages": 1.5,
            },
            {
                "section": "7. Error analysis: per-class FP/FN with hypotheses labelled as "
                "hypotheses, plus the size and overlap-target findings",
                "pages": 1.0,
            },
            {
                "section": "8. What masks add and what they cost: the four axes, no composite",
                "pages": 1.0,
            },
            {"section": "9. Video application and measured throughput", "pages": 0.5},
            {
                "section": "10. Limitations, future work, GenAI declaration, references",
                "pages": 0.5,
            },
        ],
        "rule": (
            "Every number is read from a committed artifact and cited to it. Every claim is "
            "quoted from the claim register together with its scope and its limitation."
        ),
    }


def pitch_plan() -> dict[str, Any]:
    """Assess pitch readiness and recommend the slide narrative.

    Returns:
        Readiness, the recommended slides and their timings.
    """
    return {
        "current_state": "NOT_STARTED",
        "material_readiness": (
            "the argument is ready - the claim register already carries seven claims with "
            "scope and limitation - but two visual dependencies are missing: the demo and "
            "the architecture diagram"
        ),
        "recommended_slides": [
            {"slide": 1, "title": "The problem: worn versus present", "minutes": 0.75},
            {"slide": 2, "title": "Dataset and what auditing it revealed", "minutes": 1.0},
            {"slide": 3, "title": "Split design and holdout discipline", "minutes": 0.75},
            {"slide": 4, "title": "Detector: candidates, frozen policy, D2", "minutes": 0.75},
            {"slide": 5, "title": "Segmenter: the overlap_mask experiment, S1", "minutes": 1.0},
            {"slide": 6, "title": "What masks add, on four axes", "minutes": 1.0},
            {"slide": 7, "title": "What masks cost: latency and memory", "minutes": 0.75},
            {"slide": 8, "title": "The final exam, opened once", "minutes": 0.75},
            {"slide": 9, "title": "Limitations, honestly", "minutes": 0.5},
            {"slide": 10, "title": "Live demo and conclusion", "minutes": 0.75},
        ],
        "recommended_slide_count": 10,
        "target_duration_minutes": 8.0,
        "rule": (
            "Every spoken number must match a committed artifact, and the demo shown must be "
            "real pipeline output. The rubric's failure condition for C7 is a claim that "
            "exceeds what the evaluation supports."
        ),
    }


def video_readiness() -> dict[str, Any]:
    """Audit whether the current runtime could process a real video.

    Returns:
        What exists, what is missing, and the tracking recommendation.
    """
    return {
        "current_state": "NO_VIDEO_CODE_PATH_EXISTS",
        "evidence": (
            "no reference to cv2.VideoCapture, VideoWriter or any video entry point in "
            "scripts/ or src/; the README records 'Video inference | Not implemented.'"
        ),
        "what_exists_and_is_reusable": [
            "frozen checkpoints resolvable by digest through the two freeze accessors",
            "the frozen operational inference settings: conf 0.25, NMS IoU 0.70, imgsz 768, "
            "max_det 300, retina_masks true, FP32 - already used by the phase 8C diagnostic "
            "and the phase 10C benchmark",
            "the person-PPE spatial association rule from phase 10B, with its taxonomy",
            "a measured per-image latency baseline to compare a video frame rate against",
            "opencv is already installed transitively via ultralytics",
        ],
        "must_be_built": [
            "video input and decoding, with the container and codec recorded",
            "output writing, with an explicitly chosen codec and its dependency documented",
            "device selection and an explicit refusal rather than a silent CPU fallback",
            "a detector-or-segmenter switch, since the two emit different output types",
            "overlay rendering: boxes, masks, class labels and confidences",
            "measured FPS with the hardware named, separated from the batch-1 latency figure",
            "a video provenance record with source and license",
            "at least three temporal failure modes described with timestamps",
        ],
        "blockers": [
            "no licensed source video has been acquired",
            "the frozen checkpoints are not obtainable from a clean clone (GAP-008)",
        ],
        "tracking_recommendation": (
            "Keep ByteTrack strictly optional and sequence it after the mandatory video "
            "deliverable is complete and recorded. The rubric's own failure condition for B1 "
            "is bonus work started before the mandatory criteria are done, and tracking "
            "shares the video pipeline, so an unfinished tracker must never block C5."
        ),
    }


def file_routing() -> tuple[dict[str, Any], ...]:
    """Classify the repository's files by the role they play for a reader.

    Returns:
        One entry per role, with what belongs in it and how to route to it.
    """
    return (
        {
            "role": "PUBLIC_ENTRYPOINT",
            "examples": ["README.md", "reports/final_test_evaluation.md"],
            "routing": (
                "The README is the only document a reader should need to start. It must link "
                "outward rather than contain everything."
            ),
        },
        {
            "role": "SCIENTIFIC_RECORD",
            "examples": [
                "reports/final_test_*.json",
                "reports/final_{detector,segmenter}_manifest.json",
                "reports/split_manifest.json",
                "reports/detector_segmenter_scientific_synthesis.json",
                "reports/*.provenance.json",
            ],
            "routing": (
                "The evidence every claim cites. Immutable in practice; corrected by "
                "addendum, never by edit."
            ),
        },
        {
            "role": "REPRODUCIBILITY_SUPPORT",
            "examples": ["configs/", "scripts/", "src/", "tests/", "uv.lock", ".env.example"],
            "routing": "Reachable from a single documented pipeline section in the README.",
        },
        {
            "role": "INTERNAL_HISTORICAL",
            "examples": [
                "reports/roadmap.md",
                "reports/phase1_proposal.md",
                "reports/split_candidates/",
                "reports/detection_D0_report.md",
                "reports/segmentation_S0_*",
            ],
            "routing": (
                "The phase-by-phase record. Valuable and must be kept, but it should be "
                "reached from a history index rather than from the README's main narrative."
            ),
        },
        {
            "role": "DELIVERY",
            "examples": [
                "reports/technical_report.md (missing)",
                "notebooks/*.ipynb (missing)",
                "reports/pitch_script.md (missing)",
                "reports/video_analysis.md (missing)",
                "LICENSE (missing)",
            ],
            "routing": (
                "What the assignment is graded on. Five of the six do not exist yet, which "
                "is the substance of the P0 gaps."
            ),
        },
    )


def delivery_roadmap() -> tuple[dict[str, Any], ...]:
    """Recommend the remaining phase sequence, ordered by the audit's findings.

    Returns:
        The recommended phases in execution order.
    """
    return (
        {
            "phase": "12B",
            "name": "Truth pass and delivery scaffolding",
            "why_first": (
                "The cheapest P0s and the ones that make everything else honest: retire the "
                "'no model has been trained' line and the other stale claims, add the "
                "LICENSE and the GenAI declaration, and publish the checkpoints as release "
                "assets so every later phase can load them. Small effort, unblocks 12E, 13A "
                "and the clean-room run."
            ),
            "closes": ["GAP-001", "GAP-006", "GAP-008", "GAP-011", "GAP-013"],
            "mandatory": True,
        },
        {
            "phase": "12C",
            "name": "Qualitative FP/FN gallery",
            "why_here": (
                "A P0 assignment requirement, and the error-analysis section of the report "
                "depends on it. Needs the checkpoints from 12B and validation imagery only."
            ),
            "closes": ["GAP-005"],
            "mandatory": True,
        },
        {
            "phase": "13A",
            "name": "Video application",
            "why_here": (
                "The largest missing criterion at 10%, with the longest lead time, and the "
                "pitch demo depends on it. Sequenced before the report so the report can "
                "include its results rather than being rewritten afterwards."
            ),
            "closes": ["GAP-002"],
            "mandatory": True,
        },
        {
            "phase": "12D",
            "name": "README professionalisation and visual assets",
            "why_here": (
                "Now the README can show a real demo and a real result table. Doing it "
                "earlier would mean rewriting it once the video and gallery land."
            ),
            "closes": ["GAP-010", "GAP-012", "GAP-015", "GAP-016"],
            "mandatory": False,
        },
        {
            "phase": "12E",
            "name": "Colab notebook",
            "why_here": "Depends on obtainable weights (12B) and on the demo path (12D).",
            "closes": ["GAP-004"],
            "mandatory": True,
        },
        {
            "phase": "12F",
            "name": "Reproducibility hardening and clean-room run",
            "why_here": (
                "Document the full command path, add CI, then execute the clean-room plan and "
                "publish what did and did not reproduce."
            ),
            "closes": ["GAP-009", "GAP-014", "GAP-017", "GAP-018"],
            "mandatory": False,
        },
        {
            "phase": "12G",
            "name": "Technical report",
            "why_here": (
                "Written last among the written deliverables, once the video results, the "
                "gallery and the reproduction outcome all exist to be reported."
            ),
            "closes": ["GAP-003"],
            "mandatory": True,
        },
        {
            "phase": "14",
            "name": "Delivery audit",
            "why_here": "Re-run this audit; every P0 must be closed before the pitch.",
            "closes": [],
            "mandatory": True,
        },
        {
            "phase": "15",
            "name": "Pitch script and recording",
            "why_last": (
                "Every spoken number must match a committed artifact and the demo must be "
                "real output, so it can only be recorded once everything else is final."
            ),
            "closes": ["GAP-007"],
            "mandatory": True,
        },
        {
            "phase": "16",
            "name": "Tracking bonus",
            "why_last": (
                "Optional. The rubric's failure condition is bonus work begun before the "
                "mandatory criteria are complete."
            ),
            "closes": ["GAP-021"],
            "mandatory": False,
        },
    )


def readme_plan() -> dict[str, Any]:
    """Audit the current README and recommend its future structure.

    Returns:
        Findings on the current document and the recommended sections.
    """
    return {
        "current_lines": 2434,
        "current_shape": "PHASE_LOG",
        "findings": {
            "h1_present": True,
            "tagline_present": True,
            "hero_summary_present": False,
            "result_summary_above_the_fold": False,
            "architecture_section_present": True,
            "architecture_labelled_planned": True,
            "dataset_section_present": True,
            "methodology_section_present": True,
            "methodology_labelled_planned": True,
            "results_section_present": True,
            "results_section_accurate": False,
            "reproduction_section_present": True,
            "reproduction_section_complete": False,
            "demo_section_present": False,
            "repository_structure_present": True,
            "limitations_section_present": False,
            "software_license_present": False,
            "dataset_citation_present": True,
            "genai_disclosure_present": False,
            "visual_hierarchy_adequate": False,
        },
        "too_much_detail": [
            "a fifty-line status blockquote as the first screen",
            "a sixty-row status table mixing current state with phase history",
            "roughly 1900 lines of phase-by-phase narrative between the status and the setup",
        ],
        "too_little_detail": [
            "no demo, no hero image, no architecture figure",
            "no consolidated results table",
            "no limitations section as such - limitations are scattered through the phases",
            "reproduction commands stop at phase 5C.1",
        ],
        "recommended_structure": [
            "H1 + one-line tagline + badges (python, license, tests)",
            "hero image: box versus mask on one validation image",
            "60-second summary: what it does, what was built, what was found",
            "headline results table (validation and one-shot holdout, both models)",
            "architecture diagram (DIA-01)",
            "how it works: canonical annotations, frozen split, two tracks, one evaluator",
            "what masks add and what they cost - the four axes, no composite",
            "quickstart: install, obtain weights, run the demo",
            "full reproduction pipeline, marked for GPU and provider access",
            "repository map, routing by role",
            "limitations, stated plainly",
            "scientific record and phase history, linked out",
            "license, dataset attribution, GenAI declaration",
        ],
        "rule": (
            "Move the phase narrative to a linked history document. Delete nothing: the "
            "phase record is the provenance that makes the results credible."
        ),
    }
