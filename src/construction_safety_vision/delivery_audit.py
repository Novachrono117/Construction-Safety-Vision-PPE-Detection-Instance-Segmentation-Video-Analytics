"""Phase 12A - the final repository, academic and portfolio audit.

An audit, not a change. Nothing here trains, evaluates, benchmarks, reads the
holdout or touches a scientific number; the artifacts it writes carry **no new
scientific result**, only measurements of the repository and judgements about
delivery readiness.

Five things are deliberate.

**Measurements are computed; judgements are declared.** The inventory, the
deliverable existence checks, the stale-claim probes and the
documentation-consistency checks are all derived from the repository on every
run, so the audit cannot itself go stale: fix the README and re-running reports
it fixed. The persona verdicts and gap severities are editorial, are labelled
as such, and rest on those measurements.

**Result values are read from committed artifacts by field**, never transcribed,
and then searched for verbatim in the live documentation. A headline that the
documentation does not carry is reported as a consistency gap rather than
quietly corrected.

**A stale claim is a probe, not a paraphrase.** Each records the exact string,
where it appears, why it is now false, and what it contradicts, so a later phase
can fix it without re-deriving the finding.

**Nothing is deleted and no history is flattened.** The audit classifies files
by role so a reader can be routed, and says where routing is missing. It does
not propose removing the scientific record.

**The holdout stays shut.** This module imports nothing that can reach a
checkpoint, a split accessor or an image decoder, and a test asserts that. The
audit's own counts for models executed, inference passes and holdout reads are
zero by construction.

One consequence of measuring the repository is that the audit counts its own
artifacts. The first commit that adds them changes the index the previous run
measured, so the audit must be regenerated once after being committed; it then
reaches a fixed point and stays there, which is what
``test_the_inventory_matches_the_git_index`` pins. Excluding its own outputs
from the count was the alternative and was rejected: an inventory that quietly
omits five tracked files is a worse artifact than one that needs re-running.

A self-counting total invites a wrong reconciliation, so :func:`inventory_delta`
publishes the arithmetic instead of leaving it to be guessed. It names the
commit the phase started from, measures the tracked-file count there and here,
and keeps two quantities apart that are easy to confuse: the five artifacts this
phase writes, and the nine files it adds to the index. The validator refuses a
payload whose counts do not close.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
"""Version of the audit payload schema."""

PHASE = "12A"
"""The audit phase."""

CLASSIFICATION_COMPLETE = "FINAL_REPOSITORY_AUDIT_COMPLETE"
CLASSIFICATION_INCOMPLETE = "AUDIT_INCOMPLETE"
CLASSIFICATION_STATE_MISMATCH = "SCIENTIFIC_STATE_MISMATCH"
CLASSIFICATION_VIOLATION = "HOLDOUT_ACCESS_VIOLATION"
CLASSIFICATION_BLOCKED = "BLOCKED"

ENVIRONMENT_GATE = "CSVISION_ALLOW_TEST_SPLIT"
"""The environment half of the dual holdout gate. It must be absent here."""

AUDIT_JSON = "final_repository_audit.json"
AUDIT_MARKDOWN = "final_repository_audit.md"
GAP_CSV = "final_delivery_gap_register.csv"
COMPLIANCE_CSV = "assignment_compliance_matrix.csv"
AUDIT_PROVENANCE = "final_repository_audit.provenance.json"

AUDIT_OUTPUT_ARTIFACTS: tuple[str, ...] = (
    f"reports/{COMPLIANCE_CSV}",
    f"reports/{GAP_CSV}",
    f"reports/{AUDIT_JSON}",
    f"reports/{AUDIT_MARKDOWN}",
    f"reports/{AUDIT_PROVENANCE}",
)
"""The five artifacts this phase writes.

They are five of the nine files phase 12A adds to the index, never all of them:
the phase also adds the runner, two source modules and a test. Keeping the two
quantities apart is the point of :func:`inventory_delta`.
"""

PHASE_12A_BASELINE_COMMIT = "f1c78f25b8052638cf84cf37fc119b2af0635fbd"
"""The commit phase 12A started from, declared rather than derived.

``HEAD^`` answers correctly only while ``HEAD`` is the phase 12A commit itself,
so it would silently report a different baseline once a later phase lands on
top. Naming the commit fixes the baseline to a historical fact, and an
unreachable one is an error rather than a fallback.
"""

AREAS: tuple[str, ...] = (
    "src",
    "scripts",
    "tests",
    "configs",
    "reports",
    "notebooks",
    "data",
    "academic",
    "portfolio",
    "docs",
)
"""Top-level areas counted in the inventory. Absent ones report zero."""

SEVERITIES: tuple[str, ...] = ("P0", "P1", "P2", "P3")
"""P0 blocks delivery; P3 is optional polish."""

EFFORTS: tuple[str, ...] = ("SMALL", "MEDIUM", "LARGE")

PERSONAS: tuple[str, ...] = ("PROFESSOR", "RECRUITER", "ML_ENGINEER", "FIRST_TIME_USER")

COMPLIANCE_STATES: tuple[str, ...] = ("COMPLETE", "PARTIAL", "MISSING", "NOT_APPLICABLE")

CLAIM_VERDICTS: tuple[str, ...] = (
    "SUPPORTED",
    "SUPPORTED_WITH_LIMITATION",
    "OVERSTATED",
    "STALE",
    "UNSUPPORTED",
)

FILE_ROLES: tuple[str, ...] = (
    "PUBLIC_ENTRYPOINT",
    "SCIENTIFIC_RECORD",
    "REPRODUCIBILITY_SUPPORT",
    "INTERNAL_HISTORICAL",
    "DELIVERY",
)

PROFESSOR_VERDICTS = ("PROFESSOR_READY", "PROFESSOR_READY_WITH_GAPS", "NOT_PROFESSOR_READY")
RECRUITER_VERDICTS = ("RECRUITER_READY", "RECRUITER_READY_WITH_GAPS", "NOT_RECRUITER_READY")
ENGINEERING_VERDICTS = (
    "ENGINEERING_REVIEW_READY",
    "ENGINEERING_REVIEW_READY_WITH_GAPS",
    "NOT_ENGINEERING_REVIEW_READY",
)


class AuditError(RuntimeError):
    """Raised when the audit cannot be built or does not validate."""


def digest_payload(payload: Any) -> str:
    """Hash a payload's semantic content, deterministically.

    Args:
        payload: Any JSON-shaped value.

    Returns:
        A SHA-256 hex digest over its canonical serialisation.
    """
    text = json.dumps(
        json.loads(json.dumps(payload, sort_keys=True, default=str)),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def tracked_files(root: Path) -> tuple[str, ...]:
    """List every file git tracks, as repository-relative POSIX paths.

    Args:
        root: Repository root.

    Returns:
        The tracked paths, sorted.

    Raises:
        AuditError: If git cannot enumerate the index.
    """
    try:
        completed = subprocess.run(
            ["git", "ls-files"],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        msg = f"cannot enumerate tracked files: {exc}"
        raise AuditError(msg) from None
    return tuple(sorted(line for line in completed.stdout.splitlines() if line))


def inventory(files: Sequence[str]) -> dict[str, Any]:
    """Count tracked files by repository area.

    Args:
        files: Tracked repository-relative paths.

    Returns:
        Totals plus a per-area breakdown, including areas that do not exist.
    """
    by_area = dict.fromkeys(AREAS, 0)
    root_files = 0
    other: dict[str, int] = {}
    for name in files:
        head, _, rest = name.partition("/")
        if not rest:
            root_files += 1
        elif head in by_area:
            by_area[head] += 1
        else:
            other[head] = other.get(head, 0) + 1

    reports = [name for name in files if name.startswith("reports/")]
    return {
        "total_tracked_files": len(files),
        "by_area": by_area,
        "root_files": root_files,
        "other_top_level": dict(sorted(other.items())),
        "absent_areas": sorted(area for area, count in by_area.items() if count == 0),
        "reports_breakdown": {
            "total": len(reports),
            "figures": sum(1 for name in reports if name.startswith("reports/figures/")),
            "split_candidates": sum(
                1 for name in reports if name.startswith("reports/split_candidates/")
            ),
            "flat_documents": sum(1 for name in reports if name.count("/") == 1),
            "json": sum(1 for name in reports if name.endswith(".json")),
            "markdown": sum(1 for name in reports if name.endswith(".md")),
            "csv": sum(1 for name in reports if name.endswith(".csv")),
            "provenance_records": sum(1 for name in reports if name.endswith(".provenance.json")),
        },
        "counts_are_measured": True,
    }


def _tracked_at_commit(root: Path, commit: str) -> tuple[str, ...]:
    """List the files a commit tracks, as repository-relative POSIX paths.

    Args:
        root: Repository root.
        commit: Commit-ish to read the tree of.

    Returns:
        The tracked paths at that commit, sorted.

    Raises:
        AuditError: If the commit cannot be read.
    """
    try:
        completed = subprocess.run(
            ["git", "ls-tree", "-r", "--name-only", commit],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        msg = f"cannot read the tree of {commit}: {exc}"
        raise AuditError(msg) from None
    return tuple(sorted(line for line in completed.stdout.splitlines() if line))


def _differs_from_commit(root: Path, commit: str) -> frozenset[str]:
    """Name the tracked files whose current content differs from a commit's.

    Args:
        root: Repository root.
        commit: Commit-ish to compare against.

    Returns:
        Repository-relative paths that differ.

    Raises:
        AuditError: If git cannot compute the difference.
    """
    try:
        completed = subprocess.run(
            ["git", "diff", "--name-only", commit, "--"],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        msg = f"cannot diff against {commit}: {exc}"
        raise AuditError(msg) from None
    return frozenset(line for line in completed.stdout.splitlines() if line)


def inventory_delta(root: Path, files: Sequence[str]) -> dict[str, Any]:
    """Reconcile the tracked-file count against the commit this phase started from.

    The audit counts its own outputs, so the total alone invites a wrong
    reconciliation: the five artifacts this phase writes are not the number of
    files it adds. This block measures both quantities and keeps them named
    apart, so the arithmetic can be checked rather than assumed.

    Added and deleted sets are a difference of path sets, so a rename would
    appear as one addition plus one deletion. Nothing is renamed while the
    deleted set is empty, which the payload records.

    Args:
        root: Repository root.
        files: The currently tracked paths.

    Returns:
        The reconciliation, with every count measured from git.

    Raises:
        AuditError: If the arithmetic does not close.
    """
    baseline = _tracked_at_commit(root, PHASE_12A_BASELINE_COMMIT)
    current = tuple(files)
    added = sorted(set(current) - set(baseline))
    deleted = sorted(set(baseline) - set(current))
    shared = set(baseline) & set(current)
    modified = sorted(_differs_from_commit(root, PHASE_12A_BASELINE_COMMIT) & shared)

    outputs = sorted(name for name in added if name in AUDIT_OUTPUT_ARTIFACTS)
    support = sorted(name for name in added if name not in AUDIT_OUTPUT_ARTIFACTS)

    reconciled = len(baseline) + len(added) - len(deleted)
    if reconciled != len(current):
        msg = (
            f"the inventory does not reconcile: {len(baseline)} + {len(added)} - "
            f"{len(deleted)} is {reconciled}, not {len(current)}"
        )
        raise AuditError(msg)

    return {
        "baseline_commit": PHASE_12A_BASELINE_COMMIT,
        "baseline_is_declared_not_derived": True,
        "tracked_files_at_parent": len(baseline),
        "tracked_files_at_phase_12a_head": len(current),
        "delta_tracked_files": len(current) - len(baseline),
        "added": added,
        "added_count": len(added),
        "deleted": deleted,
        "deleted_count": len(deleted),
        "modified": modified,
        "modified_count": len(modified),
        "audit_output_artifacts": outputs,
        "audit_output_artifacts_added": len(outputs),
        "implementation_support_files": support,
        "implementation_support_files_added": len(support),
        "live_docs_modified": len(modified),
        "audit_outputs_are_not_the_file_delta": len(outputs) != len(added),
        "rename_detection": "SET_DIFFERENCE_TREATS_A_RENAME_AS_ONE_ADD_PLUS_ONE_DELETE",
        "renames_or_copies_possible": bool(deleted),
        "reconciles": True,
        "counts_are_measured": True,
    }


# --- what a delivery needs, and whether it exists ---------------------------------------


DELIVERABLE_PROBES: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("software_license_file", ("LICENSE", "LICENSE.md", "LICENSE.txt", "COPYING"), "C6"),
    ("technical_report", ("reports/technical_report.md", "reports/technical_report.pdf"), "C6"),
    ("error_analysis_report", ("reports/error_analysis.md",), "C4"),
    ("video_inference_script", ("scripts/run_video_inference.py",), "C5"),
    ("video_analysis_report", ("reports/video_analysis.md",), "C5"),
    ("pitch_script", ("reports/pitch_script.md",), "C7"),
    ("genai_declaration", ("reports/genai_declaration.md", "GENAI.md"), "C6"),
    ("tracking_notes", ("reports/tracking_notes.md",), "B1"),
    ("reproducibility_audit", ("reports/reproducibility_audit.md",), "C6"),
)
"""Named deliverables, the paths that would satisfy them, and the criterion."""


def deliverables(root: Path, files: Sequence[str]) -> dict[str, Any]:
    """Check which declared delivery artifacts exist.

    Args:
        root: Repository root.
        files: Tracked repository-relative paths.

    Returns:
        Per-deliverable presence, plus notebook and qualitative-gallery checks.
    """
    tracked = set(files)
    found: dict[str, Any] = {}
    for name, candidates, criterion in DELIVERABLE_PROBES:
        hit = next((path for path in candidates if path in tracked), None)
        found[name] = {
            "criterion": criterion,
            "exists": hit is not None,
            "path": hit,
            "candidates_checked": list(candidates),
        }

    notebooks = sorted(name for name in files if name.endswith(".ipynb"))
    galleries = sorted(
        name for name in files if name.startswith("reports/figures/") and "error" in name.lower()
    )
    return {
        "probes": found,
        "notebooks_tracked": notebooks,
        "executable_notebook_exists": bool(notebooks),
        "qualitative_error_figures_tracked": galleries,
        "qualitative_error_gallery_exists": bool(galleries),
        "readme_exists": (root / "README.md").is_file(),
        "roadmap_exists": (root / "reports" / "roadmap.md").is_file(),
        "rubric_contract_exists": (root / "reports" / "rubric_contract.md").is_file(),
    }


# --- headline results, read by field then searched for in the documentation --------------


HEADLINES: tuple[tuple[str, str, str], ...] = (
    (
        "d2_test_box_map50_95",
        "final_test_detector.json",
        "canonical_box.CANONICAL_TEST_BOX_MAP50_95",
    ),
    ("d2_test_box_map50", "final_test_detector.json", "canonical_box.CANONICAL_TEST_BOX_MAP50"),
    (
        "s1_test_mask_map50_95",
        "final_test_segmenter.json",
        "canonical_mask.CANONICAL_TEST_MASK_MAP50_95",
    ),
    (
        "s1_test_box_map50_95",
        "final_test_segmenter.json",
        "canonical_box.S1_CANONICAL_TEST_BOX_MAP50_95",
    ),
    (
        "test_matched_mask_iou_mean",
        "final_test_direct_iou.json",
        "diagnostic.global.matched_mask_iou_mean",
    ),
    (
        "test_gt_normalized_mask_iou",
        "final_test_direct_iou.json",
        "diagnostic.global.gt_normalized_mask_iou",
    ),
    ("d2_test_true_positives", "final_test_detector.json", "object_level.true_positives"),
    ("s1_test_true_positives", "final_test_segmenter.json", "object_level.true_positives"),
)
"""Final-test headlines whose values the live documentation must carry verbatim."""


def _field(payload: Mapping[str, Any], dotted: str) -> Any:
    """Read one value by its dotted field path.

    Args:
        payload: A committed artifact.
        dotted: The field path.

    Returns:
        The value at that path.

    Raises:
        AuditError: If any segment is absent.
    """
    current: Any = payload
    for segment in dotted.split("."):
        if not isinstance(current, Mapping) or segment not in current:
            msg = f"expected field {dotted!r} is absent"
            raise AuditError(msg)
        current = current[segment]
    return current


def _read_json(path: Path) -> dict[str, Any]:
    """Load a committed JSON artifact.

    Args:
        path: The artifact.

    Returns:
        Its parsed content.

    Raises:
        AuditError: If it is missing or is not a JSON object.
    """
    if not path.is_file():
        msg = f"a committed artifact this audit reads is missing: {path.name}"
        raise AuditError(msg)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        msg = f"{path.name} is not a JSON object"
        raise AuditError(msg)
    return payload


def result_consistency(root: Path) -> dict[str, Any]:
    """Check that the live documentation carries the committed final numbers.

    Args:
        root: Repository root.

    Returns:
        Per-headline value, provenance and where it is rendered.
    """
    reports = root / "reports"
    live = {
        "README.md": (root / "README.md").read_text(encoding="utf-8"),
        "reports/roadmap.md": (reports / "roadmap.md").read_text(encoding="utf-8"),
        "reports/final_test_evaluation.md": (reports / "final_test_evaluation.md").read_text(
            encoding="utf-8"
        ),
    }
    cache: dict[str, dict[str, Any]] = {}
    checks: dict[str, Any] = {}
    for label, artifact, dotted in HEADLINES:
        if artifact not in cache:
            cache[artifact] = _read_json(reports / artifact)
        value = _field(cache[artifact], dotted)
        rendered = f"{value:.6f}" if isinstance(value, float) else str(value)
        present = sorted(name for name, text in live.items() if rendered in text)
        checks[label] = {
            "value": value,
            "rendered": rendered,
            "evidence_artifact": f"reports/{artifact}",
            "evidence_field": dotted,
            "documented_in": present,
            "present_in_readme": "README.md" in present,
        }
    missing = sorted(label for label, check in checks.items() if not check["present_in_readme"])
    return {
        "headlines": checks,
        "headlines_absent_from_readme": missing,
        "all_headlines_documented": not missing,
        "values_read_by_field": True,
        "values_transcribed": False,
    }


# --- claims that were true once and are not any more -------------------------------------


STALE_PROBES: tuple[dict[str, str], ...] = (
    {
        "probe_id": "STALE-01",
        "file": "README.md",
        "needle": "No model has been trained and no evaluation has been run",
        "why_false": (
            "Five models were trained (D0, D1, D2, S0, S1), two are frozen, and both the "
            "validation comparison and the one-shot holdout evaluation are complete."
        ),
        "contradicts": "the README's own status block and every phase 6-11 section below it",
    },
    {
        "probe_id": "STALE-02",
        "file": "README.md",
        "needle": "No test metric exists.",
        "why_false": (
            "Phase 11B measured the holdout once; the test metrics are committed in "
            "reports/final_test_detector.json and reports/final_test_segmenter.json."
        ),
        "contradicts": "the README's Final test metrics row in the same table",
    },
    {
        "probe_id": "STALE-03",
        "file": "README.md",
        "needle": "no test number exists",
        "why_false": (
            "True when the detector and segmenter were frozen, but stated in the present "
            "tense in a status table read as current."
        ),
        "contradicts": "the README's Final test metrics row in the same table",
    },
    {
        "probe_id": "STALE-04",
        "file": "reports/roadmap.md",
        "needle": "spatial `9877b88d...`",
        "why_false": (
            "The association-taxonomy correction rewrote the section that digest covers. The "
            "authoritative spatial fingerprint is 3988bcf6..., recorded in "
            "reports/association_taxonomy_correction.provenance.json; CLAUDE.md states the "
            "pre-correction value is superseded and must not be quoted."
        ),
        "contradicts": "reports/association_taxonomy_correction.provenance.json",
    },
    {
        "probe_id": "STALE-05",
        "file": "README.md",
        "needle": "**locked holdout**: it has never been",
        "why_false": (
            "The holdout was read once in phase 11B; it is spent, not pending, and the same "
            "blockquote goes on to quote the resulting test metrics. The sentence also "
            "asserts that every number below it is a validation number, which is no longer "
            "true."
        ),
        "contradicts": "the README's own phase 11B section and status rows",
    },
    {
        "probe_id": "STALE-06",
        "file": "README.md",
        "needle": "## Planned architecture",
        "why_false": (
            "The architecture was built, frozen and evaluated. Presenting it as planned "
            "understates the work to every reader."
        ),
        "contradicts": "phases 5-11, all complete",
    },
    {
        "probe_id": "STALE-07",
        "file": "README.md",
        "needle": "## Planned methodology",
        "why_false": "Eleven of the fourteen planned phases are complete.",
        "contradicts": "reports/roadmap.md phase status table",
    },
    {
        "probe_id": "STALE-08",
        "file": ".env.example",
        "needle": "phase 3 - not used yet",
        "why_false": "Phase 3 acquired the dataset; the key is used by four committed scripts.",
        "contradicts": "reports/dataset_provenance.md",
    },
)
"""Exact strings whose presence in live documentation is now a defect."""


def stale_claims(root: Path) -> dict[str, Any]:
    """Probe the live documentation for claims that are no longer true.

    Args:
        root: Repository root.

    Returns:
        Per-probe presence and occurrence count.
    """
    results = []
    for probe in STALE_PROBES:
        path = root / probe["file"]
        text = path.read_text(encoding="utf-8") if path.is_file() else ""
        occurrences = text.count(probe["needle"])
        results.append({**probe, "present": occurrences > 0, "occurrences": occurrences})
    present = [entry for entry in results if entry["present"]]
    return {
        "probes": results,
        "probes_total": len(results),
        "probes_present": len(present),
        "probe_ids_present": sorted(entry["probe_id"] for entry in present),
        "detection_is_computed": True,
    }


# --- the scientific lock this audit must not disturb -------------------------------------


def scientific_lock(root: Path, env: Mapping[str, str]) -> dict[str, Any]:
    """Verify the experimental state is frozen and the holdout shut.

    Args:
        root: Repository root.
        env: Environment mapping to read.

    Returns:
        The lock state, read from committed artifacts and the environment.
    """
    reports = root / "reports"
    provenance = _read_json(reports / "final_test_evaluation.provenance.json")
    accounting = _read_json(reports / "final_test_execution_accounting.json")
    policy = _field(provenance, "details.post_test_policy")
    claude = (root / "CLAUDE.md").read_text(encoding="utf-8")
    return {
        "environment_gate_variable": ENVIRONMENT_GATE,
        "environment_gate_present_in_process": env.get(ENVIRONMENT_GATE) is not None,
        "holdout": _field(provenance, "details.holdout"),
        "holdout_reads_permitted": _field(provenance, "details.holdout_reads_permitted"),
        "holdout_reads_performed": _field(provenance, "details.holdout_reads_performed"),
        "final_test_state": policy["state"],
        "model_selection": policy["model_selection"],
        "hyperparameter_tuning": policy["hyperparameter_tuning"],
        "threshold_tuning": policy["threshold_tuning"],
        "data_cleaning_for_performance": policy["data_cleaning_for_performance"],
        "training_closed_recorded_in_claude_md": "never performance-driven model development"
        in claude,
        "holdout_evaluation_attempts": _field(
            accounting, "inference_accounting.holdout_evaluation_attempts"
        ),
        "unique_models_executed": _field(accounting, "inference_accounting.unique_models_executed"),
        "total_model_inference_passes": _field(
            accounting, "inference_accounting.total_model_inference_passes"
        ),
        "final_detector": "D2 / YOLO11n / imgsz 768",
        "final_segmenter": "S1 / YOLO11n-seg / imgsz 768 / overlap_mask false",
    }


def this_audit() -> dict[str, Any]:
    """Count what the audit itself did to the models and the holdout: nothing.

    Returns:
        The self-accounting block.
    """
    return {
        "contains_new_scientific_results": False,
        "holdout_accessed": False,
        "holdout_accessor_invoked": False,
        "holdout_content_read": 0,
        "metrics_recomputed": 0,
        "model_inference_passes": 0,
        "models_executed": 0,
        "readme_rewritten": False,
        "report_generated": False,
        "scientific_results_modified": False,
        "test_identifiers_enumerated": 0,
        "thresholds_tuned": 0,
        "video_started": False,
    }


# --- assembly ---------------------------------------------------------------------------


def build(paths: Any, *, env: Mapping[str, str]) -> dict[str, Any]:
    """Assemble the phase 12A audit from measurements plus declared findings.

    Args:
        paths: A :class:`~construction_safety_vision.paths.ProjectPaths`.
        env: Environment mapping to read.

    Returns:
        The audit payload, carrying its own fingerprint.

    Raises:
        AuditError: If a committed artifact this audit reads is missing.
    """
    from construction_safety_vision import delivery_audit_findings as declared

    root = Path(paths.root)
    files = tracked_files(root)

    measured = {
        "deliverables": deliverables(root, files),
        "inventory": inventory(files),
        "inventory_delta": inventory_delta(root, files),
        "result_consistency": result_consistency(root),
        "scientific_lock": scientific_lock(root, env),
        "stale_claims": stale_claims(root),
    }
    judged = {
        "claim_audit": list(declared.claim_audit()),
        "clean_room_plan": declared.clean_room_plan(),
        "colab_plan": declared.colab_plan(),
        "compliance_matrix": list(declared.compliance_matrix()),
        "delivery_roadmap": list(declared.delivery_roadmap()),
        "diagrams": list(declared.diagrams()),
        "file_routing": list(declared.file_routing()),
        "gap_register": list(declared.gap_register()),
        "personas": list(declared.personas()),
        "pitch_plan": declared.pitch_plan(),
        "portfolio_signals": list(declared.portfolio_signals()),
        "positioning": declared.positioning(),
        "readme_plan": declared.readme_plan(),
        "report_plan": declared.report_plan(),
        "video_readiness": declared.video_readiness(),
        "visual_assets": declared.visual_assets(),
    }

    states = [row["state"] for row in judged["compliance_matrix"]]
    severities = [gap["severity"] for gap in judged["gap_register"]]
    verdicts = {entry["persona"]: entry["verdict"] for entry in judged["personas"]}

    payload: dict[str, Any] = {
        "classification": CLASSIFICATION_COMPLETE,
        "declared_findings": judged,
        "evidence_separation": {
            "declared_blocks": sorted(judged),
            "judgements_are_editorial": True,
            "measured_blocks": sorted(measured),
            "measurements_recompute_on_every_run": True,
        },
        "measurements": measured,
        "phase": PHASE,
        "readiness": {
            "engineering": verdicts["ML_ENGINEER"],
            "first_time_user": verdicts["FIRST_TIME_USER"],
            "professor": verdicts["PROFESSOR"],
            "recruiter": verdicts["RECRUITER"],
        },
        "schema_version": SCHEMA_VERSION,
        "summary": {
            "compliance": {state: states.count(state) for state in COMPLIANCE_STATES},
            "gaps": {severity: severities.count(severity) for severity in SEVERITIES},
            "mandatory_gaps_open": sum(1 for gap in judged["gap_register"] if gap["mandatory"]),
            "stale_probes_present": measured["stale_claims"]["probes_present"],
            "total_tracked_files": measured["inventory"]["total_tracked_files"],
        },
        "this_audit": this_audit(),
    }
    payload["repository_audit_sha256"] = digest_payload(payload)
    return payload


def _walk(payload: Any, path: str = "") -> list[tuple[str, str, Any]]:
    """Yield every leaf as ``(path, key, value)``.

    Args:
        payload: Any JSON-shaped value.
        path: The accumulated path.

    Returns:
        One entry per leaf.
    """
    found: list[tuple[str, str, Any]] = []
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            where = f"{path}.{key}" if path else str(key)
            found.append((where, str(key), value))
            found.extend(_walk(value, where))
    elif isinstance(payload, Sequence) and not isinstance(payload, str | bytes):
        for index, value in enumerate(payload):
            found.extend(_walk(value, f"{path}[{index}]"))
    return found


def validate(payload: Mapping[str, Any]) -> list[str]:
    """Check the audit adversarially.

    Args:
        payload: The audit payload.

    Returns:
        One description per problem found, empty when the payload is sound.
    """
    problems: list[str] = []

    if payload.get("phase") != PHASE:
        problems.append(f"phase must be {PHASE}")

    mine = payload.get("this_audit", {})
    for key, value in this_audit().items():
        if mine.get(key) != value:
            problems.append(f"this_audit.{key} must be {value!r}")

    lock = payload.get("measurements", {}).get("scientific_lock", {})
    if lock.get("environment_gate_present_in_process") is not False:
        problems.append("the holdout environment gate must be absent while auditing")
    if lock.get("final_test_state") != "FINAL_TEST_OBSERVED":
        problems.append("the final test must be recorded as observed")
    for key in (
        "model_selection",
        "hyperparameter_tuning",
        "threshold_tuning",
        "data_cleaning_for_performance",
    ):
        if lock.get(key) != "CLOSED":
            problems.append(f"scientific_lock.{key} must be CLOSED")
    if lock.get("holdout_reads_performed") != 1:
        problems.append("exactly one holdout read must be recorded")
    if lock.get("holdout_evaluation_attempts") != 1:
        problems.append("exactly one holdout evaluation attempt must be recorded")
    if lock.get("total_model_inference_passes") != 3:
        problems.append("three holdout inference passes must be recorded")

    findings = payload.get("declared_findings", {})
    for row in findings.get("compliance_matrix", []):
        if row.get("state") not in COMPLIANCE_STATES:
            problems.append(f"{row.get('requirement_id')} has an unknown compliance state")
        if not row.get("evidence_location"):
            problems.append(f"{row.get('requirement_id')} must cite where its evidence lives")
        if row.get("state") in {"PARTIAL", "MISSING"} and row.get("remaining_action") in (
            None,
            "",
            "none",
        ):
            problems.append(f"{row.get('requirement_id')} is not COMPLETE but names no action")
        if row.get("state") == "COMPLETE" and row.get("evidence_artifact") == "none":
            problems.append(f"{row.get('requirement_id')} is COMPLETE with no evidence artifact")

    personas_seen = [entry.get("persona") for entry in findings.get("personas", [])]
    if sorted(personas_seen) != sorted(PERSONAS):
        problems.append(f"all four personas must be audited, found {personas_seen}")
    for entry in findings.get("personas", []):
        if not entry.get("strengths") or not entry.get("gaps"):
            problems.append(f"{entry.get('persona')} must record both strengths and gaps")
        if not entry.get("verdict_reason"):
            problems.append(f"{entry.get('persona')} must justify its verdict")

    readiness = payload.get("readiness", {})
    if readiness.get("professor") not in PROFESSOR_VERDICTS:
        problems.append("the professor verdict must be one of the three declared values")
    if readiness.get("recruiter") not in RECRUITER_VERDICTS:
        problems.append("the recruiter verdict must be one of the three declared values")
    if readiness.get("engineering") not in ENGINEERING_VERDICTS:
        problems.append("the engineering verdict must be one of the three declared values")

    for entry in findings.get("claim_audit", []):
        if entry.get("verdict") not in CLAIM_VERDICTS:
            problems.append(f"{entry.get('claim_id')} has an unknown claim verdict")
        if not entry.get("what_the_evidence_supports"):
            problems.append(f"{entry.get('claim_id')} must say what the evidence supports")

    for gap in findings.get("gap_register", []):
        if gap.get("severity") not in SEVERITIES:
            problems.append(f"{gap.get('gap_id')} has an unknown severity")
        if gap.get("effort") not in EFFORTS:
            problems.append(f"{gap.get('gap_id')} has an unknown effort estimate")
        if not gap.get("evidence"):
            problems.append(f"{gap.get('gap_id')} must cite evidence")
        if not gap.get("recommended_fix"):
            problems.append(f"{gap.get('gap_id')} must recommend a fix")
        for persona in gap.get("persona", []):
            if persona not in PERSONAS:
                problems.append(f"{gap.get('gap_id')} names an unknown persona {persona}")

    for entry in findings.get("file_routing", []):
        if entry.get("role") not in FILE_ROLES:
            problems.append(f"unknown file role {entry.get('role')}")

    positioning = findings.get("positioning", {})
    if positioning.get("production_claim_permitted") is not False:
        problems.append("a production-readiness claim must not be permitted")
    if positioning.get("production_oriented_assessed") is not True:
        problems.append("the production-oriented wording must be explicitly assessed")

    measurements = payload.get("measurements", {})
    delta = measurements.get("inventory_delta", {})
    if not delta:
        problems.append("the inventory must reconcile against the phase baseline")
    else:
        parent = delta.get("tracked_files_at_parent")
        head = delta.get("tracked_files_at_phase_12a_head")
        added = delta.get("added_count")
        deleted = delta.get("deleted_count")
        if None in (parent, head, added, deleted):
            problems.append("the inventory delta must carry all four counts")
        elif parent + added - deleted != head:
            problems.append(
                f"the inventory delta must reconcile: {parent} + {added} - {deleted} is not {head}"
            )
        if added != len(delta.get("added", [])):
            problems.append("the declared added-file count must equal the files listed")
        if deleted != len(delta.get("deleted", [])):
            problems.append("the declared deleted-file count must equal the files listed")
        if delta.get("modified_count") != len(delta.get("modified", [])):
            problems.append("the declared modified-file count must equal the files listed")
        outputs = delta.get("audit_output_artifacts_added")
        support = delta.get("implementation_support_files_added")
        if None in (outputs, support) or outputs + support != added:
            problems.append(
                "every added file must be either an audit output or implementation support"
            )
        if outputs != len(delta.get("audit_output_artifacts", [])):
            problems.append("the audit-output count must equal the artifacts listed")
        if delta.get("live_docs_modified") != delta.get("modified_count"):
            problems.append("live_docs_modified must equal the measured modified count")
        if head != measurements.get("inventory", {}).get("total_tracked_files"):
            problems.append("the delta's head count must equal the inventory total")

    for where, key, value in _walk(payload):
        if key in {"holdout_accessed", "scientific_results_modified"} and value not in (
            False,
            None,
        ):
            problems.append(f"{where} must be false, found {value!r}")

    return problems


def classify(payload: Mapping[str, Any]) -> str:
    """Classify the audit's own outcome.

    Args:
        payload: The audit payload.

    Returns:
        One of the phase 12A classification constants.
    """
    lock = payload.get("measurements", {}).get("scientific_lock", {})
    if lock.get("environment_gate_present_in_process"):
        return CLASSIFICATION_VIOLATION
    if payload.get("this_audit", {}).get("holdout_accessed"):
        return CLASSIFICATION_VIOLATION
    if lock.get("final_test_state") != "FINAL_TEST_OBSERVED":
        return CLASSIFICATION_STATE_MISMATCH
    if validate(payload):
        return CLASSIFICATION_INCOMPLETE
    return CLASSIFICATION_COMPLETE


# --- rendering --------------------------------------------------------------------------


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    """Render a Markdown table.

    Args:
        header: Column titles.
        rows: Row cells, already stringified.

    Returns:
        The table's lines.
    """
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def _render_head(payload: Mapping[str, Any]) -> list[str]:
    """Render the header, scientific lock and inventory.

    Args:
        payload: The audit payload.

    Returns:
        Markdown lines.
    """
    measured = payload["measurements"]
    lock = measured["scientific_lock"]
    inv = measured["inventory"]
    delta = measured["inventory_delta"]
    summary = payload["summary"]
    # Sorted, not insertion-ordered: the payload round-trips through JSON with
    # sorted keys, so an insertion-ordered render would not match its own file.
    areas = [
        [f"`{area}/`", str(count)]
        for area, count in sorted(inv["by_area"].items())
        if count or area in {"academic", "portfolio", "docs", "notebooks"}
    ]
    areas.append(["root files", str(inv["root_files"])])
    return [
        "# Final repository, academic and portfolio audit",
        "",
        f"**Phase {payload['phase']}. `{payload['classification']}`. Audit only.**",
        "",
        "This phase executed no model, read no holdout content, recomputed no metric and "
        "changed no scientific result. It measures the repository and judges its delivery "
        "readiness; the two are kept apart, and the measurements re-derive on every run so "
        "the audit cannot itself go stale.",
        "",
        "## 1. Scientific state",
        "",
        *_table(
            ["", "Value"],
            [
                ["Final test", f"`{lock['final_test_state']}`"],
                ["Holdout", f"`{lock['holdout']}`"],
                [
                    "Holdout reads",
                    f"{lock['holdout_reads_performed']} of "
                    f"{lock['holdout_reads_permitted']} permitted",
                ],
                ["Evaluation attempts", str(lock["holdout_evaluation_attempts"])],
                [
                    "Models executed / inference passes",
                    f"{lock['unique_models_executed']} / {lock['total_model_inference_passes']}",
                ],
                [
                    f"`{lock['environment_gate_variable']}` in process",
                    "absent" if not lock["environment_gate_present_in_process"] else "PRESENT",
                ],
                ["Model selection", f"`{lock['model_selection']}`"],
                ["Hyperparameter tuning", f"`{lock['hyperparameter_tuning']}`"],
                ["Threshold tuning", f"`{lock['threshold_tuning']}`"],
                ["Data cleaning for performance", f"`{lock['data_cleaning_for_performance']}`"],
                ["Final detector", lock["final_detector"]],
                ["Final segmenter", lock["final_segmenter"]],
            ],
        ),
        "",
        "## 2. Repository inventory",
        "",
        f"**{inv['total_tracked_files']} tracked files.**",
        "",
        *_table(["Area", "Files"], areas),
        "",
        f"`reports/` breaks down as {inv['reports_breakdown']['flat_documents']} flat "
        f"documents, {inv['reports_breakdown']['figures']} figures and "
        f"{inv['reports_breakdown']['split_candidates']} split-candidate tables, including "
        f"{inv['reports_breakdown']['provenance_records']} provenance records. Absent areas: "
        f"{', '.join('`' + name + '/`' for name in inv['absent_areas']) or 'none'}.",
        "",
        "### 2.1 Reconciliation against the commit this phase started from",
        "",
        f"The audit counts its own outputs. The five artifacts it writes are **not** the "
        f"number of files the phase adds, so both quantities are measured and named apart. "
        f"Baseline `{delta['baseline_commit'][:12]}`.",
        "",
        *_table(
            ["Quantity", "Value"],
            [
                ["Tracked files at parent", str(delta["tracked_files_at_parent"])],
                ["Tracked files at phase 12A head", str(delta["tracked_files_at_phase_12a_head"])],
                ["Delta", f"{delta['delta_tracked_files']:+d}"],
                ["Files added", str(delta["added_count"])],
                ["Files deleted", str(delta["deleted_count"])],
                ["Live documents modified", str(delta["live_docs_modified"])],
                ["of the additions: audit outputs", str(delta["audit_output_artifacts_added"])],
                [
                    "of the additions: implementation support",
                    str(delta["implementation_support_files_added"]),
                ],
            ],
        ),
        "",
        f"{delta['tracked_files_at_parent']} + {delta['added_count']} - "
        f"{delta['deleted_count']} = {delta['tracked_files_at_phase_12a_head']}. Added and "
        f"deleted are a difference of path sets, so a rename would appear as one addition "
        f"plus one deletion; the deleted set is empty, so none occurred.",
        "",
        f"Added ({delta['added_count']}):",
        "",
        *[f"- `{name}`" for name in delta["added"]],
        "",
        f"Modified ({delta['modified_count']}):",
        "",
        *[f"- `{name}`" for name in delta["modified"]],
        "",
        f"Summary: {summary['compliance']['COMPLETE']} requirements COMPLETE, "
        f"{summary['compliance']['PARTIAL']} PARTIAL, {summary['compliance']['MISSING']} "
        f"MISSING; {summary['gaps']['P0']} P0 gaps, {summary['gaps']['P1']} P1, "
        f"{summary['gaps']['P2']} P2, {summary['gaps']['P3']} P3; "
        f"{summary['stale_probes_present']} stale-claim probes currently firing.",
        "",
    ]


def _render_compliance(payload: Mapping[str, Any]) -> list[str]:
    """Render the assignment compliance matrix and the result consistency check.

    Args:
        payload: The audit payload.

    Returns:
        Markdown lines.
    """
    findings = payload["declared_findings"]
    consistency = payload["measurements"]["result_consistency"]
    stale = payload["measurements"]["stale_claims"]
    lines = [
        "## 3. Assignment compliance",
        "",
        *_table(
            ["ID", "Requirement", "Criterion", "State", "Evidence", "Remaining action"],
            [
                [
                    row["requirement_id"],
                    row["requirement"],
                    row["criterion"],
                    f"**{row['state']}**",
                    f"`{row['evidence_artifact']}`",
                    row["remaining_action"],
                ]
                for row in findings["compliance_matrix"]
            ],
        ),
        "",
        "## 4. Result consistency",
        "",
        "Every headline is read by field from a committed artifact and then searched for "
        "verbatim in the live documentation.",
        "",
        *_table(
            ["Headline", "Value", "Source field", "Documented in"],
            [
                [
                    label,
                    consistency["headlines"][label]["rendered"],
                    f"`{consistency['headlines'][label]['evidence_field']}`",
                    ", ".join(consistency["headlines"][label]["documented_in"]) or "**nowhere**",
                ]
                # The declared order, not the dict's: the payload round-trips
                # through JSON with sorted keys and the render must match its file.
                for label, _, _ in HEADLINES
            ],
        ),
        "",
        "## 5. Stale claims",
        "",
    ]
    firing = [probe for probe in stale["probes"] if probe["present"]]
    if firing:
        lines.extend(
            _table(
                ["Probe", "File", "String", "Why it is now false"],
                [
                    [
                        probe["probe_id"],
                        f"`{probe['file']}`",
                        f"`{probe['needle']}`",
                        probe["why_false"],
                    ]
                    for probe in firing
                ],
            )
        )
    else:
        lines.append("No stale-claim probe fires. Every probed string has been retired.")
    lines.append("")
    return lines


def _render_personas(payload: Mapping[str, Any]) -> list[str]:
    """Render the four persona audits.

    Args:
        payload: The audit payload.

    Returns:
        Markdown lines.
    """
    lines = ["## 6. Four perspectives", ""]
    for entry in payload["declared_findings"]["personas"]:
        lines.extend(
            [
                f"### {entry['persona'].replace('_', ' ').title()} - **{entry['verdict']}**",
                "",
                entry["assessment"],
                "",
                "**Strengths**",
                "",
                *[f"- {item}" for item in entry["strengths"]],
                "",
                "**Gaps**",
                "",
                *[f"- {item}" for item in entry["gaps"]],
                "",
                f"*Verdict:* {entry['verdict_reason']}",
                "",
            ]
        )
    return lines


def _render_claims(payload: Mapping[str, Any]) -> list[str]:
    """Render the claim audit and the ranked portfolio signals.

    Args:
        payload: The audit payload.

    Returns:
        Markdown lines.
    """
    findings = payload["declared_findings"]
    lines = [
        "## 7. Claim audit",
        "",
        *_table(
            ["ID", "Claim", "Verdict", "What the evidence supports"],
            [
                [
                    entry["claim_id"],
                    entry["claim"],
                    f"**{entry['verdict']}**",
                    entry["what_the_evidence_supports"],
                ]
                for entry in findings["claim_audit"]
            ],
        ),
        "",
        "## 8. Portfolio signals, ranked",
        "",
    ]
    for signal in findings["portfolio_signals"]:
        lines.extend(
            [
                f"**{signal['rank']}. {signal['signal']}**",
                "",
                f"*What was done.* {signal['what_was_done']}",
                "",
                f"*Why it matters.* {signal['why_it_matters']}",
                "",
                f"*Skill demonstrated.* {signal['skill_demonstrated']}",
                "",
            ]
        )
    return lines


def _render_plans(payload: Mapping[str, Any]) -> list[str]:
    """Render positioning, the README plan, diagrams and the delivery sequence.

    Args:
        payload: The audit payload.

    Returns:
        Markdown lines.
    """
    findings = payload["declared_findings"]
    positioning = findings["positioning"]
    readme = findings["readme_plan"]
    lines = [
        "## 9. Positioning",
        "",
        *_table(
            ["", "Recommendation"],
            [
                ["H1", f"**{positioning['recommended_h1']}**"],
                ["Subtitle", positioning["recommended_subtitle"]],
                ["Tagline", positioning["recommended_tagline"]],
            ],
        ),
        "",
        f"*Elevator pitch.* {positioning['recommended_elevator_pitch']}",
        "",
        "**Alternatives assessed**",
        "",
        *[
            f"- *{alt['candidate']}* - **{alt['verdict']}**. {alt['reason']}"
            for alt in positioning["assessed_alternatives"]
        ],
        "",
        "## 10. README structure",
        "",
        f"The current README is {readme['current_lines']} lines shaped as a "
        f"`{readme['current_shape']}`. Recommended structure:",
        "",
        *[f"{index}. {item}" for index, item in enumerate(readme["recommended_structure"], 1)],
        "",
        f"*Rule.* {readme['rule']}",
        "",
        "## 11. Diagrams",
        "",
        *_table(
            ["ID", "Diagram", "Audience", "Placement", "Priority"],
            [
                [
                    entry["diagram_id"],
                    entry["name"],
                    entry["audience"],
                    entry["placement"],
                    entry["priority"],
                ]
                for entry in findings["diagrams"]
            ],
        ),
        "",
        "## 12. Delivery sequence",
        "",
        *_table(
            ["Phase", "Name", "Mandatory", "Closes"],
            [
                [
                    entry["phase"],
                    entry["name"],
                    "yes" if entry["mandatory"] else "optional",
                    ", ".join(entry["closes"]) or "-",
                ]
                for entry in findings["delivery_roadmap"]
            ],
        ),
        "",
    ]
    return lines


def _render_gaps(payload: Mapping[str, Any]) -> list[str]:
    """Render the gap register and the audit's self-accounting.

    Args:
        payload: The audit payload.

    Returns:
        Markdown lines.
    """
    findings = payload["declared_findings"]
    mine = payload["this_audit"]
    lines = ["## 13. Delivery gap register", ""]
    for severity in SEVERITIES:
        gaps = [gap for gap in findings["gap_register"] if gap["severity"] == severity]
        if not gaps:
            continue
        lines.extend(
            [
                f"### {severity}",
                "",
                *_table(
                    ["ID", "Gap", "Persona", "Effort", "Dependency", "Mandatory"],
                    [
                        [
                            gap["gap_id"],
                            gap["title"],
                            ", ".join(gap["persona"]),
                            gap["effort"],
                            gap["dependency"],
                            "yes" if gap["mandatory"] else "no",
                        ]
                        for gap in gaps
                    ],
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## 14. What this audit did",
            "",
            *_table(
                ["", "Value"],
                [
                    ["Models executed", str(mine["models_executed"])],
                    ["Model inference passes", str(mine["model_inference_passes"])],
                    ["Holdout accessed", "no"],
                    ["Holdout content read", str(mine["holdout_content_read"])],
                    ["Metrics recomputed", str(mine["metrics_recomputed"])],
                    ["Scientific results modified", "no"],
                    ["README rewritten", "no"],
                    ["Report generated", "no"],
                    ["Video started", "no"],
                ],
            ),
            "",
            f"Audit fingerprint `{payload['repository_audit_sha256']}`.",
            "",
        ]
    )
    return lines


def render(payload: Mapping[str, Any]) -> str:
    """Render the audit as Markdown.

    Args:
        payload: The audit payload.

    Returns:
        The document text.
    """
    return "\n".join(
        [
            *_render_head(payload),
            *_render_compliance(payload),
            *_render_personas(payload),
            *_render_claims(payload),
            *_render_plans(payload),
            *_render_gaps(payload),
        ]
    )


def gap_rows(payload: Mapping[str, Any]) -> list[list[str]]:
    """Render the gap register as CSV rows, header first.

    Args:
        payload: The audit payload.

    Returns:
        Rows ready for csv.writer.
    """
    header = [
        "gap_id",
        "severity",
        "persona",
        "title",
        "evidence",
        "recommended_fix",
        "effort",
        "dependency",
        "mandatory",
    ]
    order = {severity: index for index, severity in enumerate(SEVERITIES)}
    gaps = sorted(
        payload["declared_findings"]["gap_register"],
        key=lambda gap: (order[gap["severity"]], gap["gap_id"]),
    )
    rows = [header]
    rows.extend(
        [
            gap["gap_id"],
            gap["severity"],
            "; ".join(gap["persona"]),
            gap["title"],
            gap["evidence"],
            gap["recommended_fix"],
            gap["effort"],
            gap["dependency"],
            "yes" if gap["mandatory"] else "no",
        ]
        for gap in gaps
    )
    return rows


def compliance_rows(payload: Mapping[str, Any]) -> list[list[str]]:
    """Render the compliance matrix as CSV rows, header first.

    Args:
        payload: The audit payload.

    Returns:
        Rows ready for csv.writer.
    """
    header = [
        "requirement_id",
        "requirement",
        "criterion",
        "state",
        "evidence_artifact",
        "evidence_location",
        "remaining_action",
    ]
    rows = [header]
    rows.extend(
        [row[key] for key in header] for row in payload["declared_findings"]["compliance_matrix"]
    )
    return rows
