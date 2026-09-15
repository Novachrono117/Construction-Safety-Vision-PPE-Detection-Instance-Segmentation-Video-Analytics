"""Deterministic Phase 12B follow-up to the immutable Phase 12A audit."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from construction_safety_vision import delivery_audit
from construction_safety_vision.checkpoint_delivery import MANIFEST_PATH, build_manifest
from construction_safety_vision.data.canonical import scan_for_sensitive
from construction_safety_vision.provenance import ProvenanceRecord

BASELINE = "d8605645eea009fe68c6c561b56c31feae937522"
STATUS_PATH = "reports/delivery_gap_resolution_status.json"
PROVENANCE_PATH = "reports/public_delivery.provenance.json"
COMPLETE = "PUBLIC_TRUTH_AND_DELIVERY_SCAFFOLD_COMPLETE"
REPOSITORY_LICENSE = "AGPL-3.0"
LICENSE_SOURCE = "https://www.gnu.org/licenses/agpl-3.0.txt"
LICENSE_SHA256 = "0d96a4ff68ad6d4b6f1f30f713b18d5184912ba8dd389f86aa7710db079abcb0"
LICENSE_BYTES = 34523

# Editorial resolutions, separate from the immutable Phase 12A judgements.
RESOLUTIONS = {
    "GAP-001": ("RESOLVED", ["README.md"], "none"),
    "GAP-006": ("RESOLVED", ["delivery/AI_USAGE.md", "README.md"], "none"),
    "GAP-008": (
        "OPEN",
        [MANIFEST_PATH, "delivery/LICENSING.md", "scripts/verify_checkpoint.py"],
        "Complete license review and authorized publication, then implement documented retrieval.",
    ),
    "GAP-009": (
        "PARTIALLY_RESOLVED",
        ["delivery/REPRODUCTION.md"],
        "Verify the isolated validation-only command path after source and checkpoint retrieval.",
    ),
    "GAP-011": ("RESOLVED", ["LICENSE", "delivery/LICENSING.md"], "none"),
    "GAP-012": (
        "PARTIALLY_RESOLVED",
        ["README.md", "delivery/README.md"],
        "Complete the recruiter README after demo and visual assets exist; retain history.",
    ),
    "GAP-013": ("RESOLVED", ["README.md", ".env.example", "reports/roadmap.md"], "none"),
    "GAP-016": ("RESOLVED", ["reports/README.md"], "none"),
    "GAP-018": ("RESOLVED", ["README.md", "delivery/REPRODUCTION.md"], "none"),
    "GAP-019": (
        "PARTIALLY_RESOLVED",
        ["delivery/REPRODUCTION.md"],
        "Historical D0 regex retained and classified LOW; convergence remains a separate cleanup.",
    ),
    "GAP-020": (
        "OPEN",
        ["delivery/PUBLICATION.md"],
        "After visual assets exist, authorize and configure GitHub description/topics/preview.",
    ),
    "GAP-021": (
        "DEFERRED_OPTIONAL",
        ["reports/rubric_contract.md"],
        "Optional bonus only after every mandatory deliverable is complete.",
    ),
    "GAP-022": (
        "RESOLVED",
        ["README.md", "reports/README.md", "delivery/REPRODUCTION.md"],
        "none",
    ),
}


def build_status(root: Path) -> dict[str, Any]:
    """Reconcile all original gaps and only the requirements changed by delivery work."""
    with (root / "reports/final_delivery_gap_register.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        original = list(csv.DictReader(handle))
    gaps = []
    for gap in original:
        identifier = gap["gap_id"]
        status, evidence, action = RESOLUTIONS.get(
            identifier,
            ("OPEN", ["reports/final_delivery_gap_register.csv"], gap["recommended_fix"]),
        )
        gaps.append(
            {
                "gap_id": identifier,
                "original_severity": gap["severity"],
                "phase_12a_status": "OPEN",
                "current_status": status,
                "resolution_phase": "12B" if identifier in RESOLUTIONS else None,
                "evidence": evidence,
                "remaining_action": action,
            }
        )
    audit = json.loads((root / "reports/final_repository_audit.json").read_text(encoding="utf-8"))
    stale = delivery_audit.stale_claims(root)
    return {
        "schema_version": 1,
        "phase": "12B",
        "baseline_commit": BASELINE,
        "repository_license": {
            "project_repository_license": REPOSITORY_LICENSE,
            "decision_basis": "Repository compliance/documentation based on Ultralytics terms",
            "vendor_source": "https://www.ultralytics.com/license",
            "canonical_text_source": LICENSE_SOURCE,
            "canonical_text_sha256": LICENSE_SHA256,
            "canonical_text_bytes": LICENSE_BYTES,
            "dataset_license": "CC-BY-4.0; dataset images/annotations are not relicensed",
            "original_copyright_holder": "Vinicius Gomes; ownership retained",
            "evidence": "delivery/LICENSING.md",
        },
        "phase_12a_snapshot": "reports/final_repository_audit.json",
        "classification": COMPLETE if stale["probes_present"] == 0 else "TRUTH_PASS_INCOMPLETE",
        "classification_basis": "Run validate_public_delivery.py and the documented quality gates.",
        "stale_probe_ids_before": audit["measurements"]["stale_claims"]["probe_ids_present"],
        "stale_probe_ids_after": stale["probe_ids_present"],
        "additional_truth_corrections": [
            "Current holdout status row now records the completed evaluation and access lock.",
            "Removed retraining as a substitute for frozen detector checkpoint retrieval.",
            "Scoped DVFS explanation as UNTESTED_HYPOTHESIS, consistent with the synthesis.",
            "Marked the architecture's video application as pending and dataset audit as complete.",
        ],
        "gaps": gaps,
        "gap_counts": {
            status: sum(gap["current_status"] == status for gap in gaps)
            for status in ("OPEN", "PARTIALLY_RESOLVED", "RESOLVED", "DEFERRED_OPTIONAL")
        },
        "requirement_updates": [
            {
                "requirement_id": "R20",
                "phase_12a_state": "MISSING",
                "current_state": "COMPLETE",
                "evidence": "delivery/AI_USAGE.md",
                "remaining_action": "none",
            },
            {
                "requirement_id": "R16",
                "phase_12a_state": "PARTIAL",
                "current_state": "PARTIAL",
                "evidence": "README.md",
                "remaining_action": "Final delivery/recruiter restructuring.",
            },
            {
                "requirement_id": "R17",
                "phase_12a_state": "PARTIAL",
                "current_state": "PARTIAL",
                "evidence": "delivery/REPRODUCTION.md",
                "remaining_action": "Checkpoint retrieval, isolated commands and clean-room run.",
            },
        ],
        "other_requirements": "Unchanged from the immutable Phase 12A assignment matrix.",
        "agents_file": {
            "classification": "PROJECT_REPOSITORY_INSTRUCTION",
            "source": "UNKNOWN; text matches CLAUDE.md except title and tool name",
            "purpose": "Durable project protocol instructions for Codex",
            "decision": "Leave untracked and unchanged; no ignore rule added",
            "future_tracking": "Potentially appropriate after explicit human review",
        },
        "scientific_boundary": {
            "final_test": "OBSERVED",
            "holdout": "LOCKED",
            "training": "CLOSED",
            "selection_and_tuning": "CLOSED",
            "models_executed": 0,
            "holdout_content_accessed": False,
            "metrics_recomputed": 0,
            "scientific_results_changed": False,
            "accounting_scope": "Phase 12B actions; existing unit assertions use recorded values.",
        },
        "test_scope": {
            "mode": "metadata-only: PYTEST_ADDOPTS=--metadata-only; uv run pytest",
            "real_model_files": "Not read; local-artifact integration checks excluded",
            "synthetic_checkpoint_verifier": "SHA-256 and byte-size rejection tests",
            "quality_gate_results": "Reported from executed commands in the Phase 12B handoff",
        },
        "delivery_state": {
            "readme_final_professionalization": "PENDING",
            "video": "NOT_STARTED",
            "academic_report": "NOT_GENERATED",
            "colab": "NOT_GENERATED",
            "fp_fn_gallery": "NOT_STARTED",
            "pitch": "NOT_GENERATED",
            "publication_metadata": "RECOMMENDATIONS_ONLY",
            "checkpoint_distribution": "CHECKPOINT_REDISTRIBUTION_REQUIRES_LICENSE_REVIEW",
        },
    }


def historical_changes(root: Path) -> list[str]:
    """List modifications to pre-12B scientific/audit paths without opening their data."""

    def git(*args: str) -> list[str]:
        return subprocess.run(
            ["git", *args], cwd=root, check=True, capture_output=True, text=True, timeout=60
        ).stdout.splitlines()

    historical = set(git("ls-tree", "-r", "--name-only", BASELINE, "reports", "configs", "data"))
    historical.discard("reports/roadmap.md")
    changed = set(git("diff", "--name-only", BASELINE, "--", "reports", "configs", "data"))
    return sorted(historical & changed)


def sensitive_findings(root: Path, paths: list[str]) -> list[str]:
    """Apply the existing scanner to explicit public text paths, never data or weights."""
    problems = []
    for relative in paths:
        text = (root / relative).read_text(encoding="utf-8")
        problems.extend(f"{relative}: {finding}" for finding in scan_for_sensitive(text))
    return problems


def write_metadata(root: Path) -> None:
    """Derive the metadata and record its sources, code and output hashes."""
    for relative, payload in (
        (MANIFEST_PATH, build_manifest(root)),
        (STATUS_PATH, build_status(root)),
    ):
        (root / relative).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    record = ProvenanceRecord.create(
        "public_truth_and_delivery_scaffolding",
        phase=12,
        repo_root=root,
        config={
            "phase": "12B",
            "checkpoint_schema": 1,
            "gap_resolution_policy": RESOLUTIONS,
            "project_repository_license": REPOSITORY_LICENSE,
        },
        details={
            "baseline_commit": BASELINE,
            "code_identity": "Source-file hashes below identify the delivery implementation.",
            "git_commit_semantics": "Stable pre-12B baseline; delivery code uses source hashes.",
            "models_executed": 0,
            "holdout_accessed": False,
            "metrics_recomputed": 0,
            "scientific_results_modified": False,
            "phase_12a_snapshot_modified": False,
        },
    )
    # An amended delivery commit cannot name its own hash. Keep the reachable
    # scientific baseline and identify the exact delivery code through inputs.
    record.git_commit = BASELINE
    inputs = [
        "LICENSE",
        "delivery/LICENSING.md",
        "README.md",
        "delivery/README.md",
        "delivery/PUBLICATION.md",
        "reports/final_detector_manifest.json",
        "reports/final_segmenter_manifest.json",
        "reports/final_delivery_gap_register.csv",
        "reports/final_repository_audit.json",
        "reports/assignment_compliance_matrix.csv",
        "reports/detector_segmenter_spatial_comparison.json",
        "reports/final_test_evaluation.provenance.json",
        "reports/final_test_execution_accounting.json",
        "src/construction_safety_vision/checkpoint_delivery.py",
        "src/construction_safety_vision/delivery_status.py",
        "scripts/validate_public_delivery.py",
        "scripts/verify_checkpoint.py",
    ]
    for relative in inputs:
        record.add_input(root / relative, relative_to=root)
    for relative in (MANIFEST_PATH, STATUS_PATH):
        record.add_output(root / relative, relative_to=root)
    record.write_json(root / PROVENANCE_PATH)


def validate_delivery(root: Path) -> list[str]:
    """Check current public truth and historical immutability without a model/data path."""
    problems = validate_public_license(root)
    if "CSVISION_ALLOW_TEST_SPLIT" in os.environ:
        problems.append("holdout environment variable must be absent")
    if historical_changes(root):
        problems.append("SCIENTIFIC_STATE_MISMATCH: historical artifacts changed")
    stale = delivery_audit.stale_claims(root)
    if stale["probes_present"]:
        problems.append("known stale public claims remain")
    for relative, expected in (
        (MANIFEST_PATH, build_manifest(root)),
        (STATUS_PATH, build_status(root)),
    ):
        if json.loads((root / relative).read_text(encoding="utf-8")) != expected:
            problems.append(f"{relative}: regenerate from authoritative sources")
    spatial = json.loads(
        (root / "reports/detector_segmenter_spatial_comparison.json").read_text(encoding="utf-8")
    )["spatial_comparison_sha256"]
    roadmap = (root / "reports/roadmap.md").read_text(encoding="utf-8")
    if f"current spatial `{spatial}`" not in roadmap:
        problems.append("current spatial fingerprint is missing or stale")
    public = [
        "README.md",
        ".env.example",
        "LICENSE",
        "reports/README.md",
        STATUS_PATH,
        PROVENANCE_PATH,
    ]
    public += [path.relative_to(root).as_posix() for path in sorted((root / "delivery").glob("*"))]
    for relative in public:
        text = (root / relative).read_text(encoding="utf-8")
        if relative.endswith(".md"):
            for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
                if target.startswith(("https://", "http://", "#")):
                    continue
                path = ((root / relative).parent / target.split("#")[0]).resolve()
                if not path.is_relative_to(root.resolve()) or not path.exists():
                    problems.append(f"{relative}: missing or external local link {target}")
    problems.extend(sensitive_findings(root, public))
    readme = (root / "README.md").read_text(encoding="utf-8")
    for obsolete in ("Never evaluated, inspected or materialised", "obtain or retrain the weights"):
        if obsolete in readme:
            problems.append("additional stale current-state claim remains")
    for required in (
        "Python 3.12",
        "CUDA 12.8",
        "delivery/AI_USAGE.md",
        "**OBSERVED**",
        "**CLOSED**",
    ):
        if required not in readme:
            problems.append(f"README: required truth statement missing: {required}")
    for gap in build_status(root)["gaps"]:
        if not all((root / path).is_file() for path in gap["evidence"]):
            problems.append(f"{gap['gap_id']}: evidence file missing")
    lock = delivery_audit.scientific_lock(root, os.environ)
    if (
        lock["final_test_state"] != "FINAL_TEST_OBSERVED"
        or lock["holdout_evaluation_attempts"] != 1
    ):
        problems.append("SCIENTIFIC_STATE_MISMATCH: final evaluation state")
    for field in ("model_selection", "hyperparameter_tuning", "threshold_tuning"):
        if lock[field] != "CLOSED":
            problems.append(f"SCIENTIFIC_STATE_MISMATCH: {field}")
    return problems


def validate_public_license(root: Path) -> list[str]:
    """Check canonical GNU text and the live repository-license signal, offline."""
    problems = []
    license_body = (root / "LICENSE").read_bytes()
    if (
        len(license_body) != LICENSE_BYTES
        or hashlib.sha256(license_body).hexdigest() != LICENSE_SHA256
    ):
        problems.append("LICENSE differs from the canonical GNU AGPL-3.0 download")
    required = {
        "README.md": "[GNU AGPL-3.0](LICENSE)",
        "delivery/README.md": "GNU AGPL-3.0",
        "delivery/LICENSING.md": "PROJECT_REPOSITORY_LICENSE: AGPL-3.0",
        "delivery/PUBLICATION.md": "GNU AGPL-3.0",
    }
    for relative, statement in required.items():
        if statement not in (root / relative).read_text(encoding="utf-8"):
            problems.append(f"{relative}: repository license must be GNU AGPL-3.0")
    live_docs = [root / "README.md", root / "reports/README.md"]
    live_docs += sorted((root / "delivery").glob("*.md"))
    for path in live_docs:
        for line in path.read_text(encoding="utf-8").splitlines():
            # Preserve truthful third-party license labels, not a project MIT grant.
            dependency_row = path.name == "LICENSING.md" and line.startswith(
                ("| NumPy |", "| PyYAML |")
            )
            if re.search(r"\bMIT\b", line) and not dependency_row:
                problems.append(
                    f"{path.relative_to(root).as_posix()}: ambiguous live MIT reference"
                )
    return problems
