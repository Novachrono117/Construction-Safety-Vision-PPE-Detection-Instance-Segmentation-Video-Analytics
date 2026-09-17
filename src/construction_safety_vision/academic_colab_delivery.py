"""Reconcile human Colab evidence without executing models or accessing datasets."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from construction_safety_vision.data.canonical import scan_for_sensitive

REPORT = "reports/academic_colab_delivery.json"
MARKDOWN = "reports/academic_colab_delivery.md"
PROVENANCE = "reports/academic_colab_delivery.provenance.json"
COMPLETE = "EXECUTABLE_ACADEMIC_COLAB_COMPLETE"
BASELINE = "fa0bea2b072b124d65f4e9bd1b932a22d98149d2"
VALIDATED = "8ce5d0375903e3e3760873e0a75ddce37cc1149b"
NOTEBOOK = "notebooks/construction_safety_vision_demo.ipynb"
NOTEBOOK_IDENTITY = {
    "bytes": 12126,
    "sha256": "cfed2af4f197b4eba5b5bc61391197a97e5d53ef355009fe55a70b978b222773",
}
REPOSITORY = (
    "Novachrono117/Construction-Safety-Vision-PPE-Detection-Instance-Segmentation-Video-Analytics"
)
COLAB_URL = f"https://colab.research.google.com/github/{REPOSITORY}/blob/main/{NOTEBOOK}"
WARNING = "SOURCE_LICENSE_UNSPECIFIED: not ready for publication"
SOURCE_FILES = {
    "mode_b_execution": (
        "outputs/phase14a_colab_output/demo_f75991d9a6fd481fb11be6cb4f7ad895.mp4.provenance.json",
        5523,
        "f26e5a7090e3af6729dda85885416e97238defdfd904910c0e11cf641605c4bd",
    ),
    "external_clip_preparation": (
        "outputs/phase14a_colab_input/colab_demo_5s.source.json",
        2244,
        "c9a0bd915a9d84e57740fbbda7f33af083bbfc86df8ebe3607846244271a4206",
    ),
}
RUNTIME_MODULES = (
    "__init__.py",
    "config.py",
    "env.py",
    "paths.py",
    "provenance.py",
    "splits.py",
    "checkpoint_delivery.py",
    "detection_freeze.py",
    "segmentation_freeze.py",
    "video_models.py",
    "video_runtime.py",
    "video_render.py",
    "data/__init__.py",
    "data/canonical.py",
    "data/acquisition.py",
    "data/coco.py",
    "data/roboflow.py",
    "data/versioning.py",
)
TEXT_EVIDENCE = (
    "final_detector_manifest.json",
    "final_segmenter_manifest.json",
    "final_test_detector.json",
    "final_test_segmenter.json",
    "detector_segmenter_box_comparison.json",
    "qualitative_validation_gallery.json",
    "qualitative_validation_gallery.provenance.json",
    "final_real_video_demo.json",
)


def identity(data: bytes) -> dict[str, Any]:
    """Identify exact bytes without reading an artifact implicitly."""
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def git_bytes(root: Path, relative: str) -> bytes:
    """Read only an explicitly named blob from the validated revision."""
    return subprocess.check_output(["git", "show", f"{VALIDATED}:{relative}"], cwd=root)


def surface_paths(root: Path) -> dict[str, str]:
    """List project execution closure, installation inputs and displayed evidence."""
    paths = {
        f"src/construction_safety_vision/{name}": "TRANSITIVE_RUNTIME_MODULE"
        for name in RUNTIME_MODULES
    }
    paths.update(
        {
            NOTEBOOK: "CANONICAL_NOTEBOOK_WITH_DOCUMENTED_SESSION_DELTAS",
            "src/construction_safety_vision/delivery_demo.py": "STANDALONE_DELIVERY_SUPPORT",
            "scripts/run_video_demo.py": "MODE_B_ENTRYPOINT",
            "pyproject.toml": "LOCKED_INSTALLATION_INPUT",
            "uv.lock": "LOCKED_INSTALLATION_INPUT",
            ".python-version": "LOCKED_INSTALLATION_INPUT",
            "delivery/checkpoints.json": "FROZEN_CHECKPOINT_METADATA",
            "configs/detector_segmenter_comparison.yaml": "FROZEN_INFERENCE_CONFIGURATION",
        }
    )
    paths.update({f"reports/{name}": "COMMITTED_EVIDENCE_INPUT" for name in TEXT_EVIDENCE})
    gallery = json.loads(git_bytes(root, "reports/qualitative_validation_gallery.json"))
    video = json.loads(git_bytes(root, "reports/final_real_video_demo.json"))
    for path in gallery["figures"]:
        paths[path] = "APPROVED_VALIDATION_FIGURE"
    for row in video["screenshots"]["screenshots"]:
        paths[row["path"]] = "APPROVED_EXTERNAL_VIDEO_FIGURE"
    return paths


def execution_surface(root: Path) -> list[dict[str, Any]]:
    """Capture Git blob OIDs and SHA-256 of Linux checkout bytes, not CRLF variants."""
    rows = []
    for relative, role in sorted(surface_paths(root).items()):
        data = git_bytes(root, relative)
        blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        rows.append({"path": relative, "role": role, "git_blob_oid": blob, **identity(data)})
    return rows


def candidate_surface_problems(root: Path, rows: list[dict[str, Any]]) -> list[str]:
    """Require exact candidate bytes to match every recorded cloud execution input."""
    problems = []
    for row in rows:
        relative = row["path"]
        if not (root / relative).is_file():
            problems.append(f"Execution-critical file missing: {relative}")
            continue
        actual = identity((root / relative).read_bytes())
        if actual != {key: row[key] for key in ("bytes", "sha256")}:
            problems.append(f"STOP: execution-critical bytes changed: {relative}")
    return problems


def notebook_problems(data: bytes) -> list[str]:
    """Check canonical defaults and reject saved outputs, attachments or private locators."""
    problems = []
    if identity(data) != NOTEBOOK_IDENTITY:
        problems.append("Canonical notebook identity changed")
    try:
        notebook = json.loads(data)
        if notebook.get("nbformat") != 4:
            problems.append("Invalid notebook format")
        defaults, clones = {}, []
        for cell in notebook["cells"]:
            if cell.get("attachments") or cell.get("outputs"):
                problems.append("Embedded notebook output or attachment")
            if cell["cell_type"] != "code":
                continue
            if cell.get("execution_count") is not None:
                problems.append("Notebook execution count is not cleared")
            source = "".join(cell["source"])
            tree = ast.parse(source)
            for node in tree.body:
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            defaults[target.id] = node.value.value
            for node in ast.walk(tree):
                if isinstance(node, ast.List) and len(node.elts) >= 4:
                    prefix = [
                        n.value if isinstance(n, ast.Constant) else None for n in node.elts[:4]
                    ]
                    if prefix[:3] == ["git", "clone", "--branch"]:
                        clones.append(prefix[3])
        if clones != ["main"]:
            problems.append("Canonical notebook must clone main exactly once")
        for key in ("RUN_INFERENCE", "UPLOAD_RECORDED_VIDEO", "CONFIRM_EXTERNAL_VIDEO"):
            if defaults.get(key) is not False:
                problems.append(f"Unsafe notebook default: {key}")
        if defaults.get("DEVICE") != "cpu" or defaults.get("MODEL_VIEW") != "compare":
            problems.append("Canonical notebook display/device defaults changed")
    except (ValueError, TypeError, KeyError, SyntaxError):
        problems.append("Invalid notebook structure or code")
    problems.extend(public_text_problems(data.decode("utf-8")))
    return problems


def public_text_problems(text: str) -> list[str]:
    """Extend the existing sensitive scanner to personal cloud URLs and embedded media."""
    problems = scan_for_sensitive(text)
    if re.search(r"https?://(?:drive\.google\.com/|colab\.research\.google\.com/drive/)", text):
        problems.append("Private Drive/Colab URL")
    if re.search(r"data:(?:video/|application/octet-stream|image/)", text):
        problems.append("Embedded binary data URL")
    return problems


def source_records(root: Path) -> dict[str, Any]:
    """Archive exact text evidence; never alter the original generated provenance."""
    records = {}
    for key, (relative, size, digest) in SOURCE_FILES.items():
        data = (root / relative).read_bytes()
        if identity(data) != {"bytes": size, "sha256": digest}:
            raise ValueError(f"STOP: observed source record changed: {key}")
        text = data.decode("utf-8")
        if public_text_problems(text):
            raise ValueError(f"Source record is not public-safe: {key}")
        records[key] = {
            "filename": Path(relative).name,
            "identity": identity(data),
            "original_utf8_text": text,
            "record": json.loads(text),
        }
    return records


def provenance_problems(provenance: dict[str, Any]) -> list[str]:
    """Reject fabricated source records or rewritten attribution warnings."""
    problems = []
    if provenance.get("validated_revision") != VALIDATED:
        problems.append("Validated revision mismatch")
    expected_observation = {
        "executor": "HUMAN_MAINTAINER",
        "basis": "COLAB_SCREENSHOTS_AND_EXPLICIT_PHASE_14A_FINALIZATION_ATTESTATION",
        "observation_date": "2026-09-17",
        "mode_a": "PASS",
        "mode_b": "PASS",
        "agent_cloud_execution": False,
    }
    if provenance.get("human_observation") != expected_observation:
        problems.append("Human cloud observation missing or misrepresented")
    quality = provenance.get("quality_gates", {})
    pytest_result = quality.get("pytest", {})
    if (
        quality.get("ruff_check") != "PASS"
        or quality.get("ruff_format") != "PASS"
        or pytest_result.get("mode") != "metadata-only"
        or not isinstance(pytest_result.get("passed"), int)
        or pytest_result.get("passed", 0) <= 0
        or pytest_result.get("failed", 0) != 0
    ):
        problems.append("Local quality gates do not support PASS")
    try:
        records = provenance["source_records"]
        if set(records) != set(SOURCE_FILES):
            problems.append("Source evidence inventory mismatch")
        for key, (_, size, digest) in SOURCE_FILES.items():
            source = records[key]
            raw = source["original_utf8_text"].encode("utf-8")
            expected = {"bytes": size, "sha256": digest}
            if source["identity"] != expected or identity(raw) != expected:
                problems.append(f"Original source bytes changed: {key}")
            if source["record"] != json.loads(raw):
                problems.append(f"Embedded parsed source differs from original: {key}")
        run = records["mode_b_execution"]["record"]
        if WARNING not in run["warnings"] or run["source_attribution"]["license"] is not None:
            problems.append("Original source-license warning/unspecified attribution changed")
    except (KeyError, TypeError, ValueError, AttributeError):
        problems.append("Invalid source provenance structure")
    problems.extend(public_text_problems(json.dumps(provenance)))
    return problems


def build_report(provenance: dict[str, Any]) -> dict[str, Any]:
    """Derive delivery conclusions from preserved observations, never new inference."""
    records = provenance["source_records"]
    run = records["mode_b_execution"]["record"]
    prep = records["external_clip_preparation"]["record"]
    device = run["inference"]["device"]
    checkpoint_verification = {
        name: {
            **{key: model["identity"][key] for key in ("bytes", "sha256", "architecture")},
            "result": "VERIFIED_BEFORE_DESERIALIZATION",
            "basis": "HUMAN_COLAB_UPLOAD_OUTPUT_AND_MATCHING_RUNTIME_PROVENANCE",
            "frames_completed": model["frames_completed"],
        }
        for name, model in run["inference"]["models"].items()
    }
    return {
        "schema_version": 1,
        "phase": "14A",
        "classification": COMPLETE,
        "baseline_commit": BASELINE,
        "validated_temporary_revision": VALIDATED,
        "notebook": {"path": NOTEBOOK, **NOTEBOOK_IDENTITY, "canonical_branch": "main"},
        "execution_surface": provenance["execution_surface"],
        "execution_surface_identity_policy": (
            "Every candidate file must match the exact validated Git blob bytes and SHA-256. "
            "Line-ending changes are rejected, including Windows CRLF conversion."
        ),
        "cloud_validation_scope": {
            "cloud_validated_revision": VALIDATED,
            "final_candidate_commit_executed_in_cloud": False,
            "canonical_notebook_executed_verbatim": False,
            "documented_human_session_deltas": [
                "Clone target changed once in the Colab session from main "
                "to phase14a-colab-validation.",
                "Mode A kept RUN_INFERENCE=False; Mode B selected True, compare and cuda.",
                "Mode B added a separate GPU diagnostic cell and confirmed the external video.",
                "Mode B ran cells individually, skipping the already-validated Mode A displays.",
            ],
            "finalization_basis": (
                "Human cloud evidence plus unchanged validated execution surface; only delivery "
                "documentation, reports, status and verification code change during finalization."
            ),
        },
        "validation_evidence": {
            "LOCAL_IMPLEMENTATION_VERIFICATION": {
                "status": "PASS",
                "execution_kind": "AGENT_METADATA_ONLY",
                "checks": [
                    "Canonical notebook JSON, cleared outputs, main target and safe defaults",
                    "All recorded execution inputs match validated Git blobs",
                    "Exact original cloud provenance/source context preserved and reconciled",
                    "Public text scanner and focused refusal tests",
                ],
                "models_executed_during_finalization": 0,
                "holdout_accessed": False,
                "quality_gates": provenance["quality_gates"],
            },
            "REAL_COLAB_MODE_A_VALIDATION": {
                "status": "PASS",
                "result_key": "ARTIFACT_ONLY_COLAB_CLOUD_EXECUTION",
                "execution_kind": "HUMAN_EXECUTED",
                "validated_revision": VALIDATED,
                "observation_basis": provenance["human_observation"]["basis"],
                "fresh_runtime": True,
                "observed_messages": [
                    f"Repository revision: {VALIDATED}",
                    "Mode A ready. No checkpoint, dataset or inference framework loaded.",
                    "Mode A complete. No inference output was created.",
                ],
                "checkpoint_used": False,
                "roboflow_key_used": False,
                "holdout_authorization_used": False,
                "model_inference_used": False,
                "metrics": "DISPLAY_EXISTING_COMMITTED_AGGREGATES_ONLY",
                "contrast_fix": "HUMAN_CONFIRMED_READABLE_IN_REAL_COLAB",
            },
            "REAL_COLAB_MODE_B_VALIDATION": {
                "status": "PASS",
                "result_key": "MODE_B_COLAB_OPTIONAL_INFERENCE_VALIDATION",
                "execution_kind": "HUMAN_EXECUTED",
                "validated_revision": VALIDATED,
                "runtime": {
                    "gpu": device["name"],
                    "cuda_available": True,
                    "pytorch_cuda": device["cuda_build"],
                    "pytorch": device["torch"],
                    "python": run["environment"]["python_version"],
                    "ultralytics": run["inference"]["ultralytics"],
                    "opencv": run["opencv"],
                    "mode": run["mode"],
                    "requested_device": run["requested_device"],
                },
                "checkpoint_verification": checkpoint_verification,
                "input": {**run["input"], **run["source"]},
                "output": {**run["output"], **run["output_video"], "layout": run["layout"]},
                "execution_status": run["status"],
                "processed_frames": run["processed_frames"],
                "artifact_classification": "COLAB_TECHNICAL_VALIDATION_ARTIFACT",
                "publication_status": "NOT_PUBLICATION_READY",
                "original_warnings": run["warnings"],
                "original_provenance_identity": records["mode_b_execution"]["identity"],
                "verification": (
                    "Downloaded output size/hash, full decoding and layout were verified locally; "
                    "no model inference was repeated. Media remains ignored and unpublished."
                ),
            },
        },
        "source_license_distinction": {
            "original_cloud_attribution": run["source_attribution"],
            "original_cloud_warning": WARNING,
            "independent_source_context": prep["source"],
            "source_record": prep["source_record"],
            "source_preparation_identity": records["external_clip_preparation"]["identity"],
            "context_does_not_rewrite_cloud_provenance": True,
            "five_second_output_published": False,
        },
        "protocol": {
            "scientific_lifecycle": "CLOSED",
            "holdout": "OBSERVED_SPENT_LOCKED",
            "holdout_accessed_during_finalization": False,
            "holdout_authorization": "MUST_BE_UNSET",
            "models_executed_during_finalization": 0,
            "new_training_tuning_selection_or_metrics": False,
            "checkpoint_policy": (
                "Optional manual upload; exact frozen byte size and SHA-256 "
                "before deserialization; "
                "no download, pretrained substitution or training fallback."
            ),
        },
        "gap_004_previous_status": "OPEN",
        "gap_004_status": "RESOLVED",
        "assignment_requirement_id": "R18",
        "assignment_colab_status": "COMPLETE",
        "resolution_scope": (
            "The missing executable-notebook requirement (R18/C6) is fulfilled by artifact-only "
            "Run all and optional frozen-model external-video inference. The historical GAP-004 "
            "recommended fix mentioned metric reproduction; the approved closed-science delivery "
            "scope displays committed metrics and does not recompute them. The Phase 12A snapshot "
            "is unchanged. Broader clean-room reproduction remains GAP-014 OPEN."
        ),
        "other_gaps": {"GAP-008": "OPEN", "GAP-010": "PARTIALLY_RESOLVED", "GAP-014": "OPEN"},
        "colab_badge": {
            "url": COLAB_URL,
            "branch": "main",
            "repository_visibility": "PUBLIC_CONFIRMED_BEFORE_FINALIZATION",
            "url_validation": "STRUCTURALLY_VALID",
            "final_main_availability": "REQUIRES_SEPARATELY_AUTHORIZED_FINAL_COMMIT_PUSH",
        },
        "known_limitations": [
            "The final candidate commit itself has not been run in Colab; "
            "surface identities are preserved.",
            "Mode B still requires manual uploads and compatible GPU capacity; "
            "future Colab changes may matter.",
            "The five-second output is technical evidence only "
            "and must not be published in this phase.",
            "MP4/mp4v browser playback is not guaranteed; no audio/timestamp track is copied.",
            "Inference timing is a demo measurement, not a controlled scientific benchmark.",
            "Checkpoint redistribution remains unresolved; "
            "licensing differs across code, data and media.",
            "The complete Phase 13B MP4 has no public URL; six approved frames keep Mode A useful.",
            "No Release, technical report, tracking, pitch "
            "or broader README professionalization is included.",
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Produce a readable report directly from the validated text evidence."""
    evidence = report["validation_evidence"]
    mode_b = evidence["REAL_COLAB_MODE_B_VALIDATION"]
    runtime = mode_b["runtime"]
    lines = [
        "# Phase 14A - executable academic Colab",
        "",
        f"**Classification:** `{report['classification']}`.",
        "",
        "Human cloud validation and local metadata verification are separate evidence categories.",
        f"Approved scientific baseline: `{BASELINE}`. Validated temporary revision: `{VALIDATED}`.",
        "",
        "## LOCAL_IMPLEMENTATION_VERIFICATION",
        "",
        "PASS: notebook safety, source provenance, publication scan and execution-surface "
        "identity checks. No models run and no holdout access during finalization.",
        "",
        "```json",
        json.dumps(
            evidence["LOCAL_IMPLEMENTATION_VERIFICATION"]["quality_gates"],
            indent=2,
            sort_keys=True,
        ),
        "```",
        "",
        "## REAL_COLAB_MODE_A_VALIDATION",
        "",
        "**ARTIFACT_ONLY_COLAB_CLOUD_EXECUTION: PASS** - executed by the human maintainer.",
        "A fresh Colab run required no checkpoints, Roboflow key, "
        "holdout authorization or inference.",
        "It displayed architecture, committed metrics, approved validation FP/FN "
        "and an approved video-frame pair; six frames are available across three selections.",
        "The maintainer confirmed the contrast correction on the validated revision.",
        "",
        "```text",
        *evidence["REAL_COLAB_MODE_A_VALIDATION"]["observed_messages"],
        "```",
        "",
        "## REAL_COLAB_MODE_B_VALIDATION",
        "",
        "**MODE_B_COLAB_OPTIONAL_INFERENCE_VALIDATION: PASS** - executed by the human maintainer.",
        "",
        "| Runtime | Observed value |",
        "|---|---|",
        *[f"| {key} | {value} |" for key, value in runtime.items()],
        "",
        "Both checkpoints were VERIFIED_BEFORE_DESERIALIZATION after manual upload.",
        "",
        "| Checkpoint | Bytes | SHA-256 |",
        "|---|---:|---|",
        *[
            f"| {name} | {v['bytes']} | `{v['sha256']}` |"
            for name, v in mode_b["checkpoint_verification"].items()
        ],
        "",
        "The external input is a five-second, 640x360, 25 FPS excerpt (125 frames) from "
        "the approved Phase 13B real-world construction source. Output is 1280x360, with "
        "D2 left and S1 right; both models completed 125 frames.",
        "",
        "| Artifact | Bytes | SHA-256 |",
        "|---|---:|---|",
        *[
            f"| {key} | {mode_b[key]['bytes']} | `{mode_b[key]['sha256']}` |"
            for key in ("input", "output")
        ],
        "",
        mode_b["verification"],
        "",
        "## Source and licensing",
        "",
        "**COLAB_TECHNICAL_VALIDATION_ARTIFACT / NOT_PUBLICATION_READY.**",
        "",
        f"The original generated provenance retains `{WARNING}` and null attribution. "
        "Its exact UTF-8 text, byte identity and parsed content are archived in "
        "[the provenance record](academic_colab_delivery.provenance.json). "
        "It has not been rewritten.",
        "Independent preparation evidence identifies Frank Vincentz / CC BY-SA 3.0. That context "
        "does not change the original warning or authorize publication of this five-second output.",
        "Code licensing, checkpoint redistribution, dataset licensing and external-video licensing "
        "remain separate. See [licensing](../delivery/LICENSING.md).",
        "",
        "## Canonical notebook and validation scope",
        "",
        f"Notebook: `{NOTEBOOK}`; {NOTEBOOK_IDENTITY['bytes']} bytes; "
        f"SHA-256 `{NOTEBOOK_IDENTITY['sha256']}`.",
        "The canonical file still clones `main`, defaults to `RUN_INFERENCE=False`, and contains "
        "no saved outputs. It was not changed to the temporary branch locally.",
        "The human Colab session changed the clone branch, selected Mode B controls when needed "
        "and added a separate GPU diagnostic cell. The canonical notebook was therefore not "
        "executed verbatim. The final candidate commit itself was not cloud executed.",
        "Finalization relies on that disclosed cloud execution "
        "and unchanged execution-critical bytes.",
        "",
        "## Execution-surface identities",
        "",
        "Each identity is the exact Git blob content at the validated revision. All candidate "
        "files must match those raw bytes; Windows CRLF conversion also fails this check. "
        "Any execution-critical change stops finalization.",
        "",
        "| Path | Git blob OID | Bytes | SHA-256 |",
        "|---|---|---:|---|",
        *[
            f"| `{r['path']}` | `{r['git_blob_oid']}` | {r['bytes']} | `{r['sha256']}` |"
            for r in report["execution_surface"]
        ],
        "",
        "## Delivery status and publication boundary",
        "",
        "**GAP-004: OPEN -> RESOLVED. R18 executable Colab: COMPLETE.**",
        "",
        report["resolution_scope"],
        "",
        "GAP-008 remains OPEN; GAP-010 remains PARTIALLY_RESOLVED; GAP-014 remains OPEN.",
        f"The [canonical Open in Colab target]({COLAB_URL}) points to `main`. "
        "Its URL is structurally "
        "valid and the repository is public; final notebook availability on main requires a "
        "separately authorized push. This phase creates a local final commit only.",
        "No GitHub Release or full-video publication is part of this phase. The temporary "
        "validation branch remains evidence and is not merged or deleted.",
        "",
        "## Known limitations",
        "",
        *[f"- {item}" for item in report["known_limitations"]],
        "",
    ]
    return "\n".join(lines)


def validate(root: Path) -> list[str]:
    """Validate portable report evidence and candidate bytes without private files or models."""
    if "CSVISION_ALLOW_TEST_SPLIT" in os.environ:
        return ["Holdout authorization must be absent"]
    try:
        provenance = json.loads((root / PROVENANCE).read_text(encoding="utf-8"))
        report = json.loads((root / REPORT).read_text(encoding="utf-8"))
        problems = provenance_problems(provenance)
        if problems:
            return problems
        expected_surface = execution_surface(root)
        if provenance["execution_surface"] != expected_surface:
            problems.append("Execution-surface inventory/validated identities mismatch")
        problems.extend(candidate_surface_problems(root, expected_surface))
        problems.extend(notebook_problems((root / NOTEBOOK).read_bytes()))
        expected = build_report(provenance)
        if report != expected:
            problems.append("Report claims disagree with preserved validation evidence")
        markdown = (root / MARKDOWN).read_text(encoding="utf-8")
        if markdown != render_markdown(expected):
            problems.append("Markdown report differs from structured evidence")
        for relative in (REPORT, MARKDOWN, PROVENANCE):
            problems.extend(
                f"{relative}: {p}"
                for p in public_text_problems((root / relative).read_text(encoding="utf-8"))
            )
        return problems
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        return [f"Invalid academic Colab evidence: {type(error).__name__}"]


def write_reports(root: Path, quality_gates: dict[str, Any]) -> None:
    """Generate the three delivery records from fixed observed local text evidence."""
    if "CSVISION_ALLOW_TEST_SPLIT" in os.environ:
        raise ValueError("Holdout authorization must be absent")
    surface = execution_surface(root)
    problems = candidate_surface_problems(root, surface)
    problems.extend(notebook_problems((root / NOTEBOOK).read_bytes()))
    if problems:
        raise ValueError("; ".join(problems))
    provenance = {
        "schema_version": 1,
        "purpose": "DELIVERY_RECONCILIATION_OF_HUMAN_CLOUD_OBSERVATIONS",
        "baseline_commit": BASELINE,
        "validated_revision": VALIDATED,
        "human_observation": {
            "executor": "HUMAN_MAINTAINER",
            "basis": "COLAB_SCREENSHOTS_AND_EXPLICIT_PHASE_14A_FINALIZATION_ATTESTATION",
            "observation_date": "2026-09-17",
            "mode_a": "PASS",
            "mode_b": "PASS",
            "agent_cloud_execution": False,
        },
        "source_records": source_records(root),
        "execution_surface": surface,
        "quality_gates": quality_gates,
        "generator": "src/construction_safety_vision/academic_colab_delivery.py",
        "generation_policy": (
            "Explicit --write only; original local provenance is read-only. Reconciliation reads "
            "text and approved Git blobs, not models, videos or holdout data. Candidate HEAD is "
            "not embedded, avoiding self-referential final-commit identities."
        ),
    }
    problems = provenance_problems(provenance)
    if problems:
        raise ValueError("; ".join(problems))
    report = build_report(provenance)
    for relative, payload in ((PROVENANCE, provenance), (REPORT, report)):
        (root / relative).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
        )
    (root / MARKDOWN).write_text(render_markdown(report), encoding="utf-8", newline="\n")
