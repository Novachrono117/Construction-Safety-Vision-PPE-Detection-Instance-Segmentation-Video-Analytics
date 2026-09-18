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
# Phase 14C re-validated Mode A on a fresh CPU runtime after the notebook gained its
# explicit training and evaluation sections. Mode B was NOT re-executed: its evidence
# still belongs to the phase 14A revision, and the two revisions are proved
# byte-identical across every Mode B execution input below.
VALIDATED = "2e355d8012b6f1f49923c5cb3ead06760490f024"
MODE_B_VALIDATED = "8ce5d0375903e3e3760873e0a75ddce37cc1149b"
NOTEBOOK = "notebooks/construction_safety_vision_demo.ipynb"
NOTEBOOK_IDENTITY = {
    "bytes": 17891,
    "sha256": "34eed8efd8198652cb7539dc28487c9bc5cb99d41c80db4a52fb440b27ad5c92",
}
# The roles that actually execute during Mode B inference. Identity across the two
# validated revisions is what carries the phase 14A inference evidence forward. The
# notebook carries its own role and is deliberately outside this set, because it is
# the one file phase 14C changed.
MODE_B_ROLES = frozenset(
    {
        "TRANSITIVE_RUNTIME_MODULE",
        "MODE_B_ENTRYPOINT",
        "LOCKED_INSTALLATION_INPUT",
        "FROZEN_CHECKPOINT_METADATA",
        "FROZEN_INFERENCE_CONFIGURATION",
        "STANDALONE_DELIVERY_SUPPORT",
    }
)
# What the maintainer actually observed. Mode A was re-executed for phase 14C on a
# fresh CPU runtime; Mode B was not re-run, and its phase 14A result is carried
# forward only because nothing it executes changed.
HUMAN_OBSERVATION = {
    "executor": "HUMAN_MAINTAINER",
    "basis": "COLAB_SCREENSHOTS_AND_EXPLICIT_PHASE_14C_FINALIZATION_ATTESTATION",
    "observation_date": "2026-09-18",
    "mode_a": "PASS",
    "mode_a_validated_revision": VALIDATED,
    "mode_a_runtime": "CPU",
    "mode_a_reexecuted_in_phase_14c": True,
    "mode_b": "PASS",
    "mode_b_validated_revision": MODE_B_VALIDATED,
    "mode_b_observation_date": "2026-09-17",
    "mode_b_reexecuted_in_phase_14c": False,
    "agent_cloud_execution": False,
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
    # Added in phase 14C for the training and evaluation sections.
    "canonical_modeling_manifest.json",
    "detection_D2_manifest.json",
    "segmentation_S1_result_manifest.json",
    "final_test_direct_iou.json",
    "segmentation_S1_canonical_evaluation.json",
    "segmentation_S1_mask_iou.json",
)
# Committed training curves displayed by the phase 14C training section. They are
# declared by the experiments' own manifests and hold no dataset or holdout imagery.
TRAINING_FIGURES = (
    "reports/figures/detection/D2/results.png",
    "reports/figures/segmentation_S1/results.png",
)


def identity(data: bytes) -> dict[str, Any]:
    """Identify exact bytes without reading an artifact implicitly."""
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def git_bytes(root: Path, relative: str, revision: str = VALIDATED) -> bytes:
    """Read only an explicitly named blob from an explicitly named revision."""
    return subprocess.check_output(["git", "show", f"{revision}:{relative}"], cwd=root)


def mode_b_surface_identity(root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare every Mode B execution input across the two cloud-validated revisions.

    Phase 14C re-validated Mode A only. Carrying the phase 14A inference evidence
    forward is legitimate exactly when nothing Mode B executes has changed between
    the two revisions, so that is derived here rather than asserted.

    Args:
        root: Repository root.
        rows: The execution surface at the phase 14C validated revision.

    Returns:
        The compared paths and any that differ.
    """
    compared = sorted(row["path"] for row in rows if row["role"] in MODE_B_ROLES)
    changed = [
        path
        for path in compared
        if git_bytes(root, path, MODE_B_VALIDATED) != git_bytes(root, path, VALIDATED)
    ]
    return {
        "mode_a_validated_revision": VALIDATED,
        "mode_b_validated_revision": MODE_B_VALIDATED,
        "compared_paths": compared,
        "compared_count": len(compared),
        "changed_paths": changed,
        "identical": not changed,
        "roles_compared": sorted(MODE_B_ROLES),
        "notebook_excluded_because": (
            "The notebook is the one file phase 14C changed, so it is validated in its own "
            "right by the phase 14C Mode A cloud run rather than by this comparison."
        ),
        "conclusion": (
            "MODE_B_REVALIDATION_NOT_REQUIRED" if not changed else "MODE_B_REVALIDATION_REQUIRED"
        ),
    }


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
            "src/construction_safety_vision/delivery_academic.py": (
                "STANDALONE_ARTIFACT_PRESENTATION_SUPPORT"
            ),
            "scripts/run_video_demo.py": "MODE_B_ENTRYPOINT",
            "pyproject.toml": "LOCKED_INSTALLATION_INPUT",
            "uv.lock": "LOCKED_INSTALLATION_INPUT",
            ".python-version": "LOCKED_INSTALLATION_INPUT",
            "delivery/checkpoints.json": "FROZEN_CHECKPOINT_METADATA",
            "configs/detector_segmenter_comparison.yaml": "FROZEN_INFERENCE_CONFIGURATION",
        }
    )
    paths.update({f"reports/{name}": "COMMITTED_EVIDENCE_INPUT" for name in TEXT_EVIDENCE})
    paths.update(dict.fromkeys(TRAINING_FIGURES, "COMMITTED_TRAINING_CURVE_FIGURE"))
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
    if provenance.get("human_observation") != HUMAN_OBSERVATION:
        problems.append("Human cloud observation missing or misrepresented")
    identity_record = provenance.get("mode_b_surface_identity", {})
    if not identity_record.get("identical") or identity_record.get("changed_paths"):
        problems.append("Mode B execution surface changed; carried-forward evidence is invalid")
    if identity_record.get("mode_b_validated_revision") != MODE_B_VALIDATED:
        problems.append("Mode B evidence revision mismatch")
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
        "phase": "14C",
        "classification": COMPLETE,
        "assignment_wording_status": "COLAB_EXACT_WORDING_SATISFIED",
        "baseline_commit": BASELINE,
        "validated_temporary_revision": VALIDATED,
        "mode_b_surface_identity": provenance["mode_b_surface_identity"],
        "notebook_sections": [
            "1 Overview",
            "2 Dataset and classes",
            "3 TREINO / TRAINING",
            "4 AVALIACAO / EVALUATION",
            "5 Qualitative evidence",
            "6 Video evidence",
            "7 INFERENCIA / INFERENCE",
            "8 Limitations and licences",
        ],
        "training_section": {
            "label": "RECORDED_TRAINING_EVIDENCE",
            "execution": "NO_NEW_TRAINING_EXECUTED",
            "content": (
                "Executable cells read each frozen model's committed experiment manifest and "
                "render its architecture, image size, epochs, batch, seed, the optimizer that "
                "`optimizer: auto` resolved to, the full augmentation set, the checkpoint "
                "selection rule, the selected epoch, the frozen checkpoint identity and the "
                "committed training and validation curves."
            ),
            "reproducible_entry_points": [
                "uv run python scripts/train_detection_experiment.py --experiment D2",
                "uv run python scripts/train_segmentation_comparison.py",
            ],
            "default_notebook_retrains": False,
        },
        "evaluation_section": {
            "label": "FINAL TEST RESULTS - READ FROM THE COMMITTED ONE-SHOT EVALUATION",
            "content": (
                "Executable cells read the committed one-shot holdout result artifacts by field "
                "and render canonical COCO average precision, operating-point precision and "
                "recall, per-class AP, the direct instance-mask IoU diagnostic, object-level "
                "TP/FP/FN, both final-test confusion matrices from their recorded counts, and "
                "the bounded validation-versus-test comparison."
            ),
            "default_notebook_reruns_final_test": False,
            "models_invoked": 0,
            "metrics_recomputed": 0,
            "holdout_content_accessed": False,
            "executable_validation_evaluation_path": False,
        },
        "notebook": {"path": NOTEBOOK, **NOTEBOOK_IDENTITY, "canonical_branch": "main"},
        "execution_surface": provenance["execution_surface"],
        "execution_surface_identity_policy": (
            "Every candidate file must match the exact validated Git blob bytes and SHA-256. "
            "Line-ending changes are rejected, including Windows CRLF conversion."
        ),
        "cloud_validation_scope": {
            "cloud_validated_revision": VALIDATED,
            "mode_b_cloud_validated_revision": MODE_B_VALIDATED,
            "final_candidate_commit_executed_in_cloud": False,
            "canonical_notebook_executed_verbatim": False,
            "documented_human_session_deltas": [
                "Phase 14C: clone target changed once in the Colab session from main "
                "to phase14c-colab-validation.",
                "Phase 14C: Mode A ran top to bottom on a fresh CPU runtime with "
                "RUN_INFERENCE=False and the canonical compare/cpu defaults.",
                "Phase 14A: clone target changed once from main to phase14a-colab-validation.",
                "Phase 14A: Mode A kept RUN_INFERENCE=False; Mode B selected True, "
                "compare and cuda.",
                "Phase 14A: Mode B added a separate GPU diagnostic cell, which was never "
                "committed and is not part of any canonical notebook.",
                "Phase 14A: Mode B ran cells individually, skipping the already-validated "
                "Mode A displays.",
            ],
            "finalization_basis": (
                "Phase 14C human Mode A cloud evidence on the changed notebook, plus a derived "
                "byte-identity proof that every Mode B execution input is unchanged since the "
                "phase 14A inference validation; only delivery documentation, reports, status "
                "and verification code change during finalization."
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
                "observation_date": provenance["human_observation"]["observation_date"],
                "fresh_runtime": True,
                "runtime_accelerator": "CPU",
                "run_all": True,
                "traceback_observed": False,
                "observed_messages": [
                    f"Repository revision: {VALIDATED}",
                    "Mode A ready. No checkpoint, dataset or inference framework loaded.",
                    "Mode B installation skipped.",
                    "Checkpoint upload skipped.",
                    "Video upload and inference skipped.",
                    "Mode A complete. No inference output was created.",
                ],
                "sections_rendered": [
                    "dataset and classes",
                    "TREINO / TRAINING with D2 and S1 recipes and committed curves",
                    "AVALIACAO / EVALUATION with final-test metrics and both confusion matrices",
                    "qualitative evidence",
                    "real-video evidence",
                    "INFERENCIA / INFERENCE present and skipped",
                ],
                "human_readability_review": {
                    "tables": "PASS",
                    "training_curves": "PASS",
                    "evaluation_tables": "PASS",
                    "confusion_matrices": "PASS",
                    "qualitative_figures": "PASS",
                    "video_evidence": "PASS",
                },
                "gpu_required": False,
                "checkpoint_used": False,
                "roboflow_key_used": False,
                "holdout_authorization_used": False,
                "model_inference_used": False,
                "training_executed": False,
                "final_test_evaluation_executed": False,
                "metrics": "DISPLAY_EXISTING_COMMITTED_AGGREGATES_ONLY",
                "contrast_fix": "HUMAN_CONFIRMED_READABLE_IN_REAL_COLAB",
            },
            "REAL_COLAB_MODE_B_VALIDATION": {
                "status": "PASS",
                "result_key": "MODE_B_COLAB_OPTIONAL_INFERENCE_VALIDATION",
                "execution_kind": "HUMAN_EXECUTED",
                "validated_revision": MODE_B_VALIDATED,
                "observation_date": provenance["human_observation"]["mode_b_observation_date"],
                "reexecuted_in_phase_14c": False,
                "carried_forward_basis": "MODE_B_EXECUTION_SURFACE_BYTE_IDENTICAL",
                "carried_forward_conclusion": provenance["mode_b_surface_identity"]["conclusion"],
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
            "Mode B was validated once, at the phase 14A revision, and was not re-executed "
            "for phase 14C; the carry-forward rests on byte identity, not a second run.",
            "The training and evaluation sections present recorded evidence. The notebook "
            "does not retrain the models and does not rerun the spent holdout evaluation, "
            "and it offers no executable path that would.",
            "Reproducing a training run needs a CUDA GPU and the materialised dataset, "
            "neither of which the notebook provides.",
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
    mode_a = evidence["REAL_COLAB_MODE_A_VALIDATION"]
    mode_b = evidence["REAL_COLAB_MODE_B_VALIDATION"]
    runtime = mode_b["runtime"]
    identity_record = report["mode_b_surface_identity"]
    training = report["training_section"]
    evaluation = report["evaluation_section"]
    lines = [
        "# Phase 14C - executable academic Colab with explicit training and evaluation",
        "",
        f"**Classification:** `{report['classification']}`. "
        f"**Assignment wording:** `{report['assignment_wording_status']}`.",
        "",
        "Human cloud validation and local metadata verification are separate evidence categories.",
        f"Approved scientific baseline: `{BASELINE}`.",
        f"Mode A validated revision: `{VALIDATED}` (phase 14C, fresh CPU runtime).",
        f"Mode B validated revision: `{MODE_B_VALIDATED}` (phase 14A, not re-executed).",
        "",
        "## Notebook structure",
        "",
        *[f"- {name}" for name in report["notebook_sections"]],
        "",
        "### TREINO / TRAINING",
        "",
        f"`{training['label']}` / `{training['execution']}`. {training['content']}",
        "",
        "The default notebook does **not** retrain the models "
        f"(`default_notebook_retrains: {str(training['default_notebook_retrains']).lower()}`). "
        "Each model was trained exactly once under a protocol frozen before the run, and "
        "re-running it would produce a different checkpoint under the same name. The "
        "reproducible entry points are printed with each recipe:",
        "",
        *[f"- `{command}`" for command in training["reproducible_entry_points"]],
        "",
        "### AVALIACAO / EVALUATION",
        "",
        f"Banner: `{evaluation['label']}`. {evaluation['content']}",
        "",
        "The default notebook does **not** rerun the final-test evaluation "
        "(`default_notebook_reruns_final_test: "
        f"{str(evaluation['default_notebook_reruns_final_test']).lower()}`, "
        f"`models_invoked: {evaluation['models_invoked']}`, "
        f"`metrics_recomputed: {evaluation['metrics_recomputed']}`, "
        f"`holdout_content_accessed: "
        f"{str(evaluation['holdout_content_accessed']).lower()}`). "
        "The holdout was read once and is spent, so no executable evaluation path is offered "
        "for it; none is offered for validation either "
        "(`executable_validation_evaluation_path: "
        f"{str(evaluation['executable_validation_evaluation_path']).lower()}`), because that "
        "would need the materialised dataset and the frozen checkpoints.",
        "",
        "## Mode B carry-forward",
        "",
        f"**{identity_record['conclusion']}.** Phase 14C re-validated Mode A only. Every one of "
        f"the {identity_record['compared_count']} Mode B execution inputs is byte-identical "
        f"between `{MODE_B_VALIDATED}` and `{VALIDATED}`, so the phase 14A inference evidence "
        "still describes the code that would run. This is derived by comparing Git blobs, not "
        f"asserted. Changed paths: {identity_record['changed_paths'] or 'none'}. "
        f"{identity_record['notebook_excluded_because']}.",
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
        "**ARTIFACT_ONLY_COLAB_CLOUD_EXECUTION: PASS** - executed by the human maintainer "
        f"on {mode_a['observation_date']}, fresh runtime, accelerator "
        f"`{mode_a['runtime_accelerator']}`, Run all, no traceback.",
        "A fresh Colab run required no GPU, checkpoints, Roboflow key, "
        "holdout authorization or inference. It rendered:",
        "",
        *[f"- {item};" for item in mode_a["sections_rendered"]],
        "",
        "No training was executed and no final-test evaluation was rerun "
        f"(`training_executed: {str(mode_a['training_executed']).lower()}`, "
        "`final_test_evaluation_executed: "
        f"{str(mode_a['final_test_evaluation_executed']).lower()}`).",
        "",
        "| Human readability review | Result |",
        "|---|---|",
        *[
            f"| {key.replace('_', ' ')} | {value} |"
            for key, value in mode_a["human_readability_review"].items()
        ],
        "",
        "```text",
        *mode_a["observed_messages"],
        "```",
        "",
        "## REAL_COLAB_MODE_B_VALIDATION",
        "",
        "**MODE_B_COLAB_OPTIONAL_INFERENCE_VALIDATION: PASS** - executed by the human "
        f"maintainer on {mode_b['observation_date']} at `{MODE_B_VALIDATED}`. It was **not** "
        f"re-executed for phase 14C; see the carry-forward proof above "
        f"(`{mode_b['carried_forward_basis']}`).",
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
        "In each Colab session the human changed the clone branch, so the canonical notebook was "
        "not executed verbatim, and the final candidate commit itself was not cloud executed. "
        "The phase 14A session additionally added a separate GPU diagnostic cell; that cell was "
        "never committed and is absent from every canonical notebook, so a stale saved copy of "
        "it is a session artifact rather than a delivery defect.",
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
        if provenance["mode_b_surface_identity"] != mode_b_surface_identity(root, expected_surface):
            problems.append("Mode B carry-forward proof does not re-derive")
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
        "human_observation": HUMAN_OBSERVATION,
        "source_records": source_records(root),
        "execution_surface": surface,
        "mode_b_surface_identity": mode_b_surface_identity(root, surface),
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
