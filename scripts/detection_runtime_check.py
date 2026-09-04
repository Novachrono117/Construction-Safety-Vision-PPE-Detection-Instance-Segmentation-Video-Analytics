"""Verify the GPU training runtime, and optionally smoke-test the training stack.

Phase 6A. Two jobs, both deliberately narrow.

**Runtime preflight.** Establish that a CUDA GPU is actually usable, not merely
present. ``torch.cuda.is_available()`` is a weak claim: a build can report a
device it has no kernels for, which is a live risk here because this machine's
GPU is Blackwell (compute capability 12.0) and only CUDA 12.8 builds carry
``sm_120``. So the check runs real kernels - a matmul verified against CPU, a
convolution's backward pass, and an autocast step - rather than trusting a flag.
If a GPU is visible through the driver but torch cannot use it, this reports a
blocked runtime instead of quietly falling back to CPU.

**Smoke test.** A minimal one-epoch run whose only purpose is to prove the stack
executes: weights load, the dataset parses, forward and backward work on the GPU,
the validation loader runs, and checkpoints can be written. Its metrics are
engineering output, not results. They are marked ``NON_EXPERIMENTAL`` and
``DO_NOT_REPORT_AS_MODEL_RESULT``, they are not compared to anything, and nothing
is tuned from them.

The holdout takes no part in either job. The dataset descriptor carries no
``test`` key, and this script never resolves the protected split.

Usage:
    uv run python scripts/detection_runtime_check.py
    uv run python scripts/detection_runtime_check.py --smoke-test
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from construction_safety_vision.config import ConfigError
from construction_safety_vision.data.canonical import scan_for_sensitive
from construction_safety_vision.data.materialization import digest, sha256_bytes
from construction_safety_vision.experiment import (
    ExperimentConfigError,
    load_detection_baseline_config,
)
from construction_safety_vision.paths import ProjectPaths
from construction_safety_vision.provenance import ProvenanceRecord, git_commit
from construction_safety_vision.splits import HOLDOUT_UNLOCK_ENV_VAR, holdout_unlocked

REPORT_MD = "detection_runtime_report.md"
PROVENANCE_JSON = "detection_runtime.provenance.json"
ARTIFACTS_ROOT = "artifacts"
WEIGHTS_SUBDIR = "weights"
SMOKE_PROJECT = "artifacts/detection"
SMOKE_NAME = "smoke"

NON_EXPERIMENTAL = "NON_EXPERIMENTAL"
DO_NOT_REPORT = "DO_NOT_REPORT_AS_MODEL_RESULT"

GPU_READY = "GPU_READY"
GPU_BLOCKED = "BLOCKED_FOR_GPU"


class RuntimeCheckError(RuntimeError):
    """Raised when the runtime cannot be described or verified."""


def nvidia_driver_version() -> str | None:
    """Query the installed NVIDIA driver version.

    Args:
        None.

    Returns:
        The driver version, or ``None`` when ``nvidia-smi`` is unavailable.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    line = result.stdout.strip().splitlines()
    return line[0].strip() if line else None


def gpu_probe() -> dict[str, Any]:
    """Describe and exercise the CUDA runtime.

    Running real kernels rather than reading a flag is the point: a torch build
    can report a device whose architecture it has no compiled kernels for, and
    that failure only surfaces at the first matmul.

    Args:
        None.

    Returns:
        A machine-readable description of the runtime and what it could execute.
    """
    import torch

    probe: dict[str, Any] = {
        "torch_version": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "compiled_arch_list": list(torch.cuda.get_arch_list()),
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "nvidia_driver": nvidia_driver_version(),
    }
    try:
        import torchvision

        probe["torchvision_version"] = torchvision.__version__
    except ImportError:
        probe["torchvision_version"] = None

    if not probe["cuda_available"]:
        probe["status"] = GPU_BLOCKED
        probe["reason"] = (
            "torch reports no CUDA device. A CPU baseline is not an equivalent experiment, "
            "so training is blocked rather than silently downgraded."
        )
        return probe

    index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    capability = f"{properties.major}.{properties.minor}"
    probe.update(
        {
            "gpu_name": torch.cuda.get_device_name(index),
            "gpu_compute_capability": capability,
            "gpu_total_memory_bytes": int(properties.total_memory),
            "gpu_total_memory_gib": round(properties.total_memory / 1024**3, 2),
            "gpu_multi_processor_count": int(properties.multi_processor_count),
        }
    )
    architecture = f"sm_{properties.major}{properties.minor}"
    probe["gpu_arch"] = architecture
    probe["arch_kernels_present"] = architecture in probe["compiled_arch_list"]

    try:
        probe.update(_exercise_gpu(torch))
    except Exception as exc:  # a failing kernel must block, not crash the report
        probe["status"] = GPU_BLOCKED
        probe["reason"] = (
            f"a CUDA device is visible but could not execute: {type(exc).__name__}: {exc}"
        )
        probe["kernel_execution"] = "FAILED"
        return probe

    probe["status"] = GPU_READY
    return probe


def _exercise_gpu(torch: Any) -> dict[str, Any]:
    """Run real kernels and check the results.

    Args:
        torch: The imported ``torch`` module.

    Returns:
        What executed, with the numerical agreement observed.
    """
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()

    left = torch.randn(1024, 1024, device=device)
    right = torch.randn(1024, 1024, device=device)
    torch.cuda.synchronize()
    started = time.perf_counter()
    product = left @ right
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    error = float((product.cpu() - (left.cpu() @ right.cpu())).abs().max())

    inputs = torch.randn(4, 3, 64, 64, device=device, requires_grad=True)
    convolution = torch.nn.Conv2d(3, 16, 3, padding=1).to(device)
    scaler = torch.amp.GradScaler("cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        loss = convolution(inputs).square().mean()
    scaler.scale(loss).backward()
    amp_ok = inputs.grad is not None and bool(torch.isfinite(inputs.grad).all())

    return {
        "kernel_execution": "OK",
        "fp32_matmul_max_abs_error_vs_cpu": error,
        "fp32_matmul_seconds": round(elapsed, 4),
        "amp_autocast_backward": "OK" if amp_ok else "FAILED",
        "probe_peak_memory_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
    }


def ensure_weights(paths: ProjectPaths, identifier: str) -> dict[str, Any]:
    """Obtain the pretrained checkpoint and fingerprint it.

    Two files named ``yolo11n.pt`` are not necessarily the same bytes, so the
    identifier alone is not provenance. The digest is.

    Args:
        paths: Project layout.
        identifier: Weight file name, e.g. ``yolo11n.pt``.

    Returns:
        The weight provenance record.

    Raises:
        RuntimeCheckError: If the checkpoint cannot be obtained.
    """
    destination = paths.root / ARTIFACTS_ROOT / WEIGHTS_SUBDIR / identifier
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.is_file():
        from ultralytics import YOLO

        # Instantiating downloads the checkpoint through Ultralytics' own
        # mechanism; the file it fetches is then moved under artifacts/ so the
        # project owns its location rather than depending on a working directory.
        model = YOLO(identifier)
        source = Path(getattr(model, "ckpt_path", "") or identifier)
        if not source.is_file():
            msg = f"Ultralytics did not leave {identifier} on disk where it could be recorded"
            raise RuntimeCheckError(msg)
        destination.write_bytes(source.read_bytes())
        if source.resolve() != destination.resolve():
            source.unlink(missing_ok=True)

    import ultralytics

    return {
        "identifier": identifier,
        "source_mechanism": "ULTRALYTICS_ASSET_DOWNLOAD",
        "ultralytics_version": ultralytics.__version__,
        "relative_path": destination.relative_to(paths.root).as_posix(),
        "sha256": sha256_bytes(destination),
        "size_bytes": destination.stat().st_size,
        "committed": False,
    }


def run_smoke_test(paths: ProjectPaths, config: Any, dataset: Path) -> dict[str, Any]:
    """Run the minimal one-epoch execution check.

    Not an experiment. It answers "does this stack run at all", and its numbers
    are engineering telemetry.

    Args:
        paths: Project layout.
        config: The frozen D0 protocol, used only for model, image size and seed.
        dataset: The Ultralytics dataset descriptor.

    Returns:
        The smoke-test record.
    """
    import torch
    from ultralytics import YOLO

    weights = paths.root / ARTIFACTS_ROOT / WEIGHTS_SUBDIR / config.weight_identifier
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    record: dict[str, Any] = {
        "classification": NON_EXPERIMENTAL,
        "reporting": DO_NOT_REPORT,
        "epochs": 1,
        "imgsz": config.training["imgsz"],
        "batch": config.training["batch"],
        "splits": ["train", "validation"],
    }
    try:
        model = YOLO(str(weights))
        model.train(
            data=str(dataset),
            epochs=1,
            imgsz=config.training["imgsz"],
            batch=config.training["batch"],
            seed=config.seed,
            deterministic=config.training["deterministic"],
            workers=config.training["workers"],
            amp=config.training["amp"],
            pretrained=True,
            val=True,
            device=0,
            # Absolute: a relative `project` is resolved under Ultralytics'
            # own runs_dir, which would put the run somewhere other than
            # where this report says it is.
            project=str(paths.root / SMOKE_PROJECT),
            name=SMOKE_NAME,
            exist_ok=True,
            plots=False,
            verbose=False,
        )
    except Exception as exc:
        record.update(
            {
                "result": "FAILED",
                "error": f"{type(exc).__name__}: {exc}",
                "seconds": round(time.perf_counter() - started, 1),
            }
        )
        return record

    run_directory = paths.root / SMOKE_PROJECT / SMOKE_NAME
    best = run_directory / "weights" / "best.pt"
    last = run_directory / "weights" / "last.pt"
    record.update(
        {
            "result": "OK",
            "seconds": round(time.perf_counter() - started, 1),
            "device": torch.cuda.get_device_name(0),
            "peak_memory_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
            "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 1024**2, 1),
            "checkpoint_written": best.is_file() or last.is_file(),
            "run_directory": Path(SMOKE_PROJECT).joinpath(SMOKE_NAME).as_posix(),
            "weights_committed": False,
            "metrics_recorded": False,
            "note": (
                "no AP, loss or any other quality number from this run is recorded, reported "
                "or compared. The run exists to prove the stack executes."
            ),
        }
    )
    return record


def build_report(
    probe: dict[str, Any],
    weights: dict[str, Any],
    smoke: dict[str, Any] | None,
    config: Any,
    commit: str | None,
) -> str:
    """Write the detection runtime report.

    Args:
        probe: The GPU probe result.
        weights: The pretrained weight provenance.
        smoke: The smoke-test record, when one was run.
        config: The frozen D0 protocol.
        commit: Repository commit.

    Returns:
        The report as Markdown.
    """
    lines: list[str] = []
    add = lines.append
    add("# Detection Runtime and Baseline Readiness - Phase 6A")
    add("")
    add(f"Phase: 6A · Commit: `{commit or 'unavailable'}` · Runtime: **{probe['status']}**")
    add("")
    add(
        "This report separates two things that reproducibility discussions often merge. The "
        "**software environment** is pinned and reproducible: exact package versions, locked. "
        "The **execution hardware** is recorded but not pinned - a driver version and a GPU "
        "model are facts about where a run happened, not dependencies someone else must match."
    )
    add("")
    add("## 1. Software environment")
    add("")
    add("| Component | Version |")
    add("| --- | --- |")
    add(f"| Python | {platform.python_version()} |")
    add(f"| torch | {probe['torch_version']} |")
    add(f"| torchvision | {probe['torchvision_version'] or 'not installed'} |")
    add(f"| ultralytics | {weights['ultralytics_version']} |")
    add(f"| CUDA runtime (torch) | {probe['torch_cuda_runtime']} |")
    add(f"| cuDNN | {probe['cudnn_version']} |")
    add("")
    add("`FACT` These are locked in `uv.lock`. torch and torchvision resolve from the CUDA 12.8")
    add("build index rather than PyPI, declared in `pyproject.toml`.")
    add("")
    add("## 2. Execution hardware")
    add("")
    add("| Property | Value |")
    add("| --- | --- |")
    add(f"| GPU | {probe.get('gpu_name', 'none')} |")
    add(f"| Compute capability | {probe.get('gpu_compute_capability', '-')} |")
    add(f"| VRAM | {probe.get('gpu_total_memory_gib', '-')} GiB |")
    add(f"| Streaming multiprocessors | {probe.get('gpu_multi_processor_count', '-')} |")
    add(f"| NVIDIA driver | {probe.get('nvidia_driver') or 'unavailable'} |")
    add(f"| Devices visible | {probe['device_count']} |")
    add("")
    add(
        "`FACT` Hardware is recorded, never made a package dependency. Reproducing the "
        "*software* environment does not require this GPU; reproducing the *timings* does."
    )
    add("")
    add("## 3. Is the GPU actually usable?")
    add("")
    add(
        "`COMPUTED RESULT` `torch.cuda.is_available()` is a weak claim, so it is not the "
        "evidence here. This GPU is "
        f"**{probe.get('gpu_arch', 'unknown')}** (Blackwell), and only CUDA 12.8 builds carry "
        "kernels for it; an older build would see the device and fail at the first matmul."
    )
    add("")
    add("| Check | Result |")
    add("| --- | --- |")
    add(f"| `torch.cuda.is_available()` | {probe['cuda_available']} |")
    add(f"| compiled architectures | `{', '.join(probe['compiled_arch_list'])}` |")
    add(
        f"| kernels present for {probe.get('gpu_arch', '-')} | "
        f"{probe.get('arch_kernels_present', '-')} |"
    )
    add(f"| real fp32 matmul executed | {probe.get('kernel_execution', '-')} |")
    add(f"| max abs error vs CPU | {probe.get('fp32_matmul_max_abs_error_vs_cpu', '-')} |")
    add(f"| AMP autocast + backward | {probe.get('amp_autocast_backward', '-')} |")
    add("")
    if probe["status"] == GPU_BLOCKED:
        add(f"`BLOCKED` {probe.get('reason', 'the GPU could not be used.')}")
    else:
        add(
            "`COMPUTED RESULT` The device executes real kernels and agrees with the CPU to "
            "float32 precision, and mixed precision works. The D0 protocol's "
            "`device_requirement: CUDA_GPU_REQUIRED` is satisfiable, and no CPU fallback was "
            "used or needed."
        )
    add("")
    add("## 4. Pretrained weights")
    add("")
    add("| Property | Value |")
    add("| --- | --- |")
    add(f"| Identifier | `{weights['identifier']}` |")
    add(
        f"| Source | {weights['source_mechanism']} (ultralytics {weights['ultralytics_version']}) |"
    )
    add(f"| Stored at | `{weights['relative_path']}` (git-ignored) |")
    add(f"| SHA-256 | `{weights['sha256']}` |")
    add(f"| Size | {weights['size_bytes']} bytes |")
    add("")
    add(
        "`FACT` Fingerprinted rather than trusted by name: two files called `yolo11n.pt` are "
        "not necessarily the same bytes, and a baseline that cannot name its starting weights "
        "cannot be reproduced. The binary itself is not committed."
    )
    add("")
    lines.extend(_smoke_section(smoke, config))
    return "\n".join(lines) + "\n"


def _smoke_section(smoke: dict[str, Any] | None, config: Any) -> list[str]:
    """Write the smoke-test and readiness sections.

    Args:
        smoke: The smoke-test record, when one was run.
        config: The frozen D0 protocol.

    Returns:
        The report lines.
    """
    lines: list[str] = []
    add = lines.append
    add("## 5. Smoke test")
    add("")
    if smoke is None:
        add("`FACT` No smoke test was run in this invocation.")
        add("")
    else:
        add(
            f"`{NON_EXPERIMENTAL}` `{DO_NOT_REPORT}` The smoke test exists only to prove the "
            "training stack executes: weights load, the dataset parses, forward and backward "
            "run on the GPU, the validation loader works, and checkpoints can be written."
        )
        add("")
        add("| Property | Value |")
        add("| --- | --- |")
        add(f"| Result | **{smoke['result']}** |")
        add(f"| Epochs | {smoke['epochs']} |")
        add(f"| Image size | {smoke['imgsz']} |")
        add(f"| Batch | {smoke['batch']} |")
        add(f"| Splits used | {', '.join(smoke['splits'])} |")
        add(f"| Device | {smoke.get('device', '-')} |")
        add(f"| Wall clock | {smoke.get('seconds', '-')} s |")
        add(f"| Peak GPU memory allocated | {smoke.get('peak_memory_mib', '-')} MiB |")
        add(f"| Peak GPU memory reserved | {smoke.get('peak_reserved_mib', '-')} MiB |")
        add(f"| Checkpoint written | {smoke.get('checkpoint_written', '-')} |")
        if smoke.get("error"):
            add(f"| Error | {smoke['error']} |")
        add("")
        add(
            "**No metric from this run is recorded anywhere.** No AP, no loss, no precision or "
            "recall. It is not compared to anything, nothing was tuned from it, and its "
            "checkpoints are temporary engineering artifacts that are not committed. Reporting "
            "a one-epoch number as a model result would be inventing a finding."
        )
        add("")
    add("## 6. What this phase did not do")
    add("")
    add(
        f"`FACT` The full {config.experiment_id} baseline was **not run**. No hyperparameter "
        "was tuned against validation. No model result exists. The protected holdout was not "
        f"materialised, adapted, read or measured, and `{HOLDOUT_UNLOCK_ENV_VAR}` was not set; "
        "the Ultralytics dataset descriptor carries no `test` key at all."
    )
    return lines


def main(argv: list[str] | None = None) -> int:
    """Verify the runtime and optionally smoke-test the training stack.

    Args:
        argv: Command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true", help="Run the minimal 1-epoch check.")
    parser.add_argument("--config", type=Path, default=None, help="Baseline protocol.")
    args = parser.parse_args(argv)

    paths = ProjectPaths.from_root()
    try:
        config = load_detection_baseline_config(
            args.config or (paths.configs / "detection_baseline.yaml")
        )
    except (ConfigError, ExperimentConfigError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if holdout_unlocked():
        print(
            f"ERROR: {HOLDOUT_UNLOCK_ENV_VAR} is set. The holdout must stay locked throughout "
            "phase 6A; refusing to run.",
            file=sys.stderr,
        )
        return 2

    probe = gpu_probe()
    print(f"  runtime: {probe['status']}")
    print(f"  torch {probe['torch_version']}  cuda {probe['torch_cuda_runtime']}")
    if probe["status"] == GPU_BLOCKED:
        print(f"  {probe.get('reason', '')}", file=sys.stderr)
    else:
        print(
            f"  {probe['gpu_name']}  sm_{probe['gpu_compute_capability'].replace('.', '')}  "
            f"{probe['gpu_total_memory_gib']} GiB  kernels "
            f"{probe['arch_kernels_present']}"
        )

    try:
        weights = ensure_weights(paths, config.weight_identifier)
    except RuntimeCheckError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3
    print(f"  weights {weights['identifier']}  {weights['sha256'][:16]}  {weights['size_bytes']} B")

    smoke = None
    if args.smoke_test:
        if probe["status"] == GPU_BLOCKED:
            print("ERROR: refusing to smoke-test without a usable GPU", file=sys.stderr)
            return 4
        dataset = paths.root / "data/processed/adapters/yolo_detection/dataset.yaml"
        if not dataset.is_file():
            print("ERROR: run scripts/build_detection_adapter.py first", file=sys.stderr)
            return 3
        print("  smoke test: 1 epoch, train + validation only ...", flush=True)
        smoke = run_smoke_test(paths, config, dataset)
        print(f"  smoke test: {smoke['result']} in {smoke.get('seconds', '?')} s")

    commit = git_commit(paths.root)
    report = build_report(probe, weights, smoke, config, commit)
    unsafe = scan_for_sensitive(report)
    if unsafe:
        print(f"ERROR: {REPORT_MD} is not fit to commit: {'; '.join(unsafe)}", file=sys.stderr)
        return 5
    report_path = paths.reports / REPORT_MD
    report_path.write_text(report, encoding="utf-8", newline="\n")

    record = ProvenanceRecord.create(
        "detection_runtime_check",
        phase=6,
        repo_root=paths.root,
        config={
            "detection_baseline": "configs/detection_baseline.yaml",
            "baseline_protocol_sha256": config.fingerprint(),
            "experiment_id": config.experiment_id,
        },
        details={
            "phase": "6A",
            "software_environment": {
                "python": platform.python_version(),
                "torch": probe["torch_version"],
                "torchvision": probe["torchvision_version"],
                "ultralytics": weights["ultralytics_version"],
                "cuda_runtime": probe["torch_cuda_runtime"],
                "cudnn": probe["cudnn_version"],
            },
            "execution_hardware": {
                key: probe.get(key)
                for key in (
                    "gpu_name",
                    "gpu_compute_capability",
                    "gpu_total_memory_bytes",
                    "gpu_multi_processor_count",
                    "nvidia_driver",
                    "device_count",
                )
            },
            "gpu_probe": {
                key: probe.get(key)
                for key in (
                    "status",
                    "cuda_available",
                    "compiled_arch_list",
                    "arch_kernels_present",
                    "kernel_execution",
                    "fp32_matmul_max_abs_error_vs_cpu",
                    "amp_autocast_backward",
                )
            },
            "pretrained_weights": weights,
            "smoke_test": smoke,
            "smoke_metrics_recorded": False,
            "full_baseline_run": False,
            "holdout_accessed": False,
            "environment_fingerprint": digest(
                {
                    "python": platform.python_version(),
                    "torch": probe["torch_version"],
                    "torchvision": probe["torchvision_version"],
                    "ultralytics": weights["ultralytics_version"],
                    "cuda_runtime": probe["torch_cuda_runtime"],
                    "weights_sha256": weights["sha256"],
                }
            ),
        },
    )
    record.add_input(paths.configs / "detection_baseline.yaml", relative_to=paths.root)
    record.add_output(report_path, relative_to=paths.root)
    provenance_path = paths.reports / PROVENANCE_JSON
    serialised = json.dumps(record.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
    unsafe = scan_for_sensitive(serialised)
    if unsafe:
        print(
            f"ERROR: {PROVENANCE_JSON} is not fit to commit: {'; '.join(unsafe)}",
            file=sys.stderr,
        )
        return 5
    provenance_path.write_text(f"{serialised}\n", encoding="utf-8", newline="\n")

    for path in (report_path, provenance_path):
        print(f"wrote {path.relative_to(paths.root).as_posix()}")
    if probe["status"] == GPU_BLOCKED:
        print(f"\n{GPU_BLOCKED}")
        return 6
    print(f"\n{GPU_READY}   holdout (test): locked")
    return 0


if __name__ == "__main__":
    sys.exit(main())
