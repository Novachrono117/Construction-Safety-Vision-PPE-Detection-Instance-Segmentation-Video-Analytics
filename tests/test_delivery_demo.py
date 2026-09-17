"""Protect the evidence-only default and the optional checkpoint/runtime boundary."""

from __future__ import annotations

import hashlib
import json
import os
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from construction_safety_vision import delivery_demo as demo

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/construction_safety_vision_demo.ipynb"


@pytest.fixture
def public_copy(tmp_path):
    paths = ["delivery/checkpoints.json", "reports/qualitative_validation_gallery.provenance.json"]
    paths += ["reports/" + name for name in demo.REPORTS]
    data = demo.catalog(ROOT)
    paths += data["gallery"] + [row["path"] for row in data["screenshots"]]
    paths += ["src/construction_safety_vision/delivery_demo.py"]
    for relative in paths:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    return tmp_path


def test_mode_a_without_site_packages_data_weights_or_network(public_copy):
    script = (
        "import runpy, sys, pathlib; "
        "root=pathlib.Path(sys.argv[1]); "
        "m=runpy.run_path(str(root/'src/construction_safety_vision/delivery_demo.py')); "
        "data=m['catalog'](root); "
        "assert '0.427031' in m['overview_html'](root); "
        "assert not (root/'data').exists(); assert not (root/'artifacts').exists(); "
        "assert not {'torch','ultralytics','numpy','yaml'} & sys.modules.keys(); "
        "print('MODE_A_METADATA_ONLY_OK')"
    )
    result = subprocess.run(
        [sys.executable, "-S", "-c", script, str(public_copy)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "MODE_A_METADATA_ONLY_OK"


def test_figures_use_exact_committed_bytes(public_copy):
    data = demo.catalog(public_copy)
    for relative in data["gallery"] + [row["path"] for row in data["screenshots"]]:
        assert demo.figure_path(public_copy, relative).is_file()
    first = public_copy / data["gallery"][0]
    first.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SIZE_MISMATCH"):
        demo.figure_path(public_copy, data["gallery"][0])


@pytest.mark.parametrize(
    "path",
    [
        "data/processed/canonical/images/test/x.jpg",
        "reports/figures/final_test/d2/x.png",
        "../secret",
    ],
)
def test_non_gallery_content_is_rejected(public_copy, path):
    with pytest.raises(ValueError, match="Only approved"):
        demo.figure_path(public_copy, path)


def test_metric_sources_are_not_substituted(public_copy):
    data = demo.catalog(public_copy)
    expected = demo.read(public_copy, "reports/final_test_segmenter.json")
    assert (
        data["final_metrics"][2]["value"]
        == expected["canonical_mask"]["CANONICAL_TEST_MASK_MAP50_95"]
    )
    content = demo.overview_html(public_copy)
    assert "Final test" in content and "Validation" in content
    assert "not a selection rule" in content


def test_protocol_drift_stops_catalog(public_copy):
    path = public_copy / "reports/qualitative_validation_gallery.json"
    record = json.loads(path.read_text())
    record["split"] = "test"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="validation evidence only"):
        demo.catalog(public_copy)


def test_checkpoint_manifest_drift_stops_before_file_access(public_copy):
    path = public_copy / "delivery/checkpoints.json"
    record = json.loads(path.read_text())
    record["checkpoints"][0]["sha256"] = "a" * 64
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="identity mismatch"):
        demo.identities(public_copy)


@pytest.mark.parametrize("name", ["D2", "S1"])
def test_invalid_real_identity_upload_is_rejected(public_copy, name):
    path = public_copy / "invalid.pt"
    path.write_bytes(b"this is not a model")
    with pytest.raises(ValueError, match="SIZE_MISMATCH"):
        demo.stage_checkpoint(public_copy, path, name)
    assert not (public_copy / "artifacts").exists()


def test_same_size_wrong_hash_is_rejected(tmp_path):
    path = tmp_path / "wrong.pt"
    path.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="SHA256_MISMATCH"):
        demo.verify(path, {"bytes": 5, "sha256": hashlib.sha256(b"right").hexdigest()})


@pytest.fixture
def synthetic_weights(tmp_path, monkeypatch):
    # Synthetic byte identities exercise I/O ordering without any real checkpoint.
    data = b"synthetic checkpoint fixture"
    identities = {
        name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        for name in demo.WEIGHTS
    }
    monkeypatch.setattr(demo, "identities", lambda root: identities)
    upload = tmp_path / "upload.pt"
    upload.write_bytes(data)
    return upload


def test_valid_staging_is_idempotent_and_never_replaces(tmp_path, synthetic_weights):
    destination = demo.stage_checkpoint(tmp_path, synthetic_weights, "D2")
    assert destination.read_bytes() == synthetic_weights.read_bytes()
    assert demo.stage_checkpoint(tmp_path, synthetic_weights, "D2") == destination
    destination.unlink()  # Break the hard link before constructing a separate conflict.
    destination.write_bytes(b"existing unrelated checkpoint")
    with pytest.raises(ValueError, match="SIZE_MISMATCH"):
        demo.stage_checkpoint(tmp_path, synthetic_weights, "D2")
    assert destination.read_bytes() == b"existing unrelated checkpoint"


def test_holdout_opt_in_stops_delivery(monkeypatch):
    monkeypatch.setenv("CSVISION_ALLOW_TEST_SPLIT", "1")
    with pytest.raises(ValueError, match="must be unset"):
        demo.catalog(ROOT)


@pytest.mark.parametrize(
    "confirmed,mode,device",
    [(False, "compare", "cpu"), (True, "train", "cpu"), (True, "compare", "automatic")],
)
def test_invalid_runtime_request_stops_before_execution(
    tmp_path, monkeypatch, confirmed, mode, device
):
    monkeypatch.setattr(demo.subprocess, "run", lambda *a, **k: pytest.fail("Runtime reached"))
    with pytest.raises(ValueError):
        demo.run_external_video(
            tmp_path, tmp_path / "video.mp4", mode=mode, device=device, confirmed_external=confirmed
        )


def test_input_must_be_in_external_upload_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(demo.subprocess, "run", lambda *a, **k: pytest.fail("Runtime reached"))
    source = tmp_path / "data/train.mp4"
    source.parent.mkdir()
    source.write_bytes(b"synthetic")
    with pytest.raises(ValueError, match="upload cell"):
        demo.run_external_video(
            tmp_path, source, mode="compare", device="cpu", confirmed_external=True
        )


def test_runtime_dispatch_uses_frozen_cli_and_new_output(tmp_path, synthetic_weights, monkeypatch):
    for name in demo.WEIGHTS:
        demo.stage_checkpoint(tmp_path, synthetic_weights, name)
    source = tmp_path / "outputs/colab_demo/session/external_video.mp4"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"synthetic video")
    interpreter = tmp_path / (".venv/Scripts/python.exe" if os.name == "nt" else ".venv/bin/python")
    interpreter.parent.mkdir(parents=True)
    interpreter.touch()
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        assert kwargs == {"cwd": tmp_path, "check": True}
        output = Path(command[command.index("--output") + 1])
        output.write_bytes(b"synthetic output")
        output.with_suffix(".mp4.provenance.json").write_text(
            json.dumps(
                {
                    "status": "COMPLETE",
                    "output": {"bytes": output.stat().st_size, "sha256": demo.digest(output)},
                }
            )
        )

    monkeypatch.setattr(demo.subprocess, "run", fake_run)
    output = demo.run_external_video(
        tmp_path, source, mode="compare", device="cpu", confirmed_external=True
    )
    assert output != source and output.is_file()
    assert commands[0][1] == "scripts/run_video_demo.py"
    assert commands[0][-4:] == ["--mode", "compare", "--device", "cpu"]
    assert not any(x in commands[0] for x in ("--overwrite", "--conf", "--allow-test"))


def test_missing_checkpoint_never_launches_runtime(public_copy, monkeypatch):
    source = public_copy / "outputs/colab_demo/session/input.mp4"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"synthetic")
    monkeypatch.setattr(demo.subprocess, "run", lambda *a, **k: pytest.fail("Runtime reached"))
    with pytest.raises(ValueError, match="MISSING"):
        demo.run_external_video(
            public_copy, source, mode="compare", device="cpu", confirmed_external=True
        )


def test_notebook_defaults_compile_without_embedded_outputs():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    sources = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
            source = "".join(cell["source"])
            compile(source, cell["id"], "exec")
            sources.append(source)
    assert "RUN_INFERENCE = False" in sources[0]
    assert "UPLOAD_RECORDED_VIDEO = False" in "\n".join(sources)
    assert "CONFIRM_EXTERNAL_VIDEO = False" in "\n".join(sources)


def test_notebook_mode_a_skips_install_upload_and_runtime(public_copy, monkeypatch):
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    calls = []

    class Display:
        def __init__(self, *args, **kwargs):
            calls.append((args, kwargs))

    import types

    display = types.ModuleType("IPython.display")
    display.HTML = Display
    display.Image = Display
    display.display = lambda *args: None
    monkeypatch.setitem(sys.modules, "IPython", types.ModuleType("IPython"))
    monkeypatch.setitem(sys.modules, "IPython.display", display)
    module = runpy.run_path(str(public_copy / "src/construction_safety_vision/delivery_demo.py"))
    namespace = {"ROOT": public_copy, "demo": module}
    code_cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    for index, cell in enumerate(code_cells):
        if index == 1:  # Clone/bootstrap is independently checked, no network in this test.
            continue
        exec(compile("".join(cell["source"]), cell["id"], "exec"), namespace)
    assert calls
    assert namespace["result_video"] is None
    assert not (public_copy / "artifacts").exists()
    assert not (public_copy / "outputs").exists()


def test_figure_display_options_fail_closed(public_copy, monkeypatch):
    import types

    display = types.ModuleType("IPython.display")
    display.HTML = display.Image = display.display = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "IPython", types.ModuleType("IPython"))
    monkeypatch.setitem(sys.modules, "IPython.display", display)
    with pytest.raises(ValueError, match="Unknown gallery"):
        demo.show_gallery(public_copy, "test")
    with pytest.raises(ValueError, match="frame pair"):
        demo.show_video_evidence(public_copy, 4)
