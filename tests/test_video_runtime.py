import hashlib
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from construction_safety_vision import video_models as vm
from construction_safety_vision import video_runtime as vr
from construction_safety_vision.checkpoint_delivery import verify_file
from construction_safety_vision.video_fixture import create_fixture
from construction_safety_vision.video_render import COLORS, MASK_ALPHA, render_frame, render_panel

ROOT = Path(__file__).resolve().parents[1]


class StubPredictor:
    def __init__(self, root, mode, device):
        self.mode = mode
        self.count = 0

    def predict(self, frame):
        self.count += 1
        return {name: [] for name in vm.MODES[self.mode]}

    def evidence(self):
        return {"test_stub": True, "calls": self.count}

    def close(self):
        pass


@pytest.fixture
def fixture_video(tmp_path):
    source = tmp_path / "fixture.mp4"
    evidence = create_fixture(source)
    return source, evidence


def execute(tmp_path, fixture_video, **kwargs):
    source, evidence = fixture_video
    options = {
        "mode": "compare",
        "device": "cpu",
        "root": tmp_path / "repo",
        "attribution": evidence["attribution"],
        "fps_overlay": False,
        "predictor_factory": StubPredictor,
    }
    options.update(kwargs)
    return vr.run_video(source, tmp_path / "out.mp4", **options)


@pytest.mark.parametrize("mode,multiplier", [("detector", 1), ("segmenter", 1), ("compare", 2)])
def test_real_codec_roundtrip_preserves_order_count_fps_and_aspect(
    tmp_path, fixture_video, mode, multiplier
):
    record = execute(tmp_path, fixture_video, mode=mode)
    assert record["status"] == "COMPLETE"
    assert record["decoded_frames"] == record["processed_frames"] == record["encoded_frames"] == 6
    assert record["output_video"]["fps"] == record["source"]["fps"] == 10
    assert record["output_video"]["width"] == 320 * multiplier
    assert record["output_video"]["height"] == 192
    assert record["output_video"]["duration_seconds"] == 0.6
    digest = hashlib.sha256()
    cap = vr.open_capture(fixture_video[0])
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        digest.update(frame.tobytes())
        frames.append(frame)
    cap.release()
    assert record["source_decoded_sequence_sha256"] == digest.hexdigest()
    cap = vr.open_capture(tmp_path / "out.mp4")
    for original in frames:
        ok, output = cap.read()
        assert ok
        # Away from labels, compare both full-size canvases to the matching source frame.
        for panel in range(multiplier):
            actual = output[110:140, panel * 320 + 100 : panel * 320 + 200].astype(float)
            assert np.abs(actual - original[110:140, 100:200]).mean() < 8
    assert not cap.read()[0]
    cap.release()
    assert record["processing_throughput_fps"] > 0
    assert record["input_unchanged"]
    assert not list(tmp_path.glob("*.incomplete*"))
    saved = json.loads((tmp_path / "out.mp4.provenance.json").read_text())
    assert saved == record
    assert str(tmp_path) not in json.dumps(record)


@pytest.mark.parametrize("mode", ["winner", "track", "", "D2"])
def test_unknown_modes_rejected(tmp_path, fixture_video, mode):
    with pytest.raises(vm.VideoRuntimeError, match="Unknown mode"):
        execute(tmp_path, fixture_video, mode=mode)


def test_cli_requires_device_and_has_no_scientific_overrides():
    spec = importlib.util.spec_from_file_location("video_cli", ROOT / "scripts/run_video_demo.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parser = module.build_parser()
    base = ["--input", "in.mp4", "--output", "out.mp4", "--mode", "compare"]
    with pytest.raises(SystemExit):
        parser.parse_args(base)
    for override in ("--conf", "--iou", "--imgsz", "--weights", "--tracker", "--half"):
        with pytest.raises(SystemExit):
            parser.parse_args([*base, "--device", "cpu", override, "1"])
    assert parser.parse_args([*base, "--device", "cuda"]).device == "cuda"


def test_cuda_unavailable_fails_without_fallback(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(vm.VideoRuntimeError, match="CUDA_UNAVAILABLE"):
        vm.require_device("cuda")
    assert vm.require_device("cpu")["requested"] == "cpu"


def test_input_open_failure_has_failed_sidecar(tmp_path):
    source = tmp_path / "invalid.mp4"
    source.write_bytes(b"not a container")
    with pytest.raises(vm.VideoRuntimeError, match="INPUT_OPEN_FAILED"):
        vr.run_video(
            source,
            tmp_path / "out.mp4",
            mode="compare",
            device="cpu",
            root=tmp_path / "repo",
            predictor_factory=StubPredictor,
        )
    assert (
        json.loads((tmp_path / "out.mp4.failed.provenance.json").read_text())["status"] == "FAILED"
    )
    assert not (tmp_path / "out.mp4").exists()


@pytest.mark.parametrize("point", ["writer", "predict", "encode", "verify", "interrupt"])
def test_failures_remove_partial_and_never_write_success(
    tmp_path, fixture_video, monkeypatch, point
):
    class FailingPredictor(StubPredictor):
        def predict(self, frame):
            if point == "interrupt":
                raise KeyboardInterrupt
            raise RuntimeError("sensitive exception details must not be published")

    def fail(*args, **kwargs):
        raise vm.VideoRuntimeError("INJECTED_FAILURE")

    if point == "writer":
        monkeypatch.setattr(vr, "open_writer", fail)
    if point == "verify":
        monkeypatch.setattr(vr, "verify_output", fail)
    if point == "encode":
        original = vr.open_writer

        class BrokenWriter:
            def __init__(self, *args):
                self.writer = original(*args)

            def write(self, frame):
                fail()

            def release(self):
                self.writer.release()

            def getBackendName(self):  # noqa: N802 - OpenCV interface
                return self.writer.getBackendName()

        monkeypatch.setattr(vr, "open_writer", BrokenWriter)
    factory = FailingPredictor if point in {"predict", "interrupt"} else StubPredictor
    with pytest.raises(vm.VideoRuntimeError):
        execute(tmp_path, fixture_video, predictor_factory=factory)
    assert not (tmp_path / "out.mp4").exists()
    assert not (tmp_path / "out.mp4.provenance.json").exists()
    assert not list(tmp_path.glob("*.incomplete*"))
    record = json.loads((tmp_path / "out.mp4.failed.provenance.json").read_text())
    assert record["status"] == "FAILED"
    assert "sensitive exception" not in json.dumps(record)


def test_writer_open_false_rejected(monkeypatch, tmp_path):
    class ClosedWriter:
        def isOpened(self):  # noqa: N802 - OpenCV interface
            return False

        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoWriter", lambda *a: ClosedWriter())
    with pytest.raises(vm.VideoRuntimeError, match="WRITER_OPEN_FAILED"):
        vr.open_writer(tmp_path / "out.mp4", 10, (320, 192))


@pytest.mark.parametrize(
    "field,value", [("fps", 0), ("fps", float("nan")), ("width", 0), ("width", 319)]
)
def test_invalid_metadata_fails(field, value):
    fields = {
        "width": cv2.CAP_PROP_FRAME_WIDTH,
        "height": cv2.CAP_PROP_FRAME_HEIGHT,
        "fps": cv2.CAP_PROP_FPS,
    }

    class FakeCapture:
        def get(self, key):
            return float(
                value
                if key == fields[field]
                else {
                    cv2.CAP_PROP_FRAME_WIDTH: 320,
                    cv2.CAP_PROP_FRAME_HEIGHT: 192,
                    cv2.CAP_PROP_FPS: 10,
                    cv2.CAP_PROP_FRAME_COUNT: 6,
                    cv2.CAP_PROP_FOURCC: 0,
                }[key]
            )

        def getBackendName(self):  # noqa: N802 - OpenCV interface
            return "STUB"

    with pytest.raises(vm.VideoRuntimeError):
        vr.video_metadata(FakeCapture())


@pytest.mark.parametrize("violation", ["early_eof", "timestamp", "shape"])
def test_decode_contract_failures(tmp_path, fixture_video, monkeypatch, violation):
    original = vr.open_capture

    class BrokenCapture:
        def __init__(self, path):
            self.capture = original(path)
            self.n = 0

        def read(self):
            self.n += 1
            if violation == "early_eof" and self.n == 3:
                return False, None
            ok, frame = self.capture.read()
            if violation == "shape" and self.n == 2:
                frame = frame[:-2]
            return ok, frame

        def get(self, key):
            if violation == "timestamp" and key == cv2.CAP_PROP_POS_MSEC:
                return float(self.n * 250)
            return self.capture.get(key)

        def getBackendName(self):  # noqa: N802 - OpenCV interface
            return "STUB"

        def release(self):
            self.capture.release()

    monkeypatch.setattr(vr, "open_capture", BrokenCapture)
    with pytest.raises(vm.VideoRuntimeError):
        execute(tmp_path, fixture_video)
    assert not (tmp_path / "out.mp4").exists()


def test_engineering_prefix_not_claimed_full_video(tmp_path, fixture_video):
    record = execute(tmp_path, fixture_video, max_frames=2)
    assert record["status"] == "ENGINEERING_PREFIX_COMPLETE"
    assert record["processed_frames"] == 2 and not record["full_source_processed"]


def test_overwrite_protects_previous_output_on_failure(tmp_path, fixture_video):
    execute(tmp_path, fixture_video)
    before = (tmp_path / "out.mp4").read_bytes()
    with pytest.raises(vm.VideoRuntimeError, match="already exists"):
        execute(tmp_path, fixture_video)

    class Broken(StubPredictor):
        def predict(self, frame):
            raise RuntimeError("failed")

    with pytest.raises(vm.VideoRuntimeError):
        execute(tmp_path, fixture_video, overwrite=True, predictor_factory=Broken)
    assert (tmp_path / "out.mp4").read_bytes() == before


def test_frozen_thresholds_and_order():
    policy = vm.frozen_settings(ROOT)
    assert (policy["imgsz"], policy["conf"], policy["iou"], policy["max_det"]) == (
        768,
        0.25,
        0.7,
        300,
    )
    manifest = json.loads((ROOT / "reports/final_detector_manifest.json").read_text())
    assert tuple(sorted(manifest["class_map"], key=manifest["class_map"].get)) == vm.CLASSES


def test_settings_drift_rejected(tmp_path):
    path = tmp_path / vm.POLICY_PATH
    path.parent.mkdir()
    path.write_text((ROOT / vm.POLICY_PATH).read_text().replace("conf: 0.25", "conf: 0.26"))
    with pytest.raises(vm.VideoRuntimeError, match="SCIENTIFIC_STATE_MISMATCH"):
        vm.frozen_settings(tmp_path)


def test_checkpoint_corruption_rejected(tmp_path):
    path = tmp_path / "model.pt"
    path.write_bytes(b"same size")
    with pytest.raises(ValueError, match="SHA256_MISMATCH"):
        verify_file(path, sha256="0" * 64, size_bytes=9)


def test_only_frozen_copy_is_accepted(tmp_path):
    for relative in [
        "delivery/checkpoints.json",
        "reports/final_detector_manifest.json",
        "reports/final_segmenter_manifest.json",
    ]:
        path = tmp_path / relative
        path.parent.mkdir(exist_ok=True)
        path.write_bytes((ROOT / relative).read_bytes())
    with pytest.raises(vm.VideoRuntimeError, match="BLOCKED_MISSING_CHECKPOINT"):
        vm.verified_checkpoints(tmp_path, "compare")


def test_masks_are_on_original_canvas_and_compare_uses_same_pixels():
    frame = np.full((120, 240, 3), 100, np.uint8)
    mask = np.zeros((120, 240), bool)
    mask[60:80, 150:170] = True
    row = {"class_id": 4, "confidence": 0.9, "box": [145, 55, 175, 85], "mask": mask}
    rendered = render_frame(frame, {"D2": [], "S1": [row]}, "compare", 10, fps_overlay=False)
    assert np.array_equal(frame, np.full_like(frame, 100))
    assert np.array_equal(rendered[65, 155], frame[65, 155])
    expected = np.rint((1 - MASK_ALPHA) * 100 + MASK_ALPHA * np.array(COLORS[4]))
    assert np.array_equal(rendered[65, 240 + 155], expected)
    assert np.array_equal(rendered[95, 440], frame[95, 200])
    row["mask"] = np.ones((60, 120), bool)
    with pytest.raises(vm.VideoRuntimeError, match="mask/canvas"):
        render_panel(frame, [row], "S1")


def test_source_metadata_rejects_unknown_and_signed_url():
    with pytest.raises(vm.VideoRuntimeError):
        vr.source_metadata({"extra": "bad"})
    value = dict.fromkeys(vr.SOURCE_FIELDS)
    value.update(
        source_type="EXTERNAL_REAL_VIDEO", url="?".join(["https://example.org/video", "x=y"])
    )
    with pytest.raises(vm.VideoRuntimeError, match=r"public HTTPS|sensitive information"):
        vr.source_metadata(value)


def test_no_holdout_training_or_tracking_route():
    import ast

    for relative in [
        "scripts/run_video_demo.py",
        "src/construction_safety_vision/video_models.py",
        "src/construction_safety_vision/video_runtime.py",
        "src/construction_safety_vision/video_render.py",
    ]:
        tree = ast.parse((ROOT / relative).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not any(
                    s in (node.module or "") for s in ["final_holdout", "split_freeze", "tracker"]
                )
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr not in ["train", "track", "val", "image_ids"]


def test_dataset_input_rejected_before_read(tmp_path):
    with pytest.raises(vm.VideoRuntimeError, match="Dataset paths"):
        vr.validate_paths(
            tmp_path,
            tmp_path / "data/processed/protected.mp4",
            tmp_path / "outputs/out.mp4",
            "compare",
            "cpu",
            False,
            None,
        )
