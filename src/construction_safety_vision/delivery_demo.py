"""Read published evidence and stage optional frozen-model demos for a thin Colab notebook.

The metadata path uses only the standard library. Notebook display and runtime
imports happen only inside their explicit functions. No dataset API is imported.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

EVIDENCE_COMMIT = "fa0bea2b072b124d65f4e9bd1b932a22d98149d2"
REPOSITORY = (
    "https://github.com/Novachrono117/"
    "Construction-Safety-Vision-PPE-Detection-Instance-Segmentation-Video-Analytics"
)
WEIGHTS = {
    "D2": ("detector", "artifacts/frozen/detection/D2_best.pt"),
    "S1": ("segmenter", "artifacts/frozen/segmentation/S1_best.pt"),
}
MODES = {"detector": ("D2",), "segmenter": ("S1",), "compare": ("D2", "S1")}
REPORTS = (
    "final_test_detector.json",
    "final_test_segmenter.json",
    "detector_segmenter_box_comparison.json",
    "qualitative_validation_gallery.json",
    "final_real_video_demo.json",
    "final_detector_manifest.json",
    "final_segmenter_manifest.json",
    "final_test_evaluation.md",
    "detector_segmenter_scientific_synthesis.md",
    "qualitative_validation_gallery.md",
    "final_real_video_demo.md",
)
MAX_VIDEO_BYTES = 2 * 1024**3


def read(root: Path, path: str) -> dict[str, Any]:
    """Read one explicitly selected public record."""
    return json.loads((root / path).read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    """Hash a supplied artifact without deserialization."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inside(root: Path, relative: str) -> Path:
    """Refuse links or path traversal that leave the named storage root."""
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError("Path leaves the allowed storage root")
    return target


def identities(root: Path) -> dict[str, dict[str, Any]]:
    """Cross-check public identities against both immutable freeze records."""
    entries = read(root, "delivery/checkpoints.json")["checkpoints"]
    result = {entry["logical_name"]: entry for entry in entries}
    if set(result) != set(WEIGHTS) or len(entries) != 2:
        raise ValueError("Checkpoint metadata does not identify exactly D2 and S1")
    for name, (task, relative) in WEIGHTS.items():
        freeze = read(root, f"reports/final_{task}_manifest.json")
        expected = result[name]
        if (
            freeze["status"] != "FROZEN"
            or freeze["selection_status"] != "FINAL_SELECTED"
            or freeze["selected_experiment"] != name
            or freeze["model"] != expected["architecture"]
        ):
            raise ValueError("Frozen model metadata mismatch")
        for key in ("selected_checkpoint", "frozen_copy"):
            if (
                freeze[key]["sha256"] != expected["sha256"]
                or freeze[key]["size_bytes"] != expected["bytes"]
            ):
                raise ValueError("Frozen checkpoint identity mismatch")
        if freeze["frozen_copy"]["relative_path"] != relative:
            raise ValueError("Frozen checkpoint path mismatch")
    return result


def verify(path: Path, expected: dict[str, Any]) -> None:
    """Reject wrong size or SHA-256 before any model can be loaded."""
    if not path.is_file() or path.stat().st_size != expected["bytes"]:
        raise ValueError("CHECKPOINT_SIZE_MISMATCH_OR_MISSING")
    if digest(path) != expected["sha256"]:
        raise ValueError("CHECKPOINT_SHA256_MISMATCH")


def catalog(root: Path) -> dict[str, Any]:
    """Project published aggregates and approved figures, without opening data or models."""
    assert_delivery_context()
    d2 = read(root, "reports/final_test_detector.json")
    s1 = read(root, "reports/final_test_segmenter.json")
    box = read(root, "reports/detector_segmenter_box_comparison.json")
    gallery = read(root, "reports/qualitative_validation_gallery.json")
    video = read(root, "reports/final_real_video_demo.json")
    if gallery["split"] != "validation" or gallery["holdout_content_accessed"]:
        raise ValueError("Gallery must contain approved validation evidence only")
    if d2["split"] != "test" or s1["split"] != "test":
        raise ValueError("Final result metadata mismatch")
    return {
        "checkpoints": identities(root),
        "final_metrics": [
            {
                "label": "D2 · box",
                "value": d2["canonical_box"]["CANONICAL_TEST_BOX_MAP50_95"],
                "source": "reports/final_test_detector.json",
            },
            {
                "label": "S1 · box",
                "value": s1["canonical_box"]["S1_CANONICAL_TEST_BOX_MAP50_95"],
                "source": "reports/final_test_segmenter.json",
            },
            {
                "label": "S1 · mask",
                "value": s1["canonical_mask"]["CANONICAL_TEST_MASK_MAP50_95"],
                "source": "reports/final_test_segmenter.json",
            },
        ],
        "validation_box": box["metrics"]["CANONICAL_BOX_MAP50_95"],
        "validation_sensitivity": box["supported_class_sensitivity"],
        "gallery": gallery["figures"],
        "gallery_attribution": gallery["attribution"],
        "matching": gallery["policy"]["matching"],
        "inference": gallery["policy"]["inference"],
        "errors": {
            name: [
                {k: row[k] for k in ("category", "class", "available", "status")} for row in rows
            ]
            for name, rows in gallery["selections"].items()
        },
        "screenshots": video["screenshots"]["screenshots"],
        "observations": video["temporal_review"]["observations"],
        "video": {
            "frames": video["execution"]["processed_frames"],
            "output": video["execution"]["output"],
            "source": video["execution"]["source"],
            "storage": video["storage"],
        },
    }


def assert_delivery_context() -> None:
    """Stop if a holdout opt-in is present; never silently change the environment."""
    if "CSVISION_ALLOW_TEST_SPLIT" in os.environ:
        raise ValueError("CSVISION_ALLOW_TEST_SPLIT must be unset for the delivery demo")


def source_link(path: str, label: str) -> str:
    """Link to the historical evidence revision rather than a moving report."""
    return f'<a href="{REPOSITORY}/blob/{EVIDENCE_COMMIT}/{path}">{html.escape(label)}</a>'


def overview_html(root: Path) -> str:
    """Render exact published values with task, split, source and interpretation labels."""
    data = catalog(root)
    cards = "".join(
        '<article style="padding:20px;border:1px solid #ced7cc;flex:1;min-width:180px">'
        f"<h3>{html.escape(row['label'])}</h3>"
        f'<p style="font:32px monospace">{row["value"]:.6f}</p>'
        f"{source_link(row['source'], 'Committed source')}</article>"
        for row in data["final_metrics"]
    )
    models = " / ".join(
        f"{name}: {entry['architecture']} @ {entry['imgsz']}"
        for name, entry in data["checkpoints"].items()
    )
    validation = data["validation_box"]
    sensitivity = data["validation_sensitivity"]
    synthesis = source_link(
        "reports/detector_segmenter_scientific_synthesis.md", "Scientific synthesis"
    )
    # Host notebook themes style headings/links directly, overriding inherited color.
    # Scope the fixed light palette to this panel; other outputs use the host theme.
    return (
        "<style>"
        ".csv-evidence-overview h1,.csv-evidence-overview h2,.csv-evidence-overview h3"
        "{color:#202923 !important}"
        ".csv-evidence-overview a,.csv-evidence-overview a:visited"
        "{color:#075985 !important;text-decoration:underline}"
        ".csv-evidence-overview a:hover{color:#0c4a6e !important}"
        ".csv-evidence-overview a:focus-visible"
        "{outline:2px solid #075985;outline-offset:3px}"
        ".csv-evidence-overview code"
        "{color:#202923 !important;background:#e6eade !important;padding:2px 4px}"
        "</style>"
        '<section class="csv-evidence-overview" style="color-scheme:light;'
        "background:#f6f7f2;color:#202923;padding:24px;"
        'font:16px/1.6 system-ui;border-radius:10px">'
        '<p style="color:#923b10;letter-spacing:1px">MODE A / COMMITTED EVIDENCE</p>'
        "<h1>Construction Safety Vision</h1>"
        "<p>Scientific lifecycle closed. D2 and S1 frozen. No model is loaded.</p>"
        "<h2>Architecture</h2><ol>"
        "<li>Canonical COCO instance geometry; detection boxes derived from masks.</li>"
        "<li>Frozen group-aware and class-aware constrained split, aligned across tasks.</li>"
        "<li>Derived detection adapter and audited approximate segmentation adapter.</li>"
        f"<li>{html.escape(models)}. Frame-by-frame video inference; no tracking.</li>"
        "<li>Published comparisons, final evaluation and qualitative delivery evidence.</li></ol>"
        f"{synthesis}"
        "<h2>Final test · canonical COCO mAP@0.50:0.95</h2>"
        "<p>Recorded one-shot results. The holdout is observed, spent and locked.</p>"
        f'<div style="display:flex;flex-wrap:wrap;gap:12px">{cards}</div>'
        "<p>Box and mask measure different tasks. No composite score or new model ranking.</p>"
        "<h2>Validation · box comparison</h2>"
        f"<p>D2 {validation['D2']:.6f}; S1 {validation['S1']:.6f}. "
        "The rare class <code>vest_loose</code> has only one validation image. "
        "The all-class difference is sensitive to this class.</p>"
        "<p>Post-hoc descriptive supported-class sensitivity: "
        f"D2 {sensitivity['D2_supported_macro']:.6f}; "
        f"S1 {sensitivity['S1_supported_macro']:.6f}. "
        "This is not a selection rule or a significance test.</p>"
        f"{source_link('reports/detector_segmenter_box_comparison.json', 'Validation source')}"
        "<p>Final-test per-class uncertainty also remains high: vest_loose has two images. "
        "Framework-native and canonical COCO metrics are not interchangeable.</p></section>"
    )


def show_overview(root: Path) -> None:
    """Display the lightweight overview inside IPython/Colab."""
    from IPython.display import HTML, display

    display(HTML(overview_html(root)))


def figure_path(root: Path, relative: str) -> Path:
    """Allow only committed gallery/video evidence, verifying its recorded digest."""
    data = catalog(root)
    shots = {shot["path"]: shot["identity"] for shot in data["screenshots"]}
    if relative in data["gallery"] and relative.startswith("reports/figures/qualitative/"):
        provenance = read(root, "reports/qualitative_validation_gallery.provenance.json")
        record = next(row for row in provenance["outputs"] if row["path"] == relative)
        expected = {"sha256": record["sha256"], "bytes": record["size_bytes"]}
    elif relative in shots and relative.startswith("reports/figures/final_video/"):
        expected = shots[relative]
    else:
        raise ValueError("Only approved public validation/video figures may be displayed")
    path = inside(root, relative)
    verify(path, expected)
    return path


def show_gallery(root: Path, view: str = "D2") -> None:
    """Display an existing figure and its recorded counts, without recomputing errors."""
    from IPython.display import HTML, Image, display

    data = catalog(root)
    choices = {"D2": 0, "S1": 1, "mask quality": 2, "box versus mask": 3}
    if view not in choices:
        raise ValueError("Unknown gallery view")
    figure = figure_path(root, data["gallery"][choices[view]])
    credit = data["gallery_attribution"]
    display(
        HTML(
            f"<h2>Validation gallery · {html.escape(view)}</h2>"
            "<p>FP: unmatched prediction; FN: unmatched canonical object. Class-aware "
            "one-to-one box matching at IoU 0.50, confidence 0.25. "
            "Unmatched predictions do not establish that no real object exists. "
            "Causes are untested.</p>"
            f"<p>{html.escape(credit['credit'])} · {html.escape(credit['license'])}. "
            f"{html.escape(credit['changes'])}. "
            f'<a href="{html.escape(credit["source"], quote=True)}">Source</a> · '
            f'<a href="{html.escape(credit["license_url"], quote=True)}">License</a></p>'
        )
    )
    display(Image(filename=str(figure), width=1400))
    if view in data["errors"]:
        rows = "".join(
            f"<tr><td>{html.escape(row['class'])}</td><td>{row['category']}</td>"
            f"<td>{row['available']}</td><td>{row['status']}</td></tr>"
            for row in data["errors"][view]
        )
        display(
            HTML(
                "<table><caption>Recorded validation counts, not a new evaluation</caption>"
                "<tr><th>Class</th><th>Error</th><th>Available</th><th>Selection</th></tr>"
                + rows
                + "</table>"
            )
        )
    display(HTML(source_link("reports/qualitative_validation_gallery.md", "Full gallery report")))


def show_video_evidence(root: Path, pair: int = 1) -> None:
    """Show one of the three previously reviewed frame pairs, never sample new frames."""
    from IPython.display import HTML, Image, display

    if type(pair) is not int or pair not in (1, 2, 3):
        raise ValueError("Choose frame pair 1, 2 or 3")
    data = catalog(root)
    source = data["video"]["source"]
    obs = data["observations"][pair - 1]
    display(
        HTML(
            f"<h2>Real-video evidence · {source['duration_seconds']:.2f} seconds · "
            f"{data['video']['frames']} frames · {source['fps']:g} FPS playback</h2>"
            "<p>D2 boxes left; S1 masks right. Full MP4 distribution is pending. "
            "These six committed frames are available without checkpoints. "
            "Footage: Frank Vincentz / Wikimedia Commons / CC BY-SA 3.0; overlays added.</p>"
            f"{source_link('delivery/FINAL_VIDEO_ATTRIBUTION.md', 'Full attribution')}"
            f"<h3>{html.escape(obs['evidence_level'])}</h3>"
            f"<p>{html.escape(obs['visible_behavior'])}</p>"
            f"<p>{html.escape(obs['interpretation'])}</p>"
        )
    )
    for shot in data["screenshots"][2 * (pair - 1) : 2 * pair]:
        display(
            HTML(
                f"<p>{shot['timestamp_seconds']:.2f} s · "
                f"frame {shot['frame_zero_based']} (zero-based)</p>"
            )
        )
        display(Image(filename=str(figure_path(root, shot["path"])), width=1400))
    display(
        HTML(
            "<p>This is qualitative delivery evidence, not video precision/recall, "
            "tracking or a claim of real-time inference.</p>"
            + source_link("reports/final_real_video_demo.md", "Full video report")
        )
    )


def stage_checkpoint(root: Path, uploaded: Path, name: str) -> Path:
    """Publish verified bytes at the frozen path without replacing an existing artifact."""
    assert_delivery_context()
    if name not in WEIGHTS:
        raise ValueError("Only the frozen D2 and S1 checkpoints are accepted")
    expected = identities(root)[name]
    verify(uploaded, expected)
    destination = inside(root, WEIGHTS[name][1])
    if destination.exists():
        verify(destination, expected)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Uploads live under outputs in this checkout, on the same filesystem.
    # Linking is atomic and refuses to overwrite any pre-existing destination.
    destination.hardlink_to(uploaded)
    return destination


def colab_upload_checkpoints(root: Path, mode: str) -> None:
    """Ask for exact weights, writing uploads only into a fresh ignored directory."""
    assert_delivery_context()
    if mode not in MODES:
        raise ValueError("Unknown model view")
    from google.colab import files

    for name in MODES[mode]:
        expected = identities(root)[name]
        print(f"Upload {name}: {expected['bytes']} bytes; SHA-256 {expected['sha256']}")
        folder = inside(root, "outputs/colab_demo/" + uuid.uuid4().hex)
        folder.mkdir(parents=True)
        uploaded = folder / "checkpoint.pt"
        try:
            files.upload_file(str(uploaded))
            stage_checkpoint(root, uploaded, name)
            print(f"{name}: VERIFIED before deserialization")
        finally:
            uploaded.unlink(missing_ok=True)
            folder.rmdir()


def colab_upload_video(root: Path) -> Path:
    """Upload an external MP4 to a new ignored folder, never a dataset directory."""
    assert_delivery_context()
    from google.colab import files

    folder = inside(root, "outputs/colab_demo/" + uuid.uuid4().hex)
    folder.mkdir(parents=True)
    path = folder / "external_video.mp4"
    files.upload_file(str(path))
    if not 0 < path.stat().st_size <= MAX_VIDEO_BYTES:
        path.unlink()
        raise ValueError("Upload an MP4 between 1 byte and 2 GiB")
    return path


def run_external_video(
    root: Path, uploaded: Path, *, mode: str, device: str, confirmed_external: bool
) -> Path:
    """Run the approved CLI only after all frozen weights and input boundaries pass."""
    assert_delivery_context()
    if mode not in MODES or device not in ("cpu", "cuda"):
        raise ValueError("Unsupported model view or device")
    if confirmed_external is not True:
        raise ValueError("Confirm an authorized external video, outside the research dataset")
    upload_root = inside(root, "outputs/colab_demo")
    if not uploaded.resolve().is_relative_to(upload_root) or not uploaded.is_file():
        raise ValueError("Use the external-video upload cell")
    expected = identities(root)
    for name in MODES[mode]:
        verify(inside(root, WEIGHTS[name][1]), expected[name])
    interpreter = root / (".venv/Scripts/python.exe" if os.name == "nt" else ".venv/bin/python")
    if not interpreter.is_file():
        raise ValueError("Install Mode B with uv sync --locked --python 3.12 --no-dev")
    output = uploaded.parent / ("demo_" + uuid.uuid4().hex + ".mp4")
    command = [
        str(interpreter),
        "scripts/run_video_demo.py",
        "--input",
        str(uploaded),
        "--output",
        str(output),
        "--mode",
        mode,
        "--device",
        device,
    ]
    subprocess.run(command, cwd=root, check=True)
    sidecar = output.with_suffix(".mp4.provenance.json")
    record = json.loads(sidecar.read_text(encoding="utf-8"))
    if record["status"] != "COMPLETE":
        raise ValueError("Runtime did not produce a complete result")
    verify(output, record["output"])
    return output


def install_runtime(root: Path) -> None:
    """Install the existing locked environment only after the notebook Mode B opt-in."""
    assert_delivery_context()
    subprocess.run([sys.executable, "-m", "pip", "install", "uv==0.12.13"], check=True)
    subprocess.run(
        [sys.executable, "-m", "uv", "sync", "--locked", "--python", "3.12", "--no-dev"],
        cwd=root,
        check=True,
    )


def colab_view_final_video(root: Path) -> None:
    """Optionally verify and view an uploaded recorded output, with no model access."""
    assert_delivery_context()
    from google.colab import files

    data = catalog(root)
    folder = inside(root, "outputs/colab_demo/" + uuid.uuid4().hex)
    folder.mkdir(parents=True)
    path = folder / "recorded_final_video.mp4"
    print("Upload the existing final MP4. This does not execute inference.")
    try:
        files.upload_file(str(path))
        verify(path, data["video"]["output"])
    except (OSError, ValueError):
        path.unlink(missing_ok=True)
        folder.rmdir()
        raise
    print("Recorded MP4 identity verified. Browser playback depends on MP4/mp4v support.")
    files.view(str(path))
