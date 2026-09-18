"""Protect the artifact-only training and evaluation sections of the delivery notebook."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from construction_safety_vision import delivery_academic as academic
from construction_safety_vision import delivery_demo as demo

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/construction_safety_vision_demo.ipynb"
MODULES = [
    "src/construction_safety_vision/delivery_demo.py",
    "src/construction_safety_vision/delivery_academic.py",
]


@pytest.fixture
def public_copy(tmp_path):
    for relative in [*academic.SOURCES, *academic.FIGURES, *MODULES]:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    return tmp_path


def test_sections_render_without_data_weights_or_heavy_imports(public_copy):
    script = (
        "import runpy, sys, pathlib; "
        "root=pathlib.Path(sys.argv[1]); "
        "m=runpy.run_path(str(root/'src/construction_safety_vision/delivery_academic.py')); "
        "html=m['dataset_html'](root)+m['training_html'](root,'D2')"
        "+m['training_html'](root,'S1')+m['evaluation_html'](root)+m['qualitative_html'](root); "
        "assert '0.427031' in html and '0.410143' in html and '0.834548' in html; "
        "assert 'RECORDED_TRAINING_EVIDENCE' in html and 'NO_NEW_TRAINING_EXECUTED' in html; "
        "assert not (root/'data').exists() and not (root/'artifacts').exists(); "
        "assert not {'torch','ultralytics','numpy','yaml'} & sys.modules.keys(); "
        "print('ACADEMIC_METADATA_ONLY_OK')"
    )
    result = subprocess.run(
        [sys.executable, "-S", "-c", script, str(public_copy)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "ACADEMIC_METADATA_ONLY_OK"


def test_training_recipe_matches_the_committed_manifests(public_copy):
    catalog = academic.training_catalog(public_copy)
    d2 = json.loads((ROOT / "reports/detection_D2_manifest.json").read_text(encoding="utf-8"))
    s1 = json.loads(
        (ROOT / "reports/segmentation_S1_result_manifest.json").read_text(encoding="utf-8")
    )
    arguments = d2["resolved_training_arguments"]
    assert dict(catalog["D2"]["optimisation"])["imgsz"] == arguments["imgsz"]
    assert dict(catalog["S1"]["segmentation"]) == {"mask_ratio": 4, "overlap_mask": False}
    assert catalog["D2"]["optimizer"]["resolved"] == d2["resolved_optimizer"]
    assert catalog["S1"]["optimizer"]["resolved"] == s1["optimizer"]["actual_resolved_optimizer"]
    assert dict(catalog["D2"]["outcome"])["selected checkpoint epoch"] == 90
    assert dict(catalog["S1"]["outcome"])["selected checkpoint epoch"] == 77
    assert catalog["D2"]["checkpoint"]["sha256"] == d2["best_checkpoint"]["sha256"]
    assert catalog["S1"]["checkpoint"]["sha256"] == s1["checkpoints"]["best"]["sha256"]


def test_declared_training_argument_names_must_exist(public_copy):
    with pytest.raises(ValueError, match="missing"):
        academic.select({"epochs": 100}, academic.OPTIMISATION_KEYS)


def test_generalisation_rows_reproduce_the_committed_report(public_copy):
    rows = academic.evaluation_catalog(public_copy)["generalisation"]
    report = (ROOT / "reports/final_test_evaluation.md").read_text(encoding="utf-8")
    assert len(rows) == len(academic.GENERALISATION)
    for row in rows:
        cells = f"| {row['validation']:.6f} | {row['test']:.6f} | {row['difference']:.6f} |"
        assert cells in report


def test_evaluation_refuses_a_winner_or_a_composite(public_copy):
    record = json.loads(
        (public_copy / "reports/final_test_detector.json").read_text(encoding="utf-8")
    )
    record["winner_declared"] = True
    (public_copy / "reports/final_test_detector.json").write_text(
        json.dumps(record), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="winner"):
        academic.evaluation_catalog(public_copy)


def test_evaluation_refuses_a_record_from_another_split(public_copy):
    record = json.loads(
        (public_copy / "reports/final_test_segmenter.json").read_text(encoding="utf-8")
    )
    record["split"] = "validation"
    (public_copy / "reports/final_test_segmenter.json").write_text(
        json.dumps(record), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="holdout split"):
        academic.evaluation_catalog(public_copy)


def test_holdout_opt_in_stops_every_section(monkeypatch):
    monkeypatch.setenv("CSVISION_ALLOW_TEST_SPLIT", "1")
    for call in (academic.training_catalog, academic.evaluation_catalog, academic.dataset_html):
        with pytest.raises(ValueError, match="CSVISION_ALLOW_TEST_SPLIT"):
            call(ROOT)


def test_only_declared_figures_are_displayable(public_copy):
    for name in academic.EXPERIMENTS:
        assert academic.training_curve(public_copy, name).is_file()
    manifest = public_copy / "reports/detection_D2_manifest.json"
    record = json.loads(manifest.read_text(encoding="utf-8"))
    record["committed_figures"] = []
    manifest.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="declared"):
        academic.training_curve(public_copy, "D2")


def test_confusion_matrices_come_from_counts_not_from_final_test_figures(public_copy):
    data = academic.evaluation_catalog(public_copy)
    record = json.loads((ROOT / "reports/final_test_detector.json").read_text(encoding="utf-8"))
    rendered = academic.confusion_table("D2", data["detector"])
    for row in record["confusion_matrix"]["matrix"]:
        assert "".join(f"<td>{value}</td>" for value in row) in rendered
    assert "final_test" not in "".join(academic.FIGURES)


def test_a_malformed_confusion_matrix_is_refused(public_copy):
    data = academic.evaluation_catalog(public_copy)
    data["detector"]["confusion_matrix"]["matrix"] = [[1, 2], [3, 4]]
    with pytest.raises(ValueError, match="not square"):
        academic.confusion_table("D2", data["detector"])


def test_no_holdout_identifier_reaches_the_notebook_or_the_module():
    text = NOTEBOOK.read_text(encoding="utf-8")
    text += (ROOT / "src/construction_safety_vision/delivery_academic.py").read_text(
        encoding="utf-8"
    )
    for token in ("artifacts/final_test/", "holdout_sha256", "CSVISION_ALLOW_TEST_SPLIT=1"):
        assert token not in text
    assert "allow_test" not in text


def test_notebook_exposes_training_evaluation_and_inference_sections():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    markdown = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "markdown"
    )
    code = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    for heading in (
        "## 1. Visão geral",
        "## 2. Dataset e classes",
        "## 3. TREINO / TRAINING",
        "## 4. AVALIAÇÃO / EVALUATION",
        "## 5. Evidência qualitativa",
        "## 6. Evidência em vídeo",
        "## 7. INFERÊNCIA / INFERENCE",
        "## 8. Limitações e licenças",
    ):
        assert heading in markdown
    assert "NO_NEW_TRAINING_EXECUTED" in markdown
    assert "FINAL TEST RESULTS" in markdown
    joined = "\n".join(code)
    for call in ('show_training"](ROOT, "D2")', 'show_training"](ROOT, "S1")', "show_evaluation"):
        assert call in joined
    assert "evaluate_final_holdout" not in joined
    assert code[-1].strip().endswith('print("Mode A complete. No inference output was created.")')


def test_mode_b_notebook_cells_are_unchanged_from_the_validated_revision():
    validated = subprocess.check_output(
        [
            "git",
            "show",
            "a788d5303e70ddb12f5dc5ae35dd9a4b56fc6736:" + NOTEBOOK.relative_to(ROOT).as_posix(),
        ],
        cwd=ROOT,
    )
    before = [
        "".join(cell["source"])
        for cell in json.loads(validated.decode("utf-8"))["cells"]
        if cell["cell_type"] == "code"
    ]
    after = [
        "".join(cell["source"])
        for cell in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
        if cell["cell_type"] == "code"
    ]
    # Every Mode A display cell and every Mode B execution cell is carried over verbatim;
    # only the bootstrap cell changed, to load the second standalone support module.
    assert before[0] == after[0]
    assert before[2] == after[2]
    assert before[3:] == after[8:]
    assert demo.WEIGHTS and demo.MODES
