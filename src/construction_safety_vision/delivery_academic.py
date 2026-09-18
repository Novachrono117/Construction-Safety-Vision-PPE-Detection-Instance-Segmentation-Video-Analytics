"""Present the recorded training recipe and the committed results for the delivery notebook.

Artifact-only. Nothing here trains, tunes, selects, infers or recomputes a
metric: every value is read by field from a record already committed to this
repository, and the spent one-shot holdout results are quoted rather than
reproduced. The metadata path uses only the standard library, and notebook
display is imported inside the functions that need it.

The existing delivery module is loaded by path rather than copied, so its
verification helpers stay the single implementation and its bytes stay
unchanged.
"""

from __future__ import annotations

import html
import runpy
from pathlib import Path
from typing import Any

DEMO_MODULE = "src/construction_safety_vision/delivery_demo.py"
TRAINING_EVIDENCE = "RECORDED_TRAINING_EVIDENCE"
NO_NEW_TRAINING = "NO_NEW_TRAINING_EXECUTED"
FINAL_TEST_BANNER = "FINAL TEST RESULTS - READ FROM THE COMMITTED ONE-SHOT EVALUATION"
POPULATION_RECORD = "reports/canonical_modeling_manifest.json"

EXPERIMENTS = {
    "D2": {
        "task": "detection",
        "manifest": "reports/detection_D2_manifest.json",
        "freeze": "reports/final_detector_manifest.json",
        "arguments": "resolved_training_arguments",
        "curve": "reports/figures/detection/D2/results.png",
        "command": "uv run python scripts/train_detection_experiment.py --experiment D2",
        "configs": ("configs/detection_baseline.yaml", "configs/detection_experiments.yaml"),
    },
    "S1": {
        "task": "segmentation",
        "manifest": "reports/segmentation_S1_result_manifest.json",
        "freeze": "reports/final_segmenter_manifest.json",
        "arguments": "effective_training_configuration",
        "curve": "reports/figures/segmentation_S1/results.png",
        "command": "uv run python scripts/train_segmentation_comparison.py",
        "configs": ("configs/segmentation_baseline.yaml", "configs/segmentation_comparison.yaml"),
    },
}
OPTIMISATION_KEYS = (
    "epochs",
    "batch",
    "imgsz",
    "seed",
    "deterministic",
    "amp",
    "optimizer",
    "patience",
    "lr0",
    "lrf",
    "momentum",
    "weight_decay",
    "warmup_epochs",
    "cos_lr",
)
AUGMENTATION_KEYS = (
    "mosaic",
    "close_mosaic",
    "fliplr",
    "flipud",
    "scale",
    "translate",
    "degrees",
    "shear",
    "perspective",
    "hsv_h",
    "hsv_s",
    "hsv_v",
    "erasing",
    "mixup",
    "cutmix",
    "copy_paste",
    "copy_paste_mode",
    "auto_augment",
)
SEGMENTATION_KEYS = ("mask_ratio", "overlap_mask")
# The confusion matrices are rendered from their committed counts rather than
# from `reports/figures/final_test/`. That directory holds rendered final-test
# figures, which the delivery surface deliberately refuses to display and the
# metadata-only test boundary treats as protected; the counts carry the same
# information with no dependency on it.
SOURCES = (
    "delivery/checkpoints.json",
    POPULATION_RECORD,
    "reports/final_test_detector.json",
    "reports/final_test_segmenter.json",
    "reports/final_test_direct_iou.json",
    "reports/detector_segmenter_box_comparison.json",
    "reports/segmentation_S1_canonical_evaluation.json",
    "reports/segmentation_S1_mask_iou.json",
    *(entry["manifest"] for entry in EXPERIMENTS.values()),
    *(entry["freeze"] for entry in EXPERIMENTS.values()),
)
FIGURES = tuple(entry["curve"] for entry in EXPERIMENTS.values())
# Each pair existed before the holdout was read and is quoted from its own
# committed artifact by field. No new metric, subgroup or significance test.
GENERALISATION = (
    ("D2 canonical box mAP@0.50:0.95", "box", ("CANONICAL_BOX_MAP50_95", "D2"), "d2_map50_95"),
    ("D2 canonical box mAP@0.50", "box", ("CANONICAL_BOX_MAP50", "D2"), "d2_map50"),
    ("S1 canonical box mAP@0.50:0.95", "box", ("CANONICAL_BOX_MAP50_95", "S1"), "s1_box_map50_95"),
    ("S1 canonical box mAP@0.50", "box", ("CANONICAL_BOX_MAP50", "S1"), "s1_box_map50"),
    ("S1 canonical mask mAP@0.50:0.95", "mask", ("all_class_map50_95",), "s1_mask_map50_95"),
    ("S1 canonical mask mAP@0.50", "mask", ("all_class_map50",), "s1_mask_map50"),
    ("S1 direct matched_mask_iou_mean", "iou", ("matched_mask_iou_mean",), "matched_mask_iou_mean"),
    (
        "S1 direct gt_normalized_mask_iou",
        "iou",
        ("gt_normalized_mask_iou",),
        "gt_normalized_mask_iou",
    ),
    ("S1 direct gt_match_coverage", "iou", ("gt_match_coverage",), "gt_match_coverage"),
    ("S1 direct gt_iou50_coverage", "iou", ("gt_iou50_coverage",), "gt_iou50_coverage"),
    ("S1 direct gt_iou75_coverage", "iou", ("gt_iou75_coverage",), "gt_iou75_coverage"),
)
PANEL = (
    "<style>"
    ".csv-academic h1,.csv-academic h2,.csv-academic h3,.csv-academic h4{color:#202923 !important}"
    ".csv-academic a,.csv-academic a:visited{color:#075985 !important;text-decoration:underline}"
    ".csv-academic a:hover{color:#0c4a6e !important}"
    ".csv-academic a:focus-visible{outline:2px solid #075985;outline-offset:3px}"
    ".csv-academic code{color:#202923 !important;background:#e6eade !important;padding:2px 4px}"
    ".csv-academic table{border-collapse:collapse;margin:10px 0;width:100%}"
    ".csv-academic caption{text-align:left;font-weight:700;padding:6px 0}"
    ".csv-academic th,.csv-academic td"
    "{border:1px solid #ced7cc;padding:5px 9px;text-align:left;font-size:14px}"
    ".csv-academic th{background:#e6eade}"
    ".csv-academic .tag{color:#923b10;letter-spacing:1px;font-weight:700}"
    "</style>"
)
_SUPPORT: dict[str, dict[str, Any]] = {}


def support(root: Path) -> dict[str, Any]:
    """Load the existing delivery module by path, without importing the package."""
    key = str(root.resolve())
    if key not in _SUPPORT:
        _SUPPORT[key] = runpy.run_path(str(root / DEMO_MODULE))
    return _SUPPORT[key]


OPEN_PANEL = (
    '<section class="csv-academic" style="color-scheme:light;background:#f6f7f2;color:#202923;'
    'padding:24px;font:16px/1.6 system-ui;border-radius:10px">'
)


def panel(body: str) -> str:
    """Wrap rendered content in the notebook's fixed light evidence palette."""
    return PANEL + OPEN_PANEL + body + "</section>"


def cell(value: Any) -> str:
    """Render a stored value exactly; never round a recorded number for display."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    if value is None:
        return "not applicable"
    return html.escape(str(value))


def table(caption: str, headers: tuple[str, ...], rows: list[tuple[Any, ...]]) -> str:
    """Build one labelled table of already-recorded values."""
    head = "".join(f"<th>{html.escape(name)}</th>" for name in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{cell(value)}</td>" for value in row) + "</tr>" for row in rows
    )
    return f"<table><caption>{html.escape(caption)}</caption><tr>{head}</tr>{body}</table>"


def select(arguments: dict[str, Any], keys: tuple[str, ...]) -> list[tuple[str, Any]]:
    """Read exactly the declared argument names; a missing name is a defect."""
    missing = [key for key in keys if key not in arguments]
    if missing:
        raise ValueError(f"Recorded training arguments are missing: {', '.join(missing)}")
    return [(key, arguments[key]) for key in keys]


def training_catalog(root: Path) -> dict[str, dict[str, Any]]:
    """Project each frozen model's recorded training recipe and execution outcome."""
    demo = support(root)
    demo["assert_delivery_context"]()
    identities = demo["identities"](root)
    catalog = {}
    for name, entry in EXPERIMENTS.items():
        manifest = demo["read"](root, entry["manifest"])
        freeze = demo["read"](root, entry["freeze"])
        arguments = manifest[entry["arguments"]]
        if freeze["selected_experiment"] != name or freeze["selection_status"] != "FINAL_SELECTED":
            raise ValueError("Frozen model metadata mismatch")
        if manifest.get("holdout_accessed", False) or freeze.get("holdout_accessed", False):
            raise ValueError("A training record must not claim holdout access")
        catalog[name] = {
            "task": entry["task"],
            "architecture": freeze["model"],
            "checkpoint": identities[name],
            "optimisation": select(arguments, OPTIMISATION_KEYS),
            "augmentation": select(arguments, AUGMENTATION_KEYS),
            "segmentation": select(arguments, SEGMENTATION_KEYS) if name == "S1" else [],
            "optimizer": resolved_optimizer(name, manifest),
            "outcome": outcome(name, manifest, freeze),
            "command": entry["command"],
            "configs": entry["configs"],
            "curve": entry["curve"],
            "manifest": entry["manifest"],
            "freeze": entry["freeze"],
            "runtime": manifest["runtime"],
            "complexity": manifest["model_complexity"],
            "pretrained": manifest["pretrained_weights"],
        }
    return catalog


def resolved_optimizer(name: str, manifest: dict[str, Any]) -> dict[str, Any]:
    """Report what `optimizer: auto` actually resolved to, never the file's policy input."""
    if name == "S1":
        record = manifest["optimizer"]
        return {
            "declared_policy": record["declared_optimizer_policy"],
            "resolved": record["actual_resolved_optimizer"],
            "effective_lr0": record["effective_lr0"],
            "effective_momentum": record["effective_momentum"],
            "source": record["resolution_evidence"],
        }
    record = manifest["resolved_optimizer_determination"]
    return {
        "declared_policy": manifest["declared_optimizer_policy"],
        "resolved": manifest["resolved_optimizer"],
        "effective_lr0": record["effective_lr0"],
        "effective_momentum": record["effective_momentum"],
        "source": record["source"],
    }


def outcome(name: str, manifest: dict[str, Any], freeze: dict[str, Any]) -> list[tuple[str, Any]]:
    """Summarise the single recorded run and the checkpoint the frozen rule selected."""
    if name == "S1":
        execution = manifest["execution"]
        selection = manifest["checkpoint_selection"]
        epochs = f"{execution['epochs_completed']}/{execution['epochs_configured']}"
        return [
            ("epochs completed", epochs),
            ("termination", execution["termination"]),
            ("training seconds", execution["training_seconds"]),
            ("peak GPU memory reserved (GiB)", execution["peak_gpu_memory_reserved_gib"]),
            ("selected checkpoint epoch", selection["best_epoch"]),
            ("checkpoint selection policy", selection["policy"]),
            ("checkpoint selection semantics", selection["semantics"]),
            ("selected checkpoint fitness", selection["best_native_fitness"]),
            ("training runs", execution["runs"]),
        ]
    return [
        ("epochs completed", f"{manifest['epochs_completed']}/{manifest['epochs_configured']}"),
        ("termination", manifest["termination_mode"]),
        ("training seconds", manifest["training_duration_seconds"]),
        (
            "peak GPU memory reserved (GiB)",
            round(manifest["peak_gpu_memory"]["reserved_bytes"] / 1024**3, 3),
        ),
        ("selected checkpoint epoch", freeze["best_epoch"]),
        ("checkpoint selection policy", manifest["checkpoint_selection"].split(" - ")[0]),
        ("selected checkpoint fitness", manifest["best_validation_fitness"]),
        ("training runs", 1),
    ]


def training_curve(root: Path, name: str) -> Path:
    """Allow only a figure the experiment's own manifest declares as committed."""
    demo = support(root)
    entry = EXPERIMENTS[name]
    relative = entry["curve"]
    declared = demo["read"](root, entry["manifest"])["committed_figures"]
    if relative not in declared and Path(relative).name not in declared:
        raise ValueError("Only figures declared by the experiment manifest may be displayed")
    target = demo["inside"](root, relative)
    if not target.is_file():
        raise ValueError("Declared training figure is missing from this checkout")
    return target


def confusion_table(name: str, record: dict[str, Any]) -> str:
    """Render one committed confusion matrix from its counts, never from a rendered figure."""
    matrix = record["confusion_matrix"]
    labels = matrix["labels"]
    rows = matrix["matrix"]
    if len(rows) != len(labels) or any(len(row) != len(labels) for row in rows):
        raise ValueError("The recorded confusion matrix is not square over its own labels")
    return table(
        f"{name} - final-test confusion matrix at confidence {matrix['conf']} "
        f"and IoU {matrix['iou_threshold']}; rows predicted, columns ground truth",
        ("Predicted / true", *labels),
        [(label, *row) for label, row in zip(labels, rows, strict=True)],
    )


def evaluation_catalog(root: Path) -> dict[str, Any]:
    """Project the committed one-shot holdout results and their validation counterparts."""
    demo = support(root)
    demo["assert_delivery_context"]()
    detector = demo["read"](root, "reports/final_test_detector.json")
    segmenter = demo["read"](root, "reports/final_test_segmenter.json")
    direct = demo["read"](root, "reports/final_test_direct_iou.json")
    for record in (detector, segmenter, direct):
        if record["split"] != "test":
            raise ValueError("Final-test records must describe the holdout split")
    for record in (detector, segmenter):
        if record["winner_declared"] or record["composite_score"]:
            raise ValueError("No winner or composite score may be presented")
        if record["threshold_tuned_on_test"] or record["model_invoked_during_metric_computation"]:
            raise ValueError("Final-test records must remain a read of persisted predictions")
    return {
        "population": detector["population"],
        "support": detector["support_on_holdout"],
        "detector": detector,
        "segmenter": segmenter,
        "direct": direct["diagnostic"]["global"],
        "localisation": segmenter["localisation_comparison_with_detector"],
        "generalisation": generalisation(root, detector, segmenter, direct),
        "qualitative": {"D2": detector["qualitative"], "S1": segmenter["qualitative"]},
    }


def generalisation(
    root: Path, detector: dict[str, Any], segmenter: dict[str, Any], direct: dict[str, Any]
) -> list[dict[str, Any]]:
    """Pair each metric that existed before the holdout was read with its holdout counterpart."""
    demo = support(root)
    validation = {
        "box": demo["read"](root, "reports/detector_segmenter_box_comparison.json")["metrics"],
        "mask": demo["read"](root, "reports/segmentation_S1_canonical_evaluation.json")[
            "canonical"
        ],
        "iou": demo["read"](root, "reports/segmentation_S1_mask_iou.json")["global"],
    }
    test = {
        "d2_map50_95": detector["canonical_box"]["CANONICAL_TEST_BOX_MAP50_95"],
        "d2_map50": detector["canonical_box"]["CANONICAL_TEST_BOX_MAP50"],
        "s1_box_map50_95": segmenter["canonical_box"]["S1_CANONICAL_TEST_BOX_MAP50_95"],
        "s1_box_map50": segmenter["canonical_box"]["S1_CANONICAL_TEST_BOX_MAP50"],
        "s1_mask_map50_95": segmenter["canonical_mask"]["CANONICAL_TEST_MASK_MAP50_95"],
        "s1_mask_map50": segmenter["canonical_mask"]["CANONICAL_TEST_MASK_MAP50"],
    }
    test.update(direct["diagnostic"]["global"])
    rows = []
    for label, source, pointer, key in GENERALISATION:
        value = validation[source]
        for part in pointer:
            value = value[part]
        rows.append(
            {
                "metric": label,
                "validation": value,
                "test": test[key],
                "difference": round(abs(value - test[key]), 6),
            }
        )
    return rows


def dataset_html(root: Path) -> str:
    """Render the committed modelling population and class inventory."""
    demo = support(root)
    demo["assert_delivery_context"]()
    record = demo["read"](root, POPULATION_RECORD)
    population = record["population"]
    adapter = demo["read"](root, EXPERIMENTS["S1"]["manifest"])["adapter"]["counts"]
    holdout = demo["read"](root, "reports/final_test_detector.json")["population"]
    classes = table(
        "Canonical annotations by class, over the modelling population",
        ("Class", "Index", "Annotations", "Images"),
        [
            (
                name,
                index,
                population["annotations_by_class"][name],
                population["modeling_images_by_class"][name],
            )
            for name, index in sorted(record["class_map"].items(), key=lambda item: item[1])
        ],
    )
    splits = table(
        "Frozen group-aware and class-aware split, as recorded",
        ("Split", "Images", "Annotations"),
        [
            ("train", adapter["train_images"], adapter["train_instances"]),
            ("validation", adapter["validation_images"], adapter["validation_instances"]),
            ("test (holdout)", holdout["images"], holdout["annotations"]),
        ],
    )
    return panel(
        '<p class="tag">DATASET AND CLASSES / COMMITTED RECORD</p>'
        "<h2>Modelling population</h2>"
        f"<p>{population['modeling_image_population']} modelling images and "
        f"{population['canonical_annotations']} canonical annotations over "
        f"{population['groups_total']} indivisible split units "
        f"({population['groups_singleton']} singletons and "
        f"{population['groups_semantic_duplicate']} confirmed duplicate groups). "
        f"{population['excluded_images']} images were excluded as out-of-domain. "
        "Instance segmentation is the canonical annotation source and detection boxes are "
        "derived from the polygons, so both tasks use exactly the same image ids.</p>"
        f"{classes}{splits}"
        f"<p><code>vest_loose</code> is rare and its per-class figures carry an explicit "
        "small-sample caveat throughout this notebook. No class is collapsed or excluded "
        "from reporting.</p>"
        + demo["source_link"](POPULATION_RECORD, "Committed population record")
    )


def show_dataset(root: Path) -> None:
    """Display the dataset and class inventory inside IPython/Colab."""
    from IPython.display import HTML, display

    display(HTML(dataset_html(root)))


def training_html(root: Path, name: str) -> str:
    """Render one frozen model's recorded training recipe, with no execution."""
    entry = training_catalog(root)[name]
    demo = support(root)
    configs = " / ".join(demo["source_link"](path, Path(path).name) for path in entry["configs"])
    recipe = table(
        f"{name} - frozen training protocol, as recorded",
        ("Argument", "Value"),
        entry["optimisation"] + entry["segmentation"],
    )
    augmentation = table(
        f"{name} - augmentation, inherited from the frozen protocol",
        ("Argument", "Value"),
        entry["augmentation"],
    )
    optimizer = entry["optimizer"]
    run = table(f"{name} - recorded execution", ("Field", "Value"), entry["outcome"])
    checkpoint = entry["checkpoint"]
    return panel(
        f'<p class="tag">{TRAINING_EVIDENCE} &middot; {NO_NEW_TRAINING}</p>'
        f"<h2>{html.escape(name)} &middot; {html.escape(entry['architecture'])} "
        f"({html.escape(entry['task'])})</h2>"
        "<p>This cell executes no training. It reads the committed manifest of the single "
        "run that produced the frozen model and prints the recipe and the recorded outcome. "
        "Reproducing the run needs a CUDA GPU and the materialised dataset; the notebook "
        "does not attempt it.</p>"
        f"<p>Reproducible entry point: <code>{html.escape(entry['command'])}</code><br>"
        f"Frozen configuration: {configs}</p>"
        f"<p>Pretrained weights <code>{html.escape(entry['pretrained']['identifier'])}</code>, "
        f"SHA-256 <code>{html.escape(entry['pretrained']['sha256'])}</code>, "
        f"{entry['pretrained']['size_bytes']} bytes. "
        f"{entry['complexity']['parameters']} parameters.</p>"
        f"{recipe}"
        f"<p><code>optimizer: {html.escape(str(optimizer['declared_policy']))}</code> is a policy, "
        f"not a value. It resolved to <b>{html.escape(optimizer['resolved'])}</b> at lr0 "
        f"{cell(optimizer['effective_lr0'])} and momentum "
        f"{cell(optimizer['effective_momentum'])}, captured as "
        f"<code>{html.escape(optimizer['source'])}</code>. The file's declared "
        "<code>lr0</code> is therefore not what ran.</p>"
        f"{augmentation}{run}"
        f"<p>Frozen model identity: SHA-256 <code>{html.escape(checkpoint['sha256'])}</code>, "
        f"{checkpoint['bytes']} bytes, at input size {checkpoint['imgsz']}. "
        "The checkpoint binary is not committed; it is identified by digest.</p>"
        f"{demo['source_link'](entry['manifest'], 'Experiment manifest')} &middot; "
        f"{demo['source_link'](entry['freeze'], 'Freeze manifest')}"
    )


def show_training(root: Path, name: str = "D2") -> None:
    """Display one model's recorded training evidence and its committed curves."""
    from IPython.display import HTML, Image, display

    if name not in EXPERIMENTS:
        raise ValueError("Unknown experiment; choose D2 or S1")
    display(HTML(training_html(root, name)))
    display(
        HTML(
            f"<h3>{html.escape(name)} - recorded training and validation curves</h3>"
            "<p>Committed figure from the original run. Losses and validation metrics per "
            "epoch; the horizontal axis is the epoch index. Nothing is recomputed here.</p>"
        )
    )
    display(Image(filename=str(training_curve(root, name)), width=1400))


def evaluation_html(root: Path) -> str:
    """Render the committed one-shot holdout results and the bounded validation comparison."""
    data = evaluation_catalog(root)
    demo = support(root)
    detector, segmenter = data["detector"], data["segmenter"]
    d2_pr = detector["canonical_precision_recall"]
    s1_mask_pr = segmenter["canonical_mask_precision_recall"]
    s1_box_pr = segmenter["canonical_box_precision_recall"]
    headline = table(
        "Canonical COCO average precision on the holdout, read once",
        ("Model / output", "mAP@0.50", "mAP@0.50:0.95", "precision", "recall"),
        [
            (
                "D2 box",
                detector["canonical_box"]["CANONICAL_TEST_BOX_MAP50"],
                detector["canonical_box"]["CANONICAL_TEST_BOX_MAP50_95"],
                d2_pr["precision"],
                d2_pr["recall"],
            ),
            (
                "S1 box",
                segmenter["canonical_box"]["S1_CANONICAL_TEST_BOX_MAP50"],
                segmenter["canonical_box"]["S1_CANONICAL_TEST_BOX_MAP50_95"],
                s1_box_pr["precision"],
                s1_box_pr["recall"],
            ),
            (
                "S1 mask",
                segmenter["canonical_mask"]["CANONICAL_TEST_MASK_MAP50"],
                segmenter["canonical_mask"]["CANONICAL_TEST_MASK_MAP50_95"],
                s1_mask_pr["precision"],
                s1_mask_pr["recall"],
            ),
        ],
    )
    per_class = table(
        "Per-class AP@0.50:0.95 on the holdout",
        ("Class", "D2 box", "S1 box", "S1 mask", "Support"),
        [
            (
                name,
                detector["canonical_box"]["per_class"][name]["AP@0.50:0.95"],
                segmenter["canonical_box"]["per_class"][name]["AP@0.50:0.95"],
                segmenter["canonical_mask"]["per_class"][name]["AP@0.50:0.95"],
                entry["status"],
            )
            for name, entry in sorted(data["support"].items())
        ],
    )
    direct = table(
        "Direct instance-mask IoU diagnostic (S1), frozen phase 8C protocol at confidence 0.25",
        ("Quantity", "Value"),
        [
            ("matched_mask_iou_mean", data["direct"]["matched_mask_iou_mean"]),
            ("gt_normalized_mask_iou", data["direct"]["gt_normalized_mask_iou"]),
            ("gt_match_coverage", data["direct"]["gt_match_coverage"]),
            ("gt_iou50_coverage", data["direct"]["gt_iou50_coverage"]),
            ("gt_iou75_coverage", data["direct"]["gt_iou75_coverage"]),
            (
                "matched / ground-truth instances",
                f"{data['direct']['matched_count']} / {data['direct']['gt_count']}",
            ),
        ],
    )
    objects = table(
        "Object-level counts at the frozen operating point (class-aware, IoU 0.50, conf 0.25)",
        ("Model", "TP", "FP", "FN"),
        [
            (
                "D2",
                detector["object_level"]["ground_truth"]
                - detector["object_level"]["false_negatives"],
                detector["object_level"]["false_positives"],
                detector["object_level"]["false_negatives"],
            ),
            (
                "S1",
                segmenter["object_level"]["ground_truth"]
                - segmenter["object_level"]["false_negatives"],
                segmenter["object_level"]["false_positives"],
                segmenter["object_level"]["false_negatives"],
            ),
        ],
    )
    comparison = table(
        "Validation versus test, DESCRIPTIVE_GENERALIZATION_COMPARISON",
        ("Metric", "Validation", "Test", "Absolute difference"),
        [
            (row["metric"], row["validation"], row["test"], row["difference"])
            for row in data["generalisation"]
        ],
    )
    localisation = data["localisation"]
    population = data["population"]
    declined = sum(
        1
        for name in data["support"]
        if segmenter["canonical_box"]["per_class"][name]["AP@0.50:0.95"]
        < detector["canonical_box"]["per_class"][name]["AP@0.50:0.95"]
    )
    rare = next((name, entry) for name, entry in data["support"].items() if not entry["supported"])
    return panel(
        f'<p class="tag">{html.escape(FINAL_TEST_BANNER)}</p>'
        "<h2>Final holdout evaluation</h2>"
        f"<p>{population['images']} images and {population['annotations']} annotations, "
        "all evaluated, none sampled or excluded. The holdout was locked from the split "
        "freeze until phase 11B, read exactly once under a protocol frozen beforehand, and "
        "is now spent. <b>This cell recomputes nothing and executes no model</b>: the values "
        "below are read by field from the committed result artifacts, which were themselves "
        "derived from predictions persisted and fingerprinted before any metric existed.</p>"
        f"{headline}"
        "<p>Average precision is integrated at confidence 0.001, which is deliberately not an "
        "operating point; precision and recall are reported at the frozen operational 0.25 and "
        "IoU 0.50. The two confidences are never mixed. Box and mask measure different tasks "
        "and are never merged into a composite.</p>"
        f"{per_class}"
        f"<p>D2 versus S1 on boxes: mAP@0.50:0.95 {cell(localisation['detector_map50_95'])} "
        f"against {cell(localisation['segmenter_map50_95'])}, difference "
        f"{cell(localisation['delta_map50_95'])}. This is "
        f"<code>{html.escape(localisation['label'])}</code>: {declined} of the "
        f"{len(data['support'])} classes declined, no winner is declared, no significance test "
        "was run and no model selection follows. "
        f"<code>{html.escape(rare[0])}</code> has {rare[1]['images']} holdout images and "
        f"{rare[1]['instances']} instances, so its figures carry high sampling uncertainty "
        "and decide nothing.</p>"
        f"{direct}"
        "<p>The two direct-IoU headlines are not interchangeable and neither is a COCO AP. "
        "<code>matched_mask_iou_mean</code> describes mask quality where the model produced an "
        "overlapping same-class instance; <code>gt_normalized_mask_iou</code> divides the same "
        "sum by every canonical instance, so the misses lower it.</p>"
        f"{objects}{comparison}"
        "<p>Only metrics that already existed before the holdout was read are paired here. "
        "<b>Why a gap exists in either direction is UNKNOWN.</b> Each model was trained once "
        "and evaluated once per split, so run-to-run variance is unknown, and no experiment "
        "may now be run to explain a reported holdout number.</p>"
        + demo["source_link"]("reports/final_test_evaluation.md", "Full final-test report")
    )


def confusion_html(root: Path) -> str:
    """Render both final-test confusion matrices from their committed counts."""
    data = evaluation_catalog(root)
    tables = "".join(
        confusion_table(name, data[key]) for name, key in (("D2", "detector"), ("S1", "segmenter"))
    )
    return panel(
        f'<p class="tag">{html.escape(FINAL_TEST_BANNER)}</p>'
        "<h2>Final-test confusion matrices</h2>"
        "<p>Rows are predicted, columns ground truth, produced by the framework's own matrix "
        "with the semantics frozen in phase 11A. An off-diagonal matched pair counts as both a "
        "false positive and a false negative, which is the framework's behaviour and is not "
        "modified. The last row and column are the unmatched background bucket.</p>"
        f"{tables}"
        "<p>These are aggregate counts read from the committed result records. No holdout "
        "image, prediction file or image identifier is published here or anywhere in this "
        "repository.</p>"
    )


def show_evaluation(root: Path) -> None:
    """Display the committed final-test results and both confusion matrices."""
    from IPython.display import HTML, display

    display(HTML(evaluation_html(root)))
    display(HTML(confusion_html(root)))


def qualitative_html(root: Path) -> str:
    """Render where the qualitative error evidence lives and how it was selected."""
    data = evaluation_catalog(root)
    demo = support(root)
    rows = []
    for model, record in data["qualitative"].items():
        for category, entry in sorted(record["categories"].items()):
            rows.append(
                (
                    model,
                    category,
                    entry["applicable"],
                    entry.get("available", 0),
                    len(entry["selected"]),
                )
            )
    selection = table(
        "Deterministically ranked holdout examples, selected by a rule frozen before any access",
        ("Model", "Category", "Applicable", "Available", "Selected"),
        rows,
    )
    return panel(
        '<p class="tag">QUALITATIVE ERROR EVIDENCE</p>'
        "<h2>Where the FP/FN evidence lives</h2>"
        "<p>The holdout examples were chosen by a deterministic ranking frozen before the split "
        f"was read (<code>images_browsed_before_selection: "
        f"{cell(data['qualitative']['D2']['images_browsed_before_selection'])}</code>). The "
        "rendered figures and the selection manifest stay outside version control, because the "
        "protocol forbids committing holdout imagery or identifiers, and that prohibition wins "
        "over publishing them.</p>"
        f"{selection}"
        "<p>The committed, publishable qualitative gallery is the <b>validation</b> one shown "
        "in the FP/FN section of this notebook. It is separate evidence on a separate split, "
        "and it is never presented as holdout behaviour.</p>"
        + demo["source_link"]("reports/qualitative_validation_gallery.md", "Validation gallery")
    )


def show_qualitative(root: Path) -> None:
    """Display the qualitative selection accounting inside IPython/Colab."""
    from IPython.display import HTML, display

    display(HTML(qualitative_html(root)))
