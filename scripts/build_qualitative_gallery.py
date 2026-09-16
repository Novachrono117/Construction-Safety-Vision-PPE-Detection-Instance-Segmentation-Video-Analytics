"""Build Phase 12C from validation-only caches; --infer permits one frozen pass per model.

No evaluation, training, threshold search or holdout route exists here. The
default command only reuses persisted visualization predictions. A started
inference ledger is never overwritten or silently retried.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from pycocotools import mask as mask_utils

from construction_safety_vision.data.coco_materialization import decode_mask, document_fingerprint
from construction_safety_vision.data.materialization import digest
from construction_safety_vision.data.split_freeze import load_frozen_splits
from construction_safety_vision.delivery_status import STATUS_PATH, build_status
from construction_safety_vision.detection_freeze import load_checkpoint_path as detector_path
from construction_safety_vision.detection_freeze import load_final_detector
from construction_safety_vision.provenance import ProvenanceRecord, sha256_file
from construction_safety_vision.qualitative_gallery import (
    BASELINE,
    CHECKPOINTS,
    CLASSES,
    COMPLETE,
    CONFIG,
    FIGURES,
    LICENSE_URL,
    PROVENANCE,
    REPORT,
    SOURCE_URL,
    choose_examples,
    choose_hero,
    choose_masks,
    fingerprint,
    load_policy,
    mask_category,
    match_objects,
    validate_rows,
)
from construction_safety_vision.segmentation_freeze import load_checkpoint_path as segmenter_path
from construction_safety_vision.segmentation_freeze import load_final_segmenter

CACHE = "artifacts/qualitative_validation"
ROOT = Path(__file__).resolve().parents[1]
GT_COLOR = "#00e5d1"
PRED_COLOR = "#ffae42"


def read_json(path: Path) -> Any:
    """Read a JSON document explicitly named by this delivery path."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    """Write deterministic UTF-8 JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )


def encode(mask: np.ndarray) -> dict[str, Any]:
    """Persist lossless COCO RLE on the original canvas."""
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    return {"size": list(rle["size"]), "counts": rle["counts"].decode("ascii")}


def decode(rle: dict[str, Any]) -> np.ndarray:
    """Decode a persisted visualization mask without rasterization approximations."""
    return mask_utils.decode({"size": rle["size"], "counts": rle["counts"].encode("ascii")}).astype(
        bool
    )


def population(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Verify and load only the canonical validation documents and image bytes."""
    if "CSVISION_ALLOW_TEST_SPLIT" in os.environ:
        raise ValueError("Holdout gate must remain unset")
    frozen = load_frozen_splits(root / "reports/split_manifest.json")
    allowed = set(frozen.image_ids("validation", purpose="Phase 12C visual evidence"))
    # The config is the frozen declaration; its manifest also records the fingerprint.
    import yaml

    comparison = yaml.safe_load(
        (root / "configs/detector_segmenter_comparison.yaml").read_text(encoding="utf-8")
    )
    s1 = read_json(root / "reports/final_segmenter_manifest.json")
    expected = {
        "detection": comparison["population"]["source_sha256"],
        "segmentation": s1["dataset_fingerprints"]["validation_annotations_sha256"],
    }
    documents = {}
    manifest = read_json(root / "reports/task_dataset_manifest.json")
    for task in expected:
        path = root / f"data/processed/canonical/annotations/{task}_validation.coco.json"
        if sha256_file(path) != expected[task]:
            raise ValueError(f"SCIENTIFIC_STATE_MISMATCH: {task} validation bytes")
        doc = read_json(path)
        if document_fingerprint(doc) != manifest["validation"][task + "_document_sha256"]:
            raise ValueError("Canonical document fingerprint mismatch")
        if doc["info"]["split"] != "validation":
            raise ValueError("Nonvalidation document")
        documents[task] = doc
    det, seg = documents["detection"], documents["segmentation"]
    if det["images"] != seg["images"] or det["categories"] != seg["categories"]:
        raise ValueError("Canonical task views do not align")
    images = {entry["source_image_id"]: dict(entry) for entry in det["images"]}
    if set(images) != allowed or len(images) != comparison["population"]["images"]:
        raise ValueError("Frozen validation membership mismatch")
    if [c["name"] for c in det["categories"]] != list(CLASSES):
        raise ValueError("Canonical class order changed")
    for stem, image in images.items():
        if image["split"] != "validation" or Path(image["file_name"]).stem != stem:
            raise ValueError("Invalid validation image metadata")
        if Path(image["file_name"]).name != image["file_name"]:
            raise ValueError("Image filename must be a basename")
        relative = f"data/processed/canonical/images/validation/{image['file_name']}"
        if (
            not (root / relative)
            .resolve()
            .is_relative_to((root / "data/processed/canonical/images/validation").resolve())
        ):
            raise ValueError("Validation path escape")
        image.update(path=relative, sha256=sha256_file(root / relative))
    if (
        digest(sorted([stem, img["sha256"]] for stem, img in images.items()))
        != manifest["validation"]["image_content_sha256"]
    ):
        raise ValueError("Canonical validation image bytes changed")
    boxes = {a["id"]: a for a in det["annotations"]}
    if set(boxes) != {a["id"] for a in seg["annotations"]}:
        raise ValueError("Task annotation membership differs")
    truth = []
    for a in sorted(seg["annotations"], key=lambda a: a["id"]):
        image = images[a["source_image_id"]]
        d = boxes[a["id"]]
        if a["image_id"] != image["id"] or a["category_id"] != d["category_id"]:
            raise ValueError("Canonical annotation alignment mismatch")
        x, y, w, h = d["bbox"]
        truth.append(
            {
                "image_id": a["source_image_id"],
                "split": "validation",
                "annotation_id": a["id"],
                "class": CLASSES[a["category_id"]],
                "box": [x, y, x + w, y + h],
                "area": a["area"],
                "mask_rle": encode(
                    decode_mask(a["segmentation"], height=image["height"], width=image["width"])
                ),
            }
        )
    validate_rows(truth, allowed)
    if len(truth) != comparison["population"]["annotations"]:
        raise ValueError("Incomplete validation annotation population")
    return images, truth


def infer(root: Path, images: dict[str, Any], policy: dict[str, Any]) -> None:
    """Execute exactly one predict call per frozen model and persist original-canvas masks."""
    import torch
    import ultralytics
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError("BLOCKED_FOR_GPU; no CPU fallback")
    cache = root / CACHE
    cache.mkdir(parents=True, exist_ok=True)
    ledger_path = cache / "execution.json"
    if ledger_path.exists() or any((cache / f"{m}.json").exists() for m in CHECKPOINTS):
        raise RuntimeError("Existing visualization execution; reuse caches, never silently rerun")
    paths = {
        "D2": detector_path(load_final_detector(root / "reports"), root),
        "S1": segmenter_path(load_final_segmenter(root / "reports"), root),
    }
    for model, path in paths.items():
        if sha256_file(path) != CHECKPOINTS[model]:
            raise ValueError("Frozen checkpoint mismatch")
    ledger = {
        "purpose": policy["purpose"],
        "split": "validation",
        "policy_sha256": fingerprint(policy),
        "ultralytics": ultralytics.__version__,
        "torch": torch.__version__,
        "models": {},
        "test_content_accessed": False,
        "training_executed": False,
        "model_selection_changed": False,
        "threshold_changed": False,
        "implementation_sha256": sha256_file(Path(__file__)),
    }
    prior = root / "artifacts/qualitative_validation_batching_deviation/execution.json"
    if prior.exists():
        ledger["prior_execution"] = read_json(prior)
        ledger["execution_correction"] = {
            "classification": "DELIVERY_INPUT_BATCHING_DEVIATION_NOT_USED",
            "reason": (
                "Ultralytics converts a source list to LoadPilAndNumpy and batches the entire "
                "list, ignoring batch=1. The initial calls returned all 65 images with two "
                "forward hooks each (warmup plus image batch). Replaced the source route "
                "with its verified validation-directory loader to enforce batch size."
            ),
            "initial_batch_determination": "PINNED_LOADER_SOURCE_AND_RECORDED_FORWARD_COUNT",
            "initial_predictions_used_in_gallery": False,
            "selection_policy_changed": False,
            "threshold_changed": False,
            "result_driven_rerun": False,
            "source_evidence": "ultralytics/data/build.py:check_source; loaders.py:LoadPilAndNumpy",
        }
    source_directory = root / "data/processed/canonical/images/validation"
    expected_names = {image["file_name"] for image in images.values()}
    if {p.name for p in source_directory.iterdir()} != expected_names:
        raise ValueError("Validation source directory contains an unapproved entry")
    # Create exclusively before invoking a model, so an interrupted run cannot disappear.
    with ledger_path.open("x", encoding="utf-8") as handle:
        json.dump(ledger, handle, indent=2)
    for name, path in paths.items():
        state = {
            "checkpoint_sha256": CHECKPOINTS[name],
            "predict_invocations": 0,
            "images_completed": 0,
            "forward_calls_including_framework_warmup": 0,
            "status": "STARTED",
            "effective_input_dtypes": [],
            "effective_input_batch_sizes": [],
            "autocast_enabled": False,
        }
        ledger["models"][name] = state
        write_json(ledger_path, ledger)
        model = YOLO(str(path))
        handles = []

        def attach(
            predictor: Any, state: dict[str, Any] = state, handles: list[Any] = handles
        ) -> None:
            backend = predictor.model
            if backend.fp16:
                raise RuntimeError("FP32 required")
            state["parameter_dtypes"] = sorted({str(p.dtype) for p in backend.model.parameters()})
            if state["parameter_dtypes"] != ["torch.float32"]:
                raise RuntimeError("Model parameters are not FP32")

            def hook(_module: Any, args: Any) -> None:
                dtype = str(args[0].dtype)
                batch_size = int(args[0].shape[0])
                state["effective_input_batch_sizes"] = sorted(
                    set(state["effective_input_batch_sizes"]) | {batch_size}
                )
                if batch_size != 1:
                    raise RuntimeError("Effective batch differs from the declared batch=1")
                state["forward_calls_including_framework_warmup"] += 1
                state["effective_input_dtypes"] = sorted(
                    set(state["effective_input_dtypes"]) | {dtype}
                )
                active = torch.is_autocast_enabled("cuda")
                state["autocast_enabled"] |= active
                if dtype != "torch.float32" or active:
                    raise RuntimeError("Effective FP32 contract violated")

            handles.append(backend.model.register_forward_pre_hook(hook))

        model.add_callback("on_predict_start", attach)
        rows = []
        seen = []
        try:
            state["predict_invocations"] += 1
            write_json(ledger_path, ledger)
            outputs = model.predict(
                source=str(source_directory),
                stream=True,
                batch=1,
                imgsz=768,
                conf=0.25,
                iou=0.70,
                max_det=300,
                augment=False,
                quantize=32,
                device=0,
                retina_masks=name == "S1",
                verbose=False,
                save=False,
                save_txt=False,
            )
            for result in outputs:
                stem = Path(result.path).stem
                if stem not in images or stem in seen:
                    raise RuntimeError("Unexpected or duplicate inference image")
                seen.append(stem)
                if tuple(result.orig_shape) != (images[stem]["height"], images[stem]["width"]):
                    raise RuntimeError("Prediction canvas changed")
                if [result.names[i] for i in range(len(CLASSES))] != list(CLASSES):
                    raise RuntimeError("Checkpoint class labels changed")
                masks = (
                    result.masks.data.cpu().numpy().astype(bool)
                    if result.masks is not None
                    else None
                )
                for index, box in enumerate(result.boxes):
                    row = {
                        "image_id": stem,
                        "split": "validation",
                        "prediction_index": index,
                        "class": CLASSES[int(box.cls.item())],
                        "score": float(box.conf.item()),
                        "box": box.xyxy[0].cpu().tolist(),
                    }
                    if name == "S1":
                        if masks is None or masks[index].shape != result.orig_shape:
                            raise RuntimeError("Missing original-canvas S1 mask")
                        row["mask_rle"] = encode(masks[index])
                    rows.append(row)
                state["images_completed"] += 1
                print(
                    f"{name}: {state['images_completed']}/{len(images)} validation images",
                    flush=True,
                )
            if seen != sorted(images):
                raise RuntimeError("Prediction order/population mismatch")
            validate_rows(rows, set(images))
            payload = {
                "checkpoint_sha256": CHECKPOINTS[name],
                "policy_sha256": fingerprint(policy),
                "images": seen,
                "source_sha256": {k: v["sha256"] for k, v in images.items()},
                "predictions": rows,
            }
            write_json(cache / f"{name}.json", payload)
            state.update(status="COMPLETE", cache_sha256=sha256_file(cache / f"{name}.json"))
        except Exception:
            state["status"] = "FAILED_NO_AUTOMATIC_RETRY"
            raise
        finally:
            for handle in handles:
                handle.remove()
            write_json(ledger_path, ledger)
        del model
        torch.cuda.empty_cache()


def load_predictions(
    root: Path, images: dict[str, Any], policy: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read only verified delivery caches; never load a checkpoint to render."""
    ledger = read_json(root / CACHE / "execution.json")
    if ledger["policy_sha256"] != fingerprint(policy):
        raise ValueError("Policy differs from the one recorded before inference")
    predictions = {}
    for model in CHECKPOINTS:
        path = root / CACHE / f"{model}.json"
        cache = read_json(path)
        state = ledger["models"][model]
        if state["status"] != "COMPLETE" or sha256_file(path) != state["cache_sha256"]:
            raise ValueError("Incomplete or modified inference cache")
        if cache["checkpoint_sha256"] != CHECKPOINTS[model] or cache[
            "policy_sha256"
        ] != fingerprint(policy):
            raise ValueError("Cache model or policy mismatch")
        if cache["images"] != sorted(images) or cache["source_sha256"] != {
            k: v["sha256"] for k, v in images.items()
        }:
            raise ValueError("Cache validation population mismatch")
        validate_rows(cache["predictions"], set(images))
        predictions[model] = cache["predictions"]
    return predictions, ledger


def analyze(
    images: dict[str, Any], truth: list[dict[str, Any]], predictions: dict[str, Any]
) -> dict[str, Any]:
    """Classify the complete validation population and record per-instance mask geometry."""
    records = {
        model: match_objects(truth, rows, set(images)) for model, rows in predictions.items()
    }
    gt = {(r["image_id"], r["annotation_id"]): r for r in truth}
    pred = {(r["image_id"], r["prediction_index"]): r for r in predictions["S1"]}
    for r in records["S1"]:
        if r["category"] != "TRUE_POSITIVE":
            continue
        g = decode(gt[r["image_id"], r["annotation_id"]]["mask_rle"])
        p = decode(pred[r["image_id"], r["prediction_index"]]["mask_rle"])
        intersection, union = int((g & p).sum()), int((g | p).sum())
        ga, pa = int(g.sum()), int(p.sum())
        iou = intersection / union if union else 0
        relative = (pa - ga) / ga
        x1, y1, x2, y2 = r["box"]
        r.update(
            mask_iou=iou,
            relative_area_error=relative,
            mask_category=mask_category(iou, relative),
            gt_mask_pixels=ga,
            predicted_mask_pixels=pa,
            intersection_pixels=intersection,
            union_pixels=union,
            mask_box_fill=pa / ((x2 - x1) * (y2 - y1)),
        )
    return records


def panel(ax: Any, image: Any, title: str) -> None:
    """Show the complete source canvas with a readable external title."""
    ax.imshow(image)
    ax.set_title(title, fontsize=12, loc="left", pad=8, color="#18283b")
    ax.axis("off")


def draw_box(ax: Any, row: dict[str, Any], color: str, *, selected: bool, gt: bool) -> None:
    """Draw one box with explicit GT/prediction color and line style."""
    from matplotlib.patches import Rectangle

    x1, y1, x2, y2 = row["box"]
    ax.add_patch(
        Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            edgecolor=color,
            linewidth=2.5 if selected else 1.0,
            linestyle="--" if gt else "-",
        )
    )


def tint(ax: Any, mask: np.ndarray, color: str, alpha: float = 0.32) -> None:
    """Add a translucent mask without changing the source image."""
    from matplotlib.colors import to_rgba

    rgba = np.zeros((*mask.shape, 4))
    rgba[mask] = to_rgba(color, alpha)
    ax.imshow(rgba)


def footer(fig: Any) -> None:
    """Attach attribution and modification notice to every reusable figure."""
    fig.text(
        0.025,
        0.018,
        "Source: agis-workspace-8gs52 / Construction PPE Compliance Detection (Roboflow "
        "Universe) | CC BY 4.0\n"
        "Images/annotations: CC BY 4.0, not AGPL. Changes: model/GT overlays and layout. Full "
        "attribution and source link in gallery report.",
        fontsize=8,
        color="#38465a",
    )


def render(
    root: Path, payload: dict[str, Any], truth: list[dict[str, Any]], predictions: dict[str, Any]
) -> list[str]:
    """Render deterministic selections only; never browse or replace a candidate."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "white"})
    directory = root / FIGURES
    directory.mkdir(parents=True, exist_ok=True)
    assets = []
    images = payload["images"]

    def source(stem: str) -> Any:
        if stem not in images:
            raise ValueError("Render source is not frozen validation")
        with Image.open(root / images[stem]["path"]) as handle:
            return np.asarray(handle.convert("RGB"))

    def finish(fig: Any, name: str) -> None:
        footer(fig)
        relative = f"{FIGURES}/{name}.png"
        fig.savefig(
            root / relative,
            dpi=170,
            metadata={
                "Description": SOURCE_URL + " | " + LICENSE_URL + " | overlays/layout modified"
            },
        )
        plt.close(fig)
        assets.append(relative)

    for model in CHECKPOINTS:
        fig, axes = plt.subplots(5, 2, figsize=(14, 22))
        fig.subplots_adjust(
            top=0.92, bottom=0.075, left=0.025, right=0.975, hspace=0.40, wspace=0.12
        )
        fig.suptitle(
            f"{model} {'detector' if model == 'D2' else 'segmenter'} | validation FP / FN evidence",
            fontsize=20,
            x=0.025,
            ha="left",
            y=0.986,
        )
        fig.text(
            0.025,
            0.96,
            "GT: cyan dashed | Prediction: amber solid | Selected object: thick outline | Full "
            "scenes, no crops",
            fontsize=10,
        )
        for ax, selection in zip(axes.flat, payload["selections"][model], strict=True):
            r = selection["example"]
            short = "FP" if selection["category"] == "FALSE_POSITIVE" else "FN"
            title = f"{selection['class']} | {short}"
            if r is None:
                ax.axis("off")
                ax.set_title(title, loc="left", fontsize=12)
                ax.text(
                    0.5,
                    0.5,
                    "NO_VALID_EXAMPLE\nNo such error at the frozen operating point",
                    ha="center",
                    va="center",
                    fontsize=12,
                )
                continue
            stem = r["image_id"]
            subtitle = (
                f"prediction #{r['prediction_index']} | confidence {r['score']:.3f}"
                if short == "FP"
                else f"GT #{r['annotation_id']} | unmatched at box IoU 0.50"
            )
            panel(ax, source(stem), title + "\n" + subtitle)
            relevant_gt = [
                g for g in truth if g["image_id"] == stem and g["class"] == selection["class"]
            ]
            relevant_pred = [
                p
                for p in predictions[model]
                if p["image_id"] == stem and p["class"] == selection["class"]
            ]
            for g in relevant_gt:
                chosen = short == "FN" and g["annotation_id"] == r["annotation_id"]
                draw_box(ax, g, GT_COLOR, selected=chosen, gt=True)
            for p in relevant_pred:
                chosen = short == "FP" and p["prediction_index"] == r["prediction_index"]
                if model == "S1" and chosen:
                    tint(ax, decode(p["mask_rle"]), PRED_COLOR)
                draw_box(ax, p, PRED_COLOR, selected=chosen, gt=False)
            ax.text(
                0,
                -0.05,
                f"ID: {stem} | GT of class: {len(relevant_gt)} | predictions: {len(relevant_pred)}",
                transform=ax.transAxes,
                fontsize=8,
            )
        finish(fig, f"validation_fp_fn_{model.lower()}")
    fig, axes = plt.subplots(3, 3, figsize=(15, 13))
    fig.subplots_adjust(top=0.90, bottom=0.085, left=0.025, right=0.975, hspace=0.36, wspace=0.10)
    fig.suptitle("S1 | what the instance mask represents", fontsize=21, x=0.025, ha="left", y=0.975)
    fig.text(
        0.025,
        0.936,
        "Display categories reuse Phase 8D bands/area flags; these selected cases do not "
        "establish a global localization ranking.",
        fontsize=10,
    )
    for axes_row, selection in zip(axes, payload["mask_selections"], strict=True):
        r = selection["example"]
        if r is None:
            for ax in axes_row:
                ax.axis("off")
                ax.text(0.5, 0.5, selection["category"] + "\nNO_VALID_EXAMPLE", ha="center")
            continue
        stem = r["image_id"]
        g = next(
            g for g in truth if g["image_id"] == stem and g["annotation_id"] == r["annotation_id"]
        )
        p = next(
            p
            for p in predictions["S1"]
            if p["image_id"] == stem and p["prediction_index"] == r["prediction_index"]
        )
        gm, pm = decode(g["mask_rle"]), decode(p["mask_rle"])
        for ax, title in zip(
            axes_row,
            ("Canonical GT mask", "S1 predicted mask + box", "Mask disagreement"),
            strict=True,
        ):
            panel(ax, source(stem), f"{selection['category']}\n{title}")
        tint(axes_row[0], gm, GT_COLOR)
        draw_box(axes_row[0], g, GT_COLOR, selected=True, gt=True)
        tint(axes_row[1], pm, PRED_COLOR)
        draw_box(axes_row[1], p, PRED_COLOR, selected=True, gt=False)
        tint(axes_row[2], gm & ~pm, GT_COLOR, 0.60)
        tint(axes_row[2], pm & ~gm, PRED_COLOR, 0.60)
        for ax in axes_row:
            ax.text(
                0,
                -0.045,
                f"{r['class']} | GT #{r['annotation_id']} | p={r['score']:.3f} | mask "
                f"IoU={r['mask_iou']:.3f}",
                transform=ax.transAxes,
                fontsize=8,
            )
        axes_row[2].text(
            0,
            -0.10,
            "cyan: GT missed | amber: excess prediction",
            transform=axes_row[2].transAxes,
            fontsize=8,
        )
    finish(fig, "mask_quality_gallery")
    hero = payload["hero"]["example"]
    if hero is not None:
        stem = hero["image_id"]
        fig, axes = plt.subplots(1, 3, figsize=(18, 7))
        fig.subplots_adjust(top=0.85, bottom=0.17, left=0.02, right=0.98, wspace=0.06)
        fig.suptitle(
            "Boxes locate objects. Masks describe their foreground.",
            fontsize=23,
            x=0.025,
            ha="left",
            y=0.98,
        )
        fig.text(
            0.025,
            0.92,
            "VALIDATION | HERO_CANDIDATE_NOT_FINAL | deterministic selection; no claim of "
            "universal superiority",
            fontsize=11,
        )
        for ax, title in zip(
            axes,
            ("Canonical ground truth", "D2 | detector boxes", "S1 | boxes + instance masks"),
            strict=True,
        ):
            panel(ax, source(stem), title)
        for g in (g for g in truth if g["image_id"] == stem):
            draw_box(
                axes[0], g, GT_COLOR, selected=g["annotation_id"] == hero["annotation_id"], gt=True
            )
            if g["annotation_id"] == hero["annotation_id"]:
                tint(axes[0], decode(g["mask_rle"]), GT_COLOR)
        for axis, model in ((axes[1], "D2"), (axes[2], "S1")):
            for p in (p for p in predictions[model] if p["image_id"] == stem):
                draw_box(axis, p, PRED_COLOR, selected=False, gt=False)
                if model == "S1":
                    tint(axis, decode(p["mask_rle"]), PRED_COLOR, 0.25)
        fig.text(
            0.025,
            0.12,
            f"Selected anchor: {hero['class']} / GT #{hero['annotation_id']} | image {stem} | "
            f"all operating-point predictions shown",
            fontsize=10,
        )
        finish(fig, "box_vs_mask_hero_candidate")
    return assets


def main() -> int:
    """Run the explicit inference or cached-render delivery route."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--infer", action="store_true", help="run the authorized visualization passes once"
    )
    args = parser.parse_args()
    policy = load_policy(ROOT / CONFIG)
    images, truth = population(ROOT)
    if args.infer:
        infer(ROOT, images, policy)
    predictions, execution = load_predictions(ROOT, images, policy)
    snapshot = ROOT / CACHE / "accepted_inference_source.py"
    if not snapshot.exists():
        # A fresh execution may still be using this exact file; preserve it once.
        if sha256_file(Path(__file__)) != execution["implementation_sha256"]:
            raise ValueError("Original inference implementation snapshot missing")
        snapshot.write_bytes(Path(__file__).read_bytes())
    if sha256_file(snapshot) != execution["implementation_sha256"]:
        raise ValueError("Accepted inference source snapshot digest mismatch")

    def inference_ast(path: Path) -> str:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        return ast.dump(
            next(
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "infer"
            )
        )

    if inference_ast(snapshot) != inference_ast(Path(__file__)):
        raise ValueError("Inference implementation changed after execution")
    records = analyze(images, truth, predictions)
    selections = {m: choose_examples(r) for m, r in records.items()}
    masks = choose_masks(records["S1"])
    hero = choose_hero(records, images, policy["hero"])
    coverage = all(
        any(
            s["status"] == "SELECTED"
            for m in selections.values()
            for s in m
            if s["class"] == name and s["category"] == category
        )
        for name in CLASSES
        for category in ("FALSE_POSITIVE", "FALSE_NEGATIVE")
    )

    def without_masks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{k: v for k, v in row.items() if k != "mask_rle"} for row in rows]

    payload = {
        "schema_version": 1,
        "phase": "12C",
        "classification": COMPLETE,
        "purpose": policy["purpose"],
        "policy": policy,
        "policy_sha256": fingerprint(policy),
        "baseline_commit": BASELINE,
        "split": "validation",
        "images": images,
        "population_images": len(images),
        "population_annotations": len(truth),
        "model_inference_required": True,
        "execution": execution,
        "ground_truth": without_masks(truth),
        "predictions": {m: without_masks(rows) for m, rows in predictions.items()},
        "mask_pairs": [r for r in records["S1"] if "mask_iou" in r],
        "selections": selections,
        "mask_selections": masks,
        "hero": hero,
        "gap_005_status": "RESOLVED" if coverage else "PARTIALLY_RESOLVED",
        "assignment_r13_status": "COMPLETE" if coverage else "PARTIAL",
        "attribution": {
            "source": SOURCE_URL,
            "license": "CC BY 4.0",
            "license_url": LICENSE_URL,
            "credit": "agis-workspace-8gs52 / Construction PPE Compliance Detection / Roboflow "
            "Universe",
            "changes": "Original-canvas model and GT overlays; figure layout; no scene cropping",
            "image_license_is_agpl": False,
        },
        "scientific_results_modified": False,
        "headline_metrics_computed": False,
        "holdout_content_accessed": False,
        "manual_example_replacement": False,
    }
    payload["figures"] = render(ROOT, payload, truth, predictions)
    write_json(ROOT / REPORT, payload)
    write_json(ROOT / STATUS_PATH, build_status(ROOT))
    # Text and metadata validation are separate so editorial/rendering changes never rerun a model.
    from construction_safety_vision.qualitative_gallery_report import (
        render_report,
        validate_gallery,
    )

    (ROOT / "reports/qualitative_validation_gallery.md").write_text(
        render_report(payload), encoding="utf-8", newline="\n"
    )
    record = ProvenanceRecord.create(
        "qualitative_validation_gallery",
        phase=12,
        repo_root=ROOT,
        config=policy,
        details={
            "execution": execution,
            "classification": COMPLETE,
            "model_inference_required": True,
            "scientific_results_modified": False,
            "test_content_accessed": False,
        },
    )
    record.git_commit = BASELINE
    inputs = [
        CONFIG,
        "scripts/build_qualitative_gallery.py",
        "src/construction_safety_vision/qualitative_gallery.py",
        "src/construction_safety_vision/qualitative_gallery_report.py",
        "reports/final_detector_manifest.json",
        "reports/final_segmenter_manifest.json",
        "reports/task_dataset_manifest.json",
        "reports/dataset_provenance.json",
        "configs/detector_segmenter_comparison.yaml",
        "data/processed/canonical/annotations/detection_validation.coco.json",
        "data/processed/canonical/annotations/segmentation_validation.coco.json",
        f"{CACHE}/D2.json",
        f"{CACHE}/S1.json",
        f"{CACHE}/execution.json",
        f"{CACHE}/accepted_inference_source.py",
        "src/construction_safety_vision/delivery_status.py",
        "README.md",
        "reports/README.md",
        "reports/roadmap.md",
    ]
    inputs += [image["path"] for image in images.values()]
    if "prior_execution" in execution:
        inputs += [
            f"artifacts/qualitative_validation_batching_deviation/{name}"
            for name in ("execution.json", "initial_source.py", "D2.json", "S1.json")
        ]
    for relative in inputs:
        record.add_input(ROOT / relative, relative_to=ROOT)
    for relative in [
        REPORT,
        "reports/qualitative_validation_gallery.md",
        STATUS_PATH,
        *payload["figures"],
    ]:
        record.add_output(ROOT / relative, relative_to=ROOT)
    record.write_json(ROOT / PROVENANCE)
    problems = validate_gallery(ROOT)
    if problems:
        raise ValueError(problems)
    print(COMPLETE, "GAP-005:", payload["gap_005_status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
