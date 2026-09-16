"""Validation-only delivery selection, with no inference or dataset access.

The object matcher is a metric-free adapter of the established
``final_holdout_execution.object_level_outcomes`` greedy loop. Keeping the
adapter here avoids importing a holdout execution route or computing aggregate
precision/recall merely to choose illustrations. Synthetic tests verify parity
with that authoritative implementation. Historical code and results stay intact.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from construction_safety_vision.segmentation_error_analysis import (
    AREA_RATIO_TOLERANCE,
    HIGH_QUALITY_MASK,
    LOW_OVERLAP_MASK,
    MODERATE_MASK,
    outcome_band,
)

BASELINE = "73098b7c62d9cb183cb3114af6209a65f1c532e7"
CLASSES = ("helmet_loose", "helmet_on_head", "person", "vest_loose", "vest_on_body")
CHECKPOINTS = {
    "D2": "0466f872a9de22898c70d8834cf1fcfb3d77f6c9fb27e81ee6248d3d19cbc206",
    "S1": "29337d671459d0f742fb713613cc41d0eafcb8a21f5846ac0da65b2ffc024f20",
}
CONFIG = "configs/qualitative_validation_gallery.yaml"
REPORT = "reports/qualitative_validation_gallery.json"
PROVENANCE = "reports/qualitative_validation_gallery.provenance.json"
FIGURES = "reports/figures/qualitative"
COMPLETE = "QUALITATIVE_VALIDATION_GALLERY_COMPLETE"
NO_EXAMPLE = "NO_VALID_EXAMPLE"
SOURCE_URL = (
    "https://universe.roboflow.com/agis-workspace-8gs52/construction-ppe-compliance-detection"
)
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"


def fingerprint(value: Any) -> str:
    """Hash deterministic JSON content."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def load_policy(path: Path) -> dict[str, Any]:
    """Load the strict delivery policy; reject unknown keys and scientific drift."""
    policy = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version",
        "purpose",
        "split",
        "inference",
        "matching",
        "selection",
        "hero",
        "render",
    }
    if set(policy) != expected or policy["schema_version"] != 1:
        raise ValueError("Invalid gallery policy schema")
    if policy["split"] != "validation" or policy["purpose"] != "DELIVERY_VISUALIZATION_ONLY":
        raise ValueError("Only validation delivery visualization is permitted")
    if policy["inference"] != {
        "imgsz": 768,
        "conf": 0.25,
        "iou": 0.7,
        "max_det": 300,
        "augment": False,
        "tta": False,
        "precision": "FP32",
        "quantize": 32,
        "batch": 1,
        "retina_masks_s1": True,
    }:
        raise ValueError("Frozen inference settings changed")
    if policy["matching"] != {
        "box_iou": 0.5,
        "class_aware": True,
        "assignment": "GREEDY_DESCENDING_CONFIDENCE_THEN_HIGHEST_IOU",
        "prediction_tie": "ORIGINAL_PREDICTION_INDEX",
        "ground_truth_tie": "CANONICAL_ANNOTATION_ID",
    }:
        raise ValueError("Frozen matching semantics changed")
    if policy["selection"] != {
        "fp": "CONFIDENCE_DESC_PREDICTION_INDEX_ASC_IMAGE_ID_ASC",
        "fn": "CANONICAL_AREA_DESC_ANNOTATION_ID_ASC_IMAGE_ID_ASC",
        "masks": "EXISTING_8D_BANDS_AND_AREA_FLAGS_ON_BOX_MATCHED_PAIRS",
        "good_mask": "MASK_IOU_DESC_GT_AREA_DESC_ANNOTATION_ID_ASC_IMAGE_ID_ASC",
        "under_mask": "RELATIVE_AREA_ERROR_ASC_ANNOTATION_ID_ASC_IMAGE_ID_ASC",
        "over_mask": "RELATIVE_AREA_ERROR_DESC_ANNOTATION_ID_ASC_IMAGE_ID_ASC",
        "hero": "BOTH_MODELS_MATCH_PPE_HIGH_QUALITY_MASK_NONRECTANGULAR_THEN_FEWEST_ERRORS",
    }:
        raise ValueError("Selection rule not implemented")
    if policy["hero"] != {
        "minimum_gt_area_fraction": 0.005,
        "maximum_mask_box_fill": 0.85,
        "minimum_mask_box_fill": 0.1,
        "minimum_image_width": 640,
        "rank": "TOTAL_D2_S1_FP_FN_ASC_GT_AREA_FRACTION_DESC_MASK_IOU_DESC_ID_ASC",
    }:
        raise ValueError("Hero rule changed; no result-driven replacement permitted")
    if policy["render"] != {
        "font": "DejaVu Sans",
        "mask_alpha": 0.32,
        "full_image": True,
        "preserve_aspect_ratio": True,
    }:
        raise ValueError("Unknown rendering configuration")
    return policy


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute the established continuous-coordinate box intersection over union."""
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1])
    )
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def validate_rows(rows: Sequence[Mapping[str, Any]], validation_ids: set[str]) -> None:
    """Reject nonvalidation membership before rendering, decoding masks or reading images.

    The frozen accessor verifies a disjoint partition. Positive validation
    membership is therefore sufficient to exclude test, without requesting test IDs.
    """
    for row in rows:
        if row["split"] != "validation" or row["image_id"] not in validation_ids:
            raise ValueError("Source outside frozen validation membership")
        if row["class"] not in CLASSES:
            raise ValueError("Unknown class label")
        box = row["box"]
        if len(box) != 4 or not all(math.isfinite(v) for v in box):
            raise ValueError("Invalid box")
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("Nonpositive box")
        if "score" in row and not 0.25 <= row["score"] <= 1:
            raise ValueError("Prediction is outside the frozen operating point")


def match_objects(
    truth: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    validation_ids: set[str],
) -> list[dict[str, Any]]:
    """Return TP/FP/FN evidence without computing a new aggregate performance metric."""
    validate_rows(truth, validation_ids)
    validate_rows(predictions, validation_ids)
    if len({(g["image_id"], g["annotation_id"]) for g in truth}) != len(truth):
        raise ValueError("Duplicate ground-truth key")
    if len({(p["image_id"], p["prediction_index"]) for p in predictions}) != len(predictions):
        raise ValueError("Duplicate prediction key")
    records = []
    for image_id in sorted(validation_ids):
        gt = sorted(
            (g for g in truth if g["image_id"] == image_id), key=lambda g: g["annotation_id"]
        )
        pred = sorted(
            (p for p in predictions if p["image_id"] == image_id),
            key=lambda p: (-p["score"], p["prediction_index"]),
        )
        claimed = set()
        for p in pred:
            candidates = [
                (box_iou(g["box"], p["box"]), g)
                for g in gt
                if g["annotation_id"] not in claimed and g["class"] == p["class"]
            ]
            candidates = [(iou, g) for iou, g in candidates if iou >= 0.5]
            chosen = (
                min(candidates, key=lambda x: (-x[0], x[1]["annotation_id"]))
                if candidates
                else None
            )
            record = {k: v for k, v in p.items() if k != "mask_rle"}
            record["category"] = "TRUE_POSITIVE" if chosen else "FALSE_POSITIVE"
            if chosen:
                iou, g = chosen
                claimed.add(g["annotation_id"])
                record.update(annotation_id=g["annotation_id"], gt_area=g["area"], box_iou=iou)
            records.append(record)
        for g in gt:
            if g["annotation_id"] in claimed:
                continue
            overlaps = [p for p in pred if box_iou(g["box"], p["box"]) >= 0.5]
            mismatch = bool(overlaps) and not any(p["class"] == g["class"] for p in overlaps)
            record = {k: v for k, v in g.items() if k != "mask_rle"}
            record.update(category="FALSE_NEGATIVE", gt_area=g["area"])
            record["classification_mismatch"] = mismatch
            records.append(record)
    return records


def mask_category(iou: float, relative_area_error: float) -> str | None:
    """Map existing Phase 8D bands/area flags to the requested display labels."""
    band = outcome_band(matched=True, mask_iou=iou)
    if band == HIGH_QUALITY_MASK:
        return "GOOD_MASK_MATCH"
    if band in (LOW_OVERLAP_MASK, MODERATE_MASK):
        if relative_area_error < -AREA_RATIO_TOLERANCE:
            return "MASK_UNDER_COVERAGE"
        if relative_area_error > AREA_RATIO_TOLERANCE:
            return "MASK_OVER_COVERAGE"
    return None


def choose_examples(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Choose one FP and FN per class over the complete population for one model."""
    selections = []
    for name in CLASSES:
        for category in ("FALSE_POSITIVE", "FALSE_NEGATIVE"):
            candidates = [r for r in records if r["class"] == name and r["category"] == category]

            def ranking(r: Mapping[str, Any], kind: str = category) -> tuple[Any, ...]:
                if kind == "FALSE_POSITIVE":
                    return (-r["score"], r["prediction_index"], r["image_id"])
                return (-r["gt_area"], r["annotation_id"], r["image_id"])

            selections.append(
                {
                    "class": name,
                    "category": category,
                    "available": len(candidates),
                    "status": "SELECTED" if candidates else NO_EXAMPLE,
                    "example": dict(min(candidates, key=ranking)) if candidates else None,
                }
            )
    return selections


def choose_masks(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Select strong, under- and over-coverage cases using declared stable ranks."""
    selected = []
    for category in ("GOOD_MASK_MATCH", "MASK_UNDER_COVERAGE", "MASK_OVER_COVERAGE"):
        candidates = [r for r in records if r.get("mask_category") == category]

        def ranking(r: Mapping[str, Any], kind: str = category) -> tuple[Any, ...]:
            tie = (r["annotation_id"], r["image_id"], r["prediction_index"])
            if kind == "GOOD_MASK_MATCH":
                return (-r["mask_iou"], -r["gt_area"], *tie)
            value = r["relative_area_error"]
            return (value if kind == "MASK_UNDER_COVERAGE" else -value, *tie)

        selected.append(
            {
                "category": category,
                "available": len(candidates),
                "status": "SELECTED" if candidates else NO_EXAMPLE,
                "example": dict(min(candidates, key=ranking)) if candidates else None,
            }
        )
    return selected


def choose_hero(
    records: Mapping[str, Sequence[Mapping[str, Any]]],
    images: Mapping[str, Mapping[str, Any]],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Choose a shared correct PPE instance without opening any scene."""
    d2_matches = {
        (r["image_id"], r["annotation_id"])
        for r in records["D2"]
        if r["category"] == "TRUE_POSITIVE"
    }
    candidates = []
    for r in records["S1"]:
        if r.get("mask_category") != "GOOD_MASK_MATCH" or r["class"] == "person":
            continue
        image = images[r["image_id"]]
        area_fraction = r["gt_area"] / (image["width"] * image["height"])
        if (r["image_id"], r["annotation_id"]) not in d2_matches:
            continue
        if image["width"] < policy["minimum_image_width"]:
            continue
        if area_fraction < policy["minimum_gt_area_fraction"]:
            continue
        if (
            not policy["minimum_mask_box_fill"]
            <= r["mask_box_fill"]
            <= policy["maximum_mask_box_fill"]
        ):
            continue
        errors = sum(
            x["image_id"] == r["image_id"] and x["category"] != "TRUE_POSITIVE"
            for model_records in records.values()
            for x in model_records
        )
        candidates.append(dict(r, image_error_count=errors, gt_area_fraction=area_fraction))
    rank = lambda r: (  # noqa: E731
        r["image_error_count"],
        -r["gt_area_fraction"],
        -r["mask_iou"],
        r["annotation_id"],
        r["image_id"],
        r["prediction_index"],
    )
    return {
        "status": "HERO_CANDIDATE_NOT_FINAL" if candidates else NO_EXAMPLE,
        "eligible_candidates": len(candidates),
        "example": min(candidates, key=rank) if candidates else None,
        "manual_replacement": False,
    }
