"""Original-canvas video overlays and same-frame side-by-side composition."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from construction_safety_vision.video_models import CLASSES, MODES, VideoRuntimeError

# BGR, fixed per canonical class in every frame and model.
COLORS = ((80, 190, 255), (100, 220, 110), (255, 180, 70), (220, 100, 210), (60, 220, 240))
MASK_ALPHA = 0.32


def render_panel(frame: np.ndarray, predictions: list[dict[str, Any]], model: str) -> np.ndarray:
    """Draw masks on their exact original canvas, followed by labelled model boxes."""
    canvas = frame.copy()
    height, width = canvas.shape[:2]
    for row in predictions:
        class_id = row["class_id"]
        if type(class_id) is not int or class_id not in range(len(CLASSES)):
            raise VideoRuntimeError("Renderer received an invalid canonical class")
        mask = row["mask"]
        if model == "S1":
            if mask is None or mask.shape != (height, width) or mask.dtype != np.bool_:
                raise VideoRuntimeError("Renderer mask/canvas mismatch")
            canvas[mask] = np.rint(
                (1 - MASK_ALPHA) * canvas[mask] + MASK_ALPHA * np.array(COLORS[class_id])
            ).astype(np.uint8)
    for row in predictions:
        box = np.asarray(row["box"], dtype=float)
        if box.shape != (4,) or not np.isfinite(box).all() or not 0 <= row["confidence"] <= 1:
            raise VideoRuntimeError("Renderer received invalid prediction geometry/confidence")
        x1, y1, x2, y2 = np.rint(box).astype(int)
        if x2 < x1 or y2 < y1:
            raise VideoRuntimeError("Renderer received reversed box coordinates")
        x1, x2 = np.clip([x1, x2], 0, width - 1)
        y1, y2 = np.clip([y1, y2], 0, height - 1)
        color = COLORS[row["class_id"]]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"{CLASSES[row['class_id']]} {row['confidence']:.2f}"
        scale = min(0.55, max(0.30, (width - 8) / (len(label) * 12)))
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        tx, ty = min(x1, max(0, width - tw - 4)), max(th + 4, y1 - 4)
        cv2.rectangle(canvas, (tx, ty - th - 3), (tx + tw + 3, ty + 3), (20, 20, 20), -1)
        cv2.putText(
            canvas, label, (tx + 1, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA
        )
    title = "D2 - Bounding Boxes" if model == "D2" else "S1 - Instance Masks"
    cv2.rectangle(canvas, (0, 0), (width, 27), (22, 25, 30), -1)
    cv2.putText(
        canvas,
        title,
        (6, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        min(0.55, width / 430),
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def render_frame(
    frame: np.ndarray,
    predictions: dict[str, Any],
    mode: str,
    source_fps: float,
    throughput: float | None = None,
    fps_overlay: bool = True,
) -> np.ndarray:
    """Keep source aspect ratio; compare concatenates two unscaled copies horizontally."""
    if mode not in MODES or set(predictions) != set(MODES[mode]):
        raise VideoRuntimeError("Renderer mode/model mismatch")
    panels = [render_panel(frame, predictions[name], name) for name in MODES[mode]]
    canvas = np.concatenate(panels, axis=1) if mode == "compare" else panels[0]
    if fps_overlay:
        rate = "pending" if throughput is None else f"{throughput:.1f}"
        label = f"SOURCE {source_fps:.2f} fps | DEMO LOOP {rate} fps (previous frames)"
        scale = min(0.48, canvas.shape[1] / 1100)
        cv2.rectangle(
            canvas, (0, canvas.shape[0] - 22), (canvas.shape[1], canvas.shape[0]), (22, 25, 30), -1
        )
        cv2.putText(
            canvas,
            label,
            (5, canvas.shape[0] - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas
