"""Cheap deterministic geometric video fixtures, never assignment-video substitutes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from construction_safety_vision.video_runtime import file_identity, open_writer, verify_output


def create_fixture(
    path: Path, *, frames: int = 6, width: int = 320, height: int = 192, fps: float = 10.0
) -> dict[str, Any]:
    """Encode indexed geometry with no dataset, model, network or random input."""
    if path.exists():
        raise FileExistsError("Fixture exists; do not replace its evidence")
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = open_writer(path, fps, (width, height))
    try:
        for index in range(frames):
            canvas = np.full((height, width, 3), 30 + index * 20, dtype=np.uint8)
            x = 10 + index * 12
            cv2.rectangle(canvas, (x, 50), (x + 35, 90), (60, 150, 210), -1)
            cv2.circle(canvas, (width - 40, height - 40), 16, (200, 120, 50), -1)
            cv2.putText(
                canvas,
                f"SYNTHETIC {index}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            writer.write(canvas)
    finally:
        writer.release()
    meta = verify_output(path, frames=frames, fps=fps, dimensions=(width, height))
    return {
        "file": file_identity(path),
        "video": meta,
        "attribution": {
            "source_type": "SYNTHETIC_ENGINEERING_FIXTURE",
            "title": "Indexed geometric frames",
            "creator": "Repository deterministic fixture generator",
            "url": None,
            "license": "Repository-generated synthetic geometry",
            "retrieval_date": None,
            "modifications": "Model overlays; no external or dataset imagery",
        },
    }
