"""Tests for decoded-image statistics.

The behaviour that matters most is the corrupt-file path: a file that cannot be
decoded must be reported, never dropped, because silently shrinking the audited
population would falsify every count derived from it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from construction_safety_vision.data.imagestats import (
    compute_image_stats,
    sha256_of,
    summarise,
)


def save_solid(path: Path, value: int, size: tuple[int, int] = (40, 20)) -> Path:
    """Write a uniform grey image."""
    Image.fromarray(np.full((size[1], size[0]), value, dtype=np.uint8), "L").save(path)
    return path


def test_measures_dimensions_and_mode(tmp_path: Path) -> None:
    path = save_solid(tmp_path / "a.png", 128, size=(40, 20))

    stats = compute_image_stats(path)

    assert stats.decoded
    assert (stats.width, stats.height) == (40, 20)
    assert stats.mode == "L"
    assert stats.pixels == 800
    assert stats.aspect_ratio == pytest.approx(2.0)
    assert stats.size_bytes > 0


def test_hash_matches_the_standalone_helper(tmp_path: Path) -> None:
    path = save_solid(tmp_path / "a.png", 10)
    assert compute_image_stats(path).sha256 == sha256_of(path)


def test_uniform_image_has_zero_contrast(tmp_path: Path) -> None:
    stats = compute_image_stats(save_solid(tmp_path / "flat.png", 200))

    assert stats.mean_luminance == pytest.approx(200 / 255)
    assert stats.std_luminance == pytest.approx(0.0)
    assert stats.dynamic_range == pytest.approx(0.0)


def test_luminance_ordering_is_meaningful(tmp_path: Path) -> None:
    dark = compute_image_stats(save_solid(tmp_path / "dark.png", 20))
    bright = compute_image_stats(save_solid(tmp_path / "bright.png", 230))

    assert dark.mean_luminance < bright.mean_luminance
    assert 0.0 <= dark.mean_luminance <= 1.0
    assert 0.0 <= bright.mean_luminance <= 1.0


def test_high_contrast_image_has_a_wide_dynamic_range(tmp_path: Path) -> None:
    half = np.zeros((20, 40), dtype=np.uint8)
    half[:, 20:] = 255
    path = tmp_path / "split.png"
    Image.fromarray(half, "L").save(path)

    stats = compute_image_stats(path)

    assert stats.std_luminance > 0.4
    assert stats.dynamic_range == pytest.approx(1.0, abs=0.05)


def test_rgb_images_are_converted_for_luminance(tmp_path: Path) -> None:
    path = tmp_path / "rgb.png"
    Image.fromarray(np.full((10, 10, 3), 128, dtype=np.uint8), "RGB").save(path)

    stats = compute_image_stats(path)

    assert stats.mode == "RGB"
    assert stats.mean_luminance == pytest.approx(128 / 255, abs=0.01)


def test_a_corrupt_file_is_reported_not_dropped(tmp_path: Path) -> None:
    path = tmp_path / "broken.jpg"
    path.write_bytes(b"this is definitely not a jpeg")

    stats = compute_image_stats(path)

    assert stats.decoded is False
    assert stats.error
    assert stats.sha256, "the bytes are still identified even when undecodable"
    assert stats.size_bytes == 29
    assert stats.width == 0
    assert stats.aspect_ratio is None
    assert stats.dynamic_range is None


def test_a_truncated_image_is_reported(tmp_path: Path) -> None:
    good = tmp_path / "good.png"
    save_solid(good, 100)
    truncated = tmp_path / "truncated.png"
    truncated.write_bytes(good.read_bytes()[:20])

    assert compute_image_stats(truncated).decoded is False


def test_summarise_of_an_empty_sample_claims_nothing() -> None:
    assert summarise([]) == {"count": 0}


def test_summarise_reports_the_expected_fields() -> None:
    result = summarise([1.0, 2.0, 3.0, 4.0, 5.0])

    assert result["count"] == 5
    assert result["min"] == 1.0
    assert result["max"] == 5.0
    assert result["median"] == 3.0
    assert result["mean"] == pytest.approx(3.0)
    assert result["p25"] == pytest.approx(2.0)
    assert result["p75"] == pytest.approx(4.0)


def test_summarise_is_order_independent() -> None:
    assert summarise([3.0, 1.0, 2.0]) == summarise([1.0, 2.0, 3.0])
