"""Descriptive statistics computed from decoded source images.

Everything here is a measurement of pixels, deliberately kept free of judgement.
The luminance and contrast figures are *proxies*: they describe the distribution
of grey values, not whether a photograph is well lit. Reports must use neutral
language such as "lower-luminance subset" rather than "poorly lit", because a
dark image may be correctly exposed for a dark scene.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

CHUNK_SIZE = 1 << 20
"""Bytes read per iteration when hashing."""

LUMINANCE_MAX = 255.0
"""Maximum 8-bit grey value, used to normalise the proxies to 0..1."""


@dataclass(frozen=True)
class ImageStats:
    """Measurements of one decoded image.

    Attributes:
        sha256: Digest of the file bytes.
        size_bytes: File size on disk.
        width: Decoded width in pixels.
        height: Decoded height in pixels.
        mode: Pillow colour mode, e.g. ``RGB`` or ``L``.
        decoded: Whether the file decoded successfully.
        error: Failure description when ``decoded`` is ``False``.
        mean_luminance: Mean grey value, normalised to 0..1.
        std_luminance: Standard deviation of grey values, normalised. Used as
            the contrast proxy.
        p05_luminance: 5th percentile grey value, normalised.
        p95_luminance: 95th percentile grey value, normalised.
    """

    sha256: str
    size_bytes: int
    width: int = 0
    height: int = 0
    mode: str = ""
    decoded: bool = False
    error: str = ""
    mean_luminance: float | None = None
    std_luminance: float | None = None
    p05_luminance: float | None = None
    p95_luminance: float | None = None

    @property
    def pixels(self) -> int:
        """Total pixel count.

        Returns:
            ``width * height``.
        """
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float | None:
        """Width divided by height.

        Returns:
            The aspect ratio, or ``None`` when the height is unknown.
        """
        return self.width / self.height if self.height else None

    @property
    def dynamic_range(self) -> float | None:
        """Spread between the 5th and 95th luminance percentiles.

        A second contrast proxy, less sensitive to outliers than the standard
        deviation.

        Returns:
            The normalised spread, or ``None`` when the image did not decode.
        """
        if self.p05_luminance is None or self.p95_luminance is None:
            return None
        return self.p95_luminance - self.p05_luminance


def sha256_of(path: Path, *, chunk_size: int = CHUNK_SIZE) -> str:
    """Hash a file in chunks.

    Args:
        path: File to hash.
        chunk_size: Bytes read per iteration.

    Returns:
        The hexadecimal digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def compute_image_stats(path: Path) -> ImageStats:
    """Measure one image file.

    A file that cannot be decoded is reported as such rather than raising, so a
    single corrupt image never silently shrinks the audited population.

    Args:
        path: Image file to measure.

    Returns:
        The measurements, with ``decoded=False`` and a populated ``error`` when
        the file could not be read as an image.
    """
    size_bytes = path.stat().st_size
    digest = sha256_of(path)
    try:
        with Image.open(path) as image:
            image.load()
            mode = image.mode
            width, height = image.size
            grey = np.asarray(image.convert("L"), dtype=np.float64)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return ImageStats(
            sha256=digest,
            size_bytes=size_bytes,
            decoded=False,
            error=f"{type(exc).__name__}: {exc}",
        )

    percentiles = np.percentile(grey, [5, 95]) if grey.size else np.array([0.0, 0.0])
    return ImageStats(
        sha256=digest,
        size_bytes=size_bytes,
        width=width,
        height=height,
        mode=mode,
        decoded=True,
        mean_luminance=float(grey.mean() / LUMINANCE_MAX) if grey.size else None,
        std_luminance=float(grey.std() / LUMINANCE_MAX) if grey.size else None,
        p05_luminance=float(percentiles[0] / LUMINANCE_MAX) if grey.size else None,
        p95_luminance=float(percentiles[1] / LUMINANCE_MAX) if grey.size else None,
    )


def summarise(values: list[float]) -> dict[str, float]:
    """Summarise a numeric distribution.

    Args:
        values: Sample values. May be empty.

    Returns:
        Count, min, percentiles, median, mean, max and standard deviation.
        Every field is ``0.0`` for an empty sample except ``count``.
    """
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=np.float64)
    p05, p25, p50, p75, p95 = np.percentile(array, [5, 25, 50, 75, 95])
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "p05": float(p05),
        "p25": float(p25),
        "median": float(p50),
        "p75": float(p75),
        "p95": float(p95),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "std": float(array.std()),
    }
