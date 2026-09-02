"""Perceptual fingerprints for near-duplicate *candidate* generation.

A perceptual hash match is a candidate, never a verdict. Two frames of the same
video and two different photographs of the same wall can produce identical
fingerprints, and a genuine duplicate can survive a re-crop with a large
distance. Everything produced here is therefore routed to human review rather
than acted on.

Two independent fingerprints are computed so that a pair flagged by both is
stronger evidence than one flagged by either:

* **dHash** - sign of the horizontal gradient on a 9x8 grey thumbnail. Sensitive
  to structure, robust to uniform brightness shifts.
* **pHash** - sign of the low-frequency DCT coefficients of a 32x32 grey
  thumbnail, relative to their median. Robust to smooth global changes.

Both are deterministic: the same file always yields the same fingerprint, which
is what makes the candidate list reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image

DHASH_SIDE = 8
"""Fingerprint side length; dHash compares ``side + 1`` columns per row."""

PHASH_IMAGE_SIDE = 32
"""Thumbnail side used before the DCT."""

PHASH_LOW_FREQUENCY_SIDE = 8
"""Size of the retained low-frequency DCT block."""

DHASH_THRESHOLD = 10
"""Maximum dHash Hamming distance for a pair to be emitted as a candidate.

Chosen before looking at this dataset's results, from the conventional working
range for 64-bit perceptual hashes: distances up to about 10 of 64 bits (~15%)
are the usual "possibly the same scene" band. It is a recall-oriented screen -
false candidates are expected and are resolved by human review, whereas a missed
duplicate could silently leak across a split boundary.
"""

PHASH_THRESHOLD = 10
"""Maximum pHash Hamming distance for a pair to be emitted as a candidate."""


@dataclass(frozen=True)
class Fingerprint:
    """Perceptual fingerprints of one image.

    Attributes:
        image_id: Stable identifier of the image.
        dhash: 64-bit difference hash.
        phash: 64-bit perceptual hash.
    """

    image_id: str
    dhash: int
    phash: int


@dataclass(frozen=True)
class CandidatePair:
    """A pair of images that a fingerprint flagged as possibly similar.

    Attributes:
        image_a: Identifier of the first image, always the lexicographically
            smaller of the two.
        image_b: Identifier of the second image.
        dhash_distance: Hamming distance between the difference hashes.
        phash_distance: Hamming distance between the perceptual hashes.
        reasons: Which fingerprints flagged the pair.
    """

    image_a: str
    image_b: str
    dhash_distance: int
    phash_distance: int
    reasons: tuple[str, ...]

    @property
    def min_distance(self) -> int:
        """Smallest of the two distances, used for ranking.

        Returns:
            The stronger (smaller) of the two distances.
        """
        return min(self.dhash_distance, self.phash_distance)


def _grey_array(image: Image.Image, width: int, height: int) -> np.ndarray:
    """Resize an image to a grey array of the requested size.

    Args:
        image: Source image.
        width: Target width.
        height: Target height.

    Returns:
        A float array of grey values.
    """
    resized = image.convert("L").resize((width, height), Image.Resampling.LANCZOS)
    return np.asarray(resized, dtype=np.float64)


def _bits_to_int(bits: np.ndarray) -> int:
    """Pack a boolean array into an integer, most significant bit first.

    Args:
        bits: Boolean array.

    Returns:
        The packed integer.
    """
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bool(bit))
    return value


def dhash(image: Image.Image, *, side: int = DHASH_SIDE) -> int:
    """Compute the difference hash of an image.

    Args:
        image: Source image.
        side: Fingerprint side length.

    Returns:
        A ``side * side``-bit integer.
    """
    grey = _grey_array(image, side + 1, side)
    return _bits_to_int(grey[:, 1:] > grey[:, :-1])


def _dct_matrix(size: int) -> np.ndarray:
    """Build the orthonormal DCT-II basis matrix.

    Implemented directly rather than pulling in a transform library for one
    matrix multiplication.

    Args:
        size: Transform size.

    Returns:
        A ``size x size`` basis matrix.
    """
    indices = np.arange(size)
    basis = np.cos(np.pi * (2 * indices[None, :] + 1) * indices[:, None] / (2 * size))
    basis *= np.sqrt(2.0 / size)
    basis[0] *= np.sqrt(0.5)
    return basis


def phash(
    image: Image.Image,
    *,
    image_side: int = PHASH_IMAGE_SIDE,
    low_side: int = PHASH_LOW_FREQUENCY_SIDE,
) -> int:
    """Compute the DCT-based perceptual hash of an image.

    Args:
        image: Source image.
        image_side: Thumbnail side used before the transform.
        low_side: Size of the retained low-frequency block.

    Returns:
        A ``low_side * low_side``-bit integer.
    """
    grey = _grey_array(image, image_side, image_side)
    basis = _dct_matrix(image_side)
    coefficients = basis @ grey @ basis.T
    block = coefficients[:low_side, :low_side]
    # The DC term encodes overall brightness and would swamp the median.
    median = float(np.median(np.concatenate([block.flatten()[:1] * 0, block.flatten()[1:]])))
    return _bits_to_int(block > median)


def fingerprint_file(image_id: str, path: Path) -> Fingerprint:
    """Compute both fingerprints for one image file.

    Args:
        image_id: Stable identifier to attach to the result.
        path: Image file.

    Returns:
        The fingerprints.
    """
    with Image.open(path) as image:
        image.load()
        return Fingerprint(image_id=image_id, dhash=dhash(image), phash=phash(image))


def hamming(left: int, right: int) -> int:
    """Count differing bits between two fingerprints.

    Args:
        left: First fingerprint.
        right: Second fingerprint.

    Returns:
        The Hamming distance.
    """
    return int(left ^ right).bit_count()


def candidate_pairs(
    fingerprints: list[Fingerprint],
    *,
    dhash_threshold: int = DHASH_THRESHOLD,
    phash_threshold: int = PHASH_THRESHOLD,
) -> list[CandidatePair]:
    """Generate near-duplicate candidates by exhaustive pairwise comparison.

    The population is small enough that every pair can be compared, which avoids
    the recall loss of a bucketing scheme. Each unordered pair is emitted at most
    once and never paired with itself.

    Args:
        fingerprints: Fingerprints to compare.
        dhash_threshold: Maximum dHash distance to emit.
        phash_threshold: Maximum pHash distance to emit.

    Returns:
        Candidates sorted by strongest evidence first, then by identifier so the
        output is deterministic.
    """
    ordered = sorted(fingerprints, key=lambda f: f.image_id)
    pairs: list[CandidatePair] = []
    for left, right in combinations(ordered, 2):
        d_distance = hamming(left.dhash, right.dhash)
        p_distance = hamming(left.phash, right.phash)
        reasons = []
        if d_distance <= dhash_threshold:
            reasons.append("dhash")
        if p_distance <= phash_threshold:
            reasons.append("phash")
        if not reasons:
            continue
        pairs.append(
            CandidatePair(
                image_a=left.image_id,
                image_b=right.image_id,
                dhash_distance=d_distance,
                phash_distance=p_distance,
                reasons=tuple(reasons),
            )
        )
    pairs.sort(key=lambda p: (p.min_distance, p.image_a, p.image_b))
    return pairs
