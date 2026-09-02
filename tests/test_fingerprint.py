"""Tests for perceptual fingerprints and near-duplicate candidate generation.

Fixtures are synthetic images built in memory. The properties that matter are
determinism (the candidate list must be reproducible), and the structural rules
of pair generation: no self-pairs, no mirrored duplicates, stable ordering.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from construction_safety_vision.data.fingerprint import (
    CandidatePair,
    Fingerprint,
    candidate_pairs,
    dhash,
    fingerprint_file,
    hamming,
    phash,
)

RNG = np.random.default_rng(20260901)


def noise_image(seed: int, size: int = 128) -> Image.Image:
    """Build a deterministic random image."""
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (size, size, 3), dtype=np.uint8), "RGB")


def gradient_image(size: int = 128, *, flip: bool = False) -> Image.Image:
    """Build a smooth horizontal gradient."""
    row = np.linspace(0, 255, size, dtype=np.uint8)
    if flip:
        row = row[::-1]
    return Image.fromarray(np.tile(row, (size, 1)), "L").convert("RGB")


def test_hashes_are_64_bit() -> None:
    image = noise_image(1)
    assert 0 <= dhash(image) < 2**64
    assert 0 <= phash(image) < 2**64


def test_hashes_are_deterministic() -> None:
    # Reproducibility of the whole candidate list rests on this.
    image = noise_image(2)
    assert dhash(image) == dhash(noise_image(2))
    assert phash(image) == phash(noise_image(2))


def test_identical_images_have_distance_zero() -> None:
    a, b = noise_image(3), noise_image(3)
    assert hamming(dhash(a), dhash(b)) == 0
    assert hamming(phash(a), phash(b)) == 0


def test_unrelated_images_are_far_apart() -> None:
    a, b = noise_image(4), noise_image(5)
    assert hamming(dhash(a), dhash(b)) > 10
    assert hamming(phash(a), phash(b)) > 10


def test_a_rescaled_image_stays_close() -> None:
    # The point of a perceptual hash: resolution alone must not break the match.
    original = noise_image(6, size=256)
    rescaled = original.resize((128, 128), Image.Resampling.LANCZOS)
    assert hamming(dhash(original), dhash(rescaled)) <= 10
    assert hamming(phash(original), phash(rescaled)) <= 10


def test_phash_tolerates_a_uniform_brightness_shift() -> None:
    base = noise_image(7)
    brighter = Image.fromarray(
        np.clip(np.asarray(base, dtype=np.int16) + 30, 0, 255).astype(np.uint8), "RGB"
    )
    assert hamming(phash(base), phash(brighter)) <= 10


def test_dhash_detects_a_horizontal_flip() -> None:
    assert hamming(dhash(gradient_image()), dhash(gradient_image(flip=True))) > 10


def test_hamming_is_symmetric_and_zero_on_self() -> None:
    a, b = dhash(noise_image(8)), dhash(noise_image(9))
    assert hamming(a, b) == hamming(b, a)
    assert hamming(a, a) == 0


def test_fingerprint_file_reads_from_disk(tmp_path) -> None:
    path = tmp_path / "img.png"
    noise_image(10).save(path)

    result = fingerprint_file("img-1", path)

    assert result.image_id == "img-1"
    assert result == fingerprint_file("img-1", path), "must be stable across reads"


def make_fingerprints(specs: list[tuple[str, int, int]]) -> list[Fingerprint]:
    return [Fingerprint(image_id=i, dhash=d, phash=p) for i, d, p in specs]


def test_no_self_pairs_are_emitted() -> None:
    pairs = candidate_pairs(make_fingerprints([("a", 0, 0), ("b", 0, 0)]))
    assert all(p.image_a != p.image_b for p in pairs)


def test_each_unordered_pair_appears_once() -> None:
    pairs = candidate_pairs(make_fingerprints([("a", 0, 0), ("b", 0, 0), ("c", 0, 0)]))
    keys = {frozenset((p.image_a, p.image_b)) for p in pairs}
    assert len(pairs) == len(keys) == 3


def test_pairs_are_ordered_lexicographically_within_a_pair() -> None:
    pairs = candidate_pairs(make_fingerprints([("zebra", 0, 0), ("alpha", 0, 0)]))
    assert pairs[0].image_a == "alpha"
    assert pairs[0].image_b == "zebra"


def test_distant_pairs_are_not_emitted() -> None:
    far = (1 << 64) - 1
    assert candidate_pairs(make_fingerprints([("a", 0, 0), ("b", far, far)])) == []


def test_a_pair_records_which_fingerprint_flagged_it() -> None:
    # Only dHash is close; the reason list must say so rather than implying both.
    pairs = candidate_pairs(make_fingerprints([("a", 0, 0), ("b", 0b111, (1 << 64) - 1)]))
    assert len(pairs) == 1
    assert pairs[0].reasons == ("dhash",)


def test_thresholds_are_inclusive() -> None:
    at_threshold = make_fingerprints([("a", 0, 0), ("b", 0b11, 0b11)])
    assert candidate_pairs(at_threshold, dhash_threshold=2, phash_threshold=2)
    assert not candidate_pairs(at_threshold, dhash_threshold=1, phash_threshold=1)


def test_output_is_deterministic_regardless_of_input_order() -> None:
    specs = [("c", 0, 0), ("a", 1, 1), ("b", 3, 3)]
    first = candidate_pairs(make_fingerprints(specs))
    second = candidate_pairs(make_fingerprints(list(reversed(specs))))
    assert first == second


def test_candidates_are_ranked_by_strongest_evidence_first() -> None:
    pairs = candidate_pairs(make_fingerprints([("a", 0, 0), ("b", 0, 0), ("c", 0b1111, 0b1111)]))
    distances = [p.min_distance for p in pairs]
    assert distances == sorted(distances)


def test_empty_and_single_populations_produce_nothing() -> None:
    assert candidate_pairs([]) == []
    assert candidate_pairs(make_fingerprints([("only", 0, 0)])) == []


def test_candidate_pair_min_distance_picks_the_stronger_signal() -> None:
    pair = CandidatePair("a", "b", dhash_distance=9, phash_distance=2, reasons=("phash",))
    assert pair.min_distance == 2


@pytest.mark.parametrize("count", [2, 5, 12])
def test_pair_count_is_the_full_combination_when_all_match(count: int) -> None:
    fingerprints = make_fingerprints([(f"i{n:02d}", 0, 0) for n in range(count)])
    assert len(candidate_pairs(fingerprints)) == count * (count - 1) // 2
