"""Mapping the version-4 export back onto the 436 independent source images.

The export contains 742 image records. Comparing it against the provider's live
source project - to measure annotation drift, or to judge whether the frozen
snapshot is still a usable canonical source - requires knowing which export
record corresponds to which source image, and which of the two train records for
a source image is the one that was *not* augmented.

Two independent signals are used, in order.

**Identity** comes from the export's ``extra.name`` field, which carries the
original source filename. That is an exact match for 435 of the 436 images. The
exception is a filename containing non-ASCII characters, which the exporter
folds to ASCII while the source project stores it mojibake-encoded; a normalised
comparison bridges it, and the match is only accepted when it is unambiguous.

**Which representation is pristine** is not taken from the filename, from record
order, or from the export README - the README states that augmentation produced
two versions of each source image, which would mean no pristine render exists.
It is measured: the source original is re-rendered through the export's declared
preprocessing (auto-orient, then stretch to 640x640) and compared pixelwise
against each candidate. In practice one candidate per train pair reproduces the
deterministic render closely while the other does not, so the measurement
contradicts the README's wording. A pair is only resolved when that separation
is clear; anything else is reported as ambiguous rather than guessed.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from construction_safety_vision.paths import long_path

EXPORT_SIDE = 640
"""Side length the export's declared preprocessing resizes every image to."""

PRISTINE_MAE_MAX = 15.0
"""Largest mean absolute pixel error still consistent with a pristine render.

Generous relative to what is observed. The re-render cannot be bit-exact - the
provider's resampling filter and JPEG encoder are not reproduced here - so this
must absorb codec differences while staying far below an augmented render.
"""

SEPARATION_RATIO = 3.0
"""How many times worse the rejected candidate must be before a pair is resolved."""

EXACT = "exact_name"
"""Identity established by an exact ``extra.name`` match."""

NORMALISED = "normalised_name"
"""Identity established after Unicode normalisation and ASCII folding."""

UNMATCHED = "unmatched"
"""No defensible identity could be established."""

HIGH = "high"
"""The evidence separates the chosen representation clearly."""

AMBIGUOUS = "ambiguous"
"""The evidence does not separate the candidates; nothing is chosen."""


class MappingError(ValueError):
    """Raised when the mapping cannot be built as specified."""


def normalise_name(name: str) -> str:
    """Reduce a filename to a form comparable across encodings.

    The provider stores at least one filename mojibake-encoded while its exporter
    emits an ASCII-folded version of the same name. Decomposing, dropping
    combining marks and keeping only alphanumerics makes the two comparable
    without inventing a match between genuinely different files.

    Args:
        name: Filename as stored by either side.

    Returns:
        A lowercase alphanumeric key.
    """
    repaired = name
    try:
        # The source project holds this name as UTF-8 bytes that were decoded as
        # latin-1; reversing that recovers the real characters when it applies.
        candidate = name.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    else:
        repaired = candidate
    decomposed = unicodedata.normalize("NFKD", repaired)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return "".join(ch for ch in stripped.lower() if ch.isalnum())


@dataclass(frozen=True)
class V4Representation:
    """One image record inside the version-4 export.

    Attributes:
        split: Export split directory the record lives in.
        file_name: Export filename.
        coco_image_id: COCO ``id`` within that split's annotation document.
        source_name: The original filename from ``extra.name``.
        mae: Mean absolute pixel error against the deterministic re-render of
            the source original. ``None`` when it was not measured.
    """

    split: str
    file_name: str
    coco_image_id: int
    source_name: str
    mae: float | None = None


@dataclass
class SourceMapping:
    """How one source image maps into the version-4 export.

    Attributes:
        source_image_id: Provider source-image identifier.
        source_name: Original filename in the source project.
        candidates: Every export record carrying this source name.
        selected: The record judged to be the non-augmented representation.
        identity_method: How identity was established.
        confidence: Whether the representation choice is clearly separated.
        notes: Remarks, including why a choice was refused.
    """

    source_image_id: str
    source_name: str
    candidates: list[V4Representation]
    selected: V4Representation | None = None
    identity_method: str = UNMATCHED
    confidence: str = AMBIGUOUS
    notes: str = ""

    @property
    def is_augmented_selection(self) -> bool:
        """Whether the selected record looks augmented rather than pristine.

        Returns:
            ``True`` when a record was selected but does not meet the pristine
            threshold. Such a selection must never be treated as canonical.
        """
        return (
            self.selected is not None
            and self.selected.mae is not None
            and self.selected.mae > PRISTINE_MAE_MAX
        )

    def csv_row(self) -> dict[str, Any]:
        """Serialise for the committed mapping table.

        Returns:
            A mapping of column name to value. No local path is included.
        """
        return {
            "source_image_id": self.source_image_id,
            "source_name": self.source_name,
            "v4_split": self.selected.split if self.selected else "",
            "v4_image_id_or_filename": self.selected.file_name if self.selected else "",
            "v4_coco_image_id": self.selected.coco_image_id if self.selected else "",
            "mapping_method": self.identity_method,
            "mapping_confidence": self.confidence,
            "is_augmented": (
                "false" if self.selected and not self.is_augmented_selection else "true"
            ),
            "candidates": len(self.candidates),
            "selected_mae": (
                f"{self.selected.mae:.4f}"
                if self.selected and self.selected.mae is not None
                else ""
            ),
            "rejected_mae": ";".join(
                f"{c.mae:.4f}"
                for c in self.candidates
                if c is not self.selected and c.mae is not None
            ),
            "notes": self.notes,
        }


def preprocessed_render(path: Path, *, side: int = EXPORT_SIDE) -> np.ndarray:
    """Re-render a source original through the export's declared preprocessing.

    The export applies auto-orientation with EXIF stripping, then a stretch
    resize to a square. Both steps are deterministic, so the result is what an
    un-augmented export record should look like.

    Args:
        path: Source original on disk.
        side: Target side length.

    Returns:
        A ``float32`` RGB array of shape ``(side, side, 3)``.
    """
    with Image.open(long_path(path)) as handle:
        oriented = ImageOps.exif_transpose(handle)
        rgb = oriented.convert("RGB").resize((side, side), Image.BILINEAR)
        return np.asarray(rgb, dtype=np.float32)


def mean_absolute_error(left: np.ndarray, right: np.ndarray) -> float:
    """Mean absolute per-channel difference between two equally shaped images.

    Args:
        left: First image array.
        right: Second image array.

    Returns:
        The mean absolute error in 0-255 intensity units.

    Raises:
        MappingError: If the two arrays are not the same shape.
    """
    if left.shape != right.shape:
        msg = f"Cannot compare images of shape {left.shape} and {right.shape}"
        raise MappingError(msg)
    return float(np.abs(left - right).mean())


def choose_representation(
    candidates: list[V4Representation],
    *,
    pristine_max: float = PRISTINE_MAE_MAX,
    separation: float = SEPARATION_RATIO,
) -> tuple[V4Representation | None, str, str]:
    """Pick the non-augmented representation from a source image's candidates.

    Args:
        candidates: Export records sharing one source name, each already scored.
        pristine_max: Largest error still consistent with a pristine render.
        separation: How many times worse the runner-up must be to resolve a tie.

    Returns:
        The chosen record (or ``None``), the confidence label, and a note
        explaining the outcome.
    """
    scored = [c for c in candidates if c.mae is not None]
    if not scored:
        return None, AMBIGUOUS, "no candidate could be scored against the source original"

    ordered = sorted(scored, key=lambda c: c.mae or 0.0)
    best = ordered[0]
    if (best.mae or 0.0) > pristine_max:
        return (
            best,
            AMBIGUOUS,
            f"best candidate error {best.mae:.2f} exceeds the pristine threshold {pristine_max}",
        )
    if len(ordered) == 1:
        return best, HIGH, "single export record; matches the deterministic re-render"

    runner_up = ordered[1]
    if (runner_up.mae or 0.0) < (best.mae or 0.0) * separation:
        return (
            best,
            AMBIGUOUS,
            f"candidates not separated: {best.mae:.2f} vs {runner_up.mae:.2f}",
        )
    return (
        best,
        HIGH,
        f"pristine {best.mae:.2f} vs augmented {runner_up.mae:.2f}",
    )


def resolve_identity(
    source_names: dict[str, str],
    export_names: set[str],
) -> tuple[dict[str, tuple[str, str]], list[str], list[str]]:
    """Match source images to export source-names.

    Args:
        source_names: Source image id to original filename.
        export_names: Every distinct ``extra.name`` in the export.

    Returns:
        A mapping of source image id to ``(export name, method)``, the source
        ids that stayed unmatched, and the export names nothing claimed.

    Raises:
        MappingError: If a normalised key is ambiguous, which would make the
            fallback match a guess rather than a derivation.
    """
    resolved: dict[str, tuple[str, str]] = {}
    remaining_sources: dict[str, str] = {}
    for image_id, name in source_names.items():
        if name in export_names:
            resolved[image_id] = (name, EXACT)
        else:
            remaining_sources[image_id] = name

    claimed = {name for name, _ in resolved.values()}
    leftover_exports = sorted(export_names - claimed)
    if remaining_sources and leftover_exports:
        buckets: dict[str, list[str]] = {}
        for name in leftover_exports:
            buckets.setdefault(normalise_name(name), []).append(name)
        for image_id, name in sorted(remaining_sources.items()):
            options = buckets.get(normalise_name(name), [])
            if len(options) > 1:
                msg = (
                    f"Normalised name for {name!r} matches {len(options)} export records; "
                    "refusing to guess an identity"
                )
                raise MappingError(msg)
            if options:
                resolved[image_id] = (options[0], NORMALISED)

    claimed = {name for name, _ in resolved.values()}
    unmatched_sources = sorted(set(source_names) - set(resolved))
    unclaimed_exports = sorted(export_names - claimed)
    return resolved, unmatched_sources, unclaimed_exports
