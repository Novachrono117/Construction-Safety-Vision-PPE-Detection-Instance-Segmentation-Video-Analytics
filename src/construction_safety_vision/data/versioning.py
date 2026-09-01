"""Reasoning about provider dataset versions versus their source images.

A hosted dataset version may contain more images than the project it came from,
because offline augmentation generates variants. Treating those variants as
independent samples would corrupt every later split and metric, so this module
decides the question arithmetically from provider metadata rather than by
assumption.
"""

from __future__ import annotations

from typing import Any


def analyse_source_vs_generated(
    project_splits: dict[str, int],
    version_splits: dict[str, int],
    versions_per_image: int | None,
) -> dict[str, Any]:
    """Decide whether a version contains generated images beyond its source images.

    The conclusion is derived arithmetically rather than asserted: a split whose
    version count equals its source count multiplied by the declared number of
    augmented versions per image is generated; a split whose count is unchanged
    is not.

    Args:
        project_splits: Source image count per split, reported by the provider.
        version_splits: Version image count per split, reported by the provider.
        versions_per_image: Augmented versions produced per source image.

    Returns:
        A mapping describing the conclusion, the evidence and the resulting
        independent-image population.
    """
    shared = sorted(set(project_splits) & set(version_splits))
    per_split: dict[str, Any] = {}
    generated_splits: list[str] = []
    unchanged_splits: list[str] = []
    unexplained: list[str] = []

    for split in shared:
        source = project_splits[split]
        expanded = version_splits[split]
        entry: dict[str, Any] = {"source": source, "version": expanded}
        if expanded == source:
            entry["status"] = "unchanged"
            unchanged_splits.append(split)
        elif versions_per_image and source * versions_per_image == expanded:
            entry["status"] = "generated"
            entry["multiplier"] = versions_per_image
            generated_splits.append(split)
        else:
            entry["status"] = "unexplained"
            entry["ratio"] = round(expanded / source, 6) if source else None
            unexplained.append(split)
        per_split[split] = entry

    source_total = sum(project_splits.values())
    version_total = sum(version_splits.values())

    if unexplained:
        conclusion = "C_evidence_insufficient"
        summary = (
            f"Split(s) {unexplained} differ between project and version by a factor that the "
            "declared augmentation metadata does not explain."
        )
    elif generated_splits:
        conclusion = "B_source_plus_generated"
        summary = (
            f"Version contains source images plus offline-generated variants. Augmentation was "
            f"applied to {generated_splits} only ({versions_per_image} versions per source "
            f"image); {unchanged_splits} are unchanged. The {version_total} version images are "
            f"therefore NOT {version_total} independent samples: the independent source "
            f"population is {source_total}."
        )
    else:
        conclusion = "A_source_only"
        summary = "Every split matches its source count; the version contains no generated images."

    return {
        "conclusion": conclusion,
        "summary": summary,
        "per_split": per_split,
        "generated_splits": generated_splits,
        "unchanged_splits": unchanged_splits,
        "declared_versions_per_source_image": versions_per_image,
        "independent_source_images": source_total,
        "version_images": version_total,
        "independent_images_are_provider_reported": True,
    }
