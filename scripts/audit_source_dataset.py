"""Audit the 436 original source images: duplicates, leakage risk, split quality.

Everything here operates on the **436 independent source images**, never on the
742-image version-4 export, which contains two augmented copies of every training
image.

This is candidate generation and measurement. Nothing is deleted, regrouped or
re-split, and a perceptual-hash match is reported as a *candidate* for human
review, never as established leakage.

Writes:
    reports/exact_duplicate_groups.csv     (only when duplicates exist)
    reports/near_duplicate_candidates.csv
    reports/group_candidates.csv           (only when evidence exists)
    reports/empty_image_audit.csv
    reports/dataset_audit_report.md
    reports/dataset_audit.json

Usage:
    uv run python scripts/audit_source_dataset.py
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from construction_safety_vision.data.coco import load_coco_document
from construction_safety_vision.data.fingerprint import (
    DHASH_THRESHOLD,
    PHASH_THRESHOLD,
    CandidatePair,
    candidate_pairs,
    fingerprint_file,
)
from construction_safety_vision.paths import ProjectPaths
from construction_safety_vision.provenance import SCHEMA_VERSION, git_commit

PHASE = 4
"""Roadmap phase this script belongs to."""

EXPECTED_SOURCE_IMAGES = 436
"""Source population established in phase 3."""

STRONG_PAIR_LIMIT = 40
"""Strongest candidate pairs carried into the visual-review package."""

# Provider filenames come from stock-photo sources. A shared stem prefix is weak
# evidence of a related capture; it is recorded as a candidate, never as a group.
STEM_PREFIX = re.compile(r"^([A-Za-z]+[-_])")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read newline-delimited JSON.

    Args:
        path: File to read.

    Returns:
        The parsed rows, or an empty list when the file is absent.
    """
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    """Write rows as CSV with LF endings.

    Args:
        path: Destination file.
        fieldnames: Column order.
        rows: Rows to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def audit_exact_duplicates(stats: list[dict[str, Any]]) -> dict[str, Any]:
    """Group source images by content hash.

    Args:
        stats: Per-image measurements including ``sha256`` and ``split``.

    Returns:
        A summary with the duplicate groups and whether any crosses a split.
    """
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in stats:
        by_hash[row["sha256"]].append(row)
    groups = [members for members in by_hash.values() if len(members) > 1]
    cross_split = [g for g in groups if len({m["split"] for m in g}) > 1]
    return {
        "images": len(stats),
        "unique_hashes": len(by_hash),
        "duplicate_groups": len(groups),
        "images_in_duplicate_groups": sum(len(g) for g in groups),
        "cross_split_groups": len(cross_split),
        "groups": groups,
    }


def audit_near_duplicates(
    image_dir: Path, stats: list[dict[str, Any]]
) -> tuple[list[CandidatePair], dict[str, Any]]:
    """Generate near-duplicate candidates over the source originals.

    Args:
        image_dir: Directory holding the original images.
        stats: Per-image measurements, used for the split of each image.

    Returns:
        The candidate pairs and a summary keyed by same/cross split.
    """
    fingerprints = []
    for row in stats:
        if not row.get("decoded"):
            continue
        path = image_dir / f"{row['image_id']}.jpg"
        if path.is_file():
            fingerprints.append(fingerprint_file(row["image_id"], path))
    pairs = candidate_pairs(fingerprints)
    split_of = {row["image_id"]: row["split"] for row in stats}
    cross = [p for p in pairs if split_of.get(p.image_a) != split_of.get(p.image_b)]
    return pairs, {
        "images_fingerprinted": len(fingerprints),
        "pairs_compared": len(fingerprints) * (len(fingerprints) - 1) // 2,
        "candidates": len(pairs),
        "same_split_candidates": len(pairs) - len(cross),
        "cross_split_candidates": len(cross),
        "dhash_threshold": DHASH_THRESHOLD,
        "phash_threshold": PHASH_THRESHOLD,
    }


def audit_sequences(
    manifest: list[dict[str, Any]], pairs: list[CandidatePair]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Look for evidence that images come from sequences or shared captures.

    Two weak signals are combined, and both are reported as candidates:
    a shared filename prefix, and membership of a chain of near-duplicates.

    Args:
        manifest: Source image records.
        pairs: Near-duplicate candidates.

    Returns:
        Candidate group rows and a summary.
    """
    split_of = {m["image_id"]: m["split"] for m in manifest}
    name_of = {m["image_id"]: m["name"] for m in manifest}

    # Connected components over the near-duplicate candidate graph.
    parent: dict[str, str] = {m["image_id"]: m["image_id"] for m in manifest}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for pair in pairs:
        a, b = find(pair.image_a), find(pair.image_b)
        if a != b:
            parent[a] = b

    chains: dict[str, list[str]] = defaultdict(list)
    for image_id in parent:
        chains[find(image_id)].append(image_id)
    multi = {k: sorted(v) for k, v in chains.items() if len(v) > 1}

    prefixes: dict[str, list[str]] = defaultdict(list)
    for record in manifest:
        match = STEM_PREFIX.match(record["name"])
        if match:
            prefixes[match.group(1)].append(record["image_id"])

    rows: list[dict[str, Any]] = []
    for index, (_, members) in enumerate(sorted(multi.items()), start=1):
        splits = sorted({split_of.get(m, "?") for m in members})
        rows.append(
            {
                "group_id": f"chain-{index:03d}",
                "evidence": "near_duplicate_chain",
                "confidence": "candidate",
                "size": len(members),
                "splits": "|".join(splits),
                "crosses_split": len(splits) > 1,
                "image_ids": "|".join(members),
                "names": "|".join(name_of.get(m, "") for m in members),
            }
        )
    return rows, {
        "near_duplicate_chains": len(multi),
        "images_in_chains": sum(len(v) for v in multi.values()),
        "chains_crossing_splits": sum(1 for r in rows if r["crosses_split"]),
        "filename_prefixes": {k: len(v) for k, v in sorted(prefixes.items())},
    }


def audit_provider_split(
    manifest: list[dict[str, Any]],
    duplicates: dict[str, Any],
    near: dict[str, Any],
    sequences: dict[str, Any],
    cross_pairs: list[CandidatePair],
) -> dict[str, Any]:
    """Assess the provider's split without modifying it.

    Args:
        manifest: Source image records.
        duplicates: Exact-duplicate summary.
        near: Near-duplicate summary.
        sequences: Sequence-candidate summary.
        cross_pairs: Cross-split near-duplicate candidates.

    Returns:
        Per-split statistics, the concerns found, and a classification.
    """
    per_split: dict[str, dict[str, Any]] = {}
    for record in manifest:
        entry = per_split.setdefault(
            record["split"],
            {
                "images": 0,
                "instances": 0,
                "negatives": 0,
                "images_with_class": Counter(),
                "instances_per_class": Counter(),
            },
        )
        entry["images"] += 1
        entry["instances"] += record["annotation_count"]
        if record["annotation_count"] == 0:
            entry["negatives"] += 1
        for label, count in record["class_counts"].items():
            entry["images_with_class"][label] += 1
            entry["instances_per_class"][label] += count

    classes = sorted({c for m in manifest for c in m["class_counts"]})
    concerns: list[str] = []
    for split, entry in sorted(per_split.items()):
        missing = [c for c in classes if entry["instances_per_class"].get(c, 0) == 0]
        if missing:
            concerns.append(f"split {split!r} has zero instances of {missing}")
    if duplicates["cross_split_groups"]:
        concerns.append(
            f"{duplicates['cross_split_groups']} exact-duplicate group(s) cross a split boundary"
        )
    if cross_pairs:
        concerns.append(
            f"{len(cross_pairs)} near-duplicate candidate pair(s) cross a split boundary "
            "(candidates, not confirmed leakage)"
        )
    if sequences["chains_crossing_splits"]:
        concerns.append(
            f"{sequences['chains_crossing_splits']} near-duplicate chain(s) span "
            "more than one split"
        )

    if duplicates["cross_split_groups"]:
        classification = "UNSUITABLE_FOR_FINAL_PROTOCOL"
    elif concerns:
        classification = "UNDETERMINED_PENDING_VISUAL_REVIEW"
    else:
        classification = "USABLE_BUT_SUBOPTIMAL"

    return {
        "per_split": {
            split: {
                "images": entry["images"],
                "instances": entry["instances"],
                "negatives": entry["negatives"],
                "images_with_class": dict(sorted(entry["images_with_class"].items())),
                "instances_per_class": dict(sorted(entry["instances_per_class"].items())),
            }
            for split, entry in sorted(per_split.items())
        },
        "classes": classes,
        "concerns": concerns,
        "classification": classification,
        "near_duplicate_summary": near,
    }


def audit_source_vs_export(paths: ProjectPaths, manifest: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare the live source annotations against the frozen v4 export.

    The export is a snapshot. If the provider's project has been edited since it
    was generated, the two disagree - and phase 5 must then decide explicitly
    which one is canonical instead of assuming they are the same data.

    Args:
        paths: Project layout.
        manifest: Source image records.

    Returns:
        Per-split comparison and whether the two agree.
    """
    export_root = paths.data_raw / "construction-ppe-compliance-detection-v4-coco-segmentation"
    augmentation_factor = {"train": 2, "valid": 1, "test": 1}
    rows: dict[str, Any] = {}
    agree = True
    for split in ("train", "valid", "test"):
        source_instances = sum(m["annotation_count"] for m in manifest if m["split"] == split)
        path = export_root / split / "_annotations.coco.json"
        if not path.is_file():
            continue
        document = load_coco_document(path)
        export_instances = len(document["annotations"])
        factor = augmentation_factor[split]
        per_copy = export_instances / factor
        rows[split] = {
            "source_instances": source_instances,
            "export_instances": export_instances,
            "export_augmentation_factor": factor,
            "export_instances_per_source_copy": per_copy,
            "difference": round(source_instances - per_copy, 2),
            "agrees": abs(source_instances - per_copy) < 0.5,
        }
        agree = agree and rows[split]["agrees"]
    return {
        "per_split": rows,
        "source_and_export_agree": agree,
        "interpretation": (
            "The live source project and the frozen v4 export describe different "
            "annotation counts. The export is a snapshot taken before later edits, "
            "so the two are not interchangeable. Phase 5 must state which is "
            "canonical; this audit does not decide it."
            if not agree
            else "Source and export annotation counts agree."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    """Run the source audit.

    Args:
        argv: Command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="Audit the original source images.")
    parser.parse_args(argv)

    paths = ProjectPaths.from_root()
    manifest = read_jsonl(paths.reports / "source_image_manifest.jsonl")
    stats = read_jsonl(paths.data_interim / "source_image_stats.jsonl")
    annotations = read_jsonl(paths.data_interim / "source_annotations.jsonl")
    if not manifest or not stats:
        print(
            "ERROR: run scripts/fetch_source_inventory.py and "
            "scripts/download_source_images.py first.",
            file=sys.stderr,
        )
        return 2

    image_dir = paths.data_external / "source_images"
    print(f"auditing {len(manifest)} source images ...", flush=True)

    duplicates = audit_exact_duplicates(stats)
    print(f"  exact duplicates: {duplicates['duplicate_groups']} group(s)", flush=True)

    pairs, near = audit_near_duplicates(image_dir, stats)
    print(f"  near-duplicate candidates: {near['candidates']}", flush=True)

    split_of = {row["image_id"]: row["split"] for row in stats}
    name_of = {m["image_id"]: m["name"] for m in manifest}
    cross_pairs = [p for p in pairs if split_of.get(p.image_a) != split_of.get(p.image_b)]

    sequence_rows, sequences = audit_sequences(manifest, pairs)
    print(f"  near-duplicate chains: {sequences['near_duplicate_chains']}", flush=True)

    split_audit = audit_provider_split(manifest, duplicates, near, sequences, cross_pairs)
    print(f"  provider split: {split_audit['classification']}", flush=True)

    # ---- artifacts -------------------------------------------------------
    if duplicates["groups"]:
        rows = []
        for index, group in enumerate(
            sorted(duplicates["groups"], key=lambda g: g[0]["sha256"]), 1
        ):
            for member in sorted(group, key=lambda m: m["image_id"]):
                rows.append(
                    {
                        "group_id": f"dup-{index:03d}",
                        "sha256": member["sha256"],
                        "image_id": member["image_id"],
                        "name": member["name"],
                        "split": member["split"],
                        "size_bytes": member["size_bytes"],
                    }
                )
        write_csv(
            paths.reports / "exact_duplicate_groups.csv",
            ["group_id", "sha256", "image_id", "name", "split", "size_bytes"],
            rows,
        )

    write_csv(
        paths.reports / "near_duplicate_candidates.csv",
        [
            "image_a_id",
            "image_b_id",
            "provider_split_a",
            "provider_split_b",
            "dhash_distance",
            "phash_distance",
            "min_distance",
            "cross_split",
            "reason",
        ],
        [
            {
                "image_a_id": p.image_a,
                "image_b_id": p.image_b,
                "provider_split_a": split_of.get(p.image_a, "?"),
                "provider_split_b": split_of.get(p.image_b, "?"),
                "dhash_distance": p.dhash_distance,
                "phash_distance": p.phash_distance,
                "min_distance": p.min_distance,
                "cross_split": split_of.get(p.image_a) != split_of.get(p.image_b),
                "reason": "+".join(p.reasons),
            }
            for p in pairs
        ],
    )

    if sequence_rows:
        write_csv(
            paths.reports / "group_candidates.csv",
            [
                "group_id",
                "evidence",
                "confidence",
                "size",
                "splits",
                "crosses_split",
                "image_ids",
                "names",
            ],
            sequence_rows,
        )

    empty_images = [m for m in manifest if m["annotation_count"] == 0]
    write_csv(
        paths.reports / "empty_image_audit.csv",
        ["image_id", "name", "split", "width", "height", "requires_visual_review", "note"],
        [
            {
                "image_id": m["image_id"],
                "name": m["name"],
                "split": m["split"],
                "width": m["width"],
                "height": m["height"],
                "requires_visual_review": True,
                "note": (
                    "zero annotated instances; deliberate negative vs missing label is unresolved"
                ),
            }
            for m in sorted(empty_images, key=lambda m: m["image_id"])
        ],
    )

    geometry = Counter(a["geometry_type"] for a in annotations)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "report": "source_dataset_audit",
        "phase": PHASE,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(paths.root),
        "population": {
            "source_images": len(manifest),
            "expected": EXPECTED_SOURCE_IMAGES,
            "instances": sum(m["annotation_count"] for m in manifest),
            "splits": dict(Counter(m["split"] for m in manifest)),
            "note": "The 742-image v4 export is NOT this population; it augments train x2.",
        },
        "exact_duplicates": {k: v for k, v in duplicates.items() if k != "groups"},
        "near_duplicates": near,
        "sequences": sequences,
        "zero_instance_images": {
            "count": len(empty_images),
            "by_split": dict(Counter(m["split"] for m in empty_images)),
            "requires_visual_review": True,
        },
        "source_geometry_types": dict(sorted(geometry.items())),
        "source_vs_export_drift": audit_source_vs_export(paths, manifest),
        "provider_split_audit": split_audit,
        "strongest_cross_split_pairs": [
            {
                "image_a": p.image_a,
                "image_b": p.image_b,
                "name_a": name_of.get(p.image_a, ""),
                "name_b": name_of.get(p.image_b, ""),
                "split_a": split_of.get(p.image_a, "?"),
                "split_b": split_of.get(p.image_b, "?"),
                "dhash": p.dhash_distance,
                "phash": p.phash_distance,
            }
            for p in cross_pairs[:STRONG_PAIR_LIMIT]
        ],
    }
    (paths.reports / "dataset_audit.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    print("wrote reports/dataset_audit.json and the audit CSVs", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
