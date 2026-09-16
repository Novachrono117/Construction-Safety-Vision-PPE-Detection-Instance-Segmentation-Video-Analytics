"""Validate Phase 12C metadata; optionally verify validation cache/image evidence, never infer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from construction_safety_vision.provenance import sha256_file
from construction_safety_vision.qualitative_gallery import (
    CONFIG,
    PROVENANCE,
    REPORT,
    choose_examples,
    choose_hero,
    choose_masks,
    load_policy,
)
from construction_safety_vision.qualitative_gallery_report import validate_gallery


def main() -> int:
    """Validate before declaring a gallery complete."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-data", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    problems = validate_gallery(root)
    if args.with_data:
        from build_qualitative_gallery import analyze, load_predictions, population

        policy = load_policy(root / CONFIG)
        images, truth = population(root)
        predictions, _execution = load_predictions(root, images, policy)
        records = analyze(images, truth, predictions)
        payload = json.loads((root / REPORT).read_text(encoding="utf-8"))
        provenance = json.loads((root / PROVENANCE).read_text(encoding="utf-8"))
        allowed_data = {image["path"] for image in images.values()} | {
            f"data/processed/canonical/annotations/{task}_validation.coco.json"
            for task in ("detection", "segmentation")
        }
        allowed_caches = {
            f"artifacts/{directory}/{filename}"
            for directory in ("qualitative_validation", "qualitative_validation_batching_deviation")
            for filename in (
                "D2.json",
                "S1.json",
                "execution.json",
                "initial_source.py",
                "accepted_inference_source.py",
            )
        }
        for entry in provenance["inputs"]:
            relative = entry["path"]
            if relative.startswith(("data/", "artifacts/")):
                if relative not in allowed_data | allowed_caches:
                    raise ValueError("Provenance contains a nonvalidation data/cache path")
                if sha256_file(root / relative) != entry["sha256"]:
                    problems.append(f"Local evidence digest mismatch: {relative}")
        if {m: choose_examples(r) for m, r in records.items()} != payload["selections"]:
            problems.append("Cache-derived object selection mismatch")
        if choose_masks(records["S1"]) != payload["mask_selections"]:
            problems.append("Cache-derived mask selection mismatch")
        if choose_hero(records, images, policy["hero"]) != payload["hero"]:
            problems.append("Cache-derived hero selection mismatch")
    for problem in problems:
        print(problem)
    print("GALLERY_VALIDATION_FAILED" if problems else "QUALITATIVE_GALLERY_VALIDATED")
    return int(bool(problems))


if __name__ == "__main__":
    raise SystemExit(main())
