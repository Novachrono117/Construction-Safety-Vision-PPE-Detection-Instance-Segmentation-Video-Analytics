"""Verify a supplied checkpoint file against public metadata; never load or download it."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from construction_safety_vision.checkpoint_delivery import (
    MANIFEST_PATH,
    build_manifest,
    load_manifest,
    verify_file,
)


def main() -> int:
    """Validate metadata identity first, then verify only the explicitly supplied file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("D2", "S1"), required=True)
    parser.add_argument("--path", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        manifest = load_manifest(root / MANIFEST_PATH)
        if manifest != build_manifest(root):
            raise ValueError("public metadata differs from the authoritative freeze")
        entry = next(x for x in manifest["checkpoints"] if x["logical_name"] == args.model)
        verify_file(args.path, sha256=entry["sha256"], size_bytes=entry["bytes"])
    except (OSError, ValueError) as exc:
        # Do not echo local file paths or a supplied locator into public logs.
        print(f"CHECKPOINT_VERIFICATION_FAILED: {type(exc).__name__}", file=sys.stderr)
        return 1
    print(f"CHECKPOINT_BYTES_VERIFIED: {args.model}; no model loaded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
