"""Read-only checks of editorial claims against committed aggregate records."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

BASELINE = "a788d5303e70ddb12f5dc5ae35dd9a4b56fc6736"
REPORT = "academic/final_report.md"
CLAIMS = "academic/report_claims.json"
REQUIRED = {
    "d2_ap",
    "s1_box_ap",
    "s1_mask_ap",
    "matched_iou",
    "gt_iou",
    "coverage",
    "coverage50",
    "coverage75",
    "video_duration",
    "video_frames",
    "video_throughput",
    "d2_name",
    "d2_arch",
    "s1_name",
    "s1_arch",
    "test_confusion",
}
CLAIM_PATTERN = re.compile(r"<!-- claim:([a-z0-9_]+) -->(.*?)<!-- /claim -->", re.S)


def committed_bytes(root: Path, path: str) -> bytes:
    """Read a named, versioned report; never discover local model/data files."""
    if not path.startswith("reports/") or ".." in Path(path).parts:
        raise ValueError("Only explicit committed reports are accepted")
    return subprocess.check_output(["git", "show", f"{BASELINE}:{path}"], cwd=root)


def resolve(document: Any, pointer: str) -> Any:
    """Resolve an explicit JSON pointer without discovering other records."""
    for part in pointer.strip("/").split("/"):
        key = part.replace("~1", "/").replace("~0", "~")
        document = document[int(key)] if isinstance(document, list) else document[key]
    return document


def format_claim(value: Any, style: str) -> str:
    """Render stored values in Portuguese notation without recomputing metrics."""
    if style == "matrix":
        labels = ["HL", "HH", "P", "VL", "VB", "BG"]
        if len(value) != 6 or any(len(row) != 6 for row in value):
            raise ValueError("Expected the recorded six-label matrix")
        lines = [
            "| Prevista / verdadeira | " + " | ".join(labels) + " |",
            "| " + " | ".join(["---"] * 7) + " |",
        ]
        lines += [
            "| " + label + " | " + " | ".join(str(x) for x in row) + " |"
            for label, row in zip(labels, value, strict=True)
        ]
        return "\n".join(lines)
    if style == "text":
        return str(value)
    if style == "int":
        if int(value) != value:
            raise ValueError("Integer rendering would discard precision")
        return f"{int(value):,}".replace(",", ".")
    if style.startswith("decimal:"):
        return f"{value:.{int(style.split(':')[1])}f}".replace(".", ",")
    raise ValueError(f"Unknown claim style: {style}")


def expected_claims(root: Path, manifest: dict) -> dict[str, str]:
    """Verify source identities and render the declared aggregate fields."""
    if manifest["source_commit"] != BASELINE:
        raise ValueError("Scientific source baseline changed")
    sources = {}
    for path, digest in manifest["sources"].items():
        raw = committed_bytes(root, path)
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError(f"Source hash mismatch: {path}")
        if (root / path).read_bytes() != raw:
            raise ValueError(f"Historical aggregate modified: {path}")
        sources[path] = json.loads(raw)
    return {
        name: format_claim(resolve(sources[item["source"]], item["pointer"]), item["format"])
        for name, item in manifest["claims"].items()
    }


def validate(root: Path) -> dict:
    """Reject stale headline claims, changed evidence and prohibited figure paths."""
    manifest = json.loads((root / CLAIMS).read_text(encoding="utf-8"))
    expected = expected_claims(root, manifest)
    text = (root / REPORT).read_text(encoding="utf-8")
    seen = set()
    for match in CLAIM_PATTERN.finditer(text):
        name, actual = match.groups()
        if name not in expected or actual != expected[name]:
            raise ValueError(f"Report claim mismatch: {name}")
        seen.add(name)
    if seen != set(expected) or not seen >= REQUIRED:
        raise ValueError("Report is missing registered headline claims")
    if "{{" in text or "TBD" in text:
        raise ValueError("Unresolved report placeholder")
    images = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)
    for target in images:
        path = ((root / REPORT).parent / target).resolve()
        if not path.is_relative_to(root.resolve() / "reports/figures") or not path.is_file():
            raise ValueError(f"Invalid committed figure reference: {target}")
        if "final_test" in path.parts:
            raise ValueError("Test imagery is outside report scope")
        rel = path.relative_to(root.resolve()).as_posix()
        if path.read_bytes() != committed_bytes(root, rel):
            raise ValueError(f"Figure differs from committed evidence: {rel}")
    plain = CLAIM_PATTERN.sub(lambda m: m[2], text)
    plain = re.sub(r"<!--.*?-->", "", plain, flags=re.S)
    plain = re.sub(r"\]\([^)]*\)", "]", plain)
    return {
        "status": "PASS",
        "claims": len(seen),
        "claim_occurrences": len(CLAIM_PATTERN.findall(text)),
        "sources": len(manifest["sources"]),
        "figures": len(images),
        "word_count": len(re.findall(r"\b[\wÀ-ÿ]+(?:[-'][\wÀ-ÿ]+)*\b", plain)),
        "word_count_method": "visible Markdown tokens; includes tables, captions and references; "
        "excludes URLs/comments",
        "editorial_pages": text.count("<!-- pagebreak -->") + 1,
        "models_executed": 0,
        "holdout_accessed": False,
        "metrics_recomputed": 0,
    }
