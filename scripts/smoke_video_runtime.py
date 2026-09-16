"""Execute the predeclared minimal synthetic integration checks once; never a benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from construction_safety_vision.provenance import sha256_file
from construction_safety_vision.video_fixture import create_fixture
from construction_safety_vision.video_runtime import run_video, write_json


def main() -> int:
    """Generate one fixture and retain exact per-device execution provenance."""
    root = Path(__file__).resolve().parents[1]
    config_path = root / "configs/video_runtime_smoke.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if set(config) != {"schema_version", "purpose", "fixture", "executions"}:
        raise ValueError("Unknown smoke configuration field")
    directory = root / "outputs/phase13a_smoke"
    directory.mkdir(parents=True, exist_ok=True)
    ledger = directory / "execution.json"
    record: dict[str, Any] = {
        "status": "STARTED",
        "config": config,
        "config_sha256": sha256_file(config_path),
        "executions": [],
    }
    with ledger.open("x", encoding="utf-8") as handle:
        json.dump(record, handle)
    try:
        source = directory / "synthetic.mp4"
        record["fixture"] = create_fixture(source, **config["fixture"])
        write_json(ledger, record)
        for execution in config["executions"]:
            device = execution["device"]
            output = directory / f"compare_{device}.mp4"
            print(
                f"Engineering smoke: {device}, {execution['max_frames'] or 'all'} frames",
                flush=True,
            )
            result = run_video(
                source, output, root=root, attribution=record["fixture"]["attribution"], **execution
            )
            record["executions"].append(result)
            write_json(ledger, record)
        record["status"] = "COMPLETE"
    except BaseException:
        record["status"] = "FAILED"
        write_json(ledger, record)
        raise
    write_json(ledger, record)
    print("VIDEO_INTEGRATION_SMOKE_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
