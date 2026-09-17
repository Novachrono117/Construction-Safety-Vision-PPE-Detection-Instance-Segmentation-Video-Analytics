"""Validate the academic report without model or dataset access."""

import json
from pathlib import Path

from construction_safety_vision.academic_report import validate

if __name__ == "__main__":
    print(json.dumps(validate(Path(__file__).resolve().parents[1]), indent=2))
