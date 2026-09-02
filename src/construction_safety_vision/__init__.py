"""Construction Safety Vision: PPE detection, instance segmentation and video analytics.

Foundation phase. This package currently provides only the primitives that the
experimental protocol depends on:

* :mod:`construction_safety_vision.paths` - layout resolution for local and Colab runs;
* :mod:`construction_safety_vision.config` - strict, typed experiment configuration;
* :mod:`construction_safety_vision.splits` - split identifiers and the holdout guard;
* :mod:`construction_safety_vision.provenance` - artifact hashing and run records.

Dataset, training, evaluation and video modules are added by their respective
roadmap phases (see ``reports/roadmap.md``).
"""

from __future__ import annotations

from construction_safety_vision.config import (
    ConfigError,
    DatasetCandidate,
    ExperimentConfig,
    SplitRatios,
    load_experiment_config,
)
from construction_safety_vision.env import load_project_env
from construction_safety_vision.paths import ProjectPaths, find_project_root, in_colab
from construction_safety_vision.provenance import (
    ArtifactRecord,
    ProvenanceRecord,
    sha256_file,
)
from construction_safety_vision.splits import (
    DEVELOPMENT_SPLITS,
    PROTECTED_SPLITS,
    HoldoutViolationError,
    Split,
    assert_split_allowed,
)

__version__ = "0.1.0"

__all__ = [
    "DEVELOPMENT_SPLITS",
    "PROTECTED_SPLITS",
    "ArtifactRecord",
    "ConfigError",
    "DatasetCandidate",
    "ExperimentConfig",
    "HoldoutViolationError",
    "ProjectPaths",
    "ProvenanceRecord",
    "Split",
    "SplitRatios",
    "__version__",
    "assert_split_allowed",
    "find_project_root",
    "in_colab",
    "load_experiment_config",
    "load_project_env",
    "sha256_file",
]
