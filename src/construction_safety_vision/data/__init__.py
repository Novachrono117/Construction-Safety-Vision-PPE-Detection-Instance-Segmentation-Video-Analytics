"""Dataset acquisition, inspection, audit and canonical-snapshot resolution.

Nothing in this package modifies an annotation, creates a split or trains a
model. It obtains what the provider holds, measures it, and records what was
decided about it.

Re-exported below is the acquisition and inspection surface established in phase
3, which other packages call directly. The later modules are imported by their
own path rather than re-exported here, because they are used by a small number
of scripts and flattening them would say less about where each belongs:

* ``fingerprint``, ``imagestats``, ``source`` - the phase 4A source audit;
* ``geometry`` - boxes derived from segmentation, used by every later phase;
* ``manualaudit`` - the phase 4B human decisions;
* ``sourcegeometry`` - complete live geometry, recovered and verified (5A);
* ``v4mapping`` - source images to their non-augmented export representation (5A);
* ``drift`` - live annotations versus the version-4 snapshot (5A);
* ``canonical`` - the canonical-snapshot decision and its manifest (5A).
"""

from __future__ import annotations

from construction_safety_vision.data.acquisition import (
    AcquisitionError,
    DownloadError,
    DownloadResult,
    HashMismatchError,
    UnsafeArchiveError,
    download_if_absent,
    safe_extract_zip,
    stream_download,
)
from construction_safety_vision.data.coco import (
    CocoValidationError,
    SplitInspection,
    classify_segmentation,
    inspect_export,
    inspect_split,
    load_coco_document,
)
from construction_safety_vision.data.roboflow import (
    API_KEY_ENV_VAR,
    AuthenticationError,
    DatasetCoordinates,
    ExportNotReadyError,
    MissingApiKeyError,
    RoboflowClient,
    RoboflowError,
    api_key_from_env,
)
from construction_safety_vision.data.versioning import analyse_source_vs_generated

__all__ = [
    "API_KEY_ENV_VAR",
    "AcquisitionError",
    "AuthenticationError",
    "CocoValidationError",
    "DatasetCoordinates",
    "DownloadError",
    "DownloadResult",
    "ExportNotReadyError",
    "HashMismatchError",
    "MissingApiKeyError",
    "RoboflowClient",
    "RoboflowError",
    "SplitInspection",
    "UnsafeArchiveError",
    "analyse_source_vs_generated",
    "api_key_from_env",
    "classify_segmentation",
    "download_if_absent",
    "inspect_export",
    "inspect_split",
    "load_coco_document",
    "safe_extract_zip",
    "stream_download",
]
