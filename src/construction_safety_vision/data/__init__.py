"""Dataset acquisition and structural inspection.

Phase 3 scope: obtain the canonical instance-segmentation export from the
provider, record its provenance, and verify that it is internally consistent.
No annotation is modified, no split is created and no image is analysed here.
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
