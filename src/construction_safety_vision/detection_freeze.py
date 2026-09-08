"""The frozen final detector: its identity, and the checks that protect it.

Phase 7D selects one of three experiments and freezes it. From that point the
project has *a* detector rather than three candidates, and the failure mode this
module exists to prevent is quiet substitution: a later phase reaching for
``artifacts/detection/D0/weights/best.pt`` because that path was in an older
notebook, or for ``last.pt`` because it sits next to ``best.pt`` and loads just
as happily, or for a re-trained checkpoint that landed at the selected path
after someone re-ran a cell. None of those raise on their own. A YOLO checkpoint
loads whatever bytes it is given.

So the frozen detector is identified by digest, never by path, and
:func:`load_checkpoint_path` refuses rather than warns: a missing file, a digest
that does not match, or a manifest whose recorded fingerprint disagrees with the
fingerprint recomputed from its own contents all stop the caller.

Three things this module deliberately does not do.

**It never trains and never downloads.** A missing checkpoint is
:data:`BLOCKED_MISSING_MODEL_ARTIFACT` and the caller is told to obtain the
artifact, because silently re-training would produce different weights under the
selected experiment's name - the exact substitution the module exists to catch.

**It never reads the holdout.** The freeze is a selection decision made on
validation evidence, and nothing here needs a test image, a test label or a test
id. :func:`parse_final_detector` rejects a manifest that carries anything about
the protected split beyond the standard protection notice.

**It never re-derives a metric.** Every number it exposes was read from a
committed result manifest by the phase that wrote the freeze. This module
transports and verifies identity; it does not measure.

The semantic fingerprint, :func:`final_detector_fingerprint`, covers what would
make the selected model a *different* model: which experiment won, its family
and input resolution, the checkpoint bytes, the experiment's own fingerprint,
the policy that chose it, and the data it was chosen on. It deliberately
excludes timestamps, absolute paths and anything machine-specific, so two
machines that hold the same frozen detector compute the same value.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from construction_safety_vision.provenance import sha256_file

__all__ = [
    "BLOCKED_MISSING_MODEL_ARTIFACT",
    "FINAL_DETECTOR_MANIFEST",
    "FINGERPRINT_FIELDS",
    "FROZEN",
    "MODEL_ARTIFACT_MISMATCH",
    "SCHEMA_VERSION",
    "CheckpointMismatchError",
    "CheckpointMissingError",
    "FinalDetector",
    "FinalDetectorError",
    "final_detector_fingerprint",
    "holdout_leaks",
    "load_checkpoint_path",
    "load_final_detector",
    "parse_final_detector",
]

#: Version of the final-detector manifest format.
SCHEMA_VERSION = 1

#: Committed artifact this module reads.
FINAL_DETECTOR_MANIFEST = "final_detector_manifest.json"

#: The only status under which a detector counts as frozen.
FROZEN = "FROZEN"

#: Classification used when the selected checkpoint is absent from the machine.
BLOCKED_MISSING_MODEL_ARTIFACT = "BLOCKED_MISSING_MODEL_ARTIFACT"

#: Classification used when a checkpoint is present but is not the selected one.
MODEL_ARTIFACT_MISMATCH = "MODEL_ARTIFACT_MISMATCH"

#: The protected split. It has no part in the freeze and may not be named.
FORBIDDEN_SPLIT = "test"

#: Holdout status a compliant manifest carries.
HOLDOUT_STATUS = "PROTECTED_NOT_ACCESSED"

#: Fields the semantic fingerprint is computed over, in a fixed order.
#:
#: Each one, if it changed, would mean a different selected model. Order is
#: fixed here rather than taken from a mapping's iteration order so that the
#: digest is a property of the values and not of how they were assembled.
FINGERPRINT_FIELDS: tuple[str, ...] = (
    "selected_experiment",
    "model",
    "imgsz",
    "selected_checkpoint_sha256",
    "experiment_sha256",
    "phase7_policy_sha256",
    "split_assignment_sha256",
    "adapter_config_sha256",
    "class_map_sha256",
)

#: Required top-level keys of the manifest.
REQUIRED_FIELDS: tuple[str, ...] = (
    "schema_version",
    "status",
    "task",
    "selected_experiment",
    "selection_method",
    "policy_case",
    "model",
    "imgsz",
    "batch",
    "selected_checkpoint",
    "experiment_sha256",
    "phase7_policy_sha256",
    "dataset_fingerprints",
    "split_reference",
    "primary_selection_metric",
    "practical_equivalence_margin",
    "binary_distribution_status",
    "final_detector_sha256",
    "test",
)

#: Required keys of the ``selected_checkpoint`` block.
CHECKPOINT_FIELDS: tuple[str, ...] = ("relative_path", "sha256", "size_bytes", "committed")

#: Checkpoint file names that must never be accepted as the frozen detector.
#:
#: ``last.pt`` is the final epoch's weights, not the checkpoint the predeclared
#: rule selected. It sits in the same directory, has the same size, and loads
#: without complaint - which is precisely why it is named here.
REJECTED_CHECKPOINT_NAMES: frozenset[str] = frozenset({"last.pt"})


class FinalDetectorError(RuntimeError):
    """Raised when the frozen detector's identity cannot be established."""


class CheckpointMissingError(FinalDetectorError):
    """Raised when the selected checkpoint is not present on this machine."""


class CheckpointMismatchError(FinalDetectorError):
    """Raised when a checkpoint's bytes are not the selected checkpoint's."""


def _digest_text(text: str) -> str:
    """Hash a string with SHA-256.

    Args:
        text: Content to hash.

    Returns:
        The hex digest.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def final_detector_fingerprint(values: Mapping[str, Any]) -> str:
    """Compute the semantic fingerprint of a frozen detector's identity.

    The digest is taken over a canonical rendering of
    :data:`FINGERPRINT_FIELDS` - field name, then value, one per line - rather
    than over the manifest as a whole. That is the point: adding a prose note or
    a provenance timestamp to the manifest must not change the fingerprint,
    while swapping the checkpoint must.

    Args:
        values: A mapping carrying every field in :data:`FINGERPRINT_FIELDS`.

    Returns:
        The hex digest.

    Raises:
        FinalDetectorError: If a fingerprint field is missing or empty. A
            fingerprint computed over a partial identity would be a fingerprint
            of something other than the selected model.
    """
    missing = [field for field in FINGERPRINT_FIELDS if values.get(field) in (None, "")]
    if missing:
        msg = f"cannot fingerprint the final detector: missing identity field(s) {missing}"
        raise FinalDetectorError(msg)
    lines = [f"{field}={values[field]}" for field in FINGERPRINT_FIELDS]
    return _digest_text("\n".join(lines) + "\n")


def _fingerprint_values(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Collect the fingerprint inputs from a parsed manifest.

    Args:
        manifest: The manifest mapping.

    Returns:
        The fingerprint field values.
    """
    checkpoint = manifest.get("selected_checkpoint")
    checkpoint = checkpoint if isinstance(checkpoint, Mapping) else {}
    fingerprints = manifest.get("dataset_fingerprints")
    fingerprints = fingerprints if isinstance(fingerprints, Mapping) else {}
    split = manifest.get("split_reference")
    split = split if isinstance(split, Mapping) else {}
    return {
        "selected_experiment": manifest.get("selected_experiment"),
        "model": manifest.get("model"),
        "imgsz": manifest.get("imgsz"),
        "selected_checkpoint_sha256": checkpoint.get("sha256"),
        "experiment_sha256": manifest.get("experiment_sha256"),
        "phase7_policy_sha256": manifest.get("phase7_policy_sha256"),
        "split_assignment_sha256": split.get("split_assignment_sha256"),
        "adapter_config_sha256": fingerprints.get("adapter_config_sha256"),
        "class_map_sha256": fingerprints.get("class_map_sha256"),
    }


def holdout_leaks(payload: Any, *, path: str = "") -> list[str]:
    """Find every place a manifest names the protected split.

    One shape is allowed: a mapping under ``test`` carrying a ``status`` of
    :data:`HOLDOUT_STATUS`. That is how a phase records that it did not touch
    the holdout, and it is evidence worth keeping. A holdout id, count or metric
    is not.

    Args:
        payload: Any parsed value.
        path: Dotted location, used during recursion.

    Returns:
        One dotted path per leak, empty when clean.
    """
    leaks: list[str] = []
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            here = f"{path}.{key}" if path else str(key)
            if isinstance(key, str) and key.strip().lower() == FORBIDDEN_SPLIT:
                if not (isinstance(value, Mapping) and value.get("status") == HOLDOUT_STATUS):
                    leaks.append(here)
                continue
            leaks.extend(holdout_leaks(value, path=here))
    elif isinstance(payload, (list, tuple)):
        for index, item in enumerate(payload):
            leaks.extend(holdout_leaks(item, path=f"{path}[{index}]"))
    elif isinstance(payload, str) and payload.strip().lower() == FORBIDDEN_SPLIT:
        leaks.append(path)
    return leaks


@dataclass(frozen=True)
class FinalDetector:
    """The frozen detector's identity, as committed.

    Attributes:
        selected_experiment: The experiment id the policy elected.
        model: Model family and size.
        imgsz: Model input size.
        batch: Training batch size.
        checkpoint_relative_path: Where the run wrote the selected checkpoint,
            relative to the repository root.
        frozen_copy_relative_path: Where the immutable copy lives, relative to
            the repository root, or ``None`` when none was recorded.
        checkpoint_sha256: The selected checkpoint's digest.
        checkpoint_size_bytes: The selected checkpoint's size.
        experiment_sha256: The selected experiment's fingerprint.
        policy_sha256: The frozen Phase 7 policy's digest.
        split_assignment_sha256: The frozen split's digest.
        dataset_fingerprints: The adapter, class-map and population digests.
        policy_case: The selection case the frozen logic produced.
        selection_method: How the decision was reached.
        binary_distribution_status: How the weights are distributed.
        fingerprint: The recorded semantic fingerprint.
        manifest: The full parsed manifest.
    """

    selected_experiment: str
    model: str
    imgsz: int
    batch: int
    checkpoint_relative_path: str
    frozen_copy_relative_path: str | None
    checkpoint_sha256: str
    checkpoint_size_bytes: int
    experiment_sha256: str
    policy_sha256: str
    split_assignment_sha256: str
    dataset_fingerprints: dict[str, Any]
    policy_case: str
    selection_method: str
    binary_distribution_status: str
    fingerprint: str
    manifest: dict[str, Any]

    def recompute_fingerprint(self) -> str:
        """Recompute the semantic fingerprint from the manifest's own contents.

        Returns:
            The hex digest.
        """
        return final_detector_fingerprint(_fingerprint_values(self.manifest))

    def candidate_paths(self, root: str | Path) -> tuple[Path, ...]:
        """Return the places the selected checkpoint may live, in order.

        The immutable frozen copy is preferred over the training run's output
        directory, because the run directory is where a re-run would land.

        Args:
            root: Repository root.

        Returns:
            Absolute candidate paths, frozen copy first.
        """
        base = Path(root)
        candidates: list[Path] = []
        if self.frozen_copy_relative_path:
            candidates.append(base / self.frozen_copy_relative_path)
        candidates.append(base / self.checkpoint_relative_path)
        return tuple(candidates)

    def verify_checkpoint(self, path: str | Path) -> str:
        """Verify that a file is the selected checkpoint.

        Args:
            path: Checkpoint to verify.

        Returns:
            The verified digest.

        Raises:
            CheckpointMissingError: If the file does not exist.
            CheckpointMismatchError: If its name is one the freeze rejects, or
                its bytes are not the selected checkpoint's.
        """
        candidate = Path(path)
        if candidate.name in REJECTED_CHECKPOINT_NAMES:
            msg = (
                f"{MODEL_ARTIFACT_MISMATCH}: {candidate.name} is not the selected checkpoint. "
                f"The frozen detector is {self.selected_experiment}'s best.pt, chosen by the "
                "predeclared validation-fitness rule; the final epoch's weights are a different "
                "model."
            )
            raise CheckpointMismatchError(msg)
        if not candidate.is_file():
            msg = (
                f"{BLOCKED_MISSING_MODEL_ARTIFACT}: {candidate} does not exist. Obtain the "
                f"{self.selected_experiment} checkpoint with SHA-256 {self.checkpoint_sha256}; "
                "do not retrain, because a re-run produces different weights under the same "
                "experiment name."
            )
            raise CheckpointMissingError(msg)
        found = sha256_file(candidate)
        if found != self.checkpoint_sha256:
            msg = (
                f"{MODEL_ARTIFACT_MISMATCH}: {candidate} has SHA-256 {found}, but the frozen "
                f"detector is {self.checkpoint_sha256}. These are different weights; the "
                "frozen manifest is authoritative and this file is not the selected model."
            )
            raise CheckpointMismatchError(msg)
        return found


def parse_final_detector(manifest: Mapping[str, Any]) -> FinalDetector:
    """Parse and validate a final-detector manifest.

    Validation is strict on the things that decide identity and permissive on
    everything else: a missing prose field is a documentation defect, while a
    missing digest means the manifest cannot identify a model at all.

    Args:
        manifest: The parsed manifest mapping.

    Returns:
        The frozen detector's identity.

    Raises:
        FinalDetectorError: If a required field is absent, the status is not
            :data:`FROZEN`, the recorded fingerprint disagrees with the one
            recomputed from the manifest's own contents, or the manifest names
            the protected split.
    """
    missing = [field for field in REQUIRED_FIELDS if field not in manifest]
    if missing:
        msg = f"final detector manifest is missing required field(s) {missing}"
        raise FinalDetectorError(msg)

    if manifest.get("schema_version") != SCHEMA_VERSION:
        msg = (
            f"final detector manifest declares schema_version "
            f"{manifest.get('schema_version')!r}, expected {SCHEMA_VERSION}"
        )
        raise FinalDetectorError(msg)

    status = manifest.get("status")
    if status != FROZEN:
        msg = f"final detector manifest status is {status!r}, not {FROZEN!r}; no detector is frozen"
        raise FinalDetectorError(msg)

    leaks = holdout_leaks(manifest)
    if leaks:
        msg = f"final detector manifest names the protected split at {leaks}"
        raise FinalDetectorError(msg)

    checkpoint = manifest.get("selected_checkpoint")
    if not isinstance(checkpoint, Mapping):
        msg = "final detector manifest carries no selected_checkpoint block"
        raise FinalDetectorError(msg)
    missing_checkpoint = [field for field in CHECKPOINT_FIELDS if field not in checkpoint]
    if missing_checkpoint:
        msg = f"selected_checkpoint is missing field(s) {missing_checkpoint}"
        raise FinalDetectorError(msg)
    if checkpoint.get("committed") is not False:
        msg = (
            "selected_checkpoint.committed must be false: model binaries are deliberately not "
            "committed to this repository"
        )
        raise FinalDetectorError(msg)

    sha256 = checkpoint.get("sha256")
    if not isinstance(sha256, str) or len(sha256) != 64:
        msg = f"selected_checkpoint.sha256 {sha256!r} is not a SHA-256 digest"
        raise FinalDetectorError(msg)

    size_bytes = checkpoint.get("size_bytes")
    if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes <= 0:
        msg = f"selected_checkpoint.size_bytes {size_bytes!r} is not a positive integer"
        raise FinalDetectorError(msg)

    relative_path = str(checkpoint.get("relative_path", ""))
    if Path(relative_path).name in REJECTED_CHECKPOINT_NAMES:
        msg = (
            f"selected_checkpoint.relative_path points at {Path(relative_path).name}, which the "
            "freeze rejects; the selected checkpoint is the predeclared best.pt"
        )
        raise FinalDetectorError(msg)

    imgsz = manifest.get("imgsz")
    if isinstance(imgsz, bool) or not isinstance(imgsz, int):
        msg = f"final detector manifest imgsz {imgsz!r} is not an integer"
        raise FinalDetectorError(msg)
    batch = manifest.get("batch")
    if isinstance(batch, bool) or not isinstance(batch, int):
        msg = f"final detector manifest batch {batch!r} is not an integer"
        raise FinalDetectorError(msg)

    fingerprints = manifest.get("dataset_fingerprints")
    if not isinstance(fingerprints, Mapping):
        msg = "final detector manifest carries no dataset_fingerprints block"
        raise FinalDetectorError(msg)
    split_reference = manifest.get("split_reference")
    if not isinstance(split_reference, Mapping):
        msg = "final detector manifest carries no split_reference block"
        raise FinalDetectorError(msg)

    recorded = manifest.get("final_detector_sha256")
    computed = final_detector_fingerprint(_fingerprint_values(manifest))
    if recorded != computed:
        msg = (
            f"final_detector_sha256 {recorded!r} does not match the fingerprint recomputed from "
            f"the manifest's own identity fields ({computed}). One of them has been edited."
        )
        raise FinalDetectorError(msg)

    frozen_copy = manifest.get("frozen_copy")
    frozen_path: str | None = None
    if isinstance(frozen_copy, Mapping):
        value = frozen_copy.get("relative_path")
        frozen_path = str(value) if value else None

    return FinalDetector(
        selected_experiment=str(manifest["selected_experiment"]),
        model=str(manifest["model"]),
        imgsz=imgsz,
        batch=batch,
        checkpoint_relative_path=relative_path,
        frozen_copy_relative_path=frozen_path,
        checkpoint_sha256=sha256,
        checkpoint_size_bytes=size_bytes,
        experiment_sha256=str(manifest["experiment_sha256"]),
        policy_sha256=str(manifest["phase7_policy_sha256"]),
        split_assignment_sha256=str(split_reference.get("split_assignment_sha256", "")),
        dataset_fingerprints=dict(fingerprints),
        policy_case=str(manifest["policy_case"]),
        selection_method=str(manifest["selection_method"]),
        binary_distribution_status=str(manifest["binary_distribution_status"]),
        fingerprint=str(recorded),
        manifest=dict(manifest),
    )


def load_final_detector(reports: str | Path) -> FinalDetector:
    """Read and validate the committed final-detector manifest.

    Args:
        reports: The project's reports directory.

    Returns:
        The frozen detector's identity.

    Raises:
        FinalDetectorError: If the manifest is absent, unreadable or invalid.
    """
    path = Path(reports) / FINAL_DETECTOR_MANIFEST
    if not path.is_file():
        msg = (
            f"no frozen detector: {FINAL_DETECTOR_MANIFEST} does not exist. Phase 7D writes it; "
            "until then the project has candidates, not a selected model."
        )
        raise FinalDetectorError(msg)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"{FINAL_DETECTOR_MANIFEST} is not valid JSON ({exc})"
        raise FinalDetectorError(msg) from exc
    if not isinstance(payload, dict):
        msg = f"{FINAL_DETECTOR_MANIFEST} must contain a JSON object"
        raise FinalDetectorError(msg)
    return parse_final_detector(payload)


def load_checkpoint_path(detector: FinalDetector, root: str | Path) -> Path:
    """Resolve the frozen detector's checkpoint on this machine.

    Tries the immutable frozen copy first, then the training run's output
    directory, and verifies the bytes of whichever it finds. Nothing is
    downloaded and nothing is trained: a checkpoint that is not here is reported
    as missing, because producing one would produce different weights.

    Args:
        detector: The frozen detector's identity.
        root: Repository root.

    Returns:
        The absolute path of the verified checkpoint.

    Raises:
        CheckpointMissingError: If no candidate path exists.
        CheckpointMismatchError: If a candidate exists but its bytes differ.
    """
    candidates = detector.candidate_paths(root)
    present = [path for path in candidates if path.is_file()]
    if not present:
        rendered = ", ".join(str(path) for path in candidates)
        msg = (
            f"{BLOCKED_MISSING_MODEL_ARTIFACT}: none of [{rendered}] exists. The frozen detector "
            f"is {detector.selected_experiment} best.pt, SHA-256 {detector.checkpoint_sha256}, "
            f"{detector.checkpoint_size_bytes} bytes. Obtain that artifact; do not retrain."
        )
        raise CheckpointMissingError(msg)
    detector.verify_checkpoint(present[0])
    return present[0]
