"""A model-specific YOLO detection view of the canonical COCO detection dataset.

This is an **adapter**, not a second ground truth. The canonical COCO detection
dataset frozen in phase 5D remains the source of truth; everything here is a
derived representation that exists only because Ultralytics reads labels in its
own format. If the two ever disagree, the COCO file is right and the adapter is
broken.

That distinction is worth enforcing rather than asserting, because a lossy
adapter is invisible: training would simply learn slightly wrong boxes and every
downstream metric would be quietly measuring the wrong thing. So the conversion
is audited end to end - every box is converted, written, read back from the file
on disk, decoded to source pixels and compared with the canonical box.

Three properties are deliberate.

**Detection only.** No YOLO segmentation labels are produced anywhere. The
canonical segmentation state holds compressed RLE that YOLO's polygon-only
format cannot express without rasterising and re-polygonising, and that
conversion must be fidelity-audited by a later model-adapter phase before any
model sees it. Boxes have no such problem: a box is a box.

**The provider's bbox is never consulted.** The input is the canonical
segmentation-derived box from phase 5D, which is already the project's detection
ground truth.

**Development splits only.** The holdout is resolved through the same guard as
everywhere else, so ``test`` needs both opt-ins; a development run refuses it
outright.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from construction_safety_vision.config import ConfigError, check_keys

ADAPTER_TYPE = "YOLO_DETECTION"
"""The only adapter this module builds."""

CONFIG_SCHEMA_VERSION = 1
"""Schema version of ``configs/detection_adapter.yaml``."""

TEST_DISABLED = "disabled_during_development"
"""The only test-adapter policy accepted while the models are unfrozen."""

EMPTY_LABEL_FILE = "EMPTY_LABEL_FILE"
"""How an image carrying no annotation is represented."""

NOT_MATERIALIZED = "NOT_MATERIALIZED_PROTECTED_HOLDOUT"
"""Status recorded for the protected split."""


class YoloAdapterError(RuntimeError):
    """Raised when a conversion cannot be performed without losing information."""


def digest(payload: Any) -> str:
    """Hash a JSON-serialisable payload deterministically.

    Args:
        payload: Content to hash.

    Returns:
        Its SHA-256 hex digest.
    """
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AdapterTolerances:
    """Numeric slack the adapter's audits allow.

    Attributes:
        round_trip_px: Largest per-coordinate difference accepted between a
            canonical box and the same box decoded back from its YOLO label.
        normalised_bound_epsilon: Floating-point slack on the ``[0, 1]`` bound
            check. Deliberately tiny: it absorbs representation error in the
            division, not real geometry outside the canvas.
    """

    round_trip_px: float
    normalised_bound_epsilon: float

    def as_dict(self) -> dict[str, float]:
        """Serialise for the manifest and the configuration fingerprint.

        Returns:
            A JSON-serialisable mapping.
        """
        return {
            "round_trip_px": self.round_trip_px,
            "normalised_bound_epsilon": self.normalised_bound_epsilon,
        }


@dataclass(frozen=True)
class AdapterConfig:
    """Every choice the adapter makes, declared in a file.

    Attributes:
        schema_version: Version of this configuration schema.
        adapter_type: The model format produced.
        canonical_task_manifest: Repository-relative path to the phase 5D record.
        canonical_annotations_root: Repository-relative directory holding the
            canonical COCO documents.
        canonical_images_root: Repository-relative directory holding the
            canonical materialised images.
        split_directories: Adapter directory name for each canonical split.
        bbox_source: Which box is converted.
        normalisation: What the coordinates are normalised by.
        image_copy_mode: How image bytes are transferred.
        label_precision: Decimal places written per normalised coordinate.
        zero_instance_policy: How an annotation-free image is represented.
        test_adapter: Policy for the protected split.
        output_root: Repository-relative destination root.
        tolerances: Numeric slack the audits allow.
    """

    schema_version: int
    adapter_type: str
    canonical_task_manifest: str
    canonical_annotations_root: str
    canonical_images_root: str
    split_directories: dict[str, str]
    bbox_source: str
    normalisation: str
    image_copy_mode: str
    label_precision: int
    zero_instance_policy: str
    test_adapter: str
    output_root: str
    tolerances: AdapterTolerances

    def as_dict(self) -> dict[str, Any]:
        """Serialise the protocol for hashing and reporting.

        Returns:
            A JSON-serialisable mapping of the configuration content.
        """
        return {
            "schema_version": self.schema_version,
            "adapter_type": self.adapter_type,
            "canonical_task_manifest": self.canonical_task_manifest,
            "canonical_annotations_root": self.canonical_annotations_root,
            "canonical_images_root": self.canonical_images_root,
            "split_directories": self.split_directories,
            "bbox_source": self.bbox_source,
            "normalisation": self.normalisation,
            "image_copy_mode": self.image_copy_mode,
            "label_precision": self.label_precision,
            "zero_instance_policy": self.zero_instance_policy,
            "test_adapter": self.test_adapter,
            "output_root": self.output_root,
            "tolerances": self.tolerances.as_dict(),
        }

    def fingerprint(self) -> str:
        """Hash the protocol, so an adapter can name the rules that produced it.

        Returns:
            A SHA-256 hex digest over the configuration content only.
        """
        return digest(self.as_dict())


def load_adapter_config(path: str | Path) -> AdapterConfig:
    """Load and validate the detection-adapter protocol.

    Parsing is strict: an unknown key raises rather than being ignored.

    Args:
        path: Configuration file.

    Returns:
        The parsed protocol.

    Raises:
        ConfigError: If the file is missing, is not valid YAML, or is malformed.
    """
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        msg = f"Detection-adapter configuration not found: {config_path.name}"
        raise ConfigError(msg)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        msg = f"Detection-adapter configuration is not valid YAML: {config_path.name} ({exc})"
        raise ConfigError(msg) from exc
    if not isinstance(raw, Mapping):
        msg = f"Detection-adapter configuration must be a mapping: {config_path.name}"
        raise ConfigError(msg)

    check_keys(
        raw,
        required=(
            "schema_version",
            "adapter_type",
            "canonical_task_manifest",
            "canonical_annotations_root",
            "canonical_images_root",
            "split_directories",
            "bbox_source",
            "normalisation",
            "image_copy_mode",
            "label_precision",
            "zero_instance_policy",
            "test_adapter",
            "output_root",
            "tolerances",
        ),
        context="detection_adapter",
    )
    if raw["adapter_type"] != ADAPTER_TYPE:
        msg = (
            f"detection_adapter.adapter_type must be {ADAPTER_TYPE!r}, got {raw['adapter_type']!r}"
        )
        raise ConfigError(msg)
    if raw["test_adapter"] != TEST_DISABLED:
        msg = (
            f"detection_adapter.test_adapter must be {TEST_DISABLED!r} while the models are "
            f"unfrozen, got {raw['test_adapter']!r}"
        )
        raise ConfigError(msg)

    directories = raw["split_directories"]
    if not isinstance(directories, Mapping):
        msg = "detection_adapter.split_directories: must be a mapping"
        raise ConfigError(msg)
    # `test` is accepted as a key only so the explicit rejection below can
    # explain why it is refused; without this it would fail as a generic
    # unknown key and the reason would be lost.
    check_keys(
        directories,
        required=("train", "validation"),
        optional=("test",),
        context="split_directories",
    )
    if "test" in directories:
        msg = "detection_adapter.split_directories must not name the protected split"
        raise ConfigError(msg)

    tolerances = raw["tolerances"]
    if not isinstance(tolerances, Mapping):
        msg = "detection_adapter.tolerances: must be a mapping"
        raise ConfigError(msg)
    check_keys(
        tolerances,
        required=("round_trip_px", "normalised_bound_epsilon"),
        context="detection_adapter.tolerances",
    )
    try:
        parsed_tolerances = AdapterTolerances(
            round_trip_px=float(tolerances["round_trip_px"]),
            normalised_bound_epsilon=float(tolerances["normalised_bound_epsilon"]),
        )
        precision = int(raw["label_precision"])
    except (TypeError, ValueError) as exc:
        msg = f"detection_adapter: numeric field is not numeric ({exc})"
        raise ConfigError(msg) from exc
    if not 1 <= precision <= 17:
        msg = f"detection_adapter.label_precision must be between 1 and 17, got {precision}"
        raise ConfigError(msg)

    return AdapterConfig(
        schema_version=int(raw["schema_version"]),
        adapter_type=str(raw["adapter_type"]),
        canonical_task_manifest=str(raw["canonical_task_manifest"]),
        canonical_annotations_root=str(raw["canonical_annotations_root"]),
        canonical_images_root=str(raw["canonical_images_root"]),
        split_directories={
            "train": str(directories["train"]),
            "validation": str(directories["validation"]),
        },
        bbox_source=str(raw["bbox_source"]),
        normalisation=str(raw["normalisation"]),
        image_copy_mode=str(raw["image_copy_mode"]),
        label_precision=precision,
        zero_instance_policy=str(raw["zero_instance_policy"]),
        test_adapter=str(raw["test_adapter"]),
        output_root=str(raw["output_root"]),
        tolerances=parsed_tolerances,
    )


@dataclass(frozen=True)
class YoloBox:
    """One detection in Ultralytics' normalised centre-form.

    Attributes:
        class_id: Frozen canonical class index.
        centre_x: Box centre x, normalised by image width.
        centre_y: Box centre y, normalised by image height.
        width: Box width, normalised by image width.
        height: Box height, normalised by image height.
    """

    class_id: int
    centre_x: float
    centre_y: float
    width: float
    height: float


def coco_to_yolo(
    bbox: Sequence[float], *, class_id: int, image_width: int, image_height: int
) -> YoloBox:
    """Convert a canonical COCO box to Ultralytics' normalised centre-form.

    The formula, with the image ``W x H``::

        COCO  [x, y, w, h]          top-left corner plus extent, in pixels
        YOLO  [cx, cy, nw, nh]      centre plus extent, normalised

        cx = (x + w / 2) / W        nw = w / W
        cy = (y + h / 2) / H        nh = h / H

    Args:
        bbox: Canonical box as ``[x, y, width, height]`` in source pixels.
        class_id: Frozen canonical class index.
        image_width: Source image width.
        image_height: Source image height.

    Returns:
        The converted box.

    Raises:
        YoloAdapterError: If the box or the canvas is degenerate.
    """
    if image_width <= 0 or image_height <= 0:
        msg = f"cannot normalise against a {image_width}x{image_height} canvas"
        raise YoloAdapterError(msg)
    try:
        x, y, width, height = (float(value) for value in bbox)
    except (TypeError, ValueError) as exc:
        msg = f"expected four numeric bbox values, got {bbox!r}"
        raise YoloAdapterError(msg) from exc
    if width <= 0 or height <= 0:
        msg = f"box {list(bbox)} has a non-positive extent and cannot be a detection target"
        raise YoloAdapterError(msg)
    return YoloBox(
        class_id=class_id,
        centre_x=(x + width / 2.0) / image_width,
        centre_y=(y + height / 2.0) / image_height,
        width=width / image_width,
        height=height / image_height,
    )


def yolo_to_coco(box: YoloBox, *, image_width: int, image_height: int) -> list[float]:
    """Decode a normalised centre-form box back to canonical COCO pixels.

    The exact inverse of :func:`coco_to_yolo`, used by the round-trip audit.

    Args:
        box: The normalised box.
        image_width: Source image width.
        image_height: Source image height.

    Returns:
        The box as ``[x, y, width, height]`` in source pixels.
    """
    width = box.width * image_width
    height = box.height * image_height
    return [
        box.centre_x * image_width - width / 2.0,
        box.centre_y * image_height - height / 2.0,
        width,
        height,
    ]


def validate_normalised(box: YoloBox, *, epsilon: float) -> list[str]:
    """Check that a normalised box lies inside the unit square.

    Args:
        box: The normalised box.
        epsilon: Floating-point slack on the bound.

    Returns:
        One description per violation, empty when the box is valid.
    """
    problems: list[str] = []
    if box.width <= 0 or box.height <= 0:
        problems.append(f"non-positive normalised extent ({box.width}, {box.height})")
    for name, value in (
        ("centre_x", box.centre_x),
        ("centre_y", box.centre_y),
        ("width", box.width),
        ("height", box.height),
    ):
        if not -epsilon <= value <= 1.0 + epsilon:
            problems.append(f"{name} {value!r} is outside [0, 1]")
    left, right = box.centre_x - box.width / 2.0, box.centre_x + box.width / 2.0
    top, bottom = box.centre_y - box.height / 2.0, box.centre_y + box.height / 2.0
    if left < -epsilon or top < -epsilon or right > 1.0 + epsilon or bottom > 1.0 + epsilon:
        problems.append(
            f"box spans ({left}, {top}) to ({right}, {bottom}), which leaves the unit square"
        )
    return problems


def format_label_line(box: YoloBox, *, precision: int) -> str:
    """Serialise one box as an Ultralytics label line.

    Fixed-point with an explicit precision rather than ``repr``: the file must be
    byte-identical between runs and between machines, and a shortest-repr float
    is neither.

    Args:
        box: The normalised box.
        precision: Decimal places per coordinate.

    Returns:
        The label line, without a trailing newline.
    """
    values = " ".join(
        f"{value:.{precision}f}" for value in (box.centre_x, box.centre_y, box.width, box.height)
    )
    return f"{box.class_id} {values}"


def parse_label_line(line: str) -> YoloBox:
    """Parse one Ultralytics label line back into a box.

    Args:
        line: The label line.

    Returns:
        The parsed box.

    Raises:
        YoloAdapterError: If the line is not five whitespace-separated numbers.
    """
    parts = line.split()
    if len(parts) != 5:
        msg = f"label line must hold 5 fields, got {len(parts)}: {line!r}"
        raise YoloAdapterError(msg)
    try:
        return YoloBox(
            class_id=int(parts[0]),
            centre_x=float(parts[1]),
            centre_y=float(parts[2]),
            width=float(parts[3]),
            height=float(parts[4]),
        )
    except ValueError as exc:
        msg = f"label line is not numeric: {line!r} ({exc})"
        raise YoloAdapterError(msg) from exc


def label_text(boxes: Sequence[YoloBox], *, precision: int) -> str:
    """Serialise one image's labels.

    An image with no boxes yields an empty file, which is Ultralytics' own
    representation of a negative - an image deliberately containing nothing to
    detect. It is not a missing label, and the dataset scan counts it as a
    background image rather than a defect.

    Args:
        boxes: The image's boxes, already in the order they should be written.
        precision: Decimal places per coordinate.

    Returns:
        The label file's contents.
    """
    if not boxes:
        return ""
    lines = [format_label_line(box, precision=precision) for box in boxes]
    return "\n".join(lines) + "\n"


@dataclass
class RoundTrip:
    """What the box round-trip audit measured.

    Attributes:
        checked: Boxes converted, written, read back and decoded.
        within_tolerance: Boxes whose decoded form matched the canonical one.
        exact: Boxes that matched to the last bit.
        max_delta_px: Largest per-coordinate difference observed.
        mean_delta_px: Mean per-coordinate difference.
        mismatches: One description per box outside tolerance.
    """

    checked: int = 0
    within_tolerance: int = 0
    exact: int = 0
    max_delta_px: float = 0.0
    mean_delta_px: float = 0.0
    mismatches: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Serialise for the manifest and the report.

        Returns:
            A JSON-serialisable mapping.
        """
        return {
            "annotations_checked": self.checked,
            "within_tolerance": self.within_tolerance,
            "exact_matches": self.exact,
            "max_delta_px": self.max_delta_px,
            "mean_delta_px": self.mean_delta_px,
            "mismatches": len(self.mismatches),
        }


def round_trip_boxes(
    entries: Iterable[tuple[str, Sequence[float], YoloBox, int, int]],
    *,
    tolerance_px: float,
) -> RoundTrip:
    """Audit the conversion by decoding every emitted box back to pixels.

    Args:
        entries: ``(label, canonical bbox, emitted box, width, height)`` tuples,
            where the emitted box has been read back from the label file rather
            than kept in memory - the serialisation is part of what is audited.
        tolerance_px: Largest per-coordinate difference accepted.

    Returns:
        The audit result.
    """
    result = RoundTrip()
    total = 0.0
    for label, canonical, emitted, width, height in entries:
        decoded = yolo_to_coco(emitted, image_width=width, image_height=height)
        deltas = [abs(a - float(b)) for a, b in zip(decoded, canonical, strict=True)]
        delta = max(deltas)
        result.checked += 1
        total += sum(deltas) / len(deltas)
        result.max_delta_px = max(result.max_delta_px, delta)
        if delta == 0.0:
            result.exact += 1
        if delta <= tolerance_px:
            result.within_tolerance += 1
        else:
            result.mismatches.append(
                f"{label}: decoded {decoded} differs from canonical {list(canonical)} "
                f"by {delta:.9f} px"
            )
    result.mean_delta_px = total / result.checked if result.checked else 0.0
    return result


def label_fingerprint(labels: Mapping[str, str]) -> str:
    """Fingerprint every label file by content.

    Args:
        labels: Label text keyed by source image id.

    Returns:
        A SHA-256 hex digest over the sorted ``(image id, text)`` pairs.
    """
    return digest([[image_id, labels[image_id]] for image_id in sorted(labels)])


ADAPTER_ROOT_PLACEHOLDER = "{{ADAPTER_ROOT}}"
"""Stand-in for the dataset root in the committed, portable template.

Ultralytics resolves a relative ``path`` against the current working directory
rather than against the descriptor's own location, so the file it actually reads
has to carry a resolved absolute path. That file is therefore machine-specific
and git-ignored, and this placeholder is what the committed template holds in its
place.
"""


def dataset_yaml(
    *,
    path_value: str,
    train_directory: str,
    validation_directory: str,
    class_map: Mapping[str, int],
) -> str:
    """Build the Ultralytics dataset descriptor.

    No ``test`` key is emitted. Ultralytics would happily accept one, and the
    protected split must not be reachable by pointing a configuration file at it.

    Args:
        path_value: Dataset root, as Ultralytics should resolve it. Pass
            :data:`ADAPTER_ROOT_PLACEHOLDER` to build the portable template and
            a resolved absolute path to build the file Ultralytics reads.
        train_directory: Training image directory, relative to the root.
        validation_directory: Validation image directory, relative to the root.
        class_map: The frozen class map.

    Returns:
        The YAML document.
    """
    names = "\n".join(
        f"  {index}: {name}" for name, index in sorted(class_map.items(), key=lambda item: item[1])
    )
    return (
        "# Ultralytics dataset descriptor for the canonical YOLO detection adapter.\n"
        "# Generated by scripts/build_detection_adapter.py - do not hand-edit.\n"
        "#\n"
        "# There is deliberately no `test` key. The project's holdout is protected,\n"
        "# and a dataset file is exactly the kind of place it could be reached by\n"
        "# accident.\n"
        "#\n"
        "# `path` is resolved at generation time because Ultralytics resolves a\n"
        f"# relative path against the working directory, not against this file. The\n"
        f"# committed portable template carries {ADAPTER_ROOT_PLACEHOLDER} instead.\n"
        "#\n"
        "# Class indices are the frozen canonical class map and must not be reordered.\n"
        f"path: {path_value}\n"
        f"train: {train_directory}\n"
        f"val: {validation_directory}\n"
        "\n"
        "names:\n"
        f"{names}\n"
    )
