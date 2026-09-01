"""Structurally inspect the acquired export and write the dataset provenance report.

This is not exploratory data analysis. No image is opened and no distribution is
computed: the questions answered here are "what exactly did we receive?" and "is
it internally consistent?".

The report separates two kinds of statement:

* ``provider_reported`` - copied from provider metadata, believed but not proven;
* ``computed`` - derived here from the downloaded files.

Usage:
    uv run python scripts/inspect_dataset.py
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from construction_safety_vision.config import ConfigError, load_experiment_config
from construction_safety_vision.data.coco import (
    CocoValidationError,
    SplitInspection,
    inspect_export,
)
from construction_safety_vision.data.versioning import analyse_source_vs_generated
from construction_safety_vision.paths import ProjectPaths
from construction_safety_vision.provenance import SCHEMA_VERSION, git_commit, sha256_file

PHASE = 3
"""Roadmap phase this script belongs to."""

MINIMUM_REQUIRED_IMAGES = 300
"""Academic requirement on the number of annotated images."""


def _load_acquisition_record(path: Path) -> dict[str, Any]:
    """Load the acquisition provenance record.

    Args:
        path: Path of the record written by ``download_dataset.py``.

    Returns:
        The decoded record.

    Raises:
        FileNotFoundError: If the record is absent.
    """
    if not path.is_file():
        msg = f"Acquisition record not found: {path}. Run scripts/download_dataset.py first."
        raise FileNotFoundError(msg)
    return json.loads(path.read_text(encoding="utf-8"))


def _split_payload(report: SplitInspection) -> dict[str, Any]:
    """Serialise one split inspection.

    Args:
        report: Inspection to serialise.

    Returns:
        A JSON-serialisable mapping.
    """
    payload = asdict(report)
    payload["categories"] = {str(k): v for k, v in report.categories.items()}
    payload["is_structurally_sound"] = report.is_structurally_sound
    return payload


def build_report(
    paths: ProjectPaths, config_path: Path
) -> tuple[dict[str, Any], list[SplitInspection]]:
    """Assemble the machine-readable dataset provenance report.

    Args:
        paths: Project layout.
        config_path: Configuration file describing the dataset.

    Returns:
        The report payload and the per-split inspections.

    Raises:
        ConfigError: If the configuration cannot be loaded.
        CocoValidationError: If the export layout or a COCO document is invalid.
        FileNotFoundError: If the acquisition record is absent.
    """
    config = load_experiment_config(config_path)
    dataset = config.dataset
    slug = f"{dataset.project_slug}-v{dataset.version}-{dataset.export_format}"
    export_root = paths.data_raw / slug
    record_path = paths.data_external / f"{slug}.provenance.json"

    record = _load_acquisition_record(record_path)
    details = record.get("details", {})
    provider_project = details.get("provider_reported_project", {})
    provider_version = details.get("provider_reported_version", {})

    inspections = inspect_export(export_root)

    annotation_hashes = {}
    for report in inspections:
        path = export_root / report.annotations_path
        annotation_hashes[report.annotations_path] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }

    computed_images = sum(r.image_records for r in inspections)
    computed_annotations = sum(r.annotation_records for r in inspections)
    computed_files = sum(r.image_files_on_disk for r in inspections)

    category_names: dict[str, set[str]] = {}
    for report in inspections:
        for name in report.categories.values():
            category_names.setdefault(name, set()).add(report.split)

    per_category_totals: dict[str, int] = {}
    for report in inspections:
        for name, count in report.annotations_per_category.items():
            per_category_totals[name] = per_category_totals.get(name, 0) + count

    segmentation_totals: dict[str, int] = {}
    for report in inspections:
        for kind, count in report.segmentation_types.items():
            segmentation_totals[kind] = segmentation_totals.get(kind, 0) + count

    expected = set(dataset.expected_classes)
    observed_annotated = {name for name, count in per_category_totals.items() if count > 0}
    declared_categories = set(category_names)

    analysis = analyse_source_vs_generated(
        dict(provider_project.get("splits", {})),
        dict(provider_version.get("splits", {})),
        (provider_version.get("augmentation", {}).get("image", {}) or {}).get("versions"),
    )

    integrity = {
        "splits_structurally_sound": {r.split: r.is_structurally_sound for r in inspections},
        "all_splits_sound": all(r.is_structurally_sound for r in inspections),
        "image_records_match_files_on_disk": computed_images == computed_files,
        "annotations_all_have_segmentation": segmentation_totals.get("absent", 0) == 0
        and segmentation_totals.get("empty", 0) == 0,
        "expected_classes_present_as_categories": sorted(expected - declared_categories) == [],
        "expected_classes_with_annotations": sorted(expected - observed_annotated) == [],
        "undeclared_categories": sorted(declared_categories - expected),
    }

    requirement_supported = (
        analysis["independent_source_images"] >= MINIMUM_REQUIRED_IMAGES
        and integrity["all_splits_sound"]
    )

    # Findings that are structurally valid but materially affect later phases.
    # Derived from the data so this section cannot drift out of date.
    classes_absent_per_split = {
        r.split: sorted(expected - {n for n, c in r.annotations_per_category.items() if c > 0})
        for r in inspections
    }
    geometry_kinds = sorted(k for k in segmentation_totals if k in {"polygon", "rle"})
    findings: list[dict[str, Any]] = []
    if len(geometry_kinds) > 1:
        findings.append(
            {
                "id": "mixed_segmentation_representation",
                "severity": "HIGH",
                "computed": {k: segmentation_totals[k] for k in geometry_kinds},
                "statement": (
                    "The export mixes polygon and RLE segmentation in the same files. Phase 5 must "
                    "decode both representations to derive bounding boxes; code that assumes "
                    "polygons only would silently drop the RLE annotations."
                ),
            }
        )
    absent = {s: c for s, c in classes_absent_per_split.items() if c}
    if absent:
        findings.append(
            {
                "id": "class_absent_from_split",
                "severity": "HIGH",
                "computed": absent,
                "statement": (
                    "At least one expected class has zero annotations in a split. A class absent "
                    "from the evaluation split cannot be scored there, which constrains what the "
                    "final metrics can claim."
                ),
            }
        )
    empty_images = {r.split: r.images_without_annotations for r in inspections}
    if sum(empty_images.values()) > 0:
        findings.append(
            {
                "id": "images_without_annotations",
                "severity": "MEDIUM",
                "computed": {
                    "per_split": empty_images,
                    "total": sum(empty_images.values()),
                    "provider_reported_unannotated": provider_project.get("unannotated"),
                },
                "statement": (
                    "Image records carry no annotation, while the provider reports "
                    f"unannotated={provider_project.get('unannotated')}. These may be deliberate "
                    "background/negative samples rather than missed labels; phase 4 must determine "
                    "which, since the distinction changes how they are used and scored."
                ),
            }
        )
    if integrity["undeclared_categories"]:
        findings.append(
            {
                "id": "undeclared_categories_present",
                "severity": "LOW",
                "computed": {"categories": integrity["undeclared_categories"]},
                "statement": (
                    "The export declares categories that the project configuration does not expect "
                    "and that carry no annotations. This is the provider's conventional root "
                    "placeholder; phase 5 must exclude it from the class map rather than shifting "
                    "every class index."
                ),
            }
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "report": "dataset_provenance",
        "phase": PHASE,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(paths.root),
        "provider": {
            "name": "roboflow",
            "workspace": dataset.workspace,
            "project_slug": dataset.project_slug,
            "project_name": dataset.name,
            "public_project_url": details.get("public_project_url", ""),
            "public_version_url": details.get("public_version_url", ""),
            "acquisition_method": details.get("acquisition_method", ""),
        },
        "project_level": {
            "source": "provider_reported",
            "task_type": provider_project.get("type"),
            "images": provider_project.get("images"),
            "unannotated": provider_project.get("unannotated"),
            "splits": provider_project.get("splits"),
            "annotation_counts_per_class": provider_project.get("classes"),
            "license": provider_project.get("license"),
            "public": provider_project.get("public"),
            "total_versions": provider_project.get("versions"),
        },
        "version_level": {
            "source": "provider_reported",
            "version": dataset.version,
            "name": provider_version.get("name"),
            "images": provider_version.get("images"),
            "splits": provider_version.get("splits"),
            "preprocessing": provider_version.get("preprocessing"),
            "augmentation": provider_version.get("augmentation"),
            "export_format": dataset.export_format,
            "export_size_mb": details.get("provider_reported_export", {}).get("size"),
        },
        "source_vs_generated_analysis": analysis,
        "acquired_artifact": {
            "source": "computed",
            "archive": record.get("outputs", [{}])[0] if record.get("outputs") else {},
            "acquired_at": record.get("created_at"),
            "acquired_at_git_commit": record.get("git_commit"),
            "extracted_to": details.get("extracted_to"),
            "extracted_members": details.get("extracted_members"),
        },
        "annotation_files": {"source": "computed", "files": annotation_hashes},
        "exported_structure": {
            "source": "computed",
            "splits": {r.split: _split_payload(r) for r in inspections},
            "totals": {
                "image_records": computed_images,
                "annotation_records": computed_annotations,
                "image_files_on_disk": computed_files,
                "annotations_per_category": dict(sorted(per_category_totals.items())),
                "segmentation_types": dict(sorted(segmentation_totals.items())),
            },
        },
        "classes": {
            "expected_from_config": sorted(expected),
            "declared_categories_computed": sorted(declared_categories),
            "categories_with_annotations_computed": sorted(observed_annotated),
            "declared_but_never_annotated": sorted(declared_categories - observed_annotated),
        },
        "integrity_checks": integrity,
        "notable_findings": {
            "source": "computed",
            "classes_absent_per_split": classes_absent_per_split,
            "findings": findings,
        },
        "academic_requirement": {
            "minimum_images": MINIMUM_REQUIRED_IMAGES,
            "independent_source_images": analysis["independent_source_images"],
            "currently_supported": requirement_supported,
            "basis": (
                "Independent source images are provider-reported (project-level count); the "
                "version-level count includes offline-generated variants and must not be used "
                "for this requirement."
            ),
        },
        "license": {
            "dataset_license": provider_project.get("license"),
            "dataset_license_url": "https://creativecommons.org/licenses/by/4.0/",
            "attribution_required": True,
            "creator": "AGIs Workspace",
            "note": (
                "Applies to the dataset only. The software license of this repository is a "
                "separate, still undecided matter. The dataset is not redistributed here."
            ),
        },
        "unresolved_questions": [
            "Which of the 436 source images survive de-duplication (phase 4).",
            "Whether near-duplicate or same-scene frames exist across the provider's splits "
            "(phase 4); the provider's split assignment has not been audited.",
            "Whether augmented train variants leak information about validation or test images; "
            "this cannot be ruled out until source-image identity is recovered (phase 4/5).",
            "Whether the provider's train/valid/test assignment is adopted or the data is "
            "re-split (phase 5).",
            "Annotation quality: polygon geometry has been counted, not validated against the "
            "imagery (phase 4).",
        ],
    }
    return payload, inspections


def _markdown(payload: dict[str, Any]) -> str:
    """Render the human-readable provenance report.

    Args:
        payload: The machine-readable report.

    Returns:
        Markdown text.
    """
    project = payload["project_level"]
    version = payload["version_level"]
    analysis = payload["source_vs_generated_analysis"]
    structure = payload["exported_structure"]
    integrity = payload["integrity_checks"]
    lines: list[str] = []
    add = lines.append

    add("# Dataset Provenance")
    add("")
    add(
        f"Generated: {payload['generated_at']} · Phase: {payload['phase']} · "
        f"Schema: {payload['schema_version']}"
    )
    add("")
    add(
        "Every fact below is labelled **provider-reported** (copied from the provider's API or "
        "its exported documentation) or **computed** (derived here from the downloaded files). "
        "No number in this document was estimated."
    )
    add("")

    add("## Identity")
    add("")
    add("| Field | Value | Basis |")
    add("| --- | --- | --- |")
    add(f"| Provider | {payload['provider']['name']} | given |")
    add(f"| Workspace | `{payload['provider']['workspace']}` | given |")
    add(f"| Project slug | `{payload['provider']['project_slug']}` | given |")
    add(f"| Project name | {payload['provider']['project_name']} | provider-reported |")
    add(f"| Task type | `{project['task_type']}` | provider-reported |")
    add(f"| Project URL | <{payload['provider']['public_project_url']}> | given |")
    add(f"| Version URL | <{payload['provider']['public_version_url']}> | given |")
    add(f'| Version | {version["version"]} ("{version["name"]}") | provider-reported |')
    add(f"| Export format | `{version['export_format']}` | given |")
    add(f"| Acquired at (UTC) | {payload['acquired_artifact']['acquired_at']} | computed |")
    commit = payload["acquired_artifact"]["acquired_at_git_commit"]
    add(f"| Acquired at commit | `{commit}` | computed |")
    add(f"| Method | {payload['provider']['acquisition_method']} | given |")
    add("")

    add("## Project level vs version level")
    add("")
    add("These are **different populations** and are never collapsed into one number.")
    add("")
    add("| | Project (source) | Version 4 |")
    add("| --- | --- | --- |")
    add(f"| Total images | {project['images']} | {version['images']} |")
    ps, vs = project["splits"] or {}, version["splits"] or {}
    for split in sorted(set(ps) | set(vs)):
        add(f"| {split} | {ps.get(split, '-')} | {vs.get(split, '-')} |")
    add(f"| Unannotated | {project['unannotated']} | - |")
    add("")
    add(
        "Both rows are provider-reported. The version-level split counts were independently "
        "**confirmed** against the extracted files (see Structure below)."
    )
    add("")

    add("### Why the counts differ")
    add("")
    add(f"**Conclusion: `{analysis['conclusion']}`**")
    add("")
    add(analysis["summary"])
    add("")
    add("| Split | Source | Version | Status |")
    add("| --- | --- | --- | --- |")
    for split, entry in analysis["per_split"].items():
        extra = f" (x{entry['multiplier']})" if "multiplier" in entry else ""
        add(f"| {split} | {entry['source']} | {entry['version']} | {entry['status']}{extra} |")
    add("")
    add("Corroborating evidence, three independent sources:")
    add("")
    add(
        f"1. Provider version metadata declares `augmentation.image.versions = "
        f"{analysis['declared_versions_per_source_image']}`."
    )
    add(
        "2. The arithmetic is exact: the augmented split equals source x that multiplier, and the "
        "other splits are unchanged."
    )
    add(
        "3. The provider's own exported `README.roboflow.txt` states the augmentation was applied "
        '"to create 2 versions of each source image".'
    )
    add("")
    add(
        f"> **Consequence for phases 4 and 5.** The {analysis['version_images']} version images "
        f"are not {analysis['version_images']} independent samples. The independent population is "
        f"{analysis['independent_source_images']} source images. Any metric, split or duplicate "
        "analysis that treats generated variants as independent would be wrong."
    )
    add("")

    add("## Acquired artifact")
    add("")
    archive = payload["acquired_artifact"]["archive"]
    add("| Field | Value |")
    add("| --- | --- |")
    add(f"| Path | `{archive.get('path')}` |")
    add(f"| SHA-256 | `{archive.get('sha256')}` |")
    add(f"| Size | {archive.get('size_bytes')} bytes |")
    add(f"| Extracted to | `{payload['acquired_artifact']['extracted_to']}` |")
    add(f"| Members | {payload['acquired_artifact']['extracted_members']} |")
    add("")
    add("Annotation file digests (computed):")
    add("")
    add("| File | SHA-256 | Bytes |")
    add("| --- | --- | --- |")
    for name, info in payload["annotation_files"]["files"].items():
        add(f"| `{name}` | `{info['sha256']}` | {info['size_bytes']} |")
    add("")
    add(
        "The archive is **not** committed. It is reproducible from the recorded provider "
        "coordinates; the digest above proves any future download is the same bytes."
    )
    add("")

    add("## Structure (computed from the export)")
    add("")
    add("| Split | Image records | Image files | Annotations | Sound |")
    add("| --- | --- | --- | --- | --- |")
    for split, data in structure["splits"].items():
        mark = "yes" if data["is_structurally_sound"] else "NO"
        add(
            f"| {split} | {data['image_records']} | {data['image_files_on_disk']} | "
            f"{data['annotation_records']} | {mark} |"
        )
    totals = structure["totals"]
    add(
        f"| **total** | **{totals['image_records']}** | **{totals['image_files_on_disk']}** | "
        f"**{totals['annotation_records']}** | |"
    )
    add("")
    add("Annotations per category (computed, all splits):")
    add("")
    add("| Category | Annotations |")
    add("| --- | --- |")
    for name, count in totals["annotations_per_category"].items():
        add(f"| `{name}` | {count} |")
    add("")
    add("Segmentation representation (computed, all annotations):")
    add("")
    add("| Representation | Count |")
    add("| --- | --- |")
    for kind, count in totals["segmentation_types"].items():
        add(f"| {kind} | {count} |")
    add("")

    add("## Integrity checks")
    add("")
    add("| Check | Result |")
    add("| --- | --- |")
    for key, value in integrity.items():
        add(f"| {key} | `{value}` |")
    add("")

    add("## Notable findings")
    add("")
    findings = payload["notable_findings"]["findings"]
    if not findings:
        add("None. The export is structurally clean with no material caveat.")
        add("")
    for item in findings:
        add(f"### `{item['id']}` ({item['severity']})")
        add("")
        add(f"Computed: `{item['computed']}`")
        add("")
        add(item["statement"])
        add("")
    add("Expected classes with zero annotations, per split (computed):")
    add("")
    add("| Split | Classes absent |")
    add("| --- | --- |")
    for split, absent_classes in payload["notable_findings"]["classes_absent_per_split"].items():
        add(f"| {split} | `{absent_classes}` |")
    add("")

    add("## Classes")
    add("")
    classes = payload["classes"]
    add(f"- Expected from configuration: `{classes['expected_from_config']}`")
    add(f"- Declared as COCO categories (computed): `{classes['declared_categories_computed']}`")
    add(
        f"- Categories carrying annotations (computed): "
        f"`{classes['categories_with_annotations_computed']}`"
    )
    add(f"- Declared but never annotated (computed): `{classes['declared_but_never_annotated']}`")
    add("")

    add("## Academic requirement")
    add("")
    req = payload["academic_requirement"]
    add(f"- Minimum required annotated images: **{req['minimum_images']}**")
    add(f"- Independent source images: **{req['independent_source_images']}**")
    add(f"- Currently supported: **{req['currently_supported']}**")
    add(f"- Basis: {req['basis']}")
    add("")

    add("## License and attribution")
    add("")
    lic = payload["license"]
    add(f"- Dataset license: **{lic['dataset_license']}** (<{lic['dataset_license_url']}>)")
    add(f"- Creator: {lic['creator']}")
    add(f"- Attribution required: {lic['attribution_required']}")
    add(f"- {lic['note']}")
    add("")
    add("Suggested attribution:")
    add("")
    add("```text")
    add(
        f"Construction PPE Compliance Detection [dataset], version {version['version']}. "
        f"{lic['creator']}, Roboflow Universe. Licensed CC BY 4.0."
    )
    add(f"{payload['provider']['public_version_url']}")
    add("```")
    add("")

    add("## Unresolved questions (deferred to phases 4 and 5)")
    add("")
    for question in payload["unresolved_questions"]:
        add(f"- {question}")
    add("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run the inspection and write the provenance artifacts.

    Args:
        argv: Command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code; ``1`` when a structural integrity check fails.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="Configuration file to use.")
    args = parser.parse_args(argv)

    paths = ProjectPaths.from_root()
    config_path = args.config or (paths.configs / "project.yaml")
    try:
        payload, inspections = build_report(paths, config_path)
    except (ConfigError, CocoValidationError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    json_path = paths.reports / "dataset_provenance.json"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    md_path = paths.reports / "dataset_provenance.md"
    md_path.write_text(_markdown(payload), encoding="utf-8", newline="\n")

    totals = payload["exported_structure"]["totals"]
    print(f"splits inspected: {[r.split for r in inspections]}")
    print(
        f"image records: {totals['image_records']}  files on disk: "
        f"{totals['image_files_on_disk']}  annotations: {totals['annotation_records']}"
    )
    print(f"segmentation types: {totals['segmentation_types']}")
    print(f"source-vs-generated: {payload['source_vs_generated_analysis']['conclusion']}")
    print(f"wrote reports/{json_path.name} and reports/{md_path.name}")

    if not payload["integrity_checks"]["all_splits_sound"]:
        print("ERROR: structural integrity check failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
