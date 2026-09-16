"""Shared fixtures and helpers for the test suite.

The one helper here answers a question several suites need: has the single
authorised holdout evaluation happened yet?

Until phase 11B runs, no holdout image or annotation may exist on disk, and
several phases assert exactly that. Phase 11B materialises the holdout - through
the phase 5D function, under both authorisation gates, as its frozen protocol
requires - so from that point the same directories legitimately exist. Gating
those assertions on the committed evidence that the evaluation happened keeps
the protection for every other state of the repository, including the one that
matters most: a holdout materialised by accident during development.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from construction_safety_vision.paths import ProjectPaths

FINAL_EVALUATION_PROVENANCE = "final_test_evaluation.provenance.json"
"""Committed evidence that the one-shot holdout evaluation was executed."""


def holdout_has_been_evaluated(paths: ProjectPaths | None = None) -> bool:
    """Report whether phase 11B has run in this checkout.

    Args:
        paths: Project layout. Resolved from the repository root when omitted.

    Returns:
        ``True`` once the final evaluation's provenance record exists.
    """
    resolved = paths if paths is not None else ProjectPaths.from_root()
    return (resolved.reports / FINAL_EVALUATION_PROVENANCE).is_file()


def pytest_addoption(parser):
    parser.addoption(
        "--metadata-only",
        action="store_true",
        default=False,
        help="Exclude real-checkpoint integration checks and block local experiment/data reads.",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--metadata-only"):
        return
    local_artifact_checks = {
        "test_checkpoint_hashes_match_the_files_on_disk",
        "test_the_weights_on_disk_still_match_the_manifest",
        "test_the_frozen_copy_carries_the_selected_digest",
        "test_the_frozen_checkpoint_resolves_and_verifies",
        "test_the_accessor_rejects_the_reference_checkpoint",
        "test_the_accessor_rejects_last_pt",
        "test_the_frozen_copy_lives_outside_the_run_directory",
        "test_the_revalidation_corroboration_is_true_on_disk",
    }
    for item in items:
        # Restored development data must not turn a metadata run into an image/
        # annotation integration run. Keep the I/O audit active as the fail-closed
        # backstop; exclude these explicitly known local-data checks up front.
        task_data_checks = {
            "test_emitted_documents_match_the_manifest_fingerprints",
            "test_emitted_documents_are_aligned_on_disk",
            "test_emitted_counts_match_the_frozen_membership",
            "test_emitted_image_ids_are_the_frozen_membership",
            "test_the_same_image_carries_the_same_id_in_both_views",
            "test_detection_documents_carry_no_masks",
            "test_segmentation_documents_keep_both_geometry_representations",
            "test_synthetic_rectangles_are_labelled_as_synthetic",
            "test_emitted_categories_are_the_frozen_class_map",
            "test_emitted_documents_carry_no_timestamp",
            "test_emitted_documents_load_through_pycocotools",
        }
        is_task_data = (
            item.path.name == "test_task_datasets.py" and item.originalname in task_data_checks
        )
        is_canonical_fixture = (
            item.path.name == "test_segmentation_adapter_artifacts.py"
            and "canonical" in item.fixturenames
        )
        if is_task_data or is_canonical_fixture:
            item.add_marker(
                pytest.mark.skip(reason="metadata-only: real development data integration excluded")
            )
        if item.name in local_artifact_checks and item.path.name.endswith("_artifacts.py"):
            item.add_marker(
                pytest.mark.skip(reason="metadata-only: real experiment file read excluded")
            )


@pytest.fixture(scope="session", autouse=True)
def metadata_only_io_boundary(request):
    """Fail before an unexpected read of real experiment/data files in delivery checks."""
    if not request.config.getoption("--metadata-only"):
        yield
        return
    root = Path(__file__).resolve().parents[1]
    protected = [root / "artifacts", root / "reports/figures/final_test"]
    protected += [root / "data" / name for name in ("raw", "interim", "processed", "external")]
    active = True

    def audit_open(event, args):
        if active and event == "open" and isinstance(args[0], (str, bytes)):
            candidate = Path(args[0].decode() if isinstance(args[0], bytes) else args[0]).resolve()
            if any(candidate.is_relative_to(directory) for directory in protected):
                raise RuntimeError(
                    "metadata-only: unexpected local experiment/data file access blocked"
                )

    sys.addaudithook(audit_open)
    yield
    active = False
