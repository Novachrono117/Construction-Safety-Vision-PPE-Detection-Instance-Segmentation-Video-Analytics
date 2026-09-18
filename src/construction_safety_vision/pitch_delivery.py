"""Validate the video-pitch package against committed evidence.

The pitch documents are presentation, never a source of truth. Every number the
presenter says is declared in the manifest together with the artifact and the field
it came from, and this module re-reads each one. It executes no model, reads no
checkpoint and never opens holdout content.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from construction_safety_vision.data.canonical import scan_for_sensitive
from construction_safety_vision.delivery_status import PITCH_MANIFEST_PATH, PITCH_READY

DOCUMENTS = (
    "delivery/pitch/README.md",
    "delivery/pitch/full_script.md",
    "delivery/pitch/presenter_cues.md",
    "delivery/pitch/storyboard.md",
    "delivery/pitch/qa.md",
    "delivery/pitch/recording_checklist.md",
)
"""Every document the package promises, checked for presence and for content."""

SPOKEN_WORD_RANGE = (750, 950)
"""The assignment's 5-8 minutes leaves no room for a script that overruns."""

DURATION_SECONDS_RANGE = (330, 435)
"""5:30 to 7:15, the rehearsal band inside the assignment's 5:00-8:00."""


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _field(payload: dict[str, Any], dotted: str) -> Any:
    value: Any = payload
    for key in dotted.split("."):
        value = value[key]
    return value


def load_manifest(root: Path) -> dict[str, Any]:
    """Read the committed pitch manifest.

    Args:
        root: Repository root.

    Returns:
        The manifest mapping.
    """
    return _read_json(root / PITCH_MANIFEST_PATH)


def spoken_words(root: Path) -> int:
    """Count only the words the presenter actually says.

    Blockquote lines carry narration; screen directions are italicised and excluded,
    so stage notes never inflate the timing estimate.

    Args:
        root: Repository root.

    Returns:
        The number of spoken words in the full script.
    """
    text = (root / "delivery/pitch/full_script.md").read_text(encoding="utf-8")
    return sum(
        len(line[2:].split())
        for line in text.splitlines()
        if line.startswith("> ") and not line.startswith("> *[")
    )


def validate(root: Path) -> list[str]:
    """Check the pitch package against the committed evidence it quotes.

    Args:
        root: Repository root.

    Returns:
        One description per problem found, empty when the package is sound.
    """
    problems: list[str] = []
    manifest = load_manifest(root)

    if manifest["classification"] != PITCH_READY:
        problems.append(f"classification must stay {PITCH_READY} until a recording exists")
    for flag in ("recorded_video_exists", "published_link_exists"):
        if manifest[flag] is not False:
            problems.append(f"{flag} must be false: this phase records and publishes nothing")
    if manifest["publication"]["link"] is not None:
        problems.append("no published link may be declared before the human records the pitch")
    for field, expected in manifest["scientific_lock"].items():
        if expected not in (0, False):
            problems.append(f"scientific_lock.{field} must record that nothing was executed")

    documents = {name: (root / name) for name in DOCUMENTS}
    for name, path in documents.items():
        if not path.is_file():
            problems.append(f"missing pitch document: {name}")
    if set(manifest["documents"].values()) - {*DOCUMENTS, PITCH_MANIFEST_PATH}:
        problems.append("the manifest lists a document outside the declared package")
    if problems:
        return problems

    bodies = {name: path.read_text(encoding="utf-8") for name, path in documents.items()}
    script = bodies["delivery/pitch/full_script.md"]
    cues = bodies["delivery/pitch/presenter_cues.md"]

    words = spoken_words(root)
    if words != manifest["duration"]["spoken_words"]:
        problems.append(
            f"manifest spoken_words {manifest['duration']['spoken_words']} != measured {words}"
        )
    if not SPOKEN_WORD_RANGE[0] <= words <= SPOKEN_WORD_RANGE[1]:
        problems.append(f"spoken words {words} outside {SPOKEN_WORD_RANGE}")
    rate = manifest["duration"]["assumed_rate_words_per_minute"]
    seconds = words / rate * 60
    if not DURATION_SECONDS_RANGE[0] <= seconds <= DURATION_SECONDS_RANGE[1]:
        problems.append(f"estimated {seconds:.0f}s outside {DURATION_SECONDS_RANGE}")

    for claim in manifest["spoken_claims"]:
        artifact = root / claim["artifact"]
        if not artifact.is_file():
            problems.append(f"{claim['id']}: evidence artifact {claim['artifact']} is missing")
            continue
        try:
            actual = _field(_read_json(artifact), claim["field"])
        except (KeyError, TypeError):
            problems.append(f"{claim['id']}: field {claim['field']} not in {claim['artifact']}")
            continue
        if actual != claim["value"]:
            problems.append(f"{claim['id']}: manifest {claim['value']!r} != artifact {actual!r}")
        if claim["spoken"] not in script:
            problems.append(f"{claim['id']}: spoken form {claim['spoken']!r} is not in the script")

    problems.extend(_phrasing_findings(manifest["forbidden_phrasing"], script, bodies))

    segment = manifest["video_segment"]
    anchors = _read_json(root / segment["anchor_evidence"])["screenshots"]
    documented = {shot["timestamp_seconds"] for shot in anchors}
    for group in ("anchor_instants_seconds", "alternative_anchor_instants_seconds"):
        for instant in segment[group]:
            if instant not in documented:
                problems.append(f"{group}: {instant}s has no frozen screenshot record")
        window = (
            (segment["selected_start_seconds"], segment["selected_end_seconds"])
            if group == "anchor_instants_seconds"
            else (segment["alternative_start_seconds"], segment["alternative_end_seconds"])
        )
        if not all(window[0] <= instant <= window[1] for instant in segment[group]):
            problems.append(f"{group}: the presentation window does not contain its anchors")
    if segment["predictions_edited"] or segment["highlight_reel"]:
        problems.append("the shown segment must be unedited recorded output")
    if not segment["shows_visible_failure"]:
        problems.append("the shown segment must contain a narrated visible failure")
    if segment["source_output_committed"]:
        problems.append("the demonstration MP4 is an external artifact and is not committed")

    if "vest_loose" not in script or "vest_loose" not in cues:
        problems.append("the vest_loose limitation must be spoken, not only written down")
    if "três das cinco classes pioraram" not in script:
        problems.append("the script must disclose that three of five classes declined")
    if "não vou dizer que o S1 venceu" not in script:
        problems.append("the script must refuse the winner reading explicitly")

    problems.extend(_leak_findings(root, bodies))
    return problems


def _flatten(text: str) -> str:
    """Collapse wrapping and markdown emphasis so a phrase check sees whole sentences."""
    return " ".join(text.replace("*", "").replace("`", "").split()).lower()


def narration(script: str) -> str:
    """Return only what the presenter says, without stage directions."""
    return _flatten(
        " ".join(
            line[2:]
            for line in script.splitlines()
            if line.startswith("> ") and not line.startswith("> *[")
        )
    )


def _phrasing_findings(forbidden: list[str], script: str, bodies: dict[str, str]) -> list[str]:
    """Check that a forbidden claim appears only inside an explicit refusal.

    The spoken narration is judged strictly, because that is what a viewer hears: a
    forbidden phrase there needs a refusal marker beside it. The other documents may
    quote the phrases in order to forbid them, so they only need a negation in the
    same sentence.
    """
    problems: list[str] = []
    spoken = narration(script)
    refusals = ("não vou dizer", "nada aqui é", "nunca", "não é a velocidade")
    for phrase in forbidden:
        needle = phrase.lower()
        start = 0
        while (index := spoken.find(needle, start)) != -1:
            window = spoken[max(0, index - 120) : index + 120]
            if not any(mark in window for mark in refusals):
                problems.append(f"narration states a forbidden claim: ...{window}...")
            start = index + len(needle)
    for name, body in bodies.items():
        for sentence in re.split(r"(?<=[.!?:])\s+|\n", _flatten_sentences(body)):
            lowered = sentence.lower()
            for phrase in forbidden:
                if phrase.lower() in lowered and not re.search(
                    r"\b(não|nunca|nada|nenhum|nenhuma|sem|proibid)", lowered
                ):
                    problems.append(f"{name}: forbidden phrasing used as a claim: {sentence[:110]}")
    return problems


def _flatten_sentences(text: str) -> str:
    """Join wrapped lines inside a paragraph so sentence splitting works on prose.

    Blockquote, list and heading markers are dropped first: a sentence that wraps
    across two quoted lines must not be split by the quote character.
    """
    stripped = "\n".join(
        re.sub(r"^\s*(?:>\s*|[-*+]\s+|#+\s*)+", "", line) for line in text.splitlines()
    )
    paragraphs = re.split(r"\n\s*\n", stripped)
    return "\n".join(" ".join(p.replace("*", "").replace("`", "").split()) for p in paragraphs)


def _leak_findings(root: Path, bodies: dict[str, str]) -> list[str]:
    """Check the package publishes no holdout identifier, path or credential."""
    problems: list[str] = []
    assignments = root / "reports/final_split_assignments.csv"
    test_ids: set[str] = set()
    if assignments.is_file():
        import csv

        with assignments.open(encoding="utf-8", newline="") as handle:
            test_ids = {
                row["source_image_id"] for row in csv.DictReader(handle) if row["split"] == "test"
            }
    for name, body in bodies.items():
        problems.extend(f"{name}: {finding}" for finding in scan_for_sensitive(body))
        leaked = sorted(image_id for image_id in test_ids if image_id in body)
        if leaked:
            problems.append(f"{name}: publishes {len(leaked)} holdout identifier(s)")
        # A concrete personal publication URL must not be committed before it exists.
        if re.search(r"(drive\.google\.com|youtu\.be|youtube\.com)/\S{6,}", body):
            problems.append(f"{name}: a personal publication URL must not be committed")
    return problems
