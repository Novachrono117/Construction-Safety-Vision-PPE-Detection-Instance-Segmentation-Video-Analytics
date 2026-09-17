# Academic technical report

The [Portuguese Markdown](final_report.md) is the source of truth; the
[ten-page PDF](final_report.pdf) is its typeset delivery. Tables transcribe
committed results. Comments around headline values link them to
[explicit JSON fields](report_claims.json); they are hidden in rendered Markdown.

## Verification and optional PDF build

From the repository root:

```text
uv run python scripts/validate_academic_report.py
```

The renderer needs only the separately pinned document packages in
`academic/requirements.txt`. ReportLab creates the PDF; Pillow selects already
published figure panels and embeds media; pypdf verifies page count. These
packages belong to an isolated document environment, not the frozen inference
environment. No change to `pyproject.toml`, `uv.lock`, models or experiments is
required. The Codex bundled document runtime supplied these exact versions for
the delivered build; no project dependency was installed during authoring.

Optional isolated reproduction on Windows (PowerShell):

```powershell
uv venv outputs/phase14b/pdf-env
uv pip install --python outputs/phase14b/pdf-env/Scripts/python.exe -r academic/requirements.txt
outputs/phase14b/pdf-env/Scripts/python.exe scripts/build_academic_report.py
```

On Linux/macOS use `outputs/phase14b/pdf-env/bin/python` instead. Run the claim
validator before building. The renderer creates no predictions or metrics.
Source revision `a788d5303e70ddb12f5dc5ae35dd9a4b56fc6736` must be present in Git.
PDF pagination is automatic; Markdown pagebreak comments mark editorial sections,
not mandatory physical page boundaries. Body text is 11 pt, tables 9.5 pt,
captions/references 9 pt. Standard PDF fonts are used.

## Figure handling and scope

The PDF selects complete person/vest_loose panels from the published validation
FP/FN gallery and complete disagreement scene panels from the mask gallery.
Scenes and scientific overlays are preserved; titles/layout are editorial.
The Markdown links the original galleries. No test image is embedded. The test
confusion matrix is a text table copied from committed aggregate metadata.
The video frame is already published evidence; the complete MP4 is not embedded.

Dataset figures retain CC BY 4.0. The Frank Vincentz/Wikimedia video frame and
its overlays retain CC BY-SA 3.0. Code remains AGPL-3.0. See the report for full
attribution and [delivery review](../reports/final_academic_report_delivery.md)
for exact assignment compliance and the author checklist.
