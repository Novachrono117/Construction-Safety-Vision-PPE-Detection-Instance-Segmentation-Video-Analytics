"""Render the editorial Markdown as a paginated PDF; no scientific computation."""

from __future__ import annotations

import html
import io
import json
import re
from pathlib import Path

from PIL import Image as PILImage
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    Image,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
WIDTH = A4[0] - 104


def inline(text: str) -> str:
    """Convert the report's limited inline Markdown to PDF paragraph markup."""
    text = (
        html.escape(text.strip())
        .replace("≥", "&gt;=")
        .replace("→", " -&gt; ")
        .replace("\u2212", "-")
    )
    text = re.sub(
        r"\[([^\]]+)\]\((https://[^)]+)\)",
        r'<link href="\2" color="#163e60"><u>\1</u></link>',
        text,
    )
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)


def figure(target: Path) -> Image | Table:
    """Select published gallery panels; preserve complete scenes and overlays."""
    image = PILImage.open(target)
    if target.name == "validation_fp_fn_s1.png":
        # Rows for person and vest_loose, including original panel labels.
        image = image.crop((175, 1535, 2180, 2840))
    if target.name == "mask_quality_gallery.png":
        # Complete scene panels from the published mask-disagreement column.
        boxes = [(1940, 220, 2275, 706), (1740, 875, 2475, 1365), (1740, 1532, 2475, 2023)]
        labels = ["Bom ajuste", "Subcobertura", "Sobrecobertura"]
        panels = []
        for box in boxes:
            panel = image.crop(box)
            stream = io.BytesIO()
            panel.save(stream, format="PNG")
            stream.seek(0)
            scale = min((WIDTH / 3 - 9) / panel.width, 143 / panel.height)
            panels.append(Image(stream, width=panel.width * scale, height=panel.height * scale))
        table = Table([labels, panels], colWidths=[WIDTH / 3] * 3)
        table.setStyle(
            TableStyle(
                [
                    ("FONT", (0, 0), (-1, 0), "Helvetica", 9),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ]
            )
        )
        return table
    stream = io.BytesIO()
    image.convert("RGB").save(stream, format="JPEG", quality=93, subsampling=0)
    stream.seek(0)
    width = WIDTH
    return Image(stream, width=width, height=width * image.height / image.width)


def build() -> dict:
    """Build the PDF from its Markdown source and reject page-budget overflow."""
    text = (ROOT / "academic/final_report.md").read_text(encoding="utf-8")
    text = re.sub(r"<!-- claim:[a-z0-9_]+ -->|<!-- /claim -->", "", text)
    if "{{" in text:
        raise ValueError("Resolve and validate source claims before rendering")
    body = ParagraphStyle(
        "Body",
        fontName="Times-Roman",
        fontSize=11,
        leading=14,
        spaceAfter=7,
        alignment=TA_JUSTIFY,
        allowWidows=0,
        allowOrphans=0,
    )
    styles = {
        n: ParagraphStyle(
            str(n),
            parent=body,
            fontName="Helvetica-Bold",
            fontSize=size,
            leading=size + 3,
            spaceBefore=9,
            spaceAfter=8,
            keepWithNext=True,
            alignment=0,
        )
        for n, size in [(1, 19), (2, 14), (3, 11.5)]
    }
    caption = ParagraphStyle(
        "Caption", parent=body, fontSize=9, leading=11, spaceAfter=7, alignment=0
    )
    cell = ParagraphStyle("Cell", parent=body, fontSize=9.5, leading=12, spaceAfter=0, alignment=0)
    numeric_cell = ParagraphStyle("NumericCell", parent=cell, alignment=1)
    table_caption = ParagraphStyle("TableCaption", parent=caption, keepWithNext=True)
    story = []
    page_starts = []
    for number, page in enumerate(text.split("<!-- pagebreak -->"), 1):
        if number > 1:
            story.append(Spacer(1, 7))
        lines = page.strip().splitlines()
        page_starts.append(next(line for line in lines if line.strip()))
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            if line.startswith("|"):
                rows = []
                while i < len(lines) and lines[i].strip().startswith("|"):
                    row = lines[i].strip().strip("|").split("|")
                    if not all(re.fullmatch(r"[\s:-]+", item) for item in row):
                        rows.append(
                            [
                                Paragraph(inline(item), cell if j == 0 else numeric_cell)
                                for j, item in enumerate(row)
                            ]
                        )
                    i += 1
                count = len(rows[0])
                first_width = 175 if count == 3 else 135
                table = Table(
                    rows,
                    colWidths=[first_width] + [(WIDTH - first_width) / (count - 1)] * (count - 1),
                    repeatRows=1,
                    hAlign="LEFT",
                )
                table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e6edf2")),
                            ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor("#163e60")),
                            ("LINEBELOW", (0, -1), (-1, -1), 0.5, colors.grey),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("LEFTPADDING", (0, 0), (-1, -1), 5),
                            ("TOPPADDING", (0, 0), (-1, -1), 5),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                        ]
                    )
                )
                title = story.pop() if story and isinstance(story[-1], Paragraph) else None
                group = ([title] if title is not None else []) + [table, Spacer(1, 8)]
                story.append(KeepTogether(group))
                continue
            match = re.fullmatch(r"!\[[^\]]*\]\(([^)]+)\)", line)
            if match:
                target = (ROOT / "academic" / match[1]).resolve()
                if (
                    not target.is_relative_to(ROOT / "reports/figures")
                    or "final_test" in target.parts
                ):
                    raise ValueError("Only approved validation/external-video figures are allowed")
                i += 1
                while i < len(lines) and not lines[i].strip():
                    i += 1
                if i >= len(lines) or not lines[i].startswith("**Figura"):
                    raise ValueError("Every figure requires its adjacent source caption")
                story.append(
                    KeepTogether(
                        [figure(target), Spacer(1, 6), Paragraph(inline(lines[i]), caption)]
                    )
                )
                i += 1
                continue
            heading = re.match(r"^(#{1,3}) (.*)", line)
            if heading:
                story.append(Paragraph(inline(heading[2]), styles[len(heading[1])]))
                i += 1
                continue
            paragraph = [line]
            i += 1
            while i < len(lines) and lines[i].strip():
                paragraph.append(lines[i].strip())
                i += 1
            content = " ".join(paragraph)
            is_caption = content.startswith(("**Figura", "**Tabela", "Fonte:", "["))
            style = (
                table_caption if content.startswith("**Tabela") else caption if is_caption else body
            )
            story.append(Paragraph(inline(content), style))

    def footer(canvas: Canvas, doc: SimpleDocTemplate) -> None:
        canvas.setTitle("Detecção e segmentação de instâncias na análise visual de EPIs")
        canvas.setAuthor("Vinicius Pereira Gomes")
        canvas.setSubject("Final academic report; recorded scientific evidence")
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#425466"))
        canvas.drawString(
            52, 29, "Construction Safety Vision | Relatório técnico | Setembro de 2026"
        )
        canvas.drawRightString(A4[0] - 52, 29, str(doc.page))

    output = ROOT / "academic/final_report.pdf"
    doc = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        leftMargin=52,
        rightMargin=52,
        topMargin=40,
        bottomMargin=45,
        pageCompression=1,
        invariant=1,
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    reader = PdfReader(output)
    count = len(reader.pages)
    result = {
        "pages": count,
        "editorial_pages": len(page_starts),
        "bytes": output.stat().st_size,
        "body_font_pt": 11,
        "caption_font_pt": 9,
        "table_font_pt": 9.5,
    }
    print(json.dumps(result, indent=2))
    if not 6 <= count <= 10:
        raise ValueError("Editorial overflow: revise prose, not font size")
    return result


if __name__ == "__main__":
    build()
