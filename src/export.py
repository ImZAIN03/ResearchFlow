"""Exports a ResearchFlow report (markdown from report_builder) to md/docx/pdf.

The docx and pdf writers understand the subset of markdown the report uses: #-#### headings,
bullets, *italic* lines, > notes, **bold** spans, pipe tables, ![images](path) relative to the
output directory, and the critical-thinking (and Phase 2-4) markers, which are highlighted so
they stand out.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List

from docx import Document
from docx.enum.text import WD_COLOR_INDEX
from docx.shared import Inches, Pt
from fpdf import FPDF
from fpdf.enums import XPos, YPos

from .citation_formatter import CitationFormatter
from .critical_gaps import HIGHLIGHT_PATTERN
from .paper_discovery import Paper

logger = logging.getLogger(__name__)

IMAGE_LINE = re.compile(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$")
BOLD_SPAN = re.compile(r"(\*\*[^*]+\*\*)")
# Characters outside latin-1 that the core PDF fonts can't draw, with readable substitutes.
PDF_REPLACEMENTS = str.maketrans({
    "“": '"', "”": '"', "‘": "'", "’": "'", "–": "-", "—": "-", "…": "...", "•": "-",
    # statistical symbols in the Phase 3 results
    "α": "alpha", "β": "beta", "κ": "kappa", "η": "eta", "χ": "chi", "ρ": "rho", "≤": "<=", "≥": ">=",
})
TABLE_COL_WIDTHS = (34, 22, 10, 50, 46, 28)  # mm, fits the 190 mm text width of an A4 page
TABLE_FIRST_HEADER = "Paper Title"  # the widths are for the Results & Analysis table; other tables share the width evenly


def _table_rows(lines: List[str]) -> List[List[str]]:
    """Cells of a markdown pipe table, without its |---| separator row."""
    rows = []
    for line in lines:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
            continue
        rows.append(cells)
    return rows


def _blocks(content: str):
    """Yields ("table", [lines]) for runs of table lines and ("line", line) for everything else."""
    table: List[str] = []
    for line in content.splitlines():
        if line.lstrip().startswith("|"):
            table.append(line)
            continue
        if table:
            yield "table", table
            table = []
        yield "line", line
    if table:
        yield "table", table


class Exporter:
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_all(self, content: str, slug: str, formats: List[str]) -> List[Path]:
        """Writes the report in each requested format, markdown first.

        A failure in one format is logged and skipped so it can't prevent the others; markdown is
        always written (even if not requested) when another format fails, so the run's output is
        never lost.
        """
        exporters = [
            ("md", self.export_markdown),
            ("docx", self.export_docx),
            ("pdf", self.export_pdf),
        ]
        paths: List[Path] = []
        failed = False
        for fmt, export in exporters:
            if fmt not in formats:
                continue
            try:
                paths.append(export(content, f"{slug}.{fmt}"))
            except Exception as exc:
                failed = True
                logger.error("Failed to export %s: %s", fmt, exc)
        if failed and "md" not in formats:
            paths.append(self.export_markdown(content, f"{slug}.md"))
            logger.info("Wrote markdown fallback because another format failed")
        return paths

    def export_markdown(self, content: str, filename: str) -> Path:
        path = self.output_dir / filename
        path.write_text(content, encoding="utf-8")
        return path

    def _image_path(self, target: str) -> Path:
        path = Path(target)
        return path if path.is_absolute() else self.output_dir / path

    # ---- docx -----------------------------------------------------------------------------------

    @staticmethod
    def _add_runs(paragraph, text: str, italic: bool = False) -> None:
        """Adds text with **bold** spans; critical-thinking markers are bold and highlighted."""
        for part in BOLD_SPAN.split(text):
            if not part:
                continue
            bold = part.startswith("**") and part.endswith("**")
            part = part[2:-2] if bold else part
            for piece in re.split(f"({HIGHLIGHT_PATTERN.pattern})", part):
                if not piece:
                    continue
                run = paragraph.add_run(piece)
                run.bold = bold or bool(HIGHLIGHT_PATTERN.fullmatch(piece))
                run.italic = italic
                if HIGHLIGHT_PATTERN.fullmatch(piece):
                    run.font.highlight_color = WD_COLOR_INDEX.YELLOW

    def export_docx(self, content: str, filename: str) -> Path:
        document = Document()
        document.styles["Normal"].font.size = Pt(11)

        for kind, block in _blocks(content):
            if kind == "table":
                self._docx_table(document, _table_rows(block))
                continue
            line = block
            stripped = line.strip()
            heading = re.match(r"^(#{1,4}) (.*)$", line)
            image = IMAGE_LINE.match(stripped)
            if heading:
                document.add_heading(heading.group(2).replace("**", ""), level=len(heading.group(1)))
            elif image:
                path = self._image_path(image.group(2))
                if path.exists():
                    document.add_picture(str(path), width=Inches(6.0))
                else:
                    logger.warning("Image %s not found; leaving it out of the docx", path)
            elif line.startswith("- "):
                self._add_runs(document.add_paragraph(style="List Bullet"), line[2:])
            elif stripped.startswith("> "):
                self._add_runs(document.add_paragraph(), stripped[2:], italic=True)
            elif stripped.startswith("*") and stripped.endswith("*") and not stripped.startswith("**"):
                self._add_runs(document.add_paragraph(), stripped.strip("*"), italic=True)
            elif stripped:
                self._add_runs(document.add_paragraph(), line)

        path = self.output_dir / filename
        document.save(str(path))
        return path

    def _docx_table(self, document, rows: List[List[str]]) -> None:
        if not rows:
            return
        columns = max(len(r) for r in rows)
        table = document.add_table(rows=len(rows), cols=columns)
        table.style = "Table Grid"
        for r, cells in enumerate(rows):
            for c, text in enumerate(cells):
                paragraph = table.cell(r, c).paragraphs[0]
                run = paragraph.add_run(text.replace("**", ""))
                run.font.size = Pt(8)
                run.bold = r == 0
        document.add_paragraph()

    # ---- pdf ------------------------------------------------------------------------------------

    def export_pdf(self, content: str, filename: str) -> Path:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        cell_kwargs = {"new_x": XPos.LMARGIN, "new_y": YPos.NEXT}
        heading_sizes = {1: (18, 10), 2: (14, 8), 3: (12, 7), 4: (11, 6)}
        for kind, block in _blocks(content):
            if kind == "table":
                self._pdf_table(pdf, _table_rows(block))
                continue
            line = block
            text = self._sanitize_for_pdf(line)
            stripped = text.strip()
            heading = re.match(r"^(#{1,4}) (.*)$", text)
            image = IMAGE_LINE.match(stripped)
            pdf.set_text_color(0, 0, 0)
            if heading:
                size, height = heading_sizes[len(heading.group(1))]
                pdf.set_font("Helvetica", "B", size)
                pdf.multi_cell(0, height, heading.group(2).replace("**", ""), **cell_kwargs)
            elif image:
                path = self._image_path(image.group(2))
                if path.exists():
                    pdf.image(str(path), w=pdf.epw * 0.85, x=pdf.l_margin + pdf.epw * 0.075)
                else:
                    logger.warning("Image %s not found; leaving it out of the pdf", path)
            elif stripped.startswith("> ") or (stripped.startswith("*") and stripped.endswith("*") and not stripped.startswith("**")):
                pdf.set_font("Helvetica", "I", 10)
                pdf.multi_cell(0, 6, stripped.lstrip("> ").strip("*").replace("**", ""), **cell_kwargs)
            elif stripped:
                bullet = text.startswith("- ")
                body = f"  - {text[2:]}" if bullet else text
                pdf.set_font("Helvetica", "", 11)
                if HIGHLIGHT_PATTERN.search(body):
                    pdf.set_text_color(160, 70, 0)  # markers stand out for the user to fill in
                # fpdf2's markdown treats "--" and "__" as styling too, so it is only used where safe.
                markdown = "**" in body and "--" not in body and "__" not in body
                pdf.multi_cell(0, 6, body if markdown else body.replace("**", ""), markdown=markdown, **cell_kwargs)
            else:
                pdf.ln(3)

        path = self.output_dir / filename
        pdf.output(str(path))
        return path

    def _pdf_table(self, pdf: FPDF, rows: List[List[str]]) -> None:
        if not rows:
            return
        widths = TABLE_COL_WIDTHS if rows[0][0] == TABLE_FIRST_HEADER and len(rows[0]) == len(TABLE_COL_WIDTHS) else None
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Helvetica", "", 7)
        with pdf.table(col_widths=widths, text_align="LEFT", line_height=3.5, first_row_as_headings=True) as table:
            for cells in rows:
                row = table.row()
                for text in cells:
                    row.cell(self._sanitize_for_pdf(text.replace("**", "")))
        pdf.ln(3)

    @staticmethod
    def _sanitize_for_pdf(text: str) -> str:
        # Core Helvetica font only supports latin-1.
        return text.translate(PDF_REPLACEMENTS).encode("latin-1", errors="replace").decode("latin-1")

    def export_bibliography(self, papers: List[Paper], style: str, filename: str) -> Path:
        formatter = CitationFormatter()
        content = formatter.format_bibliography(papers, style=style)
        return self.export_markdown(content, filename)
