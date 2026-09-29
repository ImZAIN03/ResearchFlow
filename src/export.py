"""Exports a ResearchFlow report (outline, synthesis, summaries, bibliography) to md/docx/pdf."""
from __future__ import annotations

from pathlib import Path
from typing import List

from docx import Document
from docx.shared import Pt
from fpdf import FPDF
from fpdf.enums import XPos, YPos

from .citation_formatter import CitationFormatter
from .llm_synthesis import PaperSummary
from .outline_generator import Outline, OutlineSection
from .paper_discovery import Paper


class Exporter:
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build_report_markdown(
        self,
        topic: str,
        outline: Outline,
        synthesis: str,
        research_gaps: str,
        summaries: List[PaperSummary],
        bibliography: str,
    ) -> str:
        lines = [f"# {outline.title}", "", f"*Research report on: {topic}*", ""]

        lines.append("## Outline")
        lines.append("")
        for section in outline.sections:
            lines.extend(self._outline_section_md(section, level=3))

        lines.append("## Literature Synthesis")
        lines.append("")
        lines.append(synthesis)
        lines.append("")

        lines.append("## Research Gaps & Future Work")
        lines.append("")
        lines.append(research_gaps)
        lines.append("")

        lines.append("## Paper Summaries")
        lines.append("")
        for s in summaries:
            lines.append(f"### {s.paper.title}")
            lines.append(f"*{', '.join(s.paper.authors) or 'Unknown'} ({s.paper.year or 'n.d.'})*")
            lines.append("")
            lines.append(s.summary)
            if s.key_findings:
                lines.append("")
                for finding in s.key_findings:
                    lines.append(f"- {finding}")
            lines.append("")

        lines.append("## Bibliography")
        lines.append("")
        lines.append(bibliography)
        return "\n".join(lines)

    @staticmethod
    def _outline_section_md(section: OutlineSection, level: int) -> List[str]:
        lines = [f"{'#' * level} {section.heading}"]
        for point in section.points:
            lines.append(f"- {point}")
        lines.append("")
        for sub in section.subsections:
            lines.extend(Exporter._outline_section_md(sub, level=min(level + 1, 6)))
        return lines

    def export_markdown(self, content: str, filename: str) -> Path:
        path = self.output_dir / filename
        path.write_text(content, encoding="utf-8")
        return path

    def export_docx(self, content: str, filename: str) -> Path:
        document = Document()
        document.styles["Normal"].font.size = Pt(11)

        for line in content.splitlines():
            stripped = line.strip()
            if line.startswith("# "):
                document.add_heading(line[2:], level=1)
            elif line.startswith("## "):
                document.add_heading(line[3:], level=2)
            elif line.startswith("### "):
                document.add_heading(line[4:], level=3)
            elif line.startswith("- "):
                document.add_paragraph(line[2:], style="List Bullet")
            elif stripped.startswith("*") and stripped.endswith("*") and len(stripped) > 1:
                paragraph = document.add_paragraph()
                run = paragraph.add_run(stripped.strip("*"))
                run.italic = True
            elif stripped:
                document.add_paragraph(line)

        path = self.output_dir / filename
        document.save(str(path))
        return path

    def export_pdf(self, content: str, filename: str) -> Path:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        cell_kwargs = {"new_x": XPos.LMARGIN, "new_y": YPos.NEXT}
        for line in content.splitlines():
            text = self._sanitize_for_pdf(line)
            if line.startswith("# "):
                pdf.set_font("Helvetica", "B", 18)
                pdf.multi_cell(0, 10, text[2:], **cell_kwargs)
            elif line.startswith("## "):
                pdf.set_font("Helvetica", "B", 14)
                pdf.multi_cell(0, 8, text[3:], **cell_kwargs)
            elif line.startswith("### "):
                pdf.set_font("Helvetica", "B", 12)
                pdf.multi_cell(0, 7, text[4:], **cell_kwargs)
            elif line.startswith("- "):
                pdf.set_font("Helvetica", "", 11)
                pdf.multi_cell(0, 6, f"  - {text[2:]}", **cell_kwargs)
            elif line.strip():
                pdf.set_font("Helvetica", "", 11)
                pdf.multi_cell(0, 6, text, **cell_kwargs)
            else:
                pdf.ln(3)

        path = self.output_dir / filename
        pdf.output(str(path))
        return path

    @staticmethod
    def _sanitize_for_pdf(text: str) -> str:
        # Core Helvetica font only supports latin-1.
        return text.encode("latin-1", errors="replace").decode("latin-1")

    def export_bibliography(self, papers: List[Paper], style: str, filename: str) -> Path:
        formatter = CitationFormatter()
        content = formatter.format_bibliography(papers, style=style)
        return self.export_markdown(content, filename)
