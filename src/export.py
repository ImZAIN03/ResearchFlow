"""Exports a ResearchFlow report (outline, synthesis, summaries, bibliography) to md/docx/pdf."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List

from docx import Document
from docx.shared import Pt
from fpdf import FPDF
from fpdf.enums import XPos, YPos

from .citation_formatter import CitationFormatter
from .llm_synthesis import PaperSummary, fix_typos, split_sentences
from .outline_generator import Outline, OutlineSection
from .paper_discovery import Paper

logger = logging.getLogger(__name__)

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
        title = (outline.title if outline else "") or topic
        lines = [f"# {title}", "", f"*Research report on: {topic}*", ""]
        said: List[set] = []  # word sets of every sentence already in the report

        if outline and outline.sections:
            lines.append("## Outline")
            lines.append("")
            for section in outline.sections:
                lines.extend(self._outline_section_md(section, level=3))
                self._remember(section, said)

        lines.append("## Literature Synthesis")
        lines.append("")
        paragraphs = [self._fresh_text(p, said) for p in synthesis.split("\n\n")]
        lines.append("\n\n".join(p for p in paragraphs if p) or synthesis)
        lines.append("")

        lines.append("## Research Gaps & Future Work")
        lines.append("")
        lines.append(research_gaps)
        lines.append("")
        said.extend(self._words(s) for s in split_sentences(research_gaps))

        # Paper summaries only add what earlier sections haven't already said about each paper.
        lines.append("## Paper Summaries")
        lines.append("")
        for s in summaries:
            lines.append(f"### {s.paper.title}")
            lines.append(f"*{', '.join(s.paper.authors) or 'Unknown'} ({s.paper.year or 'n.d.'})*")
            lines.append("")
            lines.append(self._fresh_text(s.summary, said) or s.summary)
            findings = [f for f in s.key_findings if self._fresh_text(f, said)]
            if findings:
                lines.append("")
                for finding in findings:
                    lines.append(f"- {finding}")
            lines.append("")

        body = fix_typos("\n".join(lines))
        # Bibliography entries are left untouched: they must match the published titles.
        return "\n".join([body, "## Bibliography", "", bibliography])

    @staticmethod
    def _words(sentence: str) -> set:
        return set(re.findall(r"[a-z]+", sentence.lower()))

    @classmethod
    def _is_repeat(cls, sentence: str, said: List[set]) -> bool:
        """True if (nearly) all of the sentence's words already appeared together in one earlier sentence."""
        words = cls._words(sentence)
        if len(words) < 6:
            return False
        return any(len(other) >= 6 and len(words & other) / min(len(words), len(other)) >= 0.85 for other in said)

    @classmethod
    def _fresh_text(cls, text: str, said: List[set]) -> str:
        """Drops sentences already said earlier in the report and records the ones kept."""
        kept = [s for s in split_sentences(text) if not cls._is_repeat(s, said)]
        said.extend(cls._words(s) for s in kept)
        return " ".join(kept)

    @classmethod
    def _remember(cls, section: OutlineSection, said: List[set]) -> None:
        for point in section.points:
            said.extend(cls._words(s) for s in split_sentences(point))
        for sub in section.subsections:
            cls._remember(sub, said)

    @staticmethod
    def _outline_section_md(section: OutlineSection, level: int) -> List[str]:
        lines = [f"{'#' * level} {section.heading}"]
        for point in section.points:
            lines.append(f"- {point}")
        lines.append("")
        for sub in section.subsections:
            lines.extend(Exporter._outline_section_md(sub, level=min(level + 1, 6)))
        return lines

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

    def export_docx(self, content: str, filename: str) -> Path:
        document = Document()
        document.styles["Normal"].font.size = Pt(11)

        for line in content.replace("**", "").splitlines():  # markdown bold isn't rendered here
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
        for line in content.replace("**", "").splitlines():  # markdown bold isn't rendered here
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
