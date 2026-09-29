"""Uses Claude to summarize individual papers and synthesize findings across a literature set."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import anthropic

from .paper_discovery import Paper
from .text_extraction import TextExtractor

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-5"
MAX_SOURCE_CHARS = 12000


@dataclass
class PaperSummary:
    paper: Paper
    summary: str
    key_findings: List[str] = field(default_factory=list)


class LLMSynthesizer:
    """Thin wrapper around the Anthropic Messages API for research synthesis tasks."""

    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_MODEL):
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "An Anthropic API key is required. Set the ANTHROPIC_API_KEY environment "
                "variable or pass api_key= explicitly."
            )
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def complete(self, system: str, user: str, max_tokens: int = 1500) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in response.content if block.type == "text").strip()

    def summarize_paper(self, paper: Paper, full_text: str) -> PaperSummary:
        source_text = (full_text or paper.abstract)[:MAX_SOURCE_CHARS]
        system = (
            "You are a research assistant. Summarize academic papers accurately and concisely. "
            "Never invent findings that are not supported by the provided text."
        )
        user = (
            f"Paper title: {paper.title}\n"
            f"Authors: {', '.join(paper.authors) or 'Unknown'}\n"
            f"Year: {paper.year or 'Unknown'}\n\n"
            f"Text:\n{source_text}\n\n"
            "Write a 3-5 sentence summary of this paper's core contribution, methodology, and results. "
            "Then list 3-5 key findings as short bullet points, each on its own line prefixed with '- '."
        )
        raw = self.complete(system, user)
        summary, findings = self._split_summary_and_findings(raw)
        return PaperSummary(paper=paper, summary=summary, key_findings=findings)

    @staticmethod
    def _split_summary_and_findings(raw: str) -> Tuple[str, List[str]]:
        summary_lines: List[str] = []
        findings: List[str] = []
        for line in raw.splitlines():
            stripped = line.strip()
            if stripped.startswith("-") or stripped.startswith("*"):
                findings.append(stripped.lstrip("-*").strip())
            elif stripped:
                summary_lines.append(stripped)
        return " ".join(summary_lines), findings

    def synthesize_literature(self, topic: str, summaries: List[PaperSummary]) -> str:
        system = (
            "You are a research assistant writing a literature synthesis for a research paper. "
            "Organize findings thematically, note points of agreement and disagreement between "
            "sources, and cite papers inline using (Author, Year) format."
        )
        sources_block = "\n\n".join(
            f"[{s.paper.first_author_surname}, {s.paper.year or 'n.d.'}] {s.paper.title}\n"
            f"Summary: {s.summary}\n"
            f"Key findings: {'; '.join(s.key_findings) if s.key_findings else 'None extracted'}"
            for s in summaries
        )
        user = (
            f"Research topic: {topic}\n\n"
            f"Here are summaries of {len(summaries)} relevant papers:\n\n{sources_block}\n\n"
            "Write a cohesive, thematically organized literature synthesis (4-8 paragraphs) that "
            "connects these sources, highlights consensus and disagreement, and identifies open "
            "questions."
        )
        return self.complete(system, user, max_tokens=3000)

    def generate_research_gaps(self, topic: str, synthesis: str) -> str:
        system = "You are a research assistant identifying gaps and future research directions."
        user = (
            f"Research topic: {topic}\n\nLiterature synthesis:\n{synthesis}\n\n"
            "Based on this synthesis, list 4-6 specific, actionable research gaps or open questions "
            "as bullet points, each on its own line prefixed with '- '."
        )
        return self.complete(system, user, max_tokens=800)

    def summarize_all(self, papers: List[Paper], extractor: TextExtractor) -> List[PaperSummary]:
        summaries = []
        for paper in papers:
            try:
                text = extractor.extract_text_for_paper(paper)
                summaries.append(self.summarize_paper(paper, text))
            except Exception as exc:
                logger.warning("Skipping summary for %r due to error: %s", paper.title, exc)
        return summaries
