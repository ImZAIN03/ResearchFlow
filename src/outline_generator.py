"""Generates a structured research paper outline from a topic and literature synthesis."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import List

from .llm_synthesis import LLMSynthesizer


@dataclass
class OutlineSection:
    heading: str
    points: List[str] = field(default_factory=list)
    subsections: List["OutlineSection"] = field(default_factory=list)


@dataclass
class Outline:
    title: str
    sections: List[OutlineSection] = field(default_factory=list)


class OutlineGenerator:
    def __init__(self, synthesizer: LLMSynthesizer):
        self.synthesizer = synthesizer

    def generate_outline(self, topic: str, synthesis: str, research_gaps: str = "") -> Outline:
        system = (
            "You are a research assistant creating a structured outline for an academic research "
            "paper. Respond with ONLY valid JSON, no prose, no markdown code fences."
        )
        user = (
            f"Research topic: {topic}\n\n"
            f"Literature synthesis:\n{synthesis}\n\n"
            f"Research gaps:\n{research_gaps}\n\n"
            "Produce a JSON object with this exact shape:\n"
            "{\n"
            '  "title": "<a working title for the paper>",\n'
            '  "sections": [\n'
            '    {"heading": "<section heading>", "points": ["<bullet point>", ...],\n'
            '     "subsections": [{"heading": "<sub-heading>", "points": ["..."], "subsections": []}]}\n'
            "  ]\n"
            "}\n"
            "Include standard sections: Introduction, Literature Review, Methodology, "
            "Discussion/Findings, Research Gaps & Future Work, and Conclusion."
        )
        raw = self.synthesizer.complete(system, user, max_tokens=2000)
        data = self._parse_json(raw)
        return self._outline_from_dict(data)

    @staticmethod
    def _parse_json(raw: str) -> dict:
        cleaned = raw.strip()
        cleaned = re.sub(r"^```(json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
        return json.loads(cleaned)

    @classmethod
    def _outline_from_dict(cls, data: dict) -> Outline:
        return Outline(
            title=data.get("title", "Untitled"),
            sections=[cls._section_from_dict(s) for s in data.get("sections", []) or []],
        )

    @classmethod
    def _section_from_dict(cls, data: dict) -> OutlineSection:
        return OutlineSection(
            heading=data.get("heading", ""),
            points=data.get("points", []) or [],
            subsections=[cls._section_from_dict(s) for s in data.get("subsections", []) or []],
        )

    def to_markdown(self, outline: Outline) -> str:
        lines = [f"# {outline.title}", ""]
        for section in outline.sections:
            lines.extend(self._section_to_markdown(section, level=2))
        return "\n".join(lines)

    def _section_to_markdown(self, section: OutlineSection, level: int) -> List[str]:
        lines = [f"{'#' * level} {section.heading}"]
        for point in section.points:
            lines.append(f"- {point}")
        lines.append("")
        for sub in section.subsections:
            lines.extend(self._section_to_markdown(sub, level=min(level + 1, 6)))
        return lines
