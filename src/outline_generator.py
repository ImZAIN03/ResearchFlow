"""Generates a structured research paper outline from a topic and literature synthesis.

The model is first asked for a whole outline as JSON. Small models often return truncated or
malformed JSON, or fill it with placeholder text, so the draft is parsed leniently and every
standard section is validated on its own. A weak section is rewritten with a section-specific
prompt, and if that also fails it gets default content built from the papers' own titles,
abstracts and summaries, so the report never contains empty or "Lorem ipsum" sections.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from .llm_synthesis import (
    FUTURE_WORK_REMARK,
    META_REFERENCE,
    PLACEHOLDER_PATTERN,
    SYNTH_MAX_COPIED,
    LLMSynthesizer,
    PaperSummary,
    cite,
    clean_response,
    content_issue,
    copied_ratio,
    fix_against_source,
    join_items,
    method_profile,
    normalize_whitespace,
    paper_focus,
    parse_gaps,
    pick_sentences,
    shared_themes,
    sources_block,
    split_sentences,
    synthesis_issue,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SectionSpec:
    heading: str
    keywords: Tuple[str, ...]  # matched against model-supplied headings to map them onto this section
    instruction: str  # section-specific writing task; {topic} is filled in


SECTION_SPECS = [
    SectionSpec(
        "Introduction",
        ("intro", "background", "overview"),
        "Write the Introduction in about 100 words (3-4 sentences): explain what {topic} is, why it "
        "matters, and which questions the papers above investigate.",
    ),
    SectionSpec(
        "Literature Review",
        ("literature", "review", "related"),
        "Write the Literature Review in about 120 words (3-5 sentences), organized by theme rather "
        "than paper by paper: which themes the papers above share and how their perspectives differ. "
        "Name each paper by its (Author, Year) tag. Use your own words, do not copy sentences.",
    ),
    SectionSpec(
        "Methodology",
        ("method", "approach", "design", "data"),
        "Write the Methodology section in about 100 words (3-4 sentences): compare how the papers "
        "above do their research (e.g. surveys, experiments, case studies, conceptual frameworks), what "
        "evidence they rely on, and how these approaches differ in strength. Cite (Author, Year). Use "
        "your own words, do not copy sentences.",
    ),
    SectionSpec(
        "Discussion/Findings",
        ("discussion", "finding", "result", "analysis"),
        "Write the Discussion/Findings section in about 100 words (3-4 sentences): explain what the "
        "papers above show when read together, where their findings agree or disagree, and what that "
        "means for {topic}. Cite (Author, Year). Use your own words, do not copy sentences.",
    ),
    SectionSpec(
        "Research Gaps & Future Work",
        ("gap", "future", "open", "limitation"),
        "Write the Research Gaps & Future Work section: list 3 specific open problems in {topic} "
        "suggested by the papers above, each as one full sentence on its own line starting with '- '.",
    ),
    SectionSpec(
        "Conclusion",
        ("conclu", "summary", "final"),
        "Write the Conclusion in about 80 words (3 sentences): summarize what the papers above "
        "collectively show about {topic} and why it matters.",
    ),
]
FALLBACK_SECTIONS = [spec.heading for spec in SECTION_SPECS]

FINDING_KEYWORDS = (
    "we show", "we find", "we found", "results", "demonstrate", "reveal", "suggest", "indicate",
    "outperform", "improve", "found that", "shows that",
)
# Sentences that only make sense inside their original abstract ("We conclude our survey...",
# "The latter suggests...") and can't be quoted on their own.
CONTEXT_DEPENDENT = re.compile(r"^(?:we|our|in this (?:paper|work)|this (?:paper|work|survey)|the (?:latter|former)|these|it|they)\b", re.IGNORECASE)
STOPWORDS = set(
    "a an and are as at be by for from in into is it of on or the to towards toward with via using "
    "its their our this that these those based new study analysis approach approaches paper review "
    "case towards toward through between under over can how what why when".split()
)


@dataclass
class OutlineSection:
    heading: str
    points: List[str] = field(default_factory=list)
    subsections: List["OutlineSection"] = field(default_factory=list)
    origin: str = "llm"  # "llm" (outline draft), "llm-section" (section rewrite) or "fallback"


@dataclass
class Outline:
    title: str
    sections: List[OutlineSection] = field(default_factory=list)


class OutlineGenerator:
    def __init__(self, synthesizer: LLMSynthesizer):
        self.synthesizer = synthesizer

    def generate_outline(
        self,
        topic: str,
        synthesis: str,
        research_gaps: str = "",
        summaries: Sequence[PaperSummary] = (),
    ) -> Outline:
        title, drafted = self._draft_outline(topic, synthesis, research_gaps)
        context = sources_block(summaries) or synthesis[:3000]

        # Extra sections the model invents beyond the standard six are dropped: with small models
        # they are almost always restatements or hallucinated topics.
        sections: List[OutlineSection] = []
        claimed = set()
        written: List[str] = []  # points already used, so sections can't just repeat each other
        for spec in SECTION_SPECS:
            index = next(
                (i for i, s in enumerate(drafted) if i not in claimed and self._matches(spec, s.heading)), None
            )
            if index is not None:
                claimed.add(index)
            draft = drafted[index] if index is not None else None
            section = self._resolve_section(spec, draft, topic, context, research_gaps, summaries, synthesis, written)
            written.extend(section.points)
            sections.append(section)
        return Outline(title=self._valid_title(title, topic), sections=sections)

    def _draft_outline(self, topic: str, synthesis: str, research_gaps: str) -> Tuple[str, List[OutlineSection]]:
        """Asks the model for a JSON outline; returns ("", []) if nothing usable comes back."""
        system = (
            "You are a research assistant creating a structured outline for an academic research "
            "paper. Respond with ONLY valid JSON, no prose, no markdown code fences."
        )
        headings = ", ".join(f'"{h}"' for h in FALLBACK_SECTIONS)
        user = (
            f"Research topic: {topic}\n\n"
            f"Literature synthesis:\n{synthesis[:2200]}\n\n"
            f"Research gaps:\n{research_gaps[:600]}\n\n"
            "Produce a JSON object with a \"title\" string and a \"sections\" list. Each section has "
            "a \"heading\" and a \"points\" list of 2-3 full sentences about the topic, taken from "
            f"the synthesis above. Use these headings in order: {headings}. Example of one section:\n"
            '{"heading": "Introduction", "points": ["<sentence about the topic>", "<another sentence>"]}\n'
            "Replace every <...> with real content. Never write Lorem ipsum."
        )
        raw = self.synthesizer.complete(system, user, max_tokens=700)
        try:
            outline = self._outline_from_dict(self._parse_json(raw))
        except (ValueError, TypeError, AttributeError) as exc:
            logger.warning("Could not parse outline JSON from model (%s); writing sections individually", exc)
            logger.debug("Raw outline response: %r", raw)
            return "", []
        return outline.title, outline.sections

    def _resolve_section(
        self,
        spec: SectionSpec,
        draft: Optional[OutlineSection],
        topic: str,
        context: str,
        research_gaps: str,
        summaries: Sequence[PaperSummary],
        synthesis: str,
        written: Sequence[str] = (),
    ) -> OutlineSection:
        # Research gaps were already generated earlier in the run. The outline lists their names;
        # the full descriptions appear once, in the report's Research Gaps & Future Work section.
        if spec.heading.startswith("Research Gaps"):
            points = self._gap_points(research_gaps)
            if points:
                return OutlineSection(spec.heading, points, origin="gaps")

        if draft is not None:
            points = self._clean_points([fix_against_source(p, context) for p in draft.points], written)
            problem = synthesis_issue("\n".join(points), context) or content_issue(
                "\n".join(points), min_sentences=2, min_words=15, source=context
            )
            if problem is None and copied_ratio("\n".join(points), context) > SYNTH_MAX_COPIED:
                problem = "copied the source summaries"
            if problem is None:
                return OutlineSection(spec.heading, points, origin="llm")
            logger.info("Outline draft for %r rejected (%s)", spec.heading, problem)

        points = self._write_section(spec, topic, context, written)
        if points:
            return OutlineSection(spec.heading, points, origin="llm-section")

        logger.warning("Using paper-derived default content for section %r", spec.heading)
        points = self._default_points(spec, topic, summaries, synthesis, research_gaps)
        return OutlineSection(spec.heading, self._clean_points(points, written) or points, origin="fallback")

    def _write_section(self, spec: SectionSpec, topic: str, context: str, written: Sequence[str] = ()) -> List[str]:
        system = (
            "You are a research assistant writing one section of a literature-review paper. Use only "
            "facts from the provided papers. Write plain sentences, no headings, no placeholder text."
        )
        user = f"Research topic: {topic}\n\nPapers:\n{context}\n\nTask: {spec.instruction.format(topic=topic)}"

        def validate(text: str) -> Optional[str]:
            points = self._clean_points(split_sentences(text), written)
            if copied_ratio("\n".join(points), context) > SYNTH_MAX_COPIED:
                return "copied the source summaries instead of writing the section"
            return synthesis_issue(text, context) or content_issue("\n".join(points), min_sentences=2, min_words=25, source=context)

        text = self.synthesizer.complete_validated(
            system, user, max_tokens=250, min_sentences=2, validator=validate, label=f"{spec.heading} section"
        )
        return self._clean_points(split_sentences(fix_against_source(text, context)), written) if text else []

    @staticmethod
    def _gap_points(research_gaps: str) -> List[str]:
        points = []
        for title, desc in parse_gaps(research_gaps):
            first = split_sentences(desc)[:1]
            point = title or (first[0] if first else "")
            if point:
                points.append(point.rstrip("."))
        return points

    @staticmethod
    def _clean_points(points: Sequence[str], already_written: Sequence[str] = ()) -> List[str]:
        """Drops placeholder, prompt-echo and too-short points, and near-duplicates of earlier ones."""
        cleaned: List[str] = []
        seen = [set(re.findall(r"[a-z]+", p.lower())) for p in already_written]
        for point in points:
            point = clean_response(normalize_whitespace(str(point))).lstrip("-*• ").strip()
            if (
                not point
                or len(point.split()) < 4
                or PLACEHOLDER_PATTERN.search(point)
                or META_REFERENCE.search(point)
                or point.endswith(":")  # a label introducing a list ("Differences between papers:")
                or re.match(r"^themes?\s*:", point, re.IGNORECASE)
            ):
                continue
            words = set(re.findall(r"[a-z]+", point.lower()))
            if any(len(words & other) / max(1, len(words | other)) > 0.7 for other in seen):
                continue
            seen.append(words)
            cleaned.append(point)
        return cleaned

    @staticmethod
    def _matches(spec: SectionSpec, heading: str) -> bool:
        heading = heading.lower()
        return spec.heading.lower() == heading or any(k in heading for k in spec.keywords)

    @staticmethod
    def _valid_title(title: str, topic: str) -> str:
        title = normalize_whitespace(re.sub(r"^\[[^\]]*\]\s*", "", title)).strip("\"'")
        # A title that shares no word with the topic is usually about something the model invented.
        topic_words = {w for w in re.findall(r"[a-z]{3,}", topic.lower()) if w not in STOPWORDS}
        off_topic = bool(topic_words) and not topic_words & set(re.findall(r"[a-z]{3,}", title.lower()))
        if (
            not title
            or title.lower() == "untitled"
            or off_topic
            or PLACEHOLDER_PATTERN.search(title)
            or not 2 <= len(title.split()) <= 25
        ):
            return f"{topic.strip().title()}: A Review of Current Research"
        return title

    # ---- default content built from the papers themselves -------------------------------------

    def _default_points(
        self,
        spec: SectionSpec,
        topic: str,
        summaries: Sequence[PaperSummary],
        synthesis: str,
        research_gaps: str,
    ) -> List[str]:
        if not summaries:
            return self._clean_points(pick_sentences(synthesis, limit=3)) or [
                f"This section discusses {topic}; no paper summaries were available to draw on."
            ]
        if spec.heading.startswith("Research Gaps"):
            return self._default_gaps(topic, list(summaries), research_gaps)  # short names, not sentences
        builders = {
            "Introduction": self._default_introduction,
            "Literature Review": self._default_literature_review,
            "Methodology": self._default_methodology,
            "Discussion/Findings": self._default_findings,
            "Conclusion": self._default_conclusion,
        }
        return self._clean_points(builders[spec.heading](topic, list(summaries)))

    @staticmethod
    def _year_range(summaries: List[PaperSummary]) -> str:
        years = sorted(s.paper.year for s in summaries if s.paper.year)
        if not years:
            return ""
        return f" published in {years[0]}" if years[0] == years[-1] else f" published between {years[0]} and {years[-1]}"

    def _default_introduction(self, topic: str, summaries: List[PaperSummary]) -> List[str]:
        # Built from titles and metadata only, so the abstract sentences stay available for the
        # Literature Review and Discussion sections (sections may not repeat each other's points).
        titles = "; ".join(f"\"{s.paper.title}\" {cite(s.paper)}" for s in summaries[:4])
        points = [f"This report reviews {len(summaries)} papers on {topic}{self._year_range(summaries)}: {titles}."]
        themes = shared_themes(summaries, exclude=topic)
        if themes:
            points.append(f"Recurring themes across these works include {join_items(themes)}.")
        venues = sorted({s.paper.venue for s in summaries if s.paper.venue})
        if venues:
            points.append(f"The papers appeared in {', '.join(venues[:4])}.")
        return points

    # Defaults compare the papers (themes, research designs) rather than pasting abstract sentences,
    # which the report already shows under Paper Summaries.

    @staticmethod
    def _default_literature_review(topic: str, summaries: List[PaperSummary]) -> List[str]:
        points = [
            f"The literature on {topic} is approached from {len(summaries)} distinct angles: "
            + join_items(f"{paper_focus(s.paper)} {cite(s.paper)}" for s in summaries) + "."
        ]
        themes = shared_themes(summaries, exclude=topic)
        if themes:
            using = [s for s in summaries if themes[0] in f"{s.paper.title} {s.paper.abstract} {s.summary}".lower()]
            points.append(
                f"The most widely shared theme is {themes[0]}, taken up by "
                + join_items(f"{s.paper.first_author_surname} ({s.paper.year or 'n.d.'})" for s in using)
                + (f", alongside related concerns with {join_items(themes[1:])}." if themes[1:] else ".")
            )
        if len(summaries) > 1:
            points.append(
                "Rather than building directly on one another, these works develop in parallel, each "
                "framing the problem through its own domain and terminology."
            )
        return points

    @staticmethod
    def _default_methodology(topic: str, summaries: List[PaperSummary]) -> List[str]:
        sources = sorted({s.paper.source for s in summaries if s.paper.source}) or ["academic databases"]
        points = [
            f"This review retrieved papers from {', '.join(sources)} with the query \"{topic}\" and compared "
            f"the {len(summaries)} most relevant by theme and research design."
        ]
        groups: dict = {}
        for s in summaries:
            groups.setdefault(method_profile(s.paper), []).append(f"{s.paper.first_author_surname}, {s.paper.year or 'n.d.'}")
        points.append(
            "The reviewed papers fall into the following research designs: "
            + "; ".join(f"{label} ({'; '.join(names)})" for label, names in groups.items()) + "."
        )
        if any(label != "empirical studies" for label in groups):
            points.append(
                "Non-empirical designs such as reviews and frameworks map the field and set out principles, "
                "but they offer limited direct evidence of how well those principles work in practice."
            )
        if "empirical studies" in groups:
            points.append(
                "The empirical work provides measurable evidence, although its conclusions are tied to the "
                "specific data and settings it studies."
            )
        return points

    @staticmethod
    def _default_findings(topic: str, summaries: List[PaperSummary]) -> List[str]:
        points = []
        themes = shared_themes(summaries, exclude=topic, limit=2)
        if themes and len(summaries) > 1:
            points.append(
                f"Read together, the papers converge on {join_items(themes)} as the issues that matter most for {topic}."
            )
        def quotable(sentence: str) -> bool:
            return (
                len(sentence.split()) >= 4
                and not sentence.rstrip().endswith("?")
                and not CONTEXT_DEPENDENT.match(sentence)
                and not FUTURE_WORK_REMARK.search(sentence)
            )

        for s in summaries:
            candidates = (
                s.key_findings,
                pick_sentences(s.paper.abstract or s.summary, FINDING_KEYWORDS, limit=3),
                pick_sentences(s.summary, limit=4)[1:],  # past the opening sentence, which sets the scene
            )
            finding = next((f for group in candidates for f in group if quotable(f)), None)
            if finding:
                points.append(f"{finding.rstrip('.')} {cite(s.paper)}.")
        return points

    def _default_gaps(self, topic: str, summaries: List[PaperSummary], research_gaps: str) -> List[str]:
        return self._gap_points(research_gaps) or self._gap_points(
            LLMSynthesizer.fallback_research_gaps(topic, summaries)
        )

    def _default_conclusion(self, topic: str, summaries: List[PaperSummary]) -> List[str]:
        themes = shared_themes(summaries, exclude=topic)
        designs = sorted({method_profile(s.paper) for s in summaries})
        points = [
            f"Across the {len(summaries)} papers reviewed, research on {topic} centres on "
            f"{join_items(themes) if themes else 'a small set of recurring themes'}.",
            f"The evidence comes from {join_items(designs)}, so the field's conclusions rest on a "
            f"{'narrow' if len(designs) == 1 else 'mixed'} methodological base.",
            f"Together these works indicate that {topic} remains an active area where the open "
            "questions identified in this review warrant further study.",
        ]
        return points

    # ---- lenient JSON parsing -----------------------------------------------------------------

    @classmethod
    def _parse_json(cls, raw: str) -> dict:
        cleaned = (raw or "").strip()
        if not cleaned:
            raise ValueError("empty response")
        fenced = re.search(r"```(?:json)?\s*(.*?)(?:```|$)", cleaned, re.DOTALL)
        if fenced:
            cleaned = fenced.group(1).strip()
        start = cleaned.find("{")
        if start == -1:
            raise ValueError("no JSON object in response")
        candidate = cleaned[start:]
        end = candidate.rfind("}")
        attempts = [candidate, candidate[: end + 1] if end != -1 else "", cls._repair_truncated(candidate)]
        for attempt in attempts:
            if not attempt:
                continue
            try:
                data = json.loads(re.sub(r",\s*([}\]])", r"\1", attempt))  # tolerate trailing commas
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                return data
        salvaged = cls._salvage_sections(candidate)
        if salvaged["sections"]:
            logger.info("Recovered %d sections from malformed outline JSON", len(salvaged["sections"]))
            return salvaged
        raise ValueError("malformed JSON that could not be repaired")

    @staticmethod
    def _repair_truncated(text: str) -> str:
        """Closes strings/brackets left open when the model ran out of tokens mid-JSON."""
        closers: List[str] = []
        in_string = escaped = False
        for ch in text:
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
            elif ch == '"':
                in_string = True
            elif ch in "{[":
                closers.append("}" if ch == "{" else "]")
            elif ch in "}]" and closers:
                closers.pop()
        repaired = text + ('"' if in_string else "")
        repaired = re.sub(r"[\s,:]+$", "", repaired)
        if closers and closers[-1] == "}":
            repaired = re.sub(r',\s*"[^"]*"$', "", repaired)  # dangling key with no value
            repaired = re.sub(r'\{\s*"[^"]*"$', "{", repaired)
        return repaired + "".join(reversed(closers))

    @staticmethod
    def _salvage_sections(text: str) -> dict:
        """Last resort: pulls "heading"/"points" pairs out of JSON too broken to parse."""
        string = r'"((?:[^"\\]|\\.)*)"'
        title = re.search(r'"title"\s*:\s*' + string, text)
        headings = list(re.finditer(r'"heading"\s*:\s*' + string, text))
        sections = []
        for i, match in enumerate(headings):
            chunk = text[match.end() : headings[i + 1].start() if i + 1 < len(headings) else len(text)]
            points_match = re.search(r'"points"\s*:\s*\[(.*?)(?:\]|$)', chunk, re.DOTALL)
            points = re.findall(string, points_match.group(1)) if points_match else []
            sections.append({"heading": match.group(1), "points": points})
        return {"title": title.group(1) if title else "", "sections": sections}

    @classmethod
    def _outline_from_dict(cls, data: dict) -> Outline:
        if not isinstance(data, dict):
            raise ValueError(f"expected a JSON object, got {type(data).__name__}")
        return Outline(
            title=str(data.get("title") or ""),
            sections=[cls._section_from_dict(s) for s in cls._as_list(data.get("sections"))],
        )

    @classmethod
    def _section_from_dict(cls, data) -> OutlineSection:
        if not isinstance(data, dict):
            return OutlineSection(heading=str(data))
        return OutlineSection(
            heading=str(data.get("heading") or data.get("title") or ""),
            points=[str(p) for p in cls._as_list(data.get("points") or data.get("content"))],
            subsections=[cls._section_from_dict(s) for s in cls._as_list(data.get("subsections"))],
        )

    @staticmethod
    def _as_list(value) -> list:
        if value is None:
            return []
        if isinstance(value, str):
            return split_sentences(value)  # model wrote prose instead of a list
        return value if isinstance(value, list) else [value]

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
