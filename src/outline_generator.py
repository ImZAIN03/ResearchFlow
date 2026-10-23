"""Generates the model-written sections of a literature review from a topic and literature synthesis.

The user picks which report sections to generate (REPORT_SECTIONS). The Literature Review section is
the literature synthesis itself and Results is built from the papers' metadata, so only Introduction,
Methodology (study designs), Discussion and Conclusion are written here.

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
    CITATION,
    FUTURE_WORK_REMARK,
    HYBRID_KEYWORD_LEAD,
    META_REFERENCE,
    PLACEHOLDER_PATTERN,
    SYNTH_MAX_COPIED,
    SYNTH_MIN_GROUNDING,
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
    prose_issue,
    shared_themes,
    sources_block,
    split_sentences,
    synthesis_issue,
    thesis_voice,
)

logger = logging.getLogger(__name__)


# Report sections the user can choose from (--sections), in report order: key -> heading.
REPORT_SECTIONS = {
    "introduction": "Introduction",
    "literature-review": "Literature Review",
    "methodology": "Methodology",
    "results": "Results",
    "discussion": "Discussion",
    "conclusion": "Conclusion",
}


@dataclass(frozen=True)
class SectionSpec:
    key: str  # REPORT_SECTIONS key
    heading: str
    keywords: Tuple[str, ...]  # matched against model-supplied headings to map them onto this section
    instruction: str  # section-specific writing task; {topic} and {review} (e.g. "systematic review") are filled in


# Shared by every section prompt: the register a thesis examiner expects.
SECTION_RULES = (
    "Write continuous prose in the voice of a thesis: no headings, bullet points or labels; never 'we', "
    "'our', 'this paper' or 'in this paper'; name specific concepts instead of vague words such as "
    "'various', 'important' or 'discuss'; paraphrase, never copy sentences from the papers."
)

SECTION_SPECS = [
    SectionSpec(
        "introduction",
        "Introduction",
        ("intro", "background", "overview"),
        "Write the Introduction of this {review} in 5-6 sentences (about 150 words) that follow this "
        "structure: (1) establish what {topic} is and why it matters, grounded in the papers above; "
        "(2) state which questions the papers investigate, citing them as (Author, Year), with every claim "
        "taken from the papers; (3) identify the problem or gap in that work that motivates the review; "
        "(4) close with a one-sentence thesis statement beginning 'This {review} examines' or 'This "
        "{review} argues'. " + SECTION_RULES + "\nExample of the expected style (different topic, do "
        "not reuse it): \"Antibiotic resistance now limits treatment options in routine care. Studies "
        "trace it to prescribing habits (Lee, 2020) and to agricultural use (Diaz, 2021), yet how these "
        "sources interact remains unclear. This review therefore examines what the evidence shows about "
        "each pathway and where it falls short.\"",
    ),
    SectionSpec(
        "methodology",
        "Methodology",
        ("method", "approach", "design", "data"),
        "Write about 100 words (3-4 sentences) comparing how the papers above do their research (e.g. "
        "surveys, experiments, case studies, conceptual frameworks), what evidence they rely on, and how "
        "these approaches differ in strength. Cite (Author, Year). " + SECTION_RULES,
    ),
    SectionSpec(
        "discussion",
        "Discussion",
        ("discussion", "finding", "result", "analysis"),
        "Write the Discussion of this {review} in 5-6 sentences (about 140 words): (1) state what the "
        "papers above show when read together; (2) say where their findings agree and where they "
        "disagree, citing (Author, Year) on both sides; (3) explain how their methods account for those "
        "differences; (4) state what this means for {topic}. " + SECTION_RULES,
    ),
    SectionSpec(
        "conclusion",
        "Conclusion",
        ("conclu", "summary", "final"),
        "Write the Conclusion of this {review} in 4-5 sentences (about 120 words) that draw together the "
        "whole review: (1) what the papers collectively show about {topic}, citing (Author, Year); (2) how "
        "the papers' methods support or limit those conclusions; (3) which of the research gaps listed "
        "above remain most pressing; (4) what follows for future research or practice. Introduce no new "
        "facts. " + SECTION_RULES + "\nExample of the expected style (different topic, do not reuse it): "
        "\"Taken together, the studies show that prescribing habits drive resistance more consistently than "
        "agricultural use (Lee, 2020; Diaz, 2021). Because most of this evidence is observational, causal "
        "claims remain tentative. The interaction between the two pathways is the most pressing gap, and "
        "longitudinal surveillance would allow future work to close it.\"",
    ),
]
# Sections whose prompt also gets the synthesis and research gaps, so they can draw the review together.
SYNTHESIZING_SECTIONS = {"introduction", "conclusion"}
FIRST_PERSON = re.compile(r"\b(?:[Ww]e|[Oo]ur|us)\b")  # case-sensitive "us", so "US" (the country) passes
# The Introduction frames every later claim, so it is held to a higher grounding bar than other sections.
INTRODUCTION_GROUNDING = 0.85
# "This review examines ...", "This narrative review therefore sets out to ...": the thesis statement.
THESIS_STATEMENT = re.compile(
    r"\b[Tt]his (?:\w+ )?(?:review|thesis|study|chapter|meta-analysis)\b[^.]{0,80}?\b(?:examines?|argues?|"
    r"investigates?|assesses?|sets? out|aims?|asks?|evaluates?|maps?|synthesi[sz]es?|addresses?|establish(?:es)?|"
    r"identif(?:y|ies)|shows?|explores?)\b"
)


def section_grounding(spec: Optional["SectionSpec"]) -> float:
    return INTRODUCTION_GROUNDING if spec is not None and spec.key == "introduction" else SYNTH_MIN_GROUNDING

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
        sections: Sequence[str] = tuple(REPORT_SECTIONS),
        review: str = "literature review",
    ) -> Outline:
        """Writes the chosen `sections` (REPORT_SECTIONS keys); `review` names the review type in prompts."""
        specs = [spec for spec in SECTION_SPECS if spec.key in sections]
        title, drafted = self._draft_outline(topic, synthesis, research_gaps, specs)
        context = sources_block(summaries, self.synthesizer.context_chars) or synthesis[:3000]

        # Extra sections the model invents beyond the requested ones are dropped: with small models
        # they are almost always restatements or hallucinated topics.
        outline_sections: List[OutlineSection] = []
        claimed = set()
        written: List[str] = []  # points already used, so sections can't just repeat each other
        for spec in specs:
            index = next(
                (i for i, s in enumerate(drafted) if i not in claimed and self._matches(spec, s.heading)), None
            )
            if index is not None:
                claimed.add(index)
            draft = drafted[index] if index is not None else None
            section = self._resolve_section(
                spec, draft, topic, review, context, summaries, synthesis, written, research_gaps
            )
            written.extend(section.points)
            outline_sections.append(section)
        return Outline(title=self._valid_title(title, topic), sections=outline_sections)

    def _draft_outline(
        self, topic: str, synthesis: str, research_gaps: str, specs: Sequence[SectionSpec]
    ) -> Tuple[str, List[OutlineSection]]:
        """Asks the model for a JSON outline; returns ("", []) if nothing usable comes back."""
        if not specs:
            return "", []
        system = (
            "You are a research assistant creating a structured outline for an academic research "
            "paper. Respond with ONLY valid JSON, no prose, no markdown code fences."
        )
        headings = ", ".join(f'"{spec.heading}"' for spec in specs)
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
        review: str,
        context: str,
        summaries: Sequence[PaperSummary],
        synthesis: str,
        written: Sequence[str] = (),
        research_gaps: str = "",
    ) -> OutlineSection:
        """Outline draft, else a section-specific rewrite, else a model polish of the default, else the default."""
        if draft is not None:
            points = self._clean_points([fix_against_source(p, context) for p in draft.points], written)
            problem = self._section_issue("\n".join(points), context, min_words=15, spec=spec)
            if problem is None:
                return OutlineSection(spec.heading, points, origin="llm")
            logger.info("Outline draft for %r rejected (%s)", spec.heading, problem)

        extra = self._phase_context(spec, synthesis, research_gaps)
        points = self._write_section(spec, topic, review, context, written, extra)
        if points:
            return OutlineSection(spec.heading, points, origin="llm-section")

        default = self._default_points(spec, topic, summaries, synthesis, research_gaps, review)
        default = self._clean_points(default, written) or default
        polished = self._polish_default(spec, topic, review, context, default, written, extra)
        if polished:
            logger.info("Section %r written by the model from the paper-derived outline", spec.heading)
            return OutlineSection(spec.heading, polished, origin="llm-polished")
        logger.warning("Using paper-derived default content for section %r", spec.heading)
        return OutlineSection(spec.heading, default, origin="fallback")

    @staticmethod
    def _phase_context(spec: SectionSpec, synthesis: str, research_gaps: str) -> str:
        """The synthesis and research gaps, for sections that frame or close the whole review."""
        if spec.key not in SYNTHESIZING_SECTIONS:
            return ""
        parts = []
        if synthesis:
            parts.append(f"Literature synthesis (already written):\n{normalize_whitespace(synthesis)[:900]}")
        gaps = parse_gaps(research_gaps)
        if gaps:
            parts.append("Research gaps:\n" + "\n".join(
                f"- {title}: {split_sentences(desc)[0] if split_sentences(desc) else desc}" for title, desc in gaps[:4]
            ))
        return "\n\n".join(parts)

    def _section_issue(
        self, text: str, context: str, min_words: int = 25, spec: Optional[SectionSpec] = None
    ) -> Optional[str]:
        """Shared validation for model-written sections: no copying, prose only, grounded (80%, the
        Introduction 85%), and an Introduction must state what the review sets out to do."""
        if copied_ratio(text, context) > SYNTH_MAX_COPIED:
            return "copied the source summaries instead of writing the section"
        if FIRST_PERSON.search(text):
            return "written in the first person instead of thesis voice"
        if spec is not None and spec.key == "introduction" and not THESIS_STATEMENT.search(text):
            return "has no thesis statement (what this review examines or argues)"
        return (
            synthesis_issue(text, context)
            or content_issue(text, min_sentences=2, min_words=min_words, source=context)
            or prose_issue(text, context, section_grounding(spec))
        )

    def _write_section(
        self,
        spec: SectionSpec,
        topic: str,
        review: str,
        context: str,
        written: Sequence[str] = (),
        extra: str = "",
    ) -> List[str]:
        system = (
            "You are an academic writing one section of a thesis-level literature review. Use only facts "
            "from the provided papers. Write plain sentences, no headings, no placeholder text."
        )
        user = (
            f"Research topic: {topic}\n\nPapers:\n{context}\n\n"
            + (f"{extra}\n\n" if extra else "")
            + f"Task: {spec.instruction.format(topic=topic, review=review)}"
        )
        source = f"{context}\n{extra}"

        def validate(text: str) -> Optional[str]:
            points = self._clean_points(split_sentences(text), written)
            return self._section_issue("\n".join(points), source, spec=spec)

        text = self.synthesizer.complete_validated(
            system, user, max_tokens=320, min_sentences=2, validator=validate, label=f"{spec.heading} section"
        )
        return self._finish(text, source, written)

    def _polish_default(
        self,
        spec: SectionSpec,
        topic: str,
        review: str,
        context: str,
        default: Sequence[str],
        written: Sequence[str] = (),
        extra: str = "",
    ) -> List[str]:
        """Asks the model to rewrite the paper-derived default as thesis prose: an easier task than writing it.

        The rewrite keeps the default's facts and citations, so it is grounded against the papers plus
        the default itself. Returns [] if the rewrite is rejected.
        """
        outline = " ".join(default)
        system = (
            "You are an academic editor. Rewrite draft text as polished thesis prose without adding facts."
        )
        user = (
            f"Research topic: {topic}\n\nPapers:\n{context}\n\n"
            + (f"{extra}\n\n" if extra else "")
            + f"Draft {spec.heading}:\n{outline}\n\n"
            f"Task: Rewrite the draft {spec.heading} of this {review} as 4-5 sentences of fluent, connected "
            "thesis prose. Keep every fact and every (Author, Year) citation from the draft, link the "
            f"sentences into an argument about {topic}, and add nothing the papers do not support. "
            + SECTION_RULES
        )
        source = f"{context}\n{extra}\n{outline}"
        required = [author for author, _ in CITATION.findall(outline)]

        def validate(text: str) -> Optional[str]:
            points = self._clean_points(split_sentences(text), written)
            joined = "\n".join(points)
            missing = [a for a in dict.fromkeys(required) if a not in joined]
            if missing:
                return f"dropped citations from the draft ({', '.join(missing[:3])})"
            # Copying the draft back is allowed (it is ours); copying the papers is not.
            if copied_ratio(joined, context) > SYNTH_MAX_COPIED:
                return "copied the source summaries instead of writing the section"
            if FIRST_PERSON.search(joined):
                return "written in the first person instead of thesis voice"
            if spec.key == "introduction" and not THESIS_STATEMENT.search(joined):
                return "dropped the thesis statement from the draft"
            return (
                synthesis_issue(joined, source)
                or content_issue(joined, min_sentences=3, min_words=40, source=source)
                or prose_issue(joined, source, section_grounding(spec))
            )

        text = self.synthesizer.complete_validated(
            system, user, max_tokens=320, min_sentences=3, retries=0, validator=validate,
            label=f"{spec.heading} polish",
        )
        return self._finish(text, source, written)

    def _finish(self, text: str, source: str, written: Sequence[str]) -> List[str]:
        """Accepted model text as section points, in thesis voice, without repeats of earlier sections."""
        if not text:
            return []
        text = thesis_voice(fix_against_source(text, source), first_person=False)
        return self._clean_points(split_sentences(text), written)

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
        research_gaps: str = "",
        review: str = "literature review",
    ) -> List[str]:
        if not summaries:
            return self._clean_points(pick_sentences(synthesis, limit=3)) or [
                f"This section concerns {topic}; no paper summaries were available to draw on."
            ]
        summaries = list(summaries)
        gap_titles = [title for title, _ in parse_gaps(research_gaps) if title and " by " not in title]
        builders = {
            "introduction": lambda: self._default_introduction(topic, summaries, gap_titles, review),
            "methodology": lambda: self._default_methodology(topic, summaries),
            "discussion": lambda: self._default_findings(topic, summaries),
            "conclusion": lambda: self._default_conclusion(topic, summaries, gap_titles, review),
        }
        return self._clean_points(builders[spec.key]())

    @staticmethod
    def _year_range(summaries: List[PaperSummary]) -> str:
        years = sorted(s.paper.year for s in summaries if s.paper.year)
        if not years:
            return ""
        return f" published in {years[0]}" if years[0] == years[-1] else f" published between {years[0]} and {years[-1]}"

    @staticmethod
    def _lower_first(text: str) -> str:
        return text[0].lower() + text[1:] if text[1:2].islower() else text

    def _default_introduction(
        self, topic: str, summaries: List[PaperSummary], gap_titles: Sequence[str] = (), review: str = "literature review"
    ) -> List[str]:
        # Thesis voice, built from the papers' subjects, themes and gaps rather than their abstracts, so
        # the abstract sentences stay available for the Literature Review and Discussion sections
        # (sections may not repeat each other's points). Structure: context, questions, problem, aim.
        foci = join_items(f"{paper_focus(s.paper)} {cite(s.paper)}" for s in summaries[:4])
        points = [
            f"Research on {topic} has developed along several lines, from {foci}"
            + (f", with {len(summaries) - 4} further studies extending this work" if len(summaries) > 4 else "") + "."
        ]
        themes = shared_themes(summaries, exclude=topic)
        if themes:
            points.append(
                f"What connects these studies{self._year_range(summaries)} is a shared concern with "
                f"{join_items(themes)}, which frames the questions they ask about {topic}."
            )
        designs = sorted({method_profile(s.paper) for s in summaries})
        if gap_titles:
            points.append(
                f"Yet the literature leaves open {join_items(self._lower_first(t) for t in gap_titles[:3])}, "
                "so its conclusions cannot yet be taken as settled."
            )
        else:
            points.append(
                f"Yet the evidence rests on {join_items(designs)}, so it remains unclear how far its conclusions "
                "hold beyond the settings studied."
            )
        points.append(
            f"This {review} therefore examines what the existing studies establish about {topic}, how their "
            "methods shape those conclusions, and where further research is needed."
        )
        return points

    # Defaults compare the papers (themes, research designs) rather than pasting abstract sentences,
    # which the report already shows under Paper Summaries.

    @staticmethod
    def _default_methodology(topic: str, summaries: List[PaperSummary]) -> List[str]:
        # The search and screening protocol is described separately (review_methodology.protocol_paragraphs).
        points = []
        groups: dict = {}
        for s in summaries:
            groups.setdefault(method_profile(s.paper), []).append(f"{s.paper.first_author_surname}, {s.paper.year or 'n.d.'}")
        points.append(
            f"The {len(summaries)} included papers fall into the following research designs: "
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
                and not sentence.startswith(HYBRID_KEYWORD_LEAD)  # a keyword list, not a finding
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

    def _default_conclusion(
        self, topic: str, summaries: List[PaperSummary], gap_titles: Sequence[str] = (), review: str = "literature review"
    ) -> List[str]:
        # Draws the review together: what the literature shows, how its methods qualify that, which
        # gaps remain and what follows, in that order.
        themes = shared_themes(summaries, exclude=topic)
        designs = sorted({method_profile(s.paper) for s in summaries})
        cites = "; ".join(cite(s.paper).strip("()") for s in summaries[:5])
        points = [
            f"Taken together, the {len(summaries)} studies in this {review} show that research on {topic} "
            f"centres on {join_items(themes) if themes else 'a small set of recurring concerns'} ({cites}).",
            f"Because this evidence comes from {join_items(designs)}, the field's conclusions rest on a "
            f"{'narrow' if len(designs) == 1 else 'mixed'} methodological base and should be read with that in mind.",
        ]
        if gap_titles:
            points.append(
                f"The most pressing open questions concern {join_items(self._lower_first(t) for t in gap_titles[:3])}, "
                "which no reviewed study resolves."
            )
        points.append(
            f"Closing these gaps with studies designed to test the existing claims directly would give research "
            f"and practice on {topic} a firmer footing than the current literature provides."
        )
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
