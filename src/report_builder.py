"""Assembles the literature review as markdown, from the generated sections and the gap markers.

Section order follows REPORT_SECTIONS; only the sections the user chose are included. Prose comes
from the model (validated upstream) or from paper-derived fallbacks; the critical-thinking markers
from critical_gaps are inserted here, after validation, at the points where they apply.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from . import critical_gaps as gaps
from .llm_synthesis import PaperSummary, ensure_coverage, fix_typos, join_items, method_label, parse_gaps, split_sentences
from .outline_generator import REPORT_SECTIONS, Outline, OutlineSection
from .review_methodology import ReviewFlow, ReviewMethodology

MAX_CELL_CHARS = 240


@dataclass
class ReviewInputs:
    topic: str
    methodology: ReviewMethodology
    outline: Outline
    synthesis: str
    research_gaps: str
    summaries: List[PaperSummary]
    bibliography: str
    sections: Sequence[str] = tuple(REPORT_SECTIONS)
    protocol: List[str] = field(default_factory=list)  # review_methodology.protocol_paragraphs()
    flow: Optional[ReviewFlow] = None
    flowchart: Optional[str] = None  # image path relative to the report
    tensions: List[Tuple[str, str, str]] = field(default_factory=list)  # LLMSynthesizer.find_tensions()


class ReportBuilder:
    def build(self, review: ReviewInputs) -> str:
        self._said: List[set] = []  # word sets of every sentence already in the report
        for section in review.outline.sections:
            self._remember(section)
        written = {section.heading: section for section in review.outline.sections}

        builders = {
            "introduction": lambda: self._prose(written.get("Introduction")),
            "literature-review": lambda: self._literature_review(review),
            "methodology": lambda: self._methodology(review, written.get("Methodology")),
            "results": lambda: self._results(review),
            "discussion": lambda: self._discussion(review, written.get("Discussion")),
            "conclusion": lambda: self._prose(written.get("Conclusion")) + ["", gaps.conclusion_insight(review.topic)],
        }
        body: List[str] = []
        for key, heading in REPORT_SECTIONS.items():
            if key in review.sections:
                body += [f"## {heading}", "", *builders[key](), ""]

        title = review.outline.title or review.topic
        header = [f"# {title}", "", f"*A {review.methodology.name.lower()} of research on: {review.topic}*", ""]
        text = fix_typos("\n".join(body))
        # Bibliography entries are left untouched: they must match the published titles.
        return "\n".join(header + self._guide(review, text) + [text, "## References", "", review.bibliography])

    @staticmethod
    def _guide(review: ReviewInputs, body: str) -> List[str]:
        counts = gaps.marker_counts(body)
        if not any(counts.values()):
            return []
        purposes = {
            gaps.USER_ANALYSIS: "where papers contradict each other or data must be analysed",
            gaps.ADD_CRITIQUE: "where methods have limitations",
            gaps.YOUR_INSIGHT: "where gaps matter for your research",
            gaps.YOUR_OBSERVATION: "on future directions",
        }
        items = [f"**{tag}** ({n}) {purposes[tag]}" for tag, n in counts.items() if n]
        return [
            f"> **How to complete this review.** This {review.methodology.name.lower()} was generated from "
            f"{len(review.summaries)} papers. The critical analysis is left to you and marked in the text: "
            + "; ".join(items) + ". Replace each marker with your own analysis before submitting. "
            + gaps.guidance_legend(),
            "",
        ]

    # ---- sections -------------------------------------------------------------------------------

    @staticmethod
    def _prose(section: Optional[OutlineSection]) -> List[str]:
        return [" ".join(section.points)] if section and section.points else []

    def _literature_review(self, review: ReviewInputs) -> List[str]:
        paragraphs = [self._fresh_text(p) for p in review.synthesis.split("\n\n")]
        text = "\n\n".join(p for p in paragraphs if p) or review.synthesis
        # Dropping repeated sentences must not drop a paper: every included study is discussed here.
        return [ensure_coverage(text, review.summaries, review.topic)]

    def _methodology(self, review: ReviewInputs, designs: Optional[OutlineSection]) -> List[str]:
        lines: List[str] = []
        if review.protocol:
            lines += ["### Review Protocol", ""]
            for paragraph in review.protocol:
                lines += [paragraph, ""]
        if review.flowchart:
            lines += [
                f"![Figure 1. Study selection flowchart]({review.flowchart})",
                "",
                "*Figure 1. Study selection flowchart: Total Papers Found, Papers Filtered and Papers Selected.*",
                "",
            ]
        lines += ["### Characteristics of Included Studies", ""]
        lines += self._prose(designs) + [""] if designs else []
        for item in gaps.critique_markers(review.methodology, review.summaries):
            lines += [item, ""]
        return lines

    def _results(self, review: ReviewInputs) -> List[str]:
        summaries = review.summaries
        lines = [self._results_overview(summaries), ""]
        lines += ["*Table 1. Results and analysis of the included studies.*", ""]
        lines += self.results_table(summaries) + [""]
        lines += [gaps.guidance(
            gaps.USER_SHOULD_VERIFY,
            "Findings, gaps and methodology types were extracted automatically from abstracts; check each row "
            "against the paper before citing it.",
        ), ""]
        if review.methodology.empirical_only:
            lines += [gaps.meta_analysis_marker(), ""]

        # Study summaries only add what earlier sections haven't already said about each paper.
        lines += ["### Study Summaries", ""]
        for s in summaries:
            lines.append(f"#### {s.paper.title}")
            lines.append(f"*{', '.join(s.paper.authors) or 'Unknown'} ({s.paper.year or 'n.d.'})*")
            lines.append("")
            lines.append(self._fresh_text(s.summary) or s.summary)
            findings = [f for f in s.key_findings if self._fresh_text(f)]
            if findings:
                lines.append("")
                lines += [f"- {finding}" for finding in findings]
            lines.append("")
        return lines

    @staticmethod
    def _results_overview(summaries: Sequence[PaperSummary]) -> str:
        years = sorted(s.paper.year for s in summaries if s.paper.year)
        span = ""
        if years:
            span = f", published in {years[0]}" if years[0] == years[-1] else f", published between {years[0]} and {years[-1]}"
        designs: dict = {}
        for s in summaries:
            designs[method_label(s.paper)] = designs.get(method_label(s.paper), 0) + 1
        breakdown = join_items(f"{n} {label.lower()}{'' if n == 1 else 's'}" for label, n in sorted(designs.items(), key=lambda d: -d[1]))
        text = (
            f"The review includes {len(summaries)} {'study' if len(summaries) == 1 else 'studies'}{span}: {breakdown}. "
            "Table 1 lists each study's key findings, the research gap it leaves open and its methodology type."
        )
        cited = max(summaries, key=lambda s: s.paper.citation_count, default=None)
        if cited and cited.paper.citation_count:
            text += (
                f" The most cited is {cited.paper.first_author_surname} ({cited.paper.year or 'n.d.'}), "
                f"with {cited.paper.citation_count} citations."
            )
        return text

    @staticmethod
    def results_table(summaries: Sequence[PaperSummary]) -> List[str]:
        def cell(text: str) -> str:
            text = re.sub(r"\s+", " ", text.replace("|", "/")).strip()
            return text if len(text) <= MAX_CELL_CHARS else text[: MAX_CELL_CHARS - 1].rsplit(" ", 1)[0] + "…"

        def authors(names: List[str]) -> str:
            surnames = [n.split()[-1] for n in names if n.split()]
            if not surnames:
                return "Unknown"
            return f"{surnames[0]} et al." if len(surnames) > 3 else join_items(surnames)

        rows = ["| Paper Title | Authors | Year | Key Findings | Research Gaps | Methodology Type |", "|---|---|---|---|---|---|"]
        for s in summaries:
            findings = "; ".join(f.rstrip(".") for f in s.key_findings[:2]) or (split_sentences(s.summary)[:1] or [""])[0]
            rows.append(
                "| " + " | ".join(cell(c) for c in (
                    s.paper.title, authors(s.paper.authors), str(s.paper.year or "n.d."),
                    findings, s.gap or "Not stated", method_label(s.paper),
                )) + " |"
            )
        return rows

    def _discussion(self, review: ReviewInputs, discussion: Optional[OutlineSection]) -> List[str]:
        lines = self._prose(discussion) + [""] if discussion else []
        for item in gaps.contradiction_markers(review.tensions, review.summaries, review.topic):
            lines += [item, ""]

        parsed = parse_gaps(review.research_gaps)
        if parsed:
            lines += ["### Research Gaps", ""]
            for title, desc in parsed:
                lines += [f"**{title}.** {desc}" if title else desc, "", gaps.insight_marker(title, desc), ""]

        lines += ["### Future Directions", ""]
        # Gaps named after a single paper ("Open problem raised by ...") don't read as research directions.
        titles = [t[0].lower() + t[1:] if t[1:2].islower() else t for t, _ in parsed if t and " by " not in t]
        if titles:
            lines += [
                f"Building on these gaps, future research on {review.topic} should prioritise "
                f"{join_items(titles)}, ideally combining several kinds of evidence so that findings can be compared directly.",
                "",
            ]
        for item in gaps.observation_markers(review.summaries, review.topic):
            lines += [item, ""]
        return lines

    # ---- repetition control ---------------------------------------------------------------------

    @staticmethod
    def _words(sentence: str) -> set:
        return set(re.findall(r"[a-z]+", sentence.lower()))

    def _is_repeat(self, sentence: str) -> bool:
        """True if (nearly) all of the sentence's words already appeared together in one earlier sentence."""
        words = self._words(sentence)
        if len(words) < 6:
            return False
        return any(len(other) >= 6 and len(words & other) / min(len(words), len(other)) >= 0.85 for other in self._said)

    def _fresh_text(self, text: str) -> str:
        """Drops sentences already said earlier in the report and records the ones kept."""
        kept = [s for s in split_sentences(text) if not self._is_repeat(s)]
        self._said.extend(self._words(s) for s in kept)
        return " ".join(kept)

    def _remember(self, section: OutlineSection) -> None:
        for point in section.points:
            self._said.extend(self._words(s) for s in split_sentences(point))
        for sub in section.subsections:
            self._remember(sub)


def flowchart_filename(slug: str) -> str:
    return f"{slug}-flowchart.png"

