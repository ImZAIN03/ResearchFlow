"""Marks the places in a review where the user's own critical analysis is needed.

ResearchFlow writes the descriptive bulk of a literature review; judging contradictions, critiquing
methods and explaining why gaps matter is left to the user. Each marker names a tag and a specific
prompt built from the papers, so it says exactly what to analyse:

    [USER ANALYSIS NEEDED]  where papers contradict each other (or are likely to)
    [ADD YOUR CRITIQUE]     where research methods, or this review's own method, have limitations
    [YOUR INSIGHT]          where a research gap matters for the user's research
    [YOUR OBSERVATION]      on future directions suggested by the research

Markers are added when the report is assembled, after content validation, because the validators
reject bracketed placeholder text by design.
"""
from __future__ import annotations

import re
from typing import Dict, List, Sequence, Tuple

from .llm_synthesis import (
    PaperSummary,
    join_items,
    method_label,
    method_profile,
    paper_focus,
    shared_themes,
)
from .review_methodology import ReviewMethodology

USER_ANALYSIS = "[USER ANALYSIS NEEDED]"
ADD_CRITIQUE = "[ADD YOUR CRITIQUE]"
YOUR_INSIGHT = "[YOUR INSIGHT]"
YOUR_OBSERVATION = "[YOUR OBSERVATION]"
MARKERS = (USER_ANALYSIS, ADD_CRITIQUE, YOUR_INSIGHT, YOUR_OBSERVATION)
MARKER_PATTERN = re.compile("|".join(re.escape(m) for m in MARKERS))
# A marker as placed in the report (not a mention of one, e.g. in the guide at the top).
MARKER_LINE = re.compile(r"^\*\*(" + MARKER_PATTERN.pattern + r")\*\*")

# What to scrutinize in each research design (keys are llm_synthesis.METHOD_PROFILES labels).
DESIGN_CRITIQUES = {
    "empirical studies": "Assess their samples, datasets and measures: are they large and representative enough, "
    "and do the settings studied support the conclusions drawn?",
    "surveys or reviews of prior work": "Assess how comprehensive, recent and systematic their coverage of "
    "prior work is, and whether they appraise the studies they summarize.",
    "conceptual frameworks or recommendations": "Assess whether these frameworks have been tested in practice "
    "and what evidence would be needed to validate them.",
    "case studies": "Assess how representative the cases are and whether the findings can be generalized beyond them.",
    "analytical discussions": "Assess whether their arguments rest on evidence or on assumptions that remain untested.",
}


def marker(tag: str, prompt: str) -> str:
    return f"**{tag}** {prompt}"


def _tag(s: PaperSummary) -> str:
    return f"{s.paper.first_author_surname} ({s.paper.year or 'n.d.'})"


def contradiction_markers(
    tensions: Sequence[Tuple[str, str, str]], summaries: Sequence[PaperSummary], topic: str
) -> List[str]:
    """[USER ANALYSIS NEEDED] for each disagreement the model found, else for pairs worth comparing."""
    if tensions:
        return [
            marker(
                USER_ANALYSIS,
                f"Possible contradiction between {a} and {b}: {point} Check both papers, decide which "
                "position the evidence better supports, and explain why.",
            )
            for a, b, point in tensions
        ]
    summaries = list(summaries)
    if len(summaries) < 2:
        return []
    markers: List[str] = []
    # Papers that share a theme but rely on different kinds of evidence are the likeliest to disagree.
    for theme in shared_themes(summaries, exclude=topic, limit=3):
        on_theme = [s for s in summaries if theme in f"{s.paper.title} {s.paper.abstract} {s.summary}".lower()]
        pair = next(
            ((a, b) for i, a in enumerate(on_theme) for b in on_theme[i + 1 :]
             if method_profile(a.paper) != method_profile(b.paper)),
            None,
        )
        if pair:
            a, b = pair
            markers.append(marker(
                USER_ANALYSIS,
                f"{_tag(a)} ({method_label(a.paper).lower()}) and {_tag(b)} ({method_label(b.paper).lower()}) "
                f"both address {theme} with different kinds of evidence. Compare their conclusions on {theme}: "
                "do they agree, and if not, which is more convincing and why?",
            ))
        if len(markers) >= 2:
            break
    if not markers:
        dated = sorted((s for s in summaries if s.paper.year), key=lambda s: s.paper.year)
        if len(dated) >= 2 and dated[0].paper.year != dated[-1].paper.year:
            markers.append(marker(
                USER_ANALYSIS,
                f"Compare the earliest study, {_tag(dated[0])}, with the most recent, {_tag(dated[-1])}: "
                "do the newer findings confirm, refine or contradict the earlier ones?",
            ))
    return markers


def critique_markers(methodology: ReviewMethodology, summaries: Sequence[PaperSummary]) -> List[str]:
    """[ADD YOUR CRITIQUE] for the two most common research designs, and for this review's own method."""
    groups: Dict[str, List[str]] = {}
    for s in summaries:
        groups.setdefault(method_profile(s.paper), []).append(f"{s.paper.first_author_surname}, {s.paper.year or 'n.d.'}")
    markers = [
        marker(ADD_CRITIQUE, f"The {label} in this review ({'; '.join(tags)}). {DESIGN_CRITIQUES[label]}")
        for label, tags in sorted(groups.items(), key=lambda item: -len(item[1]))[:2]
    ]
    if methodology.limitations:
        markers.append(marker(
            ADD_CRITIQUE,
            "Limitations of this review's own method: " + " ".join(methodology.limitations)
            + " Discuss how these limit confidence in the conclusions and how you would address them.",
        ))
    return markers


def insight_marker(gap_title: str, gap_description: str) -> str:
    """[YOUR INSIGHT] asking why one research gap matters for the user's research."""
    subject = gap_title.rstrip(".") if gap_title else gap_description.split(".")[0]
    return marker(
        YOUR_INSIGHT,
        f"Why does the gap \"{subject}\" matter for your research question, and how could your own work help close it?",
    )


def conclusion_insight(topic: str) -> str:
    return marker(
        YOUR_INSIGHT,
        f"State in one or two sentences what this review implies for your own research on {topic}: "
        "which finding or gap shapes your research question most?",
    )


def observation_markers(summaries: Sequence[PaperSummary], topic: str) -> List[str]:
    """[YOUR OBSERVATION] on where the field is heading, anchored on the most recent studies."""
    dated = sorted((s for s in summaries if s.paper.year), key=lambda s: s.paper.year, reverse=True)
    markers = []
    if dated:
        newest = [s for s in dated if s.paper.year == dated[0].paper.year][:2]
        markers.append(marker(
            YOUR_OBSERVATION,
            f"The most recent work here ({join_items(f'{_tag(s)} on {paper_focus(s.paper)}' for s in newest)}) "
            "hints at where research is moving. Which direction do you expect to matter most next, and why?",
        ))
    markers.append(marker(
        YOUR_OBSERVATION,
        f"Which methods, data or populations would you prioritise in future studies of {topic}, "
        "given the limitations noted in this review?",
    ))
    return markers


def meta_analysis_marker() -> str:
    return marker(
        USER_ANALYSIS,
        "Extract each study's effect size, variance (or confidence interval) and sample size from the full "
        "texts, pool them (e.g. with a random-effects model), and report heterogeneity (I²) and a "
        "publication-bias check before drawing quantitative conclusions.",
    )


def marker_counts(text: str) -> Dict[str, int]:
    counts = {tag: 0 for tag in MARKERS}
    for line in text.splitlines():
        placed = MARKER_LINE.match(line)
        if placed:
            counts[placed.group(1)] += 1
    return counts


def user_input_share(text: str) -> float:
    """Share of the report's words that sit in marker prompts, i.e. are left to the user."""
    total = len(re.findall(r"\w+", text))
    marked = sum(len(re.findall(r"\w+", line)) for line in text.splitlines() if MARKER_LINE.match(line))
    return marked / total if total else 0.0
