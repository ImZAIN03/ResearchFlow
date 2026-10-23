"""Phase 1.5, step 5: puts the user's polished answers into the review and scores the result.

assemble() replaces each gap marker with the user's text, drops the "How to complete this review"
guide, and rebuilds the references from the review's papers plus any newly found papers the user
cited, in the chosen citation style. quality_report() then scores how ready the review is to submit:

    Completeness       30  every gap filled, no markers or placeholder text left
    Originality        20  answers in the user's own words
    Critical thinking  25  answers compare, critique, reason and suggest as their gaps require
    Citations          15  in-text citations resolve to references, references are cited
    Length & register  10  answers near the suggested length, academic register

The score measures form, not whether the arguments are right; the report says so.

polish_review() is the post-generation cleanup of a freshly written Phase 1 review, run before it is
exported: it repairs malformed in-text citations, corrects a cited year when the surname identifies
one reviewed paper, reports citations that resolve to no paper and papers never cited, and flags
prose sections whose key terms are under 70% grounded in the papers with a "Review needed" note
(which assemble() removes once the user has worked through the review). polish_research_gaps()
expands every research gap to at least two sentences before the review is built.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .citation_formatter import CitationFormatter
from .critical_gaps import GUIDANCE_LINE, MARKER_LINE, MARKER_PATTERN, USER_SHOULD_VERIFY
from .gap_input import MOVE_LABELS, AnswerCheck, placeholders_in
from .gap_parser import ReviewDocument, citations_in
from .gap_research import Corpus
from .llm_synthesis import (
    MIN_GAP_SENTENCES,
    REVIEW_GROUNDING,
    PaperSummary,
    expand_gap,
    format_gaps,
    parse_gaps,
    synthesis_grounding,
)
from .paper_discovery import Paper
from .text_polish import Polished, restyle_citations

GUIDE_LINE = "> **How to complete this review.**"
REVIEW_FLAG = f"> **{USER_SHOULD_VERIFY}**"
READY = "Ready for submission"
NEARLY_READY = "Nearly ready: address the issues below"
NEEDS_REVISION = "Needs revision"


class IncompleteReviewError(RuntimeError):
    """Raised when gaps are still unanswered and incomplete output wasn't allowed."""


@dataclass
class Assembly:
    markdown: str
    filled: List[int]
    unfilled: List[int]
    references: List[str]
    added_papers: List[Paper]
    residual_markers: int = 0
    placeholders: List[str] = field(default_factory=list)
    unresolved_citations: List[str] = field(default_factory=list)
    uncited_references: List[str] = field(default_factory=list)
    citation_count: int = 0

    @property
    def complete(self) -> bool:
        return not self.unfilled and not self.residual_markers and not self.placeholders


def _cite_key(paper: Paper) -> Tuple[str, str]:
    return paper.first_author_surname, str(paper.year or "n.d.")


def assemble(
    review: ReviewDocument,
    answers: Dict[int, Polished],
    corpus: Corpus,
    style: str,
    allow_incomplete: bool = False,
) -> Assembly:
    unfilled = [g.id for g in review.gaps if g.id not in answers]
    if unfilled and not allow_incomplete:
        raise IncompleteReviewError(f"Gaps {', '.join(map(str, unfilled))} have no accepted answer yet.")

    end = review.references_start if review.references_start is not None else len(review.lines)
    lines = list(review.lines[:end])
    for gap in sorted(review.gaps, key=lambda g: g.line_index, reverse=True):
        if gap.id in answers:
            lines[gap.line_index] = answers[gap.id].text
    # The completion guide sits above every gap, so removing it after the replacements is safe.
    for i, line in enumerate(lines):
        if line.startswith(GUIDE_LINE):
            del lines[i : i + 2 if i + 1 < len(lines) and not lines[i + 1].strip() else i + 1]
            break
    # Grounding notes from polish_review() are for the user while completing the review, not the reader.
    # So are the guidance notes ([USER SHOULD VERIFY], [OPTIONAL POLISH], ...) on every phase's output.
    lines = [line for line in lines if not line.startswith(REVIEW_FLAG) and not GUIDANCE_LINE.match(line.strip())]
    body = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).rstrip()

    # References: the review's papers plus the newly found papers the user cited.
    added: List[Paper] = []
    known = {p.paper.dedup_key() for p in corpus.papers}
    for polished in answers.values():
        for paper in polished.new_papers_cited:
            if paper.dedup_key() not in known:
                known.add(paper.dedup_key())
                added.append(paper)
    cited = citations_in(body)
    resolvable = {(p.surname, p.year) for p in corpus.papers} | {_cite_key(p) for p in added}
    unresolved = [f"{s} ({y})" for s, y in cited if (s, y) not in resolvable]
    cited_set = set(cited)
    uncited = [p.citation for p in corpus.papers if (p.surname, p.year) not in cited_set]

    formatter = CitationFormatter()
    format_entry = formatter._formatter_for(style)
    entries: List[Tuple[Tuple[str, str], str]] = []
    for source in corpus.papers:
        entry = format_entry(source.paper) if source.structured else source.reference
        entries.append(((source.surname, source.year), entry))
    for paper in added:
        entries.append((_cite_key(paper), format_entry(paper)))

    style = style.lower()
    order: Dict[Tuple[str, str], int] = {}
    if style == "ieee":
        first_seen = {key: i for i, key in reversed(list(enumerate(cited)))}
        entries.sort(key=lambda e: (first_seen.get(e[0], len(cited)), e[1].lower()))
        for n, (key, _) in enumerate(entries, 1):
            order.setdefault(key, n)
        references = [f"[{n}] {entry}" for n, (_, entry) in enumerate(entries, 1)]
    else:
        references = sorted((entry for _, entry in entries), key=str.lower)
    body = restyle_citations(body, style, order)

    markdown = f"{body}\n\n## References\n\n" + "\n\n".join(references) + "\n"
    placeholders = sorted(set(placeholders_in(body)))
    return Assembly(
        markdown=markdown,
        filled=[g.id for g in review.gaps if g.id in answers],
        unfilled=unfilled,
        references=references,
        added_papers=added,
        residual_markers=len(MARKER_PATTERN.findall(body)),
        placeholders=placeholders,
        unresolved_citations=list(dict.fromkeys(unresolved)),
        uncited_references=uncited,
        citation_count=len(cited),
    )


# ---- quality report -------------------------------------------------------------------------

@dataclass
class Component:
    name: str
    points: float
    maximum: int
    note: str


@dataclass
class QualityReport:
    score: int
    verdict: str
    components: List[Component]
    issues: List[str]


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 1.0


def quality_report(
    review: ReviewDocument,
    checks: Dict[int, AnswerCheck],
    polished: Dict[int, Polished],
    assembly: Optional[Assembly],
) -> QualityReport:
    gaps = review.gaps
    accepted = [checks[g.id] for g in gaps if g.id in checks and checks[g.id].accepted]
    accepted_ids = {c.gap.id for c in accepted}
    issues: List[str] = []

    filled_share = len(accepted) / len(gaps) if gaps else 1.0
    completeness = 30 * filled_share
    if assembly and (assembly.residual_markers or assembly.placeholders):
        completeness = max(0.0, completeness - 10)
        if assembly.placeholders:
            issues.append("Placeholder text remains in the review: " + ", ".join(assembly.placeholders[:5]))
    open_gaps = [g.id for g in gaps if g.id not in accepted_ids]
    if open_gaps:
        issues.append(f"Gaps still open: {', '.join(map(str, open_gaps))}.")

    per_gap = [checks[g.id] if g.id in accepted_ids else None for g in gaps]
    originality = 20 * _mean([c.originality.score if c else 0.0 for c in per_gap])
    thinking = 25 * _mean([c.thinking_score if c else 0.0 for c in per_gap])

    citation_parts = []
    if assembly:
        total = max(1, assembly.citation_count)
        citation_parts.append(1 - len(assembly.unresolved_citations) / total)
        refs = len(assembly.references)
        citation_parts.append(1 - len(assembly.uncited_references) / refs if refs else 1.0)
        if assembly.unresolved_citations:
            issues.append(
                "Citations with no matching reference (add the full reference or correct the name/year): "
                + ", ".join(assembly.unresolved_citations)
            )
        if assembly.uncited_references:
            issues.append("References never cited in the text: " + ", ".join(assembly.uncited_references))
    needs_cite = [g for g in gaps if g.cited]
    if needs_cite:
        citation_parts.append(_mean([1.0 if g.id in accepted_ids and checks[g.id].citations else 0.0 for g in needs_cite]))
    citations = 15 * _mean(citation_parts)

    in_range = []
    register = []
    for g in gaps:
        check = checks.get(g.id)
        if g.id not in accepted_ids or check is None:
            in_range.append(0.0)
            register.append(0.0)
            continue
        low, high = g.word_range
        in_range.append(1.0 if low <= check.words <= high * 1.25 else 0.5)
        register.append(0.0 if polished.get(g.id) and polished[g.id].first_person else 1.0)
    length = 7 * _mean(in_range) + 3 * _mean(register)
    if any(p.first_person for p in polished.values()):
        issues.append(
            "Some answers use first-person hedges (\"I think\", \"in my opinion\"); state the argument directly "
            "unless your field's conventions favour the first person."
        )

    components = [
        Component("Completeness", completeness, 30, f"{len(accepted)}/{len(gaps)} gaps filled with accepted answers"),
        Component("Originality", originality, 20, "answers written in your own words"),
        Component("Critical thinking", thinking, 25, "comparison, critique, reasoning and suggestions as each gap requires"),
        Component("Citations", citations, 15, "citations resolve to references and references are cited"),
        Component("Length & register", length, 10, "answers near the suggested length, academic register"),
    ]
    score = round(sum(c.points for c in components))
    complete = assembly is not None and assembly.complete and not open_gaps
    if complete and score >= 85:
        verdict = READY
    elif complete and score >= 70:
        verdict = NEARLY_READY
    else:
        verdict = NEEDS_REVISION
    return QualityReport(score=score, verdict=verdict, components=components, issues=issues)


def report_markdown(
    review: ReviewDocument,
    report: QualityReport,
    checks: Dict[int, AnswerCheck],
    polished: Dict[int, Polished],
    assembly: Optional[Assembly],
    outputs: Sequence[str] = (),
) -> str:
    lines = [
        f"# Quality report: {review.title}",
        "",
        f"**Score: {report.score}/100. {report.verdict}.**",
        "",
        "| Criterion | Score | What it measures |",
        "|---|---|---|",
        *[f"| {c.name} | {c.points:.0f}/{c.maximum} | {c.note} |" for c in report.components],
        "",
    ]
    if outputs:
        lines += ["Files written: " + ", ".join(f"`{o}`" for o in outputs), ""]
    if report.issues:
        lines += ["## Issues to fix", ""] + [f"- {i}" for i in report.issues] + [""]

    lines += [
        "## Gaps",
        "",
        "| Gap | Type | Words | Original | Critical thinking shown | Status |",
        "|---|---|---|---|---|---|",
    ]
    for gap in review.gaps:
        check = checks.get(gap.id)
        if check is None or not check.filled:
            lines.append(f"| {gap.id} | {gap.category} | 0 | - | - | not answered |")
            continue
        moves = ", ".join(MOVE_LABELS[m] for m in MOVE_LABELS if m in check.moves) or "none"
        status = "integrated" if check.accepted else "rejected"
        lines.append(
            f"| {gap.id} | {gap.category} | {check.words} | {1 - check.originality.copied_share:.0%} | {moves} | {status} |"
        )
    lines.append("")

    for gap in review.gaps:
        check = checks.get(gap.id)
        notes = []
        if check:
            notes += [f"**Error:** {e}" for e in check.errors if check.filled]
            notes += [f"Warning: {w}" for w in check.warnings]
        result = polished.get(gap.id)
        if result:
            if result.changes:
                notes.append("Polishing: " + "; ".join(result.changes) + ".")
            if result.ambiguous_names:
                notes.append(
                    "Several papers share the surname " + ", ".join(result.ambiguous_names)
                    + "; add the year so the citation is unambiguous."
                )
            if result.unknown_citations:
                notes.append("Cited but not in the references: " + ", ".join(result.unknown_citations) + ".")
        if notes:
            lines += [f"### Gap {gap.id} ({gap.tag})", ""] + [f"- {n}" for n in notes] + [""]

    if assembly and assembly.added_papers:
        lines += ["## References added", ""]
        lines += [f"- {p.first_author_surname} ({p.year or 'n.d.'}). {p.title}" for p in assembly.added_papers]
        lines.append("")
    lines += [
        "## About this score",
        "",
        "The score checks form: that every gap is answered in your own words, shows critical thinking, cites "
        "its sources and reads as academic prose. It cannot judge whether your arguments are correct or "
        "whether the review covers the right literature for your purpose. Originality is checked against "
        "the review, its papers, the papers found for each gap and the hints, not against the whole web. "
        "Have a supervisor or peer read the review before you submit it.",
        "",
    ]
    return "\n".join(lines)


# ---- post-generation polish (Phase 1) -------------------------------------------------------

# Prose sections whose grounding is checked; the others are tables, protocol text or markers.
GROUNDED_SECTIONS = ("Introduction", "Literature Review", "Discussion", "Conclusion")
NAME = r"[A-Z][\w'’-]+"
# Malformed in-text citations the model writes, and their repair to "(Surname, Year)" form.
MALFORMED_CITATIONS = (
    (re.compile(rf"\[({NAME}(?: et al\.?)?),\s*(\d{{4}}|n\.d\.)\]"), r"(\1, \2)"),  # echoed "[Surname, Year]" context tags
    (re.compile(rf"\(({NAME}(?: et al\.?)?)\s+(\d{{4}}|n\.d\.)\)"), r"(\1, \2)"),  # "(Surname 2020)": missing comma
    (re.compile(rf"\b({NAME}) et al(?:\.|\b)\s*,\s*(?=\d{{4}}|n\.d\.)"), r"\1 et al., "),  # "et al, 2020"
    (re.compile(rf"\(({NAME}(?: et al\.)?),(\d{{4}}|n\.d\.)\)"), r"(\1, \2)"),  # "(Surname,2020)": missing space
)


@dataclass
class PolishReport:
    markdown: str
    fixed: List[str] = field(default_factory=list)  # repairs made, for the log
    unresolved: List[str] = field(default_factory=list)  # citations matching no reviewed paper
    uncited: List[str] = field(default_factory=list)  # reviewed papers never cited in the text
    flagged: List[Tuple[str, float]] = field(default_factory=list)  # (section, grounding) below REVIEW_GROUNDING

    @property
    def issues(self) -> List[str]:
        issues = []
        if self.unresolved:
            issues.append("Citations matching no reviewed paper: " + ", ".join(self.unresolved))
        if self.uncited:
            issues.append("Reviewed papers never cited in the text: " + ", ".join(self.uncited))
        issues += [f"Section {name!r} is only {score:.0%} grounded in the papers; check it before submitting"
                   for name, score in self.flagged]
        return issues


def polish_research_gaps(research_gaps: str, summaries: Sequence[PaperSummary], topic: str) -> str:
    """Every gap description expanded to at least MIN_GAP_SENTENCES sentences (see expand_gap)."""
    gaps = parse_gaps(research_gaps)
    if not gaps:
        return research_gaps
    return format_gaps([(title, expand_gap(desc, summaries, topic, MIN_GAP_SENTENCES)) for title, desc in gaps])


def _fix_citation_format(body: str, fixed: List[str]) -> str:
    total = 0
    for pattern, replacement in MALFORMED_CITATIONS:
        body, count = pattern.subn(replacement, body)
        total += count
    if total:
        fixed.append(f"{total} malformed citation(s) reformatted to (Author, Year)")
    return body


def _fix_citation_years(body: str, papers: Sequence[Paper], fixed: List[str], unresolved: List[str]) -> str:
    """Corrects the year of a citation whose surname belongs to exactly one reviewed paper; reports the rest."""
    known = {_cite_key(p) for p in papers}
    by_surname: Dict[str, List[str]] = {}
    for surname, year in known:
        by_surname.setdefault(surname, []).append(year)
    for surname, year in citations_in(body):
        if (surname, year) in known:
            continue
        years = by_surname.get(surname, [])
        if len(years) == 1:
            correct = years[0]
            body = re.sub(
                rf"\b({re.escape(surname)}(?: et al\.?)?)(, | \()({re.escape(year)})(?!\w)",
                lambda m: f"{m.group(1)}{m.group(2)}{correct}", body,
            )
            fixed.append(f"{surname} ({year}) corrected to {surname} ({correct})")
        elif surname[:1].isupper() and not re.fullmatch(r"[A-Z]+", surname):  # skip acronyms ("NATO, 2020")
            unresolved.append(f"{surname} ({year})")
    return body


def _section_spans(lines: Sequence[str]) -> List[Tuple[str, int, int]]:
    """(heading, heading line index, end index) for every "## " section."""
    starts = [(i, line[3:].strip()) for i, line in enumerate(lines) if line.startswith("## ")]
    return [(name, i, starts[n + 1][0] if n + 1 < len(starts) else len(lines)) for n, (i, name) in enumerate(starts)]


def _section_prose(lines: Sequence[str]) -> str:
    return "\n".join(
        line for line in lines
        if line.strip() and not line.startswith(("#", ">", "|", "!", "*Table", "*Figure")) and not MARKER_LINE.match(line)
    )


def polish_review(markdown: str, papers: Sequence[Paper], source: str) -> PolishReport:
    """Post-generation cleanup of a Phase 1 review: citations repaired and audited, weak sections flagged.

    `source` is the papers' text (summaries and abstracts) the sections' grounding is measured against.
    The bibliography after "## References" is never changed.
    """
    body, sep, references = markdown.partition("\n## References")
    report = PolishReport(markdown=markdown)
    body = _fix_citation_format(body, report.fixed)
    body = _fix_citation_years(body, papers, report.fixed, report.unresolved)
    report.unresolved = list(dict.fromkeys(report.unresolved))
    cited = set(citations_in(body))
    report.uncited = [f"{p.first_author_surname} ({p.year or 'n.d.'})" for p in papers if _cite_key(p) not in cited]

    lines = body.splitlines()
    for name, start, end in reversed(_section_spans(lines)):
        if name not in GROUNDED_SECTIONS:
            continue
        prose = _section_prose(lines[start + 1 : end])
        if not prose:
            continue
        score = synthesis_grounding(prose, source)
        if score < REVIEW_GROUNDING:
            report.flagged.insert(0, (name, score))
            lines[start + 1 : start + 1] = [
                "",
                f"{REVIEW_FLAG} Only {score:.0%} of this section's key terms appear in the reviewed papers. Check "
                "each claim against the sources and revise or remove what they do not support.",
            ]
    report.markdown = "\n".join(lines) + ("\n" if body.endswith("\n") else "") + sep + references
    return report


def citation_audit(
    markdown: str, known: Sequence[Tuple[str, str]], extra_known: Sequence[Tuple[str, str]] = ()
) -> Dict[str, object]:
    """Checks a phase's in-text citations against the reviewed papers ((surname, year) pairs).

    Returns {"unresolved": citations matching no reviewed paper (an invented author or a wrong year),
    "uncited": reviewed papers the text never cites}; the references section is ignored.
    """
    body = markdown.split("\n## References")[0]
    known_set = {(surname, str(year)) for surname, year in known}
    cited = citations_in(body)
    allowed = known_set | {(surname, str(year)) for surname, year in extra_known}
    unresolved = [
        f"{s} ({y})" for s, y in cited
        if (s, y) not in allowed and not re.fullmatch(r"[A-Z]+", s)  # acronyms ("NATO, 2020") are not authors
    ]
    cited_set = set(cited)
    uncited = [f"{s} ({y})" for s, y in dict.fromkeys(known_set) if (s, y) not in cited_set]
    # Share of the prose's sentences that cite a paper: a measure of how far its claims are attributed.
    prose = [
        line for line in body.splitlines()
        if line.strip() and not line.startswith(("#", "|", ">", "*", "!")) and not MARKER_LINE.match(line)
    ]
    sentences = [s for line in prose for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", line) if len(s.split()) >= 6]
    with_citation = [s for s in sentences if citations_in(s)]
    share = round(len(with_citation) / len(sentences), 2) if sentences else 0.0
    return {"unresolved": list(dict.fromkeys(unresolved)), "uncited": sorted(uncited), "cited_sentence_share": share}
