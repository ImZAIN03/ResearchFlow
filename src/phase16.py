"""Phase 1.6: fills every gap Phase 1 left in a literature review with a model-written insight.

    python -m src.main --phase 1.6 --input "output/my-review.md"

Phase 1.5 helps the user write the critical analysis; Phase 1.6 writes it automatically. For each
gap marker the 2-3 papers of the review that bear most on it are digested (findings, stated
limitations, the sentences that speak to the gap), and the local model writes a 150-200 word
paragraph that answers the marker's question, with a focus set by its tag:

    [ADD YOUR CRITIQUE]      the limitations of the methods
    [YOUR INSIGHT]           why the gap matters and what it implies for research
    [YOUR OBSERVATION]       future research directions
    [USER ANALYSIS NEEDED]   compares the papers and judges which is more convincing

Every insight must pass these checks before it goes into the review:

    grounding          at least 70% of its key terms occur in the review's papers (or in the review
                       paragraph and question the gap responds to); see llm_synthesis.grounding_score
    citations          cites at least 2 of the review's papers, and no paper outside it
    originality        no sentence copy-pasted from the papers, the review or the gap prompt, and no
                       repeat of another gap's insight (gap_input.originality)
    length             150-200 words
    critical thinking  makes the moves its gap needs (gap_input.critical_moves)
    content            no placeholders, markers, prompt echoes, garbled or repetitive text

A failed insight is regenerated up to twice, with the reasons fed back to the model; if it still
fails, the gap becomes **[MANUAL REVIEW NEEDED]**. The review is then assembled as in Phase 1.5
(markers replaced, guide removed, references rebuilt in the citation style) and written as
<review>-final.md and .docx, with every check of every attempt logged in <review>-phase16-log.md
and <review>-phase16-log.json.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from dotenv import load_dotenv

from . import critical_gaps as markers
from .export import Exporter
from .gap_input import (
    MAX_ANSWER_OVERLAP,
    MAX_COPIED_RUN,
    MAX_COPIED_SHARE,
    MOVE_ADVICE,
    MOVE_LABELS,
    SourceIndex,
    build_source_index,
    critical_moves,
    originality,
)
from .gap_parser import Gap, ReviewDocument, citations_in, parse_review
from .gap_research import Corpus, Research, SourcePaper, find_evidence, load_corpus
from .llm_synthesis import (
    GROUNDING_STOPWORDS,
    LLMSynthesizer,
    OllamaConnectionError,
    clean_response,
    content_issue,
    copied_ratio,
    grounding_score,
    method_label,
    normalize_whitespace,
    split_sentences,
)
from .phase15 import _copy_images
from .review_assembly import assemble
from .text_polish import Polished, polish

logger = logging.getLogger("researchflow")

PHASE16_MODEL = "tinyllama"  # Phase 1.6's default model when --model isn't given
MANUAL_REVIEW = "[MANUAL REVIEW NEEDED]"
MIN_WORDS, MAX_WORDS = 150, 200
MIN_GROUNDING = 0.70  # share of an insight's key terms that must occur in the sources
MIN_CITED_PAPERS = 2
MAX_PAPERS_PER_GAP = 3
MAX_REGENERATIONS = 2  # attempts per gap = 1 + MAX_REGENERATIONS
COPIED_SENTENCE = 0.5  # share of a sentence's 6-word phrases found verbatim in a source: copy-pasted
SHINGLE = 6
MAX_TOKENS = 420  # ~300 words: room for a 200-word paragraph without cutting it off
CONTEXT_CHARS = 400  # of the review paragraph the gap responds to, sent with the prompt
PROMPT_OVERHEAD_CHARS = 1400  # instructions, question and context, subtracted from the model's budget

# What each marker's insight has to concentrate on.
FOCUS = {
    markers.ADD_CRITIQUE: (
        "Focus on the limitations of the methods: the samples, data, measures and research design the "
        "papers rely on, how these weaken their conclusions, and how a stronger study could address them."
    ),
    markers.YOUR_INSIGHT: (
        "Focus on why this gap is significant: explain what remains unknown because of it, why that matters "
        "for theory and practice, and what it implies for future research."
    ),
    markers.YOUR_OBSERVATION: (
        "Focus on future research directions: name the methods, data or populations future studies should "
        "prioritise, and explain why the limitations of the current papers make them necessary."
    ),
    markers.USER_ANALYSIS: (
        "Compare the papers directly: state where their findings agree and where they differ, then judge which "
        "paper is more convincing and why, based on the strength of its evidence and methods."
    ),
}
SYSTEM_PROMPT = (
    "You are an academic writing assistant who writes the critical analysis in literature reviews. You write "
    "formal academic English in the third person, and every claim you make is supported by the papers you are given."
)
LEADING_LABEL = re.compile(r"^(?:paragraph|answer|insight|analysis|response|critique|observation)\s*:\s*", re.IGNORECASE)
LIST_PREFIX = re.compile(r"^(?:[-*•]|\d+[.)])\s+")
WORD = re.compile(r"[A-Za-z][A-Za-z'’-]*")


# ---- results --------------------------------------------------------------------------------

@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    advice: str = ""  # what the model is told to fix when the check fails


@dataclass
class Attempt:
    number: int
    text: str
    checks: List[Check] = field(default_factory=list)
    grounding: float = 0.0
    words: int = 0
    cited: List[str] = field(default_factory=list)
    error: str = ""  # the model call itself failed

    @property
    def passed(self) -> bool:
        return not self.error and bool(self.checks) and all(c.passed for c in self.checks)

    @property
    def failures(self) -> List[Check]:
        return [c for c in self.checks if not c.passed]


@dataclass
class GapResult:
    gap: Gap
    papers: List[str]  # citations of the papers given to the model as context
    attempts: List[Attempt] = field(default_factory=list)

    @property
    def accepted(self) -> Optional[Attempt]:
        return next((a for a in self.attempts if a.passed), None)

    @property
    def best(self) -> Optional[Attempt]:
        """The accepted attempt, else the one that passed the most checks (kept in the log for manual review)."""
        tried = [a for a in self.attempts if a.checks]
        return self.accepted or (max(tried, key=lambda a: sum(c.passed for c in a.checks)) if tried else None)


@dataclass
class Sources:
    """What insights are validated against, built once per review."""

    grounding_text: str  # the review's papers
    index: SourceIndex  # 6-word phrases of everything an insight must not copy


def build_sources(review: ReviewDocument, corpus: Corpus) -> Sources:
    papers_text = "\n".join(f"{p.paper.title}. {p.text}" for p in corpus.papers)
    # No gap research in Phase 1.6, so the index holds the review, its gap prompts and its papers.
    index = build_source_index(review, corpus, Research(field=corpus.field, sources=[], gaps={}, signature=""))
    return Sources(grounding_text=papers_text, index=index)


# ---- context for the model ------------------------------------------------------------------

def select_papers(gap: Gap, corpus: Corpus, limit: int = MAX_PAPERS_PER_GAP) -> List[SourcePaper]:
    """The papers the gap names first, then those with the most evidence on it, then the rest."""
    chosen: List[SourcePaper] = []

    def add(source: Optional[SourcePaper]) -> None:
        if source is not None and len(chosen) < limit and all(source is not c for c in chosen):
            chosen.append(source)

    for surname, year in gap.cited:
        add(corpus.find(surname, year))
    for item in find_evidence(gap, corpus, limit=8):
        for surname, year in citations_in(item.citation):
            add(corpus.find(surname, year))
    for source in corpus.papers:
        if len(chosen) >= MIN_CITED_PAPERS:
            break
        add(source)
    return chosen


def paper_digest(source: SourcePaper, evidence: Sequence[str], max_chars: int) -> str:
    """One paper as the model sees it: citation, title, design, findings, limitation, evidence on the gap."""
    parts = [f"({source.surname}, {source.year}) \"{source.paper.title}\", {method_label(source.paper).lower()}."]
    findings = source.key_findings[:3] or split_sentences(source.summary or source.paper.abstract)[:2]
    if findings:
        parts.append("Findings: " + " ".join(f.rstrip(".") + "." for f in findings))
    if source.gap:
        parts.append(f"Limitation: {source.gap.rstrip('.')}.")
    extra = [e for e in evidence if e not in " ".join(findings)]
    if extra:
        parts.append("On this question: " + " ".join(extra))
    text = normalize_whitespace(" ".join(parts))
    return text if len(text) <= max_chars else text[:max_chars].rsplit(" ", 1)[0] + "..."


def build_prompt(
    gap: Gap, topic: str, digests: Sequence[str], papers: Sequence[SourcePaper], feedback: Sequence[str]
) -> str:
    context = gap.context if len(gap.context) <= CONTEXT_CHARS else gap.context[:CONTEXT_CHARS].rsplit(" ", 1)[0] + "..."
    examples = " and ".join(f"({p.surname}, {p.year})" for p in papers[:2])
    lines = [f"Review topic: {topic}"]
    if gap.location:
        lines.append(f"Section: {gap.location}")
    if context:
        lines.append(f"Paragraph in the review: {context}")
    lines += ["", "Papers:", *[f"- {d}" for d in digests], "", f"Question: {gap.prompt}", ""]
    task = (
        f"Task: Write one paragraph of {MIN_WORDS} to {MAX_WORDS} words that answers the question. "
        f"{FOCUS.get(gap.tag, FOCUS[markers.YOUR_INSIGHT])} Cite at least two of the papers as (Surname, Year), "
        f"for example {examples}, and cite no other sources. Use your own words: do not copy sentences from the "
        "papers. Do not use bullet points, headings or the first person."
    )
    if feedback:
        task += " Your previous paragraph was rejected. Fix these problems: " + " ".join(feedback)
    return "\n".join(lines + [task])


# ---- turning model output into an insight ---------------------------------------------------

def word_count(text: str) -> int:
    return len(WORD.findall(text or ""))


def normalize_insight(raw: str) -> str:
    """One paragraph of prose: no headings, list markers, labels or wrapping quotation marks."""
    lines = []
    for line in (raw or "").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            lines.append(LIST_PREFIX.sub("", line))
    text = LEADING_LABEL.sub("", normalize_whitespace(" ".join(lines)))
    if len(text) > 2 and text[0] in "\"“" and text[-1] in "\"”":
        text = text[1:-1].strip()
    return text


def trim_to_length(text: str, max_words: int = MAX_WORDS, min_words: int = MIN_WORDS) -> str:
    """Drops trailing sentences while the paragraph is over max_words and stays at or above min_words."""
    sentences = split_sentences(text)
    trimmed = False
    while len(sentences) > 1 and word_count(" ".join(sentences)) > max_words:
        shorter = sentences[:-1]
        if word_count(" ".join(shorter)) < min_words:
            break
        sentences, trimmed = shorter, True
    return " ".join(sentences) if trimmed else text


def prepare_insight(raw: str, task: str, corpus: Corpus, review: ReviewDocument) -> str:
    text = LLMSynthesizer._strip_echoed_instructions(clean_response(raw), task)
    text = normalize_insight(text)
    if not text:
        return ""
    # Spelling, contractions, informal wording and citation format; never the argument itself.
    text = polish(text, corpus, review_text=review.body_text).text
    return trim_to_length(text)


# ---- quality checks -------------------------------------------------------------------------

def _key_terms(text: str) -> Set[str]:
    """The words grounding_score() compares (5+ letters, minus generic research vocabulary)."""
    return {w for w in re.findall(r"[a-z]{5,}", (text or "").lower()) if w not in GROUNDING_STOPWORDS}


def _shingles(text: str) -> List[Tuple[str, ...]]:
    tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
    return [tuple(tokens[i : i + SHINGLE]) for i in range(len(tokens) - SHINGLE + 1)]


def copied_sentences(text: str, index: SourceIndex) -> List[str]:
    """Sentences of text that are mostly verbatim phrases from the indexed sources (paper titles excepted)."""
    copied = []
    for sentence in split_sentences(text):
        plain = sentence
        for title in index.titles:
            plain = re.sub(re.escape(title), " ", plain, flags=re.IGNORECASE)
        phrases = _shingles(plain)
        if len(plain.split()) >= 8 and phrases and sum(1 for p in phrases if index.match(p)) / len(phrases) >= COPIED_SENTENCE:
            copied.append(sentence)
    return copied


def check_grounding(text: str, gap: Gap, sources: Sources) -> Tuple[Check, float]:
    source = f"{sources.grounding_text}\n{gap.context}\n{gap.prompt}"
    score = grounding_score(text, source)
    missing = sorted(_key_terms(text) - _key_terms(source))
    detail = f"{score:.0%} of key terms found in the papers (minimum {MIN_GROUNDING:.0%})"
    if missing:
        detail += "; not in the papers: " + ", ".join(missing[:8]) + ("..." if len(missing) > 8 else "")
    return Check(
        "grounding", score >= MIN_GROUNDING, detail,
        "Use the terms and findings of the papers listed above, and do not introduce ideas they do not mention.",
    ), score


def check_citations(text: str, corpus: Corpus, papers: Sequence[SourcePaper]) -> Tuple[Check, List[str]]:
    resolved: List[str] = []
    unknown: List[str] = []
    for surname, year in citations_in(text):
        source = corpus.find(surname, year)
        if source is None:
            unknown.append(f"{surname} ({year})")
        elif source.citation not in resolved:
            resolved.append(source.citation)
    needed = min(MIN_CITED_PAPERS, len(corpus.papers))
    detail = f"cites {len(resolved)} of the review's papers ({', '.join(resolved) or 'none'}; minimum {needed})"
    if unknown:
        detail += "; not in the review: " + ", ".join(dict.fromkeys(unknown))
    options = ", ".join(f"({p.surname}, {p.year})" for p in papers)
    return Check(
        "citations", len(resolved) >= needed and not unknown, detail,
        f"Cite at least {needed} of these papers in the form (Surname, Year): {options}. Cite no other sources.",
    ), resolved


def check_originality(text: str, sources: Sources, earlier: Sequence[Tuple[int, str]]) -> Check:
    # Generated text gets no allowance for quotations: quoted copies count as copies.
    result = originality(re.sub(r"[\"“”]", "", text), sources.index)
    pasted = copied_sentences(text, sources.index)
    repeats = [gap_id for gap_id, other in earlier if copied_ratio(text, other) > MAX_ANSWER_OVERLAP]
    passed = (
        result.copied_share <= MAX_COPIED_SHARE and result.longest_run < MAX_COPIED_RUN and not pasted and not repeats
    )
    detail = f"{result.copied_share:.0%} of phrases copied (maximum {MAX_COPIED_SHARE:.0%}), longest copied run {result.longest_run} words"
    if result.source and result.copied_share:
        detail += f", mostly from {result.source}"
    if pasted:
        detail += f"; {len(pasted)} copy-pasted sentence(s): \"{pasted[0][:100]}\""
    if repeats:
        detail += f"; repeats the insight for gap {', '.join(map(str, repeats))}"
    return Check(
        "originality", passed, detail,
        "Write every sentence in your own words; do not copy sentences from the papers, the review or the question,"
        " and do not repeat earlier paragraphs.",
    )


def check_length(words: int) -> Check:
    return Check(
        "length", MIN_WORDS <= words <= MAX_WORDS, f"{words} words (required {MIN_WORDS}-{MAX_WORDS})",
        f"The paragraph had {words} words; write between {MIN_WORDS} and {MAX_WORDS} words.",
    )


def check_thinking(text: str, gap: Gap) -> Check:
    moves = critical_moves(text)
    missing = [m for m in gap.required_thinking if m not in moves]
    made = ", ".join(MOVE_LABELS[m] for m in MOVE_LABELS if m in moves) or "none"
    detail = f"moves made: {made}" + (f"; missing: {', '.join(MOVE_LABELS[m] for m in missing)}" if missing else "")
    advice = " ".join(MOVE_ADVICE[m].capitalize() + "." for m in missing)
    return Check("critical thinking", not missing, detail, advice)


def check_content(text: str) -> Check:
    problem = content_issue(text, min_sentences=4, min_words=1)
    if problem is None and (markers.MARKER_PATTERN.search(text) or MANUAL_REVIEW in text):
        problem = "contains a gap marker"
    return Check(
        "content", problem is None, problem or "plain academic prose",
        "Write at least four complete sentences of plain academic prose, with no placeholders, markers or "
        "repeated sentences, and do not refer to the instructions.",
    )


def check_insight(
    number: int,
    text: str,
    gap: Gap,
    corpus: Corpus,
    papers: Sequence[SourcePaper],
    sources: Sources,
    earlier: Sequence[Tuple[int, str]],
) -> Attempt:
    attempt = Attempt(number=number, text=text, words=word_count(text))
    grounding, attempt.grounding = check_grounding(text, gap, sources)
    citations, attempt.cited = check_citations(text, corpus, papers)
    attempt.checks = [
        grounding,
        citations,
        check_originality(text, sources, earlier),
        check_length(attempt.words),
        check_thinking(text, gap),
        check_content(text),
    ]
    return attempt


# ---- generation -----------------------------------------------------------------------------

def _log_attempt(gap: Gap, attempt: Attempt) -> None:
    total = MAX_REGENERATIONS + 1
    if attempt.error:
        logger.warning("Gap %d, attempt %d/%d: %s", gap.id, attempt.number, total, attempt.error)
        return
    for check in attempt.checks:
        log = logger.info if check.passed else logger.warning
        log("Gap %d, attempt %d/%d: %s %s (%s)", gap.id, attempt.number, total, check.name,
            "passed" if check.passed else "FAILED", check.detail)
    if attempt.passed:
        logger.info("Gap %d, attempt %d/%d: accepted (%d words, %.0f%% grounded, cites %s)",
                    gap.id, attempt.number, total, attempt.words, 100 * attempt.grounding, ", ".join(attempt.cited))


def fill_gap(
    gap: Gap,
    review: ReviewDocument,
    corpus: Corpus,
    model: LLMSynthesizer,
    sources: Sources,
    earlier: Sequence[Tuple[int, str]],
) -> GapResult:
    papers = select_papers(gap, corpus)
    evidence = find_evidence(gap, corpus, limit=8)
    budget = max(500, (model.context_chars - PROMPT_OVERHEAD_CHARS) // max(1, len(papers)))
    digests = [paper_digest(p, [e.sentence for e in evidence if e.citation == p.citation][:2], budget) for p in papers]
    result = GapResult(gap=gap, papers=[p.citation for p in papers])
    logger.info("Gap %d context: %s", gap.id, ", ".join(result.papers) or "no papers")

    feedback: List[str] = []
    for number in range(1, MAX_REGENERATIONS + 2):
        prompt = build_prompt(gap, corpus.topic, digests, papers, feedback)
        task = prompt.rsplit("Task:", 1)[-1]
        try:
            raw = model.complete(SYSTEM_PROMPT, prompt, max_tokens=MAX_TOKENS, temperature=0.4 + 0.2 * (number - 1))
        except OllamaConnectionError:
            raise  # the server or model is gone: no later gap can be filled either
        except Exception as exc:  # a timeout or HTTP error costs this attempt, not the run
            attempt = Attempt(number=number, text="", error=f"model call failed: {exc}")
        else:
            text = prepare_insight(raw, task, corpus, review)
            attempt = check_insight(number, text, gap, corpus, papers, sources, earlier)
        result.attempts.append(attempt)
        _log_attempt(gap, attempt)
        if attempt.passed:
            break
        feedback = [c.advice for c in attempt.failures if c.advice]
    if result.accepted is None:
        logger.warning("Gap %d failed its checks %d times; marked %s", gap.id, len(result.attempts), MANUAL_REVIEW)
    return result


def manual_review_marker(gap: Gap) -> str:
    """What replaces a gap no insight passed for: the new marker, the original tag (unbracketed) and its task."""
    return f"**{MANUAL_REVIEW}** ({gap.tag.strip('[]')}) {gap.prompt}"


# ---- logs -----------------------------------------------------------------------------------

def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def log_markdown(review: ReviewDocument, results: Sequence[GapResult], model_name: str, outputs: Sequence[str]) -> str:
    filled = [r for r in results if r.accepted]
    manual = [r.gap.id for r in results if not r.accepted]
    lines = [
        f"# Phase 1.6 quality log: {review.title}",
        "",
        f"*Model: {model_name}. Generated {dt.date.today():%d %B %Y}. {len(results)} gaps: {len(filled)} filled "
        f"automatically, {len(manual)} marked {MANUAL_REVIEW}.*",
        "",
        "> The insights in the final review were written by a language model and checked automatically for "
        "grounding, citations, originality, length and critical thinking. The checks measure form, not whether "
        "the arguments are correct: verify each insight against the papers, and disclose the use of AI "
        "assistance as your institution or publisher requires.",
        "",
    ]
    if outputs:
        lines += ["Files written: " + ", ".join(f"`{o}`" for o in outputs), ""]
    if manual:
        lines += [f"**Gaps needing manual review:** {', '.join(map(str, manual))}. Their best draft is below.", ""]
    lines += [
        "| Gap | Marker | Location | Attempts | Grounding | Papers cited | Words | Status |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        best = r.best
        lines.append(
            f"| {r.gap.id} | {r.gap.tag} | {_cell(r.gap.location or 'top level')} | {len(r.attempts)} | "
            f"{f'{best.grounding:.0%}' if best else '-'} | {', '.join(best.cited) if best and best.cited else '-'} | "
            f"{best.words if best else '-'} | {'filled' if r.accepted else 'manual review'} |"
        )
    lines.append("")

    for r in results:
        lines += [
            f"## Gap {r.gap.id}: {r.gap.tag} ({r.gap.category})",
            "",
            f"**Task.** {r.gap.prompt}",
            "",
            f"**Papers given to the model.** {', '.join(r.papers) or 'none'}",
            "",
        ]
        for attempt in r.attempts:
            status = "passed" if attempt.passed else "failed"
            lines += [f"### Attempt {attempt.number}: {status}", ""]
            if attempt.error:
                lines += [f"- {attempt.error}", ""]
                continue
            lines += ["| Check | Result | Detail |", "|---|---|---|"]
            lines += [f"| {c.name} | {'pass' if c.passed else 'FAIL'} | {_cell(c.detail)} |" for c in attempt.checks]
            lines.append("")
        best = r.best
        if best and best.text:
            label = "Inserted insight" if r.accepted else "Best draft (not inserted)"
            lines += [f"**{label}.**", "", best.text, ""]
    return "\n".join(lines)


def log_json(review: ReviewDocument, results: Sequence[GapResult], model_name: str) -> str:
    data = {
        "version": 1,
        "review": str(review.path),
        "model": model_name,
        "generated": dt.date.today().isoformat(),
        "thresholds": {
            "min_grounding": MIN_GROUNDING,
            "min_cited_papers": MIN_CITED_PAPERS,
            "words": [MIN_WORDS, MAX_WORDS],
            "max_copied_share": MAX_COPIED_SHARE,
            "max_copied_run": MAX_COPIED_RUN,
            "max_regenerations": MAX_REGENERATIONS,
        },
        "gaps": [
            {
                "id": r.gap.id,
                "tag": r.gap.tag,
                "category": r.gap.category,
                "kind": r.gap.kind,
                "location": r.gap.location,
                "prompt": r.gap.prompt,
                "papers": r.papers,
                "status": "filled" if r.accepted else "manual review",
                "attempts": [
                    {
                        "number": a.number,
                        "passed": a.passed,
                        "words": a.words,
                        "grounding": round(a.grounding, 3),
                        "cited": a.cited,
                        "error": a.error,
                        "checks": [asdict(c) for c in a.checks],
                        "text": a.text,
                    }
                    for a in r.attempts
                ],
            }
            for r in results
        ],
    }
    return json.dumps(data, indent=2, ensure_ascii=False)


# ---- workflow -------------------------------------------------------------------------------

def run_phase16(args) -> List[Path]:
    load_dotenv()
    input_path = Path(args.input)
    if not input_path.is_file():
        raise ValueError(f"Review file not found: {input_path}")
    review = parse_review(input_path)
    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = input_path.stem
    formats = [f.strip().lower() for f in args.formats.split(",") if f.strip()]

    if not review.gaps:
        logger.info("No gap markers found in %s: the review has nothing left to fill.", input_path)
        return []
    counts: Dict[str, int] = {}
    for gap in review.gaps:
        counts[gap.tag] = counts.get(gap.tag, 0) + 1
    logger.info(
        "Found %d gaps in %r: %s", len(review.gaps), review.title, ", ".join(f"{t} x{n}" for t, n in counts.items())
    )

    corpus = load_corpus(review)
    if not corpus.papers:
        raise ValueError(f"No papers could be read from {input_path} or its {input_path.stem}.researchflow.json file")
    style = (args.citation_style or corpus.citation_style or "apa").lower()
    sources = build_sources(review, corpus)

    results: List[GapResult] = []
    earlier: List[Tuple[int, str]] = []  # accepted insights, so later ones can't repeat them
    with LLMSynthesizer(model=args.model) as model:
        logger.info("Writing insights with %s (up to %d attempts per gap)...", args.model, MAX_REGENERATIONS + 1)
        for gap in review.gaps:
            logger.info("Gap %d/%d: %s in %s", gap.id, len(review.gaps), gap.tag, gap.location or "top level")
            result = fill_gap(gap, review, corpus, model, sources, earlier)
            results.append(result)
            if result.accepted:
                earlier.append((gap.id, result.accepted.text))

    answers = {
        r.gap.id: Polished(text=r.accepted.text if r.accepted else manual_review_marker(r.gap)) for r in results
    }
    assembly = assemble(review, answers, corpus, style, allow_incomplete=True)
    _copy_images(assembly.markdown, input_path.parent, output_dir)
    outputs = Exporter(output_dir=output_dir).export_all(assembly.markdown, f"{slug}-final", formats)

    log_md = output_dir / f"{slug}-phase16-log.md"
    log_md.write_text(log_markdown(review, results, args.model, [p.name for p in outputs]), encoding="utf-8")
    log_data = output_dir / f"{slug}-phase16-log.json"
    log_data.write_text(log_json(review, results, args.model), encoding="utf-8")
    written = outputs + [log_md, log_data]

    if assembly.unresolved_citations:
        logger.warning("Citations with no matching reference: %s", ", ".join(assembly.unresolved_citations))
    if assembly.uncited_references:
        logger.info("References never cited in the text: %s", ", ".join(assembly.uncited_references))
    for path in written:
        logger.info("Written: %s", path)
    filled = [r for r in results if r.accepted]
    manual = [r.gap.id for r in results if not r.accepted]
    grounding = [r.accepted.grounding for r in filled]
    logger.info(
        "Phase 1.6: %d/%d gaps filled automatically (mean grounding %.0f%%)%s. Quality checks are logged in %s.",
        len(filled), len(results), 100 * sum(grounding) / len(grounding) if grounding else 0,
        f"; gaps {', '.join(map(str, manual))} are marked {MANUAL_REVIEW}" if manual else "", log_md,
    )
    return written
