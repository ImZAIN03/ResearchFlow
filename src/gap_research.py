"""Phase 1.5, step 2: researches each gap and turns what it finds into short hints for the user.

For every gap the module
  1. mines the papers already in the review for sentences that bear on it (their abstracts, the
     Phase 1 summaries, findings and stated gaps),
  2. searches OpenAlex, arXiv, PubMed and/or Semantic Scholar for a few more papers on it, choosing
     the databases from the research field the review's papers belong to, and
  3. writes 2-3 hints of 100-150 words in total: what the reviewed studies say, how other research
     handled the issue, and the questions that remain open.

Hints point the user at evidence and ask questions; they never draft the answer, so the user's own
analysis is what ends up in the review. Results are cached next to the review (see save_research),
so the run that integrates the user's answers doesn't repeat the searches.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .citation_formatter import CitationFormatter
from .gap_parser import EVIDENCE, FUTURE, METHODOLOGICAL, Gap, ReviewDocument, citations_in
from .llm_synthesis import (
    PDF_ARTIFACT,
    PLACEHOLDER_PATTERN,
    LLMSynthesizer,
    clean_response,
    grounding_score,
    method_label,
    normalize_whitespace,
    split_sentences,
)
from .paper_discovery import Paper, PaperDiscovery
from .review_methodology import term_share

logger = logging.getLogger(__name__)

SIDECAR_SUFFIX = ".researchflow.json"  # paper metadata written by Phase 1 next to the review
RESEARCH_SUFFIX = ".gap-research.json"
NEW_PAPERS_PER_GAP = 3
SEARCH_RESULTS_PER_QUERY = 8
MIN_RELEVANCE = 0.5  # share of the query's key terms a new paper's title/abstract must contain
HINT_MAX_WORDS = 55
QUOTE_MAX_WORDS = 22

# Words that signal a sentence speaks to a kind of gap.
CATEGORY_CUES = {
    METHODOLOGICAL: ("limitation", "limited", "sample", "bias", "dataset", "data set", "validity", "generaliz",
                     "generalis", "small", "only", "design", "measure", "assum", "confound", "rigor", "rigour"),
    EVIDENCE: ("however", "contrast", "contrary", "inconsistent", "differ", "whereas", "mixed", "effect",
               "significant", "compared", "outperform", "fail", "disagree", "evidence"),
    "theoretical": ("framework", "theory", "theoretical", "implication", "important", "understanding", "model",
                    "concept", "explain", "mechanism", "critical", "essential"),
    FUTURE: ("future", "further", "need", "should", "remain", "open", "promising", "next", "emerging",
             "opportunit", "direction", "challenge"),
}
METHOD_CUES = ("we propose", "we present", "we develop", "we use", "using", "approach", "method", "framework",
               "we conduct", "we analy", "we evaluate", "we examine", "we investigate", "this study", "this paper")

# Research fields, recognized from the review's papers, and the databases that index them best.
FIELD_CUES = {
    "medicine and health": ("patient", "clinical", "disease", "health", "medical", "hospital", "therapy", "treatment",
                            "diagnos", "cancer", "nursing", "epidemiolog", "mortality", "trial", "drug", "symptom"),
    "biology and life sciences": ("cell", "protein", "gene", "genom", "species", "molecular", "organism",
                                  "ecolog", "biolog", "enzyme", "microb", "neuron", "evolution"),
    "psychology": ("psycholog", "cognitive", "behavio", "mental", "emotion", "anxiety", "depress", "wellbeing"),
    "computer science and IT": ("algorithm", "software", "network", "neural", "learning", "comput", "data",
                                "security", "attack", "model", "system", "cloud", "blockchain", "language model"),
    "engineering": ("engineer", "material", "sensor", "circuit", "mechanical", "manufactur", "energy",
                    "structural", "robot", "control system", "power", "vehicle"),
    "physics and mathematics": ("quantum", "theorem", "equation", "particle", "physics", "mathemat", "proof", "optic"),
    "economics and business": ("econom", "market", "firm", "financ", "price", "trade", "monetary", "business",
                               "investment", "labor", "labour", "consumer", "fiscal"),
    "law and policy": ("law", "legal", "court", "regulation", "legislat", "rights", "policy", "governance",
                       "jurisdiction", "liability", "constitution", "compliance"),
    "social sciences and education": ("social", "student", "education", "teacher", "learning outcome", "communit",
                                      "inequality", "gender", "political", "survey respondents", "school"),
    "humanities": ("history", "historical", "literary", "philosoph", "cultural", "religio", "artist", "museum",
                   "narrative", "ethic", "linguistic"),
    "environmental science": ("climate", "environment", "emission", "pollut", "sustainab", "biodiversity",
                              "carbon", "ecosystem", "water"),
}
FIELD_SOURCES = {
    "medicine and health": ("openalex", "pubmed"),
    "biology and life sciences": ("openalex", "pubmed", "arxiv"),
    "psychology": ("openalex", "pubmed"),
    "computer science and IT": ("openalex", "arxiv"),
    "engineering": ("openalex", "arxiv"),
    "physics and mathematics": ("openalex", "arxiv"),
    "economics and business": ("openalex", "arxiv"),
    "law and policy": ("openalex", "semantic_scholar"),
    "social sciences and education": ("openalex", "semantic_scholar"),
    "humanities": ("openalex", "semantic_scholar"),
    "environmental science": ("openalex", "arxiv"),
}
DEFAULT_SOURCES = ("openalex", "arxiv")

APA_REFERENCE = re.compile(r"^(?P<authors>.+?) \((?P<year>\d{4}|n\.d\.)\)\. (?P<rest>.+)$")
APA_AUTHOR = re.compile(r"([^,&]+?),\s*((?:[A-Z][a-z]?\.\s*-?\s*)+)")
URL = re.compile(r"\s*(https?://\S+)\s*$")


# ---- the papers behind the review ---------------------------------------------------------------

@dataclass
class SourcePaper:
    paper: Paper
    summary: str = ""
    key_findings: List[str] = field(default_factory=list)
    gap: str = ""
    reference: str = ""  # the bibliography entry as printed in the review
    structured: bool = True  # False when only partial metadata could be read from the reference text
    in_review: bool = True  # False for papers found by Phase 1.5's own searches

    @property
    def surname(self) -> str:
        return self.paper.first_author_surname

    @property
    def year(self) -> str:
        return str(self.paper.year or "n.d.")

    @property
    def citation(self) -> str:
        return f"{self.surname} ({self.year})"

    @property
    def text(self) -> str:
        parts = [self.paper.abstract, self.summary, " ".join(self.key_findings), self.gap]
        return "\n".join(p for p in parts if p)


@dataclass
class Corpus:
    topic: str
    papers: List[SourcePaper]
    citation_style: str = "apa"
    from_sidecar: bool = False
    field: str = "general"

    def find(self, surname: str, year: str) -> Optional[SourcePaper]:
        surname = surname.lower()
        return next(
            (p for p in self.papers if p.surname.lower() == surname and p.year == str(year)),
            None,
        )

    def with_surname(self, surname: str) -> List[SourcePaper]:
        return [p for p in self.papers if p.surname.lower() == surname.lower()]


def sidecar_path(review_path: Path) -> Path:
    review_path = Path(review_path)
    return review_path.with_name(review_path.stem + SIDECAR_SUFFIX)


def paper_from_dict(data: Dict) -> Paper:
    known = {f.name for f in fields(Paper)}
    return Paper(**{k: v for k, v in data.items() if k in known})


def write_sidecar(path: Path, topic: str, methodology: str, citation_style: str, summaries) -> Path:
    """Saves the papers behind a review (called by Phase 1) so Phase 1.5 can research its gaps."""
    data = {
        "version": 1,
        "topic": topic,
        "methodology": methodology,
        "citation_style": citation_style,
        "papers": [
            {**asdict(s.paper), "summary": s.summary, "key_findings": s.key_findings, "gap": s.gap}
            for s in summaries
        ],
    }
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return Path(path)


def load_corpus(review: ReviewDocument) -> Corpus:
    """The review's papers: from Phase 1's sidecar file if present, else read back from the review text."""
    path = sidecar_path(review.path)
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            papers = [
                SourcePaper(
                    paper=paper_from_dict(item),
                    summary=item.get("summary", ""),
                    key_findings=list(item.get("key_findings") or []),
                    gap=item.get("gap", ""),
                )
                for item in data.get("papers", [])
            ]
            style = data.get("citation_style") or "apa"
            formatter = CitationFormatter()
            for p in papers:
                p.reference = formatter._formatter_for(style)(p.paper)
            logger.info("Loaded %d papers from %s", len(papers), path.name)
            corpus = Corpus(topic=data.get("topic") or review.topic, papers=papers, citation_style=style, from_sidecar=True)
            corpus.field = detect_field(corpus)
            return corpus
        except (OSError, ValueError, TypeError) as exc:
            logger.warning("Could not read %s (%s); reading the papers from the review instead", path.name, exc)

    summaries = _study_summaries(review)
    papers = [_paper_from_reference(entry, summaries) for entry in review.references]
    logger.info(
        "No %s file found; recovered %d papers from the review's references (%d with summaries)",
        SIDECAR_SUFFIX, len(papers), sum(1 for p in papers if p.summary),
    )
    corpus = Corpus(topic=review.topic, papers=papers)
    corpus.field = detect_field(corpus)
    return corpus


def _title_key(title: str) -> str:
    return "".join(c.lower() for c in title if c.isalnum())


def _study_summaries(review: ReviewDocument) -> Dict[str, Tuple[str, str, List[str]]]:
    """{title key: (title, summary, findings)} from the review's "Study Summaries" subsection."""
    found: Dict[str, Tuple[str, str, List[str]]] = {}
    title, prose, bullets = None, [], []

    def close():
        if title:
            found[_title_key(title)] = (title, " ".join(prose), bullets[:])

    for line in review.lines:
        stripped = line.strip()
        if stripped.startswith("#### "):
            close()
            title, prose, bullets = stripped[5:].strip(), [], []
        elif stripped.startswith(("## ", "### ")):
            close()
            title = None
        elif title and stripped.startswith("- "):
            bullets.append(stripped[2:])
        elif title and stripped and not (stripped.startswith("*") and stripped.endswith("*")):
            prose.append(stripped)
    close()
    return found


def _paper_from_reference(entry: str, summaries: Dict[str, Tuple[str, str, List[str]]]) -> SourcePaper:
    """Best-effort Paper from an APA bibliography entry (Phase 1's default style)."""
    text = entry.strip()
    url_match = URL.search(text)
    url = url_match.group(1) if url_match else ""
    if url_match:
        text = text[: url_match.start()].strip()
    doi = url.split("doi.org/", 1)[1] if "doi.org/" in url else None

    match = APA_REFERENCE.match(text)
    if not match:
        # Another style: keep the entry as printed, with whatever can be recognized.
        year = re.search(r"\b(19|20)\d{2}\b", text)
        quoted = re.search(r"[\"“]([^\"”]+)[\"”]", text)
        surname = re.match(r"^([^\s,.]+)", text)
        paper = Paper(
            title=(quoted.group(1) if quoted else text).rstrip(".,"),
            authors=[surname.group(1)] if surname else [],
            year=int(year.group(0)) if year else None,
            url=url,
            doi=doi,
        )
        return _attach_summary(SourcePaper(paper=paper, reference=entry, structured=False), summaries)

    authors = [f"{initials.strip()} {surname.strip()}" for surname, initials in APA_AUTHOR.findall(match.group("authors"))]
    if not authors and match.group("authors").strip() != "Unknown Author":
        authors = [match.group("authors").strip()]
    rest = match.group("rest").strip()
    title, venue = rest.rstrip("."), ""
    known = next((t for t, _, _ in summaries.values() if rest.lower().startswith(t.rstrip(".").lower())), None)
    if known:
        title = known.rstrip(".")
        venue = rest[len(known):].strip(" .")
    elif ". " in rest.rstrip("."):
        title, _, venue = rest.rstrip(".").rpartition(". ")
    paper = Paper(
        title=title.strip(),
        authors=authors,
        year=int(match.group("year")) if match.group("year").isdigit() else None,
        venue=venue.strip(),
        url=url if not doi else "",
        doi=doi,
    )
    return _attach_summary(SourcePaper(paper=paper, reference=entry), summaries)


def _attach_summary(source: SourcePaper, summaries: Dict[str, Tuple[str, str, List[str]]]) -> SourcePaper:
    entry = summaries.get(_title_key(source.paper.title))
    if entry:
        _, source.summary, source.key_findings = entry
    return source


def detect_field(corpus: Corpus) -> str:
    """The research field the review's papers belong to, judged from their vocabulary."""
    text = " ".join([corpus.topic] * 3 + [f"{p.paper.title} {p.text}" for p in corpus.papers]).lower()
    scores = {
        name: sum(len(re.findall(r"\b" + re.escape(cue), text)) for cue in cues)
        for name, cues in FIELD_CUES.items()
    }
    best = max(scores, key=scores.get)
    return best if scores[best] >= 3 else "general"


def sources_for(field_name: str) -> Tuple[str, ...]:
    return FIELD_SOURCES.get(field_name, DEFAULT_SOURCES)


# ---- evidence and hints ---------------------------------------------------------------------

@dataclass
class Evidence:
    citation: str  # "Surname (Year)"
    title: str
    sentence: str
    in_review: bool = True
    score: float = 0.0


@dataclass
class Hint:
    title: str
    text: str


@dataclass
class GapResearch:
    gap_id: int
    evidence: List[Evidence] = field(default_factory=list)
    new_papers: List[Paper] = field(default_factory=list)
    hints: List[Hint] = field(default_factory=list)

    @property
    def hint_text(self) -> str:
        return "\n".join(h.text for h in self.hints)


@dataclass
class Research:
    field: str
    sources: List[str]
    gaps: Dict[int, GapResearch]
    signature: str  # identifies the review's gaps, so a cache from another version isn't reused

    def all_new_papers(self) -> List[Paper]:
        seen, papers = set(), []
        for item in self.gaps.values():
            for paper in item.new_papers:
                if paper.dedup_key() not in seen:
                    seen.add(paper.dedup_key())
                    papers.append(paper)
        return papers


def gaps_signature(review: ReviewDocument) -> str:
    digest = hashlib.sha256("\n".join(f"{g.tag}{g.prompt}" for g in review.gaps).encode("utf-8"))
    return digest.hexdigest()[:16]


def _stems(text: str) -> set:
    return {w[:6] for w in re.findall(r"[a-z]{4,}", (text or "").lower())}


def _sentence_ok(sentence: str) -> bool:
    return 40 <= len(sentence) <= 350 and not PDF_ARTIFACT.search(sentence) and not PLACEHOLDER_PATTERN.search(sentence)


def _quote(sentence: str, max_words: int = QUOTE_MAX_WORDS) -> str:
    words = normalize_whitespace(sentence).rstrip(".").split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]).rstrip(",;:") + "..."


def _score(sentence: str, gap: Gap, topic_stems: set, cited: bool) -> float:
    stems = _stems(sentence)
    keyword_hits = len(stems & _stems(" ".join(gap.keywords)))
    topic_hits = len(stems & topic_stems)
    lower = sentence.lower()
    cue_hits = sum(1 for cue in CATEGORY_CUES.get(gap.category, ()) if cue in lower)
    return keyword_hits * 2 + topic_hits + cue_hits * 1.5 + (2 if cited else 0)


def find_evidence(gap: Gap, corpus: Corpus, extra: Sequence[SourcePaper] = (), limit: int = 4) -> List[Evidence]:
    """The sentences from the papers that bear most on the gap, at most two per paper."""
    topic_stems = _stems(corpus.topic)
    cited = {(s.lower(), y) for s, y in gap.cited}
    scored: List[Evidence] = []
    for source in list(corpus.papers) + list(extra):
        is_cited = (source.surname.lower(), source.year) in cited
        seen = set()
        for sentence in split_sentences(normalize_whitespace(source.text)):
            key = sentence.lower()[:80]
            if key in seen or not _sentence_ok(sentence):
                continue
            seen.add(key)
            score = _score(sentence, gap, topic_stems, is_cited)
            if score >= 3:
                scored.append(Evidence(source.citation, source.paper.title, sentence, source.in_review, score))
    scored.sort(key=lambda e: e.score, reverse=True)
    picked: List[Evidence] = []
    per_paper: Dict[str, int] = {}
    for item in scored:
        if per_paper.get(item.citation, 0) >= 2:
            continue
        per_paper[item.citation] = per_paper.get(item.citation, 0) + 1
        picked.append(item)
        if len(picked) >= limit:
            break
    return picked


def search_query(gap: Gap, topic: str, review_type: str) -> str:
    """A short query for papers on the gap: the topic plus what the gap is about."""
    if gap.kind == "review-method":
        return f"{topic} {review_type}"
    if gap.kind == "data-synthesis":
        return f"{topic} meta-analysis"
    if gap.kind in ("trend", "priorities"):
        return f"{topic} future research challenges"
    return " ".join([topic] + [k for k in gap.keywords if k.lower() not in topic.lower()][:2])


def find_new_papers(
    gap: Gap,
    corpus: Corpus,
    review_type: str,
    discovery: PaperDiscovery,
    sources: Sequence[str],
    known: set,
    cache: Dict[str, List[Paper]],
) -> List[Paper]:
    """Up to NEW_PAPERS_PER_GAP relevant papers that are not in the review, best match first."""
    query = search_query(gap, corpus.topic, review_type)
    if query not in cache:
        logger.info("Searching %s for gap %d: %r", ", ".join(sources), gap.id, query)
        cache[query] = discovery.search_queries([query], per_query=SEARCH_RESULTS_PER_QUERY, sources=list(sources)).papers
        if "arxiv" in sources:
            time.sleep(1)  # arXiv asks clients to space out requests
    relevance_query = " ".join([corpus.topic] + gap.keywords[:2])
    ranked = []
    for paper in cache[query]:
        if paper.dedup_key() in known or not paper.abstract or not paper.authors:
            continue
        relevance = term_share(paper, relevance_query)
        if relevance >= MIN_RELEVANCE:
            ranked.append((relevance, paper.citation_count, paper))
    ranked.sort(key=lambda r: (r[0], r[1]), reverse=True)
    return [paper for _, _, paper in ranked[:NEW_PAPERS_PER_GAP]]


def _method_sentence(paper: Paper, gap: Gap) -> str:
    sentences = [s for s in split_sentences(paper.abstract) if _sentence_ok(s)]
    cue = [s for s in sentences if any(c in s.lower() for c in METHOD_CUES)]
    pool = cue or sentences
    if not pool:
        return ""
    gap_stems = _stems(" ".join(gap.keywords))
    return max(pool, key=lambda s: len(_stems(s) & gap_stems))


def _cap(text: str, max_words: int = HINT_MAX_WORDS) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).rstrip(",;:") + "..."


def _names(gap: Gap, corpus: Corpus) -> str:
    cited = [f"{s} ({y})" for s, y in gap.cited[:3]]
    if cited:
        return " and ".join(cited) if len(cited) <= 2 else f"{cited[0]}, {cited[1]} and {cited[2]}"
    return "the reviewed studies"


def open_questions(gap: Gap, corpus: Corpus) -> str:
    """Questions for the third hint, specific to the gap's kind; written for any field."""
    names, topic, focus = _names(gap, corpus), corpus.topic, gap.focus
    questions = {
        "contradiction": f"Do {names} study {focus} with comparable samples, settings and measures? If their "
                         "conclusions differ, is it method, context or timing that explains it? Which position "
                         "would you rely on, and what evidence would settle the question?",
        "data-synthesis": "Which studies report comparable outcome measures? How much do their samples and settings "
                          "differ, and what heterogeneity would you expect? Which studies would you leave out of the "
                          "pooled estimate, and why?",
        "study-methods": f"Which single weakness most limits what {names} can claim about {topic}? Would a different "
                         "sample, setting or measure change their conclusions? What would a stronger study look like?",
        "review-method": f"Which of these limitations could have excluded relevant work on {topic}? Would manual "
                         "screening, more databases or a formal quality appraisal change the conclusions? What would "
                         "you do differently?",
        "gap-significance": f"What would change in theory or practice if \"{focus}\" were resolved? Who is affected "
                            "by it: researchers, practitioners, policy-makers or the public? Which part of it could "
                            "your own study realistically address?",
        "implication": "Which single finding or gap in this review shapes your research question most? What does it "
                       "rule in or out for how you will design your own study?",
        "trend": f"What changed between the older and the newest studies on {topic}: the questions, the methods or "
                 "the data? What is driving that shift, and which direction looks most promising to you?",
        "priorities": f"Which method, data source or population is most clearly missing from research on {topic}? "
                      "Why would filling it matter most, and what would a feasible first study look like?",
    }
    return questions.get(gap.kind, f"What does the evidence on {focus} leave unresolved, and why does it matter?")


def build_hints(gap: Gap, corpus: Corpus, evidence: Sequence[Evidence], new_papers: Sequence[Paper]) -> List[Hint]:
    """2-3 short hints: what the reviewed studies say, how other research handled it, open questions."""
    hints: List[Hint] = []
    reviewed = [e for e in evidence if e.in_review][:2]
    if reviewed:
        quotes = "; ".join(f'{e.citation}: "{_quote(e.sentence, 18 if len(reviewed) > 1 else QUOTE_MAX_WORDS)}"' for e in reviewed)
        lead = {
            METHODOLOGICAL: "Look at how these claims were produced.",
            EVIDENCE: "Set these statements side by side.",
            FUTURE: "Note what the authors themselves leave open.",
        }.get(gap.category, "Ask what these statements imply beyond the papers themselves.")
        hints.append(Hint("What the reviewed studies say", _cap(f"{quotes}. {lead}")))
    elif gap.cited:
        labels = []
        for surname, year in gap.cited[:3]:
            source = corpus.find(surname, year)
            if source:
                labels.append(f"{source.citation} is a {method_label(source.paper).lower()}")
        if labels:
            hints.append(Hint(
                "What the reviewed studies say",
                _cap("; ".join(labels) + ". Consider what each kind of evidence can and cannot establish."),
            ))

    examples = []
    for paper in new_papers[:2]:
        sentence = _method_sentence(paper, gap)
        cite = f"{paper.first_author_surname} ({paper.year or 'n.d.'})"
        examples.append(f'{cite}, "{_quote(paper.title, 12)}"' + (f': "{_quote(sentence, 16)}"' if sentence else ""))
    if examples:
        hints.append(Hint(
            "How other research handled it",
            _cap("; ".join(examples) + ". Compare their approach with the studies in your review."),
        ))

    hints.append(Hint("Questions that remain open", _cap(open_questions(gap, corpus), 60)))
    return hints[:3]


def llm_questions(synthesizer: LLMSynthesizer, gap: Gap, evidence: Sequence[Evidence]) -> str:
    """Two model-written, evidence-specific questions for the third hint, or "" if none pass validation."""
    if not evidence:
        return ""
    context = "\n".join(f"- {e.citation}: {e.sentence}" for e in evidence)
    system = (
        "You are a research mentor. You help researchers think critically by asking questions. "
        "You never write their analysis for them."
    )
    user = (
        f"Gap in a literature review: {gap.prompt}\n\nEvidence from the papers:\n{context}\n\n"
        "Task: Write exactly 2 questions, each on its own line and each one sentence ending with '?', "
        "that would help the researcher analyse this gap critically. Each question must name one of the "
        "papers above by its (Author, Year) tag. Do not answer the questions."
    )
    known = {(s, y) for e in evidence for s, y in citations_in(e.citation)}
    try:
        raw = clean_response(synthesizer.complete(system, user, max_tokens=150, temperature=0.3))
    except Exception as exc:  # hints are optional; never fail the run over them
        logger.debug("Model questions for gap %d failed: %s", gap.id, exc)
        return ""
    questions = [
        normalize_whitespace(line.lstrip("-*•0123456789.) "))
        for line in raw.splitlines()
        if line.strip().endswith("?")
    ]
    questions = [
        q for q in questions
        if 8 <= len(q.split()) <= 40
        and not PLACEHOLDER_PATTERN.search(q)
        and all(c in known for c in citations_in(q))
    ][:2]
    if len(questions) < 2 or grounding_score(" ".join(questions), f"{gap.prompt} {context}") < 0.4:
        return ""
    return " ".join(questions)


def research_gaps(
    review: ReviewDocument,
    corpus: Corpus,
    search: bool = True,
    sources: Optional[Sequence[str]] = None,
    synthesizer: Optional[LLMSynthesizer] = None,
    discovery: Optional[PaperDiscovery] = None,
) -> Research:
    sources = list(sources or sources_for(corpus.field))
    logger.info("Research field: %s; searching %s", corpus.field, ", ".join(sources) if search else "nothing (--no-search)")
    discovery = discovery or (PaperDiscovery() if search else None)
    known = {p.paper.dedup_key() for p in corpus.papers}
    query_cache: Dict[str, List[Paper]] = {}
    results: Dict[int, GapResearch] = {}
    for gap in review.gaps:
        new_papers: List[Paper] = []
        if search:
            try:
                new_papers = find_new_papers(gap, corpus, review.review_type, discovery, sources, known, query_cache)
            except Exception as exc:  # a failed search must not stop the user's review from being completed
                logger.warning("Search for gap %d failed: %s", gap.id, exc)
        extra = [SourcePaper(paper=p, in_review=False) for p in new_papers]
        evidence = find_evidence(gap, corpus, extra)
        hints = build_hints(gap, corpus, evidence, new_papers)
        if synthesizer is not None:
            questions = llm_questions(synthesizer, gap, evidence)
            if questions:
                hints[-1] = Hint("Questions that remain open", _cap(questions, 60))
        results[gap.id] = GapResearch(gap.id, evidence, new_papers, hints)
        logger.info(
            "Gap %d (%s): %d evidence sentences, %d new papers, %d hints",
            gap.id, gap.category, len(evidence), len(new_papers), len(hints),
        )
    return Research(field=corpus.field, sources=sources if search else [], gaps=results, signature=gaps_signature(review))


# ---- cache ----------------------------------------------------------------------------------

def research_path(output_dir: Path, slug: str) -> Path:
    return Path(output_dir) / f"{slug}{RESEARCH_SUFFIX}"


def save_research(research: Research, path: Path) -> Path:
    data = {
        "version": 1,
        "signature": research.signature,
        "field": research.field,
        "sources": research.sources,
        "gaps": {
            str(gap_id): {
                "evidence": [asdict(e) for e in item.evidence],
                "new_papers": [asdict(p) for p in item.new_papers],
                "hints": [asdict(h) for h in item.hints],
            }
            for gap_id, item in research.gaps.items()
        },
    }
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return Path(path)


def load_research(path: Path, review: ReviewDocument) -> Optional[Research]:
    """The cached research for this review, or None if missing, unreadable or from a different review."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("signature") != gaps_signature(review):
            logger.info("%s belongs to a different version of the review; researching the gaps again", path.name)
            return None
        gaps = {
            int(gap_id): GapResearch(
                gap_id=int(gap_id),
                evidence=[Evidence(**e) for e in item.get("evidence", [])],
                new_papers=[paper_from_dict(p) for p in item.get("new_papers", [])],
                hints=[Hint(**h) for h in item.get("hints", [])],
            )
            for gap_id, item in data.get("gaps", {}).items()
        }
        return Research(field=data.get("field", "general"), sources=data.get("sources", []), gaps=gaps, signature=data["signature"])
    except (OSError, ValueError, TypeError, KeyError) as exc:
        logger.warning("Could not read %s (%s); researching the gaps again", path.name, exc)
        return None


# ---- research brief -------------------------------------------------------------------------

def research_brief(review: ReviewDocument, corpus: Corpus, research: Research) -> str:
    """Markdown brief of everything found for each gap, to read before writing the answers."""
    lines = [
        f"# Gap research: {review.title}",
        "",
        f"*Evidence and hints for the {len(review.gaps)} gaps left in the review. Research field detected: "
        f"{research.field}. Databases searched: {', '.join(research.sources) or 'none'}.*",
        "",
        "> Use this to inform **your own** analysis. Answers that copy from the hints, the review or the papers "
        "are rejected when Phase 1.5 integrates them.",
        "",
    ]
    for gap in review.gaps:
        item = research.gaps.get(gap.id) or GapResearch(gap.id)
        low, high = gap.word_range
        lines += [
            f"## Gap {gap.id}: {gap.tag} ({gap.category})",
            "",
            f"*Location: {gap.location or 'top level'}. Suggested length: {low}-{high} words.*",
            "",
            f"**Task.** {gap.prompt}",
            "",
        ]
        if gap.context:
            lines += [f"**Context in the review.** {_cap(gap.context, 90)}", ""]
        if item.evidence:
            lines += ["**Evidence from the papers**", ""]
            lines += [
                f"- {e.citation}{'' if e.in_review else ' (not yet in your review)'}: \"{normalize_whitespace(e.sentence)}\""
                for e in item.evidence
            ]
            lines.append("")
        if item.new_papers:
            lines += ["**Further papers on this gap** (cite them and they are added to your references)", ""]
            for p in item.new_papers:
                link = f"https://doi.org/{p.doi}" if p.doi else p.url
                lines.append(f"- {p.first_author_surname} ({p.year or 'n.d.'}). {p.title}. {p.venue or ''} {link}".rstrip())
            lines.append("")
        lines += ["**Hints**", ""]
        lines += [f"{n}. *{h.title}.* {h.text}" for n, h in enumerate(item.hints, 1)]
        lines.append("")
    return "\n".join(lines)
