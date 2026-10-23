"""Review methodologies (systematic, scoping, ...) and the paper screening each one applies.

The chosen methodology decides which retrieved papers are eligible (screen()), how the report's
Methodology section describes the review protocol (protocol_paragraphs()), and which inherent
limitations the user is asked to critique. ReviewFlow counts papers at each stage for the
methodology flowchart (Total Papers Found -> Papers Filtered -> Papers Selected).
"""
from __future__ import annotations

import datetime as dt
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .llm_synthesis import join_items, method_profile
from .paper_discovery import Paper

SOURCE_NAMES = {"arxiv": "arXiv", "semantic_scholar": "Semantic Scholar", "openalex": "OpenAlex", "pubmed": "PubMed"}
QUERY_STOPWORDS = set(
    "a an and are as at be by for from in into is it of on or the to with via using its their this "
    "that these those how what why when which between among towards toward based".split()
)


@dataclass(frozen=True)
class ReviewMethodology:
    key: str
    name: str
    guideline: str  # reporting guideline / framework the protocol follows
    purpose: str  # what this kind of review sets out to do; completes "aims to ..."
    synthesis: str  # how the included studies are synthesized
    require_abstract: bool  # papers without an abstract can't be screened
    min_term_share: float  # share of a query's key terms the title/abstract must contain
    recent_years: Optional[int] = None  # default publication window when --year-from isn't given
    empirical_only: bool = False
    full_text: bool = True  # False: abstracts only (no PDF downloads)
    limitations: Tuple[str, ...] = ()  # inherent limits of this review, offered for the user's critique

    def eligibility_criteria(self, year_from: Optional[int], year_to: Optional[int]) -> List[str]:
        criteria = []
        if year_from or year_to:
            criteria.append(f"it was published {_year_span(year_from, year_to)}")
        if self.require_abstract:
            criteria.append("an abstract was available for screening")
        if self.min_term_share >= 0.5:
            criteria.append("its title or abstract addressed at least half of the key terms of a search query")
        elif self.min_term_share > 0:
            criteria.append("its title or abstract mentioned at least one key term of a search query")
        if self.empirical_only:
            criteria.append("it reported empirical (quantitative or experimental) results")
        return criteria


METHODOLOGIES: Dict[str, ReviewMethodology] = {
    m.key: m
    for m in (
        ReviewMethodology(
            key="systematic",
            name="Systematic Review",
            guideline="the PRISMA 2020 reporting guideline",
            purpose="identify, screen and synthesize all eligible studies on a clearly defined question using explicit, reproducible criteria",
            synthesis="Included studies were synthesized narratively by theme and compared by research design.",
            require_abstract=True,
            min_term_share=0.5,
            limitations=(
                "Screening was automated on titles and abstracts by keyword matching; PRISMA expects two "
                "independent human reviewers and a formal risk-of-bias assessment (e.g. RoB 2, ROBINS-I or CASP).",
                "Only arXiv and Semantic Scholar were searched; a full systematic review would also cover "
                "subject databases such as Scopus, Web of Science, PubMed or IEEE Xplore, plus grey literature.",
            ),
        ),
        ReviewMethodology(
            key="scoping",
            name="Scoping Review",
            guideline="the PRISMA-ScR extension and the Arksey and O'Malley framework",
            purpose="map the extent, range and nature of the research on a topic and identify where evidence is missing",
            synthesis="Included studies were charted by year, research design and focus, and summarized descriptively.",
            require_abstract=True,
            min_term_share=0.25,
            limitations=(
                "As is standard for scoping reviews, the included studies were not critically appraised, so "
                "the review maps what has been studied rather than how well.",
                "Breadth was favoured over depth: a broad search with a fixed cap on included papers may "
                "under-represent smaller sub-areas of the topic.",
            ),
        ),
        ReviewMethodology(
            key="narrative",
            name="Narrative Review",
            guideline="no formal reporting guideline (the SANRA scale can be used to assess its quality)",
            purpose="summarize and interpret the main lines of research on a topic",
            synthesis="Included studies were discussed thematically, highlighting agreements, differences and open questions.",
            require_abstract=False,
            min_term_share=0.0,
            limitations=(
                "Papers were selected by relevance ranking and citation count rather than predefined "
                "eligibility criteria, which makes the selection open to bias toward highly cited work.",
            ),
        ),
        ReviewMethodology(
            key="rapid",
            name="Rapid Review",
            guideline="the Cochrane Rapid Reviews Methods Group guidance",
            purpose="deliver a timely evidence summary by streamlining the steps of a systematic review",
            synthesis="Included studies were summarized from their titles and abstracts and synthesized narratively.",
            require_abstract=True,
            min_term_share=0.5,
            recent_years=5,
            full_text=False,
            limitations=(
                "Studies were assessed from their abstracts only, so findings and limitations reported only "
                "in the full texts were not captured.",
                "The search was restricted to recent publications, which can omit foundational earlier studies.",
            ),
        ),
        ReviewMethodology(
            key="meta-analysis",
            name="Meta-Analysis",
            guideline="the PRISMA 2020 reporting guideline (including its meta-analysis items)",
            purpose="statistically combine the quantitative results of comparable empirical studies",
            synthesis=(
                "Included empirical studies were compared narratively; effect sizes still have to be "
                "extracted and pooled to complete the quantitative synthesis."
            ),
            require_abstract=True,
            min_term_share=0.5,
            empirical_only=True,
            limitations=(
                "Effect sizes, confidence intervals and sample sizes were not extracted automatically, so no "
                "pooled estimate, heterogeneity statistic (I²) or publication-bias test (funnel plot, Egger's test) "
                "has been computed yet.",
                "Studies were classified as empirical from their abstracts; their designs, outcomes and measures "
                "must be checked for comparability before pooling.",
            ),
        ),
    )
}
METHODOLOGY_CHOICES = tuple(METHODOLOGIES)


def get_methodology(key: str) -> ReviewMethodology:
    try:
        return METHODOLOGIES[key.lower()]
    except KeyError:
        raise ValueError(f"Unknown review methodology {key!r}. Choose from {', '.join(METHODOLOGY_CHOICES)}") from None


@dataclass
class ReviewFlow:
    """Paper counts at each review stage, for the methodology flowchart."""

    found_by_source: Dict[str, int] = field(default_factory=dict)
    duplicates: int = 0
    excluded: Counter = field(default_factory=Counter)  # screening reason -> count
    not_selected: int = 0  # eligible but ranked below the --max-papers cut-off
    not_summarized: int = 0  # selected but could not be processed

    @property
    def total_found(self) -> int:
        return sum(self.found_by_source.values())

    @property
    def filtered(self) -> int:
        return self.total_found - self.duplicates - sum(self.excluded.values())

    @property
    def selected(self) -> int:
        return self.filtered - self.not_selected - self.not_summarized


def query_terms(query: str) -> List[str]:
    terms = [w for w in re.findall(r"[a-z0-9]+", query.lower()) if len(w) >= 3 and w not in QUERY_STOPWORDS]
    return list(dict.fromkeys(terms))


def term_share(paper: Paper, query: str) -> float:
    terms = query_terms(query)
    if not terms:
        return 1.0
    text = f"{paper.title} {paper.abstract}".lower()
    # A term counts when any word in the text starts with it ("secure" matches "security" via its stem).
    words = set(re.findall(r"[a-z0-9]+", text))
    hits = sum(1 for t in terms if any(w.startswith(t[: max(4, len(t) - 2)]) for w in words))
    return hits / len(terms)


def resolve_years(
    methodology: ReviewMethodology, year_from: Optional[int], year_to: Optional[int]
) -> Tuple[Optional[int], Optional[int]]:
    if year_from is None and methodology.recent_years:
        year_from = dt.date.today().year - methodology.recent_years + 1
    return year_from, year_to


def screen(
    papers: Sequence[Paper],
    methodology: ReviewMethodology,
    queries: Sequence[str],
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
) -> Tuple[List[Paper], Counter]:
    """Applies the methodology's eligibility criteria; returns (eligible papers, exclusion reasons)."""
    excluded: Counter = Counter()
    eligible: List[Paper] = []
    for paper in papers:
        reason = None
        if (year_from or year_to) and paper.year is None:
            reason = "publication year unknown"
        elif year_from and paper.year < year_from or year_to and paper.year > year_to:
            reason = "outside the publication window"
        elif methodology.require_abstract and not paper.abstract.strip():
            reason = "no abstract to screen"
        elif methodology.min_term_share and max(term_share(paper, q) for q in queries) < methodology.min_term_share:
            reason = "off-topic"
        elif methodology.empirical_only and method_profile(paper) != "empirical studies":
            reason = "not an empirical study"
        if reason:
            excluded[reason] += 1
        else:
            eligible.append(paper)
    return eligible, excluded


# Weights of the selection score (rank_papers): how well the paper matches the queries matters most,
# then its scholarly impact, then how recent it is.
RANK_WEIGHTS = {"relevance": 0.5, "citations": 0.3, "recency": 0.2}
RECENCY_HALF_LIFE = 5  # years after which a paper's recency score halves


def rank_score(paper: Paper, queries: Sequence[str], max_citations: int, newest: int) -> Tuple[float, Dict[str, float]]:
    """(score 0-1, its components) for one paper; citations are log-scaled so a few classics don't dominate."""
    relevance = max((term_share(paper, q) for q in queries), default=1.0)
    citations = math.log1p(paper.citation_count) / math.log1p(max_citations) if max_citations else 0.0
    age = max(0, newest - paper.year) if paper.year else 2 * RECENCY_HALF_LIFE
    recency = 0.5 ** (age / RECENCY_HALF_LIFE)
    parts = {"relevance": relevance, "citations": citations, "recency": recency}
    return sum(RANK_WEIGHTS[k] * v for k, v in parts.items()), parts


def rank_papers(papers: Sequence[Paper], queries: Sequence[str]) -> List[Paper]:
    """Eligible papers ordered by a combined relevance, citation and recency score (best first)."""
    if not papers:
        return []
    max_citations = max(p.citation_count for p in papers)
    newest = max((p.year for p in papers if p.year), default=dt.date.today().year)
    scored = [(rank_score(p, queries, max_citations, newest)[0], p) for p in papers]
    scored.sort(key=lambda item: (-item[0], -(item[1].citation_count), -(item[1].year or 0)))
    return [p for _, p in scored]


def _year_span(year_from: Optional[int], year_to: Optional[int]) -> str:
    if year_from and year_to:
        return f"between {year_from} and {year_to}"
    return f"in or after {year_from}" if year_from else f"in or before {year_to}"


def protocol_paragraphs(
    methodology: ReviewMethodology,
    flow: ReviewFlow,
    queries: Sequence[str],
    sources: Sequence[str],
    year_from: Optional[int],
    year_to: Optional[int],
    max_papers: int,
) -> List[str]:
    """The review protocol for the Methodology section: search strategy, eligibility, screening, synthesis."""
    source_names = [SOURCE_NAMES.get(s, s) for s in sources]
    quoted = "; ".join(f'"{q}"' for q in queries)
    found = ", ".join(f"{n} from {SOURCE_NAMES.get(s, s)}" for s, n in flow.found_by_source.items())
    paragraphs = [
        f"This {methodology.name.lower()} aims to {methodology.purpose}, and its reporting follows "
        f"{methodology.guideline}. The databases {' and '.join(source_names)} were searched on "
        f"{dt.date.today():%d %B %Y} with the following {'query' if len(queries) == 1 else 'queries'}: {quoted}. "
        f"The search returned {flow.total_found} records ({found or 'none'}), of which {flow.duplicates} were "
        "duplicates found by more than one query or database."
    ]
    criteria = methodology.eligibility_criteria(year_from, year_to)
    screening = (
        "Records were screened automatically on their titles and abstracts. A study was eligible if "
        + join_items(criteria) + "."
        if criteria
        else "No formal eligibility criteria were applied beyond relevance to the search queries."
    )
    if flow.excluded:
        total = sum(flow.excluded.values())
        screening += f" Screening excluded {total} record{'s' if total != 1 else ''}: " + "; ".join(
            f"{reason} ({n})" for reason, n in flow.excluded.most_common()
        ) + "."
    paragraphs.append(screening)
    selection = (
        f"The {flow.filtered} eligible records were ranked by a combined score of relevance to the search queries "
        f"({RANK_WEIGHTS['relevance']:.0%}), citation count ({RANK_WEIGHTS['citations']:.0%}, log-scaled) and recency "
        f"({RANK_WEIGHTS['recency']:.0%}, halving every {RECENCY_HALF_LIFE} years), and the top "
        f"{min(max_papers, flow.filtered)} were selected for full review"
    )
    selection += " using their titles and abstracts." if not methodology.full_text else (
        "; their full texts were retrieved where an open-access PDF was available, otherwise their abstracts were used."
    )
    if flow.not_summarized:
        selection += f" {flow.not_summarized} selected record(s) could not be processed and were dropped."
    paragraphs.append(f"{selection} {methodology.synthesis} Figure 1 summarizes the selection process.")
    return paragraphs
