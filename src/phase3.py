"""Phase 3: writes a Results section from the reviewed papers' findings and plans the user's own analysis.

    python -m src.main --phase 3 --input "output/my-review.md"

Phase 1 reviews what has been studied and Phase 2 proposes how the user's study should be done; Phase 3
sets out what the reviewed studies found, and how the user's own data should be analysed so their
results can be set against them. For every paper (from the <review>.researchflow.json file Phase 1
writes) it reads, from the abstract, Phase 1 summary and findings:

    result type      quantitative, qualitative, mixed, synthesis of prior studies, conceptual
    sample           size, populations and countries (Phase 2's analysis)
    findings         result sentences from the abstract (quoted) and Phase 1's key findings (paraphrased)
    statistics       p-values, R², r, β, odds/risk ratios, confidence intervals, effect sizes, means,
                     percentages and performance metrics stated in the text
    limitations      what the paper says it leaves open, and the gap Phase 1 recorded

Findings shared by two or more papers become the section's themes. Each theme reports which papers
found it, whether their wording points the same way (consistent, mixed or contradictory) and how far
the evidence can be trusted. The field (from Phase 2's analysis, or detected again) decides which
results are highlighted: effect estimates and safety outcomes in medicine, descriptive, correlational
and thematic results in the social sciences, performance metrics in engineering and computing, case
outcomes and doctrinal findings in law. The analysis plan follows Phase 2's recommended design: the
statistical tests or qualitative method to use, sample sizes from the papers, and a power analysis.
The places only the user can fill are marked:

    [YOUR DATA ANALYSIS]     the user's own results, set against each finding
    [YOUR ANALYSIS METHOD]   the analysis the user will actually run

Extraction is rule-based, so Phase 3 runs without a model. Writes <review>-results.md and .docx, plus
<review>-results-analysis.json with every paper's results profile and the sentence each detail was read from.
"""
from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import NormalDist
from typing import Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv

from . import critical_gaps as markers
from .citation_formatter import CitationFormatter
from .export import Exporter
from .gap_research import SIDECAR_SUFFIX, Corpus, SourcePaper, load_corpus, sidecar_path
from .llm_synthesis import THEME_STOPWORDS, join_items, normalize_whitespace, split_sentences
from .phase2 import (
    ANALYSIS_LABELS,
    CONCEPTUAL,
    DESIGN_BY_KEY,
    DESIGN_SCIENCE,
    GUIDES,
    MIXED,
    QUAL,
    QUANT,
    SPELLED,
    SYNTHESIS,
    TRADITIONS,
    UNCLASSIFIED,
    Recommendation,
    StudyDesign,
    Tradition,
    _a,
    _cap,
    _cell,
    _cites,
    _clip,
    _count,
    _lc,
    _sample_sentences,
    analyse_paper,
    classify_unclear,
    detect_tradition,
    recommend,
    resolve_review,
)

logger = logging.getLogger("researchflow")

PHASE3_FORMATS = "md,docx"
DERIVED_SUFFIXES = ("-methodology", "-results")  # Phase 2/3 outputs are named after the Phase 1 review
QUOTE_MAX_WORDS = 45
MAX_THEMES = 4
MIN_SHARED_TERMS = 2  # content words two papers' findings must share to count as one finding
ALPHA, POWER = 0.05, 0.80

QUANT_RESULTS, QUAL_RESULTS, MIXED_RESULTS = "Quantitative", "Qualitative", "Mixed"
SYNTHESIS_RESULTS, CONCEPTUAL_RESULTS, UNCLEAR_RESULTS = "Synthesis of prior studies", "Conceptual / argument", "Not clear from the abstract"


# ---- statistics stated in the text ------------------------------------------------------------

@dataclass(frozen=True)
class StatKind:
    key: str
    label: str
    pattern: re.Pattern
    effect: bool = False  # an effect size or estimate (not just a significance level or a count)


NUMBER = r"-?\d+(?:\.\d+)?"
DECIMAL = r"-?\d*\.\d+"
INTERVAL = NUMBER + r"\s*(?:[-–—,;]|to)\s*" + NUMBER
# In priority order: a match inside an earlier kind's match (the % in "95% CI") is not counted again.
STAT_KINDS: Tuple[StatKind, ...] = (
    StatKind("confidence-interval", "Confidence interval", re.compile(
        r"\b9[059]\s?%\s?(?:CI|(?i:confidence intervals?))\s*[:,=]?\s*"
        r"(?:\(\s*" + INTERVAL + r"\s*\)|\[\s*" + INTERVAL + r"\s*\]|" + INTERVAL + ")"), effect=True),
    StatKind("r-squared", "Variance explained (R²)", re.compile(r"\bR(?:²|2|\^2|�)\s*=\s*" + DECIMAL), effect=True),
    StatKind("correlation", "Correlation (r)", re.compile(r"(?<![\w²�])r\s*=\s*" + DECIMAL), effect=True),
    StatKind("beta", "Regression coefficient (β)", re.compile(r"(?:β|\b(?i:beta))\s*=\s*" + NUMBER), effect=True),
    StatKind("odds-ratio", "Odds ratio", re.compile(
        r"\b(?:a?OR|(?i:odds ratios?))\s*(?:=|:|of|was|were)?\s*\d+(?:\.\d+)?"), effect=True),
    StatKind("risk-ratio", "Risk or hazard ratio", re.compile(
        r"\b(?:a?HR|a?RR|(?i:hazard ratios?|risk ratios?|relative risks?))\s*(?:=|:|of|was|were)?\s*\d+(?:\.\d+)?"),
        effect=True),
    StatKind("effect-size", "Standardised effect size", re.compile(
        r"(?:\b(?i:cohen'?s)\s+)?\b[dg]\s*=\s*" + DECIMAL + r"|(?i:effect sizes?)\s*(?:=|of|was)\s*" + DECIMAL
        + r"|(?:η²|η2|(?i:eta squared))\s*=\s*" + DECIMAL), effect=True),
    StatKind("test-statistic", "Test statistic", re.compile(
        r"\b(?:t|F|χ²|χ2|(?i:chi-square))\s*\(\s*\d+(?:\.\d+)?(?:\s*,\s*\d+(?:\.\d+)?)?\s*\)\s*=\s*" + NUMBER)),
    StatKind("p-value", "Significance (p)", re.compile(r"\b[pP]\s*(?:[<>=≤≥]|&lt;)\s*" + DECIMAL)),
    StatKind("performance", "Performance metric", re.compile(
        r"\b(?i:accuracy|precision|recall|f1(?:[- ]score)?|auc|auroc|roc[- ]auc|bleu|rouge(?:-\w+)?|rmse|mae|mse|"
        r"error rates?|detection rates?|false[- ]positive rates?)\b[^.;:]{0,30}?\d+(?:\.\d+)?\s?%?"
        r"|\b\d+(?:\.\d+)?\s?%\s+(?i:accuracy|improvement|reduction|increase|decrease|gain|fewer|faster|higher|lower)"
        r"|\b\d+(?:\.\d+)?\s?[x×]\s+(?i:faster|speed-?up|fewer|less|more|smaller|larger)"), effect=True),
    StatKind("mean", "Mean (SD)", re.compile(
        r"\b(?:M|(?i:mean))\s*=\s*" + NUMBER + r"(?:\s*,\s*SD\s*=\s*\d+(?:\.\d+)?)?")),
    StatKind("percentage", "Percentage", re.compile(r"\b\d+(?:\.\d+)?\s?%")),
)
STAT_BY_KEY = {k.key: k for k in STAT_KINDS}
INFERENTIAL_KINDS = {"confidence-interval", "r-squared", "correlation", "beta", "odds-ratio", "risk-ratio",
                     "effect-size", "test-statistic", "p-value"}
DESCRIPTIVE_KINDS = {"mean", "percentage"}

# Sentences that state a result rather than background or aims.
RESULT_CUES = (
    r"\b(?:results?|findings?|analys[ie]s|data|evidence|experiments?|evaluations?) (?:show|suggest|indicate|reveal|demonstrate|confirm|underscore|highlight)",
    r"\bwe (?:find|found|show|demonstrate|observe[d]?|report)\b", r"\b(?:found|showed|revealed|indicated|demonstrated|confirmed) that\b",
    r"\brevealed\b", r"\boutperform", r"\bsignificant(?:ly)?\b", r"\bassociated with\b", r"\bpositive (?:relationship|effect|association|correlation)",
    r"\bnegative (?:relationship|effect|association|correlation)", r"\bcan be (?:better )?understood as\b", r"\billustrates\b",
    r"\bachiev(?:es|ed|ing)\b", r"\bshowing (?:that|differences)\b", r"\bvalidate[sd]?\b",
)
# Sentences that state what a study set out to do, not what it found.
AIM_CUES = (r"\bwill\b", r"\baims? to\b", r"\bseeks? to\b", r"\b(?:purpose|objective|goal) (?:of|is|was)\b", r"\bwe propose\b")
LIMITATION_CUES = (  # the study's own limitations, not its critique of earlier work
    r"\blimitations? (?:of|in) (?:this|our|the present|the current) (?:study|work|research|paper|analysis)",
    r"\b(?:our|this study's|the study's) (?:main )?limitations?\b", r"\bis limited (?:by|to)\b",
    r"\bfuture (?:research|work|studies) (?:is |are )?(?:needed|required|should)", r"\bfurther (?:research|work|studies) (?:is|are) needed\b",
    r"\b(?:small|limited) samples?\b", r"\bnot (?:be )?generali[sz]", r"\binterpreted with caution\b", r"\bself-reported\b",
)
SUPPORTS = re.compile(
    r"\b(?:improv\w*|increas\w*|enhanc\w*|effective(?:ly|ness)?|outperform\w*|better|superior|positive(?:ly)?|benefit\w*|"
    r"strengthen\w*|validat\w*|success\w*|bolster\w*|facilitat\w*|enabl\w*|crucial|helpful|helps?|faster|higher accuracy)\b",
    re.IGNORECASE,
)
QUALIFIES = re.compile(
    r"\b(?:no (?:significant )?(?:effect|difference|association|relationship|improvement|evidence)|not (?:significant|effective|associated)|"
    r"did not|does not|fail\w*|ineffective|limited|limitations?|rare(?:ly)?|overstat\w*|outlier|rather than|obstacles?|"
    r"challenges?|barriers?|worse|negative(?:ly)?|lack\w*|shortcomings?|inconsistent|unclear|elusive|skeptic\w*)\b",
    re.IGNORECASE,
)

# Field-specific results that are statements rather than numbers: (key, label, cues).
FIELD_STATEMENTS: Dict[str, Tuple[Tuple[str, str, Tuple[str, ...]], ...]] = {
    "health": (
        ("safety", "Safety outcomes", (r"adverse (?:events?|effects?|reactions?)", r"side[- ]effects?", r"\bsafety\b", r"complications?",
                                        r"toxicit", r"\bmortality\b", r"tolerab", r"\bharms?\b")),
        ("clinical", "Clinical outcomes", (r"\boutcomes?\b", r"\bsymptoms?\b", r"\bsurvival\b", r"\brecovery\b", r"\bremission\b")),
    ),
    "social": (
        ("thematic", "Thematic findings", (r"\bthemes?\b", r"\bemerged\b", r"participants (?:described|reported|expressed|perceived|felt)",
                                           r"\bperceptions?\b", r"\bexperiences?\b", r"\battitudes?\b")),
        ("relationship", "Relationships between variables", (r"\bassociated with\b", r"\brelationship\b", r"\bpredict", r"\bcorrelat",
                                                              r"\bfactors?\b", r"\binfluence")),
    ),
    "engineering": (
        ("technical", "Technical results", (r"\boutperform", r"state[- ]of[- ]the[- ]art", r"\bbaselines?\b", r"\blatency\b", r"\bthroughput\b",
                                            r"\bscalab", r"\bruntime\b", r"\bmemory\b", r"\bbenchmark", r"\breproduc", r"\bhardware\b")),
    ),
    "law": (
        ("outcome", "Case outcomes", (r"\bheld that\b", r"\bruled\b", r"\bupheld\b", r"struck down", r"\boverrul", r"\bdismissed\b",
                                      r"\bliab(?:le|ility)\b", r"\bconvicted\b", r"\bacquitted\b", r"\b(?:un)?lawful\b", r"\bbreach")),
        ("doctrinal", "Doctrinal and precedent findings", (r"\bprecedent", r"\bdoctrine", r"\binterpret", r"\bprinciples?\b",
                                                           r"\bnorms?\b", r"international law", r"\bjurisdiction")),
    ),
    "humanities": (
        ("interpretive", "Interpretive findings", (r"can be (?:better )?understood as", r"\brather than\b", r"\breinterpret",
                                                   r"\billustrates\b", r"\breveals\b", r"\bargues?\b", r"\bin light of\b")),
    ),
}
STATEMENT_KINDS = {"safety", "outcome", "thematic", "interpretive"}  # a sentence about these is a result by itself
FIELD_SECTION = {
    "health": ("Effect Estimates and Safety Outcomes", ("odds-ratio", "risk-ratio", "confidence-interval", "p-value", "effect-size", "percentage")),
    "social": ("Descriptive, Correlational and Thematic Findings", ("mean", "percentage", "correlation", "r-squared", "beta", "p-value", "effect-size", "test-statistic")),
    "engineering": ("Performance Metrics and Technical Results", ("performance", "percentage", "confidence-interval", "p-value")),
    "law": ("Case Outcomes and Doctrinal Findings", ("percentage",)),
    "humanities": ("Interpretive Findings", ()),
}
FIELD_EXPECTATIONS = {
    "health": "In medicine and health, results are reported as effect estimates (odds, risk or hazard ratios, or mean "
              "differences) with 95% confidence intervals and p-values, and benefits are weighed against safety outcomes.",
    "social": "In the social sciences, results are reported as descriptive statistics, correlations or regression "
              "coefficients with their significance, and, for qualitative studies, as themes illustrated with quotations.",
    "engineering": "In engineering and computing, results are reported as performance metrics against baselines on named "
                   "datasets or test settings, ideally with the variation across repeated runs.",
    "law": "In law, results are the outcomes of the cases analysed and the doctrinal positions that follow from them, with "
           "the authority (court, jurisdiction, date) of each source.",
    "humanities": "In the humanities, results are interpretive claims, each supported by the sources it rests on.",
}

THEME_EXTRA_STOPWORDS = set(
    "indicate indicates indicated found reveal revealed revealing demonstrate demonstrates demonstrated demonstrating "
    "underscore underscores illuminates illuminate offers offering offer allows allow enable enables enabling proposed "
    "tailored highlight highlighted major significant significantly positive negative relationship effective effectively "
    "improve improves improved improving better greater higher lower practical valuable useful insights insight "
    "addresses address addressing unique novel comprehensive existing ability capable capability capabilities identified "
    "identifies determined determine examples ranging range suggests argues argue involved typically obscured "
    "important crucial essential state paper's study's system's model's "
    # too general to tie two findings together
    "system systems active technique techniques tools practice practices process processes factor factors aspect "
    "aspects future development application applications order perspective".split()
)


@dataclass
class Statistic:
    kind: str
    text: str
    sentence: str

    @property
    def label(self) -> str:
        return STAT_BY_KEY[self.kind].label


@dataclass
class Evidence:
    text: str
    origin: str  # "abstract": the paper's own words (quoted); "summary": Phase 1's paraphrase
    direction: str = "neutral"  # "supports", "qualifies" or "neutral", read from the wording


@dataclass(eq=False)  # one per paper, compared by identity
class ResultProfile:
    study: StudyDesign
    result_type: str
    findings: List[Evidence] = field(default_factory=list)
    statistics: List[Statistic] = field(default_factory=list)
    field_statements: List[Dict] = field(default_factory=list)  # {"kind", "label", "sentence"}
    limitations: List[Evidence] = field(default_factory=list)
    confidence: str = "Low"
    confidence_reason: str = ""

    @property
    def lead(self) -> Optional[Evidence]:
        return self.findings[0] if self.findings else None

    def effects(self) -> List[Statistic]:
        return [s for s in self.statistics if STAT_BY_KEY[s.kind].effect or s.kind == "p-value"]

    def to_dict(self) -> Dict:
        return {
            "citation": self.study.narrative,
            "title": self.study.title,
            "design": self.study.kind.label,
            "result_type": self.result_type,
            "samples": self.study.samples,
            "populations": self.study.populations,
            "countries": self.study.countries,
            "findings": [asdict(f) for f in self.findings],
            "statistics": [{**asdict(s), "label": s.label} for s in self.statistics],
            "field_statements": self.field_statements,
            "limitations": [asdict(l) for l in self.limitations],
            "confidence": self.confidence,
            "confidence_reason": self.confidence_reason,
        }


@dataclass
class Theme:
    term: str
    evidence: List[Tuple[ResultProfile, Evidence]]
    consistency: str = "mixed"  # "consistent", "mixed" or "contradictory"
    confidence: str = "Low"

    @property
    def profiles(self) -> List[ResultProfile]:
        return list(dict.fromkeys(p for p, _ in self.evidence))

    def to_dict(self) -> Dict:
        return {
            "theme": self.term,
            "studies": [p.study.narrative for p in self.profiles],
            "consistency": self.consistency,
            "confidence": self.confidence,
            "evidence": [{"study": p.study.narrative, **asdict(e)} for p, e in self.evidence],
        }


# ---- per-paper results ------------------------------------------------------------------------

def _matches(cues: Sequence[str], sentence: str) -> bool:
    lower = sentence.lower()
    return any(re.search(cue, lower) for cue in cues)


def direction(sentence: str) -> str:
    """Whether a finding's wording supports or qualifies what it is about; the first cue decides a tie."""
    support = [m.start() for m in SUPPORTS.finditer(sentence)]
    qualify = [m.start() for m in QUALIFIES.finditer(sentence)]
    if not support and not qualify:
        return "neutral"
    if len(support) != len(qualify):
        return "supports" if len(support) > len(qualify) else "qualifies"
    return "supports" if min(support, default=math.inf) < min(qualify, default=math.inf) else "qualifies"


def extract_statistics(sentences: Sequence[str]) -> List[Statistic]:
    """Statistics stated in the sentences, each counted once (a % inside a confidence interval is not a percentage)."""
    found: List[Statistic] = []
    seen = set()
    for sentence in sentences:
        taken: List[Tuple[int, int]] = []
        for kind in STAT_KINDS:
            for match in kind.pattern.finditer(sentence):
                start, end = match.span()
                if any(start < e and s < end for s, e in taken):
                    continue
                taken.append((start, end))
                text = normalize_whitespace(match.group(0)).rstrip(" ,;")
                if kind.key == "r-squared":
                    text = re.sub(r"^R(?:2|\^2)", "R²", text)
                if (kind.key, text.lower()) not in seen:
                    seen.add((kind.key, text.lower()))
                    found.append(Statistic(kind.key, text, _clip(sentence, QUOTE_MAX_WORDS)))
    return found


def _fix(text: str) -> str:
    """Repairs characters lost in the papers' metadata: "R�" was "R²", other replacement characters were quotes."""
    return normalize_whitespace(text or "").replace("R�", "R²").replace("�", "'")


def _sentences(text: str) -> List[str]:
    return split_sentences(_fix(text))


def _clean_quote(sentence: str) -> str:
    return _clip(_fix(sentence), QUOTE_MAX_WORDS)


def result_type(study: StudyDesign, statistics: Sequence[Statistic]) -> str:
    quantitative = any(s.kind in INFERENTIAL_KINDS or s.kind == "performance" for s in statistics)
    approach = study.kind.approach
    if approach == MIXED:
        return MIXED_RESULTS
    if approach == QUAL:
        return MIXED_RESULTS if quantitative else QUAL_RESULTS
    if approach in (QUANT, DESIGN_SCIENCE):
        return QUANT_RESULTS
    if approach == SYNTHESIS:
        return QUANT_RESULTS + " synthesis" if study.design == "meta-analysis" else SYNTHESIS_RESULTS
    if approach == CONCEPTUAL:
        return QUANT_RESULTS if quantitative else CONCEPTUAL_RESULTS
    return QUANT_RESULTS if quantitative else UNCLEAR_RESULTS


def rate_confidence(study: StudyDesign, statistics: Sequence[Statistic]) -> Tuple[str, str]:
    """(High/Medium/Low, why) for one study's results, from its design, sample and the statistics it reports."""
    if not study.kind.empirical:
        if study.design == "meta-analysis":
            return "Medium", "pools earlier studies, so it is only as strong as they are"
        if study.design == "systematic-review":
            return "Medium", "a systematic synthesis of earlier studies, with no new data"
        return "Low", f"{_lc(study.kind.label)}: no new data were collected"
    if study.classified_by == "model":
        return "Low", "the design was classified by the model, not stated in the abstract"
    has_sample = bool(study.samples) or "datasets" in study.collection
    inferential = [s for s in statistics if s.kind in INFERENTIAL_KINDS or s.kind == "performance"]
    if has_sample and inferential:
        return "High", "empirical, with a stated sample or dataset and reported statistics"
    if has_sample:
        return "Medium", "empirical with a stated sample or dataset, but the abstract reports no statistics"
    if inferential:
        return "Medium", "reports statistics, but the abstract does not state the sample"
    return "Low", "the abstract states neither a sample nor statistics"


def profile_paper(source: SourcePaper, study: StudyDesign, tradition: Tradition) -> ResultProfile:
    abstract = _sentences(source.paper.abstract)
    summary = _sentences(source.summary)
    findings: List[Evidence] = []
    seen = set()

    def add(text: str, origin: str) -> None:
        key = re.sub(r"\W+", " ", text.lower()).strip()
        if key and key not in seen:
            seen.add(key)
            findings.append(Evidence(_clean_quote(text), origin, direction(text)))

    for sentence in abstract:
        if _matches(RESULT_CUES, sentence) and not _matches(AIM_CUES, sentence):
            add(sentence, "abstract")
    for finding in source.key_findings:
        add(_fix(finding), "summary")
    for sentence in summary:
        if re.search(r"\bmain (?:result|finding)|\bresults? (?:show|indicate|suggest)", sentence, re.IGNORECASE):
            add(sentence, "summary")
    if not findings and abstract:
        add(abstract[-1], "abstract")  # abstracts usually end on their conclusion

    statistics = extract_statistics(abstract + [_fix(f) for f in source.key_findings] + summary)
    statements = []
    for key, label, cues in FIELD_STATEMENTS.get(tradition.key, ()):
        for sentence in abstract:
            if _matches(cues, sentence) and not _matches(AIM_CUES, sentence) and (
                key in STATEMENT_KINDS or _matches(RESULT_CUES, sentence)
            ):
                statements.append({"kind": key, "label": label, "sentence": _clean_quote(sentence)})
                break

    limitations = [Evidence(_clean_quote(s), "abstract", "qualifies") for s in abstract if _matches(LIMITATION_CUES, s)][:2]
    if source.gap:
        limitations.append(Evidence(_clean_quote(source.gap), "summary", "qualifies"))

    confidence, reason = rate_confidence(study, statistics)
    return ResultProfile(study, result_type(study, statistics), findings, statistics, statements, limitations, confidence, reason)


# ---- patterns across papers -------------------------------------------------------------------

def _stem(word: str) -> str:
    if word.endswith("ies") and len(word) > 5:
        return word[:-3] + "y"
    if word.endswith("s") and not word.endswith(("ss", "us", "is")) and len(word) > 5:
        return word[:-1]
    return word


def _content_terms(text: str, excluded: set) -> Dict[str, str]:
    """{stem: first surface form} of the content words in a finding, in order of appearance."""
    stops = THEME_STOPWORDS | THEME_EXTRA_STOPWORDS
    terms: Dict[str, str] = {}
    for word in re.findall(r"[a-z][a-z-]+", text.lower()):
        stem = _stem(word)
        if len(word) >= 5 and word not in stops and stem not in excluded:
            terms.setdefault(stem, word)
    return terms


def _theme_label(text: str, shared: Sequence[str]) -> str:
    """The shared words as they occur in a finding: adjacent ones as a phrase ("threat detection"), others joined."""
    phrases: List[List[str]] = []
    seen = set()
    last = -2
    for i, word in enumerate(re.findall(r"[a-z][a-z-]+", text.lower())):
        stem = _stem(word)
        if stem in shared and stem not in seen:
            seen.add(stem)
            if i == last + 1 and phrases:
                phrases[-1].append(word)
            else:
                phrases.append([word])
            last = i
    return join_items([" ".join(phrase) for phrase in phrases][:3])


def find_themes(profiles: Sequence[ResultProfile], topic: str, limit: int = MAX_THEMES) -> List[Theme]:
    """Findings that two or more papers report: finding sentences from different papers sharing two or more content words.

    One shared word is too weak a link ("incident" can be a cyber attack in one paper and a response time in
    another); two ("threat" and "detection") make a common finding much likelier.
    """
    excluded = {_stem(w) for w in re.findall(r"[a-z]+", topic.lower())}
    terms = {
        (i, j): _content_terms(evidence.text, excluded)
        for i, profile in enumerate(profiles) for j, evidence in enumerate(profile.findings)
    }
    pairs = []
    for (i, j), a in terms.items():
        for (k, m), b in terms.items():
            if i < k:
                shared = [t for t in a if t in b]
                if len(shared) >= MIN_SHARED_TERMS:
                    pairs.append((-len(shared), i, j, k, m, shared))
    pairs.sort(key=lambda pair: pair[:5])

    themes: List[Theme] = []
    used = set()  # (paper, finding) pairs already in a theme
    for _, i, j, k, m, shared in pairs:
        if (i, j) in used or (k, m) in used:
            continue
        members = [(i, j), (k, m)]
        vocabulary = set(terms[(i, j)]) | set(terms[(k, m)])
        for other in range(len(profiles)):
            if other in (i, k):
                continue
            match = next(
                ((other, n) for n in range(len(profiles[other].findings)) if (other, n) not in used
                 and set(shared) & set(terms[(other, n)]) and len(vocabulary & set(terms[(other, n)])) >= MIN_SHARED_TERMS),
                None,
            )
            if match:
                members.append(match)
        used.update(members)
        theme = Theme(
            _theme_label(profiles[i].findings[j].text, shared),
            [(profiles[a], profiles[a].findings[b]) for a, b in sorted(members)],
        )
        theme.consistency, theme.confidence = judge(theme)
        themes.append(theme)
        if len(themes) >= limit:
            break
    return themes


def judge(theme: Theme) -> Tuple[str, str]:
    """(consistency, confidence) of a theme, from its papers' directions, number and individual confidence."""
    directions = {e.direction for _, e in theme.evidence if e.direction != "neutral"}
    stating = sum(1 for _, e in theme.evidence if e.direction != "neutral")
    if len(directions) == 2:
        return "contradictory", "Low"
    consistency = "consistent" if directions and stating >= 2 else "mixed"
    profiles = theme.profiles
    strong = sum(1 for p in profiles if p.confidence == "High")
    empirical = sum(1 for p in profiles if p.study.kind.empirical)
    if consistency == "consistent" and len(profiles) >= 3 and strong >= 2:
        return consistency, "High"
    if (consistency == "consistent" and empirical) or strong:
        return consistency, "Medium"
    return consistency, "Low"


# ---- power analysis ---------------------------------------------------------------------------

Z_ALPHA = NormalDist().inv_cdf(1 - ALPHA / 2)
Z_POWER = NormalDist().inv_cdf(POWER)
POWER_DESIGNS = {"rct", "quasi-experimental", "observational", "survey", "experiment"}


def n_two_groups(d: float) -> int:
    """Participants per group for a two-sided independent t-test (normal approximation with Guenther's correction)."""
    return math.ceil(2 * ((Z_ALPHA + Z_POWER) / d) ** 2 + Z_ALPHA ** 2 / 4)


def n_correlation(r: float) -> int:
    """Participants to detect a correlation r (Fisher's z approximation)."""
    return math.ceil(((Z_ALPHA + Z_POWER) / math.atanh(abs(r))) ** 2 + 3)


def n_two_proportions(p1: float, p2: float) -> int:
    """Participants per group to detect a difference between two proportions."""
    mean = (p1 + p2) / 2
    root = Z_ALPHA * math.sqrt(2 * mean * (1 - mean)) + Z_POWER * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    return math.ceil(root ** 2 / (p1 - p2) ** 2)


def _value(text: str) -> Optional[float]:
    number = re.search(r"-?\d*\.\d+|-?\d+", text.split("=")[-1])
    return float(number.group(0)) if number else None


def power_rows(profiles: Sequence[ResultProfile], tradition: Tradition) -> List[Tuple[str, str, str]]:
    """(analysis, assumed effect, required sample) at alpha 0.05 and 80% power."""
    rows = [
        ("Two-group comparison (t-test)", "small effect, d = 0.2", f"{n_two_groups(0.2)} per group"),
        ("Two-group comparison (t-test)", "medium effect, d = 0.5", f"{n_two_groups(0.5)} per group"),
        ("Two-group comparison (t-test)", "large effect, d = 0.8", f"{n_two_groups(0.8)} per group"),
        ("Correlation", "small, r = 0.1", f"{n_correlation(0.1)} in total"),
        ("Correlation", "medium, r = 0.3", f"{n_correlation(0.3)} in total"),
        ("Correlation", "large, r = 0.5", f"{n_correlation(0.5)} in total"),
    ]
    if tradition.key == "health":
        rows.append(("Binary outcome (two proportions)", "event rate 20% vs 10%", f"{n_two_proportions(0.2, 0.1)} per group"))
    for profile in profiles:
        for stat in profile.statistics:
            value = _value(stat.text)
            if value is None:
                continue
            if stat.kind == "correlation" and 0.05 <= abs(value) < 1:
                rows.append(("Correlation", f"as reported by {profile.study.narrative}: {stat.text}", f"{n_correlation(value)} in total"))
            elif stat.kind == "effect-size" and re.match(r"(?i)(?:cohen'?s\s+)?[dg]\b", stat.text) and 0.05 <= abs(value) < 3:
                rows.append(("Two-group comparison (t-test)", f"as reported by {profile.study.narrative}: {stat.text}", f"{n_two_groups(abs(value))} per group"))
    return rows


# ---- analysis plan ----------------------------------------------------------------------------

QUANT_TESTS: Tuple[Tuple[str, str, str], ...] = (
    ("Compare two independent groups on a continuous outcome", "Independent-samples t-test (Welch's)", "Mann–Whitney U"),
    ("Compare three or more groups", "One-way ANOVA with post-hoc tests (e.g. Tukey)", "Kruskal–Wallis"),
    ("Compare the same participants at two or more time points", "Paired t-test or repeated-measures ANOVA", "Wilcoxon signed-rank or Friedman"),
    ("Relationship between two continuous variables", "Pearson correlation", "Spearman's rho"),
    ("Predict an outcome from several variables", "Multiple linear regression (logistic for a binary outcome)", "Robust or ordinal regression"),
    ("Association between two categorical variables", "Chi-square test of independence", "Fisher's exact test"),
    ("Test a model of latent constructs measured with multi-item scales", "Confirmatory factor analysis and structural equation modelling", "PLS-SEM for small samples"),
)
FIELD_TESTS: Dict[str, Tuple[Tuple[str, str, str], ...]] = {
    "health": (
        ("Binary outcome between groups", "Risk ratio or odds ratio with 95% CI; logistic regression to adjust for confounders", "Exact methods for rare events"),
        ("Time to an event", "Kaplan–Meier curves and Cox proportional-hazards regression", "Log-rank test"),
        ("Safety", "Adverse-event rates per group with exact confidence intervals", "Descriptive reporting when events are rare"),
    ),
    "engineering": (
        ("Compare two methods over repeated runs", "Mean ± SD (or 95% CI) over 3–10 seeds; paired t-test across seeds", "Wilcoxon signed-rank"),
        ("Compare several methods over several datasets", "Friedman test with Nemenyi post-hoc (Demšar, 2006)", "Critical-difference diagram"),
        ("Attribute the gain to components", "Ablation study: remove one component at a time", "—"),
    ),
}
QUAL_METHODS: Dict[str, Tuple[str, ...]] = {
    "interview": ("Reflexive thematic analysis (Braun and Clarke): familiarisation, coding, generating, reviewing and naming themes, writing up",
                  "Grounded theory (e.g. Charmaz) instead, if the aim is to build a theory where none exists",
                  "Interpretative phenomenological analysis instead, for lived experience in a small, homogeneous sample"),
    "ethnography": ("Iterative thematic coding of field notes and transcripts", "Thick description of practices in their context"),
    "case-study": ("Within-case analysis: pattern matching or process tracing against the propositions from the reviewed studies",
                   "Cross-case synthesis when several cases are studied",
                   "Thematic analysis of the case interviews and documents"),
    "doctrinal": ("Doctrinal interpretation (textual, purposive and systematic) of the primary sources",
                  "Comparative analysis across jurisdictions", "Case-outcome coding when many decisions are analysed"),
    "historical": ("Source criticism: provenance, reliability and bias of each source", "Process tracing of the sequence of events",
                   "Content or discourse analysis of the documents"),
}
QUAL_RIGOUR = (
    "Keep a codebook and an audit trail of coding decisions; have a second coder code a sample (report agreement, e.g. "
    "Cohen's κ) or, for reflexive approaches, discuss the codes with a peer; support themes with quotations; and use "
    "software such as NVivo, ATLAS.ti or Taguette."
)


def _strands(rec: Recommendation) -> Tuple[Optional[str], Optional[str]]:
    """The recommended design's (quantitative, qualitative) design keys."""
    if rec.components:
        return rec.components[0].key, rec.components[1].key
    if rec.design.approach in (QUANT, DESIGN_SCIENCE):
        return rec.design.key, None
    if rec.design.approach == QUAL:
        return None, rec.design.key
    return None, None


# ---- the results section ----------------------------------------------------------------------

@dataclass
class ResultsInputs:
    topic: str
    review_type: str
    slug: str
    profiles: List[ResultProfile]
    themes: List[Theme]
    tradition: Tradition
    recommendation: Recommendation
    from_phase2: bool  # the design was read from Phase 2's analysis file
    bibliography: str


def _quote(profile: ResultProfile, evidence: Evidence, cite: bool = True) -> str:
    """A finding with its source: the paper's own words in quotes, Phase 1's paraphrase marked as such."""
    text = evidence.text.rstrip(".")
    source = f" ({profile.study.cite})" if cite else ""
    if evidence.origin == "abstract":
        return f"\"{text}.\"{source}"
    return f"According to the Phase 1 summary: {text}{source}."


def _sample_brief(study: StudyDesign) -> str:
    """Size, then who and where: "300 cybersecurity practitioners; practitioners and employees; in Estonia"."""
    sizes = "; ".join(s["text"] for s in study.samples)
    if not sizes:
        sizes = "datasets (size not stated)" if "datasets" in study.collection else "size not stated"
    parts = [sizes, *study.populations[:2]]
    if study.countries:
        parts.append("in " + join_items(study.countries[:3]))
    return "; ".join(parts)


def _stats_brief(profile: ResultProfile, limit: int = 4) -> str:
    effects = profile.effects()[:limit]
    return "; ".join(s.text for s in effects) or "Not reported"


def build_results(inputs: ResultsInputs) -> str:
    topic, profiles, themes, tradition = inputs.topic, inputs.profiles, inputs.themes, inputs.tradition
    n = len(profiles)
    lines = [
        f"# Results: {topic}",
        "",
        f"*Results of the {_count(n, 'study', 'studies')} in the {inputs.review_type}, with a data analysis plan for your own study*",
        "",
        f"> **How to complete these results.** The findings below come from the reviewed studies. Where your own results "
        f"go, the text is marked **{markers.YOUR_DATA_ANALYSIS}**: report what your data show on that point and how it "
        f"compares with the studies. Where you choose your analysis, it is marked **{markers.YOUR_ANALYSIS_METHOD}**. "
        "Quotation marks enclose the papers' own words from their abstracts; findings taken from the Phase 1 summaries "
        "are paraphrases and say so.",
        "",
    ]

    # 1. Overview
    types = Counter(p.result_type for p in profiles)
    with_stats = [p for p in profiles if p.effects()]
    with_sample = [p for p in profiles if p.study.samples]
    overview = [
        f"The {_count(n, 'study', 'studies')} reviewed on {topic} report "
        + join_items(f"{_lc(t)} results ({c})" for t, c in types.most_common()) + "."
    ]
    if len(with_stats) == n:
        overview.append("Every study reports statistics or effect estimates.")
    elif with_stats:
        overview.append(
            f"{_cap(_count(len(with_stats), 'study reports', 'studies report'))} statistics or effect estimates "
            f"({_cites([p.study for p in with_stats])}); the rest report their results in words only."
        )
    else:
        overview.append("None of the abstracts reports statistics or effect sizes, so the results below are stated in words; extract the numbers from the full texts where they exist.")
    if with_sample:
        overview.append(f"{_cap(_count(len(with_sample), 'study states', 'studies state'))} {'its' if len(with_sample) == 1 else 'their'} sample size ({_cites([p.study for p in with_sample])}).")
    if themes:
        consistent = [t for t in themes if t.consistency == "consistent"]
        conflicting = [t for t in themes if t.consistency == "contradictory"]
        overview.append(
            f"{_cap(_count(len(themes), 'finding is', 'findings are'))} reported by more than one study ("
            + join_items(t.term for t in themes) + f"): {len(consistent)} consistent, {len(conflicting)} contradictory and "
            f"{len(themes) - len(consistent) - len(conflicting)} mixed."
        )
    else:
        overview.append("No finding is reported by more than one study, so each study's results are presented on their own.")
    overview.append(FIELD_EXPECTATIONS[tradition.key])
    lines += ["## 1. Overview of Findings", "", " ".join(overview), ""]

    # 2. Findings
    lines += ["## 2. Findings Across the Studies", ""]
    number = 0
    for theme in themes:
        number += 1
        studies = [p.study for p in theme.profiles]
        lines += [f"### 2.{number} {_cap(theme.term)}", ""]
        who = (f"All {SPELLED.get(n, n)} studies" if len(studies) == n
               else f"{_cap(str(SPELLED.get(len(studies), len(studies))))} of the {SPELLED.get(n, n)} studies")
        lines += [f"{who} report findings on {theme.term} ({_cites(studies)}).", ""]
        for profile, evidence in theme.evidence:
            stats = _stats_brief(profile, 3)
            detail = f"{_lc(profile.study.kind.label)}; sample: {_sample_brief(profile.study)}"
            if stats != "Not reported":
                detail += f"; {stats}"
            lines.append(f"- **{profile.study.narrative}** ({detail}): {_quote(profile, evidence)}")
        lines.append("")
        directions = Counter(e.direction for _, e in theme.evidence)
        consistency = {
            "consistent": "The studies' findings point the same way",
            "contradictory": "The studies' findings point in different directions: some support it and some qualify it",
            "mixed": "The studies report different aspects of it, so their findings can't simply be added up",
        }[theme.consistency]
        strength = (
            f"**Strength of evidence: {theme.consistency}.** {consistency} "
            f"({', '.join(f'{k}: {v}' for k, v in directions.items())}). **Confidence: {theme.confidence.lower()}**, "
            f"from {_count(len(studies), 'study', 'studies')} rated "
            + join_items(f"{p.confidence.lower()} ({p.study.cite})" for p in theme.profiles) + "."
        )
        caveats = [f"{p.study.narrative}: {p.confidence_reason}" for p in theme.profiles if p.confidence != "High"]
        if caveats:
            strength += " Limitations: " + "; ".join(caveats) + "."
        lines.append(strength)
        lines += [
            "",
            markers.marker(
                markers.YOUR_DATA_ANALYSIS,
                f"Report what your data show on {theme.term}: the result, its size (e.g. the statistic, effect size and "
                f"confidence interval, or the theme with a supporting quotation) and whether it confirms, refines or "
                f"contradicts {_cites(studies)}.",
            ),
            "",
        ]
    covered = {id(p) for t in themes for p in t.profiles}
    others = [p for p in profiles if id(p) not in covered and p.lead]
    if others:
        number += 1
        lines += [f"### 2.{number} {'Further findings' if themes else 'Findings of each study'}", ""]
        if themes:
            lines += ["These findings are reported by one study each, so they rest on single-study evidence.", ""]
        for profile in others:
            lines.append(f"- **{profile.study.narrative}** ({_lc(profile.study.kind.label)}; sample: {_sample_brief(profile.study)}; "
                         f"confidence: {profile.confidence.lower()}): {_quote(profile, profile.lead)}")
        lines += [
            "",
            markers.marker(
                markers.YOUR_DATA_ANALYSIS,
                "Report your own results on the questions these studies address, and say for each whether your data "
                "replicate it; single-study findings are the ones most in need of replication.",
            ),
            "",
        ]
    missing = [p for p in profiles if not p.lead]
    if missing:
        lines += [f"{join_items(p.study.narrative for p in missing)} state{'s' if len(missing) == 1 else ''} no findings in the text available.", ""]

    # 3. Table
    lines += [
        "## 3. Summary of Results",
        "",
        "| Study | Paper title | Key finding | Sample | Effect size / statistics | Confidence |",
        "|---|---|---|---|---|---|",
    ]
    for p in profiles:
        finding = _clip(p.lead.text, 30) if p.lead else "Not stated"
        lines.append(
            f"| {_cell(p.study.narrative)} | {_cell(p.study.title)} | {_cell(finding)} | {_cell(_sample_brief(p.study))} | "
            f"{_cell(_stats_brief(p))} | {p.confidence} |"
        )
    lines += [
        f"| Your study | {markers.YOUR_DATA_ANALYSIS} | Your main finding | Your sample | Your effect size | — |",
        "",
        "*Confidence: high = empirical, with a stated sample or dataset and reported statistics; medium = one of the two, "
        "or a systematic synthesis; low = neither, or no new data. It rates what the abstract reports, not the full study.*",
        "",
    ]

    # 4. Field-specific results
    title, kinds = FIELD_SECTION[tradition.key]
    lines += [f"## 4. {title}", "", FIELD_EXPECTATIONS[tradition.key], ""]
    rows = [(p, s) for p in profiles for s in p.statistics if s.kind in kinds]
    if rows:
        lines += ["| Study | Measure | Reported value | Context |", "|---|---|---|---|"]
        lines += [f"| {_cell(p.study.narrative)} | {s.label} | {_cell(s.text)} | {_cell(_clip(s.sentence, 25))} |" for p, s in rows]
        lines.append("")
    statements = [(p, s) for p in profiles for s in p.field_statements]
    for key, label, _ in FIELD_STATEMENTS.get(tradition.key, ()):
        group = [(p, s) for p, s in statements if s["kind"] == key]
        if group:
            lines += [f"**{label}.**", ""]
            lines += [f"- {p.study.narrative}: \"{s['sentence'].rstrip('.')}.\"" for p, s in group]
            lines.append("")
    if tradition.key in ("social", "humanities"):
        qualitative = [p for p in profiles if p.result_type in (QUAL_RESULTS, MIXED_RESULTS, CONCEPTUAL_RESULTS) and p.lead]
        if qualitative and not any(s["kind"] == "thematic" for _, s in statements):
            lines += ["**Qualitative and interpretive findings.**", ""]
            lines += [f"- {p.study.narrative}: {_quote(p, p.lead, cite=False)}" for p in qualitative]
            lines.append("")
    absent = _field_gaps(profiles, tradition)
    if absent:
        lines += [" ".join(absent), ""]

    # 5. Synthesis
    lines += ["## 5. Synthesis and Integration", "", "### Consensus findings", ""]
    consistent = [t for t in themes if t.consistency == "consistent"]
    if consistent:
        lines += [f"- **{_cap(t.term)}** ({_cites([p.study for p in t.profiles])}; confidence: {t.confidence.lower()})" for t in consistent]
    else:
        lines.append("- No finding is reported consistently by two or more studies; the evidence consists of single-study or mixed findings.")
    lines += ["", "### Mixed and contradictory findings", ""]
    divided = [t for t in themes if t.consistency != "consistent"]
    if divided:
        for t in divided:
            sides = Counter(e.direction for _, e in t.evidence)
            lines.append(f"- **{_cap(t.term)}** ({t.consistency}): " + "; ".join(
                f"{p.study.narrative} {'supports' if e.direction == 'supports' else 'qualifies' if e.direction == 'qualifies' else 'describes'} it"
                for p, e in t.evidence) + (". Look for differences in samples, settings or measures that could explain the disagreement." if sides["supports"] and sides["qualifies"] else "."))
    else:
        lines.append("- None identified from the wording of the findings.")
    lines += ["", "### Gaps in the evidence", ""]
    lines += [f"- {gap}" for gap in _evidence_gaps(profiles, themes, tradition)]
    lines += [
        "",
        "*Consistency is read from the wording of each finding (e.g. \"improves\" vs \"limited\"), so check every "
        "judgement against the papers before relying on it.*",
        "",
        "### Integrating your results",
        "",
        markers.marker(
            markers.YOUR_DATA_ANALYSIS,
            "Once your data are analysed, set your main results against the consensus and contradictory findings above: "
            "which do your results support, which do they challenge, and which gap in the evidence do they fill?",
        ),
        "",
    ]

    # 6. Analysis plan
    lines += _analysis_plan(inputs)
    lines += [
        "*Phase 3 reads findings and statistics from the papers' abstracts and the Phase 1 summaries, so results reported "
        f"only in a full text are missed. Every finding and the sentence it was read from are listed in {inputs.slug}-results-analysis.json.*",
        "",
        "## References",
        "",
        inputs.bibliography,
        "",
    ]
    return "\n".join(lines)


def _field_gaps(profiles: Sequence[ResultProfile], tradition: Tradition) -> List[str]:
    kinds = {s.kind for p in profiles for s in p.statistics}
    statements = {s["kind"] for p in profiles for s in p.field_statements}
    notes = []
    if tradition.key == "health":
        if not kinds & {"odds-ratio", "risk-ratio"}:
            notes.append("None of the abstracts reports an odds, risk or hazard ratio.")
        if "confidence-interval" not in kinds:
            notes.append("No confidence intervals are reported, so the precision of the effects can't be judged from the abstracts.")
        if "safety" not in statements:
            notes.append("No safety outcomes (adverse events, side effects) are reported; check the full texts.")
    elif tradition.key == "social":
        if not kinds & {"correlation", "r-squared", "beta", "effect-size"}:
            notes.append("No correlations, regression coefficients or effect sizes are reported.")
        if not kinds & DESCRIPTIVE_KINDS:
            notes.append("No descriptive statistics (means, percentages) are reported.")
    elif tradition.key == "engineering":
        if "performance" not in kinds:
            notes.append("None of the abstracts reports a performance figure (accuracy, F1, error rate, speed-up), so the "
                         "methods' gains can't be compared from them; take the figures from the full texts' result tables.")
        if not kinds & {"confidence-interval", "p-value"}:
            notes.append("No variation across runs or significance tests are reported.")
    elif tradition.key == "law" and "outcome" not in statements:
        notes.append("No case outcomes are stated in the abstracts; identify the decisions each study relies on from the full texts.")
    return notes


def _evidence_gaps(profiles: Sequence[ResultProfile], themes: Sequence[Theme], tradition: Tradition) -> List[str]:
    gaps = []
    no_sample = [p.study for p in profiles if p.study.kind.empirical and not p.study.samples and "datasets" not in p.study.collection]
    if no_sample:
        gaps.append(f"Sample sizes are not stated by {join_items(s.narrative for s in no_sample)}.")
    no_effects = [p.study for p in profiles if p.study.kind.empirical and not p.effects()]
    if no_effects:
        gaps.append(f"No effect sizes or statistics are reported by {join_items(s.narrative for s in no_effects)}, so the size of their effects is unknown.")
    non_empirical = [p.study for p in profiles if not p.study.kind.empirical]
    if non_empirical:
        gaps.append(f"{_cap(_count(len(non_empirical), 'study', 'studies'))} ({_cites(non_empirical)}) collected no new data; their results are arguments or syntheses.")
    countries = Counter(c for p in profiles for c in p.study.countries)
    if countries and len(countries) <= 3:
        gaps.append(f"The evidence comes from few settings ({join_items(list(countries))}); results may not transfer elsewhere.")
    elif not countries:
        gaps.append("The abstracts do not state where the studies were carried out.")
    if not themes and len(profiles) > 1:
        gaps.append("No finding has been replicated across studies.")
    for p in profiles:
        stated = next((l for l in p.limitations if l.origin == "abstract"), None)
        if stated:
            gaps.append(f"{p.study.narrative} notes: \"{stated.text.rstrip('.')}.\"")
        elif p.limitations:
            gaps.append(f"{p.study.narrative}, according to Phase 1: {p.limitations[-1].text}")
    return gaps


def _analysis_plan(inputs: ResultsInputs) -> List[str]:
    rec, tradition, profiles = inputs.recommendation, inputs.tradition, inputs.profiles
    quant, qual = _strands(rec)
    label = rec.design.label
    source = "Phase 2's recommended design" if inputs.from_phase2 else "the design Phase 2 recommends for these studies"
    lines = [
        "## 6. Data Analysis Plan",
        "",
        "### 6.1 Analysis approach",
        "",
        f"The analysis follows {source}, {_a(label)} ({rec.design.approach.lower()}). "
        + GUIDES[rec.design.key].summary,
        "",
    ]
    used = Counter(k for p in profiles for k in p.study.analysis)
    if used:
        lines += ["The reviewed studies analysed their data with " + join_items(
            f"{_lc(ANALYSIS_LABELS[k])} ({_cites([p.study for p in profiles if k in p.study.analysis])})" for k, _ in used.most_common()
        ) + "; using comparable methods lets you compare your results with theirs directly.", ""]

    if quant:
        lines += [f"**Quantitative analysis ({_lc(DESIGN_BY_KEY[quant].label)}).**", ""]
        if quant in ("computational-experiment", "simulation", "design-development") and tradition.key != "health":
            tests = FIELD_TESTS["engineering"] + QUANT_TESTS[3:5]
        else:
            tests = (FIELD_TESTS["health"] if tradition.key == "health" else ()) + QUANT_TESTS
        lines += ["| Question | Recommended test | If assumptions fail |", "|---|---|---|"]
        lines += [f"| {q} | {t} | {alt} |" for q, t, alt in tests]
        lines += [
            "",
            "Start with descriptive statistics (means and SDs, or medians and ranges, and percentages) for every variable; "
            "check the tests' assumptions (normality, equal variances, independence); report effect sizes with 95% "
            "confidence intervals alongside p-values; and correct for multiple comparisons (e.g. Holm) when running many tests.",
            "",
        ]
    if qual:
        lines += [f"**Qualitative analysis ({_lc(DESIGN_BY_KEY[qual].label)}).**", ""]
        methods = QUAL_METHODS.get(qual) or tuple(text for _, text in GUIDES[qual].analysis)
        lines += [f"- {m}" for m in methods]
        lines += ["", QUAL_RIGOUR, ""]
    if quant and qual:
        lines += [
            "**Integrating the strands.** Analyse the quantitative strand first, then use its results to select cases or "
            "participants for the qualitative strand (e.g. the highest and lowest scorers) and to focus its questions. "
            "Integrate them in a joint display that sets each statistical result beside the themes that explain it, and "
            "note where the strands agree, complement or contradict each other.",
            "",
        ]
    if not quant and not qual:
        lines += [f"- {text}" for _, text in GUIDES[rec.design.key].analysis]
        lines.append("")
    lines += [
        markers.marker(
            markers.YOUR_ANALYSIS_METHOD,
            "Specify the analysis you will run: for each research question or hypothesis, the variables or themes, the "
            "exact test or coding procedure, the software and version, and the significance level or quality criteria.",
        ),
        "",
    ]

    # 6.2 Sample size
    lines += ["### 6.2 Sample size", "", "**From the reviewed studies.** " + " ".join(_sample_sentences([p.study for p in profiles])), ""]
    for key in [k for k in (quant, qual) if k] or [rec.design.key]:
        if GUIDES[key].sample:
            lines.append(f"- **{DESIGN_BY_KEY[key].label}:** {GUIDES[key].sample}")
    lines.append("")

    # 6.3 Power analysis
    lines += ["### 6.3 Power analysis", ""]
    if quant in POWER_DESIGNS:
        lines += [
            f"Before collecting data, calculate the sample needed to detect the smallest effect that matters, at "
            f"α = {ALPHA} (two-sided) and {POWER:.0%} power. Approximate requirements:",
            "",
            "| Analysis | Assumed effect | Required sample |",
            "|---|---|---|",
        ]
        lines += [f"| {a} | {_cell(e)} | {n} |" for a, e, n in power_rows(profiles, tradition)]
        reported = [(p, s) for p in profiles for s in p.statistics if s.kind in ("r-squared", "beta", "odds-ratio", "risk-ratio")]
        lines.append("")
        if reported:
            lines += [
                "The reviewed studies report " + join_items(f"{s.text} ({p.study.cite})" for p, s in reported)
                + ". Published effects tend to be inflated, so base your calculation on a smaller effect than the largest one reported.",
                "",
            ]
        lines += [
            "For regression, ANOVA or structural equation models, compute the exact number in G*Power or R (pwr, simr) "
            "for your number of predictors or groups, and add 10–20% for drop-out or incomplete responses.",
            "",
            markers.marker(
                markers.YOUR_ANALYSIS_METHOD,
                "State the effect size you expect and where it comes from (a reviewed study or a pilot), the resulting "
                "sample size, and how many people you will invite to reach it.",
            ),
            "",
        ]
    elif quant in ("computational-experiment", "simulation", "design-development"):
        lines += [
            "Statistical power is set by the number of independent runs and test cases rather than participants: run each "
            "method with several random seeds (3–10), on more than one dataset or scenario, and test whether the "
            "differences exceed the run-to-run variation (paired tests across seeds or datasets).",
            "",
        ]
    else:
        lines += [
            "A power analysis does not apply to this design: the sample is set by "
            + ("saturation (stop when new data add no new themes)" if qual in ("interview", "ethnography") else
               "the selection of cases or sources, which you should justify explicitly") + ".",
            "",
        ]
    return lines


def analysis_json(inputs: ResultsInputs, source_path: Path) -> Dict:
    rec = inputs.recommendation
    quant, _ = _strands(rec)
    return {
        "version": 1,
        "topic": inputs.topic,
        "review": str(source_path),
        "review_type": inputs.review_type,
        "papers_analysed": len(inputs.profiles),
        "field": {"tradition": inputs.tradition.key, "tradition_name": inputs.tradition.name},
        "design": {"key": rec.design.key, "label": rec.design.label, "approach": rec.design.approach,
                   "strands": [d.key for d in rec.components], "from_phase2": inputs.from_phase2},
        "studies": [p.to_dict() for p in inputs.profiles],
        "themes": [t.to_dict() for t in inputs.themes],
        "power_analysis": [
            {"analysis": a, "assumed_effect": e, "required_sample": n}
            for a, e, n in (power_rows(inputs.profiles, inputs.tradition) if quant in POWER_DESIGNS else [])
        ],
    }


# ---- running Phase 3 ----------------------------------------------------------------------------

def resolve_input(path: Path) -> Path:
    """Maps a Phase 2/3 output (<review>-methodology.md, ...) to the Phase 1 review or paper file it was made from."""
    stem = path.stem
    for suffix in DERIVED_SUFFIXES:
        if stem.endswith(suffix) and not path.name.endswith(SIDECAR_SUFFIX):
            base = path.with_name(stem[: -len(suffix)] + ".md")
            if base.is_file():
                return base
            if sidecar_path(base).is_file():
                return sidecar_path(base)
    return path


def phase2_analysis(base: Path, output_dir: Path) -> Optional[Dict]:
    """Phase 2's <review>-methodology-analysis.json, from the output or the review's directory."""
    for directory in dict.fromkeys((output_dir, base.parent)):
        path = directory / f"{base.stem}-methodology-analysis.json"
        if path.is_file():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                logger.warning("Could not read %s (%s); recomputing the design", path.name, exc)
    return None


def apply_phase2(studies: Sequence[StudyDesign], data: Dict, field_override: Optional[str]) -> Tuple[Optional[Tradition], Optional[Recommendation]]:
    """Takes Phase 2's designs (including ones the model classified), field and recommendation."""
    designs = {s.get("title"): s for s in data.get("studies", [])}
    for study in studies:
        saved = designs.get(study.title)
        if saved and saved.get("design") in DESIGN_BY_KEY and saved["design"] != study.design:
            study.design = saved["design"]
            study.classified_by = saved.get("classified_by", study.classified_by)
    tradition = TRADITIONS.get(field_override or data.get("field", {}).get("tradition"))
    saved_rec = data.get("recommendation") or {}
    if saved_rec.get("design") not in DESIGN_BY_KEY:
        return tradition, None
    components = tuple(DESIGN_BY_KEY[s["design"]] for s in saved_rec.get("strands", []) if s.get("design") in DESIGN_BY_KEY)
    alternatives = tuple(d for d in DESIGN_BY_KEY.values() if d.label in saved_rec.get("alternatives", []))
    return tradition, Recommendation(DESIGN_BY_KEY[saved_rec["design"]], saved_rec.get("basis", "common"), components, alternatives)


@dataclass
class ResultsAnalysis:
    tradition: Tradition
    recommendation: Recommendation
    profiles: List[ResultProfile]
    themes: List[Theme]
    from_phase2: bool


def analyse_results(corpus: Corpus, topic: str, base: Path, output_dir: Path, field_override: Optional[str],
                    no_llm: bool, model: str) -> ResultsAnalysis:
    """Every paper's results profile and the shared findings, using Phase 2's design and field when available."""
    studies = [analyse_paper(source) for source in corpus.papers]
    saved = phase2_analysis(base, output_dir)
    tradition, rec = apply_phase2(studies, saved, field_override) if saved else (None, None)
    if saved:
        logger.info("Using Phase 2's analysis (%s-methodology-analysis.json)", base.stem)
    else:
        logger.info("No Phase 2 analysis found for %s; working out the design as Phase 2 would", base.stem)
    if not no_llm and any(s.design == UNCLASSIFIED.key for s in studies):
        classify_unclear(studies, corpus.papers, model)
    if tradition is None:
        tradition, _ = detect_tradition(corpus, studies, field_override)
    if rec is None:
        rec = recommend(studies, tradition)

    profiles = [profile_paper(source, study, tradition) for source, study in zip(corpus.papers, studies)]
    themes = find_themes(profiles, topic)
    for p in profiles:
        logger.info(
            "%s: %s results, %d finding(s), %d statistic(s), %s confidence",
            p.study.narrative, p.result_type.lower(), len(p.findings), len(p.statistics), p.confidence.lower(),
        )
    logger.info(
        "Field: %s. %d shared finding(s): %s. Analysis plan for: %s",
        tradition.name, len(themes), ", ".join(f"{t.term} ({t.consistency})" for t in themes) or "none", rec.design.label,
    )
    return ResultsAnalysis(tradition, rec, profiles, themes, bool(saved))


def run_phase3(args) -> List[Path]:
    load_dotenv()
    input_path = Path(args.input)
    if not input_path.is_file():
        raise ValueError(f"Review file not found: {input_path}")
    review, base = resolve_review(resolve_input(input_path))
    corpus: Corpus = load_corpus(review)
    if not corpus.papers:
        raise ValueError(
            f"No papers found for {input_path}: Phase 3 needs the review's {SIDECAR_SUFFIX} file or its References section"
        )
    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = base.stem
    formats = [f.strip().lower() for f in args.formats.split(",") if f.strip()]
    style = (args.citation_style or corpus.citation_style or "apa").lower()
    topic = corpus.topic or review.topic

    logger.info("Phase 3: extracting the results of %d papers on %r", len(corpus.papers), topic)
    analysis = analyse_results(corpus, topic, base, output_dir, getattr(args, "field", None), args.no_llm, args.model)
    inputs = ResultsInputs(
        topic=topic,
        review_type=review.review_type,
        slug=slug,
        profiles=analysis.profiles,
        themes=analysis.themes,
        tradition=analysis.tradition,
        recommendation=analysis.recommendation,
        from_phase2=analysis.from_phase2,
        bibliography=CitationFormatter().format_bibliography([p.paper for p in corpus.papers], style=style),
    )
    markdown = build_results(inputs)
    written = Exporter(output_dir=output_dir).export_all(markdown, f"{slug}-results", formats)
    analysis_path = output_dir / f"{slug}-results-analysis.json"
    analysis_path.write_text(json.dumps(analysis_json(inputs, input_path), indent=2, ensure_ascii=False), encoding="utf-8")
    written.append(analysis_path)

    placed = sum(1 for line in markdown.splitlines() if markers.RESULTS_MARKER_LINE.match(line))
    for path in written:
        logger.info("Written: %s", path)
    logger.info("%d places are marked for your own data and analysis (%s, %s).", placed,
                markers.YOUR_DATA_ANALYSIS, markers.YOUR_ANALYSIS_METHOD)
    return written
