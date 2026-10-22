"""Phase 4: writes the Conclusion, Implications and Future Directions from Phases 1-3.

    python -m src.main --phase 4 --input "output/my-review.md"

Phase 1 reviews what is known, Phase 2 proposes how the user's study is done and Phase 3 sets out what
the reviewed studies found; Phase 4 draws the three together to close the thesis or review. It reads
the review (its research gaps), the papers (from the <review>.researchflow.json file Phase 1 writes),
Phase 2's design and field (from <review>-methodology-analysis.json, or worked out again) and Phase 3's
results (recomputed the way Phase 3 computes them), and writes:

    Conclusion          what the field knows, how the proposed study builds on it, what the analysis
                        found, the contribution, the limitations, and the implications in brief
    Implications        practical, policy and theoretical, then field-specific ones: clinical significance,
                        safety and treatment guidelines in medicine; policy, social impact and behaviour
                        change in the social sciences; design standards, performance and scalability in
                        engineering; precedent, policy reform and the justice system in law; theory,
                        interpretation and cultural understanding in the humanities
    Future Directions   open questions, next steps, methodological improvements, replication and scaling,
                        interdisciplinary approaches, each with a timeline and its feasibility

Every statement is built from the analysis rather than copied from Phases 1-3, and quality checks
confirm it: no passage repeats the review, the methodology, the results or the papers verbatim, every
citation is to a reviewed paper and every paper is cited, each phase is drawn on, and every future
direction has a timeline and feasibility. What only the user's finished study can say is marked:

    [YOUR CONTRIBUTION]           what the user's own results add to the field
    [YOUR STUDY'S LIMITATIONS]    the limitations of the study as it was actually carried out
    [YOUR RESEARCH PRIORITIES]    which future direction the user would pursue first

Like Phases 2 and 3 it is rule-based and runs without a model. Writes <review>-conclusion.md and .docx,
plus <review>-conclusion-analysis.json with the directions, the evidence behind them and the checks.
"""
from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

from dotenv import load_dotenv

from . import critical_gaps as markers
from .citation_formatter import CitationFormatter
from .export import Exporter
from .gap_parser import ReviewDocument, citations_in
from .gap_research import SIDECAR_SUFFIX, Corpus, load_corpus, sidecar_path
from .llm_synthesis import join_items
from .phase2 import (
    CONCEPTUAL,
    DESIGN_BY_KEY,
    DESIGN_SCIENCE,
    MIXED,
    QUAL,
    QUANT,
    SPELLED,
    SYNTHESIS,
    FIELD_TRADITION,
    MethodologyInputs,
    StudyDesign,
    _a,
    _cap,
    _cell,
    _cites,
    _count,
    _lc,
    build_methodology,
    compare,
    resolve_review,
)
from .phase3 import (
    POWER_DESIGNS,
    QUAL_RESULTS,
    ResultProfile,
    ResultsAnalysis,
    ResultsInputs,
    Theme,
    _strands,
    analyse_results,
    build_results,
    n_two_groups,
    resolve_input as resolve_phase3_input,
)
from .review_methodology import METHODOLOGIES

logger = logging.getLogger("researchflow")

PHASE4_FORMATS = "md,docx"
SHINGLE_WORDS = 6
MAX_COPIED_SHARE = 0.10  # of the conclusion's 6-word phrases found in Phases 1-3 or the papers
MAX_COPIED_RUN = 15  # words in a row
CONCLUSION_WORDS = (350, 1600)  # expected length of the Conclusion proper (section 1)

HIGH, MEDIUM, LOW = "High", "Medium", "Low"


# ---- the Phase 1 review's research gaps -------------------------------------------------------

@dataclass
class ReviewGap:
    title: str
    kind: str  # GAP_KINDS key
    source: str  # "review": a gap the review names; otherwise the paper whose stated limitation it is ("Lindsay (2025)")


# (key, cues, unanswered question, next step, suggested design, timeline, feasibility, why feasible)
GAP_KINDS: Tuple[Tuple[str, Tuple[str, ...], str, str, str, str, str, str], ...] = (
    ("longitudinal", (r"long-term", r"longitudinal", r"over time", r"single point in time", r"over several years"),
     "Do the effects and relationships reported for {topic} persist, grow or fade over time?",
     "Follow the same participants, organisations or systems over several measurement waves",
     "Longitudinal panel or cohort study", "3–5 years", LOW,
     "multi-year follow-up exceeds a single thesis; feasible as a funded or collaborative project"),
    ("integration", (r"integration", r"in isolation", r"unifying", r"strands", r"connect"),
     "How do the separate strands of research on {topic} influence one another?",
     "Study the strands side by side with shared measures, or build and test a framework that links them",
     "Comparative study or integrative framework", "1–2 years", MEDIUM,
     "needs expertise in each strand, but can reuse their existing measures and data"),
    ("generalisability", (r"generali[sz]", r"transfer", r"across contexts", r"other countries", r"settings?", r"populations"),
     "Which findings on {topic} hold beyond the settings in which they were first observed?",
     "Repeat the strongest study's protocol in at least two new countries, sectors or populations",
     "Multi-site or cross-national replication", "1–2 years", MEDIUM,
     "requires partners or data access in each new setting"),
    ("evidence", (r"empirical", r"untested", r"tested in practice", r"validat", r"\bevidence\b"),
     "Do the frameworks and arguments put forward for {topic} hold up when tested against data?",
     "Turn the proposed frameworks into hypotheses and test them on new, independently collected data",
     "Pre-registered empirical test", "6–18 months", HIGH,
     "fits a single study, using the design proposed in Phase 2"),
    ("methods", (r"methods?\b", r"measure", r"\bbias", r"sample"),
     "How much do the conclusions on {topic} depend on how the studies were designed and measured?",
     "Compare designs or measures directly on the same question and sample",
     "Methodological comparison study", "1 year", MEDIUM,
     "needs two measurement approaches in one study, which raises its cost"),
)
GAP_KIND_BY_KEY = {k[0]: k for k in GAP_KINDS}
OPEN_KIND = ("open", (), "What remains unknown about {gap} in research on {topic}?",
             "Design a focused study whose research question addresses this gap directly",
             "Study following the design proposed in Phase 2", "1–2 years", MEDIUM, "depends on access to data on this gap")


def _gap_kind(text: str) -> str:
    lower = text.lower()
    for key, cues, *_ in GAP_KINDS:
        if any(re.search(cue, lower) for cue in cues):
            return key
    return "open"


def review_gaps(review: ReviewDocument, profiles: Sequence[ResultProfile]) -> List[ReviewGap]:
    """The gaps under the review's "Research Gaps" heading; without the review, the papers' own stated limitations."""
    gaps: List[ReviewGap] = []
    inside = False
    for line in review.lines:
        if line.startswith("#"):
            inside = line.strip().lower().lstrip("#").strip() == "research gaps"
            continue
        match = re.match(r"^\*\*([^*\[]+?)\.?\*\*\s*(.+)$", line.strip()) if inside else None
        if match:
            gaps.append(ReviewGap(match.group(1).strip(), _gap_kind(f"{match.group(1)} {match.group(2)}"), "review"))
    if gaps:
        return gaps
    for profile in profiles:
        if profile.study.stated_gap:
            gaps.append(ReviewGap(_limitation_type(profile.study.stated_gap), _gap_kind(profile.study.stated_gap),
                                  profile.study.narrative))
    return gaps


# ---- what the papers' limitations are about -----------------------------------------------------

LIMITATION_TYPES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("reliance on self-reported data", (r"self-report",)),
    ("small or narrow samples", (r"small samples?", r"limited samples?", r"sample size")),
    ("untested scalability", (r"scalab", r"large-scale", r"near-term")),
    ("limited generalisability", (r"generali[sz]", r"specific (?:context|setting|case)", r"other (?:contexts|settings|countries)")),
    ("emerging issues left unexplored", (r"emerging", r"does not (?:explore|address|consider|examine)", r"not (?:explored|addressed)")),
    ("few real-world cases", (r"\brare\b", r"remains? (?:rare|unclear)", r"few cases")),
    ("validation beyond the datasets used", (r"datasets?", r"validat", r"benchmark")),
)


def _limitation_type(text: str) -> str:
    lower = text.lower()
    return next((label for label, cues in LIMITATION_TYPES if any(re.search(c, lower) for c in cues)),
                "limitations the authors acknowledge")


# ---- future directions ----------------------------------------------------------------------------

@dataclass
class Direction:
    category: str  # one of DIRECTION_CATEGORIES
    title: str
    question: str
    rationale: str
    next_step: str
    design: str
    timeline: str
    feasibility: str
    feasibility_reason: str
    basis: str  # the phase the direction comes from


DIRECTION_CATEGORIES = (
    "Unanswered questions",
    "Methodological improvements",
    "Replication and scaling",
    "Interdisciplinary approaches",
)
# The methodological improvement each tradition most needs: (title, step, design, timeline, feasibility, why)
FIELD_IMPROVEMENTS: Dict[str, Tuple[str, str, str, str, str, str]] = {
    "health": ("Controlled designs with safety outcomes",
               "Move from associations to a randomised or well-controlled comparison with prespecified benefit and harm outcomes",
               "Randomised controlled trial or target-trial emulation", "2–4 years", LOW,
               "trials need ethics approval, registration and substantial funding; emulating a trial in existing records is cheaper"),
    "social": ("Validated instruments and triangulation",
               "Use validated, piloted scales and triangulate them with a qualitative strand",
               "Survey with validated scales plus interviews", "6–12 months", HIGH,
               "standard practice in the field and within reach of a single study"),
    "engineering": ("Reproducible evaluation",
                    "Report variance over repeated runs, compare against strong baselines on several datasets, run ablations and release the code",
                    "Benchmark study with ablations", "3–6 months", HIGH,
                    "needs compute time rather than new data"),
    "law": ("Systematic case selection",
            "Select cases and jurisdictions by stated criteria and code their outcomes so they can be compared",
            "Empirical legal study of a defined case set", "6–12 months", MEDIUM,
            "depends on access to complete case databases"),
    "humanities": ("Wider and documented corpora",
                   "Widen the corpus beyond the sources studied so far and document how it was assembled",
                   "Corpus-based or archival study", "1–2 years", MEDIUM,
                   "archive access and digitisation take time"),
}
QUALITY_FOCUS = {  # what each tradition judges studies on first
    "health": "risk of bias and clinical relevance",
    "social": "measurement quality and how representative the samples are",
    "engineering": "reproducibility and fair comparison with baselines",
    "law": "the authority of sources and transparent case selection",
    "humanities": "source criticism and transparent interpretation",
}
# Disciplines whose methods complement each tradition's: (partner, what the combination adds)
PARTNERS: Dict[str, Tuple[str, str]] = {
    "health": ("data science and health informatics",
               "causal-inference and machine-learning methods applied to routinely collected health records"),
    "social": ("computing and data science",
               "behavioural or digital-trace data analysed alongside surveys and interviews"),
    "engineering": ("the social sciences and human-computer interaction",
                    "user studies and stakeholder interviews on how the systems are adopted, trusted and governed"),
    "law": ("empirical social science and computing",
            "quantitative coding of case outcomes and practitioner interviews alongside doctrinal reading"),
    "humanities": ("the digital humanities and social sciences",
                   "computational text analysis of large corpora alongside close reading"),
}
FIELD_EXPERTISE = {  # gap_research.FIELD_CUES field -> the expertise it brings
    "computer science and IT": "computer science (system design, data and security engineering)",
    "engineering": "engineering (system design and testing)",
    "medicine and health": "clinical and health research",
    "biology and life sciences": "the life sciences",
    "psychology": "psychology (measurement of attitudes and behaviour)",
    "economics and business": "economics (cost, incentive and market analysis)",
    "law and policy": "law and policy analysis",
    "social sciences and education": "the social sciences",
    "humanities": "the humanities",
    "physics and mathematics": "mathematical modelling",
    "environmental science": "environmental science",
}


def _theme_cites(theme: Theme) -> str:
    return _cites([p.study for p in theme.profiles])


def _names(studies: Sequence[StudyDesign]) -> str:
    """Studies as the subject of a sentence: "Kumar et al. (2024) and Lindsay (2025)"."""
    return join_items([s.narrative for s in studies])


def _num(n: int) -> str:
    return "none" if n == 0 else str(SPELLED.get(n, n))


def future_directions(topic: str, gaps: Sequence[ReviewGap], analysis: ResultsAnalysis, field_name: str) -> List[Direction]:
    profiles, themes, tradition, rec = analysis.profiles, analysis.themes, analysis.tradition, analysis.recommendation
    directions: List[Direction] = []

    # Unanswered questions: the review's gaps, then the findings the studies disagree on.
    seen_kinds: Set[str] = set()
    for gap in gaps:
        if (gap.kind in seen_kinds and gap.kind != "open") or (gap.kind == "open" and gap.source != "review"):
            continue  # the papers' unclassified limitations are taken together below
        seen_kinds.add(gap.kind)
        key, _, question, step, design, timeline, feasibility, why = GAP_KIND_BY_KEY.get(gap.kind, OPEN_KIND)
        if key == "open":
            design = f"{rec.design.label}, as proposed in Phase 2"
        from_review = gap.source == "review"
        directions.append(Direction(
            "Unanswered questions", _cap(gap.title.rstrip(".")), question.format(topic=topic, gap=_lc(gap.title)),
            f"The literature review names \"{gap.title.rstrip('.')}\" as a research gap." if from_review
            else f"Phase 1 records {gap.title} as a limitation of {gap.source}.",
            step, design, timeline, feasibility, why, "Phase 1 research gap" if from_review else "Phase 1 paper limitations",
        ))
    open_issues = [g for g in gaps if g.kind == "open" and g.source != "review"]
    if open_issues:
        _, _, _, step, _, timeline, feasibility, why = OPEN_KIND
        directions.append(Direction(
            "Unanswered questions", "Issues the studies leave open",
            f"Which of the issues the reviewed studies leave open matters most for {topic}, and can a new study settle it?",
            "Phase 1 records these limitations: " + join_items([f"{g.title} ({g.source})" for g in open_issues]) + ".",
            step, f"{rec.design.label}, as proposed in Phase 2", timeline, feasibility, why, "Phase 1 paper limitations",
        ))
    for theme in themes:
        if theme.consistency == "consistent":
            continue
        directions.append(Direction(
            "Unanswered questions", f"Disagreement on {theme.term}",
            f"Under which conditions does {theme.term} hold, and why do the studies describe it differently?",
            f"The studies reporting on it ({_theme_cites(theme)}) were judged {theme.consistency} in Phase 3.",
            "Measure the outcome in the same way across the settings the studies cover, and test which differences "
            "(sample, setting, measure) explain the disagreement",
            "Comparative study with shared measures", "1–2 years", MEDIUM,
            "the settings already studied give a starting point for case selection", "Phase 3 results",
        ))

    # Methodological improvements: what the evidence lacks, then what the tradition expects.
    empirical = [p for p in profiles if p.study.kind.empirical]
    no_effects = [p for p in empirical if not p.effects()]
    no_sample = [p for p in empirical if not p.study.samples and "datasets" not in p.study.collection]
    if no_effects or no_sample:
        lacking = [x for x, group in (("effect sizes", no_effects), ("sample sizes", no_sample)) if group]
        either = [p.study for p in empirical if p in no_effects or p in no_sample]
        directions.append(Direction(
            "Methodological improvements", "Transparent reporting of effects and samples",
            f"How large are the effects behind the findings on {topic}, and how precisely are they estimated?",
            f"The abstracts of {_names(either)} give no {join_items(lacking)}.",
            "Report the size of each improvement over the baselines with its variation across runs, and state the "
            "datasets' sizes" if tradition.key == "engineering" else
            "Pre-register the analysis, justify the sample with a power analysis and report effect sizes with "
            "confidence intervals", "Any empirical design, pre-registered", "Built into the next study", HIGH,
            "costs little beyond planning", "Phase 3 evidence gaps",
        ))
    untested = [p.study for p in profiles if not p.study.kind.empirical]
    if untested:
        directions.append(Direction(
            "Methodological improvements", "Empirical tests of proposed frameworks",
            "Do the frameworks and arguments that rest on no new data stand up to testing?",
            f"{_names(untested)} collected no new data.",
            "Derive testable predictions from each framework and check them against independently collected data",
            "Hypothesis-testing empirical study", "6–18 months", MEDIUM,
            "requires operationalising the frameworks first", "Phase 2 designs",
        ))
    title, step, design, timeline, feasibility, why = FIELD_IMPROVEMENTS[tradition.key]
    directions.append(Direction(
        "Methodological improvements", title,
        f"Would the conclusions on {topic} change under the evaluation standards of {_lc(tradition.name)}?",
        f"Phase 2 places the reviewed studies in {_lc(tradition.name)}, where work is judged above all on "
        f"{QUALITY_FOCUS[tradition.key]}.",
        step, design, timeline, feasibility, why, "Phase 2 field",
    ))

    # Replication and scaling.
    covered = {id(p) for t in themes for p in t.profiles}
    single = [p.study for p in profiles if id(p) not in covered and p.lead and p.study.kind.empirical]
    if single:
        directions.append(Direction(
            "Replication and scaling", "Replication of single-study findings",
            "Which single-study findings survive an independent replication?",
            f"The findings of {_names(single)} have not been reported by any other reviewed study.",
            "Repeat each study's design and measures on a new sample, reporting the result whatever its direction",
            "Direct or conceptual replication", "6–12 months", HIGH,
            "the original protocols define what to do", "Phase 3 results",
        ))
    quant, qual = _strands(rec)
    consistent = [t for t in themes if t.consistency == "consistent"]
    target = consistent[0].term if consistent else topic
    if quant in POWER_DESIGNS:
        scale = (f"Test {target} in a larger, more varied sample; detecting a small effect (d = 0.2) at 80% power "
                 f"takes about {n_two_groups(0.2)} participants per group")
        scale_design, scale_time, scale_feasibility, scale_why = (
            "Multi-site study", "1–2 years", MEDIUM, "large samples need several recruitment sites")
    elif quant in ("computational-experiment", "simulation", "design-development"):
        scale = (f"Evaluate {target} at realistic scale: larger datasets, real deployment conditions and the "
                 "hardware it would run on, reporting cost and latency as well as accuracy")
        scale_design, scale_time, scale_feasibility, scale_why = (
            "Scaled evaluation or field deployment", "6–18 months", MEDIUM, "depends on compute and on partners for real deployments")
    else:
        scale = f"Extend the study of {target} to further cases, sites or sources chosen to differ from those already studied"
        scale_design, scale_time, scale_feasibility, scale_why = (
            "Multiple-case or comparative study", "1–2 years", MEDIUM, "each added case needs its own access and analysis")
    countries = Counter(c for p in profiles for c in p.study.countries)
    setting = (f"The evidence so far comes from {join_items(list(countries))}." if countries
               else "The studies do not say where they were carried out.")
    directions.append(Direction(
        "Replication and scaling", f"Scaling up the evidence on {target}",
        f"Does what is known about {target} hold at larger scale and in other settings?",
        setting, scale, scale_design, scale_time, scale_feasibility, scale_why, "Phases 2 and 3",
    ))

    # Interdisciplinary approaches.
    vocabulary = FIELD_TRADITION.get(field_name)
    if vocabulary and vocabulary != tradition.key and field_name in FIELD_EXPERTISE:
        directions.append(Direction(
            "Interdisciplinary approaches", f"Bridging {_lc(tradition.name)} and {field_name}",
            f"What does {topic} look like when the methods of {_lc(tradition.name)} meet the technical knowledge of {field_name}?",
            f"The papers' designs belong to {_lc(tradition.name)} but their subject matter to {field_name}.",
            f"Pair the proposed design with expertise in {FIELD_EXPERTISE[field_name]}, e.g. by co-designing "
            "instruments with technical specialists or validating self-reports against technical data",
            "Interdisciplinary mixed-methods project", "1–2 years", MEDIUM,
            "needs a co-supervisor or collaborator from the other field", "Phase 2 field",
        ))
    partner, adds = PARTNERS[tradition.key]
    directions.append(Direction(
        "Interdisciplinary approaches", f"Collaboration with {partner}",
        f"What would {partner} add to research on {topic}?",
        f"The reviewed studies work within {_lc(tradition.name)}; {partner} would add a perspective they do not cover.",
        f"Combine the field's own methods with {adds}", "Interdisciplinary project", "1–3 years", MEDIUM,
        "depends on finding collaborators and a shared research question", "Phase 2 field",
    ))
    return directions


# ---- judging the evidence ---------------------------------------------------------------------

def evidence_strength(profiles: Sequence[ResultProfile], themes: Sequence[Theme]) -> Tuple[str, str]:
    """(strong/moderate/limited/weak, why) for the evidence base as a whole."""
    ratings = Counter(p.confidence for p in profiles)
    replicated = [t for t in themes if t.consistency == "consistent"]
    if replicated and any(t.confidence == HIGH for t in replicated) and ratings[HIGH] * 2 > len(profiles):
        return "strong", "findings are replicated across studies, most of which report samples and statistics"
    if any(t.confidence in (HIGH, MEDIUM) for t in replicated) or ratings[HIGH] * 2 > len(profiles):
        return "moderate", ("at least one finding is reported consistently by more than one study" if replicated
                            else "most studies report samples and statistics, but no finding is replicated")
    if ratings[HIGH] or ratings[MEDIUM]:
        return "limited", "some studies report samples or statistics, but no finding is firmly replicated"
    return "weak", "no study reports both a sample and statistics"


def _certainty(studies: Sequence[StudyDesign], strength: str) -> str:
    """Roughly how a guideline panel (GRADE) would start rating this body of evidence."""
    designs = {s.design for s in studies}
    if designs & {"rct", "meta-analysis"}:
        start = "moderate" if strength in ("limited", "weak") else "high"
    elif designs & {"observational", "quasi-experimental"}:
        start = "low"
    else:
        start = "very low"
    return start


def _verb(term: str, confidence: str) -> str:
    return {HIGH: f"can act on the finding on {term}", MEDIUM: f"can draw on the finding on {term}, monitoring the results",
            LOW: f"should treat the finding on {term} as provisional"}[confidence]


# ---- field-specific implications ---------------------------------------------------------------

FIELD_HEADINGS: Dict[str, Tuple[str, Tuple[str, ...]]] = {
    "health": ("Clinical Implications", ("Clinical significance", "Safety implications", "Treatment guidelines")),
    "social": ("Social and Policy Implications", ("Policy recommendations", "Social impact", "Behavioural change")),
    "engineering": ("Engineering Implications", ("Design standards", "Performance improvements", "Scalability")),
    "law": ("Legal Implications", ("Precedent", "Policy reform", "Justice system")),
    "humanities": ("Humanities Implications", ("Theoretical frameworks", "Interpretation", "Cultural understanding")),
}
AUDIENCE = {
    "health": "clinicians and health services",
    "social": "practitioners and organisations",
    "engineering": "engineers and system developers",
    "law": "legal practitioners and regulators",
    "humanities": "educators, curators and cultural institutions",
}
POLICY_BODY = {
    "health": "health authorities and guideline bodies",
    "social": "governments and public agencies",
    "engineering": "standards bodies and regulators",
    "law": "legislators and law-reform bodies",
    "humanities": "cultural and education policy makers",
}


def _with(profiles: Sequence[ResultProfile], kinds: Sequence[str]) -> List[ResultProfile]:
    return [p for p in profiles if any(s.kind in kinds for s in p.statistics)]


def _stated(profiles: Sequence[ResultProfile], kind: str) -> List[ResultProfile]:
    return [p for p in profiles if any(s["kind"] == kind for s in p.field_statements)]


def _mentions(profiles: Sequence[ResultProfile], pattern: str) -> List[ResultProfile]:
    return [p for p in profiles if re.search(pattern, f"{p.study.title} {' '.join(f.text for f in p.findings)}", re.IGNORECASE)]


def field_implications(analysis: ResultsAnalysis, strength: str, topic: str) -> Dict[str, str]:
    profiles, themes, key = analysis.profiles, analysis.themes, analysis.tradition.key
    studies = [p.study for p in profiles]
    consistent = [t for t in themes if t.consistency == "consistent"]
    lead = consistent[0] if consistent else None
    if key == "health":
        effects = _with(profiles, ("odds-ratio", "risk-ratio", "confidence-interval", "effect-size"))
        safety = _stated(profiles, "safety")
        return {
            "Clinical significance": (
                f"Effect estimates are available from {_names([p.study for p in effects])}; whether they matter clinically "
                "depends on whether they exceed the minimal clinically important difference for the outcome, which "
                "statistical significance alone cannot show." if effects else
                "No study gives an effect estimate with its precision, so clinical significance cannot yet be judged: "
                "a statistically detectable effect may still be too small to change care."),
            "Safety implications": (
                f"Harms are addressed by {_names([p.study for p in safety])}, but benefits and harms are not weighed "
                "against each other across studies; any recommendation should state both." if safety else
                "The studies are silent on harms, so the safety of acting on these findings is unknown; future "
                "studies should prespecify adverse outcomes."),
            "Treatment guidelines": (
                f"A guideline panel applying GRADE would probably rate this evidence at {_certainty(studies, strength)} "
                f"certainty, given the designs used. With {strength} evidence overall, the findings can inform "
                "guideline questions but not yet a strong recommendation."),
        }
    if key == "social":
        countries = sorted({c for p in profiles for c in p.study.countries})
        populations = sorted({x for p in profiles for x in p.study.populations})
        drivers = _stated(profiles, "relationship")
        return {
            "Policy recommendations": (
                f"The clearest basis for policy is the shared finding on {lead.term} ({_theme_cites(lead)}); policy "
                f"should build on it with {'confidence' if lead.confidence == HIGH else 'pilot programmes that are evaluated'}."
                if lead else "No finding is yet shared by several studies, so policy changes should be piloted and evaluated "
                "rather than rolled out on this evidence."),
            "Social impact": (
                ("The groups studied (" + join_items(populations[:4]) + ") are the ones these findings can speak for"
                 + (f", in {join_items(countries[:4])}" if countries else "") + "; groups not studied may be affected differently.")
                if populations or countries else
                "The studies say little about whom they studied, so the social reach of the findings is uncertain."),
            "Behavioural change": (
                f"The factors related to the outcome in {_names([p.study for p in drivers])} are candidate targets for "
                "interventions, but associations do not show that changing them changes behaviour; that needs an "
                "intervention study." if drivers else
                "The studies identify no tested levers for changing behaviour, so interventions would be designed on "
                "theory rather than evidence for now."),
        }
    if key == "engineering":
        performance = _with(profiles, ("performance",))
        scale = _mentions(profiles, r"scalab|large-scale|hardware|latency|throughput|real-time")
        return {
            "Design standards": (
                f"The finding on {lead.term} ({_theme_cites(lead)}) is a candidate for design guidance once it is "
                "reproduced under a common evaluation protocol." if lead else
                "No result is yet reproduced across studies, so none is ready to become a design standard; a shared "
                "benchmark and evaluation protocol would be the first step."),
            "Performance improvements": (
                f"Reported gains ({_cites([p.study for p in performance])}) come from each paper's own test setting; they "
                "should be compared only under the same data, baselines and hardware." if performance else
                "The studies do not quantify their gains in the abstracts, so the size of any improvement over current "
                "practice is still to be established."),
            "Scalability": (
                f"Scale is raised by {_names([p.study for p in scale])}, but behaviour at production scale (cost, "
                "latency, robustness) is not demonstrated." if scale else
                "None of the studies tests its approach at production scale, which leaves cost, latency and robustness "
                "under real loads open."),
        }
    if key == "law":
        outcomes = _stated(profiles, "outcome") + _stated(profiles, "doctrinal")
        return {
            "Precedent": (
                f"The doctrinal positions in {_names([p.study for p in outcomes])} indicate how courts may approach "
                f"{topic}, but scholarly analysis is persuasive rather than binding authority." if outcomes else
                f"The studies identify no settled precedent on {topic}, which leaves room for courts to shape the law."),
            "Policy reform": (
                f"Reform proposals can draw on the shared finding on {lead.term} ({_theme_cites(lead)})." if lead else
                "The studies do not yet agree on a reform agenda, so reform proposals should set out the competing positions."),
            "Justice system": (
                "Changes to procedure or enforcement should await evidence of how the rules work in practice, which "
                "empirical legal research (case coding, practitioner interviews) can supply."),
        }
    interpretive = list(dict.fromkeys([p for p in profiles if p.result_type == QUAL_RESULTS] + _stated(profiles, "interpretive")))
    return {
        "Theoretical frameworks": (
            f"The shared reading of {lead.term} ({_theme_cites(lead)}) suggests a framework future work can test against "
            "new sources." if lead else "The studies read their material through different frameworks, which have yet to be compared."),
        "Interpretation": (
            f"Interpretations rest on the sources examined by {_names([p.study for p in interpretive] or studies)}; "
            "other sources could support different readings."),
        "Cultural understanding": (
            f"The work deepens understanding of {topic} in the contexts studied; comparison with other cultural "
            "settings would show what is particular to them."),
    }


# ---- the conclusion section -------------------------------------------------------------------

@dataclass
class ConclusionInputs:
    topic: str
    review_type: str
    review_key: str  # review_methodology key, "" if unknown
    slug: str
    analysis: ResultsAnalysis
    gaps: List[ReviewGap]
    directions: List[Direction]
    field_name: str
    bibliography: str


DESIGN_LIMITS = {  # what each design cannot show, in brief
    "survey": "measured once cannot separate cause from effect, and self-reports can be biased",
    "rct": "with strict eligibility may not reflect the people seen in routine settings",
    "observational": "remains open to confounding however carefully it adjusts",
    "quasi-experimental": "without randomisation cannot rule out differences between the groups",
    "experiment": "in controlled conditions may not carry over to real use",
    "computational-experiment": "on benchmarks may not predict behaviour in deployment",
    "simulation": "is only as realistic as its assumptions",
    "design-development": "shows that a system works where it was tested, not that it is adopted or works elsewhere",
    "case-study": "of few cases supports analytic rather than statistical generalisation",
    "interview": "captures the views of those interviewed, not how widespread they are",
    "ethnography": "is bound to the setting observed and to the observer's position",
    "doctrinal": "shows what the law says, not how it operates in practice",
    "historical": "depends on which sources survived and were accessible",
}
APPROACH_LIMITS = {
    QUANT: "can measure effects and associations but not explain them",
    QUAL: "explains in depth but cannot show how common its findings are",
    MIXED: "demands skills in two traditions and careful integration of its strands, which takes time",
    SYNTHESIS: "inherits the weaknesses of the studies it combines",
    CONCEPTUAL: "produces propositions that still need empirical testing",
    DESIGN_SCIENCE: "shows that a system works where it was tested, not that it is adopted or works elsewhere",
}


def design_limit(design) -> str:
    return DESIGN_LIMITS.get(design.key) or APPROACH_LIMITS.get(design.approach, "has limits that should be stated")


REVIEW_METHOD_LIMITS = {
    "systematic": "screening was automated rather than carried out by two independent reviewers",
    "scoping": "the included studies were mapped rather than appraised",
    "narrative": "papers were chosen by relevance and citations rather than fixed criteria",
    "rapid": "only abstracts of recent work were assessed",
    "meta-analysis": "effect sizes have not yet been pooled",
}


def _year_span(studies: Sequence[StudyDesign]) -> str:
    years = sorted(int(s.year) for s in studies if str(s.year).isdigit())
    if not years:
        return ""
    return f", published in {years[0]}" if years[0] == years[-1] else f", published between {years[0]} and {years[-1]}"


def _design_mix(studies: Sequence[StudyDesign]) -> str:
    groups: Dict[str, List[StudyDesign]] = {}
    for s in studies:
        groups.setdefault(s.design, []).append(s)
    return join_items([
        f"{_a(DESIGN_BY_KEY[k].label)} ({_cites(g)})" if len(g) == 1
        else f"{_count(len(g), 'study', 'studies')} using {_a(DESIGN_BY_KEY[k].label)} ({_cites(g)})"
        for k, g in sorted(groups.items(), key=lambda item: -len(item[1]))
    ])


BASIS_REASON = {
    "mixed": "because the reviewed studies used quantitative and qualitative designs separately, and combining them lets "
             "each strand explain or test the other",
    "common": "because it is the design the reviewed studies share, so the new results can be set directly against theirs",
    "default": "because none of the reviewed studies collected new data, and this is the standard empirical design of the field",
}


def build_conclusion(inputs: ConclusionInputs) -> str:
    a = inputs.analysis
    topic, profiles, themes, tradition, rec = inputs.topic, a.profiles, a.themes, a.tradition, a.recommendation
    studies = [p.study for p in profiles]
    n = len(profiles)
    consistent = [t for t in themes if t.consistency == "consistent"]
    divided = [t for t in themes if t.consistency != "consistent"]
    strength, why = evidence_strength(profiles, themes)
    design = rec.design.label
    lines = [
        f"# Conclusion: {topic}",
        "",
        f"*Conclusion, implications and future directions, drawing together the {inputs.review_type}, the proposed "
        "methodology and the results*",
        "",
        f"> **How to complete this conclusion.** It is built from the review, the methodology and the results. Where "
        f"only your finished study can speak, the text is marked: **{markers.YOUR_CONTRIBUTION}** for what your results "
        f"add, **{markers.YOUR_LIMITATIONS}** for the limitations of your study as carried out, and "
        f"**{markers.YOUR_PRIORITIES}** for the direction you would pursue next.",
        "",
        "## 1. Conclusion",
        "",
        "### 1.1 What the field knows",
        "",
    ]

    # 1.1 Phase 1: the literature
    known = [f"This {inputs.review_type} brought together {_count(n, 'study', 'studies')} on {topic}{_year_span(studies)}, "
             f"drawing on {_design_mix(studies)}."]
    if consistent:
        known.append("The field agrees most clearly on " + join_items(
            f"{t.term} ({_theme_cites(t)}; {t.confidence.lower()} confidence)" for t in consistent) + ".")
    if divided:
        known.append("It is divided on " + join_items(f"{t.term} ({_theme_cites(t)})" for t in divided) + ".")
    if not themes:
        known.append("The studies address different questions, so no finding is yet established by more than one of them; "
                     "the field knows a series of separate results rather than a shared body of evidence.")
    known.append(f"Taken as a whole, the evidence is {strength}: {why}.")
    review_listed = [g for g in inputs.gaps if g.source == "review"]
    if review_listed:
        known.append("What the field does not yet know centres on " + join_items(_lc(g.title) for g in review_listed) + ".")
    lines += [" ".join(known), ""]

    # 1.2 Phase 2: the methodology
    alignments = Counter(compare(s, rec)[0] for s in studies)
    addressed = {g.kind for g in inputs.gaps}
    builds = [f"The proposed study uses {_a(design)} {BASIS_REASON[rec.basis]}."]
    if rec.components:
        builds.append(f"Its strands, {_a(rec.components[0].label)} and {_a(rec.components[1].label)}, mirror the "
                      "reviewed studies' own designs.")
    same = alignments.get("Same design", 0) + sum(v for k, v in alignments.items() if k.startswith("Matches"))
    if same:
        builds.append(f"{_cap(_count(same, 'reviewed study uses', 'reviewed studies use'))} the same design"
                      f"{' or one of its strands' if rec.components else ''}, "
                      "so its results will be directly comparable.")
    targets = []
    if "generalisability" in addressed:
        targets.append("testing the findings in a new setting")
    if "evidence" in addressed or any(not s.kind.empirical for s in studies):
        targets.append("adding new empirical data where the literature rests on argument")
    if "integration" in addressed and rec.components:
        targets.append("linking strands of research that have been studied apart")
    if "longitudinal" in addressed and _strands(rec)[0] in POWER_DESIGNS:
        targets.append("laying a baseline that later waves can follow up")
    if targets:
        builds.append("It answers the review's gaps by " + join_items(targets) + ".")
    lines += ["### 1.2 How this study builds on it", "", " ".join(builds), ""]

    # 1.3 Phase 3: the results
    reported = [p for p in profiles if p.effects()]
    found = []
    if themes:
        found.append(f"Across the studies, the analysis identified {_count(len(themes), 'finding', 'findings')} reported by more "
                     f"than one study: {_num(len(consistent))} pointing the same way and {_num(len(divided))} that do not.")
        best = max(themes, key=lambda t: ({HIGH: 2, MEDIUM: 1, LOW: 0}[t.confidence], len(t.profiles)))
        found.append(f"The best-supported concerns {best.term}, backed by {_count(len(best.profiles), 'study', 'studies')} "
                     f"with {best.confidence.lower()} confidence overall.")
    else:
        found.append("The analysis found no result shared by two or more studies; each study's finding stands on its own.")
    ratings = Counter(p.confidence for p in profiles)
    found.append("Rated one by one, the studies give " + join_items(
        f"{_num(v)} {k.lower()}-confidence result{'s' if v != 1 else ''}" for k, v in sorted(ratings.items(), key=lambda kv: [HIGH, MEDIUM, LOW].index(kv[0])))
        + (f", and only {_names([p.study for p in reported])} {'quantifies its' if len(reported) == 1 else 'quantify their'} effects." if reported and len(reported) < n
           else ", and all of them quantify their effects." if reported else ", and none quantifies its effects in the abstract."))
    lines += ["### 1.3 What the analysis found", "", " ".join(found), "", markers.marker(
        markers.YOUR_CONTRIBUTION,
        "Summarise your own main results in two or three sentences: what you found on "
        + (join_items(t.term for t in themes[:3]) if themes else "your research question")
        + ", and whether it agrees with the reviewed studies.",
    ), ""]

    # 1.4 Contribution
    offers = []
    if any(not s.kind.empirical for s in studies) or strength in ("limited", "weak"):
        offers.append(f"new {rec.design.approach.lower()} evidence on a topic where the existing evidence is {strength}")
    if divided:
        offers.append("a test of the conditions under which " + join_items(t.term for t in divided) + " holds")
    single = [p for p in profiles if p.lead and not any(p in t.profiles for t in themes)]
    if single:
        offers.append("an independent check of findings so far reported by one study only")
    if review_listed:
        offers.append("progress on " + _lc(review_listed[0].title))
    contribution = [f"This study can advance research on {topic} by providing " + join_items(offers or ["a further, comparable data point"]) + "."]
    lines += ["### 1.4 Contribution", "", " ".join(contribution), "", markers.marker(
        markers.YOUR_CONTRIBUTION,
        "State the contribution your results actually make: which finding they confirm, which they challenge or "
        "refine, and which gap they close, citing the studies concerned.",
    ), ""]

    # 1.5 Limitations
    limits = []
    by_type: Dict[str, List[StudyDesign]] = {}
    for p in profiles:
        for limitation in p.limitations:
            by_type.setdefault(_limitation_type(limitation.text), [])
            if p.study not in by_type[_limitation_type(limitation.text)]:
                by_type[_limitation_type(limitation.text)].append(p.study)
    if by_type:
        limits.append("The reviewed studies point to " + join_items(f"{label} ({_cites(group)})" for label, group in by_type.items()) + ".")
    low = [p.study for p in profiles if p.confidence == LOW]
    if low:
        limits.append(f"The results of {_names(low)} rest on thin reported evidence, which weakens any conclusion built on them.")
    limits.append(f"The proposed design has limits of its own: {_a(design)} {design_limit(rec.design)}.")
    if rec.components:
        limits.append("Within it, " + join_items([f"{_a(s.label)} {design_limit(s)}" for s in rec.components]) + ".")
    review_limit = REVIEW_METHOD_LIMITS.get(inputs.review_key)
    limits.append("Finally, the review itself was assembled automatically from abstracts"
                  + (f", and {review_limit}" if review_limit else "") + ", so studies or details outside them may be missing.")
    lines += ["### 1.5 Limitations", "", " ".join(limits), "", markers.marker(
        markers.YOUR_LIMITATIONS,
        "Describe the limitations of your study as it was carried out: sample and setting, measurement, response or "
        "attrition, analysis decisions, and how each may have affected your conclusions.",
    ), ""]

    # 1.6 Closing
    audience, body = AUDIENCE[tradition.key], POLICY_BODY[tradition.key]
    first_open = next((d for d in inputs.directions if d.category == "Unanswered questions"), inputs.directions[0])
    lead = consistent[0] if consistent else None
    closing = [
        (f"For practice, {audience} {_verb(lead.term, lead.confidence)}." if lead
         else f"For practice, {audience} have no replicated finding to act on yet and should treat single-study results with caution."),
        f"For policy, {body} should {'build on the consistent findings while funding their replication' if consistent else 'commission further evidence before acting'}.",
        f"For research, the most pressing open question is this: {first_open.question}",
        "Sections 2 and 3 develop these implications and directions in turn.",
    ]
    lines += ["### 1.6 Concluding remarks", "", " ".join(closing), ""]

    # 2. Implications
    lines += ["## 2. Implications", "", "### 2.1 Practical implications", ""]
    if consistent:
        lines += [f"- **{_cap(t.term)}.** {_cap(audience)} {_verb(t.term, t.confidence)} ({_theme_cites(t)}; "
                  f"{t.confidence.lower()} confidence)." for t in consistent]
    strong_single = [p for p in single if p.confidence == HIGH]
    if strong_single:
        lines += [f"- **Single strong studies.** The results of {_names([p.study for p in strong_single])} are well "
                  "reported and may guide local practice, provided their effect is monitored after adoption."]
    if divided:
        lines += [f"- **Contested findings.** Practice on {join_items(t.term for t in divided)} should not change until the "
                  "disagreement between studies is resolved."]
    if not consistent and not strong_single and not divided:
        lines += [f"- {_cap(audience)} have no well-supported result to act on yet; the studies are better used to frame "
                  "questions for local evaluation than to change practice."]
    lines += ["", "### 2.2 Policy implications", ""]
    countries = sorted({c for p in profiles for c in p.study.countries})
    policy = [f"{_cap(body)} need evidence that is consistent across studies and settings."]
    policy.append(f"On that standard the current evidence is {strength}"
                  + (f", and it comes from {join_items(countries[:5])}" if countries else "") + ".")
    policy.append("Policies built on it should be framed as pilots with built-in evaluation" if strength != "strong"
                  else "It can support policy, with monitoring of outcomes in new settings")
    policy[-1] += ", and should fund the replication described in Section 3."
    lines += [" ".join(policy), "", "### 2.3 Theoretical implications", ""]
    theory = []
    if consistent:
        theory.append(f"The agreement on {join_items(t.term for t in consistent)} lends support to the explanations the "
                      "studies offer for it, which theories of the field should incorporate.")
    if divided:
        theory.append(f"The disagreement on {join_items(t.term for t in divided)} suggests the theories involved lack "
                      "boundary conditions: they should say when the effect occurs, not only that it does.")
    conceptual = [s for s in studies if not s.kind.empirical]
    if conceptual:
        theory.append(f"The frameworks proposed by {_names(conceptual)} are theoretical contributions that remain to be "
                      "tested empirically.")
    if not theory:
        theory.append("With no shared or contested finding, the studies do not yet challenge or confirm a common theory; "
                      "their value lies in the separate propositions each puts forward for testing.")
    lines += [" ".join(theory), ""]
    heading, subheadings = FIELD_HEADINGS[tradition.key]
    implications = field_implications(a, strength, topic)
    lines += [f"### 2.4 {heading}", ""]
    for sub in subheadings:
        lines += [f"**{sub}.** {implications[sub]}", ""]

    # 3. Future directions
    lines += ["## 3. Future Directions", "",
              "The directions below follow from the review's gaps, the proposed methodology and the results. Each "
              "has an indicative timeline and its feasibility for a single researcher (high), a collaboration "
              "(medium) or a funded programme (low).", ""]
    number = 0
    for category in DIRECTION_CATEGORIES:
        group = [d for d in inputs.directions if d.category == category]
        if not group:
            continue
        number += 1
        lines += [f"### 3.{number} {category}", ""]
        for d in group:
            lines += [
                f"**{d.title}.** {d.question} {d.rationale}",
                "",
                f"- *Next step:* {d.next_step}.",
                f"- *Design:* {d.design}. *Timeline:* {d.timeline}. *Feasibility:* {d.feasibility.lower()} ({d.feasibility_reason}).",
                "",
            ]
    number += 1
    lines += [
        f"### 3.{number} Summary of directions",
        "",
        "| Direction | Type | Suggested design | Timeline | Feasibility |",
        "|---|---|---|---|---|",
    ]
    lines += [f"| {_cell(d.title)} | {d.category} | {_cell(d.design)} | {d.timeline} | {d.feasibility} |" for d in inputs.directions]
    lines += [
        "",
        markers.marker(
            markers.YOUR_PRIORITIES,
            "Say which of these directions you would pursue first and why, given your results and resources, and what "
            "question your next study would ask.",
        ),
        "",
        "*Phase 4 builds this conclusion from the papers' abstracts, the review and the Phase 2 and 3 analyses; check "
        f"each claim against the papers. The directions and checks are listed in {inputs.slug}-conclusion-analysis.json.*",
        "",
        "## References",
        "",
        inputs.bibliography,
        "",
    ]
    return "\n".join(lines)


# ---- quality checks ---------------------------------------------------------------------------

def _words(text: str) -> List[str]:
    # Citations are left out: the same "(Siam et al., 2024)" in two texts is attribution, not copying.
    text = re.sub(r"\([^()]*\b(?:\d{4}|n\.d\.)\)", " ", text)
    return re.findall(r"[a-z0-9']+", text.lower())


def _shingles(words: Sequence[str]) -> List[Tuple[str, ...]]:
    return [tuple(words[i:i + SHINGLE_WORDS]) for i in range(len(words) - SHINGLE_WORDS + 1)]


def _prose(markdown: str) -> str:
    """The conclusion's own writing: no headings, markers, guide, table scaffolding or references."""
    kept = []
    for line in markdown.splitlines():
        if line.startswith("## References"):
            break
        if (line.startswith(("#", ">", "|---", "| Direction")) or markers.CONCLUSION_MARKER_LINE.match(line)
                or (line.startswith("*") and line.endswith("*") and not line.startswith("**"))):
            continue
        kept.append(line)
    return "\n".join(kept)


def copied_passages(text: str, sources: Dict[str, str]) -> Tuple[float, int, str]:
    """(share of 6-word phrases found in any source, longest copied run in words, the source it is from)."""
    words = _words(text)
    shingles = _shingles(words)
    if not shingles:
        return 0.0, 0, ""
    owners: Dict[Tuple[str, ...], str] = {}
    for name, source in sources.items():
        for shingle in _shingles(_words(source)):
            owners.setdefault(shingle, name)
    copied = [s in owners for s in shingles]
    longest, run, origin, best_origin = 0, 0, "", ""
    for i, hit in enumerate(copied):
        run = run + 1 if hit else 0
        origin = owners[shingles[i]] if hit and run == 1 else origin
        if run and run + SHINGLE_WORDS - 1 > longest:
            longest, best_origin = run + SHINGLE_WORDS - 1, origin
    return sum(copied) / len(shingles), longest, best_origin


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


def quality_checks(markdown: str, inputs: ConclusionInputs, sources: Dict[str, str], corpus: Corpus) -> List[Check]:
    a = inputs.analysis
    checks: List[Check] = []

    prose = _prose(markdown)
    share, run, origin = copied_passages(prose, sources)
    checks.append(Check(
        "Not verbatim", share <= MAX_COPIED_SHARE and run < MAX_COPIED_RUN,
        f"{share:.0%} of 6-word phrases occur in Phases 1-3 or the papers (limit {MAX_COPIED_SHARE:.0%}); "
        f"longest shared run {run} words{f' (from {origin})' if run else ''} (limit {MAX_COPIED_RUN - 1})",
    ))

    body = markdown.split("## References")[0]
    cited = set(citations_in(body))
    known = {(p.surname, p.year) for p in corpus.papers}
    stray = sorted(f"{s} ({y})" for s, y in cited - known)
    missing = sorted(p.citation for p in corpus.papers if (p.surname, p.year) not in cited)
    checks.append(Check("Citations", not stray and not missing,
                        "every citation is to a reviewed paper and every paper is cited" if not stray and not missing
                        else "; ".join(filter(None, [stray and f"not in the review: {', '.join(stray)}",
                                                     missing and f"never cited: {', '.join(missing)}"]))))

    lower = body.lower()
    phase1 = [g.title for g in inputs.gaps if g.source == "review"]
    absent = [t for t in phase1 if t.lower().rstrip(".") not in lower]
    absent += [] if a.recommendation.design.label.lower() in lower else [f"the design ({a.recommendation.design.label})"]
    absent += [t.term for t in a.themes if t.term.lower() not in lower]
    checks.append(Check("Draws on Phases 1-3", not absent,
                        "names the review's gaps, the proposed design and every shared finding" if not absent
                        else "not mentioned: " + ", ".join(absent)))

    headings = ["## 1. Conclusion", "### 1.1 What the field knows", "### 1.2 How this study builds on it",
                "### 1.3 What the analysis found", "### 1.4 Contribution", "### 1.5 Limitations",
                "### 1.6 Concluding remarks", "## 2. Implications", "### 2.1 Practical implications",
                "### 2.2 Policy implications", "### 2.3 Theoretical implications", "## 3. Future Directions"]
    field_heading, subs = FIELD_HEADINGS[a.tradition.key]
    expected = headings + [f"### 2.4 {field_heading}"] + [f"**{s}.**" for s in subs]
    lacking = [h.strip("#*. ") for h in expected if h not in markdown]
    checks.append(Check("Sections", not lacking, "all sections present" if not lacking else "missing: " + ", ".join(lacking)))

    incomplete = [d.title for d in inputs.directions if not (d.timeline and d.feasibility and d.next_step)]
    categories = {d.category for d in inputs.directions}
    checks.append(Check(
        "Future directions", not incomplete and len(categories) == len(DIRECTION_CATEGORIES),
        f"{len(inputs.directions)} directions in {len(categories)} of {len(DIRECTION_CATEGORIES)} categories, each with a "
        "next step, timeline and feasibility" + (f"; incomplete: {', '.join(incomplete)}" if incomplete else ""),
    ))

    placed = {m.group(1) for m in map(markers.CONCLUSION_MARKER_LINE.match, markdown.splitlines()) if m}
    checks.append(Check("Markers", placed == set(markers.CONCLUSION_MARKERS),
                        f"{len(placed)} of {len(markers.CONCLUSION_MARKERS)} kinds of marker placed"))

    section = _prose(markdown.split("## 2. Implications")[0])
    words = len(re.findall(r"\w+", section))
    low, high = CONCLUSION_WORDS
    checks.append(Check("Length", low <= words <= high, f"the Conclusion proper has {words} words (expected {low}-{high})"))
    return checks


# ---- running Phase 4 ----------------------------------------------------------------------------

def resolve_input(path: Path) -> Path:
    """Maps a Phase 2/3/4 output (<review>-conclusion.md, ...) to the Phase 1 review or paper file it was made from."""
    if path.stem.endswith("-conclusion") and not path.name.endswith(SIDECAR_SUFFIX):
        base = path.with_name(path.stem[: -len("-conclusion")] + ".md")
        if base.is_file():
            return base
        if sidecar_path(base).is_file():
            return sidecar_path(base)
    return resolve_phase3_input(path)


def phase_sources(review: ReviewDocument, corpus: Corpus, inputs: ConclusionInputs, output_dir: Path, base: Path,
                  bibliography: str) -> Dict[str, str]:
    """The texts the conclusion must not copy: the review, Phase 2 and 3's sections (as written, or rebuilt) and the papers."""
    a = inputs.analysis
    sources = {"the review": "\n".join(review.lines)}
    for name, suffix in (("the methodology", "methodology"), ("the results", "results")):
        written = next((d / f"{base.stem}-{suffix}.md" for d in dict.fromkeys((output_dir, base.parent))
                        if (d / f"{base.stem}-{suffix}.md").is_file()), None)
        if written:
            sources[name] = written.read_text(encoding="utf-8")
    if "the methodology" not in sources:
        sources["the methodology"] = build_methodology(MethodologyInputs(
            topic=inputs.topic, review_type=inputs.review_type, slug=inputs.slug, studies=[p.study for p in a.profiles],
            tradition=a.tradition, field_name=inputs.field_name, field_overridden=False,
            recommendation=a.recommendation, bibliography="",
        ))
    if "the results" not in sources:
        sources["the results"] = build_results(ResultsInputs(
            topic=inputs.topic, review_type=inputs.review_type, slug=inputs.slug, profiles=a.profiles, themes=a.themes,
            tradition=a.tradition, recommendation=a.recommendation, from_phase2=a.from_phase2, bibliography="",
        ))
    sources["the papers"] = "\n".join(p.text for p in corpus.papers)
    return sources


def analysis_json(inputs: ConclusionInputs, source_path: Path, checks: Sequence[Check]) -> Dict:
    a = inputs.analysis
    strength, why = evidence_strength(a.profiles, a.themes)
    return {
        "version": 1,
        "topic": inputs.topic,
        "review": str(source_path),
        "review_type": inputs.review_type,
        "papers_analysed": len(a.profiles),
        "field": {"tradition": a.tradition.key, "tradition_name": a.tradition.name, "vocabulary_field": inputs.field_name},
        "design": {"key": a.recommendation.design.key, "label": a.recommendation.design.label,
                   "from_phase2": a.from_phase2},
        "evidence": {
            "strength": strength,
            "reason": why,
            "study_confidence": {p.study.narrative: p.confidence for p in a.profiles},
            "shared_findings": [{"finding": t.term, "consistency": t.consistency, "confidence": t.confidence,
                                 "studies": [p.study.narrative for p in t.profiles]} for t in a.themes],
        },
        "research_gaps": [asdict(g) for g in inputs.gaps],
        "future_directions": [asdict(d) for d in inputs.directions],
        "quality_checks": [asdict(c) for c in checks],
    }


def run_phase4(args) -> List[Path]:
    load_dotenv()
    input_path = Path(args.input)
    if not input_path.is_file():
        raise ValueError(f"Review file not found: {input_path}")
    review, base = resolve_review(resolve_input(input_path))
    corpus = load_corpus(review)
    if not corpus.papers:
        raise ValueError(
            f"No papers found for {input_path}: Phase 4 needs the review's {SIDECAR_SUFFIX} file or its References section"
        )
    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = base.stem
    formats = [f.strip().lower() for f in args.formats.split(",") if f.strip()]
    style = (args.citation_style or corpus.citation_style or "apa").lower()
    topic = corpus.topic or review.topic

    logger.info("Phase 4: drawing the conclusion from the review, methodology and results for %r", topic)
    analysis = analyse_results(corpus, topic, base, output_dir, getattr(args, "field", None), args.no_llm, args.model)
    gaps = review_gaps(review, analysis.profiles)
    directions = future_directions(topic, gaps, analysis, corpus.field)
    review_key = next((k for k, m in METHODOLOGIES.items() if m.name.lower() == review.review_type.lower()), "")
    bibliography = CitationFormatter().format_bibliography([p.paper for p in corpus.papers], style=style)
    inputs = ConclusionInputs(
        topic=topic, review_type=review.review_type, review_key=review_key, slug=slug, analysis=analysis, gaps=gaps,
        directions=directions, field_name=corpus.field, bibliography=bibliography,
    )
    markdown = build_conclusion(inputs)
    checks = quality_checks(markdown, inputs, phase_sources(review, corpus, inputs, output_dir, base, bibliography), corpus)
    for check in checks:
        (logger.info if check.passed else logger.warning)("Check %s: %s (%s)", check.name, "passed" if check.passed else "FAILED", check.detail)

    written = Exporter(output_dir=output_dir).export_all(markdown, f"{slug}-conclusion", formats)
    analysis_path = output_dir / f"{slug}-conclusion-analysis.json"
    analysis_path.write_text(json.dumps(analysis_json(inputs, input_path, checks), indent=2, ensure_ascii=False), encoding="utf-8")
    written.append(analysis_path)
    for path in written:
        logger.info("Written: %s", path)
    logger.info(
        "%d future directions; %d of %d quality checks passed. Places for your own conclusions are marked (%s).",
        len(directions), sum(c.passed for c in checks), len(checks), ", ".join(markers.CONCLUSION_MARKERS),
    )
    return written
