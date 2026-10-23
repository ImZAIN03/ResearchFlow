"""Phase 2: proposes a Methodology section for the user's own study, grounded in the reviewed papers' designs.

    python -m src.main --phase 2 --input "output/my-review.md"

Phase 1 reviews what has been studied; Phase 2 turns how it was studied into a methodology for the
user's next study. For every paper in the review (read from the <review>.researchflow.json file
Phase 1 writes, or recovered from the review's references) it reads, from the title, abstract,
Phase 1 summary and findings:

    research design     e.g. survey, case study, RCT, computational experiment, doctrinal analysis
    approach            quantitative, qualitative, mixed methods, evidence synthesis, conceptual
    sample              sizes stated in the text ("300 cybersecurity practitioners")
    data collection     interviews, questionnaires, datasets, documents, legal sources, ...
    analysis            regression, thematic analysis, performance metrics, process tracing, ...
    context             the countries and populations studied

The papers' designs, and to a lesser degree their vocabulary, decide which methodological tradition
the recommendations follow: medicine and health, social sciences, engineering and computing, law, or
the humanities (--field overrides it). The recommended design is the one the papers share, mixed
methods when they use both quantitative and qualitative designs, or the tradition's standard
empirical design when none of them collected primary data. The section covers the design and its
rationale, sample, data collection, analysis, validity, ethics and timeline, and compares every
paper's design with the proposed one. The decisions only the user can make are marked:

    [YOUR METHODOLOGY CHOICE]     the design and research question
    [YOUR SAMPLE SIZE]            sample size, sampling strategy and criteria
    [YOUR DATA SOURCES]           where the data will actually come from
    [YOUR ANALYSIS PLAN]          the concrete analysis steps
    [YOUR TIMELINE & RESOURCES]   schedule, access, software, funding

Classification is rule-based, so Phase 2 runs without a model. Only when a paper's design can't be
recognized (and Ollama is running) does the local model pick one from the fixed list of designs.
Writes <review>-methodology.md and .docx, plus <review>-methodology-analysis.json with every paper's
design breakdown and the sentence each detail was read from.
"""
from __future__ import annotations

import json
import logging
import re
import statistics
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import requests
from dotenv import load_dotenv

from . import critical_gaps as markers
from .citation_formatter import CitationFormatter
from .export import Exporter
from .gap_parser import ReviewDocument, parse_review
from .gap_research import FIELD_CUES, SIDECAR_SUFFIX, Corpus, SourcePaper, load_corpus, sidecar_path
from .llm_synthesis import LLMSynthesizer, OllamaConnectionError, join_items, normalize_whitespace, split_sentences
from .review_methodology import get_methodology
from .thesis_standards import DEFAULT_DEGREE, degree_paragraph, degree_standard

logger = logging.getLogger("researchflow")

PHASE2_FORMATS = "md,docx"
DERIVED_SUFFIXES = ("-final", "-complete")  # Phase 1.5/1.6 reviews share the Phase 1 review's paper file
EVIDENCE_MAX_WORDS = 30
VOCABULARY_WEIGHT = 0.5  # the papers' designs say more about the methodological tradition than their wording

QUANT, QUAL, MIXED = "Quantitative", "Qualitative", "Mixed methods"
SYNTHESIS, CONCEPTUAL, DESIGN_SCIENCE, UNKNOWN = "Evidence synthesis", "Conceptual", "Design science", "Not stated"

NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "hundred": 100,
}
SPELLED = {n: w for w, n in NUMBER_WORDS.items() if n <= 10}


# ---- research designs -------------------------------------------------------------------------

@dataclass(frozen=True)
class Design:
    key: str
    label: str
    approach: str
    cues: Tuple[str, ...]  # regexes, matched against lower-case sentences
    families: Tuple[str, ...] = ()  # methodological traditions (TRADITIONS keys) the design is typical of
    empirical: bool = True  # collects or analyses primary data (reviews and conceptual work don't)


SURVEY_CUES = (
    r"questionnaires?", r"\brespondents?\b", r"structured surveys?", r"online surveys?",
    r"surveys? (?:was|were) (?:administered|distributed|conducted|completed|sent)", r"administered to",
    r"\bsurveyed \d", r"likert", r"cross-sectional", r"survey data", r"national survey", r"survey of \d",
)

# In priority order: a paper is classified by the first design whose cues it contains, so specific
# designs (a meta-analysis, an RCT) come before the general ones they also mention (a framework).
DESIGNS: Tuple[Design, ...] = (
    Design("meta-analysis", "Meta-analysis", SYNTHESIS,
           (r"meta-analy", r"pooled (?:effect|estimate)", r"random-effects model", r"forest plot"),
           ("health",), empirical=False),
    Design("systematic-review", "Systematic / scoping review", SYNTHESIS,
           (r"systematic (?:literature )?review", r"scoping review", r"\bprisma\b", r"umbrella review"),
           ("health",), empirical=False),
    Design("mixed-methods", "Mixed-methods design", MIXED,
           (r"mixed[- ]methods?", r"qualitative and quantitative", r"quantitative and qualitative", r"multi-method"),
           ("social",)),
    Design("rct", "Randomised controlled trial", QUANT,
           (r"randomi[sz]ed (?:controlled )?(?:clinical )?trial", r"\brcts?\b", r"randomly (?:assigned|allocated)", r"placebo"),
           ("health",)),
    Design("quasi-experimental", "Quasi-experimental design", QUANT,
           (r"quasi-experiment", r"\bpre-?test", r"\bpost-?test", r"difference-in-differences",
            r"non-?equivalent (?:control|comparison) group", r"interrupted time[- ]series"),
           ("social", "health")),
    Design("observational", "Observational study (cohort / case-control / longitudinal)", QUANT,
           (r"\bcohort\b", r"case-control", r"longitudinal (?:study|data|design|analysis)",
            r"retrospective (?:study|analysis|review|cohort)", r"prospective (?:study|cohort)", r"\bregistry\b",
            r"health records", r"panel data"),
           ("health",)),
    Design("survey", "Cross-sectional survey", QUANT, SURVEY_CUES, ("social",)),
    Design("ethnography", "Ethnography", QUAL,
           (r"ethnograph", r"participant observation", r"\bfieldwork\b", r"field notes"), ("social",)),
    Design("interview", "Qualitative interview study", QUAL,
           (r"semi-structured", r"in-depth interviews?", r"focus groups?", r"interviews? with", r"\binterviewed\b",
            r"qualitative (?:study|research|interviews?)", r"thematic analysis", r"grounded theory", r"phenomenolog"),
           ("social",)),
    Design("experiment", "Controlled experiment / user study", QUANT,
           (r"user stud(?:y|ies)", r"controlled experiments?", r"la(?:b|boratory) experiments?", r"between-subjects",
            r"within-subjects", r"experimental (?:group|condition)s?", r"participants were (?:asked|randomly|assigned|shown)",
            r"field experiments?"),
           ("social",)),
    Design("computational-experiment", "Computational experiment (benchmark evaluation)", QUANT,
           (r"benchmark", r"\bdata ?sets?\b", r"ablation", r"evaluated on", r"tested on", r"experiments? on",
            r"\bwe evaluate", r"outperform", r"\bbaselines?\b", r"cross-validation"),
           ("engineering",)),
    Design("simulation", "Simulation / modelling study", QUANT,
           (r"simulat", r"agent-based", r"monte carlo", r"numerical model", r"finite element", r"digital twin"),
           ("engineering",)),
    Design("doctrinal", "Doctrinal legal analysis", QUAL,
           (r"doctrinal", r"black-letter", r"case law", r"jurisprudence", r"\bstatutory\b", r"\blegislation\b",
            r"court (?:decision|ruling|judg(?:e)?ment)s?", r"\bjudg(?:e)?ments\b", r"\btreat(?:y|ies)\b"),
           ("law",)),
    Design("design-development", "Design and development (prototype / system)", DESIGN_SCIENCE,
           (r"prototype", r"proof[- ]of[- ]concept", r"\bwe (?:develop|implement|build|design)\b",
            r"open-source (?:framework|tool|library|software|platform|system)", r"system architecture", r"design science"),
           ("engineering",)),
    Design("case-study", "Case study", QUAL,
           (r"case stud(?:y|ies)", r"single case", r"in-depth case", r"seminal case", r"\bcase (?:analysis|research)\b"),
           ("social", "law")),
    Design("historical", "Historical / archival analysis", QUAL,
           (r"historical (?:analysis|research|method|study)", r"archiv", r"process[- ]tracing", r"historiograph",
            r"declassified", r"oral histor"),
           ("humanities",)),
    Design("literature-review", "Narrative literature review", SYNTHESIS,
           (r"literature review", r"review of (?:the )?(?:literature|research|studies|existing)", r"\bwe review\b",
            r"\bthis review\b", r"\ba (?:comprehensive |brief |short )?survey (?:of|on)\b", r"\boverview of\b",
            r"narrative review", r"state of the art (?:in|of)"),
           (), empirical=False),
    Design("conceptual", "Conceptual / theoretical analysis", CONCEPTUAL,
           (r"conceptual", r"theoretical (?:framework|model|analysis|argument)", r"\bwe argue\b", r"position paper",
            r"taxonomy", r"framework", r"\bargues?\b", r"\bessay\b"),
           (), empirical=False),
)
UNCLASSIFIED = Design("unclassified", "Unclear from the abstract", UNKNOWN, (), (), empirical=False)
DESIGN_BY_KEY: Dict[str, Design] = {d.key: d for d in DESIGNS + (UNCLASSIFIED,)}

# How data were collected and analysed: (key, label, cues).
COLLECTION_METHODS: Tuple[Tuple[str, str, Tuple[str, ...]], ...] = (
    ("interviews", "Interviews", (r"interview",)),
    ("focus-groups", "Focus groups", (r"focus groups?",)),
    ("questionnaires", "Questionnaires / surveys", SURVEY_CUES),
    ("participants-experiment", "Experiments with participants",
     (r"user stud(?:y|ies)", r"participants were", r"randomly (?:assigned|allocated)", r"\btrials?\b")),
    ("datasets", "Existing datasets / benchmarks",
     (r"\bdata ?sets?\b", r"benchmark", r"\bcorpus\b", r"publicly available data", r"\bopen data\b")),
    ("records", "Clinical or administrative records",
     (r"medical records?", r"health records?", r"\bregistry\b", r"administrative data", r"claims data")),
    ("observation", "Observation / fieldwork",
     (r"\bobservations?\b", r"participant observation", r"\bfieldwork\b", r"field (?:study|visits?)", r"site visits?")),
    ("documents", "Documents, archives and published literature",
     (r"document analysis", r"(?:official|policy|government|legal|internal) documents", r"\bdocumentary\b", r"archiv",
      r"secondary (?:sources?|data)", r"existing literature", r"evolving literature", r"recent evidence",
      r"\bnews\b", r"media reports")),
    ("legal-sources", "Legislation and case law",
     (r"case law", r"\bstatut", r"\blegislation\b", r"\btreat(?:y|ies)\b", r"\bcourts?\b", r"\bjudg(?:e)?ments?\b")),
    ("simulation", "Simulation-generated data", (r"simulat", r"synthetic data", r"monte carlo")),
    ("measurements", "Sensor, log or measurement data",
     (r"\bsensors?\b", r"\bmeasurements\b", r"log (?:data|files)", r"network traffic", r"telemetry")),
    ("hardware", "Laboratory / hardware tests", (r"\bhardware\b", r"laborator", r"\blab\b", r"testbed")),
    ("literature-databases", "Bibliographic database searches",
     (r"databases? (?:were|was) searched", r"\bscopus\b", r"web of science", r"\bpubmed\b", r"search strategy",
      r"literature search")),
)
ANALYSIS_METHODS: Tuple[Tuple[str, str, Tuple[str, ...]], ...] = (
    ("inferential", "Inferential statistics (regression, significance tests)",
     (r"regression", r"\banova\b", r"t-tests?", r"chi-square", r"correlat", r"structural equation", r"\bp ?[<=]",
      r"statistically significant", r"odds ratios?", r"hazard ratios?", r"\br(?:²|2|\^2) ?=")),
    ("descriptive", "Descriptive statistics",
     (r"descriptive (?:statistics|research|analysis|design)", r"frequenc(?:y|ies) (?:analysis|distribution)",
      r"\bpercentages\b")),
    ("meta-pooling", "Meta-analytic pooling of effect sizes",
     (r"meta-analy", r"\bpooled\b", r"random-effects", r"heterogeneity", r"forest plot")),
    ("thematic", "Thematic / qualitative coding",
     (r"thematic", r"grounded theory", r"\bnvivo\b", r"atlas\.ti", r"qualitative (?:data )?analysis", r"\bthemes\b")),
    ("content", "Content or discourse analysis",
     (r"content analysis", r"discourse analysis", r"textual analysis", r"narrative analysis")),
    ("performance", "Performance metrics and benchmark comparison",
     (r"classification accuracy", r"accuracy of (?:the )?(?:model|classifier)s?",
      r"(?:achiev|reach|obtain)(?:es|ed|ing)? (?:an? )?(?:accuracy|f1)", r"precision and recall", r"f1[- ]score",
      r"\bauc\b", r"predictive performance", r"ablation", r"cross-validation", r"benchmark", r"\bbaselines?\b")),
    ("simulation-modelling", "Simulation / computational modelling",
     (r"simulat", r"agent-based", r"monte carlo", r"optimi[sz]ation model")),
    ("assessment-frameworks", "Structured assessment frameworks (risk, threat, SWOT)",
     (r"risk assessment", r"threat model", r"vulnerability assessment", r"\bswot\b", r"kill chain", r"att&ck",
      r"cost-benefit")),
    ("comparative", "Comparative analysis",
     (r"comparative analysis", r"cross-case", r"cross-national", r"compar(?:e|ed|ing|ison) (?:of|with|across|between)")),
    ("legal-interpretation", "Legal interpretation (doctrinal analysis)",
     (r"doctrinal", r"statutory interpretation", r"case law", r"jurisprudence", r"legal interpretation")),
    ("interpretive", "Interpretive / historical analysis",
     (r"\bhistor", r"process[- ]tracing", r"interpret(?:ive|ation)\b", r"\brevisit", r"in light of")),
    ("narrative-synthesis", "Narrative synthesis of the literature",
     (r"literature review", r"systematic review", r"narrative synthesis",
      r"synthesi[sz]e[sd]? (?:the )?(?:literature|evidence|studies)")),
)
COLLECTION_LABELS = {key: label for key, label, _ in COLLECTION_METHODS}
ANALYSIS_LABELS = {key: label for key, label, _ in ANALYSIS_METHODS}
# Data that come from people: only then do the populations a paper mentions describe who it studied.
HUMAN_COLLECTION = {"interviews", "focus-groups", "questionnaires", "participants-experiment", "observation", "records"}
POPULATION_DESIGNS = {"case-study", "ethnography", "historical", "doctrinal"}

# Units a sample size can be counted in, grouped for the sample ranges.
UNIT_GROUPS: Dict[str, Tuple[str, ...]] = {
    "people": (
        r"participants?", r"respondents?", r"patients?", r"subjects?", r"individuals?", r"people", r"persons?",
        r"adults?", r"children", r"adolescents?", r"students?", r"teachers?", r"nurses?", r"physicians?",
        r"clinicians?", r"practitioners?", r"professionals?", r"experts?", r"employees?", r"workers?", r"managers?",
        r"users?", r"consumers?", r"citizens?", r"interviewees?", r"volunteers?", r"households?",
    ),
    "organisations": (
        r"firms?", r"companies", r"company", r"organi[sz]ations?", r"enterprises?", r"smes", r"schools?",
        r"hospitals?", r"universities", r"agencies", r"countries", r"nations?",
    ),
    "cases and documents": (
        r"cases?", r"incidents?", r"attacks?", r"interviews?", r"focus groups?", r"documents?", r"judg(?:e)?ments?",
        r"decisions?", r"articles?", r"papers?", r"studies", r"trials?", r"reports?", r"texts?",
    ),
    "data items": (
        r"datasets?", r"data sets?", r"images?", r"records?", r"samples?", r"tweets?", r"posts?", r"transactions?",
        r"observations?", r"instances?",
    ),
}
SAMPLE_PATTERN = re.compile(
    # "one" is left out: "one of the users" is not a sample.
    r"\b(\d{1,3}(?:,\d{3})+|\d+|" + "|".join(w for w, n in NUMBER_WORDS.items() if n > 1) + r")\s+((?:[a-z][\w-]*\s+){0,3}?)("
    + "|".join(p for patterns in UNIT_GROUPS.values() for p in patterns) + r")\b",
    re.IGNORECASE,
)
N_EQUALS = re.compile(r"\b[nN]\s*=\s*(\d{1,3}(?:,\d{3})+|\d+)")

COUNTRIES = (
    "Afghanistan", "Albania", "Algeria", "Argentina", "Armenia", "Australia", "Austria", "Azerbaijan", "Bangladesh",
    "Belarus", "Belgium", "Bolivia", "Bosnia", "Brazil", "Bulgaria", "Cambodia", "Cameroon", "Canada", "Chile",
    "China", "Colombia", "Croatia", "Cuba", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Egypt", "Estonia",
    "Ethiopia", "Finland", "France", "Georgia", "Germany", "Ghana", "Greece", "Hungary", "Iceland", "India",
    "Indonesia", "Iran", "Iraq", "Ireland", "Israel", "Italy", "Japan", "Kazakhstan", "Kenya", "Kuwait", "Latvia",
    "Lebanon", "Libya", "Lithuania", "Malaysia", "Mexico", "Moldova", "Morocco", "Myanmar", "Nepal", "Netherlands",
    "New Zealand", "Nigeria", "North Korea", "Norway", "Oman", "Pakistan", "Palestine", "Peru", "Philippines",
    "Poland", "Portugal", "Qatar", "Romania", "Russia", "Rwanda", "Saudi Arabia", "Serbia", "Singapore", "Slovakia",
    "Slovenia", "South Africa", "South Korea", "Spain", "Sri Lanka", "Sudan", "Sweden", "Switzerland", "Syria",
    "Taiwan", "Tanzania", "Thailand", "Tunisia", "Turkey", "Uganda", "Ukraine", "United Arab Emirates",
    "United Kingdom", "United States", "Uruguay", "Uzbekistan", "Venezuela", "Vietnam", "Yemen", "Zambia", "Zimbabwe",
    "Africa", "Asia", "Europe", "European Union", "Latin America", "Middle East", "Global South", "Southeast Asia",
)
COUNTRY_ALIASES = {"USA": "United States", "U.S.": "United States", "UK": "United Kingdom", "EU": "European Union"}
COUNTRY_PATTERNS = [(re.compile(r"\b" + re.escape(name) + r"\b"), name) for name in COUNTRIES] + [
    (re.compile(r"(?<![\w.])" + re.escape(alias) + r"(?![\w])"), name) for alias, name in COUNTRY_ALIASES.items()
]
POPULATIONS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("patients", (r"\bpatients?\b",)),
    ("health professionals", (r"\bnurses?\b", r"\bphysicians?\b", r"\bclinicians?\b", r"health(?:care)? professionals?")),
    ("students", (r"\bstudents?\b", r"\blearners\b", r"\bpupils\b")),
    ("teachers", (r"\bteachers?\b", r"\beducators?\b")),
    ("children and adolescents", (r"\bchildren\b", r"\badolescents?\b", r"\byouth\b")),
    ("older adults", (r"older adults", r"\belderly\b")),
    ("practitioners and employees", (r"\bpractitioners\b", r"\bprofessionals\b", r"\bemployees\b", r"\bworkers\b", r"\bmanagers\b")),
    ("experts", (r"\bexperts\b",)),
    ("organisations and firms", (r"\bfirms\b", r"\bcompanies\b", r"\borgani[sz]ations\b", r"\bsmes\b", r"\benterprises\b")),
    ("governments and states", (r"\bgovernments?\b", r"\bnations\b", r"(?<!united )\bstates\b", r"\bstate actors\b",
                                r"\bmilitar(?:y|ies)\b", r"\bdefen[cs]e (?:sector|organi[sz]ations|agencies)\b")),
    ("end users and consumers", (r"\busers\b", r"\bconsumers\b", r"\bcustomers\b")),
    ("citizens and the public", (r"\bcitizens\b", r"\bgeneral public\b", r"\bresidents\b", r"\bhouseholds\b")),
)


# ---- methodological guidance per design -------------------------------------------------------

DEFAULT_PHASES = (
    ("Research design and ethics approval", "1–2 months"),
    ("Data collection", "2–4 months"),
    ("Analysis", "1–2 months"),
    ("Writing up", "1–2 months"),
)


@dataclass(frozen=True)
class DesignGuide:
    summary: str
    strength: str  # completes "The proposed design ..."
    limitation: str  # completes "A <design> ..."
    collection: Tuple[Tuple[str, str], ...]  # (COLLECTION_METHODS key, option)
    analysis: Tuple[Tuple[str, str], ...]  # (ANALYSIS_METHODS key, option)
    sample: str  # the design's sample-size convention (not taken from the papers)
    phases: Tuple[Tuple[str, str], ...] = DEFAULT_PHASES  # (stage, indicative duration)


GUIDES: Dict[str, DesignGuide] = {
    "meta-analysis": DesignGuide(
        "Statistically pools the effect sizes of comparable quantitative studies to estimate an overall effect and explain differences between studies.",
        "gives a pooled, more precise estimate across many studies",
        "depends on the quality and comparability of the studies it pools",
        (("literature-databases", "Systematic searches of bibliographic databases (e.g. Scopus, Web of Science, PubMed) following a registered protocol"),
         ("documents", "Data extraction from the included studies' full texts with a piloted extraction form")),
        (("meta-pooling", "Random-effects meta-analysis of effect sizes, with heterogeneity (I²) and subgroup or meta-regression analyses"),
         ("inferential", "Publication-bias checks (funnel plot, Egger's test) and sensitivity analyses")),
        "All eligible studies the search finds; at least 5–10 comparable studies are usually needed for a meaningful pooled estimate.",
        (("Protocol and registration (e.g. PROSPERO or OSF)", "1 month"), ("Searching and screening", "1–2 months"),
         ("Data extraction and risk-of-bias assessment", "1–2 months"), ("Pooling and sensitivity analyses", "1 month"),
         ("Writing up", "1 month")),
    ),
    "systematic-review": DesignGuide(
        "Identifies, screens and appraises every study that meets predefined eligibility criteria, then synthesizes them.",
        "covers the evidence comprehensively and reproducibly",
        "produces no new primary data and is limited by the studies available",
        (("literature-databases", "Searches of several bibliographic databases plus grey literature, with a documented search string"),
         ("documents", "Data extraction and quality appraisal (e.g. CASP or RoB 2) of the included studies")),
        (("narrative-synthesis", "Narrative or thematic synthesis structured by the review question"),
         ("meta-pooling", "Meta-analysis where the studies are comparable enough")),
        "Every study that meets the eligibility criteria; scope the question so the number stays manageable (typically tens of studies).",
        (("Protocol and registration", "1 month"), ("Searching and screening", "1–2 months"),
         ("Data extraction and appraisal", "1–2 months"), ("Synthesis and writing up", "1–2 months")),
    ),
    "mixed-methods": DesignGuide(
        "Combines a quantitative strand (measuring how much or how often) with a qualitative strand (explaining how and why) and integrates their results.",
        "combines the breadth of quantitative data with the depth of qualitative evidence",
        "takes more time and skills, and its two strands must be integrated rather than reported side by side",
        (("questionnaires", "A structured questionnaire for the quantitative strand"),
         ("interviews", "Semi-structured interviews or case studies for the qualitative strand")),
        (("inferential", "Statistical analysis of the quantitative strand"),
         ("thematic", "Thematic analysis of the qualitative strand"),
         ("comparative", "Integration of both strands, e.g. a joint display that sets the statistical results beside the themes")),
        "Each strand is sized by its own logic: the quantitative strand by a power analysis, the qualitative strand by saturation (often 10–30 participants, purposively drawn from the quantitative sample).",
    ),
    "rct": DesignGuide(
        "Randomly allocates participants to an intervention or a control group and compares their outcomes, giving the strongest evidence of cause and effect.",
        "establishes cause and effect by controlling confounding through randomisation",
        "is costly, and its strict eligibility criteria can limit how far the results generalise",
        (("participants-experiment", "Recruitment and random allocation of participants, with blinding where possible"),
         ("measurements", "Validated outcome measures collected at baseline and at follow-up")),
        (("inferential", "Intention-to-treat comparison of the groups (e.g. t-tests, ANCOVA or regression), with effect sizes and confidence intervals"),),
        "Set by an a priori power analysis (e.g. 80% power, alpha = 0.05) for the smallest effect that matters, plus an allowance for drop-out (often 10–20%).",
        (("Protocol, ethics approval and trial registration", "2–3 months"), ("Recruitment and randomisation", "3–6 months"),
         ("Intervention and follow-up", "3–12 months"), ("Analysis", "1–2 months"), ("Writing up", "1–2 months")),
    ),
    "quasi-experimental": DesignGuide(
        "Compares groups or time points before and after an intervention without random allocation, e.g. a pre-test/post-test design with a comparison group.",
        "tests the effect of an intervention in real settings where randomisation is impractical",
        "cannot rule out confounding as fully as randomisation can",
        (("participants-experiment", "Pre- and post-intervention measurements in an intervention and a comparison group"),
         ("records", "Routinely collected or administrative data from before and after a change")),
        (("inferential", "Difference-in-differences, ANCOVA or interrupted time-series analysis, controlling for baseline differences"),),
        "Use a power analysis as for a trial and match the groups on key characteristics; time-series designs need enough time points (often 8 or more before and after).",
    ),
    "observational": DesignGuide(
        "Follows or compares groups as they naturally occur (cohort, case-control or longitudinal designs) to identify associations and risk factors.",
        "studies real-world exposures and outcomes at scale, including ones that cannot ethically be manipulated",
        "shows associations, so confounding must be controlled before inferring causes",
        (("records", "Existing records, registries or panel data"),
         ("questionnaires", "Repeated questionnaires or measurements over time")),
        (("inferential", "Multivariable regression (e.g. logistic or Cox) adjusting for confounders, with sensitivity analyses"),),
        "Typically hundreds to thousands of participants; base the size on the expected event rate and the number of confounders to adjust for.",
    ),
    "survey": DesignGuide(
        "Collects standardised data from a sample of a population at one point in time with a structured questionnaire, to describe it and test relationships between variables.",
        "measures attitudes, practices and relationships across a large, comparable sample",
        "relies on self-report at a single time point, so it shows associations rather than causes",
        (("questionnaires", "A structured questionnaire, adapted from validated scales where possible and piloted first"),
         ("datasets", "Existing national or organisational survey data, if available")),
        (("descriptive", "Descriptive statistics of the sample and key variables"),
         ("inferential", "Inferential tests and multivariate models (e.g. regression or structural equation modelling) for the hypothesised relationships")),
        "Use a power analysis or a sample-size formula (e.g. Cochran's); 200–400 complete responses is common for multivariate analysis, and response rates of 20–40% mean inviting several times that number.",
        (("Questionnaire design and ethics approval", "1–2 months"), ("Pilot test", "2–4 weeks"),
         ("Data collection", "1–3 months"), ("Data cleaning and analysis", "1–2 months"), ("Writing up", "1 month")),
    ),
    "ethnography": DesignGuide(
        "Studies a group or setting from the inside through extended participant observation, informal conversations and field notes.",
        "captures everyday practices and meanings that interviews and surveys miss",
        "is time-intensive, and its findings depend on the researcher's access and position",
        (("observation", "Participant observation with detailed field notes"),
         ("interviews", "Informal and semi-structured interviews with members of the setting"),
         ("documents", "Documents and artefacts produced in the setting")),
        (("thematic", "Iterative thematic coding of field notes and transcripts"),
         ("interpretive", "Thick description and interpretation of practices in context")),
        "One or a few sites studied in depth; the length of fieldwork (often several months) matters more than the number of people.",
        (("Access, ethics approval and preparation", "1–3 months"), ("Fieldwork", "3–12 months"),
         ("Analysis", "2–3 months"), ("Writing up", "2–3 months")),
    ),
    "interview": DesignGuide(
        "Explores participants' experiences, views and reasoning in depth through semi-structured interviews or focus groups.",
        "explains how and why people act as they do, in their own words",
        "draws on a small, purposive sample, so its findings are transferable rather than statistically generalisable",
        (("interviews", "Semi-structured interviews with a piloted interview guide"),
         ("focus-groups", "Focus groups where interaction between participants adds insight")),
        (("thematic", "Thematic analysis (e.g. Braun and Clarke's six phases), supported by software such as NVivo"),
         ("content", "Qualitative content analysis where categories are defined in advance")),
        "Purposive sampling until thematic saturation, often 12–30 interviews (or 3–6 focus groups).",
        (("Interview guide and ethics approval", "1–2 months"), ("Recruitment and interviews", "2–3 months"),
         ("Transcription and coding", "1–2 months"), ("Analysis and writing up", "1–2 months")),
    ),
    "experiment": DesignGuide(
        "Manipulates one or more conditions in a controlled setting (e.g. a user study) and measures their effect on participants' behaviour or performance.",
        "isolates the effect of specific factors under controlled conditions",
        "takes place in controlled settings that may not reflect real-world behaviour",
        (("participants-experiment", "Participants assigned to experimental conditions (between- or within-subjects)"),
         ("measurements", "Task performance, logs or validated questionnaires as outcome measures")),
        (("inferential", "t-tests, ANOVA or mixed-effects models comparing conditions, with effect sizes"),),
        "Set by a power analysis; user studies commonly have 20–40 participants per condition, and within-subjects designs need fewer.",
    ),
    "computational-experiment": DesignGuide(
        "Implements a method or model and evaluates it on benchmark datasets against baselines, using standard performance metrics.",
        "gives reproducible, quantitative comparisons against established baselines",
        "ties its results to the datasets and settings tested, which may not match real deployments",
        (("datasets", "Public benchmark datasets used by prior work, plus a dataset closer to your target setting if one is available"),
         ("simulation", "Synthetic or simulated data to test conditions the benchmarks do not cover")),
        (("performance", "Performance metrics (e.g. accuracy, F1, AUC or error rates) against baselines, with ablation studies"),
         ("inferential", "Repeated runs with different random seeds, reporting means with confidence intervals or significance tests")),
        "Use the standard benchmark splits so results are comparable, evaluate on more than one dataset, and report the number of runs (commonly 3–10 seeds).",
        (("Dataset selection and baseline reproduction", "1–2 months"), ("Method implementation", "2–3 months"),
         ("Experiments and ablations", "1–2 months"), ("Analysis and writing up", "1–2 months")),
    ),
    "simulation": DesignGuide(
        "Builds a computational model of a system and runs it under controlled scenarios to study behaviour that is hard to observe directly.",
        "explores scenarios and parameters that cannot be tested in the real world",
        "is only as valid as the model's assumptions, so it needs calibration and validation against real data",
        (("simulation", "Simulation runs across a designed set of scenarios and parameters"),
         ("datasets", "Empirical data to calibrate and validate the model")),
        (("simulation-modelling", "Scenario comparison and sensitivity analysis of the key parameters"),
         ("inferential", "Statistical summaries across replications (means, variance, confidence intervals)")),
        "Enough replications per scenario for stable estimates (often 30 or more for stochastic models) and a parameter grid covering realistic ranges.",
        (("Model design and calibration data", "1–2 months"), ("Implementation and verification", "2–3 months"),
         ("Scenario runs and sensitivity analysis", "1–2 months"), ("Writing up", "1–2 months")),
    ),
    "doctrinal": DesignGuide(
        "Analyses primary legal sources (legislation, case law, treaties) and secondary commentary to state, interpret and evaluate the law on a question.",
        "gives an authoritative, systematic account of what the law is and how it should develop",
        "describes the law on paper rather than how it works in practice",
        (("legal-sources", "Primary sources: legislation, case law and treaties from the relevant jurisdictions, via legal databases (e.g. Westlaw, LexisNexis, HeinOnline or official repositories)"),
         ("documents", "Secondary sources: academic commentary, law-reform reports and policy documents")),
        (("legal-interpretation", "Doctrinal interpretation (textual, purposive and systematic) of the sources"),
         ("comparative", "Comparative analysis across jurisdictions where relevant")),
        "Define the jurisdictions, time period and case-selection criteria; the sources are usually all leading authorities rather than a statistical sample.",
        (("Scoping the question and jurisdictions", "1 month"), ("Collecting primary and secondary sources", "2–3 months"),
         ("Analysis and argument", "2–3 months"), ("Writing up", "1–2 months")),
    ),
    "design-development": DesignGuide(
        "Designs and builds an artefact (a prototype, tool or system) to solve a defined problem, and evaluates it against requirements.",
        "produces a working artefact together with evidence of how well it solves the problem",
        "is often evaluated only in its developers' own test settings",
        (("hardware", "Prototype testing in a laboratory or testbed"),
         ("participants-experiment", "Usability or expert evaluation with intended users")),
        (("performance", "Evaluation against functional and performance requirements and against existing solutions"),
         ("thematic", "Qualitative analysis of user or expert feedback")),
        "Iterative build-and-evaluate cycles; evaluations commonly involve 5–20 expert or user testers per round, or a defined set of test scenarios.",
        (("Requirements and design", "1–2 months"), ("Building the prototype", "2–4 months"),
         ("Evaluation", "1–2 months"), ("Writing up", "1–2 months")),
    ),
    "case-study": DesignGuide(
        "Examines one or a few cases (events, organisations, incidents) in depth and in their real-world context, drawing on several sources of evidence.",
        "explains complex events in their context and can build or test theory from them",
        "may not generalise beyond the cases studied",
        (("documents", "Documents, reports and archival material about each case"),
         ("interviews", "Interviews with people involved in, or knowledgeable about, the case")),
        (("interpretive", "Within-case analysis such as process tracing or pattern matching"),
         ("comparative", "Cross-case comparison when several cases are studied")),
        "One to a few cases (multiple-case designs often use 2–6), each selected for a stated reason, e.g. a typical, extreme or most-similar case.",
        (("Case selection and case-study protocol", "1 month"), ("Data collection from several sources", "2–3 months"),
         ("Within- and cross-case analysis", "1–2 months"), ("Writing up", "1–2 months")),
    ),
    "historical": DesignGuide(
        "Reconstructs and interprets past events from primary sources such as archives, official records and testimonies.",
        "shows how present conditions arose and tests claims against the historical record",
        "depends on which sources survive and are accessible",
        (("documents", "Archival and documentary sources, including official or declassified records"),
         ("interviews", "Oral-history interviews with witnesses where possible")),
        (("interpretive", "Source criticism and historical interpretation, e.g. process tracing"),
         ("content", "Content or discourse analysis of the documents")),
        "A defined body of sources for a stated period; justify how the archive or corpus was selected.",
    ),
    "literature-review": DesignGuide(
        "Summarises and interprets existing research on a topic without a predefined protocol.",
        "maps the main ideas and debates quickly",
        "is open to selection bias and adds no new evidence",
        (("literature-databases", "Searches of bibliographic databases and citation tracking"),),
        (("narrative-synthesis", "Thematic or narrative synthesis of the studies"),),
        "The key studies on the topic; state how they were found and selected.",
    ),
    "conceptual": DesignGuide(
        "Develops or critiques a theory, model or framework through argument grounded in existing literature.",
        "clarifies concepts and proposes frameworks for later testing",
        "does not test its claims against new data",
        (("documents", "Existing literature and theory"),),
        (("interpretive", "Conceptual analysis and argument"),),
        "Not applicable: the argument rests on the literature.",
    ),
    "unclassified": DesignGuide(
        "The study's design could not be identified from its title and abstract.",
        "",
        "does not state its design in the abstract, so check the full text",
        (), (), "",
    ),
}


# ---- methodological traditions ----------------------------------------------------------------

@dataclass(frozen=True)
class Tradition:
    key: str
    name: str
    typical: Tuple[str, ...]  # its standard empirical designs, most standard first
    quality: str  # completes "studies are judged on ..."
    reporting: str  # completes "Follow ..."
    ethics: str


TRADITIONS: Dict[str, Tradition] = {
    t.key: t
    for t in (
        Tradition(
            "health", "Medicine and health sciences",
            ("observational", "rct", "survey", "interview", "quasi-experimental"),
            "internal validity (randomisation, blinding, control of confounders), risk of bias, and clinical as well as statistical significance",
            "the reporting guideline for the design: CONSORT for trials, STROBE for observational studies, COREQ or SRQR for qualitative studies and PRISMA for reviews",
            "Obtain approval from a research ethics committee or IRB, collect informed consent, protect health data under the applicable data-protection law, and register trials prospectively (e.g. on ClinicalTrials.gov).",
        ),
        Tradition(
            "social", "Social sciences",
            ("survey", "interview", "mixed-methods", "case-study", "quasi-experimental", "ethnography", "experiment"),
            "the validity and reliability of instruments (e.g. Cronbach's alpha for scales), the representativeness of the sample, and the credibility, transferability and dependability of qualitative findings",
            "recognised standards: AAPOR guidance or STROBE for surveys, COREQ or SRQR for qualitative studies, and GRAMMS for mixed-methods studies",
            "Obtain approval from an institutional ethics board, collect informed consent, keep participants anonymous or confidential, and take particular care with vulnerable groups and sensitive topics.",
        ),
        Tradition(
            "engineering", "Engineering, computing and the natural sciences",
            ("computational-experiment", "simulation", "design-development", "experiment"),
            "strong baselines and ablations, repeated runs with variance or confidence intervals, held-out test data without leakage, and results on more than one dataset or environment",
            "reproducibility practice: release code, data and configurations, report hardware and random seeds, and use a checklist such as the NeurIPS reproducibility checklist or ACM artifact badging where relevant",
            "Respect dataset licences and consent for any personal data, keep collected data private and secure, and assess dual-use and safety risks of the system; studies with human participants (e.g. user studies) still need ethics approval.",
        ),
        Tradition(
            "law", "Law",
            ("doctrinal", "case-study", "interview", "survey"),
            "the authority and currency of the sources (check that the law is still good law), transparent selection of cases and jurisdictions, and engagement with competing interpretations",
            "a recognised legal citation standard (e.g. OSCOLA or the Bluebook), stating the jurisdictions, period and case-selection criteria explicitly",
            "Doctrinal research on public legal sources rarely needs ethics approval, but empirical legal research (interviews, surveys, court observation) does; handle confidential or privileged material with care.",
        ),
        Tradition(
            "humanities", "Humanities",
            ("historical", "case-study", "interview"),
            "source criticism (provenance, reliability and bias of sources), triangulation across sources, and reflexivity about the interpretive stance",
            "a transparent description of the archive, corpus or texts, how they were selected and the interpretive framework applied",
            "Check copyright and access conditions of archives, and obtain ethics approval for oral histories or interviews.",
        ),
    )
}
# gap_research.FIELD_CUES fields -> the methodological tradition their research usually follows.
FIELD_TRADITION = {
    "medicine and health": "health",
    "biology and life sciences": "health",
    "psychology": "social",
    "computer science and IT": "engineering",
    "engineering": "engineering",
    "physics and mathematics": "engineering",
    "economics and business": "social",
    "law and policy": "law",
    "social sciences and education": "social",
    "humanities": "humanities",
    "environmental science": "engineering",
}
FIELD_CHOICES = tuple(TRADITIONS)

# What examiners and reviewers in each tradition check before anything else: (practice, how to meet it).
FIELD_BEST_PRACTICES: Dict[str, Tuple[Tuple[str, str], ...]] = {
    "health": (
        ("Pre-registration", "Register the protocol (ClinicalTrials.gov, PROSPERO or OSF) before data collection, with primary and secondary outcomes fixed in advance."),
        ("Reporting guideline", "Write to the checklist for your design (CONSORT, STROBE, COREQ/SRQR or PRISMA) and submit the completed checklist as an appendix."),
        ("Effect estimates", "Report effect sizes with 95% confidence intervals, not p-values alone, and distinguish clinical from statistical significance."),
        ("Bias control", "State how allocation, blinding and confounding are handled; analyse by intention to treat where groups are compared."),
        ("Safety", "Record adverse events in every group, even when they are not the study's focus."),
        ("Data protection", "Describe consent, anonymisation and storage of health data under the applicable law (e.g. GDPR or HIPAA)."),
    ),
    "social": (
        ("Validated instruments", "Adapt established scales where they exist, cite their source, and report reliability (Cronbach's alpha or omega of at least 0.70) for your sample."),
        ("Piloting", "Pilot every instrument or interview guide with 5-10 people from the population and report what changed."),
        ("Sampling transparency", "Describe the sampling frame, recruitment route and response rate, and compare respondents with the population on known characteristics."),
        ("Common-method bias", "When one questionnaire measures both predictors and outcomes, separate them in time or source, or test for common-method variance (e.g. Harman's single-factor test)."),
        ("Qualitative rigour", "Keep an audit trail, justify saturation, check codes with a second coder or peer, and reflect on your own position."),
        ("Integration", "In mixed-methods work, integrate the strands in a joint display rather than reporting them side by side."),
    ),
    "engineering": (
        ("Reproducibility", "Fix and report random seeds, release code, configurations and (where licences allow) data, and record hardware, software versions and runtime."),
        ("Benchmarking standards", "Use the standard splits of public benchmarks, compare with strong recent baselines re-run under the same compute budget, and report every metric the field expects."),
        ("Statistical comparison", "Run each method several times (3-10 seeds) and report mean and spread or confidence intervals; test differences across seeds or datasets."),
        ("Ablation and sensitivity", "Remove one component at a time and vary key hyperparameters, so each claimed gain is attributed and its robustness shown."),
        ("Leakage and validity", "Keep the test set untouched during development, check for train-test overlap, and state threats to internal and external validity."),
        ("Artefact quality", "Follow a reproducibility checklist (e.g. the NeurIPS checklist or ACM artifact badging) and document how to rerun every table."),
    ),
    "law": (
        ("Authority and currency", "Use primary sources from authoritative databases and confirm each is still good law at the date of writing."),
        ("Scope", "State the jurisdictions, period and case-selection criteria, and justify them against the research question."),
        ("Competing interpretations", "Set out the strongest opposing reading of each key source before arguing for your own."),
        ("Citation standard", "Cite consistently in OSCOLA, the Bluebook or the standard your faculty prescribes."),
        ("Empirical legal research", "If you interview, survey or observe, obtain ethics approval and handle privileged material carefully."),
    ),
    "humanities": (
        ("Source criticism", "Assess each source's provenance, purpose and reliability, and say how gaps in the record shape your claims."),
        ("Corpus description", "Describe the archive, corpus or texts, how they were selected and what lies outside them."),
        ("Triangulation", "Support each central claim with more than one independent source or type of source."),
        ("Interpretive framework", "Name the theoretical lens you use and reflect on how your position shapes the reading."),
        ("Access and rights", "Record archive permissions, copyright and, for oral history, consent and ethics approval."),
    ),
}

# The deliverable that shows a roadmap stage is done, matched on the stage's wording (first match wins).
ROADMAP_DELIVERABLES: Tuple[Tuple[str, str], ...] = (
    (r"protocol|registration", "Registered protocol (OSF, PROSPERO or a trial registry) and approved ethics application"),
    (r"ethics|access", "Ethics approval (or a documented exemption) and confirmed access to participants or data"),
    (r"pilot", "Pilot report and the revised instrument"),
    (r"questionnaire|instrument|interview guide|case selection", "Final instrument or case-selection rationale, with the source of every scale or question"),
    (r"search|screening", "Search log and PRISMA-style flow counts"),
    (r"extraction|appraisal|risk-of-bias", "Completed extraction and quality-appraisal forms"),
    (r"dataset selection|baseline", "Baselines reproduced on the standard splits, matching their published results"),
    (r"implementation|build|prototype|model design|calibration", "Working, version-controlled implementation with tests"),
    (r"recruit|collection|fieldwork|interviews|sources", "Clean dataset or transcripts with a data-management log"),
    (r"experiment|ablation|scenario|sensitivity", "Results tables over several seeds or runs, with ablations and sensitivity checks"),
    (r"transcription|coding", "Codebook and coded transcripts with an audit trail"),
    (r"pooling", "Pooled estimates with heterogeneity and publication-bias analyses"),
    (r"analysis|integration", "Analysis scripts and outputs, and a draft of the results chapter"),
    (r"writing", "Full draft of the chapter for supervisor review"),
)

# What to do with each kind of reviewed study when designing your own (keys: the start of compare()'s alignment).
COMPARISON_ACTIONS: Tuple[Tuple[str, str], ...] = (
    ("Same design", "Reuse or adapt their instruments, datasets or benchmarks and report the same measures, so your results sit directly beside theirs"),
    ("Matches the quantitative strand", "Treat their study as the template for your quantitative strand, and aim your qualitative strand at explaining their results"),
    ("Matches the qualitative strand", "Use their themes or coding frame as the starting point of your qualitative strand, and test how far they generalise in your quantitative strand"),
    ("Shares its", "Borrow their measures or coding frames where they fit your question, and justify the design difference in your methodology chapter"),
    ("Different design", "Treat their findings as propositions to test, and state in your rationale what your design can show that theirs cannot"),
    ("Different: no primary data", "Turn their arguments or frameworks into hypotheses, constructs or interview topics, citing them as the source"),
)


# ---- per-paper analysis -----------------------------------------------------------------------

@dataclass
class StudyDesign:
    author: str  # "Siam et al."
    year: str
    title: str
    design: str  # DESIGN_BY_KEY key
    other_designs: List[str] = field(default_factory=list)  # further designs the paper mentions
    samples: List[Dict] = field(default_factory=list)  # {"count", "unit", "group", "text"}
    collection: List[str] = field(default_factory=list)  # COLLECTION_METHODS keys
    analysis: List[str] = field(default_factory=list)  # ANALYSIS_METHODS keys
    countries: List[str] = field(default_factory=list)
    populations: List[str] = field(default_factory=list)
    stated_gap: str = ""  # the limitation Phase 1 recorded for the paper
    evidence: Dict[str, str] = field(default_factory=dict)  # detail -> the sentence it was read from
    confidence: str = "low"
    classified_by: str = "rules"
    confidence_reason: str = ""  # the evidence behind the confidence rating (design_confidence)

    @property
    def kind(self) -> Design:
        return DESIGN_BY_KEY[self.design]

    @property
    def narrative(self) -> str:
        return f"{self.author} ({self.year})"

    @property
    def cite(self) -> str:
        return f"{self.author}, {self.year}"

    @property
    def sample_text(self) -> str:
        return "; ".join(s["text"] for s in self.samples) or "Not reported"

    def to_dict(self) -> Dict:
        return {
            "citation": self.narrative,
            **asdict(self),
            "design_label": self.kind.label,
            "approach": self.kind.approach,
            "data_collection_methods": [COLLECTION_LABELS[k] for k in self.collection],
            "analysis_methods": [ANALYSIS_LABELS[k] for k in self.analysis],
        }


@lru_cache(maxsize=None)
def _rx(pattern: str) -> re.Pattern:
    return re.compile(pattern)


def _first_match(cues: Sequence[str], sentences: Sequence[str]) -> Optional[str]:
    """The first sentence containing any of the cues (matched against its lower-case form)."""
    for sentence in sentences:
        lower = sentence.lower()
        if any(_rx(cue).search(lower) for cue in cues):
            return sentence
    return None


def _clip(sentence: str, max_words: int = EVIDENCE_MAX_WORDS) -> str:
    words = sentence.split()
    return sentence if len(words) <= max_words else " ".join(words[:max_words]) + " ..."


def _paper_sentences(source: SourcePaper) -> List[str]:
    """Title, abstract, Phase 1 summary and findings. Not the stated gap: it describes what the paper didn't do."""
    paper = source.paper
    parts = [paper.title.rstrip(".") + ".", paper.abstract, source.summary, *source.key_findings]
    return [s for part in parts if part for s in split_sentences(normalize_whitespace(part))]


def _unit_group(unit: str) -> str:
    unit = normalize_whitespace(unit).lower()
    return next(
        (group for group, patterns in UNIT_GROUPS.items() if any(re.fullmatch(p, unit) for p in patterns)),
        "unspecified",
    )


def extract_samples(text: str) -> List[Dict]:
    """Sample sizes stated in the text, e.g. "300 cybersecurity practitioners" or "n = 42"."""
    found: List[Dict] = []
    seen = set()
    for match in SAMPLE_PATTERN.finditer(text):
        raw, middle, unit = match.group(1), match.group(2), match.group(3)
        count = NUMBER_WORDS[raw.lower()] if raw.lower() in NUMBER_WORDS else int(raw.replace(",", ""))
        if "," not in raw and 1900 <= count <= 2100:
            continue  # a year, not a sample
        if (count, unit.lower()) in seen:
            continue
        seen.add((count, unit.lower()))
        found.append({
            "count": count,
            "unit": normalize_whitespace(f"{middle}{unit}").lower(),
            "group": _unit_group(unit),
            "text": normalize_whitespace(match.group(0)),
        })
    for match in N_EQUALS.finditer(text):
        count = int(match.group(1).replace(",", ""))
        if all(s["count"] != count for s in found):
            found.append({"count": count, "unit": "participants or cases", "group": "unspecified", "text": f"n = {count}"})
    return found


def _author(source: SourcePaper) -> str:
    return source.surname + (" et al." if len(source.paper.authors) > 1 else "")


def analyse_paper(source: SourcePaper) -> StudyDesign:
    """Reads one paper's design, sample, data collection, analysis and context from its text."""
    sentences = _paper_sentences(source)
    text = " ".join(sentences)
    study = StudyDesign(
        author=_author(source), year=source.year, title=normalize_whitespace(source.paper.title),
        design=UNCLASSIFIED.key, stated_gap=_clip(normalize_whitespace(source.gap), 45),
    )

    matched: List[Design] = []
    for design in DESIGNS:
        sentence = _first_match(design.cues, sentences)
        if sentence:
            if not matched:
                study.evidence["design"] = _clip(sentence)
            matched.append(design)
    if matched:
        study.design = matched[0].key
        study.other_designs = [d.label for d in matched[1:4]]

    for attribute, methods in (("collection", COLLECTION_METHODS), ("analysis", ANALYSIS_METHODS)):
        for key, label, cues in methods:
            sentence = _first_match(cues, sentences)
            if sentence:
                getattr(study, attribute).append(key)
                study.evidence[f"{attribute}: {label}"] = _clip(sentence)

    study.samples = extract_samples(text)
    study.countries = list(dict.fromkeys(name for pattern, name in COUNTRY_PATTERNS if pattern.search(text)))
    if HUMAN_COLLECTION.intersection(study.collection) or study.design in POPULATION_DESIGNS:
        lower = text.lower()
        study.populations = [label for label, cues in POPULATIONS if any(_rx(c).search(lower) for c in cues)]

    study.confidence, study.confidence_reason = design_confidence(study, matched, sentences, source)
    return study


def design_confidence(
    study: StudyDesign, matched: Sequence[Design], sentences: Sequence[str], source: SourcePaper
) -> Tuple[str, str]:
    """(high/medium/low, why) for a paper's design classification, scored from the evidence for it.

        +2  the abstract or summary names the design (+1 if only the title does)
        +1  two or more of the design's cues occur, so the reading doesn't rest on one phrase
        +1  each: a stated sample, a stated data-collection method, a stated analysis method
        -1  each: a competing design from another approach is also described; a short abstract

    5 or more is high, 3-4 medium, anything less low.
    """
    if not matched:
        return "low", "no design keyword occurs in the title or abstract"
    design = matched[0]
    score, reasons = 0, []
    title = sentences[0] if sentences else ""
    lower = [s.lower() for s in sentences]
    cue_hits = sum(1 for cue in design.cues for s in lower if _rx(cue).search(s))
    if study.evidence.get("design") and _clip(title) != study.evidence["design"]:
        score += 2
        reasons.append("the abstract names the design")
    else:
        score += 1
        reasons.append("only the title names the design")
    if cue_hits >= 2:
        score += 1
        reasons.append(f"{cue_hits} cues for it agree")
    for present, label in ((study.samples, "a sample"), (study.collection, "how data were collected"),
                           (study.analysis, "how data were analysed")):
        if present:
            score += 1
            reasons.append(f"it states {label}")
    rivals = [d for d in matched[1:] if d.approach != design.approach]
    if rivals:
        score -= 1
        reasons.append(f"it also describes {_a(rivals[0].label)}")
    if len(source.paper.abstract or source.summary) < 400:
        score -= 1
        reasons.append("the abstract is short")
    label = "high" if score >= 5 else "medium" if score >= 3 else "low"
    return label, f"score {score}: " + "; ".join(reasons)


def model_design(synthesizer: LLMSynthesizer, source: SourcePaper) -> Optional[str]:
    """Asks the local model to pick one design from the fixed list; None if its answer names none."""
    listing = "\n".join(f"- {d.key}: {d.label}" for d in DESIGNS)
    text = normalize_whitespace(f"{source.paper.title}. {source.paper.abstract or source.summary}")[:1500]
    answer = synthesizer.complete(
        "You classify research papers by their research design. Reply with one design key from the list and nothing else.",
        f"Paper:\n{text}\n\nDesigns:\n{listing}\n\nTask: Which design key best describes how this paper did its research?",
        max_tokens=16,
        temperature=0.0,
    ).lower()
    hits = []
    for design in DESIGNS:
        for name in (design.key, design.label.lower()):
            index = answer.find(name)
            if index >= 0:
                hits.append((index, -len(name), design.key))  # earliest, then longest ("quasi-experimental" over "experiment")
    return min(hits)[2] if hits else None


def classify_unclear(studies: Sequence[StudyDesign], sources: Sequence[SourcePaper], model: str) -> None:
    """Lets the local model classify the papers the rules couldn't; keeps them unclassified if it can't."""
    unclear = [(study, source) for study, source in zip(studies, sources) if study.design == UNCLASSIFIED.key]
    if not unclear:
        return
    logger.info("Asking the model to classify %d paper(s) with no recognisable design...", len(unclear))
    try:
        with LLMSynthesizer(model=model) as synthesizer:
            for study, source in unclear:
                key = model_design(synthesizer, source)
                if key:
                    study.design, study.classified_by, study.confidence = key, "model", "low"
                    study.confidence_reason = "classified by the local model; the abstract names no recognisable design"
                    study.evidence["design"] = "Classified by the local model: the abstract names no recognisable design."
                    logger.info("%s classified by the model as: %s", study.narrative, DESIGN_BY_KEY[key].label)
    except (OllamaConnectionError, requests.RequestException) as exc:
        logger.warning("Could not use the model (%s); %d paper(s) stay unclassified", exc, len(unclear))


# ---- field and recommendation -----------------------------------------------------------------

def field_scores(corpus: Corpus) -> Dict[str, int]:
    """How often each research field's vocabulary occurs in the topic and papers (as gap_research.detect_field)."""
    text = " ".join([corpus.topic] * 3 + [f"{p.paper.title} {p.text}" for p in corpus.papers]).lower()
    return {
        name: sum(len(re.findall(r"\b" + re.escape(cue), text)) for cue in cues)
        for name, cues in FIELD_CUES.items()
    }


def detect_tradition(
    corpus: Corpus, studies: Sequence[StudyDesign], override: Optional[str] = None
) -> Tuple[Tradition, Dict[str, float]]:
    """The methodological tradition to follow, from the papers' designs and (half-weighted) vocabulary."""
    scores = {key: 0.0 for key in TRADITIONS}
    vocabulary = field_scores(corpus)
    total = sum(vocabulary.values())
    for name, count in vocabulary.items():
        if total:
            scores[FIELD_TRADITION[name]] += VOCABULARY_WEIGHT * count / total
    for study in studies:
        families = study.kind.families
        for family in families:
            scores[family] += 1 / len(families) / len(studies)
    scores = {key: round(value, 3) for key, value in scores.items()}
    if override:
        return TRADITIONS[override], scores
    best = max(scores, key=scores.get)
    return TRADITIONS[best if scores[best] > 0 else "social"], scores


@dataclass
class Recommendation:
    design: Design
    basis: str  # "mixed": the papers use quantitative and qualitative designs; "common": the papers' most used; "default": the tradition's standard
    components: Tuple[Design, ...] = ()  # (quantitative, qualitative) strands of a mixed-methods design
    alternatives: Tuple[Design, ...] = ()

    @property
    def keys(self) -> set:
        return {self.design.key, *(c.key for c in self.components)}


def recommend(studies: Sequence[StudyDesign], tradition: Tradition) -> Recommendation:
    empirical = [s for s in studies if s.kind.empirical]
    counts = Counter(s.design for s in empirical)
    by_use = [key for key, _ in counts.most_common()]
    approaches = {s.kind.approach for s in empirical}
    if "mixed-methods" in counts or {QUANT, QUAL} <= approaches:
        design, basis = DESIGN_BY_KEY["mixed-methods"], "mixed"
    elif empirical:
        tied = [key for key in by_use if counts[key] == counts[by_use[0]]]
        design, basis = DESIGN_BY_KEY[next((k for k in tradition.typical if k in tied), tied[0])], "common"
    else:
        design, basis = DESIGN_BY_KEY[tradition.typical[0]], "default"

    components: Tuple[Design, ...] = ()
    if design.key == "mixed-methods":
        def strand(approach: str, fallback: str) -> Design:
            candidates = [*by_use, *tradition.typical]
            return DESIGN_BY_KEY[next((k for k in candidates if DESIGN_BY_KEY[k].approach == approach), fallback)]

        components = (strand(QUANT, "survey"), strand(QUAL, "interview"))
    alternatives = [k for k in dict.fromkeys([*by_use, *tradition.typical]) if k != design.key]
    return Recommendation(design, basis, components, tuple(DESIGN_BY_KEY[k] for k in alternatives[:3]))


def compare(study: StudyDesign, rec: Recommendation) -> Tuple[str, str]:
    """(alignment, what the proposed design adds) for one reviewed study."""
    proposed = rec.design
    if study.design == proposed.key:
        return "Same design", "Directly comparable results; extend them to a new sample, setting or period"
    if study.design in rec.keys:
        quant, qual = rec.components
        if study.design == quant.key:
            return "Matches the quantitative strand", f"Adds {_a(qual.label)} that explains its results"
        return "Matches the qualitative strand", f"Adds {_a(quant.label)} that tests how far its findings generalise"
    if not study.kind.empirical:
        alignment = "Different: no primary data"
    elif study.kind.approach == proposed.approach or (proposed.approach == MIXED and study.kind.approach in (QUANT, QUAL)):
        alignment = f"Shares its {study.kind.approach.lower()} approach"
    else:
        alignment = "Different design"
    return alignment, f"{_cap(GUIDES[study.design].limitation)}; the proposed design {GUIDES[proposed.key].strength}"


# ---- writing helpers --------------------------------------------------------------------------

def _lc(label: str) -> str:
    """Lower-cases a label's first letter for use mid-sentence, leaving acronyms alone."""
    return label if label[:2].isupper() else label[:1].lower() + label[1:]


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def _a(label: str) -> str:
    text = _lc(label)
    return ("an " if text[:1].lower() in "aeiou" else "a ") + text


def _count(n: int, singular: str, plural: str) -> str:
    return f"{SPELLED.get(n, n)} {singular if n == 1 else plural}"


def _cites(studies: Sequence[StudyDesign]) -> str:
    return "; ".join(s.cite for s in studies)


def _cell(text: str) -> str:
    return normalize_whitespace(text).replace("|", "/")


def _by_key(studies: Sequence[StudyDesign], attribute: str) -> Dict[str, List[StudyDesign]]:
    """{method key: the studies using it} for "collection" or "analysis", most used first."""
    groups: Dict[str, List[StudyDesign]] = {}
    for study in studies:
        for key in getattr(study, attribute):
            groups.setdefault(key, []).append(study)
    return dict(sorted(groups.items(), key=lambda item: -len(item[1])))


def _design_counts(studies: Sequence[StudyDesign]) -> str:
    groups: Dict[str, List[StudyDesign]] = {}
    for study in studies:
        groups.setdefault(study.design, []).append(study)
    ordered = sorted(groups.items(), key=lambda item: -len(item[1]))
    return join_items(f"{_lc(DESIGN_BY_KEY[k].label)} ({len(g)}: {_cites(g)})" for k, g in ordered)


def _sample_sentences(studies: Sequence[StudyDesign]) -> List[str]:
    pairs = [(study, sample) for study in studies for sample in study.samples]
    if not pairs:
        return ["None of the abstracts states a sample size, so check the full texts before fixing your own."]
    sentences = ["Stated samples: " + join_items(f"{sample['text']} ({study.cite})" for study, sample in pairs) + "."]
    for group in UNIT_GROUPS:
        counts = [sample["count"] for _, sample in pairs if sample["group"] == group]
        if len(counts) >= 2:
            sentences.append(
                f"Samples of {group} range from {min(counts):,} to {max(counts):,} (median {statistics.median(counts):,g})."
            )
    return sentences


def _guide_items(rec: Recommendation, attribute: str) -> List[Tuple[str, str]]:
    """The recommended design's collection or analysis options; per strand for a mixed-methods design."""
    if not rec.components:
        return list(getattr(GUIDES[rec.design.key], attribute))
    quant, qual = rec.components
    items = [(k, f"Quantitative strand ({_lc(quant.label)}): {_lc(text)}") for k, text in getattr(GUIDES[quant.key], attribute)]
    items += [(k, f"Qualitative strand ({_lc(qual.label)}): {_lc(text)}") for k, text in getattr(GUIDES[qual.key], attribute)]
    if attribute == "analysis":
        items.append(GUIDES["mixed-methods"].analysis[-1])  # integrating the strands
    return items


def _phases(rec: Recommendation) -> Tuple[Tuple[str, str], ...]:
    if not rec.components:
        return GUIDES[rec.design.key].phases
    quant, qual = rec.components
    return (
        ("Research design, instruments and ethics approval", "1–2 months"),
        (f"Quantitative strand: {_lc(quant.label)} data collection", "2–3 months"),
        ("Quantitative analysis", "1 month"),
        (f"Qualitative strand: {_lc(qual.label)} data collection", "2–3 months"),
        ("Qualitative analysis and integration of both strands", "1–2 months"),
        ("Writing up", "1–2 months"),
    )


# ---- the methodology section ------------------------------------------------------------------

@dataclass
class MethodologyInputs:
    topic: str
    review_type: str
    slug: str
    studies: List[StudyDesign]
    tradition: Tradition
    field_name: str  # the field the papers' vocabulary belongs to (gap_research.detect_field)
    field_overridden: bool
    recommendation: Recommendation
    bibliography: str
    tradition_scores: Dict[str, float] = field(default_factory=dict)  # detect_tradition()'s scores, for "Why this field"
    degree: str = DEFAULT_DEGREE  # thesis_standards key


def _field_reasoning(inputs: MethodologyInputs) -> str:
    """Why the methodological tradition was chosen: the designs pointing to it, the vocabulary, the scores."""
    studies, tradition = inputs.studies, inputs.tradition
    typical = [s for s in studies if tradition.key in s.kind.families]
    parts = []
    if inputs.field_overridden:
        return f"**Why this field.** {tradition.name} was chosen with --field; the detected scores are kept in the analysis file."
    if typical:
        labels = join_items(sorted({_lc(s.kind.label) for s in typical}))
        parts.append(f"{_count(len(typical), 'of the studies uses', 'of the studies use')} designs typical of "
                     f"{tradition.name.lower()} ({labels})")
    if inputs.field_name:
        parts.append(f"their vocabulary is closest to {inputs.field_name}")
    reasoning = "**Why this field.** The field was inferred from two kinds of evidence: " + ("; ".join(parts) or "the topic") + "."
    if inputs.tradition_scores:
        ranked = sorted(inputs.tradition_scores.items(), key=lambda kv: -kv[1])[:3]
        reasoning += (" Weighting designs fully and vocabulary by half, the scores are "
                      + join_items(f"{TRADITIONS[k].name.lower()} {v:.2f}" for k, v in ranked) + ".")
        if len(ranked) > 1 and ranked[0][1] - ranked[1][1] < 0.1:
            reasoning += (f" The margin over {TRADITIONS[ranked[1][0]].name.lower()} is small, so check its conventions "
                          "too, or set the field explicitly with --field.")
    return reasoning


def _roadmap(rec: Recommendation) -> List[Tuple[str, str, str]]:
    """(stage, duration, deliverable that shows it is done) for the recommended design."""
    rows = []
    for stage, duration in _phases(rec):
        lower = stage.lower()
        deliverable = next((d for pattern, d in ROADMAP_DELIVERABLES if re.search(pattern, lower)), "Stage summary signed off by your supervisor")
        rows.append((stage, duration, deliverable))
    return rows


def _comparison_actions(comparisons: Sequence[Tuple[StudyDesign, str, str]]) -> List[str]:
    """One action per kind of alignment, naming the studies it applies to."""
    lines = []
    for prefix, action in COMPARISON_ACTIONS:
        group = [s for s, alignment, _ in comparisons if alignment.startswith(prefix)]
        if group:
            lines.append(f"- **{prefix.rstrip(':').replace('Shares its', 'Same approach')}** ({_cites(group)}): {action}.")
    return lines


def build_methodology(inputs: MethodologyInputs) -> str:
    topic, studies, rec, tradition = inputs.topic, inputs.studies, inputs.recommendation, inputs.tradition
    n = len(studies)
    label, guide = rec.design.label, GUIDES[rec.design.key]
    lines = [
        f"# Methodology: {topic}",
        "",
        f"*Proposed research methodology for a study on {topic}, based on the designs of the {_count(n, 'study', 'studies')} in the {inputs.review_type}*",
        "",
        f"> **How to complete this methodology.** It proposes a design for your own study from the designs of the "
        f"reviewed studies. The decisions only you can make are marked: **{markers.METHODOLOGY_CHOICE}** (your design "
        f"and research question), **{markers.YOUR_SAMPLE}**, **{markers.YOUR_DATA_SOURCES}**, "
        f"**{markers.YOUR_ANALYSIS_PLAN}** and **{markers.YOUR_TIMELINE}**. Replace each marker with your own "
        "decisions, and keep, adapt or delete the suggestions around it. " + markers.guidance_legend(),
        "",
    ]

    # 1. Research approach
    intro = (
        f"This section proposes a methodology for a study on {topic}. It is grounded in the research designs of the "
        f"{_count(n, 'study', 'studies')} included in the {inputs.review_type}: {_design_counts(studies)}."
    )
    vocabulary = FIELD_TRADITION.get(inputs.field_name)
    if inputs.field_overridden:
        intro += f" The recommendations follow the methodological conventions of {tradition.name.lower()}, as chosen with --field."
    elif vocabulary and vocabulary != tradition.key:
        intro += (
            f" The topic's vocabulary places it in {inputs.field_name}, but the studies investigate it with designs typical "
            f"of {tradition.name.lower()}, so the recommendations follow that tradition's methodological conventions."
        )
    else:
        intro += (
            f" The topic and the studies' designs place this research in {tradition.name.lower()}, so the "
            "recommendations follow that field's methodological conventions."
        )
    intro += f" On this basis, the proposed study takes a {rec.design.approach.lower()} approach: {_a(label)}."
    lines += ["## 1. Research Approach", "", intro, "", _field_reasoning(inputs), ""]

    # 2. Patterns in the reviewed studies
    lines += [
        "## 2. Methodological Patterns in the Reviewed Studies",
        "",
        "| Study | Design | Approach | Sample | Data collection | Analysis | Design confidence |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in studies:
        design_cell = s.kind.label + (" (classified by the model)" if s.classified_by == "model" else "")
        collection = "; ".join(COLLECTION_LABELS[k] for k in s.collection) or "Not stated"
        analysis = "; ".join(ANALYSIS_LABELS[k] for k in s.analysis) or "Not stated"
        lines.append(
            f"| {_cell(s.narrative)} | {_cell(design_cell)} | {s.kind.approach} | {_cell(s.sample_text)} | "
            f"{_cell(collection)} | {_cell(analysis)} | {_cap(s.confidence)} |"
        )
    lines.append("")
    unsure = [s for s in studies if s.confidence != "high" and s.confidence_reason]
    if unsure:
        lines += [
            "**Why some classifications are less certain.** Design confidence is scored from the evidence in each "
            "abstract (the design named, agreeing cues, a stated sample, collection and analysis method). "
            + " ".join(f"{s.narrative}: {s.confidence} ({s.confidence_reason})." for s in unsure),
            "",
        ]

    counts = Counter(s.design for s in studies)
    top_key, top_n = counts.most_common(1)[0]
    if top_n > 1 or n == 1:
        users = [s for s in studies if s.design == top_key]
        patterns = [f"The most common design is {_a(DESIGN_BY_KEY[top_key].label)} ({top_n} of {n}: {_cites(users)})."]
    else:
        patterns = ["No single design dominates: each study uses a different design, so the topic has been approached from several methodological angles."]
    approaches = Counter(s.kind.approach for s in studies)
    patterns.append("By approach, the studies are " + join_items(f"{a.lower()} ({c})" for a, c in approaches.most_common()) + ".")
    for s in studies:
        if s.other_designs:
            patterns.append(f"{s.narrative} also draws on {join_items(_a(d) for d in s.other_designs)}.")
    lines += [" ".join(patterns), ""]
    lines += [" ".join(_sample_sentences(studies)), ""]

    detail = []
    for attribute, labels, verb in (("collection", COLLECTION_LABELS, "Data were collected through"),
                                    ("analysis", ANALYSIS_LABELS, "Data were analysed with")):
        groups = _by_key(studies, attribute)
        if groups:
            detail.append(f"{verb} " + join_items(f"{_lc(labels[k])} ({_cites(g)})" for k, g in groups.items()) + ".")
        silent = [s.narrative for s in studies if not getattr(s, attribute)]
        if silent:
            what = "how data were collected" if attribute == "collection" else "how data were analysed"
            detail.append(f"{join_items(silent)} {'does' if len(silent) == 1 else 'do'} not state {what}.")
    countries = list(dict.fromkeys(c for s in studies for c in s.countries))
    populations = list(dict.fromkeys(p for s in studies for p in s.populations))
    if countries or populations:
        context = "The studies focus on " + (join_items(populations) if populations else "settings")
        detail.append(context + (f" in {join_items(countries)}" if countries else "") + ".")
    else:
        detail.append("The abstracts do not state the populations or countries studied.")
    lines += [" ".join(detail), ""]
    lines += [
        markers.guidance(
            markers.USER_SHOULD_VERIFY,
            "Designs, samples and methods were identified automatically from the titles, abstracts and Phase 1 "
            f"summaries; the sentence each was read from is listed in {inputs.slug}-methodology-analysis.json. Check "
            "them against the full texts, starting with the low-confidence classifications.",
        ),
        "",
    ]

    # 3. Proposed design
    lines += ["## 3. Proposed Research Design", "", f"**Recommended design: {label}** ({rec.design.approach.lower()}). {guide.summary}", ""]
    empirical = [s for s in studies if s.kind.empirical]
    if rec.components:
        quant, qual = rec.components
        lines += [
            f"Following the reviewed studies, the two strands would be {_a(quant.label)} (quantitative) followed by "
            f"{_a(qual.label)} (qualitative), in an explanatory sequential design: the quantitative strand shows how widespread "
            "the patterns are, and the qualitative strand explains them. A convergent design, which runs both strands in "
            "parallel and compares them, is the alternative when time is short.",
            "",
        ]
        quant_users = [s for s in empirical if s.kind.approach == QUANT]
        qual_users = [s for s in empirical if s.kind.approach == QUAL]
        mixed_users = [s for s in empirical if s.design == "mixed-methods"]
        reasons = []
        if quant_users and qual_users:
            reasons.append(f"the reviewed studies use both quantitative ({_cites(quant_users)}) and qualitative designs ({_cites(qual_users)})")
        if mixed_users:
            reasons.append(f"{join_items(s.narrative for s in mixed_users)} already combine{'s' if len(mixed_users) == 1 else ''} the two")
        basis = (
            "This design is recommended because " + " and ".join(reasons) + ". Combining the strands lets your study "
            "measure how widespread a pattern is and explain why it occurs."
        )
    elif rec.basis == "common":
        users = [s for s in studies if s.design == rec.design.key]
        if len(users) == 1 and n > 1:
            basis = (
                f"The reviewed studies each use a different design; this one ({users[0].cite}) is recommended because it is "
                f"the most established empirical design in {tradition.name.lower()} among those they use."
            )
        else:
            basis = (
                f"This design is recommended because it is the one most used by the reviewed studies ({len(users)} of {n}: "
                f"{_cites(users)}), so your results can be compared directly with theirs."
            )
    else:
        used = join_items(sorted({_lc(s.kind.label) for s in studies}))
        basis = (
            f"None of the reviewed studies collected primary data (they are: {used}), so {_a(label)}, the standard "
            f"empirical design in {tradition.name.lower()}, is proposed to test their claims with new evidence."
        )
    lines += [basis, ""]
    if rec.alternatives:
        lines += ["Alternative designs to consider:", ""]
        lines += [f"- **{d.label}**: {GUIDES[d.key].summary}" for d in rec.alternatives]
        lines.append("")
    lines += [
        markers.marker(
            markers.METHODOLOGY_CHOICE,
            f"State your research question on {topic} (and hypotheses, if your design is quantitative), then the design "
            f"you will use, {_a(label)} or one of the alternatives, and justify the choice against your question.",
        ),
        "",
    ]

    # 4. Rationale
    comparisons = [(s, *compare(s, rec)) for s in studies]
    aligned = [s for s, alignment, _ in comparisons if alignment in ("Same design",) or alignment.startswith("Matches")]
    sharing = [s for s, alignment, _ in comparisons if alignment.startswith("Shares")]
    no_data = [s for s, alignment, _ in comparisons if alignment == "Different: no primary data"]
    fit = [f"{_count(len(aligned), 'study uses', 'studies use') if aligned else 'none uses'} the proposed design or one of its strands"]
    if sharing:
        fit.append(f"{_count(len(sharing), 'shares', 'share')} its {rec.design.approach.lower()} approach")
    if no_data:
        fit.append(f"{_count(len(no_data), 'collected', 'collected')} no primary data, which the proposed design supplies")
    rationale = [f"- **Fit in numbers.** Of the {_count(n, 'reviewed study', 'reviewed studies')}, " + join_items(fit) + "."]
    if aligned:
        if n == 1:
            who = "The reviewed study uses"
        elif len(aligned) == n:
            who = f"All {SPELLED.get(n, n)} studies use"
        else:
            who = f"{_cap(str(SPELLED.get(len(aligned), len(aligned))))} of the {SPELLED.get(n, n)} studies {'uses' if len(aligned) == 1 else 'use'}"
        rationale.append(
            f"- **Fit with the reviewed studies.** {who} the proposed design or one of its strands ({_cites(aligned)}), "
            "so your study builds on established practice in this literature and its results can be compared with theirs."
        )
    else:
        rationale.append(
            "- **Fit with the reviewed studies.** None of the studies uses this design, so your study would add a "
            "methodological perspective the literature lacks."
        )
    differing = [(s, adds) for s, alignment, adds in comparisons if alignment not in ("Same design",) and not alignment.startswith("Matches")]
    if differing:
        weaknesses = " ".join(f"{s.narrative} uses {_a(s.kind.label)}, which {GUIDES[s.design].limitation}." for s, _ in differing)
        rationale.append(f"- **Methodological gaps it addresses.** {weaknesses} The proposed design {guide.strength}.")
    rationale.append(
        f"- **Fit with the field.** {_cap(_a(label))} is an established design in {tradition.name.lower()}, with "
        "recognised quality criteria and reporting standards (Section 8)."
    )
    gaps = [s for s in studies if s.stated_gap]
    if gaps:
        rationale.append(
            "- **What the studies leave open.** Phase 1 noted these limitations, which your design can target: "
            + " ".join(f"{s.narrative}: \"{s.stated_gap.rstrip('.')}.\"" for s in gaps)
        )
    options = [text for _, text in _guide_items(rec, "collection")][:2]
    if options:
        rationale.append(
            f"- **Feasibility.** The design depends on access to the data it needs ({'; '.join(_lc(o) for o in options)}); "
            "confirm that access before committing to it."
        )
    lines += ["## 4. Design Rationale", "", *rationale, ""]

    # 5. Population and sample
    lines += ["## 5. Population and Sample", ""]
    if populations or countries:
        studied = join_items(populations) if populations else "the settings described"
        lines += [
            f"The reviewed studies focus on {studied}{f' in {join_items(countries)}' if countries else ''}. Define your "
            "target population in comparable terms (who, where and when) so your findings can be set against theirs.",
            "",
        ]
    else:
        lines += ["The abstracts do not describe the populations studied; define your target population (who, where and when) explicitly.", ""]
    lines += ["**From the reviewed studies.** " + " ".join(_sample_sentences(studies)), ""]
    if rec.components:
        quant, qual = rec.components
        lines += [
            f"**Convention for {_a(label)}.** {guide.sample}",
            "",
            f"- Quantitative strand ({_lc(quant.label)}): {GUIDES[quant.key].sample}",
            f"- Qualitative strand ({_lc(qual.label)}): {GUIDES[qual.key].sample}",
            "",
        ]
    else:
        lines += [f"**Convention for {_a(label)}.** {guide.sample}", ""]
    lines += [
        markers.marker(
            markers.YOUR_SAMPLE,
            "Give your target sample size and sampling strategy (e.g. random, stratified, purposive or snowball), your "
            "inclusion and exclusion criteria, and how you arrived at the number: a power analysis, saturation, or "
            "precedent in the studies above.",
        ),
        "",
    ]

    # 6. Data collection and 7. analysis
    for number, title, attribute, labels, tag, prompt in (
        (6, "Data Collection", "collection", COLLECTION_LABELS, markers.YOUR_DATA_SOURCES,
         "Name your actual data sources: who or what you will collect data from, the instruments (questionnaire, "
         "interview guide, dataset, archive), how you will get access, and whether you will adapt an instrument used "
         "by one of the studies above."),
        (7, "Data Analysis", "analysis", ANALYSIS_LABELS, markers.YOUR_ANALYSIS_PLAN,
         "Describe your analysis plan step by step: the variables or themes, the specific tests or coding procedure, "
         "the software (e.g. SPSS, R, Python or NVivo), and how each step answers your research question."),
    ):
        groups = _by_key(studies, attribute)
        lines += [f"## {number}. {title}", "", "**Following the reviewed studies:**", ""]
        if groups:
            lines += [f"- {labels[k]} ({_cites(g)})" for k, g in groups.items()]
        else:
            lines.append("- The abstracts do not state this; check the full texts.")
        lines += ["", f"**Options for {_a(label)}:**", ""]
        for key, text in _guide_items(rec, attribute):
            used = f" (as in {_cites(groups[key])})" if key in groups else ""
            lines.append(f"- {text}{used}")
        lines += ["", markers.marker(tag, prompt), ""]

    # 8. Validity, reliability and ethics
    ethics = tradition.ethics
    if vocabulary and vocabulary != tradition.key:
        ethics += f" As the topic also concerns {inputs.field_name}, check that field's norms too: {TRADITIONS[vocabulary].ethics}"
    lines += [
        "## 8. Validity, Reliability and Ethics",
        "",
        f"- **Quality criteria.** In {tradition.name.lower()}, studies like this are judged on {tradition.quality}. "
        f"The main threat for {_a(label)} is that it {guide.limitation}; explain how your design limits it.",
        f"- **Reporting.** Follow {tradition.reporting}.",
        f"- **Ethics.** {ethics}",
        "",
    ]

    # 9. Timeline and resources
    lines += ["## 9. Timeline and Resources", "", "| Stage | Indicative duration |", "|---|---|"]
    lines += [f"| {stage} | {duration} |" for stage, duration in _phases(rec)]
    lines += [
        "",
        "*Durations are indicative for a single researcher (e.g. a master's or doctoral project) and depend on access to data and approvals.*",
        "",
        markers.marker(
            markers.YOUR_TIMELINE,
            "Set your own schedule and list the resources you need: access to participants or data, software, "
            "equipment, funding and supervision, and any approvals that could delay data collection.",
        ),
        "",
    ]

    # 10. Comparison
    lines += [
        "## 10. Comparison with the Reviewed Studies",
        "",
        "| Study | Their design | Approach | Proposed design | Alignment | What the proposed design adds |",
        "|---|---|---|---|---|---|",
    ]
    for s, alignment, adds in comparisons:
        lines.append(
            f"| {_cell(s.narrative)} | {_cell(s.kind.label)} | {s.kind.approach} | {_cell(label)} | {_cell(alignment)} | {_cell(adds)} |"
        )
    lines += ["", "**What to do with this comparison.**", "", *_comparison_actions(comparisons), ""]

    # 11. Field-specific best practices
    lines += ["## 11. Field-Specific Best Practices", "",
              f"Examiners in {tradition.name.lower()} check these practices first; build each into your design from the start.", ""]
    lines += [f"- **{practice}.** {how}" for practice, how in FIELD_BEST_PRACTICES[tradition.key]]
    if vocabulary and vocabulary != tradition.key and not inputs.field_overridden:
        practice, how = FIELD_BEST_PRACTICES[vocabulary][0]
        lines.append(f"- **{practice} ({TRADITIONS[vocabulary].name.lower()}).** {how}")
    lines.append("")

    # 12. Implementation roadmap
    lines += [
        "## 12. Implementation Roadmap",
        "",
        f"The stages below turn {_a(label)} into a plan; each ends with a deliverable that shows it is done.",
        "",
        "| Step | Stage | Indicative duration | Done when you have |",
        "|---|---|---|---|",
    ]
    lines += [f"| {i} | {_cell(stage)} | {duration} | {_cell(done)} |" for i, (stage, duration, done) in enumerate(_roadmap(rec), 1)]
    first_access = next((text for _, text in _guide_items(rec, "collection")), "")
    lines += [
        "",
        "**First three actions.**",
        "",
        f"1. Write your research question{' and hypotheses' if rec.design.approach in (QUANT, MIXED) else ''} in one "
        "paragraph and check it against the gaps in the review.",
        "2. Confirm access to your data" + (f": {_lc(first_access)}" if first_access else "") + ".",
        "3. Draft the ethics application or protocol, using Sections 5 to 8 of this chapter as its methods section.",
        "",
    ]

    # 13. Degree standards
    standard = degree_standard(inputs.degree)
    lines += [
        "## 13. Standards for Your Degree",
        "",
        degree_paragraph(inputs.degree),
        "",
        markers.guidance(
            markers.USER_MUST_ADD,
            f"Scope the study to {standard.name} level: state in your research question what is original about it "
            f"({standard.originality.split(';')[0]}).",
        ),
        "",
        markers.guidance(
            markers.USER_SHOULD_VERIFY,
            "Phase 2 recognises designs from keywords in the papers' titles and abstracts, so a design described only "
            "in a full text is missed; sample sizes are those stated in the abstracts. The field was detected from the "
            "papers (override it with --field). Check every classification before relying on it.",
        ),
        "",
        "## References",
        "",
        inputs.bibliography,
        "",
    ]
    return "\n".join(lines)


def analysis_json(inputs: MethodologyInputs, source_path: Path, scores: Dict[str, float]) -> Dict:
    rec = inputs.recommendation
    studies = inputs.studies
    return {
        "version": 1,
        "topic": inputs.topic,
        "review": str(source_path),
        "review_type": inputs.review_type,
        "papers_analysed": len(studies),
        "field": {
            "vocabulary_field": inputs.field_name,
            "tradition": inputs.tradition.key,
            "tradition_name": inputs.tradition.name,
            "tradition_scores": scores,
            "overridden": inputs.field_overridden,
        },
        "studies": [s.to_dict() for s in studies],
        "patterns": {
            "designs": dict(Counter(s.kind.label for s in studies).most_common()),
            "approaches": dict(Counter(s.kind.approach for s in studies).most_common()),
            "data_collection": {COLLECTION_LABELS[k]: len(g) for k, g in _by_key(studies, "collection").items()},
            "analysis": {ANALYSIS_LABELS[k]: len(g) for k, g in _by_key(studies, "analysis").items()},
            "samples": [{"study": s.narrative, **sample} for s in studies for sample in s.samples],
            "countries": dict(Counter(c for s in studies for c in s.countries).most_common()),
            "populations": dict(Counter(p for s in studies for p in s.populations).most_common()),
        },
        "recommendation": {
            "design": rec.design.key,
            "label": rec.design.label,
            "approach": rec.design.approach,
            "basis": rec.basis,
            "strands": [{"approach": d.approach, "design": d.key, "label": d.label} for d in rec.components],
            "alternatives": [d.label for d in rec.alternatives],
        },
        "comparison": [
            {"study": s.narrative, "their_design": s.kind.label, "proposed_design": rec.design.label,
             "alignment": alignment, "proposed_design_adds": adds}
            for s in studies
            for alignment, adds in [compare(s, rec)]
        ],
    }


# ---- running Phase 2 ----------------------------------------------------------------------------

def resolve_review(path: Path) -> Tuple[ReviewDocument, Path]:
    """The review to read papers from, and the Phase 1 review path its outputs are named after.

    Accepts the Phase 1 review, its .researchflow.json paper file, or a Phase 1.5/1.6 review
    (<review>-complete.md, <review>-final.md), whose papers are in the Phase 1 review's file.
    """
    if path.name.endswith(SIDECAR_SUFFIX):
        base = path.with_name(path.name[: -len(SIDECAR_SUFFIX)] + ".md")
        if base.is_file():
            return parse_review(base), base
        data = json.loads(path.read_text(encoding="utf-8"))
        topic = data.get("topic") or base.stem
        try:
            review_type = get_methodology(data.get("methodology") or "narrative").name.lower()
        except ValueError:
            review_type = "literature review"
        return ReviewDocument(base, [], topic, topic, review_type, [], None), base
    review = parse_review(path)
    if not sidecar_path(path).exists():
        for suffix in DERIVED_SUFFIXES:
            if path.stem.endswith(suffix):
                base = path.with_name(path.stem[: -len(suffix)] + ".md")
                if sidecar_path(base).exists():
                    return replace(review, path=base), base
    return review, path


def log_citation_check(markdown: str, corpus: Corpus, phase: str, extra_known: Sequence[Tuple[str, str]] = ()) -> Dict:
    """Verifies the phase's in-text citations against the reviewed papers (and `extra_known`
    methodological works that have their own reference entries) and logs what doesn't match."""
    from .review_assembly import citation_audit  # imported here: review_assembly imports the gap modules

    result = citation_audit(markdown, [(p.surname, p.year) for p in corpus.papers], extra_known)
    if result["unresolved"]:
        logger.warning("%s cites papers that are not in the review: %s", phase, ", ".join(result["unresolved"]))
    else:
        logger.info("%s citations: every citation matches a reviewed paper", phase)
    logger.info("%s: %.0f%% of prose sentences cite a reviewed paper", phase, 100 * result["cited_sentence_share"])
    return result


def run_phase2(args) -> List[Path]:
    load_dotenv()
    input_path = Path(args.input)
    if not input_path.is_file():
        raise ValueError(f"Review file not found: {input_path}")
    review, base = resolve_review(input_path)
    corpus = load_corpus(review)
    if not corpus.papers:
        raise ValueError(
            f"No papers found for {input_path}: Phase 2 needs the review's {SIDECAR_SUFFIX} file or its References section"
        )
    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = base.stem
    formats = [f.strip().lower() for f in args.formats.split(",") if f.strip()]
    style = (args.citation_style or corpus.citation_style or "apa").lower()
    topic = corpus.topic or review.topic

    logger.info("Phase 2: analysing the research designs of %d papers on %r", len(corpus.papers), topic)
    studies = [analyse_paper(source) for source in corpus.papers]
    if not args.no_llm:
        classify_unclear(studies, corpus.papers, args.model)
    for study in studies:
        logger.info(
            "%s: %s (%s confidence)%s", study.narrative, study.kind.label, study.confidence,
            f"; sample: {study.sample_text}" if study.samples else "",
        )

    tradition, scores = detect_tradition(corpus, studies, getattr(args, "field", None))
    rec = recommend(studies, tradition)
    logger.info("Field: %s (vocabulary: %s). Recommended design: %s", tradition.name, corpus.field, rec.design.label)

    inputs = MethodologyInputs(
        topic=topic,
        review_type=review.review_type,
        slug=slug,
        studies=studies,
        tradition=tradition,
        field_name=corpus.field,
        field_overridden=bool(getattr(args, "field", None)),
        recommendation=rec,
        bibliography=CitationFormatter().format_bibliography([p.paper for p in corpus.papers], style=style),
        tradition_scores=scores,
        degree=getattr(args, "degree", None) or DEFAULT_DEGREE,
    )
    markdown = build_methodology(inputs)
    written = Exporter(output_dir=output_dir).export_all(markdown, f"{slug}-methodology", formats)
    analysis_path = output_dir / f"{slug}-methodology-analysis.json"
    data = analysis_json(inputs, input_path, scores)
    data["citation_check"] = log_citation_check(markdown, corpus, "Phase 2")
    analysis_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    written.append(analysis_path)

    placed = sum(1 for line in markdown.splitlines() if markers.METHODOLOGY_MARKER_LINE.match(line))
    for path in written:
        logger.info("Written: %s", path)
    logger.info("%d decisions are marked for you to make (e.g. %s).", placed, markers.METHODOLOGY_CHOICE)
    return written
