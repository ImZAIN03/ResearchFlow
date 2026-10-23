"""What examiners expect of a thesis at each degree level, and how ready a generated draft is for it.

ResearchFlow writes the same evidence base for every user, but an MS thesis, an MPhil and a PhD are
judged against different standards of scope, depth and originality. DEGREE_STANDARDS states those
standards; Phase 2 sets the proposed study against them and the executive summary (export.py) reports
the draft's readiness for the chosen degree (--degree):

    readiness target   the share of an approval-ready chapter ResearchFlow aims to write; the rest is
                       the user's own data, analysis and argument, which grows with the degree's
                       demand for original contribution
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

DEFAULT_DEGREE = "ms"


@dataclass(frozen=True)
class DegreeStandard:
    key: str
    name: str
    readiness: int  # % of an approval-ready draft ResearchFlow targets at this level
    papers: str  # typical size of the literature base
    originality: str  # what counts as the thesis's contribution
    methodology: str  # depth of methodological justification expected
    analysis: str  # what the results chapter must show
    length: str  # typical length of the thesis
    examiner_focus: Tuple[str, ...]  # what examiners check first


DEGREE_STANDARDS: Dict[str, DegreeStandard] = {
    d.key: d
    for d in (
        DegreeStandard(
            "ms", "Master's (MS/MSc/MA)", 92,
            "15-40 sources, centred on the last 5-10 years",
            "a competent application or extension of established methods to a defined problem; replication in a "
            "new setting is a legitimate contribution",
            "a justified design with a clear link from research question to data and analysis; alternatives named",
            "correct, transparent analysis of one dataset or case set, with results compared against the literature",
            "15,000-30,000 words",
            ("a focused research question", "a coherent literature review", "a sound, justified method",
             "correct analysis", "conclusions that follow from the results"),
        ),
        DegreeStandard(
            "mphil", "Master of Philosophy (MPhil)", 90,
            "40-80 sources, including the field's foundational works",
            "an original contribution of limited scope: a new dataset, a new combination of methods or a substantive "
            "re-interpretation of existing evidence",
            "a design justified against competing designs, with threats to validity analysed and mitigated",
            "a robust analysis with sensitivity or robustness checks and an explicit account of its limits",
            "40,000-60,000 words",
            ("originality of the contribution", "critical rather than descriptive engagement with the literature",
             "methodological justification", "robustness of the analysis", "honest limitations"),
        ),
        DegreeStandard(
            "phd", "Doctorate (PhD/DPhil)", 88,
            "80-200+ sources, covering the field's history, debates and the current frontier",
            "a significant, original contribution to knowledge that others can build on, typically publishable as "
            "two to four peer-reviewed papers",
            "a design defended against alternatives at the level of epistemology, with full validity, reliability "
            "and ethics reasoning and a pre-specified analysis plan",
            "a multi-study or multi-method programme whose results are triangulated, stress-tested and placed in theory",
            "70,000-100,000 words (or a thesis by publication)",
            ("significance and originality of the contribution", "mastery of the field's literature and debates",
             "theoretical framing", "methodological rigour across studies", "publishability"),
        ),
    )
}
DEGREE_CHOICES = tuple(DEGREE_STANDARDS)

# Readiness each phase targets before the user's own work (the request's quality targets).
PHASE_READINESS = {"Phase 1 (literature review)": 90, "Phase 2 (methodology)": 95,
                   "Phase 3 (results and analysis plan)": 90, "Phase 4 (conclusion)": 95}


def degree_standard(key: str = "") -> DegreeStandard:
    return DEGREE_STANDARDS.get((key or DEFAULT_DEGREE).lower(), DEGREE_STANDARDS[DEFAULT_DEGREE])


def degree_paragraph(key: str = "") -> str:
    """What the chosen degree expects, as one paragraph for a methodology or summary."""
    d = degree_standard(key)
    return (
        f"For {d.name} level, examiners expect a literature base of {d.papers}; a contribution that is {d.originality}; "
        f"{d.methodology}; and {d.analysis}. Typical length: {d.length}. Examiners look first at "
        + ", ".join(d.examiner_focus[:-1]) + f" and {d.examiner_focus[-1]}."
    )
