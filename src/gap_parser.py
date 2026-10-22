"""Phase 1.5, step 1: finds the gap markers in a Phase 1 review and works out what each one asks for.

Each marker placed by critical_gaps (a line starting with **[TAG]**) becomes a Gap. A gap records the
marker's prompt, where in the review it sits, the paragraph it responds to, the papers it names, and
its category, which decides the hints it gets and what critical thinking the user's answer must show:

    methodological      [ADD YOUR CRITIQUE]: limitations of the studies' (or the review's own) methods
    evidence            [USER ANALYSIS NEEDED]: contradicting findings, or data that must be pooled
    theoretical         [YOUR INSIGHT]: why a research gap matters, what the review implies
    future direction    [YOUR OBSERVATION]: where research should go next

Nothing here depends on the field or topic: everything is read from the review itself.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from . import critical_gaps as markers
from .llm_synthesis import THEME_STOPWORDS, normalize_whitespace

METHODOLOGICAL = "methodological"
EVIDENCE = "evidence"
THEORETICAL = "theoretical"
FUTURE = "future direction"

# In-text citations as the review writes them: "Furukawa (2026)", "(Furukawa, 2026; Wang et al., 2020)".
NARRATIVE_CITATION = re.compile(r"\b([A-Z][\w'’-]+)(?: et al\.?)? \((\d{4}|n\.d\.)\)")
PARENTHETICAL_CITATION = re.compile(r"\b([A-Z][\w'’-]+)(?: et al\.?)?, (\d{4}|n\.d\.)(?!\w)")
TOPIC_LINE = re.compile(r"^\*A (.+?) of research on: (.+)\*$")
QUOTED = re.compile(r"[\"“]([^\"”]{3,120})[\"”]")
KEYWORD_STOPWORDS = THEME_STOPWORDS | set(
    "matter matters review reviewed question questions research analysis explain whether which would your "
    "their there these papers paper studies study evidence close future given noted limitations limit "
    "confidence conclusions address assess discuss findings directions direction expect prioritise prioritize "
    "methods populations hints moving state sentences implies shapes".split()
)


@dataclass
class Gap:
    id: int
    tag: str
    prompt: str
    category: str
    kind: str  # finer type: contradiction, data-synthesis, study-methods, review-method, gap-significance, implication, trend, priorities
    section: str
    subsection: str
    context: str  # the paragraph the marker responds to
    line_index: int  # index of the marker line in ReviewDocument.lines
    focus: str  # what the gap is about, in a few words
    cited: List[Tuple[str, str]] = field(default_factory=list)  # (surname, year) named by the gap
    keywords: List[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return f"Gap {self.id}"

    @property
    def location(self) -> str:
        return " > ".join(filter(None, [self.section, self.subsection]))

    @property
    def word_range(self) -> Tuple[int, int]:
        """Suggested length of the user's answer."""
        if self.kind == "implication":
            return 60, 120  # the prompt asks for one or two sentences
        return 150, 200

    @property
    def required_thinking(self) -> Tuple[str, ...]:
        """Critical-thinking moves an answer to this gap must make (see gap_input.CRITICAL_MOVES)."""
        return {
            "contradiction": ("comparison", "critique"),
            "data-synthesis": ("critique", "reasoning"),
            "study-methods": ("critique", "suggestion"),
            "review-method": ("critique", "suggestion"),
            "gap-significance": ("reasoning", "suggestion"),
            "implication": ("reasoning",),
            "trend": ("reasoning", "suggestion"),
            "priorities": ("suggestion", "reasoning"),
        }.get(self.kind, ("critique",))


@dataclass
class ReviewDocument:
    path: Path
    lines: List[str]
    title: str
    topic: str
    review_type: str  # "systematic review", "narrative review", ...
    gaps: List[Gap]
    references_start: Optional[int]  # index of the "## References" heading, if any

    @property
    def references(self) -> List[str]:
        if self.references_start is None:
            return []
        entries, current = [], []
        for line in self.lines[self.references_start + 1 :]:
            if line.startswith("## "):
                break
            if line.strip():
                current.append(line.strip())
            elif current:
                entries.append(" ".join(current))
                current = []
        if current:
            entries.append(" ".join(current))
        return entries

    @property
    def body_text(self) -> str:
        """The review without its references, marker lines and completion guide."""
        end = self.references_start if self.references_start is not None else len(self.lines)
        return "\n".join(
            line for line in self.lines[:end]
            if not markers.MARKER_LINE.match(line) and not line.startswith("> **How to complete")
        )


def citations_in(text: str) -> List[Tuple[str, str]]:
    """(surname, year) pairs cited in text, in order of first appearance."""
    found = NARRATIVE_CITATION.findall(text or "") + PARENTHETICAL_CITATION.findall(text or "")
    order = {pair: (text or "").find(pair[0]) for pair in found}
    return sorted(dict.fromkeys(found), key=lambda pair: order[pair])


def keywords_of(text: str, exclude: Sequence[str] = (), limit: int = 8) -> List[str]:
    """Most frequent content words of text (5+ letters), skipping cited surnames and generic words."""
    excluded = {w.lower() for w in exclude}
    counts = Counter(
        w for w in re.findall(r"[a-z][a-z-]{4,}", (text or "").lower())
        if w not in KEYWORD_STOPWORDS and w not in excluded
    )
    return [w for w, _ in counts.most_common(limit)]


def _plain(text: str) -> str:
    """Markdown-free text: no bold/italic markers, links or images."""
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    return normalize_whitespace(text.replace("**", "").replace("*", " "))


def classify(tag: str, prompt: str, section: str) -> Tuple[str, str]:
    """(category, kind) of a marker from its tag and prompt wording."""
    lower = prompt.lower()
    if tag == markers.ADD_CRITIQUE:
        return METHODOLOGICAL, "review-method" if "this review's own method" in lower else "study-methods"
    if tag == markers.USER_ANALYSIS:
        return EVIDENCE, "data-synthesis" if "effect size" in lower or "pool" in lower else "contradiction"
    if tag == markers.YOUR_OBSERVATION:
        return FUTURE, "trend" if "hints at where research is moving" in lower else "priorities"
    if "what this review implies" in lower or section.lower() == "conclusion":
        return THEORETICAL, "implication"
    return THEORETICAL, "gap-significance"


def _focus(kind: str, prompt: str, context: str, topic: str) -> str:
    quoted = QUOTED.search(prompt)
    if kind == "gap-significance" and quoted:
        return quoted.group(1)
    if kind == "contradiction":
        theme = re.search(r"both address (.+?) with different", prompt) or re.search(r"contradiction between .+?: (.+?)(?: Check|$)", prompt)
        if theme:
            return theme.group(1).rstrip(".")
        return "conflicting findings between studies"
    if kind == "data-synthesis":
        return "pooling effect sizes across studies"
    if kind == "study-methods":
        design = re.search(r"The (.+?) in this review", prompt)
        return f"limitations of the {design.group(1)}" if design else "limitations of the included studies"
    if kind == "review-method":
        return "limitations of this review's method"
    if kind == "trend":
        return f"emerging directions in {topic}"
    if kind == "priorities":
        return f"priorities for future studies of {topic}"
    return f"implications of this review for research on {topic}"


def parse_review(path: Path) -> ReviewDocument:
    """Reads a Phase 1 review (markdown) and extracts its gaps in document order."""
    text = Path(path).read_text(encoding="utf-8")
    lines = text.splitlines()
    title = next((l[2:].strip() for l in lines if l.startswith("# ")), Path(path).stem)
    topic, review_type = title, "literature review"
    for line in lines[:8]:
        match = TOPIC_LINE.match(line.strip())
        if match:
            review_type, topic = match.group(1).strip(), match.group(2).strip()
            break
    references_start = next(
        (i for i, l in enumerate(lines) if re.match(r"^##\s+(References|Bibliography)\s*$", l.strip(), re.IGNORECASE)),
        None,
    )

    gaps: List[Gap] = []
    section = subsection = ""
    paragraph: List[str] = []  # the last prose paragraph seen, as lines
    last_paragraph = ""
    end = references_start if references_start is not None else len(lines)
    for index, line in enumerate(lines[:end]):
        stripped = line.strip()
        if stripped.startswith("## "):
            section, subsection, last_paragraph, paragraph = stripped[3:].strip(), "", "", []
            continue
        if stripped.startswith("### "):
            subsection, last_paragraph, paragraph = stripped[4:].strip(), "", []
            continue
        placed = markers.MARKER_LINE.match(stripped)
        if placed:
            if paragraph:
                last_paragraph, paragraph = " ".join(paragraph), []
            tag = placed.group(1)
            prompt = _plain(stripped[placed.end() :])
            category, kind = classify(tag, prompt, section)
            context = _plain(last_paragraph)
            gaps.append(Gap(
                id=len(gaps) + 1,
                tag=tag,
                prompt=prompt,
                category=category,
                kind=kind,
                section=section,
                subsection=subsection,
                context=context,
                line_index=index,
                focus=_focus(kind, prompt, context, topic),
                cited=citations_in(f"{prompt} {context}"),
            ))
            continue
        if not stripped:
            if paragraph:
                last_paragraph, paragraph = " ".join(paragraph), []
        elif not stripped.startswith(("|", "![", "> ", "#")) and not (stripped.startswith("*") and stripped.endswith("*") and not stripped.startswith("**")):
            paragraph.append(stripped)

    for gap in gaps:
        surnames = [s for s, _ in gap.cited]
        gap.keywords = keywords_of(f"{gap.focus} {gap.focus} {gap.context}", exclude=surnames)
    return ReviewDocument(
        path=Path(path),
        lines=lines,
        title=title,
        topic=topic,
        review_type=review_type,
        gaps=gaps,
        references_start=references_start,
    )
