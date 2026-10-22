"""Phase 1.5, step 3: collects the user's own analysis for each gap and checks it before integration.

The user writes their answers in a plain-text file. write_template() creates it: one block per gap
with the task, its context, the hints and a suggested length (all on lines starting with '#', which
are ignored), followed by an empty ANSWER area. parse_answers() reads it back; hand-written files
with "GAP 1:" headings, or answers separated by '---' lines, are accepted too.

Every answer is then checked (check_answer):
  * originality: text copied from the review, its papers, the newly found papers or the hints is
    rejected; short quotations in "double quotes" are allowed as long as most of the text is the
    user's own,
  * critical thinking: the answer must compare, critique, reason or suggest, and make the moves its
    gap asks for (a contradiction needs comparison and critique, a future direction a suggestion...),
  * length against the suggested word count, and leftover placeholder text.
Problems that make an answer unusable are errors (the gap stays unfilled); the rest are warnings.
"""
from __future__ import annotations

import codecs
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .critical_gaps import MARKER_PATTERN
from .gap_parser import Gap, ReviewDocument, citations_in
from .gap_research import GapResearch, Research
from .llm_synthesis import PLACEHOLDER_PATTERN

SHINGLE = 6  # words per phrase compared when looking for copied text
MAX_COPIED_SHARE = 0.20  # share of the answer's 6-word phrases found verbatim in a source
MAX_COPIED_RUN = 15  # longest verbatim run of words (outside quotation marks) tolerated
MAX_QUOTED_SHARE = 0.35  # share of the answer's words that may sit inside quotation marks
MAX_ANSWER_OVERLAP = 0.5  # share of an answer repeated from another answer

TEMPLATE_HEADER = re.compile(r"^====\s*GAP\s+(\d+)\b", re.IGNORECASE)  # as write_template() writes them
GAP_HEADER = re.compile(r"^\s*(?:=+\s*)?(?:#{1,6}\s*)?gap\s+(\d+)\s*(?:[:.)|=-]|$)", re.IGNORECASE)  # hand-written
ANSWER_LINE = re.compile(r"^\s*>>>\s*answer\b", re.IGNORECASE)
SEPARATOR = re.compile(r"^\s*-{3,}\s*$")
QUOTED_SPAN = re.compile(r"[\"“]([^\"”]{1,600})[\"”]")
# "<insert citation>" is a placeholder; "p < 0.05 and n > 100" is not.
ANGLE_PLACEHOLDER = re.compile(r"^<[^\s<>]")

# Phrases that show each critical-thinking move, matched case-insensitively as whole words.
CRITICAL_MOVES: Dict[str, Tuple[str, ...]] = {
    "comparison": (
        "compared with", "compared to", "in comparison", "in contrast", "by contrast", "unlike", "whereas",
        "similarly", "likewise", "both", "neither", "while", "than", "differs", "differ", "differed",
        "difference", "differences", "different from", "same as", "on the other hand", "conversely", "alike",
        "consistent with", "in line with", "agree", "agrees", "disagree", "disagrees", "outperform", "outperforms",
    ),
    "critique": (
        "however", "but", "although", "though", "despite", "yet", "contradict", "contradicts", "contradicted",
        "contradiction", "contradictory", "inconsistent", "limitation", "limitations", "limited", "weakness",
        "weak", "flaw", "flawed", "bias", "biased", "fails", "failed", "fail to", "overlook", "overlooks",
        "overlooked", "ignores", "ignored", "questionable", "unconvincing", "not convincing", "insufficient",
        "lacks", "lacking", "lack of", "problematic", "undermines", "overstate", "overstates", "unclear",
        "untested", "unproven", "cannot", "does not", "do not", "doubt", "caution", "challenge", "challenges",
        "shortcoming", "shortcomings", "drawback", "drawbacks", "narrow", "only",
    ),
    "suggestion": (
        "should", "could", "would benefit", "needs to", "need to", "must", "recommend", "recommended",
        "propose", "proposed", "suggest", "future research", "future studies", "future work", "further research",
        "next step", "it is worth", "would help", "would allow", "would", "might", "plan to", "aim to",
        "intend to", "my study", "my research", "our study", "this study will", "prioritise", "prioritize",
        "priority", "priorities",
    ),
    "reasoning": (
        "because", "therefore", "thus", "hence", "since", "as a result", "consequently", "which means",
        "this means", "implies", "implication", "suggests that", "indicates", "so that", "due to",
        "this matters", "matters because", "leads to", "explains", "given that", "it follows",
    ),
}
MOVE_ADVICE = {
    "comparison": "compare the studies or positions directly (e.g. 'whereas X found..., Y...')",
    "critique": "say where the evidence or method falls short, or where the papers disagree",
    "suggestion": "propose what should be done next or how your own work would address it",
    "reasoning": "explain why: connect your claims with 'because', 'therefore', 'which means'",
}
MOVE_LABELS = {
    "comparison": "comparison",
    "critique": "contradiction or critique",
    "suggestion": "suggestion",
    "reasoning": "reasoning",
}


# ---- template -------------------------------------------------------------------------------

def _comment(text: str, width: int = 98, indent: str = "") -> List[str]:
    wrapped = textwrap.wrap(text, width=width - 2 - len(indent), subsequent_indent=indent) or [""]
    return [f"# {line}".rstrip() for line in wrapped]


def write_template(path: Path, review: ReviewDocument, research: Research, input_hint: str) -> Path:
    lines = [
        f"# ResearchFlow Phase 1.5: your analysis for \"{review.title}\"",
        "#",
        *_comment(
            "Write your own analysis under each '>>> ANSWER' line. Lines starting with '#' are instructions "
            "and are ignored. The hints point you to evidence; they are not text to paste: answers copied from "
            "the hints, the review or the papers are rejected. Short quotations in \"double quotes\" are fine."
        ),
        *_comment(
            "Cite papers as (Surname, Year) or Surname (Year). Papers listed under 'Further reading' are added "
            "to your references automatically when you cite them."
        ),
        "#",
        "# When you are done, run:",
        f"#   python -m src.main --phase 1.5 --input \"{input_hint}\" --filled-gaps \"{path}\"",
        "",
    ]
    for gap in review.gaps:
        item = research.gaps.get(gap.id) or GapResearch(gap.id)
        low, high = gap.word_range
        moves = " and ".join(MOVE_LABELS[m] for m in gap.required_thinking)
        lines += [
            f"==== GAP {gap.id} | {gap.tag} | {gap.category} ====",
            *_comment(f"Where: {gap.location or 'top level'}"),
            *_comment(f"Task: {gap.prompt}", indent="  "),
        ]
        if gap.context:
            context = gap.context if len(gap.context) <= 400 else gap.context[:400].rsplit(" ", 1)[0] + "..."
            lines += _comment(f"Context: {context}", indent="  ")
        if gap.cited:
            lines += _comment("Papers involved: " + ", ".join(f"{s} ({y})" for s, y in gap.cited))
        lines += _comment(f"Suggested length: {low}-{high} words. Your answer needs: {moves}.")
        lines.append("#")
        for n, hint in enumerate(item.hints, 1):
            lines += _comment(f"Hint {n} ({hint.title}): {hint.text}", indent="  ")
        if item.new_papers:
            lines.append("# Further reading:")
            for p in item.new_papers:
                link = f"https://doi.org/{p.doi}" if p.doi else p.url
                lines += _comment(f"  - {p.first_author_surname} ({p.year or 'n.d.'}). {p.title}. {link}".rstrip(), indent="    ")
        lines += [">>> ANSWER", "", ""]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    return Path(path)


# ---- parsing --------------------------------------------------------------------------------

class AnswerFileError(ValueError):
    """The filled-gaps file can't be matched to the review's gaps."""


def _clean_block(lines: Sequence[str]) -> str:
    """Answer text without comment lines, with paragraphs separated by one blank line."""
    paragraphs, current = [], []
    for line in lines:
        if line.lstrip().startswith("#") or ANSWER_LINE.match(line):
            continue
        if line.strip():
            current.append(line.strip())
        elif current:
            paragraphs.append(" ".join(current))
            current = []
    if current:
        paragraphs.append(" ".join(current))
    return "\n\n".join(paragraphs)


def _read_text(path: Path) -> str:
    """Reads the answers file whatever encoding the user's editor saved it in."""
    raw = path.read_bytes()
    if raw.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return raw.decode("utf-16")
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace")


def parse_answers(path: Path, gap_count: int) -> Dict[int, str]:
    """{gap id: answer text} ("" where the user wrote nothing)."""
    lines = _read_text(Path(path)).splitlines()
    answers: Dict[int, List[str]] = {}
    current: Optional[int] = None
    # In a template only "==== GAP n" lines are headings, so an answer may itself begin "Gap 3 matters...".
    header_pattern = TEMPLATE_HEADER if any(TEMPLATE_HEADER.match(l) for l in lines) else GAP_HEADER
    if any(header_pattern.match(l) and not l.lstrip().startswith("# ") for l in lines):
        for line in lines:
            header = header_pattern.match(line)
            if header and not line.lstrip().startswith("# "):
                current = int(header.group(1))
                answers.setdefault(current, [])
                # A hand-written "GAP 2: my answer..." may start the answer on the heading line itself.
                rest = line[header.end():].strip() if header_pattern is GAP_HEADER else ""
                if rest and not rest.startswith("="):
                    answers[current].append(rest)
                continue
            if current is not None:
                answers[current].append(line)
        unknown = sorted(n for n in answers if not 1 <= n <= gap_count)
        if unknown:
            raise AnswerFileError(
                f"The answers file has GAP {', '.join(map(str, unknown))}, but the review only has {gap_count} gaps. "
                "Was it written for a different review?"
            )
        return {n: _clean_block(answers.get(n, [])) for n in range(1, gap_count + 1)}

    content = [l for l in lines if not l.lstrip().startswith("#")]
    if any(SEPARATOR.match(l) for l in content):
        blocks, block = [], []
        for line in content:
            if SEPARATOR.match(line):
                blocks.append(block)
                block = []
            else:
                block.append(line)
        blocks.append(block)
        texts = [_clean_block(b) for b in blocks]
        texts = [t for t in texts if t] if len([t for t in texts if t]) == gap_count else texts
    else:
        texts = [t for t in _clean_block(content).split("\n\n") if t]
    if len(texts) != gap_count:
        raise AnswerFileError(
            f"Found {len(texts)} answers but the review has {gap_count} gaps. Start each answer with a line "
            "'GAP <number>:' (or separate answers with a '---' line) so each one can be matched to its gap."
        )
    return {n: text for n, text in enumerate(texts, 1)}


# ---- checks ---------------------------------------------------------------------------------

def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _shingles(tokens: Sequence[str], n: int = SHINGLE) -> List[Tuple[str, ...]]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


class SourceIndex:
    """6-word phrases of every text an answer must not copy from, labelled by where they come from."""

    def __init__(self):
        self._phrases: Dict[Tuple[str, ...], str] = {}
        self.titles: List[str] = []  # paper titles, which an answer may name without copying anything

    def add(self, label: str, text: str) -> None:
        for phrase in _shingles(_tokens(text)):
            self._phrases.setdefault(phrase, label)

    def add_title(self, title: str) -> None:
        if len(title.split()) >= 3:
            self.titles.append(title.strip().rstrip("."))

    def __len__(self) -> int:
        return len(self._phrases)

    def match(self, phrase: Tuple[str, ...]) -> Optional[str]:
        return self._phrases.get(phrase)


@dataclass
class Originality:
    copied_share: float = 0.0
    longest_run: int = 0  # words
    quoted_share: float = 0.0
    source: str = ""  # where most copied phrases come from
    excerpt: str = ""  # the longest copied passage

    @property
    def score(self) -> float:
        """1.0 = entirely the user's own words, 0.0 = at or past the copy limit."""
        return max(0.0, 1.0 - self.copied_share / MAX_COPIED_SHARE)


def originality(text: str, index: SourceIndex) -> Originality:
    """How much of the text, outside quotation marks and paper titles, is copied from the indexed sources."""
    total_words = len(_tokens(text))
    quoted_words = sum(len(_tokens(q)) for q in QUOTED_SPAN.findall(text))
    unquoted = QUOTED_SPAN.sub("\n", text)
    for title in sorted(index.titles, key=len, reverse=True):
        unquoted = re.sub(re.escape(title), "\n", unquoted, flags=re.IGNORECASE)
    # Each stretch between quotations is compared separately, so no phrase spans a removed quotation.
    phrases: List[Tuple[str, ...]] = []
    hits: List[Optional[str]] = []
    for segment in unquoted.split("\n"):
        segment_phrases = _shingles(_tokens(segment))
        if phrases and segment_phrases:
            hits.append(None)  # breaks a verbatim run at the segment boundary
            phrases.append(())
        phrases += segment_phrases
        hits += [index.match(p) for p in segment_phrases]
    checked = sum(1 for p in phrases if p)
    if not checked:
        return Originality(quoted_share=quoted_words / total_words if total_words else 0.0)
    sources: Dict[str, int] = {}
    longest = run = 0
    best_end = -1
    for i, hit in enumerate(hits):
        if hit:
            sources[hit] = sources.get(hit, 0) + 1
            run += 1
            if run > longest:
                longest, best_end = run, i
        else:
            run = 0
    copied = sum(1 for h in hits if h)
    excerpt = ""
    if longest:
        run_phrases = phrases[best_end - longest + 1 : best_end + 1]
        excerpt = " ".join(list(run_phrases[0]) + [p[-1] for p in run_phrases[1:]])
    return Originality(
        copied_share=copied / checked,
        longest_run=longest + SHINGLE - 1 if longest else 0,
        quoted_share=quoted_words / total_words if total_words else 0.0,
        source=max(sources, key=sources.get) if sources else "",
        excerpt=excerpt,
    )


def placeholders_in(text: str) -> List[str]:
    """Placeholder text ("Lorem ipsum", "[insert citation]", "<your text>"), ignoring inequalities like "p < 0.05"."""
    return [
        m.group(0) for m in PLACEHOLDER_PATTERN.finditer(text or "")
        if not m.group(0).startswith("<") or ANGLE_PLACEHOLDER.match(m.group(0))
    ]


def critical_moves(text: str) -> Dict[str, List[str]]:
    """{move: cue phrases found} for each critical-thinking move the text makes."""
    lower = " " + re.sub(r"\s+", " ", (text or "").lower()) + " "
    found: Dict[str, List[str]] = {}
    for move, cues in CRITICAL_MOVES.items():
        hits = [c for c in cues if re.search(r"(?<![a-z])" + re.escape(c) + r"(?![a-z])", lower)]
        if hits:
            found[move] = hits
    return found


@dataclass
class AnswerCheck:
    gap: Gap
    text: str
    words: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    originality: Originality = field(default_factory=Originality)
    moves: Dict[str, List[str]] = field(default_factory=dict)
    citations: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def filled(self) -> bool:
        return bool(self.text.strip())

    @property
    def accepted(self) -> bool:
        return self.filled and not self.errors

    @property
    def thinking_score(self) -> float:
        """0-1: required moves made (60%) plus breadth of moves overall (40%)."""
        if not self.filled:
            return 0.0
        required = self.gap.required_thinking
        made = sum(1 for m in required if m in self.moves) / len(required)
        breadth = len(self.moves) / len(CRITICAL_MOVES)
        return 0.6 * made + 0.4 * breadth


def check_answer(
    gap: Gap,
    text: str,
    index: SourceIndex,
    earlier: Sequence[Tuple[int, str]] = (),
) -> AnswerCheck:
    check = AnswerCheck(gap=gap, text=text.strip())
    if not check.filled:
        check.errors.append("No answer written yet.")
        return check
    check.words = len(re.findall(r"[A-Za-z][A-Za-z'’-]*", text))
    low, high = gap.word_range

    if MARKER_PATTERN.search(text) or placeholders_in(text):
        check.errors.append("Contains a gap marker or placeholder text; replace it with your own analysis.")

    if check.words < low // 2:
        check.errors.append(f"Too short to be an analysis ({check.words} words; aim for {low}-{high}).")
    elif check.words < low:
        check.warnings.append(f"Shorter than suggested ({check.words} words; aim for {low}-{high}).")
    elif check.words > high * 1.5:
        check.warnings.append(f"Much longer than suggested ({check.words} words; aim for {low}-{high}). Consider tightening it.")

    check.originality = result = originality(text, index)
    if result.copied_share > MAX_COPIED_SHARE or result.longest_run >= MAX_COPIED_RUN:
        check.errors.append(
            f"Copied text detected ({result.copied_share:.0%} of phrases match {result.source or 'the sources'}; "
            f"longest copied passage {result.longest_run} words: \"{result.excerpt[:120]}...\"). "
            "Rewrite it in your own words, or put a short quotation in \"double quotes\" with its citation."
        )
    elif result.copied_share > MAX_COPIED_SHARE / 2:
        check.warnings.append(
            f"{result.copied_share:.0%} of phrases match {result.source}; rephrase more of it in your own words."
        )
    if result.quoted_share > MAX_QUOTED_SHARE:
        check.errors.append(
            f"{result.quoted_share:.0%} of the answer is quotation; keep quotes short and make most of it your own analysis."
        )
    elif result.quoted_share and not citations_in(text):
        check.warnings.append("Contains a quotation but no citation; add (Surname, Year) after it.")

    own = _shingles(_tokens(text))
    for other_id, other in earlier:
        theirs = set(_shingles(_tokens(other)))
        if own and theirs and sum(1 for p in own if p in theirs) / len(own) > MAX_ANSWER_OVERLAP:
            check.errors.append(f"Largely repeats your answer to gap {other_id}; each gap needs its own analysis.")
            break

    check.moves = critical_moves(text)
    analytical = {m for m in check.moves if m in ("comparison", "critique", "suggestion", "reasoning")}
    if not analytical:
        check.errors.append(
            "Reads as description only: it does not compare, critique, reason or suggest. "
            + "; ".join(MOVE_ADVICE[m] for m in gap.required_thinking) + "."
        )
    else:
        for move in gap.required_thinking:
            if move not in check.moves:
                check.warnings.append(f"Missing {MOVE_LABELS[move]}: {MOVE_ADVICE[move]}.")

    check.citations = citations_in(text)
    if gap.cited and not check.citations:
        check.warnings.append(
            "Cites no papers; refer to the studies involved, e.g. "
            + " or ".join(f"({s}, {y})" for s, y in gap.cited[:2]) + "."
        )
    return check


def build_source_index(review: ReviewDocument, corpus, research: Research) -> SourceIndex:
    """Everything an answer could be copied from: the review, its papers, new papers and the hints."""
    index = SourceIndex()
    index.add("the review text", review.body_text)
    for gap in review.gaps:
        index.add("the gap prompt", gap.prompt)
    for source in corpus.papers:
        index.add(source.citation, f"{source.paper.title}. {source.text}")
        index.add_title(source.paper.title)
    for item in research.gaps.values():
        index.add("the hints", item.hint_text)
        for e in item.evidence:
            index.add(e.citation, e.sentence)
        for paper in item.new_papers:
            index.add(f"{paper.first_author_surname} ({paper.year or 'n.d.'})", f"{paper.title}. {paper.abstract}")
            index.add_title(paper.title)
    return index
