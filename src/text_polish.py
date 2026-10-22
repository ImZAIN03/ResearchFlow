"""Phase 1.5, step 4: polishes the user's answers so they read as part of the review.

Polishing never changes what the user argues. It
  * fixes spacing, capitalization, punctuation and known misspellings,
  * moves the register to academic English: expands contractions, replaces informal words and
    sentence openers ("So," -> "Therefore,"),
  * repairs citations: normalizes "(Smith 2020)" to "(Smith, 2020)", adds the year to a bare
    author name that matches exactly one paper ("Smith argues" -> "Smith (2020) argues"), and records
    which cited papers are new to the review (they are added to the references) or unknown,
  * optionally asks the local model for a grammar and clarity pass; its version is only used when it
    keeps the user's citations, vocabulary and length (see _model_edit_issue), so it can't add ideas.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .gap_parser import citations_in
from .gap_research import Corpus
from .llm_synthesis import (
    META_REFERENCE,
    PLACEHOLDER_PATTERN,
    LLMSynthesizer,
    clean_response,
    fix_typos,
)
from .paper_discovery import Paper

logger = logging.getLogger(__name__)

CONTRACTIONS = {
    "can't": "cannot", "won't": "will not", "shan't": "shall not", "n't": " not",
    "it's": "it is", "that's": "that is", "there's": "there is", "here's": "here is", "what's": "what is",
    "let's": "let us", "i'm": "I am", "i've": "I have", "i'd": "I would", "i'll": "I will",
    "we're": "we are", "we've": "we have", "we'd": "we would", "we'll": "we will",
    "they're": "they are", "they've": "they have", "they'd": "they would", "they'll": "they will",
    "you're": "you are", "you've": "you have", "you'll": "you will", "who's": "who is",
    "he's": "he is", "she's": "she is", "isn't": "is not", "aren't": "are not",
}
# Informal word or phrase -> academic replacement ("" removes a filler word).
INFORMAL = {
    "a lot of": "many", "lots of": "many", "a bunch of": "several", "tons of": "many",
    "pretty much": "largely", "kind of a": "a", "gonna": "going to", "wanna": "want to", "gotta": "have to",
    "huge": "substantial", "tiny": "very small", "really": "", "basically": "", "actually": "", "totally": "entirely",
    "super": "highly", "okay": "acceptable", "ok": "acceptable", "stuff": "material", "anyways": "in any case",
    "nowadays": "currently", "get rid of": "eliminate", "figure out": "determine", "find out": "determine",
    "look into": "examine", "come up with": "develop", "point out": "note", "go up": "increase", "go down": "decrease",
    "plus,": "in addition,", "thing is": "the issue is", "way more": "considerably more", "way less": "considerably less",
}
# Informal sentence openers -> academic ones; "So far", "So that", "And yet" etc. are left alone.
SENTENCE_OPENERS = (
    (r"So,", "Therefore,"),
    (r"So(?=\s)(?!\s+(?:far|long|that|many|much|few|called|as|too)\b)", "Therefore,"),
    (r"But(?=\s)(?!\s+(?:also|rather|for)\b)", "However,"),
    (r"And(?=\s)(?!\s+(?:yet|so|then)\b)", "In addition,"),
    (r"Also,?(?=\s)", "In addition,"),
    (r"Plus,", "In addition,"),
    (r"Anyway,", "Nevertheless,"),
)
FIRST_PERSON_HEDGES = re.compile(r"\b(?:I think|I believe|I feel|in my opinion|personally)\b", re.IGNORECASE)
LOOSE_CITATION = re.compile(r"\(([A-Z][\w'’-]+)(?: et al\.?)?\s+(\d{4}|n\.d\.)\)")
ET_AL_NO_DOT = re.compile(r"\bet al(?!\.)\b")
YEAR_AFTER = re.compile(r"^\s*(?:et al\.?\s*)?(?:\(|,\s*(?:\d{4}|n\.d\.))")
MIN_KEPT_WORDS = 0.8  # share of the user's content words a model edit must keep
MAX_NEW_WORDS = 0.2  # share of new content words a model edit may introduce


@dataclass
class Polished:
    text: str
    changes: List[str] = field(default_factory=list)
    new_papers_cited: List[Paper] = field(default_factory=list)
    unknown_citations: List[str] = field(default_factory=list)
    ambiguous_names: List[str] = field(default_factory=list)
    model_edited: bool = False
    first_person: bool = False


def _match_case(original: str, replacement: str) -> str:
    if not replacement:
        return replacement
    if original[:1].isupper():
        return replacement[0].upper() + replacement[1:]
    return replacement


def _expand_contractions(text: str) -> Tuple[str, int]:
    count = 0
    text = text.replace("’", "'")

    def replace(match: re.Match) -> str:
        nonlocal count
        word = match.group(0)
        lower = word.lower()
        if lower in CONTRACTIONS:
            count += 1
            return _match_case(word, CONTRACTIONS[lower]) if not CONTRACTIONS[lower].startswith("I ") else CONTRACTIONS[lower]
        if lower.endswith("n't"):
            count += 1
            return word[:-3] + " not"
        return word

    text = re.sub(r"\b[A-Za-z]+'(?:t|s|re|ve|d|ll|m)\b", replace, text)
    return text, count


def _formalize(text: str) -> Tuple[str, List[str]]:
    replaced: List[str] = []
    for informal, formal in sorted(INFORMAL.items(), key=lambda item: -len(item[0])):
        pattern = re.compile(r"(?<![\w-])" + re.escape(informal) + r"(?![\w-])", re.IGNORECASE)
        if pattern.search(text):
            replaced.append(f"'{informal}'" + (f" -> '{formal}'" if formal else " removed"))
            text = pattern.sub(lambda m: _match_case(m.group(0), formal), text)
    for opener, formal in SENTENCE_OPENERS:
        pattern = re.compile(r"(^|(?<=[.!?]\s))" + opener, re.MULTILINE)
        found = pattern.search(text)
        if found:
            replaced.append(f"sentence opener '{found.group(0).strip()}' -> '{formal}'")
            text = pattern.sub(lambda m: m.group(1) + formal, text)
    text = re.sub(r",\s*,", ",", text)
    return text, replaced


ABBREVIATION_END = re.compile(r"(?:\b(?:e\.g|i\.e|et al|cf|vs|etc|approx|ca|no|fig|eq)\.|\b[A-Z]\.)\s+$", re.IGNORECASE)


def _capitalize_sentence(match: re.Match) -> str:
    """Upper-cases a sentence's first letter, unless the 'sentence end' was an abbreviation (e.g., et al.)."""
    before = match.string[: match.start() + len(match.group(1))]
    if match.group(1) and ABBREVIATION_END.search(before):
        return match.group(0)
    return match.group(1) + match.group(2).upper()


def tidy(text: str) -> str:
    """Spacing, capitalization and punctuation, paragraph by paragraph."""
    paragraphs = []
    for paragraph in re.split(r"\n\s*\n", text.strip()):
        p = re.sub(r"\s+", " ", paragraph).strip()
        if not p:
            continue
        p = re.sub(r"\s+([,.;:!?)])", r"\1", p)
        p = re.sub(r"\(\s+", "(", p)
        p = re.sub(r"([,;:])(?=[A-Za-z])", r"\1 ", p)
        p = re.sub(r"(?<=[a-z])\.(?=[A-Z][a-z])", ". ", p)  # "end.Start" (not "e.g.", decimals or initials)
        p = re.sub(r"([!?])\1+", r"\1", p)
        p = re.sub(r"(?<!\.)\.\.(?!\.)", ".", p)
        p = re.sub(r"(?<![\w.'’(])i(?![\w.'’)])", "I", p)  # the pronoun, not "i.e." or an "(i)" list item
        p = re.sub(r"(^|[.!?]\s+),\s*", r"\1", p)  # a removed filler word left "Really, the..." as ", the..."
        p = re.sub(r"(^|[.!?]\s+)([a-z])", _capitalize_sentence, p)
        p = re.sub(r"\s{2,}", " ", p)
        if p[-1] not in ".!?\"”)":
            p += "."
        elif p[-1] == ")" and not re.search(r"[.!?]\)$", p) and p.count("(") == p.count(")"):
            p += "."
        paragraphs.append(p)
    return "\n\n".join(paragraphs)


def _common_words(review_text: str) -> set:
    """Surnames that are also ordinary words in the review ("Long", "Young") are not treated as citations."""
    return {w for w in re.findall(r"\b[a-z]{3,}\b", review_text)}


def repair_citations(text: str, corpus: Corpus, new_papers: Sequence[Paper], review_text: str) -> Polished:
    result = Polished(text=text)
    fixed = LOOSE_CITATION.sub(lambda m: f"({m.group(0)[1:-1].rsplit(' ', 1)[0]}, {m.group(2)})", text)
    fixed = ET_AL_NO_DOT.sub("et al.", fixed)
    if fixed != text:
        result.changes.append("citation punctuation normalized to (Surname, Year)")
    common = _common_words(review_text)

    # A bare surname of a known paper gets its year on first mention.
    for surname in sorted({p.surname for p in corpus.papers} | {p.first_author_surname for p in new_papers}, key=len, reverse=True):
        if len(surname) < 3 or surname.lower() in common or surname == "Unknown":
            continue
        candidates = {p.year for p in corpus.with_surname(surname)} | {
            str(p.year or "n.d.") for p in new_papers if p.first_author_surname == surname
        }
        mentions = list(re.finditer(r"(?<![\w(])" + re.escape(surname) + r"(?![\w'’-])", fixed))
        if not mentions:
            continue
        cited_already = any(YEAR_AFTER.match(fixed[m.end():]) for m in mentions) or any(
            s == surname for s, _ in citations_in(fixed)
        )
        inside_parens = any(fixed.rfind("(", 0, m.start()) > fixed.rfind(")", 0, m.start()) for m in mentions)
        if cited_already or inside_parens:
            continue
        if len(candidates) != 1:
            result.ambiguous_names.append(surname)
            continue
        year = next(iter(candidates))
        first = mentions[0]
        tail = fixed[first.end():]
        et_al = re.match(r"\s+et al\.", tail)
        insert_at = first.end() + (et_al.end() if et_al else 0)
        fixed = f"{fixed[:insert_at]} ({year}){fixed[insert_at:]}"
        result.changes.append(f"added year to {surname} ({year})")

    for surname, year in citations_in(fixed):
        if corpus.find(surname, year):
            continue
        match = next((p for p in new_papers if p.first_author_surname.lower() == surname.lower() and str(p.year or "n.d.") == year), None)
        if match:
            if all(p.dedup_key() != match.dedup_key() for p in result.new_papers_cited):
                result.new_papers_cited.append(match)
        else:
            result.unknown_citations.append(f"{surname} ({year})")
    result.text = fixed
    return result


def _content_words(text: str) -> set:
    return {w for w in re.findall(r"[a-z]{4,}", text.lower())}


def _model_edit_issue(original: str, edited: str) -> Optional[str]:
    """Why a model's grammar edit can't be used, or None if it preserves the user's text."""
    if not edited.strip():
        return "empty"
    if PLACEHOLDER_PATTERN.search(edited) or META_REFERENCE.search(edited):
        return "placeholder or prompt echo"
    if set(citations_in(edited)) != set(citations_in(original)):
        return "changed the citations"
    ratio = len(edited.split()) / max(1, len(original.split()))
    if not 0.8 <= ratio <= 1.2:
        return f"changed the length ({ratio:.0%})"
    before, after = _content_words(original), _content_words(edited)
    if before and len(before & after) / len(before) < MIN_KEPT_WORDS:
        return "dropped the user's wording"
    if after and len(after - before) / len(after) > MAX_NEW_WORDS:
        return "introduced new content"
    return None


def model_edit(synthesizer: LLMSynthesizer, paragraph: str) -> Optional[str]:
    system = (
        "You are an academic copy editor. You correct grammar, spelling and clarity only. You never add "
        "ideas, facts, examples or citations, and never remove the author's arguments."
    )
    user = (
        f"Paragraph:\n{paragraph}\n\n"
        "Task: Return the paragraph with grammar, spelling and awkward phrasing corrected in formal academic "
        "English. Keep every claim, every citation exactly as written, and about the same length. "
        "Return only the corrected paragraph."
    )
    try:
        edited = clean_response(synthesizer.complete(system, user, max_tokens=int(len(paragraph.split()) * 2) + 60, temperature=0.2))
    except Exception as exc:  # polishing is optional; keep the user's text
        logger.debug("Model edit failed: %s", exc)
        return None
    edited = re.sub(r"\s+", " ", edited.strip().strip('"'))
    problem = _model_edit_issue(paragraph, edited)
    if problem:
        logger.info("Kept the user's wording (model edit rejected: %s)", problem)
        return None
    return edited


def polish(
    text: str,
    corpus: Corpus,
    new_papers: Sequence[Paper] = (),
    review_text: str = "",
    synthesizer: Optional[LLMSynthesizer] = None,
) -> Polished:
    changes: List[str] = []
    working = fix_typos(text)
    if working != text:
        changes.append("spelling corrected")
    working, contractions = _expand_contractions(working)
    if contractions:
        changes.append(f"{contractions} contraction(s) expanded")
    working, replaced = _formalize(working)
    changes += [f"informal wording: {r}" for r in replaced]
    working = tidy(working)

    edited_any = False
    if synthesizer is not None:
        paragraphs = []
        for paragraph in working.split("\n\n"):
            edited = model_edit(synthesizer, paragraph)
            edited_any = edited_any or bool(edited and edited != paragraph)
            paragraphs.append(edited or paragraph)
        if edited_any:
            working = tidy(_formalize(_expand_contractions("\n\n".join(paragraphs))[0])[0])
            changes.append("grammar and clarity edited (meaning, citations and length preserved)")

    result = repair_citations(working, corpus, new_papers, review_text)
    result.text = tidy(result.text)
    result.changes = changes + result.changes
    result.model_edited = edited_any
    result.first_person = bool(FIRST_PERSON_HEDGES.search(result.text))
    return result


def restyle_citations(text: str, style: str, order: Dict[Tuple[str, str], int]) -> str:
    """Converts the review's (Surname, Year) citations to MLA (Surname) or IEEE [n] in-text form.

    APA and Chicago author-date keep (Surname, Year). `order` maps (surname, year) to the IEEE
    reference number; citations not in it are left unchanged.
    """
    style = style.lower()
    if style not in ("mla", "ieee"):
        return text

    def group(match: re.Match) -> str:
        parts = [p.strip() for p in match.group(1).split(";")]
        converted = []
        for part in parts:
            cite = re.fullmatch(r"([A-Z][\w'’-]+)( et al\.?)?, (\d{4}|n\.d\.)", part)
            if not cite:
                return match.group(0)
            surname, et_al, year = cite.groups()
            if style == "ieee":
                if (surname, year) not in order:
                    return match.group(0)
                converted.append(str(order[(surname, year)]))
            else:
                converted.append(f"{surname}{et_al or ''}")
        return f"[{', '.join(converted)}]" if style == "ieee" else f"({'; '.join(converted)})"

    text = re.sub(r"\(((?:[A-Z][\w'’-]+(?: et al\.?)?, (?:\d{4}|n\.d\.)(?:;\s*)?)+)\)", group, text)

    def narrative(match: re.Match) -> str:
        surname, et_al, year = match.group(1), match.group(2) or "", match.group(3)
        if style == "ieee":
            return f"{surname}{et_al} [{order[(surname, year)]}]" if (surname, year) in order else match.group(0)
        return f"{surname}{et_al}"

    return re.sub(r"\b([A-Z][\w'’-]+)( et al\.?)? \((\d{4}|n\.d\.)\)", narrative, text)
