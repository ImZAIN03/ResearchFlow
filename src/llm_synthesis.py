"""Uses a local Ollama model to summarize papers and synthesize findings across a literature set.

Runs entirely against a local Ollama server (https://ollama.com) instead of a hosted API, so no
API key or network egress is required beyond http://localhost:11434. Defaults to Mistral (7B),
which writes far more coherent, better-grounded prose than the small models; on low-RAM machines
"tinyllama" (~1.1B) or "phi" (Phi-2, ~2.7B) still work. Pull whichever you use first:
`ollama pull mistral`.

Small models regularly return empty, truncated, looping or placeholder ("Lorem ipsum...") text, so
every generation is validated with content_issue() and retried once with a stricter prompt. If the
model still can't produce usable text, the methods fall back to content extracted directly from the
papers' abstracts rather than returning nothing.
"""
from __future__ import annotations

import difflib
import logging
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import requests

from .paper_discovery import Paper
from .text_extraction import TextExtractor

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "mistral"  # alternatives for low-RAM machines: "phi" (Phi-2), "tinyllama"
DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_KEEP_ALIVE = "5m"  # model stays warm between the many calls within a single run
REQUEST_TIMEOUT = 600  # CPU inference on small models can be slow

# TinyLlama and Phi-2 have a 2048-token context window. Prompt + output must fit inside it, otherwise
# Ollama silently truncates the prompt and the model answers with little or nothing, so inputs are
# kept well under ~1200 tokens (~4 chars/token) and outputs are capped per task. Larger models such
# as Mistral get a bigger window, so each paper keeps more of its summary in the synthesis prompts.
NUM_CTX = 2048
MAX_SOURCE_CHARS = 3500  # per-paper text sent for summarization
MAX_CONTEXT_CHARS = 3200  # combined paper summaries sent for synthesis / section writing
SMALL_CONTEXT_MODELS = ("tinyllama", "phi")
LARGE_NUM_CTX = 8192
LARGE_SOURCE_CHARS = 8000
LARGE_CONTEXT_CHARS = 9000
MAX_RETRIES = 1  # one extra attempt per generation keeps a 3-paper run around 10 minutes on CPU

PLACEHOLDER_PATTERN = re.compile(
    r"lorem ipsum|dolor sit amet|consectetur adipiscing|vestibulum|"
    r"<[^<>\n]{2,40}>|\[(?:insert|your|add|placeholder|citation needed)[^\]]*\]|"
    r"\b(?:TODO|TBD|xxx+)\b|your (?:text|content|answer) here|"
    r"\bbullet point\b|\bsection heading\b|\bsub-heading\b",
    re.IGNORECASE,
)
# Chatty preambles small models put before the actual answer ("Sure! Here is a summary:").
PREAMBLE_PATTERN = re.compile(
    r"^(?:sure|certainly|of course|okay|here(?: is|'s| are)|below is)\b[^\n]{0,80}:\s*$", re.IGNORECASE
)
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])")
# Small models echo the prompt back: field labels ("Paper title: ... Text: ..."), the per-paper
# "[Author, Year] Title" context lines, or "Answer:" prefixes. Sentences carrying metadata labels are
# dropped and leading labels are stripped.
METADATA_LABEL = re.compile(
    r"\b(?:(?:research )?paper(?: title)?|authors?(?:\(s\))?|year|doi|publication date|number of pages|type)\s*:", re.IGNORECASE
)
LEADING_LABEL = re.compile(
    r"^(?:answer|response|summary|task|text|introduction|method(?:ology)?|main results?|results?|"
    r"conclusion|key findings?|abstract|title)\s*:\s*",
    re.IGNORECASE,
)
# Abstract sentences announcing future work ("We conclude our survey with open research questions")
# say nothing about what the paper found.
FUTURE_WORK_REMARK = re.compile(r"\bconclude (?:our|this|the) (?:survey|paper|review)|\bopen (?:research )?questions\b|\bdirections for (?:the )?future\b", re.IGNORECASE)
# Fragments of a PDF's first page (arXiv stamp, keyword list, numbered headings, links) that the
# model copies from extracted full text into its summary.
PDF_ARTIFACT = re.compile(
    r"arXiv:\s*\d{4}\.\d{4,5}|\[(?:cs|stat|math|eess|q-bio|physics)\.[A-Za-z]{2}\]|\bkeywords?\s*:|"
    r"\b\d\s+introduction\b|\bdoi\s*:|https?://",
    re.IGNORECASE,
)
LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl"}
CONTEXT_ECHO = re.compile(
    r"^(?:\[[^\]]{2,60}, (?:\d{4}|n\.d\.)\]|research topic\s*:|papers?\s*:|summaries of \d+ papers)", re.IGNORECASE
)
# Filler that talks about the prompt instead of the papers ("...from the synthesis above"), and
# instruction-tuning boilerplate TinyLlama leaks ("Based on the text material above, generate...").
META_REFERENCE = re.compile(
    r"\b(?:papers?|synthesis|text(?: material)?|summaries|sources?|literature review) (?:above|provided)\b|"
    r"\b(?:discussed|mentioned|listed|given|shown) above\b|\bpapers? #\d|"
    r"\bgenerate the response\b|\bfollowing (?:question|instruction)\b",
    re.IGNORECASE,
)
# A sentence opening with an imperative is the model restating its instructions, not content.
INSTRUCTION_ECHO = re.compile(
    r"^(?:write|list|summarize|mention|describe|state|produce|generate|compare|use only)\b", re.IGNORECASE
)
MAX_COPIED = 0.6  # share of an output's 6-word phrases copied verbatim from its input
# Words ignored when checking whether generated text is grounded in the source papers.
GROUNDING_STOPWORDS = set(
    "about above after again against among these those their there which while would could should "
    "other being between through during before under further using based paper papers study studies "
    "discusses highlights research also within without".split()
)
MIN_GROUNDING = 0.55  # share of content words that must appear in the source text
# Synthesized sections (synthesis, outline sections) must be written, not pasted: a much lower copy
# ceiling than for paper summaries, so abstracts can't reappear verbatim in several report sections.
SYNTH_MAX_COPIED = 0.3

# Misspellings that show up in reports, either from source abstracts or from the model re-typing
# them. Applied to all generated report text (never to bibliography entries, which must match the
# published titles).
TYPO_FIXES = {
    "algoithm": "algorithm", "algorithim": "algorithm", "alogrithm": "algorithm", "algorthm": "algorithm",
    "automatation": "automation", "accross": "across", "acheive": "achieve",
    "accomodate": "accommodate", "accountibility": "accountability", "anaylsis": "analysis",
    "analsis": "analysis", "aproach": "approach", "apporach": "approach", "artifical": "artificial",
    "artifcial": "artificial", "beleive": "believe", "becuase": "because", "comparision": "comparison",
    "consistant": "consistent", "definately": "definitely", "efficency": "efficiency",
    "enviroment": "environment", "evalution": "evaluation", "existance": "existence",
    "experiement": "experiment", "framwork": "framework", "goverment": "government",
    "governence": "governance", "goverance": "governance", "implmentation": "implementation",
    "infomation": "information", "intelligance": "intelligence", "inteligence": "intelligence",
    "knowlege": "knowledge", "langauge": "language", "lanugage": "language", "learing": "learning",
    "machiene": "machine", "managment": "management", "neccessary": "necessary", "occured": "occurred",
    "occurence": "occurrence", "perfomance": "performance", "preformance": "performance",
    "publically": "publicly", "reasearch": "research", "resarch": "research", "recieve": "receive",
    "recomend": "recommend", "recomendation": "recommendation", "relevent": "relevant",
    "responsibilty": "responsibility", "seperate": "separate", "statisitcs": "statistics",
    "sucessful": "successful", "teh": "the", "thier": "their", "transparancy": "transparency",
    "untill": "until", "wich": "which",
}
TYPO_PATTERN = re.compile(r"\b(" + "|".join(TYPO_FIXES) + r")(s|es|ed|ing|ic|al)?\b", re.IGNORECASE)


@dataclass
class PaperSummary:
    paper: Paper
    summary: str
    key_findings: List[str] = field(default_factory=list)
    from_llm: bool = True  # False when the model output was rejected and the abstract was used
    gap: str = ""  # what the paper leaves unaddressed, for the Results & Analysis table


class OllamaConnectionError(RuntimeError):
    """Raised when the local Ollama server can't be reached or doesn't have the requested model."""


def split_sentences(text: str) -> List[str]:
    """Splits text into sentences, treating each non-empty line (e.g. a bullet) as its own unit."""
    sentences: List[str] = []
    for line in (text or "").splitlines():
        line = line.strip().lstrip("-*•").strip()
        if line:
            sentences.extend(s.strip() for s in SENTENCE_SPLIT.split(line) if s.strip())
    return sentences


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _content_words(text: str) -> set:
    return {w for w in re.findall(r"[a-z]{5,}", (text or "").lower()) if w not in GROUNDING_STOPWORDS}


def _garbled_words(text: str) -> List[str]:
    """Words mixing Latin with look-alike Cyrillic/Greek letters, or several words run together."""
    garbled = []
    for word in re.findall(r"[^\W\d_]+", text or ""):
        mixed_script = re.search(r"[a-zA-Z]", word) and re.search(r"[Ͱ-ϿЀ-ӿ]", word)
        if mixed_script or len(word) > 22:
            garbled.append(word)
    return garbled


def grounding_score(text: str, source: str) -> float:
    """Share of the text's content words that also occur in the source; low means likely invented."""
    words = _content_words(text)
    if not words:
        return 0.0
    return len(words & _content_words(source)) / len(words)


def copied_ratio(text: str, source: str, n: int = 6) -> float:
    """Share of the text's n-word phrases that appear verbatim in the source (1.0 = pure copy)."""
    def shingles(value: str) -> set:
        tokens = re.findall(r"[a-z0-9]+", (value or "").lower())
        return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}

    own = shingles(text)
    return len(own & shingles(source)) / len(own) if own else 0.0


def content_issue(
    text: str, min_sentences: int = 2, min_words: int = 20, source: Optional[str] = None
) -> Optional[str]:
    """Returns why generated text is unusable (empty, placeholder, too short, looping, ungrounded).

    When `source` is given, the text must also share most of its vocabulary with it.
    """
    if not text or not text.strip():
        return "empty response"
    if PLACEHOLDER_PATTERN.search(text):
        return "contains placeholder text"
    if META_REFERENCE.search(text) or any(CONTEXT_ECHO.match(s) for s in split_sentences(text)):
        return "echoes the prompt instead of writing content"
    garbled = _garbled_words(text)
    if len(garbled) >= 2:
        return f"garbled words ({', '.join(garbled[:3])})"
    word_count = len(re.findall(r"[A-Za-z]{2,}", text))
    if word_count < min_words:
        return f"too short ({word_count} words, need {min_words})"
    sentences = split_sentences(text)
    if len(sentences) < min_sentences:
        return f"too few sentences ({len(sentences)}, need {min_sentences})"
    unique = {s.lower() for s in sentences}
    if len(sentences) >= 4 and len(unique) / len(sentences) < 0.6:
        return "repetitive output"
    if source is not None:
        score = grounding_score(text, source)
        if score < MIN_GROUNDING:
            return f"not grounded in the papers ({score:.0%} of key terms found in sources)"
    return None


# Synthesized prose speaks about the papers, never as one of them ("we propose", "our survey").
PAPER_VOICE = re.compile(r"\bwe (?:propose|present|adopt|conclude|attempt|show|find|argue)\b|\bour (?:survey|paper|work|study)\b", re.IGNORECASE)
CITATION = re.compile(r"[\[(]([A-Z][\w'-]+)(?: et al\.?)?,\s*(\d{4}|n\.d\.)[\])]")


def synthesis_issue(text: str, context: str, required_authors: Sequence[str] = ()) -> Optional[str]:
    """Problems specific to synthesized sections: paper voice, invented citations, list fragments.

    `context` is a sources_block(); its "[Surname, Year]" tags are the only valid citations.
    """
    if PAPER_VOICE.search(text):
        return "written in a paper's own voice instead of synthesizing it"
    known = set(CITATION.findall(context))
    invented = [f"{a}, {y}" for a, y in CITATION.findall(text) if known and (a, y) not in known]
    if invented:
        return f"cites papers not in the review ({'; '.join(invented[:2])})"
    if any(re.match(r"^\s*(?:\d+[.)]|[-*•])\s", line) or line.rstrip().endswith(":") for line in text.splitlines()):
        return "contains list fragments instead of prose"
    missing = [a for a in dict.fromkeys(required_authors) if a not in text]
    if missing:
        return f"does not discuss every paper (missing {', '.join(missing)})"
    return None


def _clean_line(line: str) -> str:
    """Strips echoed prompt labels/metadata from one line, keeping any bullet marker."""
    stripped = line.strip()
    bullet = ""
    if stripped[:1] in ("-", "*", "•"):
        bullet, stripped = "- ", stripped.lstrip("-*•").strip()
    stripped = LEADING_LABEL.sub("", stripped)
    if CONTEXT_ECHO.match(stripped):
        return ""
    kept = []
    for sentence in SENTENCE_SPLIT.split(stripped):
        if PDF_ARTIFACT.search(sentence):
            continue
        # A "[Author, Year] Title" context tag echoed mid-line ("...approach. [Bennett, 2021] AI agent's").
        sentence = re.split(r"\s\[[^\]]{2,60}, (?:\d{4}|n\.d\.)\]\s+[A-Z]", sentence)[0]
        if METADATA_LABEL.search(sentence):
            # "Title: X Authors: Y Text: actual sentence" -> keep only what follows the last label.
            tail = re.split(r"\b(?:text|introduction|abstract)\s*:\s*", sentence, flags=re.IGNORECASE)
            sentence = tail[-1] if len(tail) > 1 and not METADATA_LABEL.search(tail[-1]) else ""
        sentence = LEADING_LABEL.sub("", sentence.strip())
        if sentence:
            kept.append(sentence)
    return f"{bullet}{' '.join(kept)}" if kept else ""


def clean_response(text: str) -> str:
    """Drops chatty preambles, echoed prompt fragments and repeated lines from model output."""
    lines: List[str] = []
    seen = set()
    for line in (text or "").splitlines():
        if line.strip():
            line = _clean_line(line)
            if not line:
                continue
        key = line.strip().lower()
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        lines.append(line.rstrip())
    while lines and (not lines[0].strip() or PREAMBLE_PATTERN.match(lines[0].strip())):
        lines.pop(0)
    return _drop_truncated_tail("\n".join(lines).strip())


def _drop_truncated_tail(text: str) -> str:
    """Removes a final sentence cut off by the token limit ("...learning curve models can")."""
    if not text or text[-1] in ".!?)\"'”":
        return text
    head, _, last_line = text.rpartition("\n")
    sentences = SENTENCE_SPLIT.split(last_line)
    if len(sentences) > 1:
        return "\n".join(filter(None, [head, " ".join(sentences[:-1])]))
    return head.strip() if head.strip() else text


def pick_sentences(text: str, keywords: Sequence[str] = (), limit: int = 2, max_len: int = 300) -> List[str]:
    """Picks up to `limit` sentences from text, preferring ones that mention any of `keywords`."""
    sentences = [s for s in split_sentences(normalize_whitespace(text)) if 30 <= len(s) <= max_len]
    if keywords:
        sentences = [s for s in sentences if any(k in s.lower() for k in keywords)]
    return sentences[:limit]


def cite(paper: Paper) -> str:
    return f"({paper.first_author_surname}, {paper.year or 'n.d.'})"


def fix_typos(text: str) -> str:
    """Corrects known misspellings (see TYPO_FIXES), keeping the original capitalization."""
    def replace(match: re.Match) -> str:
        word, suffix = match.group(1), match.group(2) or ""
        fixed = TYPO_FIXES[word.lower()]
        if word.isupper() and len(word) > 1:
            fixed = fixed.upper()
        elif word[0].isupper():
            fixed = fixed[0].upper() + fixed[1:]
        return fixed + suffix

    for ligature, letters in LIGATURES.items():
        text = (text or "").replace(ligature, letters)
    return TYPO_PATTERN.sub(replace, text or "")


def fix_against_source(text: str, source: str) -> str:
    """Restores the source's spelling of words the model mistyped while paraphrasing it.

    A long word missing from the source but nearly identical to one in it ("Logiciest" next to
    "Logicist") is a retyping error. Words differing only in their ending ("evaluated" vs
    "evaluates") are left alone, since those are legitimate inflections.
    """
    vocab = {w.lower() for w in re.findall(r"[A-Za-z]{6,}", source or "")}

    def replace(match: re.Match) -> str:
        word = match.group(0)
        lower = word.lower()
        if lower in vocab:
            return word
        close = difflib.get_close_matches(lower, vocab, n=1, cutoff=0.88)
        if not close:
            return word
        candidate = close[0]
        shared = len(os.path.commonprefix([lower, candidate]))
        if shared >= min(len(lower), len(candidate)) - 1:
            return word
        return candidate.capitalize() if word[0].isupper() else candidate

    return re.sub(r"[A-Za-z]{7,}", replace, text or "")


# Rough classification of how a paper does its research, from its title and abstract. Used to
# compare methods across papers when the model can't, instead of pasting method sentences.
METHOD_PROFILES = (
    ("surveys or reviews of prior work", ("survey", "systematic review", "literature review", "scoping review", "we review", "overview of")),
    ("case studies", ("case study", "case studies")),
    ("empirical studies", ("experiment", "empirical", "dataset", "we evaluate", "interview", "questionnaire", "participants", "benchmark", "we measure")),
    ("conceptual frameworks or recommendations", ("framework", "recommendation", "guideline", "principle", "checklist", "taxonomy", "we propose", "we argue", "position")),
)
DEFAULT_METHOD_PROFILE = "analytical discussions"
METHOD_LABELS = {
    "surveys or reviews of prior work": "Survey / review",
    "case studies": "Case study",
    "empirical studies": "Empirical study",
    "conceptual frameworks or recommendations": "Conceptual framework",
    DEFAULT_METHOD_PROFILE: "Analytical discussion",
}
# What each research design cannot show; the default per-paper gap when the model names none.
METHOD_GAPS = {
    "surveys or reviews of prior work": "Synthesizes earlier work without contributing new evidence of its own.",
    "case studies": "Draws on a single case, so its findings may not generalize to other settings.",
    "empirical studies": "Findings are tied to the specific data and setting studied; transfer to other contexts is untested.",
    "conceptual frameworks or recommendations": "The proposed framework has not been validated empirically.",
    DEFAULT_METHOD_PROFILE: "Its argument is not tested against empirical data.",
}
LIMITATION_KEYWORDS = ("limitation", "remain", "open question", "challenge", "lack", "unclear", "not yet", "little is known")
THEME_STOPWORDS = GROUNDING_STOPWORDS | set(
    "provide provides provided present presents propose proposes important different number several "
    "often first including recent however various given makes making while where whose because "
    "approach approaches result results method methods findings work works field fields author authors "
    "abstract article articles report reports issue issues number order level levels understanding "
    "potential possible particular current further across toward towards along since among therefore "
    "thus overall shows shown known become becomes increasingly general specific single multiple conclude concludes construct constructs "
    "suggest suggests describe describes discuss define defines explore explores".split()
)


def method_profile(paper: Paper) -> str:
    text = f"{paper.title} {paper.abstract}".lower()
    for label, keywords in METHOD_PROFILES:
        if any(k in text for k in keywords):
            return label
    return DEFAULT_METHOD_PROFILE


def method_label(paper: Paper) -> str:
    return METHOD_LABELS[method_profile(paper)]


def default_paper_gap(paper: Paper) -> str:
    """A paper's stated limitation from its abstract, or what its research design cannot show."""
    for sentence in pick_sentences(paper.abstract, LIMITATION_KEYWORDS, limit=3, max_len=220):
        if not FUTURE_WORK_REMARK.search(sentence):
            return sentence
    return METHOD_GAPS[method_profile(paper)]


def paper_focus(paper: Paper) -> str:
    """The paper's subject as a lower-case noun phrase taken from its title ("learning curves for...")."""
    title = normalize_whitespace(paper.title).rstrip(".")
    head, sep, tail = title.partition(":")
    if sep and tail.strip():
        title = tail.strip() if len(head.split()) <= 2 else head.strip()
    words = [w.lower() if w[:1].isupper() and w[1:].islower() else w for w in title.split()]
    return " ".join(words)


def join_items(items: Sequence[str]) -> str:
    items = list(items)
    if len(items) <= 2:
        return " and ".join(items)
    return f"{', '.join(items[:-1])} and {items[-1]}"


def shared_themes(summaries: Sequence[PaperSummary], exclude: str = "", limit: int = 4) -> List[str]:
    """Content words used by at least two papers (or the most frequent ones for a single paper)."""
    excluded = set(re.findall(r"[a-z]+", exclude.lower()))
    per_paper = []
    for s in summaries:
        text = f"{s.paper.title} {s.paper.abstract} {s.summary}".lower()
        per_paper.append(Counter(w for w in re.findall(r"[a-z]{5,}", text) if w not in THEME_STOPWORDS | excluded))
    if not per_paper:
        return []
    papers_using = Counter(w for counts in per_paper for w in counts)
    totals = sum(per_paper, Counter())
    min_papers = 2 if len(per_paper) > 1 else 1
    candidates = [w for w, n in papers_using.items() if n >= min_papers]
    candidates.sort(key=lambda w: (papers_using[w], totals[w]), reverse=True)
    return candidates[:limit]


GAP_ITEM = re.compile(r"^(?:[-*•]|\d+[.)])\s+(.*)$")
# Titles the model sometimes copies from the prompt's format instead of naming the gap.
PLACEHOLDER_GAP_TITLES = {"short name", "name", "title", "short title", "gap", "gap name", "gap title", "research gap"}
# Openings that say a gap exists without naming it ("There is little research on ...").
GAP_TITLE_FILLER = re.compile(
    r"^(?:"
    r"there\s+(?:is|are)\s+(?:a\s+)?(?:little|no|limited|few|scant|insufficient)\s+(?:\w+\s+)?(?:on|of|into|about)\s+"
    r"|little\s+is\s+known\s+about\s+"
    r"|(?:few|no)\s+(?:studies|papers|works)\s+(?:have\s+)?\w+\s+"
    r"|the\s+(?:reviewed\s+)?(?:papers|literature|studies)\s+(?:do\s+not|does\s+not|rarely|seldom|lacks?)\s+(?:\w+\s+)?"
    r"|while\s+|although\s+|despite\s+"
    r")",
    re.IGNORECASE,
)
GAP_TITLE_TRAILING = {"a", "an", "the", "of", "and", "or", "to", "in", "on", "for", "with", "by", "at", "from", "as", "is", "are", "that"}


def gap_title(description: str, max_words: int = 6) -> str:
    """A short title for a gap, taken from the opening words of its description."""
    sentences = split_sentences(description)
    sentence = GAP_TITLE_FILLER.sub("", sentences[0] if sentences else description or "")
    words = re.findall(r"[A-Za-z0-9][\w'’-]*", sentence)[:max_words]
    while words and words[-1].lower() in GAP_TITLE_TRAILING:
        words.pop()
    if not words:
        return ""
    title = " ".join(words)
    return title[0].upper() + title[1:]


def parse_gaps(text: str) -> List[Tuple[str, str]]:
    """Splits a research-gaps list into (title, description) pairs; title may be "".

    A missing or placeholder title ("Short name") is replaced by one built from the description.
    """
    gaps: List[List[str]] = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        item = GAP_ITEM.match(stripped)
        if item:
            gaps.append([item.group(1)])
        elif gaps:
            gaps[-1].append(stripped)  # description continued on the next line
    parsed = []
    for parts in gaps:
        body = normalize_whitespace(" ".join(parts).replace("**", ""))
        body = re.sub(r"^(?:research\s+)?gap\s*\d*\s*[:.)-]\s*", "", body, flags=re.IGNORECASE)
        title, sep, rest = body.partition(":")
        if sep and rest.strip() and len(title.split()) <= 12 and not re.search(r"[.!?]", title):
            title, desc = title.strip(), rest.strip()
        else:
            title, desc = "", body
        if not title or title.lower().strip(" '\"") in PLACEHOLDER_GAP_TITLES:
            title = gap_title(desc)
        parsed.append((title, desc))
    return parsed


def format_gaps(gaps: Sequence[Tuple[str, str]]) -> str:
    return "\n".join(f"- **{title}**: {desc}" if title else f"- {desc}" for title, desc in gaps)


def sources_block(summaries: Iterable[PaperSummary], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """Compact per-paper context that fits a small model's context window regardless of paper count."""
    summaries = list(summaries)
    if not summaries:
        return ""
    per_paper = max(200, max_chars // len(summaries))
    entries = []
    for s in summaries:
        body = s.summary or s.paper.abstract
        if s.key_findings:
            body += " Findings: " + "; ".join(s.key_findings)
        tag = f"[{s.paper.first_author_surname}, {s.paper.year or 'n.d.'}]"
        entries.append(f"{tag} {s.paper.title}\n{normalize_whitespace(body)}"[:per_paper])
    return "\n\n".join(entries)


class LLMSynthesizer:
    """Thin wrapper around a local Ollama server's chat endpoint for research synthesis tasks.

    Use as a context manager (or call unload() explicitly when done) so the model is evicted from
    RAM as soon as a run finishes, rather than sitting loaded for Ollama's keep-alive window while
    idle:

        with LLMSynthesizer() as synthesizer:
            ...
        # model has been unloaded from RAM here
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: Optional[str] = None,
        keep_alive: str = DEFAULT_KEEP_ALIVE,
        session: Optional[requests.Session] = None,
    ):
        self.model = model
        self.base_url = (base_url or os.environ.get("OLLAMA_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.keep_alive = keep_alive
        self.session = session or requests.Session()
        small = model.split(":")[0].lower() in SMALL_CONTEXT_MODELS
        self.num_ctx = NUM_CTX if small else LARGE_NUM_CTX
        self.source_chars = MAX_SOURCE_CHARS if small else LARGE_SOURCE_CHARS
        self.context_chars = MAX_CONTEXT_CHARS if small else LARGE_CONTEXT_CHARS
        self._verify_server()

    def __enter__(self) -> "LLMSynthesizer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.unload()

    def _verify_server(self) -> None:
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
        except requests.exceptions.ConnectionError as exc:
            raise OllamaConnectionError(
                f"Could not reach Ollama at {self.base_url}. Start it with 'ollama serve' "
                f"(or install it from https://ollama.com) and make sure the model is pulled: "
                f"'ollama pull {self.model}'."
            ) from exc
        except requests.RequestException as exc:
            logger.debug("Ollama server check at %s failed non-fatally: %s", self.base_url, exc)

    def complete(self, system: str, user: str, max_tokens: int = 400, temperature: float = 0.4) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "num_predict": max_tokens,
                "num_ctx": self.num_ctx,
                "temperature": temperature,
                "repeat_penalty": 1.15,  # discourages the "Step 1... Step 1..." loops TinyLlama falls into
            },
        }
        try:
            response = self.session.post(f"{self.base_url}/api/chat", json=payload, timeout=REQUEST_TIMEOUT)
        except requests.exceptions.ConnectionError as exc:
            raise OllamaConnectionError(
                f"Lost connection to Ollama at {self.base_url}. Is 'ollama serve' still running?"
            ) from exc

        if response.status_code == 404:
            raise OllamaConnectionError(
                f"Model '{self.model}' is not available on the Ollama server at {self.base_url}. "
                f"Pull it first with: ollama pull {self.model}"
            )
        response.raise_for_status()
        data = response.json()
        return (data.get("message", {}).get("content") or "").strip()

    def complete_validated(
        self,
        system: str,
        user: str,
        max_tokens: int = 400,
        min_sentences: int = 2,
        min_words: int = 20,
        retries: int = MAX_RETRIES,
        validator: Optional[Callable[[str], Optional[str]]] = None,
        label: str = "response",
        source: Optional[str] = None,
    ) -> str:
        """Like complete(), but retries weak output and returns "" if no attempt passes validation.

        `validator` overrides the default content_issue() check; it returns a problem description
        or None. `source` is the text the output must be grounded in. Callers decide what to fall
        back to when this returns "".
        """
        validator = validator or (lambda text: content_issue(text, min_sentences, min_words, source))
        prompt = user
        task = user.rsplit("Task:", 1)[-1]
        for attempt in range(retries + 1):
            raw = clean_response(self.complete(system, prompt, max_tokens, temperature=0.4 + 0.3 * attempt))
            raw = self._strip_echoed_instructions(raw, task)
            problem = validator(raw)
            if problem is None:
                return raw
            logger.warning("Weak %s from model (attempt %d/%d): %s", label, attempt + 1, retries + 1, problem)
            logger.debug("Rejected %s: %r", label, raw)
            prompt = (
                f"{user}\n\nIMPORTANT: Your previous answer was rejected ({problem}). Write at least "
                f"{min_sentences} complete sentences of real content based only on the papers above. "
                "Do not use placeholder text such as 'Lorem ipsum'."
            )
        return ""

    @staticmethod
    def _strip_echoed_instructions(text: str, task: str) -> str:
        """Removes sentences in which the model repeats (or paraphrases) the task instruction back."""
        lines = []
        for line in text.splitlines():
            kept = [
                s
                for s in SENTENCE_SPLIT.split(line)
                if copied_ratio(s, task, n=4) < 0.5 and not INSTRUCTION_ECHO.match(s.lstrip("-*• "))
            ]
            if kept or not line.strip():
                lines.append(" ".join(kept))
        return "\n".join(lines).strip()

    def unload(self) -> None:
        """Tells Ollama to evict this model from memory immediately (keep_alive=0).

        Best-effort: failures are logged, not raised, since this typically runs during cleanup.
        """
        try:
            self.session.post(
                f"{self.base_url}/api/chat",
                json={"model": self.model, "messages": [], "stream": False, "keep_alive": 0},
                timeout=10,
            )
            logger.info("Unloaded Ollama model %s from memory", self.model)
        except requests.RequestException as exc:
            logger.debug("Could not explicitly unload model %s: %s", self.model, exc)

    def summarize_paper(self, paper: Paper, full_text: str) -> PaperSummary:
        # Abstract first: it is the densest description of the paper and survives truncation.
        source_text = normalize_whitespace(paper.abstract)
        if full_text and normalize_whitespace(full_text) != source_text:
            source_text = f"{source_text}\n\n{normalize_whitespace(full_text)}"
        source_text = source_text[: self.source_chars]
        system = (
            "You are a research assistant. Summarize academic papers accurately and concisely. "
            "Only state facts found in the provided text. Never use placeholder text."
        )
        user = (
            f"Paper title: {paper.title}\n"
            f"Authors: {', '.join(paper.authors[:5]) or 'Unknown'}\n"
            f"Year: {paper.year or 'Unknown'}\n\n"
            f"Text:\n{source_text}\n\n"
            "Task: First write a summary of exactly 3 sentences: (1) the problem the paper studies, "
            "(2) its method or approach, (3) its main result. Then write 3 key findings, each on its "
            "own line starting with '- '. Finally write one line starting with 'Gap: ' that states, in "
            "one sentence, an important question or limitation the paper leaves unaddressed."
        )
        raw = self.complete_validated(
            system,
            user,
            max_tokens=350,
            validator=lambda text: content_issue(self._split_summary_and_findings(text)[0], 2, 20, source_text),
            label=f"summary for {paper.title[:50]!r}",
        )
        if raw:
            raw, gap = self._split_gap(raw, source_text)
            summary, findings = self._split_summary_and_findings(raw)
            # TinyLlama often re-types the abstract instead of summarizing it, adding typos as it
            # goes; the clean abstract sentences are strictly better than such a copy.
            if copied_ratio(summary, source_text) > MAX_COPIED:
                logger.info("Model copied the source text for %r instead of summarizing it", paper.title)
                return self.fallback_summary(paper, full_text)
            summary = self._strip_title_and_refs(summary, paper.title)
            summary = fix_against_source(" ".join(split_sentences(summary)[:4]), source_text)
            findings = [
                fix_against_source(f, source_text)
                for f in findings
                # questions and bare headings ("Problem Formulation") aren't findings
                if not PLACEHOLDER_PATTERN.search(f) and not f.rstrip().endswith("?") and len(f.split()) >= 4
            ][:5]
            return PaperSummary(paper=paper, summary=summary, key_findings=findings, gap=gap or default_paper_gap(paper))
        return self.fallback_summary(paper, full_text)

    @staticmethod
    def _split_gap(raw: str, source_text: str) -> Tuple[str, str]:
        """Separates the 'Gap: ...' line from the summary; the gap is "" if missing or unusable."""
        kept, gap = [], ""
        for line in raw.splitlines():
            match = re.match(r"^[-*•\s]*(?:research\s+)?gap\s*:\s*(.*)$", line.strip(), re.IGNORECASE)
            if match:
                gap = gap or normalize_whitespace(match.group(1))
            else:
                kept.append(line)
        if gap and (len(gap.split()) < 6 or PLACEHOLDER_PATTERN.search(gap) or grounding_score(gap, source_text) < 0.4):
            gap = ""
        return "\n".join(kept), fix_against_source(split_sentences(gap)[0], source_text) if gap else ""

    @staticmethod
    def fallback_summary(paper: Paper, full_text: str = "") -> PaperSummary:
        """Builds a summary straight from the abstract (or paper text) when the model fails."""
        source = paper.abstract or full_text[:MAX_SOURCE_CHARS]
        summary = " ".join(pick_sentences(source, limit=3)) or normalize_whitespace(source)[:500]
        findings = [
            f
            for f in pick_sentences(
                source,
                ("we show", "we find", "results", "demonstrate", "outperform", "suggest", "reveal", "found"),
                limit=3,
            )
            if not FUTURE_WORK_REMARK.search(f)
        ]
        logger.info("Using abstract-based summary for %r", paper.title)
        return PaperSummary(paper=paper, summary=summary, key_findings=findings, from_llm=False, gap=default_paper_gap(paper))

    @staticmethod
    def _strip_title_and_refs(text: str, title: str) -> str:
        """Removes an echoed paper title at the start and numeric reference markers ("[18]")."""
        text = re.sub(r"\s?\[\d+(?:\s*[,–-]\s*\d+)*\]", "", text)
        if normalize_whitespace(text).lower().startswith(normalize_whitespace(title).lower()):
            text = normalize_whitespace(text)[len(normalize_whitespace(title)):].lstrip(" :.-")
        return text

    @staticmethod
    def _split_summary_and_findings(raw: str) -> Tuple[str, List[str]]:
        summary_lines: List[str] = []
        findings: List[str] = []
        for line in raw.splitlines():
            stripped = line.strip()
            if stripped.startswith(("-", "*", "•")):
                findings.append(stripped.lstrip("-*•").strip())
            elif re.match(r"^\d+[.)]\s", stripped) and summary_lines:
                findings.append(re.sub(r"^\d+[.)]\s*", "", stripped))
            elif stripped and not re.match(r"^(summary|key findings?)\s*:?\s*$", stripped, re.IGNORECASE):
                summary_lines.append(re.sub(r"^summary\s*:\s*", "", stripped, flags=re.IGNORECASE))
        return " ".join(summary_lines), [f for f in findings if f]

    def synthesize_literature(self, topic: str, summaries: List[PaperSummary]) -> str:
        system = (
            "You are a research assistant writing the literature synthesis of a research paper. "
            "Use only facts from the provided paper summaries and cite them inline as (Author, Year)."
        )
        user = (
            f"Research topic: {topic}\n\n"
            f"Summaries of {len(summaries)} papers:\n\n{sources_block(summaries, self.context_chars)}\n\n"
            "Task: Write 2-3 paragraphs (at least 6 sentences in total) organized by theme, not paper "
            "by paper: what themes the papers share, how their methods differ, where they agree or "
            "disagree, and what remains open. Use your own words; do not copy sentences from the "
            "summaries. Cite every paper by its (Author, Year) tag."
        )
        context = sources_block(summaries, self.context_chars)

        def validate(text: str) -> Optional[str]:
            if copied_ratio(text, context) > SYNTH_MAX_COPIED:
                return "copied the summaries instead of synthesizing them"
            authors = [s.paper.first_author_surname for s in summaries]
            return synthesis_issue(text, context, authors) or content_issue(text, min_sentences=4, min_words=60, source=context)

        text = self.complete_validated(
            system, user, max_tokens=600, min_sentences=4, validator=validate, label="literature synthesis"
        )
        return text or self.fallback_synthesis(topic, summaries)

    @staticmethod
    def fallback_synthesis(topic: str, summaries: List[PaperSummary]) -> str:
        """Cross-paper comparison built from titles, themes and methods.

        Deliberately avoids quoting abstract or summary sentences, which the report already shows
        under Paper Summaries.
        """
        logger.info("Building literature synthesis from paper themes and methods")
        years = sorted(s.paper.year for s in summaries if s.paper.year)
        span = ""
        if years:
            span = f", published in {years[0]}," if years[0] == years[-1] else f", published between {years[0]} and {years[-1]},"
        angles = join_items(f"{s.paper.first_author_surname} ({s.paper.year or 'n.d.'}) examines {paper_focus(s.paper)}" for s in summaries)
        paragraphs = [f"The {len(summaries)} papers reviewed{span} approach {topic} from different angles: {angles}."]

        themes = shared_themes(summaries, exclude=topic)
        if themes:
            paragraphs[0] += (
                f" Despite these different starting points, they share a vocabulary of {join_items(themes)}, "
                "which marks these as the concerns that connect otherwise separate lines of work."
            )
        else:
            paragraphs[0] += " The papers share little common vocabulary, which suggests the field is still fragmented."

        groups: dict = {}
        for s in summaries:
            groups.setdefault(method_profile(s.paper), []).append(f"{s.paper.first_author_surname}, {s.paper.year or 'n.d.'}")
        if len(groups) == 1:
            label, cites = next(iter(groups.items()))
            methods = (
                f"Methodologically the literature is homogeneous: all {len(summaries)} papers are {label} "
                f"({'; '.join(cites)}), so their conclusions have not yet been tested with other kinds of evidence."
            )
        else:
            methods = (
                "Methodologically the work is varied, spanning "
                + join_items(f"{label} ({'; '.join(cites)})" for label, cites in groups.items())
                + ". This mix means the papers support each other only indirectly: their claims rest on "
                "different kinds of evidence and are rarely evaluated against one another."
            )
        paragraphs.append(methods)
        paragraphs.append(
            f"Taken together, the literature treats {themes[0] if themes else topic} as a central concern for "
            f"{topic}, but it leaves several questions open, which are described in the Discussion under Research Gaps."
        )
        return "\n\n".join(paragraphs)

    def generate_research_gaps(self, topic: str, synthesis: str, summaries: Sequence[PaperSummary] = ()) -> str:
        """Returns gaps as '- **Title**: 2-3 sentence description' lines (see parse_gaps)."""
        context = sources_block(summaries, self.context_chars) or synthesis[: self.context_chars]
        system = "You are a research assistant identifying what is missing from a body of literature."
        user = (
            f"Research topic: {topic}\n\nPapers:\n{context}\n\n"
            "Task: Identify 3 research gaps: questions these papers leave unanswered. Write each gap on "
            "its own line as '- <Title>: <description>', where <Title> is a 3-6 word noun phrase naming "
            "the gap's main concept (for example 'Long-term deployment outcomes') and the description is "
            "2-3 sentences explaining what is missing from the literature and why it matters. Do not "
            "write the words 'Title' or 'Short name' literally. Do not list paper titles."
        )

        def usable(text: str) -> List[Tuple[str, str]]:
            return [
                (title, desc)
                for title, desc in parse_gaps(text)
                if len(split_sentences(desc)) >= 2
                and len(desc.split()) >= 20
                and not re.match(r'^"[^"]+"\s*(?:by\b|\()', desc)  # a paper citation is not a gap
                and copied_ratio(desc, context) <= SYNTH_MAX_COPIED
            ]

        def validate(text: str) -> Optional[str]:
            found = usable(text)
            if len(found) < 2:
                return f"only {len(found)} gap descriptions of 2+ sentences"
            # Grounded against the papers, not the synthesis: the synthesis is model output too.
            return content_issue("\n".join(d for _, d in found), 2, 40, source=context)

        text = self.complete_validated(system, user, max_tokens=450, validator=validate, label="research gaps")
        if text:
            return format_gaps(usable(text)[:4])
        return self.fallback_research_gaps(topic, summaries)

    @staticmethod
    def fallback_research_gaps(topic: str, summaries: Sequence[PaperSummary]) -> str:
        """Gap descriptions derived from what the papers state and from what the set as a whole lacks."""
        logger.info("Building research gaps from the papers' stated limitations, scope and methods")
        summaries = list(summaries)
        if not summaries:
            return format_gaps([(
                "Evidence base",
                f"No papers could be summarized for {topic}, so the state of the literature is unknown. "
                "A systematic search across more sources is needed before gaps can be identified.",
            )])
        gaps: List[Tuple[str, str]] = []

        for s in summaries:
            for sentence in pick_sentences(s.paper.abstract, LIMITATION_KEYWORDS, limit=1):
                gaps.append((
                    f"Open problem raised by {s.paper.first_author_surname} ({s.paper.year or 'n.d.'})",
                    f"{s.paper.first_author_surname} ({s.paper.year or 'n.d.'}) flags an unresolved issue: "
                    f"\"{sentence.rstrip('.')}.\" None of the reviewed papers resolves it, so it remains an "
                    f"open question for {topic} that targeted follow-up studies should address.",
                ))
                break
            if len(gaps) >= 2:
                break

        profiles = {s.paper.first_author_surname: method_profile(s.paper) for s in summaries}
        non_empirical = [s for s in summaries if profiles[s.paper.first_author_surname] != "empirical studies"]
        if non_empirical:
            cites = "; ".join(f"{s.paper.first_author_surname}, {s.paper.year or 'n.d.'}" for s in non_empirical)
            gaps.append((
                "Empirical validation",
                f"Most of the reviewed work consists of {join_items(sorted({profiles[s.paper.first_author_surname] for s in non_empirical}))} "
                f"({cites}) rather than real-world evaluation. Whether the proposed approaches actually work in "
                f"practice for {topic} is therefore largely untested. Deployment studies and measurable outcome "
                "data are needed to confirm or refute their claims.",
            ))
        else:
            gaps.append((
                "Long-term evidence",
                f"The empirical studies reviewed capture {topic} at a single point in time. How their findings "
                "hold up as technology, institutions and practices change is unknown, which calls for "
                "longitudinal studies that track outcomes over several years.",
            ))

        if len(summaries) > 1:
            gaps.append((
                "Integration across research strands",
                f"The papers study {join_items(paper_focus(s.paper) for s in summaries[:4])} largely in isolation, "
                "each with its own framing and evidence. There is little work connecting these strands, so it is "
                f"unclear how progress in one affects the others. A unifying framework or comparative study for {topic} "
                "would make the existing findings easier to build on.",
            ))

        newest = max((s.paper.year for s in summaries if s.paper.year), default=None)
        gaps.append((
            "Generalisability across contexts",
            f"Each reviewed paper draws its conclusions from a specific setting, and none compares several "
            f"settings side by side. It is therefore unclear whether the findings transfer to other countries, "
            f"sectors or populations"
            + (f", or whether they still hold given developments since {newest}" if newest else "")
            + ". Cross-context replication would show which results are general and which are local.",
        ))
        return format_gaps(gaps[:4])

    def find_tensions(self, topic: str, summaries: Sequence[PaperSummary], limit: int = 3) -> List[Tuple[str, str, str]]:
        """Asks the model where papers disagree; returns (paper A, paper B, point of tension).

        Each item must cite two different papers of the review. Returns [] when the model finds none
        (or invents citations); callers then fall back to pairs worth comparing.
        """
        if len(summaries) < 2:
            return []
        context = sources_block(summaries, self.context_chars)
        known = set(CITATION.findall(context))
        system = "You are a research assistant comparing the findings of academic papers."
        user = (
            f"Research topic: {topic}\n\nPapers:\n{context}\n\n"
            "Task: Find up to 3 places where two of these papers disagree, contradict each other, or reach "
            "different conclusions about the same issue. Write each on its own line as "
            "'- (Author, Year) vs (Author, Year): one sentence naming the issue and how their positions differ'. "
            "Use only the (Author, Year) tags of the papers above. If no papers disagree, write 'None'."
        )

        def parse(text: str) -> List[Tuple[str, str, str]]:
            found = []
            for line in text.splitlines():
                cited = list(dict.fromkeys(CITATION.findall(line)))
                if len(cited) < 2 or not all(c in known for c in cited[:2]):
                    continue
                point = normalize_whitespace(re.split(r":\s", line, maxsplit=1)[-1])
                if len(point.split()) >= 8 and not PLACEHOLDER_PATTERN.search(point):
                    (a1, y1), (a2, y2) = cited[:2]
                    found.append((f"{a1} ({y1})", f"{a2} ({y2})", point.rstrip(".") + "."))
            return found

        tensions = parse(clean_response(self.complete(system, user, max_tokens=300, temperature=0.3)))
        if not tensions:
            logger.info("Model identified no grounded disagreements between papers")
        return tensions[:limit]

    def summarize_all(self, papers: List[Paper], extractor: TextExtractor, full_text: bool = True) -> List[PaperSummary]:
        summaries = []
        for index, paper in enumerate(papers, 1):
            logger.info("Summarizing paper %d/%d: %s", index, len(papers), paper.title[:80])
            try:
                text = extractor.extract_text_for_paper(paper) if full_text else paper.abstract
                summaries.append(self.summarize_paper(paper, text))
            except OllamaConnectionError:
                raise
            except Exception as exc:
                logger.warning("Skipping summary for %r due to error: %s", paper.title, exc)
        return summaries
