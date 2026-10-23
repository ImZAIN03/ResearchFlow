"""Uses a local Ollama model to summarize papers and synthesize findings across a literature set.

Runs entirely against a local Ollama server (https://ollama.com) instead of a hosted API, so no
API key or network egress is required beyond http://localhost:11434. Defaults to Mistral (7B),
which writes far more coherent, better-grounded prose than the small models; on low-RAM machines
"tinyllama" (~1.1B) or "phi" (Phi-2, ~2.7B) still work. Pull whichever you use first:
`ollama pull mistral`.

Small models regularly return empty, truncated, looping or placeholder ("Lorem ipsum...") text, so
every generation is validated with content_issue() and retried once with a stricter prompt.
Synthesized sections are held to a higher bar (prose_issue()): prose only, no vague filler, and 80% of
their key terms grounded in the papers. If the model still can't produce usable text, the methods
degrade in steps rather than pasting abstracts: a paper summary becomes a hybrid of the abstract, the
paper's metadata and its keywords; the synthesis is written paragraph by paragraph from a structured
template (themes, methods, agreements and contradictions, open questions) and only then assembled
from the papers' themes; research gaps are expanded from each paper's stated limitations.
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
# Synthesized sections (synthesis, introduction, discussion, conclusion) must stay much closer to the
# papers than a summary needs to; sections below REVIEW_GROUNDING are flagged for the user to check.
SYNTH_MIN_GROUNDING = 0.80
REVIEW_GROUNDING = 0.70
MIN_SUMMARY_SENTENCES = 3  # contribution, limitation and relevance at the least
MAX_SUMMARY_SENTENCES = 4
MIN_GAP_SENTENCES = 2
# Academic connective vocabulary a synthesis needs that the papers themselves rarely use; ignored when
# grounding synthesized prose, so the 80% bar measures claims rather than connective tissue.
SYNTHESIS_VOCABULARY = set(
    "agree agreement agrees contrast contrasts contrasting similarly likewise whereas differ differs "
    "differing difference differences diverge diverges divergent converge converges convergent "
    "together taken theme themes thematic strand strands methodological methodologically evidence "
    "empirical conceptual literature reviewed review reviews existing remains remain unresolved "
    "unanswered unclear question questions contradiction contradictions contradict contradicts "
    "tension tensions consensus collectively combined jointly alongside rather instead whether "
    "despite indicate indicates suggesting support supports supported limited limitation limitations "
    "scope focus focuses focused examine examines examined address addresses addressed body "
    # argument and comparison vocabulary of thesis prose (and of this module's templates)
    "actually affects along angles another apparent argument arguments bears because build cannot claims "
    "clearly closing comes comparative concern concerns conclusions confirm connect connecting connects "
    "consists current define designed design designs direct directly disagreements discussion "
    "discussions easier empirically establish establishes evaluated evaluation extending firmer following "
    "footing frames frameworks framing guide included indirectly integration isolation issues kinds "
    "largely leaves leaving limits lines little makes marks matter means measurable mixed narrative offer "
    "others otherwise outcome outcomes points practice pressing progress proposed rarely recommendations "
    "reflect refute resolves separate setting settings settled shape share shared short spanning starting "
    "studied substance surveys tested tests therefore transfer untested until validated validation varied "
    "vocabulary whose strength strengths weakness weaknesses robust tentative causal consistent "
    "inconsistent broader narrower wider generalize generalise generalisability generalizability".split()
)
# Filler that signals a section says nothing specific ("various aspects are important to discuss").
VAGUE_LANGUAGE = re.compile(r"\b(?:discuss(?:es|ed|ing)?|important|various)\b", re.IGNORECASE)

# The literature synthesis follows this structure, both in the one-shot prompt and when it is written
# (or templated) paragraph by paragraph: (name, what the paragraph covers).
SYNTHESIS_STRUCTURE = (
    ("themes", "the two or three themes the papers share, saying which papers take up each theme."),
    ("methods", "how the papers' research designs and evidence differ, and what that means for the strength of their claims."),
    ("agreements and contradictions", "where the papers reach the same conclusion and where they disagree or pull in different directions."),
    ("open questions", "what the papers, read together, leave unresolved, drawing on their stated limitations."),
)
SYNTHESIS_RULES = (
    "Rules: write continuous prose with at least 2 sentences per paragraph; no headings, bullet points "
    "or paragraph labels; write about the papers in the third person (never 'we' or 'our'); name "
    "specific concepts instead of vague words such as 'various', 'important' or 'discuss'; "
    "paraphrase, never copy sentences from the summaries."
)
SYNTHESIS_EXAMPLE = (
    "Example of the expected style (different topic, do not reuse it): \"Two concerns run through this "
    "literature. (Okafor, 2019) and (Brandt, 2021) both trace slow vaccine uptake to distrust of health "
    "agencies, whereas (Ruiz, 2020) attributes it to the distance patients travel to clinics.\""
)
PARAGRAPH_LABEL = re.compile(
    r"^(?:\*\*)?(?:paragraph\s*\d+|themes?|methods?|agreements?(?: and contradictions)?|contradictions?|"
    r"open questions?)(?:\s*\([^)]*\))?(?:\*\*)?\s*[:.\-–]\s*(?:\*\*)?",
    re.IGNORECASE,
)

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


def synthesis_grounding(text: str, source: str) -> float:
    """grounding_score() for synthesized prose, ignoring the connective vocabulary a synthesis needs."""
    words = _content_words(text) - SYNTHESIS_VOCABULARY
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
    fragments = fragment_issue(text)
    if fragments:
        return fragments
    missing = [a for a in dict.fromkeys(required_authors) if a not in text]
    if missing:
        return f"does not discuss every paper (missing {', '.join(missing)})"
    return None


def fragment_issue(text: str) -> Optional[str]:
    """Rejects anything that isn't continuous prose: bullets, numbered items, headings, labels, fragments."""
    for line in (text or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"^(?:\d+[.)]|[-*•])\s", stripped) or stripped.endswith(":"):
            return "contains list fragments instead of prose"
        if stripped.startswith("#") or re.fullmatch(r"\*\*[^*]+\*\*\.?", stripped):
            return "contains headings instead of prose"
        if stripped[-1] not in ".!?)\"'”":
            return f"contains a sentence fragment ({stripped[:40]!r})"
    short = [s for s in split_sentences(text) if len(re.findall(r"[A-Za-z]+", s)) < 4]
    if short:
        return f"contains a sentence fragment ({short[0][:40]!r})"
    return None


def prose_issue(text: str, source: Optional[str] = None, min_grounding: float = SYNTH_MIN_GROUNDING) -> Optional[str]:
    """Checks for synthesized sections: prose only, no vague filler, and grounded in `source`."""
    problem = fragment_issue(text)
    if problem:
        return problem
    vague = sorted({m.group(0).lower() for m in VAGUE_LANGUAGE.finditer(text or "")})
    if vague:
        return f"uses vague placeholder language ({', '.join(vague)})"
    if source is not None:
        score = synthesis_grounding(text, source)
        if score < min_grounding:
            return f"not grounded in the papers ({score:.0%} of key terms found in sources, need {min_grounding:.0%})"
    return None


def ensure_min_sentences(text: str, extra: Sequence[str], minimum: int = MIN_SUMMARY_SENTENCES) -> str:
    """Tops `text` up with sentences from `extra` (skipping near-repeats) until it has `minimum` sentences."""
    sentences = split_sentences(text)
    seen = [set(re.findall(r"[a-z]+", s.lower())) for s in sentences]
    for candidate in extra:
        if len(sentences) >= minimum:
            break
        words = set(re.findall(r"[a-z]+", candidate.lower()))
        if not words or any(len(words & other) / max(1, len(words | other)) > 0.6 for other in seen):
            continue
        sentences.append(candidate.strip())
        seen.append(words)
    return " ".join(sentences)


# Abstract voice ("We propose...", "In this paper, ...") rewritten as the third-person voice a thesis
# uses about the work it reviews. Order matters: the longer phrases go first.
THESIS_VOICE = (
    (re.compile(r"^(?:in|throughout) this (?:paper|article|work|study|survey|review),?\s*", re.IGNORECASE), ""),
    (re.compile(r"\bthis (?:paper|article|work|manuscript)\b", re.IGNORECASE), "the study"),
    (re.compile(r"\bthis survey\b", re.IGNORECASE), "the survey"),
    (re.compile(r"\bwe (?=[a-z])", re.IGNORECASE), "the authors "),
    (re.compile(r"\bour (?=[a-z])", re.IGNORECASE), "their "),
    (re.compile(r"\b(?:to|for) us\b", re.IGNORECASE), "to the authors"),
)


def thesis_voice(text: str, subject: str = "", first_person: bool = True) -> str:
    """Rewrites abstract voice into thesis voice, naming `subject` (e.g. "Li et al. (2023)") for "we".

    With first_person=False only the "this paper" phrasing is changed; for text written as the review
    itself, where "we" would be the reviewer rather than a paper's authors.
    """
    out = []
    named = not subject
    for sentence in split_sentences(text):
        for pattern, replacement in THESIS_VOICE:
            if not first_person and replacement in ("the authors ", "their ", "to the authors"):
                continue
            if not named and replacement == "the authors " and pattern.search(sentence):
                # The first "we" names the authors; later ones stay "the authors".
                sentence, named = pattern.sub(f"{subject} ", sentence, count=1), True
            sentence = pattern.sub(replacement, sentence)
        sentence = sentence.strip()
        if sentence:
            out.append(sentence[0].upper() + sentence[1:])
    return " ".join(out)


def extract_keywords(text: str, limit: int = 5, exclude: str = "") -> List[str]:
    """Most characteristic terms of a text: repeated two-word phrases first, then frequent single words."""
    excluded = set(re.findall(r"[a-z]+", exclude.lower()))
    tokens = re.findall(r"[a-z][a-z-]+", (text or "").lower())
    stop = THEME_STOPWORDS | excluded

    def usable(word: str) -> bool:
        return len(word) >= 4 and word not in stop and word not in GROUNDING_STOPWORDS

    bigrams = Counter(f"{a} {b}" for a, b in zip(tokens, tokens[1:]) if usable(a) and usable(b))
    words = Counter(t for t in tokens if usable(t) and len(t) >= 5)
    keywords = [b for b, n in bigrams.most_common() if n >= 2][: max(1, limit // 2)]
    for word, _ in words.most_common():
        if len(keywords) >= limit:
            break
        if not any(word in k.split() for k in keywords):
            keywords.append(word)
    return keywords


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


def strip_paragraph_labels(text: str) -> str:
    """Removes structure labels the model copies from the prompt ("Paragraph 2 (methods): ...")."""
    lines = []
    for line in (text or "").splitlines():
        stripped = PARAGRAPH_LABEL.sub("", line.strip())
        if stripped or not line.strip():
            lines.append(stripped)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


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
# The same limits as clauses following a paper's name, for prose ("Li (2023) draws on a single case ...").
METHOD_GAP_CLAUSES = {
    "surveys or reviews of prior work": "synthesizes earlier work without contributing new evidence of its own",
    "case studies": "draws on a single case, so its findings may not generalize to other settings",
    "empirical studies": "ties its findings to the data and setting it studied, leaving their transfer untested",
    "conceptual frameworks or recommendations": "proposes a framework that has not yet been validated empirically",
    DEFAULT_METHOD_PROFILE: "makes an argument that has not been tested against empirical data",
}
# How a hybrid summary introduces a paper of each research design ("Li (2023) reviews prior work on ...").
HYBRID_DESIGN_VERBS = {
    "surveys or reviews of prior work": "reviews prior work on",
    "case studies": "presents a case study of",
    "empirical studies": "reports an empirical study of",
    "conceptual frameworks or recommendations": "proposes a conceptual framework for",
    DEFAULT_METHOD_PROFILE: "offers an analytical treatment of",
}
HYBRID_KEYWORD_LEAD = "The work centres on"
HYBRID_AIM_KEYWORDS = ("propose", "present", "develop", "examine", "investigate", "analy", "explore", "aim", "introduce", "design")
HYBRID_RESULT_KEYWORDS = (
    "we show", "we find", "results", "demonstrate", "outperform", "suggest", "reveal", "found", "indicate", "show that",
)
LIMITATION_KEYWORDS = ("limitation", "remain", "open question", "challenge", "lack", "unclear", "not yet", "little is known")
THEME_STOPWORDS = GROUNDING_STOPWORDS | set(
    "provide provides provided present presents propose proposes important different number several "
    "often first including recent however various given makes making while where whose because "
    "approach approaches result results method methods findings work works field fields author authors "
    "abstract article articles report reports issue issues number order level levels understanding "
    "potential possible particular current further across toward towards along since among therefore "
    "thus overall shows shown known become becomes increasingly general specific single multiple conclude concludes construct constructs "
    "suggest suggests describe describes discuss define defines explore explores "
    # hybrid-summary framing ("..., published in arXiv. The work centres on ...") and empty adjectives
    "published arxiv centres centre proposes offers reports presents treatment modern novel existing "
    "various important".split()
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
    # "Toward(s) a theory of X" names an aim, not a subject: the subject is "a theory of X".
    title = re.sub(r"^(?:towards?|on|about|rethinking|revisiting)\s+", "", title, flags=re.IGNORECASE) or title
    words = [w.lower() if w[:1].isupper() and (w[1:].islower() or w in ("A", "An")) else w for w in title.split()]
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
        # A hybrid summary only restates the abstract plus metadata, so it would add noise, not themes.
        text = f"{s.paper.title} {s.paper.abstract} {s.summary if s.from_llm else ''}".lower()
        per_paper.append(Counter(w for w in re.findall(r"[a-z]{5,}", text) if w not in THEME_STOPWORDS | excluded))
    if not per_paper:
        return []
    papers_using = Counter(w for counts in per_paper for w in counts)
    totals = sum(per_paper, Counter())
    min_papers = 2 if len(per_paper) > 1 else 1
    candidates = [w for w, n in papers_using.items() if n >= min_papers]
    candidates.sort(key=lambda w: (papers_using[w], totals[w]), reverse=True)
    themes: List[str] = []
    for word in candidates:  # one form per word: "threats" and "threat" are a single theme
        if not any(word.rstrip("s") == other.rstrip("s") for other in themes):
            themes.append(word)
    return themes[:limit]


GAP_ITEM = re.compile(r"^(?:[-*•]|\d+[.)])\s+(.*)$")
# Titles the model sometimes copies from the prompt's format instead of naming the gap.
PLACEHOLDER_GAP_TITLES = {"short name", "name", "title", "short title", "gap", "gap name", "gap title", "research gap"}
# Openings that say a gap exists without naming it ("There is little research on ...").
GAP_TITLE_FILLER = re.compile(
    r"^(?:"
    r"there\s+(?:is|are)\s+(?:a\s+)?(?:little|no|limited|few|scant|insufficient)\s+(?:\w+\s+)?(?:on|of|into|about)\s+"
    r"|little\s+is\s+known\s+about\s+"
    r"|there\s+(?:is|are|remain|remains)\s+(?:still\s+)?(?:some\s+|many\s+|several\s+)?"
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


def related_papers(text: str, summaries: Sequence[PaperSummary], min_shared: int = 2) -> List[PaperSummary]:
    """Papers sharing at least `min_shared` key terms with `text`, most overlapping first."""
    words = _content_words(text) - SYNTHESIS_VOCABULARY
    scored = []
    for s in summaries:
        shared = len(words & _content_words(f"{s.paper.title} {s.summary} {s.gap} {' '.join(s.key_findings)}"))
        if shared >= min_shared:
            scored.append((shared, s))
    scored.sort(key=lambda item: -item[0])
    return [s for _, s in scored]


def _lower_first(text: str) -> str:
    return text[0].lower() + text[1:] if text[1:2].islower() else text


def limitation_sentence(paper: Paper, gap: str) -> str:
    """The paper's main limitation as a summary sentence."""
    profile = method_profile(paper)
    if not gap or gap in METHOD_GAPS.values():
        return f"Its main limitation is that it {METHOD_GAP_CLAUSES[profile]}."
    stated = thesis_voice(gap).rstrip(".")
    stated = re.sub(r"^(?:the|this) (?:paper|study|work|article)\b", "it", stated, flags=re.IGNORECASE)
    return f"Its main limitation is that {_lower_first(stated)}."


def relevance_sentence(paper: Paper, topic: str) -> str:
    """Why the paper matters for the review's topic, from its characteristic terms."""
    keywords = extract_keywords(f"{paper.title} {paper.abstract}", limit=2, exclude=topic)
    verb = "contributes evidence on" if method_profile(paper) == "empirical studies" else "contributes a perspective on"
    if keywords:
        return f"For research on {topic}, it {verb} {join_items(keywords)}."
    return f"For research on {topic}, it {verb} {paper_focus(paper)}."


# Sentences in which authors name the work still to be done ("Future research should examine ...").
FUTURE_WORK_CUE = re.compile(
    r"\bfuture (?:work|research|studies|directions?)\b|\bfurther (?:research|work|studies|investigation)\b|"
    r"\bremains? to be (?:explored|investigated|studied|seen)\b|\bshould be (?:explored|investigated|examined|studied)\b",
    re.IGNORECASE,
)
FUTURE_WORK_LEAD = re.compile(
    r"^.*?\b(?:future (?:work|research|studies)|further (?:research|work|studies))\b\s*"
    r"(?:(?:should|could|will|may|might|must|can)\s+|(?:is|are)\s+(?:needed|required)\s+to\s+|needs?\s+to\s+)?"
    r"(?:(?:examine|explore|investigate|address|study|develop|consider|assess|evaluate|focus on|extend|test)\s+)?"
    r"(?:(?:on|into|of|in|for)\s+)?",
    re.IGNORECASE,
)


def future_work_statements(summary: PaperSummary, limit: int = 2) -> List[str]:
    """The paper's own calls for further work, from its abstract and summary."""
    text = f"{summary.paper.abstract} {summary.summary}"
    found = [s for s in split_sentences(normalize_whitespace(text)) if FUTURE_WORK_CUE.search(s) and 30 <= len(s) <= 300]
    return list(dict.fromkeys(found))[:limit]


def contradiction_gaps(
    topic: str, summaries: Sequence[PaperSummary], tensions: Sequence[Tuple[str, str, str]]
) -> List[Tuple[str, str, Tuple[str, str]]]:
    """A gap for each disagreement between papers: (title, description, (surname A, surname B))."""
    gaps = []
    for a, b, point in tensions:
        terms = extract_keywords(point, limit=2, exclude=f"{topic} {a} {b}")
        title = f"Conflicting evidence on {join_items(terms)}" if terms else f"Conflicting findings on {topic}"
        desc = (
            f"{a} and {b} reach different conclusions: {_lower_first(point.rstrip('.'))}. No reviewed study tests "
            "which position holds or under what conditions each applies. A study that measures both positions on "
            f"the same data would show whether the disagreement is substantive or an artefact of method, which "
            f"decides how far either result can guide work on {topic}."
        )
        gaps.append((title, desc, (re.sub(r"\s*\(.*$", "", a), re.sub(r"\s*\(.*$", "", b))))
    return gaps


def ensure_coverage(synthesis: str, summaries: Sequence[PaperSummary], topic: str) -> str:
    """Adds a paragraph placing every paper the synthesis never mentions, grouped by research design.

    A literature review must discuss every included study; this guarantees that whatever the model
    (or a later de-duplication step) left out is still placed within the review's themes and methods.
    """
    missing = [s for s in summaries if s.paper.first_author_surname not in synthesis]
    if not missing:
        return synthesis
    groups: dict = {}
    for s in missing:
        groups.setdefault(method_profile(s.paper), []).append(s)
    sentences = [f"The review also draws on {len(missing)} further {'study' if len(missing) == 1 else 'studies'} that extend these lines of work on {topic}."]
    for profile, group in groups.items():
        items = join_items(f"{s.paper.first_author_surname} ({s.paper.year or 'n.d.'}) on {paper_focus(s.paper)}" for s in group)
        sentences.append(f"Among the {profile}, these are {items}.")
    themes = shared_themes(list(summaries), exclude=topic, limit=2)
    if themes:
        sentences.append(f"They add further evidence on {join_items(themes)}, the concerns that connect the literature reviewed above.")
    logger.info("Synthesis coverage: added %d paper(s) the synthesis did not discuss", len(missing))
    return f"{synthesis.rstrip()}\n\n{' '.join(sentences)}"


def expand_gap(desc: str, summaries: Sequence[PaperSummary], topic: str, minimum: int = MIN_GAP_SENTENCES) -> str:
    """Brings a gap description to at least `minimum` sentences: which papers it bears on and why it matters."""
    desc = normalize_whitespace(desc)
    if desc and desc[-1] not in ".!?":
        desc += "."
    if len(split_sentences(desc)) >= minimum:
        return desc
    related = related_papers(desc, summaries)[:3]
    if related:
        names = join_items(f"{s.paper.first_author_surname} ({s.paper.year or 'n.d.'})" for s in related)
        context = f"It bears directly on the work of {names}, whose conclusions stop short of this question."
    else:
        context = "None of the reviewed papers resolves it directly." if summaries else ""
    significance = (
        f"Until it is studied, conclusions about {topic} rest on evidence that leaves this question open, "
        "which limits how far they can guide practice or further research."
    )
    return " ".join(filter(None, [desc, context, significance]))


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

    def summarize_paper(self, paper: Paper, full_text: str, topic: str = "") -> PaperSummary:
        # Abstract first: it is the densest description of the paper and survives truncation.
        source_text = normalize_whitespace(paper.abstract)
        if full_text and normalize_whitespace(full_text) != source_text:
            source_text = f"{source_text}\n\n{normalize_whitespace(full_text)}"
        source_text = source_text[: self.source_chars]
        system = (
            "You are a research assistant. Summarize academic papers accurately and concisely. "
            "Only state facts found in the provided text. Never use placeholder text."
        )
        relevance = f"for research on {topic}" if topic else "for its field"
        user = (
            f"Paper title: {paper.title}\n"
            f"Authors: {', '.join(paper.authors[:5]) or 'Unknown'}\n"
            f"Year: {paper.year or 'Unknown'}\n\n"
            f"Text:\n{source_text}\n\n"
            "Task: First write a summary of 3-4 complete sentences in the third person (\"The authors "
            "...\", never \"we\"): (1) the problem the paper studies and its method, (2) its main "
            "contribution or result, (3) its main limitation, and (4) why it matters "
            f"{relevance}. Paraphrase; do not copy sentences from the text. Then write 3 key findings, "
            "each on its own line starting with '- '. Finally write one line starting with 'Gap: ' that "
            "states, in one sentence, a question or limitation the paper leaves unaddressed."
        )

        def validate(text: str) -> Optional[str]:
            summary = self._split_summary_and_findings(self._split_gap(text, source_text)[0])[0]
            return content_issue(summary, MIN_SUMMARY_SENTENCES, 20, source_text) or (
                "copied the source text instead of summarizing it" if copied_ratio(summary, source_text) > MAX_COPIED else None
            )

        raw = self.complete_validated(
            system,
            user,
            max_tokens=350,
            min_sentences=MIN_SUMMARY_SENTENCES,
            validator=validate,
            label=f"summary for {paper.title[:50]!r}",
        )
        if not raw:
            return self.hybrid_summary(paper, full_text, topic)
        raw, gap = self._split_gap(raw, source_text)
        summary, findings = self._split_summary_and_findings(raw)
        summary = self._strip_title_and_refs(summary, paper.title)
        summary = thesis_voice(fix_against_source(" ".join(split_sentences(summary)[:4]), source_text))
        findings = [
            thesis_voice(fix_against_source(f, source_text))
            for f in findings
            # questions and bare headings ("Problem Formulation") aren't findings
            if not PLACEHOLDER_PATTERN.search(f) and not f.rstrip().endswith("?") and len(f.split()) >= 4
        ][:5]
        gap = gap or default_paper_gap(paper)
        if len(split_sentences(summary)) < MIN_SUMMARY_SENTENCES:
            # Stripping an echoed title or truncated tail can leave too few sentences; top up with the
            # paper's limitation and relevance, then from the hybrid summary.
            extra = [limitation_sentence(paper, gap), relevance_sentence(paper, topic)]
            extra += split_sentences(self.hybrid_summary(paper, full_text, topic).summary)
            summary = ensure_min_sentences(summary, extra)
        summary = " ".join(split_sentences(summary)[:MAX_SUMMARY_SENTENCES])
        return PaperSummary(paper=paper, summary=summary, key_findings=findings, gap=gap)

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
    def hybrid_summary(paper: Paper, full_text: str = "", topic: str = "") -> PaperSummary:
        """Builds a summary from the paper's metadata, keywords and abstract when the model fails.

        Rather than pasting the abstract, it frames the paper in the review's voice (who, when, what
        kind of study, on what), states its contribution (its aim and result sentences rewritten in
        the third person), its limitation and its relevance to the topic: MIN_SUMMARY_SENTENCES to
        MAX_SUMMARY_SENTENCES sentences.
        """
        source = normalize_whitespace(paper.abstract or full_text[:MAX_SOURCE_CHARS])
        who = f"{paper.first_author_surname} ({paper.year or 'n.d.'})"
        framing = f"{who} {HYBRID_DESIGN_VERBS[method_profile(paper)]} {paper_focus(paper)}"
        if paper.venue:
            framing += f", published in {normalize_whitespace(paper.venue)}"
        keywords = extract_keywords(f"{paper.title} {source}", limit=4, exclude=paper_focus(paper))
        usable = [s for s in split_sentences(source) if 30 <= len(s) <= 300 and not FUTURE_WORK_REMARK.search(s)]
        aim = next((s for s in usable if any(k in s.lower() for k in HYBRID_AIM_KEYWORDS)), None)
        result = next((s for s in usable if s != aim and any(k in s.lower() for k in HYBRID_RESULT_KEYWORDS)), None)
        contribution = [thesis_voice(s) for s in ([s for s in (aim, result) if s] or usable[:2])]
        gap = default_paper_gap(paper)
        # Framing, contribution, limitation, relevance; the keyword sentence only when there is room.
        closing = [limitation_sentence(paper, gap)] + ([relevance_sentence(paper, topic)] if topic else [])
        while contribution and 1 + len(contribution) + len(closing) > MAX_SUMMARY_SENTENCES:
            contribution.pop()
        sentences = [framing.rstrip(".") + "."] + contribution
        if keywords and len(sentences) + len(closing) < MAX_SUMMARY_SENTENCES:
            sentences.append(f"{HYBRID_KEYWORD_LEAD} {join_items(keywords)}.")
        sentences += closing
        if len(sentences) < MIN_SUMMARY_SENTENCES:
            sentences.append("Its abstract gives no further detail, so its methods and results should be checked in the full text.")

        findings = [
            thesis_voice(f)
            for f in pick_sentences(source, HYBRID_RESULT_KEYWORDS, limit=3)
            if not FUTURE_WORK_REMARK.search(f)
        ]
        logger.info("Using hybrid summary (abstract + metadata + keywords) for %r", paper.title)
        return PaperSummary(paper=paper, summary=" ".join(sentences), key_findings=findings, from_llm=False, gap=gap)

    fallback_summary = hybrid_summary  # older name, kept for callers outside this module

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
        """The literature synthesis (see _synthesize), guaranteed to discuss every paper (ensure_coverage)."""
        return ensure_coverage(self._synthesize(topic, summaries), summaries, topic)

    def _synthesize(self, topic: str, summaries: List[PaperSummary]) -> str:
        """Thematic synthesis in four paragraphs (SYNTHESIS_STRUCTURE).

        Tries the whole synthesis first; if it is rejected, writes each paragraph on its own (a much
        easier task for a small model) and fills any paragraph the model still can't write from the
        structured template, so the section keeps its shape either way.
        """
        context = sources_block(summaries, self.context_chars)
        authors = [s.paper.first_author_surname for s in summaries]
        structure = "\n".join(
            f"Paragraph {i} ({name}): {instruction}" for i, (name, instruction) in enumerate(SYNTHESIS_STRUCTURE, 1)
        )
        system = (
            "You are an academic writing the literature synthesis chapter of a thesis. Use only facts "
            "from the provided paper summaries, cite them inline as (Author, Year), and write continuous "
            "prose in the third person."
        )
        user = (
            f"Research topic: {topic}\n\n"
            f"Summaries of {len(summaries)} papers:\n\n{context}\n\n"
            f"Task: Write the literature synthesis as {len(SYNTHESIS_STRUCTURE)} paragraphs organized by "
            f"theme, not paper by paper, following this structure:\n{structure}\n"
            f"{SYNTHESIS_RULES} Cite every paper by its (Author, Year) tag at least once.\n"
            f"{SYNTHESIS_EXAMPLE}"
        )

        def validate(text: str) -> Optional[str]:
            text = strip_paragraph_labels(text)
            if copied_ratio(text, context) > SYNTH_MAX_COPIED:
                return "copied the summaries instead of synthesizing them"
            return (
                synthesis_issue(text, context, authors)
                or content_issue(text, min_sentences=6, min_words=90, source=context)
                or prose_issue(text, context)
            )

        text = self.complete_validated(
            system, user, max_tokens=700, min_sentences=6, validator=validate, label="literature synthesis"
        )
        if text:
            return strip_paragraph_labels(text)
        return self._synthesize_by_paragraph(topic, summaries, context)

    def _synthesize_by_paragraph(self, topic: str, summaries: List[PaperSummary], context: str) -> str:
        """Writes the synthesis one structured paragraph at a time; the template fills paragraphs the model can't."""
        template = self.structured_synthesis(topic, summaries)
        system = (
            "You are an academic writing one paragraph of a thesis literature synthesis. Use only facts "
            "from the provided papers and cite them as (Author, Year)."
        )
        paragraphs, written = [], 0
        for (name, instruction), fallback in zip(SYNTHESIS_STRUCTURE, template):
            user = (
                f"Research topic: {topic}\n\nPapers:\n{context}\n\n"
                f"Task: Write one paragraph of 3-4 sentences on the papers' {name}: {instruction} {SYNTHESIS_RULES}"
            )

            def validate(text: str) -> Optional[str]:
                text = strip_paragraph_labels(text)
                if copied_ratio(text, context) > SYNTH_MAX_COPIED:
                    return "copied the summaries instead of synthesizing them"
                return (
                    synthesis_issue(text, context)
                    or content_issue(text, min_sentences=2, min_words=35, source=context)
                    or prose_issue(text, context)
                )

            text = self.complete_validated(
                system, user, max_tokens=260, retries=0, validator=validate, label=f"synthesis paragraph ({name})"
            )
            if text:
                written += 1
                paragraphs.append(normalize_whitespace(strip_paragraph_labels(text)))
            else:
                paragraphs.append(fallback)
        logger.info(
            "Literature synthesis assembled from %d model-written and %d template paragraphs",
            written, len(paragraphs) - written,
        )
        if any(s.paper.first_author_surname not in "\n".join(paragraphs) for s in summaries):
            # Every paper must be placed in the synthesis; the template's opening sentence names them all.
            paragraphs[0] = f"{split_sentences(template[0])[0]} {paragraphs[0]}"
        return "\n\n".join(paragraphs)

    @staticmethod
    def fallback_synthesis(topic: str, summaries: List[PaperSummary]) -> str:
        """The structured template synthesis as one text (see structured_synthesis)."""
        return "\n\n".join(LLMSynthesizer.structured_synthesis(topic, summaries))

    @staticmethod
    def structured_synthesis(topic: str, summaries: List[PaperSummary]) -> List[str]:
        """One paragraph per SYNTHESIS_STRUCTURE part, built from titles, themes, methods and limitations.

        Deliberately avoids quoting abstract or summary sentences, which the report already shows
        under Paper Summaries.
        """
        logger.info("Building literature synthesis from the structured template (themes, methods, comparison, gaps)")
        themes_paragraph, methods_paragraph = LLMSynthesizer._themes_and_methods(topic, summaries)
        return [
            themes_paragraph,
            methods_paragraph,
            LLMSynthesizer._comparison_paragraph(topic, summaries),
            LLMSynthesizer._open_questions_paragraph(topic, summaries),
        ]

    @staticmethod
    def _comparison_paragraph(topic: str, summaries: List[PaperSummary]) -> str:
        """Where the papers overlap (shared themes, by paper) and why their positions are hard to compare."""
        def who(s: PaperSummary) -> str:
            return f"{s.paper.first_author_surname} ({s.paper.year or 'n.d.'})"

        sentences = []
        openings = ("The papers converge most clearly on {theme}, which", "A second point of overlap is {theme}, which")
        for theme, opening in zip(shared_themes(summaries, exclude=topic, limit=2), openings):
            using = [s for s in summaries if theme in f"{s.paper.title} {s.paper.abstract} {s.summary}".lower()]
            others = [s for s in summaries if s not in using]
            if len(using) >= 2:
                sentence = (
                    f"{opening.format(theme=theme)} {join_items(who(s) for s in using)} "
                    f"{'both' if len(using) == 2 else 'all'} take up"
                )
                sentence += (
                    f", while {join_items(who(s) for s in others)} {'approaches' if len(others) == 1 else 'approach'} "
                    f"{topic} without it." if others else "."
                )
                sentences.append(sentence)
        designs = {method_profile(s.paper) for s in summaries}
        if len(designs) > 1:
            sentences.append(
                "Because these works rest on different kinds of evidence, apparent disagreements between them may "
                "reflect their methods rather than their substance, and no study yet tests their claims side by side."
            )
        else:
            sentences.append(
                "Since they share one research design, their agreement may partly reflect that common method, and "
                "evidence of another kind would show whether it holds."
            )
        if len(sentences) < 2:
            sentences.insert(0, f"The papers take up largely separate questions about {topic}, so points of direct "
                                "agreement or contradiction between them are few.")
        return " ".join(sentences)

    @staticmethod
    def _open_questions_paragraph(topic: str, summaries: List[PaperSummary]) -> str:
        """What each paper leaves open, from its stated limitation or what its design cannot show."""
        clauses = []
        for s in summaries[:5]:
            who = f"{s.paper.first_author_surname} ({s.paper.year or 'n.d.'})"
            design_gap = METHOD_GAP_CLAUSES.get(method_profile(s.paper)) if s.gap in METHOD_GAPS.values() else None
            gap = design_gap or (s.gap and thesis_voice(s.gap).rstrip("."))
            if not gap:
                continue
            if design_gap:
                clauses.append(f"{who} {gap}")
            else:
                gap = re.sub(r"^(?:the|this) (?:paper|study|work|article)\b", "it", gap, flags=re.IGNORECASE)
                clauses.append(f"for {who}, {gap[0].lower() + gap[1:]}")
        sentences = [f"Each line of work also leaves questions open for {topic}."]
        if clauses:
            joined = "; ".join(clauses)
            sentences.append(joined[0].upper() + joined[1:] + ".")
        sentences.append("These limitations define the research gaps set out in the Discussion.")
        return " ".join(sentences)

    @staticmethod
    def _themes_and_methods(topic: str, summaries: List[PaperSummary]) -> Tuple[str, str]:
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
        return paragraphs[0], methods

    def generate_research_gaps(
        self,
        topic: str,
        synthesis: str,
        summaries: Sequence[PaperSummary] = (),
        tensions: Sequence[Tuple[str, str, str]] = (),
    ) -> str:
        """Returns gaps as '- **Title**: 2-3 sentence description' lines (see parse_gaps).

        Gaps are drawn from three kinds of evidence, all given to the model: the papers' stated
        limitations, the further work their authors call for, and the contradictions between papers
        (find_tensions). A gap the model describes in a single sentence is kept and expanded
        (expand_gap) rather than discarding the whole answer; every returned gap has at least
        MIN_GAP_SENTENCES sentences, and an unaddressed contradiction is always added as a gap.
        """
        summaries = list(summaries)
        papers = sources_block(summaries, self.context_chars * 3 // 5) or synthesis[: self.context_chars]
        limitations = "\n".join(
            f"[{s.paper.first_author_surname}, {s.paper.year or 'n.d.'}] {normalize_whitespace(s.gap)}"
            for s in summaries if s.gap
        )[: self.context_chars // 5]
        future = "\n".join(
            f"[{s.paper.first_author_surname}, {s.paper.year or 'n.d.'}] {sentence}"
            for s in summaries for sentence in future_work_statements(s, limit=1)
        )[: self.context_chars // 10]
        disagreements = "\n".join(f"{a} vs {b}: {point}" for a, b, point in tensions)[: self.context_chars // 10]
        context = f"{papers}\n\n{limitations}\n\n{future}\n\n{disagreements}"
        system = "You are an academic identifying what is missing from a body of literature for a thesis."
        user = (
            f"Research topic: {topic}\n\nPapers:\n{papers}\n\n"
            + (f"Limitations the papers leave open:\n{limitations}\n\n" if limitations else "")
            + (f"Further work the authors call for:\n{future}\n\n" if future else "")
            + (f"Where papers disagree:\n{disagreements}\n\n" if disagreements else "")
            + "Task: Identify 3 research gaps: questions these papers leave unanswered. Draw on the "
            "limitations, the further work the authors call for and the disagreements above, not only on "
            "single papers' limitations. Write each gap on "
            "its own line as '- <Title>: <description>', where <Title> is a 3-6 word noun phrase naming "
            "the gap's main concept (for example 'Long-term deployment outcomes'). The description must "
            "have 2-3 complete sentences: (1) what exactly is missing from the literature, (2) which "
            "papers' limitations point to it, cited as (Author, Year), and (3) why closing it matters for "
            f"{topic}. Name specific concepts; avoid vague words such as 'various', 'important' or "
            "'discuss'. Do not write the words 'Title' or 'Short name' literally. Do not list paper titles.\n"
            "Example of one line (different topic): '- Rural uptake evidence: No study measures adoption "
            "outside urban clinics, although (Okafor, 2019) flags this as its main limitation. Without such "
            "data, rollout plans cannot be judged for the settings where need is greatest.'"
        )

        def usable(text: str) -> List[Tuple[str, str]]:
            return [
                (title, desc)
                for title, desc in parse_gaps(text)
                if len(desc.split()) >= 12
                and not re.match(r'^"[^"]+"\s*(?:by\b|\()', desc)  # a paper citation is not a gap
                and copied_ratio(desc, context) <= SYNTH_MAX_COPIED
                and not VAGUE_LANGUAGE.search(desc)
                and not synthesis_issue(desc, papers)  # invented citations or paper voice
            ]

        def validate(text: str) -> Optional[str]:
            found = usable(text)
            if len(found) < 2:
                return f"only {len(found)} usable gap descriptions"
            if not any(len(split_sentences(d)) >= MIN_GAP_SENTENCES for _, d in found):
                return f"no gap description has {MIN_GAP_SENTENCES}+ sentences"
            # Grounded against the papers, not the synthesis: the synthesis is model output too.
            return content_issue("\n".join(d for _, d in found), 2, 40, source=context)

        text = self.complete_validated(system, user, max_tokens=500, validator=validate, label="research gaps")
        if not text:
            return self.fallback_research_gaps(topic, summaries, tensions)
        gaps = [(title, expand_gap(desc, summaries, topic)) for title, desc in usable(text)[:4]]
        written = " ".join(desc for _, desc in gaps)
        for title, desc, (a, b) in contradiction_gaps(topic, summaries, tensions):
            # A disagreement between papers is a gap in its own right unless the model already covers it.
            if not (a in written and b in written):
                gaps.append((title, desc))
                break
        return format_gaps(gaps[:5])

    @staticmethod
    def fallback_research_gaps(
        topic: str, summaries: Sequence[PaperSummary], tensions: Sequence[Tuple[str, str, str]] = ()
    ) -> str:
        """Gaps from the papers' stated limitations, their authors' calls for further work, the
        contradictions between papers, and then what the set as a whole lacks."""
        logger.info("Building research gaps from the papers' limitations, stated future work, contradictions and methods")
        summaries = list(summaries)
        if not summaries:
            return format_gaps([(
                "Evidence base",
                f"No papers could be summarized for {topic}, so the state of the literature is unknown. "
                "A systematic search across more sources is needed before gaps can be identified.",
            )])
        gaps: List[Tuple[str, str]] = []

        # Each paper's "limitations" field (PaperSummary.gap): what it states, or the model named, as
        # unaddressed. Design-level defaults (METHOD_GAPS) are covered by the set-level gaps below.
        for s in summaries:
            limitation = normalize_whitespace(s.gap or "")
            if not limitation or limitation in METHOD_GAPS.values() or FUTURE_WORK_REMARK.search(limitation):
                continue
            # "X works well, but challenges remain in Y": the gap is what follows the contrast.
            contrast = re.split(r",\s*(?:but|yet|although|however)\s+|;\s*however,?\s+", limitation, flags=re.IGNORECASE)
            title = gap_title(contrast[-1] if len(contrast) > 1 and len(contrast[-1].split()) >= 3 else limitation)
            if not title or any(title.lower() == t.lower() for t, _ in gaps):
                continue
            who = f"{s.paper.first_author_surname} ({s.paper.year or 'n.d.'})"
            stated = thesis_voice(limitation).rstrip(".")
            stated = re.sub(r"^(?:the|this) (?:paper|study|work|article)\b", "it", stated, flags=re.IGNORECASE)
            if stated[1:2].islower():
                stated = stated[0].lower() + stated[1:]
            others = [o for o in summaries if o is not s]
            gaps.append((title, expand_gap(f"The work of {who} leaves a limitation unresolved: {stated}.", others, topic, minimum=3)))
            if len(gaps) >= 2:
                break

        # What the authors themselves say remains to be done.
        for s in summaries:
            statement = next(iter(future_work_statements(s, limit=1)), "")
            if not statement:
                continue
            title = gap_title(FUTURE_WORK_LEAD.sub("", statement) or statement)
            if not title or any(title.lower() == t.lower() for t, _ in gaps):
                continue
            who = f"{s.paper.first_author_surname} ({s.paper.year or 'n.d.'})"
            others = [o for o in summaries if o is not s]
            stated = _lower_first(thesis_voice(statement).rstrip("."))
            gaps.append((title, expand_gap(f"{who} calls for further work: {stated}.", others, topic, minimum=3)))
            break

        gaps += [(title, desc) for title, desc, _ in contradiction_gaps(topic, summaries, tensions)[:1]]

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
        return format_gaps(gaps[:5])

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

    def summarize_all(
        self, papers: List[Paper], extractor: TextExtractor, full_text: bool = True, topic: str = ""
    ) -> List[PaperSummary]:
        summaries = []
        for index, paper in enumerate(papers, 1):
            logger.info("Summarizing paper %d/%d: %s", index, len(papers), paper.title[:80])
            try:
                text = extractor.extract_text_for_paper(paper) if full_text else paper.abstract
                summaries.append(self.summarize_paper(paper, text, topic))
            except OllamaConnectionError:
                raise
            except Exception as exc:
                logger.warning("Skipping summary for %r due to error: %s", paper.title, exc)
        return summaries
