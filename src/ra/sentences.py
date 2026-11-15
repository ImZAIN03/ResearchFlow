"""Sentence splitting, rejection rules and pattern classification."""
import re

STOP = set("""a an the of and or to in on for with by from as at is are was were be been being this that these those it its we our their they
which who whom than then there here also can may might could would should will not no but if into over under between among about via using
use used such more most less other some any each both however thus while has have had do does did study studies research work future further
paper results result show shows based new""".split())
BOILER = ("This chapter", "This book", "In this paper", "In recent years")
GAP_RE = re.compile(r"limitation|future work|further research|not yet|remains unclear|remains unknown|has not been", re.I)
VERBS = set("is are was were be been has have had do does did can may might will would show shows find found use uses used propose proposes present presents suggest suggests".split())
FIND = re.compile(r"\b(result|show|found|find|demonstrat|reveal|indicat|suggest|improv|reduc|increas|outperform|significant|achiev|confirm)", re.I)
METH = re.compile(r"\b(we (propose|use|used|conduct|apply|applied|develop|employ|collect|survey|interview|analy[sz]e)|method|experiment|dataset|survey|simulation|trial|framework|approach)", re.I)
FILLER = re.compile(r"recent years have shown|importance of|has become|plays a crucial role|in today[’']?s world", re.I)
BACK = re.compile(r"\b(is a|are a|has become|have become|widely|increasingly|is an important|plays? a)", re.I)


def words(t):
    return re.findall(r"[a-z][a-z0-9-]*", t.lower())


def content(t):
    return [w for w in words(t) if w not in STOP and len(w) > 2]


def split(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z\"'(\[])", text or "") if s.strip()]


def reject(s):
    """Return a rejection reason, or None if the sentence is acceptable."""
    toks = s.split()
    if s.startswith(BOILER):
        return "boilerplate"
    if s.startswith(("Table of Contents", "Contents")):
        return "table-of-contents"
    if len(toks) < 8:
        return "too short (<8 words)"
    if len(toks) > 80:
        return "too long (>80 words)"
    caps = sum(t[0].isupper() for t in toks) / len(toks)
    if caps > 0.4 and not any(w in VERBS or w.endswith("ed") for w in words(s)):
        return "table-of-contents"
    return None


PHRASE_SPLIT = re.compile(r"(?=Table of Contents|(?<!of )\bContents\b|" + "|".join(map(re.escape, BOILER)) + ")")


def classify(s):
    if GAP_RE.search(s):
        return "limitation"
    if FIND.search(s):
        return "finding"
    if METH.search(s):
        return "method"
    return "background"  # default: unmatched text is never promoted to a finding


def process(abstract, log):
    """Split an abstract -> [(sentence, label)]; every rejection is appended to `log`."""
    out = []
    parts = [p.strip() for s in split(abstract) for p in PHRASE_SPLIT.split(s) if p.strip()]
    for s in parts:
        why = reject(s) or ("generic filler" if FILLER.search(s) else None)  # quotes only; gaps.py keeps plain reject()
        if why:
            log.append({"sentence": s, "reason": why})
        else:
            out.append((s, classify(s)))
    return out
