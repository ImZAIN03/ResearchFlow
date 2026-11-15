"""Candidate research directions from gap sentences. Candidates only: a human must verify novelty."""
from collections import Counter
from datetime import date

from .rank import generic, stem
from .sentences import GAP_RE, VERBS, content

STATUS = "candidate (novelty NOT verified; a human must check)"
CAP = 10  # cli searches with limit=10, so 10 results means "at least 10"
NONE_PASSED = "No candidate directions passed the relevance and term filters this run"


def _vocab(question, titles):
    """Stems from the research question plus stems found in >= 2 paper titles; None means no restriction."""
    if not question and not titles:
        return None
    cnt = Counter(w for t in titles for w in {stem(x) for x in content(t)})
    return {stem(w) for w in content(question or "")} | {w for w, c in cnt.items() if c >= 2}


def _terms(text, vocab):
    # shortcut: nouns approximated without a POS tagger (drops verbs, -ed/-ly words); use a tagger if one is ever allowed
    return [w for w in content(text) if not generic(w) and w not in VERBS and not w.endswith(("ed", "ly"))
            and (vocab is None or stem(w) in vocab)]


def group(texts):
    """Greedy grouping by shared content terms -> [(top terms, [indices])]."""
    sets = [set(content(t)) for t in texts]
    left, out = set(range(len(texts))), []
    while left:
        cnt = Counter(w for i in left for w in sets[i])
        if not cnt:
            out.append(([], sorted(left)))
            break
        top = max(cnt, key=lambda w: (cnt[w], w))
        idx = sorted(i for i in left if top in sets[i])
        gc = Counter(w for i in idx for w in sets[i])
        out.append(([w for w, _ in sorted(gc.items(), key=lambda kv: (-kv[1], kv[0]))[:3]], idx))
        left -= set(idx)
    return out


def rank_directions(gaps, search, question=None, titles=()):
    """`search(query)` returns a list of papers; its length is the match count (capped by the search limit).
    Pass only gaps from on-topic papers (rank.relevant); terms are limited to question/title vocabulary."""
    real = gaps
    vocab = _vocab(question, titles)
    year = date.today().year
    out = []
    for terms, idx in group([" ".join(_terms(GAP_RE.sub(" ", g["sentence"]), vocab)) for g in real]):
        if not terms:  # nothing but generic or off-vocabulary words
            continue
        mem = [real[i] for i in idx]
        papers = {m["paper"] for m in mem}
        ys = [m["year"] for m in mem if m["year"]]
        recency = sum(max(0.0, min(1.0, (y - (year - 10)) / 10)) for y in ys) / len(ys) if ys else 0.0
        n = len(search(" ".join(terms)))
        capped = n >= CAP
        conf = "low" if capped else "high" if len(papers) >= 3 and n <= 5 else "medium" if len(papers) >= 2 else "low"
        out.append({"terms": terms, "papers": sorted({m["cite"] for m in mem}), "gap_ids": [m["id"] for m in mem],
                    "sentences": [m["sentence"] for m in mem], "match_count": n,
                    "matches": "%d+ matches (capped)" % CAP if capped else "%d matches" % n,
                    "score": round(len(papers) + recency + 1 / (1 + n), 3), "confidence": conf, "status": STATUS})
    return sorted(out, key=lambda d: -d["score"])


def render(dirs, topic, n_abstracts=0, evidence=False):
    """`evidence`: True when >= 2 papers produced gaps, so an empty list means the filters removed everything."""
    if not dirs:  # never write an empty file
        msg = NONE_PASSED if evidence else "Not enough gap evidence this run: %d papers had abstracts" % n_abstracts
        return "# Candidate research directions: %s\n\n%s\n" % (topic, msg)
    t = ["# Candidate research directions: %s\n" % topic,
         "> These are candidates, not verified gaps. Match counts come from a capped keyword search of free sources and say nothing "
         "about paywalled or unindexed work. A human must verify novelty before anything here is claimed.\n"]
    for i, d in enumerate(dirs, 1):
        t.append("## %d. %s\n- Status: %s\n- Confidence: %s (a heuristic, not evidence of novelty)\n- Search matches: %s\n- Score: %s\n- Supporting papers: %s"
                 % (i, ", ".join(d["terms"]), d["status"], d["confidence"], d["matches"], d["score"], "; ".join(d["papers"])))
        t += ['- Source sentence (%s): "%s"' % (g, s) for g, s in zip(d["gap_ids"], d["sentences"])]
        t.append("")
    return "\n".join(t)
