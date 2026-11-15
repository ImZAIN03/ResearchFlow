"""Collect limitation / future-work sentences with exact source text and citation."""
from .bib import cite
from .sentences import GAP_RE, reject, split


def find_gaps(papers, log):
    out = []
    for i, p in enumerate(papers):
        if not p["abstract"]:
            continue
        base = {"paper": i, "cite": cite(p), "year": p["year"]}
        for s in split(p["abstract"]):
            if not GAP_RE.search(s):
                continue
            why = reject(s)
            if why:
                log.append({"sentence": s, "reason": why})
            else:
                out.append(dict(base, id="G%d" % (len(out) + 1), sentence=s))
    return out
