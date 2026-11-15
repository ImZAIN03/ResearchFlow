"""Pick top finding sentences per paper and keep only those verbatim in the abstract."""
from .rank import rank


def _n(s):
    return " ".join((s or "").split())


def verify(text, abstract):
    return bool(text) and _n(text) in _n(abstract)


def top_quotes(paper, question, sents, n=3, label="finding"):
    """Return (kept, dropped) sentence lists; kept are in original abstract order and verified."""
    pool = [t for t, l in sents if l == label]
    if not pool:
        return [], []
    scores, _ = rank(question, pool)
    best = sorted(sorted(range(len(pool)), key=lambda i: -scores[i])[:n])  # indices -> abstract order
    kept = [pool[i] for i in best if verify(pool[i], paper["abstract"])]
    return kept, [pool[i] for i in best if not verify(pool[i], paper["abstract"])]


def assign(items, question):
    """Set item["section"] so each quote lands in exactly one section, best-ranked first:
    Literature Review, then Introduction (2), then Abstract (3, one per paper). Unplaced leftovers go to the Literature Review."""
    if not items:
        return items
    scores, _ = rank(question, [q["text"] for q in items])
    left = sorted(range(len(items)), key=lambda i: -scores[i])
    for name, cap in (("Literature Review", max(len(items) - 5, (len(items) + 1) // 2)), ("Introduction", 2), ("Abstract", 3)):
        take, papers = [], set()
        for i in left:
            if len(take) == cap:
                break
            if name == "Abstract" and items[i]["paper"] in papers:
                continue
            papers.add(items[i]["paper"])
            take.append(i)
        for i in take:
            items[i]["section"] = name
        left = [i for i in left if i not in take]
    for i in left:
        items[i]["section"] = "Literature Review"
    return items
