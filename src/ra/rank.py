"""Rank sentences against the research question: embeddings if available, else content-word overlap."""
import re

from .sentences import content

GENERIC = set("""adopt approach approaches adequately capability capabilities concept different existing new study paper model
system research results""".split())


def stem(w):
    return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w


def generic(w):
    return w in GENERIC or stem(w) in GENERIC


def norm(t):
    """Lowercase and turn hyphens, slashes and other punctuation into spaces."""
    return re.sub(r"[^a-z0-9]+", " ", t.lower())


def key_terms(question):
    return {stem(w) for w in content(norm(question)) if not generic(w)}


def relevant(paper, terms):
    """True if title + abstract contain >= 2 of the topic's key terms (all of them if the topic has fewer)."""
    have = {stem(w) for w in content(norm(paper["title"] + " " + (paper["abstract"] or "")))}
    return len(terms & have) >= min(2, len(terms))


_model = None
_announced = False


def rank(question, texts):
    """Return (scores aligned with texts, method name). Prints the method once per process."""
    global _model, _announced
    method = "content-word overlap"
    try:
        from sentence_transformers import SentenceTransformer, util
        _model = _model or SentenceTransformer("all-MiniLM-L6-v2")
        e = _model.encode([question] + list(texts), convert_to_tensor=True)
        scores = util.cos_sim(e[0], e[1:])[0].tolist()
        method = "sentence-transformers (all-MiniLM-L6-v2)"
    except Exception:  # not installed or model not downloadable: fall back
        q = set(content(question))
        scores = [len(q & set(content(t))) / (len(q) or 1) for t in texts]
    if not _announced:
        print("ranking method:", method)
        _announced = True
    return scores, method
