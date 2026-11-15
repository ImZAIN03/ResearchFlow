"""Keyless paper search: arXiv, OpenAlex, Semantic Scholar, Crossref, PubMed. Responses cached as JSON."""
import hashlib
import json
import re
import time
import xml.etree.ElementTree as ET
from itertools import zip_longest
from pathlib import Path

import requests

CACHE = Path("data/cache")
UA = {"User-Agent": "research-accelerator/0.1 (open-source, local)"}
MEDICAL = re.compile(r"medic|clinic|patient|disease|cancer|health|therap|drug|surg|diagnos|covid|virus|hospital|pharma|epidemi", re.I)


def _get(url, params=None, text=False, pause=1.0):
    """GET with on-disk cache keyed by URL hash; sleeps `pause` s before each live call."""
    full = requests.Request("GET", url, params=params).prepare().url
    f = CACHE / (hashlib.sha256(full.encode()).hexdigest() + ".json")
    if f.exists():
        return json.loads(f.read_text("utf-8"))["body"]
    time.sleep(pause)
    r = requests.get(full, headers=UA, timeout=30)
    r.raise_for_status()
    body = r.text if text else r.json()
    CACHE.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"url": full, "body": body}), "utf-8")
    return body


def _clean(s):
    return " ".join(re.sub(r"<[^>]+>", " ", s or "").split())


def _key(title):
    return re.sub(r"[^a-z0-9]", "", (title or "").lower())


def _inverted(inv):
    """OpenAlex abstract_inverted_index {word: [positions]} -> words joined in position order."""
    return " ".join(w for _, w in sorted((i, w) for w, ix in (inv or {}).items() for i in ix))


def _s2(url, params, pause=2):
    """Semantic Scholar GET; on HTTP 429 wait 2, 4, 8, 16 s and retry, then give up. Successes are cached by _get."""
    for wait in (2, 4, 8, 16, None):
        try:
            return _get(url, params, pause=pause)
        except requests.HTTPError as e:
            if wait is None or e.response is None or e.response.status_code != 429:
                raise
            time.sleep(wait)


def _year(v):
    m = re.search(r"(19|20)\d{2}", str(v or ""))
    return int(m.group()) if m else None


def paper(title, authors, year, doi, url, venue, abstract, type=""):
    return {"title": _clean(title), "authors": [a for a in authors if a], "year": _year(year),
            "doi": (doi or "").strip() or None, "url": url or "", "venue": venue or "", "abstract": _clean(abstract), "type": type or ""}


def arxiv(q, n):
    A = "{http://www.w3.org/2005/Atom}"
    root = ET.fromstring(_get("https://export.arxiv.org/api/query", {"search_query": "all:" + q, "max_results": n}, text=True, pause=3))
    return [paper(e.findtext(A + "title"), [a.findtext(A + "name") for a in e.findall(A + "author")],
                  e.findtext(A + "published"), e.findtext("{http://arxiv.org/schemas/atom}doi"),
                  e.findtext(A + "id"), "arXiv", e.findtext(A + "summary")) for e in root.findall(A + "entry")]


def openalex(q, n):
    out = []
    for w in _get("https://api.openalex.org/works", {"search": q, "per-page": n})["results"]:
        loc = (w.get("primary_location") or {}).get("source") or {}
        out.append(paper(w.get("title"), [(a.get("author") or {}).get("display_name") for a in w.get("authorships", [])],
                         w.get("publication_year"), re.sub(r"^https?://doi.org/", "", w.get("doi") or ""),
                         w.get("id"), loc.get("display_name"), _inverted(w.get("abstract_inverted_index")), w.get("type")))
    return out


def semantic(q, n):
    d = _s2("https://api.semanticscholar.org/graph/v1/paper/search",
            {"query": q, "limit": n, "fields": "title,authors,year,externalIds,url,venue,abstract"})
    return [paper(w.get("title"), [a.get("name") for a in w.get("authors", [])], w.get("year"),
                  (w.get("externalIds") or {}).get("DOI"), w.get("url"), w.get("venue"), w.get("abstract"))
            for w in d.get("data", [])]


def crossref(q, n):
    d = _get("https://api.crossref.org/works", {"query": q, "rows": n})["message"]["items"]
    return [paper((w.get("title") or [""])[0], [" ".join([a.get("given", ""), a.get("family", "")]).strip() for a in w.get("author", [])],
                  ((w.get("issued") or {}).get("date-parts") or [[None]])[0][0], w.get("DOI"), w.get("URL"),
                  (w.get("container-title") or [""])[0], w.get("abstract"), w.get("type")) for w in d]


def pubmed(q, n):
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    ids = _get(base + "esearch.fcgi", {"db": "pubmed", "term": q, "retmax": n, "retmode": "json"}, pause=0.5)["esearchresult"]["idlist"]
    if not ids:
        return []
    root = ET.fromstring(_get(base + "efetch.fcgi", {"db": "pubmed", "id": ",".join(ids), "retmode": "xml"}, text=True, pause=0.5))
    out = []
    for a in root.iter("PubmedArticle"):
        t = lambda path: "".join(a.find(path).itertext()) if a.find(path) is not None else ""
        doi = next((i.text for i in a.iter("ArticleId") if i.get("IdType") == "doi"), None)
        pmid = a.findtext(".//PMID")
        out.append(paper(t(".//ArticleTitle"), [(x.findtext("ForeName") or "") + " " + (x.findtext("LastName") or "") for x in a.iter("Author")],
                         t(".//PubDate/Year") or t(".//PubDate/MedlineDate"), doi, "https://pubmed.ncbi.nlm.nih.gov/%s/" % pmid,
                         t(".//Journal/Title"), " ".join("".join(x.itertext()) for x in a.iter("AbstractText"))))
    return out


def dedupe(papers):
    seen = {}
    for p in papers:
        k = (_key(p["title"]), p["year"])
        if not k[0]:
            continue
        if k not in seen:
            seen[k] = p
        elif not seen[k]["abstract"] and p["abstract"]:
            seen[k]["abstract"], seen[k]["abstract_source"] = p["abstract"], p.get("abstract_source")
    return list(seen.values())


def search(query, limit=10):
    """Search every free source (PubMed only for medical topics); round-robin merge, dedupe, cap at limit."""
    srcs = [arxiv, openalex, semantic, crossref] + ([pubmed] if MEDICAL.search(query) else [])
    results = []
    for s in srcs:
        try:
            got = s(query, limit)
        except Exception as e:  # one dead source must not kill the run
            print("warn: %s failed: %s" % (s.__name__, e))
            got = []
        for p in got:
            p["abstract_source"] = s.__name__ if p["abstract"] else None
        results.append(got)
    merged = [p for grp in zip_longest(*results) for p in grp if p]
    return dedupe(merged)[:limit]


def _abstract_openalex(p):
    if p["doi"]:
        return _inverted(_get("https://api.openalex.org/works/doi:" + p["doi"]).get("abstract_inverted_index"))
    hits = _get("https://api.openalex.org/works", {"search": p["title"], "per-page": 1})["results"]
    return _inverted(hits[0].get("abstract_inverted_index")) if hits and _key(hits[0].get("title")) == _key(p["title"]) else ""


def _abstract_crossref(p):
    return _get("https://api.crossref.org/works/" + p["doi"])["message"].get("abstract") if p["doi"] else ""


def _abstract_semantic(p):
    base = "https://api.semanticscholar.org/graph/v1/paper/"
    if p["doi"]:
        return _s2(base + "DOI:" + p["doi"], {"fields": "abstract"}).get("abstract")
    hit = (_s2(base + "search/match", {"query": p["title"], "fields": "title,abstract"}).get("data") or [{}])[0]
    return hit.get("abstract") if _key(hit.get("title")) == _key(p["title"]) else ""


def fill_abstracts(papers):
    """Fill missing abstracts from OpenAlex, then Crossref, then Semantic Scholar; record the source, else mark "no abstract"."""
    for p in papers:
        if p["abstract"]:
            continue
        p["abstract_source"] = "no abstract"  # kept for references; empty abstract yields no quotes or gaps
        for name, fn in (("openalex", _abstract_openalex), ("crossref", _abstract_crossref), ("semantic", _abstract_semantic)):
            try:
                text = _clean(fn(p))  # also strips Crossref JATS/XML tags
            except Exception as e:
                print("warn: %s abstract lookup failed for %r: %s" % (name, p["title"][:60], e))
                continue
            if text:
                p["abstract"], p["abstract_source"] = text, name
                break
    return papers
