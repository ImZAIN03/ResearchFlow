"""APA references from metadata, in-text citations and DOI checks."""
import re

DOI_RE = re.compile(r"^10\.\d{4,9}/[-._;()/:A-Za-z0-9]+$")
NO_AUTHOR = "[author not listed]"


def clean_doi(d):
    return re.sub(r"^(https?://(dx\.)?doi\.org/|doi:)", "", (d or "").strip(), flags=re.I)


def doi_ok(d):
    return bool(DOI_RE.match(clean_doi(d)))


def _sur(a):
    return a.split(",")[0].strip() if "," in a else a.split()[-1]


def cite(p):
    a = p["authors"]
    who = NO_AUTHOR if not a else _sur(a[0]) + (" & " + _sur(a[1]) if len(a) == 2 else " et al." if len(a) > 2 else "")
    return "(%s, %s)" % (who, p["year"] or "n.d.")


def _init(a):
    """'Jane A. Smith' or 'Smith, Jane A.' -> 'Smith, J. A.'; a single name is left as is."""
    if "," in a:
        sur, given = (x.strip() for x in a.split(",", 1))
    else:
        parts = a.split()
        if len(parts) < 2:
            return a
        sur, given = parts[-1], " ".join(parts[:-1])
    inits = " ".join(g[0].upper() + "." for g in re.split(r"[\s.]+", given) if g)
    return "%s, %s" % (sur, inits) if inits else sur


def authors_apa(names):
    a = [_init(x) for x in names]
    if not a:
        return NO_AUTHOR
    if len(a) > 7:
        return a[0] + ", et al."
    return a[0] if len(a) == 1 else ", ".join(a[:-1]) + ", & " + a[-1]


def _tkey(p):
    return re.sub(r"[^a-z0-9]", "", (p["title"] or "").lower())


def _kind(p):
    t = (p.get("type") or "").lower()
    return "chapter" if "chapter" in t else "book" if t in ("book", "monograph", "edited-book", "reference-book") else ""


def _same(a, b):
    if {_kind(a), _kind(b)} == {"book", "chapter"}:  # a book and its chapter stay separate entries
        return False
    da, db = clean_doi(a["doi"]).lower(), clean_doi(b["doi"]).lower()
    return bool(da and da == db) or bool(_tkey(a) and _tkey(a) == _tkey(b))


def merge(papers):
    """Merge entries sharing a DOI or normalized title; the first keeps its fields, gaps are filled from later ones."""
    out = []
    for p in papers:
        q = next((q for q in out if _same(q, p)), None)
        if q is None:
            out.append(dict(p))
            continue
        for f in ("authors", "year", "venue", "doi", "url"):
            if not q.get(f):
                q[f] = p.get(f)  # safe for doi: _same never pairs a book with a chapter
    return out


def references(papers):
    """Return (list of APA lines, list of flag messages for the readiness report)."""
    lines, flagged = [], []
    for p in sorted(merge(papers), key=lambda p: cite(p)):
        au = authors_apa(p["authors"])
        if not p["authors"]:
            flagged.append("missing author: %s" % p["title"])
        if not p["year"]:
            flagged.append("missing year (n.d.): %s" % p["title"])
        line = "%s (%s). %s. %s" % (au, p["year"] or "n.d.", p["title"].rstrip("."), ("*%s*. " % p["venue"]) if p["venue"] else "")
        if p["doi"]:
            ok = doi_ok(p["doi"])
            line += "https://doi.org/" + clean_doi(p["doi"]) + ("" if ok else " [FLAG: malformed DOI]")
            if not ok:
                flagged.append("malformed DOI: %s (%s)" % (p["title"], p["doi"]))
        else:
            line += p["url"]
        lines.append(line.strip())
    return lines, flagged
