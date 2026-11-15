import re

import requests

from src.ra import bib, discovery, draft, encyclopedia, outline, quotes, rank, sentences
from src.ra.directions import rank_directions, render
from src.ra.gaps import find_gaps

ABS = ("Cyber attacks on power grids are increasing worldwide. We propose a detection framework using network traffic data. "
       "Results show that the framework reduces false alarms by 40 percent compared with baselines. "
       "A limitation is that only simulated data were tested. Further research is needed on real grid deployments.")
P = {"title": "Grid defence", "authors": ["Jane Smith", "Li Wei", "Omar Ali"], "year": 2022, "doi": "10.1000/abc123",
     "url": "http://x", "venue": "J. Test", "abstract": ABS}


def test_toc_and_boilerplate_rejected_and_logged():
    log = []
    toc = "Introduction Background Literature Review Methods Results Discussion Conclusion References Appendix"
    bp = "In this paper we present a new method for detecting anomalies in networks."
    out = sentences.process(toc + " " + bp, log)
    reasons = " ".join(x["reason"] for x in log)
    assert "table-of-contents" in reasons and "boilerplate" in reasons and not out


def test_length_limits():
    assert sentences.reject("Too short here.")
    assert sentences.reject(" ".join(["word"] * 81))


def test_verbatim_quotes():
    sents = sentences.process(ABS, [])
    kept, dropped = quotes.top_quotes(P, "power grid detection false alarms", sents)
    assert kept and all(k in ABS for k in kept) and not dropped
    assert not quotes.verify("Invented sentence that is not in the abstract at all.", ABS)


def test_dedupe_by_title_and_year():
    a = dict(P, abstract="")
    b = dict(P, title="GRID  defence!", abstract="has one")
    c = dict(P, year=2020)
    out = discovery.dedupe([a, b, c])
    assert len(out) == 2 and out[0]["abstract"] == "has one"


def test_search_with_mocked_apis(monkeypatch, tmp_path):
    def fake(url, params=None, text=False, pause=1.0):
        if "openalex" in url:
            return {"results": [{"title": "Grid defence", "publication_year": 2022, "doi": "https://doi.org/10.1/x", "id": "u",
                                 "authorships": [{"author": {"display_name": "A B"}}], "abstract_inverted_index": {"Hello": [0], "world": [1]}}]}
        if "semanticscholar" in url:
            return {"data": [{"title": "Grid Defence", "year": 2022, "authors": [], "externalIds": {}, "abstract": "x"}]}
        raise RuntimeError("down")
    monkeypatch.setattr(discovery, "_get", fake)
    res = discovery.search("power grid", 5)
    assert len(res) == 1 and res[0]["abstract"] == "Hello world"


def test_malformed_doi_flagged():
    bad = dict(P, title="Another paper", doi="not-a-doi")  # distinct title, else it now merges with P
    lines, flagged = bib.references([P, bad])
    assert len(flagged) == 1 and any("malformed DOI" in l for l in lines)
    assert bib.doi_ok("https://doi.org/10.1000/abc123") and not bib.doi_ok("10.12/x")


def _sections():
    sents = sentences.process(ABS, [])
    q = quotes.assign([{"text": t, "cite": bib.cite(P), "paper": 0, "kind": "quote"} for t in quotes.top_quotes(P, "grid", sents)[0]], "grid")
    m = [{"text": t, "cite": bib.cite(P), "paper": 0, "kind": "method"} for t in quotes.top_quotes(P, "grid", sents, 1, "method")[0]]
    gaps = find_gaps([P, dict(P, abstract="", title="Other")], [])
    bg = [{"text": "Wikipedia says grids carry power.", "cite": "(Wikipedia: Grid - http://w)", "kind": "background"}]
    return q, outline.build(q, m, gaps, bg, ["ref"]), gaps


def test_every_ai_draft_has_citation():
    _, secs, _ = _sections()
    md = draft.render("grid", secs)
    lines = [l for l in md.splitlines() if draft.LABEL in l]
    assert lines and all(re.search(r"\([^()]+\)\s*\[AI-DRAFT", l) for l in lines)


def test_encyclopedia_never_a_quote():
    q, secs, _ = _sections()
    assert all("Wikipedia" not in x["text"] for x in q)
    assert not quotes.verify("Wikipedia says grids carry power.", ABS)
    md = draft.render("grid", secs)
    assert all("[BACKGROUND" in l for l in md.splitlines() if "Wikipedia says" in l)


def test_no_abstract_paper_excluded_from_gaps():
    _, _, gaps = _sections()
    assert gaps and all(g["paper"] == 0 for g in gaps)
    assert any("limitation" in g["sentence"].lower() for g in gaps)


def _http_error(code):
    r = requests.Response()
    r.status_code = code
    return requests.HTTPError(response=r)


def test_fill_abstracts_fallback_order(monkeypatch):
    calls = []

    def fake(url, params=None, text=False, pause=1.0):
        calls.append(url)
        if "openalex" in url:
            return {"abstract_inverted_index": None}
        if "crossref" in url:
            return {"message": {"abstract": "<jats:p>Real <jats:italic>abstract</jats:italic> text.</jats:p>"}}
        raise AssertionError("semantic scholar must not be called once crossref succeeds")
    monkeypatch.setattr(discovery, "_get", fake)
    p = discovery.fill_abstracts([dict(P, abstract="")])[0]
    assert p["abstract"] == "Real abstract text." and p["abstract_source"] == "crossref"
    assert "openalex" in calls[0] and "crossref" in calls[1]


def test_openalex_inverted_index_rebuilt(monkeypatch):
    monkeypatch.setattr(discovery, "_get", lambda *a, **k: {"abstract_inverted_index": {"world": [1], "Hello": [0], "again": [2]}})
    p = discovery.fill_abstracts([dict(P, abstract="")])[0]
    assert p["abstract"] == "Hello world again" and p["abstract_source"] == "openalex"


def test_semantic_scholar_backoff_then_success(monkeypatch):
    waits, n = [], {"s2": 0}

    def fake(url, params=None, text=False, pause=1.0):
        if "semanticscholar" not in url:
            raise _http_error(404)
        n["s2"] += 1
        if n["s2"] < 3:
            raise _http_error(429)
        return {"abstract": "Found late."}
    monkeypatch.setattr(discovery, "_get", fake)
    monkeypatch.setattr(discovery.time, "sleep", waits.append)
    p = discovery.fill_abstracts([dict(P, abstract="")])[0]
    assert waits == [2, 4] and p["abstract_source"] == "semantic" and p["abstract"] == "Found late."


def test_all_sources_fail_marks_no_abstract(monkeypatch):
    waits = []

    def fake(url, params=None, text=False, pause=1.0):
        raise _http_error(429 if "semanticscholar" in url else 500)
    monkeypatch.setattr(discovery, "_get", fake)
    monkeypatch.setattr(discovery.time, "sleep", waits.append)
    p = discovery.fill_abstracts([dict(P, abstract="")])[0]
    assert waits == [2, 4, 8, 16] and p["abstract"] == "" and p["abstract_source"] == "no abstract"
    assert not sentences.process(p["abstract"], []) and not find_gaps([p], [])


def test_each_quote_once_and_abstract_distinct_papers():
    items = [{"text": "Finding %d about grid attacks and detection results." % i, "cite": "(X, 2020)", "paper": i // 3, "kind": "quote"}
             for i in range(12)]
    quotes.assign(items, "grid attacks detection")
    assert all(q.get("section") for q in items)
    abs_papers = [q["paper"] for q in items if q["section"] == "Abstract"]
    assert len(abs_papers) == len(set(abs_papers))
    secs = outline.build(items, [], [], [], [])
    md = draft.render("grid", secs + [{"name": "Dup", "note": None, "blocks": [{"heading": None, "items": items}]}])
    assert all(md.count('"%s"' % q["text"]) == 1 for q in items)


def test_empty_directions_message():
    text = render([], "grid", 3)
    assert "Not enough gap evidence this run: 3 papers had abstracts" in text


def test_apa_authors_ampersand_and_et_al():
    assert bib.authors_apa(["Jane A. Smith", "Li Wei"]) == "Smith, J. A., & Wei, L."
    assert bib.authors_apa(["Smith, Jane", "Li Wei", "Omar Ali"]) == "Smith, J., Wei, L., & Ali, O."
    assert bib.authors_apa(["A%d Person" % i for i in range(8)]) == "Person, A., et al."
    assert bib.references([P])[0][0].startswith("Smith, J., Wei, L., & Ali, O. (2022).")


def test_author_not_listed_and_missing_year_flagged():
    lines, flagged = bib.references([dict(P, authors=[], year=None)])
    assert lines[0].startswith("[author not listed] (n.d.).") and "Anon." not in lines[0]
    assert bib.cite(dict(P, authors=[])) == "([author not listed], 2022)"
    assert any("missing author" in f for f in flagged) and any("missing year" in f for f in flagged)


def test_duplicate_references_merged_book_and_chapter_kept():
    same_doi = dict(P, title="Different title", authors=[])
    same_title = dict(P, title="GRID defence.", doi=None, venue="")
    assert len(bib.references([P, same_doi, same_title])[0]) == 1
    book = dict(P, title="Grid Handbook", type="book", doi="10.1000/book")
    chap = dict(P, title="Grid Handbook", type="book-chapter", doi=None)
    merged = bib.merge([book, chap])
    assert len(merged) == 2 and merged[1]["doi"] is None


def test_generic_term_direction_dropped():
    g = {"paper": 0, "cite": "(X, 2020)", "year": 2020, "id": "G1",
         "sentence": "Future work should adopt different approaches to existing capabilities."}
    assert rank_directions([g], lambda q: []) == []
    assert "No candidate directions passed the relevance and term filters this run" in render([], "grid", 2, evidence=True)


def test_off_topic_paper_filtered():
    key = rank.key_terms("power grid attacks")
    off = dict(P, title="Cooking pasta at home", abstract="Boil water and add salt before the pasta.")
    assert rank.relevant(P, key) and not rank.relevant(off, key)
    from src.ra import readiness
    assert "Cooking pasta at home" in readiness.report("grid", "", [], [], [], off_topic=[off["title"]]).split("Filtered as off-topic")[1]


def test_hyphenated_title_on_topic():
    p = dict(P, title="Cyber-Terrorism threats", abstract="")
    assert rank.relevant(p, rank.key_terms("cyber terrorism"))


def test_capped_match_count():
    _, _, gaps = _sections()
    dirs = rank_directions(gaps, lambda q: list(range(10)))
    assert dirs and all(d["confidence"] == "low" for d in dirs)
    assert "10+ matches (capped)" in render(dirs, "grid") and "Search matches: 10\n" not in render(dirs, "grid")


def test_generic_filler_rejected():
    log = []
    s = "Cyber security has become a central concern for every modern power grid operator."
    assert not sentences.process(s, log) and log[0]["reason"] == "generic filler"


def test_model_rewrite_acceptance():
    c = "(Smith et al., 2022)"
    src = "The framework reduces false alarms by 40 percent."
    assert draft.accept(src, "The framework reduces false alarms by 40 percent. " + c, c)
    assert not draft.accept(src, "Totally different invented claim about oceans. " + c, c)
    assert not draft.accept(src, "The framework reduces false alarms by 40 percent. (Other, 2001)", c)


def test_directions_never_confirmed():
    _, _, gaps = _sections()
    dirs = rank_directions(gaps, lambda q: [1, 2])
    text = render(dirs, "grid")
    assert dirs and "confirmed" not in text.lower() and "human must verify" in text
    assert all("confirmed" not in (d["status"] + d["confidence"]).lower() for d in dirs)
