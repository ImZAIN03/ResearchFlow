"""python -m src.ra.cli "topic" --max-papers 10 [--model NAME]"""
import argparse
import json
import re
from pathlib import Path

from . import bib, directions, discovery, draft, outline, rank, readiness
from .directions import rank_directions, render
from .encyclopedia import background
from .gaps import find_gaps
from .quotes import assign, top_quotes
from .sentences import process


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("topic")
    ap.add_argument("--max-papers", type=int, default=10)
    ap.add_argument("--model", default=None, help="optional local Ollama model for sentence rewrites")
    a = ap.parse_args(argv)
    slug = re.sub(r"[^a-z0-9]+", "-", a.topic.lower()).strip("-")
    log, quotes, methods, dropped = [], [], [], []
    papers = discovery.fill_abstracts(discovery.search(a.topic, a.max_papers))  # not in search(): direction queries only need counts
    print("found %d papers, %d without abstract" % (len(papers), sum(p["abstract_source"] == "no abstract" for p in papers)))
    for i, p in enumerate(papers):
        sents = process(p["abstract"], log)
        for label, dest in (("finding", quotes), ("method", methods)):
            kept, bad = top_quotes(p, a.topic, sents, 3 if label == "finding" else 1, label)
            dest += [{"text": t, "cite": bib.cite(p), "paper": i, "kind": "quote" if label == "finding" else "method"} for t in kept]
            dropped += bad
    assign(quotes, a.topic)
    key = rank.key_terms(a.topic)
    on = {i for i, p in enumerate(papers) if rank.relevant(p, key)}
    off_topic = [p["title"] for i, p in enumerate(papers) if i not in on]
    gaps = [g for g in find_gaps(papers, log) if g["paper"] in on]  # off-topic papers support no gap or direction
    no_abs = ["%s %s" % (bib.cite(p), p["title"]) for p in papers if not p["abstract"]]
    evidence = len({g["paper"] for g in gaps}) >= 2
    dirs = rank_directions(gaps, lambda q: discovery.search(q, directions.CAP), a.topic, [papers[i]["title"] for i in on]) if evidence else []
    bg = [{"text": b["text"], "cite": "(%s: %s - %s)" % (b["source"], b["title"], b["url"].replace("(", "%28").replace(")", "%29")), "kind": "background"} for b in background(a.topic)]
    refs, flagged = bib.references(papers)
    md = draft.render(a.topic, outline.build(quotes, methods, gaps, bg, refs), a.model)
    out = Path("output")
    out.mkdir(exist_ok=True)
    (out / (slug + "-thesis-draft.md")).write_text(md, "utf-8")
    (out / (slug + "-references.md")).write_text("# References\n\n" + "\n\n".join(refs) + "\n", "utf-8")
    (out / (slug + "-directions.md")).write_text(render(dirs, a.topic, len(papers) - len(no_abs), evidence), "utf-8")
    (out / (slug + "-readiness.md")).write_text(readiness.report(a.topic, md, flagged, dropped, dirs, no_abs, off_topic), "utf-8")
    (out / (slug + "-scan.json")).write_text(json.dumps({"topic": a.topic, "papers": papers, "quotes": quotes, "methods": methods, "dropped": dropped,
                                                          "gaps": gaps, "directions": dirs, "flagged_dois": flagged, "rejections": log}, indent=1), "utf-8")
    print("wrote output/%s-*.md and -scan.json. This is a draft skeleton, not a thesis." % slug)


if __name__ == "__main__":
    main()
