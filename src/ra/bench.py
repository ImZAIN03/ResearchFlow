"""Benchmark a scan: quote exactness, counts, and gap-label precision. Usage: python -m src.ra.bench <slug>"""
import csv
import json
import sys
from pathlib import Path

from .quotes import verify


def main(slug):
    scan = json.loads(Path("output/%s-scan.json" % slug).read_text("utf-8"))
    q = scan["quotes"]
    ok = sum(verify(x["text"], scan["papers"][x["paper"]]["abstract"]) for x in q)
    gaps = [g for g in scan["gaps"] if not g.get("none")]
    print("(a) quotes exactly matching their abstract: %d/%d (%.0f%%)" % (ok, len(q), 100 * ok / len(q) if q else 0))
    print("(b) papers: %d, gaps: %d" % (len(scan["papers"]), len(gaps)))
    f = Path("output/%s-gap-labels.csv" % slug)
    if not f.exists():
        with f.open("w", newline="", encoding="utf-8") as h:
            w = csv.writer(h)
            w.writerow(["gap_id", "label"])
            w.writerows([g["id"], ""] for g in gaps)
        print("(c) wrote blank %s: fill label with real/weak/wrong, then rerun" % f)
        return
    rows = [r for r in csv.DictReader(f.open(encoding="utf-8")) if r["label"].strip()]
    if not rows:
        print("(c) %s is still blank" % f)
        return
    real = sum(r["label"].strip().lower() == "real" for r in rows)
    print("(c) gap precision (real / labelled): %d/%d = %.0f%%" % (real, len(rows), 100 * real / len(rows)))


if __name__ == "__main__":
    main(sys.argv[1])
