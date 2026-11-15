"""Readiness report: what is drafted, what the student must still do."""
import re

from .draft import LABEL

MUST = ["Results: run your own study and write it up", "Discussion: interpret your results against the literature",
        "Methodology: choose design, sample, instruments, analysis", "Your thesis's own contribution (Abstract, Conclusion)"]


def report(topic, draft_md, flagged, unverified, dirs, no_abstract=(), off_topic=()):
    t = ["# Readiness report: %s\n" % topic, "**Not submission-ready.** This is a sourced skeleton.\n",
         "## Sections\n", "| Section | Words | Verbatim quote lines | AI-DRAFT lines | Student blanks |", "|---|---|---|---|---|"]
    blanks = []
    for part in re.split(r"^## ", draft_md, flags=re.M)[1:]:
        name, _, body = part.partition("\n")
        b = re.findall(r"\[STUDENT:[^\]]*\]", body)
        blanks += ["%s: %s" % (name, x) for x in b]
        t.append("| %s | %d | %d | %d | %d |" % (name, len(body.split()), len(re.findall(r'^- "', body, re.M)), body.count(LABEL), len(b)))
    t += ["\n## Blanks the student must fill\n"] + ["- " + m for m in MUST] + ["- " + b for b in blanks]
    t += ["\n## Flagged references\n"] + (["- " + f for f in flagged] or ["- none"])
    t += ["\n## Papers without abstracts (review manually)\n"] + (["- " + x for x in no_abstract] or ["- none"])
    t += ["\n## Unverified sentences (dropped, not in draft)\n"] + (["- " + u for u in unverified] or ["- none"])
    t += ["\n## Filtered as off-topic\n"] + (["- " + x for x in off_topic] or ["- none"])
    t += ["\n## Direction candidates needing novelty review\n"] + ["- %s (confidence %s, %s)" % (", ".join(d["terms"]), d["confidence"], d["matches"]) for d in dirs] + ([] if dirs else ["- none"])
    return "\n".join(t) + "\n"
