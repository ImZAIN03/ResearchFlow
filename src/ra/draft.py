"""Fill the outline from verified material. Verbatim by default; a local model may rewrite one sentence at a time."""
import re

import requests

from .sentences import content

LABEL = "[AI-DRAFT: verify and rewrite]"
CITE_RE = re.compile(r"\((?:[^()]+),\s*(?:\d{4}|n\.d\.)\)")


def ollama(model):
    def gen(prompt):
        try:
            r = requests.post("http://localhost:11434/api/generate", json={"model": model, "prompt": prompt, "stream": False}, timeout=120)
            return r.json().get("response", "")
        except Exception as e:
            print("warn: model call failed:", e)
            return ""
    return gen


def accept(src, out, cite):
    """Keep a rewrite only if it carries the same citation and >=70% of its content words are in the source."""
    if out.count(cite) != 1 or any(c != cite for c in CITE_RE.findall(out)):
        return False
    body = set(content(out.replace(cite, "")))
    return bool(body) and len(body & set(content(src))) / len(body) >= 0.7


def line(it, gen=None):
    t, c, k = it["text"], it["cite"], it["kind"]
    if k == "ref":
        return "- " + t
    if k == "background":
        return '- [BACKGROUND, not evidence] "%s" %s %s' % (t, c, LABEL)
    if gen:
        out = gen("Rewrite this sentence in plain academic English. Add no new facts. End with the citation %s exactly.\n%s %s" % (c, t, c)).strip()
        if accept(t, out, c):
            return "- %s %s" % (out, LABEL)
    return '- "%s" %s %s' % (t, c, LABEL)


def render(topic, sections, model=None):
    gen = ollama(model) if model else None
    md = ["# Draft skeleton: %s\n" % topic, "> NOT a submission-ready thesis. Every line marked AI-DRAFT must be verified and rewritten by you. Quoted text is verbatim from abstracts.\n"]
    seen = set()  # a quote appears at most once in the whole draft
    for s in sections:
        md.append("## %s\n" % s["name"])
        if s["note"]:
            md.append(s["note"] + "\n")
        for b in s["blocks"]:
            items = []
            for i in b["items"]:
                if i["kind"] == "quote":
                    if i["text"] in seen:
                        continue
                    seen.add(i["text"])
                items.append(i)
            if b["heading"] and items:
                md.append("### %s\n" % b["heading"])
            md += [line(i, gen) for i in items]
            md.append("")
    return "\n".join(md)
