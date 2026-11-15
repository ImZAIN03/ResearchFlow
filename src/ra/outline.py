"""Thesis outline. Items are dicts {text, cite, kind}; kind in quote|method|gap|background|ref."""
from .directions import group

BLANK_METHOD = ("[STUDENT: Research design: ____ | Sample/data source: ____ | Instruments: ____ | Analysis plan: ____ | Ethics/approvals: ____ ] "
                "Methods below are only what the reviewed papers used; choosing yours is your decision.")


def build(quotes, methods, gaps, background, refs):
    sec = lambda name: [q for q in quotes if q.get("section") == name]  # sections set by quotes.assign
    litq = sec("Literature Review")
    themes = group([q["text"] for q in litq])
    lit = [{"heading": ", ".join(t).title() or "General", "items": [litq[i] for i in idx]} for t, idx in themes]
    allg = [{"text": g["sentence"], "cite": g["cite"], "kind": "gap"} for g in gaps]
    prompts = allg
    S = lambda name, blocks=(), note=None: {"name": name, "blocks": list(blocks), "note": note}
    return [
        S("Abstract", [{"heading": None, "items": sec("Abstract")}], "[STUDENT: write this last, from your own results and contribution.]"),
        S("Introduction", [{"heading": "Background", "items": background}, {"heading": "Context from the literature", "items": sec("Introduction")}]),
        S("Literature Review", lit),
        S("Research Gaps", [{"heading": None, "items": allg}], "Candidate gaps taken from authors' own limitation sentences; verify novelty yourself."),
        S("Methodology", [{"heading": "Methods used in the reviewed papers", "items": methods}], BLANK_METHOD),
        S("Results", note="[STUDENT: blank. Results must come from your own data; nothing is drafted here.]"),
        S("Discussion", [{"heading": "Prompts: how does your work address these?", "items": prompts}], "[STUDENT: blank. Interpret your own results against the literature.]"),
        S("Conclusion", note="[STUDENT: blank. State your own contribution once Results and Discussion exist.]"),
        S("References", [{"heading": None, "items": [{"text": r, "cite": "", "kind": "ref"} for r in refs]}]),
    ]
