"""Background definitions from Wikipedia + Wikidata. Background ONLY, never evidence."""
from urllib.parse import quote

from .discovery import _get


def background(topic):
    """Return a list of {text,title,url,source,use}; empty if nothing found."""
    out = []
    try:
        d = _get("https://en.wikipedia.org/api/rest_v1/page/summary/" + quote(topic.replace(" ", "_")))
        if d.get("extract"):
            out.append({"text": d["extract"], "title": d.get("title", topic), "source": "Wikipedia",
                        "url": (d.get("content_urls") or {}).get("desktop", {}).get("page", ""), "use": "background only"})
    except Exception as e:
        print("warn: wikipedia:", e)
    try:
        d = _get("https://www.wikidata.org/w/api.php", {"action": "wbsearchentities", "search": topic, "language": "en", "format": "json", "limit": 1})
        for h in d.get("search", []):
            if h.get("description"):
                out.append({"text": "%s: %s." % (h["label"], h["description"].capitalize()), "title": h["label"], "source": "Wikidata",
                            "url": "https:" + h.get("url", "") if h.get("url", "").startswith("//") else h.get("url", ""), "use": "background only"})
    except Exception as e:
        print("warn: wikidata:", e)
    return out
