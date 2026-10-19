"""Paper discovery: search free academic APIs for papers matching a research query."""
from __future__ import annotations

import logging
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

ARXIV_API_URL = "http://export.arxiv.org/api/query"
SEMANTIC_SCHOLAR_API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
SEMANTIC_SCHOLAR_FIELDS = "title,abstract,year,venue,authors,url,openAccessPdf,externalIds,citationCount"

REQUEST_TIMEOUT = 20


@dataclass
class Paper:
    title: str
    authors: List[str] = field(default_factory=list)
    abstract: str = ""
    year: Optional[int] = None
    venue: str = ""
    url: str = ""
    pdf_url: Optional[str] = None
    doi: Optional[str] = None
    source: str = ""
    citation_count: int = 0

    @property
    def first_author_surname(self) -> str:
        if not self.authors:
            return "Unknown"
        return self.authors[0].split()[-1]

    def dedup_key(self) -> str:
        return "".join(ch.lower() for ch in self.title if ch.isalnum())


@dataclass
class SearchResult:
    papers: List[Paper] = field(default_factory=list)  # unique papers, most cited first
    found_by_source: Dict[str, int] = field(default_factory=dict)  # records returned, before deduplication
    duplicates: int = 0


class PaperDiscovery:
    """Searches multiple free academic APIs and returns deduplicated Paper objects."""

    def __init__(self, session: Optional[requests.Session] = None):
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": "ResearchFlow/1.0 (research assistant tool)"})

    def search(
        self,
        query: str,
        max_results: int = 20,
        sources: Optional[List[str]] = None,
    ) -> List[Paper]:
        return self.search_queries([query], max_results, sources).papers[:max_results]

    def search_queries(
        self,
        queries: List[str],
        per_query: int = 20,
        sources: Optional[List[str]] = None,
    ) -> "SearchResult":
        """Runs every query against every source; returns all unique papers, most cited first.

        Records found by more than one query or source are counted as duplicates (for the review's
        methodology flowchart) and kept once.
        """
        sources = sources or ["arxiv", "semantic_scholar"]
        result = SearchResult()
        seen_keys = set()

        for query in queries:
            for source in sources:
                try:
                    if source == "arxiv":
                        papers = self.search_arxiv(query, per_query)
                    elif source == "semantic_scholar":
                        papers = self.search_semantic_scholar(query, per_query)
                    else:
                        logger.warning("Unknown paper source: %s", source)
                        continue
                except requests.RequestException as exc:
                    logger.warning("Search failed for source %s (query %r): %s", source, query, exc)
                    continue

                result.found_by_source[source] = result.found_by_source.get(source, 0) + len(papers)
                for paper in papers:
                    key = paper.dedup_key()
                    if not key or key in seen_keys:
                        result.duplicates += 1
                        continue
                    seen_keys.add(key)
                    result.papers.append(paper)

        result.papers.sort(key=lambda p: (p.citation_count, p.year or 0), reverse=True)
        return result

    def search_arxiv(self, query: str, max_results: int = 20) -> List[Paper]:
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        response = self.session.get(ARXIV_API_URL, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return self._parse_arxiv_feed(response.text)

    @staticmethod
    def _parse_arxiv_feed(xml_text: str) -> List[Paper]:
        ns = {
            "atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom",
        }
        root = ET.fromstring(xml_text)
        papers = []
        for entry in root.findall("atom:entry", ns):
            title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
            title = " ".join(title.split())
            summary = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()
            summary = " ".join(summary.split())
            published = entry.findtext("atom:published", default="", namespaces=ns) or ""
            year = int(published[:4]) if published[:4].isdigit() else None
            authors = [
                " ".join((author.findtext("atom:name", default="", namespaces=ns) or "").split())
                for author in entry.findall("atom:author", ns)
            ]
            authors = [a for a in authors if a]
            entry_id = entry.findtext("atom:id", default="", namespaces=ns) or ""

            pdf_url = None
            for link in entry.findall("atom:link", ns):
                if link.get("title") == "pdf" or link.get("type") == "application/pdf":
                    pdf_url = link.get("href")
            if not pdf_url and "/abs/" in entry_id:
                pdf_url = entry_id.replace("/abs/", "/pdf/")

            doi = entry.findtext("arxiv:doi", default=None, namespaces=ns)

            if not title:
                continue

            papers.append(
                Paper(
                    title=title,
                    authors=authors,
                    abstract=summary,
                    year=year,
                    venue="arXiv",
                    url=entry_id,
                    pdf_url=pdf_url,
                    doi=doi,
                    source="arxiv",
                    citation_count=0,
                )
            )
        return papers

    def search_semantic_scholar(self, query: str, max_results: int = 20) -> List[Paper]:
        params = {
            "query": query,
            "limit": min(max_results, 100),
            "fields": SEMANTIC_SCHOLAR_FIELDS,
        }
        response = self.session.get(SEMANTIC_SCHOLAR_API_URL, params=params, timeout=REQUEST_TIMEOUT)
        # The unauthenticated API allows roughly one request per second, which several queries in a
        # row easily exceed, so back off progressively.
        for delay in (2, 5, 10):
            if response.status_code != 429:
                break
            logger.warning("Semantic Scholar rate limit hit; retrying in %ds", delay)
            time.sleep(delay)
            response = self.session.get(SEMANTIC_SCHOLAR_API_URL, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()

        papers = []
        for item in data.get("data", []) or []:
            authors = [a.get("name", "") for a in (item.get("authors") or []) if a.get("name")]
            open_access = item.get("openAccessPdf") or {}
            external_ids = item.get("externalIds") or {}
            title = item.get("title") or ""
            if not title:
                continue
            papers.append(
                Paper(
                    title=title,
                    authors=authors,
                    abstract=item.get("abstract") or "",
                    year=item.get("year"),
                    venue=item.get("venue") or "",
                    url=item.get("url") or "",
                    pdf_url=open_access.get("url"),
                    doi=external_ids.get("DOI"),
                    source="semantic_scholar",
                    citation_count=item.get("citationCount") or 0,
                )
            )
        return papers
