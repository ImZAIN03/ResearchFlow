"""Formats Paper objects into in-text citations and bibliography entries."""
from __future__ import annotations

from typing import List

from .paper_discovery import Paper


class CitationFormatter:
    """Generates in-text citations and full bibliography entries in common academic styles."""

    SUPPORTED_STYLES = ("apa", "mla", "chicago", "ieee")

    def format_bibliography(self, papers: List[Paper], style: str = "apa") -> str:
        formatter = self._formatter_for(style)
        entries = sorted((formatter(p) for p in papers), key=str.lower)
        return "\n\n".join(entries)

    def in_text_citation(self, paper: Paper, style: str = "apa") -> str:
        style = style.lower()
        surname = paper.first_author_surname
        suffix = " et al." if len(paper.authors) > 1 else ""
        year = paper.year if paper.year is not None else "n.d."
        if style in ("apa", "chicago"):
            return f"({surname}{suffix}, {year})"
        if style == "mla":
            return f"({surname}{suffix})"
        if style == "ieee":
            return f"[{surname}{suffix}, {year}]"
        raise ValueError(f"Unsupported citation style: {style}. Choose from {self.SUPPORTED_STYLES}")

    def _formatter_for(self, style: str):
        style = style.lower()
        mapping = {
            "apa": self.format_apa,
            "mla": self.format_mla,
            "chicago": self.format_chicago,
            "ieee": self.format_ieee,
        }
        if style not in mapping:
            raise ValueError(f"Unsupported citation style: {style}. Choose from {self.SUPPORTED_STYLES}")
        return mapping[style]

    @staticmethod
    def _apa_authors(authors: List[str]) -> str:
        def fmt(name: str) -> str:
            parts = name.split()
            if len(parts) < 2:
                return name
            surname = parts[-1]
            initials = " ".join(f"{p[0]}." for p in parts[:-1])
            return f"{surname}, {initials}"

        if not authors:
            return "Unknown Author"
        formatted = [fmt(a) for a in authors]
        if len(formatted) == 1:
            return formatted[0]
        if len(formatted) <= 20:
            return ", ".join(formatted[:-1]) + f", & {formatted[-1]}"
        return ", ".join(formatted[:19]) + f", ... {formatted[-1]}"

    def format_apa(self, paper: Paper) -> str:
        authors = self._apa_authors(paper.authors)
        year = paper.year if paper.year is not None else "n.d."
        title = paper.title.rstrip(".")
        venue = f" {paper.venue}." if paper.venue else ""
        link = f" https://doi.org/{paper.doi}" if paper.doi else (f" {paper.url}" if paper.url else "")
        return f"{authors} ({year}). {title}.{venue}{link}".strip()

    def format_mla(self, paper: Paper) -> str:
        if not paper.authors:
            author_str = ""
        else:
            first = paper.authors[0]
            parts = first.split()
            first_fmt = f"{parts[-1]}, {' '.join(parts[:-1])}" if len(parts) > 1 else first
            author_str = first_fmt + (", et al." if len(paper.authors) > 1 else ".")
        title = f'"{paper.title.rstrip(".")}."'
        venue = f" {paper.venue}," if paper.venue else ""
        year = f" {paper.year}." if paper.year is not None else ""
        link = f" {paper.url}" if paper.url else ""
        return f"{author_str} {title}{venue}{year}{link}".strip()

    def format_chicago(self, paper: Paper) -> str:
        if not paper.authors:
            author_str = ""
        else:
            first = paper.authors[0]
            parts = first.split()
            first_fmt = f"{parts[-1]}, {' '.join(parts[:-1])}" if len(parts) > 1 else first
            rest = ", ".join(paper.authors[1:])
            author_str = first_fmt + (f", {rest}" if rest else "") + "."
        title = f'"{paper.title.rstrip(".")}."'
        venue = f" {paper.venue}" if paper.venue else ""
        year = f" ({paper.year})" if paper.year is not None else ""
        link = f" {paper.url}" if paper.url else ""
        return f"{author_str} {title}{venue}{year}.{link}".strip()

    def format_ieee(self, paper: Paper) -> str:
        def initials_first(name: str) -> str:
            parts = name.split()
            if len(parts) < 2:
                return name
            initials = " ".join(f"{p[0]}." for p in parts[:-1])
            return f"{initials} {parts[-1]}"

        authors = ", ".join(initials_first(a) for a in paper.authors) or "Unknown Author"
        title = f'"{paper.title.rstrip(".")}"'
        venue = f", {paper.venue}" if paper.venue else ""
        year = f", {paper.year}" if paper.year is not None else ""
        return f"{authors}, {title}{venue}{year}.".strip()
