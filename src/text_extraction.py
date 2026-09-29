"""Downloads paper PDFs and extracts their text, falling back to abstracts on failure."""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import List, Optional

import requests
from pypdf import PdfReader

from .paper_discovery import Paper

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = 30
MAX_PDF_BYTES = 50 * 1024 * 1024  # 50 MB safety cap


class TextExtractionError(Exception):
    """Raised when a PDF cannot be downloaded or parsed."""


class TextExtractor:
    def __init__(self, cache_dir: Optional[Path] = None, session: Optional[requests.Session] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/pdf_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": "ResearchFlow/1.0 (research assistant tool)"})

    def _cache_path(self, url: str) -> Path:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]
        return self.cache_dir / f"{digest}.pdf"

    def download_pdf(self, url: str) -> Path:
        dest = self._cache_path(url)
        if dest.exists():
            return dest

        tmp_path = dest.with_suffix(".part")
        with self.session.get(url, timeout=REQUEST_TIMEOUT, stream=True) as response:
            response.raise_for_status()
            content_length = int(response.headers.get("Content-Length", 0) or 0)
            if content_length and content_length > MAX_PDF_BYTES:
                raise TextExtractionError(f"PDF too large ({content_length} bytes): {url}")

            downloaded = 0
            with open(tmp_path, "wb") as handle:
                for chunk in response.iter_content(chunk_size=8192):
                    downloaded += len(chunk)
                    if downloaded > MAX_PDF_BYTES:
                        handle.close()
                        tmp_path.unlink(missing_ok=True)
                        raise TextExtractionError(f"PDF exceeded size cap while downloading: {url}")
                    handle.write(chunk)

        tmp_path.replace(dest)
        return dest

    @staticmethod
    def extract_text_from_pdf(path: Path, max_pages: Optional[int] = None) -> str:
        reader = PdfReader(str(path))
        pages = reader.pages[:max_pages] if max_pages else reader.pages
        text_parts = []
        for page in pages:
            try:
                text_parts.append(page.extract_text() or "")
            except Exception as exc:  # pypdf can raise various errors on malformed PDFs
                logger.debug("Failed to extract a page from %s: %s", path, exc)
        return "\n".join(text_parts).strip()

    def extract_text_for_paper(self, paper: Paper, max_pages: Optional[int] = 40) -> str:
        """Returns full-text extracted from the paper's PDF, or its abstract if that fails."""
        if paper.pdf_url:
            try:
                pdf_path = self.download_pdf(paper.pdf_url)
                text = self.extract_text_from_pdf(pdf_path, max_pages=max_pages)
                if text:
                    return text
                logger.info("No extractable text in PDF for %r; falling back to abstract", paper.title)
            except (requests.RequestException, TextExtractionError) as exc:
                logger.info("Could not fetch PDF for %r (%s); falling back to abstract", paper.title, exc)
            except Exception as exc:
                logger.info("PDF extraction failed for %r (%s); falling back to abstract", paper.title, exc)
        return paper.abstract

    @staticmethod
    def chunk_text(text: str, max_chars: int = 8000, overlap: int = 200) -> List[str]:
        if len(text) <= max_chars:
            return [text]
        chunks = []
        start = 0
        while start < len(text):
            end = start + max_chars
            chunks.append(text[start:end])
            if end >= len(text):
                break
            start = end - overlap
        return chunks
