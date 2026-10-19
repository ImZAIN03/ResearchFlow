"""ResearchFlow CLI: discover papers, extract text, synthesize findings, and export a report.

Run with:  python -m src.main "your research topic"
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from .citation_formatter import CitationFormatter
from .export import Exporter
from .llm_synthesis import (
    DEFAULT_MODEL,
    PLACEHOLDER_PATTERN,
    LLMSynthesizer,
    OllamaConnectionError,
    PaperSummary,
    content_issue,
    sources_block,
)
from .outline_generator import Outline, OutlineGenerator
from .paper_discovery import PaperDiscovery
from .text_extraction import TextExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("researchflow")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="researchflow",
        description="Discover, synthesize, and write up literature on a research topic.",
    )
    parser.add_argument("topic", help="The research topic or question to investigate")
    parser.add_argument("--max-papers", type=int, default=15, help="Maximum number of papers to include")
    parser.add_argument(
        "--sources",
        default="arxiv,semantic_scholar",
        help="Comma-separated list of sources to search (arxiv, semantic_scholar)",
    )
    parser.add_argument(
        "--citation-style",
        default="apa",
        choices=CitationFormatter.SUPPORTED_STYLES,
        help="Citation style for the bibliography",
    )
    parser.add_argument("--formats", default="md", help="Comma-separated export formats: md, docx, pdf")
    parser.add_argument("--output-dir", default="output", help="Directory to write exported files to")
    parser.add_argument("--data-dir", default="data", help="Directory for cached downloads (e.g. PDFs)")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help="Ollama model to use for synthesis (e.g. tinyllama, phi)"
    )
    return parser.parse_args(argv)


def slugify(text: str) -> str:
    slug = "".join(c.lower() if c.isalnum() else "-" for c in text)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-")[:60] or "report"


def check_content_quality(
    outline: Outline, synthesis: str, research_gaps: str, summaries: list[PaperSummary]
) -> list[str]:
    """Logs which parts of the report are weak before export; returns the warnings.

    Doesn't block export: paper-derived fallback content is still real content, but the user should
    know which sections the model failed to write (a larger model, e.g. --model phi, usually helps).
    """
    warnings: list[str] = []
    sources = sources_block(summaries)

    for section in outline.sections:
        if section.origin == "gaps":
            continue  # names of the research gaps, which are checked in full below
        if section.origin == "fallback":
            # Extracted from the papers by construction, so the grounding check doesn't apply.
            warnings.append(
                f"Outline section {section.heading!r} was built from paper abstracts (model output rejected)"
            )
            continue
        problem = content_issue("\n".join(section.points), min_sentences=2, min_words=15, source=sources)
        if problem:
            warnings.append(f"Outline section {section.heading!r} is weak: {problem}")

    # Research gaps describe what the papers *don't* cover, so their vocabulary naturally differs from
    # the sources; model-written gaps were already grounding-checked when generated.
    for label, text, min_sentences, source in (
        ("Literature synthesis", synthesis, 4, f"{sources}\n{synthesis}"),
        ("Research gaps", research_gaps, 2, None),
    ):
        problem = content_issue(text, min_sentences=min_sentences, min_words=20, source=source)
        if problem:
            warnings.append(f"{label} is weak: {problem}")

    for s in summaries:
        if not s.from_llm:
            warnings.append(f"Summary of {s.paper.title[:60]!r} was taken from its abstract (model output rejected)")
        elif PLACEHOLDER_PATTERN.search(s.summary):
            warnings.append(f"Summary of {s.paper.title[:60]!r} contains placeholder text")

    model_sections = sum(1 for s in outline.sections if s.origin != "fallback")
    logger.info(
        "Content check: %d/%d outline sections written by the model, %d/%d summaries model-written",
        model_sections, len(outline.sections), sum(s.from_llm for s in summaries), len(summaries),
    )
    for warning in warnings:
        logger.warning("Low-quality content: %s", warning)
    if not warnings:
        logger.info("Content check passed: every section has real, non-placeholder content")
    return warnings


def run(args: argparse.Namespace) -> list[Path]:
    load_dotenv()

    topic = args.topic
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    formats = [f.strip().lower() for f in args.formats.split(",") if f.strip()]

    logger.info("Discovering papers for topic: %s", topic)
    discovery = PaperDiscovery()
    papers = discovery.search(topic, max_results=args.max_papers, sources=sources)
    if not papers:
        logger.error("No papers found for topic %r. Try a broader query.", topic)
        sys.exit(1)
    logger.info("Found %d papers", len(papers))

    extractor = TextExtractor(cache_dir=Path(args.data_dir) / "pdf_cache")

    # The Ollama model is only loaded into RAM for the duration of this block: entering the
    # context pulls it in (lazily, on first request) and exiting unloads it immediately rather
    # than leaving it resident for Ollama's default keep-alive window.
    with LLMSynthesizer(model=args.model) as synthesizer:
        logger.info("Summarizing papers...")
        summaries = synthesizer.summarize_all(papers, extractor)
        logger.info("Summarized %d/%d papers", len(summaries), len(papers))
        if not summaries:
            logger.error("Could not summarize any papers; aborting.")
            sys.exit(1)

        logger.info("Synthesizing literature review...")
        synthesis = synthesizer.synthesize_literature(topic, summaries)

        logger.info("Identifying research gaps...")
        research_gaps = synthesizer.generate_research_gaps(topic, synthesis, summaries)

        logger.info("Generating outline...")
        outline_generator = OutlineGenerator(synthesizer)
        outline = outline_generator.generate_outline(topic, synthesis, research_gaps, summaries)

    check_content_quality(outline, synthesis, research_gaps, summaries)

    logger.info("Formatting bibliography (%s)...", args.citation_style)
    citation_formatter = CitationFormatter()
    bibliography = citation_formatter.format_bibliography(papers, style=args.citation_style)

    exporter = Exporter(output_dir=Path(args.output_dir))
    report_md = exporter.build_report_markdown(
        topic=topic,
        outline=outline,
        synthesis=synthesis,
        research_gaps=research_gaps,
        summaries=summaries,
        bibliography=bibliography,
    )

    slug = slugify(topic)
    exported_paths = exporter.export_all(report_md, slug, formats)

    for path in exported_paths:
        logger.info("Exported: %s", path)

    return exported_paths


def main(argv=None) -> None:
    args = parse_args(argv)
    try:
        run(args)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(130)
    except (ValueError, OllamaConnectionError) as exc:
        logger.error("Configuration error: %s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.error("ResearchFlow failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
