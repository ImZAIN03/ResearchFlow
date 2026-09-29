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
from .llm_synthesis import DEFAULT_MODEL, LLMSynthesizer
from .outline_generator import OutlineGenerator
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
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Anthropic model to use for synthesis")
    return parser.parse_args(argv)


def slugify(text: str) -> str:
    slug = "".join(c.lower() if c.isalnum() else "-" for c in text)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-")[:60] or "report"


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
    synthesizer = LLMSynthesizer(model=args.model)

    logger.info("Summarizing papers...")
    summaries = synthesizer.summarize_all(papers, extractor)
    logger.info("Summarized %d/%d papers", len(summaries), len(papers))
    if not summaries:
        logger.error("Could not summarize any papers; aborting.")
        sys.exit(1)

    logger.info("Synthesizing literature review...")
    synthesis = synthesizer.synthesize_literature(topic, summaries)

    logger.info("Identifying research gaps...")
    research_gaps = synthesizer.generate_research_gaps(topic, synthesis)

    logger.info("Generating outline...")
    outline_generator = OutlineGenerator(synthesizer)
    outline = outline_generator.generate_outline(topic, synthesis, research_gaps)

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
    exported_paths: list[Path] = []
    if "md" in formats:
        exported_paths.append(exporter.export_markdown(report_md, f"{slug}.md"))
    if "docx" in formats:
        exported_paths.append(exporter.export_docx(report_md, f"{slug}.docx"))
    if "pdf" in formats:
        exported_paths.append(exporter.export_pdf(report_md, f"{slug}.pdf"))

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
    except ValueError as exc:
        logger.error("Configuration error: %s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.error("ResearchFlow failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
