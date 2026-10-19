"""ResearchFlow CLI: discover papers, screen them, synthesize findings, and export a literature review.

Run with:  python -m src.main "your research topic"

The review is written about 90% automatically; the critical analysis (contradictions, critiques,
why gaps matter, future directions) is marked in the text for the user to complete.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from .citation_formatter import CitationFormatter
from .critical_gaps import marker_counts, user_input_share
from .export import Exporter
from .flowchart import render_flowchart
from .llm_synthesis import (
    DEFAULT_MODEL,
    LARGE_CONTEXT_CHARS,
    PLACEHOLDER_PATTERN,
    LLMSynthesizer,
    OllamaConnectionError,
    PaperSummary,
    content_issue,
    sources_block,
)
from .outline_generator import REPORT_SECTIONS, Outline, OutlineGenerator
from .paper_discovery import PaperDiscovery
from .report_builder import ReportBuilder, ReviewInputs, flowchart_filename
from .review_methodology import (
    METHODOLOGY_CHOICES,
    ReviewFlow,
    get_methodology,
    protocol_paragraphs,
    resolve_years,
    screen,
)
from .text_extraction import TextExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("researchflow")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="researchflow",
        description="Discover, synthesize, and write up literature on a research topic.",
    )
    parser.add_argument("topic", help="The research topic or question to investigate")
    parser.add_argument(
        "--methodology",
        default="narrative",
        choices=METHODOLOGY_CHOICES,
        help="Review methodology; decides screening criteria and how the protocol is reported (default: narrative)",
    )
    parser.add_argument(
        "--query",
        action="append",
        dest="queries",
        metavar="QUERY",
        help="Custom search query; repeat for several (default: the topic itself)",
    )
    parser.add_argument(
        "--sections",
        default=",".join(REPORT_SECTIONS),
        help=f"Comma-separated report sections to generate, from: {', '.join(REPORT_SECTIONS)} (default: all)",
    )
    parser.add_argument("--year-from", type=int, help="Only include papers published in or after this year")
    parser.add_argument("--year-to", type=int, help="Only include papers published in or before this year")
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
        "--model",
        default=DEFAULT_MODEL,
        help=f"Ollama model to use for synthesis (default: {DEFAULT_MODEL}; tinyllama or phi for low-RAM machines)",
    )
    args = parser.parse_args(argv)
    args.sections = parse_sections(args.sections, parser)
    args.queries = [q.strip() for q in (args.queries or [args.topic]) if q.strip()] or [args.topic]
    return args


def parse_sections(value: str, parser: argparse.ArgumentParser) -> list[str]:
    chosen = [s.strip().lower().replace(" ", "-").replace("_", "-") for s in value.split(",") if s.strip()]
    unknown = [s for s in chosen if s not in REPORT_SECTIONS]
    if unknown or not chosen:
        parser.error(f"--sections must name at least one of: {', '.join(REPORT_SECTIONS)} (got {value!r})")
    return [key for key in REPORT_SECTIONS if key in chosen]


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
    sources = sources_block(summaries, LARGE_CONTEXT_CHARS)  # at least as much context as the model saw

    for section in outline.sections:
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
    checks = [("Literature synthesis", synthesis, 4, f"{sources}\n{synthesis}")]
    if research_gaps:  # only generated when the Discussion section was requested
        checks.append(("Research gaps", research_gaps, 2, None))
    for label, text, min_sentences, source in checks:
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
    methodology = get_methodology(args.methodology)
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    formats = [f.strip().lower() for f in args.formats.split(",") if f.strip()]
    year_from, year_to = resolve_years(methodology, args.year_from, args.year_to)

    logger.info("%s on: %s (queries: %s)", methodology.name, topic, "; ".join(args.queries))
    discovery = PaperDiscovery()
    # Retrieve a larger pool than will be reviewed, so screening and ranking have something to choose from.
    found = discovery.search_queries(args.queries, per_query=min(100, max(20, args.max_papers * 2)), sources=sources)
    eligible, excluded = screen(found.papers, methodology, args.queries, year_from, year_to)
    papers = eligible[: args.max_papers]
    flow = ReviewFlow(
        found_by_source=found.found_by_source,
        duplicates=found.duplicates,
        excluded=excluded,
        not_selected=len(eligible) - len(papers),
    )
    logger.info(
        "Found %d records, %d duplicates, %d excluded by screening, %d eligible; reviewing %d",
        flow.total_found, flow.duplicates, sum(excluded.values()), len(eligible), len(papers),
    )
    if not papers:
        logger.error(
            "No eligible papers for topic %r. Try broader queries, a wider year range or --methodology narrative.", topic
        )
        sys.exit(1)

    extractor = TextExtractor(cache_dir=Path(args.data_dir) / "pdf_cache")

    # The Ollama model is only loaded into RAM for the duration of this block: entering the
    # context pulls it in (lazily, on first request) and exiting unloads it immediately rather
    # than leaving it resident for Ollama's default keep-alive window.
    with LLMSynthesizer(model=args.model) as synthesizer:
        logger.info("Summarizing papers%s...", "" if methodology.full_text else " (abstracts only)")
        summaries = synthesizer.summarize_all(papers, extractor, full_text=methodology.full_text)
        logger.info("Summarized %d/%d papers", len(summaries), len(papers))
        if not summaries:
            logger.error("Could not summarize any papers; aborting.")
            sys.exit(1)
        flow.not_summarized = len(papers) - len(summaries)

        # The synthesis feeds the research gaps and the other sections, so it is always generated.
        logger.info("Synthesizing literature review...")
        synthesis = synthesizer.synthesize_literature(topic, summaries)

        research_gaps, tensions = "", []
        if "discussion" in args.sections:
            logger.info("Identifying research gaps...")
            research_gaps = synthesizer.generate_research_gaps(topic, synthesis, summaries)
            logger.info("Looking for contradictions between papers...")
            tensions = synthesizer.find_tensions(topic, summaries)

        logger.info("Writing sections: %s", ", ".join(args.sections))
        outline_generator = OutlineGenerator(synthesizer)
        outline = outline_generator.generate_outline(
            topic, synthesis, research_gaps, summaries, sections=args.sections, review=methodology.name.lower()
        )

    check_content_quality(outline, synthesis, research_gaps, summaries)

    logger.info("Formatting bibliography (%s)...", args.citation_style)
    citation_formatter = CitationFormatter()
    bibliography = citation_formatter.format_bibliography([s.paper for s in summaries], style=args.citation_style)

    exporter = Exporter(output_dir=Path(args.output_dir))
    slug = slugify(topic)
    flowchart = None
    protocol: list[str] = []
    if "methodology" in args.sections:
        protocol = protocol_paragraphs(methodology, flow, args.queries, sources, year_from, year_to, args.max_papers)
        flowchart = flowchart_filename(slug)
        render_flowchart(flow, exporter.output_dir / flowchart)

    report_md = ReportBuilder().build(ReviewInputs(
        topic=topic,
        methodology=methodology,
        outline=outline,
        synthesis=synthesis,
        research_gaps=research_gaps,
        summaries=summaries,
        bibliography=bibliography,
        sections=args.sections,
        protocol=protocol,
        flow=flow,
        flowchart=flowchart,
        tensions=tensions,
    ))
    counts = marker_counts(report_md)
    logger.info(
        "User input markers: %s (%.0f%% of the text left for your analysis)",
        ", ".join(f"{tag} x{n}" for tag, n in counts.items() if n) or "none",
        100 * user_input_share(report_md),
    )

    exported_paths = exporter.export_all(report_md, slug, formats)
    if flowchart:
        exported_paths.append(exporter.output_dir / flowchart)

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
