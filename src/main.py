"""ResearchFlow CLI: discover papers, screen them, synthesize findings, and export a literature review.

Run with:  python -m src.main "your research topic"

The review is written about 90% automatically; the critical analysis (contradictions, critiques,
why gaps matter, future directions) is marked in the text for the user to complete. Phase 1.5
(src/phase15.py) then helps the user fill those gaps and assembles the complete review:

    python -m src.main --phase 1.5 --input "output/<topic>.md" --filled-gaps "gaps.txt"

Phase 1.6 (src/phase16.py) instead fills every gap automatically with a validated, model-written
insight and writes the final review:

    python -m src.main --phase 1.6 --input "output/<topic>.md"

Phase 2 (src/phase2.py) analyses the research designs of the review's papers and proposes a
Methodology section for the user's own study, with the decisions only they can make marked:

    python -m src.main --phase 2 --input "output/<topic>.md"

Phase 3 (src/phase3.py) sets out what the review's papers found as a Results section and plans the
analysis of the user's own data, with the places for their results and analysis choices marked:

    python -m src.main --phase 3 --input "output/<topic>.md"

Phase 4 (src/phase4.py) draws Phases 1-3 together into the Conclusion, Implications and Future
Directions, with the places only the user's finished study can fill marked:

    python -m src.main --phase 4 --input "output/<topic>.md"

--all-phases runs every phase in sequence on a topic: Phase 1, then Phases 2, 3 and 4 on its review.
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
from .gap_input import AnswerFileError
from .gap_research import sidecar_path, write_sidecar
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
from .paper_discovery import SOURCES, PaperDiscovery
from .phase15 import run_phase15
from .phase16 import PHASE16_MODEL, run_phase16
from .phase2 import FIELD_CHOICES, PHASE2_FORMATS, run_phase2
from .phase3 import PHASE3_FORMATS, run_phase3
from .phase4 import PHASE4_FORMATS, run_phase4
from .report_builder import ReportBuilder, ReviewInputs, flowchart_filename
from .review_assembly import IncompleteReviewError
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
    parser.add_argument("topic", nargs="?", help="The research topic or question to investigate (Phase 1)")
    parser.add_argument(
        "--phase",
        default="1",
        choices=("1", "1.5", "1.6", "2", "3", "4"),
        help="1: discover papers and write the review (default). 1.5: fill the review's gaps with your own "
        "analysis. 1.6: fill them automatically with validated, model-written insights. 2: propose a "
        "methodology for your own study from the review's papers. 3: write a Results section from the papers' "
        "findings and plan your own data analysis. 4: write the Conclusion, Implications and Future Directions "
        "from Phases 1-3",
    )
    parser.add_argument(
        "--all-phases",
        action="store_true",
        help="Run every phase on the topic in sequence: Phase 1, then Phases 2, 3 and 4 on its review",
    )
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
        help=f"Comma-separated list of sources to search ({', '.join(SOURCES)}). Phase 1 default: "
        "arxiv,semantic_scholar; Phase 1.5 picks them from the review's research field",
    )
    parser.add_argument(
        "--citation-style",
        choices=CitationFormatter.SUPPORTED_STYLES,
        help="Citation style for the bibliography (default: apa; Phase 1.5 keeps the style Phase 1 used)",
    )
    parser.add_argument(
        "--formats",
        help="Comma-separated export formats: md, docx, pdf (default: md; Phase 1.5: md,docx,pdf; Phases 1.6 "
        "and 2 to 4: md,docx)",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory to write exported files to (default: output; Phases 1.5 to 4: the input's directory)",
    )
    parser.add_argument("--data-dir", default="data", help="Directory for cached downloads (e.g. PDFs)")
    parser.add_argument(
        "--model",
        help=f"Ollama model to use for synthesis (default: {DEFAULT_MODEL}; Phase 1.6: {PHASE16_MODEL}; "
        "tinyllama or phi for low-RAM machines)",
    )
    phase15 = parser.add_argument_group("Phases 1.5 to 4 (working from a Phase 1 review)")
    phase15.add_argument(
        "--input",
        help="The Phase 1 review (markdown) whose gaps to fill; for Phases 2 to 4 also its .researchflow.json "
        "file or a -complete/-final review (and for Phases 3 and 4 the files the earlier phases wrote)",
    )
    phase15.add_argument(
        "--filled-gaps",
        help="Your answers file. If it doesn't exist yet, a template with hints for every gap is written there",
    )
    phase15.add_argument("--no-search", action="store_true", help="Don't search for additional papers on each gap")
    phase15.add_argument(
        "--no-llm",
        action="store_true",
        help="Don't use the local model (Phase 1.5: hints and grammar polishing; Phases 2 to 4: classifying "
        "papers whose design the rules can't recognize)",
    )
    phase15.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Export the review even if some gaps have no accepted answer (their markers stay in place)",
    )
    phase2 = parser.add_argument_group("Phases 2 to 4 (methodology, results and conclusion)")
    phase2.add_argument(
        "--field",
        choices=FIELD_CHOICES,
        help="Methodological tradition to follow instead of the one detected from the papers (Phase 3: also how "
        f"results are reported; Phase 4: which implications are drawn): {', '.join(FIELD_CHOICES)}",
    )
    args = parser.parse_args(argv)
    unknown = [s.strip() for s in (args.sources or "").split(",") if s.strip() and s.strip() not in SOURCES]
    if unknown:
        parser.error(f"--sources must be from: {', '.join(SOURCES)} (got {', '.join(unknown)})")
    if args.all_phases and args.phase != "1":
        parser.error("--all-phases starts from Phase 1 and can't be combined with --phase")
    explicit_formats = args.formats
    if args.phase in ("2", "3", "4"):
        if not args.input:
            parser.error(f"--phase {args.phase} needs --input: the Phase 1 review (markdown) or its .researchflow.json file")
        args.formats = args.formats or {"2": PHASE2_FORMATS, "3": PHASE3_FORMATS, "4": PHASE4_FORMATS}[args.phase]
        args.model = args.model or DEFAULT_MODEL
        return args
    if args.phase == "1.6":
        if not args.input:
            parser.error("--phase 1.6 needs --input: the Phase 1 review (markdown) whose gaps to fill")
        args.formats = args.formats or "md,docx"
        args.model = args.model or PHASE16_MODEL
        return args
    args.model = args.model or DEFAULT_MODEL
    if args.phase == "1.5":
        if not args.input:
            parser.error("--phase 1.5 needs --input: the Phase 1 review (markdown) to complete")
        args.formats = args.formats or "md,docx,pdf"
        return args
    if not args.topic:
        parser.error("a research topic is required (or use --phase 1.5/1.6/2/3/4 --input REVIEW.md)")
    args.sources = args.sources or "arxiv,semantic_scholar"
    args.citation_style = args.citation_style or "apa"
    args.formats = args.formats or "md"
    args.output_dir = args.output_dir or "output"
    args.sections = parse_sections(args.sections, parser)
    args.queries = [q.strip() for q in (args.queries or [args.topic]) if q.strip()] or [args.topic]
    args.phase2_formats = explicit_formats or PHASE2_FORMATS  # used by --all-phases
    args.phase3_formats = explicit_formats or PHASE3_FORMATS
    args.phase4_formats = explicit_formats or PHASE4_FORMATS
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
    # The papers behind the review, so Phase 1.5 can research its gaps.
    review_md = exporter.output_dir / f"{slug}.md"
    exported_paths.append(
        write_sidecar(sidecar_path(review_md), topic, methodology.key, args.citation_style, summaries)
    )

    for path in exported_paths:
        logger.info("Exported: %s", path)
    if any(counts.values()) and review_md.exists():
        logger.info(
            "Next, fill the marked gaps: python -m src.main --phase 1.5 --input \"%s\" (or automatically: "
            "--phase 1.6)", review_md
        )

    return exported_paths


def run_all_phases(args: argparse.Namespace) -> list[Path]:
    """Phase 1 on the topic, then Phases 2, 3 and 4 on the review it wrote."""
    written = run(args)
    review_md = Path(args.output_dir) / f"{slugify(args.topic)}.md"
    # Without md in --formats the review may not exist as markdown, but its paper file always does.
    review_input = str(review_md if review_md.exists() else sidecar_path(review_md))
    logger.info("Phase 1 complete. Starting Phase 2 (methodology) on %s", review_input)
    written += run_phase2(argparse.Namespace(**{
        **vars(args), "phase": "2", "input": review_input, "formats": args.phase2_formats,
    }))
    # Phase 3 reads the design Phase 2 just recommended from its analysis file.
    logger.info("Phase 2 complete. Starting Phase 3 (results) on %s", review_input)
    written += run_phase3(argparse.Namespace(**{
        **vars(args), "phase": "3", "input": review_input, "formats": args.phase3_formats,
    }))
    logger.info("Phase 3 complete. Starting Phase 4 (conclusion) on %s", review_input)
    written += run_phase4(argparse.Namespace(**{
        **vars(args), "phase": "4", "input": review_input, "formats": args.phase4_formats,
    }))
    return written


def main(argv=None) -> None:
    args = parse_args(argv)
    try:
        if args.phase == "1.5":
            run_phase15(args)
        elif args.phase == "1.6":
            run_phase16(args)
        elif args.phase == "2":
            run_phase2(args)
        elif args.phase == "3":
            run_phase3(args)
        elif args.phase == "4":
            run_phase4(args)
        elif args.all_phases:
            run_all_phases(args)
        else:
            run(args)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(130)
    except IncompleteReviewError:
        sys.exit(1)  # the reasons were already logged and written to the quality report
    except AnswerFileError as exc:
        logger.error("Could not read your answers: %s", exc)
        sys.exit(1)
    except (ValueError, OllamaConnectionError) as exc:
        logger.error("Configuration error: %s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.error("ResearchFlow failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
