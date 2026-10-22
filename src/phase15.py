"""Phase 1.5: fills the gaps Phase 1 left in a literature review with the user's own analysis.

    python -m src.main --phase 1.5 --input "output/my-review.md" --filled-gaps "my-gaps.txt"

The first run (no --filled-gaps, or a --filled-gaps file that doesn't exist yet) researches every gap
and writes two files for the user: <review>-gap-research.md, with the evidence and hints per gap,
and the answers template (<review>-gaps.txt or the --filled-gaps path). The user writes their analysis
into the template and runs the same command again; that run checks each answer for originality and
critical thinking, polishes the accepted ones, and writes the complete review (<review>-complete.md,
.docx, .pdf) with a quality report (<review>-quality-report.md).
"""
from __future__ import annotations

import logging
import re
import shutil
from contextlib import ExitStack
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

from .export import IMAGE_LINE, Exporter
from .gap_input import (
    AnswerCheck,
    build_source_index,
    check_answer,
    parse_answers,
    write_template,
)
from .gap_parser import parse_review
from .gap_research import (
    load_corpus,
    load_research,
    research_brief,
    research_gaps,
    research_path,
    save_research,
)
from .llm_synthesis import LLMSynthesizer, OllamaConnectionError
from .review_assembly import IncompleteReviewError, assemble, quality_report, report_markdown
from .text_polish import Polished, polish

logger = logging.getLogger("researchflow")


def _open_model(args, stack: ExitStack) -> Optional[LLMSynthesizer]:
    """The local model for question hints and grammar edits, or None (both have model-free fallbacks)."""
    if args.no_llm:
        logger.info("Model disabled (--no-llm): hints and polishing use built-in rules only")
        return None
    try:
        return stack.enter_context(LLMSynthesizer(model=args.model))
    except OllamaConnectionError as exc:
        logger.warning("Continuing without the model (%s). Hints and polishing use built-in rules.", exc)
        return None


def _copy_images(markdown: str, source_dir: Path, output_dir: Path) -> None:
    """Copies figures the review links to (e.g. the flowchart) when writing to another directory."""
    if source_dir.resolve() == output_dir.resolve():
        return
    for line in markdown.splitlines():
        image = IMAGE_LINE.match(line.strip())
        if not image or re.match(r"^[a-z]+://", image.group(2)) or Path(image.group(2)).is_absolute():
            continue
        source, target = source_dir / image.group(2), output_dir / image.group(2)
        if source.exists() and not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def run_phase15(args) -> List[Path]:
    load_dotenv()
    input_path = Path(args.input)
    if not input_path.is_file():
        raise ValueError(f"Review file not found: {input_path}")
    review = parse_review(input_path)
    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = input_path.stem
    formats = [f.strip().lower() for f in args.formats.split(",") if f.strip()]

    if not review.gaps:
        logger.info("No gap markers found in %s: the review has nothing left to fill.", input_path)
        return []
    counts: Dict[str, int] = {}
    for gap in review.gaps:
        counts[gap.category] = counts.get(gap.category, 0) + 1
    logger.info(
        "Found %d gaps in %r: %s",
        len(review.gaps), review.title, ", ".join(f"{n} {c}" for c, n in counts.items()),
    )

    corpus = load_corpus(review)
    style = (args.citation_style or corpus.citation_style or "apa").lower()
    written: List[Path] = []

    with ExitStack() as stack:
        model = _open_model(args, stack)

        cache = research_path(output_dir, slug)
        research = load_research(cache, review)
        brief = output_dir / f"{slug}-gap-research.md"
        if research is None:
            sources = [s.strip() for s in args.sources.split(",") if s.strip()] if args.sources else None
            research = research_gaps(review, corpus, search=not args.no_search, sources=sources, synthesizer=model)
            save_research(research, cache)
            brief.write_text(research_brief(review, corpus, research), encoding="utf-8")
            written.append(brief)
        else:
            logger.info("Reusing the gap research in %s", cache.name)
            if not brief.exists():
                brief.write_text(research_brief(review, corpus, research), encoding="utf-8")

        answers_path = Path(args.filled_gaps) if args.filled_gaps else output_dir / f"{slug}-gaps.txt"
        if not answers_path.exists():
            write_template(answers_path, review, research, str(input_path))
            written.append(answers_path)
            logger.info("Wrote the gap research to %s", brief)
            logger.info(
                "Wrote your answers template to %s. Write your analysis under each '>>> ANSWER' line "
                "(about %d minutes for %d gaps), then run: python -m src.main --phase 1.5 --input \"%s\" "
                "--filled-gaps \"%s\"",
                answers_path, 5 * len(review.gaps), len(review.gaps), input_path, answers_path,
            )
            return written
        if not args.filled_gaps:
            logger.info("Using the answers in %s (pass --filled-gaps to use another file)", answers_path)

        answers = parse_answers(answers_path, len(review.gaps))
        if not any(text.strip() for text in answers.values()):
            logger.info(
                "%s has no answers yet. Write your analysis under each '>>> ANSWER' line (hints are in %s), "
                "then run the command again.", answers_path, brief,
            )
            return written

        index = build_source_index(review, corpus, research)
        checks: Dict[int, AnswerCheck] = {}
        earlier = []
        for gap in review.gaps:
            check = check_answer(gap, answers.get(gap.id, ""), index, earlier)
            checks[gap.id] = check
            if check.filled:
                earlier.append((gap.id, check.text))
            if check.accepted:
                logger.info("Gap %d accepted (%d words)%s", gap.id, check.words, f"; {len(check.warnings)} warning(s)" if check.warnings else "")
            for warning in check.warnings:
                logger.warning("Gap %d: %s", gap.id, warning)
            for error in check.errors:
                logger.error("Gap %d: %s", gap.id, error)

        new_papers = research.all_new_papers()
        polished: Dict[int, Polished] = {}
        for gap in review.gaps:
            if checks[gap.id].accepted:
                logger.info("Polishing gap %d...", gap.id)
                polished[gap.id] = polish(checks[gap.id].text, corpus, new_papers, review.body_text, model)

    report_path = output_dir / f"{slug}-quality-report.md"
    rejected = [g.id for g in review.gaps if not checks[g.id].accepted]
    assembly = None
    outputs: List[Path] = []
    if not rejected or args.allow_incomplete:
        assembly = assemble(review, polished, corpus, style, allow_incomplete=args.allow_incomplete)
        _copy_images(assembly.markdown, input_path.parent, output_dir)
        outputs = Exporter(output_dir=output_dir).export_all(assembly.markdown, f"{slug}-complete", formats)
        written += outputs

    report = quality_report(review, checks, polished, assembly)
    report_path.write_text(
        report_markdown(review, report, checks, polished, assembly, [p.name for p in outputs]), encoding="utf-8"
    )
    written.append(report_path)

    if assembly is None:
        logger.error(
            "%d of %d gaps need another pass (%s). See %s for what to fix, edit %s and run again "
            "(or add --allow-incomplete to export with those gaps still marked).",
            len(rejected), len(review.gaps), ", ".join(map(str, rejected)), report_path, answers_path,
        )
        raise IncompleteReviewError(f"Gaps {', '.join(map(str, rejected))} were not accepted")

    if assembly.added_papers:
        logger.info("Added %d newly cited paper(s) to the references", len(assembly.added_papers))
    for path in written:
        logger.info("Written: %s", path)
    logger.info("Quality score: %d/100. %s.", report.score, report.verdict)
    return written
