"""Draws the review's methodology flowchart (Total Papers Found -> Papers Filtered -> Papers Selected).

A PRISMA-style diagram rendered to PNG with Pillow (already installed as a dependency of fpdf2), so
the same image can be linked from the markdown report and embedded in the docx and pdf exports.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

from .review_methodology import SOURCE_NAMES, ReviewFlow

WIDTH, HEIGHT = 1400, 1060
MAIN_X, MAIN_W, BOX_H = 120, 600, 200
SIDE_X, SIDE_W = 820, 500
TOPS = (60, 430, 800)
INK, ACCENT, SIDE_FILL, MAIN_FILL = "#1f2933", "#2f5d8a", "#f4f6f8", "#e6eef7"
FONT_CANDIDATES = ("DejaVuSans.ttf", "arial.ttf", "Arial.ttf", "LiberationSans-Regular.ttf", "Helvetica.ttc")
BOLD_CANDIDATES = ("DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf", "LiberationSans-Bold.ttf")


def _font(size: int, bold: bool = False):
    for name in (BOLD_CANDIDATES if bold else ()) + FONT_CANDIDATES:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, width: int) -> List[str]:
    lines: List[str] = []
    for paragraph in text.split("\n"):
        line = ""
        for word in paragraph.split():
            candidate = f"{line} {word}".strip()
            if draw.textlength(candidate, font=font) <= width:
                line = candidate
            else:
                lines.append(line)
                line = word
        lines.append(line)
    return lines


def _box(draw, xy: Tuple[int, int, int, int], title: str, body: str, fill: str, title_font, body_font) -> None:
    x0, y0, x1, y1 = xy
    draw.rounded_rectangle(xy, radius=14, fill=fill, outline=ACCENT, width=3)
    rows = [(line, title_font) for line in _wrap(draw, title, title_font, x1 - x0 - 40)]
    rows += [(line, body_font) for line in _wrap(draw, body, body_font, x1 - x0 - 40)]
    heights = [font.size + 10 for _, font in rows]
    y = y0 + ((y1 - y0) - sum(heights)) / 2
    for (line, font), h in zip(rows, heights):
        draw.text(((x0 + x1) / 2, y), line, font=font, fill=INK, anchor="ma")
        y += h


def _arrow(draw, start: Tuple[int, int], end: Tuple[int, int]) -> None:
    draw.line([start, end], fill=ACCENT, width=4)
    (x0, _), (x1, y1) = start, end
    if x0 == x1:  # downward
        draw.polygon([(x1 - 12, y1 - 20), (x1 + 12, y1 - 20), (x1, y1)], fill=ACCENT)
    else:  # rightward
        draw.polygon([(x1 - 20, y1 - 12), (x1 - 20, y1 + 12), (x1, y1)], fill=ACCENT)


def render_flowchart(flow: ReviewFlow, path: Path, ranked_by: Optional[str] = "citation count and recency") -> Path:
    image = Image.new("RGB", (WIDTH, HEIGHT), "white")
    draw = ImageDraw.Draw(image)
    title_font, body_font = _font(34, bold=True), _font(26)

    sources = ", ".join(f"{SOURCE_NAMES.get(s, s)}: {n}" for s, n in flow.found_by_source.items())
    removed = flow.duplicates + sum(flow.excluded.values())
    removal_lines = [f"Duplicates: {flow.duplicates}"] + [f"{reason.capitalize()}: {n}" for reason, n in flow.excluded.most_common()]
    not_selected = [f"Ranked below the cut-off: {flow.not_selected}"]
    if flow.not_summarized:
        not_selected.append(f"Could not be processed: {flow.not_summarized}")

    mains = [
        ("Total Papers Found", f"n = {flow.total_found}\n({sources or 'no sources'})"),
        ("Papers Filtered", f"n = {flow.filtered} eligible after screening"),
        ("Papers Selected", f"n = {flow.selected} included in the review"),
    ]
    sides = [
        (f"Removed (n = {removed})", "\n".join(removal_lines)),
        (f"Not selected (n = {flow.not_selected + flow.not_summarized})",
         "\n".join(not_selected) + (f"\nRanked by {ranked_by}" if ranked_by else "")),
    ]
    for top, (title, body) in zip(TOPS, mains):
        _box(draw, (MAIN_X, top, MAIN_X + MAIN_W, top + BOX_H), title, body, MAIN_FILL, title_font, body_font)
    center = MAIN_X + MAIN_W // 2
    for i, (title, body) in enumerate(sides):
        # Side boxes sit between two main boxes, fed by a branch off the connecting arrow.
        mid = (TOPS[i] + BOX_H + TOPS[i + 1]) // 2
        _arrow(draw, (center, TOPS[i] + BOX_H), (center, TOPS[i + 1]))
        _box(draw, (SIDE_X, mid - 115, SIDE_X + SIDE_W, mid + 115), title, body, SIDE_FILL, _font(28, bold=True), _font(24))
        _arrow(draw, (center, mid), (SIDE_X, mid))

    path = Path(path)
    image.save(path)
    return path
