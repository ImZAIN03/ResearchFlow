"""Phase 4 on the Phase 1 outputs in output/ (cyber warfare, machine learning). Run with: python -m pytest tests"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pytest

from src import critical_gaps as markers
from src.gap_research import load_corpus
from src.phase2 import resolve_review, run_phase2
from src.phase3 import analyse_results
from src.phase4 import (
    DIRECTION_CATEGORIES,
    FIELD_HEADINGS,
    ConclusionInputs,
    build_conclusion,
    copied_passages,
    evidence_strength,
    future_directions,
    quality_checks,
    resolve_input,
    review_gaps,
    run_phase4,
)

OUTPUT = Path(__file__).resolve().parent.parent / "output"
CYBER = OUTPUT / "cyber-warfare.researchflow.json"  # only the Word review was exported for this topic
ML = OUTPUT / "machine-learning.md"


def args_for(path: Path, output_dir: Path, **overrides):
    return argparse.Namespace(**{
        "input": str(path), "output_dir": str(output_dir), "formats": "md,docx", "citation_style": None,
        "no_llm": True, "model": "tinyllama", "field": None, **overrides,
    })


def conclusion_for(path: Path, tmp_path: Path, field=None):
    if not path.exists():
        pytest.skip(f"{path.name} not found")
    review, base = resolve_review(path)
    corpus = load_corpus(review)
    analysis = analyse_results(corpus, corpus.topic, base, tmp_path, field, True, "tinyllama")
    gaps = review_gaps(review, analysis.profiles)
    inputs = ConclusionInputs(
        topic=corpus.topic, review_type=review.review_type, review_key="", slug=base.stem, analysis=analysis,
        gaps=gaps, directions=future_directions(corpus.topic, gaps, analysis, corpus.field), field_name=corpus.field,
        bibliography="(references)",
    )
    return review, corpus, inputs, build_conclusion(inputs)


def test_review_gaps_are_read_from_the_review(tmp_path):
    _, _, inputs, _ = conclusion_for(ML, tmp_path)
    assert [(g.title, g.kind) for g in inputs.gaps] == [
        ("Long-term evidence", "longitudinal"),
        ("Integration across research strands", "integration"),
        ("Generalisability across contexts", "generalisability"),
    ]


def test_without_the_review_the_papers_limitations_are_the_gaps(tmp_path):
    _, _, inputs, text = conclusion_for(CYBER, tmp_path)
    assert {g.source for g in inputs.gaps} == {"Kumar et al. (2024)", "Siam et al. (2025)", "Lindsay (2025)"}
    assert "### 1.4 Contribution" in text and "Issues the studies leave open" in text


@pytest.mark.parametrize("path", [CYBER, ML])
def test_conclusion_sections_and_markers(path, tmp_path):
    _, _, inputs, text = conclusion_for(path, tmp_path)
    for heading in ("## 1. Conclusion", "### 1.1 What the field knows", "### 1.2 How this study builds on it",
                    "### 1.3 What the analysis found", "### 1.4 Contribution", "### 1.5 Limitations",
                    "### 1.6 Concluding remarks", "## 2. Implications", "### 2.1 Practical implications",
                    "### 2.2 Policy implications", "### 2.3 Theoretical implications", "## 3. Future Directions",
                    "| Direction | Type | Suggested design | Timeline | Feasibility |", "## References"):
        assert heading in text
    placed = {m.group(1) for m in map(markers.CONCLUSION_MARKER_LINE.match, text.splitlines()) if m}
    assert placed == set(markers.CONCLUSION_MARKERS)
    assert inputs.analysis.recommendation.design.label.lower() in text.lower()
    for p in inputs.analysis.profiles:
        assert p.study.narrative in text or p.study.cite in text


def test_cyber_warfare_conclusion_follows_the_social_sciences(tmp_path):
    _, _, inputs, text = conclusion_for(CYBER, tmp_path)
    assert inputs.analysis.tradition.key == "social"
    assert "### 2.4 Social and Policy Implications" in text
    for sub in ("Policy recommendations", "Social impact", "Behavioural change"):
        assert f"**{sub}.**" in text
    assert "threat detection and cooperation" in text  # Phase 3's shared finding
    assert evidence_strength(inputs.analysis.profiles, inputs.analysis.themes)[0] == "moderate"
    titles = {d.title for d in inputs.directions}
    # The papers are social-science designs on a computing subject, so the fields should be bridged.
    assert "Bridging social sciences and computer science and IT" in titles
    assert "394 participants per group" in text  # scaling the survey strand: d = 0.2 at 80% power


def test_machine_learning_conclusion_follows_engineering(tmp_path):
    _, _, inputs, text = conclusion_for(ML, tmp_path)
    assert "### 2.4 Engineering Implications" in text
    assert "**Scalability.** Scale is raised by Notton et al. (2026)" in text
    assert "No result is yet reproduced across studies" in text
    assert "power analysis" not in text  # benchmark studies report variation across runs instead
    assert evidence_strength(inputs.analysis.profiles, inputs.analysis.themes)[0] == "limited"


@pytest.mark.parametrize("field", list(FIELD_HEADINGS))
def test_every_field_has_its_own_implications(field, tmp_path):
    _, _, inputs, text = conclusion_for(ML, tmp_path, field)
    heading, subs = FIELD_HEADINGS[field]
    assert f"### 2.4 {heading}" in text
    assert all(f"**{sub}.**" in text for sub in subs)


@pytest.mark.parametrize("path", [CYBER, ML])
def test_future_directions_have_timelines_and_feasibility(path, tmp_path):
    _, _, inputs, _ = conclusion_for(path, tmp_path)
    assert {d.category for d in inputs.directions} == set(DIRECTION_CATEGORIES)
    for d in inputs.directions:
        assert d.next_step and d.timeline and d.feasibility in ("High", "Medium", "Low") and d.feasibility_reason


def test_copied_passages():
    source = "the quick brown fox jumps over the lazy dog near the quiet river bank today"
    share, run, origin = copied_passages("we saw the quick brown fox jumps over the lazy dog", {"src": source})
    assert run == 9 and origin == "src" and share > 0.5
    assert copied_passages("an entirely different sentence about nothing in particular", {"src": source})[:2] == (0.0, 0)
    # Citations are attribution, not copying.
    assert copied_passages("as reported (Siam et al., 2025; Kumar et al., 2024) here", {"src": "(Siam et al., 2025; Kumar et al., 2024)"})[1] == 0


def test_quality_checks_catch_copying_and_stray_citations(tmp_path):
    review, corpus, inputs, text = conclusion_for(ML, tmp_path)
    copied = "\n".join(review.lines[8:12])  # a stretch of the literature review
    bad = text.replace("### 1.2 How this study builds on it\n", f"### 1.2 How this study builds on it\n\n{copied} (Smith, 2019).\n")
    checks = {c.name: c for c in quality_checks(bad, inputs, {"the review": "\n".join(review.lines)}, corpus)}
    assert not checks["Not verbatim"].passed
    assert not checks["Citations"].passed and "Smith (2019)" in checks["Citations"].detail


@pytest.mark.parametrize("path", [CYBER, ML])
def test_run_phase4_passes_its_quality_checks(path, tmp_path):
    if not path.exists():
        pytest.skip(f"{path.name} not found")
    written = {p.name for p in run_phase4(args_for(path, tmp_path))}
    slug = path.name.split(".")[0]
    assert written == {f"{slug}-conclusion.md", f"{slug}-conclusion.docx", f"{slug}-conclusion-analysis.json",
                       f"{slug}-executive-summary.md", f"{slug}-executive-summary.docx"}
    data = json.loads((tmp_path / f"{slug}-conclusion-analysis.json").read_text(encoding="utf-8"))
    failed = [c for c in data["quality_checks"] if not c["passed"]]
    assert not failed, failed
    assert data["future_directions"] and data["evidence"]["strength"]


def test_phase4_uses_phase2_design(tmp_path):
    if not CYBER.exists():
        pytest.skip(f"{CYBER.name} not found")
    run_phase2(args_for(CYBER, tmp_path))
    analysis = tmp_path / "cyber-warfare-methodology-analysis.json"
    data = json.loads(analysis.read_text(encoding="utf-8"))
    data["recommendation"].update(design="interview", basis="common", strands=[])
    analysis.write_text(json.dumps(data), encoding="utf-8")
    run_phase4(args_for(CYBER, tmp_path, formats="md"))
    text = (tmp_path / "cyber-warfare-conclusion.md").read_text(encoding="utf-8")
    assert "uses a qualitative interview study" in text
    assert "captures the views of those interviewed" in text  # the design's own limitation


def test_derived_files_as_input(tmp_path):
    if not ML.exists():
        pytest.skip(f"{ML.name} not found")
    for name in ("machine-learning.md", "machine-learning.researchflow.json"):
        shutil.copy(OUTPUT / name, tmp_path / name)
    for suffix in ("-methodology", "-results", "-conclusion"):
        derived = tmp_path / f"machine-learning{suffix}.md"
        derived.write_text("# Derived", encoding="utf-8")
        assert resolve_input(derived) == tmp_path / "machine-learning.md"
