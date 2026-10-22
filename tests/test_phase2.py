"""Phase 2 on the Phase 1 outputs in output/ (cyber warfare, machine learning). Run with: python -m pytest tests"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from src import critical_gaps as markers
from src.gap_research import load_corpus
from src.phase2 import (
    MethodologyInputs,
    analyse_paper,
    build_methodology,
    compare,
    detect_tradition,
    extract_samples,
    recommend,
    resolve_review,
    run_phase2,
)

OUTPUT = Path(__file__).resolve().parent.parent / "output"
CYBER = OUTPUT / "cyber-warfare.researchflow.json"  # only the Word review was exported for this topic
ML = OUTPUT / "machine-learning.md"
ML_FINAL = OUTPUT / "machine-learning-final.md"


def analyse(path: Path, field=None):
    if not path.exists():
        pytest.skip(f"{path.name} not found")
    review, base = resolve_review(path)
    corpus = load_corpus(review)
    studies = [analyse_paper(source) for source in corpus.papers]
    tradition, _ = detect_tradition(corpus, studies, field)
    return review, base, corpus, studies, tradition, recommend(studies, tradition)


def by_author(studies):
    return {s.author.split()[0]: s for s in studies}


def test_cyber_warfare_designs_match_the_papers():
    _, _, _, studies, _, _ = analyse(CYBER)
    designs = {author: s.design for author, s in by_author(studies).items()}
    assert designs == {"Kumar": "mixed-methods", "Siam": "survey", "Lindsay": "case-study"}

    siam = by_author(studies)["Siam"]
    assert [s["count"] for s in siam.samples] == [300]
    assert "questionnaires" in siam.collection
    assert "inferential" in siam.analysis  # multivariate regression, p < 0.001
    kumar = by_author(studies)["Kumar"]
    assert "interviews" in kumar.collection
    assert {"Estonia", "Georgia", "Ukraine"} <= set(kumar.countries)
    assert "Iran" in by_author(studies)["Lindsay"].countries


def test_cyber_warfare_recommends_mixed_methods_from_the_papers_strands():
    _, _, corpus, studies, tradition, rec = analyse(CYBER)
    # Survey, interviews and case studies are social-science designs, whatever the topic's vocabulary.
    assert tradition.key == "social"
    assert rec.design.key == "mixed-methods" and rec.basis == "mixed"
    assert [d.key for d in rec.components] == ["survey", "case-study"]
    alignments = {s.author.split()[0]: compare(s, rec)[0] for s in studies}
    assert alignments == {
        "Kumar": "Same design",
        "Siam": "Matches the quantitative strand",
        "Lindsay": "Matches the qualitative strand",
    }


def test_machine_learning_recommends_a_computational_experiment():
    _, _, _, studies, tradition, rec = analyse(ML)
    assert {s.design for s in studies} == {"computational-experiment"}
    assert all("datasets" in s.collection for s in studies)
    assert tradition.key == "engineering"
    assert rec.design.key == "computational-experiment" and rec.basis == "common"
    assert all(compare(s, rec)[0] == "Same design" for s in studies)


def test_final_review_uses_the_phase1_paper_file():
    review, base, corpus, studies, _, _ = analyse(ML_FINAL)
    assert base.name == "machine-learning.md"
    assert corpus.from_sidecar and len(studies) == 2


def test_field_override():
    _, _, _, _, tradition, rec = analyse(ML, field="law")
    assert tradition.key == "law"
    assert rec.design.key == "computational-experiment"  # the papers' own design still wins


@pytest.mark.parametrize("path", [CYBER, ML])
def test_methodology_has_every_marker_and_the_comparison_table(path):
    review, base, corpus, studies, tradition, rec = analyse(path)
    text = build_methodology(MethodologyInputs(
        topic=corpus.topic, review_type=review.review_type, slug=base.stem, studies=studies, tradition=tradition,
        field_name=corpus.field, field_overridden=False, recommendation=rec, bibliography="(references)",
    ))
    placed = [m.group(1) for m in map(markers.METHODOLOGY_MARKER_LINE.match, text.splitlines()) if m]
    assert placed == list(markers.METHODOLOGY_MARKERS)
    assert "| Study | Their design | Approach | Proposed design | Alignment | What the proposed design adds |" in text
    for study in studies:
        assert study.narrative in text
    assert "## References" in text


def test_extract_samples():
    samples = extract_samples("We surveyed 1,250 respondents in 2021 and interviewed twelve senior managers (n = 42).")
    assert [(s["count"], s["group"]) for s in samples] == [(1250, "people"), (12, "people"), (42, "unspecified")]
    assert extract_samples("In 2020 participants were recruited; one of the users declined.") == []


def test_run_phase2_writes_the_outputs(tmp_path):
    if not CYBER.exists():
        pytest.skip(f"{CYBER.name} not found")
    args = argparse.Namespace(
        input=str(CYBER), output_dir=str(tmp_path), formats="md,docx", citation_style=None, no_llm=True,
        model="tinyllama", field=None,
    )
    written = {p.name for p in run_phase2(args)}
    assert written == {"cyber-warfare-methodology.md", "cyber-warfare-methodology.docx", "cyber-warfare-methodology-analysis.json"}
    data = json.loads((tmp_path / "cyber-warfare-methodology-analysis.json").read_text(encoding="utf-8"))
    assert data["recommendation"]["design"] == "mixed-methods"
    assert len(data["studies"]) == len(data["comparison"]) == 3
