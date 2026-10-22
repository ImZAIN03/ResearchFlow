"""Phase 3 on the Phase 1 outputs in output/ (cyber warfare, machine learning). Run with: python -m pytest tests"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pytest

from src import critical_gaps as markers
from src.gap_research import load_corpus
from src.phase2 import analyse_paper, detect_tradition, recommend, resolve_review, run_phase2
from src.phase3 import (
    ResultsInputs,
    direction,
    extract_statistics,
    find_themes,
    n_correlation,
    n_two_groups,
    n_two_proportions,
    profile_paper,
    resolve_input,
    run_phase3,
)

OUTPUT = Path(__file__).resolve().parent.parent / "output"
CYBER = OUTPUT / "cyber-warfare.researchflow.json"  # only the Word review was exported for this topic
ML = OUTPUT / "machine-learning.md"


def profiles_for(path: Path):
    if not path.exists():
        pytest.skip(f"{path.name} not found")
    review, base = resolve_review(path)
    corpus = load_corpus(review)
    studies = [analyse_paper(source) for source in corpus.papers]
    tradition, _ = detect_tradition(corpus, studies)
    profiles = [profile_paper(source, study, tradition) for source, study in zip(corpus.papers, studies)]
    return review, base, corpus, tradition, recommend(studies, tradition), profiles


def by_author(profiles):
    return {p.study.author.split()[0]: p for p in profiles}


def args_for(path: Path, output_dir: Path, **overrides):
    return argparse.Namespace(**{
        "input": str(path), "output_dir": str(output_dir), "formats": "md,docx", "citation_style": None,
        "no_llm": True, "model": "tinyllama", "field": None, **overrides,
    })


def test_extract_statistics():
    stats = extract_statistics([
        "The regression explained variance (R² = 0.76, p < 0.001) and r = .32 between scales.",
        "Mortality fell (OR 0.62, 95% CI 0.48-0.80); d = 0.45 and 38% of respondents agreed.",
        "The model reached an accuracy of 94.2% and an F1-score of 0.91, 3x faster than the baseline.",
    ])
    found = {(s.kind, s.text) for s in stats}
    assert {("r-squared", "R² = 0.76"), ("p-value", "p < 0.001"), ("correlation", "r = .32"), ("odds-ratio", "OR 0.62"),
            ("confidence-interval", "95% CI 0.48-0.80"), ("effect-size", "d = 0.45"), ("percentage", "38%")} <= found
    assert ("percentage", "95%") not in found  # part of the confidence interval
    assert {s.kind for s in stats if s.kind == "performance"} and any("3x faster" in s.text for s in stats)


def test_direction():
    assert direction("The intervention significantly improved outcomes.") == "supports"
    assert direction("There was no significant difference between the groups.") == "qualifies"
    assert direction("Participants were recruited from three hospitals.") == "neutral"


def test_cyber_warfare_profiles():
    _, _, _, tradition, _, profiles = profiles_for(CYBER)
    assert tradition.key == "social"
    siam, kumar, lindsay = (by_author(profiles)[a] for a in ("Siam", "Kumar", "Lindsay"))
    assert {s.text for s in siam.statistics} >= {"R² = 0.76", "p < 0.001"}
    assert siam.result_type == "Quantitative" and siam.confidence == "High"
    assert kumar.result_type == "Mixed" and lindsay.result_type == "Qualitative"
    assert lindsay.confidence == "Low" and not lindsay.statistics
    # Aims ("The study will determine ...") are not findings, and only abstract sentences are quoted.
    assert not any("will determine" in f.text for f in siam.findings)
    assert siam.findings[0].origin == "abstract" and "positive relationship" in " ".join(f.text for f in siam.findings)


def test_cyber_warfare_shares_one_finding():
    _, _, corpus, _, _, profiles = profiles_for(CYBER)
    themes = find_themes(profiles, corpus.topic)
    assert [(t.term, [p.study.author.split()[0] for p in t.profiles]) for t in themes] == [
        ("threat detection and cooperation", ["Kumar", "Siam"])
    ]
    assert themes[0].consistency == "consistent" and themes[0].confidence == "Medium"


def test_machine_learning_has_no_shared_finding():
    # The two papers share words like "model" and "learning", but no finding.
    _, _, corpus, tradition, _, profiles = profiles_for(ML)
    assert tradition.key == "engineering"
    assert find_themes(profiles, corpus.topic) == []
    assert all(p.result_type == "Quantitative" and p.confidence == "Medium" for p in profiles)


def test_power_analysis_matches_standard_tables():
    # G*Power: d = 0.5 -> 64 per group, d = 0.2 -> 394, d = 0.8 -> 26; r = 0.3 -> 85 (approximations).
    assert [n_two_groups(d) for d in (0.2, 0.5, 0.8)] == [394, 64, 26]
    assert n_correlation(0.3) == 85
    assert n_two_proportions(0.2, 0.1) == 199


@pytest.mark.parametrize("path", [CYBER, ML])
def test_results_section(path):
    from src.phase3 import build_results

    review, base, corpus, tradition, rec, profiles = profiles_for(path)
    themes = find_themes(profiles, corpus.topic)
    text = build_results(ResultsInputs(
        topic=corpus.topic, review_type=review.review_type, slug=base.stem, profiles=profiles, themes=themes,
        tradition=tradition, recommendation=rec, from_phase2=False, bibliography="(references)",
    ))
    placed = {m.group(1) for m in map(markers.RESULTS_MARKER_LINE.match, text.splitlines()) if m}
    assert placed == set(markers.RESULTS_MARKERS)
    assert "| Study | Paper title | Key finding | Sample | Effect size / statistics | Confidence |" in text
    assert f"| Your study | {markers.YOUR_DATA_ANALYSIS} |" in text
    for heading in ("## 1. Overview of Findings", "## 3. Summary of Results", "## 5. Synthesis and Integration",
                    "### Consensus findings", "### Gaps in the evidence", "## 6. Data Analysis Plan", "## References"):
        assert heading in text
    for p in profiles:
        assert p.study.narrative in text
    if tradition.key == "social":
        assert "## 4. Descriptive, Correlational and Thematic Findings" in text
        assert "| Correlation | medium, r = 0.3 | 85 in total |" in text  # the survey strand gets a power analysis
        assert "**Qualitative analysis (case study).**" in text
    else:
        assert "## 4. Performance Metrics and Technical Results" in text
        assert "random seeds" in text and "| Correlation |" not in text


def test_run_phase3_uses_phase2_analysis(tmp_path):
    if not CYBER.exists():
        pytest.skip(f"{CYBER.name} not found")
    run_phase2(args_for(CYBER, tmp_path))
    analysis = tmp_path / "cyber-warfare-methodology-analysis.json"
    data = json.loads(analysis.read_text(encoding="utf-8"))
    data["recommendation"].update(design="interview", basis="common", strands=[])  # as if Phase 2 had chosen interviews
    analysis.write_text(json.dumps(data), encoding="utf-8")

    written = {p.name for p in run_phase3(args_for(CYBER, tmp_path))}
    assert written == {"cyber-warfare-results.md", "cyber-warfare-results.docx", "cyber-warfare-results-analysis.json"}
    text = (tmp_path / "cyber-warfare-results.md").read_text(encoding="utf-8")
    assert "follows Phase 2's recommended design, a qualitative interview study" in text
    assert "Reflexive thematic analysis" in text and "saturation" in text
    results = json.loads((tmp_path / "cyber-warfare-results-analysis.json").read_text(encoding="utf-8"))
    assert results["design"] == {"key": "interview", "label": "Qualitative interview study", "approach": "Qualitative",
                                 "strands": [], "from_phase2": True}
    assert len(results["studies"]) == 3 and results["power_analysis"] == []


def test_methodology_file_as_input(tmp_path):
    if not ML.exists():
        pytest.skip(f"{ML.name} not found")
    for name in ("machine-learning.md", "machine-learning.researchflow.json"):
        shutil.copy(OUTPUT / name, tmp_path / name)
    methodology = tmp_path / "machine-learning-methodology.md"
    methodology.write_text("# Methodology", encoding="utf-8")
    assert resolve_input(methodology) == tmp_path / "machine-learning.md"
    written = {p.name for p in run_phase3(args_for(methodology, tmp_path, formats="md"))}
    assert written == {"machine-learning-results.md", "machine-learning-results-analysis.json"}
