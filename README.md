# ResearchFlow
Free, open-source research acceleration tool for all education levels and fields

ResearchFlow searches arXiv and Semantic Scholar (or OpenAlex and PubMed), screens the results for the
review methodology you choose, summarizes the selected papers with a local Ollama model, and writes a
literature review in Markdown, Word and PDF. About 90% of the review is generated; the remaining
critical analysis is clearly marked for you to write. Phase 1.5 then researches each of those gaps,
gives you hints, checks and polishes what you write, and assembles the complete review.
Phase 1.6 fills the gaps automatically instead, with model-written insights that must pass
grounding, citation, originality and length checks.

## Setup

```
pip install -r requirements.txt
ollama pull mistral        # default model; use tinyllama or phi on low-RAM machines
ollama serve
```

## Usage

```
python -m src.main "federated learning in healthcare" \
    --methodology systematic \
    --query "federated learning healthcare privacy" \
    --query "federated learning medical imaging" \
    --sections introduction,literature-review,methodology,results,discussion,conclusion \
    --year-from 2019 --max-papers 15 --formats md,docx,pdf
```

| Option | What it does |
|---|---|
| `--methodology` | `systematic`, `scoping`, `narrative` (default), `rapid` or `meta-analysis`. Sets the screening criteria, the reporting guideline the protocol cites (PRISMA 2020, PRISMA-ScR, ...) and the limitations you are asked to critique. Rapid reviews use abstracts only and default to the last 5 years; meta-analyses keep only empirical studies. |
| `--query` | A custom search query. Repeat it for several; defaults to the topic. |
| `--sections` | Which sections to generate: `introduction`, `literature-review`, `methodology`, `results`, `discussion`, `conclusion`. |
| `--year-from`, `--year-to` | Publication window used in screening. |
| `--model` | Ollama model (default `mistral`). |

## What the review contains

- **Methodology**: the search and screening protocol, plus a flowchart (`<topic>-flowchart.png`)
  showing Total Papers Found → Papers Filtered → Papers Selected, with the reasons papers were removed.
- **Results**: a Results & Analysis table (Paper Title, Authors, Year, Key Findings, Research Gaps,
  Methodology Type) and per-study summaries.
- **Discussion**: findings, research gaps and future directions.

## Gaps left for you

Critical analysis is marked in the text (highlighted in the Word export) with a specific prompt:

| Marker | Where it appears |
|---|---|
| **[USER ANALYSIS NEEDED]** | Where papers contradict each other (or, for meta-analyses, where effect sizes must be pooled) |
| **[ADD YOUR CRITIQUE]** | Where the included studies' methods, or the review's own method, have limitations |
| **[YOUR INSIGHT]** | Where a research gap matters for your research |
| **[YOUR OBSERVATION]** | On future directions from the research |

Replace each marker with your own analysis before submitting, by hand, with Phase 1.5, or
automatically with Phase 1.6.

## Phase 1.5: filling the gaps

Phase 1.5 turns the review with its gaps into a complete review in about an hour, most of it your own
writing time. It works for any field and topic: everything it does is based on the review's papers.

**1. Get the research and the answers template** (about 5 minutes):

```
python -m src.main --phase 1.5 --input "output/federated-learning-in-healthcare.md" --filled-gaps "gaps.txt"
```

Because `gaps.txt` doesn't exist yet, this writes it as a template, plus
`output/<review>-gap-research.md`. For every gap, Phase 1.5:

- works out what the gap asks for (methodological, evidence, theoretical or future direction), which
  papers it involves and the paragraph it responds to;
- mines the review's papers for sentences that bear on it, and searches for a few more papers. The
  databases are picked from the research field the review's papers belong to: OpenAlex always, plus
  PubMed for medicine, biology and psychology, arXiv for computing, engineering, physics, economics
  and environmental science, or Semantic Scholar for law, social sciences and the humanities;
- writes 2-3 short hints (100-150 words in total): what the reviewed studies say, how other research
  handled the issue, and the questions that remain open.

**2. Write your analysis** (30-60 minutes) under each `>>> ANSWER` line in `gaps.txt`, about 150-200
words per gap. Lines starting with `#` are the instructions and hints and are ignored. Cite papers as
`(Surname, Year)` or `Surname (Year)`. If you cite one of the further-reading papers, it is added to the
references.

**3. Run the same command again** (about 5 minutes). Each answer is checked:

| Check | Rejects | Warns |
|---|---|---|
| Originality | over 20% of 6-word phrases, or a 15+ word passage, copied from the review, its papers, the new papers or the hints; more than 35% quotation | over 10% copied; a quotation without a citation |
| Critical thinking | no comparison, critique, reasoning or suggestion at all | missing the moves the gap needs (e.g. a contradiction needs comparison and critique, a future direction a suggestion) |
| Length | under half the suggested length | shorter or much longer than suggested |
| Placeholders | marker or placeholder text left in | |

Short quotations in "double quotes" with a citation are fine. Accepted answers are polished: spelling,
spacing and punctuation fixed, contractions and informal wording made academic, citation format
normalized and missing years added (e.g. `Smith argues` becomes `Smith (2020) argues`). If Ollama is
running, the model also fixes grammar, but its edit is only used if it keeps your citations, wording
and length, so it can't add ideas.

When every gap is accepted, you get `<review>-complete.md`, `.docx` and `.pdf` with no markers left and
the references rebuilt in your citation style, plus `<review>-quality-report.md`. The report scores the
review out of 100 (completeness 30, originality 20, critical thinking 25, citations 15, length and
register 10) and gives a verdict: **Ready for submission** (85+), **Nearly ready**, or **Needs
revision**. If an answer is rejected, the report says why. Fix it in `gaps.txt` and run again.

| Option | What it does |
|---|---|
| `--input` | The Phase 1 review (markdown). The `<review>.researchflow.json` file Phase 1 writes next to it gives Phase 1.5 the papers' full metadata; without it, the papers are read back from the review's references. |
| `--filled-gaps` | Your answers file; written as a template if it doesn't exist (default: `<review>-gaps.txt`). Hand-written files work too: start each answer with `GAP 1:`, `GAP 2:`... |
| `--citation-style` | `apa`, `mla`, `chicago` or `ieee` for the final review (default: the style Phase 1 used). |
| `--formats` | Default `md,docx,pdf`. |
| `--sources` | Override the databases searched for each gap (`openalex`, `pubmed`, `arxiv`, `semantic_scholar`). |
| `--no-search` | Only use the review's own papers. |
| `--no-llm` | Don't use Ollama; hints and polishing use built-in rules only (also the automatic fallback when Ollama isn't running). |
| `--allow-incomplete` | Export even if some gaps have no accepted answer; their markers stay in place. |

The searches run only once: their results are cached in `<review>.gap-research.json`, so the second
run checks and assembles straight away. The quality score checks form, not whether your arguments are
right, and originality is checked against the sources above, not the whole web. Have a supervisor or
peer read the review before you submit it.

## Phase 1.6: automated gap filling

Phase 1.6 takes the Phase 1 review and fills every gap marker itself, using the same local Ollama
setup (default model `tinyllama`; pass `--model mistral` or `--model phi` for better writing):

```
ollama pull tinyllama
python -m src.main --phase 1.6 --input "output/federated-learning-in-healthcare.md"
```

For each gap, Phase 1.6 reads the marker's question, where it sits and the paragraph it responds to,
picks the 2-3 papers that bear most on it (the papers the marker names first), and gives the model
their findings, stated limitations and the sentences that speak to the gap. The model writes a
150-200 word paragraph with a focus set by the marker:

| Marker | The insight focuses on |
|---|---|
| **[ADD YOUR CRITIQUE]** | Methodology limitations: samples, data, measures, design, and how a stronger study would address them |
| **[YOUR INSIGHT]** | Why the gap is significant and what it implies for research |
| **[YOUR OBSERVATION]** | Future research directions |
| **[USER ANALYSIS NEEDED]** | A comparison of the papers and which one is more convincing, and why |

Every insight is checked before it goes into the review:

| Check | Passes when |
|---|---|
| Grounding | At least 70% of its key terms occur in the review's papers (or the paragraph and question the gap responds to) |
| Citations | It cites at least 2 of the review's papers as `(Surname, Year)` and no paper outside the review |
| Originality | No sentence is copy-pasted from the papers, the review or the gap prompt (at most 20% of 6-word phrases copied, no 15-word copied run), and it doesn't repeat another gap's insight |
| Length | 150-200 words (longer drafts are trimmed at a sentence boundary first) |
| Critical thinking | It makes the moves its gap needs, e.g. comparison and critique for a contradiction |
| Content | No placeholder or marker text, prompt echoes, garbled or repeated sentences |

A failed insight is regenerated up to 2 times, with the reasons it failed given to the model. If it
still fails, the gap is replaced with **[MANUAL REVIEW NEEDED]** followed by the original task, and
the best draft is kept in the log for you to fix by hand.

The gap markers and the "How to complete this review" note are then removed, the insights inserted,
and the references rebuilt in the citation style. It writes:

- `<review>-final.md` and `<review>-final.docx`: the completed review;
- `<review>-phase16-log.md`: per gap, the papers given to the model and every check of every
  attempt, with the inserted insight (or the best draft);
- `<review>-phase16-log.json`: the same in machine-readable form, including every draft.

| Option | What it does |
|---|---|
| `--input` | The Phase 1 review (markdown). Its `<review>.researchflow.json` file gives the papers' full metadata; without it, the papers are read back from the review's references. |
| `--model` | Ollama model (default `tinyllama`). |
| `--citation-style` | `apa`, `mla`, `chicago` or `ieee` (default: the style Phase 1 used). |
| `--formats` | Default `md,docx`; add `pdf` if you want it. |
| `--output-dir` | Default: the input's directory. |

The checks measure form, not whether an argument is right: small models such as TinyLlama can
misread a paper. Read every insight against its papers before you submit, and disclose the use of
AI assistance as your institution or publisher requires.
