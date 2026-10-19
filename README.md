# ResearchFlow
Free, open-source research acceleration tool for all education levels and fields

ResearchFlow searches arXiv and Semantic Scholar, screens the results for the review methodology you
choose, summarizes the selected papers with a local Ollama model, and writes a literature review in
Markdown, Word and PDF. About 90% of the review is generated; the remaining critical analysis is
clearly marked for you to write.

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

Replace each marker with your own analysis before submitting.
