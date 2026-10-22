# ResearchFlow
Free, open-source research acceleration tool for all education levels and fields

ResearchFlow searches arXiv and Semantic Scholar (or OpenAlex and PubMed), screens the results for the
review methodology you choose, summarizes the selected papers with a local Ollama model, and writes a
literature review in Markdown, Word and PDF. About 90% of the review is generated; the remaining
critical analysis is clearly marked for you to write. Phase 1.5 then researches each of those gaps,
gives you hints, checks and polishes what you write, and assembles the complete review.
Phase 1.6 fills the gaps automatically instead, with model-written insights that must pass
grounding, citation, originality and length checks.
Phase 2 then reads the research designs of the review's papers and proposes a Methodology section
for your own study, with the decisions only you can make marked.
Phase 3 sets out what the papers found as a Results section (themes, evidence strength and a results
table) and plans the analysis of your own data, with the places for your results marked.
Phase 4 draws the three together into the Conclusion, Implications and Future Directions that close
the thesis or review.

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

## Phase 2: methodology generation

Phase 2 takes a Phase 1 review and proposes a Methodology section for **your own** study, based on
how the reviewed papers did their research. It needs no model: designs are recognized from the
papers' titles, abstracts and Phase 1 summaries.

```
python -m src.main --phase 2 --input "output/cyber-warfare.md"
```

`--input` can be the Phase 1 review, its `<review>.researchflow.json` paper file (handy if you only
exported Word), or a `-complete`/`-final` review from Phase 1.5/1.6. To run Phase 1 and Phase 2 in
one go:

```
python -m src.main "cyber warfare" --methodology systematic --all-phases
```

`--all-phases` runs every phase: Phase 1, then Phases 2, 3 and 4 on the review it wrote.

**1. Each paper's design.** For every paper Phase 2 identifies:

| Detail | Examples |
|---|---|
| Research design | meta-analysis, systematic review, mixed methods, RCT, quasi-experiment, cohort/case-control, cross-sectional survey, ethnography, interview study, controlled experiment, computational experiment (benchmarks), simulation, doctrinal legal analysis, design and development, case study, historical/archival, literature review, conceptual |
| Approach | quantitative, qualitative, mixed methods, evidence synthesis, design science, conceptual |
| Sample | sizes stated in the text, e.g. "300 cybersecurity practitioners", "n = 42" |
| Data collection | interviews, focus groups, questionnaires, experiments, datasets/benchmarks, records, observation, documents and archives, legislation and case law, simulation, sensors/logs, lab/hardware tests |
| Analysis | inferential or descriptive statistics, meta-analysis, thematic coding, content/discourse analysis, performance metrics, simulation modelling, risk/threat frameworks, comparative, legal interpretation, interpretive/historical |
| Context | countries/regions and the populations studied |

If no design can be recognized and Ollama is running, the local model picks one from the same fixed
list (marked "classified by the model"); `--no-llm` skips this.

**2. The field.** The papers' designs (and, with half the weight, their vocabulary) decide which
methodological tradition the recommendations follow:

| Tradition | Typical designs | Reporting standards cited |
|---|---|---|
| Medicine and health | cohort/case-control, RCT, survey, interviews | CONSORT, STROBE, COREQ/SRQR, PRISMA |
| Social sciences | survey, interviews, mixed methods, case study, ethnography | AAPOR/STROBE, COREQ/SRQR, GRAMMS |
| Engineering, computing, natural sciences | computational experiments, simulation, prototypes | reproducibility checklists, code/data release |
| Law | doctrinal analysis, case analysis, empirical legal studies | OSCOLA/Bluebook, explicit case selection |
| Humanities | historical/archival, case study | transparent corpus and source criticism |

So a cyber-warfare review whose papers use a survey, a case study and a mixed-methods design is
treated as social-science research even though its vocabulary is computing; the ethics section then
adds the computing field's norms too. Use `--field health|social|engineering|law|humanities` to override.

**3. The recommendation.** The proposed design is the one the papers share; mixed methods
(explanatory sequential, built from the papers' own quantitative and qualitative designs) when they
use both kinds; or the tradition's standard empirical design when none of them collected primary
data. The section contains:

1. Research Approach
2. Methodological Patterns in the Reviewed Studies (a table of every paper's design, approach,
   sample, data collection and analysis, plus the most common design, variations, sample ranges and contexts)
3. Proposed Research Design, with alternatives
4. Design Rationale (fit with the studies, the gaps it addresses, fit with the field, feasibility)
5. Population and Sample (the papers' samples and the design's sample-size convention)
6. Data Collection and 7. Data Analysis (the papers' methods, and options for the proposed design)
8. Validity, Reliability and Ethics (field-specific)
9. Timeline and Resources (indicative stages and durations)
10. Comparison with the Reviewed Studies (each paper's design vs the proposed design: alignment and
    what the proposed design adds)

**4. Your decisions.** Grounded suggestions sit around markers you replace with your own choices:

| Marker | What you specify |
|---|---|
| **[YOUR METHODOLOGY CHOICE]** | Your research question and the design you will actually use |
| **[YOUR SAMPLE SIZE]** | Sample size, sampling strategy, inclusion/exclusion criteria, and how you got the number |
| **[YOUR DATA SOURCES]** | Who or what you collect data from, the instruments, and access |
| **[YOUR ANALYSIS PLAN]** | The analysis steps, tests or coding procedure, and software |
| **[YOUR TIMELINE & RESOURCES]** | Your schedule, and the access, software, funding and approvals you need |

It writes, next to the input:

- `<review>-methodology.md` and `<review>-methodology.docx` (markers highlighted in Word);
- `<review>-methodology-analysis.json`: every paper's design breakdown with the sentence each detail
  was read from, the detected field and its scores, the patterns, the recommendation and the comparison.

| Option | What it does |
|---|---|
| `--input` | The Phase 1 review, its `.researchflow.json` file, or a `-complete`/`-final` review. |
| `--field` | `health`, `social`, `engineering`, `law` or `humanities` instead of the detected tradition. |
| `--formats` | Default `md,docx`; add `pdf` if you want it. |
| `--citation-style` | `apa`, `mla`, `chicago` or `ieee` for the references (default: the style Phase 1 used). |
| `--no-llm` | Don't ask the model to classify papers the rules can't. |
| `--output-dir` | Default: the input's directory. |

Designs are recognized from keywords in abstracts, so a design described only in a full text is
missed, and only sample sizes stated in the abstract are found. Check the analysis JSON against the
papers, and treat the proposal as a starting point to discuss with your supervisor.

## Phase 3: results and data analysis

Phase 3 takes the review and Phase 2's methodology and writes a Results section from what the
reviewed papers found, then plans how **your own** data should be analysed so your results can be set
against theirs. Like Phase 2 it needs no model.

```
python -m src.main --phase 3 --input "output/cyber-warfare.md"
```

`--input` can be the Phase 1 review, its `.researchflow.json` paper file, a `-complete`/`-final` review,
or the `-methodology` file Phase 2 wrote. If Phase 2's `<review>-methodology-analysis.json` is next to
it, Phase 3 uses the design and field Phase 2 settled on (including papers the model classified);
otherwise it works them out the same way.

**1. Each paper's results.** From the abstract, the Phase 1 summary and key findings:

| Detail | What is read |
|---|---|
| Result type | quantitative, qualitative, mixed, synthesis of prior studies, or conceptual (from the design and the statistics reported) |
| Sample | size, populations and countries (Phase 2's analysis) |
| Findings | result sentences from the abstract (aims such as "the study will..." are skipped) and Phase 1's key findings |
| Statistics | p-values, R², r, β, odds/risk/hazard ratios, 95% confidence intervals, Cohen's d and other effect sizes, test statistics, means, percentages, and performance metrics (accuracy, F1, AUC, error rates, speed-ups) |
| Limitations | what the paper says it leaves open (its own limitations, not its critique of earlier work), and the gap Phase 1 recorded |
| Confidence | **high**: empirical, with a stated sample or dataset and statistics; **medium**: one of the two, or a systematic synthesis; **low**: neither, or no new data |

**2. Patterns across papers.** Two papers report the same finding when their finding sentences share
at least two content words (a single shared word such as "model" or "incident" is too often a
coincidence). Each shared finding is rated **consistent** (the wording points the same way, e.g.
"improves"), **contradictory** (some findings support it and some qualify it, e.g. "limited", "no
significant effect") or **mixed** (they describe different aspects), with a confidence based on how
many studies report it and how strong they are.

**3. The Results section.**

1. Overview of Findings: result types, which studies report statistics and samples, the shared findings
2. Findings Across the Studies: one subsection per shared finding, with each paper's evidence, the
   number of studies, the strength of evidence and the confidence; then the findings reported by one
   study only
3. Summary of Results: a table of every paper's title, key finding, sample, effect size or statistics
   and confidence, with a row for your study
4. Field-specific results (the field is Phase 2's; override it with `--field`):

   | Field | Section | Highlights |
   |---|---|---|
   | Medicine and health | Effect Estimates and Safety Outcomes | odds/risk/hazard ratios, confidence intervals, p-values, adverse events and clinical outcomes |
   | Social sciences | Descriptive, Correlational and Thematic Findings | means, percentages, correlations, regression results, themes |
   | Engineering and computing | Performance Metrics and Technical Results | accuracy, F1, error rates, speed-ups, comparisons with baselines |
   | Law | Case Outcomes and Doctrinal Findings | holdings, rulings, precedent and doctrinal positions |
   | Humanities | Interpretive Findings | the interpretive claims and the sources they rest on |

   Each says what the abstracts do not report (e.g. no confidence intervals, no safety outcomes).
5. Synthesis and Integration: consensus findings, mixed and contradictory findings, and gaps in the
   evidence (missing samples or effect sizes, studies without new data, few settings, the papers'
   own limitations)
6. Data Analysis Plan, for Phase 2's design:
   - quantitative: a table of statistical tests by research question (t-tests, ANOVA, correlation,
     regression, chi-square, SEM) with non-parametric alternatives; medicine adds risk/odds ratios,
     survival analysis and safety; computational studies get repeated runs, paired tests and the
     Friedman test across datasets instead;
   - qualitative: thematic analysis, grounded theory or IPA for interviews, process tracing and
     cross-case synthesis for case studies, doctrinal interpretation for law, source criticism for
     historical work, with rigour steps (codebook, second coder, audit trail);
   - mixed methods: both, plus how to integrate the strands;
   - sample sizes from the papers and the design's convention, and a power analysis at α = 0.05 and 80%
     power (e.g. 64 per group for a medium effect, d = 0.5; 85 for a correlation of 0.3), including any
     effect size the papers report.

**4. Your data.** Grounded suggestions sit around markers you replace with your own work:

| Marker | What you add |
|---|---|
| **[YOUR DATA ANALYSIS]** | Your result on each finding (statistic, effect size and CI, or theme and quotation), your row in the results table, and how your results relate to the consensus and contradictions |
| **[YOUR ANALYSIS METHOD]** | The analysis you will run (tests or coding procedure, software, significance level), and the effect size and sample size behind your power analysis |

Text in quotation marks is the papers' own wording from their abstracts; findings from the Phase 1
summaries are model paraphrases and are introduced as "According to the Phase 1 summary".

It writes, next to the input:

- `<review>-results.md` and `<review>-results.docx` (markers highlighted in Word);
- `<review>-results-analysis.json`: every paper's results profile (findings, statistics, limitations,
  confidence, and the sentence each was read from), the shared findings and the power analysis.

| Option | What it does |
|---|---|
| `--input` | The Phase 1 review, its `.researchflow.json` file, a `-complete`/`-final` review or the Phase 2 `-methodology` file. |
| `--field` | `health`, `social`, `engineering`, `law` or `humanities` instead of Phase 2's (or the detected) tradition. |
| `--formats` | Default `md,docx`; add `pdf` if you want it. |
| `--citation-style` | `apa`, `mla`, `chicago` or `ieee` for the references (default: the style Phase 1 used). |
| `--no-llm` | Don't ask the model to classify papers the rules can't (only used when there is no Phase 2 analysis). |
| `--output-dir` | Default: the input's directory. |

Results are read from abstracts, so numbers given only in a full text's tables are missed, and the
consistency of a finding is judged from its wording. Check every finding, statistic and judgement
against the papers before you rely on it.

## Phase 4: conclusion and future directions

Phase 4 completes the thesis or review. It draws on the review (its research gaps), Phase 2's design
and field and Phase 3's findings, and writes the Conclusion, Implications and Future Directions. Like
Phases 2 and 3, it doesn't need a model.

```
python -m src.main --phase 4 --input "output/cyber-warfare.md"
```

`--input` can be the Phase 1 review, its `.researchflow.json` paper file, a `-complete`/`-final` review,
or any `-methodology`, `-results` or `-conclusion` file. If Phase 2's analysis file is next to it,
Phase 4 uses the design and field Phase 2 chose. It works out Phase 3's results the same way Phase 3
does. If only the paper file exists (no review markdown), the papers' own stated limitations stand in
for the review's research gaps.

**1. Conclusion.**

| Part | Drawn from |
|---|---|
| What the field knows | Phase 1: the studies, their designs and years, what they agree and disagree on, the overall strength of the evidence (strong, moderate, limited or weak) and the review's gaps |
| How this study builds on it | Phase 2: the proposed design, why it was chosen, how many reviewed studies it can be compared with, and which gaps it addresses |
| What the analysis found | Phase 3: the shared findings and how far they agree, the best-supported finding, and the confidence in each study |
| Contribution | What the proposed study can add given that evidence |
| Limitations | The papers' limitations (grouped by kind), the weakest evidence, what the proposed design can't show, and the review's own method |
| Concluding remarks | One implication each for practice, policy and research |

**2. Implications.** Practical implications (what the field's practitioners can act on, depending on
how well supported each finding is), policy implications, theoretical implications, and field-specific
ones:

| Field | Implications covered |
|---|---|
| Medicine and health | Clinical significance, safety implications, treatment guidelines (with the likely GRADE certainty) |
| Social sciences | Policy recommendations, social impact, behavioural change |
| Engineering and computing | Design standards, performance improvements, scalability |
| Law | Precedent, policy reform, justice system |
| Humanities | Theoretical frameworks, interpretation, cultural understanding |

The field is Phase 2's; override it with `--field`.

**3. Future directions.** These come in four groups: unanswered questions (the review's gaps and the
findings the studies disagree on), methodological improvements (what the evidence lacks and what the
field expects), replication and scaling (single-study findings, and the sample a larger study needs),
and interdisciplinary approaches (including bridging the papers' field and their subject matter, as
when a cyber-warfare review uses social-science designs on a computing subject). Each direction has a
next step, a suggested design, a timeline, and a feasibility rating: high (one researcher), medium (a
collaboration) or low (a funded programme). A summary table lists them all.

**4. Your conclusions.** Markers show where only your finished study can speak:

| Marker | What you add |
|---|---|
| **[YOUR CONTRIBUTION]** | Your main results, and the contribution they make: which findings they confirm or challenge and which gap they close |
| **[YOUR STUDY'S LIMITATIONS]** | The limitations of your study as you carried it out |
| **[YOUR RESEARCH PRIORITIES]** | Which future direction you would pursue first, and why |

**Quality checks.** Each run checks the conclusion and logs the results:

| Check | Passes when |
|---|---|
| Not verbatim | At most 10% of its 6-word phrases occur in the review, the methodology, the results or the papers, and no copied run reaches 15 words (citations are ignored) |
| Citations | Every citation is to a reviewed paper, and every paper is cited |
| Draws on Phases 1-3 | It names the review's gaps, the proposed design and every shared finding |
| Sections | Every conclusion, implication and field-specific section is present |
| Future directions | All four groups are covered, and every direction has a next step, timeline and feasibility |
| Markers | All three markers are placed |
| Length | The Conclusion itself is 350-1600 words |

The methodology and results compared are the `-methodology.md` and `-results.md` files if they exist
(so your edits count). Otherwise they are rebuilt.

It writes, next to the input:

- `<review>-conclusion.md` and `<review>-conclusion.docx` (markers highlighted in Word);
- `<review>-conclusion-analysis.json`: the evidence strength, the research gaps, every future
  direction and the result of every check.

| Option | What it does |
|---|---|
| `--input` | The Phase 1 review, its `.researchflow.json` file, a `-complete`/`-final` review or a Phase 2-4 output. |
| `--field` | `health`, `social`, `engineering`, `law` or `humanities` instead of Phase 2's (or the detected) tradition. |
| `--formats` | Default `md,docx`; add `pdf` if you want it. |
| `--citation-style` | `apa`, `mla`, `chicago` or `ieee` for the references (default: the style Phase 1 used). |
| `--no-llm` | Don't ask the model to classify papers the rules can't (only used when there is no Phase 2 analysis). |
| `--output-dir` | Default: the input's directory. |

The implications and directions follow rules applied to the analysis, so they are only as good as
the abstracts behind it. Timelines and feasibility are indicative. Rewrite the conclusion in your own
voice once your results are in, and check every claim against the papers.

## Tests

The tests (Phases 2 to 4) run on the cyber-warfare and machine-learning outputs in `output/`:
`pip install pytest` and then `python -m pytest tests`.
