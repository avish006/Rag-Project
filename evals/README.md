# Evaluation protocol

Each of the two PDFs has six hand-labeled questions, two per category. Gold facts, expected page and answerability are in [cases.json](cases.json). Text facts are on page 1. Chart values are raster-only on page 2. The four unanswerable questions deliberately reuse entities from the source. The PDFs contain synthetic data and can be redistributed.

## Reproduce the comparison

Install the root requirements and put a Google API key in the environment or ignored `.env`. Run `python -m evals.run both` from the repository root. The runner makes real requests to Google and updates each report only after that arm completes. Run `python -m evals.run both --output instance/experiment` for a separate experiment. Use a fresh output directory after changing prompts or implementation. A partial run for the same model, suite and PDFs can resume from its checkpoint. Provider failures are retried for transient conditions and are never graded as correct refusals.

The baseline adapter reproduces the deployed Space's original prompt, text extraction and image markers at revision `470ae7f`. Its two fixture documents each fit entirely in one original text chunk. For these fixtures the chunk selected by the legacy hybrid ranker is therefore determined exactly. This narrow adapter rejects larger inputs. The guarded arm calls the same indexing, retrieval, generation and verification functions as the live reader. The comparison holds the provider model, questions, documents, initial history, generation temperature (0.7) and maximum output tokens (1,600) constant. The verification pass uses temperature zero. The original Space's configured Gemma model returned intermittent service errors, so the published paired comparison uses Flash Lite for both arms.

## Scoring

### RAG quality metrics

Each answerable case labels its required evidence in `reference_context_ids`. For these short fixtures, `P1T1` is the first page's text window, `P2T1` the chart caption, and `P2V` the rendered chart. The baseline supplies a single text chunk. The adapter maps its actual page-text components to these IDs in document order. Its image placeholder provides no chart pixels. The guarded arm records the actual sources in the order supplied to the model. The resulting rank measures evidence presentation order, including the text and figure selection policy.

| Labeled metric | Per-question calculation |
| --- | --- |
| ID-based context precision | Number of distinct retrieved IDs in the gold set divided by distinct retrieved IDs |
| ID-based context recall | Number of distinct gold IDs retrieved divided by gold IDs |
| Reciprocal rank | One divided by the position of the first gold ID, or zero when all gold IDs are missing |

The suite averages each score across the eight questions with gold evidence. Empty retrieval earns zero for those questions. The four intentionally unanswerable questions have no relevant ID and are assessed with abstention metrics. Ragas 0.3.7 computes precision and recall through `IDBasedContextPrecision` and `IDBasedContextRecall`. [Ragas context precision](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/) · [Ragas context recall](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_recall/)

| Live measurement | Source and denominator |
| --- | --- |
| Ragas answer relevancy | Ragas `ResponseRelevancy`, strictness 3. Average cosine similarity of three questions generated from the delivered answer with the user's question, using BGE. |
| Ragas faithfulness | Ragas `Faithfulness`. Supported answer statements divided by answer statements, using retrieved text. Scored for delivered text answers. |
| Retrieved sources | Number selected per query that completes retrieval, including subsequent provider failures |
| Judged context relevance | Generator-identified relevant source IDs divided by all selected sources. Partial evidence can count. A valid empty list scores zero. |
| Draft faithfulness | Verifier-supported claims divided by claims in that draft, including drafts rejected by the guard. The verifier inspects the cited text or pixels. |
| Draft answers the question | Verifier's separate yes/no judgment of whether the draft answers the complete question |
| Cited-source share | Distinct sources cited in the delivered answer divided by all selected sources |

Live means weight each measured query equally. The denominator beside each metric counts eligible queries with a valid measurement. A refusal without claims cannot receive a faithfulness score. Historical traces without instrumentation, incomplete verdicts and failures contribute only to metrics actually measured. No question, generated answer, source ID or model explanation is written to the live metrics table. The dashboard polls every five seconds over a rolling 24-hour window. The two published fixture runs are independent of that window.

Run `python -m evals.run both --score-recorded` after generating the paired answers to evaluate those recorded outputs with Ragas. It preserves completed scores and retries missing measurements. The CLI sends only the synthetic fixtures' questions and answers to Google. It records the Ragas configuration and timestamp. A failed measurement has a status and is excluded from its average.

Ragas evaluates answers independently of the synchronous guardrail decision, using the same Google transport and local BGE encoder. The live background worker scores each delivered answer, and per-query checks cover rejected drafts and abstentions. Figure faithfulness is marked not applicable because Ragas' text faithfulness cannot verify the original chart pixels. Visual support remains a separate guardrail measurement. Correlated errors remain possible because generation, visual verification and Ragas judging use the same model. [Ragas faithfulness](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/)

Ground-truth recall, precision and correctness require labeled evidence and answers. They are displayed for the fixture suite. Live traffic provides the inputs for reference-free relevancy and text faithfulness. A faithful answer can still be incomplete, and a low relevancy score can reflect a vague follow-up question. Use per-trace evidence and measured sample counts when interpreting aggregates.

### Answer and citation metrics

- **Answer correctness** counts an answerable case when the response contains all labeled facts after case, spacing and punctuation normalization, and counts an unanswerable case when it declines. Required facts include the unit for numeric answers. For this run, the raw answers were also inspected for contradictory extra claims. The automated rule alone cannot detect every contradiction or paraphrase.
- **Citation accuracy** counts an answerable case when the set of cited pages equals the gold page set. Missing and extraneous pages fail. Baseline page citations are parsed from prose. Guarded citations come from validated source metadata. This metric measures page location, with runtime verification separately checking the evidence.
- **Insufficient-evidence recall** counts the correctly declined questions among the four unanswerable cases. **Evidence decision accuracy** also penalizes declining an answerable question. The guarded arm reports a structured status. Legacy prose uses an English refusal-pattern matcher. During review the matcher was expanded to recognize “no mention of” and “doesn't specifically mention”; both arms were rescored from the unchanged recorded outputs.
- Every report includes the model, completion time, run ID, fixture and suite hashes, case trace IDs, raw response, predicted abstention, cited pages and case grades. The guarded report also records the verification outcome and supporting evidence.

The checked-in fixtures are the reference inputs. `python -m evals.make_fixtures` rebuilds their content. Fonts and PDF library versions can change the resulting file hashes. This suite covers straightforward English text, labeled charts and near-miss questions. It does not characterize cross-page reasoning, ambiguous chart labels or retrieval quality across a long corpus.
