# Metrics reference

DocuMind evaluates labeled fixtures and measures live upload/query traffic. The dashboard refreshes every five seconds. Every mean includes its measured sample count. Missing or failed measurements remain unavailable.

## RAG evaluations

The fixture suite labels source IDs and reference answers. K is fixed at **3** for the top-K metrics. Rankings use the exact source presentation order supplied to the model, including the text/visual selection policy. Duplicate IDs count once. Binary relevance is 1 when an ID belongs to the labeled gold set.

| Metric | Formula or procedure | Scope and implementation |
| --- | --- | --- |
| Precision@3 | Relevant IDs in the first 3 positions / 3 | Labeled answerable questions. A short result list leaves unfilled positions at zero. |
| Recall@3 | Relevant IDs in the first 3 positions / number of gold IDs | Labeled answerable questions. Measures evidence coverage within the cutoff. |
| MRR | Mean of 1 / rank of the first gold ID. Zero when none is retrieved. | Full selected-source order. |
| NDCG@3 | DCG@3 / ideal DCG@3. DCG sums binary relevance / log2(rank + 1). | Gold IDs define the ideal ranking. The current labels have binary relevance. |
| Faithfulness | Ragas decomposes a delivered text answer into statements and judges support in retrieved text. Supported statements / statements. | Live background scoring and recorded evaluations through Ragas 0.3.7. Visual support is checked against actual page images by the guardrail. |
| Answer relevance | Ragas generates three questions from the answer, then averages their cosine similarity to the user's question using BGE. | Live delivered answers and recorded evaluations. Follow-up questions can be harder to evaluate in isolation. |
| Answer correctness | Ragas factual F1 compares answer claims with the labeled reference answer. A refusal on an answerable case scores zero. | Labeled answerable questions. The original gold-fact coverage pass count is also displayed. |
| Context precision | Rank-aware average precision, sum of precision at each relevant position / number of relevant retrieved IDs. | Full selected-source order. A separate Ragas ID precision score measures the relevant share of all selected sources. |
| Context recall | Gold IDs retrieved / gold IDs | Full selected-source set, computed with Ragas IDBasedContextRecall. This overlaps Recall@3 when the whole set fits the cutoff. |
| Citation accuracy | Checks the expected page set and independently asks whether each actual cited text or page image supports its answer claim. Missing citations score zero on answerable cases. | The dashboard reports strict page-plus-support accuracy and mean citation evidence support. |
| Abstention accuracy | Correct answer-or-abstain decisions / all labeled questions | Includes correct refusals and correct decisions to answer. Insufficient-evidence recall separately measures correct refusals among unanswerable questions. |

Precision, recall and NDCG are implemented in [rag_metrics.py](../rag_metrics.py). The same module aggregates the Ragas ID scores and computes rank-aware average precision and MRR. [ragas_eval.py](../ragas_eval.py) adapts the Google transport and BGE model for Ragas and independently judges citations. [evals/run.py](../evals/run.py) records responses, per-case grades, model usage, errors and the evaluation configuration.

### Interpretation and limits

- Reference-based correctness, retrieval recall, ranking quality and abstention accuracy require gold labels. Live queries supply inputs for reference-free relevancy, text faithfulness, retrieval similarity and guardrail measurements.
- The twelve-question suite has four text, four figure and four unanswerable cases. Each answerable case needs one labeled source. Larger documents, multiple evidence sources and graded relevance require additional fixtures.
- Baseline source IDs represent page-text components of its single legacy chunk. Guarded IDs represent actual text windows and visual pages. The protocol describes this mapping. [Evaluation protocol](../evals/README.md)
- Rank-aware average precision and MRR coincide when there is one relevant source. ID precision and Precision@3 can differ because they use different denominators.
- Generation, verification and Ragas judging use the same model, so errors can be correlated. Independent citation scoring is an additional call, with each claim checked against its own cited source.
- Ragas factual F1 can disagree with gold-fact coverage. In the November 27 run, guarded coverage was 12/12 while factual F1 averaged 58.4% over eight answerable cases. Claim decomposition and short reference wording affect the judge. The stored answers and per-case scores expose this disagreement.
- Legacy prose refusals are identified by an English pattern matcher. Structured guarded responses expose their abstention status directly. Inspect raw responses when reviewing a score.
- Provider failures are recorded as unavailable metrics and are excluded from quality averages. Each displayed metric includes its own denominator.

## Live request metrics

Traffic metrics count completed POST /upload and POST /query requests. Dashboard polling, trace reads, health checks, Prometheus scrapes and PDF downloads are excluded.

| Metric | Definition |
| --- | --- |
| Total recorded requests | Persisted counter. During upgrade it initializes from retained runs, then increments once per new run. Pruning old runs preserves this counter. Replacing the telemetry database starts a new counter. |
| Requests per minute | Completed requests whose start timestamp falls in the trailing 60 seconds. |
| Success rate | HTTP status below 400 / recorded requests in the rolling 24-hour window. |
| Error rate | HTTP status 400 or above / recorded requests in that window. |
| Average latency | Arithmetic mean of server-side request duration. Includes errors. |
| p50, p95, p99 | Nearest-rank percentiles of server-side duration over the rolling 24 hours. |
| Stage latency | Sample count, mean and p95 of stored OpenTelemetry spans by stage. Background Ragas spans are included under their own names. |
| Token consumption | Google usage metadata, aggregated by foreground and background phase. Shows prompt, candidate output, thinking, cached prompt and total tokens. |
| Estimated cost per query | Mean paid Standard list estimate for settled queries with complete price coverage, including generation, verification, retries and background scoring. Pending/interrupted evaluations and calls with unknown prices or missing usage are excluded from this mean. |
| Retrieval similarity | Per-selected-source cosine similarity, raw BM25 score and fused reciprocal-rank score. Traces show presentation rank and source kind. Live summaries show mean and maximum cosine. |
| Guardrail trigger rate | Queries blocked by a source, quote, verification or output-contract check / queries that reached retrieval. |
| Abstention rate | Successful insufficient-evidence responses / successful query requests. |
| Provider time to first token | Time from the first upstream generation request to its first non-thinking text chunk. Reported as mean and p95 with sample count. |

The reader receives the completed answer after evidence checks. Provider TTFT therefore describes upstream generation responsiveness. Request latency covers the guarded server response. Browser rendering and network transit to the user's device are outside these timings.

### Usage and cost accounting

Google SSE usage metadata is cumulative within a call. DocuMind keeps the latest value for each field and counts that call once. Retries have distinct call IDs. Failed streams can leave partial observed usage. An unsuccessful call receives no cost estimate. Missing metadata produces an unavailable value.

The default model's paid Standard rates, verified on November 27, 2026, are $0.30 per million input tokens, $2.50 per million output tokens including thinking, and $0.03 per million cached input tokens. The estimate is

`((prompt - cached + tool_input) × input_rate + cached × cached_rate + (output + thinking) × output_rate) / 1,000,000`.

These are list-price estimates. The visitor's free tier, discounts and actual billing can produce different charges. The app uses no paid search or other grounding tools. Unknown model prices remain unavailable. A complete process-environment override can set DOCUMIND_INPUT_USD_PER_M, DOCUMIND_OUTPUT_USD_PER_M and DOCUMIND_CACHED_USD_PER_M together. [Google pricing](https://aistudio.google.com/docs/pricing#gemini-3.5-flash-lite)

The implementation reads streamGenerateContent SSE events through the existing provider adapter. [Google streaming reference](https://aistudio.google.com/api-reference/generate-content/stream-generate-content)

## Guardrails and telemetry

Pydantic validates claim and verdict schemas. Source IDs must exist, text excerpts must occur in their cited window, every verification vote must support its claim, and the verdict must confirm question coverage. Unknown sources, quote mismatches and negative verification produce insufficient evidence. Malformed model output produces a traceable error.

The trigger-rate numerator includes unknown_source, quote_mismatch, verification_failed, provider_invalid_json, provider_invalid_claims and provider_invalid_verdict. A model's empty-claims abstention is counted in abstention outcomes. Provider outages contribute to error rate.

Public traces persist numeric usage, scores, statuses and timings. Retrieval rows contain rank, kind and numeric scores. API keys, questions, answer text, source IDs and PDF contents remain excluded from live telemetry.

SQLite schema initialization adds the new tables to existing deployments. A single run and its retained children share the trace ID. Provider call inserts are idempotent by call ID. Background calls and scores belong to the originating query trace.

Prometheus /metrics exports rolling request statistics, stage p95 values, provider TTFT, token totals, estimated costs, quality averages, sample counts and guardrail outcomes. The dashboard reads /api/metrics, and each query's trace includes individual retrieval and provider-call rows.
