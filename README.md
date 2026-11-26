# DocuMind

Read a PDF on the left and ask questions on the right. DocuMind can read chart labels and values, cite the page behind an answer, and say when the document has insufficient evidence. Click a page citation to inspect the original, or save an answer to Notes.

[Open DocuMind](https://avish006-documind.hf.space) · [Live dashboard](https://avish006-documind.hf.space/dashboard) · [Hugging Face Space](https://huggingface.co/spaces/Avish006/DocuMind)

[Watch the 53-second narrated demo](https://github.com/avish006/Rag-Project/releases/download/documind-evidence-v1/documind.mp4) · [Video in this repository](demo/documind.mp4) · [Transcript](demo/narration.txt)

The screen recording uses the local release build, a real Google model response and a synthetic PDF. It includes English captions and a synthesized male voice. Model wait times were shortened for playback.

## Try it

Download the [Harbor pilot PDF](evals/fixtures/harbor.pdf), open DocuMind, and enter your [Google AI Studio API key](https://aistudio.google.com/apikey) in Settings. Upload the PDF and ask “In Figure 1, how many crates did the west dock handle?” The chart on page 2 shows 42 crates. Then ask “How many workers unloaded those crates?” The report gives no head count.

The reader supports PDF uploads up to 50 MB, a resizable document pane, dark and light themes, Markdown and math rendering, and browser-local notes with export. The chart icon in the header opens the dashboard in a separate tab.

## Evaluation results

The two original PDFs each contain a native-text page and a chart whose values exist only in the image pixels. The same twelve labeled questions were submitted to Google's `gemini-3.5-flash-lite` in both arms, with a fresh conversation for each question.

| Metric | Before | After |
| --- | ---: | ---: |
| Answer correctness, all questions | 8/12 | 12/12 |
| Correct page citations, answerable questions | 0/8 | 8/8 |
| Insufficient-evidence recall, unanswerable questions | 4/4 | 4/4 |
| Correct answer-or-abstain decision, all questions | 8/12 | 12/12 |
| Context precision, labeled source IDs | 25.0% | 33.3% |
| Context recall, labeled source IDs | 50.0% | 100.0% |
| First relevant evidence, mean reciprocal rank | 0.500 | 0.667 |
| Ragas answer relevancy | 98.0%, n=4 | 95.5%, n=8 |
| Ragas text faithfulness | 100.0%, n=4 | 100.0%, n=4 |

The retrieval rows average over the eight answerable questions. Each needs one source. The guarded system retrieved the required evidence every time, and also supplied two other sources on each question.

Ragas scores apply to delivered answers. The baseline delivered only the four text answers. The guarded system also delivered the four figure answers, so the relevancy averages cover different question subsets. Text faithfulness covers the same four text questions in each arm.

The original pipeline answered the four text questions and declined the four chart questions. Both versions declined the four genuinely unanswerable questions. The comparison measures the **combined** effect of page-aware retrieval, visual evidence and answer checks on a small English regression suite. It does not estimate general accuracy on unfamiliar documents. See the [evaluation protocol](evals/README.md), [case labels](evals/cases.json), [baseline responses](evals/results/baseline.json) and [guarded responses](evals/results/guarded.json).

## How an answer is produced

1. PyMuPDF extracts page-local text windows (900 characters with 150 characters of overlap) and identifies pages with embedded images or vector drawings. Each source keeps a stable ID and a one-based page number.
2. BGE-small-en-v1.5 encodes the sources. Retrieval combines exact normalized-vector ranking with BM25 using reciprocal ranks. It selects up to five text windows and two visual pages. NumPy implements the exact dense search. Index data is saved with the upload.
3. DocuMind sends selected text and rendered page pixels to the model. The last six conversation turns help resolve follow-ups. The model returns claims with source IDs and evidence, or an empty list when evidence is insufficient.
4. Pydantic validates strict claim and verdict schemas. The server checks that source IDs exist and that text excerpts occur in the cited window. A second model call checks each claim against the cited text or image and the question. Every claim must be supported and the answer must address the question. An unknown citation, mismatched quote or negative verdict yields an insufficient-evidence response. Missing verdict fields, wrong types, provider outages and malformed model output produce errors with a trace ID.
5. The reader displays the verified statements and page links. Visual verification uses the same model in a separate call, so correlated mistakes remain possible. Retrieval can miss relevant pages in long documents. Scanned pages are read visually only when retrieved.

## Live operations

The dashboard refreshes every five seconds. It counts completed uploads and questions over a rolling 24 hours. The p95 is the nearest-rank 95th percentile across those requests, including errors. The errors card counts HTTP responses of 400 or above. Insufficient evidence counts successful query responses with that outcome. Dashboard polling, health checks and PDF page downloads do not add traffic to the cards.

Each upload and query returns a unique `trace_id` and `X-Trace-ID`. Click **View trace** in chat or select a recent failure on the dashboard to see the operation, stage, diagnostic, HTTP status, timestamp and duration. Select **Baseline** or **Guarded** to inspect the latest published evaluation question by question.

**Live RAG quality** updates from actual query traffic. Ragas 0.3.7 scores answer relevancy for delivered answers and faithfulness for text answers in the background. Relevancy uses three reverse-generated questions and local BGE embeddings. Faithfulness checks answer statements against retrieved text. Figure answers use the visual verifier for support, and remain eligible for Ragas answer relevancy. Each score shows its measured query count. The dashboard also shows context relevance, draft claim support, responsiveness and cited-source share from the guardrail checks, plus hourly traffic, latency and RAG quality charts built with Chart.js.

Ragas scoring starts after the request completes. It typically adds three Google generation calls for relevancy and two for text faithfulness, with limited transient retries. These calls use the same visitor-supplied Google key and count against its quota. One scoring worker handles up to eight pending/running jobs in memory. At capacity, the answer still returns and scoring is marked skipped busy. The dashboard displays pending, complete, partial and failed jobs, with detailed metric status in the trace. A restart marks unfinished jobs interrupted. User text and keys remain in memory for scoring, and public traces store only numeric scores, statuses and stage timings. See the [formulas and measurement limits](evals/README.md#rag-quality-metrics).

OpenTelemetry records indexing, retrieval, rendering, generation, verification and background scoring spans beneath the request trace. Prometheus can scrape `GET /metrics` for rolling request counts, p95 latency, scored sample counts, mean RAG scores and guardrail outcomes. SQLite keeps the application dashboard available without an external telemetry account. The single-worker Gunicorn configuration keeps the bounded scoring queue and restart recovery consistent.

| Endpoint | Purpose |
| --- | --- |
| `POST /upload` | Multipart field `file`. Returns `document_id`, `pdf_url` and `trace_id`. |
| `POST /query` | JSON with `document_id`, `query`, `api_key` and optional `history`. Returns `response`, `status`, `citations`, `reason` and `trace_id`. |
| `GET /api/metrics` | Live totals, p95 in milliseconds, 25 latest failure summaries and latest completed eval scores. |
| `GET /metrics` | Prometheus exposition of rolling 24-hour gauges. |
| `GET /traces/<id>` | Operational details for a single request. |
| `GET /evaluations/baseline`, `/evaluations/guarded` | Recorded answers, gold labels, scores and evaluation trace IDs. |
| `GET /health` | HTTP liveness check. |

The app stores the visitor's key in that browser's local storage until cleared. The server forwards the key to Google's API for that visitor's requests. Public telemetry excludes keys, questions, answers, filenames, document IDs and PDF contents. The public eval pages contain only the included synthetic fixtures. Uploaded PDFs use random bearer URLs, so anyone holding a document URL can read it during its lifetime. Access expires after 24 hours and subsequent uploads sweep old files. SQLite retains seven days of traces and prunes older entries on writes. A host with ephemeral disk can lose uploads and metrics on replacement. The checked-in evaluation reports survive restarts.

## Run locally

Use Python 3.11. On Linux or macOS, activate the environment with `source .venv/bin/activate`. On Windows, use `.venv\Scripts\Activate.ps1`.

```sh
python -m venv .venv
# Activate the environment, then install dependencies.
python -m pip install -r requirements.txt
python app.py
```

Open [localhost:5000](http://localhost:5000). The first upload downloads the BGE model. Visitors supply their own Google keys. The web server never takes a shared provider key from `.env`.

| Process environment | Default | Purpose |
| --- | --- | --- |
| `DOCUMIND_DATA` | `instance/documind` | Writable document and trace directory |
| `DOCUMIND_MODEL` | `gemini-3.5-flash-lite` | Google model ID for generation and verification |
| `DOCUMIND_EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Hub ID or local encoder directory |
| `GOOGLE_API_KEY` | unset | CLI evaluation key only. The runner can load it from an ignored `.env`. |
| `PORT` | `5000` | Local Flask entry point. The container listens on 7860. |

```sh
python -m unittest discover -s tests -v
python -m evals.run both
python -m evals.run both --score-recorded
```

## Deploy

The repository root is the canonical application for both local and Docker deployments. The image runs Gunicorn as UID 1000 on port 7860 and warms the embedding model during the build.

```sh
docker build -t documind .
docker run --rm -p 7860:7860 -v documind-data:/app/instance documind
```

The existing Space can deploy a specific public GitHub commit through one file. Run `python deploy/space_file.py <full-github-commit-sha>`, then put its output in the Space's root `Dockerfile` and commit it. This fetches the pinned source and applies the repository's Docker build instructions. To roll back, restore the previous Space Dockerfile revision. After a build, open `/health`, upload the Harbor fixture, follow a page citation and open `/dashboard`. See the [Docker Spaces configuration reference](https://huggingface.co/docs/hub/spaces-config-reference).
