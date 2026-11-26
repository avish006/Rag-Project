"""Metric definitions used by the dashboard, individual traces and labeled evals."""
from statistics import mean


LIVE = {
    "ragas_answer_relevancy": ("Ragas answer relevancy", "Reverse-question cosine similarity using three generated questions and BGE embeddings. Scored in the background on delivered answers."),
    "ragas_faithfulness": ("Ragas faithfulness", "Fraction of answer statements supported by retrieved text. Eligible for delivered text answers."),
    "retrieved_count": ("Retrieved sources", "Number of text windows and visual pages sent as context."),
    "context_relevance": ("Judged context relevance", "Share of retrieved sources the generator judged useful to the question, including partial evidence."),
    "draft_faithfulness": ("Draft faithfulness", "Share of draft claims the verifier found supported by their cited evidence. Includes rejected drafts."),
    "draft_responsiveness": ("Draft answers the question", "Verifier judged that the draft addressed the whole question. Average of yes/no judgments."),
    "source_utilization": ("Cited-source share", "Distinct sources cited in a delivered answer divided by sources retrieved."),
}
OFFLINE = {
    "context_precision": "Context precision (source IDs)",
    "context_recall": "Context recall (source IDs)",
    "reciprocal_rank": "First relevant source (MRR)",
}
EVAL_LABELS = {**OFFLINE, **{name: LIVE[name][0] for name in ("ragas_answer_relevancy", "ragas_faithfulness")}}


def retrieval_scores(retrieved, reference):
    """Macro-average these case scores over questions with labeled evidence."""
    if not reference or retrieved is None:
        return {name: None for name in OFFLINE}
    ranked = list(dict.fromkeys(retrieved))
    gold = set(reference)
    import asyncio
    from ragas_eval import id_scores
    scores = asyncio.run(id_scores(ranked, list(gold)))
    return {
        **scores,
        "reciprocal_rank": next((1 / rank for rank, sid in enumerate(ranked, 1) if sid in gold), 0.0),
    }


def summarize_metrics(rows, names):
    result = {}
    for name in names:
        values = [r[name] for r in rows if r.get(name) is not None]
        result[name] = {"mean": mean(values) if values else None, "n": len(values)}
    return result
