"""Metric definitions used by the dashboard, individual traces and labeled evals."""
from statistics import mean
import math

RETRIEVAL_K = 3
GUARD_REASONS = {"unknown_source", "quote_mismatch", "verification_failed",
                 "provider_invalid_json", "provider_invalid_claims", "provider_invalid_verdict"}


LIVE = {
    "ragas_answer_relevancy": ("Ragas answer relevancy", "Reverse-question cosine similarity using three generated questions and BGE embeddings. Scored in the background on delivered answers."),
    "ragas_faithfulness": ("Ragas faithfulness", "Fraction of answer statements supported by retrieved text. Eligible for delivered text answers."),
    "retrieved_count": ("Retrieved sources", "Number of text windows and visual pages sent as context."),
    "retrieval_cosine_mean": ("Mean retrieval cosine", "Cosine similarity across sources selected for the model, before rank fusion."),
    "retrieval_cosine_max": ("Maximum retrieval cosine", "Highest cosine similarity among selected sources."),
    "context_relevance": ("Judged context relevance", "Share of retrieved sources the generator judged useful to the question, including partial evidence."),
    "draft_faithfulness": ("Draft faithfulness", "Share of draft claims the verifier found supported by their cited evidence. Includes rejected drafts."),
    "draft_responsiveness": ("Draft answers the question", "Verifier judged that the draft addressed the whole question. Average of yes/no judgments."),
    "source_utilization": ("Cited-source share", "Distinct sources cited in a delivered answer divided by sources retrieved."),
}
OFFLINE = {
    "precision_at_k": "Precision@3",
    "recall_at_k": "Recall@3",
    "ndcg_at_k": "NDCG@3",
    "ranked_context_precision": "Context precision (rank-aware AP)",
    "context_precision": "Context precision (source IDs)",
    "context_recall": "Context recall (source IDs)",
    "reciprocal_rank": "First relevant source (MRR)",
}
EVAL_LABELS = {**OFFLINE, **{name: LIVE[name][0] for name in ("ragas_answer_relevancy", "ragas_faithfulness")},
               "ragas_factual_correctness": "Answer correctness (Ragas factual F1)",
               "citation_support": "Citation evidence support"}


def retrieval_scores(retrieved, reference, k=RETRIEVAL_K):
    """Macro-average these case scores over questions with labeled evidence."""
    if not reference or retrieved is None:
        return {name: None for name in OFFLINE}
    ranked = list(dict.fromkeys(retrieved))
    gold = set(reference)
    relevance = [int(sid in gold) for sid in ranked]
    hits = sum(relevance[:k])
    dcg = sum(value/math.log2(rank+2) for rank,value in enumerate(relevance[:k]))
    ideal = sum(1/math.log2(rank+2) for rank in range(min(k,len(gold))))
    cumulative = 0
    precision_sum = 0
    for rank, relevant in enumerate(relevance, 1):
        cumulative += relevant
        precision_sum += relevant*cumulative/rank
    import asyncio
    from ragas_eval import id_scores
    scores = asyncio.run(id_scores(ranked, list(gold)))
    return {
        **scores,
        "precision_at_k": hits/k,
        "recall_at_k": hits/len(gold),
        "ndcg_at_k": dcg/ideal,
        "ranked_context_precision": precision_sum/cumulative if cumulative else 0.0,
        "reciprocal_rank": next((1 / rank for rank, sid in enumerate(ranked, 1) if sid in gold), 0.0),
    }


def summarize_metrics(rows, names):
    result = {}
    for name in names:
        values = [r[name] for r in rows if r.get(name) is not None]
        result[name] = {"mean": mean(values) if values else None, "n": len(values)}
    return result
