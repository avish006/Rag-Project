"""Ragas adapters for the existing Google transport and local BGE encoder."""
import asyncio
import math
import os
import re
import logging
import time

os.environ["RAGAS_DO_NOT_TRACK"] = "true"
logging.getLogger("ragas").setLevel(logging.CRITICAL)

from langchain_core.embeddings import Embeddings
from langchain_core.outputs import Generation, LLMResult
from ragas import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import BaseRagasLLM
from ragas.metrics import Faithfulness, ResponseRelevancy, IDBasedContextPrecision, IDBasedContextRecall
from ragas.run_config import RunConfig

from provider import generate, ProviderError
from tracing import stage


class GoogleJudge(BaseRagasLLM):
    def __init__(self, key):
        super().__init__(run_config=RunConfig(max_retries=1, timeout=90))
        self.key = key

    def is_finished(self, response):
        return True

    def generate_text(self, prompt, n=1, temperature=0.01, stop=None, callbacks=None):
        generations = []
        for _ in range(n):
            for attempt in range(2):
                try:
                    text = generate([{"text": prompt.to_string()}], self.key,
                                    temperature or 0, max_tokens=4000, timeout=25)
                    generations.append(Generation(text=text))
                    break
                except ProviderError as exc:
                    if str(exc) not in ("provider_http_429", "provider_http_503", "provider_unreachable") or attempt == 1:
                        raise
                    time.sleep(8 * (attempt + 1))
        return LLMResult(generations=[generations])

    async def agenerate_text(self, prompt, n=1, temperature=0.01, stop=None, callbacks=None):
        return await asyncio.to_thread(self.generate_text, prompt, n, temperature, stop, callbacks)


class BGEEmbeddings(Embeddings):
    def embed_documents(self, texts):
        from rag import embeddings
        return embeddings(texts).tolist()

    def embed_query(self, text):
        return self.embed_documents([text])[0]


async def id_scores(retrieved, reference):
    sample = SingleTurnSample(retrieved_context_ids=retrieved, reference_context_ids=reference)
    return {
        "context_precision": await IDBasedContextPrecision().single_turn_ascore(sample),
        "context_recall": await IDBasedContextRecall().single_turn_ascore(sample),
    }


async def score_answer(question, response, sources, visual, key, save, *, skip=()):
    """Save each independent metric, including unavailable/error state."""
    sample = SingleTurnSample(user_input=question,
                              response=re.sub(r"\s*\[p\. \d+\]", "", response),
                              retrieved_contexts=[s["text"] for s in sources if s["kind"] == "text"])
    judge = GoogleJudge(key)
    relevance = ResponseRelevancy(llm=judge, embeddings=LangchainEmbeddingsWrapper(BGEEmbeddings()), strictness=3)
    metrics = [("ragas_answer_relevancy", relevance)]
    if not visual and sample.retrieved_contexts:
        metrics.append(("ragas_faithfulness", Faithfulness(llm=judge)))
    else:
        save("ragas_faithfulness", None, "not_applicable")

    async def measure(name, metric):
        try:
            with stage(name):
                score = float(await metric.single_turn_ascore(sample, timeout=180))
                if not math.isfinite(score):
                    save(name, None, "unavailable")
                    return
                save(name, score, "complete")
        except Exception as exc:
            # Prompts, raw provider errors and user content stay out of logs.
            code = str(exc) if isinstance(exc, ProviderError) else "error"
            save(name, None, code)

    for name, metric in metrics:
        if name not in skip:
            await measure(name, metric)
