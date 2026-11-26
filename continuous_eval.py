"""Bounded in-memory background scoring. Keys and text are never queued on disk."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
import threading

from ragas_eval import score_answer
from tracing import CAPTURE


class ContinuousEvaluator:
    def __init__(self, traces):
        self.traces = traces
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ragas")
        self.capacity = threading.BoundedSemaphore(8)
        self.futures = set()
        self.lock = threading.Lock()

    def submit(self, trace_id, question, response, sources, visual, key, *, context=None):
        if not self.capacity.acquire(blocking=False):
            self.traces.evaluation_state(trace_id, "skipped_busy")
            return
        self.traces.evaluation_state(trace_id, "pending")
        context = context or copy_context()

        def evaluate():
            collected = []
            token = CAPTURE.set(collected)
            try:
                self.traces.evaluation_state(trace_id, "running")
                states = []
                def save(name, value, status):
                    states.append(status)
                    self.traces.save_score(trace_id, name, value, status)
                asyncio.run(score_answer(question, response, sources, visual, key,
                                         save))
                failed = any(status not in ("complete", "not_applicable") for status in states)
                state = ("partial" if "complete" in states else "error") if failed else "complete"
                self.traces.evaluation_state(trace_id, state)
            except Exception:
                self.traces.evaluation_state(trace_id, "error")
            finally:
                self.traces.save_spans(trace_id, collected)
                CAPTURE.reset(token)
                self.capacity.release()

        try:
            future = self.executor.submit(context.run, evaluate)
        except RuntimeError:
            self.capacity.release()
            self.traces.evaluation_state(trace_id, "error")
            return
        with self.lock:
            self.futures.add(future)

        def done(future):
            with self.lock:
                self.futures.discard(future)
        future.add_done_callback(done)

    def close(self):
        self.executor.shutdown(wait=True)
