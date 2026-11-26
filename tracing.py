"""OpenTelemetry spans retained in the request's SQLite trace."""
from contextlib import contextmanager
from contextvars import ContextVar

from opentelemetry.sdk.trace import TracerProvider, SpanProcessor
from opentelemetry.trace import Status, StatusCode

CAPTURE = ContextVar("documind_spans", default=None)


class RequestSpanProcessor(SpanProcessor):
    def on_end(self, span):
        collected = CAPTURE.get()
        if collected is not None:
            collected.append({
                "span_id": f"{span.context.span_id:016x}",
                "parent_id": f"{span.parent.span_id:016x}" if span.parent else None,
                "name": span.name,
                "started": span.start_time / 1e9,
                "duration_ms": (span.end_time - span.start_time) / 1e6,
                "status": span.status.status_code.name,
            })


provider = TracerProvider()
provider.add_span_processor(RequestSpanProcessor())
tracer = provider.get_tracer("documind")


@contextmanager
def stage(name):
    with tracer.start_as_current_span(name, record_exception=False,
                                      set_status_on_exception=False) as span:
        try:
            yield span
        except Exception:
            span.set_status(Status(StatusCode.ERROR))
            raise
