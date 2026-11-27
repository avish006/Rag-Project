"""Gemini HTTP transport. Exceptions carry only safe diagnostic codes."""
import os
import re
import json
import math
import time
from contextvars import ContextVar
from uuid import uuid4

import requests

MODEL = os.getenv("DOCUMIND_MODEL", "gemini-3.5-flash-lite")
CALLS = ContextVar("documind_provider_calls", default=None)
PRICING_SOURCE = "https://aistudio.google.com/docs/pricing#gemini-3.5-flash-lite"


def pricing():
    """USD per million tokens, paid Standard tier. Verified 2026-11-27."""
    names = ("DOCUMIND_INPUT_USD_PER_M", "DOCUMIND_OUTPUT_USD_PER_M", "DOCUMIND_CACHED_USD_PER_M")
    if any(name in os.environ for name in names):
        try:
            rates = tuple(float(os.environ[name]) for name in names)
            if all(math.isfinite(v) and v >= 0 for v in rates):
                return rates
        except (KeyError, ValueError):
            return None
    return (0.30, 2.50, 0.03) if MODEL == "gemini-3.5-flash-lite" else None


def estimate_cost(usage):
    rates = pricing()
    if rates is None or not all(k in usage for k in ("promptTokenCount", "totalTokenCount")):
        return None
    input_rate, output_rate, cached_rate = rates
    cached = usage.get("cachedContentTokenCount", 0)
    return ((usage["promptTokenCount"]-cached+usage.get("toolUsePromptTokenCount", 0))*input_rate +
            cached*cached_rate +
            (usage.get("candidatesTokenCount", 0)+usage.get("thoughtsTokenCount", 0))*output_rate)/1_000_000


def stream_events(response):
    """SSE event boundaries may span transport chunks or contain multiple data lines."""
    data = []
    response.encoding = "utf-8"
    for line in response.iter_lines(chunk_size=1, decode_unicode=True):
        if not line:
            if data:
                payload = "\n".join(data)
                data = []
                if payload != "[DONE]":
                    yield json.loads(payload)
        elif line.startswith("data:"):
            data.append(line[5:].lstrip())
    if data and "\n".join(data) != "[DONE]":
        yield json.loads("\n".join(data))


class ProviderError(Exception):
    def __init__(self, code, retry_after=None):
        super().__init__(code)
        self.retry_after = retry_after


def generate(parts, key, temperature=0.7, *, max_tokens=1600, timeout=75):
    if not re.fullmatch(r"[\w.\-]+", MODEL):
        raise ValueError("Invalid model ID")
    started = time.perf_counter()
    record = {"call_id": uuid4().hex, "model": MODEL, "started": time.time(), "status": "error",
              "ttft_ms": None, "usage": {}, "estimated_cost_usd": None}
    fragments = []
    try:
        with requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:streamGenerateContent",
            params={"alt": "sse"}, stream=True,
            headers={"x-goog-api-key": key},
            json={"contents": [{"role": "user", "parts": parts}],
                  "generationConfig": {"maxOutputTokens": max_tokens, "temperature": temperature}},
            timeout=(8, timeout),
        ) as response:
            if not response.ok:
                try:
                    retry_after = min(60,max(0,float(response.headers.get("Retry-After",20))))
                except ValueError:
                    retry_after = 20
                raise ProviderError(f"provider_http_{response.status_code}", retry_after=retry_after)
            for event in stream_events(response):
                if not isinstance(event, dict):
                    raise ProviderError("provider_invalid_stream")
                if "error" in event:
                    raise ProviderError("provider_stream_error")
                if isinstance(event.get("usageMetadata"), dict):
                    record["usage"].update(event["usageMetadata"])
                candidates = event.get("candidates", [])
                if not candidates:
                    continue
                for part in candidates[0].get("content", {}).get("parts", []):
                    if part.get("text") and not part.get("thought"):
                        if record["ttft_ms"] is None:
                            record["ttft_ms"] = (time.perf_counter()-started)*1000
                        fragments.append(part["text"])
        answer = "".join(fragments).strip()
        if not answer:
            raise ProviderError("provider_empty_response")
        record["status"] = "ok"
        return answer
    except requests.RequestException:
        raise ProviderError("provider_unreachable") from None
    except (ValueError, KeyError, IndexError, TypeError, AttributeError):
        raise ProviderError("provider_invalid_stream") from None
    finally:
        record["duration_ms"] = (time.perf_counter()-started)*1000
        # Keep only numeric token metadata. Stream metadata is cumulative within a call.
        record["usage"] = {k:v for k,v in record["usage"].items()
                           if k in ("promptTokenCount", "candidatesTokenCount", "thoughtsTokenCount",
                                    "cachedContentTokenCount", "toolUsePromptTokenCount", "totalTokenCount")
                           and type(v) is int and v >= 0}
        if record["status"] == "ok":
            record["estimated_cost_usd"] = estimate_cost(record["usage"])
        calls = CALLS.get()
        if calls is not None:
            calls.append(record)
