import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
import requests

from provider import generate, CALLS, estimate_cost, ProviderError
from telemetry import Traces, latency_summary


class StreamResponse:
    ok = True
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def iter_lines(self, **kwargs):
        events = [
            {"candidates":[{"content":{"parts":[{"text":"Hello "}]}}],
             "usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":1,"totalTokenCount":6}},
            {"candidates":[{"content":{"parts":[{"text":"world"}]}}],
             "usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"thoughtsTokenCount":2,"totalTokenCount":10}},
        ]
        yield ": keep-alive"
        for event in events:
            yield "data: "+json.dumps(event)
            yield ""


class MeasurementTests(unittest.TestCase):
    def test_streaming_usage_is_cumulative_per_call_and_ttft_uses_first_text(self):
        records = []
        token = CALLS.set(records)
        try:
            with patch("provider.requests.post",return_value=StreamResponse()), patch(
                    "provider.time.perf_counter",side_effect=[1,1.25,1.9]):
                self.assertEqual(generate([{"text":"test"}],"secret"),"Hello world")
        finally:
            CALLS.reset(token)
        self.assertEqual(records[0]["usage"]["promptTokenCount"],5)
        self.assertEqual(records[0]["usage"]["totalTokenCount"],10)
        self.assertEqual(records[0]["ttft_ms"],250)
        self.assertNotIn("secret",str(records))
        self.assertAlmostEqual(records[0]["estimated_cost_usd"],(5*.3+5*2.5)/1e6)

    def test_unknown_model_price_is_unavailable_and_thinking_cache_are_accounted(self):
        with patch("provider.MODEL","unknown-model"):
            self.assertIsNone(estimate_cost({"promptTokenCount":10,"totalTokenCount":12}))
        self.assertAlmostEqual(estimate_cost({"promptTokenCount":100,"cachedContentTokenCount":20,
            "candidatesTokenCount":5,"thoughtsTokenCount":3,"totalTokenCount":108}), (80*.3+20*.03+8*2.5)/1e6)

    def test_percentiles_rates_and_retry_costs_survive_database_reopen(self):
        with tempfile.TemporaryDirectory() as folder:
            t=Traces(Path(folder))
            now=time.time()
            t.add(("a",now,"query",200,10,"answered","done","verified"),{"retrieved_count":3})
            t.add(("b",now,"query",200,30,"insufficient_evidence","done","verification_failed"),{"retrieved_count":3})
            t.add(("c",now,"query",500,100,"error","generation","provider_unreachable"),{"retrieved_count":3})
            call={"call_id":"retry1","model":"test","started":now,"status":"ok","duration_ms":10,
                  "ttft_ms":3,"usage":{"promptTokenCount":10,"candidatesTokenCount":4,"totalTokenCount":14},
                  "estimated_cost_usd":0.01}
            t.save_calls("a","request",[call])
            t.save_calls("a","request",[call])
            t.evaluation_state("a","pending")
            m=t.overview()
            self.assertEqual(m["usage"]["cost_per_query"]["n"],0)
            t.save_calls("a","evaluation",[{**call,"call_id":"eval1","estimated_cost_usd":0.02}])
            t.evaluation_state("a","complete")
            m=Traces(Path(folder)).overview()
            self.assertEqual(m["total_requests"],3)
            self.assertEqual(m["requests_per_minute"],3)
            self.assertAlmostEqual(m["success_rate"],2/3)
            self.assertAlmostEqual(m["guardrail_trigger_rate"],1/3)
            self.assertEqual(m["abstention_rate"],0.5)
            self.assertEqual(m["p50_ms"],30)
            self.assertEqual(m["p99_ms"],100)
            self.assertAlmostEqual(m["usage"]["cost_per_query"]["mean_usd"],0.03)
            self.assertEqual(m["usage"]["request"]["total_tokens"],14)
            self.assertEqual(m["provider_ttft"]["p95_ms"],3)

    def test_no_samples_are_unavailable(self):
        self.assertIsNone(latency_summary([])["p95_ms"])

    def test_partial_stream_failure_preserves_observed_usage_and_has_no_price(self):
        class BrokenStream(StreamResponse):
            def iter_lines(self, **kwargs):
                yield 'data: {"candidates":[{"content":{"parts":[{"text":"partial"}]}}],"usageMetadata":{"promptTokenCount":5,"totalTokenCount":6}}'
                yield ''
                raise requests.ConnectionError("private transport detail")
        records=[]
        token=CALLS.set(records)
        try:
            with patch("provider.requests.post",return_value=BrokenStream()):
                with self.assertRaisesRegex(ProviderError,"provider_unreachable"):
                    generate([{"text":"sensitive question"}],"secret")
        finally:
            CALLS.reset(token)
        self.assertIsNone(records[0]["estimated_cost_usd"])
        self.assertEqual(records[0]["usage"]["totalTokenCount"],6)
        self.assertNotIn("private transport detail",str(records))


if __name__ == "__main__":
    unittest.main()
