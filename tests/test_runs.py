import io
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from app import create_app
from rag import ask
from provider import ProviderError
from rag_metrics import retrieval_scores


class RunTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.app = create_app(self.tmp.name)
        self.addCleanup(self.app.extensions["continuous_evaluator"].close)
        self.client = self.app.test_client()

    def test_failures_are_clickable_and_polling_does_not_count(self):
        r = self.client.post("/query", json={"query": "secret question", "api_key": "secret key"})
        self.assertEqual(r.status_code, 409)
        trace = r.json["trace_id"]
        self.assertEqual(r.headers["X-Trace-ID"], trace)
        detail = self.client.get("/traces/" + trace).text
        self.assertIn("document_unavailable", detail)
        self.assertNotIn("secret", detail)
        for _ in range(2):
            metrics = self.client.get("/api/metrics").json
            self.assertEqual((metrics["requests"], metrics["errors"]), (1, 1))
            self.assertEqual(metrics["failures"][0]["trace_id"], trace)
            self.assertGreaterEqual(metrics["p95_ms"], 0)

    @patch("app.prepare_document", return_value=([{"id": "P1T1"}], np.zeros((1, 2))))
    @patch("app.ask")
    def test_documents_are_scoped_and_provider_errors_are_counted(self, answer, prepare):
        response = self.client.post("/upload", data={"file": (io.BytesIO(b"fixture"), "test.pdf")})
        self.assertEqual(response.status_code, 200)
        body = {"query": "q", "api_key": "k"}
        stranger = self.app.test_client()
        self.assertEqual(stranger.post("/query", json=body).status_code, 409)
        answer.return_value = {"status": "insufficient_evidence", "response": "Insufficient evidence",
                               "citations": [], "reason": "model_abstained"}
        self.assertEqual(self.client.post("/query", json=body).json["status"], "insufficient_evidence")
        answer.side_effect = ProviderError("provider_http_503")
        self.assertEqual(self.client.post("/query", json=body).status_code, 502)
        metrics = self.client.get("/api/metrics").json
        self.assertEqual((metrics["requests"], metrics["errors"], metrics["insufficient"]), (4, 2, 1))

    @patch("rag.evidence_parts", return_value=[])
    @patch("rag.retrieve", return_value=[{"id":"P1T1","page":1,"kind":"text","text":"Budget 50"}])
    def test_guard_rejects_bad_evidence(self, retrieve, parts):
        for source, evidence, expected in [("P9T1","Budget 50","unknown_source"),
                                          ("P1T1","Budget 80","quote_mismatch"),
                                          ("P1T1","Budget 50","verification_failed")]:
            draft = {"claims":[{"answer":"Budget 80","source":source,"evidence":evidence}], "relevant_sources":[]}
            with patch("rag.model_json", side_effect=[draft, {"supported":False, "claim_support":[False], "addresses_question":True}]):
                result = ask(None, None, "Budget?", "key", [])
                self.assertEqual((result["reason"], result["citations"]), (expected, []))

    @patch("rag.evidence_parts", return_value=[])
    @patch("rag.retrieve", return_value=[{"id":"P1T1","page":1,"kind":"text","text":"Budget 50"}])
    def test_live_quality_includes_rejected_drafts_and_preserves_missing_scores(self, retrieve, parts):
        claim = {"answer": "Budget 50", "source": "P1T1", "evidence": "Budget 50"}
        draft = {"claims": [claim, claim], "relevant_sources": ["P1T1"]}
        verdict = {"supported": False, "claim_support": [True, False], "addresses_question": True}
        with patch("app.prepare_document", return_value=([], np.zeros((1, 2)))):
            self.client.post("/upload", data={"file": (io.BytesIO(b"private"), "private.pdf")})
        with patch("rag.model_json", side_effect=[draft, verdict, {"claims": [], "relevant_sources": []}]):
            body = {"query": "confidential question", "api_key": "private key"}
            rejected = self.client.post("/query", json=body)
            self.assertEqual(rejected.json["reason"], "verification_failed")
            self.client.post("/query", json=body)
        metrics = self.client.get("/api/metrics").json
        self.assertEqual(metrics["rag_queries"], 2)
        self.assertEqual(metrics["rag_live"]["draft_faithfulness"], {"n":1, "mean":0.5})
        self.assertEqual(metrics["rag_live"]["context_relevance"], {"n":2, "mean":0.5})
        self.assertEqual(metrics["rag_live"]["source_utilization"], {"n":0, "mean":None})
        trace = self.client.get("/traces/" + rejected.json["trace_id"]).text
        self.assertIn("50.0%", trace)
        for secret in ["P1T1", "private key", "confidential question", "Budget 50"]:
            self.assertNotIn(secret, trace + str(metrics))

    def test_ranked_retrieval_uses_only_reference_evidence(self):
        self.assertEqual(retrieval_scores(["irrelevant", "gold", "gold"], ["gold", "missing"]),
                         {"context_precision":0.5, "context_recall":0.5, "reciprocal_rank":0.5})
        self.assertIsNone(retrieval_scores(["irrelevant"], [])["context_recall"])
        self.assertEqual(retrieval_scores([], ["missing"])["context_recall"], 0)

    @patch("rag.evidence_parts", return_value=[])
    @patch("rag.retrieve", return_value=[{"id":"P1T1","page":1,"kind":"text","text":"Budget 50"}])
    def test_guard_rejects_incomplete_and_wrongly_typed_verdicts(self, retrieve, parts):
        draft = {"claims":[{"answer":"Budget 50","source":"P1T1","evidence":"Budget 50"}], "relevant_sources":["P1T1"]}
        for verdict in [{"supported":True}, {"supported":"yes","claim_support":[True],"addresses_question":True},
                        {"supported":True,"claim_support":[],"addresses_question":True}]:
            with patch("rag.model_json", side_effect=[draft, verdict]):
                with self.assertRaisesRegex(ProviderError, "provider_invalid_verdict"):
                    ask(None, None, "Budget?", "key", [])

    def test_existing_database_gets_new_metrics_and_spans(self):
        import sqlite3
        from contextlib import closing
        path = self.tmp.name + "/legacy"
        from pathlib import Path
        Path(path).mkdir()
        with closing(sqlite3.connect(path+"/traces.sqlite3")) as db:
            db.execute("CREATE TABLE runs (trace_id TEXT PRIMARY KEY, started REAL, operation TEXT, status INTEGER, duration_ms REAL, outcome TEXT, stage TEXT, reason TEXT)")
        app = create_app(path)
        self.addCleanup(app.extensions["continuous_evaluator"].close)
        client = app.test_client()
        response = client.post("/query",json={})
        trace = client.get("/traces/"+response.json["trace_id"]).text
        self.assertIn("query", trace)
        self.assertIn("OpenTelemetry", trace)
        export = client.get("/metrics")
        self.assertEqual(export.status_code,200)
        self.assertIn("documind_requests_24h 1.0", export.text)


if __name__ == "__main__":
    unittest.main()
