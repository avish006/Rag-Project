import io
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from app import create_app
from rag import ask
from provider import ProviderError


class RunTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.app = create_app(self.tmp.name)
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
            draft = {"claims":[{"answer":"Budget 80","source":source,"evidence":evidence}]}
            with patch("rag.model_json", side_effect=[draft, {"supported":False}]):
                result = ask(None, None, "Budget?", "key", [])
                self.assertEqual((result["reason"], result["citations"]), (expected, []))


if __name__ == "__main__":
    unittest.main()
