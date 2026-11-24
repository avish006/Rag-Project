"""DocuMind HTTP application."""
import json
import os
from pathlib import Path
import re
import shutil
import time
from uuid import uuid4

import fitz
import numpy as np
from flask import Flask, g, jsonify, render_template, request, send_file
from werkzeug.exceptions import HTTPException

from rag import ask, prepare_document
from provider import ProviderError
from telemetry import Traces, latest_evals


def create_app(directory=None):
    app = Flask(__name__)
    store = Path(directory or os.getenv("DOCUMIND_DATA", "instance/documind")).resolve()
    documents = store / "documents"
    documents.mkdir(parents=True, exist_ok=True)
    traces = Traces(store)
    app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

    def failure(message, code, reason):
        g.outcome, g.reason = "error", reason
        return jsonify(error=message, trace_id=g.trace_id), code

    @app.before_request
    def begin():
        g.trace_id = uuid4().hex
        g.started, g.clock = time.time(), time.perf_counter()
        g.outcome, g.stage, g.reason = "ok", "validation", "ok"

    @app.after_request
    def finish(response):
        response.headers["X-Trace-ID"] = g.trace_id
        if request.path in ("/upload", "/query"):
            traces.add((g.trace_id, g.started, request.path[1:], response.status_code,
                        round((time.perf_counter() - g.clock) * 1000, 2), g.outcome, g.stage, g.reason))
        if request.path.startswith(("/api/", "/uploaded/", "/traces/")):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.errorhandler(Exception)
    def unexpected(exc):
        if isinstance(exc, HTTPException):
            return failure(exc.name, exc.code, f"http_{exc.code}")
        app.logger.error("trace=%s exception=%s", g.trace_id, type(exc).__name__)
        return failure("DocuMind could not complete this request.", 500, "internal_error")

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.post("/upload")
    def upload():
        file = request.files.get("file")
        if not file or not file.filename.lower().endswith(".pdf"):
            return failure("Choose a PDF file.", 400, "missing_or_invalid_file")
        # Opportunistic retention, scoped to generated document directories.
        cutoff = time.time() - 86400
        for item in documents.iterdir():
            if re.fullmatch("[a-f0-9]{32}", item.name) and item.is_dir():
                try:
                    if item.stat().st_mtime < cutoff:
                        shutil.rmtree(item)
                except FileNotFoundError:
                    pass
        doc_id = uuid4().hex
        folder = documents / doc_id
        folder.mkdir()
        pdf = folder / "document.pdf"
        ready = False
        try:
            file.save(pdf)
            g.stage = "indexing"
            sources, vectors = prepare_document(pdf)
            np.save(folder / "vectors.npy", vectors, allow_pickle=False)
            (folder / "sources.json").write_text(json.dumps(sources), encoding="utf-8")
            ready = True
        except (fitz.FileDataError, ValueError):
            return failure("Upload a readable, unencrypted PDF with text or figures.", 422, "unreadable_pdf")
        finally:
            if not ready:
                shutil.rmtree(folder, ignore_errors=True)
        response = jsonify(message="PDF processed!", document_id=doc_id, pdf_url=f"/uploaded/{doc_id}.pdf",
                           trace_id=g.trace_id)
        response.set_cookie("documind_document", doc_id, httponly=True, samesite="Lax", secure=request.is_secure)
        return response

    def document(doc_id):
        if isinstance(doc_id, str) and re.fullmatch("[a-f0-9]{32}", doc_id):
            folder = documents / doc_id
            if (folder / "sources.json").exists() and folder.stat().st_mtime >= time.time() - 86400:
                return folder
        return None

    @app.post("/query")
    def query():
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return failure("Send a question as JSON.", 400, "invalid_json")
        question, key = body.get("query"), body.get("api_key")
        if not isinstance(question, str) or not 0 < len(question.strip()) <= 4000:
            return failure("Enter a question of up to 4,000 characters.", 400, "invalid_question")
        if not isinstance(key, str) or not key.strip():
            return failure("Enter your Google API key in settings.", 401, "missing_key")
        folder = document(body.get("document_id", request.cookies.get("documind_document")))
        if folder is None:
            return failure("Upload a PDF before asking a question.", 409, "document_unavailable")
        history = body.get("history", [])
        if not isinstance(history, list) or not all(isinstance(t, dict) and
                all(isinstance(t.get(k), str) for k in ("query", "response")) for t in history[-6:]):
            return failure("Invalid conversation history.", 400, "invalid_history")
        history = [{k: t[k][:3000] for k in ("query", "response")} for t in history[-6:]]
        g.stage = "retrieval"
        data = (json.loads((folder / "sources.json").read_text(encoding="utf-8")),
                np.load(folder / "vectors.npy", allow_pickle=False))
        g.stage = "answer_generation"
        try:
            result = ask(folder / "document.pdf", data, question.strip(), key.strip(), history)
        except ProviderError as exc:
            reason = str(exc)
            if reason in ("provider_http_400", "provider_http_401", "provider_http_403"):
                return failure("Google rejected the request. Check your API key and model access.", 502, reason)
            if reason == "provider_http_429":
                return failure("Google's request quota was reached. Try again later.", 429, reason)
            return failure("The model could not complete this request. Please retry.", 502, reason)
        g.outcome, g.reason = result["status"], result["reason"]
        g.stage = "evidence_check"
        return jsonify(**result, trace_id=g.trace_id)

    @app.get("/uploaded/<doc_id>.pdf")
    def serve_pdf(doc_id):
        folder = document(doc_id)
        if folder is None:
            return failure("Document unavailable.", 404, "document_unavailable")
        return send_file(folder / "document.pdf", mimetype="application/pdf")

    @app.get("/health")
    def health():
        return jsonify(status="healthy")

    @app.get("/dashboard")
    def dashboard():
        return render_template("dashboard.html")

    @app.get("/evaluations/<mode>")
    def evaluation(mode):
        root = Path(app.root_path) / "evals"
        path = root / "results" / f"{mode}.json"
        if mode not in ("baseline", "guarded") or not path.is_file():
            return failure("Evaluation unavailable.", 404, "unknown_evaluation")
        report = json.loads(path.read_text(encoding="utf-8"))
        labels = {c["id"]: c for c in json.loads((root / "cases.json").read_text(encoding="utf-8"))}
        return render_template("evaluation.html", mode=mode, report=report, labels=labels)

    @app.get("/api/metrics")
    def metrics():
        return jsonify(**traces.overview(), evals=latest_evals())

    @app.get("/traces/<trace_id>")
    def trace(trace_id):
        run = traces.get(trace_id)
        if run is None:
            return render_template("trace.html", run=None), 404
        return render_template("trace.html", run=run)

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
