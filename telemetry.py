"""Local operational telemetry. Store only fields permitted in public traces."""
import json
import math
from contextlib import closing
from pathlib import Path
import sqlite3
import time


class Traces:
    def __init__(self, directory):
        self.path = directory / "traces.sqlite3"
        with closing(self.connect()) as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("""CREATE TABLE IF NOT EXISTS runs (
                trace_id TEXT PRIMARY KEY, started REAL, operation TEXT, status INTEGER,
                duration_ms REAL, outcome TEXT, stage TEXT, reason TEXT)""")
            db.execute("CREATE INDEX IF NOT EXISTS runs_started ON runs(started)")
            db.commit()

    def connect(self):
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        return db

    def add(self, values):
        with closing(self.connect()) as db, db:
            db.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?)", values)
            db.execute("DELETE FROM runs WHERE started < ?", (time.time() - 7 * 86400,))

    def get(self, trace_id):
        with closing(self.connect()) as db:
            row = db.execute("SELECT * FROM runs WHERE trace_id=?", (trace_id,)).fetchone()
            return dict(row) if row else None

    def overview(self):
        cutoff = time.time() - 86400
        with closing(self.connect()) as db:
            db.execute("BEGIN")
            totals = dict(db.execute("""SELECT COUNT(*) AS requests,
                COUNT(CASE WHEN status >= 400 THEN 1 END) AS errors,
                COUNT(CASE WHEN outcome='insufficient_evidence' THEN 1 END) AS insufficient
                FROM runs WHERE started >= ?""", (cutoff,)).fetchone())
            n = totals["requests"]
            totals["p95_ms"] = db.execute(
                "SELECT duration_ms FROM runs WHERE started>=? ORDER BY duration_ms LIMIT 1 OFFSET ?",
                (cutoff, max(0, math.ceil(.95 * n) - 1))).fetchone()[0] if n else None
            failures = db.execute("""SELECT * FROM runs WHERE started >= ? AND status>=400
                ORDER BY started DESC LIMIT 25""", (cutoff,)).fetchall()
        return {**totals, "failures": [dict(r) for r in failures]}


def latest_evals():
    directory = Path(__file__).parent / "evals" / "results"
    runs = {}
    for mode in ("baseline", "guarded"):
        path = directory / f"{mode}.json"
        if path.exists():
            report = json.loads(path.read_text(encoding="utf-8"))
            runs[mode] = {k: report[k] for k in ("run_id", "model", "scores", "finished_at", "suite_sha256")}
    return runs
