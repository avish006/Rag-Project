"""Local operational telemetry. Store only fields permitted in public traces."""
import json
import math
from contextlib import closing
from pathlib import Path
import sqlite3
import time
from rag_metrics import LIVE
from prometheus_client import CollectorRegistry, generate_latest
from prometheus_client.core import GaugeMetricFamily


class Traces:
    def __init__(self, directory):
        self.path = directory / "traces.sqlite3"
        with closing(self.connect()) as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("""CREATE TABLE IF NOT EXISTS runs (
                trace_id TEXT PRIMARY KEY, started REAL, operation TEXT, status INTEGER,
                duration_ms REAL, outcome TEXT, stage TEXT, reason TEXT)""")
            db.execute("CREATE INDEX IF NOT EXISTS runs_started ON runs(started)")
            db.execute("""CREATE TABLE IF NOT EXISTS query_metrics (
                trace_id TEXT REFERENCES runs(trace_id) ON DELETE CASCADE,
                name TEXT, value REAL NOT NULL, PRIMARY KEY(trace_id, name))""")
            db.execute("""CREATE TABLE IF NOT EXISTS spans (
                trace_id TEXT REFERENCES runs(trace_id) ON DELETE CASCADE, span_id TEXT,
                parent_id TEXT, name TEXT, started REAL, duration_ms REAL, status TEXT,
                PRIMARY KEY(trace_id,span_id))""")
            db.execute("""CREATE TABLE IF NOT EXISTS query_evals (
                trace_id TEXT PRIMARY KEY REFERENCES runs(trace_id) ON DELETE CASCADE,
                status TEXT, updated REAL)""")
            db.execute("""CREATE TABLE IF NOT EXISTS metric_status (
                trace_id TEXT REFERENCES runs(trace_id) ON DELETE CASCADE,
                name TEXT, status TEXT, PRIMARY KEY(trace_id,name))""")
            db.execute("UPDATE query_evals SET status='interrupted' WHERE status IN ('pending','running')")
            db.commit()

    def connect(self):
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        return db

    def add(self, values, measurements=None):
        with closing(self.connect()) as db, db:
            db.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?)", values)
            if measurements:
                db.executemany("INSERT INTO query_metrics VALUES (?,?,?)",
                               [(values[0], name, measurements[name]) for name in LIVE
                                if measurements.get(name) is not None])
            db.execute("DELETE FROM runs WHERE started < ?", (time.time() - 7 * 86400,))

    def get(self, trace_id):
        with closing(self.connect()) as db:
            row = db.execute("SELECT * FROM runs WHERE trace_id=?", (trace_id,)).fetchone()
            if not row:
                return None
            scores = dict(db.execute("SELECT name, value FROM query_metrics WHERE trace_id=?", (trace_id,)))
            spans = [dict(s) for s in db.execute("SELECT * FROM spans WHERE trace_id=? ORDER BY started", (trace_id,))]
            state = db.execute("SELECT status FROM query_evals WHERE trace_id=?", (trace_id,)).fetchone()
            metric_states = dict(db.execute("SELECT name,status FROM metric_status WHERE trace_id=?", (trace_id,)))
            return {**dict(row), "quality": scores, "spans": spans,
                    "eval_status": state[0] if state else "not_scheduled", "metric_status": metric_states}

    def save_spans(self, trace_id, spans):
        with closing(self.connect()) as db, db:
            db.executemany("INSERT OR IGNORE INTO spans VALUES (?,?,?,?,?,?,?)",
                           [(trace_id, s["span_id"], s["parent_id"], s["name"], s["started"],
                             s["duration_ms"], s["status"]) for s in spans])

    def evaluation_state(self, trace_id, status):
        with closing(self.connect()) as db, db:
            db.execute("INSERT OR REPLACE INTO query_evals VALUES (?,?,?)", (trace_id, status, time.time()))

    def save_score(self, trace_id, name, value, status):
        with closing(self.connect()) as db, db:
            # A worker can finish after the retention sweep.
            if not db.execute("SELECT 1 FROM runs WHERE trace_id=?", (trace_id,)).fetchone():
                return
            if value is not None:
                db.execute("INSERT OR REPLACE INTO query_metrics VALUES (?,?,?)", (trace_id, name, value))
            db.execute("INSERT OR REPLACE INTO metric_status VALUES (?,?,?)", (trace_id, name, status))

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
            live = {name: {"mean": None, "n": 0} for name in LIVE}
            for name, avg, n in db.execute("""SELECT m.name, AVG(m.value), COUNT(*)
                    FROM runs r JOIN query_metrics m USING(trace_id)
                    WHERE r.started >= ? AND r.operation='query' GROUP BY m.name""", (cutoff,)):
                live[name] = {"mean": avg, "n": n}
            query_count = db.execute("SELECT COUNT(*) FROM runs WHERE operation='query' AND started>=?",
                                     (cutoff,)).fetchone()[0]
            recent = db.execute("""SELECT trace_id, started, outcome FROM runs
                WHERE operation='query' AND started>=? ORDER BY started DESC LIMIT 8""", (cutoff,)).fetchall()
            rows = {r["trace_id"]: {**dict(r), "quality": {}} for r in recent}
            if rows:
                placeholders = ",".join("?" for _ in rows)
                for tid, name, value in db.execute(
                        f"SELECT trace_id,name,value FROM query_metrics WHERE trace_id IN ({placeholders})", tuple(rows)):
                    rows[tid]["quality"][name] = value
                for tid, status in db.execute(
                        f"SELECT trace_id,status FROM query_evals WHERE trace_id IN ({placeholders})", tuple(rows)):
                    rows[tid]["eval_status"] = status
            eval_counts = dict(db.execute("""SELECT e.status,COUNT(*) FROM query_evals e JOIN runs r USING(trace_id)
                WHERE r.started>=? GROUP BY e.status""", (cutoff,)))
            guard_counts = dict(db.execute("""SELECT reason,COUNT(*) FROM runs WHERE started>=?
                AND operation='query' AND outcome='insufficient_evidence' GROUP BY reason""", (cutoff,)))
            first_hour = int(time.time() // 3600) - 23
            timeline = [{"started": (first_hour+i)*3600, "requests":0, "errors":0, "insufficient":0, "p95_ms":None}
                        for i in range(24)]
            durations = [[] for _ in timeline]
            for started, latency, status, outcome in db.execute(
                    "SELECT started,duration_ms,status,outcome FROM runs WHERE started>=?", (first_hour*3600,)):
                bucket = int(started//3600) - first_hour
                if 0 <= bucket < 24:
                    timeline[bucket]["requests"] += 1
                    timeline[bucket]["errors"] += int(status >= 400)
                    timeline[bucket]["insufficient"] += int(outcome == "insufficient_evidence")
                    durations[bucket].append(latency)
            for bucket, values in enumerate(durations):
                if values:
                    timeline[bucket]["p95_ms"] = sorted(values)[math.ceil(.95*len(values))-1]
            for hour, name, average in db.execute("""SELECT CAST(r.started/3600 AS INTEGER),m.name,AVG(m.value)
                FROM runs r JOIN query_metrics m USING(trace_id) WHERE r.started>=?
                AND m.name IN ('ragas_answer_relevancy','ragas_faithfulness')
                GROUP BY CAST(r.started/3600 AS INTEGER),m.name""", (first_hour*3600,)):
                bucket = hour - first_hour
                if 0 <= bucket < 24:
                    timeline[bucket][name] = average
        return {**totals, "failures": [dict(r) for r in failures], "rag_queries": query_count,
                "rag_live": live, "recent_queries": list(rows.values()), "eval_jobs": eval_counts,
                "guardrails": guard_counts, "timeline": timeline}

    def prometheus(self):
        registry = CollectorRegistry()
        registry.register(MetricsCollector(self.overview()))
        return generate_latest(registry)


class MetricsCollector:
    def __init__(self, overview):
        self.overview = overview

    def collect(self):
        m = self.overview
        for key in ("requests", "errors", "insufficient", "rag_queries"):
            gauge = GaugeMetricFamily("documind_"+key+"_24h", "Rolling 24-hour "+key)
            gauge.add_metric([], m[key])
            yield gauge
        latency = GaugeMetricFamily("documind_request_p95_seconds_24h", "Rolling 24-hour request p95")
        if m["p95_ms"] is not None:
            latency.add_metric([], m["p95_ms"]/1000)
        yield latency
        scores = GaugeMetricFamily("documind_rag_score_mean_24h", "Mean quality by measured query", labels=["metric"])
        counts = GaugeMetricFamily("documind_rag_scored_queries_24h", "Quality measurement count", labels=["metric"])
        for name, value in m["rag_live"].items():
            counts.add_metric([name], value["n"])
            if value["mean"] is not None:
                scores.add_metric([name], value["mean"])
        yield scores
        yield counts
        jobs = GaugeMetricFamily("documind_eval_jobs_24h", "Background evaluation jobs", labels=["status"])
        for status, count in m["eval_jobs"].items():
            jobs.add_metric([status], count)
        yield jobs
        guards = GaugeMetricFamily("documind_guardrail_decisions_24h", "Insufficient evidence decisions", labels=["reason"])
        for reason, count in m["guardrails"].items():
            guards.add_metric([reason], count)
        yield guards


def latest_evals():
    directory = Path(__file__).parent / "evals" / "results"
    runs = {}
    for mode in ("baseline", "guarded"):
        path = directory / f"{mode}.json"
        if path.exists():
            report = json.loads(path.read_text(encoding="utf-8"))
            runs[mode] = {k: report[k] for k in ("run_id", "model", "scores", "finished_at", "suite_sha256")}
            runs[mode]["rag_scores"] = report.get("rag_scores", {})
    return runs
