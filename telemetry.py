"""Local operational telemetry. Store only fields permitted in public traces."""
import json
import math
from contextlib import closing
from pathlib import Path
import sqlite3
import time
from rag_metrics import LIVE, GUARD_REASONS
from prometheus_client import CollectorRegistry, generate_latest
from prometheus_client.core import GaugeMetricFamily, CounterMetricFamily


def latency_summary(values):
    ordered = sorted(values)
    if not ordered:
        return {"n":0, "mean_ms":None, "p50_ms":None, "p95_ms":None, "p99_ms":None}
    return {"n":len(ordered), "mean_ms":sum(ordered)/len(ordered),
            **{f"p{p}_ms":ordered[math.ceil(p/100*len(ordered))-1] for p in (50,95,99)}}


def usage_summary(rows):
    """Sums only observed metadata. Each upstream call has its own record."""
    result = {"calls":len(rows), "measured_calls":sum(r["status"]=="ok" and r["total_tokens"] is not None for r in rows),
              "priced_calls":sum(r["estimated_cost_usd"] is not None for r in rows)}
    for key in ("input_tokens","output_tokens","thinking_tokens","cached_tokens","total_tokens","estimated_cost_usd"):
        values = [r[key] for r in rows if r[key] is not None]
        result[key] = sum(values) if values else None
    result["ttft"] = latency_summary([r["ttft_ms"] for r in rows if r["ttft_ms"] is not None])
    return result


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
            db.execute("""CREATE TABLE IF NOT EXISTS provider_calls (
                trace_id TEXT REFERENCES runs(trace_id) ON DELETE CASCADE,
                call_id TEXT PRIMARY KEY, phase TEXT, model TEXT, started REAL, status TEXT,
                duration_ms REAL, ttft_ms REAL, input_tokens INTEGER, output_tokens INTEGER,
                thinking_tokens INTEGER, cached_tokens INTEGER, total_tokens INTEGER, estimated_cost_usd REAL)""")
            db.execute("CREATE INDEX IF NOT EXISTS provider_trace ON provider_calls(trace_id)")
            db.execute("""CREATE TABLE IF NOT EXISTS retrieval_scores (
                trace_id TEXT REFERENCES runs(trace_id) ON DELETE CASCADE, rank INTEGER,
                kind TEXT, cosine REAL, bm25 REAL, rrf REAL, PRIMARY KEY(trace_id,rank))""")
            db.execute("CREATE TABLE IF NOT EXISTS request_counters (name TEXT PRIMARY KEY, value INTEGER)")
            db.execute("INSERT OR IGNORE INTO request_counters SELECT 'total',COUNT(*) FROM runs")
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
            db.execute("UPDATE request_counters SET value=value+1 WHERE name='total'")
            if measurements:
                db.executemany("INSERT INTO query_metrics VALUES (?,?,?)",
                               [(values[0], name, measurements[name]) for name in LIVE
                                if measurements.get(name) is not None])
                db.executemany("INSERT INTO retrieval_scores VALUES (?,?,?,?,?,?)",
                    [(values[0],r["rank"],r["kind"],r["cosine"],r["bm25"],r["rrf"])
                     for r in measurements.get("retrieval_score_rows", [])])
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
            calls = [dict(c) for c in db.execute("SELECT * FROM provider_calls WHERE trace_id=? ORDER BY started", (trace_id,))]
            retrieval = [dict(r) for r in db.execute("SELECT rank,kind,cosine,bm25,rrf FROM retrieval_scores WHERE trace_id=? ORDER BY rank", (trace_id,))]
            return {**dict(row), "quality": scores, "spans": spans,
                    "eval_status": state[0] if state else "not_scheduled", "metric_status": metric_states,
                    "provider_calls":calls, "usage":usage_summary(calls), "retrieval":retrieval}

    def save_calls(self, trace_id, phase, calls):
        with closing(self.connect()) as db, db:
            if not db.execute("SELECT 1 FROM runs WHERE trace_id=?", (trace_id,)).fetchone():
                return
            for call in calls:
                usage = call["usage"]
                measured = "promptTokenCount" in usage and "totalTokenCount" in usage
                db.execute("INSERT OR IGNORE INTO provider_calls VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (trace_id,call["call_id"],phase,call["model"],call["started"],call["status"],
                     call["duration_ms"],call["ttft_ms"],usage.get("promptTokenCount"),
                     usage.get("candidatesTokenCount",0) if measured else None,
                     usage.get("thoughtsTokenCount",0) if measured else None,
                     usage.get("cachedContentTokenCount",0) if measured else None,
                     usage.get("totalTokenCount"),call["estimated_cost_usd"]))

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
            latencies = [r[0] for r in db.execute("SELECT duration_ms FROM runs WHERE started>=?", (cutoff,))]
            totals.update(latency_summary(latencies))
            totals["success_rate"] = (n-totals["errors"])/n if n else None
            totals["error_rate"] = totals["errors"]/n if n else None
            totals["requests_per_minute"] = db.execute("SELECT COUNT(*) FROM runs WHERE started>=?", (time.time()-60,)).fetchone()[0]
            totals["total_requests"] = db.execute("SELECT value FROM request_counters WHERE name='total'").fetchone()[0]
            failures = db.execute("""SELECT * FROM runs WHERE started >= ? AND status>=400
                ORDER BY started DESC LIMIT 25""", (cutoff,)).fetchall()
            live = {name: {"mean": None, "n": 0} for name in LIVE}
            for name, avg, n in db.execute("""SELECT m.name, AVG(m.value), COUNT(*)
                    FROM runs r JOIN query_metrics m USING(trace_id)
                    WHERE r.started >= ? AND r.operation='query' GROUP BY m.name""", (cutoff,)):
                live[name] = {"mean": avg, "n": n}
            query_count = db.execute("SELECT COUNT(*) FROM runs WHERE operation='query' AND started>=?",
                                     (cutoff,)).fetchone()[0]
            successful_queries = db.execute("SELECT COUNT(*) FROM runs WHERE operation='query' AND status<400 AND started>=?", (cutoff,)).fetchone()[0]
            totals["abstention_rate"] = totals["insufficient"]/successful_queries if successful_queries else None
            eligible = db.execute("""SELECT COUNT(*) FROM runs r JOIN query_metrics m USING(trace_id)
                WHERE r.started>=? AND r.operation='query' AND m.name='retrieved_count'""", (cutoff,)).fetchone()[0]
            placeholders_guards = ",".join("?" for _ in GUARD_REASONS)
            triggered = db.execute(f"SELECT COUNT(*) FROM runs WHERE operation='query' AND started>=? AND reason IN ({placeholders_guards})",
                                   (cutoff,*sorted(GUARD_REASONS))).fetchone()[0]
            totals["guardrail_trigger_rate"] = triggered/eligible if eligible else None
            totals["guardrail_eligible"] = eligible
            totals["guardrail_triggered"] = triggered
            calls = [dict(c) for c in db.execute("""SELECT p.* FROM provider_calls p
                JOIN runs r USING(trace_id) WHERE r.started>=? AND r.operation='query' ORDER BY p.started""", (cutoff,))]
            usage = {phase:usage_summary([c for c in calls if c["phase"]==phase]) for phase in ("request","evaluation")}
            by_query = {}
            for c in calls:
                by_query.setdefault(c["trace_id"],[]).append(c)
            active = {r[0] for r in db.execute("SELECT trace_id FROM query_evals WHERE status IN ('pending','running','interrupted')")}
            priced_queries = [sum(c["estimated_cost_usd"] for c in cs) for tid,cs in by_query.items()
                              if tid not in active and all(c["estimated_cost_usd"] is not None for c in cs)]
            usage["cost_per_query"] = {"mean_usd":sum(priced_queries)/len(priced_queries) if priced_queries else None,
                                       "n":len(priced_queries)}
            first_calls = {}
            for c in calls:
                if c["phase"]=="request":
                    first_calls.setdefault(c["trace_id"],c)
            totals["provider_ttft"] = latency_summary([c["ttft_ms"] for c in first_calls.values() if c["ttft_ms"] is not None])
            stage_values = {}
            for name,duration in db.execute("SELECT s.name,s.duration_ms FROM spans s JOIN runs r USING(trace_id) WHERE r.started>=?", (cutoff,)):
                stage_values.setdefault(name,[]).append(duration)
            stages = {name:latency_summary(values) for name,values in stage_values.items()}
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
            guard_counts = dict(db.execute(f"""SELECT reason,COUNT(*) FROM runs WHERE started>=?
                AND operation='query' AND (outcome='insufficient_evidence' OR reason IN ({placeholders_guards})) GROUP BY reason""",
                (cutoff,*sorted(GUARD_REASONS))))
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
                "guardrails": guard_counts, "timeline": timeline, "usage":usage, "stages":stages}

    def prometheus(self):
        registry = CollectorRegistry()
        registry.register(MetricsCollector(self.overview()))
        return generate_latest(registry)


class MetricsCollector:
    def __init__(self, overview):
        self.overview = overview

    def collect(self):
        m = self.overview
        total = CounterMetricFamily("documind_requests", "Requests counted since telemetry initialization")
        total.add_metric([],m["total_requests"])
        yield total
        for key in ("requests", "errors", "insufficient", "rag_queries"):
            gauge = GaugeMetricFamily("documind_"+key+"_24h", "Rolling 24-hour "+key)
            gauge.add_metric([], m[key])
            yield gauge
        for key in ("mean_ms","p50_ms","p95_ms","p99_ms"):
            latency = GaugeMetricFamily("documind_request_"+key.replace("_ms","")+"_seconds_24h", "Rolling 24-hour latency")
            if m[key] is not None:
                latency.add_metric([],m[key]/1000)
            yield latency
        for key in ("success_rate","error_rate","requests_per_minute","abstention_rate","guardrail_trigger_rate"):
            metric = GaugeMetricFamily("documind_"+key, "Live "+key)
            if m[key] is not None:
                metric.add_metric([],m[key])
            yield metric
        stages = GaugeMetricFamily("documind_stage_p95_seconds_24h","Stage p95",labels=["stage"])
        for name,value in m["stages"].items():
            stages.add_metric([name],value["p95_ms"]/1000)
        yield stages
        usage = GaugeMetricFamily("documind_provider_usage_24h","Observed tokens and priced call costs",labels=["phase","measurement"])
        for phase in ("request","evaluation"):
            for key in ("input_tokens","output_tokens","thinking_tokens","cached_tokens","total_tokens","estimated_cost_usd"):
                if m["usage"][phase][key] is not None:
                    usage.add_metric([phase,key],m["usage"][phase][key])
        yield usage
        ttft = GaugeMetricFamily("documind_provider_ttft_p95_seconds_24h","First generation call first-text p95")
        if m["provider_ttft"]["n"]:
            ttft.add_metric([],m["provider_ttft"]["p95_ms"]/1000)
        yield ttft
        cost = GaugeMetricFamily("documind_estimated_cost_per_query_usd_24h","Mean paid list price for fully priced settled queries")
        if m["usage"]["cost_per_query"]["n"]:
            cost.add_metric([],m["usage"]["cost_per_query"]["mean_usd"])
        yield cost
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
