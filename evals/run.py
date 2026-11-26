"""Run paired evaluations with real provider calls. Use python -m evals.run."""
import argparse
import hashlib
import json
import os
import re
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from provider import MODEL, ProviderError
from evals import baseline
from ragas_eval import score_answer
from rag_metrics import LIVE, OFFLINE, retrieval_scores, summarize_metrics

ROOT = Path(__file__).resolve().parent
CASES = json.loads((ROOT / "cases.json").read_text())
ABSENCE = re.compile(r"\b(?:not (?:specified|provided|given|listed|stated|mentioned|included|indicated|available)|(?:doesn.t|does not|do not|don.t) (?:contain|include|provide|specify|mention|state|have|list|indicate|give|say)|(?:cannot|can.t|unable to) (?:tell|determine|confirm|find|see)|no (?:information|details|data)|insufficient (?:information|evidence))\b", re.I)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def norm(s):
    return " ".join(re.findall(r"\w+", s.casefold().replace("kilograms", "kg")))


def score(case, result):
    raw = result["response"]
    prose = re.sub(r"\b(?:specifically|explicitly)\s+", "", raw, flags=re.I)
    prose = re.sub(r"\b(is|are|was|were)n['’]t\b", r"\1 not", prose, flags=re.I)
    abstained = (result["status"] == "insufficient_evidence" if "status" in result
                 else bool(ABSENCE.search(prose) or re.search(r"\bno mention of\b", prose, re.I)))
    pages = {c["page"] for c in result.get("citations", [])}
    if "citations" not in result:
        pages.update(int(p) for p in re.findall(r"\b(?:page|p\.)\s*(\d+)\b", raw, re.I))
    expected_absent = case["kind"] == "unanswerable"
    correct = abstained if expected_absent else (
        not abstained and all(f" {norm(f)} " in f" {norm(raw)} " for f in case["facts"]))
    return {"correct": correct, "cited_pages": sorted(pages),
            "citation_correct": None if expected_absent else pages == {case["page"]},
            "abstained": abstained, "decision_correct": abstained == expected_absent}


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(f.name, path)


def metric(values):
    return {"passed": sum(values), "total": len(values)}


def summarize(report):
    labels = {case["id"]: case for case in CASES}
    cases = report["cases"]
    for row in cases:
        row.update(score(labels[row["id"]], row))
        gold = labels[row["id"]]["reference_context_ids"]
        row["retrieval_scores"] = retrieval_scores(row.get("retrieved_context_ids"), gold)
    report["rag_scores"] = {
        **summarize_metrics([r["retrieval_scores"] for r in cases], OFFLINE),
        **summarize_metrics([r.get("quality", {}) for r in cases], LIVE),
    }
    report["evaluation_framework"] = "ragas==0.3.7"
    report["scores"] = {
        "answer_correctness": metric([r["correct"] for r in cases]),
        "citation_accuracy": metric([r["citation_correct"] for r in cases if r["citation_correct"] is not None]),
        "abstention_recall": metric([r["abstained"] for r in cases if r["kind"] == "unanswerable"]),
        "evidence_decisions": metric([r["decision_correct"] for r in cases]),
    }


def run(mode, output):
    load_dotenv()
    key = os.environ["GOOGLE_API_KEY"]
    manifest = {d: digest((ROOT / "fixtures" / f"{d}.pdf").read_bytes())
                for d in sorted({c["document"] for c in CASES})}
    report = {"mode": mode, "model": MODEL, "suite_sha256": digest((ROOT / "cases.json").read_bytes()),
              "pdf_sha256": manifest, "run_id": uuid.uuid4().hex, "cases": []}
    cache = {}
    checkpoint = output / f".{mode}-{MODEL}.partial.json"
    if checkpoint.exists():
        prior = json.loads(checkpoint.read_text(encoding="utf-8"))
        if all(prior[k] == report[k] for k in ("mode", "model", "suite_sha256", "pdf_sha256")):
            report = prior
    finished = {r["id"] for r in report["cases"]}
    if mode == "guarded":
        from rag import prepare_document, ask
    for case in CASES:
        if case["id"] in finished:
            continue
        pdf = ROOT / "fixtures" / (case["document"] + ".pdf")
        trace = uuid.uuid4().hex
        for attempt in range(4):
            try:
                if mode == "baseline":
                    result = baseline.answer(pdf, case["question"], key)
                else:
                    if pdf not in cache:
                        cache[pdf] = prepare_document(pdf)
                    observed = {}
                    result = ask(pdf, cache[pdf], case["question"], key, [], measurements=observed)
                    result["retrieved_context_ids"] = observed.pop("retrieved_context_ids")
                    result["quality"] = observed
                break
            except ProviderError as exc:
                if str(exc) not in {"provider_http_429", "provider_http_503", "provider_unreachable"} or attempt == 3:
                    raise
                time.sleep(20 * (attempt + 1))
        report["cases"].append({"id": case["id"], "kind": case["kind"], "trace_id": trace,
                                **result, **score(case, result)})
        atomic_json(checkpoint, report)
        print(mode, case["id"], flush=True)
    summarize(report)
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    atomic_json(output / f"{mode}.json", report)
    checkpoint.unlink(missing_ok=True)
    print(report["scores"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["baseline", "guarded", "both"])
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--score-recorded", action="store_true", help="Run Ragas on the recorded synthetic answers and re-score ID retrieval.")
    args = parser.parse_args()
    for mode in (["baseline", "guarded"] if args.mode == "both" else [args.mode]):
        if args.score_recorded:
            import asyncio
            from rag import prepare_document
            load_dotenv()
            report = json.loads((args.output / f"{mode}.json").read_text(encoding="utf-8"))
            by_id = {c["id"]: c for c in CASES}
            docs = {}
            for row in report["cases"]:
                case = by_id[row["id"]]
                if score(case, row)["abstained"] or case["kind"] == "unanswerable":
                    continue
                quality = row.setdefault("quality", {})
                if "ragas_answer_relevancy" in quality and ("ragas_faithfulness" in quality or case["kind"] == "figure"):
                    continue
                pdf = ROOT / "fixtures" / (case["document"]+".pdf")
                if pdf not in docs:
                    docs[pdf] = prepare_document(pdf)[0]
                retrieved = set(row.get("retrieved_context_ids", []))
                sources = [s for s in docs[pdf] if s["id"] in retrieved]
                statuses = row.setdefault("metric_status", {})
                def save(name, value, status):
                    statuses[name] = status
                    if value is not None:
                        quality[name] = value
                asyncio.run(score_answer(case["question"], row["response"], sources,
                                         case["kind"] == "figure", os.environ["GOOGLE_API_KEY"], save,
                                         skip=set(quality)))
                print(mode, row["id"], statuses, flush=True)
                atomic_json(args.output / f"{mode}.json", report)
            summarize(report)
            report["ragas_scored_at"] = datetime.now(timezone.utc).isoformat()
            report["ragas_configuration"] = {"version":"0.3.7", "judge_model":MODEL,
                                             "embeddings":"BAAI/bge-small-en-v1.5", "relevancy_strictness":3}
            atomic_json(args.output / f"{mode}.json", report)
        else:
            run(mode, args.output)
