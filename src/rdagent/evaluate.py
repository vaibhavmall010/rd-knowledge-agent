"""Evaluation harness.

Three modes:
  tools     Model-free check that the tools and the data can answer every question (runs in CI).
  agent     Runs the tool-calling agent on every question through Ollama.
  baseline  Runs single-shot RAG (retrieve top 5, answer once) through Ollama.
  compare   Runs baseline and agent and writes a side-by-side report.

Usage:  python -m rdagent.evaluate --mode compare --model llama3.1
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml

from rdagent.agent import AgentResult, BaselineRAG, RDAgent
from rdagent.config import get_settings
from rdagent.llm import OllamaClient
from rdagent.tools import KnowledgeBase, ToolExecutor

Fact = str | list[str]


@dataclass
class Question:
    id: str
    question: str
    category: str
    expected_tools: list[list[str]] = field(default_factory=list)
    oracle: list[dict] = field(default_factory=list)
    facts: list[Fact] = field(default_factory=list)
    oracle_facts: list[Fact] | None = None
    must_cite: list[list[str]] = field(default_factory=list)
    answerable: bool = True


def load_questions(path: Path) -> list[Question]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["questions"]
    questions = [Question(**q) for q in raw]
    ids = [q.id for q in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate question ids")
    return questions


# --------------------------------------------------------------------------- matching


def normalize(text: str) -> str:
    text = (text or "").lower().replace("–", "-").replace("−", "-")
    text = re.sub(r"(?<=\d)[,  ](?=\d{3}\b)", "", text)  # 26,500 -> 26500
    return re.sub(r"\s+", " ", text)


def fact_present(fact: Fact, text: str) -> bool:
    haystack = normalize(text)
    for alt in fact if isinstance(fact, list) else [fact]:
        needle = normalize(str(alt))
        if needle.isalpha() and len(needle) <= 4:
            if re.search(rf"\b{re.escape(needle)}\b", haystack):
                return True
        elif re.fullmatch(r"[\d.]+", needle):
            # numbers must not match inside longer numbers: 0.1 must not match 0.12
            if re.search(rf"(?<![\d.]){re.escape(needle)}(?![\d])", haystack):
                return True
        elif needle in haystack:
            return True
    return False


def groups_satisfied(groups: list[list[str]], ids: list[str]) -> float:
    if not groups:
        return 1.0
    have = set(ids)
    return sum(bool(have & set(g)) for g in groups) / len(groups)


# --------------------------------------------------------------------------- tool layer


def run_tool_layer(kb: KnowledgeBase, questions: list[Question]) -> dict:
    executor = ToolExecutor(kb)
    rows = []
    for q in questions:
        outputs = [executor.execute(c["tool"], c.get("args", {})) for c in q.oracle]
        payload = json.dumps(outputs, ensure_ascii=False, default=str)
        sources = [s for o in outputs for s in o.get("sources", [])]
        retrieved = [h.chunk.doc_id for h in kb.index.search(q.question, top_k=5)]
        if q.answerable:
            facts = q.oracle_facts if q.oracle_facts is not None else q.facts
            fact_cov = sum(fact_present(f, payload) for f in facts) / max(len(facts), 1)
            row = {
                "id": q.id,
                "category": q.category,
                "fact_coverage": round(fact_cov, 3),
                "source_coverage": round(groups_satisfied(q.must_cite, sources), 3),
                "bm25_recall_at_5": round(groups_satisfied(q.must_cite, retrieved), 3),
                "ok": fact_cov == 1.0 and groups_satisfied(q.must_cite, sources) == 1.0,
            }
        else:
            empty = all(o.get("error") or o.get("count") == 0 for o in outputs)
            row = {
                "id": q.id,
                "category": q.category,
                "ok": empty,
                "fact_coverage": None,
                "source_coverage": None,
                "bm25_recall_at_5": None,
            }
        rows.append(row)
    answerable = [r for r in rows if r["fact_coverage"] is not None]
    summary = {
        "questions": len(rows),
        "oracle_ok": sum(r["ok"] for r in rows),
        "fact_coverage": _mean([r["fact_coverage"] for r in answerable]),
        "source_coverage": _mean([r["source_coverage"] for r in answerable]),
        "bm25_recall_at_5": _mean([r["bm25_recall_at_5"] for r in answerable]),
    }
    return {"mode": "tools", "summary": summary, "rows": rows}


# --------------------------------------------------------------------------- model runs


def score(q: Question, r: AgentResult, with_tools: bool) -> dict:
    fact_recall = sum(fact_present(f, r.answer) for f in q.facts) / len(q.facts) if q.facts else None
    if q.answerable:
        correct = (fact_recall == 1.0) and not r.refused
    else:
        correct = r.refused
    tool_match = None
    if with_tools and q.expected_tools:
        used = set(r.tools_used)
        tool_match = all(used & set(g) for g in q.expected_tools)
    return {
        "id": q.id,
        "category": q.category,
        "answerable": q.answerable,
        "correct": correct,
        "fact_recall": fact_recall,
        "citation_recall": groups_satisfied(q.must_cite, r.citations) if q.answerable else None,
        "citation_precision": (
            (len(r.citations) - len(r.invalid_citations)) / len(r.citations) if r.citations else None
        ),
        "invalid_citations": r.invalid_citations,
        "refused": r.refused,
        "tool_match": tool_match,
        "tools_used": r.tools_used,
        "steps": len(r.steps),
        "llm_calls": r.llm_calls,
        "repaired": r.repaired,
        "latency_s": r.latency_s,
        "answer": r.answer,
    }


def run_system(system, questions: list[Question], name: str, with_tools: bool) -> dict:
    rows = []
    for i, q in enumerate(questions, 1):
        print(f"[{name}] {i}/{len(questions)} {q.id}", flush=True)
        try:
            rows.append(score(q, system.run(q.question), with_tools))
        except Exception as exc:  # keep going, record the failure
            rows.append(
                {
                    "id": q.id,
                    "category": q.category,
                    "answerable": q.answerable,
                    "correct": False,
                    "error": str(exc),
                }
            )
    return {"mode": name, "summary": summarize(rows), "rows": rows}


def _mean(values) -> float | None:
    vals = [v for v in values if v is not None]
    return round(statistics.mean(vals), 3) if vals else None


def summarize(rows: list[dict]) -> dict:
    answerable = [r for r in rows if r.get("answerable")]
    unanswerable = [r for r in rows if not r.get("answerable")]
    categories = sorted({r["category"] for r in rows})
    return {
        "questions": len(rows),
        "accuracy": _mean([float(r["correct"]) for r in rows]),
        "accuracy_answerable": _mean([float(r["correct"]) for r in answerable]),
        "refusal_accuracy_unanswerable": _mean([float(r["correct"]) for r in unanswerable]),
        "false_refusals": sum(bool(r.get("refused")) for r in answerable),
        "fact_recall": _mean([r.get("fact_recall") for r in answerable]),
        "citation_recall": _mean([r.get("citation_recall") for r in answerable]),
        "citation_precision": _mean([r.get("citation_precision") for r in rows]),
        "tool_selection_accuracy": _mean(
            [float(r["tool_match"]) for r in rows if r.get("tool_match") is not None]
        ),
        "mean_latency_s": _mean([r.get("latency_s") for r in rows]),
        "median_latency_s": (
            round(statistics.median([r["latency_s"] for r in rows if "latency_s" in r]), 3)
            if any("latency_s" in r for r in rows)
            else None
        ),
        "mean_steps": _mean([r.get("steps") for r in rows]),
        "errors": sum("error" in r for r in rows),
        "accuracy_by_category": {
            c: _mean([float(r["correct"]) for r in rows if r["category"] == c]) for c in categories
        },
    }


# --------------------------------------------------------------------------- reports


def _pct(v) -> str:
    return "n/a" if v is None else f"{100 * v:.0f}%"


def write_report(results: list[dict], out_dir: Path, model: str | None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    modes = "_".join(r["mode"] for r in results)
    (out_dir / f"eval_{modes}.json").write_text(
        json.dumps({"model": model, "created": stamp, "results": results}, indent=2, default=str),
        encoding="utf-8",
    )
    lines = [
        f"# Evaluation report: {modes}",
        "",
        f"Created {stamp}" + (f" with model `{model}`." if model else "."),
        "",
    ]
    if results[0]["mode"] == "tools":
        s = results[0]["summary"]
        lines += [
            "Model-free check of the tools and data. For each question the expert tool calls in "
            "`eval/questions.yaml` are executed and their output is searched for the expected facts "
            "and source documents.",
            "",
            "| Metric | Value |",
            "| --- | --- |",
            f"| Questions | {s['questions']} |",
            f"| Questions fully answerable from tool output | {s['oracle_ok']} / {s['questions']} |",
            f"| Fact coverage (answerable) | {_pct(s['fact_coverage'])} |",
            f"| Source coverage (answerable) | {_pct(s['source_coverage'])} |",
            f"| BM25 recall@5 of required sources, raw question | {_pct(s['bm25_recall_at_5'])} |",
            "",
            "BM25 recall is measured on the raw question with no tools. It shows how often a "
            "single-shot retriever finds the needed documents, which is the gap the agent's "
            "structured tools are meant to close.",
        ]
    else:
        keys = [
            ("accuracy", "Answer accuracy, all questions"),
            ("accuracy_answerable", "Accuracy, answerable questions"),
            ("refusal_accuracy_unanswerable", "Correct refusals, unanswerable questions"),
            ("fact_recall", "Fact recall"),
            ("citation_recall", "Citation recall"),
            ("citation_precision", "Citation precision"),
            ("tool_selection_accuracy", "Tool selection accuracy"),
        ]
        header = "| Metric | " + " | ".join(r["mode"] for r in results) + " |"
        lines += [header, "|" + " --- |" * (len(results) + 1)]
        for key, label in keys:
            lines.append(f"| {label} | " + " | ".join(_pct(r["summary"][key]) for r in results) + " |")
        for key, label in [
            ("false_refusals", "False refusals"),
            ("mean_latency_s", "Mean latency (s)"),
            ("median_latency_s", "Median latency (s)"),
            ("mean_steps", "Mean tool calls"),
            ("errors", "Run errors"),
        ]:
            lines.append(f"| {label} | " + " | ".join(str(r["summary"][key]) for r in results) + " |")
        lines += [
            "",
            "## Accuracy by category",
            "",
            "| Category | " + " | ".join(r["mode"] for r in results) + " |",
            "|" + " --- |" * (len(results) + 1),
        ]
        for cat in results[0]["summary"]["accuracy_by_category"]:
            lines.append(
                f"| {cat} | "
                + " | ".join(_pct(r["summary"]["accuracy_by_category"].get(cat)) for r in results)
                + " |"
            )
    lines += ["", "## Per question", ""]
    for r in results:
        lines += [f"### {r['mode']}", ""]
        for row in r["rows"]:
            mark = "ok" if row.get("ok", row.get("correct")) else "FAIL"
            extra = f" ({row['error']})" if row.get("error") else ""
            lines.append(f"- {row['id']} [{row['category']}]: {mark}{extra}")
        lines.append("")
    path = out_dir / f"eval_{modes}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--mode", choices=["tools", "agent", "baseline", "compare"], default="tools")
    parser.add_argument("--model", default=settings.model)
    parser.add_argument("--host", default=settings.ollama_host)
    parser.add_argument("--ids", nargs="*", help="only run these question ids")
    parser.add_argument("--out", type=Path, default=settings.reports_dir)
    args = parser.parse_args(argv)

    kb = KnowledgeBase.from_dir(settings.data_dir)
    questions = load_questions(settings.eval_file)
    if args.ids:
        questions = [q for q in questions if q.id in set(args.ids)]

    if args.mode == "tools":
        results, model = [run_tool_layer(kb, questions)], None
    else:
        llm = OllamaClient(model=args.model, host=args.host)
        model = args.model
        results = []
        if args.mode in ("baseline", "compare"):
            results.append(run_system(BaselineRAG(llm, kb), questions, "baseline", with_tools=False))
        if args.mode in ("agent", "compare"):
            agent = RDAgent(llm, kb, max_steps=settings.max_steps)
            results.append(run_system(agent, questions, "agent", with_tools=True))

    path = write_report(results, args.out, model)
    for r in results:
        print(json.dumps({"mode": r["mode"], **r["summary"]}, indent=2))
    print(f"Report written to {path}")


if __name__ == "__main__":
    main()
