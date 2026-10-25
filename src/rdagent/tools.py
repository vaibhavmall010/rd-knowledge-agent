"""The agent's tools: document search, structured test-data queries and spec checks.

Every tool returns a JSON-serialisable dict with a "sources" list of document ids. The agent may
only cite ids that a tool has returned, which is what makes its answers checkable.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rdagent.corpus import Document, chunk_document, load_documents
from rdagent.retrieval import BM25Index

DOC_TYPES = ["spec", "test_report", "change_request", "lessons_learned", "meeting_minutes"]


class ToolError(ValueError):
    """Raised for bad tool arguments. The message is returned to the model so it can retry."""


def _number(text: str) -> float | None:
    text = (text or "").strip()
    return float(text) if text else None


def read_results(path: Path) -> list[dict]:
    """Measurements as dicts, sorted by test date (oldest first)."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = [{**r, "value": float(r["value"]), "note": r.get("note") or ""} for r in csv.DictReader(f)]
    return sorted(rows, key=lambda r: r["test_date"])


def read_limits(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return [{**r, "min": _number(r["min"]), "max": _number(r["max"])} for r in csv.DictReader(f)]


@dataclass
class KnowledgeBase:
    documents: dict[str, Document]
    index: BM25Index
    results: list[dict]
    limits: list[dict]

    @classmethod
    def from_dir(cls, data_dir: Path) -> KnowledgeBase:
        data_dir = Path(data_dir)
        docs = load_documents(data_dir / "documents")
        chunks = [c for d in docs for c in chunk_document(d)]
        results = read_results(data_dir / "test_results.csv")
        limits = read_limits(data_dir / "spec_limits.csv")
        kb = cls({d.doc_id: d for d in docs}, BM25Index(chunks), results, limits)
        kb._validate()
        return kb

    def _validate(self) -> None:
        unknown = {r["report_id"] for r in self.results} - set(self.documents)
        unknown |= {r["spec_doc"] for r in self.limits} - set(self.documents)
        if unknown:
            raise ValueError(f"tables reference unknown documents: {sorted(unknown)}")

    @property
    def products(self) -> list[str]:
        return sorted({r["product"] for r in self.limits} | {r["product"] for r in self.results})

    @property
    def parameters(self) -> list[str]:
        return sorted({r["parameter"] for r in self.limits} | {r["parameter"] for r in self.results})


# --------------------------------------------------------------------------- helpers


def _norm_product(kb: KnowledgeBase, product: str | None) -> str | None:
    if product is None or product == "":
        return None
    match = {p.upper(): p for p in kb.products}.get(str(product).strip().upper())
    if match is None:
        raise ToolError(f"Unknown product '{product}'. Known products: {', '.join(kb.products)}.")
    return match


def _norm_parameter(kb: KnowledgeBase, parameter: str | None) -> str | None:
    if parameter is None or parameter == "":
        return None
    key = str(parameter).strip().lower().replace(" ", "_")
    if key not in kb.parameters:
        raise ToolError(f"Unknown parameter '{parameter}'. Known parameters: {', '.join(kb.parameters)}.")
    return key


def _clean(value: Any) -> Any:
    """Make YAML metadata JSON-friendly (dates become strings)."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return str(value)


def _pick(row: dict, keys: tuple[str, ...]) -> dict:
    return {k: row[k] for k in keys}


RESULT_COLUMNS = ("product", "parameter", "value", "unit", "test_date", "report_id", "note")


# --------------------------------------------------------------------------- tools


def search_documents(
    kb: KnowledgeBase,
    query: str,
    doc_type: str | None = None,
    product: str | None = None,
    top_k: int = 5,
) -> dict:
    if not query or not str(query).strip():
        raise ToolError("query must not be empty")
    if doc_type and doc_type not in DOC_TYPES:
        raise ToolError(f"Unknown doc_type '{doc_type}'. Use one of: {', '.join(DOC_TYPES)}.")
    product = _norm_product(kb, product) if product else None
    top_k = max(1, min(int(top_k), 8))
    hits = kb.index.search(str(query), top_k=top_k, doc_type=doc_type, product=product)
    passages = [
        {
            "doc_id": h.chunk.doc_id,
            "doc_type": h.chunk.doc_type,
            "title": h.chunk.title,
            "section": h.chunk.section,
            "score": round(h.score, 2),
            "text": h.chunk.text,
        }
        for h in hits
    ]
    return {
        "passages": passages,
        "sources": list(dict.fromkeys(p["doc_id"] for p in passages)),
        "note": "" if passages else "No matching passages found.",
    }


def get_document(kb: KnowledgeBase, doc_id: str, max_chars: int = 4000) -> dict:
    doc = kb.documents.get(str(doc_id).strip().upper())
    if doc is None:
        raise ToolError(f"Unknown document id '{doc_id}'.")
    return {
        "doc_id": doc.doc_id,
        "doc_type": doc.doc_type,
        "title": doc.title,
        "date": doc.date,
        "products": list(doc.products),
        "metadata": {k: _clean(v) for k, v in doc.metadata.items()},
        "text": doc.body[:max_chars],
        "truncated": len(doc.body) > max_chars,
        "sources": [doc.doc_id],
    }


def query_test_results(
    kb: KnowledgeBase,
    product: str | None = None,
    parameter: str | None = None,
    report_id: str | None = None,
    latest_only: bool = False,
) -> dict:
    product = _norm_product(kb, product)
    parameter = _norm_parameter(kb, parameter)
    report = str(report_id).strip().upper() if report_id else None
    rows = [
        r
        for r in kb.results
        if (not product or r["product"] == product)
        and (not parameter or r["parameter"] == parameter)
        and (not report or r["report_id"] == report)
    ]
    if latest_only:
        last = {(r["product"], r["parameter"]): id(r) for r in rows}
        rows = [r for r in rows if last[(r["product"], r["parameter"])] == id(r)]
    rows = [_pick(r, RESULT_COLUMNS) for r in rows]
    return {
        "rows": rows,
        "count": len(rows),
        "sources": list(dict.fromkeys(r["report_id"] for r in rows)),
        "note": "" if rows else "No measurements match these filters.",
    }


def _limit_row(kb: KnowledgeBase, product: str, parameter: str) -> dict:
    for row in kb.limits:
        if row["product"] == product and row["parameter"] == parameter:
            return row
    raise ToolError(f"No specification limit for {parameter} on {product}.")


def _evaluate(limit: dict, value: float) -> tuple[str, float]:
    lo, hi = limit["min"], limit["max"]
    if hi is not None:
        return ("PASS" if value <= hi else "FAIL"), round(hi - value, 3)
    return ("PASS" if value >= lo else "FAIL"), round(value - lo, 3)


def check_spec(kb: KnowledgeBase, product: str, parameter: str) -> dict:
    """Compare the latest measurement with the specification limit and show the history."""
    product = _norm_product(kb, product)
    parameter = _norm_parameter(kb, parameter)
    if not product or not parameter:
        raise ToolError("check_spec needs both product and parameter")
    limit = _limit_row(kb, product, parameter)
    history = [r for r in kb.results if r["product"] == product and r["parameter"] == parameter]
    base = {
        "product": product,
        "parameter": parameter,
        "unit": limit["unit"],
        "limit_min": limit["min"],
        "limit_max": limit["max"],
        "spec": f"{limit['spec_doc']} section {limit['section']}",
    }
    if not history:
        return {**base, "status": "NOT TESTED", "sources": [limit["spec_doc"]]}
    latest = history[-1]
    status, margin = _evaluate(limit, latest["value"])
    reports = list(dict.fromkeys(r["report_id"] for r in history))
    return {
        **base,
        "latest_value": latest["value"],
        "latest_report": latest["report_id"],
        "latest_date": latest["test_date"],
        "status": status,
        "margin": margin,
        "margin_note": "positive margin = inside the limit",
        "history": [_pick(r, ("test_date", "value", "report_id", "note")) for r in history],
        "sources": list(dict.fromkeys([*reports, limit["spec_doc"]])),
    }


def compliance_summary(kb: KnowledgeBase, product: str) -> dict:
    """Run check_spec for every specified parameter of one product."""
    product = _norm_product(kb, product)
    if not product:
        raise ToolError("compliance_summary needs a product")
    checks = [check_spec(kb, product, r["parameter"]) for r in kb.limits if r["product"] == product]
    rows = [
        {
            k: c.get(k)
            for k in (
                "parameter",
                "latest_value",
                "unit",
                "limit_min",
                "limit_max",
                "status",
                "margin",
                "latest_report",
            )
        }
        for c in checks
    ]
    passed = sum(c["status"] == "PASS" for c in checks)
    failing = [c["parameter"] for c in checks if c["status"] != "PASS"]
    sources = list(dict.fromkeys(s for c in checks for s in c["sources"]))
    summary = f"{product}: {passed} of {len(checks)} requirements pass."
    if failing:
        summary += f" Not passing: {', '.join(failing)}."
    return {"product": product, "summary": summary, "checks": rows, "sources": sources}


def list_change_requests(kb: KnowledgeBase, product: str | None = None, status: str | None = None) -> dict:
    product = _norm_product(kb, product)
    items = []
    for doc in kb.documents.values():
        if doc.doc_type != "change_request":
            continue
        if product and product not in doc.products:
            continue
        doc_status = str(doc.metadata.get("status", "unknown"))
        if status and doc_status.lower() != str(status).strip().lower():
            continue
        items.append(
            {
                "doc_id": doc.doc_id,
                "title": doc.title,
                "status": doc_status,
                "products": list(doc.products),
                "date": doc.date,
                "owner_team": doc.metadata.get("owner_team"),
            }
        )
    items.sort(key=lambda i: i["doc_id"])
    return {
        "change_requests": items,
        "count": len(items),
        "sources": [i["doc_id"] for i in items],
        "note": "" if items else "No change requests match these filters.",
    }


# --------------------------------------------------------------------------- registry


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict
    func: Callable[..., dict]

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def build_tools(kb: KnowledgeBase) -> dict[str, Tool]:
    product = {"type": "string", "enum": kb.products, "description": "Product variant"}
    parameter = {
        "type": "string",
        "enum": kb.parameters,
        "description": "Measured parameter name",
    }
    tools = [
        Tool(
            "search_documents",
            "Full-text search over specifications, test reports, change requests, lessons "
            "learned and meeting minutes. Returns passages with their document id.",
            {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Keywords to search for"},
                    "doc_type": {"type": "string", "enum": DOC_TYPES},
                    "product": product,
                    "top_k": {"type": "integer", "description": "Number of passages, 1 to 8"},
                },
                "required": ["query"],
            },
            search_documents,
        ),
        Tool(
            "get_document",
            "Read one document in full by its id, for example TR-2026-040 or ECR-0117.",
            {
                "type": "object",
                "properties": {"doc_id": {"type": "string"}},
                "required": ["doc_id"],
            },
            get_document,
        ),
        Tool(
            "query_test_results",
            "Look up measured values from the test database, filtered by product, parameter "
            "or report id. Use latest_only to get only the most recent value per parameter.",
            {
                "type": "object",
                "properties": {
                    "product": product,
                    "parameter": parameter,
                    "report_id": {"type": "string"},
                    "latest_only": {"type": "boolean"},
                },
            },
            query_test_results,
        ),
        Tool(
            "check_spec",
            "Check whether the latest measurement of a parameter meets its specification "
            "limit. Returns PASS or FAIL, the margin and the measurement history.",
            {
                "type": "object",
                "properties": {"product": product, "parameter": parameter},
                "required": ["product", "parameter"],
            },
            check_spec,
        ),
        Tool(
            "compliance_summary",
            "Check every specified requirement of one product and list which pass or fail.",
            {
                "type": "object",
                "properties": {"product": product},
                "required": ["product"],
            },
            compliance_summary,
        ),
        Tool(
            "list_change_requests",
            "List engineering change requests (ECRs) with status, filtered by product or status.",
            {
                "type": "object",
                "properties": {
                    "product": product,
                    "status": {
                        "type": "string",
                        "enum": ["Open", "In review", "Approved", "Implemented"],
                    },
                },
            },
            list_change_requests,
        ),
    ]
    return {t.name: t for t in tools}


class ToolExecutor:
    """Validates arguments, runs a tool and turns errors into messages the model can act on."""

    def __init__(self, kb: KnowledgeBase) -> None:
        self.kb = kb
        self.tools = build_tools(kb)

    def schemas(self) -> list[dict]:
        return [t.schema() for t in self.tools.values()]

    def execute(self, name: str, arguments: dict | str | None) -> dict:
        tool = self.tools.get(name)
        if tool is None:
            return {"error": f"Unknown tool '{name}'. Available: {', '.join(self.tools)}.", "sources": []}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError:
                return {"error": "Arguments are not valid JSON.", "sources": []}
        arguments = arguments or {}
        allowed = set(tool.parameters.get("properties", {}))
        clean = {k: v for k, v in arguments.items() if k in allowed and v not in (None, "")}
        missing = [k for k in tool.parameters.get("required", []) if k not in clean]
        if missing:
            return {"error": f"Missing required argument(s): {', '.join(missing)}.", "sources": []}
        try:
            return tool.func(self.kb, **clean)
        except ToolError as exc:
            return {"error": str(exc), "sources": []}
