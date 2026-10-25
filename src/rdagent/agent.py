"""The tool-calling agent loop, a single-shot RAG baseline, and citation checking."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

from rdagent.llm import LLMClient
from rdagent.tools import KnowledgeBase, ToolExecutor

REFUSAL = "not found in the R&D knowledge base"

SYSTEM_PROMPT = f"""You are an assistant for oven R&D engineers. You answer questions about
specifications, test reports, change requests (ECRs), lessons learned and meeting minutes.

Rules:
1. Use the tools to find facts. Never answer from memory.
2. For any question about whether a product meets a requirement, use check_spec or
   compliance_summary. Do not compare numbers yourself.
3. Cite every fact with the id of the document it came from in square brackets, for example
   [TR-2026-040]. Only cite ids that a tool returned.
4. If the tools do not return the information, say "{REFUSAL}" and stop. Do not guess.
5. Be short and specific: give values with units, dates and ids."""

BASELINE_PROMPT = f"""Answer the question using only the context passages below. Cite every fact
with the document id in square brackets, for example [TR-2026-040]. If the context does not
contain the answer, say "{REFUSAL}"."""

BRACKETS = re.compile(r"\[([^\[\]]{2,120})\]")
DOC_ID_SHAPE = re.compile(r"^(?:SPEC|TR|ECR|LL|MEET)-[A-Z0-9-]+$")


@dataclass
class Step:
    tool: str
    arguments: dict
    sources: list[str]
    error: str | None
    duration_s: float


@dataclass
class AgentResult:
    question: str
    answer: str
    steps: list[Step] = field(default_factory=list)
    sources_seen: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    invalid_citations: list[str] = field(default_factory=list)
    refused: bool = False
    repaired: bool = False
    llm_calls: int = 0
    latency_s: float = 0.0

    @property
    def tools_used(self) -> list[str]:
        return [s.tool for s in self.steps]

    @property
    def grounded(self) -> bool:
        """Answer cites at least one retrieved source and nothing it did not retrieve."""
        return bool(self.citations) and not self.invalid_citations


def extract_citations(answer: str) -> list[str]:
    ids: list[str] = []
    for group in BRACKETS.findall(answer or ""):
        for part in re.split(r"[,;]|\band\b", group):
            token = part.strip().upper().rstrip(".")
            token = re.sub(r"\s+SECTION.*$", "", token)
            if DOC_ID_SHAPE.match(token) and token not in ids:
                ids.append(token)
    return ids


def check_citations(answer: str, sources_seen: list[str]) -> tuple[list[str], list[str]]:
    """Return (citations, invalid). A citation is invalid if no tool returned that id in this
    run, which also catches ids the model invented."""
    cited = extract_citations(answer)
    seen = set(sources_seen)
    return cited, [c for c in cited if c not in seen]


def is_refusal(answer: str) -> bool:
    return REFUSAL.lower() in (answer or "").lower()


def _compact(result: dict, limit: int = 6000) -> str:
    text = json.dumps(result, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + ' ..."(truncated)"'


class RDAgent:
    def __init__(
        self,
        llm: LLMClient,
        kb: KnowledgeBase,
        max_steps: int = 6,
        repair: bool = True,
    ) -> None:
        self.llm, self.kb = llm, kb
        self.executor = ToolExecutor(kb)
        self.max_steps, self.repair = max_steps, repair

    def run(self, question: str) -> AgentResult:
        start = time.perf_counter()
        result = AgentResult(question=question, answer="")
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        schemas = self.executor.schemas()
        seen: list[str] = []
        answer = ""
        repair_used = False

        for _ in range(self.max_steps + 2):
            budget_left = len(result.steps) < self.max_steps
            response = self.llm.chat(messages, tools=schemas if budget_left else None)
            result.llm_calls += 1
            if response.tool_calls and not budget_left:
                # The model keeps asking for tools after the budget is spent: force an answer.
                messages.append(
                    {
                        "role": "user",
                        "content": "The tool budget is used up. Answer now with the evidence you have.",
                    }
                )
                response = self.llm.chat(messages, tools=None)
                result.llm_calls += 1
                response.tool_calls = []
            if response.tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": response.content or "",
                        "tool_calls": [
                            {"function": {"name": c.name, "arguments": c.arguments}}
                            for c in response.tool_calls
                        ],
                    }
                )
                for call in response.tool_calls:
                    t0 = time.perf_counter()
                    output = self.executor.execute(call.name, call.arguments)
                    sources = list(output.get("sources", []))
                    seen.extend(s for s in sources if s not in seen)
                    result.steps.append(
                        Step(
                            call.name,
                            dict(call.arguments),
                            sources,
                            output.get("error"),
                            round(time.perf_counter() - t0, 4),
                        )
                    )
                    messages.append({"role": "tool", "tool_name": call.name, "content": _compact(output)})
                continue

            answer = response.content.strip()
            cited, invalid = check_citations(answer, seen)
            needs_repair = invalid or (not cited and not is_refusal(answer))
            if self.repair and needs_repair and not repair_used:
                repair_used = True
                problem = (
                    f"These ids were not returned by any tool: {', '.join(invalid)}."
                    if invalid
                    else "The answer has no citations."
                )
                messages.append({"role": "assistant", "content": answer})
                messages.append(
                    {
                        "role": "user",
                        "content": f"{problem} Revise the answer. Cite only ids returned by the "
                        f"tools, or call a tool if you need more evidence. If the evidence is "
                        f'missing, say "{REFUSAL}".',
                    }
                )
                continue
            break

        result.answer = answer
        result.sources_seen = seen
        result.citations, result.invalid_citations = check_citations(answer, seen)
        result.refused = is_refusal(answer)
        result.repaired = repair_used
        result.latency_s = round(time.perf_counter() - start, 3)
        return result


class BaselineRAG:
    """Single-shot retrieval-augmented generation without tools, used as the comparison point
    in the evaluation: retrieve the top passages for the question, then answer once."""

    def __init__(self, llm: LLMClient, kb: KnowledgeBase, top_k: int = 5) -> None:
        self.llm, self.kb, self.top_k = llm, kb, top_k

    def run(self, question: str) -> AgentResult:
        start = time.perf_counter()
        hits = self.kb.index.search(question, top_k=self.top_k)
        context = "\n\n".join(
            f"[{h.chunk.doc_id}] {h.chunk.title} / {h.chunk.section}\n{h.chunk.text}" for h in hits
        )
        seen = list(dict.fromkeys(h.chunk.doc_id for h in hits))
        response = self.llm.chat(
            [
                {"role": "system", "content": BASELINE_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ]
        )
        answer = response.content.strip()
        cited, invalid = check_citations(answer, seen)
        return AgentResult(
            question=question,
            answer=answer,
            sources_seen=seen,
            citations=cited,
            invalid_citations=invalid,
            refused=is_refusal(answer),
            llm_calls=1,
            latency_s=round(time.perf_counter() - start, 3),
        )
