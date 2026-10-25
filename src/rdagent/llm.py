"""LLM clients: Ollama chat with native tool calling, and a scripted client for tests."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Protocol
from urllib.parse import urlparse


@dataclass
class ToolCall:
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLMClient(Protocol):
    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse: ...


JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def parse_text_tool_call(text: str, known_tools: Iterable[str]) -> ToolCall | None:
    """Some local models write the tool call as JSON in the message text instead of using the
    tool-calling field. Recover it when the JSON names a known tool."""
    match = JSON_OBJECT.search(text or "")
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    name = obj.get("name") or obj.get("function") or obj.get("tool")
    if isinstance(name, dict):
        obj, name = name, name.get("name")
    args = obj.get("arguments", obj.get("parameters", {}))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    if name in set(known_tools) and isinstance(args, dict):
        return ToolCall(name=name, arguments=args)
    return None


class OllamaClient:
    """Calls Ollama's /api/chat endpoint. Works with tool-capable models such as llama3.1 or
    qwen2.5. Temperature 0 keeps evaluation runs repeatable."""

    def __init__(
        self,
        model: str = "llama3.1",
        host: str = "http://localhost:11434",
        temperature: float = 0.0,
        timeout: float = 180.0,
        num_ctx: int = 8192,
    ) -> None:
        self.model, self.host = model, host.rstrip("/")
        self.temperature, self.timeout, self.num_ctx = temperature, timeout, num_ctx

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self.temperature, "num_ctx": self.num_ctx},
        }
        if tools:
            payload["tools"] = tools
        message = self._post("/api/chat", payload).get("message", {})
        calls = []
        for call in message.get("tool_calls") or []:
            fn = call.get("function", {})
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            calls.append(ToolCall(name=fn.get("name", ""), arguments=args))
        content = message.get("content", "") or ""
        if not calls and tools:
            recovered = parse_text_tool_call(content, [t["function"]["name"] for t in tools])
            if recovered:
                calls, content = [recovered], ""
        return LLMResponse(content=content, tool_calls=calls)

    def _post(self, path: str, payload: dict) -> dict:
        """POST JSON with the standard library only (no compiled dependencies)."""
        request = urllib.request.Request(
            f"{self.host}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        # Never send local traffic through a corporate proxy.
        local = urlparse(self.host).hostname in {"localhost", "127.0.0.1", "::1"}
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}) if local else urllib.request.ProxyHandler()
        )
        try:
            with opener.open(request, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            raise RuntimeError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
            raise RuntimeError(
                f"Cannot reach Ollama at {self.host}. Start it with 'ollama serve' and pull the "
                f"model with 'ollama pull {self.model}'."
            ) from exc


class ScriptedLLM:
    """Returns pre-defined responses in order, or computes them from the conversation.
    Used in tests so the agent loop can be checked without a model."""

    def __init__(self, script: list[LLMResponse] | Callable[[list[dict]], LLMResponse]) -> None:
        self.script = script
        self.calls: list[list[dict]] = []

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        self.calls.append([dict(m) for m in messages])
        if callable(self.script):
            return self.script(messages)
        if not self.script:
            raise RuntimeError("ScriptedLLM ran out of responses")
        return self.script.pop(0)
