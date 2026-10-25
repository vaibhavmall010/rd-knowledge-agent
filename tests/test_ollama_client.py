"""Runs the agent against a fake Ollama server to check the HTTP request and response format."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from rdagent.agent import RDAgent
from rdagent.llm import OllamaClient


class FakeOllama(BaseHTTPRequestHandler):
    requests: list[dict] = []

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeOllama.requests.append(body)
        if self.path != "/api/chat":
            self.send_response(404)
            self.end_headers()
            return
        if not any(m["role"] == "tool" for m in body["messages"]):
            message = {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "check_spec",
                            "arguments": {"product": "OV-600S", "parameter": "hinge_cycles"},
                        }
                    }
                ],
            }
        else:
            message = {
                "role": "assistant",
                "content": "No. The OV-600S reached 26,500 of 30,000 cycles [TR-2026-061].",
            }
        payload = json.dumps({"model": body["model"], "message": message, "done": True}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


@pytest.fixture()
def fake_ollama():
    FakeOllama.requests = []
    server = HTTPServer(("127.0.0.1", 0), FakeOllama)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def test_agent_round_trip_through_ollama_api(kb, fake_ollama):
    agent = RDAgent(OllamaClient(model="llama3.1", host=fake_ollama), kb)
    result = agent.run("Does the OV-600S hinge pass?")

    assert result.tools_used == ["check_spec"]
    assert result.citations == ["TR-2026-061"] and result.grounded
    first, second = FakeOllama.requests
    assert first["model"] == "llama3.1" and first["stream"] is False
    assert first["options"]["temperature"] == 0.0
    assert {t["function"]["name"] for t in first["tools"]} >= {"check_spec", "search_documents"}
    tool_message = second["messages"][-1]
    assert tool_message["role"] == "tool" and "26500" in tool_message["content"]


def test_unreachable_ollama_gives_a_clear_error():
    client = OllamaClient(host="http://127.0.0.1:9", timeout=2)
    with pytest.raises(RuntimeError, match="ollama serve"):
        client.chat([{"role": "user", "content": "hi"}])
