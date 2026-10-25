from rdagent.agent import REFUSAL, BaselineRAG, RDAgent, check_citations, extract_citations
from rdagent.llm import LLMResponse, ScriptedLLM, ToolCall, parse_text_tool_call


def test_agent_calls_tool_then_answers_with_valid_citation(kb):
    llm = ScriptedLLM(
        [
            LLMResponse(
                tool_calls=[ToolCall("check_spec", {"product": "OV-600P", "parameter": "fan_noise"})]
            ),
            LLMResponse(content="Yes. OV-600P measured 51.2 dB(A) against 52.0 dB(A) [TR-2026-047]."),
        ]
    )
    result = RDAgent(llm, kb).run("Does OV-600P meet fan noise?")
    assert result.tools_used == ["check_spec"]
    assert result.citations == ["TR-2026-047"]
    assert result.grounded and not result.repaired
    # the tool output was passed back to the model
    tool_msg = llm.calls[1][-1]
    assert tool_msg["role"] == "tool" and "51.2" in tool_msg["content"]


def test_agent_repairs_a_citation_it_never_retrieved(kb):
    llm = ScriptedLLM(
        [
            LLMResponse(tool_calls=[ToolCall("get_document", {"doc_id": "ECR-0117"})]),
            LLMResponse(content="It cost 18,000 EUR [ECR-0117] [TR-2026-999]."),
            LLMResponse(content="Tooling cost 18,000 EUR [ECR-0117]."),
        ]
    )
    result = RDAgent(llm, kb).run("What did ECR-0117 cost?")
    assert result.repaired
    assert result.citations == ["ECR-0117"] and not result.invalid_citations
    assert "TR-2026-999" in llm.calls[2][-1]["content"]


def test_agent_accepts_refusal_without_citations(kb):
    llm = ScriptedLLM([LLMResponse(content=f"This is {REFUSAL}.")])
    result = RDAgent(llm, kb).run("What is the EMC result?")
    assert result.refused and not result.repaired and result.llm_calls == 1


def test_agent_forces_an_answer_when_the_tool_budget_is_spent(kb):
    def policy(messages):
        if any("budget is used up" in m.get("content", "") for m in messages):
            return LLMResponse(content=f"{REFUSAL}.")
        return LLMResponse(tool_calls=[ToolCall("search_documents", {"query": "fan"})])

    result = RDAgent(ScriptedLLM(policy), kb, max_steps=2).run("Loop forever")
    assert len(result.steps) == 2 and result.refused


def test_tool_errors_are_returned_to_the_model(kb):
    llm = ScriptedLLM(
        [
            LLMResponse(tool_calls=[ToolCall("check_spec", {"product": "OV-700", "parameter": "fan_noise"})]),
            LLMResponse(content=f"{REFUSAL}."),
        ]
    )
    result = RDAgent(llm, kb).run("OV-700 noise?")
    assert result.steps[0].error and "Known products" in llm.calls[1][-1]["content"]


def test_baseline_cites_retrieved_passages(kb):
    llm = ScriptedLLM([LLMResponse(content="The spring fractured at 26,500 cycles [TR-2026-061].")])
    result = BaselineRAG(llm, kb).run("Why did the OV-600S hinge fail?")
    assert "TR-2026-061" in result.sources_seen and result.grounded


def test_citation_parsing_variants():
    text = "See [TR-2026-040, ECR-0117] and [SPEC-OV600-001 section 3.3]; not [OV-600P] or [1]."
    assert extract_citations(text) == ["TR-2026-040", "ECR-0117", "SPEC-OV600-001"]
    cited, invalid = check_citations(text, ["TR-2026-040", "SPEC-OV600-001"])
    assert invalid == ["ECR-0117"]


def test_text_tool_call_recovery():
    call = parse_text_tool_call(
        '{"name": "check_spec", "parameters": {"product": "OV-600B"}}', ["check_spec"]
    )
    assert call.name == "check_spec" and call.arguments == {"product": "OV-600B"}
    assert parse_text_tool_call('{"name": "rm_rf"}', ["check_spec"]) is None
    assert parse_text_tool_call("plain answer", ["check_spec"]) is None
