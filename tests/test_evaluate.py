from rdagent.agent import AgentResult, Step
from rdagent.evaluate import Question, fact_present, load_questions, run_tool_layer, score, summarize


def test_eval_set_is_well_formed(kb, eval_file):
    questions = load_questions(eval_file)
    assert len(questions) == 30
    assert sum(not q.answerable for q in questions) == 4
    known_tools = {
        "search_documents",
        "get_document",
        "query_test_results",
        "check_spec",
        "compliance_summary",
        "list_change_requests",
    }
    for q in questions:
        for group in q.expected_tools:
            assert set(group) <= known_tools, q.id
        for group in q.must_cite:
            assert set(group) <= set(kb.documents), q.id


def test_every_question_is_answerable_from_the_tools(kb, eval_file):
    """Regression guard: data, tools and eval set must stay consistent."""
    report = run_tool_layer(kb, load_questions(eval_file))
    failing = [r["id"] for r in report["rows"] if not r["ok"]]
    assert failing == []


def test_fact_matching_rules():
    assert fact_present("26500", "fractured at 26,500 cycles")
    assert fact_present(["5 to 7", "5-7"], "a reduction of 5–7 K")
    assert not fact_present("0.1", "costs 0.12 EUR")
    assert fact_present("0.1", "misses by 0.1 min")
    assert not fact_present("no", "it is not known")
    assert fact_present(["no", "none"], "No, all variants pass")


def test_score_and_summary():
    q = Question(
        "x", "?", "compliance", expected_tools=[["check_spec"]], facts=["51.2"], must_cite=[["TR-2026-047"]]
    )
    r = AgentResult(
        "?",
        "51.2 dB(A) [TR-2026-047]",
        steps=[Step("check_spec", {}, [], None, 0.0)],
        citations=["TR-2026-047"],
        latency_s=1.0,
    )
    row = score(q, r, with_tools=True)
    assert row["correct"] and row["tool_match"] and row["citation_recall"] == 1.0
    refusal_q = Question("y", "?", "unanswerable", answerable=False)
    refused = score(refusal_q, AgentResult("?", "not found", refused=True, latency_s=2.0), True)
    summary = summarize([row, refused])
    assert summary["accuracy"] == 1.0 and summary["refusal_accuracy_unanswerable"] == 1.0
