from rdagent.tools import ToolExecutor, check_spec, compliance_summary, query_test_results


def test_check_spec_uses_the_latest_measurement(kb):
    result = check_spec(kb, "OV-600P", "fan_noise")
    assert result["latest_value"] == 51.2
    assert result["latest_report"] == "TR-2026-047"
    assert result["status"] == "PASS"
    assert result["margin"] == 0.8
    assert [h["value"] for h in result["history"]] == [54.1, 51.2]
    assert result["sources"] == ["TR-2026-040", "TR-2026-047", "SPEC-OV600-001"]


def test_check_spec_handles_minimum_limits(kb):
    result = check_spec(kb, "OV-600S", "hinge_cycles")
    assert result["status"] == "FAIL"
    assert result["margin"] == -3500


def test_compliance_summary_lists_failures(kb):
    result = compliance_summary(kb, "ov-600s")  # case-insensitive product
    failing = {c["parameter"] for c in result["checks"] if c["status"] == "FAIL"}
    assert failing == {"preheat_time_200c", "hinge_cycles", "descale_cycles"}
    assert compliance_summary(kb, "OV-600B")["summary"] == "OV-600B: 7 of 7 requirements pass."


def test_query_latest_only(kb):
    rows = query_test_results(kb, product="OV-600S", latest_only=True)["rows"]
    by_param = {r["parameter"]: r["value"] for r in rows}
    assert by_param["energy_fan_forced"] == 0.84
    assert by_param["cavity_temp_deviation"] == 9.1


def test_executor_returns_actionable_errors(kb):
    ex = ToolExecutor(kb)
    bad_product = ex.execute("check_spec", {"product": "OV-700", "parameter": "fan_noise"})
    assert "Known products" in bad_product["error"]
    assert "Missing required" in ex.execute("check_spec", {"product": "OV-600B"})["error"]
    assert "Unknown tool" in ex.execute("delete_everything", {})["error"]
    assert ex.execute("get_document", '{"doc_id": "ecr-0117"}')["doc_id"] == "ECR-0117"


def test_executor_drops_unknown_arguments(kb):
    out = ToolExecutor(kb).execute("list_change_requests", {"status": "Open", "colour": "red"})
    assert [c["doc_id"] for c in out["change_requests"]] == ["ECR-0123"]


def test_tool_schemas_are_complete(kb):
    for schema in ToolExecutor(kb).schemas():
        fn = schema["function"]
        assert fn["name"] and fn["description"]
        assert fn["parameters"]["type"] == "object"
