import pytest

from rdagent.corpus import Document, chunk_document, parse_front_matter
from rdagent.retrieval import tokenize


def test_all_documents_load_with_metadata(kb):
    assert len(kb.documents) == 17
    types = {d.doc_type for d in kb.documents.values()}
    assert types == {"spec", "test_report", "change_request", "lessons_learned", "meeting_minutes"}
    assert kb.documents["ECR-0123"].metadata["status"] == "Open"


def test_front_matter_is_required():
    with pytest.raises(ValueError):
        parse_front_matter("# no front matter")


def test_chunks_follow_sections_and_keep_ids():
    doc = Document(
        "TR-1",
        "test_report",
        "T",
        ("OV-600B",),
        "2026-01-01",
        "# Title\n\nIntro text.\n\n## Results\n\nValue 1.\n\n## Findings\n\nOk.",
    )
    chunks = chunk_document(doc)
    assert [c.section for c in chunks] == ["Overview", "Results", "Findings"]
    assert all(c.doc_id == "TR-1" and c.products == ("OV-600B",) for c in chunks)


def test_long_sections_are_split_on_paragraphs():
    body = "## Big\n\n" + "\n\n".join("word " * 60 for _ in range(6))
    chunks = chunk_document(Document("X-1", "spec", "T", (), "d", body), max_chars=700)
    assert len(chunks) > 1
    assert all(len(c.text) <= 700 for c in chunks)


def test_tokenize_keeps_identifiers_and_their_parts():
    tokens = tokenize("Retest of OV-600P per TR-2026-047 at 0.85 kWh; decisions")
    assert {"ov-600p", "600p", "tr-2026-047", "0.85", "decision"} <= set(tokens)
    assert "of" not in tokens


def test_search_ranks_the_relevant_report_first(kb):
    hits = kb.index.search("hinge spring fracture cycles", top_k=3)
    assert hits[0].chunk.doc_id == "TR-2026-061"


def test_search_filters_by_type_and_product(kb):
    hits = kb.index.search("fan noise", doc_type="change_request", product="OV-600P")
    assert hits and all(h.chunk.doc_type == "change_request" for h in hits)
    assert all("OV-600P" in h.chunk.products for h in hits)
