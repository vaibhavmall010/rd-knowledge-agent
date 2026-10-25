"""Streamlit front end: chat with the agent and inspect every tool call and source."""

from __future__ import annotations

import json

import streamlit as st

from rdagent.agent import BaselineRAG, RDAgent
from rdagent.config import get_settings
from rdagent.llm import OllamaClient
from rdagent.tools import KnowledgeBase

EXAMPLES = [
    "Does the OV-600P currently meet the fan noise requirement?",
    "Which OV-600S requirements are still not met?",
    "Based on the simulation, would the ECR-0125 coating bring the OV-600P pyrolysis door "
    "temperature within the limit?",
    "Which change requests are still open or in review?",
]

settings = get_settings()
st.set_page_config(page_title="R&D Knowledge Agent", layout="wide")


@st.cache_resource
def load_kb() -> KnowledgeBase:
    return KnowledgeBase.from_dir(settings.data_dir)


kb = load_kb()

with st.sidebar:
    st.header("Settings")
    model = st.text_input("Ollama model", settings.model)
    host = st.text_input("Ollama host", settings.ollama_host)
    mode = st.radio("Mode", ["Agent with tools", "Single-shot RAG baseline"])
    st.caption(
        f"{len(kb.documents)} documents, {len(kb.results)} measurements, "
        f"{len(kb.limits)} specification limits. All data is fictional."
    )
    st.subheader("Try a question")
    for example in EXAMPLES:
        if st.button(example, use_container_width=True):
            st.session_state["pending"] = example

st.title("R&D Knowledge Agent")
st.write(
    "Ask about specifications, test results, change requests, lessons learned and meeting "
    "decisions. Every answer cites the documents it used, and citations are checked against what "
    "the tools actually returned."
)

question = st.chat_input("Ask a question") or st.session_state.pop("pending", None)
if question:
    st.chat_message("user").write(question)
    llm = OllamaClient(model=model, host=host)
    system = RDAgent(llm, kb, settings.max_steps) if mode.startswith("Agent") else BaselineRAG(llm, kb)
    with st.spinner("Working..."):
        try:
            result = system.run(question)
        except RuntimeError as exc:
            st.error(str(exc))
            st.stop()

    with st.chat_message("assistant"):
        st.markdown(result.answer)
        cols = st.columns(4)
        cols[0].metric("Latency", f"{result.latency_s:.1f} s")
        cols[1].metric("Tool calls", len(result.steps))
        cols[2].metric("Citations", len(result.citations))
        if result.refused:
            cols[3].metric("Grounding", "Refused")
        elif result.grounded:
            cols[3].metric("Grounding", "Checked")
        else:
            cols[3].metric("Grounding", "Problem")
        if result.invalid_citations:
            st.warning(f"Cited without retrieval: {', '.join(result.invalid_citations)}")

        if result.steps:
            with st.expander("Tool calls", expanded=False):
                for i, step in enumerate(result.steps, 1):
                    st.markdown(f"**{i}. `{step.tool}`** `{json.dumps(step.arguments)}`")
                    if step.error:
                        st.error(step.error)
                    else:
                        st.caption(f"Sources: {', '.join(step.sources) or 'none'}")
        if result.citations:
            with st.expander("Cited documents", expanded=False):
                for doc_id in result.citations:
                    doc = kb.documents.get(doc_id)
                    if doc:
                        st.markdown(f"**{doc.doc_id}**: {doc.title} ({doc.date})")
                        st.markdown(doc.body)
                        st.divider()
