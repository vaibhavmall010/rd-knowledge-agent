# R&D Knowledge Agent

A tool-calling LLM agent that answers engineering questions from specifications, test reports,
change requests, lessons learned and meeting minutes. It runs fully locally on Llama 3.1 through
Ollama, cites a document for every fact, and checks each citation against what its tools actually
returned. An evaluation harness compares it with single-shot RAG on 30 questions, and reports
where the agent wins and where it does not.

All product data is fictional: a three-variant built-in oven platform (OV-600B, OV-600P, OV-600S)
with 17 documents, 28 measurements and 23 specification limits.

## Why

Engineers in R&D spend a large part of their time finding things: the latest result of a test,
whether a change request fixed a problem, what was decided in the last review. That information
sits in two shapes. Reports and minutes are text; measurements and limits are tables. Plain RAG
handles the first and is unreliable on the second, because an LLM comparing 7.6 with 7.5 from
retrieved text gets the number or the direction of the limit wrong often enough to matter.

This agent gives the model tools for both shapes and leaves the arithmetic to code.

| Question | Tool the agent should use |
| --- | --- |
| "Does the OV-600P meet the fan noise requirement?" | `check_spec`: latest value against the limit, PASS/FAIL, margin, history |
| "Which OV-600S requirements are still not met?" | `compliance_summary`: every limit of one product |
| "What caused the fan noise failure and what fixed it?" | `search_documents`, then `get_document` |
| "Which change requests are still open?" | `list_change_requests` |
| "What fan-forced energy was measured before and after the baffle change?" | `query_test_results` |

## How it works

```mermaid
flowchart TB
    Q[Question] --> A[Agent loop<br/>Llama 3.1 via Ollama]
    A -- tool call --> T
    subgraph T[Tools]
        direction LR
        S[search_documents]
        G[get_document]
        R[query_test_results]
        C[check_spec / compliance_summary]
        E[list_change_requests]
    end
    T -- result + source ids --> A
    A -- draft answer --> V{Citation check}
    V -- cited id not retrieved --> A
    V -- ok --> Ans[Answer with citations]
```

- **Tools return source ids.** Every tool result carries the ids of the documents it came from.
- **Citations are verified.** The answer's `[TR-2026-047]`-style citations are parsed and compared
  with the ids the tools returned in this run. A citation to a document that was never retrieved,
  or one that does not exist, triggers one repair round. The UI shows the result.
- **Refusal is a valid answer.** The prompt requires the phrase "not found in the R&D knowledge
  base" when the tools return nothing. Four of the 30 evaluation questions test this.
- **Errors go back to the model.** Unknown products, parameters or ids return a message listing
  the valid values, so the model can correct its call.
- **Bounded loop.** At most six tool calls. Past that the model must answer with what it has.
- **Robust to local models.** If a model writes its tool call as JSON text instead of using the
  tool-calling field, the client recovers it.

Search is BM25 over document sections, with tokenisation that keeps identifiers such as
`OV-600P`, `TR-2026-040` and `0.85` intact. It needs no embedding model and makes exact ids easy to
find. Swapping in embeddings is a contained change in `retrieval.py`.

## Evaluation

`eval/questions.yaml` holds 30 questions in six categories: lookup, measurement, compliance,
change requests, multi-hop and unanswerable. Each question lists the facts the answer must contain,
the documents it must cite and the tools an expert would use.

**Tool layer (no model, runs in CI).** The expert tool calls for each question are executed and
their output is checked for the required facts and sources. This guards the consistency of data,
tools and evaluation set.

| Metric | Result |
| --- | --- |
| Questions fully answerable from tool output | 30 / 30 |
| Required facts present in tool output | 100% |
| Required sources returned by tools | 100% |
| BM25 recall@5 of required sources, raw question, no tools | 82% |

The last line is the motivation for the agent: a single retrieval on the raw question misses
about one in five required documents.

**Agent against baseline (needs Ollama).** `make eval` runs every question through single-shot
RAG and through the agent. Full report: [`reports/eval_baseline_agent.md`](reports/eval_baseline_agent.md).

Run on 2026-10-25 with `llama3.1` (8B), temperature 0:

| Metric | Single-shot RAG | Agent |
| --- | --- | --- |
| Answer accuracy, all 30 questions | **60%** | 30% |
| Correct refusals on unanswerable questions | **100%** | 50% |
| Citation precision | 99% | **100%** |
| Tool selection accuracy | n/a | 96% |
| False refusals | 2 | 9 |
| Median latency | **22.1 s** | 33.5 s |
| Mean tool calls | 0 | 1.37 |

**The agent lost overall.** That result is kept here deliberately, because the per-category
breakdown is the useful part:

| Category | Single-shot RAG | Agent |
| --- | --- | --- |
| compliance (does X meet its limit?) | 20% | **60%** |
| measurement | **80%** | 40% |
| lookup | **62%** | 12% |
| multi-hop | **50%** | 0% |

The agent wins by 3x exactly where it was designed to: compliance questions, where `check_spec`
compares a measured value against its limit in Python and the model never does the arithmetic.
It loses everywhere that needs more than one tool call.

**Root cause.** The agent picks the right tool 96% of the time but averages only 1.37 calls
against a budget of 6. With an 8B model, one call that returns a partial result is read as "the
tools do not have this", and rule 4 of the system prompt fires: 9 of 30 answers are false
refusals after a single call. Three examples (q11, q22, q25) refuse after one `get_document` or
`query_test_results` call that a second, differently-phrased call would have answered.

**A guardrail that does not hold.** On q28, "energy consumption of the OV-600P in steam mode"
(which does not exist), the agent answered `1.01 kWh [TR-2026-034]` — the conventional-mode
value. The citation check passed because TR-2026-034 *was* retrieved. The check validates that a
cited id was returned by a tool; it does not validate that the source supports the claim.
Catching that needs entailment checking against the retrieved passage, not id matching.

**What this says about the design.** Routing deterministic comparisons into code works and is
measurable. Multi-step tool use on a local 8B model does not, at this prompt and budget. The next
experiments are to require a second, differently-phrased attempt before a refusal is allowed, and
to re-run against a larger tool-calling model to separate a prompt problem from a model-capacity
problem.

## Quickstart

```bash
# 1. Model
ollama pull llama3.1          # any Ollama model with tool calling works, e.g. qwen2.5:7b

# 2. Install and test (the core needs only PyYAML)
python -m pip install -e . pytest
python -m pytest -q

# 3. Use it
python -m rdagent.cli "Which OV-600S requirements are still not met?"

# 4. Evaluate
python -m rdagent.evaluate --mode tools                       # no model needed
python -m rdagent.evaluate --mode compare --model llama3.1    # agent vs baseline
```

The core is pure Python (standard library plus PyYAML), so it runs on locked-down machines where
compiled packages are blocked. The optional web UI needs Streamlit, which does use compiled
packages:

```bash
python -m pip install -e ".[app]"
python -m streamlit run app/streamlit_app.py
```

With Docker: `docker compose up`, then `docker compose exec ollama ollama pull llama3.1` once, and
open http://localhost:8501.

## Project layout

```
data/documents/      17 Markdown documents with YAML front matter
data/test_results.csv  28 measurements, each linked to its test report
data/spec_limits.csv   23 limits, each linked to a specification section
eval/questions.yaml  30 evaluation questions
src/rdagent/
  corpus.py          loading and section chunking
  retrieval.py       BM25 index
  tools.py           the six tools, argument validation, JSON schemas
  llm.py             Ollama client and a scripted client for tests
  agent.py           agent loop, citation check, single-shot RAG baseline
  evaluate.py        metrics and reports
  cli.py             terminal interface
app/streamlit_app.py chat UI with tool trace and cited documents
tests/               28 tests, including a round trip through a fake Ollama server
```

## Limitations

- The data is synthetic and small. Real R&D archives need access control, document versioning
  and parsers for PDF and Office files.
- BM25 misses paraphrases. The agent is supposed to compensate with several searches, but the
  measured average is 1.37 calls per question, so in practice it rarely does; embeddings or
  hybrid search would help on a larger corpus.
- Fact matching in the evaluation is string-based, so a correct answer phrased differently can be
  scored as wrong. The per-question report shows each answer for manual review.
- The citation check verifies that a cited document id was returned by a tool, not that the
  document supports the claim. Measured consequence in the Evaluation section (q28).
- Results are from one run of one 8B model. The agent/baseline gap may be a prompt and budget
  problem rather than a property of the approach; a larger tool-calling model would separate the two.

## License

MIT