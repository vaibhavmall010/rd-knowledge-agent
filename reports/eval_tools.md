# Evaluation report: tools

Created 2026-10-25 18:34.

Model-free check of the tools and data. For each question the expert tool calls in `eval/questions.yaml` are executed and their output is searched for the expected facts and source documents.

| Metric | Value |
| --- | --- |
| Questions | 30 |
| Questions fully answerable from tool output | 30 / 30 |
| Fact coverage (answerable) | 100% |
| Source coverage (answerable) | 100% |
| BM25 recall@5 of required sources, raw question | 82% |

BM25 recall is measured on the raw question with no tools. It shows how often a single-shot retriever finds the needed documents, which is the gap the agent's structured tools are meant to close.

## Per question

### tools

- q01 [lookup]: ok
- q02 [compliance]: ok
- q03 [multi_hop]: ok
- q04 [measurement]: ok
- q05 [compliance]: ok
- q06 [change]: ok
- q07 [change]: ok
- q08 [measurement]: ok
- q09 [lookup]: ok
- q10 [multi_hop]: ok
- q11 [lookup]: ok
- q12 [lookup]: ok
- q13 [lookup]: ok
- q14 [measurement]: ok
- q15 [compliance]: ok
- q16 [measurement]: ok
- q17 [change]: ok
- q18 [multi_hop]: ok
- q19 [compliance]: ok
- q20 [lookup]: ok
- q21 [lookup]: ok
- q22 [measurement]: ok
- q23 [multi_hop]: ok
- q24 [change]: ok
- q25 [lookup]: ok
- q26 [compliance]: ok
- q27 [unanswerable]: ok
- q28 [unanswerable]: ok
- q29 [unanswerable]: ok
- q30 [unanswerable]: ok
