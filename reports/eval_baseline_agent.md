# Evaluation report: baseline_agent

Created 2026-10-25 19:08 with model `llama3.1`.

| Metric | baseline | agent |
| --- | --- | --- |
| Answer accuracy, all questions | 60% | 30% |
| Accuracy, answerable questions | 54% | 27% |
| Correct refusals, unanswerable questions | 100% | 50% |
| Fact recall | 70% | 48% |
| Citation recall | 78% | 52% |
| Citation precision | 99% | 100% |
| Tool selection accuracy | n/a | 96% |
| False refusals | 2 | 9 |
| Mean latency (s) | 23.967 | 41.052 |
| Median latency (s) | 22.067 | 33.457 |
| Mean tool calls | 0 | 1.367 |
| Run errors | 0 | 0 |

## Accuracy by category

| Category | baseline | agent |
| --- | --- | --- |
| change | 50% | 25% |
| compliance | 20% | 60% |
| lookup | 62% | 12% |
| measurement | 80% | 40% |
| multi_hop | 50% | 0% |
| unanswerable | 100% | 50% |

## Per question

### baseline

- q01 [lookup]: ok
- q02 [compliance]: FAIL
- q03 [multi_hop]: FAIL
- q04 [measurement]: ok
- q05 [compliance]: FAIL
- q06 [change]: FAIL
- q07 [change]: ok
- q08 [measurement]: ok
- q09 [lookup]: ok
- q10 [multi_hop]: FAIL
- q11 [lookup]: ok
- q12 [lookup]: FAIL
- q13 [lookup]: FAIL
- q14 [measurement]: FAIL
- q15 [compliance]: FAIL
- q16 [measurement]: ok
- q17 [change]: FAIL
- q18 [multi_hop]: ok
- q19 [compliance]: FAIL
- q20 [lookup]: ok
- q21 [lookup]: FAIL
- q22 [measurement]: ok
- q23 [multi_hop]: ok
- q24 [change]: ok
- q25 [lookup]: ok
- q26 [compliance]: ok
- q27 [unanswerable]: ok
- q28 [unanswerable]: ok
- q29 [unanswerable]: ok
- q30 [unanswerable]: ok

### agent

- q01 [lookup]: ok
- q02 [compliance]: FAIL
- q03 [multi_hop]: FAIL
- q04 [measurement]: FAIL
- q05 [compliance]: ok
- q06 [change]: FAIL
- q07 [change]: FAIL
- q08 [measurement]: ok
- q09 [lookup]: FAIL
- q10 [multi_hop]: FAIL
- q11 [lookup]: FAIL
- q12 [lookup]: FAIL
- q13 [lookup]: FAIL
- q14 [measurement]: FAIL
- q15 [compliance]: ok
- q16 [measurement]: ok
- q17 [change]: ok
- q18 [multi_hop]: FAIL
- q19 [compliance]: FAIL
- q20 [lookup]: FAIL
- q21 [lookup]: FAIL
- q22 [measurement]: FAIL
- q23 [multi_hop]: FAIL
- q24 [change]: FAIL
- q25 [lookup]: FAIL
- q26 [compliance]: ok
- q27 [unanswerable]: ok
- q28 [unanswerable]: FAIL
- q29 [unanswerable]: ok
- q30 [unanswerable]: FAIL
