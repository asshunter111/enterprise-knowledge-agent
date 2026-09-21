# Evaluation Notes

## Baseline

The production configuration is Hash Embedding + Lexical Reranker with `800/150`. The current offline runner deliberately executes a separate `200/30` chunk experiment to preserve the existing multi-chunk cases. It prints both configurations and passes one explicit experiment `Settings` object to index construction and retrieval. The result is not a production-config result.

## Metrics

The evaluation primitives support Hit@K, MRR, and paired pre/post comparisons. Existing document-level Hit@K remains a retrieval smoke signal: it does not prove answer correctness, citation correctness, or faithfulness. `expected_keywords` is dataset metadata and must not be described as an evaluated metric until an answer/ evidence scorer consumes it.

## Live evaluation

Real DeepSeek Resolver and BGE Embedding remain separate manual experiments. Their results are not comparable with the Hash baseline unless dataset, chunk settings, collection, reranker, and scoring protocol are held constant. No live result is treated as a CI regression gate.

## Known gaps

The current dataset is 23 cases with repeated questions and limited coverage. On the current Hash experiment, Verify rejects `0/3` no-answer retrieval cases; this is a known confidence-calibration failure, not a passing abstention result. q10 is also semantically arguable from the finance document, so the dataset label needs human adjudication before using it as a hard abstention threshold. Generation quality, permission cases, tool cases, memory cases, adversarial documents, and cross-document evidence require additional labeled data before their metrics can be reported.

## Abstention comparison

The retrieval Verify threshold remains `MIN_RELEVANCE_SCORE=0.05`. A separate
post-rerank evidence decision was added with `evidence_min_rerank_score=0.20`.
It runs after Reranking and before Generator, so low-confidence evidence is
abstained without changing the retrieval candidate configuration or calling the
Generator. The threshold was selected from the small experiment's score
distribution, not tuned to remove individual failed cases.

The comparison below uses the 12 single-turn cases only: 9 answerable and 3
no-answer cases. The ground truth was not changed.

| Metric | Baseline | Improved |
| --- | ---: | ---: |
| Answerable correct answer rate | 9/9 (100.00%) | 9/9 (100.00%) |
| Correct abstention | 0/3 (0.00%) | 1/3 (33.33%) |
| False refusal | 0/9 (0.00%) | 0/9 (0.00%) |
| False answer | 3/3 (100.00%) | 2/3 (66.67%) |
| Abstention precision | n/a | 1/1 (100.00%) |
| Abstention recall | 0/3 (0.00%) | 1/3 (33.33%) |

The improvement rejects q08 (`兰州有多少所大学？`), whose top rerank score
was `0.1782`. q09 (`公司今年的营业收入是多少？`) and q10
(`员工可以报销哪些费用？`) still pass the evidence threshold. q10 is
semantically arguable because the finance document contains expense-category
material, so it remains a dataset-label limitation rather than evidence that a
higher threshold should be applied blindly. This is a small offline experiment,
not a production abstention guarantee.
