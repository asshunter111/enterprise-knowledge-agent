# Evaluation Notes

## Baseline

The production configuration is Hash Embedding + Lexical Reranker with `800/150`. The current offline runner deliberately executes a separate `200/30` chunk experiment to preserve the existing multi-chunk cases. It prints both configurations and passes one explicit experiment `Settings` object to index construction and retrieval. The result is not a production-config result.

## Metrics

The evaluation primitives support Hit@K, MRR, and paired pre/post comparisons. Existing document-level Hit@K remains a retrieval smoke signal: it does not prove answer correctness, citation correctness, or faithfulness. `expected_keywords` is dataset metadata and must not be described as an evaluated metric until an answer/ evidence scorer consumes it.

## Live evaluation

Real DeepSeek Resolver and BGE Embedding remain separate manual experiments. Their results are not comparable with the Hash baseline unless dataset, chunk settings, collection, reranker, and scoring protocol are held constant. No live result is treated as a CI regression gate.

## Known gaps

The current dataset is 23 cases with repeated questions and limited coverage. On the current Hash experiment, Verify rejects `0/3` no-answer retrieval cases; this is a known confidence-calibration failure, not a passing abstention result. q10 is also semantically arguable from the finance document, so the dataset label needs human adjudication before using it as a hard abstention threshold. Generation quality, permission cases, tool cases, memory cases, adversarial documents, and cross-document evidence require additional labeled data before their metrics can be reported.
