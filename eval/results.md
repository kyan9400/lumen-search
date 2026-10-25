# Retrieval evaluation

62 questions from [questions.jsonl](questions.jsonl), produced by `lumen eval --configs bm25,dense,hybrid,hybrid+rerank --reranker cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 --out eval/results.md`.

**All questions** (62)

| Retrieval | Hit@1 | Hit@5 | Recall@10 | MRR@10 | nDCG@10 | p50 ms | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bm25 | 54.8% | 67.7% | 73.8% | 0.605 | 0.621 | 2.9 | 4.0 |
| dense | 48.4% | 75.8% | 77.8% | 0.600 | 0.624 | 42.9 | 52.3 |
| hybrid | 56.5% | 75.8% | 79.8% | 0.653 | 0.664 | 43.8 | 56.9 |
| hybrid + rerank | 62.9% | 83.9% | 79.8% | 0.704 | 0.708 | 976.3 | 1524.5 |

**en** (44)

| Retrieval | Hit@1 | Hit@5 | Recall@10 | MRR@10 | nDCG@10 | p50 ms | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bm25 | 59.1% | 72.7% | 79.0% | 0.651 | 0.662 | 2.7 | 3.5 |
| dense | 45.5% | 79.5% | 81.2% | 0.597 | 0.630 | 42.9 | 52.3 |
| hybrid | 56.8% | 79.5% | 86.4% | 0.677 | 0.696 | 43.8 | 57.4 |
| hybrid + rerank | 63.6% | 90.9% | 86.4% | 0.734 | 0.748 | 963.7 | 1492.5 |

**ru** (18)

| Retrieval | Hit@1 | Hit@5 | Recall@10 | MRR@10 | nDCG@10 | p50 ms | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bm25 | 44.4% | 55.6% | 61.1% | 0.493 | 0.520 | 3.8 | 4.2 |
| dense | 55.6% | 66.7% | 69.4% | 0.606 | 0.608 | 42.9 | 52.6 |
| hybrid | 55.6% | 66.7% | 63.9% | 0.594 | 0.586 | 44.2 | 56.0 |
| hybrid + rerank | 61.1% | 66.7% | 63.9% | 0.630 | 0.611 | 1001.2 | 1738.8 |

**cross-lingual** (7)

| Retrieval | Hit@1 | Hit@5 | Recall@10 | MRR@10 | nDCG@10 | p50 ms | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bm25 | 0.0% | 0.0% | 0.0% | 0.000 | 0.000 | 3.2 | 4.0 |
| dense | 0.0% | 14.3% | 7.1% | 0.071 | 0.055 | 45.9 | 52.6 |
| hybrid | 0.0% | 14.3% | 7.1% | 0.036 | 0.038 | 43.8 | 46.8 |
| hybrid + rerank | 14.3% | 14.3% | 7.1% | 0.143 | 0.088 | 1017.3 | 1738.8 |
**By question type**

| Retrieval | Keyword Hit@1 | Keyword Hit@5 | Paraphrase Hit@1 | Paraphrase Hit@5 |
| --- | ---: | ---: | ---: | ---: |
| bm25 | 74.2% | 90.3% | 45.8% | 58.3% |
| dense | 61.3% | 93.5% | 45.8% | 70.8% |
| hybrid | 74.2% | 96.8% | 50.0% | 66.7% |
| hybrid + rerank | 83.9% | 100.0% | 50.0% | 83.3% |

**Reranking the top 20 instead of the top 10**

| Retrieval | Hit@1 | Hit@5 | Recall@10 | MRR@10 | nDCG@10 | p50 ms | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hybrid + rerank top 20 | 62.9% | 82.3% | 85.9% | 0.705 | 0.726 | 1768.9 | 2673.5 |

Models: intfloat/multilingual-e5-small (embeddings), cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 (reranker).
Laptop CPU, 16 threads; cache off; models loaded before timing.
