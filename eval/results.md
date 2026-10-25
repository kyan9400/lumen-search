# Retrieval evaluation

62 questions from [questions.jsonl](questions.jsonl): 31 share wording with their answer (keyword), 24 do
not (paraphrase), 7 ask in one language about a document in the other (cross-lingual). Produced with

```bash
lumen eval --configs bm25,dense,hybrid,hybrid+rerank --reranker cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
```

Models: intfloat/multilingual-e5-small (embeddings) and cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
(reranker, top 10). Laptop CPU with 16 threads; cache off; models loaded before timing.

**All questions** (62)

| Retrieval | Hit@1 | Hit@5 | Recall@10 | MRR@10 | nDCG@10 | p50 ms | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bm25 | 54.8% | 67.7% | 73.8% | 0.605 | 0.621 | 2.6 | 3.2 |
| dense | 48.4% | 75.8% | 77.8% | 0.601 | 0.625 | 40.7 | 44.7 |
| hybrid | 56.5% | 75.8% | 79.8% | 0.650 | 0.662 | 40.0 | 49.5 |
| hybrid + rerank | 62.9% | 83.9% | 79.8% | 0.704 | 0.707 | 1000.5 | 1493.1 |

**en** (44)

| Retrieval | Hit@1 | Hit@5 | Recall@10 | MRR@10 | nDCG@10 | p50 ms | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bm25 | 59.1% | 72.7% | 79.0% | 0.651 | 0.662 | 2.5 | 3.2 |
| dense | 45.5% | 79.5% | 81.2% | 0.597 | 0.630 | 41.3 | 44.7 |
| hybrid | 56.8% | 79.5% | 86.4% | 0.673 | 0.693 | 40.0 | 50.9 |
| hybrid + rerank | 63.6% | 90.9% | 86.4% | 0.734 | 0.748 | 950.5 | 1493.1 |

**ru** (18)

| Retrieval | Hit@1 | Hit@5 | Recall@10 | MRR@10 | nDCG@10 | p50 ms | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bm25 | 44.4% | 55.6% | 61.1% | 0.493 | 0.520 | 3.0 | 3.5 |
| dense | 55.6% | 66.7% | 69.4% | 0.611 | 0.612 | 39.4 | 45.4 |
| hybrid | 55.6% | 66.7% | 63.9% | 0.594 | 0.586 | 40.8 | 49.5 |
| hybrid + rerank | 61.1% | 66.7% | 63.9% | 0.630 | 0.608 | 1080.6 | 1539.3 |

**cross-lingual** (7)

| Retrieval | Hit@1 | Hit@5 | Recall@10 | MRR@10 | nDCG@10 | p50 ms | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bm25 | 0.0% | 0.0% | 0.0% | 0.000 | 0.000 | 2.2 | 3.0 |
| dense | 0.0% | 14.3% | 7.1% | 0.071 | 0.055 | 39.9 | 51.7 |
| hybrid | 0.0% | 14.3% | 7.1% | 0.036 | 0.038 | 38.7 | 42.7 |
| hybrid + rerank | 14.3% | 14.3% | 7.1% | 0.143 | 0.088 | 1000.5 | 1539.3 |

**keyword** (31)

| Retrieval | Hit@1 | Hit@5 | Recall@10 | MRR@10 | nDCG@10 | p50 ms | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bm25 | 74.2% | 90.3% | 91.9% | 0.805 | 0.823 | 2.9 | 4.3 |
| dense | 61.3% | 93.5% | 98.4% | 0.761 | 0.806 | 36.7 | 40.4 |
| hybrid | 74.2% | 96.8% | 98.4% | 0.840 | 0.863 | 38.0 | 43.3 |
| hybrid + rerank | 83.9% | 100.0% | 98.4% | 0.903 | 0.916 | 918.5 | 1458.7 |

**paraphrase** (24)

| Retrieval | Hit@1 | Hit@5 | Recall@10 | MRR@10 | nDCG@10 | p50 ms | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bm25 | 45.8% | 58.3% | 71.9% | 0.525 | 0.541 | 2.9 | 4.0 |
| dense | 45.8% | 70.8% | 71.9% | 0.550 | 0.558 | 37.7 | 41.4 |
| hybrid | 50.0% | 66.7% | 77.1% | 0.584 | 0.584 | 39.5 | 45.1 |
| hybrid + rerank | 50.0% | 83.3% | 77.1% | 0.610 | 0.618 | 918.5 | 1097.4 |

**Reranking the top 20 instead of the top 10**

| Retrieval | Hit@1 | Hit@5 | Recall@10 | MRR@10 | nDCG@10 | p50 ms | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hybrid + rerank top 20 | 62.9% | 82.3% | 85.9% | 0.705 | 0.725 | 1877.7 | 2832.0 |
