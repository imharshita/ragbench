# Production concerns

A production RAG service should log retrieved passages, model version, and latency per
request. Hybrid retrieval combines lexical BM25 with dense embeddings and fuses the
ranked lists with reciprocal rank fusion. Caching frequent queries reduces cost but the
cache must be invalidated when the corpus changes. Store an immutable corpus version
with every evaluation run so results are reproducible.
