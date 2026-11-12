# Evaluating RAG

Answer faithfulness measures whether every claim in the answer is supported by the
retrieved context. Citation precision is the fraction of cited passages that actually
support the answer. Retrieval recall at k is the fraction of questions whose gold
passage appears in the top k results. Unanswerable questions test whether a system
abstains instead of hallucinating. A good evaluation set contains both answerable and
unanswerable questions.
