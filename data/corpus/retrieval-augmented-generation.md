# Retrieval-Augmented Generation

Retrieval-Augmented Generation (RAG) combines a retrieval system with a language model.
The retriever selects relevant passages from a corpus, and the generator answers using
only those passages. A grounded RAG system should cite the passages it uses. RAG is
preferred over fine-tuning when the knowledge changes often, because the corpus can be
updated without retraining.
