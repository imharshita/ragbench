# Chunking strategies

Chunking splits documents into passages before indexing. Fixed-size chunking uses a
maximum character count and an overlap between neighbouring chunks. Paragraph chunking
keeps natural paragraph boundaries and preserves topical coherence. Overlap reduces the
chance that an answer is split across two chunks. Smaller chunks improve retrieval
precision but can lose context.
