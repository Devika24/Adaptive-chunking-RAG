# examples/demo.py

import sys
sys.path.append('..')

from src.chunking import AdaptiveChunker
from src.retrieval import AdaptiveRAGSystem

# Sample documents
documents = [
    "Large language models have revolutionized natural language processing. They can understand and generate human-like text across various domains.",
    "However, these models sometimes produce inaccurate or hallucinated content. Retrieval-augmented generation helps mitigate this issue by grounding responses in factual information.",
    "Semantic chunking improves RAG performance by creating coherent context windows that preserve logical flow and topic continuity."
]

# Initialize adaptive chunker
chunker = AdaptiveChunker()

# Chunk document semantically
document = " ".join(documents)
semantic_chunks = chunker.semantic_chunking(document)
fixed_chunks = chunker.dynamic_size_chunking(document)

print("Semantic Chunks:")
for i, chunk in enumerate(semantic_chunks, 1):
    print(f"{i}. {chunk}")

print("\nFixed Chunks:")
for i, chunk in enumerate(fixed_chunks, 1):
    print(f"{i}. {chunk}")

# Build RAG system
rag_system = AdaptiveRAGSystem(chunking_strategy="semantic")
rag_system.build_knowledge_base(documents)

# Retrieve relevant information
query = "What is semantic chunking?"
results = rag_system.retrieve(query)

print(f"\nQuery: {query}")
for result in results:
    print(f"Chunk: {result['chunk']}")
    print(f"Relevance Score: {result['score']:.3f}\n")