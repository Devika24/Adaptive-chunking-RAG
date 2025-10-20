import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import json

class AdaptiveRAGSystem:
    def __init__(self, chunking_strategy="semantic"):
        self.similarity_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.chunker = AdaptiveChunker()
        self.chunking_strategy = chunking_strategy
        self.index = None
        self.chunks = []
    
    def build_knowledge_base(self, documents: List[str]):
        """Build vector store from documents"""
        all_chunks = []
        
        for doc in documents:
            if self.chunking_strategy == "semantic":
                chunks = self.chunker.semantic_chunking(doc)
            else:
                chunks = self.chunker.dynamic_size_chunking(doc)
            all_chunks.extend(chunks)
        
        self.chunks = all_chunks
        embeddings = self.similarity_model.encode(all_chunks)
        
        # Create FAISS index
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings.astype(np.float32))
        
        return len(all_chunks)
    
    def retrieve(self, query: str, k: int = 3) -> List[Dict]:
        """Retrieve most relevant chunks"""
        if self.index is None:
            raise ValueError("Knowledge base not built. Call build_knowledge_base first.")
        
        query_embedding = self.similarity_model.encode([query])
        scores, indices = self.index.search(query_embedding.astype(np.float32), k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                results.append({
                    "chunk": self.chunks[idx],
                    "score": float(score),
                    "chunk_id": int(idx)
                })
        
        return results

# Evaluation
def evaluate_rag_system():
    """Compare semantic vs fixed chunking"""
    # Sample documents (in practice, load from dataset)
    documents = [
        "Large language models transform AI. They understand context and generate text.",
        "Retrieval-augmented generation improves factuality. It combines retrieval with generation.",
        "Semantic chunking creates coherent context windows for better retrieval performance."
    ]
    
    # Test both strategies
    semantic_rag = AdaptiveRAGSystem("semantic")
    fixed_rag = AdaptiveRAGSystem("fixed")
    
    semantic_rag.build_knowledge_base(documents)
    fixed_rag.build_knowledge_base(documents)
    
    test_queries = [
        "What are large language models?",
        "How does RAG improve factuality?",
        "What is semantic chunking?"
    ]
    
    results = {}
    for query in test_queries:
        semantic_results = semantic_rag.retrieve(query)
        fixed_results = fixed_rag.retrieve(query)
        
        results[query] = {
            "semantic": [r["score"] for r in semantic_results],
            "fixed": [r["score"] for r in fixed_results]
        }
    
    return results

if __name__ == "__main__":
    performance = evaluate_rag_system()
    print("RAG Performance Comparison:", json.dumps(performance, indent=2))