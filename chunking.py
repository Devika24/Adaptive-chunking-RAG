import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict
import re

class AdaptiveChunker:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.similarity_model = SentenceTransformer(model_name)
        self.semantic_threshold = 0.7
    
    def semantic_chunking(self, text: str, max_chunk_size: int = 500) -> List[str]:
        """Adaptive chunking based on semantic coherence"""
        sentences = self._split_into_sentences(text)
        chunks = []
        current_chunk = []
        current_embeddings = []
        
        for sentence in sentences:
            if not current_chunk:
                current_chunk.append(sentence)
                current_embeddings.append(self.similarity_model.encode([sentence])[0])
                continue
            
            # Calculate semantic similarity with current chunk
            sent_embedding = self.similarity_model.encode([sentence])[0]
            similarities = cosine_similarity([sent_embedding], current_embeddings)[0]
            avg_similarity = np.mean(similarities)
            
            if (avg_similarity >= self.semantic_threshold and 
                len(' '.join(current_chunk + [sentence])) <= max_chunk_size):
                current_chunk.append(sentence)
                current_embeddings.append(sent_embedding)
            else:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_embeddings = [sent_embedding]
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def dynamic_size_chunking(self, text: str, target_chunk_size: int = 400, overlap: int = 50) -> List[str]:
        """Traditional fixed-size chunking with overlap for comparison"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), target_chunk_size - overlap):
            chunk = ' '.join(words[i:i + target_chunk_size])
            chunks.append(chunk)
            if i + target_chunk_size >= len(words):
                break
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Simple sentence splitting"""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def evaluate_chunk_quality(self, chunks: List[str]) -> Dict:
        """Evaluate chunk coherence and quality"""
        if len(chunks) < 2:
            return {"coherence_score": 1.0, "num_chunks": len(chunks)}
        
        # Calculate inter-chunk semantic similarity
        embeddings = self.similarity_model.encode(chunks)
        similarity_matrix = cosine_similarity(embeddings)
        
        # Exclude self-similarity
        np.fill_diagonal(similarity_matrix, 0)
        avg_similarity = np.sum(similarity_matrix) / (len(chunks) * (len(chunks) - 1))
        
        chunk_lengths = [len(chunk.split()) for chunk in chunks]
        length_std = np.std(chunk_lengths)
        
        return {
            "coherence_score": float(avg_similarity),
            "num_chunks": len(chunks),
            "avg_chunk_length": np.mean(chunk_lengths),
            "length_std": float(length_std)
        }

# Example usage
if __name__ == "__main__":
    chunker = AdaptiveChunker()
    
    sample_text = """
    Large language models have revolutionized natural language processing. 
    They can understand and generate human-like text across various domains. 
    However, these models sometimes produce inaccurate or hallucinated content. 
    Retrieval-augmented generation helps mitigate this issue by grounding responses in factual information.
    Semantic chunking improves RAG performance by creating coherent context windows.
    """
    
    semantic_chunks = chunker.semantic_chunking(sample_text)
    fixed_chunks = chunker.dynamic_size_chunking(sample_text)
    
    print("Semantic Chunks:", semantic_chunks)
    print("Fixed Chunks:", fixed_chunks)
    
    semantic_quality = chunker.evaluate_chunk_quality(semantic_chunks)
    fixed_quality = chunker.evaluate_chunk_quality(fixed_chunks)
    
    print("Semantic Quality:", semantic_quality)
    print("Fixed Quality:", fixed_quality)