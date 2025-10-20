# src/evaluation.py

from .chunking import AdaptiveChunker
from .retrieval import AdaptiveRAGSystem
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class RagEvaluator:
    def __init__(self):
        self.chunker = AdaptiveChunker()
    
    def evaluate_retrieval(self, queries, ground_truths, documents, strategy="semantic"):
        rag_system = AdaptiveRAGSystem(chunking_strategy=strategy)
        rag_system.build_knowledge_base(documents)
        
        precisions = []
        for query, ground_truth in zip(queries, ground_truths):
            results = rag_system.retrieve(query, k=3)
            # Simple evaluation: check if the ground_truth is in the retrieved chunks
            # In practice, you might use more complex metrics
            retrieved_texts = [r['chunk'] for r in results]
            precision = self._calculate_precision(retrieved_texts, ground_truth)
            precisions.append(precision)
        
        return np.mean(precisions)
    
    def _calculate_precision(self, retrieved_texts, ground_truth):
        # Simple check: if any of the retrieved chunks contains the ground_truth string
        # This is a simplified version. In reality, you might use semantic similarity.
        for text in retrieved_texts:
            if ground_truth in text:
                return 1.0
        return 0.0
    
    def compare_strategies(self, documents, queries, ground_truths):
        results = {}
        for strategy in ["semantic", "fixed"]:
            precision = self.evaluate_retrieval(queries, ground_truths, documents, strategy)
            results[strategy] = precision
        
        results['improvement'] = (results['semantic'] - results['fixed']) / results['fixed']
        return results