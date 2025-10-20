# generate_results.py

import sys
sys.path.append('src')

from chunking import AdaptiveChunker
from evaluation import RagEvaluator
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def generate_chunking_evaluation():
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
    
    semantic_quality = chunker.evaluate_chunk_quality(semantic_chunks)
    fixed_quality = chunker.evaluate_chunk_quality(fixed_chunks)
    
    results = {
        "semantic_chunking": semantic_quality,
        "fixed_chunking": fixed_quality
    }
    
    with open('results/chunking_evaluation.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate a bar plot for coherence score and number of chunks
    strategies = ['Semantic', 'Fixed']
    coherence_scores = [semantic_quality['coherence_score'], fixed_quality['coherence_score']]
    num_chunks = [semantic_quality['num_chunks'], fixed_quality['num_chunks']]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    ax1.bar(strategies, coherence_scores, color=['blue', 'orange'])
    ax1.set_title('Coherence Score by Strategy')
    ax1.set_ylabel('Coherence Score')
    
    ax2.bar(strategies, num_chunks, color=['blue', 'orange'])
    ax2.set_title('Number of Chunks by Strategy')
    ax2.set_ylabel('Number of Chunks')
    
    plt.tight_layout()
    plt.savefig('results/semantic_vs_fixed.png')
    plt.close()

if __name__ == "__main__":
    generate_chunking_evaluation()