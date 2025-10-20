import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from chunking import AdaptiveChunker
from retrieval import AdaptiveRAGSystem

# Set page config
st.set_page_config(
    page_title="Adaptive Chunking RAG Demo",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    .chunk-box {
        background-color: #f8f9fa;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
        border: 1px solid #e9ecef;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🔍 Adaptive Chunking RAG Demo</h1>', unsafe_allow_html=True)
    st.markdown("### Compare semantic vs fixed chunking strategies in real-time")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    chunking_strategy = st.sidebar.selectbox(
        "Chunking Strategy",
        ["semantic", "fixed"],
        help="Choose between semantic coherence-based or fixed-size chunking"
    )
    
    semantic_threshold = st.sidebar.slider(
        "Semantic Threshold", 
        min_value=0.1, max_value=0.9, value=0.7, step=0.1,
        help="Higher values create more coherent but potentially larger chunks"
    )
    
    max_chunk_size = st.sidebar.slider(
        "Max Chunk Size (words)",
        min_value=100, max_value=1000, value=500, step=50,
        help="Maximum number of words per chunk"
    )
    
    min_chunk_size = st.sidebar.slider(
        "Min Chunk Size (words)", 
        min_value=50, max_value=300, value=100, step=25,
        help="Minimum number of words per chunk"
    )
    
    # Sample documents selection
    sample_docs = st.sidebar.selectbox(
        "Sample Documents",
        ["AI Research", "Technical Documentation", "News Article", "Custom Input"],
        help="Choose from sample documents or use custom input"
    )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Input Document")
        
        # Sample documents
        sample_texts = {
            "AI Research": """
            Large language models have revolutionized natural language processing in recent years. 
            These models, based on transformer architectures, can understand and generate human-like text across various domains. 
            However, they sometimes produce inaccurate or hallucinated content, which limits their reliability in critical applications.
            
            Retrieval-augmented generation (RAG) addresses this issue by combining language models with external knowledge sources. 
            The system retrieves relevant information from a knowledge base and uses it as context for generation. 
            This approach significantly improves factuality and reduces hallucinations.
            
            Semantic chunking plays a crucial role in RAG systems by creating coherent context windows. 
            Traditional fixed-size chunking often breaks sentences and ideas arbitrarily, leading to poor retrieval performance. 
            Adaptive chunking, on the other hand, respects semantic boundaries and preserves logical flow.
            """,
            
            "Technical Documentation": """
            The AdaptiveChunker class provides intelligent document segmentation based on semantic coherence. 
            It analyzes text using sentence embeddings and cosine similarity to determine optimal chunk boundaries.
            
            Key parameters include semantic_threshold, which controls how similar sentences must be to remain in the same chunk. 
            Higher values create more conservative chunks, while lower values allow more aggressive merging.
            
            The evaluate_chunk_quality method returns metrics including coherence_score, which measures semantic consistency within chunks. 
            Additional metrics track chunk size distribution and retrieval performance.
            
            The system supports multiple embedding models and can be extended with custom similarity functions. 
            Integration with FAISS enables efficient vector similarity search for large document collections.
            """,
            
            "News Article": """
            Artificial intelligence continues to transform industries worldwide. Recent breakthroughs in large language models 
            have enabled new applications in healthcare, education, and business automation. Companies are investing heavily 
            in AI research and development to maintain competitive advantages.
            
            However, challenges remain in ensuring AI system reliability and factuality. Researchers are developing new techniques 
            like retrieval-augmented generation to improve accuracy. These systems combine the creativity of language models 
            with the precision of structured knowledge bases.
            
            The future of AI looks promising with continued advances in model architecture and training methodologies. 
            Ethical considerations and responsible deployment remain critical focus areas for the research community.
            """
        }
        
        if sample_docs == "Custom Input":
            input_text = st.text_area(
                "Enter your document:",
                height=300,
                placeholder="Paste your document text here...",
                help="Enter any text document to test the chunking strategies"
            )
        else:
            input_text = sample_texts[sample_docs]
            st.text_area(
                "Document Preview:",
                value=input_text,
                height=300,
                key="sample_doc"
            )
    
    with col2:
        st.subheader("📊 Quick Metrics")
        
        if st.button("🚀 Process Document", type="primary", use_container_width=True):
            if not input_text.strip():
                st.error("Please enter some text to process!")
                return
                
            with st.spinner("🔄 Processing document..."):
                process_document(input_text, chunking_strategy, semantic_threshold, max_chunk_size, min_chunk_size)
        
        # Display tips
        st.markdown("---")
        st.markdown("### 💡 Pro Tips")
        st.markdown("""
        - **Semantic chunking** works best for documents with clear topical sections
        - **Lower thresholds** (0.3-0.5) create smaller, more focused chunks
        - **Higher thresholds** (0.7-0.9) preserve more context in each chunk
        - Use **fixed chunking** for uniform documents without clear section breaks
        """)
    
    # Comparison section
    st.markdown("---")
    st.subheader("📈 Strategy Comparison")
    
    if st.button("🆚 Compare Both Strategies", use_container_width=True):
        if not input_text.strip():
            st.error("Please enter some text to compare!")
        else:
            with st.spinner("Running comprehensive comparison..."):
                compare_strategies(input_text, semantic_threshold, max_chunk_size, min_chunk_size)

def process_document(text, strategy, threshold, max_size, min_size):
    """Process document with selected strategy"""
    try:
        # Initialize chunker with selected parameters
        chunker = AdaptiveChunker(semantic_threshold=threshold)
        
        # Apply chunking strategy
        if strategy == "semantic":
            chunks = chunker.semantic_chunking(text, max_chunk_size=max_size)
            strategy_name = "Semantic Chunking"
        else:
            chunks = chunker.dynamic_size_chunking(text, target_chunk_size=max_size, overlap=50)
            strategy_name = "Fixed-Size Chunking"
        
        # Evaluate quality
        metrics = chunker.evaluate_chunk_quality(chunks)
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Number of Chunks", metrics['num_chunks'])
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Coherence Score", f"{metrics['coherence_score']:.3f}")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            avg_length = metrics['avg_chunk_length']
            st.metric("Avg Chunk Length", f"{avg_length:.1f} words")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Display chunks
        st.subheader(f"📄 Generated Chunks ({strategy_name})")
        
        for i, chunk in enumerate(chunks, 1):
            with st.expander(f"Chunk {i} | {len(chunk.split())} words | Coherence: {calculate_chunk_coherence(chunker, chunk):.3f}"):
                st.write(chunk)
                st.caption(f"Character count: {len(chunk)} | Word count: {len(chunk.split())}")
        
        # Test retrieval if we have chunks
        if chunks:
            test_retrieval(chunker, chunks, text, strategy)
            
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")

def calculate_chunk_coherence(chunker, chunk):
    """Calculate coherence for a single chunk"""
    sentences = [s for s in chunk.split('.') if s.strip()]
    if len(sentences) < 2:
        return 1.0
    
    try:
        embeddings = chunker.similarity_model.encode(sentences)
        similarities = []
        for i in range(len(embeddings)-1):
            sim = np.dot(embeddings[i], embeddings[i+1]) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i+1]))
            similarities.append(sim)
        return np.mean(similarities) if similarities else 1.0
    except:
        return 0.8  # Fallback value

def test_retrieval(chunker, chunks, original_text, strategy):
    """Test retrieval functionality"""
    st.subheader("🔍 Retrieval Test")
    
    query = st.text_input("Test query:", "What are the main concepts?")
    
    if st.button("Test Retrieval", key="retrieval_test"):
        try:
            # Build simple RAG system for demo
            rag_system = AdaptiveRAGSystem(chunking_strategy=strategy)
            rag_system.build_knowledge_base([original_text])
            
            # Perform retrieval
            results = rag_system.retrieve(query, k=3)
            
            st.write("**Top Results:**")
            for i, result in enumerate(results, 1):
                score_percent = result['score'] * 100
                st.markdown(f"""
                <div class="chunk-box">
                <strong>Result {i} (Score: {score_percent:.1f}%)</strong><br>
                {result['chunk'][:200]}...
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.warning(f"Retrieval test simplified due to: {str(e)}")
            # Fallback: simple keyword-based retrieval
            st.info("**Simple keyword-based results:**")
            query_words = set(query.lower().split())
            for i, chunk in enumerate(chunks[:3], 1):
                chunk_words = set(chunk.lower().split())
                common_words = query_words.intersection(chunk_words)
                if common_words:
                    score = len(common_words) / len(query_words)
                    st.markdown(f"""
                    <div class="chunk-box">
                    <strong>Chunk {i} (Keyword match: {score:.1%})</strong><br>
                    {chunk[:200]}...
                    </div>
                    """, unsafe_allow_html=True)

def compare_strategies(text, threshold, max_size, min_size):
    """Compare both chunking strategies"""
    chunker = AdaptiveChunker(semantic_threshold=threshold)
    
    # Generate chunks with both strategies
    semantic_chunks = chunker.semantic_chunking(text, max_chunk_size=max_size)
    fixed_chunks = chunker.dynamic_size_chunking(text, target_chunk_size=max_size, overlap=50)
    
    # Evaluate both
    semantic_metrics = chunker.evaluate_chunk_quality(semantic_chunks)
    fixed_metrics = chunker.evaluate_chunk_quality(fixed_chunks)
    
    # Display comparison
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        coherence_diff = semantic_metrics['coherence_score'] - fixed_metrics['coherence_score']
        st.metric(
            "Coherence Score", 
            f"{semantic_metrics['coherence_score']:.3f}",
            f"{coherence_diff:+.3f}",
            delta_color="normal" if coherence_diff >= 0 else "inverse"
        )
    
    with col2:
        chunk_diff = semantic_metrics['num_chunks'] - fixed_metrics['num_chunks']
        st.metric(
            "Number of Chunks",
            semantic_metrics['num_chunks'],
            f"{chunk_diff:+d}",
            delta_color="inverse" if chunk_diff > 0 else "normal"
        )
    
    with col3:
        length_std_diff = semantic_metrics['length_std'] - fixed_metrics['length_std']
        st.metric(
            "Length Consistency",
            f"{semantic_metrics['length_std']:.1f}",
            f"{length_std_diff:+.1f}",
            delta_color="inverse" if length_std_diff > 0 else "normal"
        )
    
    with col4:
        avg_len_diff = semantic_metrics['avg_chunk_length'] - fixed_metrics['avg_chunk_length']
        st.metric(
            "Avg Chunk Length",
            f"{semantic_metrics['avg_chunk_length']:.1f}",
            f"{avg_len_diff:+.1f}",
            delta_color="normal"
        )
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Chunk size distribution
    semantic_sizes = [len(chunk.split()) for chunk in semantic_chunks]
    fixed_sizes = [len(chunk.split()) for chunk in fixed_chunks]
    
    axes[0].hist(semantic_sizes, alpha=0.7, label='Semantic', bins=10)
    axes[0].hist(fixed_sizes, alpha=0.7, label='Fixed', bins=10)
    axes[0].set_xlabel('Chunk Size (words)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Chunk Size Distribution')
    axes[0].legend()
    
    # Metrics comparison
    metrics_names = ['Coherence', 'Chunk Count', 'Length Std']
    semantic_values = [
        semantic_metrics['coherence_score'],
        semantic_metrics['num_chunks'] / 10,  # Normalize for display
        semantic_metrics['length_std'] / 10   # Normalize for display
    ]
    fixed_values = [
        fixed_metrics['coherence_score'],
        fixed_metrics['num_chunks'] / 10,
        fixed_metrics['length_std'] / 10
    ]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    axes[1].bar(x - width/2, semantic_values, width, label='Semantic', alpha=0.7)
    axes[1].bar(x + width/2, fixed_values, width, label='Fixed', alpha=0.7)
    axes[1].set_xlabel('Metrics')
    axes[1].set_ylabel('Normalized Scores')
    axes[1].set_title('Strategy Comparison')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(metrics_names)
    axes[1].legend()
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Recommendation
    st.subheader("🎯 Recommendation")
    
    if semantic_metrics['coherence_score'] > fixed_metrics['coherence_score'] + 0.1:
        st.success("""
        **Use Semantic Chunking** - Better for this document because:
        - Higher coherence scores indicate better topic preservation
        - More natural chunk boundaries
        - Better context retention for retrieval
        """)
    else:
        st.info("""
        **Consider Fixed Chunking** - Comparable performance because:
        - Document has uniform structure
        - Semantic boundaries are less clear
        - Fixed chunking provides more predictable results
        """)

if __name__ == "__main__":
    main()