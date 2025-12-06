import streamlit as st
import os
import numpy as np
from core import config, loader, chunker, search
from core.embeddings import EmbeddingGenerator

# Page title
st.set_page_config(
    page_title="Semantic Search Engine",
    page_icon="🔍",
    layout="wide"
)

# state condistions
if 'documents_loaded' not in st.session_state:
    st.session_state.documents_loaded = False
if 'chunks' not in st.session_state:
    st.session_state.chunks = []
if 'metadatas' not in st.session_state:
    st.session_state.metadatas = []
if 'doc_embeddings' not in st.session_state:
    st.session_state.doc_embeddings = np.array([])

@st.cache_resource
def load_embedding_model():
    """Load the embedding model once."""
    return EmbeddingGenerator(config.EMBEDDING_MODEL_NAME)

def process_documents(model, chunk_size, overlap):
    """Load, chunk, and embed documents."""
    with st.spinner("Loading documents..."):
        raw_docs = loader.load_documents(config.DOCS_DIR)
    
    if not raw_docs:
        st.error(f"No .txt files found in {config.DOCS_DIR}")
        return

    all_chunks = []
    all_metadatas = []
    
    progress_bar = st.progress(0)
    total_docs = len(raw_docs)
    
    for i, doc in enumerate(raw_docs):
        chunks = chunker.chunk_text(doc['content'], chunk_size, overlap)
        for chunk_id, chunk_text in enumerate(chunks):
            all_chunks.append(chunk_text)
            all_metadatas.append({
                "filename": doc['filename'],
                "chunk_id": chunk_id
            })
        progress_bar.progress((i + 1) / total_docs)
        
    st.write(f"Generated {len(all_chunks)} chunks from {total_docs} files.")
    
    with st.spinner("Generating embeddings... (this may take a moment)"):
        embeddings = model.generate_embeddings(all_chunks)
        
    st.session_state.chunks = all_chunks
    st.session_state.metadatas = all_metadatas
    st.session_state.doc_embeddings = embeddings
    st.session_state.documents_loaded = True
    st.success("Indexing complete!")

def main():
    st.title("🔍 Semantic Search Engine")
    st.markdown("""
    This application allows you to search through text documents using semantic meaning rather than just keyword matching.
    
    **How it works:**
    1. Documents are loaded from the `docs/` folder.
    2. Text is split into overlapping chunks.
    3. Embeddings are generated using a Transformer model.
    4. Your query is compared against all chunks using Cosine Similarity.
    """)
    
    #Sidebar 
    with st.sidebar:
        st.header("Configuration")
        st.info(f"Model: {config.EMBEDDING_MODEL_NAME}")
        
        chunk_size = st.slider("Chunk Size", min_value=10, max_value=200, value=config.CHUNK_SIZE, step=10)
        overlap = st.slider("Overlap", min_value=0, max_value=100, value=config.OVERLAP, step=5)
        
        if st.button("Load & Index Documents"):
            model = load_embedding_model()
            process_documents(model, chunk_size, overlap)
            
    #Mian Interface
    if st.session_state.documents_loaded:
        query = st.text_input("Enter your search query:", placeholder="e.g., How does the transformer architecture work?")
        
        if query:
            model = load_embedding_model()
            with st.spinner("Searching..."):
                query_embedding = model.generate_embeddings([query])[0]
                top_indices = search.search(query_embedding, st.session_state.doc_embeddings, top_k=3)
                
            st.subheader("Top Results")
            for rank, idx in enumerate(top_indices):
                score = np.dot(query_embedding, st.session_state.doc_embeddings[idx]) # Re-calculating score for display since search returns only indices
                # Note: search.py uses sklearn cosine_similarity which returns 1.0 for identical vectors.
                # Since we normalized embeddings, dot product is equivalent to cosine similarity.
                
                chunk_text = st.session_state.chunks[idx]
                metadata = st.session_state.metadatas[idx]
                
                with st.expander(f"Result {rank+1} (Score: {score:.4f}) - {metadata['filename']}"):
                    st.markdown(f"**Chunk ID:** {metadata['chunk_id']}")
                    st.markdown(f"**Text:**\n\n{chunk_text}")
    else:
        st.warning("Please load documents using the sidebar button to start searching.")

if __name__ == "__main__":
    main()
