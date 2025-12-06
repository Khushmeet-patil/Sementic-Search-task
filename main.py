import streamlit as st
import os
import numpy as np
from core import config, loader, chunker, search, vectordb
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

def process_documents(model, chunk_size, overlap, source_type="default", custom_docs=None):
    """Load, chunk, and embed documents."""
    
    # Clear previous state to avoid mixing data
    st.session_state.chunks = []
    st.session_state.metadatas = []
    st.session_state.doc_embeddings = np.array([])
    st.session_state.documents_loaded = False
    
    raw_docs = []
    if source_type == "default":
        with st.spinner("Loading default documents..."):
            raw_docs = loader.load_documents(config.DOCS_DIR)
            if not raw_docs:
                st.error(f"No .txt files found in {config.DOCS_DIR}")
                return
    elif source_type == "custom":
        if not custom_docs:
             st.error("No custom documents provided.")
             return
        raw_docs = custom_docs

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
                "chunk_id": chunk_id,
                "source": source_type # Track source
            })
        progress_bar.progress((i + 1) / total_docs)
        
    st.write(f"Generated {len(all_chunks)} chunks from {total_docs} files ({source_type}).")
    
    with st.spinner("Generating embeddings... (this may take a moment)"):
        embeddings = model.generate_embeddings(all_chunks)

    if st.session_state.use_chroma:
        with st.spinner("Indexing into ChromaDB..."):
            chroma_handler = vectordb.ChromaDBHandler()
            chroma_handler.reset() # Reset for fresh index on load
            chroma_handler.add_documents(all_chunks, all_metadatas, embeddings)
            st.session_state.chroma_handler = chroma_handler # Keep handler in session
    
    st.session_state.chunks = all_chunks
    st.session_state.metadatas = all_metadatas
    st.session_state.doc_embeddings = embeddings
    st.session_state.documents_loaded = True
    st.success(f"Indexing complete! ({source_type} data)")

def main():
    st.title("🔍 Semantic Search Engine")
    st.markdown("""
    This application allows you to search through text documents using semantic meaning rather than just keyword matching.
    
    ### **How to Use:**
    1.  **Select Data Source (Sidebar):**
        *   **Default Docs:** Uses pre-loaded files from the `docs/` folder.
        *   **Custom Upload:** Allows you to upload your own `.txt` files.
    2.  **Load Data:** Click the **"Load & Index"** or **"Process Custom Data"** button in the sidebar.
    3.  **Search:** Enter your question in the search bar below to find relevant answers.
    
    ### **How it Works:**
    1.  **Chunking:** Text is split into small, overlapping segments.
    2.  **Embedding:** A Transformer model converts text into numerical vectors (meanings).
    3.  **Indexing:** Vectors are stored in memory or ChromaDB for fast retrieval.
    4.  **Similarity Search:** Your query is converted to a vector and compared against the database to find the closest matches.
    """)
    
    # Sidebar 
    with st.sidebar:
        st.header("Configuration")
        st.info(f"Model: {config.EMBEDDING_MODEL_NAME}")
        
        # 1. Select Data Source
        data_source = st.radio("Data Source", ["Default Docs", "Custom Upload"])
        
        chunk_size = st.slider("Chunk Size", min_value=10, max_value=200, value=config.CHUNK_SIZE, step=10)
        overlap = st.slider("Overlap", min_value=0, max_value=100, value=config.OVERLAP, step=5)
        
        st.session_state.use_chroma = st.checkbox("Use Vector Database (ChromaDB)", value=False)
        
        if data_source == "Default Docs":
            if st.button("Load & Index Default Docs"):
                model = load_embedding_model()
                process_documents(model, chunk_size, overlap, source_type="default")
        
        else: # Custom Upload
            uploaded_files = st.file_uploader("Upload .txt files", type=['txt'], accept_multiple_files=True)
            if uploaded_files and st.button("Process Custom Data"):
                model = load_embedding_model()
                # Create a pseudo-loader for uploaded files
                raw_docs = []
                for uploaded_file in uploaded_files:
                    string_data = uploaded_file.getvalue().decode("utf-8")
                    raw_docs.append({"filename": uploaded_file.name, "content": string_data})
                
                process_documents(model, chunk_size, overlap, source_type="custom", custom_docs=raw_docs)
            
    #Mian Interface
    if st.session_state.documents_loaded:
        query = st.text_input("Enter your search query:", placeholder="e.g., How does the transformer architecture work?")
        
        if query:
            model = load_embedding_model()
            with st.spinner("Searching..."):
                query_embedding = model.generate_embeddings([query])[0]
                
                if st.session_state.get('use_chroma', False):
                    # ChromaDB Search
                    if 'chroma_handler' not in st.session_state:
                         st.session_state.chroma_handler = vectordb.ChromaDBHandler()
                    
                    results = st.session_state.chroma_handler.query(query_embedding, n_results=3)
                    
                    # ChromaDB structure: {'ids': [[]], 'distances': [[]], 'metadatas': [[]], 'documents': [[]]}
                    # We need to parse this back to our format. 
                    # Note: Chroma returns distances, not scores (unless we did 1-dist). 
                    # But wait, we used 'cosine' space, so it returns cosine distance (1 - similarity).
                    
                    # Let's map it.
                    top_indices = [] # Not really using indices here directly, but filling lists.
                    
                    # We can iterate through the results directly
                    found_chunks = results['documents'][0]
                    found_metadatas = results['metadatas'][0]
                    found_distances = results['distances'][0]
                    
                    st.subheader("Top Results (ChromaDB)")
                    for i in range(len(found_chunks)):
                        score = 1 - found_distances[i] # Approximate similarity
                        
                        chunk_text = found_chunks[i]
                        metadata = found_metadatas[i]
                        
                        with st.expander(f"Result {i+1} (Score: {score:.4f}) - {metadata['filename']}"):
                            st.markdown(f"**Chunk ID:** {metadata['chunk_id']}")
                            st.markdown(f"**Text:**\n\n{chunk_text}")

                else:
                    # Original Numpy Search
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
