# Semantic Search Engine

## 📦 Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## 🚀 How to Run

Start the Streamlit application:

```bash
streamlit run main.py
```

## 🧠 How it Works

The application uses a **Retrieve & Rank** approach to find relevant information:

1.  **Chunking**: 
    -   The system splits large text documents into smaller, manageable segments (chunks).
    -   We use a sliding window approach with user-defined **Chunk Size** and **Overlap** to ensure context isn't lost at the edges of splits.

2.  **Embeddings**:
    -   Each text chunk is passed through a **Sentence Transformer** model (e.g., `all-MiniLM-L6-v2`).
    -   The model converts the text into a fixed-size numerical vector (embedding) that captures its semantic meaning.

3.  **Similarity Search**:
    -   When you ask a question, your query is also converted into an embedding.
    -   The system calculates the **Cosine Similarity** between your query vector and all document vectors.
    -   It returns the chunks with the highest similarity scores, effectively finding the most relevant answers based on meaning, not just keywords.

## 📅 Project Evolution

The project was developed in 4 key phases:

1.  **Phase 1: Simple Semantic Search**
    -   Basic partial implementation of embedding-based search using `sentence-transformers` and cosine similarity.

2.  **Phase 2: Dynamic Chunking**
    -   Empowered users to tune retrieval performance by editing **Chunk Size** and **Overlap** settings directly in the UI.

3.  **Phase 3: Production Readiness**
    -   Integrated **ChromaDB** to enable persistent vector storage, handling larger datasets and avoiding re-indexing on every restart.

4.  **Phase 4: Custom Data Support**
    -   Added the ability for users to upload their own `.txt` documents, making the search engine adaptable to any specific knowledge base.
