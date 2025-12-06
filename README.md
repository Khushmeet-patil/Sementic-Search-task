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
