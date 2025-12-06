import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import vectordb

def test_chromadb_handler():
    print("Initializing ChromaDBHandler...")
    handler = vectordb.ChromaDBHandler()
    
    print("Resetting collection...")
    handler.reset()
    
    # Mock data
    chunks = ["This is a test document.", "Banana is a fruit.", "Sky is blue."]
    metadatas = [
        {"filename": "test1.txt", "chunk_id": 0},
        {"filename": "test2.txt", "chunk_id": 0},
        {"filename": "test3.txt", "chunk_id": 0}
    ]
    # Mock embeddings (random for testing mechanism)
    embeddings = np.random.rand(3, 384) 
    
    print("Adding documents...")
    handler.add_documents(chunks, metadatas, embeddings)
    
    count = handler.count()
    print(f"Collection count: {count}")
    assert count == 3, f"Expected 3 documents, got {count}"
    
    print("Querying...")
    # Mock query embedding
    query_embedding = np.random.rand(384)
    results = handler.query(query_embedding, n_results=1)
    
    print("Query Results:", results)
    assert len(results['ids'][0]) == 1
    
    print("SUCCESS: ChromaDB Handler verification passed!")

if __name__ == "__main__":
    test_chromadb_handler()
