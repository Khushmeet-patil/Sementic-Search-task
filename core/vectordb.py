import chromadb
from chromadb.config import Settings
import os
from typing import List, Dict, Any
import numpy as np
from core import config

class ChromaDBHandler:
    def __init__(self):
        # Ensure the directory exists
        if not os.path.exists(config.CHROMA_DB_DIR):
            os.makedirs(config.CHROMA_DB_DIR)
            
        self.client = chromadb.PersistentClient(path=config.CHROMA_DB_DIR)
        
        self.collection = self.client.get_or_create_collection(
            name=config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"} # Use cosine similarity
        )

    def add_documents(self, chunks: List[str], metadatas: List[Dict[str, Any]], embeddings: np.ndarray):
        """
        Add documents to the collection.
        Embeddings are expected to be numpy array.
        """
        # Convert numpy embeddings to list of lists for ChromaDB
        embeddings_list = embeddings.tolist()
        
        # ID generation (simple sequential or based on chunk index)
        ids = [f"id_{meta['filename']}_{meta['chunk_id']}" for meta in metadatas]
        
        # ChromaDB requires non-empty lists
        if not ids:
            return

        self.collection.add(
            documents=chunks,
            embeddings=embeddings_list,
            metadatas=metadatas,
            ids=ids
        )

    def query(self, query_embedding: np.ndarray, n_results: int = 3) -> Dict[str, Any]:
        """
        Query the collection.
        Returns the raw result from ChromaDB.
        """
        # Convert query embedding to list
        query_embedding_list = query_embedding.tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding_list],
            n_results=n_results
        )
        return results

    def count(self):
        return self.collection.count()
        
    def reset(self):
        """
        Deletes and recreates the collection to clear data.
        """
        self.client.delete_collection(config.COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
