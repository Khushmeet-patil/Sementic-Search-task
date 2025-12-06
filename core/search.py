import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple

def search(query_embedding: np.ndarray, doc_embeddings: np.ndarray, top_k: int = 3) -> List[int]:
    if doc_embeddings.size == 0:
        return []

    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)
        
    similarities = cosine_similarity(query_embedding, doc_embeddings)
    
    scores = similarities[0]
    
    #Get indices and reverse sort
    top_indices = scores.argsort()[-top_k:][::-1]
    
    return top_indices.tolist()
