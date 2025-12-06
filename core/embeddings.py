from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

class EmbeddingGenerator:
    def __init__(self, model_name: str):
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            print("Falling back to 'all-MiniLM-L6-v2'")
            self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.array([])
            
        #Normalise
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings
