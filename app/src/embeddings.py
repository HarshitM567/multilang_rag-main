from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

class MultilingualEmbedder:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        return np.array(self.model.encode(texts, normalize_embeddings=True))

    def embed_text(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]
