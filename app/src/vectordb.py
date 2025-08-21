import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Optional
import uuid
import os

class VectorDB:
    def __init__(self, collection_name: str):
        persist_dir = os.path.join(os.path.dirname(__file__), "..", "..", "chroma_store")
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.Client(
            ChromaSettings(
                is_persistent=True,
                persist_directory=persist_dir
            )
        )
        self.col = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add(self, texts: List[str], embeddings, metadatas: Optional[List[Dict]] = None, ids: Optional[List[str]] = None):
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        if metadatas is None:
            metadatas = [{} for _ in texts]
        self.col.add(documents=texts, embeddings=embeddings.tolist(), metadatas=metadatas, ids=ids)

    def query(self, query_embedding, top_k: int = 5) -> Dict:
        res = self.col.query(query_embeddings=[query_embedding.tolist()], n_results=top_k)
        return res
