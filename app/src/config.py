from dataclasses import dataclass

@dataclass
class Settings:
    embedding_model_name: str = "intfloat/multilingual-e5-base"  # good multilingual model
    collection_name: str = "multilang_edu"
    chunk_tokens: int = 400
    chunk_overlap: int = 80
    top_k: int = 5
    rerank_top_k: int = 3
    allow_openai: bool = True
    allow_deepl: bool = True

settings = Settings()
