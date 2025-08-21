# Multi-Language RAG (Education Domain)

A retrieval-augmented generation (RAG) system that ingests documents in multiple languages
and answers in the user's preferred language, preserving context and cultural nuances.

## Features
- Multilingual embeddings via Sentence Transformers (e.g., `intfloat/multilingual-e5-base` or `paraphrase-multilingual-MiniLM-L12-v2`).
- ChromaDB vector store for local indexing & retrieval.
- Language detection and routing for queries & responses.
- Optional translation using OpenAI or DeepL; fallback to simple rule-based pass-through.
- Streamlit front-end for upload, indexing, and querying.
- Basic evaluation hooks with RAGAS.
- Domain: Education (sample K-12 docs provided in `app/data`).

## Quickstart
```bash
# 1) Create a virtual env (recommended)
python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate

# 2) Install dependencies (CPU ok)
pip install -r requirements.txt

# 3) (Optional) Set API keys for translation/generation
cp .env.example .env
# edit .env to add OPENAI_API_KEY and/or DEEPL_API_KEY

# 4) Run the app
streamlit run app/App.py
```

## How it works
- We use multilingual embeddings to index all documents directly, avoiding lossy intermediate translation.
- On query:
  1. Detect the query language.
  2. Embed the query and retrieve top-k chunks across all languages.
  3. Build a context and call the generator (OpenAI if configured; else an extractive answer is synthesized).
  4. Translate the final answer to the user's preferred language if needed.

## Chunking
- Sentence-first chunking with BlingFire and token-aware reflow.
- Defaults: 300–500 tokens with 50–100 overlap (configurable per language).

## Evaluation (RAGAS)
- The `app/src/eval/evaluate.py` provides a template to compute RAGAS metrics for a small set.
- Plug in your dataset and ground-truth answers to score faithfulness and answer relevance.

## Repo Layout
```
app/
  App.py                     # Streamlit UI
  src/
    config.py
    embeddings.py
    chunking.py
    vectordb.py
    translator.py
    generator.py
    pipeline/rag.py
    eval/evaluate.py
  data/                      # sample multilingual docs
requirements.txt
.env.example
README.md
```

## Notes
- For best retrieval, prefer `intfloat/multilingual-e5-large` if you have GPU RAM.
- Swap vector DB to Pinecone/Weaviate easily in `vectordb.py`.
- This project is designed for demo deployment on Streamlit Community Cloud or Hugging Face Spaces.
