# LangGraph RAG over PDFs (Azure OpenAI)

A minimal, working LangGraph pipeline that answers questions grounded in your own PDF documents using Azure OpenAI for chat + embeddings and Chroma as the local vector store.

## Architecture

```
PDFs (data/pdfs/) --> chunk --> embed --> Chroma vector store
                                              |
User question --> [retrieve] --> [generate] --> Answer + sources
                     (LangGraph StateGraph)
```

- `src/config.py` — environment configuration
- `src/models.py` — Azure OpenAI chat/embedding model + Chroma vector store factories
- `src/ingest.py` — loads PDFs, chunks, embeds, persists to Chroma
- `src/graph.py` — the LangGraph `StateGraph` (retrieve -> generate)
- `main.py` — CLI entrypoint (`ingest` / `chat`)

## Setup

1. Create a virtual environment and install dependencies:
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and fill in your Azure OpenAI endpoint, API key, and deployment names for a chat model (e.g. `gpt-4o`) and an embedding model (e.g. `text-embedding-3-large`).
3. Drop your PDF files into `data/pdfs/`.

## Usage

Ingest PDFs (run once, or again whenever you add/change files):
```powershell
python main.py ingest
```

Start an interactive Q&A chat grounded in your PDFs:
```powershell
python main.py chat
```

## Notes

- The vector store is persisted locally under `chroma_db/` — delete that folder to start fresh.
- Adjust chunk size/overlap in `src/ingest.py` and the number of retrieved chunks (`k`) in `src/graph.py` to tune retrieval quality.
- Extend `src/graph.py` with more nodes (e.g. query rewriting, re-ranking, conditional routing) as needed — it's a standard LangGraph `StateGraph`.
