# LangGraph RAG over PDFs (Azure OpenAI + Azure Document Intelligence)

A LangGraph pipeline that answers questions grounded in your own PDF documents — including tables and images/figures — using:

- **Azure AI Document Intelligence** (`prebuilt-layout`, markdown output) for layout-aware parsing: correct reading order, tables rendered as real markdown tables, and figure/image bounding boxes.
- **Azure OpenAI (vision-capable chat deployment, e.g. `gpt-4o`)** to caption each detected figure/chart/photo so image content becomes searchable text.
- **Azure OpenAI embeddings** + **Chroma** as the local vector store.
- **LangGraph** `StateGraph` for the retrieve → generate flow.

## Architecture

```
PDFs (data/pdfs/)
   |
   v
Azure Document Intelligence (prebuilt-layout, markdown output)
   | text (reading order) + tables (markdown) + figure bounding boxes
   v
Crop each figure (PyMuPDF) --> caption with Azure OpenAI vision --> append to markdown
   |
   v
Chunk (markdown-aware) --> embed (Azure OpenAI) --> Chroma vector store
                                              |
User question --> [retrieve] --> [generate] --> Answer + sources
                     (LangGraph StateGraph)
```

- `src/config.py` — environment configuration
- `src/models.py` — Azure OpenAI chat/embedding model + Chroma vector store factories
- `src/doc_intelligence.py` — Document Intelligence parsing + figure cropping/captioning
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
2. Copy `.env.example` to `.env` and fill in:
   - Your **Azure OpenAI** endpoint, API key, and deployment names for a vision-capable chat model (e.g. `gpt-4o`) and an embedding model (e.g. `text-embedding-3-large`).
   - Your **Azure AI Document Intelligence** endpoint and API key (create a "Document Intelligence" resource in the Azure portal or AI Foundry — the free F0 tier works for testing).
3. PDFs go in `data/pdfs/`. Two real, public-domain sample PDFs (U.S. government reports, rich in tables and charts) are included to get you started:
   - `census-income-poverty-2022.pdf` — U.S. Census Bureau income & poverty report (data tables + bar/line charts)
   - `eia-aeo2023-presentation.pdf` — U.S. EIA Annual Energy Outlook presentation (charts, diagrams, projections)

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
- Set `CAPTION_IMAGES=false` in `.env` to skip vision captioning (faster/cheaper, text + tables only).
- Adjust chunk size/overlap in `src/ingest.py` and the number of retrieved chunks (`k`) in `src/graph.py` to tune retrieval quality.
- Extend `src/graph.py` with more nodes (e.g. query rewriting, re-ranking, conditional routing) as needed — it's a standard LangGraph `StateGraph`.
