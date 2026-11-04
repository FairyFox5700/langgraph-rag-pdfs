"""Load PDFs from PDF_DIR via Azure Document Intelligence, chunk, embed,
and persist to Chroma.

Document Intelligence (prebuilt-layout, markdown output) gives us:
  - correct reading order text,
  - tables rendered as markdown tables, and
  - figure/image bounding boxes, which we crop and caption with a vision
    model so images become part of the searchable/answerable context.
"""
import sys
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config
from .doc_intelligence import analyze_pdf
from .models import get_chat_model, get_vector_store

# Markdown-aware separators so headings/tables/paragraphs stay as intact as
# possible before falling back to splitting mid-paragraph or mid-sentence.
MARKDOWN_SEPARATORS = ["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""]


def ingest() -> int:
    pdf_paths = sorted(config.PDF_DIR.glob("*.pdf"))
    if not pdf_paths:
        print(f"No PDFs found in {config.PDF_DIR}. Add some .pdf files and re-run.")
        return 0

    chat_model = get_chat_model() if config.CAPTION_IMAGES else None

    documents: List[Document] = []
    for pdf_path in pdf_paths:
        print(f"Analyzing {pdf_path.name} with Azure Document Intelligence ...")
        markdown = analyze_pdf(pdf_path, chat_model=chat_model)
        documents.append(Document(page_content=markdown, metadata={"source": pdf_path.name}))
        print(
            f"  Extracted {len(markdown):,} characters of markdown "
            "(text + tables + figure captions)."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        separators=MARKDOWN_SEPARATORS,
    )
    chunks = splitter.split_documents(documents)
    print(f"Split {len(documents)} document(s) into {len(chunks)} chunks.")

    vector_store = get_vector_store()
    vector_store.add_documents(chunks)
    print(f"Persisted embeddings to {config.CHROMA_DIR}")
    return len(chunks)


if __name__ == "__main__":
    sys.exit(0 if ingest() >= 0 else 1)
