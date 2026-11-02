"""Load PDFs from PDF_DIR, chunk them, embed them, and persist to Chroma."""
import sys

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config
from .models import get_vector_store


def ingest() -> int:
    if not config.PDF_DIR.exists() or not any(config.PDF_DIR.glob("*.pdf")):
        print(f"No PDFs found in {config.PDF_DIR}. Add some .pdf files and re-run.")
        return 0

    print(f"Loading PDFs from {config.PDF_DIR} ...")
    loader = PyPDFDirectoryLoader(str(config.PDF_DIR))
    documents = loader.load()
    print(f"Loaded {len(documents)} pages.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")

    vector_store = get_vector_store()
    vector_store.add_documents(chunks)
    print(f"Persisted embeddings to {config.CHROMA_DIR}")
    return len(chunks)


if __name__ == "__main__":
    sys.exit(0 if ingest() >= 0 else 1)
