"""Centralized configuration loaded from environment variables (.env)."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent

AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
AZURE_OPENAI_CHAT_DEPLOYMENT = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.environ.get(
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large"
)

PDF_DIR = ROOT_DIR / os.environ.get("PDF_DIR", "data/pdfs")
CHROMA_DIR = ROOT_DIR / os.environ.get("CHROMA_DIR", "chroma_db")
COLLECTION_NAME = "pdf_rag"

# Azure AI Document Intelligence — used for layout-aware PDF parsing (tables,
# reading order, and figure/image bounding boxes).
AZURE_DOC_INTEL_ENDPOINT = os.environ.get("AZURE_DOC_INTEL_ENDPOINT", "")
AZURE_DOC_INTEL_API_KEY = os.environ.get("AZURE_DOC_INTEL_API_KEY", "")

# When true, each detected figure/image is cropped and sent to the Azure
# OpenAI chat deployment (must be vision-capable, e.g. gpt-4o) to generate a
# text caption/description that gets embedded alongside the surrounding text.
CAPTION_IMAGES = os.environ.get("CAPTION_IMAGES", "true").lower() in {"1", "true", "yes"}


def require_azure_config() -> None:
    """Fail fast with a clear message if required Azure OpenAI settings are missing."""
    missing = [
        name
        for name, value in [
            ("AZURE_OPENAI_ENDPOINT", AZURE_OPENAI_ENDPOINT),
            ("AZURE_OPENAI_API_KEY", AZURE_OPENAI_API_KEY),
        ]
        if not value
    ]
    if missing:
        raise RuntimeError(
            f"Missing required environment variables: {', '.join(missing)}. "
            "Copy .env.example to .env and fill in your Azure OpenAI details."
        )


def require_doc_intel_config() -> None:
    """Fail fast with a clear message if Document Intelligence settings are missing."""
    missing = [
        name
        for name, value in [
            ("AZURE_DOC_INTEL_ENDPOINT", AZURE_DOC_INTEL_ENDPOINT),
            ("AZURE_DOC_INTEL_API_KEY", AZURE_DOC_INTEL_API_KEY),
        ]
        if not value
    ]
    if missing:
        raise RuntimeError(
            f"Missing required environment variables: {', '.join(missing)}. "
            "Copy .env.example to .env and fill in your Azure AI Document "
            "Intelligence endpoint/key (create a Document Intelligence "
            "resource in the Azure portal or AI Foundry)."
        )
