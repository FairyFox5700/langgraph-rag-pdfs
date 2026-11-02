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
