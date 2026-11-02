"""Shared Azure OpenAI model/embedding factories and the Chroma vector store."""
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_chroma import Chroma

from . import config


def get_embeddings() -> AzureOpenAIEmbeddings:
    config.require_azure_config()
    return AzureOpenAIEmbeddings(
        azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
        api_key=config.AZURE_OPENAI_API_KEY,
        api_version=config.AZURE_OPENAI_API_VERSION,
        azure_deployment=config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
    )


def get_chat_model(temperature: float = 0.0) -> AzureChatOpenAI:
    config.require_azure_config()
    return AzureChatOpenAI(
        azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
        api_key=config.AZURE_OPENAI_API_KEY,
        api_version=config.AZURE_OPENAI_API_VERSION,
        azure_deployment=config.AZURE_OPENAI_CHAT_DEPLOYMENT,
        temperature=temperature,
    )


def get_vector_store() -> Chroma:
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(config.CHROMA_DIR),
    )
