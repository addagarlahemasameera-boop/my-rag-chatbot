from langchain_ollama import OllamaEmbeddings
from src.config import EMBED_MODEL


def get_embedder() -> OllamaEmbeddings:
    """Returns the embedding model defined in config."""
    return OllamaEmbeddings(model=EMBED_MODEL)