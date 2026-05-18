import chromadb
from src.config import CHROMA_DIR, TOP_K_RESULTS
from src.embedder import get_embedder


def _get_client() -> chromadb.PersistentClient:
    """Returns a persistent ChromaDB client."""
    return chromadb.PersistentClient(path=CHROMA_DIR)


def add_documents(texts: list[str], collection_name: str) -> None:
    """
    Embeds and stores a list of texts in the specified ChromaDB collection.
    Clears the collection first to avoid duplicate entries on re-ingestion.
    """
    client     = _get_client()
    embedder   = get_embedder()

    # Delete collection if it exists so re-runs don't duplicate data
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name     = collection_name,
        metadata = {"hnsw:space": "cosine"},
    )

    print(f"  Embedding {len(texts)} documents → '{collection_name}'...")
    vectors = embedder.embed_documents(texts)

    collection.add(
        documents  = texts,
        embeddings = vectors,
        ids        = [f"{collection_name}_{i}" for i in range(len(texts))],
    )
    print(f"  Stored {len(texts)} documents.")


def search(query: str, collection_name: str, n_results: int = TOP_K_RESULTS) -> list[str]:
    """
    Searches a ChromaDB collection for documents similar to the query.
    Returns a list of matching text strings.
    """
    client     = _get_client()
    embedder   = get_embedder()

    collection    = client.get_collection(name=collection_name)
    query_vector  = embedder.embed_query(query)

    results = collection.query(
        query_embeddings = [query_vector],
        n_results        = n_results,
    )
    return results["documents"][0]