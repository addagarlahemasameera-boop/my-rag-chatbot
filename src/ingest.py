from src.config import COLLECTIONS
from src.data_loader import load_cmdb, load_graph_json
from src.vector_store import add_documents


def ingest_all() -> None:
    """
    Loads all data sources and ingests them into ChromaDB.
    Each source gets its own named collection.
    """
    print("\n" + "="*50)
    print("  HYBRID RAG — Data Ingestion")
    print("="*50)

    # CMDB (Excel)
    print("\n[1/6] app_cmdb.xlsx")
    add_documents(load_cmdb(), collection_name="cmdb")

    # All JSON sources
    json_collections = {k: v for k, v in COLLECTIONS.items() if k != "cmdb"}
    for i, (collection_name, filepath) in enumerate(json_collections.items(), start=2):
        print(f"\n[{i}/6] {filepath.split('/')[-1]}")
        add_documents(load_graph_json(filepath), collection_name=collection_name)

    print("\n" + "="*50)
    print("  Ingestion complete.")
    print("="*50 + "\n")


if __name__ == "__main__":
    ingest_all()