import os

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma")

# ── Data Files ─────────────────────────────────────────────────────────────
CMDB_FILE                    = os.path.join(DATA_DIR, "app_cmdb.xlsx")
APPLICATIONS_JSON            = os.path.join(DATA_DIR, "graphrag_applications.json")
BUSINESS_CAPABILITIES_JSON   = os.path.join(DATA_DIR, "graphrag_business_capabilities.json")
DEPENDENCIES_JSON            = os.path.join(DATA_DIR, "graphrag_dependencies.json")
INFRASTRUCTURE_JSON          = os.path.join(DATA_DIR, "graphrag_infrastructure.json")
OWNERS_JSON                  = os.path.join(DATA_DIR, "graphrag_owners.json")

# ── Models ─────────────────────────────────────────────────────────────────
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL   = "mistral"

# ── ChromaDB Collections ───────────────────────────────────────────────────
COLLECTIONS = {
    "cmdb"                  : CMDB_FILE,
    "applications"          : APPLICATIONS_JSON,
    "business_capabilities" : BUSINESS_CAPABILITIES_JSON,
    "dependencies"          : DEPENDENCIES_JSON,
    "infrastructure"        : INFRASTRUCTURE_JSON,
    "owners"                : OWNERS_JSON,
}

# ── RAG Settings ───────────────────────────────────────────────────────────
TOP_K_RESULTS = 5