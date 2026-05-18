import re
from rank_bm25 import BM25Okapi
from src.vector_store import search as vector_search
from src.config import TOP_K_RESULTS

# ---------------------------------------------------------------------------
# METADATA FILTER RULES
#
# When a query is about structured metadata (Tier, Criticality, Status,
# RTO, RPO, Owner), BM25 is unreliable because:
#   - The query says "Tier-1 and Critical"
#   - The document says "Tier: Tier-1. Criticality: Critical."
#   - Token overlap is imperfect → wrong docs win
#
# Solution: detect metadata queries and apply direct substring filters
# to the candidate pool instead of (or after) BM25.
#
# Each rule has:
#   "keywords"  — tokens in the query that trigger this rule
#   "filters"   — list of (field_pattern, value_extractor) pairs
#                 field_pattern : regex that matches the field in the doc
#                 value_extractor: callable(query) → value string to match
# ---------------------------------------------------------------------------

# Maps query keywords → the exact substring to require in matching docs
METADATA_FILTERS = {
    # Tier
    "tier-1"  : "Tier: Tier-1",
    "tier-2"  : "Tier: Tier-2",
    "tier-3"  : "Tier: Tier-3",
    # Criticality
    "critical"   : "Criticality: Critical",
    "high"        : "Criticality: High",
    "medium"      : "Criticality: Medium",
    "low"         : "Criticality: Low",
    # Status
    "active"      : "Status: Active",
    "inactive"    : "Status: Inactive",
    "development" : "Status: In Development",
}


def _apply_metadata_filters(query: str, docs: list[str]) -> list[str]:
    """
    If the query contains metadata keywords, return only docs that satisfy
    ALL the detected filter conditions.

    Example:
        query = "Which applications are Tier-1 and Critical?"
        detected filters → ["Tier: Tier-1", "Criticality: Critical"]
        only docs containing BOTH strings are returned.

    If no metadata keywords are found, returns the original list unchanged.
    """
    tokens = query.lower().split()
    required = []

    for token in tokens:
        # strip punctuation from token edges before lookup
        clean = token.strip("?,.")
        if clean in METADATA_FILTERS:
            substring = METADATA_FILTERS[clean]
            if substring not in required:
                required.append(substring)

    if not required:
        return docs   # no metadata filter applicable — return as-is

    filtered = [
        doc for doc in docs
        if all(req in doc for req in required)
    ]

    # Safety: if filters are too strict and eliminate everything, fall back
    return filtered if filtered else docs


# ---------------------------------------------------------------------------
# ROUTING RULES
# First matching rule wins. Determines which collections to search.
# ---------------------------------------------------------------------------

ROUTING_RULES = [
    {
        "keywords": {
            "depend", "depends", "dependency", "dependencies",
            "uses", "consumes", "calls", "connects", "linked",
            "upstream", "downstream", "integration", "integrated",
        },
        "collections": [
            ("dependencies", 17),
            ("cmdb",          5),
        ],
    },
    {
        "keywords": {
            "tier", "tier-1", "tier-2", "tier-3",
            "critical", "criticality",
            "status", "active", "inactive", "development",
            "rto", "rpo",
        },
        "collections": [
            ("cmdb", 16),
        ],
    },
    {
        "keywords": {
            "owner", "owns", "owned", "contact",
            "responsible", "team", "lead", "manager",
        },
        "collections": [
            ("owners", 20),
            ("cmdb",   16),
        ],
    },
    {
        "keywords": {
            "hosted", "hosting", "infrastructure", "cloud",
            "aws", "azure", "oci", "on-premise", "on-prem",
            "server", "database", "db",
        },
        "collections": [
            ("infrastructure", 16),
            ("applications",   10),
            ("cmdb",            8),
        ],
    },
]

FALLBACK_COLLECTIONS = [
    ("cmdb",           8),
    ("dependencies",   8),
    ("applications",   5),
    ("owners",         5),
    ("infrastructure", 5),
]


# ---------------------------------------------------------------------------
# Entity-name exact-match boost
# Pins documents containing the queried app name to the top of results.
# ---------------------------------------------------------------------------

def _extract_entity_names(query: str) -> list[str]:
    patterns = [
        r'\b[A-Za-z]+(?:_[A-Za-z0-9]+)+\b',   # underscore: API_Gateway
        r'\b[A-Z][a-z]+(?:[A-Z][a-z]*)+\b',    # CamelCase: WebPortal
        r'\b[A-Z]{3,}\b',                        # ALL-CAPS: IRIS, MDM
    ]
    seen, result = set(), []
    for pat in patterns:
        for e in re.findall(pat, query):
            low = e.lower()
            if low not in seen:
                seen.add(low)
                result.append(low)
    return result


def _boost_exact_matches(query: str, docs: list[str], top_k: int) -> list[str]:
    entities = _extract_entity_names(query)
    if not entities:
        return docs[:top_k]
    pinned, rest = [], []
    for doc in docs:
        doc_lower = doc.lower()
        if any(e in doc_lower for e in entities):
            pinned.append(doc)
        else:
            rest.append(doc)
    return (pinned + rest)[:top_k]


# ---------------------------------------------------------------------------
# Core pipeline helpers
# ---------------------------------------------------------------------------

def _select_collections(query: str) -> list[tuple[str, int]]:
    tokens = set(query.lower().split())
    for rule in ROUTING_RULES:
        if tokens & rule["keywords"]:
            return rule["collections"]
    return FALLBACK_COLLECTIONS


def _gather_all_docs(question: str) -> list[str]:
    collection_plan = _select_collections(question)
    seen, all_docs = {}, []
    for collection, n in collection_plan:
        try:
            docs = vector_search(question, collection_name=collection, n_results=n)
            for doc in docs:
                if doc not in seen:
                    seen[doc] = True
                    all_docs.append(doc)
        except Exception:
            pass
    return all_docs


def bm25_search(query: str, docs: list[str]) -> list[str]:
    """BM25 rerank — returns ALL docs ranked, caller trims to top_k."""
    if not docs:
        return []
    tokenized_docs  = [doc.lower().split() for doc in docs]
    tokenized_query = query.lower().split()
    bm25   = BM25Okapi(tokenized_docs)
    scores = bm25.get_scores(tokenized_query)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def hybrid_search(query: str, top_k: int = TOP_K_RESULTS) -> list[str]:
    """
    Pipeline:
        query
          → route to relevant collections
            → vector search  (semantic recall)
              → deduplicate
                → metadata filter  (exact field match for structured queries)
                  → BM25 rerank    (lexical precision on remaining candidates)
                    → entity boost (pin named app to top)
                      → top_k results
    """
    candidate_docs = _gather_all_docs(query)
    if not candidate_docs:
        return []

    # Step 1 — apply metadata filters (e.g. keep only Tier-1 + Critical docs)
    filtered_docs = _apply_metadata_filters(query, candidate_docs)

    # Step 2 — BM25 rerank the filtered pool
    bm25_ranked = bm25_search(query, filtered_docs)

    # Step 3 — pin exact entity name matches to top
    return _boost_exact_matches(query, bm25_ranked, top_k=top_k)
