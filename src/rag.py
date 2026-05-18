from langchain_ollama import OllamaLLM
from src.config import LLM_MODEL
from src.retriever import hybrid_search


def ask(question: str) -> tuple[str, list[str]]:
    """
    Full Hybrid RAG pipeline:
    1. Hybrid search (vector + BM25) finds relevant context
    2. Mistral generates answer grounded in that context

    Returns:
        answer       — Mistral's response
        context_docs — the chunks used to answer
    """
    # Step 1: Retrieve context using hybrid search
    context_docs = hybrid_search(question)

    if not context_docs:
        return "No relevant information found in the knowledge base.", []

    # Step 2: Build prompt
    context = "\n".join(context_docs)

    prompt = f"""You are an intelligent assistant for a banking IT team.
Your job is to answer questions about applications, owners, infrastructure, and dependencies.
Use ONLY the context provided below. Do not make up information.
If the answer is not in the context, say: "I don't have that information."

--- CONTEXT START ---
{context}
--- CONTEXT END ---

Question: {question}

Answer:"""

    # Step 3: Generate answer with Mistral
    llm      = OllamaLLM(model=LLM_MODEL)
    response = llm.invoke(prompt)

    return response.strip(), context_docs