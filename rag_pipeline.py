import os
import time
import lancedb
import numpy as np
from dotenv import load_dotenv
from fastembed import TextEmbedding
from groq import Groq

from rrf import reciprocal_rank_fusion
from sparse_index import bm25_search, load_bm25_index

# 1. Load environment variables
load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY is missing from your .env file!")

# 2. Configuration
LANCEDB_PATH = "./lancedb_data"
TABLE_NAME = "msmarco_chunks"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
GROQ_MODEL = "llama-3.1-8b-instant"

print("Connecting to LanceDB, loading FastEmbed model and BM25 index...")
db = lancedb.connect(LANCEDB_PATH)
table = db.open_table(TABLE_NAME)
embedder = TextEmbedding(model_name=EMBED_MODEL)
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Load BM25 sparse index once at module level
_bm25_indices = load_bm25_index()


def retrieve_context(
    query: str, top_k: int = 3, k: int = None
) -> tuple[list[dict], float, float]:
    """Retrieve top-k relevant text chunks using hybrid search (Dense HNSW + BM25 Sparse)

    fused via Reciprocal Rank Fusion (RRF). Returns:
        - fused: Top-k reranked items for context generation.
        - retrieval_ms: Elapsed retrieval wall-clock time in ms.
        - best_dense_score: Top cosine similarity score from the expanded candidate pool.
    Accepts either 'top_k' or 'k' for benchmark/guardrail compatibility.
    """
    if k is not None:
        top_k = k

    t0 = time.perf_counter()
    query_vector = np.asarray(list(embedder.embed([query]))[0], dtype=np.float32)

    # Dense search — widen the candidate pool beyond top_k so fusion has
    # real material to combine, not just the final answer set twice
    dense_hits = table.search(query_vector).limit(top_k * 4).to_list()
    dense_results = []
    for hit in dense_hits:
        distance = hit.get("_distance", 1.0)
        score = max(0.0, min(1.0, 1.0 - float(distance)))
        dense_results.append({"text": hit["text"], "score": round(score, 4)})

    # Capture absolute best dense vector similarity across candidate pool before fusion
    best_dense_score = dense_results[0]["score"] if dense_results else 0.0

    # Sparse search
    sparse_results = bm25_search(_bm25_indices, query, top_k=top_k * 4)

    # Fuse by rank (RRF) — not by raw score
    fused = reciprocal_rank_fusion(dense_results, sparse_results, top_k=top_k)

    # Preserve raw dense cosine score on fused hits where available
    dense_score_by_text = {d["text"]: d["score"] for d in dense_results}
    for item in fused:
        item["score"] = dense_score_by_text.get(item["text"], 0.0)

    retrieval_ms = (time.perf_counter() - t0) * 1000
    return fused, retrieval_ms, best_dense_score


def generate_rag_response(query: str):
    """Retrieve context and generate an LLM response via Groq."""
    print("=" * 60)
    print(f"❓ Query: {query}")

    # Step 1: Retrieval (unpacks 3-tuple return)
    retrieved_hits, retrieval_ms, best_score = retrieve_context(query, top_k=3)
    context_chunks = [item["text"] for item in retrieved_hits]
    context_block = "\n---\n".join(context_chunks)

    # Step 2: System Instructions
    system_prompt = (
        "You are an intelligent, concise AI assistant for a Voice RAG system. "
        "Answer the user's question clearly and accurately using ONLY the context provided. "
        "If the question is in Hindi, respond in natural Hindi. If in English, respond in English. "
        "Keep your response under 3 sentences so it sounds natural when spoken aloud."
    )

    user_prompt = f"Context:\n{context_block}\n\nQuestion: {query}"

    # Step 3: Fast LLM Inference via Groq
    t0 = time.perf_counter()
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=200,
    )
    generation_ms = (time.perf_counter() - t0) * 1000

    answer = response.choices[0].message.content
    total_ms = retrieval_ms + generation_ms

    print(f"⚡ Best Dense Score: {best_score:.4f}")
    print(f"⚡ Retrieval Latency : {retrieval_ms:.2f} ms")
    print(f"⚡ LLM Gen Latency   : {generation_ms:.2f} ms")
    print(f"🚀 Total RAG Latency : {total_ms:.2f} ms")
    print("-" * 60)
    print(f"💬 Answer:\n{answer}\n")
    return answer


if __name__ == "__main__":
    generate_rag_response("कॉर्पोरेशन क्या है?")
    generate_rag_response("Why did Rachel Carson write an obligation to endure?")