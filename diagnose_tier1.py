# diagnose_tier1.py
import time
from guardrails import check_unsafe_input
from rag_pipeline import retrieve_context

queries = ["What is a corporation?", "कॉर्पोरेशन क्या है?", "honesty or integrity definition"] * 5

guard_times, retrieval_times = [], []
for q in queries:
    _, _, guard_ms = check_unsafe_input(q)
    _, retrieval_ms, _ = retrieve_context(q, top_k=3)
    guard_times.append(guard_ms)
    retrieval_times.append(retrieval_ms)

print(f"Guard   — avg: {sum(guard_times)/len(guard_times):.1f}ms  min: {min(guard_times):.1f}  max: {max(guard_times):.1f}")
print(f"Retrieval — avg: {sum(retrieval_times)/len(retrieval_times):.1f}ms  min: {min(retrieval_times):.1f}  max: {max(retrieval_times):.1f}")