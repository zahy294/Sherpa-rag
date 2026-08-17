"""
Reciprocal Rank Fusion — combines dense + sparse rankings by rank
position, not raw score, since the two scoring systems aren't
comparable on the same scale.
"""


def reciprocal_rank_fusion(
    dense_results: list[dict],
    sparse_results: list[dict],
    k: int = 60,
    top_k: int = 5,
) -> list[dict]:
    fused_scores: dict[str, float] = {}
    text_lookup: dict[str, dict] = {}

    for rank, item in enumerate(dense_results):
        key = item["text"]
        fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        text_lookup[key] = item

    for rank, item in enumerate(sparse_results):
        key = item["text"]
        fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        text_lookup.setdefault(key, item)

    ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [{**text_lookup[key], "rrf_score": round(score, 6)} for key, score in ranked]