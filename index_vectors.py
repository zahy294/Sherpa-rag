"""index_vectors.py

Chunk MSMARCO-XI passages, embed with FastEmbed, index in LanceDB,
and build companion BM25 sparse index.

Chunking strategies available:
  - fixed:    naive fixed-size character windows with overlap (baseline)
  - semantic: sentence-boundary splitting only, no aggregation (baseline)
  - hybrid:   sentence-boundary detection + greedy aggregation to a target
              length + sentence-level sliding overlap (recommended, default)
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Any

import lancedb
import numpy as np
from datasets import load_dataset
from fastembed import TextEmbedding
from lancedb.index import HnswSq

HINDI_VAL = "hf://datasets/ai4bharat/MSMARCO-XI/validation/hinval.parquet"
LANCEDB_PATH = Path("./lancedb_data")
TABLE_NAME = "msmarco_chunks"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SAMPLE_ROWS = 6000

# Language-aware sentence terminators: Hindi Purna Viram (।), Latin '.', '?', '!', newline.
SENTENCE_SPLIT = re.compile(r"(?<=[।.!?\n])\s+")

DEFAULT_TARGET_CHARS = 300
DEFAULT_OVERLAP_SENTENCES = 1
DEFAULT_FIXED_CHUNK_SIZE = 300
DEFAULT_FIXED_OVERLAP = 50


# ----------------------------------------------------------------------
# Chunking strategies
# ----------------------------------------------------------------------


def fixed_chunking(
    passage: str,
    chunk_size: int = DEFAULT_FIXED_CHUNK_SIZE,
    overlap: int = DEFAULT_FIXED_OVERLAP,
) -> list[str]:
    """Baseline: naive fixed-size character windows with overlap.

    Kept only as a comparison point — not word/grapheme aware, can split
    Devanagari conjuncts mid-cluster. Do not use as the primary strategy.
    """
    text = passage.strip()
    if not text:
        return []
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def semantic_only_chunking(passage: str) -> list[str]:
    """Baseline: pure sentence-boundary splitting, no aggregation.

    Each sentence becomes its own chunk — no overlap, no length targeting. Kept
    as a comparison point; tends to over-fragment short sentences.
    """
    text = passage.strip()
    if not text:
        return []
    return [s.strip() for s in SENTENCE_SPLIT.split(text) if s.strip()]


def hybrid_semantic_window_chunking(
    passage: str,
    target_chars: int = DEFAULT_TARGET_CHARS,
    overlap_sentences: int = DEFAULT_OVERLAP_SENTENCES,
) -> list[str]:
    """Recommended strategy.

    Splits on language-aware sentence boundaries first (never cuts mid-word or
    mid-grapheme, unlike fixed_chunking), then greedily aggregates sentences
    into ~target_chars windows. The last `overlap_sentences` sentences of each
    window carry into the next chunk, so a query whose answer straddles a
    sentence boundary isn't lost.
    """
    text = passage.strip()
    if not text:
        return []

    sentences = [s.strip() for s in SENTENCE_SPLIT.split(text) if s.strip()]
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sent in sentences:
        if current and current_len + len(sent) > target_chars:
            chunks.append(" ".join(current))
            current = current[-overlap_sentences:] if overlap_sentences else []
            current_len = sum(len(s) for s in current)
        current.append(sent)
        current_len += len(sent)

    if current:
        chunks.append(" ".join(current))

    return chunks


CHUNKING_STRATEGIES = {
    "fixed": fixed_chunking,
    "semantic": semantic_only_chunking,
    "hybrid": hybrid_semantic_window_chunking,
}


# ----------------------------------------------------------------------
# Row -> chunk records
# ----------------------------------------------------------------------


def build_chunks_for_row(
    row: dict[str, Any], method: str
) -> list[dict[str, Any]]:
    """Apply the named chunking strategy to every passage in a dataset row — both

    Translated_passages (Hindi) and English_passages (English) — so English
    queries have native-language content to match against, not only Hindi.
    Previously only Hindi was indexed.
    """
    chunk_fn = CHUNKING_STRATEGIES[method]
    passages = row["passages"]
    records: list[dict[str, Any]] = []

    passage_sets = [
        ("hi", passages["Translated_passages"]),
        ("en", passages["English_passages"]),
    ]

    for content_lang, passage_list in passage_sets:
        for passage_idx, (text, selected) in enumerate(
            zip(passage_list, passages["is_selected"])
        ):
            text = text.strip()
            if not text:
                continue

            doc_id = f"{row['query_id']}_{content_lang}_p{passage_idx}"
            for chunk_idx, chunk_text in enumerate(chunk_fn(text)):
                records.append({
                    "text": chunk_text,
                    "original_text": text,
                    "chunking_method": method,
                    "content_lang": content_lang,
                    "doc_id": doc_id,
                    "chunk_id": f"{doc_id}_c{chunk_idx}",
                    "query_id": row["query_id"],
                    "Eng_Query": row["Eng_Query"],
                    "query": row["query"],
                    "passage_idx": passage_idx,
                    "chunk_idx": chunk_idx,
                    "is_selected": bool(selected),
                    "source_lang": row["source_lang"],
                    "target_lang": row["target_lang"],
                })
    return records


# ----------------------------------------------------------------------
# Data loading / embedding / indexing
# ----------------------------------------------------------------------


def load_sample_rows(num_rows: int = SAMPLE_ROWS) -> list[dict[str, Any]]:
    dataset = load_dataset(
        "parquet", data_files=HINDI_VAL, split=f"train[:{num_rows}]"
    )
    return [dict(row) for row in dataset]


def embed_chunks(
    embedder: TextEmbedding,
    records: list[dict[str, Any]],
    batch_size: int = 128,
) -> list[dict[str, Any]]:
    texts = [r["text"] for r in records]
    total = len(texts)
    vectors: list[np.ndarray] = []
    t_start = time.perf_counter()

    for i, v in enumerate(embedder.embed(texts, batch_size=batch_size)):
        vectors.append(v)
        if (i + 1) % 5000 == 0 or (i + 1) == total:
            elapsed = time.perf_counter() - t_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (total - (i + 1)) / rate if rate > 0 else 0
            print(
                f"  embedded {i + 1}/{total} ({rate:.0f}/s, ETA"
                f" {eta:.0f}s)"
            )

    return [
        {**r, "vector": np.asarray(v, dtype=np.float32)}
        for r, v in zip(records, vectors)
    ]


def _partition_count(n_rows: int) -> int:
    """Rough HNSW partition heuristic: too few partitions on a large table

    hurts search latency, too many partitions on a small table hurts
    recall and wastes index-build time. sqrt(n) is a reasonable default
    starting point for tables in the low tens-of-thousands range.
    """
    return max(1, int(n_rows**0.5))


def create_vector_table(
    db: lancedb.DBConnection,
    records: list[dict[str, Any]],
) -> lancedb.table.Table:
    table = db.create_table(TABLE_NAME, records, mode="overwrite")
    n_partitions = _partition_count(len(records))
    table.create_index(
        "vector",
        config=HnswSq(
            distance_type="cosine",
            num_partitions=n_partitions,
            m=16,
            ef_construction=200,
        ),
    )
    return table


def benchmark_search(
    table: lancedb.table.Table,
    embedder: TextEmbedding,
    queries: list[str],
    *,
    warmup_runs: int = 2,
) -> None:
    print("\n" + "=" * 60)
    print("Vector Search Benchmark (retrieval-only latency)")
    print("=" * 60)

    for query in queries:
        query_vector = np.asarray(
            list(embedder.embed([query]))[0], dtype=np.float32
        )

        for _ in range(warmup_runs):
            table.search(query_vector).limit(3).to_list()

        start = time.perf_counter()
        results = table.search(query_vector).limit(3).to_list()
        elapsed_ms = (time.perf_counter() - start) * 1000

        print(f"\nQuery: {query}")
        print(
            f"Search latency: {elapsed_ms:.2f} ms   (target: <200ms"
            " end-to-end)"
        )

        for rank, hit in enumerate(results, start=1):
            preview = hit["text"][:120] + (
                "..." if len(hit["text"]) > 120 else ""
            )
            print(
                f"  #{rank} [{hit['chunking_method']}] doc_id={hit['doc_id']}"
                f" selected={hit['is_selected']}"
            )
            print(f"      {preview}")


def build_index(
    db: lancedb.DBConnection,
    methods: list[str],
) -> tuple[lancedb.table.Table, TextEmbedding, list[dict[str, Any]]]:
    print(f"Loading {SAMPLE_ROWS} rows from {HINDI_VAL} ...")
    rows = load_sample_rows(SAMPLE_ROWS)
    print(f"Loaded {len(rows)} rows.")

    print(f"Chunking passages with strategies: {methods} ...")
    all_chunks: list[dict[str, Any]] = []
    for row in rows:
        for method in methods:
            all_chunks.extend(build_chunks_for_row(row, method))
    print(
        f"Generated {len(all_chunks)} chunks across {len(methods)}"
        " strategy/strategies."
    )

    est_vector_mb = (len(all_chunks) * 384 * 4) / (1024**2)
    print(
        f"Estimated vector memory at deploy time: ~{est_vector_mb:.0f} MB"
        " (384-dim float32)"
    )
    if est_vector_mb > 700:
        print(
            "WARNING: this may not fit comfortably in a 1GB free-tier"
            " deployment alongside the embedding model, BM25 index, and"
            " Prompt Guard model."
        )

    for method in methods:
        count = sum(1 for c in all_chunks if c["chunking_method"] == method)
        print(f"  {method}: {count} chunks")

    print("Building BM25 sparse index...")
    from sparse_index import BM25_INDEX_PATH, build_bm25_index, save_bm25_index

    bm25 = build_bm25_index(
        all_chunks
    )  # text+metadata only — no vectors needed
    save_bm25_index(bm25)
    print(f"BM25 index saved to {BM25_INDEX_PATH.resolve()}")

    print(f"Initializing FastEmbed model: {EMBED_MODEL}")
    embedder = TextEmbedding(model_name=EMBED_MODEL)

    print("Embedding chunks ...")
    t0 = time.perf_counter()
    indexed_chunks = embed_chunks(embedder, all_chunks)
    embed_secs = time.perf_counter() - t0
    print(f"Embedded {len(indexed_chunks)} chunks in {embed_secs:.1f}s.")

    print(
        f"Writing LanceDB table to {LANCEDB_PATH.resolve()} (overwriting"
        " existing table) ..."
    )
    table = create_vector_table(db, indexed_chunks)
    print(
        f"Table '{TABLE_NAME}' ready with HNSW index ({table.count_rows()}"
        " rows)."
    )
    return table, embedder, rows


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Build or open the MSMARCO vector & sparse indices."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Rebuild the indices even if a LanceDB table already exists. "
            "REQUIRED to pick up new chunking strategies or sparse index"
            " updates."
        ),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["hybrid"],
        choices=list(CHUNKING_STRATEGIES.keys()),
        help=(
            "Which chunking strategy/strategies to index. Default: hybrid only "
            "(recommended for production — indexing all three roughly triples "
            "row count and search latency for no retrieval-quality benefit). "
            "Pass multiple, e.g. --methods fixed semantic hybrid, for"
            " comparison/demo."
        ),
    )
    args = parser.parse_args()

    db = lancedb.connect(str(LANCEDB_PATH))

    if TABLE_NAME in db.list_tables() and not args.force:
        print(
            "Found existing LanceDB table. Opening table directly to skip"
            " re-embedding..."
        )
        print("(pass --force to rebuild with the current --methods selection)")
        table = db.open_table(TABLE_NAME)
        embedder = TextEmbedding(model_name=EMBED_MODEL)
        rows = load_sample_rows(2)
    else:
        table, embedder, rows = build_index(db, args.methods)

    test_queries = [
        rows[0]["query"],
        rows[1]["Eng_Query"].strip(),
        "potassium rich foods chart",
    ]
    benchmark_search(table, embedder, test_queries)


if __name__ == "__main__":
    main()