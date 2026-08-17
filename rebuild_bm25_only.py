# rebuild_bm25_only.py
import lancedb
from sparse_index import build_bm25_index, save_bm25_index

db = lancedb.connect("./lancedb_data")
table = db.open_table("msmarco_chunks")
records = table.to_pandas()[["text", "content_lang", "doc_id", "chunk_id", "query_id",
                               "Eng_Query", "query", "is_selected", "source_lang", "target_lang"]].to_dict("records")

print(f"Rebuilding BM25 from {len(records)} existing chunks (no re-embedding needed)...")
indices = build_bm25_index(records)
save_bm25_index(indices)
print("Done.")