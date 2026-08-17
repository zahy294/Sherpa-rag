# verify_bm25s_manual.py
from sparse_index import build_bm25_index, bm25_search

test_records = [
    {"text": "दिल्ली भारत की राजधानी है।", "content_lang": "hi", "doc_id": "1", "chunk_id": "1_c0",
     "query_id": 1, "Eng_Query": "", "query": "", "is_selected": False, "source_lang": "", "target_lang": ""},
    {"text": "मुंबई एक बड़ा शहर है।", "content_lang": "hi", "doc_id": "2", "chunk_id": "2_c0",
     "query_id": 2, "Eng_Query": "", "query": "", "is_selected": False, "source_lang": "", "target_lang": ""},
]

indices = build_bm25_index(test_records)
results = bm25_search(indices, "दिल्ली भारत की राजधानी", top_k=2)
for r in results:
    print(f"score={r['bm25_score']:.3f}  text={r['text']}")