"""
BM25 sparse index — bm25s backend, with bm25s's own tokenizer bypassed
entirely. We build the token-ID structure it needs ourselves, using our
verified-safe tokenize() from text_utils.py — bm25s never sees raw text,
only pre-resolved integer IDs, so its internal splitting behavior (which
corrupted Devanagari when we tried feeding it pre-cleaned text before)
can't affect correctness at all.

Still split by language: a query only scans its own language's bucket.
"""
import pickle
import re
from pathlib import Path

import bm25s
from bm25s.tokenization import Tokenized

from text_utils import tokenize

BM25_INDEX_PATH = Path("./bm25_index.pkl")
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")


def _detect_lang(text: str) -> str:
    return "hi" if _DEVANAGARI_RE.search(text) else "en"


def _tokenize_corpus(records: list[dict]) -> tuple[Tokenized, dict[str, int]]:
    vocab: dict[str, int] = {}
    ids: list[list[int]] = []
    for r in records:
        doc_ids = []
        for tok in tokenize(r["text"]):
            if tok not in vocab:
                vocab[tok] = len(vocab)
            doc_ids.append(vocab[tok])
        ids.append(doc_ids)
    return Tokenized(ids=ids, vocab=vocab), vocab


def build_bm25_index(records: list[dict]) -> dict:
    """Returns {'hi': (retriever, records, vocab), 'en': (...)}"""
    by_lang: dict[str, list[dict]] = {"hi": [], "en": []}
    for r in records:
        lang = r.get("content_lang") or _detect_lang(r["text"])
        by_lang[lang].append(r)

    indices = {}
    for lang, lang_records in by_lang.items():
        if not lang_records:
            indices[lang] = (None, [], {})
            continue
        tokenized, vocab = _tokenize_corpus(lang_records)
        retriever = bm25s.BM25()
        retriever.index(tokenized)
        indices[lang] = (retriever, lang_records, vocab)
    return indices


def save_bm25_index(indices: dict, path: Path = BM25_INDEX_PATH) -> None:
    with open(path, "wb") as f:
        pickle.dump(indices, f)


def load_bm25_index(path: Path = BM25_INDEX_PATH) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def bm25_search(indices: dict, query: str, top_k: int = 10) -> list[dict]:
    query_lang = _detect_lang(query)
    retriever, records, vocab = indices.get(query_lang, (None, [], {}))
    if retriever is None or not records:
        return []

    query_ids = [vocab[t] for t in tokenize(query) if t in vocab]
    if not query_ids:
        return []

    query_tokenized = Tokenized(ids=[query_ids], vocab=vocab)
    doc_idx, scores = retriever.retrieve(query_tokenized, k=min(top_k, len(records)))
    return [
        {
            "text": records[int(i)]["text"],
            "bm25_score": float(s),
            **{k: v for k, v in records[int(i)].items() if k not in ("text", "vector")},
        }
        for i, s in zip(doc_idx[0], scores[0])
    ]