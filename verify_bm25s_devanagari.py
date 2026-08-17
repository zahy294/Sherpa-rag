# verify_bm25s_devanagari.py — run this first, takes 2 seconds
from text_utils import tokenize
import bm25s

test = "दिल्ली भारत की राजधानी है।"
safe_tokens = tokenize(test)
print(f"Our safe tokenizer produced: {safe_tokens}")

presegmented = " ".join(safe_tokens)
result = bm25s.tokenize([presegmented], stopwords=None)
print(f"bm25s vocab after handoff: {list(result.vocab.keys())}")