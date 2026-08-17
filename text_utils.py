"""
Shared tokenization for the sparse (BM25) index.
"""
import re

# Split on whitespace and punctuation only — never on character class.
# Naive `\w`-based regexes exclude Devanagari vowel signs (matras) and
# the virama, shattering words like दिल्ली into ['द','ल','ल']. This
# splitter is script-agnostic by construction: it only ever removes
# whitespace and a fixed punctuation set, so it can't misfire on any
# script the same way `\w` does on Devanagari.
TOKEN_SPLIT = re.compile(r"[\s.,!?;:\"'()\[\]{}।॥]+")


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_SPLIT.split(text.strip().lower()) if t]