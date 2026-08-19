# Sherpa — Multilingual Voice RAG

**HH Goa 2026 — Task 02 Submission**
A voice-enabled RAG system for Hindi & English: speak a question, get a fast extracted answer instantly, then a polished, grounded answer — or a clear, honest refusal when the knowledge base doesn't have it.

`#RAGinGoa`

---

## What this is

Sherpa takes a spoken or typed question in Hindi or English, transcribes it, retrieves relevant passages from an indexed slice of [MSMARCO-XI](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI), and answers in two stages:

- **Tier 1 (extractive, instant):** the top-ranked retrieved passage, returned the moment safety and relevance checks pass — no LLM call on this path.
- **Tier 2 (generative, polished):** an LLM-generated, grounded answer in the query's own language, checked against the retrieved context before being shown. If it fails that check, Tier 1's answer stands as the answer of record instead.

Every answer — or refusal — is backed by a visible reason and a real, measured latency number.

---

## Pipeline

```
Voice input (mic)
   → Sarvam STT (saaras:v3, auto language detection)
   → [parallel] Prompt Guard (local ONNX, jailbreak/injection check)
              + Hybrid retrieval (dense HNSW + BM25 sparse, RRF fusion)
   → Gate 1: unsafe input → refuse
   → Gate 2: off-topic (retrieval score floor) → refuse (or general-knowledge
             fallback, opt-in via button, clearly labeled)
   → Tier 1: extracted answer (instant, no LLM)
   → Tier 2: LLM-generated answer (Groq) → Gate 3: groundedness check
             → grounded: shown as the polished answer
             → not grounded: Tier 1's extracted answer is kept instead
   → Sarvam TTS (auto language detection from answer script)
```

---

## Technical requirements — how each is met

### 1. Speech-to-text
Sarvam AI, `saaras:v3` model, `language_code="unknown"` for automatic Hindi/English/code-mixed detection.

### 2. Chunking — multiple strategies, not a single naive approach
Three chunking strategies are implemented and comparable in the codebase (`index_vectors.py`):

- **`fixed`** — baseline, naive fixed-size character windows with overlap. Kept only as a comparison point; not grapheme-aware.
- **`semantic`** — baseline, pure sentence-boundary splitting with no aggregation.
- **`hybrid`** *(production default)* — language-aware sentence-boundary detection (respects Hindi `।` and Latin punctuation), greedy aggregation into ~300-character windows, with sentence-level sliding overlap so answers spanning a sentence boundary aren't lost. Never splits mid-word or mid-grapheme, unlike the fixed baseline.

Each passage is indexed **in both Hindi (`Translated_passages`) and English (`English_passages`)**, tagged with `content_lang`, so English queries have native-language content to match against rather than relying solely on cross-lingual embedding similarity.

### 3. Latency target (<200ms)
The **Tier 1 path** — Prompt Guard check + hybrid retrieval + relevance gate, run concurrently — is the measured, budgeted number, and is what "under 200ms" refers to in this submission. It does not include LLM generation, STT, or TTS, each of which is reported separately as a distinct, honestly-labeled stage rather than folded into the headline claim. This mirrors how retrieval-latency claims are scoped across other Task 02 submissions we reviewed for reference, and reflects a real engineering constraint: LLM generation over a network API cannot reliably land under 200ms regardless of implementation, so it is not included in the budgeted metric.

### 4. Latency analytics (P50/P70/P100)
`benchmark.py` runs 33 test queries (a mix of real in-corpus queries pulled directly from the indexed dataset, deliberately off-topic negative controls, and jailbreak/injection test strings) and reports P50/P70/P100 per stage. See **Benchmark results** below.

### 5. Harness
`guardrails.py` wraps every model call in a structured harness:
- Pydantic-validated JSON output contracts (`GroundedAnswer`, `TwoTierUpdate`, `GeneralKnowledgeAnswer`) — no raw free-text parsing
- Retry logic with backoff, specific handling for rate limits (`RateLimitError`) vs. malformed output vs. token-budget truncation (`BadRequestError`, non-retried since it's deterministic at temperature=0.1)
- A model fallback story: the original model (`llama-3.1-8b-instant`) was decommissioned by Groq mid-development; the harness is parameterized by `model_name` specifically so this could be — and was — swapped without touching pipeline logic

### 6. Guardrails
Three independent, ordered gates, each with a visible reason surfaced in the UI:

| Gate | Mechanism | Reason codes |
|---|---|---|
| **Unsafe input** | `meta-llama/Llama-Prompt-Guard-2-86M`, quantized ONNX, run locally (no network round trip) | `unsafe_input` |
| **Off-topic** | Best raw dense cosine similarity across the full retrieval candidate pool, evaluated independently of RRF fusion rank | `off_topic` |
| **Ungrounded** | LLM self-reports `is_grounded` + `confidence`; both must clear a floor | `ungrounded`, `low_confidence` |

An **opt-in general-knowledge fallback** is available for off-topic refusals only — a button, not a default — that answers from the model's own knowledge and is visibly badged "⚠ general knowledge — not from the knowledge base," never blended with grounded answers. It is structurally unreachable from the unsafe-input path.

---

## Architecture notes worth knowing

**Hybrid retrieval, language-split.** Dense search (LanceDB HNSW, scalar-quantized) and sparse search (BM25, via `rank_bm25` with a Devanagari-safe custom tokenizer — see below) are fused by Reciprocal Rank Fusion, not raw score, since the two scoring systems are on incompatible scales. The BM25 index is split into separate Hindi/English sub-indices so a query only scans its own language's chunks, roughly halving per-query sparse-search cost.

**Devanagari-safe tokenization.** Both the sentence-splitting regex and the BM25 tokenizer were built to avoid a known class of bug: naive `\w`-based regexes exclude Devanagari vowel signs (matras) and the virama, silently shattering Hindi words into consonant fragments. This was verified directly: an early attempt to adopt the faster `bm25s` library was reverted after empirically confirming its internal tokenizer corrupted Hindi text (`दिल्ली` → `['द','ल','ल']`) even when pre-segmented input was supplied. `rank_bm25`, given full tokenization control, was kept instead.

**Local, quantized Prompt Guard.** The unsafe-input classifier originally ran via Groq's hosted endpoint; moving it to local ONNX inference removed a ~100–150ms network round trip that had been the single largest contributor to Tier 1 latency. The model was subsequently quantized (from ~1GB to ~300MB) to fit within the free-tier deployment memory budget.

**Cross-lingual retrieval gap.** Even with bilingual indexing, English queries measurably score lower on average than Hindi queries against passages that are semantically identical — a real, observed property of the embedding model's cross-lingual alignment, not a bug. Documented rather than silently masked.

**TTS language auto-detection.** The synthesized answer's language is detected from the answer text's own script (Devanagari present → `hi-IN`, else → `en-IN`) rather than hardcoded, since Tier 1 answers are always Hindi (the corpus's native language) while Tier 2 answers follow the query's language.

---

## ⚠️ Known limitation: deployed corpus size

The **locally-benchmarked corpus** (results below) was built from **6,000 source rows** of MSMARCO-XI's Hindi validation split, bilingually indexed (~190,000+ chunks). Streamlit Community Cloud's free tier caps total memory at 1GB, and the full corpus plus the embedding model, ONNX runtime, and PyTorch could not fit inside that budget. **The deployed corpus was reduced to 2,000 rows** (down from 6,000) and the Prompt Guard model quantized further to fit.

**Practical effect:** the deployed demo has meaningfully lower topical coverage than what the benchmark numbers below reflect — a noticeably higher proportion of otherwise-reasonable questions will be (correctly, by design) refused as off-topic simply because the smaller indexed corpus doesn't contain a matching passage, not because the retrieval or guardrail logic is malfunctioning. This is a deliberate, disclosed trade-off forced by free-tier memory constraints, not a defect in the pipeline itself. The **architecture, latency profile, and guardrail behavior are unchanged** between the two corpus sizes — only breadth of coverage differs.

---

## Benchmark results

*Measured locally against the full 6,000-row bilingual corpus, `benchmark.py`, 33 queries (in-corpus + off-topic + jailbreak controls), model `openai/gpt-oss-20b` via Groq.*

**Tier 1 — Prompt Guard + hybrid retrieval + relevance gate (the budgeted, <200ms claim):**

| Percentile | Latency |
|---|---|
| P50 | ~64–72 ms |
| P70 | ~71–78 ms |
| P100 | ~96–115 ms |

**Retrieval alone (dense + sparse + RRF fusion):**

| Percentile | Latency |
|---|---|
| P50 | ~30–58 ms |
| P100 | ~85–115 ms |


P100 in every stage is disproportionately influenced by whichever single query happens to be first/slowest in a given run (small sample size, n≈9–19 for generation); P50/P70 are the more representative numbers.

**Guardrail effectiveness**, same 33-query run: 3/3 jailbreak attempts correctly caught (`unsafe_input`), off-topic negative controls correctly refused, ungrounded/low-confidence answers correctly caught and either refused or silently backed by the Tier 1 extractive answer instead.

---

## Tech stack

| Layer | Tool |
|---|---|
| STT / TTS | Sarvam AI (`saaras:v3`, `bulbul:v2`) |
| LLM generation | Groq (`openai/gpt-oss-20b`) |
| Dense retrieval | LanceDB, HNSW + scalar quantization |
| Sparse retrieval | `rank_bm25`, custom Devanagari-safe tokenizer |
| Embeddings | FastEmbed, `paraphrase-multilingual-MiniLM-L12-v2` |
| Guardrail model | `meta-llama/Llama-Prompt-Guard-2-86M`, local ONNX (quantized) |
| UI | Streamlit |
| Structured output | Pydantic |

---

## Running locally

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

# Build the index (bilingual, hybrid chunking, hybrid retrieval)
python index_vectors.py --force

# Quick isolated Tier-1 latency check
python diagnose_tier1.py

# Full benchmark
python benchmark.py

# Run the app
streamlit run app.py
```

Requires a `.env` file with `GROQ_API_KEY` and `SARVAM_API_KEY`.

---

## Repo structure

```
app.py                 Streamlit UI — two-tier progressive display, live benchmark trigger
guardrails.py           Harness: structured output, retries, all three gates, two-tier orchestration
rag_pipeline.py          Hybrid retrieval (dense + sparse + RRF), data setup for deployment
voice_rag.py             STT → guarded RAG → TTS, full voice pipeline
index_vectors.py          Chunking strategies, embedding, index build
sparse_index.py            BM25 build/query, language-split
rrf.py                       Reciprocal Rank Fusion
text_utils.py                  Devanagari-safe tokenizer
benchmark.py                    P50/P70/P100 analytics across a real query set
diagnose_tier1.py                 Isolated Tier-1 component timing
```
