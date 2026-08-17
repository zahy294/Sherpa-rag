"""guardrails.py

Structured output enforcement, retry logic, prompt guard pre-checks,
two-tier progressive answering, and hallucination/off-topic detection for RAG in Goa.
"""

from concurrent.futures import ThreadPoolExecutor
from enum import Enum
import json
import logging
import os
import re
import time
from typing import Any, Dict, Generator, List, Optional

from dotenv import load_dotenv
from groq import BadRequestError, Groq, RateLimitError
from onnxruntime import SessionOptions
from optimum.onnxruntime import ORTModelForSequenceClassification
from pydantic import BaseModel, Field, ValidationError, field_validator
import torch
from transformers import AutoTokenizer

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("guardrails")

# ----------------------------------------------------------------------
# LOCAL MODEL INITIALIZATION — Prompt Guard 2 (ONNX)
# ----------------------------------------------------------------------

LOCAL_PROMPT_GUARD_MODEL = "./prompt_guard_onnx"

logger.info(
    f"Loading ONNX Prompt Guard model from {LOCAL_PROMPT_GUARD_MODEL} ..."
)
_guard_tokenizer = AutoTokenizer.from_pretrained(
    LOCAL_PROMPT_GUARD_MODEL, fix_mistral_regex=True
)

# Cap session threads so ONNX Runtime doesn't contend with FastEmbed's pool
_ort_session_options = SessionOptions()
_ort_session_options.intra_op_num_threads = 1
_ort_session_options.inter_op_num_threads = 1

_guard_model = ORTModelForSequenceClassification.from_pretrained(
    LOCAL_PROMPT_GUARD_MODEL,
    session_options=_ort_session_options,
)
logger.info("ONNX Prompt Guard model loaded.")

_warmup_inputs = _guard_tokenizer("warmup", return_tensors="pt")
_guard_model(**_warmup_inputs)
logger.info("Prompt Guard model warmed up.")

# Shared executor for concurrent prompt guard pre-checks and vector retrieval
_executor = ThreadPoolExecutor(max_workers=2)

# Initialize client with max_retries=0 to handle 429 backoffs explicitly outside latency timers
client = Groq(api_key=os.getenv("GROQ_API_KEY"), max_retries=0)

# ----------------------------------------------------------------------
# CONFIG & HELPERS
# ----------------------------------------------------------------------

MODEL_NAME = "openai/gpt-oss-20b"
JAILBREAK_SCORE_THRESHOLD = (
    0.5  # confirmed wide separation (~0.0003 benign vs ~0.999 jailbreak)
)

MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 1.5

# Retrieval-score floor below which we treat the query as off-topic
MIN_RETRIEVAL_SCORE = 0.35

# Confidence floor below which the model's own answer is refused
MIN_ANSWER_CONFIDENCE = 0.55

_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")


class RefusalReason(str, Enum):
    NONE = "none"
    UNSAFE_INPUT = "unsafe_input"  # prompt injection or jailbreak detected
    OFF_TOPIC = "off_topic"  # nothing relevant retrieved
    UNGROUNDED = "ungrounded"  # answer not supported by context
    LOW_CONFIDENCE = "low_confidence"
    PARSE_FAILURE = "parse_failure"


class GroundedAnswer(BaseModel):
    """Structured, validated output contract for every RAG response."""

    answer: str = Field(..., description="Final answer text, or refusal message")
    is_grounded: bool = Field(
        ..., description="True if answer is supported by retrieved context"
    )
    confidence: float = Field(..., ge=0.0, le=1.0)
    refusal_reason: RefusalReason = RefusalReason.NONE

    @field_validator("answer")
    @classmethod
    def answer_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("answer must not be empty")
        return v.strip()


class GuardrailResult(BaseModel):
    """Wraps the final decision returned to the caller (UI, benchmark, etc.)."""

    query: str
    response: GroundedAnswer
    retrieval_score: float
    passed_retrieval_gate: bool
    passed_groundedness_gate: bool
    latency_ms: Dict[str, float]
    attempts: int


class TwoTierUpdate(BaseModel):
    """One yielded update from run_two_tier_rag — tier 1 (extractive) or tier 2
    (LLM-polished, or a fallback notice if polish was discarded).
    """

    tier: int  # 1 or 2
    answer: str
    is_final: bool  # True if no further tiers will follow
    source_chunk: Optional[str] = None
    source_lang: Optional[str] = None
    confidence: Optional[float] = None
    refusal_reason: RefusalReason = RefusalReason.NONE
    used_tier1_fallback: bool = False
    is_general_knowledge: bool = False
    latency_ms: float = 0.0


# ----------------------------------------------------------------------
# PRE-RETRIEVAL — Local Prompt Guard (Jailbreak / Injection Detection)
# ----------------------------------------------------------------------


def check_unsafe_input(query: str) -> tuple[bool, float, float]:
    """Returns (is_safe, jailbreak_score, guard_ms).

    Runs meta-llama/Llama-Prompt-Guard-2-86M via ONNX Runtime locally (CPU) instead
    of PyTorch — eliminates interpreter overhead and optimizes CPU execution.
    """
    t0 = time.perf_counter()
    try:
        inputs = _guard_tokenizer(
            query, return_tensors="pt", truncation=True, max_length=512
        )
        logits = _guard_model(**inputs).logits
        # Label 1 = malicious/jailbreak in Prompt Guard 2's binary classification head
        probs = torch.softmax(logits, dim=-1)
        score = probs[0][1].item()
        guard_ms = (time.perf_counter() - t0) * 1000
        return score < JAILBREAK_SCORE_THRESHOLD, score, guard_ms
    except Exception as e:
        guard_ms = (time.perf_counter() - t0) * 1000
        logger.warning(
            f"Local prompt guard check failed: {e}. Failing open (allowing query through)."
        )
        return True, 0.0, guard_ms


# ----------------------------------------------------------------------
# STAGE 1 — Retrieval gate (off-topic detection)
# ----------------------------------------------------------------------


def check_retrieval_gate(best_dense_score: float) -> tuple[bool, float]:
    """Off-topic detector: refuse if the single best dense match in the full
    candidate pool doesn't clear the similarity floor. Deliberately evaluated
    independent of RRF fusion rank/window — the chunk fusion promotes into the
    returned context is not necessarily the chunk with the highest raw dense
    score, so gating on the fused top-k (even via max()) can miss a passing
    dense match that got crowded out by sparse-only signal. See
    rag_pipeline.retrieve_context.
    """
    return best_dense_score >= MIN_RETRIEVAL_SCORE, best_dense_score


# ----------------------------------------------------------------------
# STAGE 2 — Structured generation with retry & clean latency isolation
# ----------------------------------------------------------------------

STRUCTURED_SYSTEM_PROMPT = """You are a strict RAG assistant for "RAG in Goa".
You MUST answer ONLY using the provided context chunks. Do not use outside knowledge. Keep your answer concise (under 3 sentences) so it sounds natural when spoken. Respond in the same language as the question.

Respond ONLY with a single valid JSON object matching this exact schema:
{
  "answer": "<string, your concise answer or a refusal message>",
  "is_grounded": <true/false, true only if every claim in your answer is directly supported by the context>,
  "confidence": <float 0.0-1.0>,
  "refusal_reason": "<one of: none, off_topic, ungrounded, low_confidence>"
}

Rules:
- If the context does not contain the answer, set is_grounded=false, refusal_reason="ungrounded",
  and answer should politely state you don't have enough information.
- Never fabricate facts not present in the context.
- No markdown, no prose outside the JSON object.
"""


def _build_user_prompt(query: str, chunks: List[dict]) -> str:
    context_block = "\n\n".join(
        f"[{i}] {c['text']}" for i, c in enumerate(chunks)
    )
    query_is_hindi = bool(_DEVANAGARI_RE.search(query))
    language_directive = (
        "उत्तर हिंदी में दें, भले ही आपको संदर्भ का अनुवाद करना पड़े।"
        if query_is_hindi
        else "Answer in English, even though the source context may be in Hindi — translate the relevant facts."
    )
    return f"CONTEXT:\n{context_block}\n\nQUESTION: {query}\n\n{language_directive}"


def generate_structured_response(
    query: str,
    chunks: List[dict],
    model_name: Optional[str] = None,
) -> tuple[GroundedAnswer, int, float]:
    active_model = model_name or MODEL_NAME
    last_error: Optional[Exception] = None

    for attempt in range(1, MAX_RETRIES + 1):
        t_start = time.perf_counter()
        try:
            kwargs = dict(
                model=active_model,
                messages=[
                    {"role": "system", "content": STRUCTURED_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": _build_user_prompt(query, chunks),
                    },
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_completion_tokens=175,
            )
            if "gpt-oss" in active_model:
                kwargs["reasoning_effort"] = "low"

            completion = client.chat.completions.create(**kwargs)

            usage = getattr(completion, "usage", None)
            if usage:
                logger.info(
                    f"[Groq Timing] queue_time={getattr(usage, 'queue_time', '?')}s "
                    f"prompt_time={getattr(usage, 'prompt_time', '?')}s "
                    f"completion_time={getattr(usage, 'completion_time', '?')}s "
                    f"total_time={getattr(usage, 'total_time', '?')}s "
                    f"completion_tokens={getattr(usage, 'completion_tokens', '?')}"
                )

            raw = completion.choices[0].message.content
            data = json.loads(raw)
            validated = GroundedAnswer(**data)

            generation_ms = (time.perf_counter() - t_start) * 1000
            return validated, attempt, generation_ms

        except RateLimitError as e:
            logger.warning(
                f"[attempt {attempt}] rate limited on Groq API: {e}. Backing off..."
            )
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)
            continue
        except BadRequestError as e:
            last_error = e
            if "max completion tokens reached" in str(e):
                logger.warning(
                    f"[attempt {attempt}] token budget exceeded — not retrying, this is deterministic"
                )
                break
            logger.warning(f"[attempt {attempt}] Groq rejected request: {e}")
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)
            continue
        except (
            json.JSONDecodeError,
            ValidationError,
            KeyError,
            TypeError,
        ) as e:
            last_error = e
            logger.warning(f"[attempt {attempt}] structured output failed: {e}")
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)
            continue

    logger.error(
        f"All {MAX_RETRIES} attempts failed. Last error: {last_error}"
    )
    return (
        GroundedAnswer(
            answer="I'm unable to generate a reliable answer right now. Please rephrase your question.",
            is_grounded=False,
            confidence=0.0,
            refusal_reason=RefusalReason.PARSE_FAILURE,
        ),
        MAX_RETRIES,
        0.0,
    )


# ----------------------------------------------------------------------
# STAGE 3 — Groundedness gate (post-hoc hallucination check)
# ----------------------------------------------------------------------


def check_groundedness_gate(response: GroundedAnswer) -> bool:
    """Enforces the model's own self-report AND a confidence floor."""
    if not response.is_grounded:
        return False
    if response.confidence < MIN_ANSWER_CONFIDENCE:
        return False
    return True


class GeneralKnowledgeAnswer(BaseModel):
    answer: str = Field(...)
    confidence: float = Field(..., ge=0.0, le=1.0)

    @field_validator("answer")
    @classmethod
    def answer_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("answer must not be empty")
        return v.strip()


GENERAL_KNOWLEDGE_SYSTEM_PROMPT = """You are answering a question that is OUTSIDE the RAG knowledge base — no relevant context was retrieved. Answer using your own general knowledge, concisely (under 3 sentences, natural for speech). Respond in the same language as the question. If you are not confident, say so and reflect that in your confidence score.Respond ONLY with a single valid JSON object:{  "answer": "<string, your general-knowledge answer>",  "confidence": <float 0.0-1.0>}No markdown, no prose outside the JSON object."""


def generate_general_knowledge_response(
    query: str, model_name: Optional[str] = None
) -> tuple[GeneralKnowledgeAnswer, float]:
    """Answers from the model's own general knowledge. Used ONLY as an
    explicit, opt-in tier 2 fallback when retrieval found nothing
    (off_topic) — never reachable from the unsafe_input path, since that
    branch returns before the retrieval gate is ever checked.
    """
    active_model = model_name or MODEL_NAME
    t_start = time.perf_counter()
    try:
        kwargs = dict(
            model=active_model,
            messages=[
                {"role": "system", "content": GENERAL_KNOWLEDGE_SYSTEM_PROMPT},
                {"role": "user", "content": f"QUESTION: {query}"},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_completion_tokens=120,
        )
        if "gpt-oss" in active_model:
            kwargs["reasoning_effort"] = "low"
        completion = client.chat.completions.create(**kwargs)
        data = json.loads(completion.choices[0].message.content)
        validated = GeneralKnowledgeAnswer(**data)
        return validated, (time.perf_counter() - t_start) * 1000
    except Exception as e:
        logger.warning(f"General-knowledge fallback failed: {e}")
        return GeneralKnowledgeAnswer(
            answer="I don't have information on that.", confidence=0.0
        ), (time.perf_counter() - t_start) * 1000


# ----------------------------------------------------------------------
# ORCHESTRATORS — call these from voice_rag.py / app.py / benchmark.py
# ----------------------------------------------------------------------


def run_guarded_rag(
    query: str,
    retrieve_fn,
    k: int = 3,
    model_name: Optional[str] = None,
) -> GuardrailResult:
    """Full guarded pipeline."""
    latency: Dict[str, float] = {}

    t_parallel = time.perf_counter()
    guard_future = _executor.submit(check_unsafe_input, query)
    retrieve_future = _executor.submit(retrieve_fn, query, top_k=k)

    is_safe, jailbreak_score, guard_ms = guard_future.result()
    chunks, retrieval_ms, best_dense_score = retrieve_future.result()
    latency["parallel_guard_retrieval_ms"] = (
        time.perf_counter() - t_parallel
    ) * 1000
    latency["guard_ms"] = guard_ms
    latency["retrieval_ms"] = retrieval_ms

    if not is_safe:
        refusal = GroundedAnswer(
            answer="I can't process that request.",
            is_grounded=False,
            confidence=0.0,
            refusal_reason=RefusalReason.UNSAFE_INPUT,
        )
        latency["generation_ms"] = 0.0
        latency["groundedness_gate_ms"] = 0.0
        latency["total_ms"] = latency["parallel_guard_retrieval_ms"]
        return GuardrailResult(
            query=query,
            response=refusal,
            retrieval_score=0.0,
            passed_retrieval_gate=False,
            passed_groundedness_gate=False,
            latency_ms=latency,
            attempts=0,
        )

    # Retrieval Gate (Off-Topic Check)
    t1 = time.perf_counter()
    passed_retrieval, top_score = check_retrieval_gate(best_dense_score)
    latency["retrieval_gate_ms"] = (time.perf_counter() - t1) * 1000

    if not passed_retrieval:
        refusal = GroundedAnswer(
            answer=(
                "I don't have relevant information in my knowledge base to answer"
                " that."
            ),
            is_grounded=False,
            confidence=0.0,
            refusal_reason=RefusalReason.OFF_TOPIC,
        )
        latency["generation_ms"] = 0.0
        latency["groundedness_gate_ms"] = 0.0
        latency["total_ms"] = (
            latency["parallel_guard_retrieval_ms"]
            + latency["retrieval_gate_ms"]
        )

        return GuardrailResult(
            query=query,
            response=refusal,
            retrieval_score=top_score,
            passed_retrieval_gate=False,
            passed_groundedness_gate=False,
            latency_ms=latency,
            attempts=0,
        )

    # Generation
    response, attempts, generation_ms = generate_structured_response(
        query, chunks, model_name=model_name
    )
    latency["generation_ms"] = generation_ms

    # Groundedness Gate
    t3 = time.perf_counter()
    passed_groundedness = check_groundedness_gate(response)
    latency["groundedness_gate_ms"] = (time.perf_counter() - t3) * 1000

    if not passed_groundedness and response.refusal_reason == RefusalReason.NONE:
        response.refusal_reason = (
            RefusalReason.UNGROUNDED
            if not response.is_grounded
            else RefusalReason.LOW_CONFIDENCE
        )
        response.answer = (
            "I found related information but can't confidently confirm an answer"
            " from it."
        )

    latency["total_ms"] = (
        latency["parallel_guard_retrieval_ms"]
        + latency["retrieval_gate_ms"]
        + latency["generation_ms"]
        + latency["groundedness_gate_ms"]
    )

    return GuardrailResult(
        query=query,
        response=response,
        retrieval_score=top_score,
        passed_retrieval_gate=True,
        passed_groundedness_gate=passed_groundedness,
        latency_ms=latency,
        attempts=attempts,
    )


def run_two_tier_rag(
    query: str,
    retrieve_fn,
    k: int = 3,
    model_name: Optional[str] = None,
    allow_general_knowledge_fallback: bool = False,
) -> Generator[TwoTierUpdate, None, None]:
    """Two-tier answering yielded progressively."""
    latency: Dict[str, float] = {}

    t_parallel = time.perf_counter()
    guard_future = _executor.submit(check_unsafe_input, query)
    retrieve_future = _executor.submit(retrieve_fn, query, top_k=k)

    is_safe, jailbreak_score, guard_ms = guard_future.result()
    chunks, retrieval_ms, best_dense_score = retrieve_future.result()
    latency["parallel_guard_retrieval_ms"] = (
        time.perf_counter() - t_parallel
    ) * 1000

    if not is_safe:
        yield TwoTierUpdate(
            tier=1,
            answer="I can't process that request.",
            is_final=True,
            refusal_reason=RefusalReason.UNSAFE_INPUT,
            latency_ms=latency["parallel_guard_retrieval_ms"],
        )
        return

    t1 = time.perf_counter()
    passed_retrieval, top_score = check_retrieval_gate(best_dense_score)
    latency["retrieval_gate_ms"] = (time.perf_counter() - t1) * 1000

    if not passed_retrieval:
        yield TwoTierUpdate(
            tier=1,
            answer="I don't have relevant information in my knowledge base to answer that.",
            is_final=not allow_general_knowledge_fallback,
            refusal_reason=RefusalReason.OFF_TOPIC,
            latency_ms=latency["parallel_guard_retrieval_ms"]
            + latency["retrieval_gate_ms"],
        )
        if not allow_general_knowledge_fallback:
            return

        gk_response, gk_ms = generate_general_knowledge_response(
            query, model_name=model_name
        )
        yield TwoTierUpdate(
            tier=2,
            answer=gk_response.answer,
            is_final=True,
            confidence=gk_response.confidence,
            is_general_knowledge=True,
            refusal_reason=RefusalReason.OFF_TOPIC,
            latency_ms=gk_ms,
        )
        return

    # TIER 1 — extractive, no LLM call
    t_extract = time.perf_counter()
    fast_answer = chunks[0]["text"] if chunks else ""
    fast_answer_lang = (
        chunks[0].get("content_lang", "unknown") if chunks else "unknown"
    )
    extract_ms = (time.perf_counter() - t_extract) * 1000
    tier1_total_ms = (
        latency["parallel_guard_retrieval_ms"]
        + latency["retrieval_gate_ms"]
        + extract_ms
    )

    yield TwoTierUpdate(
        tier=1,
        answer=fast_answer,
        is_final=False,
        source_chunk=fast_answer,
        source_lang=fast_answer_lang,
        latency_ms=tier1_total_ms,
    )

    # TIER 2 — LLM polish, reported separately
    response, attempts, generation_ms = generate_structured_response(
        query, chunks, model_name=model_name
    )
    passed_groundedness = check_groundedness_gate(response)

    if passed_groundedness:
        yield TwoTierUpdate(
            tier=2,
            answer=response.answer,
            is_final=True,
            confidence=response.confidence,
            latency_ms=generation_ms,
        )
    else:
        yield TwoTierUpdate(
            tier=2,
            answer=fast_answer,
            is_final=True,
            used_tier1_fallback=True,
            refusal_reason=(
                RefusalReason.UNGROUNDED
                if not response.is_grounded
                else RefusalReason.LOW_CONFIDENCE
            ),
            latency_ms=generation_ms,
        )