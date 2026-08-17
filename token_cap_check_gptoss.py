"""
token_cap_check_gptoss.py
Standalone diagnostic — NOT part of the production pipeline.

Finds the real max_completion_tokens openai/gpt-oss-20b needs, by running
a representative sample of real queries through actual retrieval + generation,
at a generous cap (400) so nothing truncates, and reporting completion_tokens per query.

Usage (from repo root):
    python token_cap_check_gptoss.py
"""

import json
import os
import time

from dotenv import load_dotenv
from groq import Groq

from guardrails import STRUCTURED_SYSTEM_PROMPT, _build_user_prompt
from rag_pipeline import retrieve_context

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"), max_retries=0)
MODEL_NAME = "openai/gpt-oss-20b"
GENEROUS_CAP = 400  # high enough that nothing here should ever truncate

TEST_QUERIES = [
    ". what is a corporation?",
    "कॉर्पोरेशन क्या है?",
    "bottom front of a cargo ship",
    "मालवाहक जहाज़ के नीचे की तरफ",
    "chart for foods low in potassium.",
    "पोटेशियम में कम खाद्य पदार्थों का चार्ट।",
    "honesty or integrity definition",
    "ईमानदारी या सच्चाई की परिभाषा",
    "struthers city school district state number",
    "स्ट्रूथर्स शहर स्कूल जिला राज्य संख्या",
]


def run_one(query: str):
    chunks, retrieval_ms, best_dense_score = retrieve_context(query, top_k=3)
    if best_dense_score < 0.35:
        print(f"  (skipped — off-topic, best_dense_score={best_dense_score:.4f})")
        return None

    kwargs = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": STRUCTURED_SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(query, chunks)},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.1,
        "max_completion_tokens": GENEROUS_CAP,
    }

    if "gpt-oss" in MODEL_NAME:
        kwargs["reasoning_effort"] = "low"

    completion = client.chat.completions.create(**kwargs)

    usage = completion.usage
    raw = completion.choices[0].message.content
    try:
        parsed = json.loads(raw)
        ok = True
    except json.JSONDecodeError:
        parsed = None
        ok = False

    return {
        "query": query,
        "completion_tokens": usage.completion_tokens,
        "valid_json": ok,
        "answer_preview": (parsed.get("answer", "")[:80] if parsed else raw[:80]),
    }


def main():
    results = []
    for q in TEST_QUERIES:
        print(f"\n{q}")
        try:
            r = run_one(q)
            if r:
                results.append(r)
                print(
                    f"  completion_tokens={r['completion_tokens']}  valid_json={r['valid_json']}"
                )
                print(f"  answer: {r['answer_preview']}")
        except Exception as e:
            print(f"  ERROR: {e}")
        time.sleep(1.5)  # light pacing, avoid rate limits

    if results:
        max_tokens = max(r["completion_tokens"] for r in results)
        print(f"\n{'=' * 60}")
        print(
            f"Max completion_tokens observed across {len(results)} queries: {max_tokens}"
        )
        print(
            f"Suggested max_completion_tokens setting: {max_tokens + 30} (headroom)"
        )
        invalid = [r for r in results if not r["valid_json"]]
        if invalid:
            print(
                f"WARNING: {len(invalid)} quer(ies) produced invalid JSON even at cap={GENEROUS_CAP} — separate issue, not a token-budget problem."
            )


if __name__ == "__main__":
    main()