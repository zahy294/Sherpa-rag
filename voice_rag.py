import base64
import mimetypes
import os
import re
import time
import requests
from dotenv import load_dotenv

from guardrails import run_two_tier_rag
from rag_pipeline import retrieve_context

load_dotenv()

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
if not SARVAM_API_KEY:
    raise ValueError("SARVAM_API_KEY is missing from your .env file!")

SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"
SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"

_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")


def detect_tts_language_code(text: str) -> str:
    """Script-based detection: Devanagari present -> hi-IN, else en-IN."""
    return "hi-IN" if _DEVANAGARI_RE.search(text) else "en-IN"


def _detect_audio_format(file_path: str) -> tuple[str, str]:
    """Inspect binary header to return exact MIME type and matching filename extension."""
    with open(file_path, "rb") as f:
        header = f.read(12)

    if header.startswith(b"\x1a\x45\xdf\xa3"):
        return "audio/webm", "recording.webm"
    elif header.startswith(b"OggS"):
        return "audio/ogg", "recording.ogg"
    elif header.startswith(b"RIFF"):
        return "audio/wav", "recording.wav"
    elif header.startswith(b"ID3") or (len(header) >= 2 and header[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2")):
        return "audio/mp3", "recording.mp3"

    mime_type, _ = mimetypes.guess_type(file_path)
    ext = file_path.split(".")[-1] if "." in file_path else "wav"
    return mime_type or "audio/wav", f"recording.{ext}"


def speech_to_text(audio_file_path: str) -> tuple[str, float]:
    """Transcribe an audio file via Sarvam's REST speech-to-text endpoint."""
    t0 = time.perf_counter()
    headers = {"api-subscription-key": SARVAM_API_KEY}

    mime_type, filename = _detect_audio_format(audio_file_path)

    with open(audio_file_path, "rb") as f:
        files = {"file": (filename, f, mime_type)}
        data = {
            "model": "saaras:v3",
            "mode": "transcribe",
            "language_code": "unknown",
        }
        response = requests.post(
            SARVAM_STT_URL, headers=headers, files=files, data=data
        )

    stt_ms = (time.perf_counter() - t0) * 1000

    if response.status_code != 200:
        raise Exception(f"STT Error {response.status_code}: {response.text}")

    result = response.json()
    transcript = result.get("transcript", "").strip()
    if not transcript:
        raise ValueError(
            "Sarvam STT returned an empty transcript — please speak clearly into the mic and try again."
        )

    return transcript, stt_ms


def text_to_speech(
    text: str, output_audio_path: str = "response.mp3", language_code: str = None
) -> float:
    """Convert text answer into audio via Sarvam AI TTS."""
    if language_code is None:
        language_code = detect_tts_language_code(text)

    t0 = time.perf_counter()
    headers = {
        "api-subscription-key": SARVAM_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {
        "inputs": [text],
        "target_language_code": language_code,
        "speaker": "anushka",
        "pitch": 0,
        "pace": 1.0,
        "loudness": 1.5,
        "speech_sample_rate": 8000,
        "enable_preprocessing": True,
        "model": "bulbul:v2",
    }
    response = requests.post(SARVAM_TTS_URL, headers=headers, json=payload)
    tts_ms = (time.perf_counter() - t0) * 1000

    if response.status_code == 200:
        audio_content = response.json()["audios"][0]
        with open(output_audio_path, "wb") as f:
            f.write(base64.b64decode(audio_content))
        return tts_ms
    else:
        raise Exception(f"TTS Error {response.status_code}: {response.text}")


def run_full_voice_rag(
    audio_input_path: str, allow_general_knowledge_fallback: bool = False
):
    """Executes full Voice -> Two-Tier Guarded RAG -> Voice pipeline."""
    print("🎙️ Starting Voice RAG Pipeline...")

    transcript, stt_ms = speech_to_text(audio_input_path)
    print(f"🗣️ Transcribed Text : '{transcript}' ({stt_ms:.2f} ms)")

    for update in run_two_tier_rag(
        transcript,
        retrieve_fn=retrieve_context,
        allow_general_knowledge_fallback=allow_general_knowledge_fallback,
    ):
        print(
            f"🛡️ Tier {update.tier} | refusal={update.refusal_reason.value} |"
            f" {update.latency_ms:.2f} ms"
        )
        print(f"💬 Answer: {update.answer}")

        if update.is_final:
            audio_path = f"output_answer_tier{update.tier}.mp3"
            tts_ms = text_to_speech(update.answer, audio_path)
            print(f"🔊 Audio saved to '{audio_path}' ({tts_ms:.2f} ms)")
        else:
            audio_path = None
            tts_ms = 0.0

        yield {
            "transcript": transcript,
            "stt_ms": stt_ms,
            "tier": update.tier,
            "answer": update.answer,
            "is_final": update.is_final,
            "refusal_reason": update.refusal_reason.value,
            "confidence": update.confidence,
            "used_tier1_fallback": update.used_tier1_fallback,
            "is_general_knowledge": update.is_general_knowledge,
            "source_lang": update.source_lang,
            "generation_latency_ms": update.latency_ms,
            "tts_ms": tts_ms,
            "audio_path": audio_path,
        }


if __name__ == "__main__":
    for result in run_full_voice_rag("test_question.mp3"):
        print("-" * 60)