import base64
import os
import requests
from dotenv import load_dotenv

load_dotenv()

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
if not SARVAM_API_KEY:
    raise ValueError("SARVAM_API_KEY missing from .env file!")

url = "https://api.sarvam.ai/text-to-speech"
headers = {
    "api-subscription-key": SARVAM_API_KEY,
    "Content-Type": "application/json"
}

# The sample question we want to turn into an audio file
payload = {
    "inputs": ["कॉर्पोरेशन क्या है?"],
    "target_language_code": "hi-IN",
    "speaker": "anushka",
    "pitch": 0,
    "pace": 1.0,
    "loudness": 1.5,
    "speech_sample_rate": 8000,
    "enable_preprocessing": True,
    "model": "bulbul:v2"
}

print("Generating test question audio with Sarvam AI TTS...")
response = requests.post(url, headers=headers, json=payload)

if response.status_code == 200:
    audio_content = response.json()["audios"][0]
    output_filename = "test_question.mp3"
    with open(output_filename, "wb") as f:
        f.write(base64.b64decode(audio_content))
    print(f" Success! Created audio file: '{output_filename}'")
else:
    print(f" Error {response.status_code}: {response.text}")