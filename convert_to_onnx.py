from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

MODEL_ID = "meta-llama/Llama-Prompt-Guard-2-86M"
SAVE_DIR = "./prompt_guard_onnx"

print("Converting to ONNX (one-time, takes a minute)...")
model = ORTModelForSequenceClassification.from_pretrained(MODEL_ID, export=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print(f"Done. Saved to {SAVE_DIR}")