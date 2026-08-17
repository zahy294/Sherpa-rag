"""Load a small sample from ai4bharat/MSMARCO-XI and inspect its structure."""

import sys

from datasets import load_dataset

# Hindi validation split (smaller than the 3.7 GB train file).
HINDI_VAL = "hf://datasets/ai4bharat/MSMARCO-XI/validation/hinval.parquet"


def main() -> None:
    # Ensure Hindi/Devanagari text prints correctly on Windows terminals.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    dataset = load_dataset("parquet", data_files=HINDI_VAL, split="train[:3]")

    for i, row in enumerate(dataset, start=1):
        print(f"\n{'=' * 60}")
        print(f"Row {i}")
        print(f"{'=' * 60}")

        print(f"\nQuery (Hindi): {row['query']}")
        print(f"Answer (Hindi): {row['Answer']}")
        print(f"\nQuery (English): {row['Eng_Query']}")
        print(f"Answer (English): {row['Eng_Answer']}")

        passages = row["passages"]
        print("\nPassage structure:")
        print(f"  keys: {list(passages.keys())}")
        print(f"  num passages: {len(passages['Translated_passages'])}")

        for j, (translated, english, selected) in enumerate(
            zip(
                passages["Translated_passages"],
                passages["English_passages"],
                passages["is_selected"],
            ),
            start=1,
        ):
            print(f"\n  Passage {j} (selected={selected}):")
            preview_t = translated[:120] + ("..." if len(translated) > 120 else "")
            preview_e = english[:120] + ("..." if len(english) > 120 else "")
            print(f"    Translated: {preview_t}")
            print(f"    English:    {preview_e}")


if __name__ == "__main__":
    main()
