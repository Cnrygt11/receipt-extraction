"""Quick smoke test: read a single receipt with a local Ollama model."""
import argparse
import time
from pathlib import Path

import ollama

from config import MODEL, NUM_CTX, PROMPT

parser = argparse.ArgumentParser(description="Extract fields from a single receipt.")
parser.add_argument(
    "image", nargs="?", type=Path,
    help="Receipt image (default: first image in data/raw)",
)
args = parser.parse_args()

image = args.image or sorted(Path("data/raw").glob("*.jpg"))[0]
print("Receipt:", image.name)

start = time.time()
response = ollama.chat(
    model=MODEL,
    messages=[{"role": "user", "content": PROMPT, "images": [str(image)]}],
    format="json",
    options={"num_ctx": NUM_CTX},
)
elapsed = time.time() - start

print(response["message"]["content"])
print(f"Time: {elapsed:.1f} s")
