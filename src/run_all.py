"""Run the local Ollama model over receipts in data/raw and save the raw outputs."""
import argparse
import time
from datetime import datetime
from pathlib import Path

import ollama

from config import MODEL, NUM_CTX, PROMPT

parser = argparse.ArgumentParser(description="Extract fields from all receipts in data/raw.")
parser.add_argument("--limit", type=int, help="Only process the first N receipts")
parser.add_argument(
    "--output", type=Path,
    help="Output folder (default: results/<current date and time>)",
)
args = parser.parse_args()

run_dir = args.output or Path("results") / datetime.now().strftime("%Y-%m-%d_%H-%M")
run_dir.mkdir(parents=True, exist_ok=True)

images = sorted(Path("data/raw").glob("*.jpg"))
if args.limit:
    images = images[: args.limit]

for i, image in enumerate(images, start=1):
    start = time.time()
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": PROMPT, "images": [str(image)]}],
        format="json",                 # return plain JSON only
        options={"num_ctx": NUM_CTX},
    )
    elapsed = time.time() - start

    answer = response["message"]["content"]
    (run_dir / f"{image.stem}.json").write_text(answer, encoding="utf-8")
    print(f"[{i}/{len(images)}] {image.name} - {elapsed:.1f} s")

print("Done! Results saved to:", run_dir)
