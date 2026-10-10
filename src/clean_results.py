"""Convert raw model outputs (.txt, possibly wrapped in ```json fences) into clean .json files."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description="Clean raw model outputs into formatted JSON files.")
parser.add_argument("source", type=Path, help="Folder with raw .txt outputs")
parser.add_argument("destination", type=Path, help="Folder to write clean .json files to")
args = parser.parse_args()

args.destination.mkdir(parents=True, exist_ok=True)

ok, broken = 0, []
for f in sorted(args.source.glob("*.txt")):
    text = f.read_text(encoding="utf-8")
    start, end = text.find("{"), text.rfind("}")
    try:
        data = json.loads(text[start:end + 1])
        ok += 1
    except json.JSONDecodeError:
        # Keep broken outputs as they are instead of hiding them
        data = {"parse_error": True, "raw_output": text}
        broken.append(f.name)

    (args.destination / f"{f.stem}.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

print(f"Converted: {ok}")
print(f"Invalid: {broken}")
