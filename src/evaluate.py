"""Compare model outputs with ground-truth labels and report per-field accuracy."""
import argparse
import json
import re
from datetime import datetime
from pathlib import Path

FIELDS = ["company", "address", "date", "total"]
DATE_FORMATS = [
    "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d/%m/%y",
    "%d-%m-%y", "%Y-%m-%d", "%d %b %Y", "%d %B %Y",
]


def parse_prediction(text):
    """Return the prediction as a dict, or None if the model output is not valid JSON."""
    start, end = text.find("{"), text.rfind("}")   # skips ```json fences if present
    if start == -1 or end == -1:
        return None
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    if data.get("parse_error"):                     # marked as broken by clean_results.py
        return None
    return data


def normalize_text(s):
    s = re.sub(r"[^A-Z0-9 ]", " ", s.upper())      # keep only letters, digits and spaces
    return " ".join(s.split())                      # collapse whitespace


def normalize_date(s):
    s = s.strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass
    return s.upper()


def normalize_total(s):
    digits = re.sub(r"[^\d.]", "", s)               # "RM 1,193.00" -> "1193.00"
    try:
        return f"{float(digits):.2f}"
    except ValueError:
        return s


def normalize(field, value):
    value = str(value or "")
    if field == "date":
        return normalize_date(value)
    if field == "total":
        return normalize_total(value)
    return normalize_text(value)


parser = argparse.ArgumentParser(description="Evaluate extraction results against labels.")
parser.add_argument("results", type=Path, help="Folder with model outputs (.json or .txt)")
parser.add_argument(
    "--labels", type=Path, default=Path("data/labels"),
    help="Folder with ground-truth .txt files (default: data/labels)",
)
parser.add_argument("--quiet", action="store_true", help="Only print scores, not individual errors")
args = parser.parse_args()

pred_files = sorted(list(args.results.glob("*.json")) + list(args.results.glob("*.txt")))
if not pred_files:
    raise SystemExit(f"No .json or .txt files found in {args.results}")

correct = {f: 0 for f in FIELDS}
errors, broken, n = [], 0, 0

for pred_file in pred_files:
    label_file = args.labels / f"{pred_file.stem}.txt"
    if not label_file.exists():
        print(f"Warning: no label for {pred_file.stem}, skipping.")
        continue
    n += 1

    label = json.loads(label_file.read_text(encoding="utf-8", errors="ignore"))
    pred = parse_prediction(pred_file.read_text(encoding="utf-8"))
    if pred is None:
        broken += 1
        pred = {}   # broken output counts as wrong on every field

    for f in FIELDS:
        if normalize(f, pred.get(f)) == normalize(f, label.get(f)):
            correct[f] += 1
        else:
            errors.append((pred_file.stem, f, pred.get(f), label.get(f)))

if n == 0:
    raise SystemExit("No predictions with matching labels. Check the --labels folder.")

print(f"Evaluated: {args.results}")
print(f"Receipts: {n}  |  Invalid JSON: {broken}\n")
for f in FIELDS:
    print(f"{f:8}: {correct[f]}/{n}  ({100 * correct[f] / n:.1f}%)")
total_correct = sum(correct.values())
print(f"\noverall : {total_correct}/{n * 4}  ({100 * total_correct / (n * 4):.1f}%)")

if not args.quiet:
    print("\nERRORS")
    for name, f, p, l in errors:
        print(f"{name} | {f}\n   model: {p}\n   label: {l}")
