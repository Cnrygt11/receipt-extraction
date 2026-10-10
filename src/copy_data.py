"""Copy the first N receipts (images + labels) from the SROIE test split into data/."""
import argparse
import shutil
from pathlib import Path

parser = argparse.ArgumentParser(description="Copy SROIE receipts into data/raw and data/labels.")
parser.add_argument("source", type=Path, help="Path to the SROIE2019/test folder")
parser.add_argument("--limit", type=int, default=30, help="Number of receipts to copy (default: 30)")
args = parser.parse_args()

raw_dir = Path("data/raw")
labels_dir = Path("data/labels")
raw_dir.mkdir(parents=True, exist_ok=True)
labels_dir.mkdir(parents=True, exist_ok=True)

images = sorted((args.source / "img").glob("*.jpg"))[: args.limit]
if not images:
    raise SystemExit(f"No images found in {args.source / 'img'}. Check the path.")

copied = 0
for img in images:
    label = args.source / "entities" / f"{img.stem}.txt"
    if not label.exists():
        print(f"Skipping {img.name}: no matching label file.")
        continue
    shutil.copy(img, raw_dir)
    shutil.copy(label, labels_dir)
    copied += 1

print(f"Copied {copied} receipts to {raw_dir} and {labels_dir}.")
