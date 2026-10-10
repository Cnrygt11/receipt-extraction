import shutil
import sys
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python copy_data.py <path to SROIE2019/test folder>")
    sys.exit(1)

SOURCE = Path(sys.argv[1])

# İlk 30 makbuz fotoğrafını bul
images = sorted((SOURCE / "img").glob("*.jpg"))[:30]

for img in images:
    # Her fotoğrafın aynı isimli cevap dosyası
    label = SOURCE / "entities" / (img.stem + ".txt")
    if label.exists():
        shutil.copy(img, "data/raw/")
        shutil.copy(label, "data/labels/")

print(f"{len(images)} makbuz bulundu.")
print("raw klasöründe:", len(list(Path("data/raw").glob("*.jpg"))), "fotoğraf")
print("labels klasöründe:", len(list(Path("data/labels").glob("*.txt"))), "cevap")