# Receipt Extraction with a Fine-Tuned Vision Model

Extracting structured data (company, address, date, total) from scanned receipt images with an open-source vision-language model, measuring it, analyzing its errors, and improving it with LoRA fine-tuning.

**Result:** Fine-tuning Qwen2.5-VL-3B on 626 receipts raised field-level accuracy from **91.1% to 94.0%** on held-out test receipts and cut the number of errors by about **one third** (39 → 26), with no field getting worse.

| Field   | Base model (v3) | Fine-tuned (v4) | Change |
|---------|-----------------|-----------------|--------|
| company | 89.9%           | **95.4%**       | +5.5   |
| address | 78.0%           | **83.5%**       | +5.5   |
| date    | 97.2%           | **98.2%**       | +1.0   |
| total   | 99.1%           | 99.1%           | 0      |
| **Overall** | 91.1%       | **94.0%**       | **+2.9** |
| Invalid JSON outputs | 1  | 0               |        |

*Same 109 test receipts, same prompt, same evaluation code for both runs.*

---

## The task

Given a photo of a receipt, return:

```json
{
  "company": "OJC MARKETING SDN BHD",
  "address": "NO 2 & 4, JALAN BAYU 4, BANDAR SERI ALAM, 81750 MASAI, JOHOR",
  "date": "15/01/2019",
  "total": "193.00"
}
```

**Dataset:** [ICDAR 2019 SROIE](https://www.kaggle.com/datasets/urbikn/sroie-datasetv2) — 626 training and 347 test receipts from Malaysia, labeled with the four fields above. The dataset is not included in this repository.

**Model:** [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct), a 3-billion-parameter open-source vision-language model. The model reads the image directly; no separate OCR step.

---

## Approach

The project follows one loop throughout: **measure → analyze errors → change one thing → measure again.**

### 1. Local baseline (v1, v2)
Ran a quantized version of the model locally with [Ollama](https://ollama.com) on a laptop GPU (GTX 1650 Ti, 4 GB) over 20 test receipts.

- **v1:** 3 of 20 answers were cut off mid-JSON. Cause: the model's context window (4096 tokens) was mostly filled by the image, leaving no room for the answer.
- **v2:** Doubling the context window to 8192 fixed it. Score: 72/80 fields (90%).

### 2. Evaluation
Model answers and labels are written in different formats (`2019-01-15` vs `15/01/2019`, `RM 9.00` vs `9.00`). Both sides are normalized before comparison: dates to one format, totals to numbers, text to uppercase without punctuation. Then fields are compared by exact match.

### 3. Full-precision baseline (v3)
Moved to Google Colab (T4 GPU) and ran the original, non-quantized model with Hugging Face `transformers` on 109 test receipts. This is the baseline the fine-tuned model is compared against.

### 4. Fine-tuning (v4)
Fine-tuned the same model on all 626 training receipts with [Unsloth](https://github.com/unslothai/unsloth) and LoRA:

- LoRA rank 16 on both vision and language layers, 16-bit base model
- 1 epoch, 79 steps, effective batch size 8, learning rate 2e-4
- Same Turkish prompt and image size limit as the baseline
- ~25 minutes on a free Colab T4; adapter size 160 MB

Training loss dropped from 3.34 to ~0.05 within 30 steps and then stayed flat. One epoch was enough.

---

## Error analysis

Every remaining error after fine-tuning (26 in total) was inspected by hand and grouped:

| Type | Count | Example |
|------|-------|---------|
| **Label error** (model is right) | ~7 | Postal code labeled `B1750`; correct is `81750` |
| **Scope disagreement** | ~5 | Model includes the mall name in the address; label doesn't. Labels themselves are inconsistent here. |
| **Real misread** | ~13 | `BERUJUR` instead of `BERJUNTAI`; `$6.50` instead of `$8.50` |

Findings:
- **Real accuracy is higher than the measured 94%.** Excluding label errors, it is roughly 95–96%.
- **Recurring 6 ↔ 8 confusion** in digits (`61200`/`81200`, `$6.50`/`$8.50`, `28/04`/`26/04`). Hypothesis: images are downscaled before reading, which blurs small characters. Testing this is the next experiment.
- Before fine-tuning, two total-amount errors had the same cause: Malaysian receipts round totals to 5 sen, and the model picked the amount before rounding. Spotting patterns like this is more useful than looking at the score alone.

---

## Problems solved along the way

- **Truncated answers** → context window too small for large images (v1 → v2).
- **CUDA device-side assert** during generation in float16 → disabled sampling (`do_sample=False`), which also made results reproducible.
- **Inference suddenly 20× slower** → Colab had reconnected without a GPU; the model was running on CPU.
- **Limited local hardware** (4 GB GPU) → quick experiments locally, large evaluations and training on free Colab.

---

## Repository structure

```
├── copy_data.py            # copy N receipts from SROIE into data/
├── src/
│   ├── config.py           # model name, prompt, context size (shared settings)
│   ├── extract.py          # read a single receipt (smoke test)
│   ├── run_all.py          # read all receipts in data/raw with Ollama
│   ├── evaluate.py         # compare outputs with labels, report accuracy and errors
│   └── clean_results.py    # convert raw model outputs into formatted JSON
├── results/                # model outputs for every experiment
└── experiments.md          # full log of every experiment, settings and findings
```

## Running locally

Requirements: Python 3.11+, [Ollama](https://ollama.com), and the SROIE dataset.

```bash
python -m venv venv
venv\Scripts\activate          # Windows; on macOS/Linux: source venv/bin/activate
pip install ollama
ollama pull qwen2.5vl:3b

# copy 20 test receipts into data/
python copy_data.py path/to/SROIE2019/test --limit 20

# read them and evaluate
python src/run_all.py --output results/my_run
python src/evaluate.py results/my_run
```

To evaluate the saved Colab results against the full test labels:

```bash
python src/evaluate.py results/v4 --labels path/to/SROIE2019/test/entities --quiet
```

---

## Limitations

- The test set used for comparison is small (109 receipts). Single-field scores can move by a few points by chance; the overall direction (gains where expected, no regressions) is more reliable than any single number.
- Exact match is strict. A one-character difference in a long address counts as fully wrong.
- The fine-tuned model is ~2× slower at inference (~11 s vs ~6 s per receipt on a T4) because the LoRA adapter is not merged into the base model yet.
- All receipts are from one country and in English.

## Next steps

- **v5:** Increase image resolution to test the 6 ↔ 8 hypothesis, and measure the speed cost.
- Merge the LoRA adapter and run the fine-tuned model locally with Ollama.
- Add a character-level similarity metric alongside exact match.
- Simple web demo: upload a receipt, see the extracted fields.
- Extend to Turkish receipts and invoices.
