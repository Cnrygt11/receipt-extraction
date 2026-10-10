## v1 - qwen2.5vl:3b, num_ctx 4096
company: 16/20
address: 15/20
date:    17/20
total:   15/20
Note: The response was cut off during a process of 3 receipts (memory limit).

## v2 - qwen2.5vl:3b, num_ctx 8192
company : 17/20
address : 17/20
date    : 20/20
total   : 18/20

## v3 - Qwen2.5-VL-3B (orijinal), Colab T4, 109 test receipt

Wrong formatted JSON: 1

company : 98/109  (%89.9)
address : 85/109  (%78.0)
date    : 106/109  (%97.2)
total   : 108/109  (%99.1)

GENERAL   : 397/436  (%91.1)

## v4 - Qwen2.5-VL-3B + LoRA fine-tuned (Unsloth), Colab T4, 109 test receipts

### Training
- Data: SROIE train split, 626 receipts (img + entities); `box` files not used
- Method: LoRA (r=16, alpha=16, dropout 0), vision + language layers, 16-bit base model
- 1 epoch, 79 steps, effective batch size 8 (2 per device x 4 gradient accumulation)
- Learning rate 2e-4, linear schedule, 5 warmup steps, adamw_8bit, weight decay 0.01
- Same Turkish prompt and image size limit (MAX_PIXELS = 1280*28*28) as v3
- Target answers: plain JSON with fields in fixed order (company, address, date, total)
- Loss: 3.34 → 0.25 (step 20) → ~0.05 plateau from step 30, final 0.034 (step 75)
- Training time: ~25 min
- Adapter size: 160 MB

### Results
Wrong formatted JSON: 0

company : 104/109  (%95.4)
address : 91/109   (%83.5)
date    : 107/109  (%98.2)
total   : 108/109  (%99.1)

GENERAL : 410/436  (%94.0)

### Comparison with v3 (same 109 receipts, same prompt, same evaluation)
| Field   | v3    | v4    | Change |
|---------|-------|-------|--------|
| company | 89.9% | 95.4% | +5.5   |
| address | 78.0% | 83.5% | +5.5   |
| date    | 97.2% | 98.2% | +1.0   |
| total   | 99.1% | 99.1% | 0      |
| GENERAL | 91.1% | 94.0% | +2.9   |
| Invalid JSON | 1 | 0 |        |

- Errors dropped from 39 to 26 (about one third fewer)
- Gains came in the two weakest fields (company, address); no field got worse
- Model now always returns plain JSON without ```json fences

### Error analysis (26 errors)
- ~7 label errors (model correct): B1750, SEKYSEN, TED HENG, PARINDUSTRIAN, PERMAS JAY, PINGGLRAN/SUNGAL, JIN IBU KOTA
- ~5 scope disagreements: brand name or registration number added to company; mall name added to or missing from address. Labels themselves are inconsistent here.
- ~13 real misreads, mostly letters (BERUJUR, TEMPOI, NASAI, IASIK, SCN, BERYELEV)
- Recurring 6↔8 digit confusion: 61200/81200, $6.50/$8.50, 28/04 vs 26/04
  → hypothesis: image downscaling (MAX_PIXELS) hurts small characters

### Notes
- Inference: ~11 s/receipt vs ~6 s in v3 (LoRA adapter not merged into the base model)
- Excluding label errors, real accuracy is roughly 95-96%
- Test set is small (109 receipts); single-field scores can vary by a few points
- Next idea (v5): increase MAX_PIXELS to test the image resolution hypothesis