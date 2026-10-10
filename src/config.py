"""Shared settings for all local experiments. Change them here, not in individual scripts."""

MODEL = "qwen2.5vl:3b"

# 4096 truncated answers on large receipts (v1), 8192 fixed it (v2).
NUM_CTX = 8192

# The prompt is intentionally in Turkish: every experiment uses this exact prompt,
# and the fine-tuned model was trained with it. Changing it breaks comparability.
PROMPT = (
    "Bu makbuzdan şirket adını, adresi, tarihi ve toplam tutarı çıkar. "
    "Sadece JSON ver: company, address, date, total."
)
