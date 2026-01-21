import re
import torch
import torch.nn as nn
from transformers import AutoModel
from peft import LoraConfig, get_peft_model

MODEL_NAME = "facebook/dinov3-vitb16-pretrain-lvd1689m"  # or your variant

# 1) Load (requires recent transformers + trust_remote_code if needed)
model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True).eval()

# 2) Collect leaf Linear/Conv2d module names the way PEFT sees them
leafs = [(name, m) for name, m in model.named_modules() if isinstance(m, (nn.Linear, nn.Conv2d))]
leaf_names = [n for n, _ in leafs]

print(f"Found {len(leaf_names)} Linear/Conv2d leaves. Example:")
for n in leaf_names[:20]:
    print(" -", n)

# 3) See what the last-token names look like (often what people target)
last_tokens = sorted({n.split(".")[-1] for n in leaf_names})
print("\nUnique leaf basenames (last token):", last_tokens[:30], "...")

# 4) Candidate patterns that commonly appear across ViT variants
CANDIDATES = [
    # attention
    "qkv","q_proj","k_proj","v_proj","q","k","v","to_qkv","wqkv","kv","proj","out_proj",
    # mlp
    "fc1","fc2","mlp_fc1","mlp_fc2","w1","w2","dense","ffn","ff","linear1","linear2",
    # patch/embed
    "patch_embed","proj","patch_embed.proj",
]

# (a) exact-last-token matches
exact_hits = [t for t in CANDIDATES if t in last_tokens]

# (b) substring matches on full dotted names (more forgiving)
substr_hits = []
for t in CANDIDATES:
    if any(t in n for n in leaf_names):
        substr_hits.append(t)

# Consolidate, keep order as in CANDIDATES
TARGETS = [t for t in CANDIDATES if (t in exact_hits or t in substr_hits)]
TARGETS = sorted(set(TARGETS), key=CANDIDATES.index)

print("\nDiscovered viable target substrings (ordered):", TARGETS)

if not TARGETS:
    # Fallback: show you the top frequent leaf basenames so you can pick
    from collections import Counter
    top = Counter([n.split(".")[-1] for n in leaf_names]).most_common(20)
    print("\nNo standard targets found. Here are your top leaf basenames—pick 2–4 of these:")
    for k, v in top:
        print(f"  {k}  ({v} occurrences)")
    raise SystemExit

# 5) Quick dry run: which modules would be hit by each target?
print("\nPreview: first few matches per target")
for t in TARGETS:
    hits = [n for n in leaf_names if t in n]
    print(f"  {t}: {len(hits)} hits")
    for h in hits[:5]:  # show a few
        print("   -", h)

# 6) Apply LoRA
lora_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    target_modules=TARGETS,     # <- robust, derived from your model
)
lora_model = get_peft_model(model, lora_cfg)
lora_model.print_trainable_parameters()