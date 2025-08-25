from transformers import AutoModel
import torch
import json
from datetime import datetime
import os

model = AutoModel.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m").to(torch.device("cpu"))

# Extract model metadata
metadata = {
    "model_name": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "extraction_date": datetime.now().isoformat(),
    "config": {
        "hidden_size": model.config.hidden_size,
        "num_attention_heads": model.config.num_attention_heads,
        "num_hidden_layers": model.config.num_hidden_layers,
        "intermediate_size": model.config.intermediate_size,
        "patch_size": model.config.patch_size,
        "image_size": model.config.image_size,
        "num_channels": model.config.num_channels,
        "vocab_size": getattr(model.config, 'vocab_size', None),
    },
    "architecture": {
        "model_type": str(type(model).__name__),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "model_structure": str(model)
    }
}

# Create results directory if it doesn't exist
os.makedirs("results", exist_ok=True)

# Save metadata to JSON file
output_file = "results/dinov3_model_metadata.json"
with open(output_file, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"Model metadata saved to {output_file}")
print(f"Total parameters: {metadata['architecture']['total_parameters']:,}")
print(model)