"""
Model configuration registry and shared constants for reid experiments.
"""

import torchvision.transforms as T
from PIL import Image


# ============================================================================
# Model configuration registry
# ============================================================================

MODEL_CONFIGS = {
    "dinov3": {
        "factory": "create_dinov3_arcface_model",
        "native_size": 224,
        "patch_size": 16,
        "norm_mean": [0.485, 0.456, 0.406],
        "norm_std": [0.229, 0.224, 0.225],
        "backbone_label": "Frozen DINOv3-ViT-B/16",
        "experiment_dir": "reid_openset/dinov3",
        "default_lr": 0.0005,
    },
    "megadescriptor": {
        "factory": "create_megadescriptor_arcface_model",
        "native_size": 384,
        "patch_size": 16,
        "norm_mean": [0.5, 0.5, 0.5],
        "norm_std": [0.5, 0.5, 0.5],
        "backbone_label": "Frozen MegaDescriptor-L-384",
        "experiment_dir": "reid_openset/megadescriptor",
        "default_lr": 0.0005,
    },
    "bioclip2": {
        "factory": "create_bioclip2_arcface_model",
        "native_size": 224,
        "patch_size": 14,
        "norm_mean": [0.48145466, 0.4578275, 0.40821073],
        "norm_std": [0.26862954, 0.26130258, 0.27577711],
        "backbone_label": "Frozen BioCLIP-2 ViT-L/14",
        "experiment_dir": "reid_openset/bioclip2",
        "default_lr": 0.0005,
    },
}


# ============================================================================
# Shared constants
# ============================================================================

ARCFACE_MARGIN = 0.5
ARCFACE_SCALE = 64
BATCH_K = 8
MIN_P = 5
TARGET_SAMPLES_PER_INDIVIDUAL = 32
QUERY_QUALITY_THRESHOLDS = [0.0, 0.25, 0.5]
EVAL_BATCH_SIZE = 128
EVAL_NUM_WORKERS = 4


# ============================================================================
# Model creation dispatcher
# ============================================================================

def create_arcface_model(model_name, embedding_dim=128, image_size=None, device="cuda"):
    """Dispatch to the appropriate factory in utils/arcface.py."""
    import utils.arcface as arcface_module

    config = MODEL_CONFIGS[model_name]
    factory = getattr(arcface_module, config["factory"])

    kwargs = {"embedding_dim": embedding_dim, "device": device}
    if model_name == "bioclip2":
        kwargs["image_size"] = image_size or config["native_size"]

    return factory(**kwargs)


# ============================================================================
# Transform creation
# ============================================================================

def create_reid_transform(size, mean, std):
    """Create backbone-specific transform pipeline."""
    return T.Compose([
        T.Resize(size=(size, size), interpolation=Image.LANCZOS),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])


def create_transform_for_model(model_name, size=None):
    """Create transform using MODEL_CONFIGS defaults."""
    config = MODEL_CONFIGS[model_name]
    if size is None:
        size = config["native_size"]
    return create_reid_transform(size, config["norm_mean"], config["norm_std"])
