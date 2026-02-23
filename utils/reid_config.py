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
TRAIN_BATCH_SIZE = 36
TARGET_SAMPLES_PER_INDIVIDUAL = 64
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
    """Create backbone-specific transform pipeline (no augmentation)."""
    return T.Compose([
        T.Resize(size=(size, size), interpolation=Image.LANCZOS),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])


def create_reid_train_transform(size, mean, std, blur=False, jitter=False, ir_sim=False):
    """Create training transform with optional augmentations.

    Augmentations are applied before ToTensor/Normalize, each with p=0.5:
      - blur: Light Gaussian blur (kernel=5, sigma 0.1-2.0)
      - jitter: Color jitter (brightness/contrast/saturation=0.3, hue=0.1)
      - ir_sim: Grayscale conversion to simulate IR camera images

    Returns the base (no-augmentation) transform if all flags are False.
    """
    ops = [T.Resize(size=(size, size), interpolation=Image.LANCZOS)]

    if blur:
        ops.append(T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.5))
    if jitter:
        ops.append(T.RandomApply([T.ColorJitter(brightness=0.3, contrast=0.3,
                                                 saturation=0.3, hue=0.1)], p=0.5))
    if ir_sim:
        ops.append(T.RandomGrayscale(p=0.5))

    ops.extend([T.ToTensor(), T.Normalize(mean=mean, std=std)])
    return T.Compose(ops)


def create_transform_for_model(model_name, size=None):
    """Create eval transform using MODEL_CONFIGS defaults (no augmentation)."""
    config = MODEL_CONFIGS[model_name]
    if size is None:
        size = config["native_size"]
    return create_reid_transform(size, config["norm_mean"], config["norm_std"])


def create_train_transform_for_model(model_name, size=None,
                                      blur=False, jitter=False, ir_sim=False):
    """Create training transform with optional augmentations."""
    config = MODEL_CONFIGS[model_name]
    if size is None:
        size = config["native_size"]
    return create_reid_train_transform(size, config["norm_mean"], config["norm_std"],
                                        blur=blur, jitter=jitter, ir_sim=ir_sim)
