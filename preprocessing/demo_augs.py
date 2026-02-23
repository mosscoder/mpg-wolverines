#!/usr/bin/env python3
"""
Demo augmentation effects on random samples from the reidentification dataset.

Produces a single faceted image (10 rows × 4 columns):
  rows = sampled images, cols = baseline | blur | jitter | ir_sim

Augmentations are applied deterministically (always on) so the effect is visible,
matching the parameters used in the training pipeline (utils/reid_config.py).
"""

import os
import random
import argparse

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
from torchvision.transforms.functional import adjust_brightness, adjust_contrast, adjust_saturation, adjust_hue, to_pil_image, to_tensor
from datasets import load_dataset


# Match training pipeline parameters (utils/reid_config.py)
RESIZE = 224
BLUR_KERNEL = 5
BLUR_SIGMA = 2.0  # max of (0.1, 2.0) range
JITTER_BRIGHTNESS = 1.3
JITTER_CONTRAST = 1.3
JITTER_SATURATION = 1.3
JITTER_HUE = 0.01

AUG_LABELS = ["baseline", "blur", "jitter", "ir_sim"]


def apply_baseline(img):
    return img.resize((RESIZE, RESIZE), Image.LANCZOS)


def apply_blur(img):
    img = img.resize((RESIZE, RESIZE), Image.LANCZOS)
    return img.filter(ImageFilter.GaussianBlur(radius=BLUR_SIGMA))


def apply_jitter(img):
    img = img.resize((RESIZE, RESIZE), Image.LANCZOS)
    t = to_tensor(img)
    t = adjust_brightness(t, JITTER_BRIGHTNESS)
    t = adjust_contrast(t, JITTER_CONTRAST)
    t = adjust_saturation(t, JITTER_SATURATION)
    t = adjust_hue(t, JITTER_HUE)
    return to_pil_image(t.clamp(0, 1))


def apply_ir_sim(img):
    img = img.resize((RESIZE, RESIZE), Image.LANCZOS)
    return img.convert("L").convert("RGB")


AUGMENTATIONS = [apply_baseline, apply_blur, apply_jitter, apply_ir_sim]


def main():
    parser = argparse.ArgumentParser(description="Demo augmentation effects")
    parser.add_argument("--n", type=int, default=10, help="Number of samples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="preprocessing/results")
    args = parser.parse_args()

    random.seed(args.seed)

    print("Loading dataset...")
    dataset = load_dataset("kdoherty/wolverines", "reidentification", split="train")

    indices = random.sample(range(len(dataset)), args.n)
    n_rows = args.n
    n_cols = len(AUG_LABELS)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.2, n_rows * 2.2))

    for row, idx in enumerate(indices):
        img = dataset[idx]["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")

        for col, (label, aug_fn) in enumerate(zip(AUG_LABELS, AUGMENTATIONS)):
            ax = axes[row, col]
            ax.imshow(np.array(aug_fn(img)))
            ax.set_xticks([])
            ax.set_yticks([])

            if row == 0:
                ax.set_title(label, fontsize=11, fontweight="bold")

    plt.tight_layout(pad=0.5)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "demo_augs.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
