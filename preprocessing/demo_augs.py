#!/usr/bin/env python3
"""
Demo augmentation effects on random samples from the reidentification dataset.

Produces a single faceted image (10 rows x 10 columns):
  rows = sampled images
  cols = baseline | blur | hflip | rotation x3 | jitter x3 | ir_sim

Each jitter column is a stochastic draw from ColorJitter(b=0.3, c=0.3, s=0.3, h=0.1),
matching the training pipeline. Blur is shown at max sigma (2.0). IR sim is binary.
"""

import os
import random
import argparse

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import torchvision.transforms as T
from datasets import load_dataset


# Match training pipeline parameters (utils/reid_config.py)
RESIZE = 224
BLUR_SIGMA = 2.0
N_JITTER_SAMPLES = 3
N_ROTATION_SAMPLES = 3
ROTATION_DEGREES = 15

# Stochastic jitter matching the training pipeline exactly
JITTER_TRANSFORM = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)

ROTATION_TRANSFORM = T.RandomRotation(degrees=ROTATION_DEGREES)

COL_LABELS = [
    "baseline", "blur", "hflip",
    "rotation", "rotation", "rotation",
    "jitter", "jitter", "jitter",
    "ir_sim",
]


def apply_baseline(img):
    return img.resize((RESIZE, RESIZE), Image.LANCZOS)


def apply_blur(img, sigma):
    img = img.resize((RESIZE, RESIZE), Image.LANCZOS)
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def apply_jitter(img):
    img = img.resize((RESIZE, RESIZE), Image.LANCZOS)
    return JITTER_TRANSFORM(img)


def apply_hflip(img):
    img = img.resize((RESIZE, RESIZE), Image.LANCZOS)
    return img.transpose(Image.FLIP_LEFT_RIGHT)


def apply_rotation(img):
    img = img.resize((RESIZE, RESIZE), Image.LANCZOS)
    return ROTATION_TRANSFORM(img)


def apply_ir_sim(img):
    img = img.resize((RESIZE, RESIZE), Image.LANCZOS)
    return img.convert("L").convert("RGB")


def build_aug_row(img):
    """Return list of 10 augmented images for one source image."""
    results = [apply_baseline(img)]
    results.append(apply_blur(img, BLUR_SIGMA))
    results.append(apply_hflip(img))
    for _ in range(N_ROTATION_SAMPLES):
        results.append(apply_rotation(img))
    for _ in range(N_JITTER_SAMPLES):
        results.append(apply_jitter(img))
    results.append(apply_ir_sim(img))
    return results


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
    n_cols = len(COL_LABELS)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.2, n_rows * 2.2))

    for row, idx in enumerate(indices):
        img = dataset[idx]["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")

        aug_images = build_aug_row(img)
        for col, aug_img in enumerate(aug_images):
            ax = axes[row, col]
            ax.imshow(np.array(aug_img))
            ax.set_xticks([])
            ax.set_yticks([])

            if row == 0:
                ax.set_title(COL_LABELS[col], fontsize=9, fontweight="bold")

    plt.tight_layout(pad=0.5)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "demo_augs.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
