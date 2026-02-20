"""
Tune epoch count: train on all unfiltered data (minus val), evaluate on val each epoch.

Uses the same train/val splitting regime as the hygiene sweep (greedy temporal
validation indices from feasibility config). No quality filtering, no gallery_size
cap. Gallery = all training images for qualified individuals minus validation images.

Uses the persistent-pointer BalancedBatchSampler so epoch length is based on the
smallest class (one pass through the rarest individual per epoch).

Usage:
    cd /home/kdoherty/wolverines
    python -u reid_openset/dinov3/debug/tune_test_epoch_count.py \
        --device gpu --epochs 100 --eval-every 5 --seed 0
"""

import os
import sys
import json
import math
import time
import random
import argparse
import numpy as np
import torch
from collections import defaultdict
from datetime import datetime
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HOME"] = "/data/hf_cache"

# Ensure project root is on sys.path when run as a script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from utils.dataset import set_all_seeds
from utils.reid_config import (
    MODEL_CONFIGS, ARCFACE_MARGIN, ARCFACE_SCALE, TRAIN_BATCH_SIZE,
    EVAL_BATCH_SIZE, EVAL_NUM_WORKERS,
    create_arcface_model, create_transform_for_model,
)
from utils.reid_data import (
    ArcFaceDataset,
    load_feasibility_config, load_reidentification_dataset,
    build_metadata_cache, get_rare_individual_indices,
)
from utils.reid_evaluation import (
    evaluate_recall_with_openset,
)
from utils.reid import (
    count_trainable_parameters, load_best_hyperparams,
)


MODEL_NAME = "dinov3"


class BalancedBatchSampler(Sampler):
    """
    Balanced batch sampler with persistent pointers across epochs.

    Epoch length = ceil(min_class_size / k), so one epoch = one pass through
    the rarest individual. Large classes are consumed sequentially across
    epochs; a class is only reshuffled when its pointer wraps. Every image
    is seen exactly once before any repeats.
    """
    def __init__(self, labels, batch_size=36):
        super().__init__(None)
        self.batch_size = batch_size
        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.label_to_indices[label].append(idx)

        self.n_classes = len(self.label_to_indices)
        self.k = batch_size // self.n_classes
        if self.k == 0:
            self.k = 1
        self.remainder = batch_size - (self.k * self.n_classes)
        self.min_class_size = min(len(v) for v in self.label_to_indices.values())

        # Persistent state across epochs
        self.class_indices = {}
        self.pointers = {}
        for label, indices in self.label_to_indices.items():
            shuffled = indices.copy()
            random.shuffle(shuffled)
            self.class_indices[label] = shuffled
            self.pointers[label] = 0

    def _next_from_class(self, label):
        """Get next index, wrapping with reshuffle when exhausted."""
        indices = self.class_indices[label]
        ptr = self.pointers[label]
        if ptr >= len(indices):
            random.shuffle(indices)
            self.class_indices[label] = indices
            ptr = 0
        idx = indices[ptr]
        self.pointers[label] = ptr + 1
        return idx

    def __iter__(self):
        labels = list(self.label_to_indices.keys())
        n_batches = math.ceil(self.min_class_size / self.k)

        for _ in range(n_batches):
            batch = []
            for label in labels:
                for _ in range(self.k):
                    batch.append(self._next_from_class(label))
            if self.remainder > 0:
                extra_labels = random.sample(labels, self.remainder)
                for label in extra_labels:
                    batch.append(self._next_from_class(label))
            yield batch

    def __len__(self):
        return math.ceil(self.min_class_size / self.k)


def main():
    parser = argparse.ArgumentParser(description="Tune epoch count: unfiltered DINOv3, val eval each epoch")
    parser.add_argument("--device", type=str, choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=5,
                        help="Evaluate on val every N epochs (always eval on last epoch)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    seed = args.seed
    epochs = args.epochs
    eval_every = args.eval_every
    set_all_seeds(seed)

    output_dir = os.path.join(MODEL_CONFIGS[MODEL_NAME]["experiment_dir"], "debug")
    output_path = os.path.join(output_dir, f"tune_epoch_count_seed={seed}.json")

    if os.path.exists(output_path) and not args.overwrite:
        print(f"Result exists: {output_path}. Use --overwrite to replace.")
        return

    print("=" * 70)
    print(f"Tune Epoch Count: DINOv3 unfiltered, val eval each epoch, seed={seed}")
    print("=" * 70)

    # Load best hyperparams from opt sweeps
    best_lr, best_size, best_embedding_dim = load_best_hyperparams(MODEL_NAME)

    # Load dataset and config
    print("\nLoading dataset...")
    dataset = load_reidentification_dataset()
    metadata_cache = build_metadata_cache(dataset)

    feasibility_config = load_feasibility_config()
    if not feasibility_config:
        print("Failed to load feasibility config")
        return

    feasible_individuals = feasibility_config.get("qualified_individuals", [])
    promoted_individuals = feasibility_config.get("promoted_to_rare", [])
    excluded_individuals = feasibility_config.get("excluded_entirely", [])
    validation_indices = feasibility_config.get("validation_indices", {})

    print(f"Using {len(feasible_individuals)} individuals: {', '.join(feasible_individuals)}")

    # Split: train = all images minus val, val = greedy temporal indices
    id_to_indices = metadata_cache["id_to_indices"]
    individual_to_class = {ind: i for i, ind in enumerate(sorted(feasible_individuals))}
    class_to_name = {v: k for k, v in individual_to_class.items()}

    all_train_indices = []
    all_val_indices = []
    gallery_info = {}
    query_info = {}

    for ind_id in feasible_individuals:
        val_idx = validation_indices.get(ind_id, {}).get("indices", [])
        val_idx_set = set(val_idx)
        all_val_indices.extend(val_idx)
        query_info[ind_id] = len(val_idx)

        all_ind_indices = list(id_to_indices.get(ind_id, []))
        train_only = [i for i in all_ind_indices if i not in val_idx_set]
        all_train_indices.extend(train_only)
        gallery_info[ind_id] = len(train_only)

        print(f"  {ind_id}: {len(train_only)} train, {len(val_idx)} val")

    train_dataset = dataset.select(all_train_indices)
    val_dataset = dataset.select(all_val_indices)
    print(f"\nTotal: {len(all_train_indices)} train, {len(all_val_indices)} val")

    # Create model
    device = "cuda" if args.device == "gpu" and torch.cuda.is_available() else "cpu"
    model, emb_dim = create_arcface_model(MODEL_NAME, embedding_dim=best_embedding_dim,
                                           image_size=best_size, device=device)
    print(f"Using device: {device}")

    # Transforms and training dataloader
    transform = create_transform_for_model(MODEL_NAME, size=best_size)
    train_torch = ArcFaceDataset(train_dataset, transform, individual_to_class)

    from utils.arcface import ArcFaceLoss

    sampler = BalancedBatchSampler(
        labels=train_torch.get_labels(),
        batch_size=TRAIN_BATCH_SIZE,
    )
    print(f"  Sampler: {len(sampler)} batches/epoch, batch_size={TRAIN_BATCH_SIZE}, "
          f"k={sampler.k}/class, min_class={sampler.min_class_size}, "
          f"max_class={max(len(v) for v in sampler.label_to_indices.values())}")

    def collate_fn(batch):
        images = torch.stack([item[0] for item in batch])
        labels = torch.tensor([item[1] for item in batch])
        quality = torch.tensor([item[2] for item in batch])
        return images, labels, quality

    train_loader = DataLoader(train_torch, batch_sampler=sampler,
                              num_workers=0, collate_fn=collate_fn)

    # ArcFace loss + optimizer
    arcface_loss = ArcFaceLoss(
        num_classes=len(feasible_individuals),
        embedding_size=emb_dim,
        margin=ARCFACE_MARGIN,
        scale=ARCFACE_SCALE,
    ).to(device)

    head_params = count_trainable_parameters(model.head)
    arcface_params = count_trainable_parameters(arcface_loss)
    print(f"Trainable: head={head_params:,}, arcface={arcface_params:,}")

    optimizer = torch.optim.AdamW(
        list(model.get_trainable_parameters()) + list(arcface_loss.parameters()),
        lr=best_lr,
    )

    # Warm up the dataloader (first iteration triggers HF dataset decoding)
    print("\nWarming up dataloader (first batch)...", flush=True)
    _warmup_iter = iter(train_loader)
    _warmup_batch = next(_warmup_iter)
    del _warmup_iter, _warmup_batch
    print("Dataloader ready.")

    # Training loop with val evaluation every N epochs
    print(f"\nTraining for {epochs} epochs (eval every {eval_every} epochs)...")
    use_amp = (device == "cuda") or (isinstance(device, torch.device) and device.type == "cuda")
    start_time = time.time()
    epoch_history = []
    best_recall = 0.0
    best_recall_epoch = 0
    best_ba = 0.0
    best_ba_epoch = 0
    best_hm = 0.0
    best_hm_epoch = 0

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        arcface_loss.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:3d}/{epochs}", leave=False)
        for batch in pbar:
            images = batch[0].to(device)
            labels = batch[1].to(device)

            embeddings = model(images)
            optimizer.zero_grad()
            loss = arcface_loss(embeddings, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=f"{total_loss / num_batches:.4f}")

        train_loss = total_loss / max(num_batches, 1)

        # --- Evaluate on val every N epochs (and always on last) ---
        is_eval_epoch = ((epoch + 1) % eval_every == 0) or (epoch + 1 == epochs)

        if is_eval_epoch:
            query_quality_metrics, open_set_metrics, val_loss = evaluate_recall_with_openset(
                model, train_dataset, val_dataset, individual_to_class, transform, device,
                dataset, feasible_individuals, metadata_cache,
                criterion=arcface_loss, embedding_dim=emb_dim,
                promoted_individuals=promoted_individuals,
                excluded_individuals=excluded_individuals,
            )

            recall_1 = query_quality_metrics["q>=0.0"]["recall_at_1"]
            by_quality = open_set_metrics.get("by_quality", {})
            ba_q0 = by_quality.get("q>=0.0", {}).get("balanced_accuracy", 0.0)
            kar = by_quality.get("q>=0.0", {}).get("known_accept_rate", 0.0)
            urr = by_quality.get("q>=0.0", {}).get("unknown_reject_rate", 0.0)
            thresh = by_quality.get("q>=0.0", {}).get("cosine_threshold", 0.0)
            n_k = by_quality.get("q>=0.0", {}).get("n_known_individuals", 0)
            n_u = by_quality.get("q>=0.0", {}).get("n_unknown_individuals", 0)

            if recall_1 > best_recall:
                best_recall = recall_1
                best_recall_epoch = epoch + 1
            if ba_q0 > best_ba:
                best_ba = ba_q0
                best_ba_epoch = epoch + 1
            hm = 2 * recall_1 * ba_q0 / (recall_1 + ba_q0) if (recall_1 + ba_q0) > 0 else 0.0
            if hm > best_hm:
                best_hm = hm
                best_hm_epoch = epoch + 1

            print(f"Epoch {epoch+1:3d}/{epochs}: Loss={train_loss:.4f}, ValLoss={val_loss:.4f}, "
                  f"R@1={recall_1:.4f}, BA={ba_q0:.4f} (K={kar:.2f}[{n_k}], U={urr:.2f}[{n_u}]), "
                  f"thresh={thresh:.3f}")

            epoch_history.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "recall_at_1": recall_1,
                "balanced_accuracy": ba_q0,
                "known_accept_rate": kar,
                "unknown_reject_rate": urr,
                "cosine_threshold": thresh,
            })
        else:
            print(f"Epoch {epoch+1:3d}/{epochs}: Loss={train_loss:.4f}")
            epoch_history.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
            })

    training_time = time.time() - start_time

    print(f"\nBest R@1:    epoch {best_recall_epoch}, R@1={best_recall:.4f}")
    print(f"Best BA:     epoch {best_ba_epoch}, BA={best_ba:.4f}")
    print(f"Best H-Mean: epoch {best_hm_epoch}, H-Mean={best_hm:.4f}")

    result = {
        "config": {
            "seed": seed,
            "epochs": epochs,
            "learning_rate": best_lr,
            "image_size": best_size,
            "embedding_dim": best_embedding_dim,
            "loss": "ArcFace",
            "arcface_margin": ARCFACE_MARGIN,
            "arcface_scale": ARCFACE_SCALE,
            "backbone": MODEL_CONFIGS[MODEL_NAME]["backbone_label"],
            "quality_threshold": 0.0,
            "gallery_mode": "all_unfiltered_minus_val",
        },
        "dataset": {
            "individuals": feasible_individuals,
            "gallery_info": gallery_info,
            "query_info": query_info,
            "total_train": len(all_train_indices),
            "total_val": len(all_val_indices),
        },
        "epoch_history": epoch_history,
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "training_time_seconds": training_time,
            "best_recall_epoch": best_recall_epoch,
            "best_recall_at_1": best_recall,
            "best_ba_epoch": best_ba_epoch,
            "best_balanced_accuracy": best_ba,
            "best_hm_epoch": best_hm_epoch,
            "best_harmonic_mean": best_hm,
        },
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2,
                  default=lambda o: float(o) if isinstance(o, np.floating)
                  else int(o) if isinstance(o, np.integer) else o)

    print(f"\nSaved: {output_path}")
    print(f"Training time: {training_time:.1f}s")


if __name__ == "__main__":
    main()
