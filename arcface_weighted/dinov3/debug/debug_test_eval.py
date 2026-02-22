"""
Debug test eval: Weighted ArcFace DINOv3, all training samples, periodic eval.

No quality filtering, no gallery_size cap, no event-matched subsampling.
Trains on ALL training images for qualified individuals, evaluates every
N epochs with full R@1 + open-set BA across all query quality thresholds.

Two configs via --alpha:
  alpha=0  unweighted baseline (standard ArcFace)
  alpha=2  strong quality emphasis (weight = 1 + quality * 2)

Usage:
    cd /home/kdoherty/wolverines
    python -u arcface_weighted/dinov3/debug/debug_test_eval.py \
        --device gpu --epochs 200 --eval-every 5 --seed 0 --alpha 0
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
    QUERY_QUALITY_THRESHOLDS, EVAL_BATCH_SIZE, EVAL_NUM_WORKERS,
    create_arcface_model, create_transform_for_model,
)
from utils.reid_data import (
    ArcFaceDataset,
    load_feasibility_config, load_reidentification_dataset,
    build_metadata_cache,
)
from utils.reid_evaluation import evaluate_recall_with_openset
from utils.arcface_weighted_reid import (
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
    parser = argparse.ArgumentParser(description="Debug test eval: Weighted ArcFace DINOv3")
    parser.add_argument("--device", type=str, choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=int, required=True, choices=[0, 2],
                        help="Quality weighting strength: 0=unweighted, 2=strong")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    seed = args.seed
    epochs = args.epochs
    eval_every = args.eval_every
    alpha = args.alpha
    set_all_seeds(seed)

    experiment_dir = MODEL_CONFIGS[MODEL_NAME]["experiment_dir"].replace(
        "reid_openset/", "arcface_weighted/", 1)
    output_dir = os.path.join(experiment_dir, "debug")
    output_path = os.path.join(output_dir, f"debug_test_eval_alpha={alpha}_seed={seed}.json")

    if os.path.exists(output_path) and not args.overwrite:
        print(f"Result exists: {output_path}. Use --overwrite to replace.")
        return

    print("=" * 70)
    print(f"Debug Test Eval: Weighted ArcFace DINOv3, alpha={alpha}, all training samples, seed={seed}")
    if alpha > 0:
        print(f"  Weight formula: weight = 1 + quality * {alpha}")
    else:
        print(f"  Standard ArcFace (no quality weighting)")
    print("=" * 70)

    # Load best hyperparams from opt sweeps
    best_lr, best_size, best_embedding_dim = load_best_hyperparams(MODEL_NAME)

    # Load dataset (single split — gallery = train, query = validation held out by feasibility config)
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

    print(f"Using {len(feasible_individuals)} individuals: {', '.join(feasible_individuals)}")

    # Gallery: ALL training images for qualified individuals (no filtering)
    id_to_indices = metadata_cache["id_to_indices"]
    validation_indices = feasibility_config.get("validation_indices", {})
    individual_to_class = {ind: i for i, ind in enumerate(sorted(feasible_individuals))}

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

    # Training loop with periodic eval
    print(f"\nTraining for {epochs} epochs (eval every {eval_every} epochs, alpha={alpha})...")
    use_amp = (device == "cuda") or (isinstance(device, torch.device) and device.type == "cuda")
    start_time = time.time()
    epoch_history = []

    # Track best R@1 and BA per query quality level
    q_keys = [f"q>={q}" for q in QUERY_QUALITY_THRESHOLDS]
    best_metrics = {}
    for q_key in q_keys:
        best_metrics[q_key] = {
            "best_recall": 0.0, "best_recall_epoch": 0,
            "best_ba": 0.0, "best_ba_epoch": 0,
        }

    for epoch in range(epochs):
        model.train()
        arcface_loss.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:3d}/{epochs}", leave=False)
        for batch in pbar:
            images = batch[0].to(device)
            labels = batch[1].to(device)
            quality = batch[2].to(device)

            embeddings = model(images)
            optimizer.zero_grad()

            if alpha > 0:
                weights = 1.0 + quality * alpha
                loss = arcface_loss(embeddings, labels, sample_weights=weights)
            else:
                loss = arcface_loss(embeddings, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=f"{total_loss / num_batches:.4f}")

        train_loss = total_loss / max(num_batches, 1)

        is_eval_epoch = ((epoch + 1) % eval_every == 0) or (epoch + 1 == epochs)

        if is_eval_epoch:
            query_quality_metrics, open_set_metrics, val_loss = evaluate_recall_with_openset(
                model, train_dataset, val_dataset, individual_to_class, transform, device,
                dataset, feasible_individuals, metadata_cache,
                criterion=arcface_loss, embedding_dim=emb_dim,
                promoted_individuals=promoted_individuals,
                excluded_individuals=excluded_individuals,
            )

            cos_thresh = open_set_metrics.get("threshold_calibration", {}).get("global_threshold", 0.0)
            print(f"Epoch {epoch+1:3d}/{epochs}: Loss={train_loss:.4f}, ValLoss={val_loss:.4f}, "
                  f"cos_thresh={cos_thresh:.3f}")
            epoch_entry = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "query_quality_metrics": query_quality_metrics,
                "open_set_metrics": open_set_metrics,
            }

            for q_key in q_keys:
                r1 = query_quality_metrics.get(q_key, {}).get("recall_at_1", 0.0)
                os_q = open_set_metrics.get("by_quality", {}).get(q_key, {})
                ba = os_q.get("balanced_accuracy", 0.0)
                kar = os_q.get("known_accept_rate", 0.0)
                urr = os_q.get("unknown_reject_rate", 0.0)
                n_k = os_q.get("n_known_individuals", 0)
                n_u = os_q.get("n_unknown_individuals", 0)

                bm = best_metrics[q_key]
                if r1 > bm["best_recall"]:
                    bm["best_recall"] = r1
                    bm["best_recall_epoch"] = epoch + 1
                if ba > bm["best_ba"]:
                    bm["best_ba"] = ba
                    bm["best_ba_epoch"] = epoch + 1

                print(f"  {q_key}: R@1={r1:.4f}, BA={ba:.4f} (K={kar:.2f}[{n_k}], U={urr:.2f}[{n_u}])")

            epoch_history.append(epoch_entry)
        else:
            print(f"Epoch {epoch+1:3d}/{epochs}: Loss={train_loss:.4f}")
            epoch_history.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
            })

    training_time = time.time() - start_time

    print(f"\nBest metrics by query quality:")
    for q_key in q_keys:
        bm = best_metrics[q_key]
        print(f"  {q_key}: R@1={bm['best_recall']:.4f} (ep {bm['best_recall_epoch']}), "
              f"BA={bm['best_ba']:.4f} (ep {bm['best_ba_epoch']})")

    result = {
        "config": {
            "alpha": alpha,
            "seed": seed,
            "epochs": epochs,
            "eval_every": eval_every,
            "learning_rate": best_lr,
            "image_size": best_size,
            "embedding_dim": best_embedding_dim,
            "loss": "WeightedArcFace",
            "arcface_margin": ARCFACE_MARGIN,
            "arcface_scale": ARCFACE_SCALE,
            "backbone": MODEL_CONFIGS[MODEL_NAME]["backbone_label"],
            "quality_threshold": 0.0,
            "gallery_mode": "all_unfiltered",
        },
        "dataset": {
            "individuals": feasible_individuals,
            "gallery_info": gallery_info,
            "query_info": query_info,
            "total_train": len(all_train_indices),
            "total_val": len(all_val_indices),
        },
        "epoch_history": epoch_history,
        "best_metrics": best_metrics,
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "training_time_seconds": training_time,
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
