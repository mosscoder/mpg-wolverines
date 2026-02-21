"""
Sweep gallery quality threshold × epoch sample size.

  3 gallery filters: g>=0.00, g>=0.25, g>=0.50
  4 sample sizes:    32, 64, 128, 256 images/individual/epoch

Each epoch, randomly draw N images per individual (with replacement if
the individual has fewer than N). Eval always uses the full quality-
filtered gallery.

Each eval computes R@1 and open-set BA at all query quality levels
(q>=0.0, q>=0.25, q>=0.5).

Usage:
    cd /home/kdoherty/wolverines
    python -u reid_openset/dinov3/debug/tune_test_epoch_count.py \
        --idx 0 --device gpu --epochs 200 --eval-every 5 --seed 0
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
    EVAL_BATCH_SIZE,
    QUERY_QUALITY_THRESHOLDS,
    create_arcface_model, create_transform_for_model,
)
from utils.reid_data import (
    ArcFaceDataset,
    load_feasibility_config, load_reidentification_dataset,
    build_metadata_cache,
    filter_training_pool_by_quality,
)
from utils.reid_evaluation import evaluate_recall_with_openset
from utils.reid import count_trainable_parameters, load_best_hyperparams


MODEL_NAME = "dinov3"
GALLERY_THRESHOLDS = [0.0, 0.25, 0.5]
SAMPLE_SIZES = [32, 64, 128, 256]

# idx -> (label, gallery quality threshold, epoch_sample_size)
# 3 thresholds × 3 sample sizes = 9 jobs
JOB_CONFIGS = {}
idx = 0
for gt in GALLERY_THRESHOLDS:
    for ss in SAMPLE_SIZES:
        label = f"g{gt:.2f}_n{ss}"
        JOB_CONFIGS[idx] = (label, gt, ss)
        idx += 1


class EpochSubsampleSampler(Sampler):
    """
    Each epoch, randomly draw N images per individual from the full pool,
    then yield balanced batches from that subsample. Fresh draw each epoch.
    Uses replacement for individuals with fewer than N images.
    """
    def __init__(self, labels, batch_size=36, epoch_sample_size=64):
        super().__init__(None)
        self.batch_size = batch_size
        self.epoch_sample_size = epoch_sample_size

        self.label_to_all_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.label_to_all_indices[label].append(idx)

        self.n_classes = len(self.label_to_all_indices)
        self.k = batch_size // self.n_classes
        if self.k == 0:
            self.k = 1
        self.remainder = batch_size - (self.k * self.n_classes)

    def __iter__(self):
        # Fresh random subsample per class for this epoch
        # With replacement if pool < N, without replacement otherwise
        epoch_indices = {}
        for label, all_indices in self.label_to_all_indices.items():
            if len(all_indices) >= self.epoch_sample_size:
                sampled = random.sample(all_indices, self.epoch_sample_size)
            else:
                sampled = random.choices(all_indices, k=self.epoch_sample_size)
            random.shuffle(sampled)
            epoch_indices[label] = sampled

        labels = list(epoch_indices.keys())
        n_batches = math.ceil(self.epoch_sample_size / self.k)
        pointers = {label: 0 for label in labels}

        for _ in range(n_batches):
            batch = []
            for label in labels:
                for _ in range(self.k):
                    indices = epoch_indices[label]
                    ptr = pointers[label]
                    if ptr >= len(indices):
                        random.shuffle(indices)
                        ptr = 0
                    batch.append(indices[ptr])
                    pointers[label] = ptr + 1
            if self.remainder > 0:
                extra_labels = random.sample(labels, self.remainder)
                for label in extra_labels:
                    indices = epoch_indices[label]
                    ptr = pointers[label]
                    if ptr >= len(indices):
                        random.shuffle(indices)
                        ptr = 0
                    batch.append(indices[ptr])
                    pointers[label] = ptr + 1
            yield batch

    def __len__(self):
        return math.ceil(self.epoch_sample_size / self.k)


def build_filtered_split(dataset, metadata_cache, feasible_individuals, feasibility_config,
                         threshold, seed):
    """Quality-filtered split: all images passing threshold."""
    set_all_seeds(seed)
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
        eligible = filter_training_pool_by_quality(metadata_cache, train_only, threshold)
        all_train_indices.extend(eligible)
        gallery_info[ind_id] = len(eligible)

        print(f"  {ind_id}: {len(eligible)}/{len(train_only)} pass threshold>={threshold}, {len(val_idx)} val")

    train_ds = dataset.select(all_train_indices) if all_train_indices else None
    val_ds = dataset.select(all_val_indices)
    print(f"Total: {len(all_train_indices)} train (filtered), {len(all_val_indices)} val")

    dataset_info = {
        "gallery_info": gallery_info,
        "query_info": query_info,
        "total_train": len(all_train_indices),
        "total_val": len(all_val_indices),
        "quality_threshold": threshold,
    }
    return train_ds, val_ds, individual_to_class, dataset_info


def run_training(mode_label, train_dataset, val_dataset, individual_to_class, dataset_info,
                 dataset, metadata_cache, feasible_individuals, promoted_individuals,
                 excluded_individuals, best_lr, best_size, best_embedding_dim,
                 gallery_threshold, epoch_sample_size, epochs, eval_every, seed,
                 device_str, output_path, overwrite):
    """Train + eval loop shared across all conditions."""
    from utils.arcface import ArcFaceLoss

    if os.path.exists(output_path) and not overwrite:
        print(f"Result exists: {output_path}. Use --overwrite to replace.")
        return

    if train_dataset is None or len(train_dataset) == 0:
        print(f"No training data for {mode_label}, skipping.")
        return

    device = "cuda" if device_str == "gpu" and torch.cuda.is_available() else "cpu"
    model, emb_dim = create_arcface_model(MODEL_NAME, embedding_dim=best_embedding_dim,
                                           image_size=best_size, device=device)
    print(f"Using device: {device}")

    transform = create_transform_for_model(MODEL_NAME, size=best_size)
    train_torch = ArcFaceDataset(train_dataset, transform, individual_to_class)

    sampler = EpochSubsampleSampler(
        labels=train_torch.get_labels(),
        batch_size=TRAIN_BATCH_SIZE,
        epoch_sample_size=epoch_sample_size,
    )
    class_sizes = {l: len(v) for l, v in sampler.label_to_all_indices.items()}
    n_resampled = sum(1 for v in class_sizes.values() if v < epoch_sample_size)
    print(f"  Sampler: EpochSubsample, {len(sampler)} batches/epoch, "
          f"sample={epoch_sample_size}/ind/epoch, batch_size={TRAIN_BATCH_SIZE}, "
          f"k={sampler.k}/class, pool min={min(class_sizes.values())}, "
          f"pool max={max(class_sizes.values())}, "
          f"resampled={n_resampled}/{len(class_sizes)} individuals")

    def collate_fn(batch):
        images = torch.stack([item[0] for item in batch])
        labels = torch.tensor([item[1] for item in batch])
        quality = torch.tensor([item[2] for item in batch])
        return images, labels, quality

    train_loader = DataLoader(train_torch, batch_sampler=sampler,
                              num_workers=0, collate_fn=collate_fn)

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

    # Warm up dataloader
    print("\nWarming up dataloader (first batch)...", flush=True)
    _warmup_iter = iter(train_loader)
    _warmup_batch = next(_warmup_iter)
    del _warmup_iter, _warmup_batch
    print("Dataloader ready.")

    print(f"\nTraining for {epochs} epochs (eval every {eval_every} epochs)...")
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

            embeddings = model(images)
            optimizer.zero_grad()
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
            "mode": mode_label,
            "gallery_threshold": gallery_threshold,
            "epoch_sample_size": epoch_sample_size,
            "seed": seed,
            "epochs": epochs,
            "eval_every": eval_every,
            "learning_rate": best_lr,
            "image_size": best_size,
            "embedding_dim": best_embedding_dim,
            "loss": "ArcFace",
            "arcface_margin": ARCFACE_MARGIN,
            "arcface_scale": ARCFACE_SCALE,
            "backbone": MODEL_CONFIGS[MODEL_NAME]["backbone_label"],
        },
        "dataset": dataset_info,
        "epoch_history": epoch_history,
        "best_metrics": best_metrics,
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "training_time_seconds": training_time,
        },
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2,
                  default=lambda o: float(o) if isinstance(o, np.floating)
                  else int(o) if isinstance(o, np.integer) else o)

    print(f"\nSaved: {output_path}")
    print(f"Training time: {training_time:.1f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Sweep gallery threshold × epoch sample size (9 configs)")
    parser.add_argument("--idx", type=int, required=True,
                        help="Job index 0-11 (3 gallery thresholds × 4 sample sizes)")
    parser.add_argument("--device", type=str, choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    idx = args.idx
    if idx not in JOB_CONFIGS:
        print(f"idx={idx} has no work assigned, exiting.")
        return

    mode_label, gallery_threshold, epoch_sample_size = JOB_CONFIGS[idx]
    seed = args.seed
    epochs = args.epochs
    eval_every = args.eval_every
    set_all_seeds(seed)

    output_dir = os.path.join(MODEL_CONFIGS[MODEL_NAME]["experiment_dir"], "debug")
    output_path = os.path.join(output_dir, f"tune_epoch_count_{mode_label}_seed={seed}.json")

    if os.path.exists(output_path) and not args.overwrite:
        print(f"Result exists: {output_path}. Use --overwrite to replace.")
        return

    print("=" * 70)
    print(f"Tune Epoch Count [{mode_label}]: DINOv3, seed={seed}, idx={idx}")
    print("=" * 70)

    best_lr, best_size, best_embedding_dim = load_best_hyperparams(MODEL_NAME)

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
    print(f"\nMode: {mode_label} (threshold>={gallery_threshold}, "
          f"train {epoch_sample_size}/ind/epoch)")

    train_ds, val_ds, individual_to_class, dataset_info = build_filtered_split(
        dataset, metadata_cache, feasible_individuals, feasibility_config,
        gallery_threshold, seed,
    )
    dataset_info["mode"] = mode_label
    dataset_info["gallery_threshold"] = gallery_threshold

    run_training(
        mode_label=mode_label,
        train_dataset=train_ds,
        val_dataset=val_ds,
        individual_to_class=individual_to_class,
        dataset_info=dataset_info,
        dataset=dataset,
        metadata_cache=metadata_cache,
        feasible_individuals=feasible_individuals,
        promoted_individuals=promoted_individuals,
        excluded_individuals=excluded_individuals,
        best_lr=best_lr,
        best_size=best_size,
        best_embedding_dim=best_embedding_dim,
        gallery_threshold=gallery_threshold,
        epoch_sample_size=epoch_sample_size,
        epochs=epochs,
        eval_every=eval_every,
        seed=seed,
        device_str=args.device,
        output_path=output_path,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
