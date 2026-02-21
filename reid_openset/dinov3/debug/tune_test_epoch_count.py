"""
Tune epoch count across three gallery conditions, each on its own node.

  idx=0  Unfiltered: all training images (minus val), no quality filter
  idx=1  Best R@1 filtered: best threshold from hygiene by recall
  idx=2  Best BA filtered: best threshold from hygiene by balanced_accuracy

Trains on all data passing the filter. At each eval epoch, embeds the full
gallery once, then runs k-means (k=32, 64, 128) per individual to select
representative gallery subsets. R@1 and open-set BA are computed against
each k-subset, giving hygiene-comparable metrics across gallery sizes.

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
import torch.nn.functional as F
from collections import defaultdict
from datetime import datetime
from sklearn.cluster import KMeans
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
    filter_training_pool_by_quality,
)
from utils.reid_evaluation import (
    compute_cosine_similarity, compute_validation_loss,
    compute_rare_embeddings, compute_arcface_center_thresholds,
    compute_open_set_metrics_per_individual_threshold,
)
from utils.reid import (
    count_trainable_parameters, load_best_hyperparams, load_best_hygiene_config,
)


MODEL_NAME = "dinov3"
EVAL_K_VALUES = [32, 64, 128]

# idx -> (label, hygiene criterion or None)
JOB_CONFIGS = {
    0: ("unfiltered", None),
    1: ("best_recall", "recall"),
    2: ("best_ba", "balanced_accuracy"),
}


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


def kmeans_select_gallery(gallery_embeddings, gallery_labels, k, device):
    """
    Select k representative gallery images per individual via k-means.

    For each class, runs k-means on the L2-normalized embeddings and picks
    the image nearest each cluster center.

    Returns:
        selected_indices: indices into the gallery tensors for the subset
        subset_embeddings: (n_classes * k, emb_dim) tensor
        subset_labels: (n_classes * k,) tensor
    """
    gallery_emb_np = F.normalize(gallery_embeddings, p=2, dim=1).cpu().numpy()
    gallery_lab_np = gallery_labels.cpu().numpy()
    unique_labels = np.unique(gallery_lab_np)

    selected_indices = []

    for label in unique_labels:
        mask = gallery_lab_np == label
        class_indices = np.where(mask)[0]
        class_emb = gallery_emb_np[class_indices]

        n_samples = len(class_indices)
        actual_k = min(k, n_samples)

        if actual_k == n_samples:
            selected_indices.extend(class_indices.tolist())
            continue

        kmeans = KMeans(n_clusters=actual_k, random_state=42, n_init=10)
        kmeans.fit(class_emb)

        # For each center, find nearest image
        centers = kmeans.cluster_centers_
        for center in centers:
            dists = np.linalg.norm(class_emb - center, axis=1)
            nearest = dists.argmin()
            selected_indices.append(int(class_indices[nearest]))

    selected_indices = sorted(set(selected_indices))
    subset_embeddings = gallery_embeddings[selected_indices]
    subset_labels = gallery_labels[selected_indices]

    return selected_indices, subset_embeddings, subset_labels


def compute_recall_at_1(query_embeddings, query_labels, gallery_embeddings, gallery_labels):
    """Compute macro-averaged R@1 at q>=0.0."""
    scores = compute_cosine_similarity(query_embeddings, gallery_embeddings)
    query_labels_np = query_labels.cpu().numpy()

    pred_indices = scores.argmax(dim=1)
    pred_labels = gallery_labels[pred_indices].cpu().numpy()
    correct = (pred_labels == query_labels_np)

    per_ind_recall = []
    for label in np.unique(query_labels_np):
        m = query_labels_np == label
        per_ind_recall.append(correct[m].mean())

    return float(np.mean(per_ind_recall))


def evaluate_with_kmeans_galleries(
    model, train_dataset, val_dataset, individual_to_class, transform, device,
    dataset, feasible_individuals, metadata_cache, criterion, embedding_dim,
    promoted_individuals, excluded_individuals,
):
    """
    Embed full gallery once, then evaluate R@1 and open-set BA at multiple
    k-means gallery sizes (32, 64, 128).
    """
    model.eval()
    use_amp = (device.type == 'cuda') if isinstance(device, torch.device) else (device == 'cuda')
    class_to_name = {v: k for k, v in individual_to_class.items()}

    # --- Embed gallery ---
    train_torch = ArcFaceDataset(train_dataset, transform, individual_to_class)
    train_loader = DataLoader(train_torch, batch_size=EVAL_BATCH_SIZE, shuffle=False,
                              num_workers=EVAL_NUM_WORKERS, pin_memory=use_amp)

    gallery_emb_list, gallery_lab_list = [], []
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp):
        for images, labels, _ in train_loader:
            images = images.to(device, non_blocking=True)
            gallery_emb_list.append(model(images))
            gallery_lab_list.extend(labels.tolist())
    gallery_embeddings = torch.cat(gallery_emb_list, dim=0).float()
    gallery_labels = torch.tensor(gallery_lab_list).to(device)

    # --- Embed query ---
    val_torch = ArcFaceDataset(val_dataset, transform, individual_to_class)
    val_loader = DataLoader(val_torch, batch_size=EVAL_BATCH_SIZE, shuffle=False,
                            num_workers=EVAL_NUM_WORKERS, pin_memory=use_amp)

    query_emb_list, query_lab_list, query_qual_list = [], [], []
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp):
        for images, labels, quality in val_loader:
            images = images.to(device, non_blocking=True)
            query_emb_list.append(model(images))
            query_lab_list.extend(labels.tolist())
            query_qual_list.extend(quality.tolist())
    query_embeddings = torch.cat(query_emb_list, dim=0).float()
    query_labels = torch.tensor(query_lab_list).to(device)
    query_quality = np.array(query_qual_list)

    # --- Val loss ---
    val_loss = compute_validation_loss(query_embeddings, query_labels, criterion, device)

    # --- Embed rare/unknown ---
    rare_indices, rare_quality, rare_labels_str = get_rare_individual_indices(
        metadata_cache, feasible_individuals, quality_threshold=0.0,
        promoted_individuals=promoted_individuals,
        excluded_individuals=excluded_individuals,
    )

    if rare_indices:
        rare_emb, rare_qual_arr, rare_lab_arr = compute_rare_embeddings(
            model, dataset, rare_indices, rare_quality, rare_labels_str,
            transform, device, embedding_dim=embedding_dim,
        )
        rare_emb = rare_emb.to(device)
    else:
        rare_emb = torch.empty(0, embedding_dim).to(device)
        rare_qual_arr = np.array([])
        rare_lab_arr = np.array([])

    # --- Full gallery eval ---
    results = {}

    recall_1_full = compute_recall_at_1(query_embeddings, query_labels,
                                        gallery_embeddings, gallery_labels)
    per_ind_thresh_full, global_thresh_full = compute_arcface_center_thresholds(
        gallery_embeddings, gallery_labels, criterion, class_to_name, device,
    )
    scores_known_full = compute_cosine_similarity(query_embeddings, gallery_embeddings)
    if len(rare_emb) > 0:
        scores_unknown_full = compute_cosine_similarity(rare_emb, gallery_embeddings)
    else:
        scores_unknown_full = torch.empty(0, gallery_embeddings.shape[0]).to(device)

    ba_metrics_full = compute_open_set_metrics_per_individual_threshold(
        known_query_labels=query_labels,
        known_query_quality=query_quality,
        known_scores=scores_known_full,
        unknown_query_labels=rare_lab_arr,
        unknown_query_quality=rare_qual_arr,
        unknown_scores=scores_unknown_full,
        per_individual_thresholds=per_ind_thresh_full,
        quality_thresholds=[0.0],
        gallery_labels=gallery_labels,
        class_to_name=class_to_name,
    )
    ba_q0_full = ba_metrics_full["q>=0.0"]
    results["full"] = {
        "recall_at_1": recall_1_full,
        "balanced_accuracy": ba_q0_full.get("balanced_accuracy", 0.0),
        "known_accept_rate": ba_q0_full.get("known_accept_rate", 0.0),
        "unknown_reject_rate": ba_q0_full.get("unknown_reject_rate", 0.0),
        "n_known_individuals": ba_q0_full.get("n_known_individuals", 0),
        "n_unknown_individuals": ba_q0_full.get("n_unknown_individuals", 0),
        "cosine_threshold": global_thresh_full,
        "gallery_size": int(gallery_embeddings.shape[0]),
    }

    # --- Evaluate at each k-means gallery size ---
    for k_val in EVAL_K_VALUES:
        _, sub_emb, sub_lab = kmeans_select_gallery(
            gallery_embeddings, gallery_labels, k_val, device,
        )

        recall_1 = compute_recall_at_1(query_embeddings, query_labels, sub_emb, sub_lab)

        # Threshold calibration on the subset gallery
        per_ind_thresh, global_thresh = compute_arcface_center_thresholds(
            sub_emb, sub_lab, criterion, class_to_name, device,
        )

        # Open-set BA on the subset gallery
        scores_known = compute_cosine_similarity(query_embeddings, sub_emb)
        if len(rare_emb) > 0:
            scores_unknown = compute_cosine_similarity(rare_emb, sub_emb)
        else:
            scores_unknown = torch.empty(0, sub_emb.shape[0]).to(device)

        ba_metrics = compute_open_set_metrics_per_individual_threshold(
            known_query_labels=query_labels,
            known_query_quality=query_quality,
            known_scores=scores_known,
            unknown_query_labels=rare_lab_arr,
            unknown_query_quality=rare_qual_arr,
            unknown_scores=scores_unknown,
            per_individual_thresholds=per_ind_thresh,
            quality_thresholds=[0.0],
            gallery_labels=sub_lab,
            class_to_name=class_to_name,
        )

        ba_q0 = ba_metrics["q>=0.0"]
        actual_per_class = {}
        sub_lab_np = sub_lab.cpu().numpy()
        for label in np.unique(sub_lab_np):
            name = class_to_name[int(label)]
            actual_per_class[name] = int((sub_lab_np == label).sum())

        results[f"k={k_val}"] = {
            "recall_at_1": recall_1,
            "balanced_accuracy": ba_q0.get("balanced_accuracy", 0.0),
            "known_accept_rate": ba_q0.get("known_accept_rate", 0.0),
            "unknown_reject_rate": ba_q0.get("unknown_reject_rate", 0.0),
            "n_known_individuals": ba_q0.get("n_known_individuals", 0),
            "n_unknown_individuals": ba_q0.get("n_unknown_individuals", 0),
            "cosine_threshold": global_thresh,
            "gallery_per_class": actual_per_class,
        }

    return val_loss, results


def build_unfiltered_split(dataset, metadata_cache, feasible_individuals, feasibility_config, seed):
    """All training images minus val indices, no quality filter."""
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
        all_train_indices.extend(train_only)
        gallery_info[ind_id] = len(train_only)

        print(f"  {ind_id}: {len(train_only)} train, {len(val_idx)} val")

    train_ds = dataset.select(all_train_indices)
    val_ds = dataset.select(all_val_indices)
    print(f"Total: {len(all_train_indices)} train, {len(all_val_indices)} val")

    dataset_info = {
        "gallery_info": gallery_info,
        "query_info": query_info,
        "total_train": len(all_train_indices),
        "total_val": len(all_val_indices),
    }
    return train_ds, val_ds, individual_to_class, dataset_info


def build_filtered_split(dataset, metadata_cache, feasible_individuals, feasibility_config,
                         threshold, seed):
    """Quality-filtered split: all images passing threshold (no gallery_size cap)."""
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
                 epochs, eval_every, seed, device_str, output_path, overwrite):
    """Train + eval loop shared across all three conditions."""
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
    print(f"Eval gallery sizes (k-means): {EVAL_K_VALUES}")
    use_amp = (device == "cuda") or (isinstance(device, torch.device) and device.type == "cuda")
    start_time = time.time()
    epoch_history = []

    # Track best per gallery variant (full + k-means sizes)
    gallery_keys = ["full"] + [f"k={k_val}" for k_val in EVAL_K_VALUES]
    best_metrics = {}
    for gk in gallery_keys:
        best_metrics[gk] = {
            "best_recall": 0.0, "best_recall_epoch": 0,
            "best_ba": 0.0, "best_ba_epoch": 0,
            "best_hm": 0.0, "best_hm_epoch": 0,
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
            val_loss, eval_results = evaluate_with_kmeans_galleries(
                model, train_dataset, val_dataset, individual_to_class, transform, device,
                dataset, feasible_individuals, metadata_cache,
                criterion=arcface_loss, embedding_dim=emb_dim,
                promoted_individuals=promoted_individuals,
                excluded_individuals=excluded_individuals,
            )

            print(f"Epoch {epoch+1:3d}/{epochs}: Loss={train_loss:.4f}, ValLoss={val_loss:.4f}")
            epoch_entry = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }

            for gk in gallery_keys:
                gr = eval_results[gk]
                r1 = gr["recall_at_1"]
                ba = gr["balanced_accuracy"]
                kar = gr["known_accept_rate"]
                urr = gr["unknown_reject_rate"]
                n_k = gr["n_known_individuals"]
                n_u = gr["n_unknown_individuals"]
                thresh = gr["cosine_threshold"]

                bm = best_metrics[gk]
                if r1 > bm["best_recall"]:
                    bm["best_recall"] = r1
                    bm["best_recall_epoch"] = epoch + 1
                if ba > bm["best_ba"]:
                    bm["best_ba"] = ba
                    bm["best_ba_epoch"] = epoch + 1
                hm = 2 * r1 * ba / (r1 + ba) if (r1 + ba) > 0 else 0.0
                if hm > bm["best_hm"]:
                    bm["best_hm"] = hm
                    bm["best_hm_epoch"] = epoch + 1

                print(f"  {gk:>5s}: R@1={r1:.4f}, BA={ba:.4f} (K={kar:.2f}[{n_k}], "
                      f"U={urr:.2f}[{n_u}]), thresh={thresh:.3f}")

                epoch_entry[gk] = gr

            epoch_history.append(epoch_entry)
        else:
            print(f"Epoch {epoch+1:3d}/{epochs}: Loss={train_loss:.4f}")
            epoch_history.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
            })

    training_time = time.time() - start_time

    print(f"\nBest metrics by gallery size:")
    for gk in gallery_keys:
        bm = best_metrics[gk]
        print(f"  {gk:>5s}: R@1={bm['best_recall']:.4f} (ep {bm['best_recall_epoch']}), "
              f"BA={bm['best_ba']:.4f} (ep {bm['best_ba_epoch']}), "
              f"HM={bm['best_hm']:.4f} (ep {bm['best_hm_epoch']})")

    result = {
        "config": {
            "mode": mode_label,
            "seed": seed,
            "epochs": epochs,
            "eval_every": eval_every,
            "eval_k_values": EVAL_K_VALUES,
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
        "best_metrics": {gk: best_metrics[gk] for gk in gallery_keys},
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
    parser = argparse.ArgumentParser(description="Tune epoch count: 3 gallery conditions")
    parser.add_argument("--idx", type=int, required=True,
                        help="0=unfiltered, 1=best R@1 filtered, 2=best BA filtered")
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

    mode_label, hygiene_criterion = JOB_CONFIGS[idx]
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

    # Build train/val split based on mode
    if hygiene_criterion is None:
        # Unfiltered
        print(f"\nMode: unfiltered (all train minus val, no quality filter)")
        train_ds, val_ds, individual_to_class, dataset_info = build_unfiltered_split(
            dataset, metadata_cache, feasible_individuals, feasibility_config, seed,
        )
        dataset_info["mode"] = "unfiltered"
    else:
        # Filtered: load best hygiene config for this criterion
        print(f"\nLoading best hygiene config for criterion={hygiene_criterion}...")
        best_hygiene = load_best_hygiene_config(MODEL_NAME, criterion=hygiene_criterion)
        threshold = best_hygiene["threshold"]
        print(f"Mode: {mode_label} (threshold={threshold}, all eligible images)")

        train_ds, val_ds, individual_to_class, dataset_info = build_filtered_split(
            dataset, metadata_cache, feasible_individuals, feasibility_config,
            threshold, seed,
        )
        dataset_info["mode"] = mode_label
        dataset_info["hygiene_config"] = best_hygiene

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
        epochs=epochs,
        eval_every=eval_every,
        seed=seed,
        device_str=args.device,
        output_path=output_path,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
