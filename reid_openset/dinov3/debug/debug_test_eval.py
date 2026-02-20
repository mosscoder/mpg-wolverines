"""
Debug test eval: unfiltered DINOv3, all training samples, open-set BA.

No quality filtering, no gallery_size cap, no event-matched subsampling.
Trains on ALL training images for qualified individuals, evaluates on test split.
Prints epoch-by-epoch open-set metrics (BA, known accept, unknown reject).

Usage:
    cd /home/kdoherty/wolverines
    python -u reid_openset/dinov3/debug/debug_test_eval.py \
        --device gpu --epochs 100 --seed 0
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
from datetime import datetime
from torch.utils.data import DataLoader

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HOME"] = "/data/hf_cache"

from utils.dataset import set_all_seeds
from utils.reid_config import (
    MODEL_CONFIGS, ARCFACE_MARGIN, ARCFACE_SCALE, BATCH_K, MIN_P,
    QUERY_QUALITY_THRESHOLDS, EVAL_BATCH_SIZE, EVAL_NUM_WORKERS,
    create_arcface_model, create_transform_for_model,
)
from utils.reid_data import (
    ArcFaceDataset,
    load_feasibility_config, load_reidentification_dataset,
    load_reidentification_test_dataset, build_metadata_cache,
    get_rare_individual_indices,
)
from utils.reid_evaluation import (
    compute_rare_embeddings, compute_cosine_similarity,
    compute_arcface_center_thresholds,
    compute_open_set_metrics_per_individual_threshold,
)
from utils.reid import (
    train_epoch_arcface, count_trainable_parameters, load_best_hyperparams,
)


MODEL_NAME = "dinov3"


def main():
    parser = argparse.ArgumentParser(description="Debug test eval: unfiltered DINOv3")
    parser.add_argument("--device", type=str, choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    seed = args.seed
    epochs = args.epochs
    set_all_seeds(seed)

    output_dir = os.path.join(MODEL_CONFIGS[MODEL_NAME]["experiment_dir"], "debug")
    output_path = os.path.join(output_dir, f"debug_test_eval_seed={seed}.json")

    if os.path.exists(output_path) and not args.overwrite:
        print(f"Result exists: {output_path}. Use --overwrite to replace.")
        return

    print("=" * 70)
    print(f"Debug Test Eval: DINOv3 unfiltered, all training samples, seed={seed}")
    print("=" * 70)

    # Load best hyperparams from opt sweeps
    best_lr, best_size, best_embedding_dim = load_best_hyperparams(MODEL_NAME)

    # Load datasets
    print("\nLoading datasets...")
    train_dataset = load_reidentification_dataset()
    train_metadata_cache = build_metadata_cache(train_dataset)
    test_dataset = load_reidentification_test_dataset()
    test_metadata_cache = build_metadata_cache(test_dataset)

    feasibility_config = load_feasibility_config()
    if not feasibility_config:
        print("Failed to load feasibility config")
        return

    feasible_individuals = feasibility_config.get("qualified_individuals", [])
    promoted_individuals = feasibility_config.get("promoted_to_rare", [])
    excluded_individuals = feasibility_config.get("excluded_entirely", [])

    print(f"Using {len(feasible_individuals)} individuals: {', '.join(feasible_individuals)}")

    # Gallery: ALL training images for qualified individuals (no filtering)
    id_to_indices = train_metadata_cache["id_to_indices"]
    individual_to_class = {ind: i for i, ind in enumerate(sorted(feasible_individuals))}
    class_to_name = {v: k for k, v in individual_to_class.items()}

    all_gallery_indices = []
    gallery_info = {}
    for ind_id in feasible_individuals:
        ind_indices = list(id_to_indices.get(ind_id, []))
        all_gallery_indices.extend(ind_indices)
        gallery_info[ind_id] = len(ind_indices)
        print(f"  {ind_id}: {len(ind_indices)} gallery (all, unfiltered)")

    gallery_dataset = train_dataset.select(all_gallery_indices)

    # Query (known): ALL test images for qualified individuals
    test_id_to_indices = test_metadata_cache["id_to_indices"]
    all_query_indices = []
    query_info = {}
    for ind_id in feasible_individuals:
        test_indices = test_id_to_indices.get(ind_id, [])
        all_query_indices.extend(test_indices)
        query_info[ind_id] = len(test_indices)

    query_dataset = test_dataset.select(all_query_indices)
    print(f"\nTotal: {len(all_gallery_indices)} gallery, {len(all_query_indices)} query (known)")

    # Rare/Unknown from TEST split
    rare_indices, rare_quality_arr, rare_labels_str = get_rare_individual_indices(
        test_metadata_cache, feasible_individuals, quality_threshold=0.0,
        promoted_individuals=promoted_individuals,
        excluded_individuals=excluded_individuals,
    )
    print(f"Unknown pool: {len(rare_indices) if rare_indices else 0} images, "
          f"{len(set(rare_labels_str)) if rare_labels_str else 0} individuals")

    # Create model
    device = "cuda" if args.device == "gpu" and torch.cuda.is_available() else "cpu"
    model, emb_dim = create_arcface_model(MODEL_NAME, embedding_dim=best_embedding_dim,
                                           image_size=best_size, device=device)
    print(f"Using device: {device}")

    # Transforms and training dataloader
    transform = create_transform_for_model(MODEL_NAME, size=best_size)
    train_torch = ArcFaceDataset(gallery_dataset, transform, individual_to_class)

    from utils.arcface import ArcFaceLoss, PKBatchSampler

    min_gallery = min(gallery_info.values())
    effective_k = min(BATCH_K, min_gallery)

    pk_sampler = PKBatchSampler(
        labels=train_torch.get_labels(),
        p=min(MIN_P, len(feasible_individuals)),
        k=effective_k,
        drop_last=True,
    )

    def collate_fn(batch):
        images = torch.stack([item[0] for item in batch])
        labels = torch.tensor([item[1] for item in batch])
        quality = torch.tensor([item[2] for item in batch])
        return images, labels, quality

    train_loader = DataLoader(train_torch, batch_sampler=pk_sampler,
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

    # Training loop with per-epoch eval
    print(f"\nTraining for {epochs} epochs with per-epoch open-set eval...")
    use_amp = (device == "cuda") or (isinstance(device, torch.device) and device.type == "cuda")
    start_time = time.time()
    epoch_history = []

    for epoch in range(epochs):
        train_loss = train_epoch_arcface(model, train_loader, optimizer, arcface_loss, device)

        # --- Eval ---
        model.eval()

        # Gallery embeddings
        gallery_loader = DataLoader(
            ArcFaceDataset(gallery_dataset, transform, individual_to_class),
            batch_size=EVAL_BATCH_SIZE, shuffle=False,
            num_workers=EVAL_NUM_WORKERS, pin_memory=use_amp,
        )
        g_emb, g_lab = [], []
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
            for images, labels, _ in gallery_loader:
                images = images.to(device, non_blocking=True)
                g_emb.append(model(images))
                g_lab.extend(labels.tolist())
        g_emb = torch.cat(g_emb, dim=0).float()
        g_lab = torch.tensor(g_lab).to(device)

        # Query embeddings (known)
        query_loader = DataLoader(
            ArcFaceDataset(query_dataset, transform, individual_to_class),
            batch_size=EVAL_BATCH_SIZE, shuffle=False,
            num_workers=EVAL_NUM_WORKERS, pin_memory=use_amp,
        )
        q_emb, q_lab, q_qual = [], [], []
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
            for images, labels, quality in query_loader:
                images = images.to(device, non_blocking=True)
                q_emb.append(model(images))
                q_lab.extend(labels.tolist())
                q_qual.extend(quality.tolist())
        q_emb = torch.cat(q_emb, dim=0).float()
        q_lab_t = torch.tensor(q_lab).to(device)
        q_qual_np = np.array(q_qual)
        q_lab_np = np.array(q_lab)

        # Cosine similarity (known queries)
        scores_known = compute_cosine_similarity(q_emb, g_emb)

        # Recall@1 (closed-set, q>=0.0)
        pred_idx = scores_known.argmax(dim=1)
        pred_labels = g_lab[pred_idx].cpu().numpy()
        correct = (pred_labels == q_lab_np)
        per_ind_recall = []
        for label in np.unique(q_lab_np):
            m = q_lab_np == label
            per_ind_recall.append(correct[m].mean())
        recall_at_1 = float(np.mean(per_ind_recall))

        # Unknown embeddings
        if rare_indices:
            rare_emb, rare_qual_vals, rare_lab_arr = compute_rare_embeddings(
                model, test_dataset, rare_indices, rare_quality_arr, rare_labels_str,
                transform, device, embedding_dim=emb_dim,
            )
            rare_emb = rare_emb.to(device)
            scores_unknown = compute_cosine_similarity(rare_emb, g_emb)
        else:
            rare_qual_vals = np.array([])
            rare_lab_arr = np.array([])
            scores_unknown = torch.empty(0, g_emb.shape[0]).to(device)

        # ArcFace center thresholds (p2)
        per_ind_thresh, global_thresh = compute_arcface_center_thresholds(
            g_emb, g_lab, arcface_loss, class_to_name, device,
        )

        # Open-set BA
        ba_metrics = compute_open_set_metrics_per_individual_threshold(
            known_query_labels=q_lab_t,
            known_query_quality=q_qual_np,
            known_scores=scores_known,
            unknown_query_labels=rare_lab_arr,
            unknown_query_quality=rare_qual_vals,
            unknown_scores=scores_unknown,
            per_individual_thresholds=per_ind_thresh,
            quality_thresholds=[0.0],
            gallery_labels=g_lab,
            class_to_name=class_to_name,
        )

        ba_q0 = ba_metrics["q>=0.0"]
        ba = ba_q0["balanced_accuracy"]
        kar = ba_q0["known_accept_rate"]
        urr = ba_q0["unknown_reject_rate"]

        print(f"Epoch {epoch+1:3d}/{epochs}: Loss={train_loss:.4f}, "
              f"R@1={recall_at_1:.4f}, BA={ba:.4f} (K={kar:.2f}, U={urr:.2f}), "
              f"thresh={global_thresh:.3f}")

        epoch_history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "recall_at_1": recall_at_1,
            "balanced_accuracy": ba,
            "known_accept_rate": kar,
            "unknown_reject_rate": urr,
            "global_threshold": global_thresh,
            "per_individual_thresholds": per_ind_thresh,
        })

    training_time = time.time() - start_time

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
            "gallery_mode": "all_unfiltered",
        },
        "dataset": {
            "individuals": feasible_individuals,
            "gallery_info": gallery_info,
            "query_info": query_info,
            "total_gallery": len(all_gallery_indices),
            "total_query_known": len(all_query_indices),
            "total_query_unknown": len(rare_indices) if rare_indices else 0,
            "n_unknown_individuals": len(set(rare_labels_str)) if rare_labels_str else 0,
        },
        "epoch_history": epoch_history,
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
