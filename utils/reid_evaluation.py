"""
Evaluation functions for reid experiments: embeddings, similarity, open-set metrics.
"""

import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from torch.utils.data import DataLoader

from utils.reid_config import EVAL_BATCH_SIZE, EVAL_NUM_WORKERS, QUERY_QUALITY_THRESHOLDS
from utils.reid_data import (
    ArcFaceDataset, RareIndividualsDataset, get_rare_individual_indices
)


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_recall_simple(model, train_dataset, test_dataset, individual_to_class,
                           transform, device, batch_size=EVAL_BATCH_SIZE):
    """
    Compute macro-averaged Recall@1 using cosine similarity.
    No quality filtering, no open-set evaluation.

    Returns: recall_at_1 value
    """
    model.eval()
    use_amp = (device.type == 'cuda') if isinstance(device, torch.device) else (device == 'cuda')

    train_torch = ArcFaceDataset(train_dataset, transform, individual_to_class)
    test_torch = ArcFaceDataset(test_dataset, transform, individual_to_class)

    train_loader = DataLoader(train_torch, batch_size=batch_size, shuffle=False,
                              num_workers=EVAL_NUM_WORKERS, pin_memory=use_amp)
    test_loader = DataLoader(test_torch, batch_size=batch_size, shuffle=False,
                             num_workers=EVAL_NUM_WORKERS, pin_memory=use_amp)

    # Gallery embeddings (train)
    gallery_embeddings = []
    gallery_labels = []
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp):
        for batch in train_loader:
            images = batch[0].to(device, non_blocking=True)
            labels = batch[1]
            emb = model(images)
            gallery_embeddings.append(emb)
            gallery_labels.extend(labels.tolist())
    gallery_embeddings = torch.cat(gallery_embeddings, dim=0).float()
    gallery_labels = torch.tensor(gallery_labels).to(device)

    # Query embeddings (test)
    query_embeddings = []
    query_labels = []
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp):
        for batch in test_loader:
            images = batch[0].to(device, non_blocking=True)
            labels = batch[1]
            emb = model(images)
            query_embeddings.append(emb)
            query_labels.extend(labels.tolist())
    query_embeddings = torch.cat(query_embeddings, dim=0).float()
    query_labels_np = np.array(query_labels)

    # Compute cosine similarity
    query_emb_norm = F.normalize(query_embeddings, p=2, dim=1)
    gallery_emb_norm = F.normalize(gallery_embeddings, p=2, dim=1)
    similarity = torch.mm(query_emb_norm, gallery_emb_norm.t())

    # Macro-averaged Recall@1
    individual_correct = defaultdict(int)
    individual_total = defaultdict(int)

    for i in range(len(query_labels_np)):
        true_label = query_labels_np[i]
        pred_idx = similarity[i].argmax().item()
        pred_label = gallery_labels[pred_idx].item()

        individual_total[true_label] += 1
        if pred_label == true_label:
            individual_correct[true_label] += 1

    per_ind_recall = [
        individual_correct.get(label, 0) / individual_total[label]
        for label in individual_total
    ]

    recall_at_1 = np.mean(per_ind_recall) if per_ind_recall else 0.0
    return recall_at_1


# ============================================================================
# Hygiene sweep: open-set evaluation
# ============================================================================

def compute_rare_embeddings(model, dataset, rare_indices, rare_quality, rare_labels,
                            transform, device, embedding_dim=128, batch_size=EVAL_BATCH_SIZE):
    """Compute embeddings for rare/unknown individuals."""
    model.eval()
    use_amp = (device.type == 'cuda') if isinstance(device, torch.device) else (device == 'cuda')

    rare_dataset = RareIndividualsDataset(dataset, rare_indices, rare_quality, transform)
    rare_loader = DataLoader(rare_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=EVAL_NUM_WORKERS, pin_memory=use_amp)

    embeddings = []
    qualities = []

    with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp):
        for images, quality in rare_loader:
            images = images.to(device, non_blocking=True)
            emb = model(images)
            embeddings.append(emb.cpu())
            qualities.extend(quality.tolist())

    if embeddings:
        embeddings = torch.cat(embeddings, dim=0).float()
    else:
        embeddings = torch.empty(0, embedding_dim)

    return embeddings, np.array(qualities), np.array(rare_labels)


def compute_validation_loss(val_embeddings, val_labels, criterion, device):
    """Compute ArcFace loss on validation set."""
    criterion.eval()
    val_embeddings = val_embeddings.to(device)
    val_labels = val_labels.to(device)

    with torch.no_grad():
        loss = criterion(val_embeddings, val_labels)

    return loss.item()


def compute_cosine_similarity(query_emb, gallery_emb):
    """Compute L2-normalized cosine similarity matrix."""
    query_emb = F.normalize(query_emb, p=2, dim=1)
    gallery_emb = F.normalize(gallery_emb, p=2, dim=1)
    return torch.mm(query_emb, gallery_emb.t())


def compute_open_set_metrics_per_individual_threshold(
        known_query_labels, known_query_quality, known_scores,
        unknown_query_labels, unknown_query_quality, unknown_scores,
        per_individual_thresholds, quality_thresholds, gallery_labels,
        class_to_name):
    """
    Compute macro-averaged open-set metrics using per-individual cosine thresholds.

    Each query's accept/reject decision uses the threshold of the gallery individual
    it matched best with, rather than a single global threshold.

    Args:
        per_individual_thresholds: dict {individual_name: float} of cosine thresholds
        class_to_name: dict {class_label_int: individual_name}
    """
    if isinstance(known_query_labels, torch.Tensor):
        known_labels_np = known_query_labels.cpu().numpy()
    else:
        known_labels_np = np.array(known_query_labels)

    gallery_labels_np = gallery_labels.cpu().numpy() if isinstance(gallery_labels, torch.Tensor) else np.array(gallery_labels)

    results = {}

    for q_thresh in quality_thresholds:
        # Known accept rate (macro-averaged)
        k_mask = known_query_quality >= q_thresh
        per_ind_accept = []
        n_known_individuals = 0

        if k_mask.sum() > 0:
            known_indices = np.where(k_mask)[0]
            unique_known_labels = np.unique(known_labels_np[known_indices])

            for ind_label in unique_known_labels:
                ind_mask = (known_labels_np == ind_label) & k_mask
                ind_indices = np.where(ind_mask)[0]

                if len(ind_indices) == 0:
                    continue

                # For each query, find best gallery match and that match's individual
                max_scores, max_gallery_idx = known_scores[ind_indices].max(dim=1)
                matched_gallery_labels = gallery_labels_np[max_gallery_idx.cpu().numpy()]

                accepted = 0
                for score, matched_label in zip(max_scores, matched_gallery_labels):
                    matched_name = class_to_name[int(matched_label)]
                    thresh = per_individual_thresholds.get(matched_name, 0.5)
                    if score.item() >= thresh:
                        accepted += 1

                per_ind_accept.append(accepted / len(ind_indices))

            n_known_individuals = len(per_ind_accept)

        known_accept_rate = np.mean(per_ind_accept) if per_ind_accept else 0.0

        # Unknown reject rate (macro-averaged)
        u_mask = unknown_query_quality >= q_thresh
        per_ind_reject = []
        n_unknown_individuals = 0

        if len(unknown_query_labels) > 0 and u_mask.sum() > 0:
            unknown_indices = np.where(u_mask)[0]
            unique_unknown_labels = np.unique(unknown_query_labels[unknown_indices])

            for ind_id in unique_unknown_labels:
                ind_mask = (unknown_query_labels == ind_id) & u_mask
                ind_indices = np.where(ind_mask)[0]

                if len(ind_indices) == 0:
                    continue

                # For each unknown query, find best gallery match and use that individual's threshold
                max_scores, max_gallery_idx = unknown_scores[ind_indices].max(dim=1)
                matched_gallery_labels = gallery_labels_np[max_gallery_idx.cpu().numpy()]

                rejected = 0
                for score, matched_label in zip(max_scores, matched_gallery_labels):
                    matched_name = class_to_name[int(matched_label)]
                    thresh = per_individual_thresholds.get(matched_name, 0.5)
                    if score.item() < thresh:
                        rejected += 1

                per_ind_reject.append(rejected / len(ind_indices))

            n_unknown_individuals = len(per_ind_reject)

        unknown_reject_rate = np.mean(per_ind_reject) if per_ind_reject else 1.0

        balanced_acc = (known_accept_rate + unknown_reject_rate) / 2.0

        results[f"q>={q_thresh}"] = {
            'balanced_accuracy': balanced_acc,
            'known_accept_rate': known_accept_rate,
            'unknown_reject_rate': unknown_reject_rate,
            'n_known_individuals': n_known_individuals,
            'n_unknown_individuals': n_unknown_individuals
        }

    return results


def compute_arcface_center_thresholds(gallery_embeddings, gallery_labels, criterion,
                                      class_to_name, device, percentile=10):
    """
    Compute per-individual acceptance thresholds from ArcFace centers.

    For each individual, computes cosine similarity between their gallery
    embeddings and their learned ArcFace center. The p-th percentile
    similarity sets the acceptance threshold.

    Args:
        gallery_embeddings: Tensor of gallery embeddings
        gallery_labels: Tensor of gallery class labels
        criterion: ArcFaceLoss module (has .weight attribute for centers)
        class_to_name: dict {class_label_int: individual_name}
        device: torch device
        percentile: Percentile for threshold (default 2 = p2)

    Returns:
        (per_individual_thresholds dict, global_threshold float)
    """
    with torch.no_grad():
        centers = F.normalize(criterion.weight, p=2, dim=1)

    gallery_emb_norm = F.normalize(gallery_embeddings, p=2, dim=1)
    gallery_center_sims = torch.mm(gallery_emb_norm, centers.t())

    per_individual_thresholds = {}
    for ind_label in gallery_labels.unique():
        ind_mask = (gallery_labels == ind_label)
        ind_name = class_to_name[ind_label.item()]
        sims_to_own_center = gallery_center_sims[ind_mask, ind_label.item()]
        per_individual_thresholds[ind_name] = float(
            np.percentile(sims_to_own_center.cpu().numpy(), percentile)
        )

    valid_thresholds = list(per_individual_thresholds.values())
    global_threshold = float(np.mean(valid_thresholds)) if valid_thresholds else 0.5

    return per_individual_thresholds, global_threshold


def evaluate_recall_with_openset(model, train_dataset, val_dataset, individual_to_class,
                                  transform, device, dataset, qualified_individuals, metadata_cache,
                                  criterion, embedding_dim=128,
                                  batch_size=EVAL_BATCH_SIZE,
                                  promoted_individuals=None, excluded_individuals=None,
                                  skip_open_set=False):
    """
    Evaluate model computing Recall@1 and optionally open-set metrics.

    When skip_open_set=True, only computes R@1 and val_loss (no unknown sourcing).

    Returns:
        query_quality_metrics, open_set_metrics, val_loss
    """
    model.eval()
    use_amp = (device.type == 'cuda') if isinstance(device, torch.device) else (device == 'cuda')

    # 1. Extract Embeddings
    train_torch = ArcFaceDataset(train_dataset, transform, individual_to_class)
    val_torch = ArcFaceDataset(val_dataset, transform, individual_to_class)

    train_loader = DataLoader(train_torch, batch_size=batch_size, shuffle=False,
                              num_workers=EVAL_NUM_WORKERS, pin_memory=use_amp)
    val_loader = DataLoader(val_torch, batch_size=batch_size, shuffle=False,
                            num_workers=EVAL_NUM_WORKERS, pin_memory=use_amp)

    # Gallery
    gallery_embeddings = []
    gallery_labels = []
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp):
        for images, labels, _ in train_loader:
            images = images.to(device, non_blocking=True)
            gallery_embeddings.append(model(images))
            gallery_labels.extend(labels.tolist())
    gallery_embeddings = torch.cat(gallery_embeddings, dim=0).float()
    gallery_labels = torch.tensor(gallery_labels).to(device)

    # Query (Known)
    query_embeddings = []
    query_labels = []
    query_quality = []
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp):
        for images, labels, quality in val_loader:
            images = images.to(device, non_blocking=True)
            query_embeddings.append(model(images))
            query_labels.extend(labels.tolist())
            query_quality.extend(quality.tolist())
    query_embeddings = torch.cat(query_embeddings, dim=0).float()
    query_labels = torch.tensor(query_labels).to(device)
    query_quality = np.array(query_quality)

    # Validation loss
    val_loss = compute_validation_loss(query_embeddings, query_labels, criterion, device)

    # 2. Compute Cosine Similarity (query vs gallery)
    scores_known = compute_cosine_similarity(query_embeddings, gallery_embeddings)

    # 3. Recall@1 (Closed Set) - macro-averaged
    query_quality_metrics = {}
    query_labels_np = query_labels.cpu().numpy()

    for thresh in QUERY_QUALITY_THRESHOLDS:
        mask = query_quality >= thresh
        count = int(mask.sum())
        if count == 0:
            query_quality_metrics[f"q>={thresh}"] = {"recall_at_1": 0.0, "count": 0}
            continue

        mask_indices = np.where(mask)[0]
        individual_correct = {}
        individual_total = {}

        pred_indices = scores_known[mask_indices].argmax(dim=1)
        pred_labels = gallery_labels[pred_indices].cpu().numpy()
        true_labels = query_labels_np[mask_indices]
        correct = (pred_labels == true_labels)

        for label in np.unique(true_labels):
            label_mask = (true_labels == label)
            individual_total[label] = int(label_mask.sum())
            individual_correct[label] = int(correct[label_mask].sum())

        per_ind_recall = [
            individual_correct.get(label, 0) / individual_total[label]
            for label in individual_total
        ]
        recall = np.mean(per_ind_recall) if per_ind_recall else 0.0

        query_quality_metrics[f"q>={thresh}"] = {"recall_at_1": recall, "count": count}

    if skip_open_set:
        return query_quality_metrics, {}, val_loss

    # 4. Open-set: source rare/unknown individuals
    rare_indices, rare_quality, rare_labels_str = get_rare_individual_indices(
        metadata_cache, qualified_individuals, quality_threshold=0.0,
        promoted_individuals=promoted_individuals,
        excluded_individuals=excluded_individuals
    )

    all_rare_emb = []
    all_rare_quality = []
    all_rare_labels = []

    if rare_indices:
        emb, qual, lab = compute_rare_embeddings(
            model, dataset, rare_indices, rare_quality, rare_labels_str,
            transform, device, embedding_dim=embedding_dim
        )
        all_rare_emb.append(emb)
        all_rare_quality.append(qual)
        all_rare_labels.append(lab)

    if all_rare_emb:
        rare_emb = torch.cat(all_rare_emb, dim=0).to(device)
        rare_quality_arr = np.concatenate(all_rare_quality)
        rare_labels_arr = np.concatenate(all_rare_labels)
    else:
        rare_emb = torch.empty(0, embedding_dim).to(device)
        rare_quality_arr = np.array([])
        rare_labels_arr = np.array([])

    scores_unknown = compute_cosine_similarity(rare_emb, gallery_embeddings) if len(rare_emb) > 0 \
        else torch.empty(0, len(gallery_embeddings)).to(device)

    # B. Per-individual threshold calibration via ArcFace centers
    class_to_name = {v: k for k, v in individual_to_class.items()}

    per_individual_thresholds, global_threshold = compute_arcface_center_thresholds(
        gallery_embeddings, gallery_labels, criterion, class_to_name, device
    )

    # Compute BA metrics using per-individual thresholds
    balanced_metrics_by_quality = {}

    for q_thresh in QUERY_QUALITY_THRESHOLDS:
        q_key = f"q>={q_thresh}"
        q_metrics = compute_open_set_metrics_per_individual_threshold(
            known_query_labels=query_labels,
            known_query_quality=query_quality,
            known_scores=scores_known,
            unknown_query_labels=rare_labels_arr,
            unknown_query_quality=rare_quality_arr,
            unknown_scores=scores_unknown,
            per_individual_thresholds=per_individual_thresholds,
            quality_thresholds=[q_thresh],
            gallery_labels=gallery_labels,
            class_to_name=class_to_name
        )
        balanced_metrics_by_quality[q_key] = q_metrics[q_key]
        balanced_metrics_by_quality[q_key]['cosine_threshold'] = global_threshold

    open_set_metrics = {
        'threshold_calibration': {
            'method': 'arcface_center_p10',
            'global_threshold': global_threshold,
            'per_individual': per_individual_thresholds,
        },
        'by_quality': balanced_metrics_by_quality
    }

    return query_quality_metrics, open_set_metrics, val_loss
