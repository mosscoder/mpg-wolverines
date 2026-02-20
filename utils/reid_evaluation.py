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


def compute_loco_gap_threshold(gallery_embeddings, gallery_labels):
    """
    Post-hoc LOCO simulation to find a global gap threshold optimizing BA.

    For each gallery sample i with label L:
      Known gap (label=1): mask out i, find max sim to same-class (Top1_Known),
        max sim to best other class (Top2_Known), gap = Top1 - Top2.
      Unknown gap (label=0): mask out ALL of class L, group remaining by class,
        find max sim per class, sort desc -> Top1_Unknown, Top2_Unknown, gap = Top1 - Top2.

    Use roc_curve to find the gap threshold maximizing balanced accuracy.

    Args:
        gallery_embeddings: Tensor [N, D] of gallery embeddings
        gallery_labels: Tensor [N] of gallery class labels

    Returns:
        global_tau (float), calibration_details (dict)
    """
    from sklearn.metrics import roc_curve

    gallery_emb_norm = F.normalize(gallery_embeddings, p=2, dim=1)
    sim_matrix = torch.mm(gallery_emb_norm, gallery_emb_norm.t()).cpu().numpy()
    np.fill_diagonal(sim_matrix, -np.inf)

    labels_np = gallery_labels.cpu().numpy() if isinstance(gallery_labels, torch.Tensor) else np.array(gallery_labels)
    unique_classes = np.unique(labels_np)
    N = len(labels_np)

    # Precompute class masks
    class_masks = {c: (labels_np == c) for c in unique_classes}

    gaps = []
    gap_labels = []

    for i in range(N):
        own_class = labels_np[i]
        own_mask = class_masks[own_class].copy()

        # --- Known gap (label=1) ---
        # Max sim to same class (excluding self — already -inf on diagonal)
        same_class_sims = sim_matrix[i][own_mask]
        if np.sum(own_mask) < 2:
            # Only one sample in this class, skip known gap
            pass
        else:
            top1_known = np.max(same_class_sims)

            # Max sim to best other class
            other_mask = ~own_mask
            if np.any(other_mask):
                top2_known = np.max(sim_matrix[i][other_mask])
                gaps.append(top1_known - top2_known)
                gap_labels.append(1)

        # --- Unknown gap (label=0) ---
        # Mask out entire own class
        other_mask = ~own_mask
        if np.sum(other_mask) < 2:
            continue

        other_sims = sim_matrix[i][other_mask]
        other_labels = labels_np[other_mask]
        other_classes = np.unique(other_labels)

        if len(other_classes) < 2:
            continue

        # Max sim per other class
        per_class_max = np.array([
            np.max(other_sims[other_labels == c]) for c in other_classes
        ])
        sorted_max = np.sort(per_class_max)[::-1]
        top1_unknown = sorted_max[0]
        top2_unknown = sorted_max[1]
        gaps.append(top1_unknown - top2_unknown)
        gap_labels.append(0)

    gaps = np.array(gaps)
    gap_labels = np.array(gap_labels)

    n_known_gaps = int(np.sum(gap_labels == 1))
    n_unknown_gaps = int(np.sum(gap_labels == 0))

    if n_known_gaps == 0 or n_unknown_gaps == 0:
        # Cannot calibrate — fall back to tau=0
        return 0.0, {
            'n_known_gaps': n_known_gaps,
            'n_unknown_gaps': n_unknown_gaps,
            'best_balanced_accuracy': 0.5,
            'fallback': True,
        }

    # Use roc_curve: positive class = known (label=1), score = gap
    fpr, tpr, thresholds = roc_curve(gap_labels, gaps)
    balanced_acc = (tpr + (1 - fpr)) / 2.0
    best_idx = np.argmax(balanced_acc)
    global_tau = float(thresholds[best_idx])
    best_ba = float(balanced_acc[best_idx])

    return global_tau, {
        'n_known_gaps': n_known_gaps,
        'n_unknown_gaps': n_unknown_gaps,
        'best_balanced_accuracy': best_ba,
    }


def compute_open_set_metrics_gap(known_query_labels, known_query_quality, known_scores,
                                  unknown_query_labels, unknown_query_quality, unknown_scores,
                                  global_tau, quality_thresholds, gallery_labels, class_to_name):
    """
    Compute macro-averaged open-set metrics using gap-based accept/reject.

    For each query:
      - Group gallery similarities by identity, take max per identity
      - Sort descending -> Top1, Top2
      - Accept if (Top1 - Top2) > global_tau, else reject as unknown

    Args:
        known_query_labels: labels for known queries
        known_query_quality: quality scores for known queries
        known_scores: [n_known, n_gallery] cosine similarity matrix
        unknown_query_labels: labels for unknown queries
        unknown_query_quality: quality scores for unknown queries
        unknown_scores: [n_unknown, n_gallery] cosine similarity matrix
        global_tau: gap threshold from LOCO calibration
        quality_thresholds: list of quality thresholds to evaluate
        gallery_labels: [n_gallery] class labels for gallery
        class_to_name: dict {class_label_int: individual_name}

    Returns: dict keyed by quality threshold with BA, known_accept_rate, unknown_reject_rate
    """
    if isinstance(known_query_labels, torch.Tensor):
        known_labels_np = known_query_labels.cpu().numpy()
    else:
        known_labels_np = np.array(known_query_labels)

    gallery_labels_np = gallery_labels.cpu().numpy() if isinstance(gallery_labels, torch.Tensor) else np.array(gallery_labels)
    unique_gallery_classes = np.unique(gallery_labels_np)

    # Precompute per-class gallery indices
    class_indices = {c: np.where(gallery_labels_np == c)[0] for c in unique_gallery_classes}

    def _compute_gap(scores_row):
        """Compute gap between top-1 and top-2 per-identity max similarity."""
        per_class_max = np.array([
            float(scores_row[idx].max()) for idx in class_indices.values()
        ])
        if len(per_class_max) < 2:
            return 0.0
        sorted_max = np.sort(per_class_max)[::-1]
        return sorted_max[0] - sorted_max[1]

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

                accepted = 0
                for idx in ind_indices:
                    gap = _compute_gap(known_scores[idx])
                    if gap > global_tau:
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

                rejected = 0
                for idx in ind_indices:
                    gap = _compute_gap(unknown_scores[idx])
                    if gap <= global_tau:
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


def evaluate_recall_with_openset(model, train_dataset, val_dataset, individual_to_class,
                                  transform, device, dataset, qualified_individuals, metadata_cache,
                                  criterion, embedding_dim=128,
                                  batch_size=EVAL_BATCH_SIZE,
                                  promoted_individuals=None, excluded_individuals=None):
    """
    Evaluate model computing Recall@1 and open-set metrics with raw cosine similarity.

    Rare/unknown individuals are pooled from the train dataset only.

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

    # Rare/Unknown — pool from train dataset
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

    # 2. Compute Cosine Similarity Matrices
    scores_known = compute_cosine_similarity(query_embeddings, gallery_embeddings)

    if len(rare_emb) > 0:
        scores_unknown = compute_cosine_similarity(rare_emb, gallery_embeddings)
    else:
        scores_unknown = torch.empty(0, len(gallery_embeddings)).to(device)

    # 3. Compute Metrics

    # A. Recall@1 (Closed Set) - macro-averaged
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

    # B. Gap-based threshold calibration via LOCO simulation
    class_to_name = {v: k for k, v in individual_to_class.items()}

    global_tau, calibration_details = compute_loco_gap_threshold(
        gallery_embeddings, gallery_labels
    )

    # Compute BA metrics using gap threshold
    balanced_metrics_by_quality = {}

    for q_thresh in QUERY_QUALITY_THRESHOLDS:
        q_key = f"q>={q_thresh}"
        q_metrics = compute_open_set_metrics_gap(
            known_query_labels=query_labels,
            known_query_quality=query_quality,
            known_scores=scores_known,
            unknown_query_labels=rare_labels_arr,
            unknown_query_quality=rare_quality_arr,
            unknown_scores=scores_unknown,
            global_tau=global_tau,
            quality_thresholds=[q_thresh],
            gallery_labels=gallery_labels,
            class_to_name=class_to_name
        )
        balanced_metrics_by_quality[q_key] = q_metrics[q_key]

    open_set_metrics = {
        'threshold_calibration': {
            'method': 'loco_gap',
            'global_tau': global_tau,
            **calibration_details,
        },
        'by_quality': balanced_metrics_by_quality
    }

    return query_quality_metrics, open_set_metrics, val_loss
