#!/usr/bin/env python3
"""
Select feasible individuals for reid experiments using HuggingFace train/test splits.

Individuals qualify for closed-set experiments if they have:
  - >= min_gallery_samples train images with pelage_score > gallery_quality_threshold
  - >= min_query_samples test images with pelage_score > query_quality_threshold
  - >= min_test_events unique events (ymdh) in the test split

Usage:
    python preprocessing/assign_reid_individuals.py
"""

import os
import json
import argparse
import numpy as np
from datasets import load_dataset


def json_serialize_helper(obj):
    """Convert numpy types to JSON serializable Python types"""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')


def create_greedy_temporal_validation_split(train_ids, train_scores, train_ymdh, individuals):
    """
    For each valid individual, walk backward from most recent ymdh in the TRAIN split,
    selecting whole events until all 4 quality bins are covered.

    Quality bins: [0.75,1.0], [0.5,0.75), [0.25,0.5), [0.0,0.25)

    Returns:
        dict of {ind_id: {indices, ymdh_values, count, bins_covered, missing_bins}}
    """
    quality_bins = [
        (0.75, 1.0),
        (0.5, 0.75),
        (0.25, 0.5),
        (0.0, 0.25),
    ]
    bin_labels = ['[0.75,1.0]', '[0.5,0.75)', '[0.25,0.5)', '[0.0,0.25)']

    validation_indices = {}

    for ind_id in individuals:
        mask = train_ids == ind_id
        ind_indices = np.where(mask)[0]
        ind_scores = train_scores[ind_indices]
        ind_ymdh = train_ymdh[ind_indices]

        # Get unique ymdh values sorted descending (most recent first)
        unique_ymdh = np.sort(np.unique(ind_ymdh))[::-1]

        selected_indices = []
        selected_ymdh = []
        bins_covered = set()

        for ymdh_val in unique_ymdh:
            if len(bins_covered) == len(quality_bins):
                break

            # Get all samples from this event
            event_mask = ind_ymdh == ymdh_val
            event_indices = ind_indices[event_mask]
            event_scores = ind_scores[event_mask]

            # Check which bins this event covers
            new_bins = set()
            for b_idx, (lo, hi) in enumerate(quality_bins):
                if b_idx not in bins_covered:
                    if np.any((event_scores >= lo) & (event_scores < hi if b_idx < 3 else event_scores <= hi)):
                        new_bins.add(b_idx)

            # Always include if we haven't covered all bins yet
            selected_indices.extend(event_indices.tolist())
            selected_ymdh.append(int(ymdh_val))
            bins_covered.update(new_bins)

        missing_bins = [bin_labels[i] for i in range(len(quality_bins)) if i not in bins_covered]

        validation_indices[ind_id] = {
            'indices': selected_indices,
            'ymdh_values': selected_ymdh,
            'count': len(selected_indices),
            'bins_covered': [bin_labels[i] for i in sorted(bins_covered)],
            'missing_bins': missing_bins,
        }

        print(f"  {ind_id}: {len(selected_indices)} val samples from {len(selected_ymdh)} events, "
              f"bins covered: {len(bins_covered)}/4"
              + (f" (missing: {', '.join(missing_bins)})" if missing_bins else ""))

    return validation_indices


def assess_training_feasibility(train_ids, train_scores, validation_indices, individuals):
    """
    For each individual, count eligible training samples after excluding validation indices.
    Check at thresholds [0.0, 0.25, 0.5] and gallery sizes [2, 4, 8, 16, 32, 64].

    Returns:
        training_compatibility dict
    """
    thresholds = [0.0, 0.25, 0.5]
    gallery_sizes = [2, 4, 8, 16, 32, 64]

    training_compatibility = {}

    for ind_id in individuals:
        val_idx_set = set(validation_indices[ind_id]['indices'])
        mask = train_ids == ind_id
        ind_indices = np.where(mask)[0]
        ind_scores = train_scores[ind_indices]

        # Exclude validation indices
        train_mask = np.array([idx not in val_idx_set for idx in ind_indices])
        remaining_indices = ind_indices[train_mask]
        remaining_scores = ind_scores[train_mask]

        compat = {'total_remaining': int(len(remaining_indices))}

        for thresh in thresholds:
            eligible = int(np.sum(remaining_scores >= thresh))
            feasible_gallery_sizes = [gs for gs in gallery_sizes if eligible >= gs]
            compat[f'eligible_at_{thresh}'] = eligible
            compat[f'feasible_gallery_sizes_at_{thresh}'] = feasible_gallery_sizes

        training_compatibility[ind_id] = compat

    return training_compatibility


def main():
    parser = argparse.ArgumentParser(description='Select feasible individuals for reid experiments')
    parser.add_argument('--output_dir', type=str, default='preprocessing/results',
                       help='Output directory for results')
    parser.add_argument('--min_gallery_samples', type=int, default=64,
                       help='Minimum train images above quality threshold (default: 64)')
    parser.add_argument('--gallery_quality_threshold', type=float, default=0.5,
                       help='Quality threshold for gallery selection (default: 0.5)')
    parser.add_argument('--min_query_samples', type=int, default=1,
                       help='Minimum test images above quality threshold (default: 1)')
    parser.add_argument('--query_quality_threshold', type=float, default=0.5,
                       help='Quality threshold for query selection (default: 0.5)')
    parser.add_argument('--min_test_events', type=int, default=2,
                       help='Minimum unique events (ymdh) in test split (default: 2)')
    args = parser.parse_args()

    # Manual exclusions
    EXCLUDED_ENTIRELY = []
    PROMOTED_TO_RARE = []

    # Load both splits
    print("Loading wolverines dataset (reidentification config)...")
    train_dataset = load_dataset("kdoherty/wolverines", "reidentification", split="train")
    test_dataset = load_dataset("kdoherty/wolverines", "reidentification", split="test")
    print(f"Loaded {len(train_dataset)} train samples, {len(test_dataset)} test samples")

    # Build per-individual stats using Arrow columnar access
    print("\nAnalyzing splits...")
    train_ids = np.array(train_dataset['id'], dtype=object)
    train_scores = np.array(train_dataset['pelage_score'], dtype=np.float32)
    train_ymdh = np.array(train_dataset['ymdh'], dtype=np.int64)
    test_ids = np.array(test_dataset['id'], dtype=object)
    test_scores = np.array(test_dataset['pelage_score'], dtype=np.float32)
    test_ymdh = np.array(test_dataset['ymdh'], dtype=np.int64)

    train_stats = {}
    for ind_id in np.unique(train_ids):
        mask = train_ids == ind_id
        scores = train_scores[mask]
        ymdh_vals = train_ymdh[mask]
        train_stats[ind_id] = {
            'total': int(mask.sum()),
            'above_025': int((scores > 0.25).sum()),
            'above_threshold': int((scores > args.gallery_quality_threshold).sum()),
            'pelage_scores': scores,
            'events': int(len(np.unique(ymdh_vals))),
        }

    test_stats = {}
    for ind_id in np.unique(test_ids):
        mask = test_ids == ind_id
        scores = test_scores[mask]
        ymdh_vals = test_ymdh[mask]
        test_stats[ind_id] = {
            'total': int(mask.sum()),
            'above_025': int((scores > 0.25).sum()),
            'above_threshold': int((scores > args.query_quality_threshold).sum()),
            'events': int(len(np.unique(ymdh_vals))),
        }

    # Determine valid individuals
    all_individuals = set(train_stats.keys()) | set(test_stats.keys())
    excluded_set = set(EXCLUDED_ENTIRELY)

    valid_individuals = []
    excluded_individuals = []

    for ind_id in sorted(all_individuals):
        if ind_id in excluded_set:
            excluded_individuals.append((ind_id, "Manually excluded"))
            continue

        train_above = train_stats.get(ind_id, {}).get('above_threshold', 0)
        test_above = test_stats.get(ind_id, {}).get('above_threshold', 0)

        test_events = test_stats.get(ind_id, {}).get('events', 0)

        if (train_above >= args.min_gallery_samples
                and test_above >= args.min_query_samples
                and test_events >= args.min_test_events):
            valid_individuals.append(ind_id)
        else:
            reasons = []
            if train_above < args.min_gallery_samples:
                reasons.append(f"train above {args.gallery_quality_threshold}: {train_above} < {args.min_gallery_samples}")
            if test_above < args.min_query_samples:
                reasons.append(f"test above {args.query_quality_threshold}: {test_above} < {args.min_query_samples}")
            if test_events < args.min_test_events:
                reasons.append(f"test events: {test_events} < {args.min_test_events}")
            excluded_individuals.append((ind_id, "; ".join(reasons)))

    # Sort by mean pelage score (descending)
    valid_individuals.sort(
        key=lambda x: np.mean(train_stats[x]['pelage_scores']),
        reverse=True
    )

    # Create greedy temporal validation split from train data
    print("\nCreating greedy temporal validation split...")
    validation_indices = create_greedy_temporal_validation_split(
        train_ids, train_scores, train_ymdh, valid_individuals
    )

    # Assess training feasibility after excluding validation indices
    print("\nAssessing training feasibility...")
    training_compatibility = assess_training_feasibility(
        train_ids, train_scores, validation_indices, valid_individuals
    )

    # Build individual_pelage_scores and individual_stats
    individual_pelage_scores = {}
    individual_stats = {}

    for ind_id in valid_individuals:
        individual_pelage_scores[ind_id] = float(np.mean(train_stats[ind_id]['pelage_scores']))

    for ind_id in sorted(all_individuals):
        t = train_stats.get(ind_id, {})
        s = test_stats.get(ind_id, {})
        individual_stats[ind_id] = {
            'train': {
                'images': t.get('total', 0),
                'images_above_0.25': t.get('above_025', 0),
                f'images_above_{args.gallery_quality_threshold}': t.get('above_threshold', 0),
                'events': t.get('events', 0),
            },
            'test': {
                'images': s.get('total', 0),
                'images_above_0.25': s.get('above_025', 0),
                f'images_above_{args.query_quality_threshold}': s.get('above_threshold', 0),
                'events': s.get('events', 0),
            },
        }

    # Report
    print(f"\nFound {len(valid_individuals)} valid individuals "
          f"(train >= {args.min_gallery_samples} at > {args.gallery_quality_threshold}, "
          f"test >= {args.min_query_samples} at > {args.query_quality_threshold}):")
    for ind_id in valid_individuals:
        t = individual_stats[ind_id]
        tr, te = t['train'], t['test']
        print(f"  {ind_id}: train={tr[f'images_above_{args.gallery_quality_threshold}']}/{tr['images']} imgs, {tr['events']} events | "
              f"test={te[f'images_above_{args.query_quality_threshold}']}/{te['images']} imgs, {te['events']} events | "
              f"mean_pelage={individual_pelage_scores[ind_id]:.3f}")

    if excluded_individuals:
        print(f"\nExcluded {len(excluded_individuals)} individuals:")
        for ind_id, reason in excluded_individuals:
            t = individual_stats.get(ind_id, {})
            tr, te = t.get('train', {}), t.get('test', {})
            print(f"  {ind_id}: train={tr.get(f'images_above_{args.gallery_quality_threshold}', 0)}/{tr.get('images', 0)} imgs, {tr.get('events', 0)} events | "
                  f"test={te.get(f'images_above_{args.query_quality_threshold}', 0)}/{te.get('images', 0)} imgs, {te.get('events', 0)} events -- {reason}")

    # Save config
    config = {
        'valid_individuals': valid_individuals,
        'promoted_to_rare': PROMOTED_TO_RARE,
        'excluded_entirely': EXCLUDED_ENTIRELY,
        'selection_criteria': {
            'min_gallery_samples': args.min_gallery_samples,
            'gallery_quality_threshold': args.gallery_quality_threshold,
            'min_query_samples': args.min_query_samples,
            'query_quality_threshold': args.query_quality_threshold,
            'min_test_events': args.min_test_events,
        },
        'individual_pelage_scores': individual_pelage_scores,
        'individual_stats': individual_stats,
        'validation_indices': validation_indices,
        'training_compatibility': training_compatibility,
        'validation_strategy': 'greedy_temporal_from_train',
    }

    output_path = os.path.join(args.output_dir, 'feasible_individuals.json')
    os.makedirs(args.output_dir, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2, default=json_serialize_helper)

    print(f"\nConfiguration saved to: {output_path}")
    print("Preprocessing complete!")


if __name__ == "__main__":
    main()
