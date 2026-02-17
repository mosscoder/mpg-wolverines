#!/usr/bin/env python3
"""
Plot quality score distributions as a faceted grid: rows = individual, cols = split (train/val/test).

All individuals are shown. Qualified individuals are listed first (by mean pelage score),
followed by non-qualified individuals alphabetically.
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default='preprocessing/results/feasible_individuals.json')
    parser.add_argument('--output_dir', type=str,
                        default='preprocessing/results')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    qualified = config['qualified_individuals']
    qualified_set = set(qualified)
    validation_indices = config.get('validation_indices', {})
    all_individuals_in_stats = list(config.get('individual_stats', {}).keys())

    # Row order: qualified first (config order), then non-qualified alphabetically
    non_qualified = sorted(ind for ind in all_individuals_in_stats if ind not in qualified_set)
    individual_order = qualified + non_qualified

    # Load both splits
    print("Loading dataset...")
    train_dataset = load_dataset("kdoherty/wolverines", "reidentification", split="train")
    test_dataset = load_dataset("kdoherty/wolverines", "reidentification", split="test")

    train_ids = np.array(train_dataset['id'], dtype=object)
    train_scores = np.array(train_dataset['pelage_score'], dtype=np.float32)
    test_ids = np.array(test_dataset['id'], dtype=object)
    test_scores = np.array(test_dataset['pelage_score'], dtype=np.float32)

    # Collect scores per individual per split
    splits = ['train', 'val', 'test']
    data = {}  # {ind_id: {'train': array, 'val': array, 'test': array}}

    for ind_id in individual_order:
        train_mask = train_ids == ind_id
        all_train_idx = np.where(train_mask)[0]
        test_mask = test_ids == ind_id
        test_split_scores = test_scores[test_mask]

        if ind_id in qualified_set:
            # Qualified: train minus val indices, val from greedy temporal split
            val_idx = set(validation_indices.get(ind_id, {}).get('indices', []))
            keep = np.array([i not in val_idx for i in all_train_idx])
            train_split_scores = train_scores[all_train_idx[keep]]
            val_idx_list = sorted(val_idx)
            val_split_scores = train_scores[val_idx_list] if val_idx_list else np.array([])
        else:
            # Non-qualified: all train data serves as unknown/rare in val
            train_split_scores = np.array([])
            val_split_scores = train_scores[all_train_idx]

        data[ind_id] = {
            'train': train_split_scores,
            'val': val_split_scores,
            'test': test_split_scores,
        }
        tag = "*" if ind_id in qualified_set else " "
        print(f"  {tag} {ind_id}: train={len(train_split_scores)}, val={len(val_split_scores)}, test={len(test_split_scores)}")

    # Plot: rows=individual, cols=split
    n_rows = len(individual_order)
    n_cols = len(splits)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 2.2 * n_rows),
                             squeeze=False, sharey=False)

    bins = np.arange(0, 1.1, 0.1)
    split_colors = {'train': '#4878d0', 'val': '#ee854a', 'test': '#6acc64'}

    for row, ind_id in enumerate(individual_order):
        is_qualified = ind_id in qualified_set
        for col, split in enumerate(splits):
            ax = axes[row, col]
            scores = data[ind_id][split]

            if len(scores) > 0:
                ax.hist(scores, bins=bins, edgecolor='black', linewidth=0.4,
                        alpha=0.7, color=split_colors[split])
            else:
                ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                        transform=ax.transAxes, fontsize=8, color='0.5')

            ax.set_xlim(0, 1)
            ax.set_xticks(np.arange(0, 1.1, 0.1))
            ax.set_yscale('log')
            ax.yaxis.set_major_locator(ticker.LogLocator(base=10))
            ax.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs='auto', numticks=10))
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

            # Row labels on leftmost column
            if col == 0:
                label = ind_id if is_qualified else f"({ind_id})"
                ax.set_ylabel(label, fontsize=9, fontweight='bold' if is_qualified else 'normal')
            else:
                ax.set_ylabel('')

            # Column headers on top row
            if row == 0:
                ax.set_title(f"{split} (n={len(scores)})", fontsize=10, fontweight='bold')
            else:
                ax.set_title(f"n={len(scores)}", fontsize=8, color='0.4')

            # x-labels on bottom row only
            if row == n_rows - 1:
                ax.set_xlabel('Pelage Score', fontsize=8)
            else:
                ax.set_xlabel('')
                ax.tick_params(axis='x', labelbottom=False)

            ax.tick_params(labelsize=7)

    fig.suptitle('Quality Score Distributions by Individual and Split', fontsize=12, y=1.01)
    fig.tight_layout()

    output_path = os.path.join(args.output_dir, 'quality_histograms.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
