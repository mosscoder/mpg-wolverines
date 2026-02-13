#!/usr/bin/env python3
"""
Plot quality score distribution for test data, faceted by individual.
Includes both closed-set individuals (valid for reid) and open-set unknowns.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default='preprocessing/results/feasible_individuals.json')
    parser.add_argument('--output', type=str,
                        default='preprocessing/results/test_quality_histogram.png')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Load test split
    test_dataset = load_dataset("kdoherty/wolverines", "reidentification", split="test")

    # Use Arrow columnar access (fast)
    all_ids = np.array(test_dataset['id'], dtype=object)
    all_quality = np.array(test_dataset['pelage_score'], dtype=np.float32)

    valid_individuals_set = set(config['valid_individuals'])
    records = []

    # Extract test samples for closed-set individuals
    for ind_id in config['valid_individuals']:
        mask = all_ids == ind_id
        quality_scores = all_quality[mask]
        for q in quality_scores:
            records.append({
                'individual': ind_id,
                'pelage_score': float(q),
                'set_type': 'closed-set'
            })

    # Find open-set individuals: those NOT in valid_individuals
    unique_ids = np.unique(all_ids)
    open_set_individuals = [ind_id for ind_id in unique_ids if ind_id not in valid_individuals_set]

    for ind_id in open_set_individuals:
        mask = all_ids == ind_id
        quality_scores = all_quality[mask]
        for q in quality_scores:
            records.append({
                'individual': ind_id,
                'pelage_score': float(q),
                'set_type': 'open-set'
            })

    df = pd.DataFrame(records)

    print(f"Closed-set individuals: {len(config['valid_individuals'])}")
    print(f"Open-set individuals: {len(open_set_individuals)}")
    for ind_id in open_set_individuals:
        count = (df['individual'] == ind_id).sum()
        print(f"  {ind_id}: {count} samples")

    # Order individuals: closed-set first (sorted), then open-set (sorted)
    individual_order = sorted(config['valid_individuals']) + sorted(open_set_individuals)
    df['individual'] = pd.Categorical(df['individual'], categories=individual_order, ordered=True)

    # Create faceted histogram with independent y-axis scales
    n_individuals = len(individual_order)
    col_wrap = 4 if n_individuals > 6 else 3
    g = sns.FacetGrid(df, col='individual', col_wrap=col_wrap, height=2.5, sharey=False)
    g.map(plt.hist, 'pelage_score', bins=20, edgecolor='black', alpha=0.7)
    g.set_xlabels('Quality Score')
    g.set_ylabels('Count')
    g.fig.suptitle('Test Set Quality Score Distribution by Individual', y=1.02)
    g.tight_layout()

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    g.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
