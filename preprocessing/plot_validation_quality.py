#!/usr/bin/env python3
"""
Plot quality score distribution for validation data, faceted by individual.
Includes both known individuals and unknown wolverines.
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
                        default='preprocessing/results/validation_quality_histogram.png')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Load dataset
    dataset = load_dataset("kdoherty/wolverines", "reidentification", split="train")

    # Collect all validation indices from known individuals
    known_validation_indices = set()
    records = []

    # Extract validation samples for known individuals
    for ind_id in config['valid_individuals']:
        indices = config['validation_indices'][ind_id]['indices']
        known_validation_indices.update(indices)
        for idx in indices:
            sample = dataset[idx]
            records.append({
                'individual': ind_id,
                'pelage_score': sample['pelage_score']
            })

    # Find and add unknown wolverines (id == 'unknown')
    for idx in range(len(dataset)):
        sample = dataset[idx]
        if sample['id'] == 'unknown':
            records.append({
                'individual': 'unknown',
                'pelage_score': sample['pelage_score']
            })

    df = pd.DataFrame(records)

    # Order individuals: known first (sorted), then unknown last
    individual_order = sorted(config['valid_individuals']) + ['unknown']
    individual_order = [ind for ind in individual_order if ind in df['individual'].unique()]
    df['individual'] = pd.Categorical(df['individual'], categories=individual_order, ordered=True)

    # Create faceted histogram with independent y-axis scales
    g = sns.FacetGrid(df, col='individual', col_wrap=3, height=3, sharey=False)
    g.map(plt.hist, 'pelage_score', bins=20, edgecolor='black', alpha=0.7)
    g.set_xlabels('Quality Score')
    g.set_ylabels('Count')
    g.fig.suptitle('Validation Quality Score Distribution by Individual', y=1.02)
    g.tight_layout()

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    g.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
