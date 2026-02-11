"""
Generate figures and tables from final test evaluation results.

Outputs:
  - figures/test_R@1.png: Barplot of R@1 per backbone
  - tables/reid_hyperparams.csv: All hyperparameters per backbone
  - tables/test_R@1.json: Numerical R@1 per backbone

Usage:
    python test_best_configs/scripts/plot_results.py
"""

import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append('.')

from utils.reid import MODEL_CONFIGS


RESULTS_DIR = 'test_best_configs/results'
FIGURES_DIR = 'test_best_configs/figures'
TABLES_DIR = 'test_best_configs/tables'

MODELS = list(MODEL_CONFIGS.keys())


def load_results():
    """Load result JSONs for all backbones."""
    results = {}
    for model_name in MODELS:
        path = os.path.join(RESULTS_DIR, f'{model_name}_test.json')
        if not os.path.exists(path):
            print(f"WARNING: Missing result file: {path}")
            continue
        with open(path, 'r') as f:
            results[model_name] = json.load(f)
        print(f"Loaded {model_name}: R@1 = {results[model_name]['results']['test_recall_at_1']:.4f}")
    return results


def create_barplot(results):
    """Create barplot: backbone on x-axis, R@1 on y-axis."""
    os.makedirs(FIGURES_DIR, exist_ok=True)

    names = []
    r1_values = []
    for model_name in MODELS:
        if model_name not in results:
            continue
        r = results[model_name]
        label = r['backbone'].replace('Frozen ', '')
        names.append(label)
        r1_values.append(r['results']['test_recall_at_1'])

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(names, r1_values, color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(names)],
                  width=0.5, edgecolor='black', linewidth=0.8)

    # Value labels on bars
    for bar, val in zip(bars, r1_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('Test Recall at rank 1', fontsize=13)
    ax.set_ylim(0, 1.0)
    ax.set_title('Final Test Performance by Backbone', fontsize=14)
    ax.tick_params(axis='x', labelsize=11)
    ax.tick_params(axis='y', labelsize=11)

    plt.tight_layout()
    output_path = os.path.join(FIGURES_DIR, 'test_R@1.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved barplot: {output_path}")


def create_hyperparams_csv(results):
    """Create CSV with all hyperparameters per backbone."""
    os.makedirs(TABLES_DIR, exist_ok=True)

    rows = []
    for model_name in MODELS:
        if model_name not in results:
            continue
        r = results[model_name]
        cfg = r['config']
        rows.append({
            'model': model_name,
            'backbone': r['backbone'].replace('Frozen ', ''),
            'lr': cfg['learning_rate'],
            'image_size': cfg['image_size'],
            'embedding_dim': cfg['embedding_dim'],
            'epochs': cfg['epochs'],
            'gallery_threshold': cfg['gallery_threshold'],
            'query_threshold': cfg['query_threshold'],
            'cosine_similarity_threshold': round(cfg['cosine_similarity_threshold'], 4),
            'train_images': r['dataset']['train_total'],
            'test_images': r['dataset']['test_total'],
            'test_r1': round(r['results']['test_recall_at_1'], 4),
        })

    df = pd.DataFrame(rows)
    output_path = os.path.join(TABLES_DIR, 'reid_hyperparams.csv')
    df.to_csv(output_path, index=False)
    print(f"Saved hyperparams CSV: {output_path}")
    print(df.to_string(index=False))


def create_r1_json(results):
    """Create JSON with R@1 per backbone."""
    os.makedirs(TABLES_DIR, exist_ok=True)

    data = {}
    for model_name in MODELS:
        if model_name not in results:
            continue
        r = results[model_name]
        data[model_name] = {
            'backbone': r['backbone'].replace('Frozen ', ''),
            'test_r1': round(r['results']['test_recall_at_1'], 4),
        }

    output_path = os.path.join(TABLES_DIR, 'test_R@1.json')
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved R@1 JSON: {output_path}")


def main():
    print("=" * 60)
    print("Final Test Results - Figures & Tables")
    print("=" * 60)

    results = load_results()

    if not results:
        print("No results found. Run sweep_best.py first.")
        return

    create_barplot(results)
    create_hyperparams_csv(results)
    create_r1_json(results)

    print(f"\nDone. Generated {len(results)} backbone results.")


if __name__ == '__main__':
    main()
