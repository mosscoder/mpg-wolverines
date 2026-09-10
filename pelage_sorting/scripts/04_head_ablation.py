#!/usr/bin/env python3
"""
Script 04: Head Ablation (R1.4)
Linear versus two-layer nonlinear head on identical frozen DINOv3 features,
for the reviewer-requested ablation. Runs locally, touches no existing
experiment: the production pipeline (scripts 00-02) is deterministic up to
the head (fixed resize, no augmentation, frozen backbone), so per-image CLS
features are constants. They are extracted once, cached, and both heads
train on the same cached features under the production protocol:
AdamW lr=0.001 weight_decay=0.01, batch 32, CrossEntropyLoss, macro-F1
metrics, epochs selected by 5-fold CV exactly as in scripts 00/01.

Heads:
  linear   Linear(768 -> 2), the published architecture. Fidelity control:
           its test F1 must reproduce results/01_test within seed noise.
  mlp      Linear(768 -> h) -> ReLU -> Linear(h -> 2), the reviewer's two
           fully connected layers. h selected on CV from {256, 768}.

Final runs use seeds 0-4 on the full train split for the CV-selected epoch
count, reporting final-epoch test metrics (matching final_test_f1 in
results/01_test).

    python scripts/04_head_ablation.py            # from pelage_sorting/

Outputs to results/04_head_ablation/: per-run JSONs, summary.json, and a
gitignored features.npz cache (reproducible in one forward pass).
"""

import sys
import os
import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold

script_dir = os.path.dirname(os.path.abspath(__file__))
wolverines_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(wolverines_root)

from utils.dataset import set_all_seeds
from utils.preprocessing import get_standard_transform
from utils.training import compute_metrics

ARROW_DIR = os.path.join(wolverines_root, 'hugging_face_dataset/v2/data/pelage_dataset')
PUBLISHED = os.path.join(script_dir, '../results/01_test/final_test_performance.json')

# Production protocol (scripts 00/01)
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.01
BATCH_SIZE = 32
CV_EPOCHS = 50
CV_FOLDS = 5
CV_SEED = 0
FINAL_SEEDS = [0, 1, 2, 3, 4]
MLP_HIDDEN_GRID = [256, 768]
FEATURE_DIM = 768


def extract_features(npz_path, device):
    """One frozen forward pass over train/test; cache CLS features."""
    from datasets import load_from_disk
    from utils.models import create_model

    model = create_model(device=device)
    model.eval()
    transform = get_standard_transform(resize_size=224)

    out = {}
    for split in ['train', 'test']:
        ds = load_from_disk(os.path.join(ARROW_DIR, split))
        feats, labels = [], []
        batch = []
        print(f"extracting {split}: {len(ds)} images")
        for i, row in enumerate(ds):
            batch.append(transform(row['image'].convert('RGB')))
            labels.append(1 if row['label'] == 2 else 0)  # binary: full vs rest
            if len(batch) == 64 or i == len(ds) - 1:
                x = torch.stack(batch).to(model.backbone.device)
                with torch.no_grad():
                    cls = model.backbone(x).last_hidden_state[:, 0, :]
                feats.append(cls.cpu().float().numpy())
                batch = []
        out[f'{split}_x'] = np.concatenate(feats)
        out[f'{split}_y'] = np.array(labels, dtype=np.int64)

    assert out['train_x'].shape == (2431, FEATURE_DIM), out['train_x'].shape
    assert out['test_x'].shape == (270, FEATURE_DIM), out['test_x'].shape
    assert int(out['train_y'].sum()) == 176 and int(out['test_y'].sum()) == 19
    np.savez(npz_path, **out)
    print(f"cached features to {npz_path}")
    return out


def make_head(name, hidden=None):
    if name == 'linear':
        return nn.Sequential(nn.Linear(FEATURE_DIM, 2))
    return nn.Sequential(nn.Linear(FEATURE_DIM, hidden), nn.ReLU(),
                         nn.Linear(hidden, 2))


def train_head(head, train_x, train_y, eval_x, eval_y, epochs):
    """Production training loop on cached features; returns per-epoch eval metrics."""
    loader = DataLoader(TensorDataset(torch.from_numpy(train_x),
                                      torch.from_numpy(train_y)),
                        batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    opt = torch.optim.AdamW(head.parameters(), lr=LEARNING_RATE,
                            weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    ex, ey = torch.from_numpy(eval_x), eval_y
    history = []
    for _ in range(epochs):
        head.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = criterion(head(xb), yb)
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            preds = head(ex).argmax(dim=1).numpy()
        history.append(compute_metrics(preds, ey))
    return history, preds


def cv_folds(labels):
    """Replicate create_kfold_splits: per-label KFold(shuffle, seed), concatenated."""
    folds = None
    for label in [0, 1]:
        indices = np.where(labels == label)[0]
        kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=CV_SEED)
        label_folds = [(indices[tr], indices[va]) for tr, va in kf.split(indices)]
        if folds is None:
            folds = label_folds
        else:
            folds = [(np.concatenate([a, c]), np.concatenate([b, d]))
                     for (a, b), (c, d) in zip(folds, label_folds)]
    return folds


def select_epochs(head_name, hidden, train_x, train_y):
    """Script 00/01 rule: mean val F1 per epoch across folds, argmax (1-indexed)."""
    histories = []
    for fold, (tr, va) in enumerate(cv_folds(train_y)):
        set_all_seeds(fold)  # script 00: seed = fold
        head = make_head(head_name, hidden)
        history, _ = train_head(head, train_x[tr], train_y[tr],
                                train_x[va], train_y[va], CV_EPOCHS)
        histories.append([h['f1_score'] for h in history])
    mean_f1 = np.mean(histories, axis=0)
    best_epoch = int(np.argmax(mean_f1)) + 1
    return best_epoch, float(mean_f1[best_epoch - 1])


def binary_metrics(preds, labels):
    from sklearn.metrics import f1_score, precision_score, recall_score
    return {
        'f1_score': f1_score(labels, preds, pos_label=1),
        'precision': precision_score(labels, preds, pos_label=1, zero_division=0),
        'recall': recall_score(labels, preds, pos_label=1, zero_division=0),
    }


def main():
    parser = argparse.ArgumentParser(description='R1.4 head ablation on cached features')
    parser.add_argument('--output_dir', type=str, default='results/04_head_ablation')
    parser.add_argument('--device', type=str, default='mps',
                        help='device for feature extraction only; heads train on cpu')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    npz_path = os.path.join(args.output_dir, 'features.npz')
    if os.path.exists(npz_path):
        data = dict(np.load(npz_path))
        print(f"loaded cached features from {npz_path}")
    else:
        data = extract_features(npz_path, args.device)
    train_x, train_y = data['train_x'], data['train_y']
    test_x, test_y = data['test_x'], data['test_y']

    # ---- Epoch (and width) selection by 5-fold CV, per head ----
    configs = {}
    epochs, cv_f1 = select_epochs('linear', None, train_x, train_y)
    configs['linear'] = {'hidden': None, 'optimal_epochs': epochs, 'cv_f1': cv_f1}
    print(f"linear: optimal_epochs={epochs}, cv_f1={cv_f1:.4f}")

    mlp_grid = {}
    for h in MLP_HIDDEN_GRID:
        epochs, cv_f1 = select_epochs('mlp', h, train_x, train_y)
        mlp_grid[h] = {'optimal_epochs': epochs, 'cv_f1': cv_f1}
        print(f"mlp h={h}: optimal_epochs={epochs}, cv_f1={cv_f1:.4f}")
    best_h = max(mlp_grid, key=lambda h: mlp_grid[h]['cv_f1'])
    configs['mlp'] = {'hidden': best_h, 'cv_grid': {str(h): v for h, v in mlp_grid.items()},
                      **mlp_grid[best_h]}
    print(f"mlp: selected h={best_h}")

    # ---- Final runs: full train, CV-selected epochs, seeds 0-4 ----
    summary = {'protocol': {'learning_rate': LEARNING_RATE, 'weight_decay': WEIGHT_DECAY,
                            'batch_size': BATCH_SIZE, 'cv_epochs': CV_EPOCHS,
                            'cv_folds': CV_FOLDS, 'cv_seed': CV_SEED,
                            'final_seeds': FINAL_SEEDS, 'mlp_hidden_grid': MLP_HIDDEN_GRID},
               'configs': configs, 'heads': {}}
    for name, cfg in configs.items():
        macro_f1s, bin_f1s = [], []
        for seed in FINAL_SEEDS:
            set_all_seeds(seed)
            head = make_head(name, cfg['hidden'])
            start = time.time()
            history, preds = train_head(head, train_x, train_y,
                                        test_x, test_y, cfg['optimal_epochs'])
            result = {
                'head': name, 'hidden': cfg['hidden'], 'seed': seed,
                'optimal_epochs': cfg['optimal_epochs'],
                'trainable_parameters': sum(p.numel() for p in head.parameters()),
                'final_test': history[-1],
                'final_test_binary': binary_metrics(preds, test_y),
                'best_test_f1': max(h['f1_score'] for h in history),
                'test_history': history,
                'training_time': time.time() - start,
            }
            tag = f"head={name}" + (f"_h={cfg['hidden']}" if cfg['hidden'] else '')
            with open(os.path.join(args.output_dir, f"{tag}_seed={seed}.json"), 'w') as f:
                json.dump(result, f, indent=2)
            macro_f1s.append(history[-1]['f1_score'])
            bin_f1s.append(result['final_test_binary']['f1_score'])
            print(f"{tag} seed={seed}: test macro F1 {macro_f1s[-1]:.4f}, "
                  f"binary F1 {bin_f1s[-1]:.4f}")
        summary['heads'][name] = {
            'hidden': cfg['hidden'], 'optimal_epochs': cfg['optimal_epochs'],
            'final_test_f1_mean': float(np.mean(macro_f1s)),
            'final_test_f1_sd': float(np.std(macro_f1s, ddof=1)),
            'final_test_f1_per_seed': macro_f1s,
            'final_test_binary_f1_mean': float(np.mean(bin_f1s)),
            'final_test_binary_f1_sd': float(np.std(bin_f1s, ddof=1)),
            'final_test_binary_f1_per_seed': bin_f1s,
        }

    with open(PUBLISHED) as f:
        summary['published_reference'] = json.load(f)['final_performance']

    with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n=== SUMMARY ===")
    for name, s in summary['heads'].items():
        label = name if not s['hidden'] else f"{name} (h={s['hidden']})"
        print(f"{label}: "
              f"test macro F1 {s['final_test_f1_mean']:.4f} +/- {s['final_test_f1_sd']:.4f} "
              f"({s['optimal_epochs']} epochs)")
    print(f"published final test F1: {summary['published_reference']['final_test_f1']:.4f}")


if __name__ == '__main__':
    main()
