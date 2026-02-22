"""
Aggregate AdaFace hygiene sweep results.

The shared plotting module (utils.reid_plotting) expects results with a float
'threshold' config key and hardcodes output paths to reid_openset/. This script:
1. Loads adaface results and injects a synthetic threshold field
   (quality_aware=1.0, quality_ignorant=0.0)
2. Writes patched copies with the expected filename pattern to a temp dir
3. Calls the individual plotting functions with adaface/ output paths
"""

import os
import sys
import json
import glob
import shutil
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

LOSS_MODE_TO_THRESHOLD = {
    "quality_aware": 1.0,
    "quality_ignorant": 0.0,
}

MODEL_NAME = "dinov3"


def main():
    src_dir = "adaface/dinov3/results/hygiene"

    src_files = glob.glob(os.path.join(src_dir, "mode=*_gallery=*_seed=*.json"))
    if not src_files:
        print(f"No adaface hygiene results found in {src_dir}")
        sys.exit(1)

    print(f"Found {len(src_files)} adaface hygiene result files")

    # Create temp dir with patched copies
    tmp_dir = tempfile.mkdtemp(prefix="adaface_plot_")

    try:
        for src_file in src_files:
            with open(src_file, 'r') as f:
                data = json.load(f)

            loss_mode = data['config']['loss_mode']
            threshold = LOSS_MODE_TO_THRESHOLD[loss_mode]
            gallery_size = data['config']['gallery_size']
            seed = data['config']['seed']

            # Inject synthetic threshold so the plotter can group by it
            data['config']['threshold'] = threshold

            dst_name = f"threshold={threshold:.2f}_gallery={gallery_size}_seed={seed}.json"
            dst_path = os.path.join(tmp_dir, dst_name)

            with open(dst_path, 'w') as f:
                json.dump(data, f, indent=2)

        print(f"Wrote {len(src_files)} patched files to {tmp_dir}")

        # Import plotting components individually to control output paths
        from utils.reid_plotting import (
            load_hygiene_results, print_summary_table,
            collect_strategies_data, save_scores_table,
            save_thresholds_table, save_summary_table,
            plot_recall_curves, plot_validation_loss_curves,
            plot_cosine_threshold,
        )

        results = load_hygiene_results(tmp_dir)
        if len(results) == 0:
            print("No results loaded. Exiting.")
            return

        print_summary_table(results)

        # Tables -> adaface/tables/dinov3/
        tables_dir = os.path.join("adaface", "tables", MODEL_NAME)
        os.makedirs(tables_dir, exist_ok=True)

        strategies = collect_strategies_data(results)
        save_scores_table(strategies, tables_dir)
        save_thresholds_table(strategies, tables_dir)
        save_summary_table(results, tables_dir)

        # Figures -> adaface/figures/dinov3/
        figures_dir = os.path.join("adaface", "figures", MODEL_NAME)
        os.makedirs(figures_dir, exist_ok=True)

        plot_recall_curves(results, os.path.join(figures_dir, "recall_curves.png"))
        plot_validation_loss_curves(results, os.path.join(figures_dir, "validation_loss_curves.png"))
        plot_cosine_threshold(results, os.path.join(figures_dir, "cosine_threshold.png"))

        print(f"\nDiagnostics complete for {MODEL_NAME}!")
        print(f"  Tables:  {tables_dir}")
        print(f"  Figures: {figures_dir}")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
