"""
Aggregate AdaFace hygiene sweep results.

The shared plotting module (utils.reid_plotting) expects results with a float
'threshold' config key. This script loads adaface results, injects a synthetic
threshold field (quality_aware=1.0, quality_ignorant=0.0), writes them to a
temp directory with the expected filename pattern, then invokes the plotter.
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


def main():
    src_dir = "adaface/dinov3/results/hygiene"
    output_dir = "adaface"

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

        # Invoke the shared plotter
        from utils.reid_plotting import run_single_model
        run_single_model("dinov3", tmp_dir, output_dir)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
