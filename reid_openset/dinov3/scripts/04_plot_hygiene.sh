#!/bin/bash
# Plot hygiene sweep results for DINOv3
# Run locally (no sbatch needed): bash reid_openset/dinov3/scripts/04_plot_hygiene.sh

cd /home/kdoherty/wolverines

python -u utils/reid_plotting.py --model dinov3
