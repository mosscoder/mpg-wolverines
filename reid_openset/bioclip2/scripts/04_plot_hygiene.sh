#!/bin/bash
# Plot hygiene sweep results for BioCLIP-2
# Run locally (no sbatch needed): bash reid_openset/bioclip2/scripts/04_plot_hygiene.sh

cd /home/kdoherty/wolverines

python -u utils/reid_plotting.py --model bioclip2
