#!/bin/bash
# Plot hygiene sweep results for MegaDescriptor
# Run locally (no sbatch needed): bash reid_openset/megadescriptor/scripts/04_plot_hygiene.sh

cd /home/kdoherty/wolverines

python -u -m utils.reid_plotting --model megadescriptor
