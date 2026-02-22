#!/bin/bash
cd /home/kdoherty/wolverines
python -u -m utils.reid_plotting --model dinov3 \
    --results_dir adaface/dinov3/results/hygiene \
    --output_dir adaface
