#!/bin/bash
# Aggregate AdaFace hygiene sweep results using the shared plotting module.
cd /home/kdoherty/wolverines
python -u adaface/aggregate_results.py "$@"
