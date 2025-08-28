#!/bin/bash

# Individual ID Feature Importance Workflow
# Submits all jobs in correct dependency order

echo "=== Individual ID Feature Importance Workflow ==="
echo "Submitting jobs to SLURM..."

# Submit CV jobs
echo "1. Submitting cross-validation jobs (array 0-4)..."
CV_JOB=$(sbatch --parsable individual_id/sbatch/03_best_epochs.sbatch)
echo "   CV Job ID: $CV_JOB"

# Submit final training (depends on CV completion)
echo "2. Submitting final training job..."
FINAL_JOB=$(sbatch --parsable --dependency=afterok:$CV_JOB individual_id/sbatch/04_train_final.sbatch)
echo "   Final Job ID: $FINAL_JOB"

# Submit visualization (depends on final training completion)
echo "3. Submitting visualization job..."
VIZ_JOB=$(sbatch --parsable --dependency=afterok:$FINAL_JOB individual_id/sbatch/05_visualize.sbatch)
echo "   Visualization Job ID: $VIZ_JOB"

echo ""
echo "All jobs submitted successfully!"
echo ""
echo "Job Dependencies:"
echo "  CV (Array): $CV_JOB"
echo "  Final:      $FINAL_JOB (waits for CV)"
echo "  Viz:        $VIZ_JOB (waits for Final)"
echo ""
echo "Monitor progress with:"
echo "  squeue -u \$USER"
echo ""
echo "Expected outputs:"
echo "  - individual_id/results/best_epoch_cv/fold_{0-4}.json"
echo "  - individual_id/results/test_performance.json"
echo "  - individual_id/models/individual_id_final.pth"  
echo "  - individual_id/figures/feature_importance_grid.png"