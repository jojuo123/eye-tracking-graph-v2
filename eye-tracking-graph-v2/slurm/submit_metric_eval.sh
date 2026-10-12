#!/bin/bash
# Queues the full metric evaluation for each split: damage curves and model scoring
# run in parallel, then the overlay plot once both succeed. Run from
# eye-tracking-graph-v2/:
#   ./slurm/submit_metric_eval.sh                  # TEST and TRAIN, every sample
#   MAX_SAMPLES=600 ./slurm/submit_metric_eval.sh TRAIN
# Any variable from slurm/_env.sh (VENV, H5, CKPT, ...) can be overridden the same way.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p slurm-output/metric-eval

splits=("$@")
if [ ${#splits[@]} -eq 0 ]; then
    splits=(TEST TRAIN)
fi

for split in "${splits[@]}"; do
    damage=$(sbatch --parsable --export=ALL,SPLIT=$split --job-name="metric-damage-$split" slurm/metric_damage.sh)
    model=$(sbatch --parsable --export=ALL,SPLIT=$split --job-name="metric-model-$split" slurm/metric_model.sh)
    plot=$(sbatch --parsable --export=ALL,SPLIT=$split --job-name="metric-plot-$split" \
        --dependency=afterok:$damage:$model slurm/metric_plot.sh)
    echo "$split: damage=$damage model=$model plot=$plot (after both)"
done
