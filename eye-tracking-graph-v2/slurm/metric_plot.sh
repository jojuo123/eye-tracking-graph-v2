#!/bin/bash
# Damage curves with the model's scores as horizontal lines, one figure per damage
# type. Needs metric_damage.sh and metric_model.sh to have finished for the same
# SPLIT (submit_metric_eval.sh chains it with --dependency=afterok).
#SBATCH --job-name=metric-plot
#SBATCH --nodes=1
#SBATCH --ntasks=1 --cpus-per-task=1 --mem=4000M
#SBATCH --time=0-00:30:00
#SBATCH --output=slurm-output/metric-eval/plot-%j.out

source slurm/_env.sh

python -u metric-evaluation/plot_damage_vs_model.py \
    --damage-dir "$DAMAGE_DIR" --model-dir "$MODEL_DIR"
