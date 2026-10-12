#!/bin/bash
# Synthetic-damage sensitivity curves for every shortlisted metric, full-length
# scanpaths, parallel over samples. Submit from eye-tracking-graph-v2/:
#   SPLIT=TEST sbatch --export=ALL slurm/metric_damage.sh
#
# Resumable: SLURM sends SIGUSR1 5 min before the walltime, the script checkpoints
# to $DAMAGE_DIR/checkpoint.json and exits cleanly; resubmitting the same job picks
# up where it stopped (see run_damage_evaluation.py's docstring).
#SBATCH --job-name=metric-damage
#SBATCH --nodes=1
#SBATCH --ntasks=1 --cpus-per-task=32 --mem=32000M
#SBATCH --time=0-06:00:00
#SBATCH --output=slurm-output/metric-eval/damage-%j.out
#SBATCH --signal=B:USR1@300

source slurm/_env.sh
# --workers 0 = one worker per CPU SLURM allotted (the script caps BLAS threads at 1).

# exec: python replaces the batch shell, so --signal=B:USR1 reaches it directly.
exec python -u metric-evaluation/run_damage_evaluation.py \
    --h5-path "$H5" --coord-scale "$COORD_SCALE" --split "$SPLIT" ${SAMPLE_ARGS[@]+"${SAMPLE_ARGS[@]}"} \
    --max-fixations 0 --n-repeats 3 --workers 0 \
    --output-dir "$DAMAGE_DIR"
