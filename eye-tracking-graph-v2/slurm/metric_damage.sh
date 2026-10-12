#!/bin/bash
# Synthetic-damage sensitivity curves for every shortlisted metric, full-length
# scanpaths, parallel over samples. Submit from eye-tracking-graph-v2/:
#   SPLIT=TEST sbatch --export=ALL slurm/metric_damage.sh
#SBATCH --job-name=metric-damage
#SBATCH --nodes=1
#SBATCH --ntasks=1 --cpus-per-task=32 --mem=32000M
#SBATCH --time=0-06:00:00
#SBATCH --output=slurm-output/metric-eval/damage-%j.out

source slurm/_env.sh
export OMP_NUM_THREADS=1   # the metrics are pure Python; one worker per CPU

python -u metric-evaluation/run_damage_evaluation.py \
    --h5-path "$H5" --coord-scale "$COORD_SCALE" --split "$SPLIT" ${SAMPLE_ARGS[@]+"${SAMPLE_ARGS[@]}"} \
    --max-fixations 0 --n-repeats 3 --workers "${SLURM_CPUS_PER_TASK:-1}" \
    --output-dir "$DAMAGE_DIR"
