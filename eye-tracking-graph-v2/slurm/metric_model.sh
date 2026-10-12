#!/bin/bash
# Scores the trained model's predicted fixation orders (plus random and top-to-bottom
# baselines) with every shortlisted metric. CPU only: model inference is cheap next
# to the pure-Python metrics. Submit from eye-tracking-graph-v2/:
#   SPLIT=TEST sbatch --export=ALL slurm/metric_model.sh
#SBATCH --job-name=metric-model
#SBATCH --nodes=1
#SBATCH --ntasks=1 --cpus-per-task=4 --mem=16000M
#SBATCH --time=0-12:00:00
#SBATCH --output=slurm-output/metric-eval/model-%j.out

source slurm/_env.sh
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

python -u metric-evaluation/run_model_evaluation.py \
    --checkpoint "$CKPT" --device cpu \
    --h5-path "$H5" --coord-scale "$COORD_SCALE" --split "$SPLIT" ${SAMPLE_ARGS[@]+"${SAMPLE_ARGS[@]}"} \
    --n-random 10 --output-dir "$MODEL_DIR"
