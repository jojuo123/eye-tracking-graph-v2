#!/bin/bash
# Shared setup for the metric-evaluation SLURM jobs: sourced (not run) by each job
# script, from eye-tracking-graph-v2/. Override any variable at submit time, e.g.
#   VENV=~/python-venv/other sbatch --export=ALL slurm/metric_damage.sh
#
# The venv needs the project requirements plus matplotlib:
#   pip install -r ../requirements.txt matplotlib

: "${VENV:=$HOME/python-venv/eyegraph}"
: "${H5:=erda2/eye-tracking/normalized_data_softGT_scaled20.h5}"
: "${COORD_SCALE:=20}"   # *_scaled20.h5 stores x/y_position_norm x 20; the metrics need [0, 1]
: "${CKPT:=work_dir/loss_ablation/doubly_stochastic_cross_entropy_soft/checkpoints/checkpoint_best.pt}"
: "${SPLIT:=TEST}"
: "${MAX_SAMPLES:=}"     # empty = every sample in the split; otherwise a seed-0 random subset
: "${OUT_ROOT:=work_dir}"

set -euo pipefail

# Mount ERDA only if this node doesn't have it yet; never unmount here, since other
# jobs on the same node may be reading from it.
if ! mountpoint -q erda2; then
    ./mount_erda.sh
fi
if [ ! -f "$H5" ]; then
    echo "H5 file not found: $H5 (is ERDA mounted?)" >&2
    exit 1
fi

module load python/3.12.8
source "$VENV/bin/activate"

split_lc=$(echo "$SPLIT" | tr '[:upper:]' '[:lower:]')
DAMAGE_DIR="$OUT_ROOT/metric_damage_curves_scaled20_${split_lc}"
MODEL_DIR="$OUT_ROOT/metric_model_evaluation_scaled20_${split_lc}"
SAMPLE_ARGS=()
if [ -n "$MAX_SAMPLES" ]; then
    SAMPLE_ARGS=(--max-samples "$MAX_SAMPLES")
fi
