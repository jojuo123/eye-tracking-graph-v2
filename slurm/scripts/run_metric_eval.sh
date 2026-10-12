#!/bin/bash
#The partition is the queue you want to run on. standard is gpu and can be ommitted.
#SBATCH --job-name=eyegraph-metric-eval
#SBATCH --nodes=1
#number of independent tasks we are going to start in this script
#SBATCH --ntasks=1
#number of cpus we want to allocate for each program
#SBATCH --cpus-per-task=32
#No GPU: model inference is cheap on cpu; the metrics run on all cpus
#Expected runtime: ~1 h for TEST + TRAIN together; limit leaves margin
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=0-04:00:00
#SBATCH --output=slurm/outputs/metric-eval/metric-eval-log-%j.out
#Skipping many options! see man sbatch
# From here on, we can start our program

# Everything in one job, for each split: damage curves -> model evaluation -> plot of
# the damage curves with the model as a horizontal line.
# Usage (from new-src/):
#   mkdir -p slurm/outputs/metric-eval
#   sbatch slurm/scripts/run_metric_eval.sh              # TEST and TRAIN
#   sbatch slurm/scripts/run_metric_eval.sh TEST         # one split only
# If the job runs out of time, submit the same command again: the damage sweep resumes
# from its checkpoint and finished model evaluations are skipped.
SPLITS=${@:-TEST TRAIN}

H5=../erda2/eye-tracking/normalized_data_softGT_scaled20.h5   # relative to metric-evaluation/
COORD_SCALE=20   # the scaled20 file stores coordinates x20; the metrics need [0, 1]
CHECKPOINT=../work_dir/loss_ablation/doubly_stochastic_cross_entropy_soft/checkpoints/checkpoint_best.pt
WORKERS=${SLURM_CPUS_PER_TASK:-1}

module load python/3.12.8

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt  metric evaluation, splits: $SPLITS"

source /home/lht444/python-venv/eye-tracking/bin/activate
cd eye-tracking-graph-v2
./mount_erda.sh
cd metric-evaluation

for SPLIT in $SPLITS; do
    split=$(echo "$SPLIT" | tr '[:upper:]' '[:lower:]')
    DAMAGE_OUT=../work_dir/metric_damage_curves_scaled20_$split
    MODEL_OUT=../work_dir/metric_model_evaluation_scaled20_$split

    echo "=== $SPLIT: damage curves ($(date '+%H:%M:%S'))"
    # Full-length scanpaths (--max-fixations 0), the same ones the model is scored on.
    python run_damage_evaluation.py \
            --split "$SPLIT" \
            --h5-path "$H5" \
            --coord-scale "$COORD_SCALE" \
            --max-fixations 0 \
            --n-repeats 3 \
            --output-dir "$DAMAGE_OUT" \
            --checkpoint-every 300 \
            --workers "$WORKERS"

    # The sweep exits 0 even when stopped early, so ask its checkpoint whether it finished.
    if ! python -c "
import json, sys
s = json.load(open('$DAMAGE_OUT/checkpoint.json'))
sys.exit(0 if s['cursor'] >= s['units_total'] else 1)
"; then
        echo "$SPLIT damage sweep incomplete -- submit this script again to resume"
        break
    fi

    echo "=== $SPLIT: model evaluation ($(date '+%H:%M:%S'))"
    if [ -f "$MODEL_OUT/summary.json" ]; then
        echo "already done ($MODEL_OUT/summary.json exists) -- skipping"
    else
        python run_model_evaluation.py \
                --checkpoint "$CHECKPOINT" \
                --split "$SPLIT" \
                --h5-path "$H5" \
                --coord-scale "$COORD_SCALE" \
                --n-random 10 \
                --workers "$WORKERS" \
                --output-dir "$MODEL_OUT"
    fi

    echo "=== $SPLIT: plot ($(date '+%H:%M:%S'))"
    python plot_damage_vs_model.py --damage-dir "$DAMAGE_OUT" --model-dir "$MODEL_OUT"
done

cd ..
./unmount_erda.sh
echo "finished $(date '+%d/%m/%Y %H:%M:%S')"
