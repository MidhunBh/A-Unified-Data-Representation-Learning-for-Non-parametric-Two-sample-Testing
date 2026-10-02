#!/usr/bin/env bash

# Avoid conda deactivate scripts crashing on unset variables.
set +u

ROOT="/system/user/publicwork/mbhaskar/A-Unified-Data-Representation-Learning-for-Non-parametric-Two-sample-Testing/Experiments"

cd "$ROOT"

source /system/user/publicwork/mbhaskar/miniconda3/etc/profile.d/conda.sh
conda activate rltst

mkdir -p output/sensitivity_split_ratio
mkdir -p result/split_ratio_cells

echo "============================================================"
echo "HDGM d=10 split-ratio sensitivity"
echo "Started: $(date)"
echo "Python: $(which python)"
python --version
echo
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader
echo "============================================================"


###############################################################################
# BATCH 1
###############################################################################

echo
echo "===== BATCH 1 ====="
echo "400/600 -> GPU 2"
echo "500/500 -> GPU 3"

CUDA_VISIBLE_DEVICES=2 \
PYTHONUNBUFFERED=1 \
python -u factorial/factorial_split_ratio_HDGM_d10_cell.py \
    --n-train 400 \
    --n-test 600 \
    > output/sensitivity_split_ratio/400_600.log 2>&1 &

PID1=$!

CUDA_VISIBLE_DEVICES=3 \
PYTHONUNBUFFERED=1 \
python -u factorial/factorial_split_ratio_HDGM_d10_cell.py \
    --n-train 500 \
    --n-test 500 \
    > output/sensitivity_split_ratio/500_500.log 2>&1 &

PID2=$!

echo "PIDs: $PID1 $PID2"

wait "$PID1"
STATUS1=$?

wait "$PID2"
STATUS2=$?

echo "Batch 1 exit codes: $STATUS1 $STATUS2"

if [ "$STATUS1" -ne 0 ] || [ "$STATUS2" -ne 0 ]; then
    echo "ERROR: Batch 1 failed."
    echo "Inspect:"
    echo "  output/sensitivity_split_ratio/400_600.log"
    echo "  output/sensitivity_split_ratio/500_500.log"
    exit 1
fi


###############################################################################
# BATCH 2
###############################################################################

echo
echo "===== BATCH 2 ====="
echo "600/400 -> GPU 2"
echo "700/300 -> GPU 3"

CUDA_VISIBLE_DEVICES=2 \
PYTHONUNBUFFERED=1 \
python -u factorial/factorial_split_ratio_HDGM_d10_cell.py \
    --n-train 600 \
    --n-test 400 \
    > output/sensitivity_split_ratio/600_400.log 2>&1 &

PID3=$!

CUDA_VISIBLE_DEVICES=3 \
PYTHONUNBUFFERED=1 \
python -u factorial/factorial_split_ratio_HDGM_d10_cell.py \
    --n-train 700 \
    --n-test 300 \
    > output/sensitivity_split_ratio/700_300.log 2>&1 &

PID4=$!

echo "PIDs: $PID3 $PID4"

wait "$PID3"
STATUS3=$?

wait "$PID4"
STATUS4=$?

echo "Batch 2 exit codes: $STATUS3 $STATUS4"

if [ "$STATUS3" -ne 0 ] || [ "$STATUS4" -ne 0 ]; then
    echo "ERROR: Batch 2 failed."
    echo "Inspect:"
    echo "  output/sensitivity_split_ratio/600_400.log"
    echo "  output/sensitivity_split_ratio/700_300.log"
    exit 1
fi


###############################################################################
# MERGE
###############################################################################

echo
echo "===== MERGING ====="

python factorial/merge_split_ratio_HDGM_d10.py \
  | tee output/sensitivity_split_ratio/merged_summary.log

echo
echo "===== COMPLETE ====="
echo "Finished: $(date)"
echo
cat result/factorial_split_ratio_HDGM_d10_N4000.json
