#!/usr/bin/env bash
set -u

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

declare -a PIDS=()

run_cell () {
    TRAIN="$1"
    TEST="$2"
    GPU="$3"
    LABEL="${TRAIN}_${TEST}"
    LOG="output/sensitivity_split_ratio/${LABEL}.log"

    echo "Launching ${LABEL} on physical GPU ${GPU}"
    echo "Log: ${LOG}"

    CUDA_VISIBLE_DEVICES="${GPU}" \
    PYTHONUNBUFFERED=1 \
    python -u factorial/factorial_split_ratio_HDGM_d10_cell.py \
        --n-train "${TRAIN}" \
        --n-test "${TEST}" \
        > "${LOG}" 2>&1 &

    PIDS+=("$!")
}

# Existing 300/700 result is preserved; run only the four missing cells.
run_cell 400 600 0
run_cell 500 500 1
run_cell 600 400 2
run_cell 700 300 3

echo
echo "PIDs: ${PIDS[*]}"
echo "All four cells launched."
echo

FAILED=0

for PID in "${PIDS[@]}"; do
    if ! wait "${PID}"; then
        echo "WARNING: process ${PID} failed."
        FAILED=1
    fi
done

echo
echo "All child processes finished at $(date)."

if [ "${FAILED}" -eq 0 ]; then
    echo
    echo "Merging cells..."
    python factorial/merge_split_ratio_HDGM_d10.py \
        | tee output/sensitivity_split_ratio/merged_summary.log

    echo
    echo "===== FINAL JSON ====="
    cat result/factorial_split_ratio_HDGM_d10_N4000.json
else
    echo
    echo "At least one split failed."
    echo "No merge attempted."
    echo "Inspect output/sensitivity_split_ratio/*.log"
fi

echo
echo "===== FINAL GIT STATUS ====="
git status --short
echo
echo "Overnight split-ratio runner finished."
