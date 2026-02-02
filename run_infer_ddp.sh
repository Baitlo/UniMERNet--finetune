#!/bin/bash
# UniMERNet DDP Inference Script
# 
# This script runs distributed inference on CMER_BENCH dataset
#
# Usage examples:
#   # Single GPU
#   ./run_infer_ddp.sh 1
#
#   # 8 GPUs
#   ./run_infer_ddp.sh 8
#
#   # Custom checkpoint path
#   ./run_infer_ddp.sh 8 /path/to/checkpoint.pth

NUM_GPUS=8
CHECKPOINT_PATH="/home/ubuntu/baiweikang/diskdata/Models/UniMERNet/weights_1/20260130162/checkpoint_latest.pth"

# Change to script directory
cd "$(dirname "$0")"

# Configuration file
CONFIG_FILE="configs/infer_ddp.yaml"

# Output file
OUTPUT_FILE="infer_results/para_result_2.jsonl"

# Data root
DATA_ROOT="/home/ubuntu/bigdiskdata/baiweikang/CMER_BENCH_1_0_for_unimer"

# Build command
if [ "$NUM_GPUS" -gt 1 ]; then
    CMD="torchrun --nproc_per_node=$NUM_GPUS infer_ddp.py"
else
    CMD="python infer_ddp.py"
fi

CMD="$CMD --cfg-path $CONFIG_FILE"
CMD="$CMD --data-root $DATA_ROOT"
CMD="$CMD --output $OUTPUT_FILE"
CMD="$CMD --batch-size 16"
CMD="$CMD --num-workers 8"

# Add custom checkpoint path if provided
if [ -n "$CHECKPOINT_PATH" ]; then
    CMD="$CMD --options model.finetuned=$CHECKPOINT_PATH"
fi

echo "Running command: $CMD"
echo "===================="

$CMD

echo "===================="
echo "Results saved to: $OUTPUT_FILE"
