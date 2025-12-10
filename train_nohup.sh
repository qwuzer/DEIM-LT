#!/bin/bash
# Training script with nohup for persistent SSH connections (alternative to screen/tmux)
# Usage: ./train_nohup.sh [config_file]

CONFIG=${1:-"configs/deim_dfine/deim_hgnetv2_s_coco_lt.yml"}
LOG_DIR="./outputs/$(basename $CONFIG .yml)"
LOG_FILE="${LOG_DIR}/training.log"
PID_FILE="${LOG_DIR}/training.pid"

# Create log directory
mkdir -p "$LOG_DIR"

# NCCL environment variables for WSL2 compatibility
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

# Check if training is already running
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "⚠ Training is already running (PID: $OLD_PID)"
        echo "   Kill it first: kill $OLD_PID"
        exit 1
    else
        echo "Removing stale PID file..."
        rm -f "$PID_FILE"
    fi
fi

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Start training with nohup
cd "$SCRIPT_DIR"
nohup bash -c "
CUDA_VISIBLE_DEVICES=0 torchrun --master_port=7777 --nproc_per_node=1 train.py -c $CONFIG --seed=0
" > "$LOG_FILE" 2>&1 &

TRAIN_PID=$!
echo $TRAIN_PID > "$PID_FILE"

echo "✓ Training started in background (PID: $TRAIN_PID)"
echo "  Config: $CONFIG"
echo "  Log file: $LOG_FILE"
echo "  PID file: $PID_FILE"
echo ""
echo "Useful commands:"
echo "  View logs:            tail -f $LOG_FILE"
echo "  Check if running:    ps -p $TRAIN_PID"
echo "  Stop training:       kill $TRAIN_PID"
echo "  View GPU usage:       watch -n 1 nvidia-smi"

