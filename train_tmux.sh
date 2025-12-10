#!/bin/bash
# Training script with tmux session for persistent SSH connections
# Usage: ./train_tmux.sh [config_file] [session_name]

CONFIG=${1:-"configs/deim_dfine/deim_hgnetv2_s_coco_lt.yml"}
SESSION_NAME=${2:-"deim_training"}
LOG_DIR="./outputs/$(basename $CONFIG .yml)"
LOG_FILE="${LOG_DIR}/training.log"

# Create log directory
mkdir -p "$LOG_DIR"

# Set tmux socket directory (WSL2 compatibility)
export TMUX_TMPDIR="$HOME/.tmux"
mkdir -p "$TMUX_TMPDIR"

# Kill any existing tmux session with same name
tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true

# NCCL environment variables for WSL2 compatibility
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Start training in a tmux session
tmux new-session -d -s "$SESSION_NAME" -c "$SCRIPT_DIR" "
echo '========================================'
echo 'Training started at $(date)'
echo "Config: $CONFIG"
echo "Session: $SESSION_NAME"
echo "Log file: $LOG_FILE"
echo '========================================'
echo ''

# Run training and log output
CUDA_VISIBLE_DEVICES=0 torchrun --master_port=7777 --nproc_per_node=1 train.py -c $CONFIG --seed=0 2>&1 | tee $LOG_FILE

echo ''
echo '========================================'
echo 'Training completed at $(date)'
echo '========================================'
bash
"

echo "✓ Training started in tmux session '$SESSION_NAME'"
echo ""
echo "Useful commands:"
echo "  Attach to session:    tmux attach -t $SESSION_NAME"
echo "  Detach from session:  Press Ctrl+B then D"
echo "  List all sessions:   tmux ls"
echo "  Kill session:        tmux kill-session -t $SESSION_NAME"
echo ""
echo "View logs in real-time:"
echo "  tail -f $LOG_FILE"
echo ""
echo "View logs with timestamps:"
echo "  tail -f $LOG_FILE | while read line; do echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] \$line\"; done"

