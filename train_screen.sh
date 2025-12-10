#!/bin/bash
# Training script with screen session for persistent SSH connections
# Usage: ./train_screen.sh [config_file] [session_name]

CONFIG=${1:-"configs/deim_dfine/deim_hgnetv2_n_coco_lt.yml"}  # Changed from _s_ to _n_
SESSION_NAME=${2:-"deim_training"}
LOG_DIR="./outputs/$(basename $CONFIG .yml)"
LOG_FILE="${LOG_DIR}/training.log"

# Create log directory
mkdir -p "$LOG_DIR"

# Fix screen directory permission issue (WSL2 compatibility)
export SCREENDIR="$HOME/.screen"
mkdir -p "$SCREENDIR"
chmod 700 "$SCREENDIR" 2>/dev/null || true

# Kill any existing screen session with same name
screen -S "$SESSION_NAME" -X quit 2>/dev/null || true

# NCCL environment variables for WSL2 compatibility
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Start training in a screen session
SCREENDIR="$HOME/.screen" screen -S "$SESSION_NAME" -dm bash -c "
cd \"$SCRIPT_DIR\"
export SCREENDIR=\"\$HOME/.screen\"
echo '========================================'
echo 'Training started at \$(date)'
echo \"Config: $CONFIG\"
echo \"Session: $SESSION_NAME\"
echo \"Log file: $LOG_FILE\"
echo '========================================'
echo ''

# Run training and log output
CUDA_VISIBLE_DEVICES=0 torchrun --master_port=7777 --nproc_per_node=1 train.py -c $CONFIG --seed=0 2>&1 | tee $LOG_FILE

echo ''
echo '========================================'
echo 'Training completed at \$(date)'
echo '========================================'
exec bash
"

echo "✓ Training started in screen session '$SESSION_NAME'"
echo ""
echo "Useful commands:"
echo "  Attach to session:    screen -r $SESSION_NAME"
echo "  Detach from session:   Press Ctrl+A then D"
echo "  List all sessions:     screen -ls"
echo "  Kill session:          screen -S $SESSION_NAME -X quit"
echo ""
echo "View logs in real-time:"
echo "  tail -f $LOG_FILE"
echo ""
echo "View logs with timestamps:"
echo "  tail -f $LOG_FILE | while read line; do echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] \$line\"; done"

