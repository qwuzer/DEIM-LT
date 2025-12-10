# Persistent Training Guide

This guide explains how to run training that persists even when your SSH connection is lost.

## Quick Start

### Option 1: Using Screen (Recommended)
```bash
# Start training with default COCO-LT config
./train_screen.sh

# Start training with custom config
./train_screen.sh configs/deim_dfine/deim_hgnetv2_s_lvis.yml

# Start training with custom session name
./train_screen.sh configs/deim_dfine/deim_hgnetv2_s_coco_lt.yml my_training
```

**Screen Commands:**
- Attach to session: `screen -r deim_training`
- Detach from session: Press `Ctrl+A` then `D`
- List all sessions: `screen -ls`
- Kill session: `screen -S deim_training -X quit`

### Option 2: Using Tmux
```bash
# Start training
./train_tmux.sh

# With custom config
./train_tmux.sh configs/deim_dfine/deim_hgnetv2_s_lvis.yml
```

**Tmux Commands:**
- Attach to session: `tmux attach -t deim_training`
- Detach from session: Press `Ctrl+B` then `D`
- List all sessions: `tmux ls`
- Kill session: `tmux kill-session -t deim_training`

### Option 3: Using Nohup (Simplest)
```bash
# Start training in background
./train_nohup.sh

# With custom config
./train_nohup.sh configs/deim_dfine/deim_hgnetv2_s_lvis.yml
```

**Nohup Commands:**
- View logs: `tail -f outputs/deim_hgnetv2_s_coco_lt/training.log`
- Check if running: `ps -p $(cat outputs/deim_hgnetv2_s_coco_lt/training.pid)`
- Stop training: `kill $(cat outputs/deim_hgnetv2_s_coco_lt/training.pid)`

## Comparison

| Method | Pros | Cons |
|--------|------|------|
| **Screen** | Easy to attach/detach, see live output | Requires learning screen commands |
| **Tmux** | Modern, powerful, split panes | Slightly more complex than screen |
| **Nohup** | Simplest, no learning curve | Can't see live output easily |

## Monitoring Training

### View Logs
```bash
# Real-time log viewing
tail -f outputs/deim_hgnetv2_s_coco_lt/training.log

# Last 100 lines
tail -n 100 outputs/deim_hgnetv2_s_coco_lt/training.log

# Search for errors
grep -i error outputs/deim_hgnetv2_s_coco_lt/training.log
```

### Check GPU Usage
```bash
# Continuous monitoring
watch -n 1 nvidia-smi

# One-time check
nvidia-smi
```

### Check Training Status
```bash
# For screen/tmux
screen -ls    # or tmux ls

# For nohup
ps -p $(cat outputs/deim_hgnetv2_s_coco_lt/training.pid)
```

## Resuming Training

If training was interrupted, you can resume from a checkpoint:

```bash
# Edit train_screen.sh (or train_tmux.sh) and add -r flag to train.py
# Example:
# train.py -c configs/deim_dfine/deim_hgnetv2_s_coco_lt.yml --seed=0 -r outputs/deim_hgnetv2_s_coco_lt/last.pth
```

## Troubleshooting

### Session Already Exists
```bash
# Kill existing screen session
screen -S deim_training -X quit

# Kill existing tmux session
tmux kill-session -t deim_training
```

### Can't Attach to Session
```bash
# Force attach (if session is attached elsewhere)
screen -r -d deim_training
# or
tmux attach -d -t deim_training
```

### Training Not Starting
- Check if GPU is available: `nvidia-smi`
- Check if port 7777 is in use: `lsof -i :7777`
- Check logs for errors: `tail -n 50 outputs/deim_hgnetv2_s_coco_lt/training.log`

## Best Practices

1. **Always use persistent sessions** for long training jobs
2. **Check logs regularly** to catch errors early
3. **Monitor GPU usage** to ensure training is progressing
4. **Save checkpoints frequently** (configured in your config file)
5. **Use descriptive session names** when running multiple experiments

