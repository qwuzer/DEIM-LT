# DEIM-LT: Long-Tailed Object Detection Training Guide

This guide provides comprehensive instructions for setting up datasets, configuring training, and running experiments for the DEIM-LT long-tailed object detection framework.

## Quick Start

**For experienced users who just need the commands:**

```bash
# 1. Setup dataset (see Dataset Setup section for details)
# 2. Start training with persistent session
./train_screen.sh configs/deim_dfine/deim_hgnetv2_s_coco_lt.yml

# 3. Attach to session to see progress
screen -r deim_training

# 4. View logs
tail -f outputs/deim_hgnetv2_s_coco_lt/training.log
```

**For detailed instructions, continue reading below.**

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Dataset Setup](#dataset-setup)
   - [LVIS Dataset](#lvis-dataset)
   - [COCO-LT Dataset](#coco-lt-dataset)
4. [Configuration Files](#configuration-files)
   - [Dataset Configs](#dataset-configs)
   - [Training Configs](#training-configs)
   - [Config Hierarchy](#config-hierarchy)
5. [Running Training](#running-training)
   - [Basic Training](#basic-training)
   - [Persistent Training Sessions](#persistent-training-sessions)
   - [Resuming Training](#resuming-training)
6. [Monitoring Training](#monitoring-training)
7. [Troubleshooting](#troubleshooting)
8. [Common Issues and Solutions](#common-issues-and-solutions)

---

## Overview

DEIM-LT is a framework for training object detection models on long-tailed datasets. It supports:
- **LVIS**: Large Vocabulary Instance Segmentation (1203 classes)
- **COCO-LT**: Long-tailed version of COCO (80 classes)
- Multiple model sizes: Small (S), Medium (M), Large (L), XLarge (X), Nano (N)

---

## Prerequisites

### Path Placeholders

Throughout this README, we use the following placeholders:
- `{DATA_DIR}`: Your dataset directory (e.g., `/data`, `/home/user/data`, `/mnt/datasets`)
- `{PROJECT_ROOT}`: Path to the DEIM-LT project directory (where you cloned the repository)

**Important**: Replace these placeholders with your actual paths when following the instructions.

### System Requirements
- Linux (tested on WSL2/Ubuntu)
- CUDA-capable GPU
- Python 3.8+
- PyTorch with CUDA support

### Environment Setup

```bash
# Activate your conda/virtual environment
conda activate deim  # or your environment name

# Install dependencies (if not already installed)
pip install -r requirements.txt
```

### Required Tools
- `screen` or `tmux` (for persistent training sessions)
- `gdown` (for downloading from Google Drive)

---

## Dataset Setup

### LVIS Dataset

#### Directory Structure
```
{DATA_DIR}/lvis/
├── lvis_v1_train/
│   ├── train2017/          # Training images (118,287 images)
│   └── lvis_v1_train_with_filenames.json
└── lvis_v1_val/
    ├── val2017/             # Validation images (15,000+ images)
    └── lvis_v1_val_with_filenames.json
```

**Note**: Replace `{DATA_DIR}` with your actual data directory path (e.g., `/data`, `/home/user/data`, etc.)

#### Setup Steps

1. **Download LVIS annotations**:
   ```bash
   # Download LVIS v1.0 annotations from the official LVIS website
   # Place them in: {DATA_DIR}/lvis/lvis_v1_train/ and {DATA_DIR}/lvis/lvis_v1_val/
   ```

2. **Download COCO images** (LVIS uses COCO images):
   ```bash
   # Download COCO train2017 and val2017 images from COCO website
   # Extract to appropriate directories
   ```

3. **Verify dataset structure**:
   ```bash
   # Check image counts
   ls {DATA_DIR}/lvis/lvis_v1_train/train2017/*.jpg | wc -l  # Should be ~118,287
   ls {DATA_DIR}/lvis/lvis_v1_val/val2017/*.jpg | wc -l      # Should be ~15,000
   
   # Verify annotation files exist
   ls -lh {DATA_DIR}/lvis/lvis_v1_train/lvis_v1_train_with_filenames.json
   ls -lh {DATA_DIR}/lvis/lvis_v1_val/lvis_v1_val_with_filenames.json
   ```

4. **Update paths in config**:
   Edit `configs/dataset/lvis_detection.yml` and replace the paths with your actual dataset locations.

---

### COCO-LT Dataset

#### Directory Structure
```
{DATA_DIR}/coco/
├── annotations/
│   ├── coco_lt_train.json          # COCO-LT training annotations
│   ├── instances_train2017.json    # Standard COCO training annotations
│   └── instances_val2017.json      # Standard COCO validation annotations
├── train2017/                      # Training images (118,287 images)
└── val2017/                        # Validation images (5,000 images)
```

**Note**: Replace `{DATA_DIR}` with your actual data directory path (e.g., `/data`, `/home/user/data`, etc.)

#### Setup Steps

1. **Create directory structure**:
   ```bash
   mkdir -p {DATA_DIR}/coco/{annotations,train2017,val2017}
   ```

2. **Download COCO-LT annotations**:
   ```bash
   cd {DATA_DIR}/coco/annotations
   
   # Install gdown if not available
   pip install gdown
   
   # Download COCO-LT training annotations
   gdown https://drive.google.com/uc?id=1cQM7BclPRjikWDhaUGoJNwQD3ia4StlQ -O coco_lt_train.json
   ```

3. **Download COCO images**:
   ```bash
   # Download COCO train2017 and val2017 images from COCO website
   # Extract to:
   # - {DATA_DIR}/coco/train2017/
   # - {DATA_DIR}/coco/val2017/
   ```

4. **Download COCO standard annotations** (for validation):
   ```bash
   cd {DATA_DIR}/coco/annotations
   
   # Download from COCO website:
   # - instances_train2017.json
   # - instances_val2017.json
   ```

5. **Verify dataset**:
   ```bash
   # Check image counts
   ls {DATA_DIR}/coco/train2017/*.jpg | wc -l  # Should be 118,287
   ls {DATA_DIR}/coco/val2017/*.jpg | wc -l     # Should be 5,000
   
   # Verify annotation files
   ls -lh {DATA_DIR}/coco/annotations/coco_lt_train.json
   ls -lh {DATA_DIR}/coco/annotations/instances_val2017.json
   ```

6. **Verify annotation format**:
   ```bash
   python3 << EOF
    import json

   # Check COCO-LT (replace {DATA_DIR} with your path)
   with open('{DATA_DIR}/coco/annotations/coco_lt_train.json', 'r') as f:
       data = json.load(f)
       print(f"COCO-LT: {len(data['categories'])} categories, {len(data['images'])} images, {len(data['annotations'])} annotations")
   
   # Check COCO val
   with open('{DATA_DIR}/coco/annotations/instances_val2017.json', 'r') as f:
       data = json.load(f)
       print(f"COCO Val: {len(data['categories'])} categories, {len(data['images'])} images")
   EOF
   ```

---

## Configuration Files

### Dataset Configs

Dataset configurations define the dataset paths, number of classes, and data loading settings.

**Location**: `configs/dataset/`

#### LVIS Detection Config (`lvis_detection.yml`)

   

#### COCO-LT Detection Config (`coco_lt_detection.yml`)

```yaml
    task: detection

    evaluator:
    type: CocoEvaluator
    iou_types: ['bbox', ]

    num_classes: 80  # COCO-LT has 80 classes
    remap_mscoco_category: False

        train_dataloader:
    type: DataLoader
    dataset:
        type: CocoDetection
        img_folder: {DATA_DIR}/coco/train2017
        ann_file: {DATA_DIR}/coco/annotations/coco_lt_train.json
        return_masks: False
    shuffle: True
    num_workers: 4
    drop_last: True

    val_dataloader:
    type: DataLoader
    dataset:
        type: CocoDetection
        img_folder: {DATA_DIR}/coco/val2017
        ann_file: {DATA_DIR}/coco/annotations/instances_val2017.json
        return_masks: False
    shuffle: False
    num_workers: 4
    drop_last: False
```

**Key Parameters**:
- `num_classes`: Number of classes (80 for COCO-LT)
- `remap_mscoco_category`: Set to `False` (dataset handles category ID mapping automatically)

---

### Training Configs

Training configurations define model architecture, optimizer, learning rate schedule, and training hyperparameters.

**Location**: `configs/deim_dfine/`

#### Example: COCO-LT Small Model (`deim_hgnetv2_s_coco_lt.yml`)

```yaml
    __include__: [
    './dfine_hgnetv2_s_coco.yml',
    '../base/deim.yml'
    ]

    # Modified runtime settings
    print_freq: 1          # Print training stats every N iterations
    checkpoint_freq: 1     # Save checkpoint every N epochs

    output_dir: ./outputs/deim_hgnetv2_s_coco_lt

        optimizer:
        type: AdamW
        params:
            -
        params: '^(?=.*backbone)(?!.*bn).*$'  # Backbone (non-BN layers)
        lr: 0.0002
            -
        params: '^(?=.*(?:norm|bn)).*$'       # Normalization layers
            weight_decay: 0.
    lr: 0.0004           # Main learning rate
        betas: [0.9, 0.999]
    weight_decay: 0.0001

    # Training epochs
    epoches: 132  # 120 + 4n format

    # Learning rate scheduler
    flat_epoch: 64    # Flat learning rate for first N epochs
    no_aug_epoch: 12  # No augmentation for last N epochs

    # Data augmentation
        train_dataloader:
        dataset:
            transforms:
        policy:
            epoch: [4, 64, 120]  # Augmentation policy epochs
        collate_fn:
        mixup_epochs: [4, 64]    # MixUp active during these epochs
        stop_epoch: 120          # Stop augmentation at this epoch
        total_batch_size: 8        # Total batch size (across all GPUs)

    val_dataloader:
        total_batch_size: 8        # Validation batch size
```

**Key Parameters**:
- `__include__`: Base configs to inherit from
- `output_dir`: Where to save checkpoints and logs
- `total_batch_size`: Total batch size (will be divided by number of GPUs)
- `epoches`: Total training epochs
- `print_freq`: How often to print training stats
- `checkpoint_freq`: How often to save checkpoints

#### Available Model Configs

| Config File | Model Size | Dataset | Classes |
|------------|------------|---------|---------|
| `deim_hgnetv2_s_lvis.yml` | Small | LVIS | 1203 |
| `deim_hgnetv2_s_coco_lt.yml` | Small | COCO-LT | 80 |
| `deim_hgnetv2_n_coco_lt.yml` | Nano | COCO-LT | 80 |
| `deim_hgnetv2_m_coco.yml` | Medium | COCO | 80 |
| `deim_hgnetv2_l_coco.yml` | Large | COCO | 80 |
| `deim_hgnetv2_x_coco.yml` | XLarge | COCO | 80 |

---

### Config Hierarchy

The configuration system uses YAML file inclusion with hierarchical structure:

```
deim_hgnetv2_s_coco_lt.yml
├── dfine_hgnetv2_s_coco.yml
│   ├── lvis_detection.yml (or coco_lt_detection.yml)
│   ├── runtime.yml
│   ├── base/dataloader.yml
│   ├── base/optimizer.yml
│   └── base/dfine_hgnetv2.yml
└── base/deim.yml
```

**Important**: Later files override earlier ones. Settings in your main config file override everything.

---

## Running Training

### Basic Training

#### Direct Training (for testing)

```bash
cd {PROJECT_ROOT}  # Navigate to DEIM-LT project directory

# Set NCCL environment variables (WSL2 compatibility)
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

# Run training
CUDA_VISIBLE_DEVICES=0 torchrun --master_port=7777 --nproc_per_node=1 \
    train.py -c configs/deim_dfine/deim_hgnetv2_s_coco_lt.yml --seed=0
```

#### Using the Training Script

```bash
cd {PROJECT_ROOT}
./train.sh
```

**Note**: This will stop if your SSH connection is lost. Use persistent sessions for long training jobs.

---

### Persistent Training Sessions

For long training jobs, use persistent sessions that survive SSH disconnections. 

**See [README_TRAINING.md](README_TRAINING.md) for detailed instructions** on using screen, tmux, or nohup for persistent training sessions.

Quick reference:
- **Screen**: `./train_screen.sh [config_file] [session_name]`
- **Tmux**: `./train_tmux.sh [config_file] [session_name]`
- **Nohup**: `./train_nohup.sh [config_file]`

---

### Resuming Training

To resume from a checkpoint:

1. **Edit the training script** or create a resume script:

```bash
# Example: Resume from last checkpoint
cd {PROJECT_ROOT}
CUDA_VISIBLE_DEVICES=0 torchrun --master_port=7777 --nproc_per_node=1 \
    train.py -c configs/deim_dfine/deim_hgnetv2_s_coco_lt.yml \
    --seed=0 \
    -r outputs/deim_hgnetv2_s_coco_lt/last.pth
```

2. **Or modify the config file** to include resume path:

```yaml
    resume: outputs/deim_hgnetv2_s_coco_lt/last.pth
```

3. **Checkpoint files**:
   - `last.pth`: Latest checkpoint
   - `epoch_*.pth`: Epoch-specific checkpoints (if `checkpoint_freq: 1`)

---

## Monitoring Training

### View Logs

```bash
# Real-time log viewing
tail -f outputs/deim_hgnetv2_s_coco_lt/training.log

# Last 100 lines
tail -n 100 outputs/deim_hgnetv2_s_coco_lt/training.log

# Search for errors
grep -i error outputs/deim_hgnetv2_s_coco_lt/training.log

# Search for specific metrics
grep "loss" outputs/deim_hgnetv2_s_coco_lt/training.log | tail -20
```

### Check GPU Usage

```bash
# Continuous monitoring
watch -n 1 nvidia-smi

# One-time check
nvidia-smi

# Check GPU memory usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

### Check Training Status

```bash
# For screen/tmux
screen -ls    # or tmux ls

# For nohup
ps -p $(cat outputs/deim_hgnetv2_s_coco_lt/training.pid)

# Check if process is using GPU
nvidia-smi | grep python
```

### Monitor Training Metrics

Training logs include:
- Loss values (classification, bbox, giou, etc.)
- Learning rate schedule
- Epoch progress
- Validation metrics (if validation is enabled)

---

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Symptoms**: `RuntimeError: CUDA out of memory`

**Solutions**:
- Reduce `total_batch_size` in config
- Reduce image size in transforms
- Use gradient accumulation (if supported)

```yaml
train_dataloader:
  total_batch_size: 4  # Reduce from 8
```

#### 2. Dataset Not Found

**Symptoms**: `FileNotFoundError: No such file or directory`

**Solutions**:
- Verify dataset paths in config files
- Check that images and annotations exist
- Ensure paths are absolute or relative to project root

```bash
# Verify paths (replace {DATA_DIR} with your actual path)
ls -lh {DATA_DIR}/coco/train2017 | head -5
ls -lh {DATA_DIR}/coco/annotations/coco_lt_train.json
```

#### 3. Class Index Mismatch

**Symptoms**: `CUDA error: device-side assert triggered` or `indexSelectLargeIndex: Assertion failed`

**Solutions**:
- Ensure `num_classes` matches your dataset
- Check `remap_mscoco_category` setting
- Verify annotation file format

```yaml
# For COCO-LT
num_classes: 80
remap_mscoco_category: False

# For LVIS
num_classes: 1203
remap_mscoco_category: False
```

#### 4. NCCL Errors (WSL2)

**Symptoms**: `NCCL error` or distributed training failures

**Solutions**:
- Use the provided NCCL environment variables
- Ensure single GPU training uses `--nproc_per_node=1`
- Check WSL2 network configuration

```bash
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
```

#### 5. Screen/Tmux Permission Errors

**Symptoms**: `Cannot make directory '/run/screen': Permission denied`

**Solutions**:
- The scripts automatically handle this by using `$HOME/.screen`
- If issues persist, manually set:

```bash
export SCREENDIR="$HOME/.screen"
mkdir -p "$SCREENDIR"
chmod 700 "$SCREENDIR"
```

---

## Common Issues and Solutions

### Issue: Training Stops When SSH Disconnects

**Solution**: Always use persistent sessions (`train_screen.sh`, `train_tmux.sh`, or `train_nohup.sh`)

### Issue: Can't Attach to Screen Session

**Solution**: 
```bash
# Force attach (if session is attached elsewhere)
screen -r -d deim_training
```

### Issue: Port Already in Use

**Symptoms**: `Address already in use` for port 7777

**Solution**:
```bash
# Use a different port
CUDA_VISIBLE_DEVICES=0 torchrun --master_port=7778 --nproc_per_node=1 train.py ...
```

### Issue: Config Merge Conflicts

**Symptoms**: `AssertionError: batch_size or total_batch_size should be choosed one`

**Solution**: Ensure only one batch size parameter is set:
```yaml
train_dataloader:
  total_batch_size: 8  # Use this, not batch_size
```

### Issue: Slow Data Loading

**Solution**:
- Increase `num_workers` (but not more than CPU cores)
- Use SSD storage for datasets
- Pre-load dataset into memory (if enough RAM)

```yaml
train_dataloader:
  num_workers: 8  # Increase if you have more CPU cores
```

---

## Quick Reference

### Training Commands

```bash
# Quick start (COCO-LT)
cd {PROJECT_ROOT}
./train_screen.sh

# LVIS training
./train_screen.sh configs/deim_dfine/deim_hgnetv2_s_lvis.yml

# Attach to session
screen -r deim_training

# View logs
tail -f outputs/deim_hgnetv2_s_coco_lt/training.log
```

**Note**: For detailed persistent training instructions, see [README_TRAINING.md](README_TRAINING.md)

### Config File Locations

- **Dataset configs**: `configs/dataset/`
- **Training configs**: `configs/deim_dfine/`
- **Base configs**: `configs/base/`

### Output Locations

- **Checkpoints**: `outputs/{config_name}/`
- **Logs**: `outputs/{config_name}/training.log`
- **TensorBoard logs**: `outputs/{config_name}/summary/` (if enabled)

---

## Additional Resources

- **Model Architecture**: See `engine/deim/` for model implementations
- **Data Processing**: See `engine/data/` for dataset and transform implementations
- **Training Loop**: See `engine/solver/` for training logic

---

## Support

For issues or questions:
1. Check this README and troubleshooting section
2. Review training logs for error messages
3. Verify dataset setup and config file paths
4. Check GPU availability and CUDA installation

---

**Last Updated**: December 2024
