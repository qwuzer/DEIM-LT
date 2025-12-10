#!/bin/bash
# NCCL environment variables for WSL2 compatibility
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

CUDA_VISIBLE_DEVICES=0 torchrun --master_port=7777 --nproc_per_node=1 train.py -c configs/deim_dfine/deim_hgnetv2_n_coco_lt.yml --seed=0