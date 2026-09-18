#!/bin/bash
# 检查 P2P 能力与 NCCL 实际走的传输路径
nvidia-smi topo -p2p r
nvidia-smi nvlink -s | head -4
source ~/anaconda3/etc/profile.d/conda.sh && conda activate lerobot
D=$(dirname "$(realpath "$0")")
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NCCL_DEBUG=INFO \
  torchrun --nproc_per_node=8 --master_port=29599 "$D/allreduce_bench.py" 2>&1 \
  | grep -E "via |P2P is disabled|RESULT" | sort -u
