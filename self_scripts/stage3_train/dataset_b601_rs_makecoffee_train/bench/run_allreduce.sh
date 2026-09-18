#!/bin/bash
# 裸测 allreduce：本模型梯度体积(196.8MB)在本机拓扑上的耗时
source ~/anaconda3/etc/profile.d/conda.sh && conda activate lerobot
D=$(dirname "$(realpath "$0")")
for N in 1 2 4 8; do
  CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((N-1))) \
    torchrun --nproc_per_node=$N --master_port=$((29500+N)) "$D/allreduce_bench.py" 2>&1 | grep RESULT
done
