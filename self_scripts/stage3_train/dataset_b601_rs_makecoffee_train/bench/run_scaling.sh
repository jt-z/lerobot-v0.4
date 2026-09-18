#!/bin/bash
# 扩展性 sweep：固定 per-GPU batch=8，N=1/2/4/8
# 用法: bash run_scaling.sh [steps]
source ~/anaconda3/etc/profile.d/conda.sh && conda activate lerobot
export HF_ENDPOINT=https://hf-mirror.com
cd /home/ksa/lerobot/self_scripts
D=$(dirname "$(realpath "$0")")
STEPS=${1:-600}
OUT=${BENCH_OUT:-/tmp/act_bench}; mkdir -p "$OUT"
for N in 1 2 4 8; do
  python3 "$D/mkcfg.py" $N 8 $STEPS >/dev/null
  echo "########## N=$N bs=8 steps=$STEPS ##########"
  CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((N-1))) accelerate launch --num_processes=$N \
    $(which lerobot-train) --config_path="$OUT/n${N}_bs8.json" > "$OUT/log_n${N}_bs8.txt" 2>&1
  rm -rf "$OUT/out_n${N}_bs8"
done
echo "SCALING SWEEP DONE"
