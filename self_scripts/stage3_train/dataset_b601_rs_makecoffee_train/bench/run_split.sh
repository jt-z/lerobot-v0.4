#!/bin/bash
# 把 8 张卡拆成 8/G 个「独立的 G 卡任务」并发跑。
#
# 用途：
#   1) 跑多个独立实验（超参扫描 / 多任务）—— 效率远高于 8 卡 DDP（见
#      multigpu_scaling_analysis.md §5.4 / §8.4）
#   2) 复现组拆分 benchmark 数据
#
# 用法:
#   bash run_split.sh <组大小 G> [总步数] [标签]
#
#   bash run_split.sh 1 600    # 8 组单卡   -> 合计 335.2 samples/s（效率 97.2%）
#   bash run_split.sh 2 600    # 4 组双卡   -> 合计 279.8 samples/s（效率 81.1%）
#   bash run_split.sh 4 600    # 2 组四卡   -> 合计 232.0 samples/s（效率 67.3%）
#   bash run_split.sh 8 600    # 1 组八卡   -> 合计 228.8 samples/s（效率 66.4%，即现状 DDP）
#
# 输出：日志写到 $BENCH_OUT（默认 /tmp/act_bench）下的 log_<标签>_g<i>.txt
#       读数请看 ot_train.py:444 行的 updt_s / data_s，丢弃前 ~100 步热机
#
# ⚠️ 注意：G<8 时是「多个不同的模型」，不是「一个模型用多卡」。
#    每组全局 batch = G×8，8×1 时只有 8（而非 64），优化动态不同，lr 需另行确定。

set -u
source ~/anaconda3/etc/profile.d/conda.sh && conda activate lerobot
export HF_ENDPOINT=https://hf-mirror.com
cd /home/ksa/lerobot/self_scripts

G=${1:-1}
STEPS=${2:-600}
TAG=${3:-split${G}}
OUT=${BENCH_OUT:-/tmp/act_bench}
NG=$((8 / G))

if [ $((8 % G)) -ne 0 ]; then
    echo "错误：组大小 G 必须整除 8（收到 G=$G）" >&2
    exit 1
fi

mkdir -p "$OUT"

# 生成 NG 份配置，只有 output_dir / job_name 不同
python3 - "$NG" "$G" "$STEPS" "$OUT" "$TAG" <<'PY'
import json, sys
NG, G, STEPS, OUT, TAG = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
base = {
    "dataset": {
        "repo_id": "hellozjt/b601_20260910_164106",
        "root": "/data/share/b601_20260910_164106",
        "revision": "main",
        "streaming": False,
    },
    "policy": {"type": "act", "device": "cuda", "push_to_hub": False, "optimizer_lr": 8e-5},
    "steps": STEPS,
    "eval_freq": 0,
    "batch_size": 8,          # per-GPU，与 scaling sweep 保持一致
    "num_workers": 6,
    "log_freq": 50,
    "save_freq": 10**9,
    "save_checkpoint": False,  # 不落盘，纯测吞吐
    "seed": 1000,
}
for i in range(NG):
    c = dict(base)
    c["output_dir"] = f"{OUT}/out_{TAG}_g{i}"
    c["job_name"] = f"{TAG}_g{i}"
    json.dump(c, open(f"{OUT}/{TAG}_g{i}.json", "w"), indent=2)
print(f"gen {NG} configs: G={G}, steps={STEPS}, tag={TAG}")
PY

echo "启动 $NG 组 × $G 卡（合计 $((NG * G)) 卡）"
for i in $(seq 0 $((NG - 1))); do
    GPU_LIST=$(seq -s, $((i * G)) $((i * G + G - 1)))
    CUDA_VISIBLE_DEVICES=$GPU_LIST accelerate launch --num_processes=$G \
        --main_process_port=$((29800 + i)) "$(which lerobot-train)" \
        --config_path="$OUT/${TAG}_g$i.json" > "$OUT/log_${TAG}_g$i.txt" 2>&1 &
done

wait
for i in $(seq 0 $((NG - 1))); do
    rm -rf "$OUT/out_${TAG}_g$i"
done
echo "$TAG DONE"
