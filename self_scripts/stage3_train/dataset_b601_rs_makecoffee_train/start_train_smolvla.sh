#!/bin/bash
# 启动训练：单臂 seeed_b601_rs_follower 的 SmolVLA（make coffee 任务）
#
# 与同目录 start_train_act_v2.sh 的关系
# ------------------------------------
# 数据集、目标、步数全部相同（223 ep / 533,168 帧 / 100K 步 = 12.0 pass），
# 只是把策略从 ACT 换成 SmolVLA —— 两者在同一数据集、同等训练量下可比。
#
#   ACT v2     : 100K 步，bs8×8卡 = 12.0 pass，实测 2.83 it/s → 约 9.8 h
#   SmolVLA 本 : 100K 步，同 12.0 pass，实测 0.97 it/s → 约 28.5 h（慢 2.9 倍）
#
# 与 ACT 的一个重要区别：**SmolVLA 不需要改 lerobot 源码**。
# ACT 的 get_scheduler_preset() 恒返回 None（要靠本地补丁才能加 lr 衰减），
# 而 SmolVLA 自带余弦衰减 preset，直接调参即可。
#
# ⚠️ scheduler_decay_steps 必须显式设为 steps
# ------------------------------------------
# SmolVLA 的默认值是 30000。若 steps=100000 而不改它，余弦在 30K 步就走完了，
# 剩下 70K 步 lr 会**一直停在 decay_lr = 2.5e-6**（峰值的 1/20）—— 等于白烧 70K 步。
# 实测对比：
#   decay_steps=30000  → lr @30K/50K/75K/100K = 2.5e-6, 2.5e-6, 2.5e-6, 2.5e-6
#   decay_steps=100000 → lr @30K/50K/75K/100K = 4.0e-5, 2.6e-5, 9.5e-6, 2.5e-6
# 本配置已设为 100000。（lerobot 只在 steps < decay_steps 时自动缩放，变长不管。）
#
# 权重来源
# --------
# 用 --policy.pretrained_path 加载 lerobot/smolvla_base（本地 modelscope 缓存），
# 而不是 --policy.path。区别：pretrained_path 只加载权重，policy 的输入特征
# （本数据集 3 路相机 hand/front/top + 7 维 state/action）按数据集自动推断；
# --policy.path 会连 base 的配置一起加载（它期望 camera1/2/3，且维度不同）。
# base 只训了 camera1/2/3 三路，本数据集也正好 3 路，但**名字不同** ——
# 这一点沿用你 coffee_cup 那次跑通的做法（4 路、名字也不同，未加 rename_map/empty_cameras）。
#
# ⚠️ checkpoint 很大
# ----------------
# 单个 ckpt 1.5 GB（ACT 是 591MB）。save_freq=5000 → 20 个 + last ≈ **30 GB**。
# 磁盘只剩 121G，够但要留意别和别的 run 挤。
#
# 前置检查（任一不过就拒绝开训）
# ------------------------------
#   1. 数据集完整性  总行数 == meta total_frames（+ index 无重复 + episode 边界 + 视频不截断）
#   2. 传输静默期    数据集在 QUIET_MINS 分钟内没被写过
# 详见 datastet_notes/DISTILLED.md：这个损坏训练时不报错，会**静默学错**，
# 所以不能靠「训练没崩」判断数据正常。
#
# 训练后复查
# ----------
# 钩子的守卫 3 是「有训练在跑就跳过」→ 训练期间源端若重传，数据会被静默改坏而没人修。
# 所以跑完会再数一遍总行数，对不上就明确报出来。
#
# 用法
# ----
#   cd /home/ksa/lerobot/self_scripts
#   bash stage3_train/dataset_b601_rs_makecoffee_train/start_train_smolvla.sh
#
#   ... --resume              # 从本配置自己的 last checkpoint 续训
#   ... --skip-data-check     # 跳过前置校验（不推荐）
#   ... --dry-run             # 只跑前置检查 + 打印将执行的命令
#
# 可调环境变量
#   DS_ROOT      覆盖数据集根目录（默认从配置里读）
#   QUIET_MINS   传输静默期分钟数（默认 5）
#   SMOLVLA_BASE 预训练权重路径
#
# 退出码：0=训练正常结束  1=数据校验未通过  2=传输中/被守卫拦截  3=环境或用法错误

set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "$(realpath "$0")")" && pwd)
SELF_SCRIPTS_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
CONFIG="$SCRIPT_DIR/smolvla_train_config.json"
VERIFY="$SCRIPT_DIR/bench/verify_dataset.py"

SMOLVLA_BASE=${SMOLVLA_BASE:-/home/ksa/.cache/modelscope/hub/models/lerobot/smolvla_base}
QUIET_MINS=${QUIET_MINS:-5}
RESUME=0; SKIP_CHECK=0; DRY_RUN=0

while [ $# -gt 0 ]; do
    case "$1" in
        --resume)          RESUME=1; shift ;;
        --skip-data-check) SKIP_CHECK=1; shift ;;
        --dry-run)         DRY_RUN=1; shift ;;
        -h|--help)         sed -n '2,60p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "未知参数: $1（-h 看用法）" >&2; exit 3 ;;
    esac
done

die() { echo "✗ $*" >&2; exit "${EXIT_CODE:-3}"; }
step() { echo; echo "── $* ──"; }

# ---------------- 环境 ----------------
[ -f "$CONFIG" ] || die "找不到配置：$CONFIG"

if [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "$CONDA_DEFAULT_ENV" != "lerobot" ]; then
    echo "激活 lerobot conda 环境…"
    eval "$(conda shell.bash hook)" || die "conda 不可用"
    conda activate lerobot || die "conda activate lerobot 失败"
fi
PY=$(which python) || die "找不到 python"

# 使用 HF 国内镜像，避免 huggingface.co 网络不可达
export HF_ENDPOINT=https://hf-mirror.com

# output_dir 在配置里是相对路径（相对 self_scripts/），所以必须在这里跑
cd "$SELF_SCRIPTS_DIR" || die "cd $SELF_SCRIPTS_DIR 失败"

# ---------------- 预训练权重 ----------------
step "检查预训练权重"
[ -d "$SMOLVLA_BASE" ] || die "找不到 smolvla_base：$SMOLVLA_BASE
   （可用 SMOLVLA_BASE=/path/to/smolvla_base 指定）"
[ -f "$SMOLVLA_BASE/model.safetensors" ] || die "$SMOLVLA_BASE 里没有 model.safetensors"
[ -f "$SMOLVLA_BASE/config.json" ] || die "$SMOLVLA_BASE 里没有 config.json"
echo "  ✓ $SMOLVLA_BASE"

# ---------------- 前置检查 1：数据集完整性 ----------------
DS_ROOT=${DS_ROOT:-$("$PY" -c "import json;print(json.load(open('$CONFIG'))['dataset']['root'])")}
[ -n "$DS_ROOT" ] || die "无法从配置里解析 dataset.root"
[ -d "$DS_ROOT" ] || die "数据集目录不存在：$DS_ROOT"

ROWS_BEFORE=""
if [ "$SKIP_CHECK" = 1 ]; then
    echo "⚠ 已跳过数据校验（--skip-data-check）——数据若损坏将静默学错，风险自负"
else
    step "前置检查 1/2：数据集完整性"
    [ -f "$VERIFY" ] || die "找不到校验脚本：$VERIFY"

    VJSON=$(mktemp) || die "mktemp 失败"
    if ! "$PY" "$VERIFY" --root "$DS_ROOT" --json >"$VJSON" 2>&1; then
        echo "✗ 数据集校验未通过，拒绝开训。详细报告：" >&2
        echo >&2
        "$PY" "$VERIFY" --root "$DS_ROOT" >&2 || true
        echo >&2
        echo "修复：$PY $VERIFY --root $DS_ROOT --fix" >&2
        rm -f "$VJSON"
        EXIT_CODE=1 die "前置校验失败"
    fi

    REPORT=$("$PY" - "$VJSON" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(f"  总行数        : {d['total_rows']:,}")
print(f"  meta 应有行数 : {d['meta_total_frames']:,}")
print(f"  数据文件      : {d['n_files']} 个")
print(f"  重复行        : {d['n_bogus_rows']}  ← 0 才算正常")
print(f"  错位行        : {d['n_misaligned_rows']}")
print("  ✓ 全部不变量成立")
print(d["total_rows"])
PY
)
    ROWS_BEFORE=$(printf '%s\n' "$REPORT" | tail -n 1)
    printf '%s\n' "$REPORT" | sed '$d'
    rm -f "$VJSON"
fi

# ---------------- 前置检查 2：传输静默期 ----------------
step "前置检查 2/2：传输静默期（${QUIET_MINS} 分钟）"
if pgrep -x rsync >/dev/null 2>&1 || pgrep -x scp >/dev/null 2>&1; then
    EXIT_CODE=2 die "检测到 rsync/scp 正在运行，传输可能未结束。稍后重试，或 QUIET_MINS=0 强开"
fi
recent=$(find "$DS_ROOT" -newermt "-${QUIET_MINS} minutes" -type f 2>/dev/null | head -1)
if [ -n "$recent" ]; then
    EXIT_CODE=2 die "数据集在 ${QUIET_MINS} 分钟内有写入（$(basename "$recent")），传输可能未结束。
   等一会儿再跑，或 QUIET_MINS=0 跳过本检查。"
fi
echo "  ✓ ${QUIET_MINS} 分钟内无写入"

# ---------------- 启动 ----------------
OUT_DIR=$("$PY" -c "import json;print(json.load(open('$CONFIG'))['output_dir'])")
CKPT_CFG="$SELF_SCRIPTS_DIR/$OUT_DIR/checkpoints/last/pretrained_model/train_config.json"
# ⚠️ 日志**不能**放在 output_dir 里面。lerobot 的 TrainPipelineConfig.validate() 在
# 「output_dir 已存在且非 resume」时会直接抛 FileExistsError —— 而我们要写日志就得先
# mkdir，那个 mkdir 会先把 output_dir 建出来，于是紧接着 lerobot 自己就把自己拦下了。
# 所以日志放到 output_dir 的**同级**目录：output_lerobot_train/logs/<run名>.log
LOG_FILE="$SELF_SCRIPTS_DIR/$(dirname "$OUT_DIR")/logs/$(basename "$OUT_DIR").log"

step "启动训练"

if [ "$RESUME" = 1 ] && [ -f "$CKPT_CFG" ]; then
    echo "从 checkpoint 续训：$CKPT_CFG"
    echo "（续训用 checkpoint 里的配置，pretrained_path 已存在里面，不用再传）"
    LAUNCH=(accelerate launch --multi_gpu --num_processes=8 "$(which lerobot-train)"
            --config_path="$CKPT_CFG" --resume=true)
elif [ "$RESUME" = 1 ]; then
    die "--resume 指定了，但找不到 checkpoint：$CKPT_CFG"
else
    if [ -d "$SELF_SCRIPTS_DIR/$OUT_DIR" ]; then
        EXIT_CODE=3 die "输出目录已存在：$OUT_DIR
   换个配置（改 output_dir），或用 --resume 续训。"
    fi
    echo "从头训练（新 run，输出到 $OUT_DIR）"
    echo "日志：$LOG_FILE"
    LAUNCH=(accelerate launch --multi_gpu --num_processes=8 "$(which lerobot-train)"
            --policy.pretrained_path="$SMOLVLA_BASE"
            --config_path="$CONFIG"
            --policy.push_to_hub=false)
fi

echo "  ${LAUNCH[*]}"
echo

if [ "$DRY_RUN" = 1 ]; then
    echo "（--dry-run：前置检查已通过，不实际启动训练）"
    exit 0
fi

# 日志目录在这里才创建，有两层原因：
#   1. 必须在「输出目录是否存在」的判断之后 —— 否则 mkdir 会先把 output_dir 建出来，
#      紧接着的存在性检查就会把自己的创建动作当成冲突而误报。
#   2. 必须在 DRY_RUN 提前退出之后 —— 否则跑一次 --dry-run 就留下 output_dir，
#      真正开训时反而被「目录已存在」挡住。
mkdir -p "$(dirname "$LOG_FILE")" || die "无法创建日志目录"

# 训练输出同时落日志。用 PIPESTATUS[0] 取 lerobot-train 的退出码（而非 tee 的）
set -o pipefail
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "${LAUNCH[@]}" 2>&1 | tee -a "$LOG_FILE"
TRAIN_RC=${PIPESTATUS[0]}

# ---------------- 训练后复查：数据在训练期间被改过吗 ----------------
if [ -n "$ROWS_BEFORE" ]; then
    step "训练后复查：数据集是否在训练期间被改动"
    ROWS_AFTER=$("$PY" - "$DS_ROOT" <<'PY' 2>/dev/null
import glob, os, sys
import pyarrow.parquet as pq
files = sorted(glob.glob(os.path.join(sys.argv[1], "data/chunk-*/*.parquet")))
print(sum(pq.ParquetFile(f).metadata.num_rows for f in files))
PY
)
    if [ -z "$ROWS_AFTER" ]; then
        echo "  （无法复查，跳过）"
    elif [ "$ROWS_AFTER" = "$ROWS_BEFORE" ]; then
        echo "  ✓ 行数未变（$ROWS_BEFORE），训练期间数据没被改动"
    else
        echo "  ✗✗ 数据在训练期间被改动：$ROWS_BEFORE → $ROWS_AFTER 行"
        echo "    源端大概率在训练中途重传，把 file-009 覆盖回了损坏版（钩子的守卫 3"
        echo "    在训练期间会跳过修复）。这份 checkpoint 学到的是错配的数据，别直接上真机。"
        echo "    现在先跑：$PY $VERIFY --root $DS_ROOT --fix"
    fi
fi

exit "$TRAIN_RC"
