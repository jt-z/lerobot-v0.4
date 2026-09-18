#!/bin/bash
# 启动训练 v2：单臂 seeed_b601_rs_follower 的 ACT 模型（make coffee 任务）
#
# 与 v1（start_train_act.sh）的关系
# --------------------------------
# v2 的超参与 v1 **逐字一致**，只改了两处：
#   - output_dir / job_name（v1 的目录已有 21 个 checkpoint，不覆盖，否则 validate() 会抛
#     FileExistsError）
#   - 数据集从 180 ep / 453,139 帧 增长到 223 ep / 533,168 帧（源端 9-18 推了两批）
# 目的：一次只动「数据」这一个变量，loss 曲线可与 v1 直接比。
#
# cosine lr 衰减这次**不加** —— 笔记里那条建议的前提是「真机效果不行」，而 v1 真机验证过
# 可用。真要试时用 --config act_train_config_v2_cosine.json（已单独备好并验证过）。
#
# 为什么要加前置校验门禁
# ----------------------
# 本数据集的源端在录制 ep161 时崩过一次，残留的 writer 缓冲让
# `data/chunk-000/file-009.parquet` 末尾多出 11,005 行废数据。这会让「物理行号 == index」
# 失效 → ep162 之后约 21% 的帧 state/action 取到错误 episode 的行（图文错配）。
#
# 关键在于**它是静默的**：9-16 那次数据小、越界，step 0 就崩；9-18 数据变大后不越界，
# 训练全程无报错、loss 曲线正常，学的却是错的。所以**不能靠「训练没崩」判断数据正常**。
# 详见 datastet_notes/DISTILLED.md。
#
# 而且这个损坏会在**每次源端重传时复现**（源端一直没修，重传会把文件覆盖回损坏版）。
# 因此：开训前必须自己验一遍，验不过就不开训。
#
# 前置检查（任一不过就拒绝开训，退出码 1/2）
# --------------------------------------------
#   1. 数据集完整性  总行数 == meta total_frames（+ index 无重复 + episode 边界自洽 + 视频不截断）
#   2. 传输静默期    数据集在 QUIET_MINS 分钟内没被写过，否则可能还在传
#
# 训练后复查（重要）
# ------------------
# bench/post_sync_check.sh 的守卫 3 是「有 lerobot-train 在跑就跳过」。也就是说：
# **训练期间源端若重传，file-009 会被改回损坏版，而钩子那时不会去修** —— 数据在你训练
# 到一半时静默变坏。所以训练结束后这里会再验一次，行数变了就明确告诉你，
# 免得拿一个「训到一半数据被换过」的 checkpoint 去上真机。
#
# 用法
# ----
#   cd /home/ksa/lerobot/self_scripts
#   bash stage3_train/dataset_b601_rs_makecoffee_train/start_train_act_v2.sh
#
#   ... --config act_train_config_v2_cosine.json   # 换配置（相对脚本目录）
#   ... --resume                                   # 从本配置自己的 last checkpoint 续训
#   ... --skip-data-check                          # 跳过前置校验（不推荐）
#   ... --dry-run                                  # 只跑前置检查 + 打印将执行的命令
#
# 可调环境变量
#   DS_ROOT      覆盖数据集根目录（默认从配置里读）
#   QUIET_MINS   传输静默期分钟数（默认 5）
#
# 退出码：0=训练正常结束  1=数据校验未通过  2=传输中/被守卫拦截  3=环境或用法错误

set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "$(realpath "$0")")" && pwd)
SELF_SCRIPTS_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
VERIFY="$SCRIPT_DIR/bench/verify_dataset.py"

CONFIG_NAME="act_train_config_v2.json"
QUIET_MINS=${QUIET_MINS:-5}
RESUME=0; SKIP_CHECK=0; DRY_RUN=0

while [ $# -gt 0 ]; do
    case "$1" in
        --config)          CONFIG_NAME="$2"; shift 2 ;;
        --config=*)        CONFIG_NAME="${1#*=}"; shift ;;
        --resume)          RESUME=1; shift ;;
        --skip-data-check) SKIP_CHECK=1; shift ;;
        --dry-run)         DRY_RUN=1; shift ;;
        -h|--help)         sed -n '2,50p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "未知参数: $1（-h 看用法）" >&2; exit 3 ;;
    esac
done

# 相对路径按脚本目录解析
case "$CONFIG_NAME" in
    /*) CONFIG="$CONFIG_NAME" ;;
    *)  CONFIG="$SCRIPT_DIR/$CONFIG_NAME" ;;
esac

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
        echo "（或确认源端已修好，见 datastet_notes/05-预防与治本方案.md）" >&2
        rm -f "$VJSON"
        EXIT_CODE=1 die "前置校验失败"
    fi

    ROWS_BEFORE=$("$PY" - "$VJSON" <<'PY'
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
    # 最后一行是回传的行数，前面是给人看的报告
    ROWS_SAVED=$(printf '%s\n' "$ROWS_BEFORE" | tail -n 1)
    printf '%s\n' "$ROWS_BEFORE" | sed '$d'
    ROWS_BEFORE="$ROWS_SAVED"
    rm -f "$VJSON"
fi

# ---------------- 前置检查 2：传输静默期 ----------------
# 源端是按需增量推送的，没有固定「结束」事件。数据集还在被写就开训，
# dataloader 会读到半成品 —— 这里用静默期当「传输已结束」的近似信号。
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

step "启动训练"
if [ "$RESUME" = 1 ] && [ -f "$CKPT_CFG" ]; then
    echo "从 checkpoint 续训：$CKPT_CFG"
    LAUNCH=(accelerate launch --num_processes=8 "$(which lerobot-train)"
            --config_path="$CKPT_CFG" --resume=true)
elif [ "$RESUME" = 1 ]; then
    die "--resume 指定了，但找不到 checkpoint：$CKPT_CFG"
else
    if [ -d "$SELF_SCRIPTS_DIR/$OUT_DIR" ]; then
        EXIT_CODE=3 die "输出目录已存在：$OUT_DIR
   换个配置（改 output_dir），或用 --resume 续训。"
    fi
    echo "从头训练（新 run，输出到 $OUT_DIR）"
    LAUNCH=(accelerate launch --num_processes=8 "$(which lerobot-train)"
            --config_path="$CONFIG")
fi

echo "  ${LAUNCH[*]}"
echo

if [ "$DRY_RUN" = 1 ]; then
    echo "（--dry-run：前置检查已通过，不实际启动训练）"
    exit 0
fi

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "${LAUNCH[@]}"
TRAIN_RC=$?

# ---------------- 训练后复查：数据在训练期间被改过吗 ----------------
# 见文件头「训练后复查」。行数对不上就说明源端在训练中途重传了，正在跑的这份 checkpoint
# 学到的东西不可信 —— 必须明确报出来，不能让它悄悄过去。
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
