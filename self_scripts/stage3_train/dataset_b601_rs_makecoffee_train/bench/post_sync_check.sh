#!/bin/bash
# 数据集同步后自动校验 + 修复钩子。
#
# 解决的问题
# ----------
# 源端只要再增量重传，`data/chunk-000/file-009.parquet` 就会被覆盖回损坏版本
# （9-16、9-18 已各复现一次，且两次逐字节相同）。不修的话训练不会崩，
# 而是**静默错配**：state/action 取到错误 episode 的行，视频仍按 meta 查 → 图文错配，
# loss 曲线看着正常，学的却是错的。
#
# 这个钩子把「检测 → 修复 → 复检 → 记日志」固化下来，挂到 cron / systemd timer 上，
# 每次同步后自动跑，不用再手工发现（9-18 那次就是因为手工做、发现得晚）。
#
# 三重守卫（任一命中就跳过，退出码 2，不做任何事）
# --------------------------------------------------
#   1. 已有另一个实例在跑          （flock 非阻塞）
#   2. 数据集在 QUIET_MINS 分钟内被写过  → 传输还没结束，绝不能动
#   3. 正在训练（lerobot-train 在跑）    → 避免改写 parquet 影响 dataloader
#
# 用法
# ----
#   bash bench/post_sync_check.sh              # 常规调用（cron/timer 用这个）
#   bash bench/post_sync_check.sh --force      # 跳过守卫 2/3，手工立即跑
#   bash bench/post_sync_check.sh --no-fix     # 只检测，不自动修
#   bash bench/post_sync_check.sh --verbose    # 同时输出到 stdout
#
# 退出码
#  0 = 通过（含「检出并已修复后通过」）
#  1 = 存在问题（修复失败，或需要人工介入）
#  2 = 被守卫跳过（正常情况，不是错误）
#  3 = 环境/用法错误
#
# 可调环境变量
#  DS_ROOT     数据集根目录（默认 /data/share/b601_20260910_164106）
#  QUIET_MINS  数据集需静默多少分钟才认为传输结束（默认 5）
#  LOG_DIR     日志目录（默认 bench/logs/）
#  ALLOW_FIX   1=允许自动修复（默认 1）
#  SKIP_VIDEO  1=跳过视频检查，快一些（默认 0）

set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "$(realpath "$0")")" && pwd)
VERIFY="$SCRIPT_DIR/verify_dataset.py"

DS_ROOT=${DS_ROOT:-/data/share/b601_20260910_164106}
QUIET_MINS=${QUIET_MINS:-5}
LOG_DIR=${LOG_DIR:-$SCRIPT_DIR/logs}
ALLOW_FIX=${ALLOW_FIX:-1}
SKIP_VIDEO=${SKIP_VIDEO:-0}

LOG="$LOG_DIR/post_sync.log"
LOCK="$LOG_DIR/.post_sync.lock"
STATE="$LOG_DIR/.post_sync.state"

FORCE=0; DO_FIX=$ALLOW_FIX; VERBOSE=0
for a in "$@"; do
    case "$a" in
        --force|-f)   FORCE=1 ;;
        --no-fix)     DO_FIX=0 ;;
        --verbose|-v) VERBOSE=1 ;;
        -h|--help)    sed -n '2,50p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "未知参数: $a（-h 看用法）" >&2; exit 3 ;;
    esac
done

mkdir -p "$LOG_DIR" || exit 3

log() {
    printf '%s %s\n' "$(date '+%F %T')" "$*" >> "$LOG"
    [ "$VERBOSE" = 1 ] && printf '%s\n' "$*"
}
die() { log "$*"; exit "${1:-3}"; }

# 日志超过 2MB 就只留尾部，避免无限增长
if [ -f "$LOG" ] && [ "$(stat -c%s "$LOG")" -gt 2097152 ]; then
    tail -n 2000 "$LOG" > "$LOG.tmp" && mv "$LOG.tmp" "$LOG"
fi

# ---------------- 守卫 1：单实例 ----------------
exec 9>"$LOCK" || die 3 "无法创建锁文件 $LOCK"
if ! flock -n 9; then
    log "跳过：已有另一个实例在跑"
    exit 2
fi

# ---------------- 环境 ----------------
[ -d "$DS_ROOT" ] || die 3 "数据集目录不存在：$DS_ROOT"
[ -f "$VERIFY" ]  || die 3 "找不到校验脚本：$VERIFY"

# 解释器解析：cron 的 PATH 极简，`python3` 会落到 /usr/bin/python3 而不是 anaconda。
# 这里显式挑一个带 pyarrow+numpy 的，避免依赖调用方的 PATH。
PY=${PYTHON:-}
if [ -z "$PY" ]; then
    for c in "${HOME:-/home/ksa}/anaconda3/bin/python3" \
             "${HOME:-/home/ksa}/.conda/envs/lerobot/bin/python3" \
             /usr/bin/python3 python3; do
        if "$c" -c 'import pyarrow, numpy' >/dev/null 2>&1; then PY=$c; break; fi
    done
fi
if [ -z "$PY" ] || ! "$PY" -c 'import pyarrow, numpy' >/dev/null 2>&1; then
    die 3 "找不到带 pyarrow+numpy 的 python3（可用 PYTHON=/path/to/python3 指定）"
fi

if [ "$FORCE" = 0 ]; then
    # ---------------- 守卫 2：传输是否还在进行 ----------------
    if pgrep -x rsync >/dev/null 2>&1 || pgrep -x scp >/dev/null 2>&1; then
        log "跳过：检测到 rsync/scp 正在运行"
        exit 2
    fi
    # sftp-server 会一直挂着不退出（实测空闲时也不例外），所以不拿它当判据，
    # 只靠 mtime 静默期判断 —— 这才是可靠的「传输结束」信号。
    recent=$(find "$DS_ROOT" -newermt "-${QUIET_MINS} minutes" -type f 2>/dev/null | head -1)
    if [ -n "$recent" ]; then
        log "跳过：数据集在 ${QUIET_MINS} 分钟内被写过（传输可能未结束）：$(basename "$recent")"
        exit 2
    fi

    # ---------------- 守卫 3：训练是否在跑 ----------------
    if pgrep -f "lerobot-train" >/dev/null 2>&1; then
        log "跳过：检测到 lerobot-train 正在运行，避免改写 parquet 影响 dataloader"
        exit 2
    fi
fi

# ---------------- 指纹短路：数据集没变就不重复做昂贵的校验 ----------------
# 轮询每 10 分钟一次，但同步只在少数时间发生。用「路径+大小+mtime」的哈希做指纹，
# 没变化就直接退出，让常态轮询几乎零成本（只需一次 find）。
FP=$(find "$DS_ROOT" -type f -printf '%p %s %T@\n' 2>/dev/null | sort | md5sum | cut -d' ' -f1)
prev_fp=""
[ -f "$STATE" ] && prev_fp=$(head -n 1 "$STATE")

if [ "$FORCE" = 0 ] && [ -n "$prev_fp" ] && [ "$prev_fp" = "$FP" ]; then
    log "跳过：数据集自上次校验以来无变化（fp ${FP:0:12}）"
    exit 0
fi

# ---------------- 记录数据集规模，识别增量批次 ----------------
read -r NFRAMES NEP <<<"$("$PY" - "$DS_ROOT" <<'PY'
import json, sys, pathlib
p = pathlib.Path(sys.argv[1]) / "meta" / "info.json"
try:
    d = json.loads(p.read_text())
    print(d.get("total_frames", -1), d.get("total_episodes", -1))
except Exception:
    print(-1, -1)
PY
)"

prev="${prev_fp:+$([ -f "$STATE" ] && sed -n '2p' "$STATE")}"
cur="$NFRAMES $NEP"
if [ -n "$prev" ] && [ "$prev" != "$cur" ]; then
    log "检测到增量：${prev} → ${cur}（frames episodes）"
fi

# ---------------- 检测 ----------------
V_BASE=(--root "$DS_ROOT")
V_OPTS=(--json)
[ "$SKIP_VIDEO" = 1 ] && V_OPTS+=(--no-video)

out=$("$PY" "$VERIFY" "${V_BASE[@]}" "${V_OPTS[@]}" 2>>"$LOG")
rc=$?

get() { printf '%s' "$out" | "$PY" -c "import json,sys;print(json.load(sys.stdin).get('$1',''))" 2>/dev/null; }

# 记录「本次校验通过时」的指纹 + 规模。修复会改写文件，所以调用时现场重算指纹。
save_state() {
    local f
    f=$(find "$DS_ROOT" -type f -printf '%p %s %T@\n' 2>/dev/null | sort | md5sum | cut -d' ' -f1)
    printf '%s\n%s\n' "$f" "$NFRAMES $NEP" > "$STATE"
}

if [ -z "$out" ]; then
    log "校验脚本无输出（rc=$rc），见下方错误"
    exit 1
fi

ok=$(get ok)
n_bogus=$(get n_bogus_rows)
n_rows=$(get total_rows)
n_mis=$(get n_misaligned_rows)

if [ "$ok" = "True" ]; then
    save_state
    log "OK：$n_rows 行 / ${NEP} episodes / 重复行 0"
    exit 0
fi

log "检出问题：重复行 $n_bogus / 错位行 $n_mis（共 $n_rows 行）"

if [ "$DO_FIX" != 1 ]; then
    log "未修复（--no-fix）：需要人工处理"
    exit 1
fi

# ---------------- 修复 ----------------
log "开始自动修复…"
fix_out=$("$PY" "$VERIFY" "${V_BASE[@]}" "${V_OPTS[@]}" --fix -y 2>>"$LOG")
fget() { printf '%s' "$fix_out" | "$PY" -c "import json,sys;print(json.load(sys.stdin).get('$1',''))" 2>/dev/null; }
f_ok=$(fget ok)
f_fixed=$(fget fixed)
f_rows=$(fget total_rows)

if [ "$f_ok" = "True" ]; then
    save_state          # 修复改写了文件，重算指纹再记录，避免下一轮误判为「无变化」
    log "✓ 修复成功：丢弃 $n_bogus 行废数据，现 $f_rows 行（fixed=$f_fixed）"
    log "  备份在 $DS_ROOT/data/chunk-000/file-009.parquet.bak（原始损坏版，勿删）"
    exit 0
fi

log "✗ 修复失败或修复后仍不自洽，需要人工介入"
log "  可用：python3 $VERIFY --root $DS_ROOT        # 看详细报告"
log "  回滚：把 \$DS_ROOT/data/chunk-*/*.parquet.bak 拷回原文件名"
exit 1
