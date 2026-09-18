#!/bin/bash
# 安装 / 卸载「数据集同步后自动校验」钩子（post_sync_check.sh）。
#
# 用法
# ----
#   bash bench/install_hook.sh status              # 看当前装了没、最近日志
#   bash bench/install_hook.sh install [分钟]      # 装 cron（默认每 10 分钟）
#   bash bench/install_hook.sh install --systemd   # 装 systemd user timer
#   bash bench/install_hook.sh run                 # 立即跑一次（--force）
#   bash bench/install_hook.sh uninstall           # 卸载（cron + systemd 都清）
#   bash bench/install_hook.sh ... --dry-run       # 只打印将要做什么，不动系统
#
# 为什么用轮询而不是事件触发
# --------------------------
# 本机没装 inotify-tools（无 inotifywait），而且同步是**增量重传**、没有固定的
# 「结束」事件。所以用周期轮询 + 钩子内部的三重守卫（单实例 / 传输中 / 训练中）——
# 守卫让「跑得太频繁」是安全的，代价只是一次 find + 一次结构校验。

set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "$(realpath "$0")")" && pwd)
HOOK="$SCRIPT_DIR/post_sync_check.sh"
MARK="b601-dataset-sync-check"
UNIT="b601-dataset-sync-check"
UNIT_DIR="${HOME:-/home/ksa}/.config/systemd/user"

[ -x "$HOOK" ] || { echo "找不到可执行的 $HOOK" >&2; exit 3; }

DRY=0; MODE="cron"; INTERVAL=10; ACTION=""
for a in "$@"; do
    case "$a" in
        status|install|uninstall|run) ACTION="$a" ;;
        --systemd) MODE="systemd" ;;
        --cron)    MODE="cron" ;;
        --dry-run) DRY=1 ;;
        ''|*[!0-9]*) echo "未知参数: $a（看文件头注释）" >&2; exit 3 ;;
        *) INTERVAL="$a" ;;
    esac
done
[ -n "$ACTION" ] || ACTION="status"

# 是否本机 cron 已有 / systemd 已装
cron_installed() { crontab -l 2>/dev/null | grep -qF "$MARK"; }
systemd_installed() { [ -f "$UNIT_DIR/$UNIT.timer" ]; }

do_it() {                      # do_it <描述> <命令...>
    if [ "$DRY" = 1 ]; then
        echo "  [dry-run] $1"
    else
        shift
        "$@"
    fi
}

cron_line() {
    printf '*/%s * * * * /bin/bash %s >/dev/null 2>&1  # %s' "$INTERVAL" "$HOOK" "$MARK"
}

# --------------------------------------------------------------------------- #
case "$ACTION" in

status)
    echo "钩子脚本 : $HOOK"
    echo
    if cron_installed; then
        echo "cron     : ✓ 已安装"
        crontab -l 2>/dev/null | grep -F "$MARK" | sed 's/^/           /'
    else
        echo "cron     : ✗ 未安装"
    fi
    if systemd_installed; then
        echo "systemd  : ✓ 已安装 ($UNIT_DIR/$UNIT.timer)"
        systemctl --user is-active "$UNIT.timer" 2>/dev/null | sed 's/^/           状态: /' || true
    else
        echo "systemd  : ✗ 未安装"
    fi
    echo
    echo "守护条件 : 单实例 / 数据静默 5 分钟 / 无 lerobot-train 在跑"
    echo
    LOG="$SCRIPT_DIR/logs/post_sync.log"
    if [ -f "$LOG" ]; then
        echo "最近日志 ($(wc -l < "$LOG") 行)："
        tail -n 8 "$LOG" | sed 's/^/  /'
    else
        echo "还没有日志（$LOG）—— 钩子尚未跑过"
    fi
    ;;

run)
    echo "立即执行（--force，跳过传输/训练守卫）…"
    bash "$HOOK" --force --verbose
    ;;

install)
    if [ "$MODE" = "cron" ]; then
        line=$(cron_line)
        echo "安装 cron（每 $INTERVAL 分钟）："
        echo "  $line"
        if [ "$DRY" = 0 ]; then
            tmp=$(mktemp) || exit 3
            crontab -l 2>/dev/null | grep -vF "$MARK" > "$tmp"
            printf '%s\n' "$line" >> "$tmp"
            crontab "$tmp" && echo "  ✓ 已写入 crontab"
            rm -f "$tmp"
        else
            echo "  [dry-run] 不动 crontab"
        fi
    else
        echo "安装 systemd user timer（每 $INTERVAL 分钟）："
        [ "$DRY" = 0 ] && mkdir -p "$UNIT_DIR"
        do_it "写 $UNIT_DIR/$UNIT.service" \
            bash -c "cat > '$UNIT_DIR/$UNIT.service'" <<EOF
[Unit]
Description=b601 数据集同步后校验修复

[Service]
Type=oneshot
ExecStart=/bin/bash $HOOK
EOF
        do_it "写 $UNIT_DIR/$UNIT.timer" \
            bash -c "cat > '$UNIT_DIR/$UNIT.timer'" <<EOF
[Unit]
Description=周期触发 b601 数据集校验

[Timer]
OnBootSec=5min
OnUnitActiveSec=${INTERVAL}min
Persistent=true

[Install]
WantedBy=timers.target
EOF
        do_it "daemon-reload + enable --now" \
            systemctl --user daemon-reload
        [ "$DRY" = 0 ] && systemctl --user enable --now "$UNIT.timer" \
            && echo "  ✓ 已启用"
        echo
        echo "提示：要让 timer 在登出后仍能跑，需要（只需一次）："
        echo "    sudo loginctl enable-linger ${USER:-ksa}"
    fi
    echo
    echo "装好后用 'bash bench/install_hook.sh status' 确认。"
    ;;

uninstall)
    echo "卸载钩子："
    if cron_installed; then
        if [ "$DRY" = 0 ]; then
            tmp=$(mktemp) || exit 3
            crontab -l 2>/dev/null | grep -vF "$MARK" > "$tmp"
            crontab "$tmp" && echo "  ✓ 已从 crontab 移除"
            rm -f "$tmp"
        else
            echo "  [dry-run] 从 crontab 移除"
        fi
    else
        echo "  cron: 本来就没装"
    fi
    if systemd_installed; then
        do_it "停用并删除 systemd unit" \
            systemctl --user disable --now "$UNIT.timer"
        if [ "$DRY" = 0 ]; then
            rm -f "$UNIT_DIR/$UNIT.timer" "$UNIT_DIR/$UNIT.service"
            systemctl --user daemon-reload
            echo "  ✓ 已删除 systemd unit"
        fi
    else
        echo "  systemd: 本来就没装"
    fi
    ;;

*) echo "未知动作: $ACTION" >&2; exit 3 ;;
esac
