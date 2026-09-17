#!/usr/bin/env python
"""解析 act_100k.log 的 INFO step 汇总行, 绘制训练曲线总览。

用法:
    python plot_train_log.py [日志路径] [-o 输出图片]

示例行:
    INFO 2026-09-16 19:01:24 ot_train.py:444 step:200 smpl:102K ep:41 epch:0.23 \
        loss:4.082 grdn:111.924 lr:8.0e-05 updt_s:0.285 data_s:0.025 pct:0.20% eta:7h53m

关于 epoch:
    日志里的 epch/ep 是 **8 个 rank 累加** 的值 (rank 数 = 有效 batch 的进程数),
    真实的数据集 pass 数 = step * effective_batch / num_frames。
    本脚本只画后者, 并在图上标注实测倍数, 避免把 113 当成 113 个 epoch 来读。
"""
import argparse
import datetime as dt
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- 调色板 (dataviz 参考调色板的 slot 1/2, 浅色底) ---------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
S1 = "#2a78d6"  # categorical slot 1 (blue)
S2 = "#eb6834"  # categorical slot 2 (orange)

EMA_SPAN = 50  # 每 200 步一条日志 -> span 50 约等于 10k 步的滑动平均

PAT = re.compile(
    r"INFO\s+\S+\s+\S+\s+\S+\.py:\d+\s+"
    r"step:(?P<step>[\d.]+[kKmM]?)\s+"
    r"smpl:(?P<smpl>\S+)\s+"
    r"ep:(?P<ep>[\d.]+[kKmM]?)\s+"
    r"epch:(?P<epch>[\d.]+)\s+"
    r"loss:(?P<loss>[\d.eE+-]+)\s+"
    r"grdn:(?P<grdn>[\d.eE+-]+)\s+"
    r"lr:(?P<lr>[\d.eE+-]+)\s+"
    r"updt_s:(?P<updt>[\d.]+)\s+"
    r"data_s:(?P<data>[\d.]+)"
)
TS = re.compile(r"^INFO\s+(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")
_MULT = {"k": 1e3, "m": 1e6, "g": 1e9}


def _num(s: str) -> float:
    s = s.strip().lower()
    if s and s[-1] in _MULT:
        return float(s[:-1]) * _MULT[s[-1]]
    return float(s)


def _header(text: str) -> dict:
    """从日志头部捞训练配置 (num_frames / 有效 batch / job 名 等)。"""
    out = {}
    for line in text.splitlines()[:130]:
        for key, pat in (
            ("frames", r"dataset\.num_frames=(\d+)"),
            ("episodes", r"dataset\.num_episodes=(\d+)"),
            ("eff_batch", r"Effective batch size:\s*\d+\s*x\s*(\d+)\s*=\s*(\d+)"),
            ("params", r"num_learnable_params=(\d+)"),
            ("steps", r"cfg\.steps=(\d+)"),
            ("repo_id", r"'repo_id':\s*'([^']+)'"),
            ("job_name", r"'job_name':\s*'([^']+)'"),
        ):
            m = re.search(pat, line)
            if not m:
                continue
            if key == "eff_batch":
                out["n_ranks"], out["eff_batch"] = int(m[1]), int(m[2])
            elif key in ("repo_id", "job_name"):
                out[key] = m[1]
            else:
                out[key] = int(m[1])
    return out


def _ema(vals, span):
    alpha = 2.0 / (span + 1.0)
    out, prev = [], None
    for v in vals:
        prev = v if prev is None else alpha * v + (1 - alpha) * prev
        out.append(prev)
    return out


def _hms(seconds):
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelsize=9, length=3)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_color(INK_2)
    ax.title.set_color(INK)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)


def legend(ax, **kw):
    lg = ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, **kw)
    return lg


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="?", default=str(here / "act_100k.log"))
    ap.add_argument("-o", "--out", default=str(here / "act_100k_curves.png"))
    args = ap.parse_args()

    log_path = Path(args.log)
    text = log_path.read_text(encoding="utf-8", errors="replace")

    rows = [m.groupdict() for m in (PAT.search(l) for l in text.splitlines()) if m]
    if not rows:
        raise SystemExit(f"未在 {log_path} 中解析到 step 汇总行")

    steps = [int(_num(r["step"])) for r in rows]
    losses = [float(r["loss"]) for r in rows]
    grdns = [float(r["grdn"]) for r in rows]
    updts = [float(r["updt"]) for r in rows]
    datas = [float(r["data"]) for r in rows]
    lrs = [float(r["lr"]) for r in rows]
    epchs = [float(r["epch"]) for r in rows]

    loss_ema = _ema(losses, EMA_SPAN)
    grdn_ema = _ema(grdns, EMA_SPAN)

    hdr = _header(text)
    n_frames, eff_batch = hdr.get("frames"), hdr.get("eff_batch")

    # 真实数据集 pass 数; 拿不到头部信息就退化成用 step 归一化
    if n_frames and eff_batch:
        passes = [s * eff_batch / n_frames for s in steps]
        note = ""
    else:
        passes = [s / steps[-1] for s in steps]
        note = " (num_frames 未解析到, x 轴为相对进度)"

    # 日志 epch 与真实 pass 数的实测倍数 —— 用来提醒读者别把 epch 当 epoch
    mult = epchs[-1] / passes[-1] if passes[-1] else float("nan")

    stamps = [dt.datetime.strptime(m[1], "%Y-%m-%d %H:%M:%S")
              for m in (TS.match(l) for l in text.splitlines()) if m]
    wall = (stamps[-1] - stamps[0]).total_seconds() if len(stamps) >= 2 else float("nan")

    tail = int(len(losses) * 0.8)  # 后 20% 视为收敛段
    tail_loss = losses[tail:]
    tail_mean = sum(tail_loss) / len(tail_loss)
    tail_std = (sum((v - tail_mean) ** 2 for v in tail_loss) / len(tail_loss)) ** 0.5

    final_step, final_loss = steps[-1], loss_ema[-1]
    lr_const = len(set(lrs)) == 1
    avg_its = final_step / wall if wall == wall else float("nan")

    print(f"共解析 {len(steps)} 条日志 (step {steps[0]} -> {steps[-1]})")
    print(f"数据集 pass: {passes[-1]:.2f}  |  日志 epch 末值 {epchs[-1]:.2f} "
          f"= {mult:.2f}x  (rank 累加)")
    print(f"loss: {losses[0]:.3f} -> 末段 {tail_mean:.4f} ± {tail_std:.4f} "
          f"(best {min(losses):.3f})")
    print(f"wall clock {_hms(wall)}, avg {avg_its:.2f} it/s")

    # ------------------------------------------------------------------ 画图
    plt.rcParams["font.family"] = "sans-serif"
    fig, axes = plt.subplots(3, 2, figsize=(15, 13.5), facecolor=SURFACE)
    fig.suptitle(f"ACT Training — {hdr.get('job_name', 'train')}"
                 f"   ({log_path.name})", fontsize=16, color=INK, y=0.985)

    # --- (0,0) loss, 线性坐标: 整体下降幅度
    ax = axes[0, 0]
    ax.plot(steps, losses, color=S1, alpha=0.35, linewidth=0.8, label="raw")
    ax.plot(steps, loss_ema, color=S1, linewidth=2.0, label=f"EMA({EMA_SPAN})")
    ax.set_ylim(bottom=0)
    ax.set_title("Loss — full range (linear)", fontsize=12)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.annotate(f"{final_loss:.3f}", xy=(steps[-1], loss_ema[-1]),
                xytext=(-52, 18), textcoords="offset points",
                color=S1, fontsize=10, fontweight="bold")
    style(ax)
    legend(ax, loc="upper right", ncol=2)

    # --- (0,1) loss, log 坐标: 后段细节 (线性图里 tail 被压扁)
    ax = axes[0, 1]
    ax.plot(steps, losses, color=S1, alpha=0.35, linewidth=0.8, label="raw")
    ax.plot(steps, loss_ema, color=S1, linewidth=2.0, label=f"EMA({EMA_SPAN})")
    ax.set_yscale("log")
    ax.axhline(tail_mean, color=MUTED, linewidth=1.0, linestyle="--")
    ax.annotate(f"last 20% mean {tail_mean:.3f}", xy=(steps[0], tail_mean),
                xytext=(6, 6), textcoords="offset points", color=INK_2, fontsize=9)
    ax.set_title("Loss — log scale (tail detail)", fontsize=12)
    ax.set_xlabel("step")
    ax.set_ylabel("loss (log)")
    style(ax)
    legend(ax, loc="upper right", ncol=2)

    # --- (1,0) 梯度范数, log 坐标 (111.9 -> ~1.2)
    ax = axes[1, 0]
    ax.plot(steps, grdns, color=S2, alpha=0.35, linewidth=0.8, label="raw")
    ax.plot(steps, grdn_ema, color=S2, linewidth=2.0, label=f"EMA({EMA_SPAN})")
    ax.set_yscale("log")
    ax.set_title("Gradient norm (log)", fontsize=12)
    ax.set_xlabel("step")
    ax.set_ylabel("grad norm (log)")
    ax.annotate(f"{grdns[0]:.0f}", xy=(steps[0], grdns[0]),
                xytext=(0, 8), textcoords="offset points", color=S2, fontsize=10)
    ax.annotate(f"{grdn_ema[-1]:.2f}", xy=(steps[-1], grdn_ema[-1]),
                xytext=(-46, 10), textcoords="offset points",
                color=S2, fontsize=10, fontweight="bold")
    style(ax)
    legend(ax, loc="upper right", ncol=2)

    # --- (1,1) 单步耗时拆解 (同一单位 -> 同一坐标轴, 不做双轴)
    # data_s 每个 epoch 边界尖一下 —— 检测出来标上, 否则这条线看着"没事发生"
    d_mean = sum(datas) / len(datas)
    d_std = (sum((v - d_mean) ** 2 for v in datas) / len(datas)) ** 0.5
    spikes = [i for i, v in enumerate(datas) if v > d_mean + 3 * d_std]

    ax = axes[1, 1]
    ax.plot(steps, updts, color=S1, linewidth=1.8, label="update (GPU)")
    ax.plot(steps, datas, color=S2, linewidth=1.8, label="data loading")
    ax.plot([steps[i] for i in spikes], [datas[i] for i in spikes], "o",
            color=S2, markersize=4.5, zorder=5)
    ax.set_ylim(0, max(updts) * 1.12)
    ax.set_title("Step time breakdown (same axis, seconds)", fontsize=12)
    ax.set_xlabel("step")
    ax.set_ylabel("seconds / step")
    share = sum(datas) / (sum(datas) + sum(updts)) * 100
    ax.annotate(f"summed over run: data = {share:.1f}% of step time",
                xy=(0.33, 0.60), xycoords="axes fraction", color=INK_2, fontsize=9.5)

    # 尖峰的间距是否等于一个 epoch 的步数 -> 是的话就是 dataloader 换 epoch 重填
    if len(spikes) >= 3:
        # spikes 存的是下标, 间距要换算回 step (经 steps[] 取真实值, 容忍不等距日志)
        gaps = sorted(steps[b] - steps[a] for a, b in zip(spikes, spikes[1:]))
        gap = gaps[len(gaps) // 2]
        per_epoch = n_frames / eff_batch if n_frames and eff_batch else None
        if per_epoch and abs(gap - per_epoch) / per_epoch < 0.15:
            ratio = sum(datas[i] for i in spikes) / len(spikes) / d_mean
            ax.annotate(f"spiked points = epoch boundaries every {gap:,} steps\n"
                        f"({len(spikes)} of them; data_s {ratio:.2f}x normal — "
                        f"loader refill)",
                        xy=(0.33, 0.44), xycoords="axes fraction",
                        color=MUTED, fontsize=9)
    style(ax)
    legend(ax, loc="center left")

    # --- (2,0) loss vs 数据集 pass 数 (真实 epoch)
    ax = axes[2, 0]
    ax.plot(passes, losses, ".", color=S1, alpha=0.3, markersize=3, label="raw")
    ax.plot(passes, loss_ema, color=S1, linewidth=2.0, label=f"EMA({EMA_SPAN})")
    ax.set_yscale("log")
    ax.set_title("Loss vs dataset passes", fontsize=12)
    ax.set_xlabel("dataset passes (step x eff_batch / num_frames)" + note)
    ax.set_ylabel("loss (log)")
    if mult == mult and abs(mult - round(mult)) < 0.01 and round(mult) > 1:
        ax.annotate(f"log `epch` ends at {epchs[-1]:.1f} = {mult:.0f}x these passes\n"
                    f"(per-rank episode counts summed)",
                    xy=(0.03, 0.10), xycoords="axes fraction",
                    color=MUTED, fontsize=9,
                    bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.85, pad=2))
    style(ax)
    legend(ax, loc="upper right", ncol=2)

    # --- (2,1) 收敛段 loss 分布: 噪声底噪有多宽
    ax = axes[2, 1]
    counts, _, _ = ax.hist(tail_loss, bins=30, color=S1, alpha=0.85,
                           edgecolor=SURFACE, linewidth=0.5)
    top = counts.max() * 1.5  # 留白给标注, 不然文字压在柱子上
    ax.set_ylim(0, top)
    ax.axvline(tail_mean, color=S2, linewidth=1.8)
    ax.annotate(f"mean {tail_mean:.3f}\nσ {tail_std:.3f}\nbest {min(tail_loss):.3f}",
                xy=(tail_mean, top), xytext=(7, -4), textcoords="offset points",
                va="top", color=S2, fontsize=9.5)
    ax.set_title(f"Loss distribution — last {len(tail_loss)} logs "
                 f"(step {steps[tail]:,}+)", fontsize=12)
    ax.set_xlabel("loss")
    ax.set_ylabel("count")
    style(ax)

    # --- 顶部关键指标
    stats = (
        f"steps {final_step:,}   ·   dataset passes {passes[-1]:.1f}   ·   "
        f"wall clock {_hms(wall)}   ·   avg {avg_its:.2f} it/s",
        f"final loss (EMA) {final_loss:.3f}   ·   best {min(losses):.3f}   ·   "
        f"last 20% {tail_mean:.3f} ± {tail_std:.3f}   ·   grad norm "
        f"{grdns[0]:.0f} → {grdn_ema[-1]:.2f}",
        f"lr {lrs[0]:.2e}{' (constant, no schedule)' if lr_const else ''}   ·   "
        f"effective batch {eff_batch or '?'} "
        f"({hdr.get('n_ranks', '?')} ranks)   ·   "
        f"{(hdr.get('params') or 0) / 1e6:.0f}M params   ·   "
        f"frames {n_frames:,}" if n_frames else "",
    )
    for i, line in enumerate(s for s in stats if s):
        fig.text(0.5, 0.952 - i * 0.0165, line, ha="center", fontsize=10,
                 color=INK_2 if i else INK)

    fig.tight_layout(rect=(0.01, 0.01, 0.99, 0.925))
    fig.savefig(args.out, dpi=150, facecolor=SURFACE)
    print(f"曲线已保存: {args.out}")


if __name__ == "__main__":
    main()
