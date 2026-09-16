#!/usr/bin/env python
"""解析 train_act.log 中的 INFO step 汇总行并绘制训练曲线。"""
import re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG = Path(__file__).parent / "train_act.log"
OUT = Path(__file__).parent / "train_act_curves.png"

# 示例: INFO 2026-08-27 18:45:51 ot_train.py:444 step:200 smpl:102K ep:102 epch:0.93 loss:4.087 grdn:107.307 lr:8.0e-05 updt_s:0.520 data_s:0.028 pct:0.40% eta:7h11m
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

_MULT = {"k": 1e3, "m": 1e6}


def _num(s: str) -> float:
    s = s.strip().lower()
    if s and s[-1] in _MULT:
        return float(s[:-1]) * _MULT[s[-1]]
    return float(s)


steps, losses, grdns, lrs, updts, datas = [], [], [], [], [], []
for line in LOG.read_text(encoding="utf-8").splitlines():
    m = PAT.search(line)
    if not m:
        continue
    steps.append(int(_num(m["step"])))
    losses.append(float(m["loss"]))
    grdns.append(float(m["grdn"]))
    lrs.append(float(m["lr"]))
    updts.append(float(m["updt"]))
    datas.append(float(m["data"]))

print(f"共解析 {len(steps)} 条日志 (step {steps[0]} -> {steps[-1]})" if steps else "未解析到数据")
if not steps:
    raise SystemExit(1)

fig, axes = plt.subplots(3, 2, figsize=(14, 12))
fig.suptitle("ACT Training Curves (train_act.log)", fontsize=15)

# (ax, x, y, title, ylabel, color)
plots = [
    (axes[0, 0], steps, losses, "Loss", "loss", "tab:red"),
    (axes[0, 1], steps, grdns, "Gradient Norm", "grad norm", "tab:blue"),
    (axes[1, 0], steps, lrs, "Learning Rate", "lr", "tab:green"),
    (axes[1, 1], steps, updts, "Update Time (s)", "s/step", "tab:purple"),
    (axes[2, 0], steps, datas, "Data Loading Time (s)", "s/step", "tab:orange"),
]
for ax, x, y, title, ylabel, color in plots:
    ax.plot(x, y, "-o", color=color, markersize=3, linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(min(x), max(x))

# 只画 loss 的子图, 放最大图便于观察
axes[2, 1].plot(steps, losses, "-", color="tab:red", linewidth=1.2)
axes[2, 1].set_title("Loss (raw, every log_freq)")
axes[2, 1].set_xlabel("step")
axes[2, 1].set_ylabel("loss")
axes[2, 1].grid(True, alpha=0.3)
axes[2, 1].set_xlim(min(steps), max(steps))

fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(OUT, dpi=150)
print(f"曲线已保存: {OUT}")
