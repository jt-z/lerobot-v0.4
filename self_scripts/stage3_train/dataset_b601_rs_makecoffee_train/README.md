# 单臂 b601_rs — make coffee ACT 训练

训练**单臂** `seeed_b601_rs_follower` 的 ACT 模型。

⚠️ 与隔壁 `dataset_new_makecoffee_train/`（**双臂** `bi_b601_so101_follower`）是两套
独立训练，本体不同、checkpoint 不通用，见下表。

## 数据集 `/data/share/b601_20260910_164106`

| | 本目录（单臂） | `dataset_new_makecoffee_train/`（双臂） |
|---|---|---|
| robot_type | `seeed_b601_rs_follower` | `bi_b601_so101_follower` |
| action / state | **7** 维 | **13** 维 |
| 相机 | 3 路：hand / front / top | 4 路：left_hand / left_top / left_front / right_hand |
| episodes | 180 | 101 |
| frames | 453,139 | 188,418 |
| fps | 30 | 30 |

任务：`Pick up the paper cup, place it on the silver tray of the coffee machine, pick up the cube,
press the button with the cube (red light on), wait about 4 seconds, release the button (red light
off), put the cube on the table first, then move the cup from the coffee machine to the table`

`root` 直接指向 `/data/share`，**不需要往 HF cache 拷贝**（`/data` 与 `/` 同属一个 LV，本地盘，无网络开销）。

## 超参依据

按 epoch 对齐旧的双臂 run（50k 步 ≈ 17 epochs）：

- 旧：188,418 ÷ (8卡 × bs8) = 2,944 steps/epoch
- 新：453,139 ÷ 64 = 7,080 steps/epoch → **100k 步 ≈ 14 epochs**

| 参数 | 值 | 说明 |
|---|---|---|
| `steps` | 100000 | 想严格对齐 17 epochs 可改 120000 |
| `batch_size` | 8 | 保持与旧 run 可比 |
| `save_freq` | 5000 | 单个 ckpt ≈ **591MB**，实测 21 个（20 + `last`）≈ **12GB**（原估 4.6GB 偏低 2.6 倍，见训练结果分析） |
| `num_workers` | 6 | 48 核，8 进程 × 6 = 48 吃满；CPU 争抢时可回落 4 |

**备选**：本数据集只有 3 路相机（旧的 4 路），显存占用更低，`batch_size` 提到 16/GPU
（全局 128）大概率放得下，wall-clock 约减半 —— 但需同步把 `optimizer_lr` 从 `8e-5`
线性放大到约 `1.2e-4`。

## 运行

```bash
cd /home/ksa/lerobot/self_scripts
bash stage3_train/dataset_b601_rs_makecoffee_train/start_train_act.sh
```

输出目录：`self_scripts/output_lerobot_train/b601_20260910_164106_act/`

## 数据集修复记录（2026-09-16）

首次训练在 step 0 就崩了：

```
RuntimeError: Invalid frame index=10332 for streamIndex=0; must be less than 8832
```

**根因**：`data/chunk-000/file-009.parquet` 末尾混入了 **11,005 行废弃数据**——episode 161
第一遍录制（崩过一次，重录了）残留的 writer 缓冲。该块是文件最后一个 row group
（row groups 为 `[2034, 2180, 2925, 2458, 2679, 2447, 2481, 2142, 2247, 2230, 11005]`）。

证据：
- parquet 实际 464,144 行 vs meta `total_frames` 453,139，差值**恰好 11,005**
- 180 个 episode 中**只有 ep161** 对不上：parquet 13,115 vs meta 2,110
- 全表**唯一一处** `index` 回退：物理行 429,365 处 `429364 → 418360`
- 70 个视频文件与 meta **完全吻合**（0 截断），所以只需修数据 parquet

`LeRobotDataset` 依赖「物理行号 == `index` 值」来定位 episode，这个块把后面所有 episode
的物理位置整体后移 11,005，于是 `__getitem__` 取到错误的 episode → 查错视频时间戳。

**修复**：丢弃 `file-009.parquet` 最后一个 row group（按 row group 边界无损删除）。

- 备份：`data/chunk-000/file-009.parquet.bak`（原始 sha256 `81518caf...f0068`）
- 修复后：file-009 `394537..418359`(23,823 行)，file-010 从 `418360` 起 → `index` 完全连续
- 校验：总行数 453,139 = meta；`index` 单调且唯一 0..453,138；180 episodes 结构无误；
  全部 180 个 episode 的边界帧 + 400 个随机采样帧解码 0 失败

> 若日后从源端重传此数据集，该损坏很可能**原样复现**（残留的
> `tmpb_cyevai/observation.images.top_161.mp4` 同样指向 ep161，说明崩溃在源端录制阶段）。

## 训练结果分析（100K 步 run，2026-09-16 19:00 → 09-17 04:48）

日志：`act_100k.log`（500 条 `ot_train.py:444` 汇总行，每 200 步一条）

```bash
python plot_train_log.py            # -> act_100k_curves.png（6 图：loss 线性/log、
                                    #    梯度范数、单步耗时、loss vs pass、收敛段分布）
```

**结论**：跑完、没崩、收敛正常；但 **loss 在最后 30K 步已进入平台期**，且
**本次 run 全程没有验证指标** —— 模型能否完成任务，这份日志回答不了。

### 基本盘

| 项 | 值 |
|---|---|
| 步数 | 100,000 |
| 数据集 | 453,139 帧 / 180 episodes |
| 有效 batch | 64（8 卡 × bs8） |
| **真实 epoch** | **14.12 个 pass**（与「超参依据」预测的 14 epochs 吻合） |
| 墙钟 | 9h48m，均 2.83 it/s |
| 参数量 | 52M |
| lr | 8e-5 **恒定，无 schedule** |

### loss 分段（原始 loss，每段 50 个日志点）

| step 区间 | mean | std | 阶段 |
|---|---|---|---|
| 0–10K | 0.383 | 0.617 | 陡降 4.082 → 0.163 |
| 10–20K | 0.162 | 0.0039 | |
| 20–30K | 0.145 | 0.0083 | |
| 30–40K | 0.121 | 0.0078 | |
| 40–50K | 0.109 | 0.0078 | 慢速线性下降 |
| 50–60K | 0.096 | 0.0110 | |
| 60–70K | 0.095 | 0.0087 | |
| 70–80K | 0.080 | 0.0159 | |
| 80–90K | 0.080 | 0.0141 | 平台 |
| 90–100K | **0.077** | 0.0144 | |

**学习几乎全发生在前 10K 步**（一个多 pass 就吃掉了绝大部分收益）；10–70K 慢速下降
（50K 步只降 0.067）；70–100K 进入平台。

**后 30K 还在不在降**（窗口均值 ± 标准误）：

```
50–60K  0.0953 ± 0.0016      80–90K  0.0800 ± 0.0021
60–70K  0.0934 ± 0.0014      90–100K 0.0765 ± 0.0019
70–80K  0.0815 ± 0.0023
```

末段 80–90K → 90–100K 降 0.0035，合并标准误约 0.0028，**仅 ~1.25σ**：还在降，但已降到
噪声级别，继续训的边际收益很小。

### 观察：70K 之后噪声底噪翻倍

去掉趋势（相对 EMA 的残差）后的 std：

```
40–50K 0.0075   50–60K 0.0094   60–70K 0.0084
70–80K 0.0156 ← 翻倍   80–90K 0.0137   90–100K 0.0140
```

**约 step 70K 处，batch 间 loss 波动从 ~0.008 跳到 ~0.014，之后再没降回去。**
拐点原因未知（lr 全程没变、数据顺序没变，日志里看不出）。它解释了后段为何会出现
0.027 / 0.044 这种孤立低值。

> ⚠️ **别把 `best 0.027` 当指标**：全程 loss<0.05 的只有 4 个孤立点
> （71000、78000×2、98000），全是单 batch 噪声，不代表模型能力。

### 梯度裁剪只在前 5K 步起作用

`grad_clip_norm=10.0`：

- 起点范数 **111.9**，step ≤ 5000 的 23 条日志全部 > 10 —— **这段一直被裁剪**，
  实际更新被压到范数 10
- **step 5000 之后（末条 10.11）再没超过阈值，裁剪完全不触发**
- 末值 1.22，全程无爆炸、无 NaN

前 5K 步靠裁剪硬稳住是 ACT 从零训的常态，不是问题。

### 单步耗时：吞吐缓慢劣化，但末段已稳住

| step 区间 | update | data | it/s |
|---|---|---|---|
| 0–20K | 0.3072s | 0.0154s | 3.10 |
| 20–40K | 0.3217s | 0.0154s | 2.97 |
| 40–60K | 0.3390s | 0.0153s | 2.82 |
| 60–80K | 0.3566s | 0.0146s | 2.69 |
| 80–100K | 0.3558s | 0.0145s | 2.70 |

- update 时间单调上涨 **+16%**，与 step 相关系数 **r = 0.892**（斜率 0.64 ms/1K 步）
  —— 系统性劣化而非抖动，成因日志看不出（散热 / 显存碎片 / 抢卡均可能）
- 末 20K 稳住（0.3566 → 0.3558），不是失控退化
- **data_s 全程只占 4.3%**，dataloader 不是瓶颈
- `data_s` 每 **~7,000 步**尖一次（15 次，1.68×），间距 = 一个 epoch 的步数
  （7,080）→ 是**换 epoch 时 worker 重填 buffer**，不是 checkpoint
  （已验证：`step % 5000 == 0` 处的 data_s 与其余点无差异，比值 0.99）

### 两个坑

**1. 日志里的 `epch` 是真实 epoch 的 8 倍。**

末行 `epch:112.99` 看着像训了 113 个 epoch，实际：

```
112.99 ÷ (100000 × 64 / 453139) = 8.000     ← 8 个 rank 的 episode 计数累加
```

**真实是 14.12 个 pass。** 用双臂 run 交叉验证同样精确等于 8.000（28.98 pass vs
`epch` 231.85），所以是 lerobot 该日志格式的固有行为，不是本 run 异常。
（`plot_train_log.py` 已按真实 pass 数画 x 轴并标注该倍数。）

**2. checkpoint 比原估计大 2.6 倍。**

```
21 个目录（005000…100000 + last），单个 591MB，合计 12G
```

原估「230MB / 4.6GB」，实际 **12GB**，磁盘规划需按 2.6 倍算。

### 日志回答不了的问题

**`eval_freq: 0` —— 全程没跑过任何验证。**

- 没有 val loss，**过拟合完全无法判断**（14 个 pass、52M 参数、453K 帧，风险真实存在
  但无证据）
- 训练 loss 0.077 只是拟合训练集的损失，**与任务成功率无直接关系**；ACT 的常见失败
  模式（长时序中间步骤崩、按钮按不到、杯子放歪）在 loss 曲线上全都看不见
- **能否可用只能上真机 rollout**

**与双臂 run 的粗略对照**（数据集不同，不可直接比）：twin 那个 50K 步 run 末段
loss 0.066–0.10，本 run 100K 步 0.077，同一量级。但 twin 是 **28.98 个 pass**
（数据小），本 run 才 **14.12 个 pass** —— **按 pass 数算本 run 训练量反而更少**。

### 下一步建议

1. **先别急着重训/续训**，拿 `checkpoints/100000` 上真机 rollout。对比
   `080000` vs `100000` 两个 ckpt，可直接看出后 20K 训练有没有用（这是唯一能决定
   下一步的动作）
2. **若真机效果不行**，问题不在步数（曲线已平台），而在：
   - 加 lr 衰减（现全程 8e-5 无 schedule，末期仍在全 lr 上抖）
   - 或按「超参依据」的备选：bs 提到 16、lr 提到 1.2e-4，墙钟减半
   - 或查数据质量（70K 后噪声翻倍那条线索值得回头查）
3. 补验证：`eval_freq` 调大跑少量 val，胜过现在「盲训 100K」
4. 下轮训练加 `nvidia-smi` / 功耗采样，定位 update 时间 +16% 的来源

## 备注

- 数据集根目录下的 `tmp*/` 是录制时的残留编码临时目录，不影响训练。
- **读日志时注意**：`ot_train.py:444` 行的 `epch` / `ep` 是 8 个 rank 累加值，
  除以 8 才是真实 epoch；或直接用 `step × 64 ÷ 453139`。
