# SmolVLA 训练说明（单臂 b601）

在同一个数据集（`/data/share/b601_20260910_164106`，223 ep / 533,168 帧）上训练
**SmolVLA**，与 ACT v2 同数据集、同训练量，可直接对比两个策略。

| 文件 | 用途 |
|---|---|
| [`smolvla_train_config.json`](smolvla_train_config.json) | 训练配置 |
| [`start_train_smolvla.sh`](start_train_smolvla.sh) | 启动器（带前置数据校验门禁） |

ACT 版见 [`ACT_v2_训练说明.md`](ACT_v2_训练说明.md)；数据集侧的损坏/修复背景见
[`datastet_notes/`](datastet_notes/)。

---

## 1. 与 ACT v2 的对照

| | ACT v2 | SmolVLA |
|---|---|---|
| 数据集 | b601 单臂（223 ep / 533,168 帧） | 同 |
| 步数 / pass | 100K / **12.0** | 100K / **12.0** |
| 有效 batch | 64（8 卡 × bs8） | 同 |
| 实测吞吐 | 2.83 it/s | **2.32 it/s** |
| 预计墙钟 | ~9.8 h | **~12 h** |
| 单个 ckpt | 591 MB | **1.5 GB**（save_freq 5000 → 20 个 ≈ 30 GB） |
| 可学习参数 | 52M 全量 | **100M / 450M**（expert-only） |
| lr schedule | 恒定 8e-5（无 schedule） | 余弦 5e-5 → 2.5e-6 |
| 需要改 lerobot 源码？ | 只有 cosine 变体需要 | **不需要** |

**关键结构差异**：ACT 上游 `get_scheduler_preset()` 恒返回 `None`（所以 ACT 要加 lr 衰减
得打源码补丁，见 ACT 文档第 5 节）；**SmolVLA 自带余弦衰减 preset**，直接调参即可。

## 2. ⚠️ `scheduler_decay_steps` 必须显式设成 `steps`

SmolVLA 的 `scheduler_decay_steps` **默认值是 30000**。若 `steps=100000` 而不改它，
余弦在 30K 步就走完了，剩下 70K 步 lr 会**一直停在 `decay_lr = 2.5e-6`**（峰值的 1/20）
—— 等于白烧 70K 步。实测：

```
decay_steps=30000  → lr @30K/50K/75K/100K = 2.5e-6, 2.5e-6, 2.5e-6, 2.5e-6   ← 后 70% 全平
decay_steps=100000 → lr @30K/50K/75K/100K = 4.0e-5, 2.6e-5, 9.5e-6, 2.5e-6   ← 正常铺满
```

**lerobot 的自动缩放只在 `steps < decay_steps`（训得比衰减短）时生效**，训得**长**它不管
（`CosineDecayWithWarmupSchedulerConfig.build()` 里 `if num_training_steps < self.num_decay_steps`）。
本配置已设为 100000。

## 3. 权重来源：用 `--policy.pretrained_path` 而非 `--policy.path`

```
--policy.pretrained_path=/home/ksa/.cache/modelscope/hub/models/lerobot/smolvla_base
```

| | 行为 |
|---|---|
| `pretrained_path` | 只加载**权重**，policy 的输入特征按**数据集**自动推断 |
| `policy.path` | 连 base 的**配置**一起加载（它期望 `camera1/2/3`，且维度与 b601 不同） |

base 是在 `camera1/2/3` 三路相机上训的，b601 是 `hand/front/top` 三路，**名字不同**。
沿用你 coffee_cup 那次跑通的做法（4 路、名字也不同，未加 `rename_map` / `empty_cameras`）。

**这一点已用 smoke test 实测验证**，见第 4 节。

语言指令来自 `meta/tasks.parquet`（LeRobot v3 用 `.parquet` 不是 `.jsonl`），b601 有 1 条：

> `Pick up the paper cup, place it on the silver tray of the coffee machine, pick up the cube,
> press the button with the cube (red light on), wait about 4 seconds, release the button
> (red light off), put the cube on the table first, then move the cup ...`

动作维度 7 会被 padding 到 `max_action_dim=32`，与 coffee_cup 那次 12 维的处理一致。

## 4. Smoke test（开长跑前的验证）

正式开跑前用 `--steps=20 --save_freq=20 --output_dir=/tmp/smolvla_smoke` 验了 20 步：

| 项 | 结果 |
|---|---|
| policy 构建 | ✅ **无相机名报错**（担心的风险点，已排除） |
| 参数量 | `num_learnable_params=99,880,992 (100M)` / `num_total_params=450,046,176 (450M)` |
| loss | 0.395 → 0.191（20 步） |
| checkpoint | 正常落盘（1.2 GB） |
| 报错 | 0 |

**⚠️ smoke test 的一个误导**：那 20 步里 lr 是从峰值开始**下降**的，看着像 warmup 没生效。
实际是 `steps=20 < scheduler_decay_steps=100000` 触发了自动缩放：

```
scale_factor = 20/100000 = 0.0002
actual_warmup_steps = int(1000 × 0.0002) = 0     ← warmup 被压成 0
actual_decay_steps  = 20                          ← 余弦压缩进 20 步
```

**用很短的 `steps` 做 smoke test 时，lr 轨迹不能代表真实 run。** 真实 run 里
`steps == decay_steps`，不触发缩放，warmup 正常 —— 实测首步 `lr=9.99e-08`
（= 5e-5 × lambda(1)，lambda(1)=0.002），随后稳步爬升。

## 5. 踩过的坑：日志目录会把 `output_dir` 提前建出来

第一次启动直接失败：

```
FileExistsError: Output directory output_lerobot_train/smolvla_b601_20260910_164106
already exists and resume is False.
```

原因是**我自己造的**：为了把日志写到 `$OUT_DIR/logs/train_smolvla.log`，脚本得先
`mkdir -p` —— 而那个 mkdir 会**先把 `output_dir` 建出来**，紧接着 lerobot 的
`TrainPipelineConfig.validate()` 检查到目录已存在，直接拒绝。

> 即：启动器先把目录建好，然后被 lerobot 当成「要覆盖已有 run」拦下。

**修法**：日志放到 `output_dir` 的**同级**目录，不再碰 output_dir：

```
output_lerobot_train/logs/smolvla_b601_20260910_164106.log
```

顺带修掉的第二个问题：`--dry-run` 也会走 mkdir，导致跑过一次 dry-run 后
真正开训反被「目录已存在」挡住。现在 mkdir 挪到了 dry-run 提前退出之后。

> ⚠️ **`dataset_new_makecoffee_train/start_train_smolvla.sh` 有同一个坑**：
> 它在分支判断之前就 `mkdir -p "$OUTPUT_DIR/logs"`。那次能跑起来可能是因为目录
> 当时已存在（走了 resume 分支），但**全新训练时会同样卡住**。

## 6. 实测吞吐与墙钟

历史速率（从 `smolvla_coffee_cup_button_20260826_232220` 的 checkpoint mtime 反推）：

```
step 2000 -> 16000  用时 3.99 h   3,510 steps/h   (1.026 s/step, 0.97 it/s)
```

按此外推 100K 步要 **28.5 小时** —— 但**这个外推是错的**。b601 上的实测：

```
333/100000 [02:23<11:55:57, 2.32it/s, loss=0.142, lr=1.34e-05]
```

**2.32 it/s ≈ 8,352 步/h，100K 步约 12 小时**，比外推快 2.4 倍。差异大概来自
coffee_cup 那次是 4 路相机、且当时机器负载不同。

> 教训：拿别的数据集/别的配置的历史吞吐外推长跑墙钟，误差可以到 2 倍以上。
> 以 smoke test 之后前几百步的实测为准。

## 7. 用法

```bash
cd /home/ksa/lerobot/self_scripts

bash stage3_train/dataset_b601_rs_makecoffee_train/start_train_smolvla.sh --dry-run  # 只看
bash stage3_train/dataset_b601_rs_makecoffee_train/start_train_smolvla.sh            # 开训

#   --resume             从本配置自己的 last checkpoint 续训
#   --skip-data-check    跳过前置校验（不推荐）
#   --dry-run            只跑前置检查 + 打印将执行的命令
```

可调环境变量：`DS_ROOT`、`QUIET_MINS`、`SMOLVLA_BASE`。

退出码：`0`=训练正常结束、`1`=数据校验未通过、`2`=传输中/被守卫拦截、`3`=环境或用法错误。

### 前置检查

与 ACT v2 相同的两道：

1. **数据集完整性** —— 总行数 == `meta total_frames`（+ index 无重复 + episode 边界 + 视频不截断）
2. **传输静默期** —— 数据集在 5 分钟内没被写过

另加一项**预训练权重存在性检查**（`smolvla_base` 目录 + `model.safetensors` + `config.json`），
在数据校验之前跑，环境问题快速失败。

训练结束后同样会复查总行数，检测数据是否在训练期间被改动。

### 监控

```bash
# 实时进度
tr '\r' '\n' < /home/ksa/lerobot/self_scripts/output_lerobot_train/logs/smolvla_b601_20260910_164106.log | tail -1

# checkpoint（step 5000 起，每个 1.5GB）
ls /home/ksa/lerobot/self_scripts/output_lerobot_train/smolvla_b601_20260910_164106/checkpoints/

# 停止
pkill -f start_train_smolvla.sh; pkill -f "lerobot-train --policy"
```

## 8. 已知局限

- 与 ACT 同样：`eval_freq` 未启用（lerobot 的 eval 需要 `cfg.env`，无 gym 环境时不执行），
  模型能否完成任务只能上真机 rollout 验证。
- `--resume` 分支只验过配置解析往返，**没有真跑过续训**。
- 磁盘：本 run 约 30 GB checkpoint。启动时根分区剩余 121 GB。
