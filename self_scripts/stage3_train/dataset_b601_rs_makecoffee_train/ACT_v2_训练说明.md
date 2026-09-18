# ACT v2 训练说明

本文记录 v2 这一版 ACT 训练的设计决定与实测验证。数据集侧的损坏/修复背景见
[`datastet_notes/`](datastet_notes/)，本文不重复。

| 文件 | 用途 |
|---|---|
| [`act_train_config_v2.json`](act_train_config_v2.json) | **本次使用的配置** |
| [`start_train_act_v2.sh`](start_train_act_v2.sh) | 启动器（带前置数据校验门禁） |
| [`act_train_config_v2_cosine.json`](act_train_config_v2_cosine.json) | cosine lr 衰减变体，**本次未用**，备好待 A/B |
| [`start_train_act.sh`](start_train_act.sh) | v1 启动器，**未改动**，保留 |

---

## 1. 为什么要有 v2（而不是直接重跑 v1）

**v1 现在直接重跑会崩**，实测：

```
✗ validate() 抛错: FileExistsError
  Output directory output_lerobot_train/b601_20260910_164106_act already exists
  and resume is False. Please change your output directory ...
```

v1 的输出目录里有 21 个 checkpoint（12GB），而 `TrainPipelineConfig.validate()`
拒绝覆盖已存在的 output_dir。所以 v1 的 benchmark 原样保留，v2 用新目录。

## 2. 核心决定：只换数据，超参一个字不动

理由来自「v1 真机验证过可用」这个事实：

> `datastet_notes/05-预防与治本方案.md` 里那条「加 lr 衰减」的建议，**前提写的是
> 「若真机效果不行」**。这个前提没有发生。

| | 有实测支撑？ |
|---|---|
| v1 超参（lr 8e-5 恒定） | ✅ 训练跑完 + 真机推理可用 |
| cosine lr 衰减 | ❌ 笔记里的推测性建议，未验证 |

一次只动一个变量，loss 曲线才能与 v1 直接比。

**v1 与 v2 的逐字段对照**（v2 = v1 逐字复制，只改两处）：

| 字段 | v1 | v2 |
|---|---|---|
| `dataset` | b601（同一 root，内容已增长） | 同 |
| `policy` | act / cuda / lr 8e-5 | 同 |
| `steps` | 100000 | 同 |
| `batch_size` | 8 | 同 |
| `save_freq` | 5000 | 同 |
| `num_workers` | 6 | 同 |
| `eval_freq` | 0 | 同 |
| `output_dir` | `..._act` | **`..._act_v2`** ← 差异 |
| `job_name` | `..._single_arm` | **`..._single_arm_v2`** ← 差异 |

真正的变量只有一个：**数据集从 180 ep / 453,139 帧 增长到 223 ep / 533,168 帧**（+24%）。

> 100K 步在 533,168 帧下 = `100000 × 64 ÷ 533168` = **12.0 个 pass**
> （v1 是 453,139 帧下的 14.12 个 pass）。墙钟按 v1 实测 2.83 it/s 约 **9.8 小时**。

## 3. 前置数据校验门禁（本次最重要的改动）

`start_train_act_v2.sh` 在开训前跑两道检查，**任一不过就拒绝开训**：

| # | 检查 | 判据 |
|---|---|---|
| 1 | 数据集完整性 | 总行数 == `meta total_frames`（+ index 无重复 + episode 边界自洽 + 视频不截断） |
| 2 | 传输静默期 | 数据集在 `QUIET_MINS`（默认 5）分钟内没被写过 |

### 为什么必须要

笔记 `DISTILLED.md` 的核心结论：这个损坏**训练时不报错**。

- 9-16 那次数据小（180 ep），错位后越界 → step 0 崩，立刻暴露
- 9-18 那次数据变大（223 ep），错位后**不越界** → 全程无报错、loss 正常，
  但 ep162 之后约 21% 的帧 state/action 取到错误 episode 的行（图文错配）

**所以「训练没崩」不能作为数据正常的证据**，必须有一个独立的、开训前跑的门禁。

实现上复用 [`bench/verify_dataset.py`](bench/verify_dataset.py)（`--json` 模式，退出码
0=通过 / 1=有问题），失败时再跑一次人读模式把报告打给用户。

## 4. 训练后复查：钩子守卫 3 的盲区

`bench/post_sync_check.sh` 的守卫 3 是「**有 `lerobot-train` 在跑就跳过**」。反过来读：

> 训练那 10 小时里，源端只要重传一次，`file-009` 就被覆盖回损坏版，
> 而钩子**那时不会去修** —— 数据在训练中途静默变坏，训练照跑不误。

前置校验管的是**开跑那一刻**，管不了中途。所以启动器在训练结束后会自动再数一遍总行数，
和开跑前对不上就明确报出来：

```
✓ 行数未变（533168），训练期间数据没被改动
   —— 或者 ——
✗✗ 数据在训练期间被改动：533168 → 544173 行
   源端大概率在训练中途重传 ... 这份 checkpoint 学到的是错配的数据，别直接上真机。
```

> 2026-09-18 已把根因在**源端**修掉（`192.168.60.183` 上的 `file-009.parquet`
> 丢弃末尾 11,005 行 row group），所以此后同步带下来的都是正确数据，
> 这个中途变坏的风险已经消除。此项复查作为兜底保留。

## 5. cosine 变体（备好但本次不用）

真要试 lr 衰减时，直接：

```bash
bash stage3_train/dataset_b601_rs_makecoffee_train/start_train_act_v2.sh \
     --config act_train_config_v2_cosine.json
```

### ⚠️ 它依赖一处 lerobot 源码改动

`src/lerobot/policies/act/configuration_act.py` 新增了 `lr_scheduler` 字段
（默认 `None` = 与上游逐字一致，**对本次 v2 训练无影响**）。

**为什么非改不可**：ACT 上游的 `get_scheduler_preset()` 恒返回 `None`。而 lerobot
里唯一能自定义 scheduler 的路径是 `use_policy_training_preset=false`，那条路会走
`policy.parameters()` —— 实测会把参数分组并掉：

```
用 preset（get_optim_params）:  group0 40,432,327 params @ 8e-5
                              group1  11,166,912 params @ 1e-5   ← ResNet18
用 policy.parameters()      :  合成一组，backbone 也吃 8e-5（8 倍）
```

`modeling_act.py:72` 那句 `# TODO: As of now, lr_backbone == lr` **与代码不符**，
实际 `optimizer_lr=8e-5`、`optimizer_lr_backbone=1e-5`（后者是 `ACTConfig` 的默认值）。
所以走 `false` 会在「加 schedule」之外**偷偷把 backbone lr 提到 8 倍**，
真机效果变化时无法归因。

新字段返回 scheduler，`get_optim_params()` 照常走，参数分组原样保留。

### 实测验证

```
group0:   40,432,327 params  lr 8e-5   ← 分组保住
group1:   11,166,912 params  lr 1e-5   ← ResNet18，仍是 1/8
lr 倍率 @ [0, 2000, 10000, 50000, 100000] = [0.0005, 0.999, 0.978, 0.55, 0.1]
```

用 v1 的真实 checkpoint 加载配置、走 `make_optimizer_and_scheduler` 真路径验证；
resume 往返（`lr_scheduler` 存进 checkpoint 的 `train_config.json` 再读回）也已验证。

## 6. 用法

```bash
cd /home/ksa/lerobot/self_scripts

bash stage3_train/dataset_b601_rs_makecoffee_train/start_train_act_v2.sh --dry-run  # 只看
bash stage3_train/dataset_b601_rs_makecoffee_train/start_train_act_v2.sh            # 开训

# 其他参数
#   --config <文件>        换配置（相对脚本目录），如 act_train_config_v2_cosine.json
#   --resume               从本配置自己的 last checkpoint 续训
#   --skip-data-check      跳过前置校验（不推荐）
#   --dry-run              只跑前置检查 + 打印将执行的命令
```

可调环境变量：`DS_ROOT`（覆盖数据集根目录）、`QUIET_MINS`（静默期分钟数）。

退出码：`0`=训练正常结束、`1`=数据校验未通过、`2`=传输中/被守卫拦截、`3`=环境或用法错误。

## 7. 已做的验证

| 项 | 结果 |
|---|---|
| 配置解析 + `validate()` | 通过 |
| 真实数据跑 `--dry-run` | 两道检查通过，打印正确命令 |
| **数据损坏时是否拦得住** | 造了「meta 说 100 帧、实际 150 行」的假数据集 → **rc=1，拒绝开训**并打印人读报告 |
| 输出目录已存在 | 报错并提示改用 `--resume`（rc=3） |
| `--dry-run` 是否留残留 | 不留（早期版本会留，已修，见下） |

### 踩过的坑：`--dry-run` 会留下 output_dir

早期版本的日志目录建在 `$OUT_DIR/logs/`，为了写日志得先 `mkdir` —— 而那个 mkdir
会先把 `output_dir` 建出来，于是**紧接着的存在性检查把自己的创建动作当成了冲突**，
且跑过一次 `--dry-run` 后真正开训反而被「目录已存在」挡住。

ACT 版目前没有内置日志重定向（直接输出到终端），不受影响；
**SmolVLA 版有日志文件，已在那边把日志移到 `output_dir` 同级目录**，
详见 [`SmolVLA_训练说明.md`](SmolVLA_训练说明.md) 第 5 节。

## 8. 已知局限

- `eval_freq` 仍是 0。笔记建议的「调大 eval_freq 跑少量 val」**在当前 lerobot 里做不到**：
  `lerobot_train.py:230` 与 `:482` 都要求 `cfg.env` 非空（gym 环境）才会执行 eval，
  没配 env 时 `eval_freq` 再大也不生效。要验证只能上真机 rollout。
- 启动器的 `--resume` 分支只验过配置解析往返，**没有真跑过续训**。首次使用前建议确认。
