# SmolVLA 模型构成与显存分析 —— 为什么单卡只占 4.5 GB

**分析对象**：`b601_20260910_164106` 单臂 SmolVLA（`smolvla_train_config.json` 那个 run）
**日期**：2026-09-18
**关联**：[`SmolVLA_训练说明.md`](SmolVLA_训练说明.md)（训练流程）、
[`bench/model_param_analysis.md`](bench/model_param_analysis.md)（ACT 侧同类分析）、
[`bench/multigpu_scaling_analysis.md`](bench/multigpu_scaling_analysis.md)（多卡扩展性）

> 环境：`transformers 4.53.3` / `torch 2.7.1+cu126` / `safetensors 0.8.0`，8×RTX 3090（24 GB）。
> 数字分三类：**【实测】** = 直接量出来的（重建模型统计、前反向探针、落盘文件大小、
> `nvidia-smi`）；**【推算】** = 由实测数据加减得到的；**【文献】** = 引自 SmolVLA 论文
> / 模型卡，未在本机复现。复现脚本见 §7。

---

## 0. 结论摘要

| 问题 | 结论 |
|---|---|
| 总参数量？ | **450.0M**，常驻权重仅 **1.198 GB**（混合精度，不是全 fp32） |
| 实际训练多少？ | **99.88M（22.2%）** —— 只有 action expert + 投影头 |
| **有预训练权重的参数？** | **450M / 450M（100%）**，与 `smolvla_base` 逐比特相同，**没有任何随机初始化** |
| 和 ACT 一样「大部分从零训」吗？ | **完全相反** —— ACT 78.4% 从零训，SmolVLA **0%**（见 §1.1） |
| 冻结 SigLIP/SmolLM2 会让 b601 效果差吗？ | **不必然** —— 冻结是设计而非妥协；真风险在数据量（见 §8） |
| 冻结 `lm_head` 呢？ | **无影响** —— 它在 forward 里根本不执行（见 §1.2） |
| 指令有没有问题？ | ⚠️ **被静默截断 66→48，丢的是任务主目标**（见 §9） |
| 冻结了 350M 参数，所以省显存？ | **只省了优化器状态，激活一点没省** |
| 为什么只占 4.5 GB？ | **只训 22% 参数** → AdamW 状态只建 99.88M 的量 |
| 4.5 GB 里最大的是什么？ | **前向激活 2.14 GB**（bs=8/卡），比权重本身还大 |
| 显存还有 20 GB 富余，能开大 batch 提速吗？ | **不建议**，理由见 §6，与 ACT 那边实测结论一致 |

**一句话**：4.5 GB 是「只微调 action expert」这个设计的结果，不是模型本身小。
而且**冻结 VLM 并没有省掉它的激活** —— cross-attn 结构决定了 VLM 每层的 hidden state
必须为反向保留，这 2 GB 是省不掉的。

---

## 1. 模型构成【实测】

`load_vlm_weights=False`（**你的 JSON 配置覆盖了 base 的 `True`**，见 §5 的坑），
所以 VLM 是按 config 新建的，**dtype 并不统一**：

| 组件 | 参数量 | 占比 | dtype | 显存 | 训练 |
|---|---:|---:|---|---:|---|
| vision encoder (SigLIP) | 86.4M | 19.2% | **fp32** | 0.346 GB | ✗ |
| vlm_text (SmolLM2，取前 16 层，hidden 960) | 204.6M | 45.5% | **bf16** | 0.409 GB | ✗ |
| lm_head | 47.3M | 10.5% | **fp32** | 0.189 GB | ✗ |
| **action expert**（16 层，hidden 720） | 98.2M | 21.8% | bf16 97M + fp32 2M | 0.200 GB | **✓** |
| proj heads / connector | 13.4M | 3.0% | fp32 | 0.054 GB | 部分 ✓ |
| **合计** | **450.0M** | 100% | 混合 | **1.198 GB** | **99.88M** |

- 训练日志实测：`num_learnable_params=99880992` / `num_total_params=450046176` ✔ 吻合
- 1.198 GB 与早先 run 落盘的
  `smolvla_coffee_cup_button_20260826_232220/checkpoints/004000/pretrained_model/model.safetensors`
  **大小逐字节一致** —— 说明重建是准的，模型在显存里就是这个 dtype 组合

> `vlm_text 204.6M` 对应 SmolLM2-360M 截断到 16 层（`num_vlm_layers=16`）；
> expert hidden = 960 × `expert_width_multiplier 0.75` = 720。
> 语言 embedding 与 lm_head **未 tying**，所以 lm_head 单独占 47.3M。

### 1.1 ⚠️ 权重来源：全部 450M 都来自预训练，**没有随机初始化**

日志里出现 `load_vlm_weights=False`，很容易读成「VLM 权重不加载 = 随机初始化」——**不是**。
两条加载路径是**冗余**的，都能拿到 VLM 权重：

| 路径 | 来源 | 本 run |
|---|---|---|
| `load_vlm_weights=True` | 从 HF 拉原始 `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` | ✗ 关闭（且在本环境会崩，见 §5） |
| `--policy.pretrained_path` | 从 `lerobot/smolvla_base` 拉**完整 450M**（已含 VLM + expert） | ✓ 生效 |

即 `False` 只是省掉「先从 HF 拉一遍 SmolVLM2、再被 smolvla_base 覆盖」的重复 I/O；
最终权重一样，而且**来自 smolvla_base 的那个更好**（是已经过机器人数据预训练的版本，
不是原始 VLM）。另外训练日志里 8 个 rank 的 `log_model_loading_keys` **一条 missing /
unexpected key 的 warning 都没打**。

**【实测】重建模型 vs `smolvla_base/model.safetensors`，逐张量 bit 级比对**（复现见 §7.5）：

```
文件 key 数 500   模型 key 数 500
模型有/文件无（会随机初始化）: 0
文件有/模型无              : 0
逐张量 bit 级比对: 完全一致 500/500  覆盖 450.0M 参数
可训练张量 155 个，全部来自 base 文件: True
```

**与 ACT 的对比 —— 这是两者最大的结构性差异**：

| | ACT | SmolVLA |
|---|---:|---:|
| 有预训练权重的参数 | 11.2M / 51.6M（**21.6%**，只有 ResNet18 backbone） | **450M / 450M（100%）** |
| 本次 run 之内从零训 | **40.4M（78.4%）** | **0** |
| 本次 run 可训练 | 51.6M（全量，含从零的 transformer） | 99.88M（只有 expert + 投影头） |
| 你的 223 ep / 533k 帧 的作用 | **训 78% 的参数** | **微调 22% 的参数** |

> ⚠️ **`bench/model_param_analysis.md` 里 ACT 那句「78.4% 从零训」不能类推到 SmolVLA。**
> 两个 run 走的是相反的范式：ACT 基本从零学，SmolVLA 是纯微调。

**`smolvla_base` 本身是怎么来的**【文献，SmolVLA 论文 arXiv:2506.01844 / HF blog】：

- 骨干 SmolVLM2-500M（SigLIP + SmolLM2）来自 VLM 预训练
- action expert ~100M **没有可继承的预训练权重**（flow-matching expert 在 VLM 生态里
  不存在现成来源），所以在 SmolVLA 预训练**开始时**确实是随机的 —— 但随后吃了约
  **481 个 LeRobot 社区数据集 / ~22.9k episodes / ~10.6M 帧**（以 SO-100/SO-101 为主，30 FPS）
- 论文消融：**去掉**社区预训练时 SO100 成功率 51.7%，**有**预训练后 78.3%（+26.6 个点）

> **对本项目的两个含义**：
> 1. b601_rs **不是** SO-100/SO-101 系列，所以这仍是一次**跨本体迁移** —— 预训练权重
>    不是「同款手臂」的，但量级上远好过从零训。
> 2. 起点是预训练好的策略，SmolVLA **很可能比 ACT 更早收敛**，100K 步里可能有一段
>    是在平台上空转。要确认只能上真机 rollout 对比不同 ckpt（与 ACT 那边结论一致）。

### 1.2 `lm_head` 冻结无影响 —— 它**根本不执行**（47.3M 死重）

容易被列进「冻结导致效果差」的名单，但它其实从不参与计算。
`smolvlm_with_expert.py:413` 的 forward 只取这两个子模块：

```python
models = [self.get_vlm_model().text_model, self.lm_expert]
```

全程不碰 `lm_head`。那 47.3M 参数 / 0.189 GB 是**死重** ——
与 ACT 里「推理时完全不执行的 VAE encoder」（`bench/model_param_analysis.md` §4）同类。

> 所以「冻结 `lm_head` 导致 b601 效果差」这条因果链**不存在**：
> 不参与计算的东西，冻结与否对输出没有任何影响。

---

## 2. 为什么优化器状态这么小 —— 只训 22%

`SmolVLAConfig` 的三个默认值决定了这一切：

```python
freeze_vision_encoder: bool = True     # SigLIP 全冻
train_expert_only:     bool = True     # VLM 全冻（self.vlm.eval() + requires_grad=False）
train_state_proj:      bool = True     # 只有这个 32→960 的投影可训
```

`train_expert_only=True` 走到 `smolvlm_with_expert.py:144-147`：

```python
if self.train_expert_only:
    self.vlm.eval()
    for params in self.vlm.parameters():
        params.requires_grad = False
```

于是可训参数只剩 **99.88M**，而 PyTorch 只为 `requires_grad=True` 的参数建
梯度与 AdamW 状态：

| 项 | 全量微调（假想） | **本 run（实测）** |
|---|---:|---:|
| 权重常驻 | 1.20 GB | **1.21 GB**（冻结的也要在显存里） |
| 梯度 | 0.83 GB | **0.32 GB** |
| AdamW (m+v) | 1.66 GB | **0.41 GB** |
| 小计 | **3.69 GB** | **1.94 GB** |

**AdamW 0.41 GB 得到了独立验证**：早先 run 落盘的
`training_state/optimizer_state.safetensors` = **412.66 MB**，
而 2 × 0.206 GB = 0.412 GB ✔ 完全对上。

> 若全量微调，光静态状态就 3.69 GB，再加 2.14 GB 激活 → 接近 6 GB，仍能塞进 3090，
> 但**「省」从来不是 4.5 GB 的原因**，激活才是大头（下一节）。

---

## 3. 显存账【实测】

单卡 bs=8 的前向 + 反向探针（随机权重，合成数据），与真实 run 对照：

| 项 | 实测 | 说明 |
|---|---:|---|
| 权重常驻 | 1.21 GB | 450M 混合精度 |
| **前向激活** | **2.14 GB** | ← **最大头**，比权重还大 |
| 梯度 | 0.32 GB | 只为 99.88M 可训参数建 |
| AdamW (m+v) | 0.41 GB | 与落盘 optimizer state 吻合 |
| CUDA context / 分配器开销 | ~0.5 GB | cuBLAS/cuDNN handle、NCCL buffer、碎片 |
| **合计** | **≈ 4.5 GB** | `nvidia-smi` 实测 **4550–4592 MiB** ✔ |

单步峰值（`torch.cuda.max_memory_allocated`）：bs=8 → **3.48 GB**（不含优化器状态，
优化器在首个 step 之后才分配，加上即 ~3.9 GB）。

---

## 4. 关键机制：冻结了 VLM，为什么还有 2 GB 激活？

这一节是整个分析里最反直觉的部分，我**一开始判断错了，被实测纠正**。

### 4.1 先验假设（错的）

`state_proj` 可训 → `state_emb` 带梯度 → 它被 `torch.cat` 进 prefix
→ 整个 prefix `requires_grad=True` → 16 层 VLM 全都建图。

### 4.2 实测否掉了它

| 配置 | 前向激活 | `prefix.requires_grad` |
|---|---:|---|
| 基线（`train_state_proj=True`） | 2.14 GB | True |
| `train_state_proj=False` | **2.12 GB** | **False** |

**白冻了 —— 激活几乎没变。** 假设不成立。

### 4.3 真正的原因：expert 的 k/v 投影吃的是 VLM 的 hidden state

`smolvlm_with_expert.py:109-123`，cross_attn 模式下 expert 的 k_proj / v_proj
被**重建**成从 VLM 维度取值：

```python
if "cross" in attention_mode:
    for layer_idx in range(len(self.lm_expert.layers)):
        if self.self_attn_every_n_layers > 0 and layer_idx % self.self_attn_every_n_layers == 0:
            continue                       # 每隔 2 层留一个真正的 self-attn
        self.lm_expert.layers[layer_idx].self_attn.k_proj = nn.Linear(
            config.text_config.num_key_value_heads * config.text_config.head_dim,   # 960 ← VLM 的维度
            lm_expert_config.num_key_value_heads * lm_expert_config.head_dim,       # 720 ← expert 的维度
        )
```

即：expert 在 15/16 层上做的是**对 VLM 每层 hidden state 的 cross-attention**，
而这两个投影是**可训练**的。autograd 要算它们的梯度，就必须保留 VLM 每层的输出。

**VLM 参数冻结只是不产生 VLM 自己的权重梯度，激活一分没省。**
再加上每层保留的 attention 矩阵与 MLP 中间量，16 层叠起来就是 2 GB。

> 这也解释了 §3 里「激活 2.14 GB > 权重 1.21 GB」——并不是异常，是结构决定的。

### 4.4 激活随 batch 严格线性【实测】

| bs | 前向激活 | 单步峰值 |
|---:|---:|---:|
| 1 | 0.29 GB | — |
| 2 | 0.54 GB | — |
| 4 | 1.08 GB | — |
| 8 | 2.14 GB | 3.48 GB |
| 16 | 4.24 GB | 5.73 GB |
| 32 | 8.41 GB | 10.16 GB |

完全线性 → 没有隐藏的常数项，也无法靠"小 batch 摊薄"。

---

## 5. 训练参数（live run 实际解析出来的）【实测】

```
lr 5e-5          betas (0.9, 0.95)   eps 1e-8
weight_decay 1e-10   grad_clip_norm 10
cosine: warmup 1000 → decay 100000 → decay_lr 2.5e-6
bs 8/卡 × 8 卡 = 64    steps 100000    chunk_size 50    n_action_steps 50
```

序列长度（实测）：**prefix = 241 tokens** = 3 图 × 64 + 48 语言 + 1 state，
suffix（expert）= 50。**图片 token 占 prefix 的 80%** —— 512×512 经 SigLIP(16) +
pixel-shuffle(4) 后每路只剩 8×8=64 个 token，图像信息被压得很狠。

### ⚠️ 坑：`--policy.path` 在本环境会直接崩

| | 行为 | 本环境 |
|---|---|---|
| `--policy.pretrained_path`（**当前用的**） | 只加载权重，policy 特征按数据集推断；配置走你的 JSON → `load_vlm_weights=False` | ✅ 正常 |
| `--policy.path` | 连 base 的 `config.json` 一起加载 → `load_vlm_weights=True` | ❌ **TypeError** |

因为 `load_vlm_weights=True` 时 `smolvlm_with_expert.py:78-83` 会走：

```python
AutoModelForImageTextToText.from_pretrained(model_id, device_map=device,
                                            dtype=torch.float32, ...)
```

而 `transformers 4.53.3` 不认 `dtype=`（那是 4.56+ 的写法，此处应为 `torch_dtype=`）→
`TypeError: SmolVLMForConditionalGeneration.__init__() got an unexpected keyword argument 'dtype'`。

> 所以 `SmolVLA_训练说明.md` §3 说的「用 pretrained_path 而不是 policy.path，是因为相机名
> 不同」只是其中一个理由；**在本环境它还多救了一次命**。别换回去。
>
> 另外注意：这也意味着**模型实际是「按 config 新建 + 加载 base 权重」**，
> 而不是"加载 base 的配置"。VLM 那 350M 的权重仍然来自 smolvla_base 的 safetensors，
> 没有随机初始化的问题。

---

## 6. 还有 20 GB 富余，要不要开大 batch？

**不建议**，两个独立理由：

1. **换不来墙钟。** 隔壁 ACT 实测（`bench/multigpu_scaling_analysis.md` §5.2）：
   单卡 bs 8→16→24 吞吐只有 43.1 → 45.5 → 45.9 samples/s（**+6.5%**），耗时几乎严格翻倍。
   显存余量是**冗余，不是机会** —— 每卡算力早已饱和。SmolVLA 激活线性增长这一点
   （§4.4）说明它同样不会例外。
2. **会破坏本 run 的核心价值。** 这份配置是特意与 ACT v2 对齐的
   （同数据集、同 12.0 pass、同有效 batch 64），bs 一改有效 batch 就变，
   ACT vs SmolVLA 的对比就不成立了。

真的想利用余量，等这轮跑完作为**独立实验**做，并相应重调 lr。

### 其他可留意项

- **AdamW 状态是 bf16 而非 fp32。【实测】** PyTorch 的 `exp_avg`/`exp_avg_sq` 用
  `zeros_like(p)`，dtype 跟参数走；expert 是 bf16，所以优化器状态也是 bf16
  （412.66 MB = 2 × 0.206 GB 印证）。bf16 的 `exp_avg_sq` 配 `eps=1e-8` 数值条件偏差，
  是常见坑，**但当前 run 的 loss 曲线（0.395→0.191→0.051 @1177 步）没有异常**，
  不建议中途动它。
- **`max_action_dim=32` 对 7 维动作有 78% 是 padding**（与 coffee_cup 那次 12 维的处理一致）。
  只涉及 1.6M 参数，可忽略，但要知道动作投影有 3/4 在算零。
- **checkpoint 组成**：模型 1.2 GB + 优化器 0.41 GB。`SmolVLA_训练说明.md` 记的
  "单个 ckpt 1.5 GB" 是这两部分之和，与磁盘估算（20 个 ≈ 30 GB）一致。

---

## 7. 复现方法

所有探针都**不需要 GPU 上跑训练**；下面的构造/统计在 CPU 上即可，
只有 §7.3 的内存探针需要一张空闲卡（我用 GPU 0，几个 batch 的瞬时占用）。

### 7.1 参数量 / dtype 拆解（CPU）

关键点：**必须显式设 `load_vlm_weights=False`**，否则会踩 §5 的 `dtype=` 崩溃。
不要用 `SmolVLAConfig.from_pretrained()`（它会读 base config.json 并带 `type` 字段，draccus 会报错）。

```python
import torch, collections
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

cfg = SmolVLAConfig(device="cpu", load_vlm_weights=False, train_expert_only=True,
                    freeze_vision_encoder=True, train_state_proj=True,
                    attention_mode="cross_attn", num_vlm_layers=16,
                    expert_width_multiplier=0.75)
m = SmolVLAPolicy(cfg).model
for k, v in m.state_dict().items():
    print(f"{k:70s} {str(v.dtype):16s} {v.numel()}")
```

### 7.2 交叉验证：落盘 ckpt 的 dtype 分布

```python
import json, struct
p = ".../checkpoints/004000/pretrained_model/model.safetensors"
with open(p, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n))
print(sorted({(v["dtype"], v["data_offsets"][1]-v["data_offsets"][0]) for k, v in hdr.items()}))
```

### 7.3 前向/反向显存探针（需 1 张空闲卡）

```python
import torch
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

torch.cuda.set_device(0)          # 换成空闲卡
B, D, CH, L = 8, 32, 50, 48
cfg = SmolVLAConfig(device="cuda", load_vlm_weights=False, train_expert_only=True,
                    freeze_vision_encoder=True, attention_mode="cross_attn",
                    num_vlm_layers=16, expert_width_multiplier=0.75)
m = SmolVLAPolicy(cfg).model.cuda()

torch.cuda.empty_cache(); torch.cuda.synchronize()
a0 = torch.cuda.memory_allocated()
inp = dict(images=[torch.randn(B, 3, 512, 512, device=0) for _ in range(3)],
           img_masks=[torch.ones(B, dtype=torch.bool, device=0) for _ in range(3)],
           lang_tokens=torch.randint(0, 1000, (B, L), device=0),
           lang_masks=torch.ones(B, L, dtype=torch.bool, device=0),
           state=torch.randn(B, D, device=0), actions=torch.randn(B, CH, D, device=0))
with torch.autocast("cuda", dtype=torch.bfloat16):
    loss = m.forward(**inp).mean()
torch.cuda.synchronize(); print("fwd activ", (torch.cuda.memory_allocated()-a0)/1e9)
loss.backward()
torch.cuda.synchronize(); print("fwd+bwd  ", (torch.cuda.memory_allocated()-a0)/1e9)
```

> §4.2 的 ablation 就是把 `train_state_proj` 换成 `False` 再跑一遍，
> 并额外打印 `m.embed_prefix(...)[0].requires_grad`。

### 7.4 真实占用对照

```bash
nvidia-smi    # 训练中每个 rank 约 4550–4592 MiB
```

### 7.5 权重来源校验（CPU，§1.1 的 bit 级比对）

决定性证据：模型里的参数是否真的来自 `smolvla_base`。**注意用 `p.state_dict()` 而不是
`p.model.state_dict()`** —— 落盘/加载用的是带 `model.` 前缀的那套 key，用 `p.model` 比会
得到 0/500 全不匹配的假阴性。

```python
import torch
from safetensors import safe_open
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

B = "/home/ksa/.cache/modelscope/hub/models/lerobot/smolvla_base"
cfg = SmolVLAConfig(device="cpu", load_vlm_weights=False, train_expert_only=True,
                    freeze_vision_encoder=True, attention_mode="cross_attn",
                    num_vlm_layers=16, expert_width_multiplier=0.75)
p = SmolVLAPolicy.from_pretrained(B, config=cfg)
sd = p.state_dict()
with safe_open(f"{B}/model.safetensors", "pt") as f:
    fkeys = set(f.keys())
print("模型有/文件无（会随机初始化）:", len(set(sd) - fkeys))   # 期望 0
ok = n = 0
with safe_open(f"{B}/model.safetensors", "pt") as f:
    for k in sd:
        a, b = f.get_tensor(k), sd[k]
        n += 1
        ok += bool(a.shape == b.shape and torch.equal(a.to(b.dtype), b))
print(f"bit 级一致 {ok}/{n}")                                    # 期望 500/500
```

> 对照：同目录 `bench/model_param_analysis.md` §3 用的是**另一套**判据（对比
> `checkpoints/100000` 与随机初始化），因为 ACT 那边确实有大量从零训的参数。
> SmolVLA 这里不存在这个问题，一次全量比对就够。

---

## 8. 冻结 VLM 会让 b601 效果变差吗？

**结论：不必然 —— 而且我认为它排在风险列表的最后一位。**

### 8.1 冻结是设计，不是妥协

`train_expert_only=True` 是 lerobot 与 SmolVLA 论文的**默认微调配方**，
论文的整个效率主张（单卡可训、消费级可部署）就建立在这上面。
它**不是**"为了省显存才退而求其次"的降级方案。

结构上，VLM 在这里是**特征提取器**而非任务求解器：它把 3 路图像 + 语言压成
241 个 token 的 hidden state，本体相关的活由那 99.88M 可训 expert 来干。
关键点：**读取冻结特征的 k_proj / v_proj 本身是可训练的**（§4.3），
所以「冻结特征 → b601 动作」这个映射是专门学出来的，不是照搬 SO-101。

### 8.2 风险排序

| # | 风险 | 判断 |
|---|---|---|
| 1 | **223 ep 训 8 阶段长时序任务**（抓杯→放托盘→抓方块→按按钮→等 4 秒→松开→放方块→移杯子） | 我认为**这才是主要风险**，对 ACT 和 SmolVLA 一样难 |
| 2 | 跨本体差距（b601 **7 维** / SO-101 **6 维**，外观与标定都不同） | 真实但可控：action 投影头可训，维度差由 `max_action_dim=32` padding 吸收 |
| 3 | 指令被截断 | 单任务下大概率无害，**多任务时致命**（§9） |
| 4 | 冻结 SigLIP + SmolLM2 | **最不用担心** —— 见 §8.1 |

支撑第 4 点的另一个结构证据：b601 是 hand/front/top 三路（**含腕部视角**），
与预训练数据的 top/wrist/side 拓扑相近，不是毫无重叠的域；
而「夹爪 / 杯子 / 按钮 / 边缘」这类视觉特征跨本体迁移性本来就很好。

> 唯一能定论的还是**真机 rollout**。loss 曲线回答不了这个问题
> （`README.md` 已就 ACT 写过同样的话）。

### 8.3 如果真机确实不行：升级路径（代价从低到高）

1. **先 rollout 定位问题**，别盲目改动
2. `freeze_vision_encoder=False` —— 只解冻 SigLIP（86.4M），代价小
3. `train_expert_only=False` —— 解冻整个 VLM。**显存是够的**（§2：全量微调静态
   3.69 GB + 激活，24 GB 卡放得下）。但 223 ep 喂 450M 参数有过拟合 / 遗忘预训练
   的风险，**可能反而更差**

---

## 9. ⚠️ 指令被静默截断：66 → 48，丢的是任务主目标

查 §8 时实测到的具体缺陷。**跑得通、不报错、loss 正常，但模型读到的指令是残缺的。**

```
任务字符串词数      : 56
完整 tokenize 长度  : 66
tokenizer_max_length: 48            → 截断
tokenizer.truncation_side = left    → 丢的是开头
```

模型**真正读到**的：

```
 up the cube, press the button with the cube (red light on), wait about 4 seconds,
 release the button (red light off), put the cube on the table first, then move the
 cup from the coffee machine to the table
```

**被丢掉的是任务的主目标**：

```
Pick up the paper cup, place it on the silver tray of the coffee machine, pick up
```

路径已确认：`processor_smolvla.py:73-78` 只传了 `max_length=48`、没传 `truncation`
→ 落到 `TokenizerProcessorStep` 的默认 `truncation=True`
（`processor/tokenizer_processor.py:83`），再叠加 tokenizer 自身的
`truncation_side="left"`。

### 为什么这次大概率无害 —— 以及为什么仍要修

本数据集只有 **1 条任务**，所有样本的语言输入是**同一个常量**。模型会很快学到
「语言不携带信息」转而纯靠视觉 —— 截断在单任务下被掩盖了，只是白白浪费容量。

**它是一颗哑弹，会在两种情况下引爆：**

- **多任务训练** —— 语言是唯一区分信号，主目标被截断 = 灾难
- **推理时换指令** —— 换了也不生效（仍被截到同一段残缺文本）

### 顺带：即使不截断，这条指令也在分布外

SmolVLA 预训练时任务文本是被 Qwen2.5-VL **改写成简短动词短语**的【文献，§1.1】。
56 词的多阶段长句本来就不是预训练见过的形态。

### 修法（按推荐顺序）

1. **改数据集 `meta/tasks.parquet` 里的任务串**，改成简短单句
   （如 `Move the paper cup from the coffee machine to the table`）—— 治本，单/多任务都好
2. 调大 `tokenizer_max_length` —— **有风险**：base 是在 48 上预训练并验证的，
   位置编码 / 注意力分布都按这个长度调过，改大可能引入新的分布外问题
3. 什么都不做 —— 仅当确定**永远单任务**

> 注：方案 1 改的是数据集本身，需要连带走一遍校验（`bench/verify_dataset.py`）
> 并对齐 `meta` 的统计。

复现该结论：

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
print(tok.truncation_side)                       # left
ids = tok(task, add_special_tokens=False)["input_ids"]
print(len(ids))                                  # 66
print(tok.decode(ids[:48]))                      # 会丢掉开头的模型输入 = ids[-48:]
```

---

## 附：关键数字速查

```
总参数            450.0M        常驻权重 1.198 GB（fp32/bf16 混合）
预训练覆盖         450.0M (100%)   与 smolvla_base 逐比特相同，零随机初始化
可训参数           99.88M (22.2%)  只有 action expert + 投影头
冻结              350.17M (77.8%)  SigLIP + SmolLM2 + lm_head
                └ lm_head 47.3M 在 forward 里**从不执行**（§1.2），冻结无影响
（对照 ACT：预训练 21.6% / 从零训 78.4% —— 两者范式相反）

单卡显存（bs=8）    ≈ 4.5 GB      nvidia-smi 实测 4550–4592 MiB
  ├ 权重            1.21 GB
  ├ 前向激活        2.14 GB   ← 最大头，且冻结省不掉（§4）
  ├ 梯度            0.32 GB
  ├ AdamW           0.41 GB   （= 落盘 optimizer_state 412.66 MB）
  └ context/碎片   ~0.5 GB

激活随 batch       严格线性：bs8 2.14 / bs16 4.24 / bs32 8.41 GB
prefix             241 tokens = 3×64(图) + 48(语言) + 1(state)，图占 80%
suffix (expert)    50 tokens
指令               meta/tasks.parquet 里 66 token → 被 tokenizer_max_length=48 截断
                  （truncation_side=left，丢掉开头含任务主目标）见 §9

optimizer          lr 5e-5, betas (0.9,0.95), eps 1e-8, wd 1e-10, clip 10
scheduler          cosine 1000 warmup → 100000 decay → 2.5e-6
```
