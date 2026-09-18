# ACT 模型参数构成分析 —— 哪些是预训练、哪些从零训

**分析对象**：`b601_20260910_164106` 单臂 ACT，`checkpoints/100000`（100K 步 run 的产物）
**日期**：2026-09-17
**关联**：`README.md`（训练结果）、`multigpu_scaling_analysis.md`（多卡/通信）

> 所有数字均为**实测**（从 `model.safetensors` 与实例化后的模型统计得出），
> 复现脚本见 §5。少数标注【推算】的除外。

---

## 0. 结论摘要

| 问题 | 结论 |
|---|---|
| 哪些是预训练权重？ | **只有 ResNet18 backbone，11,166,912 个（21.6%）** |
| 哪些是从零训练？ | **其余全部，40,432,327 个（78.4%）** |
| 主要是 transformer 部分从零训吗？ | **是，而且比直觉更彻底** —— transformer 相关部分合计 78.4% |
| 预训练部分被冻结了吗？ | **没有**，backbone 在微调，只是用 1/8 的学习率 |
| 有没有真正冻结的东西？ | **有** —— backbone 的全部 BatchNorm（15,616 个值） |

**一句话**：这个模型有超过 **3/4 的参数是随机初始化后训出来的**，只有 21.6% 享受了
ImageNet 迁移；而且那 21.6% 里的 BN 参数还完全没动。

---

## 1. 参数量精确拆解【实测】

总可训参数 **51,599,239**（与训练日志 `num_learnable_params=51599239` 完全吻合）：

| 部件 | 参数量 | 占比 | 初始化 | 含哪些子模块 |
|---|---|---|---|---|
| Transformer **encoder** | 17,617,408 | **34.1%** | 🔴 随机 | `encoder.layers` + 各类 `*_input_proj` |
| **VAE encoder** | 17,374,272 | **33.7%** | 🔴 随机 | `vae_encoder.layers` + `cls_embed` + 投影层 |
| ResNet18 **backbone**（卷积） | 11,166,912 | **21.6%** | 🟢 **ImageNet** | `conv1` + `layer1`–`layer4` |
| Transformer **decoder** | 5,437,056 | **10.5%** | 🔴 随机 | `decoder.layers` + `decoder_pos_embed` + `norm` |
| `action_head` | 3,591 | 0.007% | 🔴 随机 | 输出层 |
| **合计** | **51,599,239** | 100% | | |

```
从零训练 :  40,432,327  =  78.4%
预训练迁移:  11,166,912  =  21.6%
```

**注意**：encoder / decoder / vae_encoder / backbone 这几行的数字**已经包含了各自的
投影层和位置编码**，不要重复相加。

> ⚠️ 一个反直觉的对称性：**encoder（34.1%）和 VAE encoder（33.7%）几乎一样大**。
> 因为 `n_vae_encoder_layers == n_encoder_layers == 4`，两者是同构的 transformer。
> 而 VAE encoder 推理时根本不用（见 §4）。

---

## 2. backbone 预训练了，但**没有冻结**

`modeling_act.py:75-90` 把参数分成两组，用**不同的学习率**：

```python
self.optimizer_grouped_parameters = [
    {
        "params": [p for n, p in self.named_parameters()
                   if not n.startswith("model.backbone") and p.requires_grad]
        #                                  ^^^^^^^^^^^^^^ lr = optimizer_lr
    },
    {
        "params": [p for n, p in self.named_parameters()
                   if n.startswith("model.backbone") and p.requires_grad],
        "lr": self.config.optimizer_lr_backbone,
    },
]
```

| 组 | lr | 理由 |
|---|---|---|
| backbone 之外全部 | **8e-5** | 从零训，需要正常学习率 |
| backbone | **1e-5** | 已预训练，小 lr 保护特征、避免灾难性遗忘 |

实测**冻结参数 = 0**：backbone 的 11.2M 卷积权重全都 `requires_grad=True`，在微调。

> 这个 8× 的差分 lr 是为「预训练 vs 从零训」设计的，但它同时意味着
> **从零训的部分以 8 倍速度在学**。`multigpu_scaling_analysis.md` §3.2 里提到
> `optimizer_lr=8e-5` 是配置里离 ACT 参考实现最远的一项（默认 1e-5），
> 根因就在这里。

---

## 3. 真正被冻结的：backbone 的全部 BatchNorm

构造 backbone 时传了 `norm_layer=FrozenBatchNorm2d`（`modeling_act.py:325`）：

```python
backbone_model = getattr(torchvision.models, config.vision_backbone)(
    replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
    weights=config.pretrained_backbone_weights,      # ResNet18_Weights.IMAGENET1K_V1
    norm_layer=FrozenBatchNorm2d,                    # <-- 关键
)
```

实测结果：

```
backbone 卷积权重(可训)   : 11,166,912
backbone BN 可训参数      : 0            ← 一个都没有
backbone BN buffer(冻结)  : 15,616
```

torchvision 的 `FrozenBatchNorm2d` 把 BN 的 **`weight` / `bias` / `running_mean` /
`running_var` 全部注册成 `buffer` 而不是 `parameter`**。实测确认它们全部落在
`named_buffers()` 里，因此：

- 这四个量**全部冻结在 ImageNet 原值**，一个都不更新
- 它们仍然会存进 checkpoint（`safetensors` 比可训参数多 **71,424** 个值，就是这些 buffer）

**这不是偷懒，是必需的**：本任务 batch 只有 8/卡，如果 BN 的 running stats 还在滚动更新，
统计量会被小 batch 严重污染、训练不稳。冻结是正确做法。

---

## 4. VAE encoder 占 33.7%，但**推理时完全不执行**

门控在 `modeling_act.py:397`：

```python
if self.config.use_vae and self.training:        # ← 只有训练时进这个分支
    ...
    cls_token_out = self.vae_encoder(...)
    latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
    mu = latent_pdf_params[:, : self.config.latent_dim]
    log_sigma_x2 = latent_pdf_params[:, self.config.latent_dim :]
    # 重参数化采样
    latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
else:
    # When not using the VAE encoder, we set the latent to be all zeros.
    latent_sample = torch.zeros([batch_size, self.config.latent_dim]).to(...)
```

**推理时 `latent` 恒为 0，VAE encoder 一次都不跑。**

含义：

| | 训练时 | 推理时 |
|---|---|---|
| VAE encoder（17,374,272，33.7%） | ✅ 执行 | ❌ 不执行 |
| `latent` | 从 VAE 分布采样 | 全 0（确定性策略） |
| 实际参与前向的参数 | 51,599,239 | **34,224,967（66.3%）** |

这就是 ACT 论文里「推理用确定性策略 z=0」的实现 —— VAE 只是训练期的正则化手段。

**实用推论**：

- **模型有 1/3 的参数只在训练时存在**，部署时是纯死重
- checkpoint 里这 17.4M × 4B ≈ **70 MB** 白白存着
- 若要导出部署模型，可以把整个 `vae_encoder.*` 剪掉，体积减少约 34%

---

## 5. 复现方法

```python
import json, collections, torch
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.configs.types import FeatureType, PolicyFeature

CKPT = "output_lerobot_train/b601_20260910_164106_act/checkpoints/100000/pretrained_model"
d = json.load(open(f"{CKPT}/train_config.json"))
cfg = ACTConfig(**{k: v for k, v in d["policy"].items()
                   if k in ACTConfig.__dataclass_fields__})
cfg.input_features = {
    "observation.state":        PolicyFeature(type=FeatureType.STATE,  shape=(7,)),
    "observation.images.hand":  PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
    "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
    "observation.images.top":   PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
}
cfg.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))}
pol = ACTPolicy(cfg)

def grp(n):
    for p in ["backbone", "encoder", "decoder", "vae_encoder", "action_head"]:
        if f"model.{p}" in n:
            return p
    return "other"

P, T, B = collections.Counter(), collections.Counter(), collections.Counter()
for n, p in pol.named_parameters():
    P[grp(n)] += p.numel()
    T[grp(n)] += p.numel() if p.requires_grad else 0
for n, b in pol.named_buffers():
    B[grp(n)] += b.numel()

for g in ["backbone", "encoder", "vae_encoder", "decoder", "action_head"]:
    print(f"{g:12s} trainable={P[g]:>12,}  frozen={P[g]-T[g]:>8,}  buffers={B[g]:>8,}")
print(f"{'TOTAL':12s} trainable={sum(P.values()):>12,}")
```

> 注意 `grp()` 的匹配顺序很重要：`model.encoder` 是 `model.vae_encoder` 的子串问题
> 不存在（前缀不同），但 `model.encoder` 会匹配到 `model.encoder_*` 的所有投影层 ——
> 这是**有意的**，投影层确实属于 encoder 组。

---

## 6. 与训练曲线的呼应

78.4% 从零初始化这个比例，直接解释了 `act_100k.log` 里的两个现象：

**1. 起始梯度范数 111.9，前 5000 步一直被裁剪**

```
step 200 : grdn:111.924    ← 大部分网络是随机的，早期梯度自然巨大
step 100K: grdn:1.183
```

`grad_clip_norm=10` 在 step ≤ 5000 的 23 条日志里**全部触发**，5000 步之后
（末条 10.11）再没超过阈值。用裁剪硬稳住前 5K 步，是 ACT 从零训的常态，不是问题。

**2. 学习几乎全发生在前 10K 步**

```
0–10K   : loss 4.082 → 0.163   ← 一个多 pass 吃掉绝大部分收益
10–70K  : 0.162 → 0.095        ← 慢速精修
70–100K : 0.080 → 0.077        ← 平台
```

先快速拟合随机初始化的部分，之后才是精修 —— 与「大部分参数从零开始」的图像一致。

---

## 7. 对后续工作的含义

| 场景 | 含义 |
|---|---|
| **想换任务/数据集重训** | 只有 21.6% 可复用（backbone），**没有"微调大模型"那种省算力的空间** |
| **部署** | 可剪掉 `vae_encoder.*`，模型小 34%，推理结果**完全不变**（z 恒为 0） |
| **想减少训练量** | 从零训的比例决定了 100K 步是合理量级；若换更小的数据集，可参考 pass 数而非绝对步数 |
| **想加数据继续训** | 属于续训（`--resume`），不是从零 —— 78.4% 的参数已经学到东西了 |
| **上真机的显存** | 推理只需 34.2M 参数（66.3%），比训练时轻 |

---

## 附：关键数字速查

```
总可训参数            51,599,239
  ├─ 从零训练         40,432,327  (78.4%)
  │    ├─ encoder     17,617,408  (34.1%)
  │    ├─ vae_encoder 17,374,272  (33.7%)   ← 推理不用
  │    ├─ decoder      5,437,056  (10.5%)
  │    └─ action_head      3,591  (0.007%)
  └─ ImageNet 预训练  11,166,912  (21.6%)   ← 微调，非冻结
       └─ 其中 BN      15,616 个值真冻结（buffer，非 parameter）

冻结 parameter        0
buffer（不可训）      71,424
推理实际参与          34,224,967  (66.3%)
checkpoint 文件       206,707,932 B ≈ 197 MB (fp32)
```
