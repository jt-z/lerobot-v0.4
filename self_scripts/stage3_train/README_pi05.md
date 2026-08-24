# Pi0.5 训练配置说明

## 环境要求

Pi0.5 需要特殊版本的 transformers 库：

```bash
pip install "transformers @ git+https://github.com/huggingface/transformers.git@fix/lerobot_openpi" scipy
```

## 文件说明

- **配置文件**: [configs/pi05_train_config.json](configs/pi05_train_config.json)
- **启动脚本**: [start_train_pi05.sh](start_train_pi05.sh)

## 主要配置参数

### 模型配置
- `pretrained_path`: `lerobot/pi05_base` - 使用官方预训练模型
- `dtype`: `bfloat16` - 混合精度训练
- `compile_model`: `true` - 启用模型编译加速
- `gradient_checkpointing`: `true` - 减少显存占用
- `freeze_vision_encoder`: `false` - 不冻结视觉编码器
- `train_expert_only`: `false` - 训练所有参数

### 训练配置
- `steps`: `3000` - Pi0.5 推荐的微调步数
- `batch_size`: `8` - 每个 GPU 的批次大小
- `save_freq`: `500` - 每 500 步保存一次 checkpoint

### 数据标准化
Pi0.5 使用 QUANTILES 标准化（与 SmolVLA 的 MEAN_STD 不同）：
```json
"normalization_mapping": {
  "ACTION": "QUANTILES",
  "STATE": "QUANTILES", 
  "VISUAL": "IDENTITY"
}
```

## 训练步数说明

Pi0.5 推荐 3000 步是因为：
1. **充分预训练**: Pi0.5 在大规模异构数据上预训练（多模态网络数据、跨embodiment机器人数据等）
2. **强泛化能力**: 基于 Gemma 的 VLM + Action Expert 架构
3. **微调任务**: 3000 步适用于在新数据集上的微调

根据数据集规模可调整：
- 小数据集/简单任务: 3000-5000 步
- 中等数据集: 5000-10000 步
- 大数据集/复杂任务: 10000-20000 步

## 使用方法

```bash
cd /home/ksa/lerobot/self_scripts/stage3_train
./start_train_pi05.sh
```

脚本会自动检测 checkpoint：
- 如果存在 checkpoint，则继续训练
- 否则从预训练模型开始新训练

## 与 SmolVLA 的主要区别

| 特性 | Pi0.5 | SmolVLA |
|------|-------|---------|
| 预训练模型 | lerobot/pi05_base | 本地 smolvla_base |
| 数据标准化 | QUANTILES | MEAN_STD |
| 推荐步数 | 3000 | 20000 |
| 训练类型 | 微调 | 较长训练 |
| transformers 版本 | 特殊分支 | 标准版本 |

## 注意事项

1. **transformers 版本冲突**: Pi0.5 的 transformers 与其他模型（SmolVLA、GROOT 等）冲突，不能同时使用
2. **显存要求**: 使用 8 GPU 训练，batch_size=8，需要足够的显存
3. **数据集要求**: 如果数据集没有 quantiles 统计，需要先转换或使用 MEAN_STD 标准化
