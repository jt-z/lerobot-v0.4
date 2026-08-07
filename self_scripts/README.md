# LeRobot 训练脚本使用指南

## 设计理念

**简单、透明、可验证**
- ✅ 配置文件可见（`configs/` 目录）
- ✅ 脚本简单（只处理 GPU 配置）
- ✅ 易于调试（知道用的是哪个配置）
- ✅ 避免冗余（基础配置 + 工具生成变体）

## 快速开始

### 1. 使用预设配置

```bash
# 稳定训练配置（推荐）
./train.sh configs/stable.json

# 基础配置
./train.sh configs/base.json

# 无VAE配置
./train.sh configs/no-vae.json
```

### 2. GPU 配置

```bash
# 使用 4 GPU
./train.sh configs/stable.json --gpus 4

# 使用单个 GPU（GPU 7）
./train.sh configs/stable.json --gpus 1 --gpu 7

# 调试模式
./train.sh configs/stable.json --gpus 1 --gpu 7 --debug
```

### 3. 从 checkpoint 恢复

```bash
./train.sh configs/stable.json --resume
```

## 配置文件管理

### 预设配置（`configs/` 目录）

- **base.json** - 基础配置（batch_size=32, lr=8e-5）
- **stable.json** - 稳定训练配置（batch_size=16, lr=5e-5, 梯度裁剪）
- **no-vae.json** - 禁用VAE配置

### 创建新配置

**方法1：从现有配置复制修改（推荐）**

```bash
# 复制并编辑
cp configs/stable.json configs/my_task.json
vim configs/my_task.json  # 修改 dataset.repo_id, output_dir 等

# 使用
./train.sh configs/my_task.json
```

**方法2：使用 make_config.py 工具**

```bash
# 从基础配置生成新配置
./make_config.py configs/base.json \
    --set dataset.repo_id=hellozjt/my_new_task \
    --set dataset.root=/home/ksa/.cache/huggingface/lerobot/hellozjt/my_new_task \
    --set output_dir=output_lerobot_train/my_new_task \
    --set batch_size=16 \
    -o configs/my_new_task.json

# 查看但不保存
./make_config.py configs/stable.json --set batch_size=8 --dry-run

# 从稳定配置修改
./make_config.py configs/stable.json \
    --set dataset.repo_id=hellozjt/another_task \
    --set batch_size=8 \
    -o configs/another_task.json
```

## 常见场景

### 场景1：新任务训练

```bash
# 复制稳定配置
cp configs/stable.json configs/cap_pen.json

# 编辑配置文件，修改：
# - dataset.repo_id
# - dataset.root
# - output_dir
vim configs/cap_pen.json

# 开始训练
./train.sh configs/cap_pen.json --gpus 8
```

### 场景2：调试训练问题

```bash
# 单GPU + 调试模式
./train.sh configs/stable.json --gpus 1 --gpu 7 --debug
```

### 场景3：GPU 0 损坏

```bash
# 使用其他GPU（比如GPU 7）
./train.sh configs/stable.json --gpus 1 --gpu 7
```

### 场景4：快速实验不同batch size

```bash
# 生成临时配置
./make_config.py configs/stable.json --set batch_size=8 -o configs/stable_bs8.json

# 训练
./train.sh configs/stable_bs8.json --gpus 4
```

## 配置文件示例

### 最小配置示例

```json
{
  "dataset": {
    "repo_id": "hellozjt/my_dataset",
    "root": "/home/ksa/.cache/huggingface/lerobot/hellozjt/my_dataset",
    "revision": "main",
    "streaming": false
  },
  "policy": {
    "type": "act",
    "device": "cuda",
    "push_to_hub": false,
    "optimizer_lr": 8e-5
  },
  "output_dir": "output_lerobot_train/my_dataset",
  "job_name": "training_job",
  "steps": 37500,
  "batch_size": 32,
  "save_freq": 2000
}
```

### 稳定配置示例

```json
{
  "dataset": { ... },
  "policy": {
    "type": "act",
    "device": "cuda",
    "push_to_hub": false,
    "optimizer_lr": 5e-5,
    "use_vae": true,
    "kl_weight": 1.0,
    "n_vae_encoder_layers": 2
  },
  "optimizer": {
    "grad_clip_norm": 5.0
  },
  "output_dir": "output_lerobot_train/demo_data_dual_stable",
  "job_name": "training_job",
  "steps": 37500,
  "batch_size": 16,
  "save_freq": 2000
}
```

## 工具脚本

### 数据集工具

```bash
# 检查数据集有效性
python check_dataset_validity.py

# 转换为视频格式（压缩）
python convert_to_video_format.py
```

### GPU 重置

```bash
# GPU卡住时使用
./reset_gpu_and_train.sh
```

## 优势

✅ **透明**: 配置文件清晰可见，训练前知道用什么参数  
✅ **简单**: 启动脚本只处理GPU逻辑，没有复杂判断  
✅ **灵活**: 需要新配置就复制修改，或用工具生成  
✅ **少量维护**: 只保留3-5个常用配置，不算冗余  
✅ **易调试**: 出问题时配置文件在那里，容易排查

## 文件结构

```
self_scripts/
├── configs/                    # 配置文件目录
│   ├── base.json              # 基础配置
│   ├── stable.json            # 稳定配置
│   ├── no-vae.json            # 无VAE配置
│   └── my_task.json           # 你的任务配置...
├── train.sh                   # 训练启动脚本（简化版）
├── make_config.py             # 配置生成工具
├── check_dataset_validity.py # 数据集验证
├── convert_to_video_format.py # 格式转换
└── legacy_archive/            # 旧文件归档
```

## 迁移说明

如果你之前有配置文件，直接移到 `configs/` 目录即可：

```bash
mv train_config_*.json configs/
```

然后用新的方式调用：

```bash
# 旧方式
./start_train_act_stable.sh

# 新方式
./train.sh configs/stable.json
```
