#!/bin/bash
# 启动 Pi0.5 训练（使用预训练模型）
cd /home/ksa/lerobot/self_scripts/

CHECKPOINT_CONFIG="output_lerobot_train/pi05_cap_pen_lora/checkpoints/last/pretrained_model/train_config.json"
# export HF_ENDPOINT=https://hf-mirror.com

if [ -f "$CHECKPOINT_CONFIG" ]; then
    echo "检测到checkpoint，将从上次训练继续"
    accelerate launch --multi_gpu --num_processes=8 $(which lerobot-train) --config_path="$CHECKPOINT_CONFIG" --resume=true
else
    echo "未检测到checkpoint，开始新训练（使用预训练权重）"
    accelerate launch --multi_gpu --num_processes=8 $(which lerobot-train) \
        --policy.pretrained_path=/home/ksa/.cache/modelscope/models/lerobot--pi05_base/snapshots/master \
        --config_path=/home/ksa/lerobot/self_scripts/stage3_train/configs/pi05_train_config.json
fi