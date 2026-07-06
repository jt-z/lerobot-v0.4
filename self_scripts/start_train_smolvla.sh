#!/bin/bash
# 启动 SmolVLA 训练（支持从checkpoint继续训练）
cd /home/ksa/lerobot/self_scripts/

CHECKPOINT_CONFIG="output_lerobot_train/smolvla_put_ball2cup/checkpoints/last/pretrained_model/train_config.json"

if [ -f "$CHECKPOINT_CONFIG" ]; then
    echo "检测到checkpoint，将从上次训练继续"
    accelerate launch --multi_gpu --num_processes=8 $(which lerobot-train) --config_path="$CHECKPOINT_CONFIG" --resume=true
else
    echo "未检测到checkpoint，开始新训练"
    accelerate launch --multi_gpu --num_processes=8 $(which lerobot-train) \
        --policy.path=/home/ksa/.cache/modelscope/hub/models/lerobot/smolvla_base \
        --config_path=/home/ksa/lerobot/self_scripts/train_config_smolvla.json
fi