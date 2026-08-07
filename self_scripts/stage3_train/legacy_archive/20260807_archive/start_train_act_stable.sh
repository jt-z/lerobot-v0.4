#!/bin/bash
# 使用稳定配置正常训练（不开启调试模式）
cd /home/ksa/lerobot/self_scripts/

echo "使用稳定配置开始训练（降低KL权重、减小batch size、更保守的学习率）"
accelerate launch --multi_gpu --num_processes=8 $(which lerobot-train) \
    --config_path=train_config_act_stable.json
