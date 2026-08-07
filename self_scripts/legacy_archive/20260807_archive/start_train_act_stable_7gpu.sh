#!/bin/bash
# 使用7张GPU训练（GPU0已失效）
cd /home/ksa/lerobot/self_scripts/

echo "使用稳定配置开始训练（7张GPU，降低KL权重、减小batch size）"
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch --multi_gpu --num_processes=7 $(which lerobot-train) \
    --config_path=train_config_act_stable.json
