#!/bin/bash
# 调试模式训练脚本 - 使用更稳定的配置
cd /home/ksa/lerobot/self_scripts/

# 启用CUDA调试模式 会导致训练机器慢，只用用于调试
# export CUDA_LAUNCH_BLOCKING=1
# export TORCH_USE_CUDA_DSA=1

# 使用稳定配置
echo "使用稳定配置开始训练（降低KL权重、减小batch size、更保守的学习率）"
accelerate launch --multi_gpu --num_processes=4 $(which lerobot-train) \
    --config_path=train_config_act_stable.json
