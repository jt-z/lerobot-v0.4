#!/bin/bash
# 启动训练：单臂 seeed_b601_rs_follower 的 ACT 模型（make coffee 任务）
#
# 注意：这是「单臂」训练，与 stage3_train/dataset_new_makecoffee_train/ 下的
# 「双臂」训练（bi_b601_so101_follower）是完全独立的两套，互不影响。
#   - 双臂: 13维 action/state, 4路相机 (left_hand/left_top/left_front/right_hand)
#   - 单臂:  7维 action/state, 3路相机 (hand/front/top)   <- 本脚本
# 两者本体不同，checkpoint 不通用，不能互相续训。

cd /home/ksa/lerobot/self_scripts/

# 激活 conda 环境
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "lerobot" ]; then
    echo "激活 lerobot conda 环境..."
    eval "$(conda shell.bash hook)"
    conda activate lerobot
fi

# 使用HF国内镜像，避免 huggingface.co 网络不可达
export HF_ENDPOINT=https://hf-mirror.com

# 从头训练。本数据集是新的单臂本体，没有可续训的 checkpoint。
# 将来要续训时，把下面这行取消注释即可（指向本次自己产生的 checkpoint）：
# CHECKPOINT_CONFIG="output_lerobot_train/b601_20260910_164106_act/checkpoints/last/pretrained_model/train_config.json"

if [ -f "$CHECKPOINT_CONFIG" ]; then
    echo "检测到checkpoint，将从checkpoint继续"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --num_processes=8 $(which lerobot-train) --config_path="$CHECKPOINT_CONFIG" --resume=true
else
    echo "未检测到checkpoint，开始新训练（单臂 b601_rs）"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --num_processes=8 $(which lerobot-train) --config_path=stage3_train/dataset_b601_rs_makecoffee_train/act_train_config.json
fi
