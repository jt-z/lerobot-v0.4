#!/bin/bash
# 启动 SmolVLA 训练（支持从checkpoint继续训练）
# 数据集: hellozjt/coffee_cup_button_20260826_232220 (4路摄像头: left_hand/left_top/left_front/right_hand, 12维动作)
cd /home/ksa/lerobot/self_scripts/

# 激活 conda 环境
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "lerobot" ]; then
    echo "激活 lerobot conda 环境..."
    eval "$(conda shell.bash hook)"
    conda activate lerobot
fi

# 使用HF国内镜像，避免 huggingface.co 网络不可达
export HF_ENDPOINT=https://hf-mirror.com

OUTPUT_DIR="output_lerobot_train/task2_make_coffee_new_dataset/smolvla_coffee_cup_button_20260826_232220"
CHECKPOINT_CONFIG="$OUTPUT_DIR/checkpoints/last/pretrained_model/train_config.json"
LOG_FILE="$OUTPUT_DIR/logs/train_smolvla.log"

# 保证 tee 管道退出码等于 lerobot-train 的退出码（而非 tee 的）
set -o pipefail
mkdir -p "$OUTPUT_DIR/logs"

if [ -f "$CHECKPOINT_CONFIG" ]; then
    echo "检测到checkpoint，将从上次训练继续（日志: $LOG_FILE）"
    accelerate launch --multi_gpu --num_processes=8 $(which lerobot-train) --config_path="$CHECKPOINT_CONFIG" --resume=true 2>&1 | tee -a "$LOG_FILE"
else
    echo "未检测到checkpoint，开始新训练（日志: $LOG_FILE）"
    # 注意：
    # 1) 使用 --policy.pretrained_path（而非 --policy.path）加载 smolvla_base 权重，
    #    这样 policy 的输入特征（4路摄像头、12维state/action）会按数据集自动推断，
    #    与之前 task2_make_coffee/smolvla_cap_pen_and_put_into_holder 的成功训练配置一致。
    # 2) 本数据集4路摄像头名与模型输入完全匹配，无需 --rename_map 和 --policy.empty_cameras。
    accelerate launch --multi_gpu --num_processes=8 $(which lerobot-train) \
        --policy.pretrained_path=/home/ksa/.cache/modelscope/hub/models/lerobot/smolvla_base \
        --config_path=/home/ksa/lerobot/self_scripts/stage3_train/dataset_new_makecoffee_train/smolvla_train_config.json \
        --policy.push_to_hub=false 2>&1 | tee -a "$LOG_FILE"
fi
