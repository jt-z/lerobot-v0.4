#!/bin/bash
# 简化的训练启动脚本
# 用法:
#   ./train.sh configs/stable.json                    # 使用指定配置，8 GPU
#   ./train.sh configs/stable.json --gpus 4           # 4 GPU
#   ./train.sh configs/stable.json --gpus 1 --gpu 7   # 单GPU（GPU 7）
#   ./train.sh configs/stable.json --resume           # 从checkpoint恢复
#    ./train.sh  /home/ksa/lerobot/self_scripts/stage3_train/configs/stable.json

set -e

# 激活 conda 环境
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "lerobot" ]; then
    echo "激活 lerobot conda 环境..."
    eval "$(conda shell.bash hook)"
    conda activate lerobot
fi

# 默认参数
CONFIG_FILE=""
NUM_GPUS=8
GPU_ID=""
RESUME=false
DEBUG=false

# 解析参数
if [[ $# -eq 0 ]] || [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    echo "用法: $0 <config_file> [选项]"
    echo ""
    echo "参数:"
    echo "  config_file          配置文件路径（如 configs/stable.json）"
    echo ""
    echo "选项:"
    echo "  --gpus N            使用 N 张 GPU（默认：8）"
    echo "  --gpu ID            使用单个GPU（如 --gpu 7）"
    echo "  --resume            从最新checkpoint恢复训练"
    echo "  --debug             启用CUDA调试模式"
    echo ""
    echo "示例:"
    echo "  $0 configs/stable.json"
    echo "  $0 configs/stable.json --gpus 4"
    echo "  $0 configs/stable.json --gpus 1 --gpu 7"
    echo "  $0 configs/stable.json --resume"
    echo ""
    echo "可用配置:"
    if [ -d "configs" ]; then
        ls -1 configs/*.json 2>/dev/null | sed 's/^/  /'
    fi
    echo ""
    echo "创建新配置:"
    echo "  ./make_config.py configs/base.json --set dataset.repo_id=hellozjt/my_task -o configs/my_task.json"
    exit 0
fi

CONFIG_FILE="$1"
shift

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --gpu)
            GPU_ID="$2"
            NUM_GPUS=1
            shift 2
            ;;
        --resume)
            RESUME=true
            shift
            ;;
        --debug)
            DEBUG=true
            shift
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

# 检查配置文件（支持相对路径和绝对路径）
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 配置文件不存在: $CONFIG_FILE"
    echo ""
    echo "可用配置:"
    if [ -d "configs" ]; then
        ls -1 configs/*.json 2>/dev/null | sed 's/^/  /'
    fi
    exit 1
fi

# 转换为绝对路径
CONFIG_FILE=$(realpath "$CONFIG_FILE")

# 启用调试模式
if [ "$DEBUG" = true ]; then
    echo "⚠️  启用 CUDA 调试模式（训练会变慢）"
    export CUDA_LAUNCH_BLOCKING=1
    export TORCH_USE_CUDA_DSA=1
fi

# 设置GPU
GPU_PREFIX=""
if [ -n "$GPU_ID" ]; then
    GPU_PREFIX="CUDA_VISIBLE_DEVICES=$GPU_ID"
    echo "使用 GPU $GPU_ID"
fi

# 处理resume
RESUME_FLAG=""
if [ "$RESUME" = true ]; then
    echo "✓ 从checkpoint恢复训练模式"
    RESUME_FLAG="--resume=true"
fi

# 显示配置
echo ""
echo "========================================="
echo "训练配置:"
echo "  配置文件: $CONFIG_FILE"
echo "  GPU 数量: $NUM_GPUS"
if [ -n "$GPU_ID" ]; then
    echo "  GPU ID:   $GPU_ID"
fi
echo "  恢复训练: $RESUME"
echo "  调试模式: $DEBUG"
echo "========================================="
echo ""

# 构建训练命令
if [ "$NUM_GPUS" -eq 1 ]; then
    TRAIN_CMD="$GPU_PREFIX accelerate launch --num_processes=1 $(which lerobot-train) --config_path=$CONFIG_FILE $RESUME_FLAG"
else
    TRAIN_CMD="$GPU_PREFIX accelerate launch --multi_gpu --num_processes=$NUM_GPUS $(which lerobot-train) --config_path=$CONFIG_FILE $RESUME_FLAG"
fi

echo "执行命令: $TRAIN_CMD"
echo ""

# 执行训练
eval $TRAIN_CMD
