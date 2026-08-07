#!/bin/bash
# 归档冗余的旧脚本和配置文件到 legacy_archive/ 目录
# 使用前请先确认新的 train.sh 工作正常

set -e
cd /home/ksa/lerobot/self_scripts/

# 创建存档目录（带时间戳）
ARCHIVE_DIR="legacy_archive/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$ARCHIVE_DIR"

echo "========================================="
echo "归档冗余文件"
echo "========================================="
echo ""
echo "存档目录: $ARCHIVE_DIR"
echo ""

# 要删除的文件列表
LEGACY_SCRIPTS=(
    "start_train_act_stable.sh"
    "start_train_act_debug.sh"
    "start_train_act_stable_7gpu.sh"
)

LEGACY_CONFIGS=(
    "train_config_act_stable.json"
    "train_config_act_no_vae.json"
    "train_config_act_put_ball2cup.json"
    "train_config_act_two_simulate.json"
)

OPTIONAL_FILES=(
    "start_train_act.sh"
    "train_config_act.json"
)

echo "将要归档以下文件："
echo ""
echo "冗余训练脚本:"
for file in "${LEGACY_SCRIPTS[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file"
    fi
done

echo ""
echo "冗余配置文件:"
for file in "${LEGACY_CONFIGS[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file"
    fi
done

echo ""
echo "可选归档（如果不再使用）:"
for file in "${OPTIONAL_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  ? $file"
    fi
done

echo ""
read -p "确认归档以上文件？(y/N): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "取消操作"
    rmdir "$ARCHIVE_DIR" 2>/dev/null || true
    exit 0
fi

# 归档冗余脚本
echo ""
echo "归档冗余训练脚本..."
for file in "${LEGACY_SCRIPTS[@]}"; do
    if [ -f "$file" ]; then
        mv -v "$file" "$ARCHIVE_DIR/"
    fi
done

# 归档冗余配置
echo ""
echo "归档冗余配置文件..."
for file in "${LEGACY_CONFIGS[@]}"; do
    if [ -f "$file" ]; then
        mv -v "$file" "$ARCHIVE_DIR/"
    fi
done

# 询问是否归档可选文件
echo ""
read -p "是否也归档 start_train_act.sh 和 train_config_act.json？(y/N): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    for file in "${OPTIONAL_FILES[@]}"; do
        if [ -f "$file" ]; then
            mv -v "$file" "$ARCHIVE_DIR/"
        fi
    done
fi

# 创建归档说明文件
cat > "$ARCHIVE_DIR/README.txt" << 'EOF'
这些文件已被新的统一训练脚本系统替代

新系统包括:
- train.sh: 统一的训练启动脚本
- config_generator.py: 动态配置生成器

旧脚本与新命令的对照:
- start_train_act_stable.sh       -> ./train.sh --preset stable
- start_train_act_debug.sh        -> ./train.sh --preset stable --gpus 4 --debug
- start_train_act_stable_7gpu.sh  -> ./train.sh --preset stable --gpus 7 --gpu-ids 1,2,3,4,5,6,7
- start_train_act.sh              -> ./train.sh

配置文件现在通过 config_generator.py 动态生成，无需手动维护 JSON 文件。

如需恢复这些文件，只需将它们移回上级目录即可。

归档时间: $(date)
EOF

echo ""
echo "========================================="
echo "✓ 归档完成"
echo "========================================="
echo ""
echo "归档位置: $ARCHIVE_DIR"
echo ""
echo "保留的文件："
echo "  - train.sh (统一训练脚本)"
echo "  - config_generator.py (配置生成器)"
echo "  - check_dataset_validity.py (数据集验证)"
echo "  - convert_to_video_format.py (格式转换)"
echo "  - reset_gpu_and_train.sh (GPU重置)"
echo "  - 其他工具脚本..."
echo ""
echo "现在可以使用 ./train.sh 进行训练了！"
echo "查看帮助: ./train.sh --help"
echo ""
echo "如需恢复归档文件："
echo "  mv $ARCHIVE_DIR/* ."
