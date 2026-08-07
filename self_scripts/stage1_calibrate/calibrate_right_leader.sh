#!/bin/bash

# ========================================
# 单独校准右臂 Leader
# ========================================
# 此脚本仅校准右臂 Leader 机械臂
# ========================================

set -e  # 遇到错误立即退出

# ========================================
# 配置区域 - 根据你的实际串口修改
# ========================================

# 右臂 Leader 串口
RIGHT_LEADER_PORT="/dev/ttyRightLeader"  # 请根据实际情况修改

# 校准文件 ID
LEADER_ID="jt_leader_arm_right"

# ========================================
# 颜色输出
# ========================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ========================================
# 主流程
# ========================================

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}单独校准：右臂 Leader${NC}"
echo -e "${BLUE}========================================${NC}\n"

echo -e "${YELLOW}配置信息：${NC}"
echo "  串口: $RIGHT_LEADER_PORT"
echo "  ID: $LEADER_ID"
echo "  校准文件: ${LEADER_ID}.json"
echo ""

# 检查串口
if [ -e "$RIGHT_LEADER_PORT" ]; then
    echo -e "${GREEN}✓ 找到设备: $RIGHT_LEADER_PORT${NC}\n"
else
    echo -e "${RED}✗ 设备不存在: $RIGHT_LEADER_PORT${NC}"
    echo -e "\n${YELLOW}提示：运行以下命令查看可用串口：${NC}"
    echo "  ls -l /dev/ttyUSB* /dev/ttyACM*"
    exit 1
fi

# 重要提示
echo -e "${RED}========================================${NC}"
echo -e "${RED}⚠️  重要提示${NC}"
echo -e "${RED}========================================${NC}"
echo -e "${YELLOW}开始校准前，请确保：${NC}"
echo "  1. 右臂 Leader 已连接到 $RIGHT_LEADER_PORT"
echo "  2. 将机械臂移动到 ${GREEN}中间位置${NC}"
echo "  3. 确保机械臂可以自由移动到各个极限位置"
echo ""

# 询问用户是否继续
echo -e "${YELLOW}机械臂是否已在中间位置？是否继续校准？(y/n)${NC}"
read -r response
if [[ ! "$response" =~ ^[Yy]$ ]]; then
    echo "取消校准"
    exit 0
fi

echo ""
echo -e "${BLUE}开始校准...${NC}\n"

# 执行校准
lerobot-calibrate \
    --teleop.type=so101_leader \
    --teleop.port="$RIGHT_LEADER_PORT" \
    --teleop.id="$LEADER_ID"

# 检查结果
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}✓ 右臂 Leader 校准完成！${NC}"
    echo -e "${GREEN}========================================${NC}\n"

    echo "校准文件已生成："
    echo "  ${LEADER_ID}.json"
    echo ""
    echo "文件位置："
    echo "  ~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/"
    echo ""

    # 显示校准文件
    CALIB_FILE=~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/${LEADER_ID}.json
    if [ -f "$CALIB_FILE" ]; then
        echo -e "${GREEN}校准文件详情：${NC}"
        ls -lh "$CALIB_FILE"
    fi
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}✗ 校准失败！${NC}"
    echo -e "${RED}========================================${NC}\n"
    echo -e "${YELLOW}可能的原因：${NC}"
    echo "  1. 机械臂未在中间位置"
    echo "  2. 串口连接有问题"
    echo "  3. 设备权限不足（尝试: sudo chmod 666 $RIGHT_LEADER_PORT）"
    exit 1
fi
