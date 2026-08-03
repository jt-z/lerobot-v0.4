#!/bin/bash

# ========================================
# 双臂 SO-101 校准脚本（方案B：分别校准）
# ========================================
# 此脚本会分别校准 4 个机械臂：
# - 左臂 Follower
# - 右臂 Follower
# - 左臂 Leader
# - 右臂 Leader
# ========================================

set -e  # 遇到错误立即退出

# ========================================
# 配置区域 - 根据你的实际串口修改
# ========================================

# Follower 臂串口
LEFT_FOLLOWER_PORT="/dev/ttyLeftFollower"  # 请根据实际情况修改
RIGHT_FOLLOWER_PORT="/dev/ttyRightFollower"  # 请根据实际情况修改

# Leader 臂串口
LEFT_LEADER_PORT="/dev/ttyLeftLeader"  # 请根据实际情况修改
RIGHT_LEADER_PORT="/dev/ttyRightLeader"  # 请根据实际情况修改

# 校准文件 ID 前缀（不要修改后缀 _left/_right）
FOLLOWER_ID_PREFIX="jt_follower_arm"
LEADER_ID_PREFIX="jt_leader_arm"

# ========================================
# 颜色输出
# ========================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ========================================
# 辅助函数
# ========================================

print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# ========================================
# 检查串口设备
# ========================================

check_ports() {
    print_header "检查串口设备"

    local all_ports_ok=true

    for port in "$LEFT_FOLLOWER_PORT" "$RIGHT_FOLLOWER_PORT" "$LEFT_LEADER_PORT" "$RIGHT_LEADER_PORT"; do
        if [ -e "$port" ]; then
            print_success "找到设备: $port"
        else
            print_error "设备不存在: $port"
            all_ports_ok=false
        fi
    done

    if [ "$all_ports_ok" = false ]; then
        echo -e "\n${YELLOW}提示：运行以下命令查看可用串口：${NC}"
        echo "  ls -l /dev/ttyUSB* /dev/ttyACM*"
        exit 1
    fi

    echo -e "\n${GREEN}所有串口设备检查通过！${NC}"
}

# ========================================
# 校准单个设备
# ========================================

calibrate_device() {
    local device_type=$1
    local device_name=$2
    local port=$3
    local id=$4

    print_header "校准: $device_name"

    echo -e "${YELLOW}即将校准: $device_name${NC}"
    echo "  类型: $device_type"
    echo "  串口: $port"
    echo "  ID: $id"
    echo ""

    if [ "$device_type" = "follower" ]; then
        lerobot-calibrate \
            --robot.type=so101_follower \
            --robot.port="$port" \
            --robot.id="$id"
    else
        lerobot-calibrate \
            --teleop.type=so101_leader \
            --teleop.port="$port" \
            --teleop.id="$id"
    fi

    if [ $? -eq 0 ]; then
        print_success "$device_name 校准完成！"
    else
        print_error "$device_name 校准失败！"
        exit 1
    fi
}

# ========================================
# 主流程
# ========================================

main() {
    print_header "双臂 SO-101 校准脚本"

    echo "此脚本将依次校准以下设备："
    echo "  1. 左臂 Follower ($LEFT_FOLLOWER_PORT)"
    echo "  2. 右臂 Follower ($RIGHT_FOLLOWER_PORT)"
    echo "  3. 左臂 Leader ($LEFT_LEADER_PORT)"
    echo "  4. 右臂 Leader ($RIGHT_LEADER_PORT)"
    echo ""

    # 检查串口
    check_ports

    # 询问用户是否继续
    echo -e "\n${YELLOW}是否继续校准？(y/n)${NC}"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "取消校准"
        exit 0
    fi

    # ========================================
    # 1. 校准左臂 Follower
    # ========================================
    calibrate_device "follower" "左臂 Follower" "$LEFT_FOLLOWER_PORT" "${FOLLOWER_ID_PREFIX}_left"

    echo -e "\n${YELLOW}按 Enter 继续校准右臂 Follower...${NC}"
    read -r

    # ========================================
    # 2. 校准右臂 Follower
    # ========================================
    calibrate_device "follower" "右臂 Follower" "$RIGHT_FOLLOWER_PORT" "${FOLLOWER_ID_PREFIX}_right"

    echo -e "\n${YELLOW}按 Enter 继续校准左臂 Leader...${NC}"
    read -r

    # ========================================
    # 3. 校准左臂 Leader
    # ========================================
    calibrate_device "leader" "左臂 Leader" "$LEFT_LEADER_PORT" "${LEADER_ID_PREFIX}_left"

    echo -e "\n${YELLOW}按 Enter 继续校准右臂 Leader...${NC}"
    read -r

    # ========================================
    # 4. 校准右臂 Leader
    # ========================================
    calibrate_device "leader" "右臂 Leader" "$RIGHT_LEADER_PORT" "${LEADER_ID_PREFIX}_right"

    # ========================================
    # 完成
    # ========================================
    print_header "校准完成！"

    echo "已生成以下校准文件："
    echo "  - ${FOLLOWER_ID_PREFIX}_left.json"
    echo "  - ${FOLLOWER_ID_PREFIX}_right.json"
    echo "  - ${LEADER_ID_PREFIX}_left.json"
    echo "  - ${LEADER_ID_PREFIX}_right.json"
    echo ""
    echo "校准文件位置："
    echo "  ~/.cache/huggingface/lerobot/calibration/robots/so_follower/"
    echo "  ~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/"
    echo ""

    # 列出校准文件
    print_header "校准文件列表"
    echo -e "${GREEN}Follower 校准文件：${NC}"
    ls -lh ~/.cache/huggingface/lerobot/calibration/robots/so_follower/ 2>/dev/null || echo "  目录不存在"
    echo ""
    echo -e "${GREEN}Leader 校准文件：${NC}"
    ls -lh ~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/ 2>/dev/null || echo "  目录不存在"

    print_success "所有校准完成！现在可以进行数据采集了。"
}

# 运行主函数
main
