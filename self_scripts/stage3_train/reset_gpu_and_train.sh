#!/bin/bash
# 重置GPU并重启训练

echo "1. 杀死所有Python进程..."
pkill -9 -f python

echo "2. 等待进程清理..."
sleep 3

echo "3. 重置NVIDIA驱动..."
sudo nvidia-smi --gpu-reset

echo "4. 验证GPU状态..."
nvidia-smi

echo "5. 测试PyTorch是否能访问CUDA..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

echo ""
echo "如果上面显示CUDA可用，按回车继续训练..."
read

cd /home/ksa/lerobot/self_scripts/
bash start_train_act_stable.sh
