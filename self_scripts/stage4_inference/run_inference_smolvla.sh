#!/bin/bash
# SmolVLA 推理脚本 - so101 put_ball2cup 任务
# 使用 lerobot-record + --policy.path 在真实机器人上运行策略

# 清理上次评估缓存（每次重新开始）
rm -rf /home/jt/.cache/huggingface/lerobot/hellozjt/eval_smolvla_so101_put_ball2cup

# 确认 checkpoint 路径存在
POLICY_PATH=/home/jt/dev/lerobot/output_lerobot_train/smolvla_put_ball2cup/checkpoints/020000/pretrained_model
if [ ! -d "$POLICY_PATH" ]; then
    echo "错误: 找不到模型 checkpoint: $POLICY_PATH"
    echo "请确认训练已完成且路径正确"
    exit 1
fi

lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=jt_follower_arm \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: 'MJPG'}, side: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: 'MJPG'}}" \
  --display_data=true \
  --dataset.push_to_hub=false \
  --dataset.repo_id=hellozjt/eval_smolvla_so101_put_ball2cup \
  --dataset.single_task="Put ball to the cup" \
  --dataset.episode_time_s=200 \
  --dataset.num_episodes=5 \
  --dataset.rename_map='{"observation.images.front": "observation.images.camera1", "observation.images.side": "observation.images.camera2"}' \
  --policy.path=$POLICY_PATH \
  --policy.device=cuda \
  --policy.empty_cameras=1
