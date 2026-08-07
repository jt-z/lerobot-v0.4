
rm -r /home/jt/.cache/huggingface/lerobot/hellozjt/eval_lerobot_so101_put_ball2cup

lerobot-record  \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: 'MJPG'}, side: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: 'MJPG'}}" \
  --robot.id=jt_follower_arm \
  --display_data=true \
  --dataset.push_to_hub=false \
  --dataset.repo_id=hellozjt/eval_lerobot_so101_put_ball2cup \
  --dataset.single_task="Put ball to the cup" \
  --dataset.episode_time_s=1000 \
  --policy.path=/home/jt/dev/lerobot/output_lerobot_train/put_ball2cup/checkpoints/008000/pretrained_model
