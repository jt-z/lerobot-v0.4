#!/usr/bin/env python3
"""
测试多个摄像头并拍摄图片以确认端口号
"""
import cv2
import os
from datetime import datetime

# 从 v4l2-ctl 输出看到的视频设备
# 通常每个摄像头有两个设备，第一个是视频流
camera_devices = [0, 2, 4, 6]  # /dev/video0, /dev/video2, /dev/video4

output_dir = "camera_test_images"
os.makedirs(output_dir, exist_ok=True)

print("开始测试摄像头...\n")

for device_id in camera_devices:
    print(f"正在测试 /dev/video{device_id}...")

    cap = cv2.VideoCapture(device_id)

    if not cap.isOpened():
        print(f"  ✗ 无法打开 /dev/video{device_id}\n")
        continue

    # 设置分辨率（可选）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 读取几帧以让摄像头稳定
    for _ in range(5):
        cap.read()

    # 捕获图像
    ret, frame = cap.read()

    if ret and frame is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{output_dir}/camera_video{device_id}_{timestamp}.jpg"
        cv2.imwrite(filename, frame)

        height, width = frame.shape[:2]
        print(f"  ✓ 成功捕获图像")
        print(f"    分辨率: {width}x{height}")
        print(f"    保存至: {filename}\n")
    else:
        print(f"  ✗ 无法从 /dev/video{device_id} 读取图像\n")

    cap.release()

print(f"\n完成！所有图片已保存到 {output_dir}/ 目录")
print("请查看图片内容以确认每个端口对应的摄像头位置。")
