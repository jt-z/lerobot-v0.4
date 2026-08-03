#!/usr/bin/env python3
"""
摄像头调焦测试脚本
用于读取 /dev/video4 并实时显示画面，方便手动调焦
"""

import cv2
import sys

def main():
    # 视频设备路径
    device_path = "/dev/video4"

    print(f"正在打开摄像头: {device_path}")
    cap = cv2.VideoCapture(device_path)

    if not cap.isOpened():
        print(f"错误: 无法打开摄像头 {device_path}")
        print("请确认:")
        print("1. 设备是否存在")
        print("2. 是否有访问权限")
        print("3. 设备是否被其他程序占用")
        sys.exit(1)

    # 设置摄像头分辨率（可选）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 获取实际分辨率
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"摄像头已打开")
    print(f"分辨率: {width}x{height}")
    print(f"帧率: {fps}")
    print("\n操作说明:")
    print("- 按 'q' 或 ESC 键退出")
    print("- 按 's' 键保存当前帧")
    print("- 现在可以手动调焦了\n")

    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("错误: 无法读取帧")
            break

        frame_count += 1

        # 在画面上显示帧数和分辨率信息
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Resolution: {width}x{height}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'q' to quit, 's' to save", (10, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 显示画面
        cv2.imshow('Camera Focus Test - /dev/video4', frame)

        # 处理按键
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:  # 'q' 或 ESC
            print("退出程序")
            break
        elif key == ord('s'):  # 's' 保存截图
            filename = f"focus_test_frame_{frame_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"已保存截图: {filename}")

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print("摄像头已关闭")

if __name__ == "__main__":
    main()
