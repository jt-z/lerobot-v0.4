# 本目录状态：已归档（停止新建/维护）

> **重要**：本目录 `/home/ksa/lerobot/self_scripts` 是**旧版**工作脚本库，从 2026-09-30 起**只作存档**，不再新建、不再更新脚本。
> 以后所有训练、采集、遥操作、校准、推理等一律使用**新版代码库**：
>
> ```bash
> # 新版（主用）
> /home/ksa/devdata/Projects/lerobot/self_scripts
> ```
>
> 本目录文件保留的唯一目的是**历史对照 / 排障参考**，不要直接执行这里的脚本。

---

## 为什么切换

新版代码库对旧版做了系统性重构，主要差异：

| 方面 | 旧版（本目录） | 新版（主用） |
|---|---|---|
| 目录组织 | 按阶段（stage1_calibrate / stage3_train …） | 按硬件方案分组（b601_common / b601_single / so101_single / so101_bimanual / b601_so101_bimanual），组内编号即执行顺序 |
| 串口引用 | `/dev/ttyLeftFollower`、`/dev/ttyACM0` 等不稳定路径 | `/dev/serial/by-id/...` 稳定路径（USB 序列号） |
| 摄像头引用 | `/dev/videoN`（随插拔顺序变化） | `/dev/v4l/by-id/...` 稳定路径 |
| 校准 | 逐臂单独脚本 | 双机械臂一次校准（`bi_so_follower` / `bi_so_leader`） |
| 采集 | 固定时长 episode | 按键结束 episode（`n` 保存 / `r` 重录 / `q` 退出），支持 `--resume` 续录 |
| 训练 | 分散、路径硬编码 | 脚本自定位目录；SmolVLA v2 含数据集完整性校验 + GPU 健康检查 + 按 pass 反算 steps |
| 索引 | CHEATSHEET.md | `tools/cheatsheet.sh` |

新版还收录了：
- `b601_common/`：B601 夹爪零点探测、相机对齐、数据集回放（rerun）等新工具
- `b601_so101_bimanual/`：B601 + SO-101 异构双臂方案
- `tools/`：动作特征可视化、动作追踪、训练/推理日志分析等通用工具

## 已从本目录迁移到新版的内容

| 内容 | 新版位置 |
|---|---|
| 校准 / 遥操作 / 采集脚本（SO-101 双臂） | `so101_bimanual/01_calibrate.sh`、`02_teleoperate.sh`、`03_collect_make_coffee.sh` |
| 笔帽盖笔推理（旧数据集） | `so101_bimanual/inference/old_dataset/` |
| ACT 训练（双臂 make_coffee） | `so101_bimanual/train/start_train_act.sh` + `act_train_config.json` |
| PI0.5 LoRA 微调 | `so101_bimanual/train/start_train_pi05.sh` + `configs/pi05_train_config_20000steps.json` |
| SmolVLA 训练 v2（含数据校验） | `b601_single/train/smolvla/`（脚本 + 配置 + `bench/verify_dataset.py` + `datastet_notes/`） |
| 数据集校验工具 | `tools/check_dataset_validity.py`、`tools/convert_to_video_format.py` |
| 全部知识文档 | `notes/`（SO101 硬件、ACT NaN 排查、SmolVLA 架构等） |
| 速查表 | `tools/cheatsheet.sh` |

## 未迁移（已废弃/历史任务）

- `stage4_inference/run_inference_grab_ball.sh`、`run_inference_smolvla.sh`：put_ball2cup 历史任务，硬编码 `/home/jt/...` 旧路径
- `stage3_train/legacy_archive/`、`output_lerobot_train/`：历史训练产物，仅存档
- `stage3_train/dataset_b601_rs_makecoffee_train/bench/` 下除 `verify_dataset.py` 外的多卡基准脚本：一次性诊断工具

## 注意事项

- 本目录下 `output_lerobot_train/`、`train_logs/` 等是历史产物，**占空间较大**，确认无用后可自行清理（不影响新版）。
- 迁移时保留的机器相关绝对路径（modelscope 缓存、`/data/share/...` 数据集根目录等）在新版下仍有效。
