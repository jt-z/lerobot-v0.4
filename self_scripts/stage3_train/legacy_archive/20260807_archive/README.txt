这些文件已被新的统一训练脚本系统替代

新系统包括:
- train.sh: 统一的训练启动脚本
- config_generator.py: 动态配置生成器

旧脚本与新命令的对照:
- start_train_act_stable.sh       -> ./train.sh --preset stable
- start_train_act_debug.sh        -> ./train.sh --preset stable --gpus 4 --debug
- start_train_act_stable_7gpu.sh  -> ./train.sh --preset stable --gpus 7 --gpu-ids 1,2,3,4,5,6,7

配置文件现在通过 config_generator.py 动态生成，无需手动维护 JSON 文件。
预设包括: stable, no-vae, put-ball2cup, cap-pen

如需恢复这些文件，只需将它们移回上级目录即可。

归档时间: 2026-08-07
