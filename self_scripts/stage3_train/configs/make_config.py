#!/usr/bin/env python3
"""
配置文件生成工具 - 使用基础配置 + 覆盖的方式
用法：
    # 从基础配置创建新配置
    ./make_config.py base.json --set dataset.repo_id=hellozjt/my_data --output my_config.json

    # 从现有配置修改
    ./make_config.py stable.json --set batch_size=8 --output stable_small.json

    # 查看合并后的配置（不保存）
    ./make_config.py stable.json --set batch_size=8 --dry-run
"""

import json
import sys
import argparse
from pathlib import Path
from copy import deepcopy

def set_nested_value(config, key_path, value):
    """设置嵌套的配置值，如 'dataset.repo_id' """
    keys = key_path.split('.')
    current = config

    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]

    # 类型转换
    final_key = keys[-1]
    if value.lower() == 'true':
        current[final_key] = True
    elif value.lower() == 'false':
        current[final_key] = False
    elif value.replace('.', '').replace('e-', '').replace('e+', '').isdigit():
        # 尝试转换为数字
        try:
            if 'e' in value.lower() or '.' in value:
                current[final_key] = float(value)
            else:
                current[final_key] = int(value)
        except:
            current[final_key] = value
    else:
        current[final_key] = value

def main():
    parser = argparse.ArgumentParser(
        description="配置文件生成工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 从基础配置创建新任务配置
  %(prog)s configs/base.json --set dataset.repo_id=hellozjt/my_task --set batch_size=16 -o my_task.json

  # 修改现有配置
  %(prog)s configs/stable.json --set batch_size=8 -o stable_small.json

  # 查看但不保存
  %(prog)s configs/stable.json --set policy.optimizer_lr=1e-4 --dry-run
"""
    )

    parser.add_argument("base_config", help="基础配置文件路径")
    parser.add_argument("--set", "-s", action="append", dest="overrides",
                        help="覆盖配置项，格式: key.subkey=value (可多次使用)")
    parser.add_argument("--output", "-o", help="输出文件路径")
    parser.add_argument("--dry-run", action="store_true", help="只显示配置，不保存")

    args = parser.parse_args()

    # 读取基础配置
    base_path = Path(args.base_config)
    if not base_path.exists():
        print(f"❌ 配置文件不存在: {args.base_config}")
        sys.exit(1)

    with open(base_path) as f:
        config = json.load(f)

    # 应用覆盖
    if args.overrides:
        print(f"从 {base_path.name} 应用覆盖...")
        for override in args.overrides:
            if '=' not in override:
                print(f"⚠️  忽略无效的覆盖: {override}")
                continue
            key, value = override.split('=', 1)
            print(f"  {key} = {value}")
            set_nested_value(config, key, value)

    # 显示配置
    print("\n最终配置:")
    print(json.dumps(config, indent=2))

    # 保存
    if args.dry_run:
        print("\n(--dry-run 模式，未保存)")
    elif args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"\n✓ 已保存到: {output_path}")
    else:
        print("\n⚠️  未指定输出文件 (使用 --output 或 --dry-run)")

if __name__ == "__main__":
    main()
