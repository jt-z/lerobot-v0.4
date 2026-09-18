# b601 单臂 makecoffee 数据集笔记

主题：`/data/share/b601_20260910_164106`（单臂 `seeed_b601_rs_follower`，7 维 action/state，
3 路相机 hand/front/top，30 fps）的**源端增量传输状态**、**`file-009.parquet` 损坏复现与修复**、
**校验方法**。

配套训练工程（ACT 训练配置、100K run 分析、多卡扩展性）见上级目录
[`../README.md`](../README.md)，本目录不重复那部分内容。

## 文件索引

| 文件 | 内容 |
|---|---|
| [`00-项目元数据.md`](00-项目元数据.md) | 基本信息、完整时间线、问题与解决、关键决策、路径速查 |
| [`01-数据集现状与传输状态.md`](01-数据集现状与传输状态.md) | 当前规模、增量批次、传输是否完成的判断方法 |
| [`02-损坏机理-物理行号等于index.md`](02-损坏机理-物理行号等于index.md) | 根因、为什么必须在数据 parquet 层修、影响面 |
| [`03-file009-修复操作.md`](03-file009-修复操作.md) | 可复现的修复步骤与逐项校验门槛 |
| [`04-校验方法与结果.md`](04-校验方法与结果.md) | 结构校验 + 解码校验的方法与实测结果 |
| [`05-预防与治本方案.md`](05-预防与治本方案.md) | 为什么每次重传都会复现、两个治本选项 |
| [`DISTILLED.md`](DISTILLED.md) | 蒸馏版：规律、决策、踩坑精华 |

## 脚本

| 脚本 | 用途 | 用法 |
|---|---|---|
| [`scripts/fix_file009.py`](scripts/fix_file009.py) | 幂等修复：丢弃 file-009 末尾多的 11,005 行 row group，先备份再原子替换 | `python scripts/fix_file009.py --root /data/share/b601_20260910_164106`（加 `--dry-run` 只检查） |
| [`scripts/verify_dataset.py`](scripts/verify_dataset.py) | 结构校验（不需要 GPU）；`--decode` 追加 `LeRobotDataset` 实拉解码校验 | `python scripts/verify_dataset.py --root ... [--decode]` |

> ⚠️ 已知本目录名 `datastet_notes` 是 `dataset_notes` 的笔误（按用户给定路径原样创建）。

## 快速自检

```bash
# 一行判断数据集有没有坏（不等就说明重传又把损坏带回来了）
python scripts/verify_dataset.py --root /data/share/b601_20260910_164106
```

## 一句话结论

**源端每次重传都会把 `data/chunk-000/file-009.parquet` 覆盖回损坏版（末尾多 11,005 行）**；
`LeRobotDataset` 靠「物理行号 == `index` 值」定位 episode，这个块会让 **ep162 之后约
112,698 帧（≈21%）** 静默错配——训练不报错、loss 曲线正常，但学的是错的。
**同步后先跑 `verify_dataset.py`，`总行数 == meta total_frames` 成立再开训。**
