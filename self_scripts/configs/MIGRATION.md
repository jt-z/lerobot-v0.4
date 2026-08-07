# 迁移到新的简化训练脚本系统

## 变更内容

### ❌ 删除的复杂系统
- `config_generator.py` - 过于复杂，有大量预设和逻辑
- `cleanup_legacy.sh` - 不再需要
- 动态配置生成逻辑 - 不透明，难以调试

### ✅ 新的简化系统
- `train.sh` - 简单的启动脚本，只处理GPU配置
- `make_config.py` - 轻量级配置工具（可选使用）
- `configs/` - 可见的配置文件目录（3个预设）

## 核心理念

**之前（过度设计）：**
- 动态生成配置，看不到最终JSON
- 复杂的参数解析和预设逻辑
- 心智负担：要记住各种参数组合

**现在（简单实用）：**
- 配置文件可见、可验证
- 脚本只处理GPU，不碰配置内容
- 需要新配置？复制修改，或用工具生成

## 使用对比

### 旧方式（归档）
```bash
./start_train_act_stable.sh
./start_train_act_debug.sh --gpus 4
```

### 新方式（推荐）
```bash
./train.sh configs/stable.json
./train.sh configs/stable.json --gpus 4 --debug
```

## 创建新配置

### 方法1：复制修改（最直观）
```bash
cp configs/stable.json configs/my_task.json
vim configs/my_task.json  # 修改需要的字段
./train.sh configs/my_task.json
```

### 方法2：使用工具（可选）
```bash
./make_config.py configs/base.json \
    --set dataset.repo_id=hellozjt/my_task \
    --set batch_size=16 \
    -o configs/my_task.json
```

## 优势

✅ **透明** - 配置文件就在那里，清晰可见  
✅ **简单** - 脚本逻辑简单，易于理解和维护  
✅ **可调试** - 出问题时知道用的是什么配置  
✅ **灵活** - 复制、修改、版本控制都很方便
