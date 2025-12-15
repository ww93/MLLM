# 生成 LLM 偏好指南

## 📋 概述

`llm_generator.py` 现在可以直接运行，自动为 ML-100K 数据集生成用户偏好和物品描述。

---

## 🚀 快速开始

### 1. 设置 API 密钥

```bash
# 使用阿里云 DashScope（推荐）
export DASHSCOPE_API_KEY="your-api-key-here"

# 或使用 OpenAI
export OPENAI_API_KEY="your-api-key-here"
```

获取 DashScope API 密钥：https://dashscope.aliyuncs.com/

### 2. 运行生成脚本

```bash
cd /Users/admin/Desktop/MLLM

# 方法1: 直接运行（生成全部数据）
python UR4Rec/models/llm_generator.py

# 方法2: 生成部分数据（测试用）
python UR4Rec/models/llm_generator.py --num_users 10 --num_items 50

# 方法3: 只生成用户偏好
python UR4Rec/models/llm_generator.py --skip_items

# 方法4: 只生成物品描述
python UR4Rec/models/llm_generator.py --skip_users
```

---

## ⚙️ 命令行参数

### 基础参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--data_dir` | str | `UR4Rec/data/Multimodal_Datasets` | 数据集目录 |
| `--output_dir` | str | `data/llm_generated` | 输出目录 |
| `--model_name` | str | `qwen-flash` | LLM 模型名称 |

### 数量控制

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--num_users` | int | None | 生成的用户数（None=全部938个） |
| `--num_items` | int | None | 生成的物品数（None=全部1659个） |

### 功能开关

| 参数 | 类型 | 说明 |
|------|------|------|
| `--enable_thinking` | flag | 启用深度思考模式（消耗更多tokens） |
| `--skip_users` | flag | 跳过用户偏好生成 |
| `--skip_items` | flag | 跳过物品描述生成 |

---

## 📝 使用示例

### 示例 1: 完整生成（全部数据）

```bash
export DASHSCOPE_API_KEY="sk-xxx"

python UR4Rec/models/llm_generator.py
```

**输出**:
```
============================================================
ML-100K 数据集 LLM 偏好生成
============================================================

使用 DashScope API (模型: qwen-flash)

[1/4] 加载数据集...
✓ 数据加载完成
  - 物品数: 1659
  - 用户序列数: 938

[2/4] 创建 LLM 生成器...
✓ 生成器创建完成

[3/4] 生成用户偏好...
生成用户偏好: 100%|████████| 938/938 [15:00<00:00, 1.04it/s]
✓ 用户偏好已保存到: data/llm_generated/user_preferences.json

[4/4] 生成物品描述...
生成物品描述: 100%|████████| 1659/1659 [20:00<00:00, 1.38it/s]
✓ 物品描述已保存到: data/llm_generated/item_descriptions.json

============================================================
生成完成！
============================================================
```

**生成的文件**:
- `data/llm_generated/user_preferences.json` - 938 个用户偏好
- `data/llm_generated/item_descriptions.json` - 1659 个物品描述
- `data/llm_generated/llm_cache/llm_cache.json` - 缓存文件

### 示例 2: 小批量测试（推荐第一次运行）

```bash
export DASHSCOPE_API_KEY="sk-xxx"

# 只生成 10 个用户和 50 个物品
python UR4Rec/models/llm_generator.py --num_users 10 --num_items 50
```

**优势**:
- 快速验证 API 密钥是否正确
- 快速查看生成质量
- 节省 API 调用费用
- 约 2-3 分钟完成

### 示例 3: 分步生成（避免超时）

```bash
export DASHSCOPE_API_KEY="sk-xxx"

# 步骤1: 只生成用户偏好
python UR4Rec/models/llm_generator.py --skip_items

# 步骤2: 只生成物品描述
python UR4Rec/models/llm_generator.py --skip_users
```

**适用场景**:
- 网络不稳定时
- API 有速率限制
- 需要分批处理

### 示例 4: 使用不同模型

```bash
export DASHSCOPE_API_KEY="sk-xxx"

# 使用 qwen-plus（更高质量）
python UR4Rec/models/llm_generator.py --model_name qwen-plus --num_users 10

# 使用 qwen-turbo（更快速度）
python UR4Rec/models/llm_generator.py --model_name qwen-turbo --num_users 10

# 使用 qwen-max（最高质量）
python UR4Rec/models/llm_generator.py --model_name qwen-max --num_users 10
```

### 示例 5: 启用深度思考模式

```bash
export DASHSCOPE_API_KEY="sk-xxx"

# 启用思考模式（仅 qwen-flash 支持）
python UR4Rec/models/llm_generator.py \
    --enable_thinking \
    --num_users 10
```

**注意**: 思考模式会消耗 30-50% 更多的 tokens，但质量更高。

---

## 📊 输出格式

### user_preferences.json

```json
{
  "298": "该用户偏好动作冒险类电影，尤其喜欢科幻题材...",
  "253": "该用户喜欢经典剧情片，关注人物情感和故事深度...",
  ...
}
```

**格式**: `{user_id: preference_text}`

### item_descriptions.json

```json
{
  "1": "《玩具总动员》是一部经典动画电影，适合家庭观看...",
  "2": "《黄金眼》是一部动作惊悚片，节奏紧凑...",
  ...
}
```

**格式**: `{item_id: description_text}`

---

## 🔄 缓存机制

### 自动缓存

脚本会自动缓存所有 LLM 调用结果：

```
data/llm_generated/llm_cache/llm_cache.json
```

**优势**:
- 重复运行不会重复调用 API
- 如果中断可以从缓存恢复
- 节省 API 费用

### 查看缓存

```bash
# 查看缓存内容
cat data/llm_generated/llm_cache/llm_cache.json | head -50

# 查看缓存统计
python -c "
import json
with open('data/llm_generated/llm_cache/llm_cache.json') as f:
    cache = json.load(f)
    print(f'用户偏好: {len(cache[\"user_preferences\"])} 条')
    print(f'物品描述: {len(cache[\"item_descriptions\"])} 条')
"
```

### 清除缓存

```bash
# 清除所有缓存
rm -rf data/llm_generated/llm_cache/

# 只清除用户偏好缓存
python -c "
import json
from pathlib import Path

cache_file = Path('data/llm_generated/llm_cache/llm_cache.json')
if cache_file.exists():
    with open(cache_file, 'r') as f:
        cache = json.load(f)
    cache['user_preferences'] = {}
    with open(cache_file, 'w') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    print('用户偏好缓存已清除')
"
```

---

## 💰 成本估算

基于 DashScope API 定价（2024年）：

### qwen-flash
- **输入**: ¥0.001 / 1K tokens
- **输出**: ¥0.002 / 1K tokens

### 估算（ML-100K 全量）

| 任务 | 数量 | 输入 tokens | 输出 tokens | 费用 |
|------|------|------------|-------------|------|
| 用户偏好 | 938 | ~200/个 | ~150/个 | ~¥0.5 |
| 物品描述 | 1659 | ~100/个 | ~100/个 | ~¥0.5 |
| **总计** | 2597 | ~350K | ~250K | **~¥1** |

**实际费用可能因模型和生成长度而异**

### 小批量测试费用

```bash
# 10 用户 + 50 物品 ≈ ¥0.05 (5分钱)
python UR4Rec/models/llm_generator.py --num_users 10 --num_items 50
```

---

## 🐛 故障排查

### 问题 1: API 密钥错误

**错误信息**:
```
❌ 错误: 未设置 API 密钥
```

**解决方法**:
```bash
# 检查环境变量
echo $DASHSCOPE_API_KEY

# 重新设置
export DASHSCOPE_API_KEY="your-key"

# 验证设置
python -c "import os; print(os.getenv('DASHSCOPE_API_KEY'))"
```

### 问题 2: 数据集找不到

**错误信息**:
```
❌ 错误: 找不到 movies.dat: UR4Rec/data/Multimodal_Datasets/M_ML-100K/movies.dat
```

**解决方法**:
```bash
# 检查数据集是否存在
ls UR4Rec/data/Multimodal_Datasets/M_ML-100K/

# 或使用自定义路径
python UR4Rec/models/llm_generator.py --data_dir /path/to/your/data
```

### 问题 3: API 调用失败

**错误信息**:
```
Error: 401 Unauthorized
```

**解决方法**:
1. 检查 API 密钥是否正确
2. 检查账户余额
3. 检查网络连接
4. 使用代理（如需要）

### 问题 4: 生成中断

**解决方法**:

脚本支持自动恢复，只需重新运行：

```bash
# 重新运行，会从缓存继续
python UR4Rec/models/llm_generator.py
```

已生成的内容会从缓存读取，不会重复调用 API。

---

## 🔧 进阶用法

### 1. 批处理生成

```bash
#!/bin/bash
# batch_generate.sh

export DASHSCOPE_API_KEY="sk-xxx"

# 分批生成用户偏好
for i in {0..9}; do
    start=$((i * 100))
    echo "生成用户 $start - $((start + 100))"
    python UR4Rec/models/llm_generator.py \
        --num_users $((start + 100)) \
        --skip_items
    sleep 5  # 避免速率限制
done

# 生成物品描述
python UR4Rec/models/llm_generator.py --skip_users
```

### 2. 自定义 Prompt（修改代码）

编辑 `UR4Rec/models/llm_generator.py`:

```python
# 第 195-205 行
prompt = f"""基于用户的历史交互物品，总结该用户的偏好特征。

用户历史交互的物品：
{items_str}

请用2-3句话总结该用户的偏好，包括：
1. 偏好的类型/风格
2. 关注的主要特征
3. 可能的兴趣方向

用户偏好总结："""
```

修改为自定义内容。

### 3. 并行生成（多进程）

```python
# parallel_generate.py
from multiprocessing import Pool
from UR4Rec.models.llm_generator import LLMPreferenceGenerator

def generate_user_batch(user_batch):
    generator = LLMPreferenceGenerator(...)
    for user_id, history in user_batch:
        generator.generate_user_preference(user_id, history, item_metadata)

# 使用进程池
with Pool(4) as pool:
    pool.map(generate_user_batch, user_batches)
```

---

## 📚 生成后的使用

### 加载生成的偏好

```python
import json
from pathlib import Path

# 加载用户偏好
with open('data/llm_generated/user_preferences.json', 'r') as f:
    user_preferences = json.load(f)

# 加载物品描述
with open('data/llm_generated/item_descriptions.json', 'r') as f:
    item_descriptions = json.load(f)

# 使用
user_id = "298"
print(f"用户 {user_id} 的偏好: {user_preferences[user_id]}")
```

### 集成到训练流程

```python
from UR4Rec.models.text_preference_retriever import TextPreferenceRetriever

# 创建检索器时传入生成的文本
retriever = TextPreferenceRetriever(
    user_preferences=user_preferences,
    item_descriptions=item_descriptions,
    ...
)
```

---

## ✅ 总结

✅ **llm_generator.py 现在可以直接运行**

**最简单的使用方式**:
```bash
export DASHSCOPE_API_KEY="your-key"
python UR4Rec/models/llm_generator.py --num_users 10 --num_items 50
```

**完整流程**:
1. 设置 API 密钥
2. 运行生成脚本
3. 检查生成的文件
4. 集成到训练流程

**关键特性**:
- 🚀 一键生成用户偏好和物品描述
- 💾 自动缓存，支持断点续传
- 🔧 灵活配置，支持部分生成
- 💰 成本可控，小批量测试
- 📊 完整的进度显示和统计

---

*创建时间: 2025-12-09*
