# qwen-flash 使用指南

## 快速开始

### 1. 设置 API 密钥

```bash
export DASHSCOPE_API_KEY="your-api-key-here"
```

### 2. 基础使用（推荐）

```python
from models.llm_generator import LLMPreferenceGenerator
import os

# 创建生成器
generator = LLMPreferenceGenerator(
    llm_backend="openai",
    model_name="qwen-flash",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 生成用户偏好
item_metadata = {
    101: {"title": "流浪地球", "genres": "科幻|动作"},
    205: {"title": "疯狂的石头", "genres": "喜剧|犯罪"}
}

user_pref = generator.generate_user_preference(
    user_id=1,
    user_history=[101, 205],
    item_metadata=item_metadata
)

print(user_pref)
```

---

## 功能对比

### 标准模式 vs 思考模式

| 特性 | 标准模式 | 思考模式 (`enable_thinking=True`) |
|------|---------|----------------------------------|
| **速度** | ⚡️ 快 | 🐢 较慢 |
| **Token消耗** | 💰 低 | 💰💰 高（包含思考过程） |
| **输出质量** | ✓ 高质量 | ✓✓ 更高质量（带推理过程） |
| **适用场景** | 批量生成、生产环境 | 调试、需要解释性的场景 |

### 推荐配置

```python
# ✅ 推荐：批量生成用户偏好（标准模式）
generator = LLMPreferenceGenerator(
    llm_backend="openai",
    model_name="qwen-flash",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    enable_thinking=False  # 标准模式，快速且经济
)
```

```python
# ⚠️ 可选：需要看思考过程时（思考模式）
generator = LLMPreferenceGenerator(
    llm_backend="openai",
    model_name="qwen-flash",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    enable_thinking=True  # 思考模式，会显示推理过程
)
```

---

## 完整示例

### 示例1：生成用户偏好

```python
from models.llm_generator import LLMPreferenceGenerator
import os

generator = LLMPreferenceGenerator(
    llm_backend="openai",
    model_name="qwen-flash",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 物品元数据
item_metadata = {
    101: {"title": "流浪地球", "genres": "科幻|动作"},
    205: {"title": "疯狂的石头", "genres": "喜剧|犯罪"},
    303: {"title": "让子弹飞", "genres": "动作|喜剧"}
}

# 生成偏好
preference = generator.generate_user_preference(
    user_id=1,
    user_history=[101, 205, 303],
    item_metadata=item_metadata
)

print(f"用户偏好: {preference}")
```

### 示例2：生成物品描述

```python
# 生成单个物品描述
item_desc = generator.generate_item_description(
    item_id=101,
    item_metadata=item_metadata[101]
)

print(f"物品描述: {item_desc}")
```

### 示例3：批量生成

```python
# 准备用户数据
users_data = [
    {"user_id": 1, "user_history": [101, 205, 303]},
    {"user_id": 2, "user_history": [205, 303]},
    {"user_id": 3, "user_history": [101, 303]}
]

# 批量生成用户偏好
generator.batch_generate_user_preferences(
    users_data=users_data,
    item_metadata=item_metadata,
    save_path="data/user_preferences.json"
)

print("✓ 批量生成完成！")
```

---

## 使用脚本生成

### 命令行方式

```bash
# 激活虚拟环境
cd /Users/admin/Desktop/MLLM
source venv/bin/activate

# 设置API密钥
export DASHSCOPE_API_KEY="your-api-key"

# 运行生成脚本
python UR4Rec/scripts/generate_llm_data.py \
    --config configs/your_config.yaml \
    --data_dir data/processed \
    --output_dir data/llm_generated \
    --llm_backend openai \
    --model_name qwen-flash \
    --api_key $DASHSCOPE_API_KEY
```

### 添加 base_url 参数

如果脚本需要指定 base_url，可以修改 `generate_llm_data.py`：

```python
generator = LLMPreferenceGenerator(
    llm_backend=args.llm_backend,
    model_name=args.model_name,
    api_key=args.api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 添加这行
    cache_dir=args.cache_dir
)
```

---

## 快速测试

运行测试脚本：

```bash
cd /Users/admin/Desktop/MLLM
source venv/bin/activate

# 设置API密钥
export DASHSCOPE_API_KEY="your-api-key"

# 运行测试
python test_qwen_flash.py
```

测试脚本会：
1. 测试基础生成功能
2. （可选）测试思考模式
3. （可选）测试批量生成

---

## 其他可用的 qwen 模型

DashScope 支持多个通义千问模型：

| 模型名称 | 特点 | 推荐场景 |
|---------|------|---------|
| **qwen-turbo** | 快速响应，经济实惠 | 简单对话、批量处理 |
| **qwen-plus** | 性能均衡，推荐使用 | 通用推荐系统任务 |
| **qwen-max** | 最强性能，最高质量 | 复杂推理、关键任务 |
| **qwen-flash** | 支持深度思考模式 | 需要推理过程的任务 |

### 切换模型

只需修改 `model_name` 参数：

```python
# 使用 qwen-plus（推荐日常使用）
generator = LLMPreferenceGenerator(
    llm_backend="openai",
    model_name="qwen-plus",  # ← 改这里
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
```

---

## 成本优化建议

1. **使用缓存**（已自动开启）
   - 相同输入会从缓存读取，不重复调用API
   - 缓存位置：`data/llm_cache/llm_cache.json`

2. **选择合适的模型**
   - 开发测试：`qwen-turbo`（最便宜）
   - 生产环境：`qwen-plus`（性价比高）
   - 高质量需求：`qwen-max`

3. **不要开启思考模式**（批量生成时）
   - `enable_thinking=False` 可节省 30-50% tokens

4. **控制生成长度**
   - 用户偏好：`max_tokens=200`（默认已设置）
   - 物品描述：`max_tokens=150`（默认已设置）

---

## 常见问题

### Q1: API密钥错误

```
错误: 401 Unauthorized
```

**解决方法**：
- 检查环境变量是否正确设置：`echo $DASHSCOPE_API_KEY`
- 确认API密钥格式：`sk-xxx`
- 登录 https://dashscope.aliyuncs.com/ 重新生成密钥

### Q2: 连接超时

```
错误: Connection timeout
```

**解决方法**：
- 检查网络连接
- 确认 base_url 正确：`https://dashscope.aliyuncs.com/compatible-mode/v1`
- 尝试增加超时设置

### Q3: 模型不存在

```
错误: Model 'qwen-flash' not found
```

**解决方法**：
- 确认模型名称拼写正确
- 查看支持的模型列表：https://help.aliyun.com/zh/dashscope/

### Q4: 缓存问题

如果需要清除缓存：

```bash
rm -rf data/llm_cache/llm_cache.json
```

---

## 参考资料

- [阿里云百炼官网](https://dashscope.aliyuncs.com/)
- [通义千问API文档](https://help.aliyun.com/zh/dashscope/)
- [DashScope详细指南](UR4Rec/docs/DASHSCOPE_GUIDE.md)
- [LLM API通用指南](UR4Rec/docs/LLM_API_GUIDE.md)

---

## 总结

✅ **您现在可以直接使用 qwen-flash！**

最简单的使用方式：

```python
from models.llm_generator import LLMPreferenceGenerator
import os

generator = LLMPreferenceGenerator(
    llm_backend="openai",
    model_name="qwen-flash",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 开始生成！
```

**不需要**修改其他代码，**不需要**额外配置，直接运行即可！
