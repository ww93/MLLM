# 阿里云百炼（DashScope）API使用指南

## 概述

UR4Rec完全支持阿里云百炼的OpenAI兼容API，包括：
- ✅ 标准对话模式
- ✅ **深度思考模式**（`enable_thinking`）
- ✅ 流式响应
- ✅ 多种通义千问模型（Qwen系列）

---

## 快速开始

### 1. 获取API密钥

1. 访问 [阿里云百炼](https://dashscope.aliyuncs.com/)
2. 注册/登录账号
3. 获取API Key（格式：`sk-xxx`）

### 2. 设置环境变量

```bash
export DASHSCOPE_API_KEY="your-api-key-here"
```

或在代码中直接设置：
```python
import os
os.environ["DASHSCOPE_API_KEY"] = "your-api-key"
```

---

## 基本使用

### 方式1: 使用UR4Rec封装

```python
from models import OpenAILLM
import os

# 创建客户端
llm = OpenAILLM(
    model="qwen-plus",  # 或 qwen-turbo, qwen-max, qwen-flash
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 生成文本
response = llm.generate("你是谁？")
print(response)
```

### 方式2: 直接使用OpenAI客户端

```python
from openai import OpenAI
import os

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

response = client.chat.completions.create(
    model="qwen-plus",
    messages=[{"role": "user", "content": "你是谁？"}]
)

print(response.choices[0].message.content)
```

---

## 可用模型

| 模型名称 | 说明 | 适用场景 | 成本 |
|---------|------|---------|------|
| **qwen-turbo** | 快速响应，经济实惠 | 简单对话、批量处理 | 💰 |
| **qwen-plus** | 性能均衡，推荐使用 | 通用推荐系统任务 | 💰💰 |
| **qwen-max** | 最强性能，最高质量 | 复杂推理、关键任务 | 💰💰💰 |
| **qwen-flash** | 支持深度思考模式 | 需要推理过程的任务 | 💰💰 |

---

## 深度思考模式 🆕

阿里云百炼的特色功能，模型会展示思考过程。

### 示例1: 启用思考模式

```python
from models import OpenAILLM
import os

llm = OpenAILLM(
    model="qwen-flash",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 启用思考模式
response = llm.generate(
    "分析用户喜欢的电影类型有哪些共同特征？",
    extra_body={"enable_thinking": True}  # 🆕 关键参数
)

print(response)
```

### 示例2: 显示思考过程（流式）

```python
from openai import OpenAI
import os

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

messages = [{"role": "user", "content": "设计一个推荐算法"}]

completion = client.chat.completions.create(
    model="qwen-flash",
    messages=messages,
    extra_body={"enable_thinking": True},  # 启用思考
    stream=True  # 流式响应
)

is_answering = False

print("=" * 20 + " 思考过程 " + "=" * 20)

for chunk in completion:
    delta = chunk.choices[0].delta

    # 打印思考过程
    if hasattr(delta, "reasoning_content") and delta.reasoning_content:
        if not is_answering:
            print(delta.reasoning_content, end="", flush=True)

    # 打印最终答案
    if hasattr(delta, "content") and delta.content:
        if not is_answering:
            print("\n" + "=" * 20 + " 完整回复 " + "=" * 20)
            is_answering = True
        print(delta.content, end="", flush=True)

print()
```

---

## 流式响应

### 示例: 流式生成

```python
from models import OpenAILLM
import os

llm = OpenAILLM(
    model="qwen-plus",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 使用流式响应
response = llm.generate(
    "解释协同过滤算法",
    stream=True  # 启用流式
)

print(response)  # 会自动处理流式数据并返回完整文本
```

---

## 在UR4Rec中使用

### 1. LLM生成器

```python
from models.llm_generator import LLMPreferenceGenerator
import os

# 创建生成器
generator = LLMPreferenceGenerator(
    llm_backend="openai",
    model_name="qwen-plus",
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

print(f"用户偏好: {user_pref}")
```

---

## 配置文件

### config.yaml

```yaml
# LLM配置
llm_backend: openai
llm_model: qwen-plus
llm_api_key: ${DASHSCOPE_API_KEY}  # 从环境变量读取
llm_base_url: https://dashscope.aliyuncs.com/compatible-mode/v1

# 可选：启用深度思考
llm_extra_body:
  enable_thinking: true

# 其他配置...
embedding_dim: 256
```

### Python代码

```python
import yaml
import os

# 加载配置
with open('config.yaml') as f:
    config = yaml.safe_load(f)

# 创建LLM客户端
from models import OpenAILLM

llm = OpenAILLM(
    model=config['llm_model'],
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=config['llm_base_url']
)
```

---

## 完整示例脚本

保存为 `test_dashscope.py`:

```python
from openai import OpenAI
import os

def test_dashscope():
    """测试阿里云百炼连接"""
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    try:
        response = client.chat.completions.create(
            model="qwen-turbo",
            messages=[{"role": "user", "content": "你好"}],
            max_tokens=50
        )

        print("✓ 连接成功！")
        print(f"回复: {response.choices[0].message.content}")
        return True

    except Exception as e:
        print(f"✗ 连接失败: {e}")
        return False

if __name__ == "__main__":
    test_dashscope()
```

运行测试:
```bash
export DASHSCOPE_API_KEY=your-key
python test_dashscope.py
```

---

## 性能优化

### 1. 使用缓存

```python
from models.llm_generator import LLMPreferenceGenerator

# 自动缓存LLM响应
generator = LLMPreferenceGenerator(
    llm_backend="openai",
    model_name="qwen-plus",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    cache_dir="data/dashscope_cache"  # 缓存目录
)

# 第一次调用会请求API
result1 = generator.generate_user_preference(...)

# 第二次调用会从缓存读取
result2 = generator.generate_user_preference(...)  # 瞬间返回
```

### 2. 批量处理

```python
# 批量生成用户偏好
users_data = [
    {"user_id": 1, "user_history": [101, 102]},
    {"user_id": 2, "user_history": [201, 202]},
    # ...
]

generator.batch_generate_user_preferences(
    users_data=users_data,
    item_metadata=item_metadata,
    save_path="data/user_preferences.json"
)
```

### 3. 选择合适的模型

| 任务类型 | 推荐模型 | 原因 |
|---------|---------|------|
| 简单文本生成 | qwen-turbo | 快速且经济 |
| 用户偏好分析 | qwen-plus | 质量与速度平衡 |
| 复杂推理 | qwen-max | 最高质量 |
| 需要推理过程 | qwen-flash + thinking | 可解释性强 |

---

## 成本估算

基于阿里云百炼定价（仅供参考）：

| 模型 | 输入价格 | 输出价格 | 每1000次调用 |
|------|---------|---------|-------------|
| qwen-turbo | ¥0.0008/1K tokens | ¥0.002/1K tokens | ~¥5-10 |
| qwen-plus | ¥0.004/1K tokens | ¥0.012/1K tokens | ~¥20-40 |
| qwen-max | ¥0.04/1K tokens | ¥0.12/1K tokens | ~¥200-400 |

**节省成本技巧**:
- 使用缓存机制
- 批量处理请求
- 根据任务选择合适模型
- 控制max_tokens参数

---

## 错误处理

```python
from models import OpenAILLM
import os

llm = OpenAILLM(
    model="qwen-plus",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

try:
    response = llm.generate("你好")
    print(response)

except Exception as e:
    if "401" in str(e):
        print("❌ API密钥无效")
    elif "429" in str(e):
        print("❌ 请求频率过高，请稍后重试")
    elif "500" in str(e):
        print("❌ 服务器错误")
    else:
        print(f"❌ 未知错误: {e}")
```

---

## 常见问题

### Q1: 如何获取API密钥？

访问 https://dashscope.aliyuncs.com/ 注册并创建API Key。

### Q2: 支持哪些模型？

所有通义千问系列模型：qwen-turbo, qwen-plus, qwen-max, qwen-flash等。

### Q3: 思考模式有什么用？

可以看到模型的推理过程，提高可解释性，适合复杂任务。

### Q4: 流式响应的优势？

实时反馈，提升用户体验，特别适合生成长文本。

### Q5: 如何切换到其他模型？

只需修改`model`参数，无需改动其他代码。

---

## 完整运行示例

```bash
# 1. 设置API密钥
export DASHSCOPE_API_KEY=your-key

# 2. 激活虚拟环境
cd /Users/admin/Desktop/MLLM
source venv/bin/activate

# 3. 运行示例
cd UR4Rec/examples
python dashscope_example.py

# 4. 查看输出
# 会看到6个不同的使用示例
```

---

## 参考链接

- [阿里云百炼官网](https://dashscope.aliyuncs.com/)
- [通义千问API文档](https://help.aliyun.com/zh/dashscope/)
- [OpenAI兼容模式文档](https://help.aliyun.com/zh/dashscope/developer-reference/compatibility-of-openai-with-dashscope/)

---

## 总结

✅ **UR4Rec已完全支持阿里云百炼API**

支持的功能：
- ✓ 标准对话
- ✓ 深度思考模式（`enable_thinking`）
- ✓ 流式响应（`stream`）
- ✓ 所有通义千问模型
- ✓ 自动缓存
- ✓ 批量处理

**推荐配置**:
- 开发/测试: qwen-turbo
- 生产环境: qwen-plus
- 高质量需求: qwen-max
- 可解释性: qwen-flash + thinking mode
