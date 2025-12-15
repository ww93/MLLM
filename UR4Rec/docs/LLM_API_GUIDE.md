# LLM API使用指南

## 概述

UR4Rec项目支持多种LLM后端，包括：
- ✅ **OpenAI官方API** (GPT-3.5, GPT-4等)
- ✅ **OpenAI兼容API** (vLLM, LocalAI, LM Studio, Ollama等)
- ✅ **Anthropic API** (Claude系列)
- ✅ **本地模型** (通过Transformers库)

---

## 1. OpenAI官方API

### 设置API密钥

```bash
export OPENAI_API_KEY="your-api-key-here"
```

### Python代码示例

```python
from models import OpenAILLM

# 使用官方OpenAI API
llm = OpenAILLM(
    model="gpt-3.5-turbo",
    api_key="your-api-key"  # 或者使用环境变量
)

response = llm.generate("Summarize user preferences based on their history.")
print(response)
```

---

## 2. OpenAI兼容API 🆕

### 支持的服务

| 服务 | 默认端口 | base_url示例 |
|------|---------|-------------|
| **vLLM** | 8000 | `http://localhost:8000/v1` |
| **LocalAI** | 8080 | `http://localhost:8080/v1` |
| **LM Studio** | 1234 | `http://localhost:1234/v1` |
| **Ollama** | 11434 | `http://localhost:11434/v1` |
| **Text Generation WebUI** | 5000 | `http://localhost:5000/v1` |

### 示例1: 使用vLLM

```python
from models import OpenAILLM

# 连接到本地vLLM服务
llm = OpenAILLM(
    model="meta-llama/Llama-2-7b-chat-hf",  # vLLM加载的模型
    api_key="dummy-key",  # 本地服务通常不需要真实key
    base_url="http://localhost:8000/v1"
)

response = llm.generate("Generate user preference description.")
print(response)
```

#### vLLM服务器启动

```bash
# 安装vLLM
pip install vllm

# 启动OpenAI兼容服务器
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-chat-hf \
    --port 8000
```

### 示例2: 使用LM Studio

```python
from models import OpenAILLM

# LM Studio提供OpenAI兼容API
llm = OpenAILLM(
    model="local-model",  # LM Studio中加载的模型
    base_url="http://localhost:1234/v1"
)

response = llm.generate("Describe this item.")
```

#### LM Studio设置
1. 下载并安装LM Studio
2. 加载本地模型
3. 在设置中启用"Local Server"
4. 默认地址: `http://localhost:1234/v1`

### 示例3: 使用Ollama

```python
from models import OpenAILLM

# Ollama需要启用OpenAI兼容模式
llm = OpenAILLM(
    model="llama3.2",
    base_url="http://localhost:11434/v1"
)

response = llm.generate("Analyze user behavior.")
```

#### Ollama OpenAI兼容服务器

```bash
# 安装Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 拉取模型
ollama pull llama3.2

# 启动OpenAI兼容服务（默认端口11434）
ollama serve
```

---

## 3. 在UR4Rec中使用LLM

### 3.1 LLM Generator（离线生成用户偏好）

```python
from models.llm_generator import LLMPreferenceGenerator

# 创建生成器（使用本地LM Studio）
generator = LLMPreferenceGenerator(
    llm_backend="openai",
    model_name="local-model",
    api_key="dummy-key",
    base_url="http://localhost:1234/v1"
)

# 生成用户偏好
item_metadata = {
    101: {"title": "Movie A", "genres": "Action|Adventure"},
    205: {"title": "Movie B", "genres": "Comedy|Drama"}
}

user_pref = generator.generate_user_preference(
    user_id=1,
    user_history=[101, 205],
    item_metadata=item_metadata
)

print(f"User preference: {user_pref}")
```

---

## 4. 完整训练示例

### 配置文件 (config.yaml)

```yaml
# LLM配置
llm_backend: openai
llm_model: meta-llama/Llama-2-7b-chat-hf
llm_api_key: dummy-key
llm_base_url: http://localhost:8000/v1  # vLLM地址

# 模型配置
embedding_dim: 256
num_heads: 8
# ... 其他配置
```

### 训练脚本

```python
import yaml
from models import UR4RecV2

# 加载配置
with open('config.yaml') as f:
    config = yaml.safe_load(f)

# 创建模型
model = UR4RecV2(
    num_items=10000,
    # ... 其他参数
)

# LLM仅用于离线生成用户偏好和物品描述
# 参考 scripts/generate_llm_data.py
```

---

## 5. 环境变量配置

### .env文件示例

```bash
# OpenAI官方
OPENAI_API_KEY=sk-xxx

# Anthropic
ANTHROPIC_API_KEY=sk-ant-xxx

# 本地服务地址
VLLM_BASE_URL=http://localhost:8000/v1
LM_STUDIO_BASE_URL=http://localhost:1234/v1
OLLAMA_BASE_URL=http://localhost:11434/v1
```

### 在代码中使用环境变量

```python
import os
from models.llm_generator import LLMPreferenceGenerator

# 从环境变量读取配置
generator = LLMPreferenceGenerator(
    llm_backend="openai",
    model_name=os.getenv("LLM_MODEL", "gpt-3.5-turbo"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("VLLM_BASE_URL")  # 可选，用于本地服务
)
```

---

## 6. 性能对比

| 服务类型 | 延迟 | 成本 | 隐私 | 推荐场景 |
|---------|------|------|------|---------|
| **OpenAI API** | 低-中 | 按量计费 | 数据上传 | 原型开发、小规模应用 |
| **vLLM (本地)** | 极低 | 仅硬件 | 完全本地 | 生产环境、大规模推理 |
| **LM Studio** | 低 | 免费 | 完全本地 | 个人开发、测试 |
| **Ollama** | 低 | 免费 | 完全本地 | 快速实验、轻量应用 |

---

## 7. 故障排查

### 问题1: 连接本地服务失败

```python
# 测试连接
import requests

response = requests.get("http://localhost:8000/v1/models")
print(response.json())
```

**解决方法**:
- 确认服务已启动
- 检查端口号是否正确
- 确认防火墙设置

### 问题2: API密钥错误

对于本地服务，通常不需要真实API密钥：

```python
generator = LLMPreferenceGenerator(
    llm_backend="openai",
    model_name="local-model",
    api_key="dummy-key",  # 任意字符串
    base_url="http://localhost:8000/v1"
)
```

### 问题3: 模型不存在

```python
# 列出可用模型
response = requests.get("http://localhost:8000/v1/models")
models = response.json()
print("Available models:", models)
```

---

## 8. 最佳实践

### 8.1 开发环境
- 使用**LM Studio**或**Ollama**进行快速原型开发
- 模型推荐: `llama3.2`, `qwen2.5`, `gemma2`

### 8.2 生产环境
- 使用**vLLM**部署，性能最优
- 启用批处理和连续批处理(continuous batching)
- 使用GPU加速

### 8.3 成本优化
- 缓存常见查询结果（已在LLMPreferenceGenerator中实现）
- 使用更小的模型(7B而非70B)处理简单任务
- 批量处理请求

### 8.4 安全性
- 本地部署避免数据泄露
- 使用API密钥管理工具(如Vault)
- 限制API访问频率

---

## 9. 快速测试

### 测试脚本

```python
# test_llm_api.py
from models.llm_generator import LLMPreferenceGenerator

def test_llm_connection(base_url=None):
    """测试LLM连接"""
    try:
        generator = LLMPreferenceGenerator(
            llm_backend="openai",
            model_name="gpt-3.5-turbo" if not base_url else "local-model",
            api_key="dummy-key",
            base_url=base_url
        )

        # 测试简单生成
        test_metadata = {
            1: {"title": "Test Item", "genres": "Test"}
        }
        result = generator.generate_user_preference(
            user_id=1,
            user_history=[1],
            item_metadata=test_metadata
        )

        print(f"✓ Connection successful!")
        print(f"Response: {result[:100]}...")
        return True

    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return False

# 测试不同服务
print("Testing vLLM...")
test_llm_connection("http://localhost:8000/v1")

print("\nTesting LM Studio...")
test_llm_connection("http://localhost:1234/v1")
```

运行测试:
```bash
source venv/bin/activate
python test_llm_api.py
```

---

## 10. 参考资源

### 官方文档
- [OpenAI API文档](https://platform.openai.com/docs)
- [vLLM文档](https://docs.vllm.ai)
- [LM Studio](https://lmstudio.ai)
- [Ollama文档](https://ollama.com/docs)

### 模型选择
- **快速原型**: `llama3.2` (3B), `gemma2` (2B)
- **平衡性能**: `llama3.2` (7B), `qwen2.5` (7B)
- **最佳质量**: `llama3.2` (70B), `qwen2.5` (72B)

---

## 总结

UR4Rec现在完全支持：
- ✅ OpenAI官方API
- ✅ 所有OpenAI兼容API (vLLM, LocalAI, LM Studio, Ollama等)
- ✅ Anthropic Claude API
- ✅ 本地Transformers模型

**推荐配置**:
- 开发: LM Studio + llama3.2
- 生产: vLLM + llama3.2 (7B)
- 实验: OpenAI API (gpt-3.5-turbo)
