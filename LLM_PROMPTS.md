# LLM Prompts 说明

## 📋 概述

`llm_generator.py` 使用的 prompt 已更新为**英文版本**，确保生成的用户偏好和物品描述为英文。

---

## 🔤 Prompt 模板

### 1. 用户偏好生成 Prompt

**位置**: `llm_generator.py:195-205`

```
Based on the user's historical interactions, summarize the user's preference characteristics.

User's historical interactions:
- Toy Story (1995)
- GoldenEye (1995)
- Four Rooms (1995)
...

Please summarize the user's preferences in 2-3 sentences, including:
1. Preferred types/genres
2. Main features of interest
3. Potential interest directions

User preference summary:
```

**输入**:
- `user_history`: 用户历史交互的物品 ID 列表
- `item_metadata`: 物品元数据（包含 title、genres 等）

**输出示例**:
```
This user prefers action-adventure movies, especially those with sci-fi themes.
They show strong interest in visual effects and fast-paced storytelling.
Their viewing patterns suggest an appreciation for blockbuster franchises and
high-budget productions.
```

**参数**:
- `max_tokens`: 200
- `temperature`: 0.7

---

### 2. 物品描述生成 Prompt

**位置**: `llm_generator.py:241-249`

```
Generate a concise description for the following item for a recommendation system.

Item information:
- Title: Toy Story (1995)
- Genres: Animation|Children's|Comedy

Please describe the item's core features and target audience in 1-2 sentences.

Item description:
```

**输入**:
- `item_id`: 物品 ID
- `item_metadata`: 物品元数据
  - `title`: 电影标题
  - `genres`: 电影类型

**输出示例**:
```
A groundbreaking computer-animated film that tells the story of toys coming to life,
combining humor and heart. Perfect for families and animation enthusiasts who appreciate
innovative storytelling and memorable characters.
```

**参数**:
- `max_tokens`: 150
- `temperature`: 0.7

---

## 🔄 语言选择原因

### 为什么使用英文？

1. **数据集语言**: ML-100K 数据集中的电影标题主要是英文
2. **模型性能**: 大多数 LLM 在英文上的表现更好
3. **国际化**: 英文输出更容易被国际用户理解
4. **一致性**: 电影标题、类型都是英文，描述也用英文保持一致

### 中英文对比

| 方面 | 中文 Prompt | 英文 Prompt |
|------|-----------|-----------|
| **模型理解** | 可能略差 | 更好 |
| **生成质量** | 可能有混杂 | 更一致 |
| **Token 消耗** | 更多（中文字符） | 更少 |
| **数据一致性** | 标题英文，描述中文 | 全部英文 ✓ |

---

## 📊 生成示例

### 用户偏好示例

**User ID: 298**

**历史交互**:
- The Shawshank Redemption (1994)
- The Godfather (1972)
- Pulp Fiction (1994)
- The Dark Knight (2008)
- Fight Club (1999)

**生成的偏好 (英文)**:
```
This user demonstrates a strong preference for critically acclaimed dramas
with complex narratives and strong character development. They gravitate
towards films that explore darker themes, moral ambiguity, and psychological
depth. Their taste suggests appreciation for auteur-driven cinema and
storytelling that challenges conventional narratives.
```

---

### 物品描述示例

**Item ID: 1 - Toy Story (1995)**

**元数据**:
- Title: Toy Story (1995)
- Genres: Animation, Children's, Comedy

**生成的描述 (英文)**:
```
Pixar's revolutionary computer-animated film follows the secret life of toys
when humans aren't watching, combining cutting-edge animation with heartfelt
storytelling. Perfect for families, animation enthusiasts, and anyone who
appreciates innovative filmmaking and timeless themes of friendship and identity.
```

---

**Item ID: 50 - The Usual Suspects (1995)**

**元数据**:
- Title: The Usual Suspects (1995)
- Genres: Crime, Thriller

**生成的描述 (英文)**:
```
A masterfully crafted crime thriller featuring an intricate plot and one of
cinema's most iconic twist endings. Ideal for fans of sophisticated mysteries
who appreciate complex narratives and stellar ensemble performances.
```

---

## 🔧 自定义 Prompt

### 如何修改 Prompt

如果需要自定义 prompt，编辑 `UR4Rec/models/llm_generator.py`:

#### 修改用户偏好 Prompt

```python
# 第 195-205 行
prompt = f"""Based on the user's historical interactions, summarize the user's preference characteristics.

User's historical interactions:
{items_str}

Please summarize the user's preferences in 2-3 sentences, including:
1. Preferred types/genres
2. Main features of interest
3. Potential interest directions

User preference summary:"""
```

**可以修改为**:
```python
# 更简洁的版本
prompt = f"""Analyze the user's movie preferences based on their history:
{items_str}

Provide a 2-sentence summary focusing on their genre preferences and viewing patterns."""

# 或更详细的版本
prompt = f"""As a movie recommendation expert, analyze this user's viewing history:
{items_str}

Create a detailed profile (3-4 sentences) covering:
- Genre preferences and patterns
- Thematic interests (e.g., action, drama, comedy)
- Era preferences (classic vs modern)
- Likely demographic characteristics

User profile:"""
```

#### 修改物品描述 Prompt

```python
# 第 241-249 行
prompt = f"""Generate a concise description for the following item for a recommendation system.

Item information:
- Title: {title}
- Genres: {genres}

Please describe the item's core features and target audience in 1-2 sentences.

Item description:"""
```

**可以修改为**:
```python
# 更市场化的版本
prompt = f"""Write a compelling 2-sentence description for:
Title: {title}
Genres: {genres}

Focus on what makes this movie unique and who would enjoy it."""

# 或更技术化的版本
prompt = f"""Create a structured description for movie recommendation:
- Title: {title}
- Genres: {genres}

Format: [Genre appeal] + [Target audience] + [Key features]
Length: 1-2 sentences."""
```

---

## 💡 Prompt 优化建议

### 1. 输出长度控制

```python
# 短输出 (50-100 tokens)
"Summarize in ONE sentence:"

# 中等输出 (100-200 tokens)
"Provide a 2-3 sentence summary:"

# 长输出 (200-300 tokens)
"Create a detailed 4-5 sentence profile:"
```

### 2. 风格控制

```python
# 正式风格
"Provide a professional analysis of..."

# 口语化风格
"Describe in a conversational tone..."

# 营销风格
"Write a compelling pitch that..."
```

### 3. 结构化输出

```python
# JSON 格式
"Output in JSON format: {\"genres\": [...], \"appeal\": \"...\", \"audience\": \"...\"}"

# 列表格式
"Provide bullet points covering: \n- Genre preferences\n- Key themes\n- Target demographic"

# 段落格式
"Write a cohesive paragraph covering all aspects."
```

---

## 🧪 测试 Prompt

### 快速测试

```bash
# 生成 5 个用户和 10 个物品测试效果
export DASHSCOPE_API_KEY="your-key"
python UR4Rec/models/llm_generator.py --num_users 5 --num_items 10

# 查看生成结果
cat data/llm_generated/user_preferences.json | python -m json.tool | head -30
cat data/llm_generated/item_descriptions.json | python -m json.tool | head -30
```

### 评估生成质量

```python
import json

# 读取生成结果
with open('data/llm_generated/user_preferences.json', 'r') as f:
    user_prefs = json.load(f)

with open('data/llm_generated/item_descriptions.json', 'r') as f:
    item_descs = json.load(f)

# 检查平均长度
user_pref_lengths = [len(p.split()) for p in user_prefs.values()]
item_desc_lengths = [len(d.split()) for d in item_descs.values()]

print(f"用户偏好平均词数: {sum(user_pref_lengths)/len(user_pref_lengths):.1f}")
print(f"物品描述平均词数: {sum(item_desc_lengths)/len(item_desc_lengths):.1f}")

# 查看示例
print("\n用户偏好示例:")
for user_id, pref in list(user_prefs.items())[:3]:
    print(f"\nUser {user_id}:")
    print(f"  {pref}")

print("\n物品描述示例:")
for item_id, desc in list(item_descs.items())[:3]:
    print(f"\nItem {item_id}:")
    print(f"  {desc}")
```

---

## 🛡️ 错误处理

### 内容审核兜底逻辑

LLM API 可能因内容审核失败返回 400 错误。已添加兜底逻辑自动处理：

**位置**: `llm_generator.py:208-219` 和 `262-274`

**错误类型**: `openai.BadRequestError` with error code 400 (data_inspection_failed)

**处理逻辑**:

```python
try:
    preference_text = self._call_llm(prompt, max_tokens=200)
except Exception as e:
    # 检查是否是内容审核错误
    error_str = str(e).lower()
    if ("400" in error_str or "bad request" in error_str) and \
       ("data_inspection_failed" in error_str or "inappropriate content" in error_str):
        logger.warning(f"用户 {user_id} 触发内容审核，使用兜底文本")
        preference_text = "User has no obvious preferences."
    else:
        # 其他错误继续抛出
        raise
```

**兜底文本**:
- **用户偏好**: `"User has no obvious preferences."`
- **物品描述**: `"No description available."`

**优势**:
- ✅ 自动处理内容审核失败
- ✅ 不会中断批量生成流程
- ✅ 记录警告日志便于排查
- ✅ 其他错误正常抛出便于调试

**示例日志**:
```
WARNING:__main__:用户 123 触发内容审核，使用兜底文本
WARNING:__main__:物品 456 (Movie Title) 触发内容审核，使用兜底文本
```

---

## 📚 相关文档

- [GENERATE_LLM_PREFERENCES.md](GENERATE_LLM_PREFERENCES.md) - LLM 生成完整指南
- [QWEN_FLASH_USAGE.md](QWEN_FLASH_USAGE.md) - qwen-flash 使用指南
- [llm_generator.py](UR4Rec/models/llm_generator.py) - 源代码

---

## 📝 更新日志

### 2025-12-10
- ✅ 添加内容审核错误兜底逻辑
- ✅ 用户偏好和物品描述都支持错误处理

### 2025-12-09
- ✅ 更新用户偏好 prompt 为英文
- ✅ 更新物品描述 prompt 为英文
- ✅ 创建本说明文档

### 原版本 (中文 Prompt)

如果需要恢复中文版本：

```python
# 用户偏好 (中文)
prompt = f"""基于用户的历史交互物品，总结该用户的偏好特征。

用户历史交互的物品：
{items_str}

请用2-3句话总结该用户的偏好，包括：
1. 偏好的类型/风格
2. 关注的主要特征
3. 可能的兴趣方向

用户偏好总结："""

# 物品描述 (中文)
prompt = f"""请为以下物品生成一个简洁的描述，用于推荐系统。

物品信息：
- 标题：{title}
- 类型：{genres}

请用1-2句话描述该物品的核心特征和适合的用户群体。

物品描述："""
```

---

*创建时间: 2025-12-09*
*版本: 1.0 (English Prompts)*
