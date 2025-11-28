# 物品描述处理改进说明

## 📝 改进内容

### 1. 支持从 text.xls 读取物品描述

**文件**: `scripts/preprocess_multimodal_dataset.py`

**改进**:
- ✅ 支持通过列名读取：`movie-id` 和 `review`
- ✅ 回退支持位置索引（第1列 = ID，第2列 = 描述）
- ✅ 过滤空值（`nan`）
- ✅ 自动保存到 `item_metadata.json` 的 `description` 字段

**代码片段**:
```python
# 优先使用列名匹配
if 'movie-id' in df.columns and 'review' in df.columns:
    print("使用列名: 'movie-id' 和 'review'")
    for idx, row in df.iterrows():
        movie_id = int(row['movie-id'])
        description = str(row['review'])
        if pd.notna(description) and description != 'nan':
            descriptions[movie_id] = description
```

---

### 2. 默认使用已有描述

**文件**: `scripts/generate_llm_data.py`

**改进**:
- ✅ 默认行为：直接使用 `item_metadata.json` 中的 `description` 字段
- ✅ 无需调用 LLM API，速度快，零成本
- ✅ 新增参数 `--use_existing_descriptions`（默认开启）
- ✅ 新增参数 `--regenerate_descriptions` 用于 LLM 重新生成

**使用示例**:

**方式 1: 使用 text.xls 描述（推荐）**
```bash
python scripts/generate_llm_data.py \
    --config configs/movielens_100k.yaml \
    --data_dir data/ml-100k-multimodal \
    --output_dir data/ml-100k-multimodal/llm_generated \
    --llm_backend mock
```

**方式 2: 使用 LLM 重新生成**
```bash
python scripts/generate_llm_data.py \
    --config configs/movielens_100k.yaml \
    --data_dir data/ml-100k-multimodal \
    --output_dir data/ml-100k-multimodal/llm_generated \
    --llm_backend openai \
    --api_key YOUR_API_KEY \
    --regenerate_descriptions
```

---

## 📂 数据流程

### Step 1: 预处理数据
```bash
python scripts/preprocess_multimodal_dataset.py \
    --dataset ml-100k \
    --data_dir data/Multimodal_Datasets \
    --output_dir data/ml-100k-multimodal \
    --copy_images
```

**输出**: `data/ml-100k-multimodal/item_metadata.json`

```json
{
  "1": {
    "title": "Toy Story (1995)",
    "genres": ["Animation", "Children's", "Comedy"],
    "description": "A cowboy doll is profoundly threatened...",
    "original_id": 1
  }
}
```

### Step 2: 生成 LLM 数据
```bash
python scripts/generate_llm_data.py \
    --config configs/movielens_100k.yaml \
    --data_dir data/ml-100k-multimodal \
    --output_dir data/ml-100k-multimodal/llm_generated \
    --llm_backend mock
```

**输出**: `data/ml-100k-multimodal/llm_generated/item_descriptions.json`

```json
{
  "1": "A cowboy doll is profoundly threatened...",
  "2": "When an alien artifact...",
  ...
}
```

---

## 🔍 验证

运行测试脚本验证描述是否正确提取：

```bash
cd /Users/admin/Desktop/MLLM/UR4Rec
source venv/bin/activate
python test_text_extraction.py
```

**预期输出**:
```
✅ 找到文件: data/ml-100k-multimodal/item_metadata.json
📊 物品总数: 1,682
  ✅ 有文本描述 (来自 text.xls): 1,682
  📈 描述覆盖率: 100.0%
```

---

## 📊 性能对比

| 方式 | 速度 | 成本 | 描述质量 |
|------|------|------|---------|
| **使用 text.xls（推荐）** | ⚡ 极快 (~5秒) | 💰 $0 | ⭐⭐⭐⭐ 原始高质量 |
| **LLM 重新生成** | 🐢 慢 (~10分钟) | 💸 ~$10-50 | ⭐⭐⭐⭐⭐ 可定制 |

---

## 🎯 总结

### 优势
1. ✅ **零成本**: 直接使用数据集提供的描述，无需 API
2. ✅ **高质量**: Multimodal_Datasets 提供的是人工审核的描述
3. ✅ **快速**: 从预处理到生成只需几秒钟
4. ✅ **灵活**: 支持回退到 LLM 生成（如果需要定制）

### 数据来源
- `data/Multimodal_Datasets/M_ML-100K/text.xls`
- `data/Multimodal_Datasets/M_ML-1M/text.xls`

### 列格式
| 列名 | 说明 |
|------|------|
| `movie-id` | 电影 ID |
| `review` | 电影描述/评论文本 |

---

**更新日期**: 2025-11-28
