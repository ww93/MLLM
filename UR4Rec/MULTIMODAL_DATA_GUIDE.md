# 多模态数据集使用指南

本文档说明如何使用 `data/Multimodal_Datasets` 目录下已有的多模态 MovieLens 数据。

---

## 📂 数据结构

```
data/Multimodal_Datasets/
├── M_ML-100K/              # MovieLens-100K 多模态版本
│   ├── movies.dat          # 电影元数据 (id::title::genres)
│   ├── ratings.dat         # 用户评分 (user::movie::rating::timestamp)
│   ├── text.xls            # 电影文本描述 (Excel格式)
│   ├── user.dat            # 用户信息
│   └── image/              # 电影海报图片
│       ├── 1.png           # 电影 ID 对应的图片
│       ├── 2.png
│       └── ...
│
├── M_ML-1M/                # MovieLens-1M 多模态版本
│   ├── movies.dat
│   ├── ratings.dat
│   ├── text.xls
│   ├── users.dat
│   └── image/              # 3,883 张电影海报
│
└── M_Douban/               # 豆瓣数据集（可选）
```

---

## 🚀 快速开始

### Step 1: 预处理数据

```bash
# 预处理 MovieLens-100K
python scripts/preprocess_multimodal_dataset.py \
    --dataset ml-100k \
    --data_dir data/Multimodal_Datasets \
    --output_dir data/ml-100k-multimodal \
    --min_rating 4.0 \
    --min_seq_len 5 \
    --copy_images

# 预处理 MovieLens-1M
python scripts/preprocess_multimodal_dataset.py \
    --dataset ml-1m \
    --data_dir data/Multimodal_Datasets \
    --output_dir data/ml-1m-multimodal \
    --min_rating 4.0 \
    --min_seq_len 5 \
    --copy_images
```

**参数说明**:
- `--dataset`: 数据集名称 (ml-100k 或 ml-1m)
- `--data_dir`: Multimodal_Datasets 目录路径
- `--output_dir`: 输出目录
- `--min_rating`: 最小评分阈值 (只保留高评分交互)
- `--min_seq_len`: 最小序列长度 (过滤太短的序列)
- `--copy_images`: 是否复制并重命名图片

**输出文件**:
```
data/ml-100k-multimodal/
├── train_sequences.npy     # 训练序列 {user_id: [item_ids]}
├── val_sequences.npy       # 验证序列
├── test_sequences.npy      # 测试序列
├── item_metadata.json      # 物品元数据 (包含从 text.xls 提取的描述) ⭐
├── user_map.json           # 用户ID映射 {old_id: new_id}
├── item_map.json           # 物品ID映射 {old_id: new_id}
├── stats.json              # 数据统计
└── images/                 # 重命名后的图片 (使用新ID)
    ├── 1.png
    ├── 2.png
    └── ...
```

**item_metadata.json 示例**:
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

---

### Step 2: 提取图片特征

```bash
# 使用 CLIP 提取图片特征
python scripts/preprocess_images.py \
    --image_dir data/ml-100k-multimodal/images \
    --output_path data/ml-100k-multimodal/image_features.pt \
    --mode clip \
    --batch_size 32
```

---

### Step 3: 生成 LLM 数据

**方案 A: 使用 text.xls 中的描述（推荐，默认行为）**

```bash
# 直接使用 Multimodal_Datasets 中的物品描述
python scripts/generate_llm_data.py \
    --config configs/movielens_100k.yaml \
    --data_dir data/ml-100k-multimodal \
    --output_dir data/ml-100k-multimodal/llm_generated \
    --llm_backend openai \
    --api_key YOUR_API_KEY 
```

**说明**：
- ✅ 默认行为：直接使用 `text.xls` 中的物品描述（已在预处理时保存到 `item_metadata.json`）
- ✅ 无需 API 调用，速度快
- ✅ 使用原始数据集提供的高质量描述

**方案 B: 使用 LLM 重新生成描述（可选）**

```bash
# 使用 LLM 重新生成物品描述（忽略 text.xls）
python scripts/generate_llm_data.py \
    --config configs/movielens_100k.yaml \
    --data_dir data/ml-100k-multimodal \
    --output_dir data/ml-100k-multimodal/llm_generated \
    --llm_backend openai \
    --api_key YOUR_API_KEY \
    --regenerate_descriptions
```

**说明**：
- 🔄 使用 LLM 重新生成更详细的描述
- 💰 需要 API 调用（有成本）
- 适合需要定制化描述的场景

---

### Step 4: 训练多模态模型

```bash
# 训练文本模态模型
python scripts/train_v2.py \
    --config configs/movielens_100k.yaml \
    --data_dir data/ml-100k-multimodal \
    --llm_data_dir data/ml-100k-multimodal/llm_generated \
    --output_dir outputs/ml-100k-multimodal

# 训练多模态模型（文本+图像）
python scripts/train_v2.py \
    --config configs/movielens_100k.yaml \
    --data_dir data/ml-100k-multimodal \
    --llm_data_dir data/ml-100k-multimodal/llm_generated \
    --output_dir outputs/ml-100k-mm-full \
    --use_multimodal
```

---

## 📊 数据格式详解

### 1. movies.dat

格式: `movie_id::title::genres`

示例:
```
1::Toy Story (1995)::Animation|Children's|Comedy
2::GoldenEye (1995)::Action|Adventure|Thriller
```

### 2. ratings.dat

格式: `user_id::movie_id::rating::timestamp`

示例:
```
196::242::3::881250949
186::302::3::891717742
```

### 3. text.xls

Excel 格式，包含电影的文本描述。

**列格式**:
- **列名**: `movie-id` 和 `review`
- **movie-id**: 电影 ID（整数）
- **review**: 电影描述/评论文本（字符串）

**示例**:
| movie-id | review |
|----------|--------|
| 1 | A cowboy doll is profoundly threatened and jealous when a new spaceman figure... |
| 2 | When an alien artifact discovered on Earth is found to have come from Venus... |

**预处理行为**:
1. ✅ 优先使用列名 `movie-id` 和 `review` 读取
2. ✅ 如果列名不匹配，回退到位置索引（第1列=ID，第2列=描述）
3. ✅ 自动过滤空值（`nan`）
4. ✅ 保存到 `item_metadata.json` 的 `description` 字段

### 4. images/

每部电影的海报图片，命名格式: `{movie_id}.png`

**图片规格**:
- 格式: PNG
- 分辨率: 不固定（需要预处理）
- 推荐调整为: 224x224 (CLIP 标准)

---

## 🔄 与原始 MovieLens 数据的区别

| 特性 | 原始 MovieLens | 多模态 MovieLens |
|------|---------------|-----------------|
| 评分数据 | ✅ | ✅ |
| 电影元数据 | ✅ (基础) | ✅ (详细) |
| 文本描述 | ❌ | ✅ (text.xls) |
| 电影海报 | ❌ | ✅ (image/) |
| 格式 | CSV/DAT | DAT + XLS + PNG |

---

## ⚙️ 高级用法

### 1. 只使用图片，不复制

```bash
python scripts/preprocess_multimodal_dataset.py \
    --dataset ml-100k \
    --data_dir data/Multimodal_Datasets \
    --output_dir data/ml-100k-multimodal \
    # 不添加 --copy_images

# 然后直接使用原始图片目录
python scripts/preprocess_images.py \
    --image_dir data/Multimodal_Datasets/M_ML-100K/image \
    --output_path data/ml-100k-multimodal/image_features.pt \
    --mode clip
```

**优点**: 节省磁盘空间
**缺点**: 需要手动管理 ID 映射

---

### 2. 自定义过滤条件

```bash
# 只保留评分 >= 4.5 的交互
python scripts/preprocess_multimodal_dataset.py \
    --dataset ml-100k \
    --data_dir data/Multimodal_Datasets \
    --output_dir data/ml-100k-strict \
    --min_rating 4.5 \
    --min_seq_len 10 \
    --copy_images
```

---

### 3. 批量处理多个数据集

```bash
#!/bin/bash

# 定义数据集列表
datasets=("ml-100k" "ml-1m")

for dataset in "${datasets[@]}"; do
    echo "处理 $dataset..."

    python scripts/preprocess_multimodal_dataset.py \
        --dataset $dataset \
        --data_dir data/Multimodal_Datasets \
        --output_dir data/${dataset}-multimodal \
        --min_rating 4.0 \
        --min_seq_len 5 \
        --copy_images

    python scripts/preprocess_images.py \
        --image_dir data/${dataset}-multimodal/images \
        --output_path data/${dataset}-multimodal/image_features.pt \
        --mode clip

    python scripts/generate_llm_data.py \
        --config configs/${dataset//-/_}.yaml \
        --data_dir data/${dataset}-multimodal \
        --output_dir data/${dataset}-multimodal/llm_generated \
        --llm_backend mock

    echo "$dataset 处理完成！"
    echo "---"
done
```

---

## 🐛 常见问题

### Q1: 无法读取 text.xls 文件

**错误**: `xlrd.biffh.XLRDError: Excel xlsx file; not supported`

**原因**: text.xls 是旧格式的 Excel 文件

**解决**:
```bash
# 安装 xlrd (支持旧格式)
pip install xlrd==1.2.0

# 或者安装 openpyxl (如果是 xlsx)
pip install openpyxl
```

如果仍然失败，脚本会回退到使用电影标题和类型作为描述。

---

### Q2: 图片复制很慢

**原因**: 1,682 (ML-100K) 或 3,883 (ML-1M) 张图片需要时间

**解决**:
```bash
# 方案1: 不复制图片，直接使用原始目录
# 省略 --copy_images 参数

# 方案2: 使用硬链接（如果在同一文件系统）
# 修改 preprocess_multimodal_dataset.py 中的 shutil.copy2 为 os.link
```

---

### Q3: 预处理后图片 ID 对不上

**原因**: ID 被重新映射了

**解决**: 使用脚本自动复制图片（`--copy_images`），它会处理 ID 映射。

或者手动查看映射:
```python
import json

with open('data/ml-100k-multimodal/item_map.json', 'r') as f:
    item_map = json.load(f)

# old_id -> new_id
print(item_map)
```

---

### Q4: 某些电影没有图片

**现象**: 图片目录中缺少某些电影的图片

**影响**: 不影响训练，这些电影会被跳过或使用占位图

**检查**:
```bash
# 统计有多少电影有图片
ls data/Multimodal_Datasets/M_ML-100K/image/*.png | wc -l

# 与电影总数对比
wc -l data/Multimodal_Datasets/M_ML-100K/movies.dat
```

---

## 📊 数据统计

### MovieLens-100K 多模态

```
用户数: 943
电影数: 1,682
评分数: 100,000
图片数: 1,682 (100% 覆盖)
文本描述: 1,682 (100% 覆盖)
```

### MovieLens-1M 多模态

```
用户数: 6,040
电影数: 3,706
评分数: 1,000,209
图片数: 3,883 (>100%, 某些 ID 可能无评分数据)
文本描述: 3,706 (100% 覆盖)
```

---

## 🎯 与其他脚本的对比

| 特性 | preprocess_movielens.py | preprocess_multimodal_dataset.py |
|------|------------------------|----------------------------------|
| 数据源 | 下载原始 MovieLens | 使用本地多模态数据 |
| 文本描述 | 无（只有类型） | ✅ 来自 text.xls |
| 图片 | 需要额外下载 | ✅ 已包含 |
| 图片处理 | 需要单独脚本 | ✅ 集成复制和重命名 |
| ID 映射 | 简单 | 复杂（需要对应原始图片） |

**建议**: 如果有 `Multimodal_Datasets` 数据，优先使用 `preprocess_multimodal_dataset.py`。

---

## 📚 参考文档

- [WORKFLOW.md](WORKFLOW.md) - 完整工作流程
- [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - 训练指南
- [RETRIEVER_ANALYSIS.md](RETRIEVER_ANALYSIS.md) - 多模态检索器设计

---

**最后更新**: 2025-11-27
