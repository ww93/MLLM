# 数据处理记录

## 2025-01-07: Amazon Beauty & Clothing 数据集处理

### 处理目标
为 FedMem 训练准备多模态数据，包括：
- 用户-物品交互序列
- CLIP 图像特征（512维）
- Sentence-BERT 文本特征（384维）

### 数据集信息

#### Beauty 数据集
- **原始文件**:
  - `ratings_Beauty.csv`: 用户评分数据
  - `meta_Beauty.json`: 产品元数据（Python dict格式）
  - `reviews_Beauty_5.json`: 评论数据（5-core）

- **元数据字段**:
  - `asin`: 产品ID
  - `title`: 产品标题
  - `description`: 产品描述（详细，用于文本embedding）
  - `imUrl`: 产品图片URL
  - `categories`: 产品类别
  - `salesRank`: 销售排名

#### Clothing 数据集
- **原始文件**:
  - `ratings_Clothing_Shoes_and_Jewelry.csv`: 用户评分数据
  - `meta_Clothing_Shoes_and_Jewelry.json`: 产品元数据
  - `reviews_Clothing_Shoes_and_Jewelry_5.json`: 评论数据

- **元数据字段**:
  - `asin`: 产品ID
  - `title`: 产品标题
  - `brand`: 品牌
  - `imUrl`: 产品图片URL
  - `categories`: 产品类别
  - **注意**: 无 `description` 字段

### 处理方案

#### 脚本: `process_amazon_dataset.py`

通用处理脚本，支持 Beauty 和 Clothing 数据集。

**主要步骤**:

1. **交互数据处理**
   - 加载 ratings CSV 文件
   - K-core 过滤（默认 k=5）：递归过滤掉交互少于5次的用户和物品
   - ID 映射：将原始 ID 映射为连续数字（1-based，0保留给padding）
   - 按时间排序保存为 FedMem 格式

2. **图像特征提取**
   - 从 meta 文件提取图片 URL
   - 多线程下载图片（32线程）
   - 使用 CLIP ViT-B/32 提取图像特征
   - L2 归一化
   - 保存为 `[num_items+1, 512]` 的 tensor

3. **文本特征提取**
   - **Beauty**: 使用 `title + description`（description 包含详细产品信息）
   - **Clothing**: 使用 `title + brand + categories`（因为无 description）
   - 使用 Sentence-BERT (all-MiniLM-L6-v2) 提取文本 embedding
   - L2 归一化
   - 保存为 `[num_items+1, 384]` 的 tensor

4. **输出文件**
   - `subset_ratings.dat`: 交互序列（user_id item_id rating timestamp）
   - `clip_features.pt`: CLIP 图像特征
   - `text_features.pt`: 文本特征
   - `item_map.json`: item_id 到 idx 的映射
   - `images/`: 下载的图片目录

### 使用方法

```bash
# 处理 Beauty 数据集
python UR4Rec/data/process_amazon_dataset.py --dataset Beauty

# 处理 Clothing 数据集
python UR4Rec/data/process_amazon_dataset.py --dataset Clothing

# 指定自定义参数
python UR4Rec/data/process_amazon_dataset.py \
  --dataset Beauty \
  --data_dir /path/to/data \
  --min_interactions 10 \
  --device cuda
```

### 训练 FedMem

处理完成后，使用以下命令训练：

```bash
# Beauty 数据集
python UR4Rec/scripts/train_fedmem.py \
  --data_dir UR4Rec/data/Beauty \
  --visual_file clip_features.pt \
  --text_file text_features.pt \
  --save_dir checkpoints/fedmem_beauty

# Clothing 数据集
python UR4Rec/scripts/train_fedmem.py \
  --data_dir UR4Rec/data/Clothing \
  --visual_file clip_features.pt \
  --text_file text_features.pt \
  --save_dir checkpoints/fedmem_clothing
```

### 技术细节

#### 数据格式兼容性
- **原始格式**: Python dict（使用 `ast.literal_eval` 解析）
- **FedMem 格式**: 空格分隔的文本文件

#### K-core 过滤算法
递归过滤直到收敛：
1. 统计每个用户和物品的交互次数
2. 移除交互次数 < k 的用户和物品
3. 重复直到没有变化

#### 多模态特征对齐
- **图像**: CLIP ViT-B/32 → 512维
- **文本**: Sentence-BERT → 384维
- **索引对齐**: 使用相同的 item_idx 映射
- **Padding**: 索引0保留为全零向量

#### 缺失数据处理
- **无图片**: 保持全零向量（模型会处理）
- **无 description**: 使用 title + brand + categories

### 预期效果

处理后的数据可以直接用于 FedMem 训练，支持：
- 纯 ID 推荐（Stage 1）
- 多模态特征对齐（Stage 2）
- MoE 集成推荐（Stage 3）

### 注意事项

1. **内存消耗**: CLIP 和 Sentence-BERT 需要 GPU 内存（建议 8GB+）
2. **下载时间**: 图片下载可能需要较长时间（取决于网络和物品数量）
3. **文本长度**: 自动截断至 512 字符，避免 BERT 溢出
4. **L2 归一化**: 所有特征都进行 L2 归一化，便于计算相似度

---

**处理完成标志**: 当所有4个输出文件生成后，即可开始训练。

---

## 2025-01-08: 模态一致性分析工具升级

### 修改目标
扩展 `modality_compare.py` 脚本，使其同时支持 Amazon 和 ML-1M 数据集格式，用于分析不同数据集的多模态一致性。

### ML-1M 数据集格式

**目录结构**: `UR4Rec/data/Multimodal_Datasets/M_ML-1M/`

**文件说明**:
- `movies.dat`: 电影元数据（格式：`MovieID::Title::Genres`，用 `::` 分隔）
- `ratings.dat`: 用户评分数据（格式：`UserID::MovieID::Rating::Timestamp`）
- `text.xls`: 电影文本描述（Excel格式，包含 MovieID 和 description 列）
- `image/`: 电影海报图片目录（格式：`{MovieID}.png`）

**数据特点**:
- 本地存储的图片（无需在线下载）
- 基于用户时间序列的交互数据
- 3883 部电影，约 3706 张海报图片

### 修改内容

#### 1. 数据集配置扩展

添加对两种数据集类型的支持：

```python
DATA_PATHS = {
    # Amazon 数据集（基于 co-occurrence: also_bought）
    'Movies_Amazon': {
        'type': 'amazon',
        'path': 'UR4Rec/data/Movies_and_TV/meta_Movies_and_TV.json'
    },
    'Beauty': {
        'type': 'amazon',
        'path': 'UR4Rec/data/Beauty/meta_Beauty.json'
    },

    # ML-1M 数据集（基于用户交互序列）
    'ML-1M': {
        'type': 'ml1m',
        'base_dir': 'UR4Rec/data/Multimodal_Datasets/M_ML-1M'
    }
}
```

#### 2. 新增 ML-1M 处理函数

**函数**: `compute_ml1m_sequence_consistency()`

**分析方法**:
- **Amazon 数据集**: 计算 co-occurrence pairs（also_bought 关系）的模态相似度
- **ML-1M 数据集**: 计算用户交互序列中相邻电影的模态相似度

**处理流程**:
1. 加载电影元数据（movies.dat）
2. 加载文本描述（text.xls）
3. 加载用户交互序列（ratings.dat）
4. 按时间排序用户序列
5. 过滤短序列用户（min_seq_len >= 10）
6. 采样用户（默认 500 个）
7. 对每个用户，随机选择序列中的相邻电影对（最多 5 对）
8. 计算相邻电影对的视觉和语义相似度

#### 3. 新增辅助函数

**`load_local_image(image_path)`**:
- 从本地文件加载图片（用于 ML-1M）
- 支持 PNG 格式
- 自动转换为 RGB 模式

#### 4. 主流程修改

支持根据数据集类型调用不同的处理函数：

```python
for cat, config in DATA_PATHS.items():
    if config['type'] == 'amazon':
        v_score, s_score = compute_co_occurrence_consistency(config['path'], cat)
    elif config['type'] == 'ml1m':
        v_score, s_score = compute_ml1m_sequence_consistency(config['base_dir'], cat)
```

### 使用方法

```bash
# 运行模态一致性分析（同时处理 Amazon 和 ML-1M 数据集）
python UR4Rec/data/modality_compare.py
```

**配置选项**:
- `SAMPLE_ITEMS`: 采样数量（Amazon: also_bought对数；ML-1M: 用户数）
- `ML1M_MIN_SEQ_LEN`: ML-1M 最小序列长度（默认 10）
- `DEVICE`: 计算设备（cuda/cpu）

### 输出结果

**终端输出**:
- 每个数据集的平均视觉相似度（Visual Similarity）
- 每个数据集的平均语义相似度（Semantic Similarity）
- 处理统计信息

**可视化输出**:
- `modality_consistency_comparison.pdf`: 对比柱状图
- 显示不同数据集和模态的相似度得分

### 技术对比

| 特性 | Amazon 数据集 | ML-1M 数据集 |
|------|--------------|--------------|
| 关系类型 | Co-occurrence (also_bought) | 时间序列（用户观影顺序） |
| 图片来源 | 在线下载（URL） | 本地文件（.png） |
| 文本来源 | meta文件（title + description） | text.xls（description） |
| 相似度计算 | 共同购买的物品对 | 用户序列中的相邻电影对 |
| 采样单位 | 物品对 | 用户 |

### 预期效果

该工具可用于：
1. **动机分析**: 证明不同模态在推荐任务中的一致性
2. **数据集对比**: 分析不同数据集的多模态特性
3. **模型设计**: 为多模态推荐系统提供理论依据

### 关键发现

通过对比分析可以发现：
- **视觉一致性**: 相关物品/电影在视觉上的相似程度
- **语义一致性**: 相关物品/电影在语义上的相似程度
- **模态差异**: 不同模态捕捉的信息互补性

---

**修改完成**: modality_compare.py 现已支持 Amazon 和 ML-1M 两种数据集格式。
