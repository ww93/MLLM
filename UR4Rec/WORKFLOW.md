# UR4Rec V2 完整工作流程

本文档描述 UR4Rec V2 的完整数据处理和训练流程。

---

## 📋 总体流程图

```
Step 1: 数据预处理
    ├── 下载原始数据
    ├── 构建用户序列
    ├── 划分训练/验证/测试集
    └── 生成物品元数据
         ↓
Step 2: 图片数据准备（可选）
    ├── 下载物品图片
    └── 提取 CLIP 特征
         ↓
Step 3: LLM 数据生成
    ├── 生成用户偏好文本
    └── 生成物品描述文本
         ↓
Step 4: 模型训练
    ├── Stage 1: 预训练 SASRec
    ├── Stage 2: 预训练检索器
    ├── Stage 3: 联合微调
    └── Stage 4: 端到端优化
         ↓
Step 5: 模型评估
    ├── 测试集评估
    └── 生成推荐结果
```

---

## 🚀 详细步骤

### Step 1: 数据预处理

#### 1.1 MovieLens 数据集

```bash
# MovieLens-100K
python scripts/preprocess_movielens.py \
    --dataset ml-100k \
    --output_dir data/ml-100k \
    --num_candidates 100

# MovieLens-1M
python scripts/preprocess_movielens.py \
    --dataset ml-1m \
    --output_dir data/ml-1m \
    --num_candidates 100
```

**输出文件**：
```
data/ml-100k/
├── train_sequences.npy      # 训练序列 {user_id: [item_ids]}
├── val_sequences.npy        # 验证序列
├── test_sequences.npy       # 测试序列
├── item_metadata.json       # 物品元数据 {item_id: {title, genres}}
├── item_map.json           # 物品ID映射 {original_id: new_id}
├── user_map.json           # 用户ID映射
└── stats.json              # 数据统计
```

#### 1.2 Amazon Beauty 数据集

```bash
python scripts/preprocess_beauty.py \
    --input_file data/raw/beauty.json \
    --output_dir data/beauty \
    --num_candidates 100
```

**输出文件结构同上**

---

### Step 2: 图片数据准备（可选，用于多模态）

#### 2.1 下载图片

**MovieLens（需要 TMDB API）**：

```bash
# 获取 TMDB API 密钥: https://www.themoviedb.org/settings/api

python scripts/download_images.py \
    --dataset movielens \
    --item_metadata data/ml-100k/item_metadata.json \
    --output_dir data/ml-100k/images \
    --tmdb_api_key YOUR_API_KEY

# 或使用占位图片（无需 API）
python scripts/download_images.py \
    --dataset movielens \
    --item_metadata data/ml-100k/item_metadata.json \
    --output_dir data/ml-100k/images
```

**Amazon Beauty（从元数据中的 URL 下载）**：

```bash
python scripts/download_images.py \
    --dataset amazon \
    --item_metadata data/beauty/item_metadata.json \
    --output_dir data/beauty/images
```

**输出文件**：
```
data/ml-100k/images/
├── 1.jpg                  # 物品 1 的图片
├── 2.jpg
├── ...
└── download_log.json      # 下载日志
```

#### 2.2 提取图片特征（使用 CLIP）

```bash
# 提取 CLIP 特征
python scripts/preprocess_images.py \
    --image_dir data/ml-100k/images \
    --output_path data/ml-100k/image_features.pt \
    --mode clip \
    --batch_size 32

# 或创建调整大小后的图片缓存
python scripts/preprocess_images.py \
    --image_dir data/ml-100k/images \
    --output_path data/ml-100k/images_224 \
    --mode resize \
    --target_size 224 224
```

**输出文件**：
```
data/ml-100k/
├── image_features.pt         # CLIP 特征 {embeddings, ids, features_dict}
└── images_224/               # 或调整大小后的图片
    ├── 1.jpg
    ├── 2.jpg
    └── ...
```

---

### Step 3: LLM 数据生成

#### 3.1 使用 Mock 生成器（推荐用于测试）

```bash
python scripts/generate_llm_data.py \
    --config configs/movielens_100k.yaml \
    --data_dir data/ml-100k \
    --output_dir data/ml-100k/llm_generated \
    --llm_backend mock
```

#### 3.2 使用 OpenAI GPT

```bash
export OPENAI_API_KEY="your-api-key"

python scripts/generate_llm_data.py \
    --config configs/movielens_100k.yaml \
    --data_dir data/ml-100k \
    --output_dir data/ml-100k/llm_generated \
    --llm_backend openai \
    --model_name gpt-3.5-turbo \
    --api_key $OPENAI_API_KEY
```

#### 3.3 使用 Anthropic Claude

```bash
export ANTHROPIC_API_KEY="your-api-key"

python scripts/generate_llm_data.py \
    --config configs/movielens_100k.yaml \
    --data_dir data/ml-100k \
    --output_dir data/ml-100k/llm_generated \
    --llm_backend anthropic \
    --model_name claude-3-haiku-20240307 \
    --api_key $ANTHROPIC_API_KEY
```

#### 3.4 限制生成数量（用于快速测试）

```bash
python scripts/generate_llm_data.py \
    --config configs/movielens_100k.yaml \
    --data_dir data/ml-100k \
    --output_dir data/ml-100k/llm_generated \
    --llm_backend mock \
    --max_users 100 \
    --max_items 500
```

**输出文件**：
```
data/ml-100k/llm_generated/
├── user_preferences.json     # {"user_1": "该用户喜欢动作和科幻电影..."}
└── item_descriptions.json    # {"item_1": "一部经典的科幻动作片..."}
```

---

### Step 4: 模型训练

#### 4.1 文本模态训练（基础版本）

```bash
python scripts/train_v2.py \
    --config configs/movielens_100k.yaml \
    --data_dir data/ml-100k \
    --llm_data_dir data/ml-100k/llm_generated \
    --output_dir outputs/ml-100k \
    --epochs_per_stage 10 \
    --device cuda
```

#### 4.2 多模态训练（文本+图像）

```bash
python scripts/train_v2.py \
    --config configs/movielens_100k.yaml \
    --data_dir data/ml-100k \
    --llm_data_dir data/ml-100k/llm_generated \
    --output_dir outputs/ml-100k-multimodal \
    --use_multimodal \
    --epochs_per_stage 10 \
    --device cuda
```

#### 4.3 自定义训练阶段

```bash
# 只预训练 SASRec
python scripts/train_v2.py \
    --config configs/movielens_100k.yaml \
    --data_dir data/ml-100k \
    --llm_data_dir data/ml-100k/llm_generated \
    --output_dir outputs/ml-100k-sasrec \
    --stages pretrain_sasrec \
    --epochs_per_stage 20

# 完整四阶段训练
python scripts/train_v2.py \
    --config configs/movielens_100k.yaml \
    --data_dir data/ml-100k \
    --llm_data_dir data/ml-100k/llm_generated \
    --output_dir outputs/ml-100k-full \
    --stages pretrain_sasrec pretrain_retriever joint_finetune end_to_end \
    --epochs_per_stage 15 \
    --patience 5
```

**输出文件**：
```
outputs/ml-100k/
├── pretrain_sasrec_best.pt      # 阶段1最佳模型
├── pretrain_retriever_best.pt   # 阶段2最佳模型
├── joint_finetune_best.pt       # 阶段3最佳模型
├── end_to_end_best.pt           # 阶段4最佳模型
├── final_model.pt               # 最终模型
├── results.json                 # 训练结果和指标
└── checkpoints/                 # 训练检查点
```

---

### Step 5: 模型评估

训练脚本会自动在测试集上评估，结果保存在 `results.json` 中。

**评估指标**：
- `hit@5`, `hit@10`, `hit@20` - 命中率
- `ndcg@5`, `ndcg@10`, `ndcg@20` - 归一化折扣累积增益
- `mrr` - 平均倒数排名

---

## 🔄 完整流程示例

### MovieLens-100K 完整训练流程

```bash
# 1. 数据预处理
python scripts/preprocess_movielens.py \
    --dataset ml-100k \
    --output_dir data/ml-100k \
    --num_candidates 100

# 2. （可选）下载图片
python scripts/download_images.py \
    --dataset movielens \
    --item_metadata data/ml-100k/item_metadata.json \
    --output_dir data/ml-100k/images \
    --tmdb_api_key YOUR_KEY  # 或省略使用占位图片

# 3. （可选）提取图片特征
python scripts/preprocess_images.py \
    --image_dir data/ml-100k/images \
    --output_path data/ml-100k/image_features.pt \
    --mode clip

# 4. 生成 LLM 数据
python scripts/generate_llm_data.py \
    --config configs/movielens_100k.yaml \
    --data_dir data/ml-100k \
    --output_dir data/ml-100k/llm_generated \
    --llm_backend mock

# 5. 训练模型（文本模态）
python scripts/train_v2.py \
    --config configs/movielens_100k.yaml \
    --data_dir data/ml-100k \
    --llm_data_dir data/ml-100k/llm_generated \
    --output_dir outputs/ml-100k \
    --epochs_per_stage 10

# 6. （可选）训练多模态模型
python scripts/train_v2.py \
    --config configs/movielens_100k.yaml \
    --data_dir data/ml-100k \
    --llm_data_dir data/ml-100k/llm_generated \
    --output_dir outputs/ml-100k-multimodal \
    --use_multimodal \
    --epochs_per_stage 10
```

---

## ⚙️ 核心模块说明

### 数据处理模块

| 脚本 | 功能 | 输入 | 输出 |
|-----|------|------|------|
| `preprocess_movielens.py` | MovieLens 数据预处理 | 原始评分数据 | 序列 + 元数据 |
| `preprocess_beauty.py` | Amazon 数据预处理 | JSON 数据 | 序列 + 元数据 |
| `download_images.py` | 下载物品图片 | 元数据 | 图片文件 |
| `preprocess_images.py` | 提取图片特征 | 图片文件 | CLIP 特征 |
| `generate_llm_data.py` | LLM 数据生成 | 序列 + 元数据 | 文本描述 |

### 模型模块

| 文件 | 功能 | 说明 |
|-----|------|------|
| `models/sasrec.py` | SASRec 序列模型 | 基于 Transformer 的序列推荐 |
| `models/text_preference_retriever.py` | 文本检索器 | 使用 Sentence-BERT 编码文本 |
| `models/multimodal_retriever.py` | 多模态检索器 | 文本+图像跨模态检索 |
| `models/ur4rec_v2.py` | UR4Rec 整合模型 | 融合 SASRec 和检索器 |
| `models/llm_generator.py` | LLM 生成器 | 离线生成偏好描述 |

### 训练模块

| 文件 | 功能 | 说明 |
|-----|------|------|
| `models/multimodal_loss.py` | 多模态损失函数 | 检索/一致性/对比/多样性损失 |
| `models/joint_trainer.py` | 联合训练器 | 多阶段训练管理 |
| `scripts/train_v2.py` | 主训练脚本 | 完整训练流程 |

---

## 🐛 常见问题

### Q1: 数据预处理失败

**问题**: `FileNotFoundError: [Errno 2] No such file or directory`

**解决**:
```bash
# 检查原始数据是否存在
ls data/raw/

# MovieLens 会自动下载，确保有网络连接
# Amazon 需要手动下载数据文件
```

### Q2: 图片下载失败

**问题**: TMDB API 返回 401 Unauthorized

**解决**:
```bash
# 检查 API 密钥是否正确
echo $TMDB_API_KEY

# 或使用占位图片（无需 API）
python scripts/download_images.py \
    --dataset movielens \
    --item_metadata data/ml-100k/item_metadata.json \
    --output_dir data/ml-100k/images
    # 不提供 --tmdb_api_key
```

### Q3: CLIP 特征提取内存不足

**问题**: `CUDA out of memory`

**解决**:
```bash
# 减小批次大小
python scripts/preprocess_images.py \
    --image_dir data/ml-100k/images \
    --output_path data/ml-100k/image_features.pt \
    --mode clip \
    --batch_size 8  # 从 32 降低到 8

# 或使用 CPU
python scripts/preprocess_images.py \
    --image_dir data/ml-100k/images \
    --output_path data/ml-100k/image_features.pt \
    --mode clip \
    --device cpu
```

### Q4: LLM 数据生成太慢

**问题**: OpenAI API 调用很慢

**解决**:
```bash
# 方案1: 使用 Mock 生成器
--llm_backend mock

# 方案2: 限制生成数量
--max_users 500 --max_items 1000

# 方案3: 使用更快的模型
--model_name gpt-3.5-turbo  # 代替 gpt-4
```

### Q5: 训练显存不足

**问题**: `RuntimeError: CUDA out of memory`

**解决**: 在配置文件中调整参数
```yaml
training:
  train_batch_size: 16  # 从 32 降低
  eval_batch_size: 32   # 从 64 降低

model:
  sasrec_hidden_dim: 128  # 从 256 降低
  retriever_output_dim: 128  # 从 256 降低
```

---

## 📊 数据流详解

### 训练时的数据流

```
1. 加载序列数据
   train_sequences.npy → {user_id: [item_1, item_2, ..., item_n]}

2. 采样训练样本
   用户序列 → 输入序列(前n-1) + 目标物品(第n个) + 负样本(随机)

3. LLM 文本数据
   user_preferences.json → 查找用户偏好文本
   item_descriptions.json → 查找物品描述文本

4. 模型前向传播
   SASRec: 输入序列 → 序列表示 → 候选物品分数
   检索器: 偏好文本 → 偏好向量 → 物品匹配分数
   融合: SASRec分数 + 检索器分数 → 最终分数

5. 损失计算
   BPR损失: -log(sigmoid(正样本分数 - 负样本分数))
   检索损失: BCE(检索分数, 标签)

6. 反向传播
   根据训练阶段更新对应参数
```

### 推理时的数据流

```
1. 输入
   - 用户ID
   - 用户历史序列
   - 候选物品列表

2. SASRec 分数
   历史序列 → Transformer → 序列表示 → 候选物品分数

3. 检索器分数
   用户ID → 查找偏好文本 → 文本编码 → 偏好向量
   候选物品 → 物品嵌入 → 物品向量
   余弦相似度(偏好向量, 物品向量) → 检索分数

4. 融合
   加权融合: α * SASRec分数 + β * 检索分数 → 最终分数

5. 排序
   根据最终分数对候选物品排序 → Top-K 推荐列表
```

---

## 🎯 最佳实践

### 1. 快速测试流程

```bash
# 使用最小数据集快速验证
python scripts/preprocess_movielens.py --dataset ml-100k --output_dir data/ml-100k
python scripts/generate_llm_data.py --llm_backend mock --max_users 100 --max_items 500 ...
python scripts/train_v2.py --epochs_per_stage 2 ...
```

### 2. 完整训练流程

```bash
# 使用完整数据 + 真实 LLM
python scripts/preprocess_movielens.py --dataset ml-1m --output_dir data/ml-1m
python scripts/generate_llm_data.py --llm_backend openai --api_key $OPENAI_API_KEY ...
python scripts/train_v2.py --epochs_per_stage 20 --patience 10 ...
```

### 3. 多模态训练建议

```bash
# 1. 先训练纯文本模型（验证基础架构）
python scripts/train_v2.py --output_dir outputs/text_only

# 2. 下载并预处理图片
python scripts/download_images.py ...
python scripts/preprocess_images.py ...

# 3. 训练多模态模型
python scripts/train_v2.py --use_multimodal --output_dir outputs/multimodal
```

---

## 📚 相关文档

- [README_CN.md](README_CN.md) - 项目总览
- [QUICKSTART_CN.md](QUICKSTART_CN.md) - 快速开始指南
- [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - 详细训练教程
- [REFACTORING_PROGRESS.md](REFACTORING_PROGRESS.md) - 重构进度和架构说明
- [RETRIEVER_ANALYSIS.md](RETRIEVER_ANALYSIS.md) - 检索器设计分析

---

**最后更新**: 2025-11-27
