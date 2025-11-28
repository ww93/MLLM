# UR4Rec V2 训练指南

本文档详细介绍如何训练 UR4Rec V2 模型。

## 📋 目录

1. [训练流程概览](#训练流程概览)
2. [环境准备](#环境准备)
3. [数据准备](#数据准备)
4. [LLM 数据生成](#llm-数据生成)
5. [模型训练](#模型训练)
6. [训练阶段详解](#训练阶段详解)
7. [配置文件说明](#配置文件说明)
8. [常见问题](#常见问题)

---

## 训练流程概览

UR4Rec V2 的完整训练流程分为以下步骤：

```
1. 数据预处理
   ↓
2. LLM 离线生成用户偏好和物品描述
   ↓
3. 多阶段训练
   ├─ 阶段1: 预训练 SASRec
   ├─ 阶段2: 预训练检索器
   ├─ 阶段3: 联合微调
   └─ 阶段4: 端到端优化
   ↓
4. 模型评估和保存
```

---

## 环境准备

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖包括：
- PyTorch >= 2.0.0
- sentence-transformers
- transformers
- numpy, pandas
- pyyaml
- tqdm

### 2. （可选）配置 LLM API

如果使用真实的 LLM（非 Mock）：

**OpenAI**:
```bash
export OPENAI_API_KEY="your-api-key"
```

**Anthropic**:
```bash
export ANTHROPIC_API_KEY="your-api-key"
```

---

## 数据准备

### 1. 下载和预处理数据集

#### MovieLens-100K

```bash
python scripts/preprocess_movielens.py \
    --dataset ml-100k \
    --output_dir data/ml-100k \
    --num_candidates 100
```

#### MovieLens-1M

```bash
python scripts/preprocess_movielens.py \
    --dataset ml-1m \
    --output_dir data/ml-1m \
    --num_candidates 100
```

#### Amazon Beauty

```bash
python scripts/preprocess_beauty.py \
    --input_file data/raw/beauty.json \
    --output_dir data/beauty \
    --num_candidates 100
```

### 2. 验证数据

预处理完成后，检查数据目录是否包含以下文件：

```
data/ml-100k/
├── train_sequences.npy      # 训练序列
├── val_sequences.npy        # 验证序列
├── test_sequences.npy       # 测试序列
├── item_metadata.json       # 物品元数据
├── item_map.json           # 物品ID映射
└── user_map.json           # 用户ID映射
```

---

## LLM 数据生成

### 1. 使用 Mock 生成器（无需 API）

推荐用于快速测试：

```bash
python scripts/generate_llm_data.py \
    --config configs/movielens_100k.yaml \
    --data_dir data/ml-100k \
    --output_dir data/ml-100k/llm_generated \
    --llm_backend mock
```

### 2. 使用 OpenAI

```bash
python scripts/generate_llm_data.py \
    --config configs/movielens_100k.yaml \
    --data_dir data/ml-100k \
    --output_dir data/ml-100k/llm_generated \
    --llm_backend openai \
    --model_name gpt-3.5-turbo \
    --api_key $OPENAI_API_KEY
```

### 3. 使用 Anthropic

```bash
python scripts/generate_llm_data.py \
    --config configs/movielens_100k.yaml \
    --data_dir data/ml-100k \
    --output_dir data/ml-100k/llm_generated \
    --llm_backend anthropic \
    --model_name claude-3-haiku-20240307 \
    --api_key $ANTHROPIC_API_KEY
```

### 4. 生成选项

```bash
# 限制生成数量（用于测试）
python scripts/generate_llm_data.py \
    --config configs/movielens_100k.yaml \
    --data_dir data/ml-100k \
    --output_dir data/ml-100k/llm_generated \
    --llm_backend mock \
    --max_users 100 \
    --max_items 500

# 只生成用户偏好
python scripts/generate_llm_data.py \
    --config configs/movielens_100k.yaml \
    --data_dir data/ml-100k \
    --output_dir data/ml-100k/llm_generated \
    --llm_backend mock \
    --skip_items

# 只生成物品描述
python scripts/generate_llm_data.py \
    --config configs/movielens_100k.yaml \
    --data_dir data/ml-100k \
    --output_dir data/ml-100k/llm_generated \
    --llm_backend mock \
    --skip_users
```

### 5. 验证生成结果

检查输出目录：

```
data/ml-100k/llm_generated/
├── user_preferences.json    # 用户偏好描述
└── item_descriptions.json   # 物品文本描述
```

查看示例：

```bash
# 查看用户偏好示例
head -20 data/ml-100k/llm_generated/user_preferences.json

# 查看物品描述示例
head -20 data/ml-100k/llm_generated/item_descriptions.json
```

---

## 模型训练

### 1. 基础训练（文本模态）

```bash
python scripts/train_v2.py \
    --config configs/movielens_100k.yaml \
    --data_dir data/ml-100k \
    --llm_data_dir data/ml-100k/llm_generated \
    --output_dir outputs/ml-100k \
    --epochs_per_stage 10
```

### 2. 多模态训练（文本+图像）

```bash
python scripts/train_v2.py \
    --config configs/movielens_100k.yaml \
    --data_dir data/ml-100k \
    --llm_data_dir data/ml-100k/llm_generated \
    --output_dir outputs/ml-100k-multimodal \
    --use_multimodal \
    --epochs_per_stage 10
```

### 3. 自定义训练阶段

只训练某些阶段：

```bash
# 只预训练 SASRec
python scripts/train_v2.py \
    --config configs/movielens_100k.yaml \
    --data_dir data/ml-100k \
    --llm_data_dir data/ml-100k/llm_generated \
    --output_dir outputs/ml-100k-sasrec \
    --stages pretrain_sasrec

# 预训练后联合微调
python scripts/train_v2.py \
    --config configs/movielens_100k.yaml \
    --data_dir data/ml-100k \
    --llm_data_dir data/ml-100k/llm_generated \
    --output_dir outputs/ml-100k-joint \
    --stages pretrain_sasrec pretrain_retriever joint_finetune

# 完整四阶段训练
python scripts/train_v2.py \
    --config configs/movielens_100k.yaml \
    --data_dir data/ml-100k \
    --llm_data_dir data/ml-100k/llm_generated \
    --output_dir outputs/ml-100k-full \
    --stages pretrain_sasrec pretrain_retriever joint_finetune end_to_end
```

### 4. 训练参数调整

```bash
python scripts/train_v2.py \
    --config configs/movielens_100k.yaml \
    --data_dir data/ml-100k \
    --llm_data_dir data/ml-100k/llm_generated \
    --output_dir outputs/ml-100k \
    --epochs_per_stage 20 \      # 每阶段训练轮数
    --patience 10 \               # 早停耐心值
    --device cuda \               # 使用 GPU
    --seed 42                     # 随机种子
```

---

## 训练阶段详解

### 阶段 1: 预训练 SASRec

**目标**：训练序列推荐的基础能力

**训练内容**：
- 只训练 SASRec 参数
- 检索器参数冻结
- 使用 BPR 损失

**损失函数**：
```python
loss = -log(sigmoid(pos_score - neg_score))
```

**何时使用**：
- 首次训练模型
- 需要强化序列建模能力

**检查点**：`pretrain_sasrec_best.pt`

---

### 阶段 2: 预训练检索器

**目标**：训练文本偏好匹配能力

**训练内容**：
- SASRec 参数冻结
- 只训练检索器参数
- 使用检索损失（BCE）

**损失函数**：
```python
loss = BCE(retriever_scores, labels)
```

**何时使用**：
- SASRec 已预训练好
- 需要优化文本检索性能

**检查点**：`pretrain_retriever_best.pt`

---

### 阶段 3: 联合微调

**目标**：协调两个模块的输出

**训练内容**：
- 交替训练 SASRec 和检索器
- 奇数 batch 训练 SASRec
- 偶数 batch 训练检索器

**损失函数**：
```python
loss = sasrec_weight * sasrec_loss + retriever_weight * retriever_loss
```

**何时使用**：
- 两个模块都已预训练
- 需要平衡两路输出

**检查点**：`joint_finetune_best.pt`

---

### 阶段 4: 端到端优化

**目标**：全局最优化

**训练内容**：
- 所有参数一起训练
- 使用完整损失函数
- 支持不确定性加权

**损失函数**：

如果使用不确定性加权：
```python
loss = Σ (1/(2σ_i²)) * L_i + log(σ_i²)
```

否则：
```python
loss = α * sasrec_loss + β * retriever_loss
```

如果多模态：
```python
loss = retrieval_loss +
       consistency_weight * consistency_loss +
       contrastive_weight * contrastive_loss +
       diversity_weight * diversity_loss
```

**何时使用**：
- 前三阶段都已完成
- 追求最佳性能

**检查点**：`end_to_end_best.pt`

---

## 配置文件说明

### 配置文件结构

```yaml
# configs/movielens_100k.yaml

dataset:
  name: "MovieLens-100K"
  num_users: 943
  num_items: 1682

model:
  # SASRec 参数
  sasrec_hidden_dim: 256
  sasrec_num_blocks: 2
  sasrec_num_heads: 4
  sasrec_dropout: 0.1

  # 文本编码器参数
  text_model_name: "all-MiniLM-L6-v2"
  text_embedding_dim: 384
  retriever_output_dim: 256

  # 融合参数
  fusion_method: "weighted"  # weighted | rank | cascade
  sasrec_weight: 0.5
  retriever_weight: 0.5

  # 序列参数
  max_seq_len: 50

training:
  # 优化器
  sasrec_lr: 0.001
  retriever_lr: 0.0001
  weight_decay: 0.00001

  # 损失函数
  use_uncertainty_weighting: true
  retrieval_loss_weight: 1.0
  consistency_weight: 0.1
  contrastive_weight: 0.1
  diversity_weight: 0.01

  # 训练策略
  gradient_clip: 1.0
  warmup_steps: 100

  # 批次大小
  train_batch_size: 32
  eval_batch_size: 64

  # 负采样
  num_negatives: 5
  num_candidates: 100
```

### 关键参数说明

#### SASRec 参数

| 参数 | 说明 | 推荐值 |
|-----|------|--------|
| `sasrec_hidden_dim` | 隐藏层维度 | 128-512 |
| `sasrec_num_blocks` | Transformer 层数 | 2-4 |
| `sasrec_num_heads` | 注意力头数 | 2-8 |
| `sasrec_dropout` | Dropout 率 | 0.1-0.3 |

#### 检索器参数

| 参数 | 说明 | 推荐值 |
|-----|------|--------|
| `text_model_name` | Sentence-BERT 模型 | all-MiniLM-L6-v2 |
| `text_embedding_dim` | 文本嵌入维度 | 384 (由模型决定) |
| `retriever_output_dim` | 检索器输出维度 | 128-512 |

#### 融合方法

- `weighted`: 加权融合（推荐）
  ```python
  final_score = α * sasrec_score + β * retriever_score
  ```

- `rank`: 基于排名融合
  ```python
  final_score = α * rank_score(sasrec) + β * rank_score(retriever)
  ```

- `cascade`: 级联融合
  ```python
  final_score = sasrec_score + 0.5 * retriever_score
  ```

#### 训练策略

| 参数 | 说明 | 推荐值 |
|-----|------|--------|
| `sasrec_lr` | SASRec 学习率 | 1e-3 |
| `retriever_lr` | 检索器学习率 | 1e-4 |
| `use_uncertainty_weighting` | 自动任务加权 | true |
| `gradient_clip` | 梯度裁剪 | 1.0 |
| `warmup_steps` | 预热步数 | 100 |

---

## 常见问题

### Q1: 训练时内存不足怎么办？

**A**: 降低批次大小或模型维度

```yaml
# 在配置文件中调整
training:
  train_batch_size: 16  # 从 32 降低到 16
  eval_batch_size: 32   # 从 64 降低到 32

model:
  sasrec_hidden_dim: 128  # 从 256 降低到 128
  retriever_output_dim: 128  # 从 256 降低到 128
```

### Q2: LLM API 调用成本太高怎么办？

**A**: 使用 Mock 生成器或限制生成数量

```bash
# 使用 Mock（无需 API）
--llm_backend mock

# 限制生成数量
--max_users 1000
--max_items 2000
```

### Q3: 训练太慢怎么办？

**A**: 几种加速方法

1. **使用 GPU**:
```bash
--device cuda
```

2. **减少训练轮数**:
```bash
--epochs_per_stage 5
```

3. **跳过某些阶段**:
```bash
# 只做端到端训练
--stages end_to_end
```

4. **增加批次大小**（如果内存允许）:
```yaml
training:
  train_batch_size: 64
```

### Q4: 如何恢复中断的训练？

**A**: 使用检查点恢复

```python
# 在训练脚本中加载检查点
trainer.load_checkpoint('outputs/ml-100k/pretrain_sasrec_best.pt')
```

### Q5: 如何调试模型性能？

**A**: 查看训练日志和指标

训练过程中会输出：
```
Epoch 1/10
训练指标:
  total_loss: 0.5432
  sasrec_loss: 0.3210
  retriever_loss: 0.2222
  lr_sasrec: 0.001
  lr_retriever: 0.0001

验证指标:
  hit@5: 0.1234
  hit@10: 0.2345
  hit@20: 0.3456
  ndcg@5: 0.0987
  ndcg@10: 0.1543
  ndcg@20: 0.2109
  mrr: 0.1876
```

### Q6: 多模态训练失败怎么办？

**A**: 确认 CLIP 模型已安装

```bash
pip install transformers pillow
```

如果仍然失败，先训练文本模态版本：
```bash
# 不使用 --use_multimodal 标志
python scripts/train_v2.py \
    --config configs/movielens_100k.yaml \
    --data_dir data/ml-100k \
    --llm_data_dir data/ml-100k/llm_generated \
    --output_dir outputs/ml-100k
```

### Q7: 如何选择最佳模型？

**A**: 根据验证集指标

训练脚本会自动保存每个阶段的最佳模型：
- `pretrain_sasrec_best.pt`
- `pretrain_retriever_best.pt`
- `joint_finetune_best.pt`
- `end_to_end_best.pt`

通常 `end_to_end_best.pt` 性能最好。

测试集评估：
```python
# 加载最佳模型
model.load_state_dict(torch.load('outputs/ml-100k/end_to_end_best.pt'))

# 在测试集上评估
test_metrics = trainer.evaluate(test_loader)
```

### Q8: 如何调整融合权重？

**A**: 在配置文件中修改

```yaml
model:
  fusion_method: "weighted"
  sasrec_weight: 0.7      # 增加 SASRec 权重
  retriever_weight: 0.3   # 降低检索器权重
```

或者使用不确定性加权自动学习：
```yaml
training:
  use_uncertainty_weighting: true
```

### Q9: 训练完成后如何使用模型？

**A**: 加载模型进行推理

```python
import torch
from models.ur4rec_v2 import UR4RecV2

# 加载模型
model = UR4RecV2(num_items=1682, ...)
model.load_state_dict(torch.load('outputs/ml-100k/final_model.pt'))
model.eval()

# 加载 LLM 生成的数据
model.load_llm_generated_data(
    'data/ml-100k/llm_generated/user_preferences.json',
    'data/ml-100k/llm_generated/item_descriptions.json'
)

# 推理
with torch.no_grad():
    ranked_items = model.predict(
        user_ids=[1, 2, 3],
        input_seq=input_seq,
        candidate_items=candidate_items
    )
```

---

## 💡 最佳实践

### 1. 数据准备
- ✅ 确保数据预处理正确
- ✅ 验证 LLM 生成的文本质量
- ✅ 检查数据分布（用户活跃度、物品流行度）

### 2. 训练策略
- ✅ 先快速训练一个小规模版本（减少用户/物品数）
- ✅ 验证代码和流程没问题后再全量训练
- ✅ 使用早停避免过拟合
- ✅ 定期保存检查点

### 3. 超参数调优
- ✅ 先用默认参数训练
- ✅ 然后调整学习率
- ✅ 最后调整模型维度和层数

### 4. 性能优化
- ✅ 使用 GPU 加速
- ✅ 合理设置批次大小
- ✅ 使用多进程数据加载

### 5. 结果分析
- ✅ 对比不同阶段的性能
- ✅ 分析 SASRec 和检索器的贡献
- ✅ 可视化训练曲线

---

## 📚 相关文档

- [README_CN.md](README_CN.md) - 项目总览
- [QUICKSTART_CN.md](QUICKSTART_CN.md) - 快速开始
- [REFACTORING_PROGRESS.md](REFACTORING_PROGRESS.md) - 重构进度
- [PROJECT_SUMMARY_CN.md](PROJECT_SUMMARY_CN.md) - 技术细节

---

**最后更新**: 2025-11-27
