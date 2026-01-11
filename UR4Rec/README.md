# FedDMMR: Federated Deep Multimodal Memory Recommendation

基于场景自适应异构混合专家(Scenario-Adaptive Heterogeneous MoE)的联邦推荐系统

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 项目结构

```
UR4Rec/
├── data/                          # 数据目录
│   └── ml-1m/                     # MovieLens-1M数据集
│       ├── subset_ratings.dat     # 用户交互数据（1000用户子集）
│       ├── clip_features.pt       # CLIP视觉特征（512维）
│       └── text_features.pt       # SBERT文本特征（384维）
│
├── models/                        # 模型定义
│   ├── sasrec.py                  # SASRec序列模型
│   ├── ur4rec_v2_moe.py          # FedDMMR主模型（Residual Enhancement架构）
│   ├── fedmem_client.py          # 联邦学习客户端
│   └── fedmem_server.py          # 联邦学习服务器
│
├── scripts/                       # 训练脚本
│   ├── preprocess_ml1m_subset.py # 数据预处理
│   ├── train_fedmem.py           # 主训练脚本
│   ├── train_stage1_backbone.py  # Stage 1训练（SASRec骨干）
│   ├── train_stage2_fixed.py     # Stage 2训练（模态对齐，修复版）
│   ├── train_stage3_skip_stage2.py # Stage 3训练（跳过Stage 2）
│   └── train_stage3_moe.py       # Stage 3训练（标准版）
│
├── checkpoints/                   # 模型checkpoint
│   ├── stage1_backbone/           # Stage 1模型
│   ├── stage2_alignment/          # Stage 2模型（已过拟合，不推荐使用）
│   ├── stage2_fixed/              # Stage 2修复版模型
│   └── stage3_skip_stage2/        # Stage 3模型（跳过Stage 2）
│
└── Stage/                         # 文档
    └── pitfalls.md                # Bug修复历史
```

## 核心架构：Residual Enhancement

```python
# 残差增强融合
fused_repr = seq_out_norm + gating_weight * auxiliary_repr

# auxiliary_repr = w_vis * vis_expert_out + w_sem * sem_expert_out
# Router只控制辅助专家权重，SASRec作为骨干直接保留
```

**关键特性**：
- ✅ SASRec输出直接保留（避免被路由器稀释）
- ✅ 可学习的gating_weight控制多模态信息注入强度
- ✅ 三阶段训练策略（可灵活调整）

## 快速开始

### 环境准备

```bash
# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install torch torchvision torchaudio
pip install numpy pandas scikit-learn tqdm
pip install sentence-transformers  # 用于文本特征
```

### 1. 数据预处理

```bash
# 准备MovieLens-1M数据（1000用户子集）
python UR4Rec/scripts/preprocess_ml1m_subset.py \
    --data_dir UR4Rec/data/ml-1m \
    --top_k 1000 \
    --output_file subset_ratings.dat

# 生成多模态特征（CLIP视觉 + SBERT文本）
# 注：这一步需要原始电影海报和描述数据
# 如果已有特征文件，可跳过此步
```

**预期输出**：
```
UR4Rec/data/ml-1m/
├── subset_ratings.dat      # 515,329条交互
├── clip_features.pt        # [3953, 512] 视觉特征
└── text_features.pt        # [3953, 384] 文本特征
```

### 2. 模型训练

#### 推荐方案：两阶段训练（跳过Stage 2）

**Stage 1: 训练SASRec骨干**
```bash
python UR4Rec/scripts/train_stage1_backbone.py
```

**训练配置**：
- 模型：纯SASRec（无多模态）
- 轮数：20轮
- 学习率：1e-3
- 预期性能：HR@10 ≈ 0.39-0.41

**输出**：
```
checkpoints/stage1_backbone/
├── fedmem_model.pt         # 最佳模型
├── train_history.json      # 训练历史
└── config.json             # 配置文件
```

---

**Stage 3: 训练MoE（跳过Stage 2）**
```bash
python UR4Rec/scripts/train_stage3_skip_stage2.py
```

**训练配置**：
- 加载：Stage 1的SASRec
- 训练：Projectors（从随机初始化）+ Router + SASRec
- 冻结：Item Embedding
- 轮数：30轮
- gating_init：0.01（让投影层快速学习）
- contrastive_lambda：0.5（增强对齐信号）
- 学习率：1e-3
- 预期性能：HR@10 ≈ 0.45-0.50+

**输出**：
```
checkpoints/stage3_skip_stage2/
├── fedmem_model.pt         # 最终模型
├── train_history.json      # 训练历史
└── config.json             # 配置文件
```

**预期训练曲线**：
```
Round 1-5:   HR@10 ≈ 0.41   (依赖Stage 1的SASRec)
Round 10-20: HR@10 ≈ 0.45   (投影层开始对齐)
Round 20-30: HR@10 ≈ 0.50+  (多模态融合生效)
```

---

#### 备选方案：三阶段训练（修复Stage 2）

**Stage 1**: 同上

**Stage 2: 模态对齐（修复版）**
```bash
python UR4Rec/scripts/train_stage2_fixed.py
```

**训练配置**：
- 加载：Stage 1的SASRec
- 训练：Projectors + Router + Gating Weight
- 冻结：SASRec + Item Embedding
- 轮数：10轮
- gating_init：0.01
- contrastive_lambda：0.5
- 学习率：5e-4
- 预期性能：HR@10 ≈ 0.43-0.45

**Stage 3: MoE集成微调**
```bash
# 需要先修改train_stage3_moe.py中的stage2_checkpoint路径
python UR4Rec/scripts/train_stage3_moe.py
```

**训练配置**：
- 加载：Stage 1的SASRec + Stage 2的Projectors
- 训练：所有组件（微调）
- 冻结：Item Embedding
- 轮数：20轮
- 学习率：5e-4
- 预期性能：HR@10 ≈ 0.50+

### 3. 模型评估

评估在训练过程中自动进行，每轮结束后评估验证集。

**查看训练历史**：
```bash
# 查看JSON格式
cat checkpoints/stage3_skip_stage2/train_history.json

# 或使用Python解析
python -c "
import json
with open('checkpoints/stage3_skip_stage2/train_history.json') as f:
    hist = json.load(f)
    for i, metrics in enumerate(hist['val_metrics'], 1):
        print(f'Round {i}: HR@10 = {metrics[\"HR@10\"]:.3f}')
"
```

**评估指标说明**：
- **HR@K** (Hit Rate): Top-K推荐命中率
- **NDCG@K**: 归一化折损累积增益
- **MRR**: 平均倒数排名

### 4. 使用训练好的模型

```python
import torch
from models.ur4rec_v2_moe import UR4RecV2MoE

# 加载模型
checkpoint = torch.load('checkpoints/stage3_skip_stage2/fedmem_model.pt')
model = UR4RecV2MoE(
    num_items=3953,
    sasrec_hidden_dim=128,
    # ... 其他参数
)
model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
model.eval()

# 推理
with torch.no_grad():
    scores = model(
        user_ids=None,
        input_seq=input_seq,      # [B, L] 用户历史序列
        target_items=candidate_items,  # [B, N] 候选物品
        target_visual=visual_feats,    # [B, N, 512] 视觉特征
        memory_visual=memory_vis,      # [B, TopK, 512] 记忆视觉
        memory_text=memory_text,       # [B, TopK, 384] 记忆文本
        training_mode=False
    )
    # scores: [B, N] 每个候选物品的得分
```

## 关键参数说明

### 模型参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `sasrec_hidden_dim` | 128 | SASRec隐藏层维度 |
| `sasrec_num_blocks` | 2 | Transformer层数 |
| `sasrec_num_heads` | 4 | 注意力头数 |
| `max_seq_len` | 50 | 最大序列长度 |
| `moe_num_heads` | 8 | 语义专家的注意力头数 |

### 训练参数

| 参数 | Stage 1 | Stage 2 (修复) | Stage 3 (跳过Stage 2) |
|------|---------|---------------|----------------------|
| `learning_rate` | 1e-3 | 5e-4 | 1e-3 |
| `gating_init` | N/A | 0.01 | 0.01 |
| `contrastive_lambda` | N/A | 0.5 | 0.5 |
| `num_rounds` | 20 | 10 | 30 |
| `batch_size` | 64 | 64 | 64 |
| `num_negatives` | 100 | 100 | 100 |

### 关键参数解释

**`gating_init`**：
- 控制多模态信息注入强度的初始值
- **太小（0.0001）**：保护SASRec但阻止投影层学习 ❌
- **适中（0.01）**：平衡保护与学习 ✅
- **太大（0.1+）**：可能破坏SASRec ⚠️

**`contrastive_lambda`**：
- 对比学习损失的权重
- 增大此值可加强多模态对齐
- Stage 2/3推荐使用0.5

**`training_mode`**：
- `True`：批内负采样模式，输出[B,B]矩阵
- `False`：标准评估模式，输出[B,N]得分

## 常见问题

### Q1: Stage 2训练后性能下降怎么办？

**A**: 使用推荐方案（跳过Stage 2），直接运行`train_stage3_skip_stage2.py`。

原因：Stage 2的gating_init=0.0001太小，投影层无法有效学习，导致过拟合。

### Q2: 训练速度太慢怎么办？

**A**:
- 使用GPU：在配置中设置`device="cuda"`
- 减少客户端数量：修改`client_fraction`参数
- 使用更小的batch_size：减少内存占用
- 参考：原项目文档中的加速指南

### Q3: 如何在自己的数据集上使用？

**A**:
1. 准备数据格式：每行一条交互记录（user_id, item_id, timestamp, rating）
2. 提取多模态特征：CLIP（图像）+ SBERT（文本）
3. 修改配置中的`num_items`参数
4. 运行训练脚本

### Q4: 为什么不使用原始的Stage 2？

**A**: 原始Stage 2（`checkpoints/stage2_alignment/`）存在过拟合问题：
- Round 1: HR@10 = 0.411 ✓
- Round 20: HR@10 = 0.338 ↓ (-17.8%)

原因是`gating_init=0.0001`太小。请使用修复版或直接跳过。

## 性能基准

基于MovieLens-1M子集（1000用户，3953物品）：

| 阶段 | HR@5 | HR@10 | HR@20 | NDCG@10 |
|------|------|-------|-------|---------|
| Stage 1 (SASRec) | 0.249 | 0.388 | 0.558 | 0.209 |
| Stage 3 (跳过Stage 2) | ~0.30 | ~0.45-0.50 | ~0.65 | ~0.28 |

**注意**：实际性能可能因随机种子、硬件等因素有所不同。

## 技术架构

### Residual Enhancement Fusion

```python
# 1. 获取各专家输出
seq_out = SASRec(input_seq)           # [B, D] 序列专家
vis_out = VisualExpert(visual_feat)   # [B, D] 视觉专家
sem_out = SemanticExpert(text_feat)   # [B, D] 语义专家

# 2. LayerNorm归一化
seq_out_norm = LayerNorm(seq_out)
vis_out_norm = LayerNorm(vis_out)
sem_out_norm = LayerNorm(sem_out)

# 3. Router决定辅助专家权重
router_weights = Router(target_item_emb)  # [B, 2]
w_vis, w_sem = router_weights[:, 0], router_weights[:, 1]

# 4. 残差增强融合
auxiliary_repr = w_vis * vis_out_norm + w_sem * sem_out_norm
fused_repr = seq_out_norm + gating_weight * auxiliary_repr

# 5. 评分
scores = dot_product(fused_repr, target_item_embs)
```

### 三阶段训练策略

| 阶段 | 训练对象 | 冻结对象 | 目标 |
|------|---------|---------|------|
| Stage 1 | SASRec + Item Embedding | - | 学习ID序列模式 |
| Stage 2 (可选) | Projectors + Router | SASRec + Item Embedding | 多模态对齐 |
| Stage 3 | All (微调) | Item Embedding | 学习Router + 整体优化 |

**灵活性**：
- 可以跳过Stage 2，直接从Stage 1到Stage 3
- Stage 3会同时训练投影层和Router

## 相关文档

- **[Stage/pitfalls.md](Stage/pitfalls.md)** - Bug修复历史（必读）
- **[models/FedDMMR_README.md](models/FedDMMR_README.md)** - 模型架构详解
- **[docs/drift_adaptive_learning.md](docs/drift_adaptive_learning.md)** - 漂移自适应学习

## 引用

如果使用本项目，请引用：
```bibtex
@article{feddmmr2026,
  title={FedDMMR: Federated Deep Multimodal Memory Recommendation with Scenario-Adaptive Heterogeneous MoE},
  author={...},
  journal={...},
  year={2026}
}
```

## 许可证

MIT License

---

**最后更新**: 2026-01-07
**维护状态**: ✅ 活跃维护
**推荐Python版本**: 3.8+
**推荐PyTorch版本**: 2.0+
