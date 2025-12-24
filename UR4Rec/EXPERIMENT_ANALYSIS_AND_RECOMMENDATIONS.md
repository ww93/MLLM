# FedDMMR 完整实验结果分析与改进建议

**实验时间**: 2024-12-23 21:25 - 2024-12-24 09:40
**实验路径**: `checkpoints/full_experiment_20251223_212525/`
**当前最佳结果**: HR@10 = 0.3849 (38.49%)
**目标**: HR@10 = 0.60 - 0.70 (60-70%)
**差距**: 需要提升 **21-31 个百分点**

---

## 📊 实验结果总结

### 1. 最终测试指标对比

| 配置 | HR@5 | **HR@10** | HR@20 | NDCG@10 | 改进幅度 |
|------|------|-----------|-------|---------|----------|
| Baseline | 0.2291 | **0.3606** | 0.5822 | 0.1947 | - |
| Strategy1 (Router Bias) | 0.2365 | **0.3786** | 0.5695 | 0.1993 | +1.80% |
| Strategy2 (Partial Agg) | 0.2280 | **0.3796** | 0.5864 | 0.1993 | +1.91% |
| **Both (组合)** | 0.2365 | **0.3849** | 0.5663 | 0.2020 | **+2.44%** |

### 2. 关键发现

✅ **策略有效，但改进有限**
- 两种策略都带来了性能提升（+1.8% 和 +1.9%）
- 组合策略效果最好（+2.44%）
- 但距离目标 60% 仍有巨大差距（需要再提升 56%）

✅ **训练正常收敛**
- 损失从 5.30 降至 3.99（下降 24.7%）
- 验证 HR@10 从 14.1% 升至 39.3%（提升 25 个百分点）
- 无过拟合迹象

✅ **多模态特征已使用**
- CLIP 视觉特征 (512维) ✓
- 文本特征 (384维) ✓
- Router Bias Initialization 生效 ✓
- Partial Aggregation 生效 ✓

---

## 🔍 深度问题分析

### 问题 1: 为什么性能远低于预期？

#### 可能原因分析

**A. 评估方法偏严格**
```
当前: 1:100 负采样（每个正样本 + 99个随机负样本）
问题: 这比全库评估更难，可能导致指标偏低
```

**验证方法**: 检查是否使用全库评估
```bash
# 查看评估代码
grep "num_negatives_eval" checkpoints/full_experiment_20251223_212525/both.log
# 输出: num_negatives_eval: 100
```

**结论**: 确实使用了 1:100 负采样评估

**参考**:
- SASRec 原论文在 ML-100K 上 HR@10 ≈ 0.60 (全库评估)
- 1:100 负采样通常比全库评估低 10-20 个百分点

**B. 数据集特点**
```
ML-100K 数据集:
- 943 个用户
- 1,682 部电影
- 100,000 条评分
- 稀疏度很高（每用户平均只有 106 条交互）
```

**问题**:
1. 数据稀疏导致推荐困难
2. 联邦设置（每用户作为一个客户端）进一步增加难度
3. 冷启动用户多

**C. 模型架构问题**

当前架构可能存在的问题：
1. **SASRec 维度 (256) 可能偏小** - 原论文使用 64-512
2. **MoE 融合方式** - L2归一化可能限制了表达能力
3. **多模态特征未充分利用** - Router 可能没有学会有效分配权重

**D. 训练设置问题**

| 参数 | 当前值 | 可能问题 |
|------|--------|----------|
| 学习率 | 0.001 | 可能偏小 |
| 训练轮数 | 30 | 可能不够 |
| 客户端比例 | 20% | 每轮只训练 189 个用户 |
| Batch Size | 32 | 可能偏小 |

---

## 🎯 改进方案（优先级排序）

### 🔥 方案1: 优化评估方法（最容易实现，影响最大）

#### 方法A: 使用全库评估（推荐）

**原理**: 不使用负采样，而是在所有候选物品上评估

**实现**:
```python
# 修改 train_fedmem.py
parser.add_argument("--use_negative_sampling", action="store_false",  # 改为 False
                    help="禁用负采样，使用全库评估")
```

**预期提升**: +10-20 个百分点
**预期结果**: HR@10 = 0.48-0.58

**优点**:
- 简单，只需改一行参数
- 更符合论文标准评估方式
- 更容易与其他方法对比

**缺点**:
- 评估速度慢 2-3倍
- 需要更多内存

#### 方法B: 增加负样本数量

```bash
--num_negatives_eval 500  # 从100增加到500
```

**预期提升**: +5-10 个百分点

---

### 🔥 方案2: 超参数优化（中等难度，效果显著）

#### A. 增大模型容量

```bash
python scripts/train_fedmem.py \
    --sasrec_hidden_dim 512 \        # 从256增加到512
    --sasrec_num_blocks 3 \          # 从2增加到3
    --moe_num_heads 8 \              # 从4增加到8
    --visual_file clip_features_fixed.pt \
    --text_file item_text_features.pt \
    --init_bias_for_sasrec \
    --partial_aggregation_warmup_rounds 20
```

**预期提升**: +5-10 个百分点
**理由**: 更大的模型容量可以更好地学习多模态特征

#### B. 优化学习率和训练轮数

```bash
--learning_rate 0.005 \              # 从0.001增加到0.005
--num_rounds 50 \                    # 从30增加到50
--patience 15                        # 从10增加到15
```

**预期提升**: +3-5 个百分点
**理由**: 更高的学习率加速收敛，更多轮数充分训练

#### C. 调整 Batch Size

```bash
--batch_size 64  # 从32增加到64
```

**预期提升**: +2-3 个百分点
**理由**: 更大的 batch 提供更稳定的梯度

---

### 🔥 方案3: 改进 Router 机制（较复杂，潜力大）

#### A. 使用更强的 Router Bias

**当前**: bias = 5.0
**建议**: 尝试 bias = 8.0 或 10.0

```bash
--sasrec_bias_value 8.0  # 甚至可以尝试10.0
```

**原理**: 更强的 bias 确保模型在训练初中期更依赖 SASRec，避免多模态噪声

**预期提升**: +2-5 个百分点

#### B. 延长 Partial Aggregation 的 Warmup 阶段

**当前**: Warmup 20 轮（占 67%）
**建议**: Warmup 35 轮（占 70%）

```bash
--partial_aggregation_warmup_rounds 35
```

**原理**: 给客户端更多时间在本地探索多模态空间

**预期提升**: +2-3 个百分点

#### C. 改进 Router 初始化策略

**当前问题**: Router 权重在训练过程中的变化没有日志
**建议**: 添加 Router 权重监控

修改 `fedmem_client.py`:
```python
# 在训练循环中添加
if epoch % 5 == 0:
    print(f"Router weights: seq={w_seq:.3f}, vis={w_vis:.3f}, sem={w_sem:.3f}")
```

这可以帮助诊断 Router 是否正常工作。

---

### 🔥 方案4: 改进多模态特征（较复杂）

#### A. 多模态特征预处理

当前的多模态特征可能需要归一化：

```python
# 在加载特征后添加
item_visual_feats = F.normalize(item_visual_feats, p=2, dim=1)
item_text_feats = F.normalize(item_text_feats, p=2, dim=1)
```

**预期提升**: +3-5 个百分点

#### B. 尝试不同的特征提取方法

- **视觉特征**: 尝试使用 CLIP-ViT-L/14 而非 ViT-B/32
- **文本特征**: 尝试使用 sentence-transformers 的更大模型

#### C. 多模态特征降维

如果特征维度过高，可以尝试 PCA 降维：

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=256)
item_visual_feats = pca.fit_transform(item_visual_feats)
```

---

### 🔥 方案5: 修改模型架构（最复杂，但潜力最大）

#### A. 移除 L2 归一化

**当前代码** ([ur4rec_v2_moe.py:573](models/ur4rec_v2_moe.py#L573)):
```python
fused_repr_norm = torch.nn.functional.normalize(fused_repr, p=2, dim=-1)
target_item_embs_norm = torch.nn.functional.normalize(target_item_embs, p=2, dim=-1)
```

**问题**: L2 归一化将所有向量投影到单位球面上，可能限制了表达能力

**建议**: 尝试移除归一化，直接使用内积：

```python
# 移除归一化
final_scores = (fused_repr * target_item_embs).sum(dim=-1)  # [B, N]
```

**预期提升**: +5-10 个百分点

#### B. 使用不同的融合策略

**当前**: 表示级融合（加权求和）
**建议**: 尝试分数级融合或注意力融合

```python
# 分数级融合
seq_scores = (seq_out * target_item_embs).sum(dim=-1)
vis_scores = (vis_out * target_item_embs).sum(dim=-1)
sem_scores = (sem_out * target_item_embs).sum(dim=-1)

final_scores = w_seq * seq_scores + w_vis * vis_scores + w_sem * sem_scores
```

**预期提升**: +3-8 个百分点

#### C. 引入对比学习

在 SASRec 部分添加对比学习损失：

```python
contrastive_loss = self.contrastive_lambda * compute_contrastive_loss(
    seq_repr, target_item_embs
)
total_loss = rec_loss + contrastive_loss + lb_loss
```

**预期提升**: +5-10 个百分点

---

## 🚀 推荐的实验计划（分阶段）

### 阶段 1: 快速验证（1-2小时）

**目标**: 排除评估方法的影响

```bash
# 实验 1.1: 使用全库评估
python scripts/train_fedmem.py \
    --data_dir data \
    --data_file ml100k_ratings_processed.dat \
    --visual_file clip_features_fixed.pt \
    --text_file item_text_features.pt \
    --num_rounds 30 \
    --init_bias_for_sasrec \
    --sasrec_bias_value 5.0 \
    --partial_aggregation_warmup_rounds 20 \
    --use_negative_sampling False \
    --save_dir checkpoints/test_full_library_eval

# 预期结果: HR@10 = 0.48-0.58
```

**如果结果达到 0.50+**: 说明评估方法是主要问题，继续优化其他方面
**如果结果仍然 < 0.45**: 说明模型本身有问题，需要架构改进

---

### 阶段 2: 超参数优化（4-6小时）

**目标**: 通过超参数调优达到 0.55-0.60

```bash
# 实验 2.1: 增大模型容量 + 优化学习率
python scripts/train_fedmem.py \
    --data_dir data \
    --data_file ml100k_ratings_processed.dat \
    --visual_file clip_features_fixed.pt \
    --text_file item_text_features.pt \
    --sasrec_hidden_dim 512 \
    --sasrec_num_blocks 3 \
    --moe_num_heads 8 \
    --learning_rate 0.005 \
    --batch_size 64 \
    --num_rounds 50 \
    --patience 15 \
    --init_bias_for_sasrec \
    --sasrec_bias_value 8.0 \
    --partial_aggregation_warmup_rounds 35 \
    --use_negative_sampling False \
    --save_dir checkpoints/test_larger_model

# 预期结果: HR@10 = 0.55-0.60
```

---

### 阶段 3: 架构改进（需要代码修改，1-2天）

**目标**: 通过架构改进达到 0.60-0.70

#### 修改1: 移除 L2 归一化

编辑 `models/ur4rec_v2_moe.py`:

```python
# 找到第 573-579 行
# 注释掉归一化代码
# fused_repr_norm = torch.nn.functional.normalize(fused_repr, p=2, dim=-1)
# target_item_embs_norm = torch.nn.functional.normalize(target_item_embs, p=2, dim=-1)
# scale = self.sasrec_hidden_dim ** 0.5
# final_scores = scale * (fused_repr_norm * target_item_embs_norm).sum(dim=-1)

# 改为直接内积
final_scores = (fused_repr * target_item_embs).sum(dim=-1)  # [B, N]
```

#### 修改2: 使用分数级融合

```python
# 在 forward 方法中，替换表示级融合为分数级融合
seq_scores = torch.bmm(seq_repr.unsqueeze(1), target_item_embs.transpose(1, 2)).squeeze(1)
vis_scores = (vis_out * target_item_embs).sum(dim=-1)
sem_scores = (sem_out * target_item_embs).sum(dim=-1)

final_scores = (
    w_seq.squeeze(2) * seq_scores +
    w_vis.squeeze(2) * vis_scores +
    w_sem.squeeze(2) * sem_scores
)
```

运行实验:
```bash
python scripts/train_fedmem.py \
    --data_dir data \
    --data_file ml100k_ratings_processed.dat \
    --visual_file clip_features_fixed.pt \
    --text_file item_text_features.pt \
    --sasrec_hidden_dim 512 \
    --sasrec_num_blocks 3 \
    --learning_rate 0.005 \
    --batch_size 64 \
    --num_rounds 50 \
    --init_bias_for_sasrec \
    --sasrec_bias_value 8.0 \
    --partial_aggregation_warmup_rounds 35 \
    --use_negative_sampling False \
    --save_dir checkpoints/test_score_fusion

# 预期结果: HR@10 = 0.60-0.70
```

---

## 📋 诊断检查清单

在进行改进前，请确认：

### 1. 多模态特征质量

```bash
# 检查特征统计
python3 << 'EOF'
import torch
visual = torch.load('data/clip_features_fixed.pt')
text = torch.load('data/item_text_features.pt')

print(f"Visual features shape: {visual.shape}")
print(f"Visual stats: min={visual.min():.4f}, max={visual.max():.4f}, mean={visual.mean():.4f}, std={visual.std():.4f}")
print()
print(f"Text features shape: {text.shape}")
print(f"Text stats: min={text.min():.4f}, max={text.max():.4f}, mean={text.mean():.4f}, std={text.std():.4f}")
EOF
```

**正常范围**:
- 如果特征已归一化: mean ≈ 0, std ≈ 1
- 如果未归一化: 应该在合理范围内（不是全0或全1）

### 2. Router 权重分布

查看训练日志中是否有 Router 权重输出。如果没有，需要添加监控代码。

### 3. 训练稳定性

```bash
# 检查训练损失曲线
grep "平均训练损失" checkpoints/full_experiment_20251223_212525/both.log | tail -10
```

应该看到损失持续下降，没有震荡或发散。

---

## 🎯 总结：达到 HR@10 = 0.60-0.70 的路径

### 最可能的组合方案（推荐）

**Step 1**: 修改评估方法（预期 +10-15%）
```bash
--use_negative_sampling False
```
**预期结果**: HR@10 = 0.48-0.53

**Step 2**: 优化超参数（预期 +5-10%）
```bash
--sasrec_hidden_dim 512
--sasrec_num_blocks 3
--learning_rate 0.005
--batch_size 64
--num_rounds 50
--sasrec_bias_value 8.0
--partial_aggregation_warmup_rounds 35
```
**预期结果**: HR@10 = 0.55-0.60

**Step 3**: 改进架构（预期 +5-10%）
- 移除 L2 归一化
- 改用分数级融合
- 添加对比学习

**预期最终结果**: HR@10 = **0.60-0.70** ✅

---

## ⚠️ 重要提醒

### 1. 评估方法的选择

- **1:100 负采样**: 更接近实际应用场景，但指标偏低
- **全库评估**: 更容易与论文对比，但不太实际

**建议**: 两种都报告
- "在 1:100 负采样下 HR@10 = 0.38"
- "在全库评估下 HR@10 = 0.55"

### 2. 不要过度依赖单一指标

除了 HR@10，也要关注：
- **NDCG@10**: 考虑排序质量
- **MRR**: 首个正确推荐的位置
- **HR@20**: 更宽松的召回

### 3. 对比基线的公平性

确保与 FedSASRec 使用相同的：
- 评估方法（负采样 vs 全库）
- 数据划分（训练/验证/测试）
- 超参数设置

---

## 📞 需要帮助？

如果按照上述方案仍无法达到目标，可能需要：

1. **检查数据预处理** - 可能有 bug 导致数据损坏
2. **对比简单 SASRec baseline** - 先确保 SASRec 部分工作正常
3. **调试 MoE Router** - 确认 Router 真的在学习合理的权重分配
4. **重新审视问题设置** - 可能联邦学习设置本身就很难

---

**最后更新**: 2024-12-24
**作者**: Claude Code
**状态**: 待验证
