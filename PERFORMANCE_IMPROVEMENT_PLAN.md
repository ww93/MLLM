# UR4Rec Performance Improvement Plan
## 将HR@10从0.4提升至0.7-0.8

---

## 问题分析

### 当前状态
- **当前指标**: HR@10 ≈ 0.4 (40%)
- **目标指标**: HR@10 ≈ 0.7-0.8 (70-80%)
- **差距**: 需要提升30-40个百分点（75-100%的相对提升）
- **观察**: 使用CLIP特征后性能提升不明显

### 核心问题诊断

#### 1. **多模态融合不充分** ⚠️
```python
# 当前问题：简单加权融合
final_scores = sasrec_weight * sasrec_scores + retriever_weight * retriever_scores
```

**为什么不理想**：
- SASRec和Retriever可能学到不同的信息
- 简单线性加权无法capture复杂的交互模式
- CLIP特征在MoE中的贡献可能被淹没

**数据支持**：
- CLIP加入前后指标差异不大 → 说明CLIP信号没有被有效利用
- retriever_weight=0.6 但可能实际贡献更小

#### 2. **负样本采样不足** ⚠️⚠️
```yaml
num_negatives: 20  # 从1682个候选中只sample 20个
```

**为什么这是瓶颈**：
- MovieLens-100K有1682个物品
- 只采样20个负样本 → 只覆盖1.2%的物品空间
- 模型never sees 98%的hard negatives
- 导致ranking能力弱

**理论支持**：
- InfoNCE loss需要足够的负样本才能学到好的表示
- 推荐领域最佳实践：negative samples ≥ 100-500

#### 3. **序列建模能力不足** ⚠️
```yaml
sasrec_num_blocks: 3  # 只有3层Transformer
sasrec_num_heads: 8
max_seq_len: 50
```

**为什么限制性能**：
- MovieLens用户行为序列较长
- 3层Transformer难以捕获长期依赖
- 注意力机制可能under-parameterized

#### 4. **CLIP特征未被充分利用** ⚠️⚠️⚠️
```python
# 当前：CLIP特征只是替换了trainable embeddings
item_embed_vectors = self.clip_features[item_ids]  # 被动使用

# 问题：
# - CLIP特征是静态的，没有fine-tuning
# - 没有visual-text alignment
# - MoE权重可能忽视视觉信号
```

**数据支持**：
- CLIP图片特征已成功提取（1681/1682物品）
- 但性能提升不明显 → **特征利用方式有问题**

---

## 改进方案

### 🚀 Phase 1: 短期改进 (预期+10-15%)

#### 1.1 增强负样本采样
```python
# From: num_negatives: 20
# To:   num_negatives: 200-500
```

**实施**：
```yaml
# configs/ur4rec_moe_100k.yaml
num_negatives: 500  # 增加25倍

# 同时增加batch training negatives
batch_size: 32  # 增大batch以容纳更多negatives
use_in_batch_negatives: true  # 使用batch内的其他样本作为负样本
```

**预期提升**: +5-10%
**成本**: 训练时间增加2-3倍

---

#### 1.2 改进多模态融合机制

**当前问题**：
```python
# 简单加权融合 - 不optimal
final_scores = 0.4 * sasrec_scores + 0.6 * retriever_scores
```

**改进方案A: Gating Fusion**
```python
class AdaptiveGatingFusion(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # 学习每个用户/物品的动态权重
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # [sasrec_weight, retriever_weight]
            nn.Softmax(dim=-1)
        )

    def forward(self, sasrec_repr, retriever_repr, sasrec_scores, retriever_scores):
        # 基于表示学习gate weights
        combined = torch.cat([sasrec_repr, retriever_repr], dim=-1)
        weights = self.gate(combined)  # [B, 2]

        # 动态融合
        final_scores = (
            weights[:, 0:1] * sasrec_scores +
            weights[:, 1:2] * retriever_scores
        )
        return final_scores, weights
```

**改进方案B: Cross-Attention Fusion**
```python
class CrossAttentionFusion(nn.Module):
    def __init__(self, hidden_dim, num_heads=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads)
        self.fusion_proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, sasrec_repr, retriever_repr):
        # SASRec attends to Retriever
        sasrec_enhanced, _ = self.cross_attn(
            query=sasrec_repr,
            key=retriever_repr,
            value=retriever_repr
        )

        # Retriever attends to SASRec
        retriever_enhanced, _ = self.cross_attn(
            query=retriever_repr,
            key=sasrec_repr,
            value=sasrec_repr
        )

        # Combine
        fused = self.fusion_proj(torch.cat([sasrec_enhanced, retriever_enhanced], dim=-1))
        return fused
```

**预期提升**: +3-5%
**成本**: 增加少量参数和计算

---

#### 1.3 Fine-tune CLIP特征投影层

**当前问题**：
```python
# CLIP特征是frozen的，无法adapt to recommendation task
self.clip_features = torch.load(clip_features_path)  # Static
item_embed = self.clip_features[item_id]  # 直接使用
```

**改进方案**：
```python
class AdaptiveCLIPProjection(nn.Module):
    def __init__(self, clip_dim=512, output_dim=512):
        super().__init__()
        # 可训练的投影层
        self.projection = nn.Sequential(
            nn.Linear(clip_dim, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, clip_features):
        # 将CLIP特征adapt到推荐任务
        return self.projection(clip_features)

# 在训练中使用
self.clip_projection = AdaptiveCLIPProjection().to(device)
item_embed = self.clip_projection(self.clip_features[item_id])
```

**Key Insight**: 让模型学习哪些视觉特征对推荐有用

**预期提升**: +2-5%
**成本**: 增加少量参数，训练时间几乎不变

---

### 🔥 Phase 2: 中期改进 (预期+15-20%)

#### 2.1 实施Hierarchical MoE架构

**当前问题**：
- Flat MoE: 3个expert (user_pref, item_desc, image)
- 每个模态只有一个expert → 表达能力有限

**改进方案**: 参考已有的hierarchical_moe_guide.md

```
Level 1: Within-Modality MoE (9 experts total)
├─ User Preference MoE (3 sub-experts)
│   ├─ Genre Expert (类型偏好)
│   ├─ Mood Expert (情绪偏好)
│   └─ Style Expert (风格偏好)
├─ Item Description MoE (3 sub-experts)
│   ├─ Content Expert (内容理解)
│   ├─ Theme Expert (主题分析)
│   └─ Quality Expert (质量评估)
└─ CLIP Image MoE (3 sub-experts)
    ├─ Composition Expert (视觉构图)
    ├─ Color/Texture Expert (颜色纹理)
    └─ Object Expert (物体识别)

Level 2: Cross-Modal Fusion
└─ Learn to combine 3 modality outputs dynamically
```

**为什么这能提升性能**：
1. **更细粒度的特征**: 9个expert vs 3个 → 3x表达能力
2. **专业化**: 每个sub-expert focus on specific aspect
3. **Better CLIP utilization**: 3个视觉expert能学到不同的视觉模式

**实施**：
```python
# models/hierarchical_moe.py已实现
# 需要创建对应的config并训练
```

**预期提升**: +8-12%
**成本**: 训练时间增加50%，参数量增加2x

---

#### 2.2 增强序列建模能力

```yaml
# 当前
sasrec_num_blocks: 3
sasrec_hidden_dim: 512

# 改进
sasrec_num_blocks: 6  # 增加到6层
sasrec_hidden_dim: 768  # 增大隐藏层
sasrec_num_heads: 12  # 增加注意力头
```

**额外增强**：
```python
# 添加Position-wise Feed-Forward增强
class EnhancedSASRec(nn.Module):
    def __init__(self, ...):
        super().__init__()
        self.blocks = nn.ModuleList([
            EnhancedTransformerBlock(  # 替换原来的simple block
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                use_glu=True,  # Gated Linear Unit
                use_relative_position=True  # Relative position encoding
            )
            for _ in range(num_blocks)
        ])
```

**预期提升**: +3-5%
**成本**: 训练时间增加30%

---

#### 2.3 改进训练策略

**A. 对比学习增强**
```python
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, user_repr, pos_item_repr, neg_items_repr):
        # InfoNCE loss with more negatives
        pos_score = torch.sum(user_repr * pos_item_repr, dim=-1) / self.temperature
        neg_scores = torch.matmul(user_repr, neg_items_repr.T) / self.temperature

        # Large-scale contrastive loss
        logits = torch.cat([pos_score.unsqueeze(-1), neg_scores], dim=-1)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

        return F.cross_entropy(logits, labels)
```

**B. 多任务学习**
```python
# 同时优化多个目标
total_loss = (
    1.0 * ranking_loss +           # 主任务：ranking
    0.3 * contrastive_loss +        # 对比学习
    0.2 * visual_text_alignment_loss  # CLIP对齐
)
```

**预期提升**: +4-8%
**成本**: 训练复杂度增加

---

### ⚡ Phase 3: 高级改进 (预期+5-10%)

#### 3.1 CLIP-Text Alignment

**核心思想**: 让CLIP视觉特征和文本描述对齐

```python
class CLIPTextAlignmentLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, clip_features, text_features):
        # Normalize
        clip_norm = F.normalize(clip_features, dim=-1)
        text_norm = F.normalize(text_features, dim=-1)

        # Cosine similarity
        sim_matrix = torch.matmul(clip_norm, text_norm.T) / self.temperature

        # Contrastive loss (same item's image and text should match)
        labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
        loss = (
            F.cross_entropy(sim_matrix, labels) +
            F.cross_entropy(sim_matrix.T, labels)
        ) / 2

        return loss
```

**效果**: 确保视觉和文本特征互补而non-redundant

**预期提升**: +2-4%

---

#### 3.2 冷启动物品增强

**问题**: 测试集中有23个物品(1682-1659)没有在训练集中出现

**方案**：
```python
# 使用CLIP和text features进行zero-shot推荐
def handle_cold_start_items(self, item_ids):
    cold_items = [i for i in item_ids if i not in self.training_items]

    if cold_items:
        # 使用CLIP+text特征做zero-shot
        clip_repr = self.clip_features[cold_items]
        text_repr = self.item_description_embeddings[cold_items]

        # Weighted combination
        cold_item_repr = 0.7 * clip_repr + 0.3 * text_repr
        return cold_item_repr
    else:
        return None
```

**预期提升**: +1-3%

---

#### 3.3 用户行为序列增强

**Data Augmentation for sequences**:
```python
def augment_sequence(seq):
    # 1. Item masking (like BERT)
    masked_seq = mask_random_items(seq, mask_ratio=0.15)

    # 2. Item reordering (shuffle subsequences)
    reordered_seq = reorder_subsequence(seq, shuffle_ratio=0.1)

    # 3. Item substitution (replace with similar items)
    substituted_seq = substitute_similar_items(seq, sub_ratio=0.1)

    return [seq, masked_seq, reordered_seq, substituted_seq]
```

**预期提升**: +2-3%

---

## 实施优先级和预期效果

### Priority 1: Quick Wins (1-2 days)
1. ✅ **增加负样本数量** (num_negatives: 20 → 500)
   - 预期: +5-10%
   - 工作量: 修改配置文件，重新训练

2. ✅ **Fine-tune CLIP投影层**
   - 预期: +2-5%
   - 工作量: 添加可训练投影层

3. ✅ **改进融合机制** (实施Gating Fusion)
   - 预期: +3-5%
   - 工作量: 实现新的融合模块

**小计**: +10-20% → HR@10: 0.4 → 0.48-0.52

---

### Priority 2: Medium-term (3-5 days)
4. ✅ **实施Hierarchical MoE**
   - 预期: +8-12%
   - 工作量: 使用已有代码，创建配置，训练

5. ✅ **增强SASRec** (6层，更多heads)
   - 预期: +3-5%
   - 工作量: 修改模型配置

6. ✅ **对比学习增强**
   - 预期: +4-8%
   - 工作量: 实现对比loss

**小计**: +15-25% → HR@10: 0.48-0.52 → 0.63-0.77

---

### Priority 3: Advanced (optional, 2-3 days)
7. ✅ **CLIP-Text Alignment**
   - 预期: +2-4%
   - 工作量: 实现alignment loss

8. ✅ **序列增强**
   - 预期: +2-3%
   - 工作量: 实现数据增强

**小计**: +4-7% → HR@10: 0.63-0.77 → 0.67-0.84

---

## 综合预期

| 阶段 | 实施方案 | 预期HR@10 | 累计提升 |
|-----|---------|----------|---------|
| **Baseline** | 当前状态 | 0.40 | - |
| **Phase 1** | Quick Wins | 0.48-0.52 | +20-30% |
| **Phase 2** | Medium-term | 0.63-0.77 | +58-93% |
| **Phase 3** | Advanced | 0.67-0.84 | +68-110% |

**最终预期**: **HR@10 = 0.67-0.84** ✅达成目标0.7-0.8

---

## 立即行动项

### 第一步：修复IndexError ✅ (已完成)
- [x] 使用max(item_ids)而不是len(item_map)

### 第二步：Quick Wins实施 (今日完成)
1. [配置修改] 增加负样本到500
2. [代码添加] 实现AdaptiveCLIPProjection
3. [代码添加] 实现AdaptiveGatingFusion
4. [训练] 使用新配置重新训练

### 第三步：Medium-term实施 (2-3天)
1. [配置修改] 启用Hierarchical MoE
2. [配置修改] 增强SASRec (6 blocks, 768 dim)
3. [代码添加] 实现对比学习loss
4. [训练] 完整训练新架构

---

## 总结

**核心瓶颈**：
1. ❌ 负样本不足 (只有20个)
2. ❌ 简单融合机制
3. ❌ CLIP特征未充分利用
4. ❌ 序列建模能力弱

**解决方案**：
1. ✅ 500个负样本 + in-batch negatives
2. ✅ Gating/Cross-Attention融合
3. ✅ Fine-tunable CLIP投影 + Hierarchical MoE
4. ✅ 更深的SASRec + 对比学习

**预期结果**：
- **Phase 1**: HR@10 = 0.48-0.52 (+20-30%)
- **Phase 2**: HR@10 = 0.63-0.77 (+58-93%)
- **Phase 3**: HR@10 = 0.67-0.84 (+68-110%) ✅ **达成目标**

---

## 下一步

我将依次实施以上改进。首先从Quick Wins开始，因为它们能快速带来显著提升。

是否现在开始实施Phase 1的改进？
