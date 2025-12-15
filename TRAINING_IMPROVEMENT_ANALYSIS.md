# Training Performance Analysis & Improvements

## 当前问题分析

### 观察到的症状
```
Validation metrics (Epoch 7):
  hit@5: 0.0544     (5.4% - 非常低)
  ndcg@10: 0.0465   (4.6% - 非常低)
  mrr: 0.0542       (平均排名 ~18)
  avg_rank: 49.09   (在100个候选中排名49)

Training metrics:
  SASRec loss: 1.20 → 0.37 (下降70%)
  Retrieval loss: ~22 (几乎不变)

Other observations:
  - patience: 5/5 (即将early stop)
  - train=SAS, switch=0 (一直训练SASRec)
  - avg_interactions: 0.0 (Memory未使用)
```

### 根本原因

#### 1. **过拟合（Overfitting）**
**证据**:
- SASRec loss大幅下降（1.2→0.37），但validation metrics不提升
- 训练集很小（938个用户，1659个items）
- Dropout=0.0，模型容易记住训练数据

**原因**:
- 数据集太小（ML-100K）
- 模型参数太多（27M参数 vs 938个用户）
- 没有正则化（dropout=0）

#### 2. **Retrieval模块未学习**
**证据**:
- Retrieval loss维持在~22，完全不变
- 一直显示`train=SAS`，从未切换到`train=RET`
- `adaptive_switch_count=0`

**原因**:
- Adaptive alternating的threshold（0.01）太高，永远不会切换
- SASRec loss从1.2降到0.37，变化远大于0.01，所以一直训练SASRec
- Retrieval模块完全没有学习机会

#### 3. **学习率太保守**
**证据**:
- 当前lr=0.0001
- 在小数据集上，需要更激进的学习率才能快速收敛

**原因**:
- 默认学习率适合大数据集
- 小数据集需要更大的学习率（0.001-0.01）

#### 4. **Memory机制未启用**
**证据**:
- `avg_interactions: 0.0`
- `avg_memory_history_size: 0.0`

**原因**:
- `interaction_threshold=20`太高
- 每个epoch只有30个batch，很难达到20次交互

#### 5. **训练策略问题**
**证据**:
- Batch size=32，对于938个用户来说太大（每个epoch只有30个batch）
- 负样本数=5，对于1659个items来说太少

---

## 改进方案

### 方案A: 保守改进（推荐用于测试）

**改进的配置**：`ur4rec_moe_100k_improved.yaml`

#### 关键改动：

1. **降低模型复杂度，增加正则化**
```yaml
sasrec_hidden_dim: 128       # 256→128 (减少50%参数)
retriever_output_dim: 128    # 256→128 (匹配SASRec)
sasrec_dropout: 0.2          # 0.0→0.2 (防止过拟合)
moe_dropout: 0.2             # 0.1→0.2
moe_num_heads: 4             # 8→4 (减少参数)
```

2. **提高学习率**
```yaml
sasrec_lr: 0.001             # 0.0001→0.001 (提高10倍)
retriever_lr: 0.0005         # 0.0001→0.0005 (提高5倍)
```

3. **改进训练策略**
```yaml
batch_size: 16               # 32→16 (每个epoch 60个batch，更新更频繁)
num_negatives: 10            # 5→10 (增加训练难度)
use_adaptive_alternating: false  # 禁用自适应，避免一直训练SASRec
use_uncertainty_weighting: false # 使用固定权重
```

4. **调整融合权重**
```yaml
sasrec_weight: 0.7           # 0.5→0.7 (SASRec学得更好，给它更多权重)
retriever_weight: 0.3        # 0.5→0.3
```

5. **简化训练阶段**
```yaml
stages:
  - pretrain_sasrec          # 先专注训练SASRec
  - joint_finetune           # 然后联合微调
# 移除 pretrain_retriever 和 end_to_end
```

6. **降低Memory threshold**
```yaml
interaction_threshold: 5     # 20→5 (更早触发memory更新)
```

---

### 方案B: 激进改进（如果保守方案效果不好）

```yaml
# 更激进的配置
sasrec_lr: 0.005             # 更高学习率
retriever_lr: 0.002
sasrec_dropout: 0.3          # 更强正则化
batch_size: 8                # 更小batch
num_negatives: 20            # 更多负样本
sasrec_hidden_dim: 64        # 更小模型
```

---

## 预期改进

### 使用改进配置后，预期：

1. **Metrics提升**
   - hit@5: 0.05 → 0.10-0.15 (翻倍)
   - ndcg@10: 0.045 → 0.08-0.12
   - mrr: 0.05 → 0.10-0.15

2. **训练行为**
   - 每个epoch有60个batch（vs 30）
   - 更新更频繁，收敛更快
   - Retrieval模块也会训练

3. **泛化能力**
   - Dropout防止过拟合
   - 训练loss和validation metrics同步提升

---

## 运行命令

### 测试改进配置（保守方案）

```bash
# 清理旧输出
rm -rf outputs/ur4rec_moe_improved/*

# 运行改进配置（短时间测试）
python UR4Rec/scripts/train_ur4rec_moe.py \
    --config UR4Rec/configs/ur4rec_moe_100k_improved.yaml \
    --data_dir UR4Rec/data/Multimodal_Datasets \
    --llm_data_dir data/llm_generated \
    --output_dir outputs/ur4rec_moe_improved \
    --epochs_per_stage 20 \
    --patience 10
```

### 如果效果好，运行完整训练

```bash
python UR4Rec/scripts/train_ur4rec_moe.py \
    --config UR4Rec/configs/ur4rec_moe_100k_improved.yaml \
    --data_dir UR4Rec/data/Multimodal_Datasets \
    --llm_data_dir data/llm_generated \
    --output_dir outputs/ur4rec_moe_final \
    --epochs_per_stage 50 \
    --patience 10
```

---

## 监控指标

### 训练时应该看到：

1. **Loss变化**
   ```
   Epoch 0:
     total_loss: 15-20 (不是23+)
     sasrec: 0.8-1.5 (正常)
     retrieval: 14-19 (应该下降!)

   Epoch 10:
     total_loss: 10-15
     sasrec: 0.3-0.6
     retrieval: 9-14 (明显下降)
   ```

2. **Metrics提升**
   ```
   Epoch 1: hit@5=0.06
   Epoch 5: hit@5=0.09
   Epoch 10: hit@5=0.12
   ```

3. **训练行为**
   ```
   - 60 batches/epoch (vs 30)
   - joint_finetune阶段会切换 train=SAS/RET
   - memory会启用（avg_interactions > 0）
   ```

---

## Troubleshooting

### 如果改进配置仍然效果不好：

#### 1. Metrics仍然很低（<0.08）
**可能原因**：数据集本身太小/太稀疏

**解决**：
```yaml
# 进一步降低模型复杂度
sasrec_hidden_dim: 64
sasrec_num_blocks: 1  # 只用1层transformer
batch_size: 8
```

#### 2. Loss不收敛（NaN或爆炸）
**可能原因**：学习率太高

**解决**：
```yaml
sasrec_lr: 0.0005  # 降低学习率
retriever_lr: 0.0002
```

#### 3. 过拟合更严重
**可能原因**：Dropout不够

**解决**：
```yaml
sasrec_dropout: 0.3  # 增加dropout
# 或者添加L2正则化
weight_decay: 0.01
```

#### 4. Retrieval loss仍然不变
**可能原因**：Retriever太复杂或LLM数据质量不好

**解决**：
- 检查LLM生成的数据质量
- 降低retriever复杂度
- 或者暂时只训练SASRec

---

## 技术细节

### 为什么降低hidden_dim可以提升性能？

对于小数据集：
- **参数过多** → 容易记住训练数据 → 泛化差
- **参数适中** → 学习general patterns → 泛化好

计算：
- 原配置：27M参数 / 938用户 = 28,782参数/用户（过剩！）
- 新配置：~7M参数 / 938用户 = 7,462参数/用户（仍然很多但更合理）

### 为什么提高学习率？

对于小数据集：
- **更新次数少**：每个epoch只有30-60个batch
- **需要快速收敛**：否则100 epochs也学不好
- **数据简单**：不需要特别小心的学习率

### 为什么增加negative samples？

- **训练信号更强**：10个负样本 vs 5个
- **更难的任务**：模型需要学习更精细的区分
- **更接近评估**：评估时有100个candidates

---

## 总结

### 核心改进
1. ✅ **降低模型复杂度** - 防止过拟合
2. ✅ **增加Dropout** - 提高泛化
3. ✅ **提高学习率** - 加快收敛
4. ✅ **减小Batch size** - 更频繁更新
5. ✅ **增加负样本** - 增加训练难度
6. ✅ **简化训练流程** - 专注于有效的阶段
7. ✅ **禁用自适应交替** - 确保Retriever也被训练

### 预期结果
- Metrics提升2-3倍（from 0.05 to 0.10-0.15）
- 训练更稳定
- 泛化能力更强

---

**创建时间**: 2025-12-10
**适用场景**: 小数据集（<1000用户，<2000 items）的推荐系统
