# FedDMMR 预训练加速指南

## 🚀 为什么需要预训练？

FedDMMR包含多个复杂组件：
- **SASRec骨干** - 序列建模（慢）
- **视觉/语义专家** - 多模态融合（慢）
- **本地记忆** - 动态更新（慢）
- **对比学习** - 漂移自适应（慢）

**问题**: 从零开始训练需要50-100轮，耗时4-8小时（CPU）

**解决方案**: 先预训练SASRec骨干（15轮，30-60分钟），再迁移学习FedDMMR（10-15轮，1-2小时）

**总耗时**: 预训练(1小时) + 微调(1-2小时) = **2-3小时** ✅

---

## 📋 两阶段训练流程

### 阶段1: 预训练FedSASRec（快速骨干初始化）

**特点**:
- ✅ 仅使用ID嵌入（无多模态特征）
- ✅ 禁用对比学习（contrastive_lambda=0）
- ✅ 无辅助专家参与（MoE待机）
- ✅ 快速收敛配置（15轮，大学习率，大batch）

**执行命令**:
```bash
cd UR4Rec
python scripts/pretrain_fedsasrec.py
```

**预期输出**:
```
✓ 预训练完成！
预训练权重已保存到: checkpoints/fedsasrec_pretrain/
  - 模型: fedmem_model.pt
  - 训练历史: train_history.json
```

**预期时长**:
- CPU: 30-60分钟
- GPU: 10-20分钟

**预期性能**: HR@10 ≈ 0.55-0.58（纯SASRec基线）

---

### 阶段2: 微调FedDMMR（完整多模态训练）

**特点**:
- ✅ 加载预训练的SASRec骨干
- ✅ 启用多模态特征（CLIP + 文本）
- ✅ 启用漂移自适应对比学习
- ✅ 启用本地动态记忆
- ✅ 跳过Warmup，直接全量聚合

**方法1: 自动加载（推荐）**

编辑 `scripts/train_ml1m_drift_adaptive.py`，取消注释预训练路径：

```python
config = {
    # ... 其他配置 ...

    # [NEW] 预训练权重路径（取消注释以启用）
    "pretrained_path": "checkpoints/fedsasrec_pretrain/fedmem_model.pt",  # ✅ 取消注释

    # ... 其他配置 ...
}
```

然后运行：
```bash
cd UR4Rec
python scripts/train_ml1m_drift_adaptive.py
```

**方法2: 手动指定路径**

```bash
cd UR4Rec
python scripts/train_fedmem.py \
  --data_dir data/ml-1m \
  --data_file subset_ratings.dat \
  --visual_file clip_features.pt \
  --text_file text_features.pt \
  --pretrained_path checkpoints/fedsasrec_pretrain/fedmem_model.pt \
  --learning_rate 1e-4 \
  --num_rounds 15 \
  --partial_aggregation_warmup_rounds 0
```

**预期输出**:
```
[2.5/4] 加载预训练权重...
  路径: checkpoints/fedsasrec_pretrain/fedmem_model.pt
  ✓ 成功加载 45 个参数
    主要模块: SASRec骨干、Item嵌入、Router

  📌 预训练权重已加载，建议:
    - 使用较小的学习率（如1e-4）
    - 减少训练轮数（当前15轮）
    - 直接跳过Warmup（设置partial_aggregation_warmup_rounds=0）
```

**预期时长**:
- CPU: 1-2小时（15轮）
- GPU: 20-40分钟

**预期性能**: HR@10 ≈ 0.60-0.65（完整FedDMMR + 所有修复）

---

## 🔍 权重加载详情

### 加载策略

预训练权重加载是**选择性的**和**兼容性的**：

```python
# 优先加载SASRec骨干参数
if 'sasrec' in key or 'item_emb' in key:
    current_state[key] = pretrained_state[key]

# 可选加载Router和LayerNorm（如果形状匹配）
elif 'router' in key or 'layernorm' in key:
    if shape_matches:
        current_state[key] = pretrained_state[key]

# 跳过辅助专家参数（视觉/语义专家从头训练）
```

### 加载的参数

| 模块 | 是否加载 | 原因 |
|------|---------|------|
| SASRec骨干 | ✅ 加载 | 序列建模能力可迁移 |
| Item嵌入 | ✅ 加载 | ID表示可迁移 |
| Router | ✅ 加载 | 路由权重可作为初始化 |
| LayerNorm | ✅ 加载 | 归一化参数可迁移 |
| 视觉专家 | ❌ 跳过 | 预训练时未使用多模态 |
| 语义专家 | ❌ 跳过 | 预训练时未使用多模态 |

### 为什么这样设计？

1. **SASRec是核心**: 序列建模能力最重要，训练最慢，必须预训练
2. **专家快速适应**: 视觉/语义专家参数较少，可以在微调阶段快速学习
3. **避免冲突**: 预训练时未见过多模态特征，强制加载可能导致负迁移

---

## ⚡ 性能对比

### 训练时长

| 训练方式 | 预训练阶段 | 微调阶段 | 总时长 (CPU) |
|---------|-----------|---------|-------------|
| 从零训练 | - | 50轮 | 4-8小时 |
| 预训练+微调 | 15轮 | 15轮 | **2-3小时** ✅ |

**加速比**: ~2-3x

### 模型性能

| 训练方式 | HR@10 (预期) | NDCG@10 (预期) |
|---------|-------------|---------------|
| 从零训练 (50轮) | 0.58-0.62 | 0.38-0.42 |
| 预训练+微调 (15+15轮) | 0.60-0.65 | 0.39-0.43 |

**性能差异**: 接近或略优（预训练提供更好的初始化）

---

## 📝 配置建议

### 预训练阶段 (pretrain_fedsasrec.py)

```python
config = {
    "num_rounds": 15,           # 15轮足够
    "client_fraction": 0.2,     # 每轮20%客户端（加速）
    "learning_rate": 5e-3,      # 5倍学习率（快速收敛）
    "batch_size": 64,           # 2倍batch（加速）
    "contrastive_lambda": 0.0,  # ❗关键：禁用对比学习
    # 不加载visual_file和text_file（仅ID嵌入）
}
```

### 微调阶段 (train_ml1m_drift_adaptive.py)

```python
config = {
    "num_rounds": 15,                              # 减少到15轮
    "client_fraction": 0.1,                        # 恢复到10%
    "learning_rate": 1e-4,                         # ❗降低学习率（微调）
    "batch_size": 32,                              # 恢复到32
    "contrastive_lambda": 0.1,                     # 启用对比学习
    "visual_file": "clip_features.pt",             # 加载视觉特征
    "text_file": "text_features.pt",               # 加载文本特征
    "pretrained_path": "checkpoints/fedsasrec_pretrain/fedmem_model.pt",  # ❗加载预训练
    "partial_aggregation_warmup_rounds": 0,        # ❗跳过Warmup
}
```

**关键差异**:
- 预训练用**大学习率**（5e-3）快速收敛
- 微调用**小学习率**（1e-4）精细调整
- 微调**跳过Warmup**（预训练已完成骨干训练）

---

## ❓ 常见问题

### Q1: 预训练权重文件不存在怎么办？

**A**: 训练会自动回退到从零开始：

```
✗ 预训练权重文件不存在: checkpoints/fedsasrec_pretrain/fedmem_model.pt
将使用随机初始化继续训练
```

解决方法：先运行预训练脚本
```bash
python scripts/pretrain_fedsasrec.py
```

### Q2: 可以只运行阶段2（微调）吗？

**A**: 可以，但不推荐。如果跳过预训练：
- 需要更多轮数（40-50轮）
- 训练时间更长（4-6小时）
- 可能收敛到次优解

### Q3: 预训练和微调使用的数据一样吗？

**A**: 是的，都是ML-1M子集（1000用户，3646物品）。差异在于：
- **预训练**: 仅ID嵌入，无多模态特征
- **微调**: 完整多模态（CLIP + 文本）

### Q4: 可以用其他数据集预训练吗？

**A**: 可以，但需要注意：
- **Item数量必须匹配**（或使用更大的预训练集）
- **隐藏维度必须匹配**（sasrec_hidden_dim）
- **预训练数据应该相似**（推荐领域、序列分布）

### Q5: 预训练权重会被微调更新吗？

**A**: 会的。预训练只提供初始化，微调会继续更新所有参数。

---

## 🎯 快速开始（3步走）

### Step 1: 预训练SASRec骨干

```bash
cd UR4Rec
python scripts/pretrain_fedsasrec.py
```

⏱️ 等待30-60分钟...

### Step 2: 提取多模态特征（如果未提取）

```bash
cd UR4Rec
python scripts/extract_ml1m_real_features.py
```

⏱️ 等待5分钟...

### Step 3: 微调FedDMMR

编辑 `scripts/train_ml1m_drift_adaptive.py`，取消注释：
```python
"pretrained_path": "checkpoints/fedsasrec_pretrain/fedmem_model.pt",
```

运行：
```bash
python scripts/train_ml1m_drift_adaptive.py
```

⏱️ 等待1-2小时...

🎉 **完成！总耗时2-3小时，性能HR@10≈0.60-0.65**

---

## 📊 验证预训练效果

### 检查预训练模型

```bash
python -c "
import torch
state = torch.load('checkpoints/fedsasrec_pretrain/fedmem_model.pt')
print(f'参数数量: {len(state)}')
print(f'包含SASRec: {any(\"sasrec\" in k for k in state.keys())}')
"
```

预期输出：
```
参数数量: 50-60
包含SASRec: True
```

### 检查微调是否加载预训练

查看训练日志，应该看到：
```
[2.5/4] 加载预训练权重...
  ✓ 成功加载 45 个参数
```

如果看不到这条信息，说明未加载预训练权重。

---

## 🔧 故障排除

### 问题1: 预训练收敛很慢

**可能原因**: 学习率太小或batch太小

**解决方法**:
```python
"learning_rate": 5e-3,  # 确保是5e-3而非1e-3
"batch_size": 64,       # 确保是64而非32
```

### 问题2: 微调损失不下降

**可能原因**: 学习率太大

**解决方法**:
```python
"learning_rate": 1e-4,  # 微调必须用小学习率
```

### 问题3: 加载权重后性能下降

**可能原因**: 预训练数据与微调数据差异太大

**解决方法**:
- 使用相同数据集预训练和微调
- 或降低学习率，延长微调轮数

---

## 📚 技术细节

### 预训练 vs 从零训练

| 方面 | 预训练+微调 | 从零训练 |
|------|-----------|---------|
| 骨干初始化 | 预训练权重 ✅ | 随机初始化 |
| 收敛速度 | 快（15轮微调） | 慢（50轮） |
| 最终性能 | 高 | 中等 |
| 训练稳定性 | 稳定 ✅ | 可能不稳定 |

### 为什么预训练有效？

1. **骨干是瓶颈**: SASRec包含80%参数，训练最慢
2. **ID表示可迁移**: 预训练的物品嵌入包含序列模式知识
3. **多模态快速适应**: 专家参数少，可以快速学习新特征

### 迁移学习理论

```
预训练阶段:  ID → SASRec → 序列表示 → 推荐
                ↓
            学习序列模式

微调阶段:    ID + CLIP + Text → MoE → 融合表示 → 推荐
                ↓
            SASRec复用预训练知识
            专家学习多模态融合
```

---

**总结**: 预训练加速是一个简单但有效的策略，能够将训练时间从4-8小时缩短到2-3小时，同时保持或提升模型性能。强烈推荐在正式实验中使用！

🚀 **Happy Training!**
