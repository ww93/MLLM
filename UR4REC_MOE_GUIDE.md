# UR4Rec V2 with MoE 使用指南

## 📋 概述

**UR4Rec V2 with MoE** 是一个结合了以下特性的推荐系统：

1. **SASRec**: 序列模式学习
2. **MoE-enhanced Retriever**: 使用混合专家（MoE）机制增强的文本偏好检索器
3. **User Memory**: 动态用户记忆机制，追踪用户偏好变化
4. **LLM Integration**: 使用大语言模型生成的用户偏好和物品描述
5. **Multi-stage Training**: 四阶段训练策略

---

## 🎯 架构特点

### 核心组件

```
UR4RecV2MoE
├── SASRec (序列建模)
│   └── Transformer-based sequential recommendation
│
└── TextPreferenceRetrieverMoE (MoE增强检索器)
    ├── Text Encoder (预训练句子编码器)
    ├── RetrieverMoEMemory
    │   ├── MoE Block (多专家融合)
    │   │   ├── Expert 1: User Preference
    │   │   ├── Expert 2: Item Description
    │   │   └── Expert 3: Item Embedding
    │   └── User Memory (动态记忆机制)
    │       ├── GRU-based memory update
    │       ├── Drift detection
    │       └── Memory persistence
    └── Fusion Layer (加权融合)
```

### 关键改进

1. **MoE 替代简单融合**: 使用多专家机制动态融合多个信息源
2. **用户记忆**: 追踪用户长期和短期偏好变化
3. **记忆持久化**: 支持保存和加载用户记忆状态
4. **自适应更新**: 基于交互次数、偏好漂移或时间的记忆更新策略

---

## 🚀 快速开始

### 1. 准备环境

```bash
cd /Users/admin/Desktop/MLLM
source venv/bin/activate

# 确保已安装必要的包
pip install sentence-transformers
```

### 2. 生成 LLM 数据（如果还没有）

```bash
export DASHSCOPE_API_KEY="your-api-key"

# 生成用户偏好和物品描述
python UR4Rec/models/llm_generator.py
```

生成的文件：
- `data/llm_generated/user_preferences.json`
- `data/llm_generated/item_descriptions.json`

### 3. 准备训练数据

确保数据目录包含：
```
UR4Rec/data/Multimodal_Datasets/
├── train_sequences.npy
├── val_sequences.npy
├── test_sequences.npy
└── item_map.json
```

### 4. 开始训练

```bash
python UR4Rec/scripts/train_ur4rec_moe.py \
    --config UR4Rec/configs/ur4rec_moe_100k.yaml \
    --data_dir UR4Rec/data/Multimodal_Datasets \
    --llm_data_dir data/llm_generated \
    --output_dir outputs/ur4rec_moe \
    --epochs_per_stage 10 \
    --patience 5
```

---

## ⚙️ 配置说明

### Memory 配置

```yaml
# Memory mechanism parameters
max_memory_size: 10          # 保存的历史状态数量
update_trigger: "INTERACTION_COUNT"  # 更新触发器
interaction_threshold: 10    # 交互次数阈值
drift_threshold: 0.3         # 偏好漂移阈值
decay_factor: 0.95          # 记忆衰减因子
```

**更新触发器选项**:
- `INTERACTION_COUNT`: 每 N 次交互后更新
- `DRIFT_THRESHOLD`: 当偏好漂移超过阈值时更新
- `TIME_BASED`: 定期更新
- `EXPLICIT`: 仅手动触发

### MoE 配置

```yaml
# MoE parameters
moe_num_heads: 8            # 注意力头数
moe_dropout: 0.1            # Dropout 率
moe_num_proxies: 4          # 代理 token 数量
```

### Fusion 配置

```yaml
# Fusion parameters
fusion_method: "weighted"   # 融合方法
sasrec_weight: 0.5         # SASRec 权重
retriever_weight: 0.5      # Retriever 权重
```

**融合方法选项**:
- `weighted`: 固定权重加权求和
- `learned`: 学习融合权重
- `adaptive`: 基于表示自适应融合

---

## 📊 训练阶段

### 阶段 1: 预训练 SASRec

```
目标: 训练序列建模能力
优化: 仅 SASRec 参数
冻结: Retriever 参数
```

### 阶段 2: 预训练 Retriever

```
目标: 训练文本偏好匹配能力
优化: 仅 Retriever 参数
冻结: SASRec 参数
特性: 开始构建用户记忆
```

### 阶段 3: 联合微调（自适应交替）

```
目标: 协同优化两个模块
优化: 交替优化 SASRec 和 Retriever
策略: 基于损失变化自适应切换
特性: 持续更新用户记忆
```

### 阶段 4: 端到端训练

```
目标: 全局优化
优化: 同时优化所有参数
特性: 最终融合层调整
```

---

## 💾 模型保存与加载

### 保存

训练过程自动保存：

```
outputs/ur4rec_moe/
├── pretrain_sasrec_best.pt        # 阶段 1 最佳模型
├── pretrain_sasrec_memories.pt    # 阶段 1 记忆
├── pretrain_retriever_best.pt     # 阶段 2 最佳模型
├── pretrain_retriever_memories.pt # 阶段 2 记忆
├── joint_finetune_best.pt         # 阶段 3 最佳模型
├── joint_finetune_memories.pt     # 阶段 3 记忆
├── end_to_end_best.pt             # 阶段 4 最佳模型
├── end_to_end_memories.pt         # 阶段 4 记忆
├── final_model.pt                 # 最终模型
├── final_memories.pt              # 最终记忆
└── results.json                   # 训练结果
```

### 加载

```python
from UR4Rec.models.ur4rec_v2_moe import UR4RecV2MoE

# 创建模型
model = UR4RecV2MoE(...)

# 加载模型权重
model.load_state_dict(torch.load('outputs/ur4rec_moe/final_model.pt'))

# 加载用户记忆
model.load_memories('outputs/ur4rec_moe/final_memories.pt')
```

---

## 📈 监控与调试

### 查看记忆统计

```python
# 获取记忆统计信息
memory_stats = model.get_memory_stats()
print(memory_stats)

# 输出示例:
# {
#     'total_users': 938,
#     'users_with_memory': 756,
#     'avg_memory_size': 7.2,
#     'avg_interaction_count': 15.3
# }
```

### TensorBoard 日志

训练过程会生成 TensorBoard 日志：

```bash
tensorboard --logdir outputs/ur4rec_moe/logs
```

可视化内容：
- 训练/验证损失
- 评估指标（Hit@K, NDCG@K, MRR）
- 记忆统计
- MoE 专家权重分布

---

## 🔧 高级用法

### 自定义记忆更新策略

```python
from UR4Rec.models.retriever_moe_memory import MemoryConfig, UpdateTrigger

# 基于偏好漂移的更新
memory_config = MemoryConfig(
    memory_dim=256,
    max_memory_size=15,
    update_trigger=UpdateTrigger.DRIFT_THRESHOLD,
    drift_threshold=0.25,  # 更敏感的漂移检测
    decay_factor=0.9       # 更快的遗忘
)
```

### 自定义融合策略

```python
# 使用自适应融合
model = UR4RecV2MoE(
    ...,
    fusion_method='adaptive',  # 基于表示自适应融合
    # sasrec_weight 和 retriever_weight 将被忽略
)
```

### 推理时的记忆管理

```python
# 推理时不更新记忆
scores, info = model(
    user_ids=user_ids,
    input_seq=input_seq,
    target_items=target_items,
    update_memory=False  # 关键：推理时不更新
)

# 手动更新记忆（如果需要）
model.preference_retriever.moe_retriever._update_memory(
    user_memory=...,
    current_repr=...,
    force=True
)
```

---

## 🧪 实验建议

### 消融实验

#### 1. 测试 MoE 的贡献

```yaml
# 禁用 MoE，使用简单平均
moe_num_proxies: 0  # 设为 0 禁用自注意力
```

#### 2. 测试记忆机制的贡献

```yaml
# 禁用记忆
max_memory_size: 0  # 设为 0 禁用记忆
```

#### 3. 测试不同融合策略

```yaml
# 尝试不同的融合方法
fusion_method: "weighted"  # vs "learned" vs "adaptive"
```

### 超参数调优

#### 重要超参数

1. **记忆更新频率**:
   ```yaml
   interaction_threshold: [5, 10, 15, 20]
   ```

2. **记忆衰减速度**:
   ```yaml
   decay_factor: [0.9, 0.95, 0.99]
   ```

3. **融合权重**:
   ```yaml
   sasrec_weight: [0.3, 0.5, 0.7]
   retriever_weight: [0.3, 0.5, 0.7]
   ```

4. **MoE 专家数量**:
   ```yaml
   moe_num_proxies: [2, 4, 8]
   ```

---

## 📝 对比：train_v2 vs train_moe_memory vs train_ur4rec_moe

| 特性 | train_v2 | train_moe_memory | **train_ur4rec_moe** (新) |
|------|----------|------------------|---------------------------|
| **模型** | UR4RecV2 | UR4RecMoEMemory | **UR4RecV2MoE** |
| **SASRec** | ✅ | ❌ | ✅ |
| **Text Retriever** | ✅ (简单) | ❌ | ✅ (MoE-enhanced) |
| **MoE 机制** | ❌ | ✅ | ✅ |
| **User Memory** | ❌ | ✅ | ✅ |
| **LLM 数据** | ✅ | ❌ | ✅ |
| **多阶段训练** | ✅ | ❌ | ✅ |
| **记忆持久化** | ❌ | ✅ | ✅ |
| **推荐** | 最全面 | - | **✅ 推荐** |

**结论**: `train_ur4rec_moe` 是 `train_v2` 和 `train_moe_memory` 的最佳结合，推荐使用！

---

## 🐛 常见问题

### Q1: 如何处理大规模数据集？

**A**: 调整批次大小和记忆大小：

```yaml
batch_size: 64  # 减小批次
max_memory_size: 5  # 减少记忆状态数量
```

### Q2: 记忆更新太频繁/太少怎么办？

**A**: 调整更新触发器：

```yaml
# 更频繁
interaction_threshold: 5

# 更少
interaction_threshold: 20

# 或使用漂移检测
update_trigger: "DRIFT_THRESHOLD"
drift_threshold: 0.2  # 更敏感
```

### Q3: 如何平衡 SASRec 和 Retriever 的贡献？

**A**: 调整融合权重或使用学习型融合：

```yaml
# 方法 1: 调整权重
sasrec_weight: 0.7  # 更依赖序列模式
retriever_weight: 0.3

# 方法 2: 使用学习型融合
fusion_method: "learned"
```

### Q4: 训练时间太长？

**A**:
1. 减少每阶段的 epoch 数
2. 使用更小的模型
3. 增加 patience 值（早停）

```yaml
epochs_per_stage: 5  # 减少 epoch
patience: 3  # 更早停止
sasrec_num_blocks: 1  # 更小的模型
```

---

## 📚 相关文档

- [LLM_PROMPTS.md](LLM_PROMPTS.md) - LLM Prompt 说明
- [GENERATE_LLM_PREFERENCES.md](GENERATE_LLM_PREFERENCES.md) - LLM 生成指南
- [DATA_LOADING_GUIDE.md](DATA_LOADING_GUIDE.md) - 数据加载指南
- [ADAPTIVE_TRAINING_GUIDE.md](ADAPTIVE_TRAINING_GUIDE.md) - 自适应训练指南

---

## 🎉 总结

**UR4Rec V2 with MoE** 成功整合了：
- ✅ **序列建模** (SASRec)
- ✅ **文本语义理解** (TextEncoder + LLM)
- ✅ **多专家融合** (MoE)
- ✅ **动态用户记忆** (User Memory)
- ✅ **多阶段训练** (4-stage training)

**适用场景**:
- 需要结合序列和语义信息的推荐任务
- 用户偏好会随时间变化的动态场景
- 已有 LLM 生成的用户/物品描述
- 追求 SOTA 性能的研究项目

---

*创建时间: 2025-12-10*
*版本: 1.0*
