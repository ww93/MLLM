# 自适应交替训练使用指南

## 概述

自适应交替训练（Adaptive Alternating Training）是一种动态训练策略，能够根据损失变化自动决定训练哪个模块（SASRec 或 Retriever）。

### 核心思想

传统的交替训练按固定频率切换模块（例如每隔一个batch切换一次），但这忽略了：
- 不同模块的收敛速度可能不同
- 某个模块可能已经收敛，但仍在被强制训练
- 另一个模块可能需要更多训练步数

**自适应交替训练**通过监控每个模块的损失变化率来动态决策：
- **损失变化率 < 阈值** → 该模块趋于收敛，切换到另一模块
- **损失变化率 ≥ 阈值** → 该模块仍在优化，继续训练

---

## 快速开始

### 1. 启用自适应交替训练

在创建 `JointTrainer` 时设置参数：

```python
from UR4Rec.models.joint_trainer import JointTrainer

trainer = JointTrainer(
    model=model,
    device="cuda",
    # 启用自适应交替训练
    use_adaptive_alternating=True,
    adaptive_switch_threshold=0.01,  # 损失变化率阈值
    adaptive_min_steps=5,            # 每个模块最少连续训练步数
)
```

### 2. 设置训练阶段

自适应交替训练**仅在联合微调阶段生效**：

```python
# 阶段1: 预训练 SASRec
trainer.set_training_stage("pretrain_sasrec")
trainer.train_epoch(train_loader, epoch=1)

# 阶段2: 预训练 Retriever
trainer.set_training_stage("pretrain_retriever")
trainer.train_epoch(train_loader, epoch=2)

# 阶段3: 联合微调（启用自适应交替训练）
trainer.set_training_stage("joint_finetune")
for epoch in range(3, 10):
    metrics = trainer.train_epoch(train_loader, epoch=epoch)
    print(f"Epoch {epoch} - 切换次数: {metrics.get('adaptive_switch_count', 0)}")

# 阶段4: 端到端训练
trainer.set_training_stage("end_to_end")
trainer.train_epoch(train_loader, epoch=10)
```

---

## 参数详解

### `use_adaptive_alternating` (bool, default=True)

是否启用自适应交替训练。

- `True`: 使用自适应策略，根据损失变化动态切换
- `False`: 使用传统策略，按步数奇偶性切换

### `adaptive_switch_threshold` (float, default=0.01)

损失变化率阈值。

计算方式：
```
change_rate = |loss[t] - loss[t-k]| / loss[t-k]
```

- **阈值越小**：更早判定收敛，切换更频繁
- **阈值越大**：更晚判定收敛，切换更少

**推荐值**：
- `0.005`: 对损失变化敏感，快速切换（适合小数据集）
- `0.01`: 平衡（默认，推荐）
- `0.02`: 对损失变化保守，较少切换（适合大数据集）

### `adaptive_min_steps` (int, default=5)

每个模块最少连续训练的步数。

防止过于频繁地切换模块，保证每个模块有足够的优化步数。

**推荐值**：
- `3`: 最小限制，适合快速实验
- `5`: 平衡（默认，推荐）
- `10`: 较强限制，适合大batch size

---

## 训练监控

### 进度条显示

训练时会显示当前训练的模块和切换次数：

```
Epoch 3: 100%|██████| 100/100 [00:30<00:00, 3.33it/s, loss=0.4521, lr_s=1.0e-03, lr_r=1.0e-04, train=SAS, switch=3]
```

字段说明：
- `train=SAS`: 当前训练 SASRec
- `train=RET`: 当前训练 Retriever
- `train=ALL`: 同时训练两个模块（仅在end_to_end阶段）
- `switch=3`: 本epoch内切换了3次

### 训练日志

每次切换模块时会打印详细信息：

```
[AdaptiveAlternating] Step 234: 切换训练模块
  从 sasrec → retriever
  原因: 损失趋于稳定 (变化率: 0.0078 < 0.01)
  总切换次数: 5
```

### 统计信息

可以通过 `get_stats()` 获取详细统计：

```python
if trainer.use_adaptive_alternating:
    stats = trainer.adaptive_alternating.get_stats()
    print(f"总步数: {stats['total_steps']}")
    print(f"当前模块: {stats['current_module']}")
    print(f"切换次数: {stats['switch_count']}")
    print(f"SASRec 训练比例: {stats['sasrec_training_ratio']:.2%}")
    print(f"Retriever 训练比例: {stats['retriever_training_ratio']:.2%}")
```

---

## 最佳实践

### 1. 超参数调优顺序

1. **先固定阈值** (`adaptive_switch_threshold=0.01`)
2. **调整最小步数** (`adaptive_min_steps`)
   - 观察训练日志，如果切换过于频繁，增加此值
3. **微调阈值**
   - 如果某个模块训练不足，降低阈值（更频繁切换）
   - 如果切换过于频繁影响收敛，提高阈值

### 2. 与其他策略组合

可以同时启用多个训练策略：

```python
trainer = JointTrainer(
    model=model,
    # 自适应交替训练
    use_adaptive_alternating=True,
    adaptive_switch_threshold=0.01,
    adaptive_min_steps=5,

    # 课程学习（可选）
    use_curriculum_learning=True,

    # Memory Bank 对比学习（可选）
    use_memory_bank=True,
    memory_bank_size=65536,

    # 知识蒸馏（可选）
    use_knowledge_distillation=True,
    kd_temperature=4.0,
    kd_weight=0.1
)
```

### 3. 调试技巧

如果训练不稳定，可以：

1. **检查切换次数**：
   - 过少（<2次/epoch）：降低阈值或最小步数
   - 过多（>10次/epoch）：提高阈值或最小步数

2. **检查训练比例**：
   ```python
   stats = trainer.adaptive_alternating.get_stats()
   print(f"SASRec: {stats['sasrec_training_ratio']:.1%}")
   print(f"Retriever: {stats['retriever_training_ratio']:.1%}")
   ```
   理想情况：两者接近 50%-50%

3. **可视化损失曲线**：
   ```python
   import matplotlib.pyplot as plt

   # 获取损失历史
   sasrec_losses = list(trainer.adaptive_alternating.sasrec_loss_history)
   retriever_losses = list(trainer.adaptive_alternating.retriever_loss_history)

   plt.plot(sasrec_losses, label='SASRec')
   plt.plot(retriever_losses, label='Retriever')
   plt.legend()
   plt.show()
   ```

---

## 与传统方法对比

| 特性 | 传统交替训练 | 自适应交替训练 |
|------|-------------|---------------|
| **切换频率** | 固定（每N步） | 动态（基于收敛状态） |
| **训练效率** | 可能浪费步数 | 自动聚焦需要优化的模块 |
| **收敛速度** | 取决于固定频率 | 更快（根据实际需求调整） |
| **超参数调优** | 需要调整切换频率N | 只需调整阈值和最小步数 |
| **适用场景** | 两模块收敛速度相似 | 两模块收敛速度不同 |

---

## 预期效果

根据实验（详见 [TRAINING_IMPROVEMENT_PROPOSAL.md](TRAINING_IMPROVEMENT_PROPOSAL.md)），启用自适应交替训练后：

- **Hit@10**: +3% ~ +5% 提升
- **NDCG@10**: +4% ~ +6% 提升
- **训练步数**: 减少 10% ~ 15%
- **收敛稳定性**: 显著提升

---

## 故障排查

### Q1: 提示 "模块未初始化"

**错误**：
```
AttributeError: 'JointTrainer' object has no attribute 'adaptive_alternating'
```

**解决**：
确认创建 trainer 时启用了该功能：
```python
trainer = JointTrainer(model, use_adaptive_alternating=True)
```

### Q2: 切换次数为 0

**原因**：
- 可能在 pretrain 阶段（不会切换）
- 最小步数设置过大
- 阈值设置过小，损失变化率一直高于阈值

**解决**：
1. 确认在 `joint_finetune` 阶段
2. 降低 `adaptive_min_steps`
3. 提高 `adaptive_switch_threshold`

### Q3: 切换过于频繁

**原因**：
- 阈值设置过高
- 最小步数设置过小
- 损失震荡严重

**解决**：
1. 降低 `adaptive_switch_threshold`（如 0.01 → 0.005）
2. 提高 `adaptive_min_steps`（如 5 → 10）
3. 检查学习率是否过高

---

## 完整示例

```python
import torch
from torch.utils.data import DataLoader
from UR4Rec.models.ur4rec_v2 import UR4RecV2
from UR4Rec.models.joint_trainer import JointTrainer

# 1. 创建模型
model = UR4RecV2(
    num_items=10000,
    sasrec_hidden_dim=256,
    text_embedding_dim=384,
    retriever_output_dim=256
)

# 2. 创建训练器（启用自适应交替训练）
trainer = JointTrainer(
    model=model,
    device="cuda",
    sasrec_lr=1e-3,
    retriever_lr=1e-4,
    # 自适应交替训练
    use_adaptive_alternating=True,
    adaptive_switch_threshold=0.01,
    adaptive_min_steps=5
)

# 3. 四阶段训练
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

# 阶段1: 预训练 SASRec
print("\n=== 阶段1: 预训练 SASRec ===")
trainer.set_training_stage("pretrain_sasrec")
for epoch in range(1, 6):
    metrics = trainer.train_epoch(train_loader, epoch)
    print(f"Epoch {epoch} - Loss: {metrics['total_loss']:.4f}")

# 阶段2: 预训练 Retriever
print("\n=== 阶段2: 预训练 Retriever ===")
trainer.set_training_stage("pretrain_retriever")
for epoch in range(6, 11):
    metrics = trainer.train_epoch(train_loader, epoch)
    print(f"Epoch {epoch} - Loss: {metrics['total_loss']:.4f}")

# 阶段3: 联合微调（自适应交替训练）
print("\n=== 阶段3: 联合微调（自适应交替） ===")
trainer.set_training_stage("joint_finetune")
for epoch in range(11, 21):
    metrics = trainer.train_epoch(train_loader, epoch)

    # 打印统计信息
    stats = trainer.adaptive_alternating.get_stats()
    print(f"Epoch {epoch}:")
    print(f"  Loss: {metrics['total_loss']:.4f}")
    print(f"  切换次数: {stats['switch_count']}")
    print(f"  当前模块: {stats['current_module']}")
    print(f"  训练比例 - SASRec: {stats['sasrec_training_ratio']:.1%}, "
          f"Retriever: {stats['retriever_training_ratio']:.1%}")

    # 每5个epoch验证一次
    if epoch % 5 == 0:
        val_metrics = trainer.evaluate(val_loader, k_list=[5, 10, 20])
        print(f"  Validation - Hit@10: {val_metrics['hit@10']:.4f}, "
              f"NDCG@10: {val_metrics['ndcg@10']:.4f}")

# 阶段4: 端到端训练
print("\n=== 阶段4: 端到端训练 ===")
trainer.set_training_stage("end_to_end")
for epoch in range(21, 26):
    metrics = trainer.train_epoch(train_loader, epoch)
    val_metrics = trainer.evaluate(val_loader)
    print(f"Epoch {epoch} - Loss: {metrics['total_loss']:.4f}, "
          f"Hit@10: {val_metrics['hit@10']:.4f}")

# 5. 保存最终模型
trainer.save_checkpoint("checkpoints/final_model.pt", epoch=25, metrics=val_metrics)
```

---

## 参考资料

- [TRAINING_IMPROVEMENT_PROPOSAL.md](TRAINING_IMPROVEMENT_PROPOSAL.md) - 完整的训练改进策略提案
- [training_strategies.py](UR4Rec/models/training_strategies.py) - 实现源码
- [joint_trainer.py](UR4Rec/models/joint_trainer.py) - 联合训练器

---

## 总结

✅ **自适应交替训练已实现并集成到 JointTrainer**

**关键优势**：
- 🚀 自动决策训练哪个模块，无需手动调整切换频率
- 📈 根据收敛状态动态优化，训练更高效
- 📊 实时监控和统计，便于分析和调试
- 🎯 预期效果：Hit@10 +3~5%, NDCG@10 +4~6%

**使用方式**：
```python
trainer = JointTrainer(model, use_adaptive_alternating=True)
trainer.set_training_stage("joint_finetune")
trainer.train_epoch(train_loader, epoch)
```

**不需要**修改数据加载、模型定义等其他代码，直接启用即可！
