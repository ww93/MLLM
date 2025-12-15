# 训练错误修复总结

## 修复日期
2025-12-10

## 修复的错误

### 1. UncertaintyWeightedLoss 初始化错误

**错误信息**:
```
TypeError: UncertaintyWeightedLoss.__init__() got an unexpected keyword argument 'loss_components'
```

**位置**: [joint_trainer.py:146-149](UR4Rec/models/joint_trainer.py#L146-L149)

**原因**: `UncertaintyWeightedLoss` 的 `__init__` 方法只接受 `num_tasks` 参数，但代码传入了 `loss_components` 和 `device`

**修复**:
```python
# 修复前
self.criterion = UncertaintyWeightedLoss(
    loss_components=loss_components,
    device=device
)

# 修复后
self.criterion = UncertaintyWeightedLoss(
    num_tasks=len(loss_components)
)
```

---

### 2. MultiModalRetrievalLoss 参数名称不匹配

**错误信息**: 参数名不匹配导致初始化失败

**位置**: [joint_trainer.py:152-158](UR4Rec/models/joint_trainer.py#L152-L158)

**原因**: `MultiModalRetrievalLoss` 使用不同的参数名（`temperature`, `alpha_consistency`, etc.）

**修复**:
```python
# 修复前
self.criterion = MultiModalRetrievalLoss(
    retrieval_loss_weight=retrieval_loss_weight,
    consistency_weight=consistency_weight,
    contrastive_weight=contrastive_weight,
    diversity_weight=diversity_weight,
    device=device
)

# 修复后
self.criterion = MultiModalRetrievalLoss(
    temperature=0.07,
    alpha_consistency=consistency_weight,
    beta_contrastive=contrastive_weight,
    gamma_diversity=diversity_weight
)
```

---

### 3. 模型前向传播参数名错误

**错误信息**:
```
TypeError: UR4RecV2MoE.forward() got an unexpected keyword argument 'candidate_items'
```

**位置**: [joint_trainer.py:352-358](UR4Rec/models/joint_trainer.py#L352-L358)

**原因**: 模型的 `forward` 方法接受 `target_items` 而不是 `candidate_items`

**修复**:
```python
# 修复前
final_scores, scores_dict = self.model(
    user_ids=user_ids,
    input_seq=input_seq,
    candidate_items=candidate_items,
    seq_padding_mask=seq_padding_mask
)

# 修复后
final_scores, scores_dict = self.model(
    user_ids=user_ids,
    input_seq=input_seq,
    target_items=candidate_items,
    seq_padding_mask=seq_padding_mask,
    return_components=True
)
```

**相同修复应用于**: [joint_trainer.py:623-628](UR4Rec/models/joint_trainer.py#L623-L628) (evaluate 方法)

---

### 4. SASRec 调用方法错误

**错误信息**:
```
TypeError: SASRec.forward() got multiple values for argument 'padding_mask'
```

**位置**: [ur4rec_v2_moe.py:178-182](UR4Rec/models/ur4rec_v2_moe.py#L178-L182)

**原因**: 错误地直接调用 `sasrec()` 并传入 `target_items`，但 `SASRec.forward()` 不接受此参数

**修复**:
```python
# 修复前
sasrec_logits = self.sasrec(
    input_seq,
    target_items,
    padding_mask=seq_padding_mask
)

# 修复后
sasrec_logits = self.sasrec.predict(
    input_seq,
    candidate_items=target_items,
    padding_mask=seq_padding_mask
)
```

---

### 5. 检索器返回形状不匹配

**错误信息**:
```
RuntimeError: The size of tensor a (32) must match the size of tensor b (6) at non-singleton dimension 1
```

**位置**: [retriever_moe_memory.py:305](UR4Rec/models/retriever_moe_memory.py#L305)

**原因**: `enhanced_score` 的形状是 `[1]` 而不是标量 `[]`，导致堆叠后形状为 `[batch, num_candidates, 1]`

**实际形状**:
- `sasrec_logits`: `[32, 6]` ✓
- `retriever_scores`: `[32, 6, 1]` ✗

**修复**:
```python
# 修复前
enhanced_score = (memory_integrated * target_item[i]).sum(dim=-1)

# 修复后
enhanced_score = (memory_integrated * target_item[i]).sum(dim=-1).squeeze()
```

---

### 6. 损失函数期望 Tensor 但收到 Dict

**错误信息**:
```
TypeError: unsupported operand type(s) for *: 'Tensor' and 'dict'
```

**位置**: [joint_trainer.py:395](UR4Rec/models/joint_trainer.py#L395)

**原因**: `UncertaintyWeightedLoss.forward()` 期望 tensor 输入，但收到 dict

**修复**:
```python
# 修复前
total_loss, loss_dict = self.criterion(losses)

# 修复后
# Convert dict to tensor in the order expected by UncertaintyWeightedLoss
loss_components = ['retrieval', 'sasrec']
if self.use_multimodal:
    loss_components.extend(['consistency', 'contrastive', 'diversity'])
losses_tensor = torch.stack([losses[key] for key in loss_components])

total_loss, weights = self.criterion(losses_tensor)

# Create loss_dict for logging
loss_dict = {key: losses[key].item() for key in losses}
loss_dict['total'] = total_loss.item()
```

---

### 7. 指标聚合时类型错误

**错误信息**:
```
TypeError: unsupported operand type(s) for +=: 'int' and 'str'
```

**位置**: [joint_trainer.py:553](UR4Rec/models/joint_trainer.py#L553)

**原因**: metrics 字典包含字符串值（如 'training_module'），无法累加

**修复**:
```python
# 修复前
for key, value in metrics.items():
    if key not in total_metrics:
        total_metrics[key] = 0
    total_metrics[key] += value

# 修复后
# 累积指标（只累积数值类型）
for key, value in metrics.items():
    if isinstance(value, (int, float)):
        if key not in total_metrics:
            total_metrics[key] = 0
        total_metrics[key] += value
```

---

## 测试结果

运行命令：
```bash
python UR4Rec/scripts/train_ur4rec_moe.py \
    --config UR4Rec/configs/ur4rec_moe_100k.yaml \
    --data_dir UR4Rec/data/Multimodal_Datasets \
    --llm_data_dir data/llm_generated \
    --output_dir outputs/ur4rec_moe_test \
    --epochs_per_stage 1 \
    --patience 1
```

**状态**: 训练可以启动并运行，所有参数传递错误已修复 ✓

---

## 已知问题

### NaN Loss

**观察**: 训练过程中出现 `loss=nan`

**可能原因**:
1. 学习率过高
2. 梯度爆炸
3. 数值不稳定（log(0), 除零等）
4. 输入数据问题

**建议解决方案**:
1. 检查数据预处理和归一化
2. 降低学习率
3. 添加梯度裁剪（已有：`gradient_clip=1.0`）
4. 检查损失函数中的数值稳定性
5. 添加调试输出以定位 NaN 来源

---

## 修复文件列表

1. [UR4Rec/models/joint_trainer.py](UR4Rec/models/joint_trainer.py)
   - 修复 UncertaintyWeightedLoss 初始化
   - 修复 MultiModalRetrievalLoss 初始化
   - 修复模型调用参数名
   - 修复损失字典到tensor转换
   - 修复metrics聚合类型检查

2. [UR4Rec/models/ur4rec_v2_moe.py](UR4Rec/models/ur4rec_v2_moe.py)
   - 修复 SASRec 调用方法

3. [UR4Rec/models/retriever_moe_memory.py](UR4Rec/models/retriever_moe_memory.py)
   - 修复 enhanced_score 形状

---

## 后续建议

1. **调试 NaN Loss**:
   - 添加 tensor 数值范围检查
   - 在每个损失计算后检查 NaN
   - 使用 `torch.autograd.detect_anomaly()` 定位问题

2. **验证数据质量**:
   - 检查序列长度分布
   - 验证用户偏好和物品描述的编码质量
   - 确认负采样策略合理性

3. **改进训练稳定性**:
   - 考虑使用更保守的学习率调度
   - 添加早期梯度监控
   - 实施损失缩放（loss scaling）

---

*生成时间: 2025-12-10*
*修复者: Claude*
