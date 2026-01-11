# Stage训练脚本兼容性说明

## 🔍 问题回答

### Q1: 能直接运行 `train_stage2_alignment.py` 吗？

**❌ 不能，不兼容**

原因：
1. **架构冲突**: 旧脚本使用"方案2"（保持512/384维），新代码使用轻量级投影（512→128, 384→128）
2. **Memory参数冲突**: 旧脚本 `memory_capacity=50`，新代码默认200 (Two-tier LT容量)
3. **参数量不匹配**: 旧脚本训练~4M参数，新方案只训练~139K参数

**解决方案**: 使用新创建的 `train_stage2_lightweight.py` ✅

### Q2: 刚刚的修改是否会影响Stage 1？

**✅ 不会影响，向后兼容**

理由：
1. **Stage 1是纯ID训练**: 不使用多模态特征，不受投影层修改影响
2. **Memory修改向后兼容**:
   - `update()` API兼容（参数顺序调整但不影响功能）
   - Two-tier架构对Stage 1透明（ST+LT vs 单一buffer，对训练无影响）
3. **默认参数更新**:
   - `memory_capacity`: 50→200 (更大容量，对Stage 1有利)
   - `surprise_threshold`: 0.7→0.5 (仍然有效)

**实测建议**: Stage 1可以直接使用新代码训练，预期性能 HR@10 ≈ 0.60-0.70 不变

### Q3: 能否直接从Stage 2开始运行？

**❌ 不推荐，必须先运行Stage 1**

原因：
1. **Stage 2依赖Stage 1 checkpoint**:
   - 需要加载预训练的SASRec骨干
   - 需要加载训练好的Item Embedding
2. **没有Stage 1，投影层无法对齐**:
   - 投影层的目标是将多模态特征对齐到**预训练的ID空间**
   - 如果ID空间是随机的，对齐没有意义
3. **性能会很差**:
   - 从零开始训练Stage 2 → 随机SASRec + 随机投影层 → HR@10 < 0.30

**必须流程**:
```
Stage 1 (纯ID) → Stage 2 (对齐) → Stage 3 (MoE)
```

**如果已有Stage 1 checkpoint**: 可以直接运行Stage 2 ✅

## 📋 新旧脚本对比

### 旧脚本（不兼容）

| 脚本 | 方案 | 参数量 | Memory | 状态 |
|------|------|--------|--------|------|
| `train_stage2_alignment.py` | 方案2 (512/384维) | ~4M | capacity=50 | ❌ 不兼容 |
| `train_stage2_fixed.py` | 方案2变体 | ~4M | capacity=50 | ❌ 不兼容 |
| `train_stage3_moe.py` | 依赖旧Stage 2 | 全部 | capacity=50 | ❌ 不兼容 |

### 新脚本（兼容）

| 脚本 | 方案 | 参数量 | Memory | 状态 |
|------|------|--------|--------|------|
| `train_stage1_backbone.py` | 纯ID SASRec | 全部 | LT=200 | ✅ 兼容 |
| `train_stage2_lightweight.py` | 轻量级投影 | ~139K | LT=200 | ✅ **新创建** |
| `train_stage3_lightweight.py` | MoE微调 | 全部-item_emb | LT=200 | ✅ **新创建** |

## 🚀 正确的使用流程

### 方案A: 使用独立脚本（推荐，更简单）

```bash
# Step 1: Stage 1 训练（纯ID SASRec）
python UR4Rec/scripts/train_stage1_backbone.py

# Step 2: Stage 2 训练（轻量级对齐）
python UR4Rec/scripts/train_stage2_lightweight.py

# Step 3: Stage 3 训练（MoE微调）
python UR4Rec/scripts/train_stage3_lightweight.py
```

### 方案B: 使用 train_fedmem.py（更灵活）

```bash
# Step 1: Stage 1
python UR4Rec/scripts/train_fedmem.py \
  --stage pretrain_sasrec \
  --data_dir UR4Rec/data/ml-1m \
  --save_dir UR4Rec/checkpoints/stage1_backbone \
  --num_rounds 50 \
  --learning_rate 1e-3

# Step 2: Stage 2
python UR4Rec/scripts/train_fedmem.py \
  --stage align_projectors \
  --stage1_checkpoint UR4Rec/checkpoints/stage1_backbone/fedmem_model.pt \
  --visual_file clip_features.pt \
  --text_file text_features.pt \
  --data_dir UR4Rec/data/ml-1m \
  --save_dir UR4Rec/checkpoints/stage2_lightweight \
  --num_rounds 30 \
  --learning_rate 1e-3

# Step 3: Stage 3
python UR4Rec/scripts/train_fedmem.py \
  --stage finetune_moe \
  --stage1_checkpoint UR4Rec/checkpoints/stage1_backbone/fedmem_model.pt \
  --stage2_checkpoint UR4Rec/checkpoints/stage2_lightweight/fedmem_model.pt \
  --visual_file clip_features.pt \
  --text_file text_features.pt \
  --data_dir UR4Rec/data/ml-1m \
  --save_dir UR4Rec/checkpoints/stage3_moe \
  --num_rounds 50 \
  --learning_rate 5e-4
```

## 📊 预期性能

| Stage | 目标 | 预期HR@10 | 训练时间 | 参数量 |
|-------|------|----------|---------|--------|
| **Stage 1** | 纯ID预训练 | 0.60-0.70 | 基准 | 全部 |
| **Stage 2** | 轻量级对齐 | 0.60-0.67 | **快20倍** | ~139K |
| **Stage 3** | MoE微调 | 0.65-0.75 | 基准 | 全部-item_emb |

**成功标志**:
- Stage 2 HR@10 接近 Stage 1 (±0.05) ✓
- Stage 3 HR@10 > Stage 1 (多模态有效) ✓
- 训练损失稳定下降 ✓

## 🔧 修改内容总结

### 代码修改

1. **模型架构** (`ur4rec_v2_moe.py`):
   - ✅ 添加 `visual_proj` (512→128)
   - ✅ 添加 `text_proj` (384→128)
   - ✅ 添加 `align_gating` MLP
   - ✅ 设置 `preserve_multimodal_dim=False`
   - ✅ 统一LayerNorms为128维

2. **训练脚本** (`train_fedmem.py`):
   - ✅ Stage 2冻结策略：只训练投影层+gating
   - ✅ Stage 3冻结策略：冻结item_emb，训练其他
   - ✅ 更新Memory参数默认值 (LT=200)

3. **Memory机制** (`fedmem_client.py`):
   - ✅ `update()` 参数顺序调整
   - ✅ `retrieve_multimodal_memory_batch()` 返回值包装
   - ✅ Two-tier架构透明集成

### 新创建的文件

1. ✅ `train_stage2_lightweight.py` - 轻量级Stage 2脚本
2. ✅ `train_stage3_lightweight.py` - 对应的Stage 3脚本
3. ✅ `STAGE2_3_MODIFICATIONS.md` - Stage 2/3修改详细说明
4. ✅ `MEMORY_UPDATE_SUMMARY.md` - Memory机制更新说明
5. ✅ `MEMORY_COMPATIBILITY_UPDATE.md` - Memory兼容性快速参考
6. ✅ `STAGE_SCRIPTS_COMPATIBILITY.md` - 本文档

## 🐛 常见问题

### Q: Stage 2报错 "FileNotFoundError: stage1_checkpoint not found"

**原因**: 没有运行Stage 1

**解决**:
```bash
python UR4Rec/scripts/train_stage1_backbone.py
```

### Q: Stage 2性能比Stage 1低很多（HR@10 < 0.50）

**可能原因**:
1. Stage 1 checkpoint加载失败（权重是随机的）
2. 投影层学习率过大，破坏了特征
3. 多模态特征文件损坏或格式错误

**调试**:
```python
# 在train_fedmem.py的Stage 2加载后添加：
for name, param in global_model.named_parameters():
    if 'sasrec.item_embedding' in name:
        print(f"Embedding mean: {param.mean():.4f}, std: {param.std():.4f}")
        break
# 如果mean和std接近0，说明没有加载Stage 1权重
```

**解决**:
1. 检查Stage 1 checkpoint路径
2. 降低学习率: `--learning_rate 5e-4`
3. 检查多模态特征文件

### Q: Stage 3性能不如Stage 1

**可能原因**:
1. 多模态特征质量低（噪声多）
2. Router学习不充分（lb_loss很大）
3. 学习率过大，破坏了Stage 2的对齐

**解决**:
1. 降低学习率: `--learning_rate 1e-4`
2. 增加训练轮数: `--num_rounds 80`
3. 检查Router权重分布（应该有差异，不是均匀0.5/0.5）
4. 考虑只使用质量更好的模态（禁用另一个）

### Q: Memory统计异常（st_size=0, lt_size=0）

**原因**: `update()` 调用失败，id_emb为None

**解决**: 检查 `_get_item_id_emb()` 是否返回有效tensor

## ✅ 检查清单

在运行训练前，请确认：

- [ ] **Stage 1**:
  - [ ] 数据文件存在: `UR4Rec/data/ml-1m/subset_ratings.dat`
  - [ ] 运行 `train_stage1_backbone.py`
  - [ ] 验证 HR@10 ≈ 0.60-0.70
  - [ ] 保存到: `UR4Rec/checkpoints/stage1_backbone/`

- [ ] **Stage 2**:
  - [ ] Stage 1 checkpoint存在
  - [ ] 多模态特征文件存在:
    - [ ] `UR4Rec/data/ml-1m/clip_features.pt`
    - [ ] `UR4Rec/data/ml-1m/text_features.pt`
  - [ ] 运行 `train_stage2_lightweight.py`
  - [ ] 验证 HR@10 ≈ 0.60-0.67 (接近Stage 1)
  - [ ] 验证可训练参数 ~139K
  - [ ] 保存到: `UR4Rec/checkpoints/stage2_lightweight/`

- [ ] **Stage 3**:
  - [ ] Stage 1 & 2 checkpoints都存在
  - [ ] 运行 `train_stage3_lightweight.py`
  - [ ] 验证 HR@10 > Stage 1 (多模态有效)
  - [ ] 检查Router权重有差异
  - [ ] 保存到: `UR4Rec/checkpoints/stage3_moe/`

## 📚 相关文档

- **Stage 2/3详细修改**: [STAGE2_3_MODIFICATIONS.md](STAGE2_3_MODIFICATIONS.md)
- **Memory机制更新**: [MEMORY_UPDATE_SUMMARY.md](MEMORY_UPDATE_SUMMARY.md)
- **Memory兼容性**: [MEMORY_COMPATIBILITY_UPDATE.md](MEMORY_COMPATIBILITY_UPDATE.md)

---

**总结**: 使用新创建的 `train_stage2_lightweight.py` 和 `train_stage3_lightweight.py` 脚本，按照 Stage 1 → 2 → 3 的顺序运行。旧的 `train_stage2_alignment.py` 已过时，不兼容当前代码。
