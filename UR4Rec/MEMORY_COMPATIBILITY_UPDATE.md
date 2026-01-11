# Memory兼容性更新 - 快速参考

## ✅ 已完成的修改

### 1. 核心API适配 (`fedmem_client.py`)

#### 修改1: `update()` 调用 (Line 498-504)
```python
# ❌ 旧版本
self.local_memory.update(
    item_id=item_id,
    loss_val=loss_val,
    text_emb=self._get_item_text_emb(item_id),
    img_emb=self._get_item_img_emb(item_id),      # 旧参数名
    id_emb=self._get_item_id_emb(item_id)
)

# ✅ 新版本
self.local_memory.update(
    item_id=item_id,
    id_emb=self._get_item_id_emb(item_id),         # 移到前面
    visual_emb=self._get_item_img_emb(item_id),    # 改名
    text_emb=self._get_item_text_emb(item_id),
    loss_val=loss_val                              # 移到最后
)
```

#### 修改2: `retrieve_multimodal_memory_batch()` 适配 (Line 570-578)
```python
# 新版本返回4个值，wrapper保持向后兼容
mem_vis, mem_txt, mem_id, mask = self.local_memory.retrieve_multimodal_memory_batch(
    batch_size=batch_size,
    top_k=top_k
)
# 只返回前2个（向后兼容）
return mem_vis, mem_txt
```

### 2. 参数文档更新

#### `fedmem_client.py` (Line 131-135)
```python
# 记忆参数 (Two-tier: ST + LT)
memory_capacity: int = 200,         # 从50改为200 ✓
surprise_threshold: float = 0.5,    # 兼容参数 ✓
```

#### `train_fedmem.py` (Line 480-487)
```python
# FedMem参数 (Two-tier Memory: ST + LT)
parser.add_argument("--memory_capacity", type=int, default=200,  # 从50改为200 ✓
                    help="LT (long-term) 记忆容量，推荐200 (ML-1M), ST固定50")
parser.add_argument("--surprise_threshold", type=float, default=0.5,  # 从0.7改为0.5 ✓
                    help="兼容参数，新版本主要使用novelty-based写入 (默认0.583)")
```

### 3. 打印信息更新 (`train_fedmem.py`, Line 416-418)
```python
print(f"    - 记忆架构: Two-tier (ST: 50, LT: {args.memory_capacity})")
print(f"    - LT写入策略: Novelty-based (threshold=0.583)")
print(f"    - 兼容参数 surprise_threshold: {args.surprise_threshold}")
```

## 📊 关键变化总结

| 方面 | 旧版本 | 新版本 | 影响 |
|------|--------|--------|------|
| **架构** | 单一buffer | Two-tier (ST+LT) | 更快响应+稳定存储 |
| **LT容量** | 50 | 200 | 更大存储空间 |
| **写入策略** | Surprise-based | Novelty-based (p90≈0.583) | ~10%写入率 |
| **update()参数** | (id, loss, text, img, id_emb) | (id, id_emb, visual, text, loss) | API变化 |
| **retrieve()返回** | 2个值 (vis, txt) | 4个值 (vis, txt, id, mask) | 向后兼容包装 |
| **surprise_threshold** | 0.7 | 0.5 (兼容参数) | Fallback用途 |

## 🔍 验证清单

运行以下命令验证修改是否生效：

```bash
# 1. 基本训练测试 (Stage 1)
python UR4Rec/scripts/train_fedmem.py \
  --data_dir UR4Rec/data/ml-1m \
  --stage pretrain_sasrec \
  --num_rounds 5 \
  --save_dir /tmp/test_memory

# 2. 检查日志输出，应该看到：
# - "记忆架构: Two-tier (ST: 50, LT: 200)"
# - "LT写入策略: Novelty-based (threshold=0.583)"
```

## 🐛 常见问题

### Q1: 报错 "update() missing 1 required positional argument: 'id_emb'"
**原因**: `id_emb` 现在是必需参数，不能为None

**解决**: 确保 `_get_item_id_emb()` 总是返回一个有效的tensor

### Q2: Memory统计显示 `st_size=0, lt_size=0`
**原因**: `update()` 没有被调用，或者id_emb为None

**调试**:
```python
# 在train_local_model()的update调用前添加
print(f"Updating memory: item={item_id}, id_emb={self._get_item_id_emb(item_id) is not None}")
```

### Q3: LT写入率 > 20%（异常高）
**原因**: 数据集特征不同，或者多模态特征质量较低

**解决**: 调整novelty阈值
```python
LocalDynamicMemory(
    capacity=200,
    lt_novelty_threshold=0.65,  # 提高阈值降低写入率
    ...
)
```

## 📚 相关文档

- **详细文档**: [MEMORY_UPDATE_SUMMARY.md](MEMORY_UPDATE_SUMMARY.md)
- **实现代码**: [local_dynamic_memory.py](models/local_dynamic_memory.py)
- **使用示例**: [train_fedmem.py](scripts/train_fedmem.py)
- **Stage 2/3修改**: [STAGE2_3_MODIFICATIONS.md](STAGE2_3_MODIFICATIONS.md)

## ✨ 下一步

1. ✅ 所有代码已更新，可以直接运行训练
2. 📊 监控 `st_size`, `lt_size`, LT写入率等统计指标
3. 🔧 根据实际数据集调优 `lt_novelty_threshold` (如需要)
4. 🚀 继续Stage 2/3轻量级对齐训练
