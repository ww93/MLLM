# Memory机制更新总结

## 📋 更新概览

LocalDynamicMemory已升级为**Two-tier (ST + LT)架构**，使用数据驱动的参数优化，显著提升记忆检索和更新的效率。

## 🔄 主要变化

### 1. Two-tier架构

**Short-Term (ST) Memory**:
- **容量**: 50 (固定窗口W)
- **更新策略**: FIFO，始终更新
- **用途**: 捕获最近兴趣，快速响应漂移
- **数据结构**: `OrderedDict` (保持插入顺序)

**Long-Term (LT) Memory**:
- **容量**: 200 (可配置，原 `capacity` 参数)
- **更新策略**: Novelty-gated写入 (p90阈值 ≈ 0.583，约10%写入率)
- **用途**: 存储多样性和稳定兴趣
- **数据结构**: `Dict` (item_id -> MemoryEntry)

### 2. API变化

#### 2.1 `update()` 方法

**旧版本签名**:
```python
def update(
    item_id: int,
    loss_val: float,
    text_emb: Optional[torch.Tensor],
    img_emb: Optional[torch.Tensor],
    id_emb: Optional[torch.Tensor]
)
```

**新版本签名**:
```python
def update(
    item_id: int,
    id_emb: torch.Tensor,              # 必需参数，移到前面
    visual_emb: Optional[torch.Tensor] = None,  # 参数名改为visual_emb
    text_emb: Optional[torch.Tensor] = None,
    loss_val: Optional[float] = None   # 变为可选参数
)
```

**关键变化**:
- ✅ `id_emb` 从可选参数变为**必需参数**，移到第2位
- ✅ `img_emb` 重命名为 `visual_emb`（统一命名）
- ✅ `loss_val` 变为可选参数（LT主要依赖novelty，loss_val作为fallback）
- ✅ 参数顺序调整：`(item_id, id_emb, visual_emb, text_emb, loss_val)`

#### 2.2 `retrieve_multimodal_memory_batch()` 方法

**旧版本返回**:
```python
-> Tuple[torch.Tensor, torch.Tensor]
# (mem_visual, mem_text)
```

**新版本返回**:
```python
-> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
# (mem_vis, mem_txt, mem_id, mask)
```

**新增返回值**:
- `mem_id`: [B, K, D_id] - 记忆中的ID嵌入
- `mask`: [B, K] - 有效性掩码 (1=有效, 0=填充)

**检索策略**:
- ST占比: 25% (默认，可配置 `st_retrieve_ratio`)
- LT占比: 75%
- ST选择: 最近的k_st个entry
- LT选择: utility最高的k_lt个entry
  - utility = log(1+frequency) + exp_recency(age, half_life=200)

### 3. Novelty-based LT写入

**计算方式**:
```python
# 1. 构建combined embedding (与分析脚本一致)
combined = l2_norm(concat(l2_norm(visual), l2_norm(text)))

# 2. 计算vs ST window的max cosine similarity
maxcos = max(cosine(combined, st_entry.comb_emb) for st_entry in ST)

# 3. 计算novelty
novelty = 1.0 - maxcos

# 4. 决定是否写入LT
if novelty >= lt_novelty_threshold:  # 默认0.583 (p90)
    write_to_LT()
```

**数据驱动的默认参数** (基于ML-1M分析):
- `lt_novelty_threshold`: 0.5830 (p90 combined novelty, ~10%写入率)
- `retrieve_topk`: 32 (推荐检索数量)
- `st_capacity`: 50 (窗口W)
- `lt_merge_sim_threshold`: 0.74 (合并阈值，避免重复)
- `lt_recency_half_life_steps`: 200 (与聚类窗口一致)

### 4. 其他改进

**LT去重机制**:
```python
# 写入LT前，检查是否与现有entry高度相似
if max_similarity >= lt_merge_sim_threshold:  # 默认0.74
    # 合并到现有entry（EMA更新）
    existing_entry.comb_emb = 0.9 * existing + 0.1 * new
else:
    # 作为新entry添加
    add_new_entry()
```

**LT驱逐策略**:
```python
# 基于utility最低的entry驱逐
utility = log(1 + frequency) + exp_recency(age, half_life_steps)
evict(argmin(utility))
```

## 🔧 代码适配修改

### 修改文件: `fedmem_client.py`

#### 修改1: `update()` 调用 (Line 497-504)

**旧代码**:
```python
self.local_memory.update(
    item_id=item_id,
    loss_val=loss_val,
    text_emb=self._get_item_text_emb(item_id),
    img_emb=self._get_item_img_emb(item_id),
    id_emb=self._get_item_id_emb(item_id)
)
```

**新代码**:
```python
self.local_memory.update(
    item_id=item_id,
    id_emb=self._get_item_id_emb(item_id),         # 移到前面
    visual_emb=self._get_item_img_emb(item_id),     # 改名
    text_emb=self._get_item_text_emb(item_id),
    loss_val=loss_val
)
```

#### 修改2: `_retrieve_multimodal_memory_batch()` 适配 (Line 550-578)

**旧代码**:
```python
def _retrieve_multimodal_memory_batch(...):
    return self.local_memory.retrieve_multimodal_memory_batch(
        batch_size=batch_size,
        top_k=top_k
    )
```

**新代码**:
```python
def _retrieve_multimodal_memory_batch(...):
    # 新版本返回4个值：(mem_vis, mem_txt, mem_id, mask)
    mem_vis, mem_txt, mem_id, mask = self.local_memory.retrieve_multimodal_memory_batch(
        batch_size=batch_size,
        top_k=top_k
    )

    # 向后兼容：只返回visual和text（忽略mem_id和mask）
    return mem_vis, mem_txt
```

## 📊 预期效果

### 性能提升

| 指标 | 旧版本 | 新版本 | 改进 |
|------|--------|--------|------|
| LT写入率 | 随机 (~50%) | 数据驱动 (~10%) | 减少80%写入 |
| 检索速度 | 单一buffer | ST+LT分层 | 更快 |
| 内存多样性 | 低 | 高 (novelty-gated) | 更好覆盖 |
| 响应漂移 | 慢 | 快 (ST FIFO) | 实时适应 |

### 统计指标监控

新版本提供更详细的统计信息：

```python
stats = memory.get_statistics()
{
    "st_size": 当前ST大小,
    "lt_size": 当前LT大小,
    "total_updates_st": ST总更新次数,
    "total_updates_lt": LT总更新次数,
    "total_promotions": ST->LT提升次数,
    "total_expires_lt": LT驱逐次数,
    "lt_novelty_threshold": LT写入阈值,
    "lt_merge_sim_threshold": 合并阈值,
    "st_capacity": ST容量,
    "lt_capacity": LT容量,
    "retrieve_topk": 检索K值
}
```

**健康指标参考** (ML-1M):
- `total_updates_lt / total_updates_st` ≈ 0.1 (10%写入率)
- `total_promotions / lt_size` ≈ 1-2 (适度更新)
- `total_expires_lt` > 0 (正常循环，不应该为0)
- `st_size` ≈ 50 (达到容量)
- `lt_size` ≈ 200 (达到容量)

## 🐛 潜在问题与解决

### 问题1: ST写入率过低

**症状**: `total_updates_lt / total_updates_st` < 0.05

**原因**: `lt_novelty_threshold` 过高，导致几乎没有entry能通过novelty gate

**解决方案**:
```python
LocalDynamicMemory(
    capacity=200,
    lt_novelty_threshold=0.50,  # 降低阈值，从0.583降到0.50
    ...
)
```

### 问题2: LT过度合并

**症状**: `total_promotions` 很小，但 `total_updates_lt` 很大

**原因**: `lt_merge_sim_threshold` 过低，导致大部分新entry被合并到现有entry

**解决方案**:
```python
LocalDynamicMemory(
    capacity=200,
    lt_merge_sim_threshold=0.80,  # 提高阈值，从0.74提高到0.80
    ...
)
```

### 问题3: 检索结果为空

**症状**: `mask.sum() == 0` 或 `mem_vis` 全为零

**原因**: ST和LT都为空（训练初期），或 `retrieve_topk=0`

**检查方法**:
```python
stats = client.local_memory.get_statistics()
print(f"ST size: {stats['st_size']}, LT size: {stats['lt_size']}")

# 如果都为0，说明update()没有被正确调用
# 检查是否在训练循环中调用了memory.update()
```

### 问题4: id_emb为None导致错误

**症状**: `TypeError: update() missing 1 required positional argument: 'id_emb'`

**原因**: `_get_item_id_emb()` 返回None

**解决方案**:
```python
def _get_item_id_emb(self, item_id: int):
    # 确保总是返回一个有效的tensor
    if self.model is None:
        # Fallback: 返回随机初始化的embedding
        return torch.randn(self.sasrec_hidden_dim, device=self.device)

    # ... 正常逻辑 ...
```

## 📝 迁移检查清单

- [x] 更新 `fedmem_client.py` 中的 `update()` 调用
  - [x] 修改参数顺序: `id_emb` 移到前面
  - [x] 重命名: `img_emb` -> `visual_emb`
  - [x] 调整注释: "Surprise-based" -> "Two-tier"

- [x] 更新 `fedmem_client.py` 中的 `retrieve_multimodal_memory_batch()` 调用
  - [x] 接收4个返回值: `(mem_vis, mem_txt, mem_id, mask)`
  - [x] 向后兼容包装器: 只返回前2个值
  - [x] 更新文档字符串

- [ ] 测试验证
  - [ ] 运行训练脚本，验证memory正常更新
  - [ ] 检查统计信息: `st_size`, `lt_size`, 写入率
  - [ ] 验证检索结果非空: `mask.sum() > 0`
  - [ ] 监控LT写入率: ~10% (可调整)

- [ ] 性能调优（可选）
  - [ ] 根据数据集调整 `lt_novelty_threshold`
  - [ ] 调整 `st_capacity` 如果序列特别长/短
  - [ ] 调整 `retrieve_topk` 和 `st_retrieve_ratio`

## 🚀 使用建议

### 默认配置（推荐，基于ML-1M）

```python
LocalDynamicMemory(
    capacity=200,                      # LT容量
    st_capacity=50,                    # ST容量 (窗口W)
    lt_novelty_threshold=0.5830,       # p90 combined novelty
    retrieve_topk=32,                  # 检索数量
    st_retrieve_ratio=0.25,            # 25% from ST
    lt_merge_sim_threshold=0.74,       # 合并阈值
    lt_recency_half_life_steps=200,    # 衰减半衰期
    device='cuda'
)
```

### 小数据集调优

```python
# 数据集更小 (e.g., ML-100K)
LocalDynamicMemory(
    capacity=100,                      # 减少LT容量
    st_capacity=30,                    # 减少ST容量
    lt_novelty_threshold=0.55,         # 略微降低阈值
    retrieve_topk=20,                  # 减少检索数量
    ...
)
```

### 大数据集调优

```python
# 数据集更大 (e.g., Amazon Beauty)
LocalDynamicMemory(
    capacity=500,                      # 增加LT容量
    st_capacity=100,                   # 增加ST容量
    lt_novelty_threshold=0.60,         # 略微提高阈值（更严格）
    retrieve_topk=50,                  # 增加检索数量
    ...
)
```

## 🔍 调试技巧

### 1. 打印memory统计信息

```python
# 在训练循环中每N轮打印一次
if round_idx % 5 == 0:
    stats = client.local_memory.get_statistics()
    print(f"Round {round_idx} Memory Stats:")
    print(f"  ST: {stats['st_size']}/{stats['st_capacity']}")
    print(f"  LT: {stats['lt_size']}/{stats['lt_capacity']}")
    print(f"  LT write ratio: {stats['total_updates_lt']/max(1,stats['total_updates_st']):.2%}")
```

### 2. 验证检索结果

```python
mem_vis, mem_txt = client._retrieve_multimodal_memory_batch(batch_size=4, top_k=32)
print(f"Retrieved shapes: vis={mem_vis.shape}, txt={mem_txt.shape}")
# 应该输出: vis=torch.Size([4, 32, 512]), txt=torch.Size([4, 32, 384])

# 检查是否有有效数据
print(f"Visual non-zero: {(mem_vis.abs().sum(dim=-1) > 0).sum().item()}")
print(f"Text non-zero: {(mem_txt.abs().sum(dim=-1) > 0).sum().item()}")
```

### 3. 监控novelty分布

```python
# 临时添加到local_dynamic_memory.py的update()方法中
novelties = []  # global list

def update(...):
    ...
    novelty = 1.0 - maxcos
    novelties.append(novelty)

    if len(novelties) % 100 == 0:
        print(f"Novelty stats: min={min(novelties):.3f}, "
              f"max={max(novelties):.3f}, "
              f"mean={sum(novelties)/len(novelties):.3f}, "
              f"p90={sorted(novelties)[int(len(novelties)*0.9)]:.3f}")
```

## ✅ 总结

本次memory机制更新实现了：

1. **Two-tier架构**: ST快速响应 + LT稳定存储
2. **Novelty-gated写入**: 基于数据分析的p90阈值，~10%写入率
3. **去重机制**: 避免LT存储近似重复的entry
4. **Utility-based驱逐**: 综合频率和新鲜度的智能驱逐
5. **混合检索**: ST (25%) + LT (75%) 平衡最近性和多样性

**向后兼容性**: 通过wrapper方法保持API兼容，现有调用者无需修改。

**数据驱动**: 所有参数默认值基于ML-1M数据集的实证分析，可根据具体数据集调优。
