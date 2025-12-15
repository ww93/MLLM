# 数据准备总结

## ✅ 完成状态

所有训练所需的数据文件已成功生成！

---

## 📁 生成的文件

### 1. 序列数据 (UR4Rec/data/Multimodal_Datasets/)

```
✓ train_sequences.npy (831 KB)
  - 训练用户序列
  - 938 个用户
  - 平均序列长度: 46.5

✓ val_sequences.npy (932 KB)
  - 验证用户序列
  - 938 个用户

✓ test_sequences.npy (1.0 MB)
  - 测试用户序列
  - 938 个用户

✓ item_map.json (24 KB)
  - 物品ID映射
  - 1659 个物品
```

### 2. LLM 生成数据 (data/llm_generated/)

```
✓ user_preferences.json (682 KB)
  - 938 个用户偏好描述
  - 100% 覆盖率
  - 英文描述

✓ item_descriptions.json (466 KB)
  - 1659 个物品描述
  - 完整覆盖
  - 英文描述
```

### 3. 配置文件

```
✓ UR4Rec/configs/ur4rec_moe_100k.yaml
  - 完整训练配置
  - MoE + Memory 参数
  - 30 epochs per stage
```

---

## 📊 数据统计

### 原始数据 (M_ML-100K)
- **总评分**: 99,309 条
- **用户数**: 943
- **物品数**: 1,659
- **评分范围**: 1.0 - 5.0

### 过滤后数据 (评分 >= 4.0)
- **高评分**: 55,024 条 (55.4%)
- **用户数**: 942
- **活跃物品**: 1,428
- **有效用户**: 938 (序列长度 >= 5)

### 序列统计
- **平均序列长度**: 58.6
- **最短序列**: 5 个物品
- **最长序列**: 376 个物品
- **训练序列长度**: 46.5 (平均)

### 数据划分
- **训练集**: 80% (前 80% 的交互)
- **验证集**: 10% (前 90% 的交互)
- **测试集**: 10% (全部交互)

---

## 🔧 数据处理流程

### 执行的脚本

```bash
# 1. 数据预处理
python UR4Rec/scripts/prepare_ml100k_data.py

# 处理步骤:
# - 加载 movies.dat (1,659 部电影)
# - 加载 ratings.dat (99,309 条评分)
# - 过滤高评分 (>=4.0, 作为正样本)
# - 构建用户序列 (按时间排序)
# - 过滤短序列 (min_seq_len=5)
# - 划分 train/val/test (80/10/10)
# - 保存为 .npy 和 .json
```

### LLM 数据生成（已完成）

```bash
# 2. LLM 偏好生成 (已完成)
export DASHSCOPE_API_KEY="your-key"
python UR4Rec/models/llm_generator.py

# 生成内容:
# - 938 个用户偏好描述 (英文)
# - 1,659 个物品描述 (英文)
# - 自动缓存和错误处理
```

---

## 🚀 开始训练

### 方法 1: 完整训练（推荐）

```bash
source venv/bin/activate

python UR4Rec/scripts/train_ur4rec_moe.py \
    --config UR4Rec/configs/ur4rec_moe_100k.yaml \
    --data_dir UR4Rec/data/Multimodal_Datasets \
    --llm_data_dir data/llm_generated \
    --output_dir outputs/ur4rec_moe \
    --epochs_per_stage 30 \
    --patience 5
```

**训练阶段**:
1. Pretrain SASRec (30 epochs)
2. Pretrain Retriever with MoE + Memory (30 epochs)
3. Joint Finetune with Adaptive Alternating (30 epochs)
4. End-to-End Training (30 epochs)

**预计时间**: 根据硬件，可能需要数小时

### 方法 2: 快速测试

```bash
# 减少 epochs 进行快速测试
python UR4Rec/scripts/train_ur4rec_moe.py \
    --config UR4Rec/configs/ur4rec_moe_100k.yaml \
    --data_dir UR4Rec/data/Multimodal_Datasets \
    --llm_data_dir data/llm_generated \
    --output_dir outputs/ur4rec_moe_test \
    --epochs_per_stage 2 \
    --patience 1
```

---

## 📈 训练输出

训练完成后，将生成：

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
├── results.json                   # 训练结果
└── logs/                          # TensorBoard 日志
```

---

## 🔍 验证数据

随时可以运行验证脚本：

```bash
python verify_data.py
```

输出示例：
```
============================================================
Data Verification for UR4Rec MoE Training
============================================================

1. Checking sequence data...
  ✓ Train sequences: UR4Rec/data/Multimodal_Datasets/train_sequences.npy (831.1 KB)
  ✓ Val sequences: UR4Rec/data/Multimodal_Datasets/val_sequences.npy (932.4 KB)
  ✓ Test sequences: UR4Rec/data/Multimodal_Datasets/test_sequences.npy (1042.2 KB)
  ✓ Item mapping: UR4Rec/data/Multimodal_Datasets/item_map.json (23.8 KB)

2. Checking LLM generated data...
  ✓ User preferences: data/llm_generated/user_preferences.json (682.4 KB)
  ✓ Item descriptions: data/llm_generated/item_descriptions.json (465.5 KB)

3. Checking configuration...
  ✓ Config file: UR4Rec/configs/ur4rec_moe_100k.yaml (1.3 KB)

4. Verifying data content...
  ✓ Train users: 938
  ✓ Val users: 938
  ✓ Test users: 938
  ✓ Avg train seq length: 46.5
  ✓ Total items: 1659
  ✓ User preferences: 938
  ✓ Item descriptions: 1659
  ✓ LLM user coverage: 100.0%

============================================================
✅ All data files are ready!
============================================================
```

---

## 📝 数据格式说明

### train_sequences.npy 格式

```python
{
    user_id (int): [item_id1, item_id2, ..., item_idn]  # 按时间排序
}
```

**示例**:
```python
{
    1: [242, 302, 377, ...],  # 用户 1 的序列
    2: [51, 346, 89, ...],    # 用户 2 的序列
    ...
}
```

### user_preferences.json 格式

```json
{
    "user_id": "User preference description in English"
}
```

**示例**:
```json
{
    "1": "This user prefers action and adventure movies...",
    "2": "This user enjoys romantic comedies and dramas..."
}
```

### item_descriptions.json 格式

```json
{
    "item_id": "Item description in English"
}
```

**示例**:
```json
{
    "1": "Toy Story is a groundbreaking animated film...",
    "2": "GoldenEye is an action-packed thriller..."
}
```

---

## 🎯 关键配置参数

### Memory 参数 (已调整)

```yaml
max_memory_size: 20           # 记忆历史状态数量
interaction_threshold: 20     # 每 20 次交互更新记忆
update_trigger: "INTERACTION_COUNT"
```

### 训练参数 (已调整)

```yaml
epochs_per_stage: 30          # 每阶段 30 个 epoch
patience: 5                   # 早停耐心值
batch_size: 32                # 批次大小
```

### MoE 参数

```yaml
moe_num_heads: 8             # MoE 注意力头数
moe_num_proxies: 4           # 代理 token 数量
```

---

## 📚 相关文档

- [UR4REC_MOE_GUIDE.md](UR4REC_MOE_GUIDE.md) - 完整使用指南
- [MERGE_SUMMARY.md](MERGE_SUMMARY.md) - 模型合并说明
- [DATA_LOADING_GUIDE.md](DATA_LOADING_GUIDE.md) - 数据加载详解
- [LLM_PROMPTS.md](LLM_PROMPTS.md) - LLM Prompt 说明

---

## ✅ 检查清单

在开始训练前，确保：

- [x] 序列数据已生成 (train/val/test_sequences.npy)
- [x] 物品映射已生成 (item_map.json)
- [x] LLM 用户偏好已生成 (user_preferences.json)
- [x] LLM 物品描述已生成 (item_descriptions.json)
- [x] 配置文件已准备 (ur4rec_moe_100k.yaml)
- [x] 虚拟环境已激活
- [x] 必要的包已安装 (torch, sentence-transformers, etc.)

---

## 🎉 总结

**所有数据准备工作已完成！**

- ✅ **938 个用户序列** (训练/验证/测试)
- ✅ **1,659 个物品** (完整映射)
- ✅ **938 个 LLM 用户偏好** (100% 覆盖)
- ✅ **1,659 个 LLM 物品描述** (完整覆盖)
- ✅ **配置文件** (MoE + Memory)

**现在可以开始训练 UR4Rec MoE 模型了！** 🚀

---

*生成时间: 2025-12-10*
*数据集: MovieLens-100K*
*模型: UR4Rec V2 with MoE*
