# FedMem: 基于LLM的联邦推荐系统（带本地动态多模态记忆）

> 本项目实现了一个创新的联邦学习推荐系统，结合了本地动态记忆机制和多模态表示学习。

## 架构概览

FedMem是标准联邦推荐系统的升级版本，核心创新包括：

### 1. **本地动态记忆 (Local Dynamic Memory)**
   - **Surprise机制**：客户端维护本地记忆缓冲区，基于"惊喜"（预测误差）机制动态更新
   - **多模态支持**：支持文本/图像/ID三种模态的嵌入表示
   - **效用驱动过期**：基于 `utility = α * recency + β * frequency` 的智能过期机制
   - **隐私保护**：记忆存储在本地，不直接上传原始数据

### 2. **多模态MoE (Multimodal Mixture-of-Experts)**
   - **场景感知路由器**：根据目标物品类别/嵌入动态分配专家权重
   - **三个专家模块**：
     - **视觉专家**：处理图像特征（CLIP嵌入）
     - **文本专家**：处理LLM生成的文本偏好
     - **序列专家**：处理ID序列模式（SASRec）
   - **自适应融合**：动态权重分配，适应不同场景

### 3. **原型聚合 (Prototype Aggregation)**
   - **K-Means聚类**：将客户端记忆聚类为K个原型中心点
   - **全局抽象记忆**：服务器聚合原型构建全局知识
   - **知识蒸馏**：下发给客户端辅助本地推荐
   - **隐私友好**：仅传输聚类中心，不泄露原始交互数据

## 核心组件

### 1. LocalDynamicMemory (`UR4Rec/models/local_dynamic_memory.py`)

**功能**：
- 存储重要交互记忆
- Surprise机制：`loss > threshold` → 加入记忆
- Expire机制：`utility = α * recency + β * frequency`
- K-Means聚类提取原型

**关键方法**：
```python
# 查询记忆
memory.query(target_item, k=5)

# 更新记忆（基于Surprise）
memory.update(
    item_id=item_id,
    loss_val=loss_val,  # Surprise指标
    text_emb=text_emb,
    img_emb=img_emb,
    id_emb=id_emb
)

# 提取原型
prototypes = memory.get_memory_prototypes(k=5)
```

### 2. FedMemClient (`UR4Rec/models/fedmem_client.py`)

**功能**：
- 本地训练集成记忆查询
- Surprise-based记忆更新
- 对比学习损失（对齐用户偏好与物品）
- 上传模型参数 + 记忆原型

**训练流程**：
```python
# 1. 前向传播
retrieved_memory = query_memory_batch(target_items)
scores = model(user_ids, input_seq, target_items, retrieved_memory)

# 2. 计算损失
rec_loss = BPR_loss(pos_scores, neg_scores)
contrastive_loss = align_user_preference_with_item(user_ids, target_items)
total_loss = rec_loss + λ * contrastive_loss

# 3. 反向传播
optimizer.step()

# 4. Surprise-based记忆更新
for each sample:
    if sample_loss > threshold:
        memory.update(item_id, sample_loss, embeddings)
```

**关键方法**：
```python
# 训练
metrics = client.train_local_model(verbose=True)
# 返回: {'loss', 'rec_loss', 'contrastive_loss', 'memory_size', 'memory_updates'}

# 提取原型
prototypes = client.get_memory_prototypes()  # [K, emb_dim]

# 接收全局记忆
client.set_global_abstract_memory(global_prototypes)
```

### 3. FedMemServer (待完整实现)

**功能**：
- 聚合模型参数（FedAvg）
- 聚合记忆原型（平均）
- 下发全局模型 + 全局抽象记忆

**Prototype聚合逻辑**：
```python
def aggregate_prototypes(client_prototypes: List[torch.Tensor]):
    """
    聚合客户端原型 → 全局抽象记忆

    Args:
        client_prototypes: List of [K, emb_dim] tensors

    Returns:
        global_prototypes: [K, emb_dim]
    """
    # 简单平均
    global_prototypes = torch.stack(client_prototypes).mean(dim=0)
    return global_prototypes
```

## 训练流程

### FedMem联邦学习完整流程

```
初始化：
- 服务器创建全局UR4RecV2MoE模型
- 为每个用户创建FedMemClient（含本地记忆）

每轮训练（Round r）：
┌───────────────────────────────────────────────────────┐
│ 1. 服务器选择客户端（client_fraction = 10%）          │
│ 2. 下发全局模型参数 + 全局抽象记忆                      │
├───────────────────────────────────────────────────────┤
│ 3. 客户端本地训练：                                     │
│    For each batch:                                     │
│      - 查询本地记忆                                     │
│      - 前向传播（注入记忆）                             │
│      - 计算rec_loss + contrastive_loss                 │
│      - 反向传播，更新模型                               │
│      - 基于Surprise更新本地记忆                         │
├───────────────────────────────────────────────────────┤
│ 4. 客户端提取：                                         │
│    - 模型参数（state_dict）                            │
│    - 记忆原型（K-Means中心点）                          │
│ 5. 上传到服务器                                         │
├───────────────────────────────────────────────────────┤
│ 6. 服务器聚合：                                         │
│    - FedAvg聚合模型参数                                │
│    - 平均聚合记忆原型 → 全局抽象记忆                    │
│ 7. 更新全局模型                                         │
├───────────────────────────────────────────────────────┤
│ 8. 验证集评估                                           │
│ 9. Early stopping判断                                  │
└───────────────────────────────────────────────────────┘

最终测试：
- 在测试集上评估全局模型
- 输出：HR@K, NDCG@K, MRR
```

## 关键参数配置

```yaml
# FedMem配置示例
fedmem:
  # 联邦学习参数
  num_rounds: 50
  client_fraction: 0.1
  local_epochs: 1
  federated_lr: 0.001

  # 记忆参数
  memory_capacity: 50              # 记忆容量
  surprise_threshold: 0.5          # 惊喜阈值
  recency_weight: 0.6              # 近期性权重
  frequency_weight: 0.4            # 频率权重

  # MoE参数
  num_memory_prototypes: 5         # 原型数量
  contrastive_lambda: 0.1          # 对比学习权重

  # 模型参数
  sasrec_hidden_dim: 512
  retriever_output_dim: 512
  moe_num_heads: 8
  fusion_method: 'weighted'
```

## 实验指标

### 对比实验

1. **集中式SASRec（Baseline）**
   - HR@10: 0.40-0.41
   - NDCG@10: ~0.25

2. **联邦SASRec（无记忆）**
   - HR@10: 目标≥0.35

3. **FedMem（完整系统）**
   - HR@10: 目标≥0.40
   - NDCG@10: 目标≥0.25
   - 记忆效用指标：
     - 平均记忆大小
     - 记忆更新频率
     - 原型覆盖度

### 关键创新点

1. **Surprise机制**：
   - 自适应记忆更新
   - 只记忆难以预测的item
   - 提高记忆效率

2. **Prototype聚合**：
   - 隐私保护（不传输原始数据）
   - 知识蒸馏（全局抽象记忆）
   - 辅助个性化推荐

3. **多模态融合**：
   - 文本（LLM生成）
   - 图像（CLIP特征）
   - 序列（ID嵌入）

## 代码结构

```
UR4Rec/
├── models/
│   ├── local_dynamic_memory.py        # ✅ 本地动态记忆
│   ├── fedmem_client.py               # ✅ FedMem客户端
│   ├── fedmem_server.py               # ✅ FedMem服务器
│   ├── ur4rec_v2_moe.py               # ✅ UR4Rec MoE模型（已增强）
│   ├── federated_aggregator.py        # 联邦聚合器
│   ├── sasrec.py                      # SASRec序列模型
│   └── text_preference_retriever_moe.py  # 文本偏好检索器
├── scripts/
│   ├── train_fedmem.py                # ✅ FedMem训练脚本
│   ├── train_federated_ur4rec_moe.py  # 标准联邦UR4Rec训练
│   └── train_sasrec_centralized.py    # 集中式SASRec基线
└── utils/
    ├── data_loader.py                 # 数据加载工具
    └── metrics.py                     # 评估指标
```

## 快速开始

### 安装依赖

```bash
cd UR4Rec
pip install -r requirements.txt
```

### 准备数据

```bash
# 下载MovieLens-1M数据集
wget http://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip ml-1m.zip -d data/

# 预处理数据
python scripts/preprocess_movielens.py --input data/ml-1m/ratings.dat --output data/ml-1m/
```

### 训练FedMem模型

```bash
# 基础训练（使用默认参数）
python scripts/train_fedmem.py \
    --data_dir data/ml-1m \
    --data_file ratings.dat \
    --save_dir checkpoints/fedmem \
    --enable_prototype_aggregation \
    --verbose

# 完整配置训练
python scripts/train_fedmem.py \
    --data_dir data/ml-1m \
    --data_file ratings.dat \
    --save_dir checkpoints/fedmem_full \
    --num_rounds 50 \
    --client_fraction 0.1 \
    --local_epochs 1 \
    --memory_capacity 50 \
    --surprise_threshold 0.5 \
    --contrastive_lambda 0.1 \
    --num_memory_prototypes 5 \
    --enable_prototype_aggregation \
    --learning_rate 0.001 \
    --batch_size 32 \
    --device cuda \
    --verbose
```

### 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num_rounds` | 50 | 联邦学习轮数 |
| `--client_fraction` | 0.1 | 每轮参与的客户端比例 |
| `--memory_capacity` | 50 | 本地记忆容量 |
| `--surprise_threshold` | 0.5 | Surprise阈值（超过此值才加入记忆） |
| `--contrastive_lambda` | 0.1 | 对比学习损失权重 |
| `--num_memory_prototypes` | 5 | 记忆原型数量（K-Means聚类中心数） |
| `--enable_prototype_aggregation` | False | 是否启用原型聚合 |

## 开发完成状态

### ✅ 已完成：

1. **LocalDynamicMemory** (`local_dynamic_memory.py`) ✅
   - [x] Surprise-based记忆更新机制
   - [x] 效用驱动的过期机制
   - [x] K-Means原型提取
   - [x] 多模态嵌入支持

2. **FedMemClient** (`fedmem_client.py`) ✅
   - [x] 集成LocalDynamicMemory
   - [x] Surprise-based记忆更新逻辑
   - [x] 对比学习损失计算
   - [x] 原型提取与上传

3. **FedMemServer** (`fedmem_server.py`) ✅
   - [x] 原型聚合（aggregate_prototypes）
   - [x] 全局抽象记忆分发（distribute_global_abstract_memory）
   - [x] FedAvg参数聚合
   - [x] 训练循环与评估

4. **UR4RecV2MoE增强** (`ur4rec_v2_moe.py`) ✅
   - [x] retrieved_memory参数支持
   - [x] compute_contrastive_loss方法
   - [x] get_item_embeddings方法

5. **训练脚本** (`train_fedmem.py`) ✅
   - [x] 完整的数据加载流程
   - [x] 客户端和服务器创建
   - [x] 训练循环
   - [x] 结果保存与可视化

### 📋 可选增强：

1. **数据增强**
   - [ ] 多数据集支持（Amazon, Yelp等）
   - [ ] 数据划分策略（IID vs Non-IID）

2. **模型优化**
   - [ ] 场景感知路由器（根据物品类别动态路由）
   - [ ] 更复杂的记忆检索机制
   - [ ] 视觉特征集成（CLIP图像嵌入）

3. **实验分析**
   - [ ] 对比实验脚本（Baseline vs FedMem）
   - [ ] 消融实验（记忆/原型/对比学习独立测试）
   - [ ] 可视化工具（记忆演化、原型分布等）

## 使用示例

```python
from UR4Rec.models.fedmem_client import FedMemClient
from UR4Rec.models.local_dynamic_memory import LocalDynamicMemory
from UR4Rec.models.ur4rec_v2_moe import UR4RecV2MoE

# 创建全局模型
global_model = UR4RecV2MoE(...)

# 创建FedMem客户端
client = FedMemClient(
    client_id=user_id,
    model=global_model,
    user_sequence=[1, 2, 3, 4, 5],
    memory_capacity=50,
    surprise_threshold=0.5
)

# 本地训练
metrics = client.train_local_model(verbose=True)
print(f"Loss: {metrics['loss']:.4f}, Memory: {metrics['memory_size']}")

# 提取原型
prototypes = client.get_memory_prototypes()  # [K, emb_dim]

# 评估
eval_metrics = client.evaluate(split='test')
print(f"HR@10: {eval_metrics['HR@10']:.4f}")
```

## 参考文献

本实现基于以下研究思路：

1. **Federated Learning**: McMahan et al. "Communication-Efficient Learning of Deep Networks from Decentralized Data"
2. **Dynamic Memory Networks**: Kumar et al. "Ask Me Anything: Dynamic Memory Networks for NLP"
3. **Surprise-based Learning**: Achille et al. "Information Dropout: Learning Optimal Representations Through Noisy Computation"
4. **Prototype Learning**: Snell et al. "Prototypical Networks for Few-shot Learning"

## License

本项目遵循MIT License。

## 联系方式

如有问题，请提交Issue或联系开发团队。
