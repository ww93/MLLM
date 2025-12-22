# FedMem 实现详解

> 本文档详细说明FedMem系统的实现细节、关键算法和代码结构

## 目录

1. [系统架构](#系统架构)
2. [核心算法](#核心算法)
3. [代码实现](#代码实现)
4. [训练流程](#训练流程)
5. [关键创新点](#关键创新点)

---

## 系统架构

### 总体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                      FedMem Server                          │
│  ┌────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │  Global Model  │  │ Global Abstract │  │  Aggregator  │ │
│  │ (UR4RecV2MoE)  │  │     Memory      │  │   (FedAvg)   │ │
│  └────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
           │                    │                    │
           ├────────────────────┼────────────────────┤
           ↓                    ↓                    ↓
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  FedMem Client 1 │  │  FedMem Client 2 │  │  FedMem Client N │
│ ┌──────────────┐ │  │ ┌──────────────┐ │  │ ┌──────────────┐ │
│ │ Local Model  │ │  │ │ Local Model  │ │  │ │ Local Model  │ │
│ └──────────────┘ │  │ └──────────────┘ │  │ └──────────────┘ │
│ ┌──────────────┐ │  │ ┌──────────────┐ │  │ ┌──────────────┐ │
│ │Local Dynamic │ │  │ │Local Dynamic │ │  │ │Local Dynamic │ │
│ │   Memory     │ │  │ │   Memory     │ │  │ │   Memory     │ │
│ └──────────────┘ │  │ └──────────────┘ │  │ └──────────────┘ │
│  User Data: u1   │  │  User Data: u2   │  │  User Data: uN   │
└──────────────────┘  └──────────────────┘  └──────────────────┘
```

### 模型架构

```
UR4RecV2MoE (全局模型)
│
├── SASRec (序列模型)
│   ├── Item Embedding
│   ├── Position Embedding
│   └── Transformer Blocks (Self-Attention)
│
├── Text Preference Retriever MoE
│   ├── Text Encoder (Sentence-BERT)
│   ├── MoE Layer
│   │   ├── Expert 1 (Text)
│   │   ├── Expert 2 (Visual)
│   │   └── Expert 3 (Sequential)
│   └── Attention Fusion
│
└── Fusion Module
    ├── Weighted Fusion
    ├── Learned Fusion
    └── Adaptive Fusion
```

---

## 核心算法

### 1. Surprise-based Memory Update

```python
# 伪代码
def update_memory(item_id, loss_val, embeddings):
    """
    基于Surprise机制更新本地记忆

    核心思想：只有当模型对某个item的预测误差较大时，
    才将其加入记忆，这样记忆存储的是"难以预测"的item
    """
    # Step 1: Surprise判断
    if loss_val < surprise_threshold:
        return  # 如果损失小于阈值，不记忆

    # Step 2: 检查是否已在记忆中
    if item_id in memory_buffer:
        # 更新现有记忆条目
        memory_buffer[item_id].update({
            'timestamp': current_time,
            'frequency': frequency + 1,
            'surprise': max(old_surprise, loss_val),
            'embeddings': embeddings
        })
    else:
        # Step 3: 如果记忆已满，执行过期逻辑
        if len(memory_buffer) >= capacity:
            # 计算所有条目的效用分数
            utilities = {}
            for item, mem in memory_buffer.items():
                recency = exp(-time_diff / half_life)
                frequency_score = log(1 + mem['frequency'])
                utility = α * recency + β * frequency_score
                utilities[item] = utility

            # 移除效用最低的条目
            least_useful = min(utilities, key=utilities.get)
            del memory_buffer[least_useful]

        # Step 4: 添加新记忆条目
        memory_buffer[item_id] = {
            'text_emb': embeddings.text,
            'img_emb': embeddings.image,
            'id_emb': embeddings.id,
            'timestamp': current_time,
            'frequency': 1,
            'surprise': loss_val
        }
```

**数学公式**：

效用函数：
```
Utility(item) = α × Recency(item) + β × Frequency(item)

Recency(item) = exp(-Δt / τ)  # 指数衰减，τ为半衰期

Frequency(item) = log(1 + freq) / log(1 + freq + 10)  # 归一化频率
```

### 2. Prototype Aggregation

```python
# 伪代码
def aggregate_prototypes(client_prototypes):
    """
    聚合客户端记忆原型 → 全局抽象记忆

    输入: List[Tensor[K, D]]  # 每个客户端的K个原型
    输出: Tensor[K, D]         # 全局K个原型
    """
    # 方法1: 简单平均（隐私友好）
    global_prototypes = mean(client_prototypes, dim=0)

    # 方法2: 加权平均（根据客户端数据量）
    weighted_prototypes = sum(w_i * P_i) / sum(w_i)

    # 方法3: 重新聚类（更精确但计算量大）
    all_prototypes = concat(client_prototypes)  # [N*K, D]
    global_prototypes = kmeans(all_prototypes, K)

    return global_prototypes
```

**K-Means聚类原型提取**：

```python
def extract_prototypes(memory_buffer, K):
    """
    从本地记忆中提取K个原型（中心点）
    """
    # Step 1: 收集所有嵌入
    embeddings = [mem['text_emb'] for mem in memory_buffer.values()]
    X = stack(embeddings)  # [N, D]

    # Step 2: K-Means聚类
    centroids = random_init(K, D)
    for iter in range(max_iters):
        # 分配到最近的中心
        distances = cdist(X, centroids)  # [N, K]
        assignments = argmin(distances, dim=1)  # [N]

        # 更新中心
        for k in range(K):
            mask = (assignments == k)
            if mask.sum() > 0:
                centroids[k] = X[mask].mean(dim=0)

        # 检查收敛
        if converged:
            break

    return centroids  # [K, D]
```

### 3. Contrastive Loss

```python
def compute_contrastive_loss(user_ids, pos_items, neg_items, τ):
    """
    InfoNCE对比学习损失

    目标：拉近用户偏好与正样本item，推远负样本
    """
    # Step 1: 获取嵌入
    user_embeds = get_user_preference_embeddings(user_ids)  # [B, D]
    pos_embeds = get_item_embeddings(pos_items)  # [B, D]
    neg_embeds = get_item_embeddings(neg_items)  # [B, N, D]

    # Step 2: 归一化
    user_embeds = normalize(user_embeds, p=2, dim=1)
    pos_embeds = normalize(pos_embeds, p=2, dim=1)
    neg_embeds = normalize(neg_embeds, p=2, dim=2)

    # Step 3: 计算相似度
    pos_sim = sum(user_embeds * pos_embeds, dim=1) / τ  # [B]
    neg_sim = bmm(neg_embeds, user_embeds.unsqueeze(2)).squeeze(2) / τ  # [B, N]

    # Step 4: InfoNCE损失
    logits = cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # [B, 1+N]
    labels = zeros(B)  # 正样本在第0位
    loss = cross_entropy(logits, labels)

    return loss
```

**数学公式**：

InfoNCE损失：
```
L_contrastive = -log(
    exp(sim(u, i⁺) / τ) /
    (exp(sim(u, i⁺) / τ) + Σ_j exp(sim(u, i⁻_j) / τ))
)

其中：
- u: 用户偏好嵌入（文本）
- i⁺: 正样本item嵌入（ID/图像）
- i⁻: 负样本item嵌入
- τ: 温度参数
- sim(·,·): 余弦相似度
```

---

## 代码实现

### 关键类关系图

```
FederatedAggregator
       ↑
       │ uses
       │
FedMemServer ──────┐
   │               │
   │ manages       │ contains
   │               │
   ↓               ↓
FedMemClient ←── LocalDynamicMemory
   │
   │ trains
   │
   ↓
UR4RecV2MoE
   │
   ├── SASRec
   └── TextPreferenceRetrieverMoE
```

### 核心方法调用流程

#### 训练循环

```python
# train_fedmem.py
def main():
    # 1. 初始化
    global_model = UR4RecV2MoE(...)
    clients = [FedMemClient(...) for user in users]
    server = FedMemServer(global_model, clients, ...)

    # 2. 训练循环
    for round in range(num_rounds):
        # 2.1 选择客户端
        selected = server.select_clients()

        # 2.2 下发全局模型和全局抽象记忆
        server.distribute_model(selected)
        server.distribute_global_abstract_memory(selected)

        # 2.3 客户端本地训练
        for client in selected:
            client.train_local_model()
            # 内部执行：
            #   - 前向传播
            #   - 计算rec_loss + contrastive_loss
            #   - 反向传播
            #   - Surprise-based记忆更新

        # 2.4 收集模型参数和记忆原型
        params = [c.get_model_parameters() for c in selected]
        prototypes = [c.get_memory_prototypes() for c in selected]

        # 2.5 聚合
        global_params = server.aggregate_parameters(params)
        global_memory = server.aggregate_prototypes(prototypes)

        # 2.6 更新全局模型
        server.update_global_model(global_params, global_memory)

        # 2.7 评估
        val_metrics = server.evaluate_global_model(split='val')
```

#### Client训练步骤

```python
# FedMemClient.train_local_model()
def train_local_model(self):
    for epoch in range(local_epochs):
        for batch in train_loader:
            # 1. 数据准备
            user_ids, item_seqs, target_items = batch
            neg_items = negative_sampling(target_items)

            # 2. 记忆查询（可选）
            retrieved_memory = memory.query(target_items)

            # 3. 前向传播
            scores = model(
                user_ids, item_seqs,
                cat([target_items, neg_items]),
                retrieved_memory=retrieved_memory
            )

            # 4. 计算损失
            rec_loss = BPR_loss(scores[:, 0], scores[:, 1:])
            contrastive_loss = model.compute_contrastive_loss(
                user_ids, target_items
            )
            total_loss = rec_loss + λ * contrastive_loss

            # 5. 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # 6. Surprise-based记忆更新
            for i, item in enumerate(target_items):
                loss_val = compute_sample_loss(scores[i])
                if loss_val > surprise_threshold:
                    embeddings = extract_embeddings(item)
                    memory.update(item, loss_val, embeddings)
```

---

## 训练流程

### 完整训练流程图

```
┌─────────────────────────────────────────────────────────┐
│ 初始化阶段                                                │
├─────────────────────────────────────────────────────────┤
│ 1. 加载数据 (user_sequences)                             │
│ 2. 创建全局模型 (UR4RecV2MoE)                            │
│ 3. 创建客户端 (FedMemClient × N)                         │
│    - 每个客户端初始化LocalDynamicMemory                   │
│ 4. 创建服务器 (FedMemServer)                             │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ Round r (r = 1, 2, ..., R)                              │
├─────────────────────────────────────────────────────────┤
│ 【服务器端】                                              │
│ 1. 选择客户端 (client_fraction = 10%)                    │
│ 2. 下发：                                                 │
│    - 全局模型参数 (θ_global)                             │
│    - 全局抽象记忆 (M_global) [如果启用]                  │
├─────────────────────────────────────────────────────────┤
│ 【客户端并行训练】                                         │
│ For each selected client c:                             │
│   3. 加载全局模型和记忆                                   │
│   4. 本地训练 (local_epochs轮):                          │
│      For each batch:                                     │
│        a. 前向传播 (注入retrieved_memory)                │
│        b. 计算total_loss = rec_loss + λ*contrastive_loss│
│        c. 反向传播，更新本地模型                          │
│        d. Surprise更新本地记忆                           │
│   5. 提取：                                               │
│      - 本地模型参数 (θ_c)                                │
│      - 记忆原型 (P_c = KMeans(M_c, K))                   │
│   6. 上传到服务器                                         │
│   7. 释放本地模型内存                                     │
├─────────────────────────────────────────────────────────┤
│ 【服务器聚合】                                            │
│ 8. FedAvg聚合模型:                                       │
│    θ_global = Σ(w_c * θ_c) / Σw_c                       │
│ 9. 聚合原型 [FedMem核心]:                                │
│    M_global = mean([P_1, P_2, ..., P_m])                │
│ 10. 更新全局模型和记忆                                    │
├─────────────────────────────────────────────────────────┤
│ 【评估】                                                  │
│ 11. 在验证集上评估全局模型                                │
│ 12. 计算指标 (HR@K, NDCG@K, MRR)                        │
│ 13. Early stopping检查                                   │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ 最终评估                                                  │
├─────────────────────────────────────────────────────────┤
│ 14. 恢复最佳模型 (基于验证集HR@10)                       │
│ 15. 在测试集上最终评估                                    │
│ 16. 保存模型、训练历史、全局记忆                          │
└─────────────────────────────────────────────────────────┘
```

### 数据流图

```
用户交互序列 (user_sequences)
    │
    ├─→ Leave-one-out划分
    │   ├─→ 训练集: [:-2]
    │   ├─→ 验证集: [-2]
    │   └─→ 测试集: [-1]
    │
    └─→ 创建ClientDataset
        │
        └─→ 滑动窗口生成训练样本
            │
            ├─→ Input: [item_1, ..., item_t]
            └─→ Target: item_{t+1}
```

---

## 关键创新点

### 1. Surprise机制的优势

**传统方法**：
- 记录所有交互或随机采样
- 问题：记忆利用率低，噪声多

**FedMem方法**：
```python
if loss_val > surprise_threshold:
    memory.add(item)  # 只记忆"惊喜"的item
```

**优势**：
- ✅ **自适应性**：自动聚焦于难以预测的item
- ✅ **效率**：记忆容量有限时，优先存储重要信息
- ✅ **泛化**：记忆的是"特殊情况"，帮助模型处理长尾item

### 2. 原型聚合的隐私保护

**直接聚合记忆的问题**：
```
Client → Server: 原始记忆 {item_1, item_2, ...}
❌ 问题：泄露用户交互历史
```

**FedMem原型聚合**：
```
Client: M_c (原始记忆50个item)
    ↓ K-Means (K=5)
Client → Server: P_c (5个聚类中心点)
    ↓ 聚合
Server: M_global (全局5个抽象原型)
```

**优势**：
- ✅ **隐私友好**：仅传输聚类中心，不泄露具体item
- ✅ **通信高效**：50 items → 5 prototypes (10×压缩)
- ✅ **知识蒸馏**：抽象化记忆，提取共性知识

### 3. 多模态对比学习

**目标**：对齐不同模态的表示

```
User Preference (Text) ←─ align ─→ Item (ID/Image)
        ↑                              ↑
    LLM生成                      CLIP/Embedding
```

**效果**：
- ✅ 增强跨模态表示学习
- ✅ 改善冷启动问题（新item有文本描述）
- ✅ 提高推荐多样性

### 4. 完整损失函数

```python
Total_Loss = L_rec + λ * L_contrastive

L_rec = BPR_loss = -log σ(score_pos - score_neg)

L_contrastive = InfoNCE(user_pref, item_pos, items_neg)

其中:
- L_rec: 推荐损失（排序损失）
- L_contrastive: 对比学习损失（对齐损失）
- λ: 权重系数（默认0.1）
```

---

## 性能分析

### 计算复杂度

| 组件 | 复杂度 | 说明 |
|------|--------|------|
| SASRec前向 | O(L·D²) | L=序列长度, D=隐藏维度 |
| Retriever前向 | O(N·D²) | N=候选item数 |
| 记忆查询 | O(M·D) | M=记忆大小 |
| K-Means聚类 | O(M·K·I·D) | I=迭代次数 |
| 原型聚合 | O(C·K·D) | C=客户端数 |

### 通信开销

**标准FedAvg**：
```
每轮通信: 2 × |θ| (下发 + 上传模型参数)
```

**FedMem**：
```
每轮通信: 2 × |θ| + K × D (模型参数 + 原型)

额外开销: K × D ≈ 5 × 256 = 1280维
相比模型参数(百万级): 可忽略不计
```

---

## 实验建议

### 消融实验设计

1. **Baseline**：集中式SASRec
2. **FedAvg-SASRec**：标准联邦SASRec（无记忆）
3. **FedMem-NoProto**：有记忆但不聚合原型
4. **FedMem-NoContrast**：无对比学习损失
5. **FedMem-Full**：完整FedMem系统

### 超参数调优

关键超参数：
- `memory_capacity`: [20, 50, 100]
- `surprise_threshold`: [0.3, 0.5, 0.7]
- `contrastive_lambda`: [0.05, 0.1, 0.2]
- `num_memory_prototypes`: [3, 5, 10]
- `client_fraction`: [0.05, 0.1, 0.2]

---

## 常见问题

### Q1: 如何处理新用户（冷启动）？

**A**: 新用户可以使用全局抽象记忆作为初始化：
```python
if len(user_sequence) < min_seq_len:
    # 使用全局记忆辅助推荐
    user_memory.initialize_from_global(global_abstract_memory)
```

### Q2: 记忆容量如何设置？

**A**: 经验值：
- 小数据集(ML-1M): 20-50
- 中数据集(Amazon): 50-100
- 大数据集(淘宝): 100-200

### Q3: Surprise阈值如何调整？

**A**:
- 阈值太低 → 记忆噪声多
- 阈值太高 → 记忆更新少
- 建议：在验证集上搜索最优值（0.3-0.7）

### Q4: 是否支持Non-IID数据？

**A**: 是的，FedMem特别适合Non-IID场景：
- 本地记忆捕获用户个性化偏好
- 原型聚合提取共性知识
- 对比学习增强泛化能力

---

## 参考文献

1. **Federated Learning**: McMahan et al. "Communication-Efficient Learning of Deep Networks from Decentralized Data" (AISTATS 2017)
2. **SASRec**: Kang & McAuley. "Self-Attentive Sequential Recommendation" (ICDM 2018)
3. **Prototypical Networks**: Snell et al. "Prototypical Networks for Few-shot Learning" (NeurIPS 2017)
4. **Contrastive Learning**: Chen et al. "A Simple Framework for Contrastive Learning of Visual Representations" (ICML 2020)

---

## 附录

### A. 完整参数列表

见 `configs/fedmem_config.yaml`

### B. 评估指标说明

- **HR@K** (Hit Ratio): Top-K中是否包含目标item
- **NDCG@K** (Normalized DCG): 考虑排名位置的指标
- **MRR** (Mean Reciprocal Rank): 目标item排名的倒数平均值

### C. 故障排除

常见错误及解决方案：

1. **OOM (Out of Memory)**
   - 减小 `batch_size`
   - 减小 `client_fraction`
   - 使用 `release_model()` 及时释放内存

2. **训练不收敛**
   - 降低 `learning_rate`
   - 增加 `local_epochs`
   - 检查数据质量

3. **记忆未更新**
   - 降低 `surprise_threshold`
   - 检查损失计算是否正确
   - 打印 `memory_updates` 指标
