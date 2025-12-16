# 联邦学习实现总结

## 实现状态

### ✅ 已完成的核心组件

1. **federated_aggregator.py** - FedAvg参数聚合器
   - 实现了FedAvg经典算法
   - 支持FedProx（带近端正则化）
   - 支持自定义权重聚合
   - 提供梯度压缩功能

2. **federated_client.py** - 联邦学习客户端
   - 每个user作为一个client
   - Leave-one-out数据划分（训练/验证/测试）
   - 本地训练与参数上传
   - 支持负采样训练

3. **federated_server.py** - 联邦学习服务器
   - 管理全局模型
   - 协调客户端训练
   - FedAvg参数聚合
   - 早停机制
   - 全局模型评估

4. **train_federated.py** - 联邦训练脚本
   - 完整的训练流程
   - 支持配置化训练
   - 自动保存模型和结果

5. **ur4rec_federated.yaml** - 联邦学习配置
   - 模型参数配置
   - 联邦学习超参数
   - 客户端训练参数

## 遇到的问题与解决

### 问题：依赖文件缺失

在项目清理时，一些旧文件（`retriever_moe.py`, `retriever_moe_memory.py`）被删除，但UR4RecV2MoE模型依赖这些文件。

### 解决方案

有两种方案：

**方案A：简化联邦学习模型**（推荐）
- 使用SASRec作为联邦学习的基础模型
- SASRec已经过验证，性能稳定（HR@10 ≈ 0.40）
- 避免复杂的MoE和Memory机制
- 更符合联邦学习的轻量化需求

**方案B：修复依赖**
- 需要检查并修复所有被删除文件的导入
- 或者重新实现简化版的UR4RecV2MoE
- 工作量较大

## 建议的实现路径

### 1. 使用SASRec进行联邦学习

修改`train_federated.py`，将模型从UR4RecV2MoE改为SASRec：

```python
from UR4Rec.models.sasrec import SASRec

# 创建全局模型（SASRec）
global_model = SASRec(
    num_items=config['num_items'],
    hidden_dim=config['sasrec_hidden_dim'],
    num_blocks=config['sasrec_num_blocks'],
    num_heads=config['sasrec_num_heads'],
    dropout=config['sasrec_dropout'],
    max_seq_len=config['max_seq_len']
)
```

### 2. 相应修改federated_client.py

Client使用SASRec而不是UR4RecV2MoE：
- 简化前向传播
- 只训练序列推荐部分
- 移除MoE和Retriever相关代码

### 3. 训练与验证

训练流程：
```bash
python UR4Rec/scripts/train_federated.py \
    --config UR4Rec/configs/ur4rec_federated.yaml \
    --data_dir UR4Rec/data/Multimodal_Datasets \
    --output_dir outputs/federated \
    --num_rounds 50 \
    --local_epochs 1 \
    --client_fraction 0.1 \
    --patience 10
```

### 4. 性能对比

预期结果：
- 中心化SASRec: HR@10 ≈ 0.40
- 联邦SASRec: HR@10 ≈ 0.38-0.40 (目标：不低于中心化)
- 联邦学习可能有1-5%的性能损失（正常现象）

## 核心算法：FedAvg

```
服务器端：
1. 初始化全局模型 w_0
2. For round t = 1 to T:
   a. 选择 K 个客户端（随机采样）
   b. 下发全局模型 w_t 给选中的客户端
   c. 并行执行客户端训练
   d. 收集客户端模型 {w_t^1, w_t^2, ..., w_t^K}
   e. 聚合：w_{t+1} = Σ (n_k / n_total) * w_t^k
   f. 评估全局模型性能
   g. 检查早停条件
3. 返回最佳全局模型

客户端端（第k个客户端）：
1. 接收全局模型 w_t
2. 使用本地数据 D_k 训练 E 个epoch
3. 本地SGD更新：w_t^k = w_t - η * ∇L(w_t, D_k)
4. 上传更新后的模型 w_t^k
```

## 关键参数说明

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| num_rounds | 联邦学习总轮数 | 50 |
| local_epochs | 客户端本地训练轮数 | 1 |
| client_fraction | 每轮参与的客户端比例 | 0.1 (10%) |
| learning_rate | 客户端学习率 | 0.001 |
| patience | 早停patience | 10 |
| batch_size | 客户端批大小 | 16 |
| num_negatives | 负样本数量 | 100 |

## 数据划分：Leave-One-Out

每个用户的交互序列划分：
- **训练集**：除最后2个item外的所有历史
- **验证集**：倒数第2个item
- **测试集**：最后1个item

示例：
```
用户序列：[1, 3, 5, 7, 9, 11, 13]
训练：[1, 3, 5, 7, 9]
验证：[11]
测试：[13]
```

## 实现优势

1. **隐私保护**：用户数据不离开本地
2. **可扩展**：支持大规模用户
3. **鲁棒性**：单个客户端故障不影响全局
4. **通信高效**：只传输模型参数，不传输数据
5. **个性化潜力**：可扩展为个性化联邦学习

## 后续优化方向

1. **聚合算法**：
   - FedProx（已实现）
   - FedNova
   - Scaffold

2. **客户端选择**：
   - 基于数据量的选择
   - 基于模型质量的选择
   - 公平性约束的选择

3. **通信优化**：
   - 梯度压缩（已实现Top-K）
   - 量化
   - 知识蒸馏

4. **个性化**：
   - Per-FedAvg
   - 联邦元学习
   - 混合模型（全局+本地）

## 文件结构

```
UR4Rec/
├── models/
│   ├── federated_aggregator.py    # FedAvg聚合器
│   ├── federated_client.py        # 客户端实现
│   ├── federated_server.py        # 服务器实现
│   └── sasrec.py                  # SASRec模型
├── scripts/
│   └── train_federated.py         # 联邦训练脚本
└── configs/
    └── ur4rec_federated.yaml      # 联邦配置文件
```

## 参考文献

1. McMahan et al. (2017). "Communication-Efficient Learning of Deep Networks from Decentralized Data." AISTATS.
2. Kang & McAuley (2018). "Self-Attentive Sequential Recommendation." ICDM.
3. Li et al. (2020). "Federated Optimization in Heterogeneous Networks." MLSys.

---

**实现日期**: 2025-12-15
**状态**: 核心组件已完成，需简化模型（SASRec）进行测试
