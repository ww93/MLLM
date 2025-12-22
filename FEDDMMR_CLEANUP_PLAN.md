# FedDMMR Project Cleanup Plan

## 架构确认

基于 `/Users/admin/Desktop/MLLM/UR4Rec/scripts/train_fedmem.py` 的分析，确认项目符合 FedDMMR 架构：

### ✅ 核心组件已验证

1. **模型架构** (`ur4rec_v2_moe.py`)
   - ✅ SASRec 骨干
   - ✅ VisualExpert（轻量级注意力）
   - ✅ SemanticExpert（多头交叉注意力）
   - ✅ ItemCentricRouter（基于目标物品嵌入）
   - ✅ Representation-Level Fusion（加权和专家表示）
   - ✅ 损失函数：Rec_Loss + Load_Balancing_Loss

2. **本地内存** (`local_dynamic_memory.py`)
   - ✅ Key-Value 存储（Item ID -> {Visual, Text}）
   - ✅ Surprise-based Update（基于阈值）
   - ✅ Utility-based Expiration（Recency + Frequency）
   - ✅ Prototype Extraction（K-Means）

3. **联邦学习** (`fedmem_client.py`, `fedmem_server.py`)
   - ✅ 客户端加载多模态特征
   - ✅ 服务器聚合（FedAvg + Prototype Aggregation）

## 文件分类

### 🟢 KEEP - FedDMMR 核心文件

#### 模型文件 (UR4Rec/models/)
- ✅ **ur4rec_v2_moe.py** - FedDMMR 主模型
- ✅ **sasrec.py** - SASRec 骨干
- ✅ **local_dynamic_memory.py** - 本地动态记忆
- ✅ **fedmem_client.py** - 联邦客户端
- ✅ **fedmem_server.py** - 联邦服务器
- ✅ **multimodal_loss.py** - 多模态损失函数（对比损失）

#### 训练脚本 (UR4Rec/scripts/)
- ✅ **train_fedmem.py** - 主训练脚本
- ✅ **extract_clip_features.py** - CLIP 特征提取
- ✅ **generate_text_features.py** - 文本特征生成
- ✅ **process_ml100k.py** - ML-100K 数据预处理
- ✅ **process_ml1m.py** - ML-1M 数据预处理

#### 工具文件 (UR4Rec/utils/)
- ✅ **metrics.py** - 评估指标（含负采样评估）
- ✅ **data_loader.py** - 数据加载工具

#### 数据文件 (UR4Rec/data/)
- ✅ **dataset_loader.py** - 数据集加载器
- ✅ `ml100k_ratings_processed.dat` - 预处理的交互数据
- ✅ `clip_features_fixed.pt` - CLIP 视觉特征
- ✅ `item_text_features.pt` - 文本特征
- ✅ `item_metadata.json` - 物品元数据

### 🔴 DELETE - 与 FedDMMR 无关的文件

#### 1. README/文档文件（保留核心说明）
- ❌ FEDMEM_ADAPTATION_COMPLETED.md
- ❌ FEDMEM_PROJECT_SUMMARY.md
- ❌ FEDMEM_NEG_SAMPLING_IMPLEMENTATION.md
- ❌ FedMem_ADAPTATION_GUIDE.md
- ❌ FedMem_README.md
- ❌ DIAGNOSTIC_SUMMARY.md
- ❌ FINAL_DIAGNOSIS.md
- ❌ CONFIG_PARAMETER_REMOVED.md
- ❌ FEDERATED_VS_CENTRALIZED_PERFORMANCE.md
- ❌ DIAGNOSIS_REPORT.md
- ❌ debug_training_issue.md
- ❌ UR4Rec/PROJECT_COMPLETE.md
- ❌ UR4Rec/FEDMEM_IMPLEMENTATION.md
- ❌ UR4Rec/MULTIMODAL_DATA_GUIDE.md
- ❌ UR4Rec/CHANGELOG_TEXT_DESCRIPTIONS.md
- ❌ UR4Rec/RETRIEVER_ANALYSIS.md
- ❌ UR4Rec/WORKFLOW.md
- ❌ UR4Rec/TRAINING_GUIDE.md
- ❌ UR4Rec/DOCS_INDEX.md
- ⚠️  **保留**: UR4Rec/models/FedDMMR_README.md（FedDMMR 架构说明）

#### 2. 不相关的模型文件
- ❌ **hierarchical_moe.py** - 旧的分层 MoE（已被 ur4rec_v2_moe.py 取代）
- ❌ **ur4rec_v2.py** - 旧版本（已被 ur4rec_v2_moe.py 取代）
- ❌ **user_preference_retriever.py** - 静态用户档案（违反 Item-Centric 原则）
- ❌ **text_preference_retriever.py** - 静态用户偏好检索
- ❌ **text_preference_retriever_moe.py** - 静态用户偏好 MoE
- ❌ **retriever_moe_memory.py** - 旧的检索器实现
- ❌ **multimodal_retriever.py** - 旧的多模态检索器
- ❌ **clip_image_encoder.py** - 独立的 CLIP 编码器（已集成到特征提取脚本）
- ❌ **llm_generator.py** - LLM 生成器（用于数据预处理，非在线推理）
- ❌ **joint_trainer.py** - 联合训练器（非联邦）
- ❌ **federated_aggregator.py** - 旧的聚合器（已被 fedmem_server.py 取代）
- ❌ **federated_server.py** - 旧的服务器（已被 fedmem_server.py 取代）
- ❌ **federated_client.py** - 旧的客户端（已被 fedmem_client.py 取代）
- ❌ **federated_client_ur4rec.py** - 旧的 UR4Rec 客户端
- ❌ **sasrec_fixed.py** - 修复版本（已合并到 sasrec.py）

#### 3. 不相关的训练脚本
- ❌ **train_federated.py** - 旧的联邦训练
- ❌ **train_federated_ur4rec_moe.py** - 旧的 MoE 训练
- ❌ **train_ur4rec_moe.py** - 非联邦 MoE 训练
- ❌ **train_v2.py** - V2 版本训练
- ❌ **train_sasrec_centralized.py** - 中心化训练（非联邦）
- ❌ **train_sasrec_fixed.py** (根目录) - 修复版本训练

#### 4. LLM 相关脚本（数据预处理，非核心）
- ❌ **generate_llm_data.py** - LLM 数据生成
- ❌ **test_llm_connection.py** - LLM 连接测试
- ❌ **extract_ml1m_descriptions.py** - 描述提取
- ❌ **LLM_DATA_GENERATION_README.md** - LLM 文档
- ❌ QWEN_FLASH_INTEGRATION.md (根目录)

#### 5. 诊断/测试脚本
- ❌ **diagnose_training_eval_mismatch.py**
- ❌ **diagnose_id_mapping_bug.py** (根目录)
- ❌ **diagnose_router_weights.py** (根目录)
- ❌ **diagnose_scoring.py** (根目录)
- ❌ **diagnostic_check_embedding_update.py**
- ❌ **test_model_forward.py** (根目录)
- ❌ **test_negative_sampling.py** (根目录)
- ❌ **test_item_pop_baseline.py**
- ❌ **test_text_extraction.py** (UR4Rec/)
- ❌ **analyze_training.py** (根目录)
- ❌ **analyze_expert_contributions.py** (根目录)

#### 6. 其他数据预处理脚本（保留核心的）
- ⚠️  **保留**: process_ml100k.py, process_ml1m.py
- ❌ **prepare_ml100k_data.py** - 重复功能
- ❌ **prepare_ml1m_data.py** - 重复功能
- ❌ **process_ml100k_4star.py** - 特殊版本
- ❌ **preprocess_multimodal_dataset.py** - 通用预处理
- ❌ **preprocess_movielens.py** - 旧版预处理
- ❌ **preprocess_images.py** - 图像预处理（功能已集成）
- ❌ **preprocess_beauty.py** - Beauty 数据集（非目标数据集）
- ❌ **download_images.py** - 图像下载
- ❌ **extract_ml1m_clip_features.py** - 重复功能（保留 extract_clip_features.py）
- ❌ **extract_ml1m_text_features.py** - 重复功能（保留 generate_text_features.py）

#### 7. 配置文件（保留核心的）
- ⚠️  **保留**: UR4Rec/configs/fedmem_config.yaml
- ❌ UR4Rec/configs/ur4rec_moe_100k.yaml - 非联邦配置
- ❌ UR4Rec/configs/ur4rec_hierarchical_balanced.yaml - 旧架构配置
- ❌ UR4Rec/configs/ur4rec_federated.yaml - 旧配置
- ❌ UR4Rec/config_ml100k.yaml - 重复配置

#### 8. 示例文件
- ❌ UR4Rec/examples/quick_start.py - 示例代码
- ❌ UR4Rec/demo.py - 演示代码

#### 9. Checkpoints（全部删除，保留目录结构说明）
- ❌ UR4Rec/checkpoints/* - 所有 checkpoint 目录
  - ml100k_no_l2norm/
  - ml100k_multimodal/
  - centralized_test/
  - sasrec_baseline/
  - fedmem_test/
  - ... (所有其他 checkpoint)

#### 10. 其他杂项
- ❌ UR4Rec/setup.py - 打包文件（非必需）
- ❌ UR4Rec/data/multimodal_dataset.py - 旧数据集类

## 删除执行计划

### 阶段 1: 删除文档文件
```bash
# 根目录文档
rm -f FEDMEM_ADAPTATION_COMPLETED.md
rm -f FEDMEM_PROJECT_SUMMARY.md
rm -f FEDMEM_NEG_SAMPLING_IMPLEMENTATION.md
rm -f FedMem_ADAPTATION_GUIDE.md
rm -f FedMem_README.md
rm -f DIAGNOSTIC_SUMMARY.md
rm -f FINAL_DIAGNOSIS.md
rm -f CONFIG_PARAMETER_REMOVED.md
rm -f FEDERATED_VS_CENTRALIZED_PERFORMANCE.md
rm -f DIAGNOSIS_REPORT.md
rm -f debug_training_issue.md
rm -f QWEN_FLASH_INTEGRATION.md

# UR4Rec 文档
rm -f UR4Rec/PROJECT_COMPLETE.md
rm -f UR4Rec/FEDMEM_IMPLEMENTATION.md
rm -f UR4Rec/MULTIMODAL_DATA_GUIDE.md
rm -f UR4Rec/CHANGELOG_TEXT_DESCRIPTIONS.md
rm -f UR4Rec/RETRIEVER_ANALYSIS.md
rm -f UR4Rec/WORKFLOW.md
rm -f UR4Rec/TRAINING_GUIDE.md
rm -f UR4Rec/DOCS_INDEX.md
```

### 阶段 2: 删除旧模型文件
```bash
cd UR4Rec/models
rm -f hierarchical_moe.py
rm -f ur4rec_v2.py
rm -f user_preference_retriever.py
rm -f text_preference_retriever.py
rm -f text_preference_retriever_moe.py
rm -f retriever_moe_memory.py
rm -f multimodal_retriever.py
rm -f clip_image_encoder.py
rm -f llm_generator.py
rm -f joint_trainer.py
rm -f federated_aggregator.py
rm -f federated_server.py
rm -f federated_client.py
rm -f federated_client_ur4rec.py
rm -f sasrec_fixed.py
cd ../..
```

### 阶段 3: 删除旧训练脚本
```bash
cd UR4Rec/scripts
rm -f train_federated.py
rm -f train_federated_ur4rec_moe.py
rm -f train_ur4rec_moe.py
rm -f train_v2.py
rm -f train_sasrec_centralized.py
rm -f generate_llm_data.py
rm -f test_llm_connection.py
rm -f extract_ml1m_descriptions.py
rm -f LLM_DATA_GENERATION_README.md
rm -f diagnose_training_eval_mismatch.py
rm -f diagnostic_check_embedding_update.py
rm -f test_item_pop_baseline.py
rm -f prepare_ml100k_data.py
rm -f prepare_ml1m_data.py
rm -f process_ml100k_4star.py
rm -f preprocess_multimodal_dataset.py
rm -f preprocess_movielens.py
rm -f preprocess_images.py
rm -f preprocess_beauty.py
rm -f download_images.py
rm -f extract_ml1m_clip_features.py
rm -f extract_ml1m_text_features.py
rm -f evaluate.py
cd ../..
```

### 阶段 4: 删除根目录诊断脚本
```bash
rm -f train_sasrec_fixed.py
rm -f diagnose_id_mapping_bug.py
rm -f diagnose_router_weights.py
rm -f diagnose_scoring.py
rm -f test_model_forward.py
rm -f test_negative_sampling.py
rm -f analyze_training.py
rm -f analyze_expert_contributions.py
```

### 阶段 5: 删除配置文件
```bash
cd UR4Rec/configs
rm -f ur4rec_moe_100k.yaml
rm -f ur4rec_hierarchical_balanced.yaml
rm -f ur4rec_federated.yaml
cd ..
rm -f config_ml100k.yaml
cd ..
```

### 阶段 6: 删除示例和其他
```bash
rm -rf UR4Rec/examples
rm -f UR4Rec/demo.py
rm -f UR4Rec/test_text_extraction.py
rm -f UR4Rec/setup.py
rm -f UR4Rec/data/multimodal_dataset.py
```

### 阶段 7: 删除 Checkpoints
```bash
rm -rf UR4Rec/checkpoints/*
# 保留目录结构
mkdir -p UR4Rec/checkpoints
echo "# FedDMMR Checkpoints Directory" > UR4Rec/checkpoints/README.md
echo "Training checkpoints will be saved here." >> UR4Rec/checkpoints/README.md
```

## 保留的最终文件结构

```
MLLM/
├── UR4Rec/
│   ├── models/
│   │   ├── ur4rec_v2_moe.py          # FedDMMR 主模型
│   │   ├── sasrec.py                 # SASRec 骨干
│   │   ├── local_dynamic_memory.py   # 本地动态记忆
│   │   ├── fedmem_client.py          # 联邦客户端
│   │   ├── fedmem_server.py          # 联邦服务器
│   │   ├── multimodal_loss.py        # 多模态损失
│   │   ├── FedDMMR_README.md         # 架构说明
│   │   └── __init__.py
│   ├── scripts/
│   │   ├── train_fedmem.py           # 主训练脚本
│   │   ├── extract_clip_features.py  # CLIP 特征提取
│   │   ├── generate_text_features.py # 文本特征生成
│   │   ├── process_ml100k.py         # ML-100K 预处理
│   │   └── process_ml1m.py           # ML-1M 预处理
│   ├── utils/
│   │   ├── metrics.py                # 评估指标
│   │   ├── data_loader.py            # 数据加载
│   │   └── __init__.py
│   ├── data/
│   │   ├── dataset_loader.py         # 数据集加载器
│   │   ├── ml100k_ratings_processed.dat
│   │   ├── clip_features_fixed.pt
│   │   ├── item_text_features.pt
│   │   └── item_metadata.json
│   ├── configs/
│   │   └── fedmem_config.yaml        # FedMem 配置
│   ├── checkpoints/
│   │   └── README.md                 # Checkpoint 说明
│   └── README.md                     # 项目说明
└── README.md                          # 根目录说明
```

## 验证清单

清理后，确保以下功能正常：

1. ✅ 可以运行 `train_fedmem.py` 进行训练
2. ✅ 支持 ML-100K 和 ML-1M 数据集
3. ✅ 支持多模态特征（CLIP + 文本）
4. ✅ 1:100 负采样评估正常工作
5. ✅ 本地动态记忆更新正常
6. ✅ 联邦聚合（FedAvg + Prototype）正常

## 执行命令

执行所有删除操作：
```bash
# 在 /Users/admin/Desktop/MLLM 目录下执行
bash FEDDMMR_CLEANUP_SCRIPT.sh
```
