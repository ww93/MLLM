# UR4Rec V2 with Hierarchical MoE

基于MovieLens-100K的多模态推荐系统，结合SASRec序列建模和Hierarchical MoE (Mixture of Experts) 架构。

## 项目特性

- **Hierarchical MoE架构**: 3个模态（用户偏好、物品描述、CLIP图像）各有3个子专家，共9个专家
- **多模态融合**: 集成文本（LLM生成）和视觉（CLIP）特征
- **四阶段训练**: pretrain_sasrec → pretrain_retriever → joint_finetune → end_to_end
- **性能**: HR@10 ≈ 0.41 (baseline: 0.40)

## 快速开始

### 1. 环境配置

```bash
# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备

```bash
# 准备MovieLens-100K数据
python UR4Rec/scripts/prepare_ml100k_data.py \
    --data_dir UR4Rec/data/Multimodal_Datasets

# 生成LLM偏好数据（需要Qwen API）
python UR4Rec/scripts/generate_llm_data.py \
    --data_dir UR4Rec/data/Multimodal_Datasets \
    --output_dir data/llm_generated

# 提取CLIP图像特征
python UR4Rec/scripts/extract_clip_features.py \
    --data_dir UR4Rec/data/Multimodal_Datasets \
    --output_path UR4Rec/data/clip_features.pt \
    --clip_model ViT-B/32
```

### 3. 开始训练

```bash
# 使用平衡版配置（推荐）
python UR4Rec/scripts/train_ur4rec_moe.py \
    --config UR4Rec/configs/ur4rec_hierarchical_balanced.yaml \
    --data_dir UR4Rec/data/Multimodal_Datasets \
    --llm_data_dir data/llm_generated \
    --output_dir outputs/ur4rec_balanced \
    --epochs_per_stage 25

# 或使用基础配置
python UR4Rec/scripts/train_ur4rec_moe.py \
    --config UR4Rec/configs/ur4rec_moe_100k.yaml \
    --data_dir UR4Rec/data/Multimodal_Datasets \
    --llm_data_dir data/llm_generated \
    --output_dir outputs/ur4rec_base \
    --epochs_per_stage 50
```

## 项目结构

```
MLLM/
├── UR4Rec/
│   ├── models/              # 模型实现
│   │   ├── __init__.py
│   │   ├── sasrec.py        # SASRec序列模型
│   │   ├── hierarchical_moe.py  # Hierarchical MoE实现
│   │   ├── text_preference_retriever_moe.py  # MoE检索器
│   │   ├── ur4rec_v2_moe.py # 主模型
│   │   ├── ur4rec_v2.py     # 基础模型
│   │   ├── joint_trainer.py # 训练器
│   │   ├── llm_generator.py # LLM数据生成
│   │   └── clip_image_encoder.py  # CLIP编码器
│   ├── scripts/             # 脚本
│   │   ├── train_ur4rec_moe.py      # 主训练脚本
│   │   ├── generate_llm_data.py     # LLM数据生成
│   │   ├── extract_clip_features.py # CLIP特征提取
│   │   └── prepare_ml100k_data.py   # 数据准备
│   └── configs/             # 配置文件
│       ├── ur4rec_hierarchical_balanced.yaml  # 平衡版配置（推荐）
│       └── ur4rec_moe_100k.yaml               # 基础配置
├── requirements.txt         # Python依赖
├── QUICK_START.md          # 详细使用指南
└── README.md               # 本文件
```

## 核心配置说明

### ur4rec_hierarchical_balanced.yaml（推荐）

适中的模型大小和训练速度：
- SASRec: 4层 × 512维
- 负样本数: 100
- Batch size: 16
- 训练时间: ~12-24小时（CPU）
- 预期性能: HR@10 ≈ 0.45-0.48

### ur4rec_moe_100k.yaml（基础）

较小模型，更快训练：
- SASRec: 3层 × 512维
- 负样本数: 20
- Batch size: 16
- 训练时间: ~6-12小时（CPU）
- 基线性能: HR@10 ≈ 0.40

## 技术栈

- **PyTorch** 2.0+: 深度学习框架
- **SentenceTransformers**: 文本编码（all-MiniLM-L6-v2）
- **CLIP**: 图像特征提取（ViT-B/32）
- **Qwen API**: LLM偏好数据生成

## 训练监控

训练过程会输出以下指标：
- **HR@5/10/20**: Hit Rate（命中率）
- **NDCG@5/10/20**: Normalized Discounted Cumulative Gain
- **MRR**: Mean Reciprocal Rank
- **Loss**: 训练损失

## 常见问题

### 1. 内存不足？

减少batch_size或负样本数：
```yaml
batch_size: 8  # 从16降至8
num_negatives: 50  # 从100降至50
```

### 2. 训练太慢？

- 使用GPU（如果可用）
- 减少epochs_per_stage
- 使用基础配置而非平衡版

### 3. 性能不理想？

- 增加负样本数（100 → 200）
- 增加训练epochs
- 尝试不同的学习率

## 引用

如果使用本项目，请引用：

```bibtex
@misc{ur4rec2025,
  title={UR4Rec V2: Hierarchical Mixture-of-Experts for Multi-Modal Recommendation},
  author={Your Name},
  year={2025}
}
```

## 许可证

MIT License

## 联系方式

如有问题，请提Issue或联系：[你的邮箱]
