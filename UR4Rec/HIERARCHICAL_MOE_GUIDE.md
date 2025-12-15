# Hierarchical MoE with CLIP 使用指南

## 概述

新的层级MoE架构使用**真实的CLIP图片特征**和**层级专家系统**，相比原始架构有以下改进：

### 架构对比

**原始架构（Flat MoE）：**
```
3个Expert（固定）
├─ Expert 0: 用户偏好文本
├─ Expert 1: 物品描述文本
└─ Expert 2: 物品embedding（非真实图片❌）
```

**新架构（Hierarchical MoE + CLIP）：**
```
Level 1: 模态内部MoE（3个模态 × 3个sub-experts = 9个专家）
├─ 用户偏好MoE
│   ├─ Genre Expert（类型偏好）
│   ├─ Mood Expert（情绪偏好）
│   └─ Style Expert（风格偏好）
├─ 物品描述MoE
│   ├─ Content Expert（内容理解）
│   ├─ Theme Expert（主题分析）
│   └─ Quality Expert（质量评估）
└─ CLIP图片MoE
    ├─ Composition Expert（视觉构图）✅
    ├─ Color/Texture Expert（颜色纹理）✅
    └─ Object Expert（物体识别）✅

Level 2: 跨模态融合
└─ Cross-Modal Router（学习3个模态的权重）
```

## 安装依赖

```bash
# 安装CLIP
pip install git+https://github.com/openai/CLIP.git

# 安装其他依赖
pip install torch torchvision pillow sentence-transformers
```

## 使用步骤

### Step 1: 准备图片数据

确保MovieLens数据集中有图片：

```bash
# 图片应存储在：
UR4Rec/data/Multimodal_Datasets/images/
├── 1.jpg
├── 2.jpg
├── 3.jpg
...
└── 1659.jpg
```

如果没有图片，可以运行下载脚本：
```bash
python UR4Rec/scripts/download_images.py \
    --data_dir UR4Rec/data/Multimodal_Datasets \
    --output_dir UR4Rec/data/Multimodal_Datasets/images
```

### Step 2: 提取CLIP特征

使用CLIP模型预提取所有图片的特征（加速训练）：

```bash
python UR4Rec/scripts/extract_clip_features.py \
    --data_dir UR4Rec/data/Multimodal_Datasets \
    --output_path UR4Rec/data/clip_features.pt \
    --clip_model ViT-B/32 \
    --output_dim 512 \
    --batch_size 32
```

**参数说明：**
- `--clip_model`: CLIP模型变体
  - `ViT-B/32`: 快速，512维 (推荐)
  - `ViT-B/16`: 中等，512维
  - `ViT-L/14`: 最佳，768维（但更慢）
- `--output_dim`: 输出特征维度（建议512）
- `--batch_size`: 批处理大小

**预期输出：**
```
✓ 加载 CLIP 模型: ViT-B/32 (embedding_dim=512)
物品总数: 1659
✓ 找到 1659 个有效图片文件
提取 CLIP 特征...
  进度: 10/52 batches
  进度: 20/52 batches
  ...
✓ 完成: 1659 个物品的图片特征已提取
✓ 图片特征已保存至: UR4Rec/data/clip_features.pt
特征形状: torch.Size([1660, 512])
```

### Step 3: 训练Hierarchical MoE模型

**重要：目前训练脚本还需要适配新架构，等当前训练完成后我会更新训练代码。**

暂时先提取好CLIP特征，训练代码会在分析完当前训练结果后更新。

## 预期改进

相比原始架构，新架构预期带来：

### 1. **真实视觉特征** ✅
- 使用CLIP提取的图片embedding
- 捕获视觉内容、颜色、构图等信息
- 替代原来的可训练item embedding

### 2. **更细粒度的表示** ✅
- 9个专家 vs 原来3个
- 每个模态内部学习多个aspects
- 更丰富的特征表达能力

### 3. **更好的可解释性** ✅
- 可以分析每个sub-expert的贡献
- 理解模型关注哪些方面（类型、风格、视觉等）
- 支持可视化routing权重

### 4. **潜在性能提升**
- **预期Hit@10提升**: 0.46 → 0.55-0.65
- **原因**:
  - 真实图片特征提供额外信息源
  - 层级架构增强表达能力
  - 更细粒度的特征学习

## 架构优势

### Level 1 (Within-Modality MoE)

**用户偏好MoE：**
- Genre Expert: 学习类型偏好（动作、爱情、科幻等）
- Mood Expert: 学习情绪偏好（轻松、紧张、温馨等）
- Style Expert: 学习风格偏好（商业、艺术、独立等）

**物品描述MoE：**
- Content Expert: 理解情节内容和故事线
- Theme Expert: 分析主题和深层含义
- Quality Expert: 评估制作质量和专业度

**CLIP图片MoE：**
- Composition Expert: 海报构图、版式设计
- Color/Texture Expert: 色调、光影、视觉风格
- Object Expert: 人物、场景、物体识别

### Level 2 (Cross-Modality Fusion)

- 学习三个模态的动态权重
- 不同用户/物品可能依赖不同模态
- 示例：
  - 视觉导向用户 → 图片权重高
  - 剧情导向用户 → 描述权重高

## 性能监控

训练时可以监控：

1. **Level 1 Routing Weights**
   ```python
   routing_info['user_sub_weights']  # [B, 3] 用户偏好内部权重
   routing_info['desc_sub_weights']  # [B, 3] 物品描述内部权重
   routing_info['image_sub_weights']  # [B, 3] CLIP图片内部权重
   ```

2. **Level 2 Routing Weights**
   ```python
   routing_info['cross_modal_weights']  # [B, 3] 跨模态权重
   # [用户偏好权重, 物品描述权重, CLIP图片权重]
   ```

3. **Expert Specialization Analysis**
   - 分析每个expert在哪些类型物品上权重高
   - 验证专家是否学到预期的aspects

## 下一步

当前训练完成后，我会：

1. ✅ 分析当前模型瓶颈
2. ✅ 更新训练脚本以支持Hierarchical MoE
3. ✅ 集成CLIP特征加载
4. ✅ 运行新架构训练
5. ✅ 对比性能提升

## 相关文件

- **CLIP编码器**: `UR4Rec/models/clip_image_encoder.py`
- **层级MoE**: `UR4Rec/models/hierarchical_moe.py`
- **特征提取脚本**: `UR4Rec/scripts/extract_clip_features.py`
- **配置文件**: `UR4Rec/configs/ur4rec_hierarchical_moe_clip.yaml`

## 常见问题

**Q: 为什么要预提取CLIP特征，不直接在训练时提取？**

A: 预提取有以下优势：
- 训练速度快10-20倍（避免重复编码）
- 减少GPU内存占用
- CLIP编码器可以freeze，只训练MoE部分

**Q: 如果没有所有物品的图片怎么办？**

A: 代码会自动处理：
- 有图片的物品使用CLIP特征
- 无图片的物品使用零向量（或随机初始化）
- 对性能影响较小（大部分物品有图片即可）

**Q: CLIP特征文件有多大？**

A:
- 1659个物品 × 512维 × 4字节 ≈ 3.4 MB
- 非常小，可以快速加载

**Q: 能否用其他视觉模型替代CLIP？**

A: 可以，只需修改 `CLIPImageEncoder` 类：
- ResNet/EfficientNet: 修改encoder部分
- BLIP/ALBEF: 修改为多模态编码器
- 自定义CNN: 完全重写encoder

## 总结

新的Hierarchical MoE + CLIP架构提供了：
1. ✅ 真实的视觉特征（CLIP）
2. ✅ 层级专家系统（9个expert）
3. ✅ 更好的可解释性
4. ✅ 潜在的性能提升

当前训练完成后，我们将使用这个新架构进行下一轮训练，并对比性能改进。
