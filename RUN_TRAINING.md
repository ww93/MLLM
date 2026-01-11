# 三阶段训练运行指南

## ✅ 所有Bug已修复

1. ✅ Stage 1不再加载不兼容的预训练模型
2. ✅ Stage 2/3禁用了Warmup机制
3. ✅ 客户端模型引用同步已修复
4. ✅ PyTorch 2.6兼容性已修复
5. ✅ Python解释器路径已修复
6. ✅ 空参数处理已修复

## 🚀 运行训练

### 准备工作

```bash
cd /Users/admin/Desktop/MLLM

# 激活虚拟环境
source venv/bin/activate

# 验证环境
python --version  # 应该显示 Python 3.13.9
python -c "import torch; print(torch.__version__)"  # 应该显示 2.9.1
```

### 三阶段训练

#### Stage 1: Backbone Pre-training

```bash
python UR4Rec/scripts/train_stage1_backbone.py
```

**预期效果**：
- 训练30轮
- 最终HR@10应达到0.60-0.70
- 保存在`UR4Rec/checkpoints/stage1_backbone/`

#### Stage 2: Modality Alignment

```bash
# 在Stage 1完成后运行
python UR4Rec/scripts/train_stage2_alignment.py
```

**预期效果**：
- 第1轮HR@10应接近0.60-0.70（继承Stage 1）
- 训练20轮
- 保存在`UR4Rec/checkpoints/stage2_alignment/`

#### Stage 3: MoE Fine-tuning

```bash
# 在Stage 2完成后运行
python UR4Rec/scripts/train_stage3_moe.py
```

**预期效果**：
- 第1轮HR@10应接近Stage 2最终值
- 训练20轮
- 最终HR@10应达到0.70+
- 保存在`UR4Rec/checkpoints/stage3_moe/`

### 一键运行所有阶段

```bash
python UR4Rec/scripts/train_all_stages.py
```

## 📊 验证清单

### Stage 1验证
- [ ] 日志显示"从零训练，不加载预训练模型"
- [ ] 第1轮HR@10 > 0.10
- [ ] 第10轮HR@10 > 0.50
- [ ] 第30轮HR@10 ≈ 0.60-0.70

### Stage 2验证
- [ ] 日志显示"✓ 已同步Stage 1权重到 1000 个客户端"
- [ ] 日志显示"自动禁用warmup"
- [ ] 日志**不**显示"[Warmup阶段 X/20]"
- [ ] 第1轮HR@10 ≈ 0.60-0.70
- [ ] 聚合参数数量 = 64个（不是28个）

### Stage 3验证
- [ ] 日志显示"✓ 已同步Stage 1+2权重到 1000 个客户端"
- [ ] 日志显示"自动禁用warmup"
- [ ] 第1轮HR@10 ≈ Stage 2最终值
- [ ] 最终HR@10 > 0.70

## 🐛 故障排查

### 问题1: ModuleNotFoundError: No module named 'torch'

**解决方案**：
```bash
source venv/bin/activate
```

### 问题2: HR@10仍然很低

**检查**：
- 查看日志是否显示"✓ 已同步Stage X权重到客户端"
- 确认没有"[Warmup阶段 X/20]"
- 检查Stage 1是否达到0.60-0.70

### 问题3: Python缓存未更新

**解决方案**：
```bash
find UR4Rec -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find UR4Rec -name "*.pyc" -delete 2>/dev/null
```

## 📈 预期训练轨迹

### Stage 1 (30轮)
```
轮次    HR@10    说明
1      ~0.15    初始
10     ~0.50    快速上升
20     ~0.60    稳定
30     ~0.65    收敛
```

### Stage 2 (20轮)
```
轮次    HR@10    说明
1      ~0.65    继承Stage 1 ✓
10     ~0.64    对齐中
20     ~0.65    对齐完成
```

### Stage 3 (20轮)
```
轮次    HR@10    说明
1      ~0.65    继承Stage 2 ✓
10     ~0.68    Router学习
20     ~0.72    最佳性能
```

## 📝 相关文档

- [FINAL_FIX_INSTRUCTIONS.md](UR4Rec/FINAL_FIX_INSTRUCTIONS.md) - 详细修复说明
- [STAGE2_STAGE3_COMPLETE_FIX.md](UR4Rec/STAGE2_STAGE3_COMPLETE_FIX.md) - 技术分析
- [THREE_STAGE_TRAINING_GUIDE.md](UR4Rec/THREE_STAGE_TRAINING_GUIDE.md) - 完整训练指南

---

**重要提示**：
- 必须在虚拟环境中运行
- Stage 2/3依赖前一阶段的checkpoint
- 训练过程中不要中断，否则需要重新开始该阶段
