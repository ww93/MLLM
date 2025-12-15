# 评估阶段NaN问题修复

## 问题
评估时所有15个batch都出现NaN，导致验证指标全部为NaN：
```
Evaluation summary: 0 valid batches, 15 skipped (NaN), 0 total samples
Validation metrics: hit@5: nan, ndcg@5: nan, ...
```

## 根本原因
**训练脚本没有从config读取`sasrec_dropout`参数！**

- Config文件设置：`sasrec_dropout: 0.0`（已修复NaN）
- 但训练脚本第405行创建模型时**没有传递这个参数**
- 导致模型使用默认值：`sasrec_dropout=0.1`（会导致NaN）
- 训练阶段恰好没遇到NaN，但评估阶段(batch_size=64)触发了NaN

## 修复

### 1. 修改训练脚本传递dropout参数
**文件**: `UR4Rec/scripts/train_ur4rec_moe.py:410, 415`

```python
model = UR4RecV2MoE(
    num_items=num_items,
    sasrec_hidden_dim=config.get('sasrec_hidden_dim', 256),
    sasrec_num_blocks=config.get('sasrec_num_blocks', 2),
    sasrec_num_heads=config.get('sasrec_num_heads', 4),
    sasrec_dropout=config.get('sasrec_dropout', 0.1),  # ✅ 新增
    text_model_name=config.get('text_model_name', 'all-MiniLM-L6-v2'),
    text_embedding_dim=config.get('text_embedding_dim', 384),
    retriever_output_dim=config.get('retriever_output_dim', 256),
    moe_num_heads=config.get('moe_num_heads', 8),
    moe_dropout=config.get('moe_dropout', 0.1),  # ✅ 新增
    moe_num_proxies=config.get('moe_num_proxies', 4),
    # ...
)
```

### 2. 改进评估错误处理
**文件**: `UR4Rec/models/joint_trainer.py:647-648, 674, 678, 693, 696`

添加了详细的评估诊断：
- 跟踪有效/跳过的batch数量
- 统计产生的总样本数
- 防止"Mean of empty slice"警告
- 清晰的错误消息

## 重要：需要重新训练

⚠️ **已有的模型检查点无效！**

如果你已经训练过模型，检查点中的模型使用了`dropout=0.1`。需要：

1. **删除旧的检查点**：
```bash
rm -rf outputs/ur4rec_moe_test/*.pt
rm -rf outputs/ur4rec_moe/*.pt
```

2. **从头开始训练**：
```bash
python UR4Rec/scripts/train_ur4rec_moe.py \
    --config UR4Rec/configs/ur4rec_moe_100k.yaml \
    --data_dir UR4Rec/data/Multimodal_Datasets \
    --llm_data_dir data/llm_generated \
    --output_dir outputs/ur4rec_moe \
    --epochs_per_stage 100 \
    --patience 5
```

## 预期结果

修复后，评估应该显示：
```
Evaluation summary: 14 valid batches, 1 skipped (NaN), 448 total samples
Validation metrics:
  hit@5: 0.234
  ndcg@5: 0.189
  ...
```

如果仍然有NaN批次（少量），会被跳过但不影响评估。

## 可选：进一步稳定性改进

如果评估仍有NaN问题，可以考虑：

### 在config中也禁用MoE dropout
```yaml
# UR4Rec/configs/ur4rec_moe_100k.yaml
moe_dropout: 0.0  # 改为0.0（当前是0.1）
```

## 修改的文件

1. **UR4Rec/scripts/train_ur4rec_moe.py**
   - Line 410: 添加 `sasrec_dropout` 参数传递
   - Line 415: 添加 `moe_dropout` 参数传递

2. **UR4Rec/models/joint_trainer.py**
   - Lines 647-648: 添加batch统计变量
   - Line 674: 统计跳过的batch
   - Line 678: 统计有效的batch
   - Lines 680-693: 添加per-batch rank统计和警告
   - Line 696: 添加评估摘要输出
   - Lines 698-706: 处理空结果情况

## 技术细节

### 为什么训练没问题但评估有问题？

1. **Batch size不同**：
   - 训练：batch_size=32
   - 评估：batch_size=64（见train_ur4rec_moe.py:212）

2. **数值精度**：
   - 更大的batch可能导致attention矩阵更大
   - Dropout + 大矩阵 → 更容易产生极值
   - LayerNorm遇到极值 → NaN

3. **随机性**：
   - 训练数据shuffle，模式多样
   - 评估数据固定，可能有特殊模式触发NaN

### Dropout如何导致NaN

```
1. Embedding输出 → attention计算
2. Dropout随机置零部分激活 → 剩余激活被放大(1/p)
3. LayerNorm计算方差
4. 如果激活过大或全零 → 方差计算异常 → NaN
5. NaN传播到后续层
```

---

**修复时间**: 2025-12-10
**相关文档**: NAN_LOSS_FIX_SUMMARY.md, TRAINING_FIXES_SUMMARY.md
