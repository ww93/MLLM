#!/bin/bash
# 完整实验脚本 - 使用实际数据文件和多模态特征

echo "=========================================="
echo "FedDMMR 完整实验"
echo "=========================================="
echo ""
echo "实验配置："
echo "  - 数据集: ML-100K"
echo "  - 训练轮数: 30"
echo "  - 多模态特征: CLIP视觉 + 文本"
echo "  - 对比4种配置"
echo ""

cd /Users/admin/Desktop/MLLM/UR4Rec

# 激活虚拟环境
source /Users/admin/Desktop/MLLM/venv/bin/activate

# 数据文件路径
DATA_DIR="data"
DATA_FILE="ml100k_ratings_processed.dat"
VISUAL_FILE="clip_features_fixed.pt"
TEXT_FILE="item_text_features.pt"

# 检查数据文件是否存在
if [ ! -f "$DATA_DIR/$DATA_FILE" ]; then
    echo "❌ 错误: 数据文件不存在: $DATA_DIR/$DATA_FILE"
    echo ""
    echo "可用的数据文件："
    ls -la $DATA_DIR/*.dat 2>/dev/null || echo "  未找到数据文件"
    exit 1
fi

echo "✓ 数据文件: $DATA_DIR/$DATA_FILE"

# 检查多模态特征文件
if [ -f "$DATA_DIR/$VISUAL_FILE" ]; then
    echo "✓ 视觉特征: $DATA_DIR/$VISUAL_FILE"
    USE_VISUAL="--visual_file $VISUAL_FILE"
else
    echo "⚠️  视觉特征文件不存在，将不使用视觉特征"
    USE_VISUAL=""
fi

if [ -f "$DATA_DIR/$TEXT_FILE" ]; then
    echo "✓ 文本特征: $DATA_DIR/$TEXT_FILE"
    USE_TEXT="--text_file $TEXT_FILE"
else
    echo "⚠️  文本特征文件不存在，将不使用文本特征"
    USE_TEXT=""
fi

echo ""

# 基础参数
COMMON_ARGS="--data_dir $DATA_DIR \
    --data_file $DATA_FILE \
    $USE_VISUAL \
    $USE_TEXT \
    --num_rounds 30 \
    --client_fraction 0.2 \
    --local_epochs 1 \
    --learning_rate 0.001 \
    --batch_size 32 \
    --memory_capacity 50 \
    --enable_prototype_aggregation \
    --patience 10 \
    --seed 42"

# 保存目录
SAVE_DIR_BASE="checkpoints/full_experiment_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$SAVE_DIR_BASE"

echo "结果将保存到: $SAVE_DIR_BASE"
echo ""

# 记录实验配置
cat > "$SAVE_DIR_BASE/experiment_config.txt" << EOF
实验时间: $(date)
数据集: $DATA_FILE
视觉特征: ${USE_VISUAL:-未使用}
文本特征: ${USE_TEXT:-未使用}
训练轮数: 30
客户端比例: 0.2
学习率: 0.001
EOF

echo "=========================================="
echo "[1/4] Baseline (无优化)"
echo "=========================================="
python scripts/train_fedmem.py \
    $COMMON_ARGS \
    --save_dir "$SAVE_DIR_BASE/baseline" \
    | tee "$SAVE_DIR_BASE/baseline.log"

if [ $? -ne 0 ]; then
    echo "❌ Baseline 训练失败！"
    exit 1
fi
echo "✅ Baseline 完成"

echo ""
echo "=========================================="
echo "[2/4] Strategy1 (Router Bias Initialization)"
echo "=========================================="
python scripts/train_fedmem.py \
    $COMMON_ARGS \
    --save_dir "$SAVE_DIR_BASE/strategy1" \
    --init_bias_for_sasrec \
    --sasrec_bias_value 5.0 \
    | tee "$SAVE_DIR_BASE/strategy1.log"

if [ $? -ne 0 ]; then
    echo "❌ Strategy1 训练失败！"
    exit 1
fi
echo "✅ Strategy1 完成"

echo ""
echo "=========================================="
echo "[3/4] Strategy2 (Partial Aggregation)"
echo "=========================================="
python scripts/train_fedmem.py \
    $COMMON_ARGS \
    --save_dir "$SAVE_DIR_BASE/strategy2" \
    --partial_aggregation_warmup_rounds 20 \
    | tee "$SAVE_DIR_BASE/strategy2.log"

if [ $? -ne 0 ]; then
    echo "❌ Strategy2 训练失败！"
    exit 1
fi
echo "✅ Strategy2 完成"

echo ""
echo "=========================================="
echo "[4/4] Strategy1+2 (组合策略)"
echo "=========================================="
python scripts/train_fedmem.py \
    $COMMON_ARGS \
    --save_dir "$SAVE_DIR_BASE/both" \
    --init_bias_for_sasrec \
    --sasrec_bias_value 5.0 \
    --partial_aggregation_warmup_rounds 20 \
    | tee "$SAVE_DIR_BASE/both.log"

if [ $? -ne 0 ]; then
    echo "❌ Strategy1+2 训练失败！"
    exit 1
fi
echo "✅ Strategy1+2 完成"

echo ""
echo "=========================================="
echo "实验完成！"
echo "=========================================="
echo ""
echo "结果汇总："
echo ""

# 创建结果对比脚本
python << 'PYTHON_SCRIPT'
import json
import os

base_dir = os.environ.get('SAVE_DIR_BASE', 'checkpoints/full_experiment_latest')

configs = ['baseline', 'strategy1', 'strategy2', 'both']
config_names = {
    'baseline': 'Baseline (无优化)',
    'strategy1': 'Strategy1 (Router Bias)',
    'strategy2': 'Strategy2 (Partial Agg)',
    'both': 'Strategy1+2 (组合)'
}

print(f"{'配置':<25} {'HR@5':<10} {'HR@10':<10} {'NDCG@10':<10}")
print("="*60)

results = {}
for config in configs:
    history_file = f"{base_dir}/{config}/train_history.json"
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            history = json.load(f)
            test_metrics = history.get('test_metrics', {})
            hr5 = test_metrics.get('HR@5', 0)
            hr10 = test_metrics.get('HR@10', 0)
            ndcg10 = test_metrics.get('NDCG@10', 0)

            results[config] = {
                'HR@5': hr5,
                'HR@10': hr10,
                'NDCG@10': ndcg10
            }

            print(f"{config_names[config]:<25} {hr5:<10.4f} {hr10:<10.4f} {ndcg10:<10.4f}")
    else:
        print(f"{config_names[config]:<25} {'N/A':<10} {'N/A':<10} {'N/A':<10}")

print("")
print("="*60)

# 找出最佳配置
if results:
    best_config = max(results.items(), key=lambda x: x[1]['HR@10'])
    print(f"\n最佳配置: {config_names[best_config[0]]}")
    print(f"  HR@10: {best_config[1]['HR@10']:.4f}")
    print(f"  NDCG@10: {best_config[1]['NDCG@10']:.4f}")

# 保存结果
summary_file = f"{base_dir}/results_summary.json"
with open(summary_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n结果已保存到: {summary_file}")

PYTHON_SCRIPT

export SAVE_DIR_BASE="$SAVE_DIR_BASE"
python << 'PYTHON_SCRIPT'
import json
import os

base_dir = os.environ.get('SAVE_DIR_BASE', 'checkpoints/full_experiment_latest')

configs = ['baseline', 'strategy1', 'strategy2', 'both']
config_names = {
    'baseline': 'Baseline (无优化)',
    'strategy1': 'Strategy1 (Router Bias)',
    'strategy2': 'Strategy2 (Partial Agg)',
    'both': 'Strategy1+2 (组合)'
}

print(f"{'配置':<25} {'HR@5':<10} {'HR@10':<10} {'NDCG@10':<10}")
print("="*60)

results = {}
for config in configs:
    history_file = f"{base_dir}/{config}/train_history.json"
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            history = json.load(f)
            test_metrics = history.get('test_metrics', {})
            hr5 = test_metrics.get('HR@5', 0)
            hr10 = test_metrics.get('HR@10', 0)
            ndcg10 = test_metrics.get('NDCG@10', 0)

            results[config] = {
                'HR@5': hr5,
                'HR@10': hr10,
                'NDCG@10': ndcg10
            }

            print(f"{config_names[config]:<25} {hr5:<10.4f} {hr10:<10.4f} {ndcg10:<10.4f}")
    else:
        print(f"{config_names[config]:<25} {'N/A':<10} {'N/A':<10} {'N/A':<10}")

print("")
print("="*60)

# 找出最佳配置
if results:
    best_config = max(results.items(), key=lambda x: x[1]['HR@10'])
    print(f"\n最佳配置: {config_names[best_config[0]]}")
    print(f"  HR@10: {best_config[1]['HR@10']:.4f}")
    print(f"  NDCG@10: {best_config[1]['NDCG@10']:.4f}")

# 保存结果
summary_file = f"{base_dir}/results_summary.json"
with open(summary_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n结果已保存到: {summary_file}")

PYTHON_SCRIPT

echo ""
echo "详细日志保存在:"
echo "  $SAVE_DIR_BASE/*.log"
echo ""
echo "训练历史保存在:"
echo "  $SAVE_DIR_BASE/*/train_history.json"
echo ""
echo "=========================================="
