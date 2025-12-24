#!/bin/bash
# 测试策略2: Partial Aggregation
# 对比实验：baseline vs strategy2

echo "=========================================="
echo "策略2测试: Partial Aggregation"
echo "=========================================="
echo ""
echo "实验设置："
echo "  - 数据集: ML-100K"
echo "  - 策略: Partial Aggregation (前20轮只聚合SASRec参数)"
echo "  - 目的: 让Client在本地自由微调多模态部分，避免联邦聚合破坏Router"
echo ""

# 设置数据路径
DATA_DIR="data/ml-100k"
SAVE_DIR_BASELINE="checkpoints/strategy2_baseline"
SAVE_DIR_STRATEGY2="checkpoints/strategy2_partial_agg"

# 检查数据是否存在
if [ ! -d "$DATA_DIR" ]; then
    echo "错误: 数据目录不存在: $DATA_DIR"
    echo "请先准备 ML-100K 数据集"
    exit 1
fi

# 创建保存目录
mkdir -p "$SAVE_DIR_BASELINE"
mkdir -p "$SAVE_DIR_STRATEGY2"

echo "=========================================="
echo "[1/2] 运行 Baseline (全量聚合)"
echo "=========================================="
python scripts/train_fedmem.py \
    --data_dir "$DATA_DIR" \
    --data_file "u.data" \
    --save_dir "$SAVE_DIR_BASELINE" \
    --num_rounds 30 \
    --client_fraction 0.2 \
    --local_epochs 1 \
    --learning_rate 0.001 \
    --batch_size 32 \
    --memory_capacity 50 \
    --enable_prototype_aggregation \
    --patience 10 \
    --seed 42

echo ""
echo "=========================================="
echo "[2/2] 运行 Strategy2 (前20轮Partial Aggregation)"
echo "=========================================="
python scripts/train_fedmem.py \
    --data_dir "$DATA_DIR" \
    --data_file "u.data" \
    --save_dir "$SAVE_DIR_STRATEGY2" \
    --num_rounds 30 \
    --client_fraction 0.2 \
    --local_epochs 1 \
    --learning_rate 0.001 \
    --batch_size 32 \
    --memory_capacity 50 \
    --enable_prototype_aggregation \
    --patience 10 \
    --seed 42 \
    --partial_aggregation_warmup_rounds 20

echo ""
echo "=========================================="
echo "策略2测试完成！"
echo "=========================================="
echo ""
echo "结果比较："
echo ""
echo "Baseline (全量聚合) 测试结果:"
if [ -f "$SAVE_DIR_BASELINE/train_history.json" ]; then
    python -c "
import json
with open('$SAVE_DIR_BASELINE/train_history.json', 'r') as f:
    history = json.load(f)
    test_metrics = history.get('test_metrics', {})
    print(f\"  HR@5:  {test_metrics.get('HR@5', 0.0):.4f}\")
    print(f\"  HR@10: {test_metrics.get('HR@10', 0.0):.4f}\")
    print(f\"  NDCG@5:  {test_metrics.get('NDCG@5', 0.0):.4f}\")
    print(f\"  NDCG@10: {test_metrics.get('NDCG@10', 0.0):.4f}\")
"
fi

echo ""
echo "Strategy2 (Partial Aggregation) 测试结果:"
if [ -f "$SAVE_DIR_STRATEGY2/train_history.json" ]; then
    python -c "
import json
with open('$SAVE_DIR_STRATEGY2/train_history.json', 'r') as f:
    history = json.load(f)
    test_metrics = history.get('test_metrics', {})
    print(f\"  HR@5:  {test_metrics.get('HR@5', 0.0):.4f}\")
    print(f\"  HR@10: {test_metrics.get('HR@10', 0.0):.4f}\")
    print(f\"  NDCG@5:  {test_metrics.get('NDCG@5', 0.0):.4f}\")
    print(f\"  NDCG@10: {test_metrics.get('NDCG@10', 0.0):.4f}\")
"
fi

echo ""
echo "完整结果保存在:"
echo "  Baseline:  $SAVE_DIR_BASELINE/train_history.json"
echo "  Strategy2: $SAVE_DIR_STRATEGY2/train_history.json"
