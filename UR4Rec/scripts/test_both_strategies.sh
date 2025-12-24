#!/bin/bash
# 综合测试：同时启用策略1和策略2
# 对比实验：baseline vs strategy1 vs strategy2 vs strategy1+2

echo "=========================================="
echo "综合测试: Router Bias + Partial Aggregation"
echo "=========================================="
echo ""
echo "实验设置："
echo "  - 数据集: ML-100K"
echo "  - 对比4个配置："
echo "    1. Baseline (无优化)"
echo "    2. Strategy1 (仅Router Bias Initialization)"
echo "    3. Strategy2 (仅Partial Aggregation)"
echo "    4. Strategy1+2 (两种策略结合)"
echo ""

# 设置数据路径
DATA_DIR="data/ml-100k"
SAVE_DIR_BASELINE="checkpoints/combined_baseline"
SAVE_DIR_S1="checkpoints/combined_strategy1"
SAVE_DIR_S2="checkpoints/combined_strategy2"
SAVE_DIR_BOTH="checkpoints/combined_both"

# 检查数据是否存在
if [ ! -d "$DATA_DIR" ]; then
    echo "错误: 数据目录不存在: $DATA_DIR"
    echo "请先准备 ML-100K 数据集"
    exit 1
fi

# 创建保存目录
mkdir -p "$SAVE_DIR_BASELINE"
mkdir -p "$SAVE_DIR_S1"
mkdir -p "$SAVE_DIR_S2"
mkdir -p "$SAVE_DIR_BOTH"

# 通用参数
COMMON_ARGS="--data_dir $DATA_DIR \
    --data_file u.data \
    --num_rounds 30 \
    --client_fraction 0.2 \
    --local_epochs 1 \
    --learning_rate 0.001 \
    --batch_size 32 \
    --memory_capacity 50 \
    --enable_prototype_aggregation \
    --patience 10 \
    --seed 42"

echo "=========================================="
echo "[1/4] Baseline (无优化)"
echo "=========================================="
python scripts/train_fedmem.py \
    $COMMON_ARGS \
    --save_dir "$SAVE_DIR_BASELINE"

echo ""
echo "=========================================="
echo "[2/4] Strategy1 (Router Bias Initialization)"
echo "=========================================="
python scripts/train_fedmem.py \
    $COMMON_ARGS \
    --save_dir "$SAVE_DIR_S1" \
    --init_bias_for_sasrec \
    --sasrec_bias_value 5.0

echo ""
echo "=========================================="
echo "[3/4] Strategy2 (Partial Aggregation)"
echo "=========================================="
python scripts/train_fedmem.py \
    $COMMON_ARGS \
    --save_dir "$SAVE_DIR_S2" \
    --partial_aggregation_warmup_rounds 20

echo ""
echo "=========================================="
echo "[4/4] Strategy1+2 (两种策略结合)"
echo "=========================================="
python scripts/train_fedmem.py \
    $COMMON_ARGS \
    --save_dir "$SAVE_DIR_BOTH" \
    --init_bias_for_sasrec \
    --sasrec_bias_value 5.0 \
    --partial_aggregation_warmup_rounds 20

echo ""
echo "=========================================="
echo "综合测试完成！"
echo "=========================================="
echo ""
echo "结果比较："
echo ""

# 定义一个函数来打印结果
print_results() {
    local dir=$1
    local name=$2

    echo "$name:"
    if [ -f "$dir/train_history.json" ]; then
        python -c "
import json
with open('$dir/train_history.json', 'r') as f:
    history = json.load(f)
    test_metrics = history.get('test_metrics', {})
    print(f\"  HR@5:    {test_metrics.get('HR@5', 0.0):.4f}\")
    print(f\"  HR@10:   {test_metrics.get('HR@10', 0.0):.4f}\")
    print(f\"  NDCG@5:  {test_metrics.get('NDCG@5', 0.0):.4f}\")
    print(f\"  NDCG@10: {test_metrics.get('NDCG@10', 0.0):.4f}\")
"
    else
        echo "  结果文件不存在"
    fi
    echo ""
}

print_results "$SAVE_DIR_BASELINE" "Baseline (无优化)"
print_results "$SAVE_DIR_S1" "Strategy1 (Router Bias)"
print_results "$SAVE_DIR_S2" "Strategy2 (Partial Agg)"
print_results "$SAVE_DIR_BOTH" "Strategy1+2 (结合)"

echo "=========================================="
echo "详细结果保存在:"
echo "  Baseline:    $SAVE_DIR_BASELINE/train_history.json"
echo "  Strategy1:   $SAVE_DIR_S1/train_history.json"
echo "  Strategy2:   $SAVE_DIR_S2/train_history.json"
echo "  Strategy1+2: $SAVE_DIR_BOTH/train_history.json"
echo "=========================================="
