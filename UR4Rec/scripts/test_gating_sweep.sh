#!/bin/bash
# Gating Weight Sweep Experiment
# 测试不同gating_init值对模型性能的影响

echo "=========================================="
echo "Gating Weight Sweep Experiment"
echo "目标: 找到最佳的gating_init值"
echo "=========================================="
echo ""

cd /Users/admin/Desktop/MLLM/UR4Rec

# 激活虚拟环境
source /Users/admin/Desktop/MLLM/venv/bin/activate

# 数据文件
DATA_DIR="data"
DATA_FILE="ml100k_ratings_processed.dat"
VISUAL_FILE="clip_features_fixed.pt"
TEXT_FILE="item_text_features.pt"

# 实验配置
NUM_ROUNDS=30
CLIENT_FRACTION=0.2
LEARNING_RATE=0.001
BATCH_SIZE=32

# 保存目录
SAVE_DIR_BASE="checkpoints/gating_sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$SAVE_DIR_BASE"

echo "结果将保存到: $SAVE_DIR_BASE"
echo ""

# 测试不同的gating_init值
GATING_VALUES=(0.0 0.01 0.05 0.1 0.2 0.5)

for GATING in "${GATING_VALUES[@]}"; do
    echo "=========================================="
    echo "测试 gating_init = $GATING"
    echo "=========================================="

    python scripts/train_fedmem.py \
        --data_dir "$DATA_DIR" \
        --data_file "$DATA_FILE" \
        --visual_file "$VISUAL_FILE" \
        --text_file "$TEXT_FILE" \
        --num_rounds $NUM_ROUNDS \
        --client_fraction $CLIENT_FRACTION \
        --learning_rate $LEARNING_RATE \
        --batch_size $BATCH_SIZE \
        --gating_init $GATING \
        --save_dir "$SAVE_DIR_BASE/gating_${GATING}" \
        | tee "$SAVE_DIR_BASE/gating_${GATING}.log"

    if [ $? -ne 0 ]; then
        echo "❌ gating_init=$GATING 失败"
        exit 1
    fi
    echo "✅ gating_init=$GATING 完成"
    echo ""
done

# 汇总结果
echo "=========================================="
echo "实验完成！汇总结果："
echo "=========================================="
echo ""

export SAVE_DIR_BASE="$SAVE_DIR_BASE"

python3 << 'EOF'
import json
import os

base_dir = os.environ.get('SAVE_DIR_BASE', '')
gating_values = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]

print(f"{'Gating Init':<15} {'HR@10':<10} {'NDCG@10':<10} {'状态'}")
print("="*60)

best_hr10 = 0
best_gating = None
results = []

for gating in gating_values:
    history_file = f"{base_dir}/gating_{gating}/train_history.json"
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            history = json.load(f)
            hr10 = history['test_metrics']['HR@10']
            ndcg10 = history['test_metrics']['NDCG@10']

            results.append({
                'gating': gating,
                'hr10': hr10,
                'ndcg10': ndcg10
            })

            if hr10 > best_hr10:
                best_hr10 = hr10
                best_gating = gating

            status = "✅ 最佳" if gating == best_gating else ""
            print(f"{gating:<15.2f} {hr10:<10.4f} {ndcg10:<10.4f} {status}")
    else:
        print(f"{gating:<15.2f} {'N/A':<10} {'N/A':<10} ❌ 失败")

print()
print("="*60)
print(f"最佳配置: gating_init = {best_gating}")
print(f"最佳HR@10 = {best_hr10:.4f}")
print("="*60)
print()

# 保存结果
results_file = f"{base_dir}/sweep_results.json"
with open(results_file, 'w') as f:
    json.dump({
        'best_gating': best_gating,
        'best_hr10': best_hr10,
        'all_results': results
    }, f, indent=2)

print(f"详细结果已保存到: {results_file}")

# 分析趋势
print("\n建议:")
if best_gating == 0.0:
    print("  - gating=0.0最佳，说明多模态信息可能引入噪声")
    print("  - 建议检查多模态特征质量")
elif best_gating >= 0.5:
    print("  - gating>=0.5最佳，说明多模态信息非常有用")
    print("  - 建议尝试更大的gating值（0.8, 1.0）")
elif best_gating <= 0.05:
    print("  - 最佳gating值很小，说明需要谨慎注入多模态信息")
    print("  - 建议保持保守的gating策略")
else:
    print("  - gating值在中等范围，说明架构设计合理")
    print(f"  - 建议在{best_gating}附近微调")

EOF

echo ""
echo "详细日志保存在: $SAVE_DIR_BASE/*.log"
echo "训练历史保存在: $SAVE_DIR_BASE/*/train_history.json"
echo ""
echo "=========================================="
