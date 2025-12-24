#!/bin/bash
# 优化实验脚本 - 目标 HR@10 = 0.60-0.70

echo "=========================================="
echo "FedDMMR 优化实验"
echo "目标: HR@10 = 0.60-0.70"
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

# 保存目录
SAVE_DIR_BASE="checkpoints/optimized_experiment_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$SAVE_DIR_BASE"

echo "结果将保存到: $SAVE_DIR_BASE"
echo ""

# ========================================
# 实验 1: 全库评估 (预期 HR@10 ≈ 0.50)
# ========================================
echo "=========================================="
echo "[1/3] 实验1: 使用全库评估（预期提升10-15%）"
echo "=========================================="

python scripts/train_fedmem.py \
    --data_dir "$DATA_DIR" \
    --data_file "$DATA_FILE" \
    --visual_file "$VISUAL_FILE" \
    --text_file "$TEXT_FILE" \
    --num_rounds 30 \
    --client_fraction 0.2 \
    --learning_rate 0.001 \
    --batch_size 32 \
    --init_bias_for_sasrec \
    --sasrec_bias_value 5.0 \
    --partial_aggregation_warmup_rounds 20 \
    --use_negative_sampling False \
    --save_dir "$SAVE_DIR_BASE/exp1_full_library" \
    | tee "$SAVE_DIR_BASE/exp1_full_library.log"

if [ $? -ne 0 ]; then
    echo "❌ 实验1失败"
    exit 1
fi
echo "✅ 实验1完成"
echo ""

# ========================================
# 实验 2: 优化超参数 (预期 HR@10 ≈ 0.58)
# ========================================
echo "=========================================="
echo "[2/3] 实验2: 优化超参数（增大模型+学习率）"
echo "=========================================="

python scripts/train_fedmem.py \
    --data_dir "$DATA_DIR" \
    --data_file "$DATA_FILE" \
    --visual_file "$VISUAL_FILE" \
    --text_file "$TEXT_FILE" \
    --sasrec_hidden_dim 512 \
    --sasrec_num_blocks 3 \
    --moe_num_heads 8 \
    --num_rounds 50 \
    --client_fraction 0.2 \
    --learning_rate 0.005 \
    --batch_size 64 \
    --patience 15 \
    --init_bias_for_sasrec \
    --sasrec_bias_value 8.0 \
    --partial_aggregation_warmup_rounds 35 \
    --use_negative_sampling False \
    --save_dir "$SAVE_DIR_BASE/exp2_optimized_hyperparams" \
    | tee "$SAVE_DIR_BASE/exp2_optimized_hyperparams.log"

if [ $? -ne 0 ]; then
    echo "❌ 实验2失败"
    exit 1
fi
echo "✅ 实验2完成"
echo ""

# ========================================
# 实验 3: 组合优化 (预期 HR@10 ≈ 0.60+)
# ========================================
echo "=========================================="
echo "[3/3] 实验3: 最佳组合配置（目标 HR@10 > 0.60）"
echo "=========================================="

python scripts/train_fedmem.py \
    --data_dir "$DATA_DIR" \
    --data_file "$DATA_FILE" \
    --visual_file "$VISUAL_FILE" \
    --text_file "$TEXT_FILE" \
    --sasrec_hidden_dim 512 \
    --sasrec_num_blocks 3 \
    --sasrec_num_heads 8 \
    --moe_num_heads 8 \
    --num_rounds 50 \
    --client_fraction 0.3 \
    --learning_rate 0.005 \
    --batch_size 64 \
    --patience 15 \
    --weight_decay 1e-6 \
    --memory_capacity 100 \
    --init_bias_for_sasrec \
    --sasrec_bias_value 10.0 \
    --partial_aggregation_warmup_rounds 40 \
    --use_negative_sampling False \
    --save_dir "$SAVE_DIR_BASE/exp3_best_config" \
    | tee "$SAVE_DIR_BASE/exp3_best_config.log"

if [ $? -ne 0 ]; then
    echo "❌ 实验3失败"
    exit 1
fi
echo "✅ 实验3完成"
echo ""

# ========================================
# 结果汇总
# ========================================
echo "=========================================="
echo "实验完成！汇总结果："
echo "=========================================="
echo ""

python3 << 'EOF'
import json
import os

base_dir = os.environ.get('SAVE_DIR_BASE', '')

experiments = [
    ('exp1_full_library', '实验1: 全库评估', '0.50'),
    ('exp2_optimized_hyperparams', '实验2: 优化超参数', '0.58'),
    ('exp3_best_config', '实验3: 最佳组合', '0.60+')
]

print(f"{'实验':<40} {'HR@10':<10} {'预期':<10} {'状态'}")
print("="*80)

for exp_dir, exp_name, expected in experiments:
    history_file = f"{base_dir}/{exp_dir}/train_history.json"
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            history = json.load(f)
            hr10 = history['test_metrics']['HR@10']
            status = "✅ 达标" if hr10 >= 0.60 else "⚠️  偏低"
            print(f"{exp_name:<40} {hr10:<10.4f} {expected:<10} {status}")
    else:
        print(f"{exp_name:<40} {'N/A':<10} {expected:<10} ❌ 失败")

print()
print("="*80)
EOF

export SAVE_DIR_BASE="$SAVE_DIR_BASE"

python3 << 'EOF'
import json
import os

base_dir = os.environ.get('SAVE_DIR_BASE', '')

experiments = [
    ('exp1_full_library', '实验1: 全库评估', 0.50),
    ('exp2_optimized_hyperparams', '实验2: 优化超参数', 0.58),
    ('exp3_best_config', '实验3: 最佳组合', 0.60)
]

print(f"{'实验':<40} {'HR@10':<10} {'预期':<10} {'状态'}")
print("="*80)

best_hr10 = 0
best_exp = None

for exp_dir, exp_name, expected in experiments:
    history_file = f"{base_dir}/{exp_dir}/train_history.json"
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            history = json.load(f)
            hr10 = history['test_metrics']['HR@10']
            if hr10 > best_hr10:
                best_hr10 = hr10
                best_exp = exp_name
            status = "✅ 达标" if hr10 >= 0.60 else "⚠️  偏低" if hr10 >= 0.50 else "❌ 偏低"
            print(f"{exp_name:<40} {hr10:<10.4f} {expected:<10} {status}")
    else:
        print(f"{exp_name:<40} {'N/A':<10} {expected:<10} ❌ 失败")

print()
print("="*80)
print(f"最佳结果: {best_exp} - HR@10 = {best_hr10:.4f}")
print("="*80)
print()

if best_hr10 >= 0.60:
    print("🎉 恭喜！已达到目标 HR@10 >= 0.60")
elif best_hr10 >= 0.50:
    print("✅ 进展良好，已提升至 HR@10 >= 0.50")
    print("建议：继续尝试架构改进（移除L2归一化、分数级融合）")
else:
    print("⚠️  结果仍然偏低，建议：")
    print("  1. 检查数据预处理是否正确")
    print("  2. 验证多模态特征质量")
    print("  3. 考虑架构级别的改进")
EOF

echo ""
echo "详细日志保存在: $SAVE_DIR_BASE/*.log"
echo "训练历史保存在: $SAVE_DIR_BASE/*/train_history.json"
echo ""
echo "=========================================="
