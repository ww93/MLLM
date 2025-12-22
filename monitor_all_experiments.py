#!/usr/bin/env python3
"""
监测所有实验结果
"""
import json
import os
from pathlib import Path

experiments = [
    ("bugfix_full_test", "50 negs, 50 rounds"),
    ("sasrec_baseline", "50 negs, 30 rounds"),
    ("ml100k_no_l2norm", "50 negs, 10 rounds, no_l2norm"),
    ("bugfix_test", "50 negs, 10 rounds"),
    ("neg500_test", "500 negs, 20 rounds"),
    ("4star_test", "50 negs, 10 rounds, 4-star"),
    ("small_model_test", "32d model, 20 rounds"),
    ("router_fix_validation", "50 negs, 10 rounds, router weight mask fix"),
    ("sasrec_fallback_test", "50 negs, 10 rounds, SASRec fallback bypass"),
    ("centralized_test", "50 negs, 10 rounds, 100% clients (centralized)"),
]

print("=" * 100)
print("所有ML-100K实验结果监测")
print("=" * 100)
print()

# ItemPop baseline
print(f"{'ItemPop Baseline':<25} | HR@10: 8.80% (实测)")
print("-" * 100)

for exp_name, description in experiments:
    checkpoint_path = f"UR4Rec/checkpoints/{exp_name}/train_history.json"

    if not os.path.exists(checkpoint_path):
        print(f"{exp_name:<25} | {description:<40} | ❌ 未完成")
        continue

    try:
        with open(checkpoint_path, 'r') as f:
            data = json.load(f)

        test_metrics = data.get('test_metrics', {})
        hr10 = test_metrics.get('HR@10', 0) * 100
        hr5 = test_metrics.get('HR@5', 0) * 100
        hr20 = test_metrics.get('HR@20', 0) * 100
        ndcg10 = test_metrics.get('NDCG@10', 0) * 100
        mrr = test_metrics.get('MRR', 0) * 100

        # 最后一轮的val metrics
        val_metrics = data.get('val_metrics', [])
        if val_metrics:
            last_val = val_metrics[-1]
            val_hr10 = last_val.get('HR@10', 0) * 100
        else:
            val_hr10 = 0

        # 训练轮数
        num_rounds = len(data.get('round', []))

        # 状态标记
        if hr10 > 8.80:
            status = "✅"
        elif hr10 > 5:
            status = "⚠️ "
        else:
            status = "❌"

        print(f"{exp_name:<25} | {description:<40} | {status} HR@10: {hr10:5.2f}% | Val: {val_hr10:5.2f}% | Rounds: {num_rounds}")

    except Exception as e:
        print(f"{exp_name:<25} | {description:<40} | ⚠️  Error: {str(e)[:30]}")

print("=" * 100)
print()
print("图例:")
print("  ✅ HR@10 > 8.80% (优于ItemPop)")
print("  ⚠️  5% < HR@10 < 8.80% (有进展但仍低于ItemPop)")
print("  ❌ HR@10 < 5% (性能很差)")
print()
print("我的诊断结论:")
print("  根本原因: 训练使用50个负样本，但评估使用1682个全排序")
print("  解决方案: 使用500+个负样本接近全排序")
