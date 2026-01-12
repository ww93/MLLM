"""
验证Stage 3性能崩溃的假设：
1. Round 0 (纯checkpoint): 应该接近Stage 2 final (~0.51)
2. Round 1 (训练1轮后): 性能下降到0.38
3. 检查SASRec参数的变化量

如果Round 0性能高，Round 1性能低，说明问题是**训练过程破坏了Stage 2的配合**
"""
import torch
import numpy as np

print("="*60)
print("Stage 3 性能崩溃分析")
print("="*60)

# 1. 检查Stage 2 checkpoint
print("\n1. Stage 2 checkpoint (基线):")
stage2_ckpt = torch.load('UR4Rec/checkpoints/stage2_lightweight/fedmem_model.pt',
                          map_location='cpu', weights_only=False)
stage2_state = stage2_ckpt.get('model_state_dict', stage2_ckpt)

print(f"  gating_weight: {stage2_state['gating_weight'].item():.6f}")
print(f"  Test HR@10: {stage2_ckpt.get('test_metrics', {}).get('HR@10', 'N/A')}")

# 2. 检查Stage 3 Round 1 checkpoint (如果存在)
import os
if os.path.exists('UR4Rec/checkpoints/stage3_moe/fedmem_model.pt'):
    print("\n2. Stage 3 Round 1 checkpoint:")
    stage3_ckpt = torch.load('UR4Rec/checkpoints/stage3_moe/fedmem_model.pt',
                              map_location='cpu', weights_only=False)
    stage3_state = stage3_ckpt.get('model_state_dict', stage3_ckpt)

    print(f"  gating_weight: {stage3_state['gating_weight'].item():.6f}")
    print(f"  变化: {stage3_state['gating_weight'].item() - stage2_state['gating_weight'].item():.6f}")

    # 3. 计算SASRec参数的变化量
    print("\n3. SASRec参数变化分析:")
    sasrec_changes = []
    for key in stage2_state.keys():
        if 'sasrec' in key.lower() and key in stage3_state:
            if stage2_state[key].shape == stage3_state[key].shape:
                diff = torch.norm(stage3_state[key] - stage2_state[key]).item()
                relative_diff = diff / (torch.norm(stage2_state[key]).item() + 1e-8)
                sasrec_changes.append((key, diff, relative_diff))

    if sasrec_changes:
        sasrec_changes.sort(key=lambda x: x[2], reverse=True)  # 按相对变化排序
        print(f"  SASRec参数总数: {len(sasrec_changes)}")
        print(f"  Top 5 变化最大的参数:")
        for key, abs_diff, rel_diff in sasrec_changes[:5]:
            print(f"    {key}: 相对变化={rel_diff:.6f}, 绝对变化={abs_diff:.6f}")

        # 平均变化
        avg_rel_change = np.mean([x[2] for x in sasrec_changes])
        print(f"  平均相对变化: {avg_rel_change:.6f}")

        # 判断
        if avg_rel_change > 0.01:
            print(f"\n⚠️  SASRec参数显著变化（平均相对变化 > 1%）")
            print(f"  → 这解释了为什么性能下降：投影层不再匹配新的SASRec输出")
        else:
            print(f"\n✓ SASRec参数变化较小（平均相对变化 < 1%）")
            print(f"  → 问题可能在其他地方（Router? Experts?）")

    # 4. 检查投影层变化
    print("\n4. 投影层参数变化分析:")
    proj_changes = []
    for key in ['visual_proj.weight', 'visual_proj.bias',
                'text_proj.weight', 'text_proj.bias']:
        if key in stage2_state and key in stage3_state:
            diff = torch.norm(stage3_state[key] - stage2_state[key]).item()
            relative_diff = diff / (torch.norm(stage2_state[key]).item() + 1e-8)
            proj_changes.append((key, diff, relative_diff))
            print(f"  {key}: 相对变化={relative_diff:.6f}")

    avg_proj_change = np.mean([x[2] for x in proj_changes]) if proj_changes else 0
    print(f"  投影层平均相对变化: {avg_proj_change:.6f}")

    # 5. 总结
    print("\n" + "="*60)
    print("诊断结论:")
    print("="*60)

    if avg_rel_change > 0.01:
        print("✗ 问题确认：SASRec参数在Stage 3训练中显著改变")
        print("  → Stage 2学到的投影层是针对'冻结的SASRec'优化的")
        print("  → Stage 3解冻SASRec后，SASRec输出分布改变")
        print("  → 投影层权重不再匹配，导致融合效果变差")
        print()
        print("推荐解决方案：")
        print("  方案1: Stage 3继续冻结SASRec（至少前几轮）")
        print("  方案2: Stage 3使用极小的学习率（1e-5）")
        print("  方案3: Stage 3分阶段：先训练Router（冻结投影层+SASRec），再逐步解冻")
    else:
        print("? 问题未明确：SASRec参数变化不大，但性能仍下降")
        print("  可能原因：")
        print("  1. Router权重初始化不当（需要检查Router变化）")
        print("  2. Experts参数变化（需要检查Experts变化）")
        print("  3. 评估时的随机性（客户端采样、negative sampling）")
else:
    print("\n2. Stage 3 checkpoint尚未保存")
    print("  → 请等待Round 1训练完成")
