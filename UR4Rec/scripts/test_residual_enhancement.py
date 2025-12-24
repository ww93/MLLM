#!/usr/bin/env python3
"""
快速测试脚本：验证Residual Enhancement架构是否正确工作

测试内容:
1. 模型能否正确初始化
2. Router是否输出2个权重（Visual, Semantic）
3. Gating weight是否存在且可学习
4. Forward pass是否正确工作
5. 输出形状是否符合预期
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from models.ur4rec_v2_moe import UR4RecV2MoE

def test_residual_enhancement():
    """测试Residual Enhancement架构"""

    print("=" * 80)
    print("Residual Enhancement 架构测试")
    print("=" * 80)

    # 测试参数
    num_items = 100
    batch_size = 4
    seq_len = 10
    num_candidates = 10
    visual_dim = 512
    text_dim = 384
    hidden_dim = 64

    print(f"\n[1/5] 初始化模型...")
    print(f"  - num_items: {num_items}")
    print(f"  - hidden_dim: {hidden_dim}")
    print(f"  - gating_init: 0.1")

    model = UR4RecV2MoE(
        num_items=num_items,
        sasrec_hidden_dim=hidden_dim,
        sasrec_num_blocks=2,
        sasrec_num_heads=2,
        visual_dim=visual_dim,
        text_dim=text_dim,
        moe_hidden_dim=hidden_dim,
        moe_num_heads=2,
        gating_init=0.1,
        device="cpu"
    )

    print("✓ 模型初始化成功")
    print(f"  总参数数量: {sum(p.numel() for p in model.parameters()):,}")

    # 验证架构
    print(f"\n[2/5] 验证架构组件...")

    # 检查gating_weight
    assert hasattr(model, 'gating_weight'), "❌ 缺少gating_weight"
    assert isinstance(model.gating_weight, torch.nn.Parameter), "❌ gating_weight不是Parameter"
    print(f"✓ Gating weight存在: {model.gating_weight.item():.4f}")

    # 检查LayerNorm
    assert hasattr(model, 'seq_layernorm'), "❌ 缺少seq_layernorm"
    assert hasattr(model, 'vis_layernorm'), "❌ 缺少vis_layernorm"
    assert hasattr(model, 'sem_layernorm'), "❌ 缺少sem_layernorm"
    print("✓ 所有LayerNorm存在")

    # 检查Router
    assert model.router.num_experts == 2, f"❌ Router专家数应为2，实际为{model.router.num_experts}"
    print(f"✓ Router控制专家数: {model.router.num_experts}")

    print(f"\n[3/5] 测试Router输出...")
    # 测试Router输出形状
    dummy_item_emb = torch.randn(batch_size, num_candidates, hidden_dim)
    router_output = model.router(dummy_item_emb)
    expected_shape = (batch_size, num_candidates, 2)
    assert router_output.shape == expected_shape, \
        f"❌ Router输出形状错误: 期望{expected_shape}，实际{router_output.shape}"
    print(f"✓ Router输出形状正确: {router_output.shape}")
    print(f"  示例权重: Visual={router_output[0, 0, 0]:.4f}, Semantic={router_output[0, 0, 1]:.4f}")

    print(f"\n[4/5] 测试完整forward pass...")

    # 准备输入数据
    input_seq = torch.randint(1, num_items, (batch_size, seq_len))
    target_items = torch.randint(1, num_items, (batch_size, num_candidates))
    target_visual = torch.randn(batch_size, num_candidates, visual_dim)
    memory_visual = torch.randn(batch_size, 5, visual_dim)
    memory_text = torch.randn(batch_size, 5, text_dim)

    # Forward pass
    scores, info = model(
        user_ids=None,
        input_seq=input_seq,
        target_items=target_items,
        target_visual=target_visual,
        memory_visual=memory_visual,
        memory_text=memory_text,
        return_components=True
    )

    print(f"✓ Forward pass成功")
    print(f"  输出形状: {scores.shape}")
    assert scores.shape == (batch_size, num_candidates), \
        f"❌ 输出形状错误: 期望{(batch_size, num_candidates)}，实际{scores.shape}"

    print(f"\n[5/5] 验证返回信息...")

    # 检查必要的keys
    required_keys = [
        'router_weights', 'expert_usage', 'lb_loss',
        'w_vis', 'w_sem', 'gating_weight',
        'seq_out', 'vis_out', 'sem_out',
        'seq_out_norm', 'vis_out_norm', 'sem_out_norm',
        'auxiliary_repr', 'fused_repr'
    ]

    for key in required_keys:
        assert key in info, f"❌ 缺少返回信息: {key}"

    print("✓ 所有必要的返回信息都存在")

    # 检查形状
    assert info['router_weights'].shape == (batch_size, num_candidates, 2), \
        f"❌ router_weights形状错误"
    assert info['expert_usage'].shape == (2,), \
        f"❌ expert_usage形状错误"

    print(f"\n关键信息:")
    print(f"  - Gating Weight: {info['gating_weight']:.4f}")
    print(f"  - Visual Expert 平均权重: {info['w_vis']:.4f}")
    print(f"  - Semantic Expert 平均权重: {info['w_sem']:.4f}")
    print(f"  - Expert Usage: Visual={info['expert_usage'][0]:.4f}, Semantic={info['expert_usage'][1]:.4f}")
    print(f"  - Load Balance Loss: {info['lb_loss']:.6f}")

    # 验证没有w_seq（因为SASRec不再参与路由）
    assert 'w_seq' not in info, "❌ info中不应该包含w_seq"
    print(f"  ✓ SASRec不参与路由竞争（没有w_seq）")

    print("\n" + "=" * 80)
    print("✅ 所有测试通过！Residual Enhancement架构工作正常")
    print("=" * 80)

    # 打印架构总结
    print("\n架构总结:")
    print("  1. SASRec作为骨干，输出直接保留")
    print("  2. Router仅控制2个辅助专家（Visual, Semantic）")
    print("  3. Gating weight控制辅助信息注入强度")
    print("  4. 融合公式: final = seq_norm + gating * (w_vis*vis_norm + w_sem*sem_norm)")
    print("  5. Load balance loss仅考虑2个辅助专家")
    print()

if __name__ == "__main__":
    test_residual_enhancement()
