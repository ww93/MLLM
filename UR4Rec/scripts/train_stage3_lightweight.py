"""
Stage 3: MoE Fine-tuning (MoE微调)
第三阶段：MoE全局微调

目标: 学习Router权重，实现多模态场景自适应融合
对象:
  - 冻结: Item Embedding（保持ID空间稳定）
  - 训练: SASRec Transformer + Projectors + Experts + CrossModalFusion + Router
预期: HR@10 > Stage 1 (0.65-0.75)，多模态信息充分利用

使用方法:
    python UR4Rec/scripts/train_stage3_lightweight.py
"""
import os
import sys
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    """第三阶段：MoE全局微调"""

    print("="*60)
    print("Stage 3: MoE Fine-tuning")
    print("第三阶段：MoE全局微调")
    print("="*60)
    print()

    # 检查Stage 1 & 2 checkpoints
    script_dir = Path(__file__).parent.parent
    stage1_checkpoint = str(script_dir / "checkpoints" / "stage1_backbone" / "fedmem_model.pt")
    stage2_checkpoint = str(script_dir / "checkpoints" / "stage2_lightweight" / "fedmem_model.pt")

    missing_checkpoints = []
    if not os.path.exists(stage1_checkpoint):
        missing_checkpoints.append(("Stage 1", stage1_checkpoint))
    if not os.path.exists(stage2_checkpoint):
        missing_checkpoints.append(("Stage 2", stage2_checkpoint))

    if missing_checkpoints:
        print(f"❌ 错误: 以下checkpoint缺失:")
        for name, path in missing_checkpoints:
            print(f"  - {name}: {path}")
        print()
        print("请先运行前序阶段:")
        if not os.path.exists(stage1_checkpoint):
            print(f"  1. python UR4Rec/scripts/train_stage1_backbone.py")
        if not os.path.exists(stage2_checkpoint):
            print(f"  2. python UR4Rec/scripts/train_stage2_lightweight.py")
        return 1

    print(f"✓ 找到Stage 1 checkpoint: {stage1_checkpoint}")
    print(f"✓ 找到Stage 2 checkpoint: {stage2_checkpoint}")
    print()

    config = {
        # 数据配置
        "data_dir": "UR4Rec/data/ml-1m",
        "data_file": "subset_ratings.dat",

        # [Stage 3] 加载多模态特征
        "visual_file": "clip_features.pt",
        "text_file": "text_features.pt",

        # [Stage 3] 训练阶段
        "stage": "finetune_moe",
        "stage1_checkpoint": stage1_checkpoint,
        "stage2_checkpoint": stage2_checkpoint,

        # 模型配置
        "model_type": "moe",
        "num_items": 3953,
        "sasrec_hidden_dim": 128,
        "sasrec_num_blocks": 2,
        "sasrec_num_heads": 4,
        "max_seq_len": 50,

        # MoE参数
        "moe_num_heads": 8,
        "retriever_output_dim": 128,
        "gating_init": 0.1,              # Stage 3可以用更大的初始值

        # FedMem参数 (Two-tier Memory)
        "memory_capacity": 200,
        "surprise_threshold": 0.5,
        "contrastive_lambda": 0.05,      # Stage 3降低对齐损失权重
        "num_memory_prototypes": 5,

        # 联邦学习参数
        "num_rounds": 50,                # Stage 3需要更多轮数学习Router
        "client_fraction": 0.2,
        "local_epochs": 1,
        "patience": 15,                  # 更大的patience
        "partial_aggregation_warmup_rounds": 0,  # Stage 3禁用warmup

        # [Stage 3] 训练参数 - 更小的学习率
        "learning_rate": 5e-4,           # 比Stage 2小，避免破坏已对齐的投影层
        "weight_decay": 1e-4,
        "batch_size": 64,
        "num_negatives": 100,

        # 评估参数
        "use_negative_sampling": "true",
        "num_negatives_eval": 100,

        # 其他参数
        "seed": 42,
        "save_dir": "UR4Rec/checkpoints/stage3_moe",
        "verbose": ""
    }

    print("训练配置:")
    print(f"  阶段: Stage 3 - MoE Fine-tuning")
    print(f"  目标: 学习Router权重，全局微调所有组件")
    print()
    print("训练对象:")
    print(f"  ❄️  冻结: Item Embedding（保持ID空间稳定）")
    print(f"  🔥 训练: SASRec Transformer + Projectors + Experts + CrossModalFusion + Router")
    print()
    print("训练参数:")
    print(f"  轮数: {config['num_rounds']} (需要更多轮数学习Router)")
    print(f"  学习率: {config['learning_rate']:.0e} (小学习率，避免破坏Stage 2)")
    print(f"  Memory: Two-tier (ST: 50, LT: {config['memory_capacity']})")
    print(f"  对比学习权重: {config['contrastive_lambda']} (降低)")
    print(f"  早停patience: {config['patience']}")
    print()
    print("预期效果:")
    print(f"  Round 1-10: HR@10 ≈ 0.62-0.67 (继承Stage 2)")
    print(f"  Round 20-30: HR@10 ≈ 0.67-0.72 (Router开始生效)")
    print(f"  Round 40-50: HR@10 ≈ 0.70-0.75 (多模态充分融合)")
    print()
    print("关键指标:")
    print(f"  ✓ Router权重分布: 应该有差异化（不是均匀0.5/0.5）")
    print(f"  ✓ 负载均衡损失: < 0.1")
    print(f"  ✓ HR@10 > Stage 1 (验证多模态有效性)")
    print()
    print(f"保存位置: {config['save_dir']}/")
    print()

    # 检查数据文件
    data_path = os.path.join(config["data_dir"], config["data_file"])
    visual_path = os.path.join(config["data_dir"], config["visual_file"])
    text_path = os.path.join(config["data_dir"], config["text_file"])

    missing_files = []
    if not os.path.exists(data_path):
        missing_files.append(data_path)
    if not os.path.exists(visual_path):
        missing_files.append(visual_path)
    if not os.path.exists(text_path):
        missing_files.append(text_path)

    if missing_files:
        print(f"❌ 错误: 以下数据文件缺失:")
        for f in missing_files:
            print(f"  - {f}")
        print()
        print("请确保以下文件存在:")
        print(f"  1. {config['data_dir']}/{config['data_file']}")
        print(f"  2. {config['data_dir']}/{config['visual_file']} (CLIP特征)")
        print(f"  3. {config['data_dir']}/{config['text_file']} (SBERT特征)")
        return 1

    print("✓ 所有数据文件已就绪")
    print()

    # 构建命令
    cmd = ["python", "UR4Rec/scripts/train_fedmem.py"]

    for key, value in config.items():
        if value == "":
            cmd.append(f"--{key}")
        else:
            cmd.append(f"--{key}")
            cmd.append(str(value))

    print("执行命令:")
    print(" ".join(cmd))
    print()
    print("="*60)
    print("开始训练...")
    print("="*60)
    print()

    # 执行训练
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print()
        print("="*60)
        print("✓ Stage 3训练完成！")
        print("="*60)
        print()
        print("三阶段训练全部完成！")
        print()
        print("结果分析:")
        print(f"  1. 查看训练历史: {config['save_dir']}/train_history.json")
        print(f"  2. 对比三个阶段的性能:")
        print(f"     - Stage 1 (纯ID): HR@10 ≈ 0.60-0.70")
        print(f"     - Stage 2 (对齐): HR@10 ≈ 0.60-0.67")
        print(f"     - Stage 3 (MoE): HR@10 ≈ 0.65-0.75")
        print()
        print("如果Stage 3性能低于Stage 1:")
        print(f"  ⚠️  可能原因:")
        print(f"    1. 多模态特征质量较低")
        print(f"    2. Router权重学习不充分（查看lb_loss）")
        print(f"    3. 学习率过大，破坏了Stage 2的对齐")
        print(f"  解决方案:")
        print(f"    1. 降低学习率: --learning_rate 1e-4")
        print(f"    2. 增加训练轮数: --num_rounds 80")
        print(f"    3. 检查多模态特征文件")
    else:
        print()
        print("="*60)
        print("✗ 训练失败")
        print("="*60)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
