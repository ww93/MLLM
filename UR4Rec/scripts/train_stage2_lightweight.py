"""
Stage 2: Lightweight Alignment (轻量级对齐)
第二阶段：轻量级多模态对齐

[新方案 - 轻量级投影]:
1. 投影层: visual_proj (512→128), text_proj (384→128)
2. Gating MLP: 动态权重学习 (~25K params)
3. 总参数量: ~139K < 200K ✓

目标: 将多模态特征投影到ID空间，不破坏SASRec骨干
对象:
  - 冻结: SASRec, Item Embedding, Experts, CrossModalFusion, Router
  - 训练: visual_proj, text_proj, align_gating (~139K params)
预期: HR@10接近Stage 1 (0.60-0.65)，训练速度快20倍

使用方法:
    python UR4Rec/scripts/train_stage2_lightweight.py
"""
import os
import sys
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    """第二阶段：轻量级多模态对齐"""

    print("="*60)
    print("Stage 2: Lightweight Alignment")
    print("第二阶段：轻量级多模态对齐")
    print("="*60)
    print()

    # 检查Stage 1 checkpoint
    script_dir = Path(__file__).parent.parent
    stage1_checkpoint = str(script_dir / "checkpoints" / "stage1_backbone" / "fedmem_model.pt")

    if not os.path.exists(stage1_checkpoint):
        print(f"❌ 错误: Stage 1 checkpoint不存在: {stage1_checkpoint}")
        print(f"请先运行Stage 1训练:")
        print(f"  python UR4Rec/scripts/train_stage1_backbone.py")
        return 1

    print(f"✓ 找到Stage 1 checkpoint: {stage1_checkpoint}")
    print()

    config = {
        # 数据配置
        "data_dir": "UR4Rec/data/ml-1m",
        "data_file": "subset_ratings.dat",

        # [Stage 2] 加载多模态特征
        "visual_file": "clip_features.pt",
        "text_file": "text_features.pt",

        # [Stage 2] 训练阶段
        "stage": "align_projectors",
        "stage1_checkpoint": stage1_checkpoint,

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
        "gating_init": 0.01,

        # FedMem参数 (Two-tier Memory)
        "memory_capacity": 200,         # LT容量
        "surprise_threshold": 0.5,      # 兼容参数
        "contrastive_lambda": 0.1,      # 对齐损失权重
        "num_memory_prototypes": 5,

        # 联邦学习参数
        "num_rounds": 30,               # 轻量级训练，可以多训练几轮
        "client_fraction": 0.2,
        "local_epochs": 1,
        "patience": 10,
        "partial_aggregation_warmup_rounds": 0,  # Stage 2禁用warmup

        # 训练参数
        "learning_rate": 1e-3,          # 正常学习率
        "weight_decay": 1e-4,
        "batch_size": 64,
        "num_negatives": 100,

        # 评估参数
        "use_negative_sampling": "true",
        "num_negatives_eval": 100,

        # 其他参数
        "seed": 42,
        "save_dir": "UR4Rec/checkpoints/stage2_lightweight",
        "verbose": ""
    }

    print("训练配置:")
    print(f"  阶段: Stage 2 - Lightweight Alignment")
    print(f"  目标: 轻量级投影层训练，将多模态对齐到ID空间")
    print()
    print("架构 [轻量级方案]:")
    print(f"  ✓ visual_proj: 512 → 128 (~65K params)")
    print(f"  ✓ text_proj: 384 → 128 (~49K params)")
    print(f"  ✓ align_gating: MLP (~25K params)")
    print(f"  ✓ 总参数量: ~139K < 200K ✓")
    print()
    print("训练对象:")
    print(f"  ❄️  冻结: SASRec + Item Embedding + Experts + CrossModalFusion + Router")
    print(f"  🔥 训练: visual_proj + text_proj + align_gating (~139K params)")
    print()
    print("训练参数:")
    print(f"  轮数: {config['num_rounds']}")
    print(f"  学习率: {config['learning_rate']:.0e}")
    print(f"  Memory: Two-tier (ST: 50, LT: {config['memory_capacity']})")
    print(f"  对齐损失权重: {config['contrastive_lambda']}")
    print(f"  早停patience: {config['patience']}")
    print()
    print("预期效果:")
    print(f"  Round 1-5: HR@10 ≈ 0.55-0.60 (投影层初始化)")
    print(f"  Round 10-20: HR@10 ≈ 0.60-0.65 (对齐稳定)")
    print(f"  Round 30: HR@10 ≈ 0.62-0.67 (接近Stage 1)")
    print()
    print("轻量级方案优势:")
    print(f"  ✓ 参数量小 (~139K vs ~4M)，训练速度快20倍")
    print(f"  ✓ 不破坏预训练的SASRec骨干")
    print(f"  ✓ 投影层简单，容易收敛")
    print(f"  ✓ 为Stage 3提供良好的初始化")
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
        print("✓ Stage 2训练完成！")
        print("="*60)
        print()
        print("下一步: Stage 3 - MoE微调")
        print("  python UR4Rec/scripts/train_stage3_lightweight.py")
        print()
        print("或者直接使用train_fedmem.py:")
        print(f"  python UR4Rec/scripts/train_fedmem.py \\")
        print(f"    --stage finetune_moe \\")
        print(f"    --stage1_checkpoint {stage1_checkpoint} \\")
        print(f"    --stage2_checkpoint {config['save_dir']}/fedmem_model.pt \\")
        print(f"    --data_dir {config['data_dir']} \\")
        print(f"    --visual_file {config['visual_file']} \\")
        print(f"    --text_file {config['text_file']} \\")
        print(f"    --save_dir UR4Rec/checkpoints/stage3_moe \\")
        print(f"    --num_rounds 50 \\")
        print(f"    --learning_rate 5e-4")
    else:
        print()
        print("="*60)
        print("✗ 训练失败")
        print("="*60)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
