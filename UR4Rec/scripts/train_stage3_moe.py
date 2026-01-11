"""
Stage 3: MoE Fine-tuning
第三阶段：MoE集成微调 - 学习"什么时候该用谁"

目标: 学习Router（什么时候用谁）
对象:
  - 冻结: Item Embedding (锚点)
  - 微调 (小LR): SASRec Transformer, Visual/Semantic Projectors
  - 全速训练: MoE Router
Loss: Rec + Contrastive + LB
预期HR@10: 0.70+ (冲击最佳性能)

使用方法:
    python UR4Rec/scripts/train_stage3_moe.py
"""
import os
import sys
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    """第三阶段：MoE集成微调"""

    print("="*60)
    print("Stage 3: MoE Fine-tuning")
    print("第三阶段：MoE集成微调")
    print("="*60)
    print()

    # 检查Stage 1 和 Stage 2 checkpoints
    script_dir = Path(__file__).parent.parent
    stage1_checkpoint = str(script_dir / "checkpoints" / "stage1_backbone" / "fedmem_model.pt")
    stage2_checkpoint = str(script_dir / "checkpoints" / "stage2_alignment" / "fedmem_model.pt")

    # [修复] Stage 1必须存在，Stage 2可选
    if not os.path.exists(stage1_checkpoint):
        print(f"❌ 错误: 缺少Stage 1 checkpoint: {stage1_checkpoint}")
        print(f"    请先运行: python UR4Rec/scripts/train_stage1_backbone.py")
        print()
        return 1

    print(f"✓ 找到Stage 1 checkpoint: {stage1_checkpoint}")

    # [修复] Stage 2是可选的（如果Stage 2失败，可以跳过）
    if os.path.exists(stage2_checkpoint):
        print(f"✓ 找到Stage 2 checkpoint: {stage2_checkpoint}")
        print(f"  将加载Stage 2的多模态投影层")
    else:
        print(f"⚠️  未找到Stage 2 checkpoint: {stage2_checkpoint}")
        print(f"  将跳过Stage 2，多模态投影层保持随机初始化")
        print(f"  Stage 3将从头训练多模态组件（推荐做法）")
        stage2_checkpoint = None  # 设置为None，train_fedmem.py会跳过加载
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
        # [关键] gating_init=0.01（pitfalls.md推荐值）
        # 原因：0.0001梯度太小，投影层无法学习对齐（参考pitfalls.md问题2）
        # 注意：这会触发归一化评分（gating_weight>=0.001），与Stage 1不同
        "gating_init": 0.01,

        # FedMem参数
        "memory_capacity": 50,
        "surprise_threshold": 0.3,
        "contrastive_lambda": 0.5,  # [Stage 3] 增强对比学习（帮助多模态对齐）
        "num_memory_prototypes": 5,

        # 联邦学习参数
        "num_rounds": 20,  # Stage 3微调
        "client_fraction": 0.2,
        "local_epochs": 1,
        "patience": 10,
        "partial_aggregation_warmup_rounds": 0,  # [关键] Stage 3禁用warmup

        # [Stage 3] 训练参数
        "learning_rate": 5e-4,  # 小学习率微调
        "weight_decay": 1e-4,
        "batch_size": 64,
        "num_negatives": 100,  # 标准负采样（保持稳定性）

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
    if stage2_checkpoint is None:
        print(f"  策略: 跳过Stage 2，直接从Stage 1开始训练（推荐）")
        print(f"  原因: Stage 2的对齐困难导致性能下降（参考pitfalls.md）")
    else:
        print(f"  策略: 加载Stage 2的投影层，继续微调")
    print(f"  目标: 同时训练SASRec、多模态层和Router")
    print()
    print("训练对象:")
    print(f"  ❄️  冻结: Item Embedding (锚点)")
    print(f"  🔥 可训练: SASRec Transformer + Visual/Semantic Experts + Router")
    print(f"  关键: SASRec可训练，能与多模态特征自然对齐")
    print()
    print("训练参数:")
    print(f"  轮数: {config['num_rounds']}")
    print(f"  学习率: {config['learning_rate']}")
    print(f"  对比学习权重: {config['contrastive_lambda']}")
    print(f"  门控初始值: {config['gating_init']}")
    print()
    print("预期效果:")
    print(f"  Round 1-3: HR@10 ≈ 0.35-0.38 (接近Stage 1)")
    print(f"  Round 10-15: HR@10 ≈ 0.40-0.43 (多模态逐步融入)")
    print(f"  Round 20: HR@10 ≈ 0.43-0.45 (稳定收敛)")
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
        print(f"❌ 错误: 以下数据文件不存在:")
        for f in missing_files:
            print(f"  - {f}")
        print()
        print(f"请先运行数据预处理:")
        print(f"  python UR4Rec/scripts/preprocess_ml1m_subset.py --top_k 1000")
        return 1

    print(f"✓ 所有数据文件已就绪")
    print()

    # 构建命令 - 使用当前Python解释器
    cmd = [sys.executable, "UR4Rec/scripts/train_fedmem.py"]

    for key, value in config.items():
        if value == "":
            # [修复] 空字符串参数特殊处理
            if key in ["verbose"]:  # 只有verbose是flag参数
                cmd.append(f"--{key}")
            # 其他空字符串参数不传递，使用train_fedmem.py的默认值
        else:
            cmd.append(f"--{key}")
            cmd.append(str(value))

    print("执行命令:")
    print(" ".join(cmd))
    print()
    print("="*60)
    print("开始训练 Stage 3...")
    print("="*60)
    print()

    # 运行训练
    try:
        result = subprocess.run(cmd, check=True)
        print()
        print("="*60)
        print("✓ Stage 3 训练完成！")
        print("="*60)
        print()
        print(f"最终模型保存位置: {config['save_dir']}/fedmem_model.pt")
        print()
        print("三阶段训练全部完成！")
        print()
        print("性能对比:")
        print(f"  Stage 1 (纯ID): HR@10 ≈ 0.60-0.70")
        print(f"  Stage 3 (MoE): HR@10 ≈ 0.70+ (期望)")
        print()
        print("查看训练历史:")
        print(f"  cat {config['save_dir']}/train_history.json")
        return 0

    except subprocess.CalledProcessError as e:
        print()
        print("="*60)
        print(f"❌ Stage 3 训练失败，退出码: {e.returncode}")
        print("="*60)
        return e.returncode
    except KeyboardInterrupt:
        print()
        print("="*60)
        print("⚠️ 训练被用户中断")
        print("="*60)
        return 130


if __name__ == "__main__":
    exit(main())
