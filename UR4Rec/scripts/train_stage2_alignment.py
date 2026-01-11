"""
Stage 2: Modality Alignment (方案2：保持原始维度 + 注意力融合)
第二阶段：模态对齐

[方案2改进]:
1. 保持预训练特征维度（CLIP 512维, SBERT 384维）
2. 使用CrossModalFusion层进行注意力融合
3. 避免维度压缩导致的信息损失

目标: 让多模态特征通过注意力机制与SASRec输出融合
对象:
  - 冻结: SASRec, Item Embedding
  - 训练: Visual Expert (512→512), Semantic Expert (384→384), CrossModalFusion层
预期: 保持预训练特征完整性，提升多模态融合效果

使用方法:
    python UR4Rec/scripts/train_stage2_alignment.py
"""
import os
import sys
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    """第二阶段：多模态投影层对齐"""

    print("="*60)
    print("Stage 2: Modality Alignment")
    print("第二阶段：模态对齐")
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
        "gating_init": 0.01,  # [方案2] 无需极小值，CrossModalFusion自适应融合

        # FedMem参数
        "memory_capacity": 50,
        "surprise_threshold": 0.3,
        "contrastive_lambda": 0.3,  # [方案2] 中等强度，辅助特征对齐
        "num_memory_prototypes": 5,

        # 联邦学习参数
        "num_rounds": 20,  # [方案2] 略微减少轮数，避免过拟合
        "client_fraction": 0.2,
        "local_epochs": 1,
        "patience": 8,  # [方案2] 更激进的早停
        "partial_aggregation_warmup_rounds": 0,

        # [方案2] 训练参数 - 正常学习率，注意力机制需要足够的更新
        "learning_rate": 5e-4,  # 略高于原版，让CrossModalFusion快速收敛
        "weight_decay": 1e-4,
        "batch_size": 64,
        "num_negatives": 100,

        # 评估参数
        "use_negative_sampling": "true",
        "num_negatives_eval": 100,

        # 其他参数
        "seed": 42,
        "save_dir": "UR4Rec/checkpoints/stage2_alignment",
        "verbose": ""
    }

    print("训练配置:")
    print(f"  阶段: Stage 2 - Modality Alignment (方案2)")
    print(f"  目标: 保持预训练特征完整性，使用注意力融合")
    print()
    print("架构改进 [方案2]:")
    print(f"  ✓ Visual Expert: 512维输出（保持CLIP原始维度）")
    print(f"  ✓ Semantic Expert: 384维输出（保持SBERT原始维度）")
    print(f"  ✓ CrossModalFusion: 注意力机制融合异构特征")
    print(f"  ✓ 避免维度压缩导致的信息损失")
    print()
    print("训练对象:")
    print(f"  ❄️  冻结: SASRec + Item Embedding")
    print(f"  🔥 训练: VisualExpert (512→512) + SemanticExpert (384→384) + CrossModalFusion")
    print()
    print("训练参数 [方案2优化]:")
    print(f"  轮数: {config['num_rounds']}")
    print(f"  学习率: {config['learning_rate']:.0e} (中等，让注意力快速收敛)")
    print(f"  对比学习权重: {config['contrastive_lambda']} (中等强度)")
    print(f"  门控初始值: {config['gating_init']} (无需极小值)")
    print(f"  早停patience: {config['patience']} (更激进)")
    print()
    print("预期效果:")
    print(f"  Round 1-3: HR@10 ≈ 0.35-0.40 (与Stage 1接近)")
    print(f"  Round 8-10: HR@10 ≈ 0.40-0.43 (注意力融合开始生效)")
    print(f"  Round 15: HR@10 ≈ 0.43-0.45 (多模态信息充分利用)")
    print()
    print("方案2优势:")
    print(f"  ✓ 保持CLIP/SBERT预训练特征的完整性")
    print(f"  ✓ 注意力机制自适应加权融合")
    print(f"  ✓ 避免随机初始化投影层破坏特征结构")
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
    print("开始训练 Stage 2...")
    print("="*60)
    print()

    # 运行训练
    try:
        result = subprocess.run(cmd, check=True)
        print()
        print("="*60)
        print("✓ Stage 2 训练完成！")
        print("="*60)
        print()
        print(f"模型保存位置: {config['save_dir']}/fedmem_model.pt")
        print()
        print("下一步:")
        print(f"  运行 Stage 3 (MoE集成微调):")
        print(f"  python UR4Rec/scripts/train_stage3_moe.py")
        return 0

    except subprocess.CalledProcessError as e:
        print()
        print("="*60)
        print(f"❌ Stage 2 训练失败，退出码: {e.returncode}")
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
