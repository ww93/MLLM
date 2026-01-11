"""
Stage 1: Backbone Pre-training
第一阶段：夯实地基 - 纯ID SASRec预训练

目标: 训练高质量的纯ID SASRec
对象: SASRec (Embedding + Transformer)
数据: 仅Item ID序列
预期HR@10: 0.60-0.70

使用方法:
    python UR4Rec/scripts/train_stage1_backbone.py
"""
import os
import sys
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    """第一阶段：纯ID SASRec预训练"""

    print("="*60)
    print("Stage 1: Backbone Pre-training")
    print("第一阶段：夯实地基 - 纯ID SASRec预训练")
    print("="*60)
    print()

    config = {
        # 数据配置
        "data_dir": "UR4Rec/data/ml-1m",
        "data_file": "subset_ratings.dat",

        # [Stage 1] 禁用多模态（会在train_fedmem中自动处理）
        "visual_file": "clip_features.pt",  # 会被stage参数覆盖
        "text_file": "text_features.pt",     # 会被stage参数覆盖

        # [Stage 1] 训练阶段
        "stage": "pretrain_sasrec",

        # 模型配置
        "model_type": "moe",  # 使用MoE架构（但Stage 1只训练SASRec部分）
        "num_items": 3953,
        "sasrec_hidden_dim": 128,
        "sasrec_num_blocks": 2,
        "sasrec_num_heads": 4,
        "max_seq_len": 50,

        # MoE参数（Stage 1不使用，但需要初始化）
        "moe_num_heads": 8,
        "retriever_output_dim": 128,

        # FedMem参数
        "memory_capacity": 50,
        "surprise_threshold": 0.3,
        "contrastive_lambda": 0.0,  # Stage 1不使用对比学习
        "num_memory_prototypes": 5,

        # 联邦学习参数
        "num_rounds": 30,  # Stage 1训练更多轮以确保收敛
        "client_fraction": 0.2,  # 每轮20%客户端（200个）
        "local_epochs": 1,
        "patience": 15,

        # [Stage 1] 训练参数 - 对齐FedSASRec预训练配置
        "learning_rate": 5e-3,  # 高学习率加速收敛（与FedSASRec一致）
        "weight_decay": 1e-5,   # 降低正则化
        "batch_size": 64,
        "num_negatives": 100,   # 标准负采样（与FedSASRec一致，而非批内负采样）
        "partial_aggregation_warmup_rounds": 0,

        # 评估参数
        "use_negative_sampling": "true",
        "num_negatives_eval": 100,

        # 其他参数
        "seed": 42,
        "save_dir": "UR4Rec/checkpoints/stage1_backbone",
        "pretrained_path": "",  # [关键修复] Stage 1从零训练，不加载预训练模型
        "verbose": ""
    }

    print("训练配置:")
    print(f"  阶段: Stage 1 - Backbone Pre-training")
    print(f"  目标: 训练纯ID SASRec")
    print(f"  数据: 仅Item ID序列（禁用多模态）")
    print()
    print("模型配置:")
    print(f"  架构: SASRec")
    print(f"  物品数: {config['num_items']}")
    print(f"  隐藏维度: {config['sasrec_hidden_dim']}")
    print(f"  Transformer块: {config['sasrec_num_blocks']}")
    print()
    print("训练参数:")
    print(f"  轮数: {config['num_rounds']}")
    print(f"  学习率: {config['learning_rate']} (5x加速)")
    print(f"  客户端比例: {config['client_fraction']}")
    print(f"  负样本数: {config['num_negatives']} (标准负采样)")
    print()
    print("⚠️  重要修复:")
    print(f"  已对齐FedSASRec预训练配置（5e-3学习率 + 100负样本）")
    print(f"  避免使用批内负采样（4负样本太少，导致HR@10仅0.09）")
    print()
    print("预期效果:")
    print(f"  HR@10: 0.60-0.70 (纯ID baseline)")
    print(f"  这是你的保底分数！")
    print()
    print(f"保存位置: {config['save_dir']}/")
    print(f"  模型: fedmem_model.pt")
    print(f"  训练历史: train_history.json")
    print()

    # 检查数据文件
    data_path = os.path.join(config["data_dir"], config["data_file"])
    if not os.path.exists(data_path):
        print(f"❌ 错误: 数据文件不存在: {data_path}")
        print(f"请先运行数据预处理:")
        print(f"  python UR4Rec/scripts/preprocess_ml1m_subset.py --top_k 1000")
        return 1

    print(f"✓ 数据文件已就绪")
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
    print("开始训练 Stage 1...")
    print("="*60)
    print()

    # 运行训练
    try:
        result = subprocess.run(cmd, check=True)
        print()
        print("="*60)
        print("✓ Stage 1 训练完成！")
        print("="*60)
        print()
        print(f"模型保存位置: {config['save_dir']}/fedmem_model.pt")
        print()
        print("下一步:")
        print(f"  运行 Stage 2 (多模态对齐):")
        print(f"  python UR4Rec/scripts/train_stage2_alignment.py")
        return 0

    except subprocess.CalledProcessError as e:
        print()
        print("="*60)
        print(f"❌ Stage 1 训练失败，退出码: {e.returncode}")
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
