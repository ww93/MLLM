"""
FedSASRec 预训练脚本（无MoE、无Memory、无Contrastive）

目标: 快速预训练SASRec骨干网络，为FedDMMR提供初始化权重
特点:
1. 仅使用ID嵌入（无多模态特征）
2. 禁用记忆模块（contrastive_lambda=0）
3. 禁用对比学习（无视觉/文本专家参与）
4. 快速收敛配置（10-15轮）
5. 保存SASRec权重供FedDMMR加载

使用方法:
    python scripts/pretrain_fedsasrec.py

预期时长: CPU 30-60分钟，GPU 10-20分钟
"""
import os
import sys
import subprocess
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    """运行FedSASRec预训练"""

    print("="*60)
    print("FedSASRec 预训练（为FedDMMR提供骨干初始化）")
    print("="*60)
    print()

    # 配置参数
    config = {
        # 数据配置
        "data_dir": "UR4Rec/data/ml-1m",
        "data_file": "subset_ratings.dat",

        # [关键] 不加载多模态特征（仅ID嵌入）
        # 注意: 不传递visual_file和text_file参数，让它们保持默认值None

        # 模型配置（与FedDMMR保持一致）
        "num_items": 3953,
        "sasrec_hidden_dim": 128,
        "sasrec_num_blocks": 2,
        "sasrec_num_heads": 4,
        "max_seq_len": 50,

        # [关键] FedMem参数 - 禁用对比学习
        "memory_capacity": 50,  # 记忆容量保留（但不会触发对比学习）
        "surprise_threshold": 0.5,
        "contrastive_lambda": 0.0,  # ❗关键: 设为0禁用对比学习
        "num_memory_prototypes": 5,
        # 不启用prototype_aggregation（默认False）

        # [优化] 联邦学习参数 - 快速收敛
        "num_rounds": 20,  # 预训练只需15轮（比FedDMMR的20轮少）
        "client_fraction": 0.2,  # 每轮20%客户端（200个，加速训练）
        "local_epochs": 1,
        "patience": 10,

        # [优化] 训练参数 - 较大学习率加速收敛
        "learning_rate": 5e-3,  # 5倍学习率（FedDMMR用1e-3）
        "weight_decay": 1e-5,
        "batch_size": 64,  # 2倍batch size（加速训练）
        "num_negatives": 100,

        # 评估参数
        "use_negative_sampling": "true",
        "num_negatives_eval": 100,

        # Residual Enhancement参数（虽然没有辅助专家，但参数保留）
        "gating_init": 1.0,

        # 其他参数
        "seed": 42,
        "save_dir": "UR4Rec/checkpoints/fedsasrec_pretrain",  # 专用保存目录
        "verbose": ""  # 显示详细信息
    }

    print("预训练配置:")
    print(f"  数据集: ML-1M长序列子集（1000用户）")
    print(f"  物品数: {config['num_items']}")
    print(f"  联邦轮数: {config['num_rounds']} (快速预训练)")
    print(f"  客户端比例: {config['client_fraction']} (每轮{int(1000*config['client_fraction'])}个)")
    print(f"  学习率: {config['learning_rate']} (5x加速)")
    print(f"  批大小: {config['batch_size']} (2x加速)")
    print(f"  保存目录: {config['save_dir']}")
    print()
    print("关键特性:")
    print("  ✅ 仅ID嵌入（无多模态特征）")
    print("  ✅ 无对比学习（contrastive_lambda=0）")
    print("  ✅ 无辅助专家（MoE处于待机状态）")
    print("  ✅ 快速收敛配置（15轮，大学习率，大batch）")
    print()
    print("⏱️  预计时长: 30-60分钟 (CPU), 10-20分钟 (GPU)")
    print()

    # 检查数据文件是否存在
    data_path = os.path.join(config["data_dir"], config["data_file"])
    if not os.path.exists(data_path):
        print(f"❌ 错误: 数据文件不存在: {data_path}")
        print(f"请先运行子集生成脚本:")
        print(f"  python UR4Rec/scripts/preprocess_ml1m_subset.py --top_k 1000")
        return 1

    print(f"✓ 数据文件已找到: {data_path}")
    print()

    # 构建命令
    cmd = ["python3", "UR4Rec/scripts/train_fedmem.py"]

    for key, value in config.items():
        if value == "":  # 标志参数（如--verbose）
            cmd.append(f"--{key}")
        else:
            cmd.append(f"--{key}")
            cmd.append(str(value))

    print("执行命令:")
    print(" ".join(cmd))
    print()
    print("="*60)
    print("开始预训练...")
    print("="*60)
    print()

    # 运行训练
    try:
        result = subprocess.run(cmd, check=True)
        print()
        print("="*60)
        print("✓ 预训练完成！")
        print("="*60)
        print()
        print(f"预训练权重已保存到: {config['save_dir']}/")
        print(f"  - 模型: fedmem_model.pt")
        print(f"  - 训练历史: train_history.json")
        print()
        print("下一步: 使用预训练权重训练FedDMMR")
        print(f"  python UR4Rec/scripts/train_ml1m_drift_adaptive.py \\")
        print(f"    --pretrained_path {config['save_dir']}/fedmem_model.pt")
        return 0

    except subprocess.CalledProcessError as e:
        print()
        print("="*60)
        print(f"❌ 预训练失败，退出码: {e.returncode}")
        print("="*60)
        return e.returncode
    except KeyboardInterrupt:
        print()
        print("="*60)
        print("⚠️ 预训练被用户中断")
        print("="*60)
        return 130


if __name__ == "__main__":
    exit(main())
