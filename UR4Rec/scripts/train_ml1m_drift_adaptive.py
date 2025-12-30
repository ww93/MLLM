"""
ML-1M 长序列子集训练脚本 - 漂移自适应对比学习

专门用于测试FedDMMR的漂移自适应对比学习功能
使用Top-1000活跃用户的长序列子集（515K交互）

使用方法：
    python scripts/train_ml1m_drift_adaptive.py

特性：
1. ✅ 漂移自适应对比学习（自动集成）
2. ✅ ML-1M长序列子集（1000用户，平均序列515）
3. ✅ 本地动态记忆（FedMem）
4. ✅ Surprise-based记忆更新
"""
import os
import sys
import subprocess
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    """运行ML-1M漂移自适应训练"""

    print("="*60)
    print("ML-1M 长序列子集 - 漂移自适应对比学习训练")
    print("="*60)
    print()

    # 配置参数
    config = {
        # 数据配置
        "data_dir": "UR4Rec/data/ml-1m",
        "data_file": "subset_ratings.dat",

        # [NEW] 多模态特征文件（真实CLIP+文本特征）
        "visual_file": "clip_features.pt",  # CLIP特征 [3953, 512]
        "text_file": "text_features.pt",    # 文本特征 [3953, 384]

        # 模型配置（适配ML-1M）
        "num_items": 3953,  # ML-1M子集的最大物品ID+1
        "sasrec_hidden_dim": 128,  # 较小的维度，快速训练
        "sasrec_num_blocks": 2,
        "sasrec_num_heads": 4,
        "max_seq_len": 50,

        # FedMem参数
        "memory_capacity": 50,
        "surprise_threshold": 0.3,
        "contrastive_lambda": 0.1,  # 对比学习损失权重
        "num_memory_prototypes": 5,
        "enable_prototype_aggregation": "",  # 启用原型聚合

        # 联邦学习参数
        "num_rounds": 20,  # 快速测试：20轮
        "client_fraction": 0.1,  # 每轮10%的客户端（100个）
        "local_epochs": 1,
        "patience": 10,

        # 训练参数
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "batch_size": 32,
        "num_negatives": 100,

        # 评估参数
        "use_negative_sampling": "true",
        "num_negatives_eval": 100,

        # Residual Enhancement参数
        "gating_init": 1.0,  # [FIX 1] 修复后默认值

        # [NEW] 预训练权重路径（可选，如果存在则自动加载）
        # "pretrained_path": "checkpoints/fedsasrec_pretrain/fedmem_model.pt",

        # 其他参数
        "seed": 42,
        "save_dir": "checkpoints/ml1m_drift_adaptive",
        "verbose": ""  # 显示详细信息
    }

    print("训练配置:")
    print(f"  数据集: ML-1M长序列子集（1000用户）")
    print(f"  物品数: {config['num_items']}")
    print(f"  联邦轮数: {config['num_rounds']}")
    print(f"  客户端比例: {config['client_fraction']} (每轮{int(1000*config['client_fraction'])}个)")
    print(f"  对比学习权重: {config['contrastive_lambda']}")
    print(f"  保存目录: {config['save_dir']}")
    print()
    print("多模态特征:")
    print(f"  ✅ CLIP视觉特征: {config['visual_file']} (3646个电影海报)")
    print(f"  ✅ 文本特征: {config['text_file']} (3646个电影描述)")
    print()
    print("核心特性:")
    print("  ✅ 漂移自适应对比学习（基于surprise的温度调节）")
    print("  ✅ 本地动态记忆（容量: {})".format(config['memory_capacity']))
    print("  ✅ 长序列训练（平均序列长度: 515）")
    print("  ✅ 真实多模态特征（CLIP + Sentence-BERT）")
    print()

    # 检查数据文件是否存在
    data_path = os.path.join(config["data_dir"], config["data_file"])
    if not os.path.exists(data_path):
        print(f"❌ 错误: 数据文件不存在: {data_path}")
        print(f"请先运行子集生成脚本:")
        print(f"  python scripts/preprocess_ml1m_subset.py --top_k 1000")
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
    print("开始训练...")
    print("="*60)
    print()

    # 运行训练
    try:
        result = subprocess.run(cmd, check=True)
        print()
        print("="*60)
        print("✓ 训练完成！")
        print("="*60)
        print()
        print(f"结果已保存到: {config['save_dir']}/")
        print(f"  - 模型: fedmem_model.pt")
        print(f"  - 训练历史: train_history.json")
        print(f"  - 配置: config.json")
        return 0

    except subprocess.CalledProcessError as e:
        print()
        print("="*60)
        print(f"❌ 训练失败，退出码: {e.returncode}")
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
