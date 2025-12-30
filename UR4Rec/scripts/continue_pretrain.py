"""
从checkpoint继续预训练FedSASRec

目标: 从已有的预训练checkpoint继续训练，直到充分收敛
使用场景: 当初始预训练轮数不够时（如15轮→25轮）

使用方法:
    python scripts/continue_pretrain.py

特点:
1. 从现有checkpoint加载权重
2. 继续训练10轮（总共25轮）
3. 保持相同的超参数配置
4. 更新原checkpoint文件
"""
import os
import sys
import subprocess
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    """从checkpoint继续预训练"""

    print("="*60)
    print("从Checkpoint继续预训练FedSASRec")
    print("="*60)
    print()

    # 检查checkpoint是否存在
    checkpoint_path = "UR4Rec/checkpoints/fedsasrec_pretrain/fedmem_model.pt"
    if not os.path.exists(checkpoint_path):
        print(f"❌ 错误: Checkpoint不存在: {checkpoint_path}")
        print(f"请先运行初始预训练:")
        print(f"  python UR4Rec/scripts/pretrain_fedsasrec.py")
        return 1

    print(f"✓ 找到预训练checkpoint: {checkpoint_path}")
    print()

    # 配置参数（与初始预训练保持一致，但从checkpoint加载）
    config = {
        # 数据配置
        "data_dir": "UR4Rec/data/ml-1m",
        "data_file": "subset_ratings.dat",

        # [关键] 不加载多模态特征（仅ID嵌入）
        # 注意: 不传递visual_file和text_file参数

        # 模型配置（与初始预训练保持一致）
        "num_items": 3953,
        "sasrec_hidden_dim": 128,
        "sasrec_num_blocks": 2,
        "sasrec_num_heads": 4,
        "max_seq_len": 50,

        # FedMem参数
        "memory_capacity": 50,
        "surprise_threshold": 0.3,
        "contrastive_lambda": 0.0,  # ❗关键: 禁用对比学习
        "num_memory_prototypes": 5,

        # [关键] 联邦学习参数 - 继续训练10轮
        "num_rounds": 15,  # 额外训练10轮（总共15+10=25轮）
        "client_fraction": 0.2,
        "local_epochs": 1,
        "patience": 10,

        # 训练参数（保持一致）
        "learning_rate": 5e-3,  # 保持大学习率
        "weight_decay": 1e-5,
        "batch_size": 64,
        "num_negatives": 100,

        # 评估参数
        "use_negative_sampling": "true",
        "num_negatives_eval": 100,

        # Residual Enhancement参数
        "gating_init": 1.0,

        # [NEW] 从checkpoint加载
        "pretrained_path": checkpoint_path,

        # 其他参数
        "seed": 42,
        "save_dir": "UR4Rec/checkpoints/fedsasrec_pretrain",  # 保存到同一位置（覆盖）
        "verbose": ""
    }

    print("继续训练配置:")
    print(f"  加载checkpoint: {checkpoint_path}")
    print(f"  继续训练轮数: {config['num_rounds']} (额外)")
    print(f"  总轮数: 15 + {config['num_rounds']} = 25")
    print(f"  学习率: {config['learning_rate']} (保持不变)")
    print(f"  客户端比例: {config['client_fraction']}")
    print()
    print("预期效果:")
    print(f"  当前HR@10: 0.36")
    print(f"  目标HR@10: 0.50-0.55 (25轮后)")
    print()
    print("⏱️  预计时长: 20-30分钟")
    print()

    # 构建命令
    cmd = ["python3", "UR4Rec/scripts/train_fedmem.py"]

    for key, value in config.items():
        if value == "":  # 标志参数
            cmd.append(f"--{key}")
        else:
            cmd.append(f"--{key}")
            cmd.append(str(value))

    print("执行命令:")
    print(" ".join(cmd))
    print()
    print("="*60)
    print("开始继续训练...")
    print("="*60)
    print()

    # 运行训练
    try:
        result = subprocess.run(cmd, check=True)
        print()
        print("="*60)
        print("✓ 继续训练完成！")
        print("="*60)
        print()
        print(f"更新的预训练权重: {config['save_dir']}/fedmem_model.pt")
        print(f"总训练轮数: 25轮")
        print()
        print("下一步: 使用更新后的预训练权重训练FedDMMR")
        print(f"  python UR4Rec/scripts/train_ml1m_drift_adaptive.py \\")
        print(f"    (取消注释pretrained_path配置)")
        return 0

    except subprocess.CalledProcessError as e:
        print()
        print("="*60)
        print(f"❌ 继续训练失败，退出码: {e.returncode}")
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
