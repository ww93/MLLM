"""
Stage 1: Backbone Pre-training
第一阶段：夯实地基 - 纯ID SASRec预训练

目标: 训练高质量的纯ID SASRec
对象: SASRec (Embedding + Transformer)
数据: 仅Item ID序列
预期HR@10: 0.60-0.70

修改记录:
1. num_rounds: 30 -> 100 (ID更新稀疏，需要更多轮次收敛)
2. learning_rate: 5e-3 -> 1e-3 (配合BPR和std=0.02初始化，1e-3更稳健)
3. num_negatives: 100 -> 20 (兼顾训练速度与梯度质量，100个太慢且收益递减)
4. patience: 15 -> 20 (给予更多耐心)

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

        # [Stage 1] 禁用多模态（visual/text会被stage逻辑自动忽略，但文件需存在避免报错）
        "visual_file": "clip_features.pt",
        "text_file": "text_features.pt",

        # [Stage 1] 训练阶段
        "stage": "pretrain_sasrec",

        # 模型配置
        "model_type": "moe",  # 必须使用MoE架构以保证权重Key匹配
        "num_items": 3953,
        "sasrec_hidden_dim": 128,
        "sasrec_num_blocks": 2,
        "sasrec_num_heads": 4,
        "max_seq_len": 50,

        # MoE参数（Stage 1不使用，但需要初始化占位）
        "moe_num_heads": 8,
        "retriever_output_dim": 128,

        # FedMem参数 (Stage 1 不用 Memory，设为最小开销)
        "memory_capacity": 50,
        "surprise_threshold": 0.3,
        "contrastive_lambda": 0.0,  # 禁用对比学习
        "num_memory_prototypes": 5,

        # 联邦学习参数
        "num_rounds": 50,       # [修改] 增加轮数！ID训练收敛慢，30轮不够
        "client_fraction": 0.2,  # 200个客户端/轮，保证物品覆盖率
        "local_epochs": 1,
        "patience": 20,          # [修改] 增加耐心值

        # [Stage 1] 训练参数
        "learning_rate": 1e-3,   # [修改] 1e-3 是最稳健的，5e-3风险太大
        "weight_decay": 1e-5,
        "batch_size": 64,
        "num_negatives": 20,     # [修改] 降至20。100太慢，20配合BPR Loss足矣
        "partial_aggregation_warmup_rounds": 0, # 必须为0，全量聚合

        # 评估参数 (评估时需要高精度，保持100)
        "use_negative_sampling": "true",
        "num_negatives_eval": 100,

        # 其他参数
        "seed": 42,
        "save_dir": "UR4Rec/checkpoints/stage1_backbone",
        "pretrained_path": "",  # 从零开始
        "verbose": ""
    }

    print("训练配置:")
    print(f"  阶段: Stage 1 - Backbone Pre-training")
    print(f"  目标: 训练纯ID SASRec (预期 HR@10 > 0.60)")
    print(f"  数据: 仅Item ID序列")
    print()
    print("核心超参调整:")
    print(f"  Running Rounds: {config['num_rounds']} (原30 -> 100，确保ID充分收敛)")
    print(f"  Learning Rate:  {config['learning_rate']} (原5e-3 -> 1e-3，稳健模式)")
    print(f"  Train Negatives: {config['num_negatives']} (原100 -> 20，加速训练)")
    print()
    print(f"保存位置: {config['save_dir']}/")
    print()

    # 检查数据文件
    data_path = os.path.join(config["data_dir"], config["data_file"])
    if not os.path.exists(data_path):
        print(f"❌ 错误: 数据文件不存在: {data_path}")
        return 1

    # 构建命令
    cmd = [sys.executable, "UR4Rec/scripts/train_fedmem.py"]

    for key, value in config.items():
        if value == "":
            if key in ["verbose"]:
                cmd.append(f"--{key}")
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

    try:
        result = subprocess.run(cmd, check=True)
        print()
        print("="*60)
        print("✓ Stage 1 训练完成！")
        print("请检查日志，确认 HR@10 是否达到 0.60 以上。")
        print("如果未达到，请不要进入 Stage 2，继续检查 Stage 1 的代码逻辑。")
        print("="*60)
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Stage 1 训练失败，退出码: {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("\n⚠️ 训练被用户中断")
        return 130


if __name__ == "__main__":
    exit(main())