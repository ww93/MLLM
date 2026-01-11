"""
Stage 2: Modality Alignment (修复版)
第二阶段：模态对齐 - 修复过拟合问题

修复内容:
1. gating_init 从 0.0001 增大到 0.01 (增大100倍)
2. contrastive_lambda 从 0.1 增大到 0.5 (增强对齐信号)
3. num_rounds 从 20 减少到 10 (减少过拟合风险)
4. learning_rate 从 1e-4 增大到 5e-4 (加速收敛)

目标: 让投影层快速对齐到ID空间
预期: Round 5-10达到HR@10 ≈ 0.43-0.45

使用方法:
    python UR4Rec/scripts/train_stage2_fixed.py
"""
import os
import sys
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    """第二阶段：模态对齐（修复版）"""

    print("="*60)
    print("Stage 2: Modality Alignment (修复版)")
    print("第二阶段：模态对齐（修复过拟合问题）")
    print("="*60)
    print()

    # 检查Stage 1 checkpoint
    script_dir = Path(__file__).parent.parent
    stage1_checkpoint = str(script_dir / "checkpoints" / "stage1_backbone" / "fedmem_model.pt")

    if not os.path.exists(stage1_checkpoint):
        print(f"❌ 错误: 缺少 Stage 1 checkpoint: {stage1_checkpoint}")
        print(f"    请先运行: python UR4Rec/scripts/train_stage1_backbone.py")
        print()
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

        # [修复1] gating_init: 0.0001 → 0.01 (增大100倍)
        "gating_init": 0.01,

        # FedMem参数
        "memory_capacity": 50,
        "surprise_threshold": 0.3,
        # [修复2] contrastive_lambda: 0.1 → 0.5 (增强对齐信号)
        "contrastive_lambda": 0.5,
        "num_memory_prototypes": 5,

        # 联邦学习参数
        # [修复3] num_rounds: 20 → 10 (减少过拟合风险)
        "num_rounds": 10,
        "client_fraction": 0.2,
        "local_epochs": 1,
        "patience": 5,  # 更激进的早停
        "partial_aggregation_warmup_rounds": 0,

        # [修复4] learning_rate: 1e-4 → 5e-4 (加速收敛)
        "learning_rate": 5e-4,
        "weight_decay": 1e-4,
        "batch_size": 64,
        "num_negatives": 100,

        # 评估参数
        "use_negative_sampling": "true",
        "num_negatives_eval": 100,

        # 其他参数
        "seed": 42,
        "save_dir": "UR4Rec/checkpoints/stage2_fixed",
        "verbose": ""
    }

    print("训练配置:")
    print(f"  阶段: Stage 2 - Modality Alignment (修复版)")
    print(f"  目标: 让多模态特征对齐到ID空间")
    print()
    print("训练对象:")
    print(f"  ❄️  冻结: SASRec + Item Embedding")
    print(f"  🔥 训练: Visual/Semantic Projectors + Router + Gating Weight")
    print()
    print("修复内容:")
    print(f"  [修复1] gating_init: 0.0001 → {config['gating_init']:.2f} (增大100倍)")
    print(f"          理由: 让投影层接收到足够的梯度信号")
    print(f"  [修复2] contrastive_lambda: 0.1 → {config['contrastive_lambda']:.1f} (增强5倍)")
    print(f"          理由: 增强多模态对齐的监督信号")
    print(f"  [修复3] num_rounds: 20 → {config['num_rounds']} (减少50%)")
    print(f"          理由: 减少过拟合风险，尽早停止")
    print(f"  [修复4] learning_rate: 1e-4 → {config['learning_rate']:.0e} (增大5倍)")
    print(f"          理由: 加速投影层收敛")
    print()
    print("预期效果:")
    print(f"  Round 1-3: HR@10 ≈ 0.41 (依赖Stage 1的SASRec)")
    print(f"  Round 5-7: HR@10 ≈ 0.43-0.45 (投影层开始对齐)")
    print(f"  Round 8-10: HR@10 ≈ 0.45+ (多模态融合生效)")
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
            if key in ["verbose"]:
                cmd.append(f"--{key}")
        else:
            cmd.append(f"--{key}")
            cmd.append(str(value))

    print("执行命令:")
    print(" ".join(cmd))
    print()
    print("="*60)
    print("开始训练 Stage 2 (修复版)...")
    print("="*60)
    print()

    # 运行训练
    try:
        result = subprocess.run(cmd, check=True)
        print()
        print("="*60)
        print("✓ Stage 2 (修复版) 训练完成！")
        print("="*60)
        print()
        print(f"最终模型保存位置: {config['save_dir']}/fedmem_model.pt")
        print()
        print("下一步:")
        print(f"  运行 Stage 3: python UR4Rec/scripts/train_stage3_moe.py")
        print(f"  (记得修改 stage2_checkpoint 路径为: {config['save_dir']}/fedmem_model.pt)")
        print()
        print("查看训练历史:")
        print(f"  cat {config['save_dir']}/train_history.json")
        return 0

    except subprocess.CalledProcessError as e:
        print()
        print("="*60)
        print(f"❌ Stage 2 (修复版) 训练失败，退出码: {e.returncode}")
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
