"""
三阶段训练一键运行脚本
Three-Stage Training Master Script

自动运行三个阶段:
1. Stage 1: Backbone Pre-training (纯ID SASRec)
2. Stage 2: Modality Alignment (多模态对齐)
3. Stage 3: MoE Fine-tuning (MoE集成微调)

使用方法:
    python UR4Rec/scripts/train_all_stages.py

可选参数:
    --start-from {1,2,3}    从指定阶段开始
    --skip-stage {1,2,3}    跳过指定阶段

示例:
    # 运行所有阶段
    python UR4Rec/scripts/train_all_stages.py

    # 从Stage 2开始
    python UR4Rec/scripts/train_all_stages.py --start-from 2

    # 跳过Stage 1
    python UR4Rec/scripts/train_all_stages.py --skip-stage 1
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent))


def run_stage(stage_num: int, script_name: str) -> bool:
    """
    运行指定阶段的训练

    Args:
        stage_num: 阶段编号 (1, 2, 3)
        script_name: 脚本文件名

    Returns:
        是否成功
    """
    print()
    print("="*80)
    print(f"Stage {stage_num} 开始")
    print("="*80)
    print()

    start_time = time.time()

    try:
        cmd = ["python3", f"UR4Rec/scripts/{script_name}"]
        result = subprocess.run(cmd, check=True)

        elapsed_time = time.time() - start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)

        print()
        print("="*80)
        print(f"✓ Stage {stage_num} 完成！")
        print(f"耗时: {hours}小时 {minutes}分钟 {seconds}秒")
        print("="*80)
        print()

        return True

    except subprocess.CalledProcessError as e:
        print()
        print("="*80)
        print(f"✗ Stage {stage_num} 失败，退出码: {e.returncode}")
        print("="*80)
        return False

    except KeyboardInterrupt:
        print()
        print("="*80)
        print(f"⚠️ Stage {stage_num} 被用户中断")
        print("="*80)
        return False


def check_checkpoint(checkpoint_path: str, stage_name: str) -> bool:
    """检查checkpoint是否存在"""
    if os.path.exists(checkpoint_path):
        print(f"  ✓ 找到 {stage_name} checkpoint: {checkpoint_path}")
        return True
    else:
        print(f"  ✗ 未找到 {stage_name} checkpoint: {checkpoint_path}")
        return False


def main():
    parser = argparse.ArgumentParser(description="三阶段训练一键运行")
    parser.add_argument("--start-from", type=int, choices=[1, 2, 3], default=1,
                        help="从指定阶段开始 (默认: 1)")
    parser.add_argument("--skip-stage", type=int, choices=[1, 2, 3], default=None,
                        help="跳过指定阶段")
    args = parser.parse_args()

    print("="*80)
    print("三阶段训练一键运行脚本")
    print("Three-Stage Training Master Script")
    print("="*80)
    print()
    print("训练计划:")
    print("  Stage 1: Backbone Pre-training (纯ID SASRec, 预计1-2小时)")
    print("  Stage 2: Modality Alignment (多模态对齐, 预计40分钟-1小时)")
    print("  Stage 3: MoE Fine-tuning (MoE集成微调, 预计40分钟-1小时)")
    print()
    print(f"开始阶段: Stage {args.start_from}")
    if args.skip_stage:
        print(f"跳过阶段: Stage {args.skip_stage}")
    print()

    # 定义三个阶段
    stages = {
        1: ("train_stage1_backbone.py", "Stage 1 (Backbone)"),
        2: ("train_stage2_alignment.py", "Stage 2 (Alignment)"),
        3: ("train_stage3_moe.py", "Stage 3 (MoE)")
    }

    # checkpoint路径
    script_dir = Path(__file__).parent.parent
    checkpoints = {
        1: str(script_dir / "checkpoints" / "stage1_backbone" / "fedmem_model.pt"),
        2: str(script_dir / "checkpoints" / "stage2_alignment" / "fedmem_model.pt"),
        3: str(script_dir / "checkpoints" / "stage3_moe" / "fedmem_model.pt")
    }

    # 检查前置checkpoint
    print("检查前置checkpoint...")
    for stage_num in range(1, args.start_from):
        if not check_checkpoint(checkpoints[stage_num], f"Stage {stage_num}"):
            print()
            print(f"❌ 错误: Stage {args.start_from} 需要 Stage {stage_num} 的checkpoint")
            print(f"请先运行: python UR4Rec/scripts/{stages[stage_num][0]}")
            return 1

    print()
    input("按Enter键开始训练...")
    print()

    # 记录总时间
    total_start_time = time.time()

    # 运行各阶段
    for stage_num in range(args.start_from, 4):
        # 跳过指定阶段
        if args.skip_stage == stage_num:
            print()
            print("="*80)
            print(f"跳过 Stage {stage_num}")
            print("="*80)
            print()
            continue

        # 运行阶段
        script_name, stage_name = stages[stage_num]
        success = run_stage(stage_num, script_name)

        if not success:
            print()
            print("="*80)
            print("训练中断")
            print("="*80)
            return 1

        # 阶段间休息
        if stage_num < 3:
            print(f"Stage {stage_num} 完成，准备下一阶段...")
            time.sleep(2)

    # 总结
    total_elapsed_time = time.time() - total_start_time
    hours = int(total_elapsed_time // 3600)
    minutes = int((total_elapsed_time % 3600) // 60)

    print()
    print("="*80)
    print("🎉 三阶段训练全部完成！")
    print("="*80)
    print()
    print(f"总耗时: {hours}小时 {minutes}分钟")
    print()
    print("模型保存位置:")
    print(f"  Stage 1: {checkpoints[1]}")
    print(f"  Stage 2: {checkpoints[2]}")
    print(f"  Stage 3: {checkpoints[3]} (最终模型)")
    print()
    print("性能对比:")
    print("  查看各阶段的 train_history.json:")
    print(f"    cat checkpoints/stage1_backbone/train_history.json")
    print(f"    cat checkpoints/stage2_alignment/train_history.json")
    print(f"    cat checkpoints/stage3_moe/train_history.json")
    print()
    print("预期性能:")
    print("  Stage 1 (纯ID): HR@10 ≈ 0.60-0.70")
    print("  Stage 3 (MoE): HR@10 ≈ 0.70+")
    print()

    return 0


if __name__ == "__main__":
    exit(main())
