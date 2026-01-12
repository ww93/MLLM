"""
FedMem训练脚本：带本地动态记忆和原型聚合的联邦推荐系统（支持多模态数据）

使用方法：
    # 基本用法（仅ID特征）
    python scripts/train_fedmem.py --data_dir data/ml-1m --save_dir checkpoints/fedmem

    # 完整多模态用法
    python scripts/train_fedmem.py \
        --data_dir data/ml-1m \
        --visual_file item_images.npy \
        --text_file item_llm_texts.npy \
        --save_dir checkpoints/fedmem

核心特性：
1. **Two-tier本地动态记忆** (ST: 最近兴趣 + LT: 稳定多样性)
2. **Novelty-based LT写入** (数据驱动阈值，~10%写入率)
3. 记忆原型聚合（Prototype Aggregation，从LT提取）
4. 对比学习损失（Contrastive Loss）
5. 多模态特征加载（视觉 + 文本）
6. **[NEW] 轻量级Stage 2对齐** (投影层 <200K params)
"""
import os
import sys
import json
import argparse

# Debug print switch (set FEDMEM_DEBUG=1 to enable)
DEBUG = bool(int(os.environ.get('FEDMEM_DEBUG', '0')))

def dprint(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

import torch
import numpy as np
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.ur4rec_v2_moe import UR4RecV2MoE
from models.fedmem_simple import FedMemSimple  # [NEW] 简化架构
from models.fedmem_client import FedMemClient
from models.fedmem_server import FedMemServer



def str2bool(v):
    """Robust bool parser for argparse."""
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {v}")


def set_seed(seed: int = 42):
    """设置随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# [NEW] 多模态特征加载函数
def load_multimodal_features(
    data_dir: str,
    visual_file: Optional[str],
    text_file: Optional[str],
    num_items: int,
    device: str = 'cpu'
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], int, int]:
    """
    加载预提取的多模态特征

    Args:
        data_dir: 数据目录
        visual_file: 视觉特征文件名 (e.g., 'item_images.npy' or 'item_images.pt')
        text_file: 文本特征文件名 (e.g., 'item_llm_texts.npy' or 'item_llm_texts.pt')
        num_items: 物品总数
        device: 计算设备

    Returns:
        item_visual_feats: [num_items, img_dim] 或 None
        item_text_feats: [num_items, text_dim] 或 None
        img_dim: 视觉特征维度
        text_dim: 文本特征维度
    """
    item_visual_feats = None
    item_text_feats = None
    img_dim = 512  # 默认维度
    text_dim = 768  # 默认维度

    print(f"\n{'='*60}")
    print("加载多模态特征")
    print(f"{'='*60}")

    # ========== 加载视觉特征 ==========
    if visual_file:
        visual_path = os.path.join(data_dir, visual_file)

        if os.path.exists(visual_path):
            try:
                # 支持.npy和.pt格式
                if visual_path.endswith('.npy'):
                    visual_np = np.load(visual_path)
                    item_visual_feats = torch.from_numpy(visual_np).float().to(device)
                elif visual_path.endswith('.pt') or visual_path.endswith('.pth'):
                    item_visual_feats = torch.load(visual_path, map_location=device, weights_only=False)
                else:
                    raise ValueError(f"不支持的视觉特征文件格式: {visual_path}")

                # 验证形状
                if item_visual_feats.shape[0] != num_items:
                    print(f"⚠️ 警告: 视觉特征数量 ({item_visual_feats.shape[0]}) 与物品数量 ({num_items}) 不匹配")
                    print(f"   将创建零填充特征以匹配物品数量")

                    # 创建零填充特征
                    img_dim = item_visual_feats.shape[1]
                    padded_feats = torch.zeros(num_items, img_dim, device=device)
                    min_items = min(num_items, item_visual_feats.shape[0])
                    padded_feats[:min_items] = item_visual_feats[:min_items]
                    item_visual_feats = padded_feats
                else:
                    img_dim = item_visual_feats.shape[1]

                print(f"✓ 成功加载视觉特征: {visual_path}")
                print(f"  形状: {item_visual_feats.shape}")
                print(f"  数据类型: {item_visual_feats.dtype}")
                print(f"  统计: min={item_visual_feats.min():.4f}, max={item_visual_feats.max():.4f}, mean={item_visual_feats.mean():.4f}")

            except Exception as e:
                print(f"✗ 加载视觉特征失败: {e}")
                dprint(f"  将使用随机初始化的视觉特征（仅用于调试）")
                item_visual_feats = None
        else:
            print(f"⚠️ 警告: 视觉特征文件不存在: {visual_path}")
            dprint(f"  将使用随机初始化的视觉特征（仅用于调试）")

    # 如果没有加载成功，使用随机特征
    if visual_file and item_visual_feats is None:
        print(f"\n[DEBUG] 创建随机视觉特征: [{num_items}, {img_dim}]")
        item_visual_feats = torch.randn(num_items, img_dim, device=device) * 0.01
        dprint(f"⚠️ 警告: 使用随机视觉特征！这仅用于调试，不适合正式训练！")

    # ========== 加载文本特征 ==========
    if text_file:
        text_path = os.path.join(data_dir, text_file)

        if os.path.exists(text_path):
            try:
                # 支持.npy和.pt格式
                if text_path.endswith('.npy'):
                    text_np = np.load(text_path)
                    item_text_feats = torch.from_numpy(text_np).float().to(device)
                elif text_path.endswith('.pt') or text_path.endswith('.pth'):
                    item_text_feats = torch.load(text_path, map_location=device, weights_only=False)
                else:
                    raise ValueError(f"不支持的文本特征文件格式: {text_path}")

                # 验证形状
                if item_text_feats.shape[0] != num_items:
                    print(f"⚠️ 警告: 文本特征数量 ({item_text_feats.shape[0]}) 与物品数量 ({num_items}) 不匹配")
                    print(f"   将创建零填充特征以匹配物品数量")

                    # 创建零填充特征
                    text_dim = item_text_feats.shape[1]
                    padded_feats = torch.zeros(num_items, text_dim, device=device)
                    min_items = min(num_items, item_text_feats.shape[0])
                    padded_feats[:min_items] = item_text_feats[:min_items]
                    item_text_feats = padded_feats
                else:
                    text_dim = item_text_feats.shape[1]

                print(f"\n✓ 成功加载文本特征: {text_path}")
                print(f"  形状: {item_text_feats.shape}")
                print(f"  数据类型: {item_text_feats.dtype}")
                print(f"  统计: min={item_text_feats.min():.4f}, max={item_text_feats.max():.4f}, mean={item_text_feats.mean():.4f}")

            except Exception as e:
                print(f"✗ 加载文本特征失败: {e}")
                dprint(f"  将使用随机初始化的文本特征（仅用于调试）")
                item_text_feats = None
        else:
            print(f"\n⚠️ 警告: 文本特征文件不存在: {text_path}")
            dprint(f"  将使用随机初始化的文本特征（仅用于调试）")

    # 如果没有加载成功，使用随机特征
    if text_file and item_text_feats is None:
        print(f"\n[DEBUG] 创建随机文本特征: [{num_items}, {text_dim}]")
        item_text_feats = torch.randn(num_items, text_dim, device=device) * 0.01
        dprint(f"⚠️ 警告: 使用随机文本特征！这仅用于调试，不适合正式训练！")

    # ========== 总结 ==========
    print(f"\n{'='*60}")
    print("多模态特征加载总结")
    print(f"{'='*60}")
    print(f"视觉特征: {'✓ 已加载' if item_visual_feats is not None else '✗ 未加载'}")
    if item_visual_feats is not None:
        print(f"  维度: {img_dim}")
    print(f"文本特征: {'✓ 已加载' if item_text_feats is not None else '✗ 未加载'}")
    if item_text_feats is not None:
        print(f"  维度: {text_dim}")
    print(f"{'='*60}\n")

    return item_visual_feats, item_text_feats, img_dim, text_dim


# [UPDATED] 更新后的加载用户序列函数
def load_user_sequences(
    data_path: str,
    data_dir: str,
    visual_file: Optional[str] = None,
    text_file: Optional[str] = None,
    device: str = 'cpu'
) -> Tuple[Dict[int, List[int]], int, Optional[torch.Tensor], Optional[torch.Tensor], int, int]:
    """
    加载用户交互序列和多模态特征

    Args:
        data_path: 交互数据文件路径
        data_dir: 数据目录（用于加载多模态特征）
        visual_file: 视觉特征文件名
        text_file: 文本特征文件名
        device: 计算设备

    Returns:
        user_sequences: {user_id: [item_id1, item_id2, ...]}
        num_items: 物品总数
        item_visual_feats: [num_items, img_dim] 或 None
        item_text_feats: [num_items, text_dim] 或 None
        img_dim: 视觉特征维度
        text_dim: 文本特征维度
    """
    user_sequences = {}
    max_item_id = 0

    print(f"\n{'='*60}")
    print("加载用户交互序列")
    print(f"{'='*60}")

    # 检测数据格式并加载
    with open(data_path, 'r') as f:
        first_line = f.readline().strip()
        f.seek(0)  # 重置到文件开头

        parts = first_line.split()

        # 判断格式：
        # 格式1: user_id item_1 item_2 item_3 ... (一行多个items)
        # 格式2: user_id item_id rating timestamp (每行一条交互)

        if len(parts) > 4:
            # 格式1: 每行是一个用户的完整序列
            print("检测到格式: 每行一个用户序列")
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue

                user_id = int(parts[0])
                items = [int(x) for x in parts[1:]]
                user_sequences[user_id] = items

                if items:
                    max_item_id = max(max_item_id, max(items))
        else:
            # 格式2: 每行是一条交互记录，需要聚合
            print("检测到格式: 每行一条交互记录")
            user_interactions = {}

            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue

                user_id = int(parts[0])
                item_id = int(parts[1])

                # 如果有timestamp（第4个字段），用于排序
                timestamp = int(parts[3]) if len(parts) >= 4 else 0

                if user_id not in user_interactions:
                    user_interactions[user_id] = []

                user_interactions[user_id].append((timestamp, item_id))
                max_item_id = max(max_item_id, item_id)

            # 按时间排序并提取item序列
            for user_id, interactions in user_interactions.items():
                interactions.sort(key=lambda x: x[0])  # 按timestamp排序
                user_sequences[user_id] = [item_id for _, item_id in interactions]

    num_items = max_item_id + 1

    # 打印过滤前的统计
    print(f"✓ 原始用户数: {len(user_sequences)}")
    print(f"✓ 物品总数: {num_items}")

    if len(user_sequences) == 0:
        raise ValueError(
            f"❌ 没有加载到任何用户数据！\n"
            f"   请检查数据文件格式: {data_path}\n"
            f"   预期格式: user_id item_1 item_2 item_3 ...\n"
            f"   每行一个用户，空格分隔"
        )

    # 过滤掉序列太短的用户（至少需要5个item：train, val, test）
    original_user_count = len(user_sequences)
    user_sequences = {
        uid: seq for uid, seq in user_sequences.items()
        if len(seq) >= 5
    }

    if len(user_sequences) == 0:
        raise ValueError(
            f"❌ 过滤后没有符合条件的用户！\n"
            f"   原始用户数: {original_user_count}\n"
            f"   过滤条件: 序列长度 >= 5\n"
            f"   建议: 检查数据文件 {data_path} 的格式是否正确"
        )

    print(f"✓ 过滤后用户数: {len(user_sequences)} (过滤掉 {original_user_count - len(user_sequences)} 个)")

    # 计算统计信息
    seq_lengths = [len(seq) for seq in user_sequences.values()]
    print(f"  序列长度统计:")
    print(f"    最小: {min(seq_lengths)}")
    print(f"    最大: {max(seq_lengths)}")
    print(f"    平均: {sum(seq_lengths)/len(seq_lengths):.1f}")
    print(f"    总交互数: {sum(seq_lengths):,}")
    print(f"{'='*60}\n")

    # [NEW] 加载多模态特征
    item_visual_feats, item_text_feats, img_dim, text_dim = load_multimodal_features(
        data_dir=data_dir,
        visual_file=visual_file,
        text_file=text_file,
        num_items=num_items,
        device=device
    )

    return user_sequences, num_items, item_visual_feats, item_text_feats, img_dim, text_dim


# [UPDATED] 更新后的创建客户端函数
def create_fedmem_clients(
    user_sequences: Dict[int, List[int]],
    global_model: UR4RecV2MoE,
    item_visual_feats: Optional[torch.Tensor],
    item_text_feats: Optional[torch.Tensor],
    args: argparse.Namespace
) -> List[FedMemClient]:
    """
    创建FedMem客户端（支持多模态特征）

    Args:
        user_sequences: 用户交互序列
        global_model: 全局模型
        item_visual_feats: 物品视觉特征 [num_items, img_dim]
        item_text_feats: 物品文本特征 [num_items, text_dim]
        args: 训练参数

    Returns:
        clients: FedMemClient列表
    """
    clients = []

    print(f"\n{'='*60}")
    print("创建 FedMem 客户端")
    print(f"{'='*60}")

    for user_id, sequence in user_sequences.items():
        client = FedMemClient(
            client_id=user_id,
            model=global_model,
            user_sequence=sequence,
            device=args.device,
            # [NEW] 多模态特征
            item_visual_feats=item_visual_feats,
            item_text_feats=item_text_feats,
            # 训练参数
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            local_epochs=args.local_epochs,
            batch_size=args.batch_size,
            max_seq_len=args.max_seq_len,
            # 负采样
            num_negatives=args.num_negatives,
            num_items=args.num_items,
            # FedMem记忆参数
            memory_capacity=args.memory_capacity,
            surprise_threshold=args.surprise_threshold,
            contrastive_lambda=args.contrastive_lambda,
            num_memory_prototypes=args.num_memory_prototypes,
            # 负采样评估参数
            use_negative_sampling=args.use_negative_sampling,
            num_negatives_eval=args.num_negatives_eval
        )
        clients.append(client)

    print(f"✓ 创建了 {len(clients)} 个 FedMem 客户端")
    print(f"  每个客户端:")
    print(f"    - 视觉特征: {'启用' if item_visual_feats is not None else '禁用'}")
    print(f"    - 文本特征: {'启用' if item_text_feats is not None else '禁用'}")
    print(f"    - 记忆架构: Two-tier (ST: 50, LT: {args.memory_capacity})")
    print(f"    - LT写入策略: Novelty-based (threshold=0.583)")
    print(f"    - 兼容参数 surprise_threshold: {args.surprise_threshold}")
    print(f"{'='*60}\n")

    return clients


def main():
    parser = argparse.ArgumentParser(description="FedMem训练脚本（支持多模态）")

    # 数据参数
    parser.add_argument("--data_dir", type=str, default="UR4Rec/data/ml-1m",
                        help="数据目录")
    parser.add_argument("--data_file", type=str, default="subset_ratings.dat",
                        help="交互数据文件名")

    # [NEW] 多模态特征文件参数
    parser.add_argument("--visual_file", type=str, default="clip_features.pt",
                        help="视觉特征文件名 (e.g., 'item_images.npy' or 'item_images.pt')")
    parser.add_argument("--text_file", type=str, default="text_features.pt",
                        help="文本特征文件名 (e.g., 'item_llm_texts.npy' or 'item_llm_texts.pt')")

    # 模型参数
    parser.add_argument("--model_type", type=str, default="moe",
                        choices=["moe", "simple"],
                        help="模型架构类型: 'moe' (MoE架构) 或 'simple' (简化架构)")

    # [NEW] 三阶段训练参数
    parser.add_argument("--stage", type=str, default="full",
                        choices=["pretrain_sasrec", "align_projectors", "finetune_moe", "full"],
                        help="训练阶段:\n"
                             "  pretrain_sasrec: 第一阶段，纯ID SASRec预训练\n"
                             "  align_projectors: 第二阶段，多模态投影层对齐\n"
                             "  finetune_moe: 第三阶段，MoE集成微调\n"
                             "  full: 完整训练（默认）")
    parser.add_argument("--stage1_checkpoint", type=str, default=None,
                        help="第一阶段checkpoint路径（用于stage2和stage3）")
    parser.add_argument("--stage2_checkpoint", type=str, default=None,
                        help="第二阶段checkpoint路径（用于stage3）")

    parser.add_argument("--num_items", type=int, default=1682,
                        help="物品总数（自动检测如果未指定）")
    parser.add_argument("--sasrec_hidden_dim", type=int, default=256,
                        help="SASRec隐藏层维度")
    parser.add_argument("--sasrec_num_blocks", type=int, default=2,
                        help="SASRec Transformer块数量")
    parser.add_argument("--sasrec_num_heads", type=int, default=4,
                        help="SASRec注意力头数量")
    parser.add_argument("--retriever_output_dim", type=int, default=256,
                        help="Retriever输出维度")
    parser.add_argument("--moe_num_heads", type=int, default=8,
                        help="MoE注意力头数量")
    parser.add_argument("--max_seq_len", type=int, default=50,
                        help="最大序列长度")

    # [NEW] 简化架构专用参数
    parser.add_argument("--id_emb_dim", type=int, default=128,
                        help="[简化架构] ID嵌入维度")
    parser.add_argument("--visual_proj_dim", type=int, default=64,
                        help="[简化架构] 视觉特征投影维度")
    parser.add_argument("--text_proj_dim", type=int, default=64,
                        help="[简化架构] 文本特征投影维度")

    # FedMem参数 (Two-tier Memory: ST + LT)
    parser.add_argument("--memory_capacity", type=int, default=200,
                        help="LT (long-term) 记忆容量，推荐200 (ML-1M), ST固定50")
    parser.add_argument("--surprise_threshold", type=float, default=0.5,
                        help="兼容参数，新版本主要使用novelty-based写入 (默认0.583)")
    parser.add_argument("--contrastive_lambda", type=float, default=0.05,
                        help="对比学习损失权重")
    parser.add_argument("--num_memory_prototypes", type=int, default=5,
                        help="记忆原型数量（从LT提取）")
    parser.add_argument("--enable_prototype_aggregation", action="store_true",
                        help="启用原型聚合")

    # 联邦学习参数
    parser.add_argument("--num_rounds", type=int, default=50,
                        help="联邦学习轮数")
    parser.add_argument("--client_fraction", type=float, default=0.1,
                        help="每轮参与的客户端比例")
    parser.add_argument("--local_epochs", type=int, default=1,
                        help="客户端本地训练轮数")
    parser.add_argument("--aggregation_method", type=str, default="fedavg",
                        choices=["fedavg", "fedprox"],
                        help="聚合方法")
    parser.add_argument("--patience", type=int, default=10,
                        help="早停patience")

    # 训练参数
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="权重衰减")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="批大小")
    parser.add_argument("--num_negatives", type=int, default=4,
                        help="负样本数量")

    # 负采样评估参数
    parser.add_argument("--use_negative_sampling", type=str2bool, default=True,
                        help="使用1:100负采样评估（对齐NCF/SASRec论文）")
    parser.add_argument("--num_negatives_eval", type=int, default=100,
                        help="评估时的负样本数量（默认100）")

    # 【残差增强】Residual Enhancement 参数
    parser.add_argument("--gating_init", type=float, default=0.1,
                        help="门控权重初始值（推荐0.0-0.1），控制辅助信息注入强度")

    # 【策略1】Router Bias Initialization 参数 [已废弃，保留向后兼容]
    parser.add_argument("--init_bias_for_sasrec", action="store_true",
                        help="[已废弃] 启用Router Bias Initialization（策略1）")
    parser.add_argument("--sasrec_bias_value", type=float, default=5.0,
                        help="[已废弃] SASRec expert的bias初始值")

    # 【策略2】Partial Aggregation 参数
    parser.add_argument("--partial_aggregation_warmup_rounds", type=int, default=20,
                        help="Warmup轮数，前N轮只聚合SASRec参数（策略2），0表示禁用")

    # 其他参数
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="计算设备")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--save_dir", type=str, default="checkpoints/fedmem",
                        help="模型保存目录")
    parser.add_argument("--verbose", action="store_true",
                        help="打印详细训练信息")

    # [NEW] 预训练权重加载
    parser.add_argument("--pretrained_path", type=str, default=None,
                        help="预训练模型路径（用于迁移学习）。加载SASRec骨干权重，跳过Warmup阶段")

    args = parser.parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 保存配置
    with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    print(f"\n{'='*60}")
    print("FedMem训练配置")
    print(f"{'='*60}")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print(f"{'='*60}\n")

    # ============================================
    # 1. [UPDATED] 加载数据（包含多模态特征）
    # ============================================
    print("\n[1/4] 加载数据...")
    data_path = os.path.join(args.data_dir, args.data_file)

    if not os.path.exists(data_path):
        print(f"错误：数据文件不存在 {data_path}")
        print("请确保数据文件存在或使用正确的路径")
        return

    # [三阶段训练] 第一阶段：纯ID训练，禁用多模态
    if args.stage == "pretrain_sasrec":
        print(f"  [Stage 1] 纯ID SASRec预训练 - 禁用多模态特征加载")
        visual_file_to_load = None
        text_file_to_load = None

        # [Stage 1关键修复] Stage 1目标是复现FedSASRec性能；partial warmup若只聚合'sasrec'会遗漏item/pos embedding，导致global模型无法对齐。
        if args.partial_aggregation_warmup_rounds != 0:
            print(f"  [Stage 1] 自动关闭partial warmup: {args.partial_aggregation_warmup_rounds} -> 0")
            args.partial_aggregation_warmup_rounds = 0
    else:
        visual_file_to_load = args.visual_file
        text_file_to_load = args.text_file

    # [NEW] 加载交互序列 + 多模态特征
    user_sequences, num_items, item_visual_feats, item_text_feats, img_dim, text_dim = load_user_sequences(
        data_path=data_path,
        data_dir=args.data_dir,
        visual_file=visual_file_to_load,
        text_file=text_file_to_load,
        device=args.device
    )
    args.num_items = num_items  # 更新num_items

    # ============================================
    # 2. [UPDATED] 创建全局模型（根据model_type选择架构）
    # ============================================
    # [NEW] 使用从数据加载得到的实际维度
    # 如果没有加载多模态特征，使用默认维度
    actual_text_dim = text_dim if item_text_feats is not None else 384
    actual_img_dim = img_dim if item_visual_feats is not None else 512

    # 根据model_type选择模型架构
    if args.model_type == "moe":
        print("\n[2/4] 创建全局 UR4RecV2MoE 模型（MoE架构）...")
        print(f"  模型配置:")
        print(f"    - 架构: MoE (Mixture of Experts)")
        print(f"    - 物品数: {args.num_items}")
        print(f"    - 文本特征维度: {actual_text_dim}")
        print(f"    - 图像特征维度: {actual_img_dim}")
        print(f"    - SASRec隐藏维度: {args.sasrec_hidden_dim}")
        print(f"    - MoE隐藏维度: {args.sasrec_hidden_dim}")

        global_model = UR4RecV2MoE(
            num_items=args.num_items,
            # SASRec参数
            sasrec_hidden_dim=args.sasrec_hidden_dim,
            sasrec_num_blocks=args.sasrec_num_blocks,
            sasrec_num_heads=args.sasrec_num_heads,
            sasrec_dropout=0.1,
            max_seq_len=args.max_seq_len,
            # 多模态特征维度
            visual_dim=actual_img_dim,  # CLIP特征维度
            text_dim=actual_text_dim,   # Sentence-BERT特征维度
            # MoE参数
            moe_hidden_dim=args.sasrec_hidden_dim,  # 与SASRec保持一致
            moe_num_heads=args.moe_num_heads,
            moe_dropout=0.1,
            router_hidden_dim=128,
            # 残差增强参数
            gating_init=args.gating_init,
            # 负载均衡
            load_balance_lambda=0.01,
            # 【策略1】Router Bias Initialization [已废弃，保留向后兼容]
            init_bias_for_sasrec=args.init_bias_for_sasrec,
            sasrec_bias_value=args.sasrec_bias_value,
            # 设备
            device=args.device
        )

    elif args.model_type == "simple":
        print("\n[2/4] 创建全局 FedMemSimple 模型（简化架构）...")

        # 计算总的输入维度
        total_input_dim = args.id_emb_dim + args.visual_proj_dim + args.text_proj_dim

        print(f"  模型配置:")
        print(f"    - 架构: Simple (直接拼接)")
        print(f"    - 物品数: {args.num_items}")
        print(f"    - ID嵌入维度: {args.id_emb_dim}")
        print(f"    - 视觉投影维度: {actual_img_dim} → {args.visual_proj_dim}")
        print(f"    - 文本投影维度: {actual_text_dim} → {args.text_proj_dim}")
        print(f"    - 拼接后总维度: {total_input_dim}")
        print(f"    - SASRec输入维度: {total_input_dim}")

        global_model = FedMemSimple(
            num_items=args.num_items,
            # ID embedding维度
            id_emb_dim=args.id_emb_dim,
            # 多模态特征维度
            visual_dim=actual_img_dim,      # CLIP特征
            text_dim=actual_text_dim,       # Sentence-BERT特征
            # 投影维度
            visual_proj_dim=args.visual_proj_dim,
            text_proj_dim=args.text_proj_dim,
            # SASRec参数
            sasrec_num_blocks=args.sasrec_num_blocks,
            sasrec_num_heads=args.sasrec_num_heads,
            sasrec_dropout=0.1,
            max_seq_len=args.max_seq_len,
            # 设备
            device=args.device
        )

    else:
        raise ValueError(f"未知的model_type: {args.model_type}. 支持: 'moe', 'simple'")

    print(f"\n✓ 模型创建成功!")
    print(f"  总参数数量: {sum(p.numel() for p in global_model.parameters()):,}")
    trainable_params = sum(p.numel() for p in global_model.parameters() if p.requires_grad)
    print(f"  可训练参数: {trainable_params:,}")

    # ============================================
    # 2.5. [NEW] 加载预训练权重（可选）
    # ============================================
    if args.pretrained_path is not None:
        print(f"\n[2.5/4] 加载预训练权重...")
        print(f"  路径: {args.pretrained_path}")

        if os.path.exists(args.pretrained_path):
            try:
                # 加载预训练模型（PyTorch 2.6+需要weights_only=False来加载包含numpy对象的checkpoint）
                pretrained_state = torch.load(args.pretrained_path, map_location=args.device, weights_only=False)

                # 只加载SASRec骨干权重（兼容性加载）
                current_state = global_model.state_dict()
                loaded_keys = []
                skipped_keys = []

                for key, value in pretrained_state.items():
                    # 优先加载SASRec相关参数
                    if 'sasrec' in key.lower() or 'item_emb' in key.lower():
                        if key in current_state and current_state[key].shape == value.shape:
                            current_state[key] = value
                            loaded_keys.append(key)
                        else:
                            skipped_keys.append(key)
                    # 可选：加载Router和LayerNorm（如果形状匹配）
                    elif 'router' in key.lower() or 'layernorm' in key.lower():
                        if key in current_state and current_state[key].shape == value.shape:
                            current_state[key] = value
                            loaded_keys.append(key)
                        else:
                            skipped_keys.append(key)
                    else:
                        skipped_keys.append(key)

                # 应用加载的权重
                global_model.load_state_dict(current_state)

                print(f"  ✓ 成功加载 {len(loaded_keys)} 个参数")
                print(f"    主要模块: SASRec骨干、Item嵌入、Router")
                if len(skipped_keys) > 0:
                    print(f"  ⚠️  跳过 {len(skipped_keys)} 个参数（形状不匹配或非骨干参数）")

                # 重要提示
                print(f"\n  📌 预训练权重已加载，建议:")
                print(f"    - 使用较小的学习率（如1e-4, 当前{args.learning_rate}）")
                print(f"    - 减少训练轮数（当前{args.num_rounds}轮）")
                print(f"    - 直接跳过Warmup（设置partial_aggregation_warmup_rounds=0）")

            except Exception as e:
                print(f"  ✗ 加载预训练权重失败: {e}")
                print(f"  将使用随机初始化继续训练")
        else:
            print(f"  ✗ 预训练权重文件不存在: {args.pretrained_path}")
            print(f"  将使用随机初始化继续训练")

    # ============================================
    # 3. [三阶段训练] Checkpoint加载与模型更新 (在创建客户端之前)
    # ============================================
    print(f"\n[3/4] 三阶段训练策略 - Checkpoint加载...")

    if args.stage == "pretrain_sasrec":
        # ===== 第一阶段：纯ID SASRec预训练 =====
        print(f"  [Stage 1: Backbone Pre-training]")
        print(f"  目标: 训练高质量的纯ID SASRec (预期 HR@10 ≈ 0.60-0.70)")
        print(f"  训练对象: SASRec (Embedding + Transformer)")
        print(f"  数据: 仅Item ID序列")
        print(f"  冻结: 无")
        print(f"  ✓ 所有参数可训练")

    elif args.stage == "align_projectors":
        # ===== 第二阶段：多模态投影层对齐 =====
        print(f"  [Stage 2: Modality Alignment]")
        print(f"  目标: 让多模态特征对齐到ID空间")
        print(f"  训练对象: Visual/Semantic Projectors")
        print(f"  冻结: SASRec + Item Embedding")

        # [关键修复] Stage 2禁用warmup
        # 原因：warmup只聚合SASRec，但Stage 2冻结了SASRec，训练的是投影层
        if args.partial_aggregation_warmup_rounds > 0:
            print(f"  ⚠️  警告: Stage 2应该禁用warmup（当前设置={args.partial_aggregation_warmup_rounds}）")
            print(f"  原因: warmup只聚合SASRec，但Stage 2训练的是投影层")
            print(f"  自动禁用warmup...")
            args.partial_aggregation_warmup_rounds = 0

        # 加载Stage 1 checkpoint
        if args.stage1_checkpoint and os.path.exists(args.stage1_checkpoint):
            print(f"  加载Stage 1 checkpoint: {args.stage1_checkpoint}")
            try:
                checkpoint = torch.load(args.stage1_checkpoint, map_location=args.device, weights_only=False)
                state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint

                # [方案2修复] Stage 2时不加载expert和LayerNorm参数，因为维度已改变
                # Stage 1的expert是128维，方案2的expert是512/384维
                # Stage 1的LayerNorm也是128维，需要跳过避免覆盖新的512/384维LayerNorm
                filtered_state_dict = {}
                skipped_keys = []
                for key, value in state_dict.items():
                    # 跳过visual_expert、semantic_expert、cross_modal_fusion和相关LayerNorm的参数
                    if any(pattern in key for pattern in [
                        'visual_expert', 'semantic_expert', 'cross_modal_fusion',
                        'vis_layernorm', 'sem_layernorm'
                    ]):
                        skipped_keys.append(key)
                        continue
                    filtered_state_dict[key] = value

                if skipped_keys:
                    dprint(f"  [方案2] 跳过加载expert和LayerNorm参数（维度已改变）: {len(skipped_keys)}个")
                    for key in skipped_keys[:3]:
                        print(f"     - {key}")
                    if len(skipped_keys) > 3:
                        print(f"     - ... 还有{len(skipped_keys)-3}个")

                # [STAGE 2/3 FIX] strict=False允许部分加载，忽略missing keys（如gating_weight）
                missing_keys, unexpected_keys = global_model.load_state_dict(filtered_state_dict, strict=False)

                print(f"  ✓ 成功加载Stage 1权重到global_model")
                if missing_keys:
                    dprint(f"  ℹ️  新增参数（Stage 1 checkpoint中不存在）: {len(missing_keys)}个")
                    for key in missing_keys[:3]:  # 只显示前3个
                        print(f"     - {key} (将使用随机初始化)")
                    if len(missing_keys) > 3:
                        print(f"     - ... 还有{len(missing_keys)-3}个")

                # [调试] 验证权重确实被加载 - 检查关键参数
                param_stats = []
                for name, param in global_model.named_parameters():
                    if 'item_emb' in name.lower() or 'sasrec' in name.lower():
                        param_stats.append((name, param.mean().item(), param.std().item(), param.abs().max().item()))
                        if len(param_stats) >= 3:  # 只打印前3个关键参数
                            break

                dprint(f"  [调试] 关键参数统计（验证是否真的加载了训练好的权重）:")
                for name, mean, std, max_val in param_stats:
                    print(f"    {name}: mean={mean:.4f}, std={std:.4f}, max={max_val:.4f}")
                dprint(f"  [调试] 如果是训练好的权重，mean和max应该有明显的非零值")

                # [方案2调试] 验证expert和LayerNorm的维度
                dprint(f"\n  [方案2调试] 验证模型维度设置:")
                print(f"    preserve_multimodal_dim: {global_model.preserve_multimodal_dim}")
                print(f"    visual_expert.output_dim: {global_model.visual_expert.output_dim}")
                print(f"    semantic_expert.output_dim: {global_model.semantic_expert.output_dim}")
                print(f"    vis_layernorm.normalized_shape: {global_model.vis_layernorm.normalized_shape}")
                print(f"    sem_layernorm.normalized_shape: {global_model.sem_layernorm.normalized_shape}")

            except Exception as e:
                print(f"  ✗ 加载失败: {e}")
        else:
            print(f"  ⚠️  警告: 未提供Stage 1 checkpoint，使用随机初始化")

        print(f"  ✓ Checkpoint加载完成，冻结策略将在创建客户端后应用")

    elif args.stage == "finetune_moe":
        # ===== 第三阶段：MoE集成微调 =====
        print(f"  [Stage 3: MoE Fine-tuning]")
        print(f"  目标: 学习Router (什么时候用谁)")
        print(f"  微调 (小LR): SASRec Transformer, Visual/Semantic Projectors")
        print(f"  全速训练: MoE Router")
        print(f"  冻结: Item Embedding (锚点)")

        # [关键修复] Stage 3禁用warmup
        # 原因：warmup只聚合SASRec，但Stage 3需要聚合Transformer、投影层、Router
        if args.partial_aggregation_warmup_rounds > 0:
            print(f"  ⚠️  警告: Stage 3应该禁用warmup（当前设置={args.partial_aggregation_warmup_rounds}）")
            print(f"  原因: warmup只聚合SASRec，但Stage 3需要聚合多个组件")
            print(f"  自动禁用warmup...")
            args.partial_aggregation_warmup_rounds = 0

        # 加载Stage 1 checkpoint (backbone)
        if args.stage1_checkpoint and os.path.exists(args.stage1_checkpoint):
            print(f"  加载Stage 1 checkpoint: {args.stage1_checkpoint}")
            try:
                checkpoint = torch.load(args.stage1_checkpoint, map_location=args.device, weights_only=False)
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint

                # [关键修复] 加载SASRec + Item Embedding
                # 原因：Stage 3冻结item_emb，必须加载训练好的embedding
                # 否则会出现"训练好的SASRec + 随机的embedding"的不匹配
                current_state = global_model.state_dict()
                loaded = 0
                for key, value in state_dict.items():
                    # 加载 SASRec 和 item_emb
                    if ('sasrec' in key.lower() or 'item_emb' in key.lower()) and key in current_state:
                        current_state[key] = value
                        loaded += 1
                global_model.load_state_dict(current_state)
                print(f"  ✓ 成功加载Stage 1权重到global_model ({loaded}个参数)")
                print(f"     包括: SASRec骨干 + Item Embedding")
            except Exception as e:
                print(f"  ✗ 加载Stage 1失败: {e}")

        # 加载Stage 2 checkpoint (projectors)
        if args.stage2_checkpoint and os.path.exists(args.stage2_checkpoint):
            print(f"  加载Stage 2 checkpoint: {args.stage2_checkpoint}")
            try:
                checkpoint = torch.load(args.stage2_checkpoint, map_location=args.device, weights_only=False)
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint

                # [关键修复] Stage 3只加载投影层和gating，跳过随机的MoE组件
                # 原因：Stage 2训练时MoE组件（router/expert/fusion）是冻结的（随机状态）
                #       Stage 3不应该加载这些随机参数，应该用自己的初始化
                current_state = global_model.state_dict()
                loaded = 0
                skipped_shape = []
                skipped_random_moe = []
                for key, value in state_dict.items():
                    # [Stage 3关键] 只加载Stage 2训练过的组件
                    # ✓ 加载: visual_proj, text_proj, align_gating, gating_weight
                    # ✗ 跳过: router, expert, cross_modal_fusion (这些在Stage 2是冻结/随机的)
                    should_load = (
                        ('proj' in key.lower() and 'expert' not in key.lower()) or  # 投影层（非Expert内部的proj）
                        'align_gating' in key.lower() or    # Stage 2训练的对齐门控
                        'gating_weight' in key.lower()      # 残差融合权重（核心！）
                    )

                    # 记录跳过的MoE组件（用于调试）
                    is_moe_component = (
                        'router' in key.lower() or
                        'expert' in key.lower() or
                        'cross_modal_fusion' in key.lower() or
                        ('layernorm' in key.lower() and any(x in key.lower() for x in ['vis_', 'sem_', 'seq_']))
                    )
                    if is_moe_component:
                        skipped_random_moe.append(key)

                    if should_load and key in current_state:
                        # 形状检查：只加载形状匹配的参数
                        if current_state[key].shape == value.shape:
                            current_state[key] = value
                            loaded += 1
                        else:
                            skipped_shape.append(f"{key} (ckpt:{value.shape} vs model:{current_state[key].shape})")

                # 打印跳过的MoE组件（重要调试信息）
                if skipped_random_moe:
                    print(f"  ℹ️  跳过Stage 2中随机的MoE组件 ({len(skipped_random_moe)}个):")
                    print(f"     原因: 这些组件在Stage 2是冻结的（未训练），不应该加载")
                    print(f"     跳过: router ({sum(1 for k in skipped_random_moe if 'router' in k)}), "
                          f"expert ({sum(1 for k in skipped_random_moe if 'expert' in k)}), "
                          f"fusion ({sum(1 for k in skipped_random_moe if 'fusion' in k)}), "
                          f"layernorm ({sum(1 for k in skipped_random_moe if 'layernorm' in k)})")

                if skipped_shape:
                    print(f"  ℹ️  跳过形状不匹配的参数 ({len(skipped_shape)}个):")
                    for item in skipped_shape[:5]:  # 只显示前5个
                        print(f"     - {item}")
                    if len(skipped_shape) > 5:
                        print(f"     - ... 还有{len(skipped_shape)-5}个")

                global_model.load_state_dict(current_state)
                print(f"  ✓ 成功加载Stage 2权重到global_model ({loaded}个参数)")
                print(f"     加载: visual_proj, text_proj, align_gating, gating_weight")
                print(f"     保持Stage 3自己的初始化: router, experts, fusion")
                # [关键验证] 打印gating_weight实际值
                if hasattr(global_model, 'gating_weight'):
                    print(f"  ✓ 验证: gating_weight = {global_model.gating_weight.item():.6f}")
            except Exception as e:
                print(f"  ✗ 加载Stage 2失败: {e}")

        print(f"  ✓ Checkpoint加载完成，冻结策略将在创建客户端后应用")

    elif args.stage == "full":
        # ===== 完整训练（原有逻辑） =====
        if args.pretrained_path is not None and os.path.exists(args.pretrained_path):
            print(f"  [Full Training] 使用小学习率微调（不冻结embedding）")
            print(f"  原因: 需要embedding与多模态特征对齐")
            print(f"  ✓ 所有参数保持可训练，使用学习率{args.learning_rate}")
        else:
            print(f"  [Full Training] 从零开始训练")
            print(f"  ✓ 所有参数可训练")

    # ============================================
    # 3.5. [UPDATED] 创建FedMem客户端 (在checkpoint加载和模型更新之后)
    # ============================================
    print("\n[3.5/4] 创建 FedMem 客户端...")

    # [NEW] 传递多模态特征到客户端
    clients = create_fedmem_clients(
        user_sequences=user_sequences,
        global_model=global_model,
        item_visual_feats=item_visual_feats,  # [NEW]
        item_text_feats=item_text_feats,      # [NEW]
        args=args
    )

    # ============================================
    # 3.6. [三阶段训练] 参数冻结策略 (客户端已创建)
    # ============================================
    if args.stage == "align_projectors":
        print(f"\n[3.6/4] 应用Stage 2冻结策略（轻量级对齐）...")
        print(f"  ✓ 目标: 训练投影层，将多模态特征对齐到ID空间")
        print(f"  ✓ 参数量: <200K (vs 原方案 ~4M)")
        print(f"  冻结: SASRec + Item Embedding + Experts + CrossModalFusion + Router")
        print(f"  训练: visual_proj (512→128) + text_proj (384→128) + align_gating MLP")

        # 应用冻结策略到所有客户端
        for client in clients:
            client._ensure_model_initialized()
            frozen_params = []
            trainable_params_names = []

            for name, param in client.model.named_parameters():
                k = name.lower()
                # [Stage 2核心] 只训练投影层和对齐门控
                if 'visual_proj' in k or 'text_proj' in k or 'align_gating' in k:
                    param.requires_grad = True
                    trainable_params_names.append(name)
                else:
                    # 冻结其他所有参数：SASRec, Experts, CrossModalFusion, Router, LayerNorms, Gating Weight
                    param.requires_grad = False
                    frozen_params.append(name)

            # 重建优化器（只包含可训练参数）
            trainable_params = [p for p in client.model.parameters() if p.requires_grad]
            client.optimizer = torch.optim.Adam(
                trainable_params,
                lr=client.learning_rate,
                weight_decay=client.weight_decay
            )

            # 统计可训练参数数量
            if client.client_id == list(user_sequences.keys())[0]:  # 只打印第一个客户端
                num_trainable = sum(p.numel() for p in trainable_params)
                print(f"  示例客户端 {client.client_id}:")
                print(f"    - 冻结参数: {len(frozen_params)}个")
                print(f"    - 可训练参数: {len(trainable_params_names)}个 (~{num_trainable:,} params)")
                print(f"    - 可训练层: {', '.join(trainable_params_names)}")

        print(f"  ✓ 所有 {len(clients)} 个客户端已应用Stage 2轻量级冻结策略")

    elif args.stage == "finetune_moe":
        print(f"\n[3.6/4] Stage 3: 三阶段渐进式解冻策略")
        print(f"  [方案1] 渐进式解冻将在训练过程中动态应用：")
        print(f"    Stage 3a (Round 0-9):  冻结 SASRec+投影层+Experts+Fusion, 训练 Router")
        print(f"    Stage 3b (Round 10-29): 冻结 SASRec+投影层, 训练 Router+Experts+Fusion")
        print(f"    Stage 3c (Round 30-49): 冻结 item_emb, 训练 所有其他参数 (LR=1e-5)")
        print(f"  ✓ 目标: 渐进解冻，避免破坏Stage 2学到的SASRec-投影层配合")

        # [关键验证] 检查gating_weight是否正确加载
        sample_client = clients[0]
        sample_client._ensure_model_initialized()
        if hasattr(sample_client.model, 'gating_weight'):
            print(f"  ✓ 验证: 客户端 {sample_client.client_id} gating_weight = {sample_client.model.gating_weight.item():.6f}")

        print(f"  ✓ 渐进式冻结策略将在每轮训练时动态应用")

        # 注：原有的静态冻结策略已移除，改为在FedMemServer.train_round中动态应用

    # ============================================
    # 4. 创建FedMem服务器并开始训练
    # ============================================
    print("\n[4/4] 创建 FedMem 服务器并开始训练...")

    server = FedMemServer(
        global_model=global_model,
        clients=clients,
        device=args.device,
        # 联邦学习参数
        aggregation_method=args.aggregation_method,
        client_fraction=args.client_fraction,
        num_rounds=args.num_rounds,
        local_epochs=args.local_epochs,
        patience=args.patience,
        # FedMem参数
        enable_prototype_aggregation=args.enable_prototype_aggregation,
        num_memory_prototypes=args.num_memory_prototypes,
        # 【策略2】Partial Aggregation
        partial_aggregation_warmup_rounds=args.partial_aggregation_warmup_rounds,
        # [方案1] 渐进式解冻
        stage=args.stage
    )

    # 开始训练（传递user_sequences用于负采样评估）
    train_history = server.train(user_sequences=user_sequences, verbose=args.verbose or True)

    # ============================================
    # 5. 保存模型和结果
    # ============================================
    print("\n保存模型和训练历史...")

    # 保存模型
    model_path = os.path.join(args.save_dir, 'fedmem_model.pt')
    server.save_model(model_path)

    # 保存训练历史
    history_path = os.path.join(args.save_dir, 'train_history.json')
    with open(history_path, 'w') as f:
        # 将tensor转换为list以便JSON序列化
        history_serializable = {}
        for key, value in train_history.items():
            if isinstance(value, list):
                history_serializable[key] = [
                    {k: float(v) if isinstance(v, (int, float)) else v
                     for k, v in item.items()}
                    if isinstance(item, dict) else item
                    for item in value
                ]
            elif isinstance(value, dict):
                history_serializable[key] = {
                    k: float(v) if isinstance(v, (int, float)) else v
                    for k, v in value.items()
                }
            else:
                history_serializable[key] = value

        json.dump(history_serializable, f, indent=2)

    print(f"✓ 模型已保存到: {model_path}")
    print(f"✓ 训练历史已保存到: {history_path}")

    # ============================================
    # 6. 打印最终结果
    # ============================================
    print(f"\n{'='*60}")
    print("最终结果")
    print(f"{'='*60}")

    test_metrics = train_history['test_metrics']
    print("\n测试集指标:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.4f}")

    best_metrics = server.get_best_metrics()
    dprint(f"\n最佳验证轮次: {best_metrics.get('round', -1) + 1}")
    dprint("最佳验证指标:")
    for key, value in best_metrics.items():
        if key != 'round':
            print(f"  {key}: {value:.4f}")

    # [NEW] 打印多模态使用情况
    print(f"\n{'='*60}")
    print("多模态使用情况")
    print(f"{'='*60}")
    print(f"视觉特征: {'✓ 使用' if item_visual_feats is not None else '✗ 未使用'}")
    print(f"文本特征: {'✓ 使用' if item_text_feats is not None else '✗ 未使用'}")
    if item_visual_feats is None and item_text_feats is None:
        print("\n⚠️ 注意: 未加载任何多模态特征！")
        print("   建议使用 --visual_file 和 --text_file 参数加载多模态数据")
        print("   以获得更好的推荐效果。")

    print(f"\n{'='*60}")
    print("训练完成！")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()