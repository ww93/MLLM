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
1. 本地动态记忆（LocalDynamicMemory）
2. Surprise-based记忆更新
3. 记忆原型聚合（Prototype Aggregation）
4. 对比学习损失（Contrastive Loss）
5. **[NEW] 多模态特征加载（视觉 + 文本）**
"""
import os
import sys
import json
import argparse
import torch
import numpy as np
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.ur4rec_v2_moe import UR4RecV2MoE
from models.fedmem_client import FedMemClient
from models.fedmem_server import FedMemServer


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
                    item_visual_feats = torch.load(visual_path, map_location=device)
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
                print(f"  将使用随机初始化的视觉特征（仅用于调试）")
                item_visual_feats = None
        else:
            print(f"⚠️ 警告: 视觉特征文件不存在: {visual_path}")
            print(f"  将使用随机初始化的视觉特征（仅用于调试）")

    # 如果没有加载成功，使用随机特征
    if visual_file and item_visual_feats is None:
        print(f"\n[DEBUG] 创建随机视觉特征: [{num_items}, {img_dim}]")
        item_visual_feats = torch.randn(num_items, img_dim, device=device) * 0.01
        print(f"⚠️ 警告: 使用随机视觉特征！这仅用于调试，不适合正式训练！")

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
                    item_text_feats = torch.load(text_path, map_location=device)
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
                print(f"  将使用随机初始化的文本特征（仅用于调试）")
                item_text_feats = None
        else:
            print(f"\n⚠️ 警告: 文本特征文件不存在: {text_path}")
            print(f"  将使用随机初始化的文本特征（仅用于调试）")

    # 如果没有加载成功，使用随机特征
    if text_file and item_text_feats is None:
        print(f"\n[DEBUG] 创建随机文本特征: [{num_items}, {text_dim}]")
        item_text_feats = torch.randn(num_items, text_dim, device=device) * 0.01
        print(f"⚠️ 警告: 使用随机文本特征！这仅用于调试，不适合正式训练！")

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
    print(f"    - 记忆容量: {args.memory_capacity}")
    print(f"    - Surprise阈值: {args.surprise_threshold}")
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

    # FedMem参数
    parser.add_argument("--memory_capacity", type=int, default=50,
                        help="本地记忆容量")
    parser.add_argument("--surprise_threshold", type=float, default=0.3,
                        help="Surprise阈值")
    parser.add_argument("--contrastive_lambda", type=float, default=0.2,
                        help="对比学习损失权重")
    parser.add_argument("--num_memory_prototypes", type=int, default=5,
                        help="记忆原型数量")
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
    parser.add_argument("--learning_rate", type=float, default=5e-3,
                        help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="权重衰减")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="批大小")
    parser.add_argument("--num_negatives", type=int, default=100,
                        help="负样本数量")

    # 负采样评估参数
    parser.add_argument("--use_negative_sampling", default=True,
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

    # [NEW] 加载交互序列 + 多模态特征
    user_sequences, num_items, item_visual_feats, item_text_feats, img_dim, text_dim = load_user_sequences(
        data_path=data_path,
        data_dir=args.data_dir,
        visual_file=args.visual_file,
        text_file=args.text_file,
        device=args.device
    )
    args.num_items = num_items  # 更新num_items

    # ============================================
    # 2. [UPDATED] 创建全局模型（使用实际的特征维度）
    # ============================================
    print("\n[2/4] 创建全局 UR4RecV2MoE 模型...")

    # [NEW] 使用从数据加载得到的实际维度
    # 如果没有加载多模态特征，使用默认维度
    actual_text_dim = text_dim if item_text_feats is not None else 384
    actual_img_dim = img_dim if item_visual_feats is not None else 512

    print(f"  模型配置:")
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
    # 3. [UPDATED] 创建FedMem客户端（传入多模态特征）
    # ============================================
    print("\n[3/4] 创建 FedMem 客户端...")

    # [NEW] 传递多模态特征到客户端
    clients = create_fedmem_clients(
        user_sequences=user_sequences,
        global_model=global_model,
        item_visual_feats=item_visual_feats,  # [NEW]
        item_text_feats=item_text_feats,      # [NEW]
        args=args
    )

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
        partial_aggregation_warmup_rounds=args.partial_aggregation_warmup_rounds
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
    print(f"\n最佳验证轮次: {best_metrics.get('round', -1) + 1}")
    print("最佳验证指标:")
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
