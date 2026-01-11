"""
Train Federated SASRec: Pure SASRec in Federated Learning Setting

训练联邦学习版本的SASRec
- 用于对比集中式vs联邦式的性能差异
- 不包含memory、MoE等复杂组件
- 使用与FedMem相同的联邦学习框架(FedAvg)
"""

import os
import sys
import argparse
import json
import time
from typing import Dict, List
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.federated_client_sasrec import FederatedSASRecClient
from models.fedmem_server import FedMemServer


def load_user_sequences(data_path: str) -> Dict[int, List[int]]:
    """
    加载用户序列数据

    Args:
        data_path: 数据文件路径

    Returns:
        user_sequences: {user_id: [item_id, ...]}
    """
    user_sequences = {}

    with open(data_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue

            user_id = int(parts[0])
            item_ids = [int(x) for x in parts[1:]]

            user_sequences[user_id] = item_ids

    return user_sequences


def create_federated_sasrec_clients(
    user_sequences: Dict[int, List[int]],
    num_items: int,
    args
) -> List[FederatedSASRecClient]:
    """
    创建联邦SASRec客户端

    Args:
        user_sequences: 用户序列
        num_items: 物品总数
        args: 命令行参数

    Returns:
        clients: 客户端列表
    """
    print("\n创建联邦SASRec客户端...")
    print(f"  用户数量: {len(user_sequences)}")
    print(f"  物品数量: {num_items}")

    clients = []
    for user_id, sequence in user_sequences.items():
        if len(sequence) < 3:
            continue

        client = FederatedSASRecClient(
            client_id=user_id,
            user_sequence=sequence,
            num_items=num_items,
            device=args.device,
            # 模型参数
            hidden_dim=args.hidden_dim,
            num_blocks=args.num_blocks,
            num_heads=args.num_heads,
            dropout=args.dropout,
            max_seq_len=args.max_seq_len,
            # 训练参数
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            local_epochs=args.local_epochs,
            batch_size=args.batch_size,
            # 负采样
            num_negatives=args.num_negatives,
            # 评估参数
            use_negative_sampling=args.use_negative_sampling,
            num_negatives_eval=args.num_negatives_eval
        )
        clients.append(client)

    print(f"✓ 创建了 {len(clients)} 个联邦SASRec客户端")
    print(f"  每个客户端:")
    print(f"    - 隐藏维度: {args.hidden_dim}")
    print(f"    - Transformer块数: {args.num_blocks}")
    print(f"    - 注意力头数: {args.num_heads}")
    print(f"    - 最大序列长度: {args.max_seq_len}")
    print(f"    - 负采样评估: {'启用' if args.use_negative_sampling else '禁用'}")

    return clients


def main():
    parser = argparse.ArgumentParser(description="Train Federated SASRec")

    # 数据参数
    parser.add_argument("--data_dir", type=str, default="UR4Rec/data",
                        help="数据目录")
    parser.add_argument("--data_file", type=str, default="ml1m_ratings_processed.dat",
                        help="数据文件名")
    parser.add_argument("--save_dir", type=str, default="UR4Rec/checkpoints/federated_sasrec",
                        help="模型保存目录")

    # 联邦学习参数
    parser.add_argument("--num_rounds", type=int, default=30,
                        help="联邦学习轮数")
    parser.add_argument("--local_epochs", type=int, default=1,
                        help="本地训练轮数")
    parser.add_argument("--client_fraction", type=float, default=0.1,
                        help="每轮参与训练的客户端比例")
    parser.add_argument("--patience", type=int, default=5,
                        help="早停patience")

    # 模型参数
    parser.add_argument("--hidden_dim", type=int, default=128,
                        help="隐藏维度")
    parser.add_argument("--num_blocks", type=int, default=2,
                        help="Transformer块数量")
    parser.add_argument("--num_heads", type=int, default=4,
                        help="注意力头数量")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout率")
    parser.add_argument("--max_seq_len", type=int, default=50,
                        help="最大序列长度")

    # 训练参数
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="权重衰减")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="批大小")
    parser.add_argument("--num_negatives", type=int, default=50,
                        help="训练时负样本数量")

    # 评估参数
    parser.add_argument("--use_negative_sampling", default=True,
                        help="使用1:100负采样评估（对齐SASRec论文）")
    parser.add_argument("--num_negatives_eval", type=int, default=100,
                        help="评估时的负样本数量（默认100）")

    # 其他参数
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="计算设备")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--verbose", action="store_true",
                        help="打印详细信息")

    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("\n" + "=" * 60)
    print("联邦SASRec训练")
    print("=" * 60)
    print(f"\n配置:")
    print(f"  数据文件: {args.data_file}")
    print(f"  联邦学习轮数: {args.num_rounds}")
    print(f"  客户端采样比例: {args.client_fraction}")
    print(f"  本地训练轮数: {args.local_epochs}")
    print(f"  隐藏维度: {args.hidden_dim}")
    print(f"  Transformer块数: {args.num_blocks}")
    print(f"  注意力头数: {args.num_heads}")
    print(f"  负采样评估: {'启用' if args.use_negative_sampling else '禁用'}")
    print(f"  设备: {args.device}")

    # ============================================
    # 1. 加载数据
    # ============================================
    print("\n" + "=" * 60)
    print("1. 加载数据")
    print("=" * 60)

    data_path = os.path.join(args.data_dir, args.data_file)
    user_sequences = load_user_sequences(data_path)

    print(f"✓ 加载了 {len(user_sequences)} 个用户序列")

    # 计算物品总数
    all_items = set()
    for seq in user_sequences.values():
        all_items.update(seq)
    num_items = max(all_items)

    print(f"✓ 物品总数: {num_items}")
    print(f"✓ 平均序列长度: {np.mean([len(seq) for seq in user_sequences.values()]):.1f}")

    # ============================================
    # 2. 创建客户端
    # ============================================
    print("\n" + "=" * 60)
    print("2. 创建联邦SASRec客户端")
    print("=" * 60)

    clients = create_federated_sasrec_clients(user_sequences, num_items, args)

    # ============================================
    # 3. 创建全局模型
    # ============================================
    print("\n" + "=" * 60)
    print("3. 创建全局SASRec模型")
    print("=" * 60)

    from models.sasrec_fixed import SASRecFixed

    global_model = SASRecFixed(
        num_items=num_items,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len
    ).to(args.device)

    print(f"✓ 创建全局SASRec模型")
    print(f"  参数量: {sum(p.numel() for p in global_model.parameters()):,}")

    # ============================================
    # 4. 创建联邦学习服务器并训练
    # ============================================
    print("\n" + "=" * 60)
    print("4. 开始联邦学习训练")
    print("=" * 60)

    # 复用FedMemServer的联邦学习逻辑
    server = FedMemServer(
        global_model=global_model,
        clients=clients,
        device=args.device,
        num_rounds=args.num_rounds,
        client_fraction=args.client_fraction,
        local_epochs=args.local_epochs,
        patience=args.patience,
        # FedMem特有参数设为False/0（不使用）
        enable_prototype_aggregation=False,
        num_memory_prototypes=0
    )

    # 训练
    train_history = server.train(user_sequences=user_sequences, verbose=args.verbose or True)

    # ============================================
    # 5. 保存模型和结果
    # ============================================
    print("\n" + "=" * 60)
    print("5. 保存模型和训练历史")
    print("=" * 60)

    os.makedirs(args.save_dir, exist_ok=True)

    # 保存模型
    model_path = os.path.join(args.save_dir, 'federated_sasrec_model.pt')
    torch.save(server.global_model.state_dict(), model_path)
    print(f"✓ 模型已保存到: {model_path}")

    # 保存训练历史
    history_path = os.path.join(args.save_dir, 'train_history.json')
    with open(history_path, 'w') as f:
        json.dump(train_history, f, indent=2)
    print(f"✓ 训练历史已保存到: {history_path}")

    # ============================================
    # 6. 打印最终结果
    # ============================================
    print("\n" + "=" * 60)
    print("6. 训练结果总结")
    print("=" * 60)

    best_metrics = server.get_best_val_metrics()
    test_metrics = train_history['test_metrics']

    print(f"\n最佳验证集结果 (Round {best_metrics['round']}):")
    for key, value in best_metrics.items():
        if key != 'round' and key != 'train_loss':
            print(f"  {key}: {value:.4f}")

    print(f"\n最终测试集结果:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.4f}")

    # 保存配置
    config = vars(args)
    config_path = os.path.join(args.save_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\n✓ 配置已保存到: {config_path}")

    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)

    # 对比说明
    print("\n" + "=" * 60)
    print("实验说明")
    print("=" * 60)
    print("\n本实验训练了联邦学习版本的纯SASRec模型。")
    print("\n对比实验设计:")
    print("  1. 集中式SASRec: 所有用户数据在中心服务器训练")
    print("  2. 联邦式SASRec: 每个用户作为一个客户端，使用FedAvg聚合")
    print("  3. 联邦式FedMem: 在联邦SASRec基础上增加memory和MoE")
    print("\n通过对比1和2，可以判断联邦学习架构本身是否导致性能下降。")
    print("通过对比2和3，可以判断memory和MoE组件的影响。")


if __name__ == "__main__":
    main()
