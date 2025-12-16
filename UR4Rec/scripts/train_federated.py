"""
Federated Learning Training Script for UR4Rec

联邦学习训练脚本：
- 每个user作为一个client
- 使用FedAvg聚合算法
- Leave-one-out数据划分
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import numpy as np
import argparse
import yaml
from pathlib import Path
from typing import Dict, List
import json
from tqdm import tqdm

from UR4Rec.models.sasrec import SASRec
from UR4Rec.models.federated_client import FederatedClient
from UR4Rec.models.federated_server import FederatedServer


def load_movielens_data(data_dir: str) -> Dict[int, List[int]]:
    """
    加载MovieLens数据集

    Returns:
        user_sequences: {user_id: [item_id1, item_id2, ...]}
    """
    data_path = Path(data_dir) / "M_ML-100K"

    # 读取评分数据
    ratings_file = data_path / "ratings.dat"

    if not ratings_file.exists():
        raise FileNotFoundError(f"Data file not found: {ratings_file}")

    # 读取数据: user_id::item_id::rating::timestamp
    user_sequences = {}

    with open(ratings_file, 'r') as f:
        for line in f:
            parts = line.strip().split('::')
            user_id = int(parts[0])
            item_id = int(parts[1])
            rating = float(parts[2])
            timestamp = int(parts[3])

            # 只保留评分>=4的正样本
            if rating >= 4.0:
                if user_id not in user_sequences:
                    user_sequences[user_id] = []
                user_sequences[user_id].append((item_id, timestamp))

    # 按时间排序
    for user_id in user_sequences:
        user_sequences[user_id].sort(key=lambda x: x[1])
        user_sequences[user_id] = [item_id for item_id, _ in user_sequences[user_id]]

    # 过滤掉序列太短的用户（至少需要3个交互：train/val/test）
    user_sequences = {
        user_id: seq for user_id, seq in user_sequences.items()
        if len(seq) >= 3
    }

    print(f"Loaded {len(user_sequences)} users with valid sequences")
    print(f"Avg sequence length: {np.mean([len(seq) for seq in user_sequences.values()]):.2f}")

    return user_sequences


def create_federated_clients(
    user_sequences: Dict[int, List[int]],
    global_model: SASRec,
    config: Dict,
    device: str
) -> List[FederatedClient]:
    """
    创建联邦学习客户端

    Args:
        user_sequences: 用户交互序列
        global_model: 全局模型
        config: 配置
        device: 设备

    Returns:
        clients: 客户端列表
    """
    clients = []

    for user_id, sequence in tqdm(user_sequences.items(), desc="Creating clients"):
        client = FederatedClient(
            client_id=user_id,
            model=global_model,
            user_sequence=sequence,
            device=device,
            learning_rate=config.get('federated_lr', 1e-3),
            weight_decay=config.get('weight_decay', 1e-5),
            local_epochs=config.get('local_epochs', 1),
            batch_size=config.get('batch_size', 32),
            max_seq_len=config.get('max_seq_len', 50),
            num_negatives=config.get('num_negatives', 100),
            num_items=config.get('num_items', 1682)
        )
        clients.append(client)

    print(f"Created {len(clients)} federated clients")

    return clients


def main():
    parser = argparse.ArgumentParser(description="Federated Learning Training for UR4Rec")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='outputs/federated', help='Output directory')
    parser.add_argument('--num_rounds', type=int, default=50, help='Number of federated rounds')
    parser.add_argument('--local_epochs', type=int, default=1, help='Local training epochs')
    parser.add_argument('--client_fraction', type=float, default=0.1, help='Fraction of clients per round')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 覆盖配置参数
    config['local_epochs'] = args.local_epochs

    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 加载数据
    print("\n" + "="*60)
    print("Loading MovieLens-100K Data")
    print("="*60)
    user_sequences = load_movielens_data(args.data_dir)

    # 创建全局模型
    print("\n" + "="*60)
    print("Creating Global Model")
    print("="*60)

    global_model = SASRec(
        num_items=config['num_items'],
        hidden_dim=config['sasrec_hidden_dim'],
        num_blocks=config['sasrec_num_blocks'],
        num_heads=config['sasrec_num_heads'],
        max_seq_len=config['max_seq_len'],
        dropout=config['sasrec_dropout']
    )

    num_params = sum(p.numel() for p in global_model.parameters())
    print(f"Model Parameters: {num_params:,}")

    # 创建联邦客户端
    print("\n" + "="*60)
    print("Creating Federated Clients")
    print("="*60)
    clients = create_federated_clients(user_sequences, global_model, config, device)

    # 创建联邦服务器
    print("\n" + "="*60)
    print("Creating Federated Server")
    print("="*60)
    server = FederatedServer(
        global_model=global_model,
        clients=clients,
        device=device,
        aggregation_method='fedavg',
        client_fraction=args.client_fraction,
        num_rounds=args.num_rounds,
        local_epochs=args.local_epochs,
        patience=args.patience
    )

    # 开始联邦训练
    print("\n" + "="*60)
    print("Starting Federated Training")
    print("="*60)
    print(f"Total Rounds: {args.num_rounds}")
    print(f"Clients per Round: {int(len(clients) * args.client_fraction)}")
    print(f"Local Epochs: {args.local_epochs}")
    print(f"Patience: {args.patience}")
    print("="*60 + "\n")

    train_history = server.train(verbose=True)

    # 保存模型和结果
    model_path = output_dir / "federated_model.pt"
    server.save_model(str(model_path))

    # 保存训练历史
    history_path = output_dir / "train_history.json"
    with open(history_path, 'w') as f:
        json.dump(train_history, f, indent=2)

    print(f"\nModel saved to: {model_path}")
    print(f"Training history saved to: {history_path}")

    # 打印最终结果
    print("\n" + "="*60)
    print("Final Results")
    print("="*60)

    best_metrics = server.get_best_metrics()
    print("\nBest Validation Metrics:")
    for key, value in best_metrics.items():
        if key != 'round':
            print(f"  {key}: {value:.4f}")

    test_metrics = train_history['test_metrics']
    print("\nTest Set Metrics:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.4f}")

    # 保存最终结果
    results = {
        'best_val_metrics': best_metrics,
        'test_metrics': test_metrics,
        'config': config,
        'num_clients': len(clients),
        'num_rounds': args.num_rounds
    }

    results_path = output_dir / "final_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nFinal results saved to: {results_path}")
    print("\n" + "="*60)
    print("Federated Training Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
