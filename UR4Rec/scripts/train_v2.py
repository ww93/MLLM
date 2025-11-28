"""
UR4Rec V2 训练脚本

使用联合训练器进行多阶段训练：
1. 预训练 SASRec
2. 预训练检索器
3. 联合微调
4. 端到端优化
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from typing import Dict, List, Tuple
import random

from models.ur4rec_v2 import UR4RecV2
from models.multimodal_retriever import MultiModalPreferenceRetriever
from models.joint_trainer import JointTrainer


class UR4RecDataset(Dataset):
    """
    UR4Rec 数据集
    """

    def __init__(
        self,
        sequences: Dict,
        num_items: int,
        max_seq_len: int = 50,
        num_negatives: int = 5,
        num_candidates: int = 100,
        mode: str = 'train'
    ):
        """
        Args:
            sequences: 用户序列字典 {user_id: [item_ids]}
            num_items: 物品总数
            max_seq_len: 最大序列长度
            num_negatives: 负样本数量（训练时）
            num_candidates: 候选物品数量（评估时）
            mode: 'train' | 'val' | 'test'
        """
        self.sequences = sequences
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.num_negatives = num_negatives
        self.num_candidates = num_candidates
        self.mode = mode

        # 用户列表
        self.user_ids = list(sequences.keys())

        # 物品集合（用于负采样）
        self.all_items = set(range(1, num_items + 1))

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        sequence = self.sequences[user_id]

        if self.mode == 'train':
            # 训练模式：取前 n-1 作为输入，第 n 个作为目标
            if len(sequence) < 2:
                # 序列太短，填充
                input_seq = [0] * (self.max_seq_len - 1) + [sequence[0]]
                target_item = sequence[0]
            else:
                input_seq = sequence[:-1]
                target_item = sequence[-1]

            # 截断/填充输入序列
            if len(input_seq) > self.max_seq_len:
                input_seq = input_seq[-self.max_seq_len:]
            else:
                padding_len = self.max_seq_len - len(input_seq)
                input_seq = [0] * padding_len + input_seq

            # 负采样
            user_items = set(sequence)
            negative_items = []
            while len(negative_items) < self.num_negatives:
                neg_item = random.randint(1, self.num_items)
                if neg_item not in user_items:
                    negative_items.append(neg_item)

            # 填充掩码
            seq_padding_mask = [1 if item != 0 else 0 for item in input_seq]

            return {
                'user_ids': user_id,
                'input_seq': torch.LongTensor(input_seq),
                'target_items': torch.LongTensor([target_item]),
                'negative_items': torch.LongTensor(negative_items),
                'seq_padding_mask': torch.BoolTensor(seq_padding_mask)
            }

        else:
            # 评估模式：取前 n-1 作为输入，第 n 个作为目标
            if len(sequence) < 2:
                input_seq = [0] * (self.max_seq_len - 1) + [sequence[0]]
                target_item = sequence[0]
            else:
                input_seq = sequence[:-1]
                target_item = sequence[-1]

            # 截断/填充
            if len(input_seq) > self.max_seq_len:
                input_seq = input_seq[-self.max_seq_len:]
            else:
                padding_len = self.max_seq_len - len(input_seq)
                input_seq = [0] * padding_len + input_seq

            # 候选物品：目标物品 + 随机负样本
            user_items = set(sequence)
            candidate_items = [target_item]
            while len(candidate_items) < self.num_candidates:
                cand_item = random.randint(1, self.num_items)
                if cand_item not in candidate_items:
                    candidate_items.append(cand_item)

            # 填充掩码
            seq_padding_mask = [1 if item != 0 else 0 for item in input_seq]

            return {
                'user_ids': user_id,
                'input_seq': torch.LongTensor(input_seq),
                'target_items': torch.LongTensor([target_item]),
                'candidate_items': torch.LongTensor(candidate_items),
                'seq_padding_mask': torch.BoolTensor(seq_padding_mask)
            }


def collate_fn(batch):
    """批处理函数"""
    user_ids = [item['user_ids'] for item in batch]
    input_seqs = torch.stack([item['input_seq'] for item in batch])
    target_items = torch.cat([item['target_items'] for item in batch])
    seq_padding_masks = torch.stack([item['seq_padding_mask'] for item in batch])

    result = {
        'user_ids': user_ids,
        'input_seq': input_seqs,
        'target_items': target_items,
        'seq_padding_mask': seq_padding_masks
    }

    # 训练模式：负样本
    if 'negative_items' in batch[0]:
        negative_items = torch.stack([item['negative_items'] for item in batch])
        result['negative_items'] = negative_items

    # 评估模式：候选物品
    if 'candidate_items' in batch[0]:
        candidate_items = torch.stack([item['candidate_items'] for item in batch])
        result['candidate_items'] = candidate_items

    return result


def load_data(data_dir: str, max_seq_len: int, num_items: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    加载训练、验证、测试数据

    Args:
        data_dir: 数据目录
        max_seq_len: 最大序列长度
        num_items: 物品总数

    Returns:
        train_loader, val_loader, test_loader
    """
    data_path = Path(data_dir)

    # 加载序列
    train_seq = np.load(data_path / 'train_sequences.npy', allow_pickle=True).item()
    val_seq = np.load(data_path / 'val_sequences.npy', allow_pickle=True).item()
    test_seq = np.load(data_path / 'test_sequences.npy', allow_pickle=True).item()

    print(f"训练用户: {len(train_seq)}")
    print(f"验证用户: {len(val_seq)}")
    print(f"测试用户: {len(test_seq)}")

    # 创建数据集
    train_dataset = UR4RecDataset(
        train_seq, num_items, max_seq_len,
        num_negatives=5, mode='train'
    )

    val_dataset = UR4RecDataset(
        val_seq, num_items, max_seq_len,
        num_candidates=100, mode='val'
    )

    test_dataset = UR4RecDataset(
        test_seq, num_items, max_seq_len,
        num_candidates=100, mode='test'
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )

    return train_loader, val_loader, test_loader


def train_stage(
    trainer: JointTrainer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    stage_name: str,
    num_epochs: int,
    save_dir: Path,
    patience: int = 5
):
    """
    训练一个阶段

    Args:
        trainer: 联合训练器
        train_loader: 训练数据
        val_loader: 验证数据
        stage_name: 阶段名称
        num_epochs: 训练轮数
        save_dir: 保存目录
        patience: 早停耐心值
    """
    print(f"\n{'='*60}")
    print(f"开始训练阶段: {stage_name}")
    print(f"{'='*60}")

    # 设置训练阶段
    trainer.set_training_stage(stage_name)

    best_metric = 0.0
    patience_counter = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # 训练
        train_metrics = trainer.train_epoch(train_loader, epoch)

        print(f"训练指标:")
        for key, value in train_metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")

        # 验证
        val_metrics = trainer.evaluate(val_loader, k_list=[5, 10, 20])

        print(f"验证指标:")
        for key, value in val_metrics.items():
            print(f"  {key}: {value:.4f}")

        # 保存最佳模型
        current_metric = val_metrics.get('ndcg@10', val_metrics.get('hit@10', 0))

        if current_metric > best_metric:
            best_metric = current_metric
            patience_counter = 0

            # 保存检查点
            checkpoint_path = save_dir / f'{stage_name}_best.pt'
            trainer.save_checkpoint(
                save_path=str(checkpoint_path),
                epoch=epoch,
                metrics=val_metrics
            )

            print(f"✓ 保存最佳模型 (指标: {current_metric:.4f})")

        else:
            patience_counter += 1
            print(f"✗ 指标未提升 (耐心: {patience_counter}/{patience})")

            if patience_counter >= patience:
                print(f"早停！最佳指标: {best_metric:.4f}")
                break

    # 加载最佳模型
    best_checkpoint = save_dir / f'{stage_name}_best.pt'
    if best_checkpoint.exists():
        trainer.load_checkpoint(str(best_checkpoint))
        print(f"已加载最佳模型: {best_checkpoint}")


def main():
    parser = argparse.ArgumentParser(description='UR4Rec V2 训练脚本')

    parser.add_argument('--config', type=str, required=True,
                        help='配置文件路径')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='数据目录')
    parser.add_argument('--llm_data_dir', type=str, required=True,
                        help='LLM 生成数据目录')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='输出目录')

    parser.add_argument('--use_multimodal', action='store_true',
                        help='使用多模态检索器')

    parser.add_argument('--stages', nargs='+',
                        default=['pretrain_sasrec', 'pretrain_retriever', 'joint_finetune', 'end_to_end'],
                        help='训练阶段列表')

    parser.add_argument('--epochs_per_stage', type=int, default=10,
                        help='每阶段训练轮数')
    parser.add_argument('--patience', type=int, default=5,
                        help='早停耐心值')

    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='设备')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')

    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print("UR4Rec V2 训练")
    print("=" * 60)
    print(f"配置文件: {args.config}")
    print(f"数据目录: {args.data_dir}")
    print(f"LLM 数据目录: {args.llm_data_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"多模态: {args.use_multimodal}")
    print(f"设备: {args.device}")
    print(f"训练阶段: {args.stages}")

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取物品数量
    item_map_path = Path(args.data_dir) / 'item_map.json'
    with open(item_map_path, 'r') as f:
        item_map = json.load(f)
    num_items = len(item_map)

    print(f"\n物品总数: {num_items}")

    # 加载数据
    print("\n加载数据...")
    max_seq_len = config.get('max_seq_len', 50)
    train_loader, val_loader, test_loader = load_data(
        args.data_dir, max_seq_len, num_items
    )

    # 创建模型
    print("\n创建模型...")
    if args.use_multimodal:
        # TODO: 实现多模态模型初始化
        raise NotImplementedError("多模态模型尚未完全集成")
    else:
        model = UR4RecV2(
            num_items=num_items,
            sasrec_hidden_dim=config.get('sasrec_hidden_dim', 256),
            sasrec_num_blocks=config.get('sasrec_num_blocks', 2),
            sasrec_num_heads=config.get('sasrec_num_heads', 4),
            text_model_name=config.get('text_model_name', 'all-MiniLM-L6-v2'),
            text_embedding_dim=config.get('text_embedding_dim', 384),
            retriever_output_dim=config.get('retriever_output_dim', 256),
            fusion_method=config.get('fusion_method', 'weighted'),
            max_seq_len=max_seq_len,
            device=args.device
        )

    print(f"模型参数: {sum(p.numel() for p in model.parameters()):,}")

    # 加载 LLM 生成的数据
    print("\n加载 LLM 生成的数据...")
    user_pref_path = Path(args.llm_data_dir) / 'user_preferences.json'
    item_desc_path = Path(args.llm_data_dir) / 'item_descriptions.json'

    model.load_llm_generated_data(
        user_preferences_path=str(user_pref_path),
        item_descriptions_path=str(item_desc_path)
    )

    # 创建训练器
    print("\n创建训练器...")
    trainer = JointTrainer(
        model=model,
        device=args.device,
        sasrec_lr=config.get('sasrec_lr', 1e-3),
        retriever_lr=config.get('retriever_lr', 1e-4),
        use_multimodal=args.use_multimodal,
        use_uncertainty_weighting=config.get('use_uncertainty_weighting', True)
    )

    # 多阶段训练
    for stage in args.stages:
        train_stage(
            trainer=trainer,
            train_loader=train_loader,
            val_loader=val_loader,
            stage_name=stage,
            num_epochs=args.epochs_per_stage,
            save_dir=output_dir,
            patience=args.patience
        )

    # 最终测试
    print("\n" + "=" * 60)
    print("最终测试")
    print("=" * 60)

    test_metrics = trainer.evaluate(test_loader, k_list=[5, 10, 20])

    print("\n测试指标:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.4f}")

    # 保存最终模型
    final_model_path = output_dir / 'final_model.pt'
    torch.save(model.state_dict(), final_model_path)
    print(f"\n最终模型已保存至: {final_model_path}")

    # 保存配置和指标
    results = {
        'config': config,
        'args': vars(args),
        'test_metrics': test_metrics,
        'train_history': trainer.train_history
    }

    results_path = output_dir / 'results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"结果已保存至: {results_path}")

    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
