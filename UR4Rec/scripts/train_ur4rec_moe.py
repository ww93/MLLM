"""
UR4Rec V2 with MoE Training Script

Combines:
1. Multi-stage training from train_v2
2. MoE-enhanced retriever with user memory
3. LLM-generated preferences and descriptions
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
from datetime import datetime

from models.ur4rec_v2_moe import UR4RecV2MoE
from models.retriever_moe_memory import MemoryConfig, UpdateTrigger
from models.joint_trainer import JointTrainer


class UR4RecDataset(Dataset):
    """
    UR4Rec Dataset for training
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
            sequences: User sequences {user_id: [item_ids]}
            num_items: Total number of items
            max_seq_len: Maximum sequence length
            num_negatives: Number of negative samples (training)
            num_candidates: Number of candidate items (evaluation)
            mode: 'train' | 'val' | 'test'
        """
        self.sequences = sequences
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.num_negatives = num_negatives
        self.num_candidates = num_candidates
        self.mode = mode

        self.user_ids = list(sequences.keys())
        self.all_items = set(range(1, num_items + 1))

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        sequence = self.sequences[user_id]

        if self.mode == 'train':
            # Training: use first n-1 as input, nth as target
            if len(sequence) < 2:
                input_seq = [0] * (self.max_seq_len - 1) + [sequence[0]]
                target_item = sequence[0]
            else:
                input_seq = sequence[:-1]
                target_item = sequence[-1]

            # Truncate/pad input sequence
            if len(input_seq) > self.max_seq_len:
                input_seq = input_seq[-self.max_seq_len:]
            else:
                padding_len = self.max_seq_len - len(input_seq)
                input_seq = [0] * padding_len + input_seq

            # Negative sampling
            user_items = set(sequence)
            negative_items = []
            while len(negative_items) < self.num_negatives:
                neg_item = random.randint(1, self.num_items)
                if neg_item not in user_items:
                    negative_items.append(neg_item)

            seq_padding_mask = [1 if item != 0 else 0 for item in input_seq]

            return {
                'user_ids': user_id,
                'input_seq': torch.LongTensor(input_seq),
                'target_items': torch.LongTensor([target_item]),
                'negative_items': torch.LongTensor(negative_items),
                'seq_padding_mask': torch.BoolTensor(seq_padding_mask)
            }

        else:
            # Evaluation: same structure
            if len(sequence) < 2:
                input_seq = [0] * (self.max_seq_len - 1) + [sequence[0]]
                target_item = sequence[0]
            else:
                input_seq = sequence[:-1]
                target_item = sequence[-1]

            if len(input_seq) > self.max_seq_len:
                input_seq = input_seq[-self.max_seq_len:]
            else:
                padding_len = self.max_seq_len - len(input_seq)
                input_seq = [0] * padding_len + input_seq

            # Candidate items: target + random negatives
            user_items = set(sequence)
            candidate_items = [target_item]
            while len(candidate_items) < self.num_candidates:
                cand_item = random.randint(1, self.num_items)
                if cand_item not in candidate_items:
                    candidate_items.append(cand_item)

            seq_padding_mask = [1 if item != 0 else 0 for item in input_seq]

            return {
                'user_ids': user_id,
                'input_seq': torch.LongTensor(input_seq),
                'target_items': torch.LongTensor([target_item]),
                'candidate_items': torch.LongTensor(candidate_items),
                'seq_padding_mask': torch.BoolTensor(seq_padding_mask)
            }


def collate_fn(batch):
    """Batch collation function"""
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

    if 'negative_items' in batch[0]:
        negative_items = torch.stack([item['negative_items'] for item in batch])
        result['negative_items'] = negative_items

    if 'candidate_items' in batch[0]:
        candidate_items = torch.stack([item['candidate_items'] for item in batch])
        result['candidate_items'] = candidate_items

    return result


def load_data(data_dir: str, max_seq_len: int, num_items: int, config: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load training, validation, test data

    Args:
        data_dir: Data directory
        max_seq_len: Maximum sequence length
        num_items: Total number of items
        config: Configuration dictionary

    Returns:
        train_loader, val_loader, test_loader
    """
    data_path = Path(data_dir)

    # Load sequences
    train_seq = np.load(data_path / 'train_sequences.npy', allow_pickle=True).item()
    val_seq = np.load(data_path / 'val_sequences.npy', allow_pickle=True).item()
    test_seq = np.load(data_path / 'test_sequences.npy', allow_pickle=True).item()

    print(f"训练用户: {len(train_seq)}")
    print(f"验证用户: {len(val_seq)}")
    print(f"测试用户: {len(test_seq)}")

    # Create datasets
    train_dataset = UR4RecDataset(
        train_seq, num_items, max_seq_len,
        num_negatives=config.get('num_negatives', 5),  # 从config读取
        mode='train'
    )

    val_dataset = UR4RecDataset(
        val_seq, num_items, max_seq_len,
        num_candidates=100, mode='val'
    )

    test_dataset = UR4RecDataset(
        test_seq, num_items, max_seq_len,
        num_candidates=100, mode='test'
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 32),  # 从config读取
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.get('num_workers', 4)  # 从config读取
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
    Train one stage

    Args:
        trainer: Joint trainer
        train_loader: Training data
        val_loader: Validation data
        stage_name: Stage name
        num_epochs: Number of epochs
        save_dir: Save directory
        patience: Early stopping patience
    """
    print(f"\n{'='*60}")
    print(f"Training Stage: {stage_name}")
    print(f"{'='*60}")

    trainer.set_training_stage(stage_name)

    best_metric = 0.0
    patience_counter = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Train
        train_metrics = trainer.train_epoch(train_loader, epoch)

        print(f"Training metrics:")
        for key, value in train_metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")

        # Validate
        val_metrics = trainer.evaluate(val_loader, k_list=[5, 10, 20])

        print(f"Validation metrics:")
        for key, value in val_metrics.items():
            print(f"  {key}: {value:.4f}")

        # Save best model
        current_metric = val_metrics.get('ndcg@10', val_metrics.get('hit@10', 0))

        if current_metric > best_metric:
            best_metric = current_metric
            patience_counter = 0

            checkpoint_path = save_dir / f'{stage_name}_best.pt'
            trainer.save_checkpoint(
                save_path=str(checkpoint_path),
                epoch=epoch,
                metrics=val_metrics
            )

            # Save memories
            if hasattr(trainer.model, 'save_memories'):
                memory_path = save_dir / f'{stage_name}_memories.pt'
                trainer.model.save_memories(str(memory_path))

            print(f"✓ Saved best model (metric: {current_metric:.4f})")

        else:
            patience_counter += 1
            print(f"✗ No improvement (patience: {patience_counter}/{patience})")

            if patience_counter >= patience:
                print(f"Early stopping! Best metric: {best_metric:.4f}")
                break

        # Show memory stats
        if hasattr(trainer.model, 'get_memory_stats'):
            memory_stats = trainer.model.get_memory_stats()
            print(f"Memory stats: {memory_stats}")

    # Load best model
    best_checkpoint = save_dir / f'{stage_name}_best.pt'
    if best_checkpoint.exists():
        trainer.load_checkpoint(str(best_checkpoint))
        print(f"Loaded best model: {best_checkpoint}")

        # Load memories
        if hasattr(trainer.model, 'load_memories'):
            memory_path = save_dir / f'{stage_name}_memories.pt'
            if memory_path.exists():
                trainer.model.load_memories(str(memory_path))
                print(f"Loaded memories: {memory_path}")


def main():
    parser = argparse.ArgumentParser(description='UR4Rec V2 with MoE Training')

    parser.add_argument('--config', type=str, required=True,
                        help='Configuration file path')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Data directory')
    parser.add_argument('--llm_data_dir', type=str, required=True,
                        help='LLM generated data directory')
    parser.add_argument('--output_dir', type=str, default='outputs/ur4rec_moe',
                        help='Output directory')

    parser.add_argument('--stages', nargs='+',
                        default=['pretrain_sasrec', 'pretrain_retriever', 'joint_finetune', 'end_to_end'],
                        help='Training stages')

    parser.add_argument('--epochs_per_stage', type=int, default=10,
                        help='Epochs per stage')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience')

    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Load config
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print("UR4Rec V2 with MoE Training")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Data directory: {args.data_dir}")
    print(f"LLM data directory: {args.llm_data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")
    print(f"Training stages: {args.stages}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get number of items
    item_map_path = Path(args.data_dir) / 'item_map.json'
    with open(item_map_path, 'r') as f:
        item_map = json.load(f)
    # 使用最大item ID而不是item_map长度，确保所有测试集物品都能被索引
    item_ids = [int(float(k)) for k in item_map.keys()]
    num_items = max(item_ids) if item_ids else len(item_map)

    print(f"\nTotal items: {num_items} (max_id: {num_items}, count: {len(item_map)})")

    # Load data
    print("\nLoading data...")
    max_seq_len = config.get('max_seq_len', 50)
    train_loader, val_loader, test_loader = load_data(
        args.data_dir, max_seq_len, num_items, config
    )

    # Create memory config
    memory_config = MemoryConfig(
        memory_dim=config.get('retriever_output_dim', 256),
        max_memory_size=config.get('max_memory_size', 10),
        update_trigger=UpdateTrigger[config.get('update_trigger', 'INTERACTION_COUNT')],
        interaction_threshold=config.get('interaction_threshold', 10),
        drift_threshold=config.get('drift_threshold', 0.3),
        decay_factor=config.get('decay_factor', 0.95),
        enable_persistence=True
    )

    # Create model
    print("\nCreating model...")
    model = UR4RecV2MoE(
        num_items=num_items,
        sasrec_hidden_dim=config.get('sasrec_hidden_dim', 256),
        sasrec_num_blocks=config.get('sasrec_num_blocks', 2),
        sasrec_num_heads=config.get('sasrec_num_heads', 4),
        sasrec_dropout=config.get('sasrec_dropout', 0.1),  # 添加dropout参数
        text_model_name=config.get('text_model_name', 'all-MiniLM-L6-v2'),
        text_embedding_dim=config.get('text_embedding_dim', 384),
        retriever_output_dim=config.get('retriever_output_dim', 256),
        moe_num_heads=config.get('moe_num_heads', 8),
        moe_dropout=config.get('moe_dropout', 0.1),  # 添加moe_dropout参数
        moe_num_proxies=config.get('moe_num_proxies', 4),
        memory_config=memory_config,
        fusion_method=config.get('fusion_method', 'weighted'),
        sasrec_weight=config.get('sasrec_weight', 0.5),
        retriever_weight=config.get('retriever_weight', 0.5),
        max_seq_len=max_seq_len,
        device=args.device
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load LLM generated data
    print("\nLoading LLM generated data...")
    user_pref_path = Path(args.llm_data_dir) / 'user_preferences.json'
    item_desc_path = Path(args.llm_data_dir) / 'item_descriptions.json'

    model.load_llm_generated_data(
        user_preferences_path=str(user_pref_path),
        item_descriptions_path=str(item_desc_path)
    )

    # Create trainer
    print("\nCreating trainer...")
    trainer = JointTrainer(
        model=model,
        device=args.device,
        sasrec_lr=config.get('sasrec_lr', 1e-3),
        retriever_lr=config.get('retriever_lr', 1e-4),
        use_uncertainty_weighting=config.get('use_uncertainty_weighting', True),
        use_adaptive_alternating=config.get('use_adaptive_alternating', True)
    )

    # Multi-stage training
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

    # Final testing
    print("\n" + "=" * 60)
    print("Final Testing")
    print("=" * 60)

    test_metrics = trainer.evaluate(test_loader, k_list=[5, 10, 20])

    print("\nTest metrics:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.4f}")

    # Save final model
    final_model_path = output_dir / 'final_model.pt'
    torch.save(model.state_dict(), final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")

    # Save memories
    if hasattr(model, 'save_memories'):
        final_memory_path = output_dir / 'final_memories.pt'
        model.save_memories(str(final_memory_path))
        print(f"Final memories saved to: {final_memory_path}")

    # Save config and results
    results = {
        'config': config,
        'args': vars(args),
        'test_metrics': test_metrics,
        'train_history': trainer.train_history,
        'memory_config': {
            'memory_dim': memory_config.memory_dim,
            'max_memory_size': memory_config.max_memory_size,
            'update_trigger': memory_config.update_trigger.value,
            'interaction_threshold': memory_config.interaction_threshold,
        },
        'timestamp': datetime.now().isoformat()
    }

    results_path = output_dir / 'results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Results saved to: {results_path}")

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
