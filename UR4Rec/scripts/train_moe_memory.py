"""Training script for UR4RecMoEMemory model.

This script provides:
1. Complete training pipeline with validation
2. Support for different memory update strategies
3. Evaluation metrics (NDCG, Hit Rate, MRR)
4. Model checkpointing and memory persistence
5. TensorBoard logging
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import yaml
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple

from models.ur4rec_moe_memory import UR4RecMoEMemory, MemoryConfig, UpdateTrigger


class RecommendationDataset(Dataset):
    """Dataset for recommendation training."""

    def __init__(
        self,
        sequences: Dict[int, List[int]],
        num_items: int,
        max_seq_len: int = 50,
        num_negatives: int = 4,
        mode: str = 'train'
    ):
        """
        Args:
            sequences: Dict mapping user_id -> list of item_ids
            num_items: Total number of items
            max_seq_len: Maximum sequence length
            num_negatives: Number of negative samples per positive
            mode: 'train' or 'eval'
        """
        self.sequences = sequences
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.num_negatives = num_negatives
        self.mode = mode

        self.user_ids = list(sequences.keys())
        self.all_items = set(range(1, num_items + 1))

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        sequence = self.sequences[user_id]

        if len(sequence) < 2:
            sequence = [0] * (2 - len(sequence)) + sequence

        # Split into history and target
        history = sequence[:-1]
        target = sequence[-1]

        # Truncate/pad history
        if len(history) > self.max_seq_len:
            history = history[-self.max_seq_len:]
        else:
            history = [0] * (self.max_seq_len - len(history)) + history

        history_mask = [1 if item != 0 else 0 for item in history]

        # Sample negatives
        user_items = set(sequence)
        negatives = []
        while len(negatives) < self.num_negatives:
            neg = np.random.randint(1, self.num_items + 1)
            if neg not in user_items:
                negatives.append(neg)

        # Create candidates (1 positive + N negatives)
        candidates = [target] + negatives
        labels = [1.0] + [0.0] * self.num_negatives

        return {
            'user_id': user_id,
            'history': torch.LongTensor(history),
            'history_mask': torch.BoolTensor(history_mask),
            'candidates': torch.LongTensor(candidates),
            'labels': torch.FloatTensor(labels)
        }


def collate_fn(batch):
    """Collate function for DataLoader."""
    return {
        'user_ids': [item['user_id'] for item in batch],
        'history': torch.stack([item['history'] for item in batch]),
        'history_mask': torch.stack([item['history_mask'] for item in batch]),
        'candidates': torch.stack([item['candidates'] for item in batch]),
        'labels': torch.stack([item['labels'] for item in batch])
    }


def compute_metrics(scores: torch.Tensor, labels: torch.Tensor, k_list: List[int] = [5, 10, 20]) -> Dict:
    """Compute ranking metrics.

    Args:
        scores: [batch, num_candidates] prediction scores
        labels: [batch, num_candidates] binary labels

    Returns:
        Dictionary of metrics
    """
    batch_size = scores.size(0)

    # Get rankings
    _, indices = torch.sort(scores, dim=1, descending=True)

    metrics = {}

    for k in k_list:
        hit_count = 0
        ndcg_sum = 0.0
        mrr_sum = 0.0

        for i in range(batch_size):
            # Get top-k predictions
            top_k = indices[i, :k]

            # Find positive items
            pos_items = (labels[i] > 0).nonzero(as_tuple=True)[0]

            if len(pos_items) == 0:
                continue

            # Hit rate
            if any(idx in top_k for idx in pos_items):
                hit_count += 1

            # NDCG
            dcg = 0.0
            idcg = 0.0
            for rank, idx in enumerate(top_k):
                if labels[i, idx] > 0:
                    dcg += 1.0 / np.log2(rank + 2)
            for rank in range(min(k, len(pos_items))):
                idcg += 1.0 / np.log2(rank + 2)

            if idcg > 0:
                ndcg_sum += dcg / idcg

            # MRR
            for rank, idx in enumerate(top_k):
                if labels[i, idx] > 0:
                    mrr_sum += 1.0 / (rank + 1)
                    break

        metrics[f'hit@{k}'] = hit_count / batch_size
        metrics[f'ndcg@{k}'] = ndcg_sum / batch_size
        metrics[f'mrr@{k}'] = mrr_sum / batch_size

    return metrics


def train_epoch(
    model: UR4RecMoEMemory,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: str,
    epoch: int
) -> Dict:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    num_batches = 0

    criterion = nn.BCEWithLogitsLoss()

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch in progress_bar:
        # Move to device
        user_ids = batch['user_ids']
        history = batch['history'].to(device)
        history_mask = batch['history_mask'].to(device)
        candidates = batch['candidates'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        scores, info = model(
            user_ids=user_ids,
            history_items=history,
            target_items=candidates,
            history_mask=history_mask,
            update_memory=True  # Update memory during training
        )

        # Compute loss
        loss = criterion(scores, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Update metrics
        total_loss += loss.item()
        num_batches += 1

        progress_bar.set_postfix({'loss': total_loss / num_batches})

    return {
        'loss': total_loss / num_batches
    }


def evaluate(
    model: UR4RecMoEMemory,
    dataloader: DataLoader,
    device: str,
    k_list: List[int] = [5, 10, 20]
) -> Dict:
    """Evaluate model."""
    model.eval()

    all_scores = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            user_ids = batch['user_ids']
            history = batch['history'].to(device)
            history_mask = batch['history_mask'].to(device)
            candidates = batch['candidates'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass (no memory update during eval)
            scores, _ = model(
                user_ids=user_ids,
                history_items=history,
                target_items=candidates,
                history_mask=history_mask,
                update_memory=False
            )

            all_scores.append(scores)
            all_labels.append(labels)

    # Concatenate all batches
    all_scores = torch.cat(all_scores, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Compute metrics
    metrics = compute_metrics(all_scores, all_labels, k_list)

    return metrics


def main(args):
    """Main training function."""
    print("="*60)
    print("UR4RecMoEMemory Training")
    print("="*60)

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Device: {device}")

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print(f"Config: {args.config}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # TensorBoard
    writer = SummaryWriter(output_dir / 'logs')

    # Load data
    print("\nLoading data...")
    data_dir = Path(args.data_dir)

    with open(data_dir / 'train_sequences.json', 'r') as f:
        train_sequences = {int(k): v for k, v in json.load(f).items()}
    with open(data_dir / 'val_sequences.json', 'r') as f:
        val_sequences = {int(k): v for k, v in json.load(f).items()}
    with open(data_dir / 'test_sequences.json', 'r') as f:
        test_sequences = {int(k): v for k, v in json.load(f).items()}

    # Get num_items
    all_items = set()
    for seq in list(train_sequences.values()) + list(val_sequences.values()) + list(test_sequences.values()):
        all_items.update(seq)
    num_items = max(all_items)

    print(f"  Num items: {num_items}")
    print(f"  Train users: {len(train_sequences)}")
    print(f"  Val users: {len(val_sequences)}")
    print(f"  Test users: {len(test_sequences)}")

    # Create datasets
    train_dataset = RecommendationDataset(
        train_sequences, num_items,
        max_seq_len=config['max_seq_len'],
        num_negatives=config['num_negatives'],
        mode='train'
    )

    val_dataset = RecommendationDataset(
        val_sequences, num_items,
        max_seq_len=config['max_seq_len'],
        num_negatives=config['num_negatives'],
        mode='eval'
    )

    test_dataset = RecommendationDataset(
        test_sequences, num_items,
        max_seq_len=config['max_seq_len'],
        num_negatives=config['num_negatives'],
        mode='eval'
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'] * 2,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'] * 2,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers
    )

    # Create model
    print("\nCreating model...")

    memory_config = MemoryConfig(
        memory_dim=config['embedding_dim'],
        max_memory_size=config.get('max_memory_size', 10),
        update_trigger=UpdateTrigger[config.get('update_trigger', 'INTERACTION_COUNT')],
        interaction_threshold=config.get('interaction_threshold', 10),
        drift_threshold=config.get('drift_threshold', 0.3),
        decay_factor=config.get('decay_factor', 0.95),
        enable_persistence=True
    )

    model = UR4RecMoEMemory(
        num_items=num_items,
        embedding_dim=config['embedding_dim'],
        num_heads=config['num_heads'],
        dropout=config['dropout'],
        num_proxies=config.get('num_proxies', 4),
        memory_config=memory_config,
        device=device
    )

    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )

    # Training loop
    best_ndcg = 0.0
    patience_counter = 0

    for epoch in range(1, config['num_epochs'] + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{config['num_epochs']}")
        print(f"{'='*60}")

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch)
        print(f"Train loss: {train_metrics['loss']:.4f}")

        writer.add_scalar('Loss/train', train_metrics['loss'], epoch)

        # Evaluate
        if epoch % args.eval_every == 0:
            print("\nValidation:")
            val_metrics = evaluate(model, val_loader, device, config['k_values'])

            for key, value in val_metrics.items():
                print(f"  {key}: {value:.4f}")
                writer.add_scalar(f'Metrics/val_{key}', value, epoch)

            # Check improvement
            current_ndcg = val_metrics['ndcg@10']

            if current_ndcg > best_ndcg:
                best_ndcg = current_ndcg
                patience_counter = 0

                # Save best model
                print(f"✓ New best model (NDCG@10: {best_ndcg:.4f})")
                model.save_model(
                    str(output_dir / 'best_model.pt'),
                    save_memories=True
                )

            else:
                patience_counter += 1
                print(f"No improvement ({patience_counter}/{args.patience})")

            # Scheduler step
            scheduler.step(current_ndcg)

            # Early stopping
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

        # Memory statistics
        if epoch % 5 == 0:
            memory_stats = model.get_memory_stats()
            print(f"\nMemory stats: {memory_stats}")
            for key, value in memory_stats.items():
                if isinstance(value, (int, float)):
                    writer.add_scalar(f'Memory/{key}', value, epoch)

    # Final evaluation on test set
    print("\n" + "="*60)
    print("Final Test Evaluation")
    print("="*60)

    model.load_model(str(output_dir / 'best_model.pt'), load_memories=True)
    test_metrics = evaluate(model, test_loader, device, config['k_values'])

    print("\nTest metrics:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.4f}")

    # Save results
    results = {
        'config': config,
        'test_metrics': test_metrics,
        'best_val_ndcg': float(best_ndcg),
        'timestamp': datetime.now().isoformat()
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    writer.close()
    print(f"\n✓ Training completed. Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train UR4RecMoEMemory')

    parser.add_argument('--config', type=str, required=True,
                        help='Path to config YAML file')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing sequence data')
    parser.add_argument('--output_dir', type=str, default='outputs/moe_memory',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'], help='Device')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--eval_every', type=int, default=1,
                        help='Evaluate every N epochs')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()
    main(args)
