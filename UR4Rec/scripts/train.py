"""
Training script for UR4Rec.
"""
import os
import sys
import argparse
import yaml
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models import UR4Rec
from utils import create_dataloaders, evaluate_ranking, print_metrics


def train_epoch(model, dataloader, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch in progress_bar:
        # Move to device
        user_histories = batch['user_histories'].to(device)
        candidates = batch['candidates'].to(device)
        history_masks = batch['history_masks'].to(device)
        candidate_masks = batch['candidate_masks'].to(device)
        ground_truths = batch['ground_truths']

        # Forward pass
        loss = model.compute_loss(
            user_histories,
            candidates,
            ground_truths,
            history_masks,
            candidate_masks
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Update metrics
        total_loss += loss.item()
        num_batches += 1

        # Update progress bar
        progress_bar.set_postfix({'loss': total_loss / num_batches})

    return total_loss / num_batches


def evaluate(model, dataloader, device, k_values=[5, 10, 20]):
    """Evaluate model on validation/test set."""
    model.eval()
    all_predictions = []
    all_ground_truths = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move to device
            user_histories = batch['user_histories'].to(device)
            candidates = batch['candidates'].to(device)
            history_masks = batch['history_masks'].to(device)
            candidate_masks = batch['candidate_masks'].to(device)
            ground_truths = batch['ground_truths']

            # Get predictions (without LLM for efficiency during training)
            predictions = model.predict(
                user_histories,
                candidates,
                history_masks,
                candidate_masks,
                use_llm=False
            )

            all_predictions.extend(predictions)
            all_ground_truths.extend(ground_truths)

    # Compute metrics
    metrics = evaluate_ranking(all_predictions, all_ground_truths, k_values)

    return metrics


def main(args):
    """Main training function."""
    # Load config if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        # Override with command line arguments
        for key, value in vars(args).items():
            if value is not None:
                config[key] = value
    else:
        config = vars(args)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize tensorboard
    writer = SummaryWriter(output_dir / 'logs')

    # Create dataloaders
    print("Loading data...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_path=config['train_data'],
        val_path=config['val_data'],
        test_path=config['test_data'],
        batch_size=config['batch_size'],
        max_history_len=config['max_history_len'],
        max_candidates=config['max_candidates'],
        num_workers=config['num_workers']
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Create model
    print("Creating model...")
    model = UR4Rec(
        num_items=config['num_items'],
        embedding_dim=config['embedding_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        dropout=config['dropout'],
        max_seq_len=config['max_seq_len'],
        use_llm=False  # Don't use LLM during training
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=3,
        verbose=True
    )

    # Training loop
    best_ndcg = 0
    patience_counter = 0

    for epoch in range(1, config['num_epochs'] + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{config['num_epochs']}")
        print(f"{'='*50}")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        print(f"Train Loss: {train_loss:.4f}")

        # Log to tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)

        # Evaluate on validation set
        if epoch % config['eval_every'] == 0:
            print("\nValidation:")
            val_metrics = evaluate(model, val_loader, device, config['k_values'])
            print_metrics(val_metrics, "Validation Results")

            # Log metrics
            for metric_name, value in val_metrics.items():
                writer.add_scalar(f'Metrics/val_{metric_name}', value, epoch)

            # Check for improvement
            current_ndcg = val_metrics[f'NDCG@{config["k_values"][0]}']

            if current_ndcg > best_ndcg:
                best_ndcg = current_ndcg
                patience_counter = 0

                # Save best model
                checkpoint_path = output_dir / 'best_model.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_ndcg': best_ndcg,
                    'config': config
                }, checkpoint_path)
                print(f"✓ New best model saved (NDCG@{config['k_values'][0]}: {best_ndcg:.4f})")
            else:
                patience_counter += 1
                print(f"No improvement ({patience_counter}/{config['patience']})")

            # Update learning rate
            scheduler.step(current_ndcg)

            # Early stopping
            if patience_counter >= config['patience']:
                print(f"\nEarly stopping after {epoch} epochs")
                break

        # Save checkpoint
        if epoch % config['save_every'] == 0:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config
            }, checkpoint_path)

    # Load best model and evaluate on test set
    print("\n" + "="*50)
    print("Final Evaluation on Test Set")
    print("="*50)

    checkpoint = torch.load(output_dir / 'best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics = evaluate(model, test_loader, device, config['k_values'])
    print_metrics(test_metrics, "Test Results")

    # Save test results
    with open(output_dir / 'test_results.yaml', 'w') as f:
        yaml.dump(test_metrics, f)

    writer.close()
    print(f"\nTraining completed. Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train UR4Rec model")

    # Data arguments
    parser.add_argument('--train_data', type=str, default='data/train.json',
                        help='Path to training data')
    parser.add_argument('--val_data', type=str, default='data/val.json',
                        help='Path to validation data')
    parser.add_argument('--test_data', type=str, default='data/test.json',
                        help='Path to test data')

    # Model arguments
    parser.add_argument('--num_items', type=int, default=10000,
                        help='Total number of items')
    parser.add_argument('--embedding_dim', type=int, default=256,
                        help='Embedding dimension')
    parser.add_argument('--num_layers', type=int, default=4,
                        help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=1024,
                        help='Feed-forward dimension')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--max_seq_len', type=int, default=100,
                        help='Maximum sequence length')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--max_history_len', type=int, default=50,
                        help='Maximum history length')
    parser.add_argument('--max_candidates', type=int, default=20,
                        help='Maximum number of candidates')

    # Evaluation arguments
    parser.add_argument('--k_values', type=int, nargs='+', default=[5, 10, 20],
                        help='K values for evaluation metrics')
    parser.add_argument('--eval_every', type=int, default=1,
                        help='Evaluate every N epochs')

    # Other arguments
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Output directory')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file')

    args = parser.parse_args()
    main(args)
