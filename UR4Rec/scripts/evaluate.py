"""
Evaluation script for UR4Rec with LLM reranking.
"""
import os
import sys
import argparse
import yaml
import torch
from tqdm import tqdm
from pathlib import Path
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models import UR4Rec
from utils import create_dataloaders, evaluate_ranking, print_metrics


def evaluate_with_llm(model, dataloader, device, use_llm=True, k_values=[5, 10, 20]):
    """
    Evaluate model with optional LLM reranking.

    Args:
        model: UR4Rec model
        dataloader: DataLoader
        device: Device to use
        use_llm: Whether to use LLM reranking
        k_values: K values for metrics

    Returns:
        metrics: Dictionary of evaluation metrics
    """
    model.eval()
    all_predictions = []
    all_ground_truths = []
    all_retriever_scores = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move to device
            user_histories = batch['user_histories'].to(device)
            candidates = batch['candidates'].to(device)
            history_masks = batch['history_masks'].to(device)
            candidate_masks = batch['candidate_masks'].to(device)
            ground_truths = batch['ground_truths']

            # Get predictions
            predictions = model.predict(
                user_histories,
                candidates,
                history_masks,
                candidate_masks,
                use_llm=use_llm
            )

            # Also get retriever scores for analysis
            preference_scores, _ = model(
                user_histories,
                candidates,
                history_masks,
                candidate_masks
            )

            all_predictions.extend(predictions)
            all_ground_truths.extend(ground_truths)
            all_retriever_scores.extend(preference_scores.cpu().numpy())

    # Compute metrics
    metrics = evaluate_ranking(all_predictions, all_ground_truths, k_values)

    return metrics, all_predictions, all_retriever_scores


def main(args):
    """Main evaluation function."""
    # Load config
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Try to load from checkpoint
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        config = checkpoint.get('config', {})

    # Override config with command line arguments
    for key, value in vars(args).items():
        if value is not None and key not in ['config', 'checkpoint']:
            config[key] = value

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print("Loading data...")
    _, _, test_loader = create_dataloaders(
        train_path=config.get('train_data', 'data/train.json'),
        val_path=config.get('val_data', 'data/val.json'),
        test_path=config.get('test_data', 'data/test.json'),
        batch_size=args.batch_size,
        max_history_len=config.get('max_history_len', 50),
        max_candidates=config.get('max_candidates', 20),
        num_workers=args.num_workers
    )

    print(f"Test batches: {len(test_loader)}")

    # Create model
    print("Creating model...")
    model = UR4Rec(
        num_items=config.get('num_items', 10000),
        embedding_dim=config.get('embedding_dim', 256),
        num_layers=config.get('num_layers', 4),
        num_heads=config.get('num_heads', 8),
        d_ff=config.get('d_ff', 1024),
        dropout=config.get('dropout', 0.1),
        max_seq_len=config.get('max_seq_len', 100),
        llm_backend=args.llm_backend,
        llm_model_name=args.llm_model,
        llm_api_key=args.llm_api_key,
        use_llm=args.use_llm
    ).to(device)

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    # Evaluate without LLM (retriever only)
    print("\n" + "="*50)
    print("Evaluation: Retriever Only")
    print("="*50)

    metrics_no_llm, predictions_no_llm, scores_no_llm = evaluate_with_llm(
        model, test_loader, device, use_llm=False, k_values=args.k_values
    )
    print_metrics(metrics_no_llm, "Retriever Only Results")

    # Evaluate with LLM (if enabled)
    if args.use_llm:
        print("\n" + "="*50)
        print("Evaluation: With LLM Reranking")
        print("="*50)

        metrics_with_llm, predictions_with_llm, scores_with_llm = evaluate_with_llm(
            model, test_loader, device, use_llm=True, k_values=args.k_values
        )
        print_metrics(metrics_with_llm, "LLM Reranking Results")

        # Compute improvement
        print("\n" + "="*50)
        print("Improvement with LLM")
        print("="*50)

        for key in metrics_no_llm:
            baseline = metrics_no_llm[key]
            improved = metrics_with_llm[key]
            improvement = ((improved - baseline) / baseline * 100) if baseline > 0 else 0
            print(f"{key:15s}: {baseline:.4f} → {improved:.4f} ({improvement:+.2f}%)")

        # Save results
        results = {
            'retriever_only': metrics_no_llm,
            'with_llm': metrics_with_llm,
            'config': config
        }
    else:
        results = {
            'retriever_only': metrics_no_llm,
            'config': config
        }

    # Save results
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        yaml.dump(results, f)

    print(f"\nResults saved to {output_path}")

    # Save predictions if requested
    if args.save_predictions:
        pred_path = output_path.parent / 'predictions.json'
        pred_data = {
            'retriever_only': predictions_no_llm,
            'with_llm': predictions_with_llm if args.use_llm else None
        }

        with open(pred_path, 'w') as f:
            json.dump(pred_data, f, indent=2)

        print(f"Predictions saved to {pred_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate UR4Rec model")

    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')

    # Data arguments
    parser.add_argument('--test_data', type=str, default='data/test.json',
                        help='Path to test data')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers')

    # LLM arguments
    parser.add_argument('--use_llm', action='store_true',
                        help='Use LLM for reranking')
    parser.add_argument('--llm_backend', type=str, default='openai',
                        choices=['openai', 'anthropic', 'local'],
                        help='LLM backend to use')
    parser.add_argument('--llm_model', type=str, default=None,
                        help='LLM model name')
    parser.add_argument('--llm_api_key', type=str, default=None,
                        help='API key for LLM')

    # Evaluation arguments
    parser.add_argument('--k_values', type=int, nargs='+', default=[5, 10, 20],
                        help='K values for evaluation')

    # Output arguments
    parser.add_argument('--output_file', type=str, default='outputs/eval_results.yaml',
                        help='Output file for results')
    parser.add_argument('--save_predictions', action='store_true',
                        help='Save predictions to file')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file')

    args = parser.parse_args()
    main(args)
