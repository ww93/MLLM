"""
Simple demo script for UR4Rec.

This script demonstrates:
1. Creating sample data
2. Training a simple model
3. Making predictions
"""
import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from models import UR4Rec
from utils import prepare_sample_data, RecommendationDataset, CollatorWithPadding
from torch.utils.data import DataLoader


def main():
    print("="*60)
    print("UR4Rec Demo".center(60))
    print("="*60)

    # Step 1: Prepare sample data
    print("\n[Step 1] Preparing sample data...")
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    prepare_sample_data("data/demo_train.json", num_samples=50)
    prepare_sample_data("data/demo_test.json", num_samples=10)

    print("✓ Sample data created")

    # Step 2: Create model
    print("\n[Step 2] Creating UR4Rec model...")

    model = UR4Rec(
        num_items=1000,
        embedding_dim=128,
        num_layers=2,
        num_heads=4,
        d_ff=512,
        dropout=0.1,
        use_llm=False  # Start without LLM for demo
    )

    print(f"✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Step 3: Load data
    print("\n[Step 3] Loading data...")

    train_dataset = RecommendationDataset("data/demo_train.json")
    test_dataset = RecommendationDataset("data/demo_test.json")

    collator = CollatorWithPadding()

    train_loader = DataLoader(
        train_dataset, batch_size=8, shuffle=True, collate_fn=collator
    )
    test_loader = DataLoader(
        test_dataset, batch_size=8, shuffle=False, collate_fn=collator
    )

    print(f"✓ Train samples: {len(train_dataset)}")
    print(f"✓ Test samples: {len(test_dataset)}")

    # Step 4: Train for a few epochs
    print("\n[Step 4] Training model (5 epochs)...")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    print(f"Using device: {device}")

    for epoch in range(1, 6):
        model.train()
        total_loss = 0
        num_batches = 0

        for batch in train_loader:
            user_histories = batch['user_histories'].to(device)
            candidates = batch['candidates'].to(device)
            history_masks = batch['history_masks'].to(device)
            candidate_masks = batch['candidate_masks'].to(device)
            ground_truths = batch['ground_truths']

            # Forward pass
            loss = model.compute_loss(
                user_histories, candidates, ground_truths,
                history_masks, candidate_masks
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"  Epoch {epoch}/5 - Loss: {avg_loss:.4f}")

    print("✓ Training completed")

    # Step 5: Make predictions
    print("\n[Step 5] Making predictions on test set...")

    model.eval()
    all_predictions = []
    all_ground_truths = []

    with torch.no_grad():
        for batch in test_loader:
            user_histories = batch['user_histories'].to(device)
            candidates = batch['candidates'].to(device)
            history_masks = batch['history_masks'].to(device)
            candidate_masks = batch['candidate_masks'].to(device)
            ground_truths = batch['ground_truths']

            # Get predictions
            predictions = model.predict(
                user_histories, candidates,
                history_masks, candidate_masks,
                use_llm=False
            )

            all_predictions.extend(predictions)
            all_ground_truths.extend(ground_truths)

    # Show some examples
    print("\n✓ Predictions made. Here are some examples:")
    print("-" * 60)

    for i in range(min(3, len(all_predictions))):
        pred = all_predictions[i]
        gt = all_ground_truths[i]

        print(f"\nExample {i+1}:")
        print(f"  Predicted ranking (top 5): {pred[:5]}")
        print(f"  Ground truth: {gt}")

        # Check if any ground truth items are in top 5
        top_5 = set(pred[:5])
        gt_set = set(gt)
        hits = top_5 & gt_set

        if hits:
            print(f"  ✓ Hit! Found {len(hits)} relevant item(s) in top 5: {hits}")
        else:
            print(f"  ✗ Miss - no relevant items in top 5")

    # Step 6: Compute metrics
    print("\n[Step 6] Computing evaluation metrics...")

    from utils import evaluate_ranking, print_metrics

    metrics = evaluate_ranking(all_predictions, all_ground_truths, k_values=[5, 10])
    print_metrics(metrics, "Demo Results")

    # Step 7: Save model
    print("\n[Step 7] Saving model...")

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    model.save_retriever("outputs/demo_model.pt")
    print("✓ Model saved to outputs/demo_model.pt")

    # Summary
    print("\n" + "="*60)
    print("Demo completed successfully!".center(60))
    print("="*60)
    print("\nNext steps:")
    print("1. Train on larger datasets using scripts/train.py")
    print("2. Enable LLM reranking by setting use_llm=True")
    print("3. Evaluate with different LLM backends")
    print("\nFor more information, see README.md")


if __name__ == "__main__":
    main()
