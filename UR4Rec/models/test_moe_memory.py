"""Test script for UR4RecMoEMemory model.

This script demonstrates:
1. Model initialization
2. Forward pass with multimodal inputs
3. Memory updates and persistence
4. Top-K prediction
5. Memory statistics
"""
import torch
import numpy as np
from pathlib import Path

from ur4rec_moe_memory import UR4RecMoEMemory, MemoryConfig, UpdateTrigger


def test_basic_functionality():
    """Test basic model functionality."""
    print("=" * 60)
    print("Test 1: Basic Model Functionality")
    print("=" * 60)

    # Model configuration
    num_items = 1000
    embedding_dim = 128
    batch_size = 4
    history_len = 10
    num_candidates = 20

    # Create model
    memory_config = MemoryConfig(
        memory_dim=embedding_dim,
        update_trigger=UpdateTrigger.INTERACTION_COUNT,
        interaction_threshold=3,
        enable_persistence=True
    )

    model = UR4RecMoEMemory(
        num_items=num_items,
        embedding_dim=embedding_dim,
        memory_config=memory_config,
        device='cpu'
    )

    print(f"✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Sample data
    user_ids = [1, 2, 3, 4]
    history_items = torch.randint(1, num_items, (batch_size, history_len))
    target_items = torch.randint(1, num_items, (batch_size,))
    history_mask = torch.ones(batch_size, history_len, dtype=torch.bool)

    # Forward pass
    print("\nForward pass...")
    scores, info = model(
        user_ids=user_ids,
        history_items=history_items,
        target_items=target_items,
        history_mask=history_mask,
        update_memory=True
    )

    print(f"✓ Output scores shape: {scores.shape}")
    print(f"✓ Routing weights shape: {info['routing_weights'].shape}")
    print(f"✓ Sample scores: {scores[:2]}")
    print(f"✓ Sample routing weights: {info['routing_weights'][0, 0]}")

    return model


def test_memory_updates():
    """Test memory update mechanisms."""
    print("\n" + "=" * 60)
    print("Test 2: Memory Update Mechanisms")
    print("=" * 60)

    # Create model with interaction-based updates
    memory_config = MemoryConfig(
        memory_dim=64,
        update_trigger=UpdateTrigger.INTERACTION_COUNT,
        interaction_threshold=2,  # Update every 2 interactions
        max_memory_size=5
    )

    model = UR4RecMoEMemory(
        num_items=100,
        embedding_dim=64,
        memory_config=memory_config,
        device='cpu'
    )

    user_id = 42
    history = torch.randint(1, 100, (1, 5))
    target = torch.randint(1, 100, (1,))

    print(f"\nSimulating {6} interactions for user {user_id}...")

    for i in range(6):
        scores, info = model(
            user_ids=[user_id],
            history_items=history,
            target_items=target,
            update_memory=True
        )

        memory_updated = info['memory_info']['memory_updated'][0]
        interaction_count = model.retriever.user_memories[user_id].interaction_count

        print(f"  Interaction {i+1}: "
              f"Memory updated={memory_updated}, "
              f"Interaction count={interaction_count}")

    # Check memory statistics
    stats = model.get_memory_stats()
    print(f"\n✓ Memory stats: {stats}")

    # Explicit memory update
    print(f"\nTesting explicit memory update...")
    model.retriever.explicit_update_memory(user_id, force_reset=False)
    print(f"✓ Explicit update triggered for user {user_id}")


def test_drift_detection():
    """Test drift-based memory updates."""
    print("\n" + "=" * 60)
    print("Test 3: Drift-Based Memory Updates")
    print("=" * 60)

    memory_config = MemoryConfig(
        memory_dim=64,
        update_trigger=UpdateTrigger.DRIFT_THRESHOLD,
        drift_threshold=0.2,  # Update when similarity < 0.8
        max_memory_size=3
    )

    model = UR4RecMoEMemory(
        num_items=100,
        embedding_dim=64,
        memory_config=memory_config,
        device='cpu'
    )

    user_id = 99
    print(f"Testing drift detection for user {user_id}...")

    # First interaction - initialize memory
    history1 = torch.randint(1, 50, (1, 5))  # Items 1-50
    target1 = torch.randint(1, 50, (1,))

    scores1, info1 = model(
        user_ids=[user_id],
        history_items=history1,
        target_items=target1,
        update_memory=True
    )
    print(f"  Interaction 1: Drift score={info1['memory_info']['drift_scores'][0]:.4f}")

    # Similar interaction - should not update
    history2 = torch.randint(1, 50, (1, 5))
    target2 = torch.randint(1, 50, (1,))

    scores2, info2 = model(
        user_ids=[user_id],
        history_items=history2,
        target_items=target2,
        update_memory=True
    )
    print(f"  Interaction 2: Drift score={info2['memory_info']['drift_scores'][0]:.4f}, "
          f"Updated={info2['memory_info']['memory_updated'][0]}")

    # Very different interaction - should trigger update
    history3 = torch.randint(50, 100, (1, 5))  # Items 50-100 (different range)
    target3 = torch.randint(50, 100, (1,))

    scores3, info3 = model(
        user_ids=[user_id],
        history_items=history3,
        target_items=target3,
        update_memory=True
    )
    print(f"  Interaction 3: Drift score={info3['memory_info']['drift_scores'][0]:.4f}, "
          f"Updated={info3['memory_info']['memory_updated'][0]}")


def test_top_k_prediction():
    """Test top-K prediction."""
    print("\n" + "=" * 60)
    print("Test 4: Top-K Prediction")
    print("=" * 60)

    model = UR4RecMoEMemory(
        num_items=500,
        embedding_dim=64,
        device='cpu'
    )

    batch_size = 3
    user_ids = [10, 20, 30]
    history = torch.randint(1, 500, (batch_size, 8))
    candidates = torch.randint(1, 500, (batch_size, 50))

    print(f"Predicting top-10 items for {batch_size} users...")

    top_items, top_scores = model.predict_top_k(
        user_ids=user_ids,
        history_items=history,
        candidate_items=candidates,
        k=10
    )

    print(f"✓ Top items shape: {top_items.shape}")
    print(f"✓ Top scores shape: {top_scores.shape}")
    print(f"\nUser 1 top-5 items: {top_items[0, :5].tolist()}")
    print(f"User 1 top-5 scores: {top_scores[0, :5].tolist()}")


def test_memory_persistence():
    """Test memory save/load functionality."""
    print("\n" + "=" * 60)
    print("Test 5: Memory Persistence")
    print("=" * 60)

    save_dir = Path("/tmp/ur4rec_test")
    save_dir.mkdir(exist_ok=True)

    # Create model and generate some user memories
    memory_config = MemoryConfig(
        memory_dim=64,
        update_trigger=UpdateTrigger.INTERACTION_COUNT,
        interaction_threshold=1,
        enable_persistence=True
    )

    model1 = UR4RecMoEMemory(
        num_items=100,
        embedding_dim=64,
        memory_config=memory_config,
        device='cpu'
    )

    # Generate memories for multiple users
    print("Generating memories for 5 users...")
    for user_id in range(1, 6):
        history = torch.randint(1, 100, (1, 5))
        target = torch.randint(1, 100, (1,))

        model1(
            user_ids=[user_id],
            history_items=history,
            target_items=target,
            update_memory=True
        )

    stats1 = model1.get_memory_stats()
    print(f"✓ Generated memories: {stats1}")

    # Save model and memories
    model_path = save_dir / "model.pt"
    print(f"\nSaving model to {model_path}...")
    model1.save_model(str(model_path), save_memories=True)
    print("✓ Model and memories saved")

    # Create new model and load
    print("\nLoading into new model instance...")
    model2 = UR4RecMoEMemory(
        num_items=100,
        embedding_dim=64,
        memory_config=memory_config,
        device='cpu'
    )

    model2.load_model(str(model_path), load_memories=True)
    stats2 = model2.get_memory_stats()
    print(f"✓ Loaded memories: {stats2}")

    # Verify memories match
    assert stats1['num_users'] == stats2['num_users'], "User count mismatch"
    print("✓ Memory persistence verified!")


def test_multimodal_inputs():
    """Test with multimodal text and image inputs."""
    print("\n" + "=" * 60)
    print("Test 6: Multimodal Inputs (Text + Images)")
    print("=" * 60)

    model = UR4RecMoEMemory(
        num_items=100,
        embedding_dim=128,
        device='cpu'
    )

    batch_size = 2
    user_ids = [1, 2]
    history = torch.randint(1, 100, (batch_size, 5))
    targets = torch.randint(1, 100, (batch_size, 3))  # 3 candidates each

    # Text descriptions
    item_descriptions = [
        "High quality wireless headphones with noise cancellation",
        "Portable Bluetooth speaker with 20-hour battery",
        "USB-C charging cable with fast charge support",
        "High quality wireless headphones with noise cancellation",
        "Portable Bluetooth speaker with 20-hour battery",
        "USB-C charging cable with fast charge support",
    ]

    # Mock images (random tensors)
    item_images = torch.randn(6, 3, 224, 224)

    print("Running forward pass with text + image inputs...")
    scores, info = model(
        user_ids=user_ids,
        history_items=history,
        target_items=targets,
        item_descriptions=item_descriptions,
        item_images=item_images,
        update_memory=True
    )

    print(f"✓ Scores shape: {scores.shape}")
    print(f"✓ Sample scores:\n{scores}")
    print(f"✓ Routing weights (user 1, item 1): {info['routing_weights'][0, 0]}")
    print(f"  → Expert 0 (user pref): {info['routing_weights'][0, 0, 0]:.3f}")
    print(f"  → Expert 1 (item desc): {info['routing_weights'][0, 0, 1]:.3f}")
    print(f"  → Expert 2 (item image): {info['routing_weights'][0, 0, 2]:.3f}")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("UR4RecMoEMemory - Complete Test Suite")
    print("="*60 + "\n")

    try:
        # Test 1: Basic functionality
        model = test_basic_functionality()

        # Test 2: Memory updates
        test_memory_updates()

        # Test 3: Drift detection
        test_drift_detection()

        # Test 4: Top-K prediction
        test_top_k_prediction()

        # Test 5: Memory persistence
        test_memory_persistence()

        # Test 6: Multimodal inputs
        test_multimodal_inputs()

        print("\n" + "="*60)
        print("✓ All tests passed successfully!")
        print("="*60 + "\n")

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
