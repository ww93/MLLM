#!/usr/bin/env python3
"""
Extract text embeddings from ML-1M item descriptions

This script reads item descriptions from item_descriptions.json,
uses Sentence-BERT to extract text embeddings, and saves as .pt file
for FedMem training.

Usage:
    python UR4Rec/scripts/extract_ml1m_text_features.py

Output:
    UR4Rec/data/ml1m/text_features.pt - [num_items, 384] text feature tensor
"""

import json
import torch
import os
from typing import Dict
from sentence_transformers import SentenceTransformer


def load_item_descriptions(json_path: str) -> Dict[int, str]:
    """
    Load item descriptions from JSON file

    Args:
        json_path: Path to JSON file

    Returns:
        Dictionary {item_id: description_text}
    """
    print(f"📖 Loading item descriptions: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Convert string keys to integers
    descriptions = {int(k): v for k, v in data.items()}
    print(f"✅ Loaded {len(descriptions)} item descriptions")
    return descriptions


def generate_text_embeddings(
    descriptions: Dict[int, str],
    model_name: str = 'all-MiniLM-L6-v2',
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Generate text embeddings using Sentence-BERT

    Args:
        descriptions: {item_id: description_text}
        model_name: Sentence-BERT model name
        device: 'cpu', 'cuda', or 'mps'

    Returns:
        Text feature tensor [num_items, embedding_dim]
    """
    print(f"\n🤖 Loading Sentence-BERT model: {model_name}")
    model = SentenceTransformer(model_name, device=device)
    embedding_dim = model.get_sentence_embedding_dimension()
    print(f"✅ Model loaded, embedding dimension: {embedding_dim}")

    # Determine max item_id to create correct tensor size
    max_item_id = max(descriptions.keys())
    num_items = max_item_id + 1

    print(f"\n🔢 Item ID range: 1 - {max_item_id}")
    print(f"📦 Creating feature matrix: [{num_items}, {embedding_dim}]")

    # Initialize feature matrix (zero-padded)
    text_features = torch.zeros(num_items, embedding_dim, dtype=torch.float32)

    # Prepare texts and corresponding item_ids for batch encoding
    item_ids = []
    texts = []
    for item_id in sorted(descriptions.keys()):
        item_ids.append(item_id)
        texts.append(descriptions[item_id])

    print(f"\n🚀 Generating text embeddings for {len(texts)} items...")

    # Batch encoding (more efficient)
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_ids = item_ids[i:i+batch_size]

        # Generate embeddings
        embeddings = model.encode(
            batch_texts,
            convert_to_tensor=True,
            show_progress_bar=False,
            device=device
        )

        # Fill feature matrix
        for j, item_id in enumerate(batch_ids):
            text_features[item_id] = embeddings[j].cpu()

        if (i // batch_size + 1) % 10 == 0:
            print(f"  Progress: {i + len(batch_texts)}/{len(texts)} items")

    print(f"✅ Text embedding generation complete!")

    # Statistics
    num_nonzero = (text_features.abs().sum(dim=1) > 0).sum().item()
    print(f"\n📊 Statistics:")
    print(f"  - Valid features: {num_nonzero}/{num_items}")
    print(f"  - Zero-padded: {num_items - num_nonzero}")
    print(f"  - Feature shape: {text_features.shape}")
    print(f"  - Feature range: [{text_features.min():.4f}, {text_features.max():.4f}]")

    return text_features


def main():
    # Path configuration
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    json_path = os.path.join(project_root, 'UR4Rec/data/ml1m', 'item_descriptions.json')
    output_path = os.path.join(project_root, 'UR4Rec/data/ml1m', 'text_features.pt')

    print("=" * 70)
    print("📝 Extracting text features from ML-1M item descriptions")
    print("=" * 70)

    # Check input file
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"❌ Item description file not found: {json_path}")

    # Load descriptions
    descriptions = load_item_descriptions(json_path)

    # Select device
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"\n🖥️  Using device: {device.upper()}")

    # Generate embeddings
    text_features = generate_text_embeddings(descriptions, device=device)

    # Save features
    print(f"\n💾 Saving features to: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(text_features, output_path)

    # Verify save
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✅ Save successful! File size: {file_size_mb:.2f} MB")

    # Test loading
    print(f"\n🧪 Verifying file can be loaded correctly...")
    loaded_features = torch.load(output_path)
    print(f"✅ Load successful! Shape: {loaded_features.shape}")

    print("\n" + "=" * 70)
    print("🎉 Text feature extraction complete!")
    print("=" * 70)
    print(f"\n💡 Usage:")
    print(f"   python UR4Rec/scripts/train_fedmem.py \\")
    print(f"       --data_dir UR4Rec/data \\")
    print(f"       --data_file ml1m_ratings_processed.dat \\")
    print(f"       --visual_file ml1m/clip_features.pt \\")
    print(f"       --text_file ml1m/text_features.pt \\")
    print(f"       --num_rounds 50 \\")
    print(f"       --device cpu")
    print()


if __name__ == "__main__":
    main()
