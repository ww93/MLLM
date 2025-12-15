"""
Prepare ML-100K data for UR4Rec training

This script:
1. Loads movies.dat and ratings.dat
2. Filters high ratings (>=4.0) as positive samples
3. Builds user sequences sorted by timestamp
4. Splits into train/val/test sets
5. Saves as .npy and item_map.json
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import json
from collections import defaultdict
from typing import Dict, List, Tuple
import random

def load_movies(data_dir: Path) -> Dict[int, Dict]:
    """Load movies.dat"""
    print("Loading movies.dat...")

    movies = {}
    with open(data_dir / 'movies.dat', 'r', encoding='latin-1') as f:
        for line in f:
            parts = line.strip().split('::')
            if len(parts) >= 3:
                movie_id = int(parts[0])
                title = parts[1]
                genres = parts[2].split('|')

                movies[movie_id] = {
                    'title': title,
                    'genres': genres,
                    'genres_str': parts[2]
                }

    print(f"  Loaded {len(movies)} movies")
    return movies


def load_ratings(data_dir: Path) -> pd.DataFrame:
    """Load ratings.dat"""
    print("Loading ratings.dat...")

    ratings = []
    with open(data_dir / 'ratings.dat', 'r', encoding='latin-1') as f:
        for line in f:
            parts = line.strip().split('::')
            if len(parts) >= 4:
                user_id = int(parts[0])
                movie_id = int(parts[1])
                rating = float(parts[2])
                timestamp = int(parts[3])

                ratings.append({
                    'user_id': user_id,
                    'item_id': movie_id,
                    'rating': rating,
                    'timestamp': timestamp
                })

    df = pd.DataFrame(ratings)
    print(f"  Loaded {len(df)} ratings")
    print(f"  Users: {df['user_id'].nunique()}")
    print(f"  Items: {df['item_id'].nunique()}")
    print(f"  Rating range: {df['rating'].min():.1f} - {df['rating'].max():.1f}")

    return df


def filter_ratings(df: pd.DataFrame, min_rating: float = 4.0) -> pd.DataFrame:
    """Filter high ratings as positive samples"""
    print(f"\nFiltering ratings >= {min_rating}...")

    filtered = df[df['rating'] >= min_rating].copy()

    print(f"  Before: {len(df)} ratings")
    print(f"  After: {len(filtered)} ratings ({len(filtered)/len(df)*100:.1f}%)")
    print(f"  Users: {filtered['user_id'].nunique()}")
    print(f"  Items: {filtered['item_id'].nunique()}")

    return filtered


def build_user_sequences(
    df: pd.DataFrame,
    min_seq_len: int = 5
) -> Dict[int, List[int]]:
    """Build user interaction sequences"""
    print(f"\nBuilding user sequences (min_seq_len={min_seq_len})...")

    # Group by user and sort by timestamp
    user_sequences = defaultdict(list)

    for _, row in df.iterrows():
        user_sequences[row['user_id']].append((row['item_id'], row['timestamp']))

    # Sort by timestamp
    for user_id in user_sequences:
        user_sequences[user_id].sort(key=lambda x: x[1])
        user_sequences[user_id] = [item_id for item_id, _ in user_sequences[user_id]]

    # Filter short sequences
    user_sequences = {
        uid: seq for uid, seq in user_sequences.items()
        if len(seq) >= min_seq_len
    }

    print(f"  Valid users: {len(user_sequences)}")
    print(f"  Avg sequence length: {np.mean([len(seq) for seq in user_sequences.values()]):.1f}")
    print(f"  Min sequence length: {min([len(seq) for seq in user_sequences.values()])}")
    print(f"  Max sequence length: {max([len(seq) for seq in user_sequences.values()])}")

    return dict(user_sequences)


def split_sequences(
    user_sequences: Dict[int, List[int]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1
) -> Tuple[Dict, Dict, Dict]:
    """Split sequences into train/val/test"""
    print(f"\nSplitting sequences (train={train_ratio}, val={val_ratio}, test={1-train_ratio-val_ratio})...")

    train_seq = {}
    val_seq = {}
    test_seq = {}

    for user_id, sequence in user_sequences.items():
        seq_len = len(sequence)

        # Calculate split points
        train_end = int(seq_len * train_ratio)
        val_end = int(seq_len * (train_ratio + val_ratio))

        # Ensure at least 1 item in val and test
        if train_end < 1:
            train_end = 1
        if val_end <= train_end:
            val_end = train_end + 1
        if val_end >= seq_len:
            val_end = seq_len - 1

        # Split
        train_seq[user_id] = sequence[:train_end]
        val_seq[user_id] = sequence[:val_end]  # Include training data
        test_seq[user_id] = sequence  # Full sequence

    print(f"  Train users: {len(train_seq)}")
    print(f"  Val users: {len(val_seq)}")
    print(f"  Test users: {len(test_seq)}")

    return train_seq, val_seq, test_seq


def create_item_map(movies: Dict[int, Dict]) -> Dict[str, int]:
    """Create item ID mapping"""
    print("\nCreating item map...")

    # Get all item IDs
    item_ids = sorted(movies.keys())

    # Create mapping: item_id -> continuous_id
    item_map = {str(item_id): item_id for item_id in item_ids}

    print(f"  Items: {len(item_map)}")

    return item_map


def save_data(
    train_seq: Dict,
    val_seq: Dict,
    test_seq: Dict,
    item_map: Dict,
    output_dir: Path
):
    """Save processed data"""
    print(f"\nSaving data to {output_dir}...")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save sequences
    np.save(output_dir / 'train_sequences.npy', train_seq)
    np.save(output_dir / 'val_sequences.npy', val_seq)
    np.save(output_dir / 'test_sequences.npy', test_seq)

    # Save item map
    with open(output_dir / 'item_map.json', 'w') as f:
        json.dump(item_map, f, indent=2)

    print("  ✓ train_sequences.npy")
    print("  ✓ val_sequences.npy")
    print("  ✓ test_sequences.npy")
    print("  ✓ item_map.json")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Prepare ML-100K data for UR4Rec')
    parser.add_argument('--data_dir', type=str,
                       default='UR4Rec/data/Multimodal_Datasets/M_ML-100K',
                       help='Raw data directory')
    parser.add_argument('--output_dir', type=str,
                       default='UR4Rec/data/Multimodal_Datasets',
                       help='Output directory')
    parser.add_argument('--min_rating', type=float, default=4.0,
                       help='Minimum rating for positive samples')
    parser.add_argument('--min_seq_len', type=int, default=5,
                       help='Minimum sequence length')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Train set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                       help='Validation set ratio')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    print("="*60)
    print("ML-100K Data Preparation for UR4Rec")
    print("="*60)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Min rating: {args.min_rating}")
    print(f"Min sequence length: {args.min_seq_len}")
    print(f"Split ratio: {args.train_ratio}/{args.val_ratio}/{1-args.train_ratio-args.val_ratio}")

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    # Step 1: Load movies
    movies = load_movies(data_dir)

    # Step 2: Load ratings
    ratings = load_ratings(data_dir)

    # Step 3: Filter high ratings
    filtered_ratings = filter_ratings(ratings, args.min_rating)

    # Step 4: Build user sequences
    user_sequences = build_user_sequences(filtered_ratings, args.min_seq_len)

    # Step 5: Split sequences
    train_seq, val_seq, test_seq = split_sequences(
        user_sequences,
        args.train_ratio,
        args.val_ratio
    )

    # Step 6: Create item map
    item_map = create_item_map(movies)

    # Step 7: Save data
    save_data(train_seq, val_seq, test_seq, item_map, output_dir)

    print("\n" + "="*60)
    print("Data preparation complete!")
    print("="*60)

    print("\nGenerated files:")
    print(f"  {output_dir}/train_sequences.npy")
    print(f"  {output_dir}/val_sequences.npy")
    print(f"  {output_dir}/test_sequences.npy")
    print(f"  {output_dir}/item_map.json")

    print("\nData statistics:")
    print(f"  Total items: {len(item_map)}")
    print(f"  Train users: {len(train_seq)}")
    print(f"  Val users: {len(val_seq)}")
    print(f"  Test users: {len(test_seq)}")
    print(f"  Avg train seq length: {np.mean([len(seq) for seq in train_seq.values()]):.1f}")

    print("\nNext steps:")
    print("  1. Generate LLM preferences:")
    print("     export DASHSCOPE_API_KEY='your-key'")
    print("     python UR4Rec/models/llm_generator.py")
    print("\n  2. Train the model:")
    print("     python UR4Rec/scripts/train_ur4rec_moe.py \\")
    print("         --config UR4Rec/configs/ur4rec_moe_100k.yaml \\")
    print("         --data_dir UR4Rec/data/Multimodal_Datasets \\")
    print("         --llm_data_dir data/llm_generated \\")
    print("         --output_dir outputs/ur4rec_moe")


if __name__ == "__main__":
    main()
