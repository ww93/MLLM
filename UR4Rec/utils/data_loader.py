"""
Data loading and preprocessing utilities for UR4Rec.
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional
import pickle
import json
from pathlib import Path


class RecommendationDataset(Dataset):
    """
    Dataset for recommendation reranking task.
    Each sample contains:
    - user_id: user identifier
    - user_history: sequence of user's historical interactions
    - candidate_items: list of candidate items to rerank
    - ground_truth: ground truth ranking or relevant items
    """

    def __init__(
        self,
        data_path: str,
        max_history_len: int = 50,
        max_candidates: int = 20,
        tokenizer=None
    ):
        """
        Args:
            data_path: Path to the dataset file
            max_history_len: Maximum length of user history sequence
            max_candidates: Maximum number of candidate items
            tokenizer: Optional tokenizer for text processing
        """
        self.max_history_len = max_history_len
        self.max_candidates = max_candidates
        self.tokenizer = tokenizer

        # Load data
        self.data = self._load_data(data_path)

    def _load_data(self, data_path: str) -> List[Dict]:
        """Load dataset from file."""
        path = Path(data_path)

        if path.suffix == '.json':
            with open(path, 'r') as f:
                data = json.load(f)
        elif path.suffix == '.pkl':
            with open(path, 'rb') as f:
                data = pickle.load(f)
        elif path.suffix == '.csv':
            df = pd.read_csv(path)
            data = df.to_dict('records')
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns a single sample with user history and candidate items.
        """
        sample = self.data[idx]

        # Process user history
        user_history = sample.get('user_history', [])
        if len(user_history) > self.max_history_len:
            user_history = user_history[-self.max_history_len:]

        # Process candidate items
        candidates = sample.get('candidates', [])
        if len(candidates) > self.max_candidates:
            candidates = candidates[:self.max_candidates]

        # Get ground truth
        ground_truth = sample.get('ground_truth', [])

        return {
            'user_id': sample.get('user_id', idx),
            'user_history': user_history,
            'candidates': candidates,
            'ground_truth': ground_truth,
            'metadata': sample.get('metadata', {})
        }


class CollatorWithPadding:
    """
    Custom collator for batching recommendation data.
    """

    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: List[Dict]) -> Dict:
        """
        Collate batch of samples.
        """
        user_ids = [item['user_id'] for item in batch]

        # Pad user histories
        max_history_len = max(len(item['user_history']) for item in batch)
        user_histories = []
        history_masks = []

        for item in batch:
            history = item['user_history']
            padding_len = max_history_len - len(history)

            padded_history = history + [self.pad_token_id] * padding_len
            mask = [1] * len(history) + [0] * padding_len

            user_histories.append(padded_history)
            history_masks.append(mask)

        # Pad candidates
        max_candidates = max(len(item['candidates']) for item in batch)
        candidates_list = []
        candidate_masks = []

        for item in batch:
            cands = item['candidates']
            padding_len = max_candidates - len(cands)

            padded_cands = cands + [self.pad_token_id] * padding_len
            mask = [1] * len(cands) + [0] * padding_len

            candidates_list.append(padded_cands)
            candidate_masks.append(mask)

        # Ground truth
        ground_truths = [item['ground_truth'] for item in batch]

        return {
            'user_ids': torch.tensor(user_ids, dtype=torch.long),
            'user_histories': torch.tensor(user_histories, dtype=torch.long),
            'history_masks': torch.tensor(history_masks, dtype=torch.bool),
            'candidates': torch.tensor(candidates_list, dtype=torch.long),
            'candidate_masks': torch.tensor(candidate_masks, dtype=torch.bool),
            'ground_truths': ground_truths
        }


def create_dataloaders(
    train_path: str,
    val_path: str,
    test_path: str,
    batch_size: int = 32,
    max_history_len: int = 50,
    max_candidates: int = 20,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.

    Args:
        train_path: Path to training data
        val_path: Path to validation data
        test_path: Path to test data
        batch_size: Batch size
        max_history_len: Maximum user history length
        max_candidates: Maximum number of candidates
        num_workers: Number of workers for data loading

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = RecommendationDataset(
        train_path, max_history_len, max_candidates
    )
    val_dataset = RecommendationDataset(
        val_path, max_history_len, max_candidates
    )
    test_dataset = RecommendationDataset(
        test_path, max_history_len, max_candidates
    )

    # Create collator
    collator = CollatorWithPadding()

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator
    )

    return train_loader, val_loader, test_loader


def prepare_sample_data(output_path: str = "data/sample_data.json", num_samples: int = 100):
    """
    Generate sample data for testing the framework.

    Args:
        output_path: Path to save sample data
        num_samples: Number of samples to generate
    """
    import random

    data = []
    num_items = 1000

    for i in range(num_samples):
        # Generate random user history
        history_len = random.randint(10, 50)
        user_history = random.sample(range(1, num_items), history_len)

        # Generate candidate items
        num_candidates = random.randint(10, 20)
        candidates = random.sample(range(1, num_items), num_candidates)

        # Generate ground truth (relevant items)
        num_relevant = random.randint(1, 5)
        ground_truth = random.sample(candidates, num_relevant)

        sample = {
            'user_id': i,
            'user_history': user_history,
            'candidates': candidates,
            'ground_truth': ground_truth,
            'metadata': {
                'timestamp': f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}"
            }
        }
        data.append(sample)

    # Save to file
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Sample data saved to {output_path}")
    return data


if __name__ == "__main__":
    # Test data loading
    print("Generating sample data...")
    prepare_sample_data("../data/train.json", num_samples=100)
    prepare_sample_data("../data/val.json", num_samples=20)
    prepare_sample_data("../data/test.json", num_samples=20)

    print("\nTesting data loader...")
    train_loader, val_loader, test_loader = create_dataloaders(
        "../data/train.json",
        "../data/val.json",
        "../data/test.json",
        batch_size=8
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Test one batch
    batch = next(iter(train_loader))
    print("\nSample batch:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        else:
            print(f"  {key}: {len(value)} items")
