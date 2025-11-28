"""
Evaluation metrics for recommendation reranking.
"""
import numpy as np
from typing import List, Dict, Union
import torch


def ndcg_at_k(predicted_ranking: List, ground_truth: List, k: int = 10) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at K.

    Args:
        predicted_ranking: Predicted ranking of items
        ground_truth: List of relevant items
        k: Cutoff rank

    Returns:
        NDCG@K score
    """
    if len(predicted_ranking) == 0 or len(ground_truth) == 0:
        return 0.0

    # Calculate DCG
    dcg = 0.0
    for i, item in enumerate(predicted_ranking[:k]):
        if item in ground_truth:
            dcg += 1.0 / np.log2(i + 2)

    # Calculate IDCG (ideal DCG)
    idcg = 0.0
    for i in range(min(len(ground_truth), k)):
        idcg += 1.0 / np.log2(i + 2)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def hit_rate_at_k(predicted_ranking: List, ground_truth: List, k: int = 10) -> float:
    """
    Calculate Hit Rate at K.

    Args:
        predicted_ranking: Predicted ranking of items
        ground_truth: List of relevant items
        k: Cutoff rank

    Returns:
        HR@K score (1 if any relevant item in top-k, 0 otherwise)
    """
    if len(predicted_ranking) == 0 or len(ground_truth) == 0:
        return 0.0

    top_k = set(predicted_ranking[:k])
    relevant = set(ground_truth)

    return 1.0 if len(top_k & relevant) > 0 else 0.0


def mrr_at_k(predicted_ranking: List, ground_truth: List, k: int = 10) -> float:
    """
    Calculate Mean Reciprocal Rank at K.

    Args:
        predicted_ranking: Predicted ranking of items
        ground_truth: List of relevant items
        k: Cutoff rank

    Returns:
        MRR@K score
    """
    if len(predicted_ranking) == 0 or len(ground_truth) == 0:
        return 0.0

    relevant = set(ground_truth)

    for i, item in enumerate(predicted_ranking[:k]):
        if item in relevant:
            return 1.0 / (i + 1)

    return 0.0


def precision_at_k(predicted_ranking: List, ground_truth: List, k: int = 10) -> float:
    """
    Calculate Precision at K.

    Args:
        predicted_ranking: Predicted ranking of items
        ground_truth: List of relevant items
        k: Cutoff rank

    Returns:
        P@K score
    """
    if len(predicted_ranking) == 0 or len(ground_truth) == 0:
        return 0.0

    top_k = set(predicted_ranking[:k])
    relevant = set(ground_truth)

    return len(top_k & relevant) / k


def recall_at_k(predicted_ranking: List, ground_truth: List, k: int = 10) -> float:
    """
    Calculate Recall at K.

    Args:
        predicted_ranking: Predicted ranking of items
        ground_truth: List of relevant items
        k: Cutoff rank

    Returns:
        R@K score
    """
    if len(predicted_ranking) == 0 or len(ground_truth) == 0:
        return 0.0

    top_k = set(predicted_ranking[:k])
    relevant = set(ground_truth)

    return len(top_k & relevant) / len(relevant)


def map_at_k(predicted_ranking: List, ground_truth: List, k: int = 10) -> float:
    """
    Calculate Mean Average Precision at K.

    Args:
        predicted_ranking: Predicted ranking of items
        ground_truth: List of relevant items
        k: Cutoff rank

    Returns:
        MAP@K score
    """
    if len(predicted_ranking) == 0 or len(ground_truth) == 0:
        return 0.0

    relevant = set(ground_truth)
    score = 0.0
    num_hits = 0

    for i, item in enumerate(predicted_ranking[:k]):
        if item in relevant:
            num_hits += 1
            score += num_hits / (i + 1)

    if num_hits == 0:
        return 0.0

    return score / min(len(relevant), k)


def evaluate_ranking(
    predicted_rankings: List[List],
    ground_truths: List[List],
    k_values: List[int] = [5, 10, 20]
) -> Dict[str, float]:
    """
    Evaluate ranking performance with multiple metrics.

    Args:
        predicted_rankings: List of predicted rankings
        ground_truths: List of ground truth relevant items
        k_values: List of K values for evaluation

    Returns:
        Dictionary of metric scores
    """
    results = {}

    for k in k_values:
        ndcg_scores = []
        hr_scores = []
        mrr_scores = []
        precision_scores = []
        recall_scores = []
        map_scores = []

        for pred, gt in zip(predicted_rankings, ground_truths):
            ndcg_scores.append(ndcg_at_k(pred, gt, k))
            hr_scores.append(hit_rate_at_k(pred, gt, k))
            mrr_scores.append(mrr_at_k(pred, gt, k))
            precision_scores.append(precision_at_k(pred, gt, k))
            recall_scores.append(recall_at_k(pred, gt, k))
            map_scores.append(map_at_k(pred, gt, k))

        results[f'NDCG@{k}'] = np.mean(ndcg_scores)
        results[f'HR@{k}'] = np.mean(hr_scores)
        results[f'MRR@{k}'] = np.mean(mrr_scores)
        results[f'P@{k}'] = np.mean(precision_scores)
        results[f'R@{k}'] = np.mean(recall_scores)
        results[f'MAP@{k}'] = np.mean(map_scores)

    return results


def print_metrics(metrics: Dict[str, float], title: str = "Evaluation Results"):
    """
    Pretty print evaluation metrics.

    Args:
        metrics: Dictionary of metric scores
        title: Title for the output
    """
    print(f"\n{'='*50}")
    print(f"{title:^50}")
    print(f"{'='*50}")

    for metric_name, score in metrics.items():
        print(f"{metric_name:15s}: {score:.4f}")

    print(f"{'='*50}\n")


if __name__ == "__main__":
    # Test metrics
    print("Testing evaluation metrics...")

    # Example rankings
    predicted = [1, 3, 5, 7, 9, 2, 4, 6, 8, 10]
    ground_truth = [1, 2, 3]

    print(f"Predicted ranking: {predicted}")
    print(f"Ground truth: {ground_truth}")

    # Calculate individual metrics
    print(f"\nNDCG@5: {ndcg_at_k(predicted, ground_truth, 5):.4f}")
    print(f"HR@5: {hit_rate_at_k(predicted, ground_truth, 5):.4f}")
    print(f"MRR@5: {mrr_at_k(predicted, ground_truth, 5):.4f}")
    print(f"P@5: {precision_at_k(predicted, ground_truth, 5):.4f}")
    print(f"R@5: {recall_at_k(predicted, ground_truth, 5):.4f}")
    print(f"MAP@5: {map_at_k(predicted, ground_truth, 5):.4f}")

    # Batch evaluation
    predicted_rankings = [
        [1, 3, 5, 7, 9],
        [2, 4, 6, 8, 10],
        [1, 2, 3, 4, 5]
    ]
    ground_truths = [
        [1, 2, 3],
        [2, 5, 7],
        [3, 6, 9]
    ]

    metrics = evaluate_ranking(predicted_rankings, ground_truths, k_values=[3, 5])
    print_metrics(metrics, "Batch Evaluation")
