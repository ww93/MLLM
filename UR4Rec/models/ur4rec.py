"""
UR4Rec: User preference Retrieval for Recommendation.

Main framework that integrates the User Preference Retriever with LLM-based reranking.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Union
import logging

from .user_preference_retriever import UserPreferenceRetriever
from .llm_reranker import LLMReranker


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UR4Rec(nn.Module):
    """
    UR4Rec Framework.

    Enhances reranking for recommendation with LLMs through user preference retrieval.
    """

    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 100,
        llm_backend: str = "openai",
        llm_model_name: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        item_metadata: Optional[Dict] = None,
        use_llm: bool = True,
        temperature: float = 0.3
    ):
        """
        Args:
            num_items: Total number of items in catalog
            embedding_dim: Dimension of item embeddings
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            d_ff: Feed-forward network dimension
            dropout: Dropout rate
            max_seq_len: Maximum sequence length
            llm_backend: LLM backend ('openai', 'anthropic', 'local')
            llm_model_name: Name of LLM model
            llm_api_key: API key for LLM
            item_metadata: Item metadata for LLM prompts
            use_llm: Whether to use LLM reranking (if False, uses retriever scores only)
            temperature: Temperature for LLM generation
        """
        super().__init__()

        self.num_items = num_items
        self.use_llm = use_llm
        self.temperature = temperature

        # User Preference Retriever
        self.preference_retriever = UserPreferenceRetriever(
            num_items=num_items,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
            max_seq_len=max_seq_len
        )

        # LLM Reranker (optional)
        self.llm_reranker = None
        if use_llm:
            try:
                self.llm_reranker = LLMReranker(
                    llm_backend=llm_backend,
                    model_name=llm_model_name,
                    api_key=llm_api_key,
                    item_metadata=item_metadata
                )
                logger.info(f"LLM reranker initialized with backend: {llm_backend}")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM reranker: {e}")
                logger.warning("Falling back to retriever-only mode")
                self.use_llm = False

    def forward(
        self,
        user_history: torch.Tensor,
        candidates: torch.Tensor,
        history_mask: Optional[torch.Tensor] = None,
        candidate_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: retrieve user preferences and compute scores.

        Args:
            user_history: [batch_size, history_len] - User interaction history
            candidates: [batch_size, num_candidates] - Candidate items
            history_mask: [batch_size, history_len] - History padding mask
            candidate_mask: [batch_size, num_candidates] - Candidate padding mask

        Returns:
            preference_scores: [batch_size, num_candidates] - Preference scores
            rankings: [batch_size, num_candidates] - Item rankings (indices)
        """
        # Get preference scores from retriever
        preference_scores, candidate_embeds, history_repr = self.preference_retriever(
            user_history, candidates, history_mask, candidate_mask
        )

        # Apply softmax to get probabilities
        if candidate_mask is not None:
            masked_scores = preference_scores.masked_fill(~candidate_mask, -1e9)
        else:
            masked_scores = preference_scores

        # Rank by preference scores
        rankings = torch.argsort(masked_scores, dim=1, descending=True)

        return preference_scores, rankings

    def rerank_with_llm(
        self,
        user_history: Union[List[int], torch.Tensor],
        candidates: Union[List[int], torch.Tensor],
        preference_scores: torch.Tensor,
        temperature: Optional[float] = None
    ) -> List[int]:
        """
        Rerank candidates using LLM.

        Args:
            user_history: User interaction history
            candidates: Candidate items
            preference_scores: Scores from preference retriever
            temperature: LLM temperature (uses default if None)

        Returns:
            Reranked list of item IDs
        """
        if not self.use_llm or self.llm_reranker is None:
            # Fall back to retriever scores
            logger.warning("LLM not available, using retriever scores only")
            if isinstance(candidates, torch.Tensor):
                candidates = candidates.tolist()
            if isinstance(preference_scores, torch.Tensor):
                scores = preference_scores.tolist()
            else:
                scores = preference_scores

            # Sort by scores
            ranked_pairs = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
            return [item for item, _ in ranked_pairs]

        temp = temperature if temperature is not None else self.temperature

        return self.llm_reranker.rerank(
            user_history, candidates, preference_scores, temperature=temp
        )

    def predict(
        self,
        user_history: torch.Tensor,
        candidates: torch.Tensor,
        history_mask: Optional[torch.Tensor] = None,
        candidate_mask: Optional[torch.Tensor] = None,
        use_llm: Optional[bool] = None
    ) -> List[List[int]]:
        """
        Make predictions for a batch.

        Args:
            user_history: [batch_size, history_len]
            candidates: [batch_size, num_candidates]
            history_mask: [batch_size, history_len]
            candidate_mask: [batch_size, num_candidates]
            use_llm: Override whether to use LLM

        Returns:
            List of ranked item lists for each sample in batch
        """
        # Get preference scores
        preference_scores, rankings = self.forward(
            user_history, candidates, history_mask, candidate_mask
        )

        # Decide whether to use LLM
        should_use_llm = use_llm if use_llm is not None else self.use_llm

        if should_use_llm and self.llm_reranker is not None:
            # Rerank with LLM
            batch_size = user_history.size(0)
            results = []

            for i in range(batch_size):
                history = user_history[i]
                cands = candidates[i]
                scores = preference_scores[i]

                # Remove padding
                if history_mask is not None:
                    history = history[history_mask[i]]
                if candidate_mask is not None:
                    cands = cands[candidate_mask[i]]
                    scores = scores[candidate_mask[i]]

                # Rerank with LLM
                ranked = self.rerank_with_llm(history, cands, scores)
                results.append(ranked)

            return results
        else:
            # Use retriever rankings directly
            results = []
            batch_size = candidates.size(0)

            for i in range(batch_size):
                cands = candidates[i]
                ranking = rankings[i]

                # Remove padding
                if candidate_mask is not None:
                    mask = candidate_mask[i]
                    valid_cands = cands[mask]
                    valid_ranking = ranking[mask]
                else:
                    valid_cands = cands
                    valid_ranking = ranking

                # Get ranked items
                ranked_items = valid_cands[valid_ranking].tolist()
                results.append(ranked_items)

            return results

    def compute_loss(
        self,
        user_history: torch.Tensor,
        candidates: torch.Tensor,
        ground_truth: List[List[int]],
        history_mask: Optional[torch.Tensor] = None,
        candidate_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute training loss.

        Uses a ranking loss based on preference scores and ground truth.

        Args:
            user_history: [batch_size, history_len]
            candidates: [batch_size, num_candidates]
            ground_truth: List of relevant items for each sample
            history_mask: [batch_size, history_len]
            candidate_mask: [batch_size, num_candidates]

        Returns:
            loss: Scalar loss tensor
        """
        # Get preference scores
        preference_scores, _ = self.forward(
            user_history, candidates, history_mask, candidate_mask
        )

        batch_size = candidates.size(0)
        num_candidates = candidates.size(1)

        # Create labels: 1 for relevant items, 0 for others
        labels = torch.zeros_like(preference_scores)

        for i in range(batch_size):
            cands = candidates[i].tolist()
            gt = ground_truth[i]

            for j, cand in enumerate(cands):
                if cand in gt:
                    labels[i, j] = 1.0

        # Binary cross-entropy loss
        loss = F.binary_cross_entropy_with_logits(
            preference_scores,
            labels,
            reduction='none'
        )

        # Apply mask if provided
        if candidate_mask is not None:
            loss = loss * candidate_mask.float()
            loss = loss.sum() / candidate_mask.sum()
        else:
            loss = loss.mean()

        return loss

    def save_retriever(self, path: str):
        """Save preference retriever weights."""
        torch.save(self.preference_retriever.state_dict(), path)
        logger.info(f"Preference retriever saved to {path}")

    def load_retriever(self, path: str):
        """Load preference retriever weights."""
        self.preference_retriever.load_state_dict(torch.load(path))
        logger.info(f"Preference retriever loaded from {path}")


if __name__ == "__main__":
    print("Testing UR4Rec Framework...")

    # Model parameters
    num_items = 10000
    batch_size = 2
    history_len = 15
    num_candidates = 8

    # Create model (without LLM for testing)
    model = UR4Rec(
        num_items=num_items,
        embedding_dim=128,
        num_layers=2,
        num_heads=4,
        use_llm=False
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Sample data
    user_history = torch.randint(1, num_items, (batch_size, history_len))
    candidates = torch.randint(1, num_items, (batch_size, num_candidates))
    history_mask = torch.ones(batch_size, history_len, dtype=torch.bool)
    candidate_mask = torch.ones(batch_size, num_candidates, dtype=torch.bool)
    ground_truth = [
        [candidates[0, 0].item(), candidates[0, 2].item()],
        [candidates[1, 1].item()]
    ]

    print(f"\nInput shapes:")
    print(f"  User history: {user_history.shape}")
    print(f"  Candidates: {candidates.shape}")

    # Forward pass
    with torch.no_grad():
        preference_scores, rankings = model(
            user_history, candidates, history_mask, candidate_mask
        )

    print(f"\nOutput shapes:")
    print(f"  Preference scores: {preference_scores.shape}")
    print(f"  Rankings: {rankings.shape}")

    print(f"\nSample preference scores:")
    print(preference_scores[0])

    print(f"\nSample rankings (indices):")
    print(rankings[0])

    # Compute loss
    loss = model.compute_loss(
        user_history, candidates, ground_truth, history_mask, candidate_mask
    )
    print(f"\nLoss: {loss.item():.4f}")

    # Make predictions
    predictions = model.predict(user_history, candidates, history_mask, candidate_mask)
    print(f"\nPredictions:")
    for i, pred in enumerate(predictions):
        print(f"  Sample {i}: {pred[:5]}...")  # First 5 items
