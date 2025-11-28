"""
User Preference Retriever Model.

A transformer-based model that retrieves user preferences from behavior sequences
to provide essential knowledge for LLM-based reranking.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: [batch_size, seq_len, d_model]
            key: [batch_size, seq_len, d_model]
            value: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len]

        Returns:
            output: [batch_size, seq_len, d_model]
            attention_weights: [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size = query.size(0)

        # Linear projections
        Q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_linear(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_linear(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply mask
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
            scores = scores.masked_fill(mask == 0, -1e9)

        # Attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        context = torch.matmul(attention_weights, V)

        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # Final linear projection
        output = self.out_linear(context)

        return output, attention_weights


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len]

        Returns:
            output: [batch_size, seq_len, d_model]
        """
        # Self-attention with residual connection
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class UserPreferenceRetriever(nn.Module):
    """
    Transformer-based User Preference Retriever.

    This model processes user behavior sequences and candidate items
    to retrieve relevant user preferences for LLM-based reranking.
    """

    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 100
    ):
        """
        Args:
            num_items: Total number of items in the catalog
            embedding_dim: Dimension of item embeddings
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            d_ff: Dimension of feed-forward network
            dropout: Dropout rate
            max_seq_len: Maximum sequence length
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_items = num_items

        # Item embeddings
        self.item_embedding = nn.Embedding(
            num_items + 1,  # +1 for padding token
            embedding_dim,
            padding_idx=0
        )

        # Positional encoding
        self.pos_encoding = PositionalEncoding(embedding_dim, max_seq_len, dropout)

        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(embedding_dim, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Cross-attention for candidate items
        self.cross_attention = MultiHeadAttention(embedding_dim, num_heads, dropout)

        # Output projection for preference scores
        self.preference_scorer = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, 1)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(
        self,
        user_history: torch.Tensor,
        candidates: torch.Tensor,
        history_mask: Optional[torch.Tensor] = None,
        candidate_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the User Preference Retriever.

        Args:
            user_history: [batch_size, history_len] - User's historical interactions
            candidates: [batch_size, num_candidates] - Candidate items to rerank
            history_mask: [batch_size, history_len] - Mask for user history
            candidate_mask: [batch_size, num_candidates] - Mask for candidates

        Returns:
            preference_scores: [batch_size, num_candidates] - Preference scores for candidates
            candidate_embeddings: [batch_size, num_candidates, embedding_dim] - Candidate embeddings
            history_representation: [batch_size, embedding_dim] - Aggregated history representation
        """
        batch_size = user_history.size(0)

        # Embed user history
        history_embeds = self.item_embedding(user_history)  # [batch_size, history_len, embedding_dim]

        # Add positional encoding
        history_embeds = history_embeds.transpose(0, 1)  # [history_len, batch_size, embedding_dim]
        history_embeds = self.pos_encoding(history_embeds)
        history_embeds = history_embeds.transpose(0, 1)  # [batch_size, history_len, embedding_dim]

        # Apply transformer encoder layers
        encoded_history = history_embeds
        for layer in self.encoder_layers:
            encoded_history = layer(encoded_history, history_mask)

        # Aggregate history representation (mean pooling with mask)
        if history_mask is not None:
            mask_expanded = history_mask.unsqueeze(-1).float()
            sum_embeddings = (encoded_history * mask_expanded).sum(dim=1)
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
            history_representation = sum_embeddings / sum_mask
        else:
            history_representation = encoded_history.mean(dim=1)

        # Embed candidate items
        candidate_embeds = self.item_embedding(candidates)  # [batch_size, num_candidates, embedding_dim]

        # Cross-attention: candidates attend to user history
        attended_candidates, attention_weights = self.cross_attention(
            query=candidate_embeds,
            key=encoded_history,
            value=encoded_history,
            mask=history_mask
        )

        # Compute preference scores
        preference_scores = self.preference_scorer(attended_candidates).squeeze(-1)  # [batch_size, num_candidates]

        # Apply candidate mask if provided
        if candidate_mask is not None:
            preference_scores = preference_scores.masked_fill(~candidate_mask, -1e9)

        return preference_scores, candidate_embeds, history_representation

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        """
        Get embeddings for specific items.

        Args:
            item_ids: [batch_size, num_items]

        Returns:
            embeddings: [batch_size, num_items, embedding_dim]
        """
        return self.item_embedding(item_ids)


if __name__ == "__main__":
    # Test the model
    print("Testing User Preference Retriever...")

    # Model parameters
    num_items = 10000
    batch_size = 4
    history_len = 20
    num_candidates = 10

    # Create model
    model = UserPreferenceRetriever(
        num_items=num_items,
        embedding_dim=256,
        num_layers=4,
        num_heads=8
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create sample data
    user_history = torch.randint(1, num_items, (batch_size, history_len))
    candidates = torch.randint(1, num_items, (batch_size, num_candidates))
    history_mask = torch.ones(batch_size, history_len, dtype=torch.bool)
    candidate_mask = torch.ones(batch_size, num_candidates, dtype=torch.bool)

    # Forward pass
    with torch.no_grad():
        preference_scores, candidate_embeds, history_repr = model(
            user_history, candidates, history_mask, candidate_mask
        )

    print(f"\nInput shapes:")
    print(f"  User history: {user_history.shape}")
    print(f"  Candidates: {candidates.shape}")

    print(f"\nOutput shapes:")
    print(f"  Preference scores: {preference_scores.shape}")
    print(f"  Candidate embeddings: {candidate_embeds.shape}")
    print(f"  History representation: {history_repr.shape}")

    print(f"\nSample preference scores:")
    print(preference_scores[0])
