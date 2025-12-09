"""Retriever block with Mixture-of-Experts cross-modal reasoning.

The module builds on the Retriever Block shown in Figure 2 by introducing a
Mixture-of-Experts (MoE) design that lets different modalities specialize:
- user preference embeddings
- item description embeddings
- item image embeddings

The experts are combined through a learned router so the final dot-product score
with a target item embedding reflects user affinity.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalExpert(nn.Module):
    """A single expert that fuses a context modality with a target query."""

    def __init__(self, embedding_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )

    def forward(self, query: torch.Tensor, context: torch.Tensor, context_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            query: [batch, 1, dim] target item embedding.
            context: [batch, seq, dim] modality tokens.
            context_mask: [batch, seq] optional padding mask.
        Returns:
            fused: [batch, 1, dim] expert-enhanced representation.
        """
        attn_output, _ = self.attention(query, context, context, key_padding_mask=None if context_mask is None else ~context_mask)
        fused = self.feed_forward(query + attn_output)
        return fused


class RetrieverMoEBlock(nn.Module):
    """Cross-attention MoE retriever that merges three modality experts."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_proxies: int = 4,
    ) -> None:
        super().__init__()
        self.experts = nn.ModuleList(
            [CrossModalExpert(embedding_dim, num_heads, dropout) for _ in range(3)]
        )
        self.router = nn.Sequential(
            nn.LayerNorm(embedding_dim * 4),
            nn.Linear(embedding_dim * 4, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, len(self.experts)),
        )
        self.proxy_tokens = nn.Parameter(torch.randn(num_proxies, embedding_dim))
        self.self_attn = nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout, batch_first=True)
        self.output_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def _expert_pool(self, embeddings: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return embeddings.mean(dim=1)
        masked = embeddings * mask.unsqueeze(-1).float()
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        return masked.sum(dim=1) / denom

    def forward(
        self,
        user_pref: torch.Tensor,
        item_desc: torch.Tensor,
        item_image: torch.Tensor,
        target_item: torch.Tensor,
        user_pref_mask: Optional[torch.Tensor] = None,
        item_desc_mask: Optional[torch.Tensor] = None,
        item_image_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run MoE cross attention and return a scalar score per sample."""
        batch_size = target_item.size(0)
        query = target_item.unsqueeze(1)

        expert_outputs: List[torch.Tensor] = [
            self.experts[0](query, user_pref, user_pref_mask),
            self.experts[1](query, item_desc, item_desc_mask),
            self.experts[2](query, item_image, item_image_mask),
        ]

        pooled_inputs = [
            self._expert_pool(user_pref, user_pref_mask),
            self._expert_pool(item_desc, item_desc_mask),
            self._expert_pool(item_image, item_image_mask),
        ]
        router_input = torch.cat([query.squeeze(1)] + pooled_inputs, dim=-1)
        routing_logits = self.router(router_input)
        routing_weights = F.softmax(routing_logits, dim=-1)

        stacked = torch.stack(expert_outputs, dim=2)  # [B, 1, num_experts, D]
        mixture = (stacked * routing_weights.unsqueeze(1).unsqueeze(-1)).sum(dim=2)

        proxy_tokens = self.proxy_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        attn_input = torch.cat([mixture, proxy_tokens], dim=1)
        attn_output, _ = self.self_attn(mixture, attn_input, attn_input)
        refined = self.output_norm(mixture + self.dropout(attn_output))

        scores = (refined.squeeze(1) * target_item).sum(dim=-1)
        return scores, routing_weights


__all__ = ["RetrieverMoEBlock", "CrossModalExpert"]
