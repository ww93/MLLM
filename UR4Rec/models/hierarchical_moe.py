"""
Hierarchical Mixture-of-Experts for Multi-modal Retrieval.

Architecture:
- Level 1: Within each modality, 3 sub-experts learn different aspects
  - User Preference MoE: 3 experts for genre/mood/style
  - Item Description MoE: 3 experts for content/theme/quality
  - Image Feature MoE: 3 experts for visual aspects
- Level 2: Cross-modality fusion with learned routing
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
        self.attention = nn.MultiheadAttention(
            embedding_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            query: [batch, 1, dim] target item embedding
            context: [batch, seq, dim] modality tokens
            context_mask: [batch, seq] optional padding mask

        Returns:
            fused: [batch, 1, dim] expert-enhanced representation
        """
        attn_output, _ = self.attention(
            query, context, context,
            key_padding_mask=None if context_mask is None else ~context_mask
        )
        fused = self.feed_forward(query + attn_output)
        return fused


class ModalityMoE(nn.Module):
    """
    Modality-specific Mixture of Experts.

    Each modality has 3 sub-experts that learn different aspects:
    - User Preference: genre preferences, mood preferences, style preferences
    - Item Description: content understanding, thematic analysis, quality assessment
    - Image Features: visual composition, color/texture, object recognition
    """

    def __init__(
        self,
        embedding_dim: int,
        num_sub_experts: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()

        self.num_sub_experts = num_sub_experts
        self.embedding_dim = embedding_dim

        # Sub-experts for this modality
        self.sub_experts = nn.ModuleList([
            CrossModalExpert(embedding_dim, num_heads, dropout)
            for _ in range(num_sub_experts)
        ])

        # Router for sub-experts (within modality)
        self.sub_router = nn.Sequential(
            nn.LayerNorm(embedding_dim * 2),  # query + context pooled
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, num_sub_experts)
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def _pool_context(
        self,
        context: torch.Tensor,
        mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Pool context sequence to single vector."""
        if mask is None:
            return context.mean(dim=1)
        masked = context * mask.unsqueeze(-1).float()
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        return masked.sum(dim=1) / denom

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: [batch, 1, dim] target item
            context: [batch, seq, dim] modality features
            context_mask: [batch, seq] optional mask

        Returns:
            output: [batch, 1, dim] modality-enhanced representation
            routing_weights: [batch, num_sub_experts] sub-expert weights
        """
        batch_size = query.size(0)

        # Compute routing weights for sub-experts
        context_pooled = self._pool_context(context, context_mask)
        router_input = torch.cat([query.squeeze(1), context_pooled], dim=-1)
        routing_logits = self.sub_router(router_input)
        routing_weights = F.softmax(routing_logits, dim=-1)  # [B, num_sub_experts]

        # Apply each sub-expert
        expert_outputs = []
        for expert in self.sub_experts:
            expert_out = expert(query, context, context_mask)
            expert_outputs.append(expert_out)

        # Weighted combination of sub-experts
        stacked = torch.stack(expert_outputs, dim=2)  # [B, 1, num_sub_experts, D]
        mixture = (stacked * routing_weights.unsqueeze(1).unsqueeze(-1)).sum(dim=2)  # [B, 1, D]

        # Output projection
        output = self.output_proj(mixture)

        return output, routing_weights


class HierarchicalRetrieverMoE(nn.Module):
    """
    Hierarchical MoE for multi-modal retrieval.

    Architecture:
    1. Level 1 (Within-modality): Each of 3 modalities has 3 sub-experts
       - User Preference MoE: 3 experts
       - Item Description MoE: 3 experts
       - Image Feature MoE: 3 experts
    2. Level 2 (Cross-modality): Learn to fuse 3 modality outputs
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_proxies: int = 8,
        num_sub_experts: int = 3
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_sub_experts = num_sub_experts

        # Level 1: Modality-specific MoEs (3 modalities)
        self.user_pref_moe = ModalityMoE(
            embedding_dim, num_sub_experts, num_heads, dropout
        )
        self.item_desc_moe = ModalityMoE(
            embedding_dim, num_sub_experts, num_heads, dropout
        )
        self.image_feat_moe = ModalityMoE(
            embedding_dim, num_sub_experts, num_heads, dropout
        )

        # Level 2: Cross-modality router
        self.cross_modal_router = nn.Sequential(
            nn.LayerNorm(embedding_dim * 4),  # query + 3 modality representations
            nn.Linear(embedding_dim * 4, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 3)  # 3 modalities
        )

        # Self-attention with proxy tokens for final refinement
        self.proxy_tokens = nn.Parameter(torch.randn(num_proxies, embedding_dim))
        self.self_attn = nn.MultiheadAttention(
            embedding_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.output_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def _pool_modality(
        self,
        embeddings: torch.Tensor,
        mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Pool modality embeddings."""
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
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Args:
            user_pref: [batch, seq, dim] user preference embeddings
            item_desc: [batch, seq, dim] item description embeddings
            item_image: [batch, seq, dim] item image embeddings
            target_item: [batch, dim] target item embedding
            *_mask: Optional masks for each modality

        Returns:
            scores: [batch] scalar scores
            routing_info: Dict with routing weights at both levels
        """
        batch_size = target_item.size(0)
        query = target_item.unsqueeze(1)  # [B, 1, D]

        # Level 1: Within-modality MoE
        user_output, user_sub_weights = self.user_pref_moe(
            query, user_pref, user_pref_mask
        )
        desc_output, desc_sub_weights = self.item_desc_moe(
            query, item_desc, item_desc_mask
        )
        image_output, image_sub_weights = self.image_feat_moe(
            query, item_image, item_image_mask
        )

        # Level 2: Cross-modality routing
        modality_outputs = [
            user_output.squeeze(1),
            desc_output.squeeze(1),
            image_output.squeeze(1)
        ]
        router_input = torch.cat([query.squeeze(1)] + modality_outputs, dim=-1)
        cross_modal_logits = self.cross_modal_router(router_input)
        cross_modal_weights = F.softmax(cross_modal_logits, dim=-1)  # [B, 3]

        # Weighted fusion of modalities
        stacked_modalities = torch.stack([
            user_output.squeeze(1),
            desc_output.squeeze(1),
            image_output.squeeze(1)
        ], dim=1)  # [B, 3, D]
        fused = (stacked_modalities * cross_modal_weights.unsqueeze(-1)).sum(dim=1)  # [B, D]
        fused = fused.unsqueeze(1)  # [B, 1, D]

        # Self-attention refinement with proxy tokens
        proxy_tokens = self.proxy_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        attn_input = torch.cat([fused, proxy_tokens], dim=1)
        attn_output, _ = self.self_attn(fused, attn_input, attn_input)
        refined = self.output_norm(fused + self.dropout(attn_output))

        # Compute similarity scores
        refined_normalized = F.normalize(refined.squeeze(1), p=2, dim=-1)
        target_normalized = F.normalize(target_item, p=2, dim=-1)
        scores = (refined_normalized * target_normalized).sum(dim=-1)

        # Return routing information for analysis
        routing_info = {
            'cross_modal_weights': cross_modal_weights,  # [B, 3]
            'user_sub_weights': user_sub_weights,  # [B, num_sub_experts]
            'desc_sub_weights': desc_sub_weights,  # [B, num_sub_experts]
            'image_sub_weights': image_sub_weights,  # [B, num_sub_experts]
        }

        return scores, cross_modal_weights, routing_info


__all__ = ["HierarchicalRetrieverMoE", "ModalityMoE", "CrossModalExpert"]
