"""Unified UR4Rec architecture and losses in a single module.

This module consolidates the UR4Rec architecture from Figure 2 into one file,
including the recommendation backbone, the preference-item matching head, and
preference-item contrastive learning. It keeps the implementation torch-only so
it can be dropped into training scripts without additional dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional mask support."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
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
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = query.size(0)

        Q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_linear(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_linear(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_linear(context)
        return output, attention_weights


class TransformerEncoderLayer(nn.Module):
    """Minimal transformer encoder layer for the preference extractor."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


@dataclass
class UR4RecLossOutput:
    preference_matching_loss: torch.Tensor
    contrastive_loss: torch.Tensor
    total_loss: torch.Tensor


class UR4RecUnified(nn.Module):
    """Self-contained UR4Rec architecture with matching and contrastive losses."""

    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 200,
        contrastive_temperature: float = 0.07,
        contrastive_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_items = num_items
        self.contrastive_temperature = contrastive_temperature
        self.contrastive_weight = contrastive_weight

        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.pos_encoding = PositionalEncoding(embedding_dim, max_len=max_seq_len, dropout=dropout)
        self.encoder_layers = nn.ModuleList(
            [TransformerEncoderLayer(embedding_dim, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        self.cross_attention = MultiHeadAttention(embedding_dim, num_heads, dropout)
        self.matching_projection = nn.Linear(embedding_dim, embedding_dim)
        self.score_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(embedding_dim, 1))
        self.preference_norm = nn.LayerNorm(embedding_dim)
        self.candidate_norm = nn.LayerNorm(embedding_dim)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)

    def encode_history(
        self, user_history: torch.Tensor, history_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        history_embeds = self.item_embedding(user_history)
        history_embeds = history_embeds.transpose(0, 1)
        history_embeds = self.pos_encoding(history_embeds)
        encoded_history = history_embeds.transpose(0, 1)
        for layer in self.encoder_layers:
            encoded_history = layer(encoded_history, history_mask)

        if history_mask is not None:
            mask = history_mask.unsqueeze(-1).float()
            summed = (encoded_history * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1e-6)
            history_repr = summed / denom
        else:
            history_repr = encoded_history.mean(dim=1)
        history_repr = self.preference_norm(history_repr)
        return encoded_history, history_repr

    def forward(
        self,
        user_history: torch.Tensor,
        candidate_items: torch.Tensor,
        history_mask: Optional[torch.Tensor] = None,
        candidate_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded_history, history_repr = self.encode_history(user_history, history_mask)
        candidate_embeddings = self.candidate_norm(self.item_embedding(candidate_items))

        attended_candidates, _ = self.cross_attention(
            query=candidate_embeddings,
            key=encoded_history,
            value=encoded_history,
            mask=history_mask,
        )

        preference_query = self.matching_projection(history_repr).unsqueeze(1)
        match_features = attended_candidates + preference_query
        logits = self.score_head(match_features).squeeze(-1)

        if candidate_mask is not None:
            logits = logits.masked_fill(~candidate_mask, float("-inf"))

        contrastive_logits = torch.matmul(history_repr, candidate_embeddings.transpose(1, 2))
        contrastive_logits = contrastive_logits / self.contrastive_temperature

        return logits, contrastive_logits, candidate_embeddings

    def compute_loss(
        self,
        user_history: torch.Tensor,
        candidate_items: torch.Tensor,
        labels: torch.Tensor,
        history_mask: Optional[torch.Tensor] = None,
        candidate_mask: Optional[torch.Tensor] = None,
    ) -> UR4RecLossOutput:
        logits, contrastive_logits, _ = self.forward(user_history, candidate_items, history_mask, candidate_mask)

        matching_loss = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction="none")
        if candidate_mask is not None:
            matching_loss = (matching_loss * candidate_mask.float()).sum() / candidate_mask.float().sum().clamp(min=1e-6)
        else:
            matching_loss = matching_loss.mean()

        with torch.no_grad():
            positive_indices = labels.argmax(dim=1)
            valid_positive = labels.max(dim=1).values > 0

        if valid_positive.any():
            targets = positive_indices[valid_positive]
            valid_logits = contrastive_logits[valid_positive]
            contrastive_loss = F.cross_entropy(valid_logits, targets)
        else:
            contrastive_loss = torch.zeros((), device=logits.device, dtype=logits.dtype)

        total_loss = matching_loss + self.contrastive_weight * contrastive_loss
        return UR4RecLossOutput(matching_loss, contrastive_loss, total_loss)


__all__ = [
    "UR4RecUnified",
    "UR4RecLossOutput",
]
