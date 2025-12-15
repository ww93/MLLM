"""Retriever MoE Block with User Memory Mechanism.

Extends the MoE retriever with a user-local memory system that:
1. Stores dynamic user preference representations
2. Updates memory based on configurable triggers (interaction count, time, drift)
3. Persists memory state across sessions
4. Captures evolving user preferences over time
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .retriever_moe import RetrieverMoEBlock, CrossModalExpert


class UpdateTrigger(Enum):
    """Memory update trigger types."""
    INTERACTION_COUNT = "interaction_count"  # Update after N interactions
    DRIFT_THRESHOLD = "drift_threshold"      # Update when preference drifts
    TIME_BASED = "time_based"                 # Update periodically
    EXPLICIT = "explicit"                     # Manual trigger only


@dataclass
class MemoryConfig:
    """Configuration for memory mechanism."""
    memory_dim: int = 256
    max_memory_size: int = 10  # Number of historical states to keep
    update_trigger: UpdateTrigger = UpdateTrigger.INTERACTION_COUNT
    interaction_threshold: int = 10  # Update after N interactions
    drift_threshold: float = 0.3     # Cosine similarity threshold
    decay_factor: float = 0.95       # Exponential decay for old memories
    enable_persistence: bool = True  # Save/load memory from disk


@dataclass
class UserMemory:
    """User memory state."""
    user_id: int
    memory_vector: torch.Tensor  # Current memory representation
    memory_history: List[torch.Tensor]  # Historical states
    interaction_count: int = 0
    last_update_step: int = 0
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class RetrieverMoEMemory(nn.Module):
    """MoE Retriever with user-local memory mechanism.

    This module extends RetrieverMoEBlock with:
    - Per-user memory storage for dynamic preferences
    - Configurable update triggers (interaction count, drift detection, etc.)
    - Memory persistence across sessions
    - Adaptive memory updates based on user behavior
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_proxies: int = 4,
        memory_config: Optional[MemoryConfig] = None,
    ) -> None:
        """
        Args:
            embedding_dim: Embedding dimension for all modalities
            num_heads: Number of attention heads
            dropout: Dropout rate
            num_proxies: Number of proxy tokens for self-attention
            memory_config: Configuration for memory mechanism
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.memory_config = memory_config or MemoryConfig(memory_dim=embedding_dim)

        # Core MoE retriever block
        self.moe_block = RetrieverMoEBlock(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            num_proxies=num_proxies,
        )

        # Memory-related components
        self.user_memories: Dict[int, UserMemory] = {}
        self.global_step = 0

        # Memory integration module
        self.memory_gate = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.Sigmoid()
        )

        # Memory update network
        self.memory_update = nn.GRUCell(embedding_dim, self.memory_config.memory_dim)

        # Memory projection (if memory_dim != embedding_dim)
        if self.memory_config.memory_dim != embedding_dim:
            self.memory_projection = nn.Linear(self.memory_config.memory_dim, embedding_dim)
        else:
            self.memory_projection = nn.Identity()

        # Drift detector
        self.drift_detector = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, 1),
            nn.Sigmoid()
        )

    def _initialize_memory(self, user_id: int) -> UserMemory:
        """Initialize memory for a new user."""
        memory_vector = torch.zeros(
            self.memory_config.memory_dim,
            device=next(self.parameters()).device
        )
        return UserMemory(
            user_id=user_id,
            memory_vector=memory_vector,
            memory_history=[],
            interaction_count=0,
            last_update_step=self.global_step,
            metadata={}
        )

    def _get_user_memory(self, user_id: int) -> UserMemory:
        """Get or create user memory."""
        if user_id not in self.user_memories:
            self.user_memories[user_id] = self._initialize_memory(user_id)
        return self.user_memories[user_id]

    def _should_update_memory(
        self,
        user_memory: UserMemory,
        current_repr: torch.Tensor
    ) -> bool:
        """Determine if memory should be updated based on configured trigger."""
        trigger = self.memory_config.update_trigger

        if trigger == UpdateTrigger.EXPLICIT:
            return False  # Only update when explicitly called

        if trigger == UpdateTrigger.INTERACTION_COUNT:
            return user_memory.interaction_count >= self.memory_config.interaction_threshold

        if trigger == UpdateTrigger.DRIFT_THRESHOLD:
            # Calculate drift using cosine similarity
            memory_proj = self.memory_projection(user_memory.memory_vector)
            similarity = F.cosine_similarity(
                memory_proj.unsqueeze(0),
                current_repr.unsqueeze(0),
                dim=1
            ).item()
            return similarity < (1.0 - self.memory_config.drift_threshold)

        if trigger == UpdateTrigger.TIME_BASED:
            steps_since_update = self.global_step - user_memory.last_update_step
            return steps_since_update >= self.memory_config.interaction_threshold

        return False

    def _update_memory(
        self,
        user_memory: UserMemory,
        current_repr: torch.Tensor,
        force: bool = False
    ) -> None:
        """Update user memory with current representation."""
        # Store current memory in history
        if len(user_memory.memory_history) >= self.memory_config.max_memory_size:
            user_memory.memory_history.pop(0)
        user_memory.memory_history.append(user_memory.memory_vector.clone())

        # Update memory using GRU
        new_memory = self.memory_update(
            current_repr.unsqueeze(0),
            user_memory.memory_vector.unsqueeze(0)
        ).squeeze(0)

        # Apply decay to old memory
        if not force:
            decay = self.memory_config.decay_factor
            user_memory.memory_vector = decay * user_memory.memory_vector + (1 - decay) * new_memory
        else:
            user_memory.memory_vector = new_memory

        # Update metadata
        user_memory.interaction_count = 0
        user_memory.last_update_step = self.global_step
        user_memory.metadata['last_update'] = self.global_step

    def _integrate_memory(
        self,
        current_repr: torch.Tensor,
        memory_repr: torch.Tensor
    ) -> torch.Tensor:
        """Integrate current representation with memory using gating."""
        # Project memory if needed
        memory_proj = self.memory_projection(memory_repr)

        # Compute gate
        combined = torch.cat([current_repr, memory_proj], dim=-1)
        gate = self.memory_gate(combined)

        # Gated fusion
        integrated = gate * current_repr + (1 - gate) * memory_proj
        return integrated

    def forward(
        self,
        user_ids: List[int],
        user_pref: torch.Tensor,
        item_desc: torch.Tensor,
        item_image: torch.Tensor,
        target_item: torch.Tensor,
        user_pref_mask: Optional[torch.Tensor] = None,
        item_desc_mask: Optional[torch.Tensor] = None,
        item_image_mask: Optional[torch.Tensor] = None,
        update_memory: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Forward pass with memory integration.

        Args:
            user_ids: List of user IDs for memory lookup
            user_pref: [batch, seq, dim] user preference embeddings
            item_desc: [batch, seq, dim] item description embeddings
            item_image: [batch, seq, dim] item image embeddings
            target_item: [batch, dim] target item embedding
            user_pref_mask: Optional mask for user preferences
            item_desc_mask: Optional mask for item descriptions
            item_image_mask: Optional mask for item images
            update_memory: Whether to update memory (disable during inference)

        Returns:
            scores: [batch] scalar scores for each sample
            routing_weights: [batch, num_experts] expert weights
            memory_info: Dictionary with memory-related information
        """
        batch_size = target_item.size(0)
        device = target_item.device

        # Get base MoE scores
        base_scores, routing_weights = self.moe_block(
            user_pref, item_desc, item_image, target_item,
            user_pref_mask, item_desc_mask, item_image_mask
        )

        # Process each user's memory
        memory_enhanced_scores = []
        memory_info = {
            'memory_used': [],
            'memory_updated': [],
            'drift_scores': []
        }

        for i, user_id in enumerate(user_ids):
            # Get user memory
            user_memory = self._get_user_memory(user_id)

            # Extract current representation (use refined output from MoE)
            # We need to compute the refined representation
            query = target_item[i:i+1].unsqueeze(1)
            expert_outputs = [
                self.moe_block.experts[0](query, user_pref[i:i+1], None if user_pref_mask is None else user_pref_mask[i:i+1]),
                self.moe_block.experts[1](query, item_desc[i:i+1], None if item_desc_mask is None else item_desc_mask[i:i+1]),
                self.moe_block.experts[2](query, item_image[i:i+1], None if item_image_mask is None else item_image_mask[i:i+1]),
            ]

            stacked = torch.stack(expert_outputs, dim=2)
            mixture = (stacked * routing_weights[i:i+1].unsqueeze(1).unsqueeze(-1)).sum(dim=2)

            # Self-attention with proxies
            proxy_tokens = self.moe_block.proxy_tokens.unsqueeze(0)
            attn_input = torch.cat([mixture, proxy_tokens], dim=1)
            attn_output, _ = self.moe_block.self_attn(mixture, attn_input, attn_input)
            current_repr = self.moe_block.output_norm(mixture + self.moe_block.dropout(attn_output))
            current_repr = current_repr.squeeze(1)

            # Check if memory exists and integrate
            memory_used = False
            if user_memory.memory_vector.abs().sum() > 0:  # Memory is initialized
                memory_integrated = self._integrate_memory(
                    current_repr, user_memory.memory_vector
                )
                memory_used = True
            else:
                memory_integrated = current_repr

            # Normalize vectors before computing similarity score
            memory_normalized = F.normalize(memory_integrated, p=2, dim=-1)
            target_normalized = F.normalize(target_item[i], p=2, dim=-1)

            # Compute enhanced score with memory
            enhanced_score = (memory_normalized * target_normalized).sum(dim=-1).squeeze()
            memory_enhanced_scores.append(enhanced_score)

            # Update memory if needed
            memory_updated = False
            if update_memory:
                user_memory.interaction_count += 1
                if self._should_update_memory(user_memory, current_repr):
                    self._update_memory(user_memory, current_repr)
                    memory_updated = True

            # Record memory info
            memory_info['memory_used'].append(memory_used)
            memory_info['memory_updated'].append(memory_updated)

            # Compute drift score for monitoring
            if memory_used:
                memory_proj = self.memory_projection(user_memory.memory_vector)
                drift_input = torch.cat([current_repr, memory_proj], dim=-1)
                drift_score = self.drift_detector(drift_input).item()
                memory_info['drift_scores'].append(drift_score)
            else:
                memory_info['drift_scores'].append(0.0)

        # Stack enhanced scores
        enhanced_scores = torch.stack(memory_enhanced_scores)

        # Increment global step
        self.global_step += 1

        return enhanced_scores, routing_weights, memory_info

    def explicit_update_memory(self, user_id: int, force_reset: bool = False) -> None:
        """Explicitly trigger memory update for a user.

        Args:
            user_id: User ID to update
            force_reset: If True, completely reset memory instead of gradual update
        """
        if user_id not in self.user_memories:
            return

        user_memory = self.user_memories[user_id]

        if force_reset:
            # Reset memory
            self.user_memories[user_id] = self._initialize_memory(user_id)
        else:
            # Mark for update on next forward pass
            user_memory.interaction_count = self.memory_config.interaction_threshold

    def save_memories(self, save_path: str) -> None:
        """Save all user memories to disk."""
        if not self.memory_config.enable_persistence:
            return

        save_data = {}
        for user_id, memory in self.user_memories.items():
            save_data[str(user_id)] = {
                'memory_vector': memory.memory_vector.cpu().tolist(),
                'memory_history': [m.cpu().tolist() for m in memory.memory_history],
                'interaction_count': memory.interaction_count,
                'last_update_step': memory.last_update_step,
                'metadata': memory.metadata
            }

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(save_data, f)

    def load_memories(self, load_path: str) -> None:
        """Load user memories from disk."""
        if not Path(load_path).exists():
            print(f"Warning: Memory file not found: {load_path}")
            return

        with open(load_path, 'r') as f:
            save_data = json.load(f)

        device = next(self.parameters()).device

        for user_id_str, data in save_data.items():
            # Handle both int and float format strings (e.g., '487' or '487.0')
            user_id = int(float(user_id_str))
            memory_vector = torch.tensor(data['memory_vector'], device=device)
            memory_history = [torch.tensor(m, device=device) for m in data['memory_history']]

            self.user_memories[user_id] = UserMemory(
                user_id=user_id,
                memory_vector=memory_vector,
                memory_history=memory_history,
                interaction_count=data['interaction_count'],
                last_update_step=data['last_update_step'],
                metadata=data['metadata']
            )

        print(f"Loaded memories for {len(self.user_memories)} users from {load_path}")

    def get_memory_stats(self) -> Dict:
        """Get statistics about memory usage."""
        if not self.user_memories:
            return {
                'num_users': 0,
                'avg_interactions': 0,
                'avg_memory_history_size': 0
            }

        return {
            'num_users': len(self.user_memories),
            'avg_interactions': sum(m.interaction_count for m in self.user_memories.values()) / len(self.user_memories),
            'avg_memory_history_size': sum(len(m.memory_history) for m in self.user_memories.values()) / len(self.user_memories),
            'global_step': self.global_step
        }


__all__ = ["RetrieverMoEMemory", "MemoryConfig", "UpdateTrigger", "UserMemory"]
