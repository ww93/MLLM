"""UR4Rec with MoE Memory: Complete multimodal recommendation system.

This module integrates:
1. Multimodal encoders (text + image)
2. MoE retriever with user memory
3. Dynamic preference tracking
4. Dot-product scoring with target items
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .retriever_moe_memory import RetrieverMoEMemory, MemoryConfig, UpdateTrigger


class MultiModalEncoder(nn.Module):
    """Encoder for multimodal inputs (text + image)."""

    def __init__(
        self,
        text_model_name: str = "all-MiniLM-L6-v2",
        image_model_name: str = "openai/clip-vit-base-patch32",
        output_dim: int = 256,
        freeze_encoders: bool = True
    ):
        super().__init__()
        self.output_dim = output_dim

        # Text encoder
        try:
            from sentence_transformers import SentenceTransformer
            self.text_encoder = SentenceTransformer(text_model_name)
            if freeze_encoders:
                for param in self.text_encoder.parameters():
                    param.requires_grad = False
            text_dim = self.text_encoder.get_sentence_embedding_dimension()
        except ImportError:
            print("ERROR: sentence-transformers not installed!")
            print("Please install: pip install sentence-transformers")
            print("Falling back to random embeddings (NOT for production use)")
            self.text_encoder = None
            text_dim = 384

        # Image encoder (CLIP)
        try:
            from transformers import CLIPVisionModel
            self.image_encoder = CLIPVisionModel.from_pretrained(image_model_name)
            if freeze_encoders:
                for param in self.image_encoder.parameters():
                    param.requires_grad = False
            image_dim = self.image_encoder.config.hidden_size
        except ImportError:
            print("ERROR: transformers not installed!")
            print("Please install: pip install transformers")
            print("Falling back to random embeddings (NOT for production use)")
            self.image_encoder = None
            image_dim = 512

        # Projections
        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

        self.image_projection = nn.Sequential(
            nn.Linear(image_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Encode text descriptions."""
        if self.text_encoder:
            with torch.no_grad():
                embeddings = self.text_encoder.encode(
                    texts, convert_to_tensor=True, show_progress_bar=False
                )
        else:
            # Fallback: Random embeddings (NOT for production!)
            print("WARNING: Using random text embeddings - install sentence-transformers for real inference")
            batch_size = len(texts)
            device = next(self.parameters()).device
            embeddings = torch.randn(batch_size, 384, device=device)

        return self.text_projection(embeddings)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images.

        Args:
            images: [batch, 3, H, W] image tensors
        """
        if self.image_encoder:
            with torch.no_grad():
                outputs = self.image_encoder(pixel_values=images)
                embeddings = outputs.pooler_output
        else:
            # Fallback: Random embeddings (NOT for production!)
            print("WARNING: Using random image embeddings - install transformers for real inference")
            batch_size = images.size(0)
            device = images.device
            embeddings = torch.randn(batch_size, 512, device=device)

        return self.image_projection(embeddings)


class UR4RecMoEMemory(nn.Module):
    """Complete UR4Rec system with MoE and user memory.

    This model:
    1. Encodes multimodal inputs (text descriptions + images)
    2. Uses MoE to fuse: user preferences, item descriptions, item images
    3. Maintains per-user memory for dynamic preference tracking
    4. Computes dot-product scores with target items
    5. Supports memory persistence and adaptive updates
    """

    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 256,
        # Encoder settings
        text_model_name: str = "all-MiniLM-L6-v2",
        image_model_name: str = "openai/clip-vit-base-patch32",
        freeze_encoders: bool = True,
        # MoE settings
        num_heads: int = 8,
        dropout: float = 0.1,
        num_proxies: int = 4,
        # Memory settings
        memory_config: Optional[MemoryConfig] = None,
        # Device
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Args:
            num_items: Total number of items in catalog
            embedding_dim: Dimension for all embeddings
            text_model_name: Pretrained text model name
            image_model_name: Pretrained image model name
            freeze_encoders: Whether to freeze pretrained encoders
            num_heads: Number of attention heads in MoE
            dropout: Dropout rate
            num_proxies: Number of proxy tokens
            memory_config: Configuration for memory mechanism
            device: Device to run on
        """
        super().__init__()

        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.device = device

        # Multimodal encoders
        self.encoder = MultiModalEncoder(
            text_model_name=text_model_name,
            image_model_name=image_model_name,
            output_dim=embedding_dim,
            freeze_encoders=freeze_encoders
        )

        # Item embeddings (learnable, initialized from text/image)
        self.item_embeddings = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        nn.init.normal_(self.item_embeddings.weight, mean=0, std=0.02)

        # MoE retriever with memory
        self.retriever = RetrieverMoEMemory(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            num_proxies=num_proxies,
            memory_config=memory_config or MemoryConfig()
        )

        # User preference aggregator (for history)
        self.history_aggregator = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dim_feedforward=embedding_dim * 4,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=2
        )

        self.to(device)

    def encode_user_history(
        self,
        history_items: torch.Tensor,
        history_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode user interaction history into preference representation.

        Args:
            history_items: [batch, history_len] item IDs
            history_mask: [batch, history_len] mask (True for valid items)

        Returns:
            history_repr: [batch, seq_len, embedding_dim]
        """
        # Get item embeddings
        history_embeds = self.item_embeddings(history_items)

        # Apply transformer encoder
        if history_mask is not None:
            # Create attention mask (True = ignore)
            attn_mask = ~history_mask
            history_repr = self.history_aggregator(
                history_embeds,
                src_key_padding_mask=attn_mask
            )
        else:
            history_repr = self.history_aggregator(history_embeds)

        return history_repr

    def forward(
        self,
        user_ids: List[int],
        history_items: torch.Tensor,
        target_items: torch.Tensor,
        item_descriptions: Optional[List[str]] = None,
        item_images: Optional[torch.Tensor] = None,
        history_mask: Optional[torch.Tensor] = None,
        update_memory: bool = True
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass: compute scores for target items.

        Args:
            user_ids: List of user IDs (for memory lookup)
            history_items: [batch, history_len] user interaction history
            target_items: [batch] or [batch, num_candidates] target item IDs
            item_descriptions: Optional list of item descriptions (batch*num_items)
            item_images: Optional [batch*num_items, 3, H, W] item images
            history_mask: [batch, history_len] mask for history
            update_memory: Whether to update user memory

        Returns:
            scores: [batch] or [batch, num_candidates] preference scores
            info_dict: Dictionary with intermediate results
        """
        batch_size = history_items.size(0)
        device = history_items.device

        # Encode user history as preferences
        user_pref = self.encode_user_history(history_items, history_mask)

        # Handle target items shape
        if target_items.dim() == 1:
            # Single target per user: [batch]
            target_items = target_items.unsqueeze(1)
            num_targets = 1
        else:
            # Multiple targets per user: [batch, num_candidates]
            num_targets = target_items.size(1)

        # Get target item embeddings
        target_embeds = self.item_embeddings(target_items)  # [batch, num_targets, dim]

        # Encode item descriptions (if provided)
        if item_descriptions is not None:
            item_desc_embeds = self.encoder.encode_text(item_descriptions)
            item_desc_embeds = item_desc_embeds.view(batch_size, num_targets, -1)
        else:
            # Use learnable embeddings as fallback
            item_desc_embeds = target_embeds

        # Encode item images (if provided)
        if item_images is not None:
            item_image_embeds = self.encoder.encode_image(item_images)
            item_image_embeds = item_image_embeds.view(batch_size, num_targets, -1)
        else:
            # Use learnable embeddings as fallback
            item_image_embeds = target_embeds

        # Expand for multiple targets
        user_pref_expanded = user_pref.unsqueeze(1).expand(-1, num_targets, -1, -1)
        user_pref_expanded = user_pref_expanded.reshape(batch_size * num_targets, -1, self.embedding_dim)

        item_desc_expanded = item_desc_embeds.reshape(batch_size * num_targets, 1, self.embedding_dim)
        item_image_expanded = item_image_embeds.reshape(batch_size * num_targets, 1, self.embedding_dim)
        target_expanded = target_embeds.reshape(batch_size * num_targets, self.embedding_dim)

        # Expand user_ids for each target
        user_ids_expanded = []
        for user_id in user_ids:
            user_ids_expanded.extend([user_id] * num_targets)

        # Pass through MoE retriever with memory
        scores, routing_weights, memory_info = self.retriever(
            user_ids=user_ids_expanded,
            user_pref=user_pref_expanded,
            item_desc=item_desc_expanded,
            item_image=item_image_expanded,
            target_item=target_expanded,
            update_memory=update_memory
        )

        # Reshape scores back to [batch, num_targets]
        scores = scores.reshape(batch_size, num_targets)

        # If single target, squeeze
        if num_targets == 1:
            scores = scores.squeeze(1)

        # Reshape routing weights
        routing_weights = routing_weights.reshape(batch_size, num_targets, -1)

        info_dict = {
            'routing_weights': routing_weights,
            'memory_info': memory_info,
            'user_pref': user_pref,
            'target_embeds': target_embeds
        }

        return scores, info_dict

    def predict_top_k(
        self,
        user_ids: List[int],
        history_items: torch.Tensor,
        candidate_items: torch.Tensor,
        k: int = 10,
        history_mask: Optional[torch.Tensor] = None,
        item_descriptions: Optional[List[str]] = None,
        item_images: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict top-K items for each user.

        Args:
            user_ids: List of user IDs
            history_items: [batch, history_len] user history
            candidate_items: [batch, num_candidates] candidate item IDs
            k: Number of top items to return
            history_mask: Optional history mask
            item_descriptions: Optional item descriptions
            item_images: Optional item images

        Returns:
            top_k_items: [batch, k] top-K item IDs
            top_k_scores: [batch, k] corresponding scores
        """
        with torch.no_grad():
            scores, _ = self.forward(
                user_ids=user_ids,
                history_items=history_items,
                target_items=candidate_items,
                item_descriptions=item_descriptions,
                item_images=item_images,
                history_mask=history_mask,
                update_memory=False  # Don't update during inference
            )

        # Get top-K
        top_k_scores, top_k_indices = torch.topk(scores, k=min(k, scores.size(1)), dim=1)

        # Map indices back to item IDs
        batch_indices = torch.arange(candidate_items.size(0)).unsqueeze(1).expand_as(top_k_indices)
        top_k_items = candidate_items[batch_indices, top_k_indices]

        return top_k_items, top_k_scores

    def update_item_embeddings_from_text(
        self,
        item_ids: torch.Tensor,
        item_texts: List[str]
    ) -> None:
        """Update item embeddings using text descriptions."""
        text_embeds = self.encoder.encode_text(item_texts)
        with torch.no_grad():
            self.item_embeddings.weight[item_ids] = text_embeds

    def update_item_embeddings_from_images(
        self,
        item_ids: torch.Tensor,
        item_images: torch.Tensor
    ) -> None:
        """Update item embeddings using images."""
        image_embeds = self.encoder.encode_image(item_images)
        with torch.no_grad():
            self.item_embeddings.weight[item_ids] = image_embeds

    def save_model(self, save_path: str, save_memories: bool = True) -> None:
        """Save model weights and optionally user memories."""
        save_dict = {
            'model_state_dict': self.state_dict(),
            'config': {
                'num_items': self.num_items,
                'embedding_dim': self.embedding_dim,
            }
        }
        torch.save(save_dict, save_path)

        if save_memories:
            memory_path = save_path.replace('.pt', '_memories.json')
            self.retriever.save_memories(memory_path)

    def load_model(self, load_path: str, load_memories: bool = True) -> None:
        """Load model weights and optionally user memories."""
        checkpoint = torch.load(load_path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])

        if load_memories:
            memory_path = load_path.replace('.pt', '_memories.json')
            self.retriever.load_memories(memory_path)

    def get_memory_stats(self) -> Dict:
        """Get user memory statistics."""
        return self.retriever.get_memory_stats()


__all__ = ["UR4RecMoEMemory", "MemoryConfig", "UpdateTrigger"]
