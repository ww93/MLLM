"""
UR4Rec V2 with MoE-enhanced Retriever

Combines:
1. SASRec for sequential pattern modeling
2. MoE-enhanced Text Preference Retriever (with user memory)
3. Weighted fusion of both modules
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import json
from pathlib import Path

from .sasrec import SASRec
from .text_preference_retriever_moe import TextPreferenceRetrieverMoE
from .retriever_moe_memory import MemoryConfig, UpdateTrigger


class UR4RecV2MoE(nn.Module):
    """
    UR4Rec V2 with MoE-enhanced Retriever

    Architecture:
    1. Offline stage: LLM generates user preferences and item descriptions
    2. Online stage:
       - SASRec: Sequential pattern learning
       - MoE Retriever: Text-based preference matching with user memory
       - Fusion: Weighted combination of both predictions
    """

    def __init__(
        self,
        num_items: int,
        # SASRec parameters
        sasrec_hidden_dim: int = 256,
        sasrec_num_blocks: int = 2,
        sasrec_num_heads: int = 4,
        sasrec_dropout: float = 0.1,
        # Text retriever parameters
        text_model_name: str = "all-MiniLM-L6-v2",
        text_embedding_dim: int = 384,
        retriever_output_dim: int = 256,
        # MoE parameters
        moe_num_heads: int = 8,
        moe_dropout: float = 0.1,
        moe_num_proxies: int = 4,
        # Memory parameters
        memory_config: Optional[MemoryConfig] = None,
        # Fusion parameters
        fusion_method: str = "weighted",  # 'weighted', 'learned', 'adaptive'
        sasrec_weight: float = 0.5,
        retriever_weight: float = 0.5,
        # Other
        max_seq_len: int = 50,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Args:
            num_items: Total number of items
            sasrec_*: SASRec parameters
            text_*: Text encoder parameters
            retriever_output_dim: Retriever output dimension
            moe_*: MoE parameters
            memory_config: Memory mechanism configuration
            fusion_method: How to fuse SASRec and Retriever scores
            *_weight: Fusion weights (if fusion_method='weighted')
            max_seq_len: Maximum sequence length
            device: Device to run on
        """
        super().__init__()

        self.num_items = num_items
        self.fusion_method = fusion_method
        self.sasrec_weight = sasrec_weight
        self.retriever_weight = retriever_weight
        self.device = device

        # 1. SASRec: Sequential pattern modeling
        self.sasrec = SASRec(
            num_items=num_items,
            hidden_dim=sasrec_hidden_dim,
            num_blocks=sasrec_num_blocks,
            num_heads=sasrec_num_heads,
            dropout=sasrec_dropout,
            max_seq_len=max_seq_len
        )

        # 2. MoE-enhanced Text Preference Retriever
        self.preference_retriever = TextPreferenceRetrieverMoE(
            num_items=num_items,
            text_model_name=text_model_name,
            text_embedding_dim=text_embedding_dim,
            output_dim=retriever_output_dim,
            num_heads=moe_num_heads,
            dropout=moe_dropout,
            num_proxies=moe_num_proxies,
            memory_config=memory_config or MemoryConfig(
                memory_dim=retriever_output_dim,
                max_memory_size=10,
                update_trigger=UpdateTrigger.INTERACTION_COUNT,
                interaction_threshold=10,
                enable_persistence=True
            ),
            device=device
        )

        # 3. Learned fusion (if fusion_method='learned')
        if fusion_method == 'learned':
            self.fusion_layer = nn.Sequential(
                nn.Linear(2, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, 2),
                nn.Softmax(dim=-1)
            )
        elif fusion_method == 'adaptive':
            # Adaptive fusion based on user/item features
            self.fusion_predictor = nn.Sequential(
                nn.Linear(sasrec_hidden_dim + retriever_output_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 2),
                nn.Softmax(dim=-1)
            )

        self.to(device)

    def load_llm_generated_data(
        self,
        user_preferences_path: str,
        item_descriptions_path: str
    ):
        """
        Load LLM-generated user preferences and item descriptions

        Args:
            user_preferences_path: Path to user preferences JSON
            item_descriptions_path: Path to item descriptions JSON
        """
        self.preference_retriever.load_llm_generated_data(
            user_preferences_path=user_preferences_path,
            item_descriptions_path=item_descriptions_path
        )

    def forward(
        self,
        user_ids: List[int],
        input_seq: torch.Tensor,
        target_items: torch.Tensor,
        seq_padding_mask: Optional[torch.Tensor] = None,
        update_memory: bool = False,
        return_components: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Forward pass

        Args:
            user_ids: User IDs [batch_size]
            input_seq: Input sequence [batch_size, seq_len]
            target_items: Target items [batch_size] or [batch_size, num_candidates]
            seq_padding_mask: Padding mask [batch_size, seq_len]
            update_memory: Whether to update user memory
            return_components: Whether to return component scores

        Returns:
            scores: [batch_size] or [batch_size, num_candidates] prediction scores
            (optional) info: Dictionary with component scores and other info
        """
        batch_size = input_seq.size(0)

        # Ensure target_items is 2D
        if target_items.dim() == 1:
            target_items = target_items.unsqueeze(1)  # [B, 1]

        # 1. SASRec prediction
        sasrec_logits = self.sasrec.predict(
            input_seq,
            candidate_items=target_items,
            padding_mask=seq_padding_mask
        )  # [B, num_candidates]

        # 2. MoE Retriever prediction
        retriever_scores, retriever_info = self.preference_retriever(
            user_ids=user_ids,
            item_ids=target_items,
            user_history=input_seq,
            update_memory=update_memory
        )  # [B, num_candidates]

        # 3. Fusion
        if self.fusion_method == 'weighted':
            # Simple weighted sum
            final_scores = (
                self.sasrec_weight * sasrec_logits +
                self.retriever_weight * retriever_scores
            )

        elif self.fusion_method == 'learned':
            # Learned fusion weights
            # Stack scores: [B, N, 2]
            stacked = torch.stack([sasrec_logits, retriever_scores], dim=-1)
            # Get fusion weights: [2]
            weights = self.fusion_layer(torch.tensor([1.0, 1.0], device=self.device))
            # Apply weights
            final_scores = (stacked * weights.unsqueeze(0).unsqueeze(0)).sum(dim=-1)

        elif self.fusion_method == 'adaptive':
            # Adaptive fusion based on representations
            # Get last hidden state from SASRec
            sasrec_repr = self.sasrec.get_sequence_representation(
                input_seq, seq_padding_mask
            )  # [B, D]
            # Get fused representation from retriever
            retriever_repr = retriever_info['fused_repr']  # [B, D]

            # Predict fusion weights
            concat_repr = torch.cat([sasrec_repr, retriever_repr], dim=-1)  # [B, 2D]
            weights = self.fusion_predictor(concat_repr)  # [B, 2]

            # Apply weights
            stacked = torch.stack([sasrec_logits, retriever_scores], dim=-1)  # [B, N, 2]
            final_scores = (stacked * weights.unsqueeze(1)).sum(dim=-1)  # [B, N]

        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

        if return_components:
            info = {
                'sasrec_scores': sasrec_logits,
                'retriever_scores': retriever_scores,
                'final_scores': final_scores,
                'sasrec_weight': self.sasrec_weight,
                'retriever_weight': self.retriever_weight,
                **retriever_info
            }
            return final_scores, info
        else:
            return final_scores

    def get_top_k_recommendations(
        self,
        user_ids: List[int],
        input_seq: torch.Tensor,
        k: int = 10,
        exclude_items: Optional[List[List[int]]] = None,
        seq_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get top-k recommendations for users

        Args:
            user_ids: User IDs
            input_seq: Input sequence [batch_size, seq_len]
            k: Number of recommendations
            exclude_items: Items to exclude for each user
            seq_padding_mask: Padding mask

        Returns:
            top_k_items: [batch_size, k] top-k item IDs
            top_k_scores: [batch_size, k] top-k scores
        """
        batch_size = len(user_ids)

        # Get all items
        all_items = torch.arange(1, self.num_items + 1, device=self.device)
        all_items = all_items.unsqueeze(0).expand(batch_size, -1)  # [B, num_items]

        # Get scores for all items
        scores = self.forward(
            user_ids=user_ids,
            input_seq=input_seq,
            target_items=all_items,
            seq_padding_mask=seq_padding_mask,
            update_memory=False
        )  # [B, num_items]

        # Exclude items
        if exclude_items is not None:
            for i, exclude_list in enumerate(exclude_items):
                if exclude_list:
                    exclude_indices = torch.tensor(exclude_list, device=self.device) - 1
                    scores[i, exclude_indices] = float('-inf')

        # Top-k
        top_k_scores, top_k_indices = torch.topk(scores, k, dim=1)
        top_k_items = all_items.gather(1, top_k_indices)

        return top_k_items, top_k_scores

    def get_memory_stats(self) -> Dict:
        """Get user memory statistics"""
        return self.preference_retriever.get_memory_stats()

    def save_memories(self, save_path: str):
        """Save user memories"""
        self.preference_retriever.save_memories(save_path)

    def load_memories(self, load_path: str):
        """Load user memories"""
        self.preference_retriever.load_memories(load_path)
