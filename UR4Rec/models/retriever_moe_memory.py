"""
MoE-enhanced Retriever with User Memory

实现带有用户记忆机制的MoE检索器：
- 动态用户记忆存储
- 多专家路由机制
- 跨模态信息融合
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from enum import Enum
import json
from pathlib import Path


class UpdateTrigger(Enum):
    """用户记忆更新触发器"""
    ALWAYS = "always"  # 每次交互都更新
    INTERACTION_COUNT = "interaction_count"  # 达到交互阈值时更新
    TIME_BASED = "time_based"  # 基于时间更新
    NEVER = "never"  # 不更新（仅推理）


class MemoryConfig:
    """用户记忆配置"""

    def __init__(
        self,
        memory_dim: int = 256,
        max_memory_size: int = 10,
        update_trigger: UpdateTrigger = UpdateTrigger.INTERACTION_COUNT,
        interaction_threshold: int = 10,
        enable_persistence: bool = True,
        memory_decay: float = 0.9
    ):
        """
        Args:
            memory_dim: 记忆向量维度
            max_memory_size: 最大记忆条目数
            update_trigger: 更新触发机制
            interaction_threshold: 交互阈值（用于INTERACTION_COUNT模式）
            enable_persistence: 是否启用记忆持久化
            memory_decay: 记忆衰减系数
        """
        self.memory_dim = memory_dim
        self.max_memory_size = max_memory_size
        self.update_trigger = update_trigger
        self.interaction_threshold = interaction_threshold
        self.enable_persistence = enable_persistence
        self.memory_decay = memory_decay


class UserMemoryBank:
    """用户记忆库"""

    def __init__(self, config: MemoryConfig, device: str = "cuda"):
        self.config = config
        self.device = device

        # 存储每个用户的记忆
        self.user_memories: Dict[int, torch.Tensor] = {}  # {user_id: [memory_size, memory_dim]}
        self.user_interaction_counts: Dict[int, int] = {}

    def get_memory(self, user_id: int) -> Optional[torch.Tensor]:
        """获取用户记忆"""
        return self.user_memories.get(user_id, None)

    def update_memory(
        self,
        user_id: int,
        new_memory: torch.Tensor,
        force_update: bool = False
    ):
        """
        更新用户记忆

        Args:
            user_id: 用户ID
            new_memory: 新的记忆向量 [memory_dim]
            force_update: 强制更新
        """
        # 更新交互计数
        self.user_interaction_counts[user_id] = self.user_interaction_counts.get(user_id, 0) + 1

        # 检查是否应该更新
        should_update = force_update
        if not should_update:
            if self.config.update_trigger == UpdateTrigger.ALWAYS:
                should_update = True
            elif self.config.update_trigger == UpdateTrigger.INTERACTION_COUNT:
                should_update = (self.user_interaction_counts[user_id] % self.config.interaction_threshold == 0)

        if not should_update:
            return

        # 更新记忆
        if user_id not in self.user_memories:
            # 创建新记忆
            self.user_memories[user_id] = new_memory.unsqueeze(0)  # [1, memory_dim]
        else:
            # 添加到现有记忆
            current_memory = self.user_memories[user_id]

            # 应用衰减
            if self.config.memory_decay < 1.0:
                current_memory = current_memory * self.config.memory_decay

            # 拼接新记忆
            updated_memory = torch.cat([current_memory, new_memory.unsqueeze(0)], dim=0)

            # 限制记忆大小
            if updated_memory.size(0) > self.config.max_memory_size:
                updated_memory = updated_memory[-self.config.max_memory_size:]

            self.user_memories[user_id] = updated_memory

    def get_stats(self) -> Dict:
        """获取记忆统计信息"""
        return {
            'num_users_with_memory': len(self.user_memories),
            'avg_memory_size': sum(m.size(0) for m in self.user_memories.values()) / max(len(self.user_memories), 1),
            'total_interactions': sum(self.user_interaction_counts.values())
        }

    def save(self, save_path: str):
        """保存记忆到文件"""
        if not self.config.enable_persistence:
            return

        save_dict = {
            'user_memories': {uid: mem.cpu() for uid, mem in self.user_memories.items()},
            'user_interaction_counts': self.user_interaction_counts
        }
        torch.save(save_dict, save_path)
        print(f"User memories saved to {save_path}")

    def load(self, load_path: str):
        """从文件加载记忆"""
        if not Path(load_path).exists():
            print(f"Warning: Memory file not found: {load_path}")
            return

        save_dict = torch.load(load_path, map_location=self.device)
        self.user_memories = {uid: mem.to(self.device) for uid, mem in save_dict['user_memories'].items()}
        self.user_interaction_counts = save_dict['user_interaction_counts']
        print(f"User memories loaded from {load_path}")


class RetrieverMoEMemory(nn.Module):
    """
    MoE-enhanced Retriever with User Memory

    Features:
    1. Multi-expert routing for different modalities
    2. User memory for personalized preference tracking
    3. Cross-modal attention mechanism
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_proxies: int = 4,
        memory_config: Optional[MemoryConfig] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Args:
            embedding_dim: 嵌入维度
            num_heads: 注意力头数
            dropout: Dropout率
            num_proxies: 代理token数量
            memory_config: 记忆配置
            device: 计算设备
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_proxies = num_proxies
        self.device = device

        # Memory configuration
        self.memory_config = memory_config or MemoryConfig(memory_dim=embedding_dim)
        self.memory_bank = UserMemoryBank(self.memory_config, device=device)

        # Expert routing network (gate)
        self.num_experts = 3  # user_pref, item_desc, item_image
        self.router = nn.Sequential(
            nn.Linear(embedding_dim * self.num_experts, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, self.num_experts),
            nn.Softmax(dim=-1)
        )

        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Proxy tokens for aggregation
        self.proxy_tokens = nn.Parameter(torch.randn(num_proxies, embedding_dim))
        nn.init.xavier_uniform_(self.proxy_tokens)

        # Memory integration layer
        self.memory_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.Dropout(dropout)
        )

        self.to(device)

    def forward(
        self,
        user_ids: List[int],
        user_pref: torch.Tensor,  # [B, 1, D] 用户偏好
        item_desc: torch.Tensor,  # [B, 1, D] 物品描述
        item_image: torch.Tensor,  # [B, 1, D] 物品图像
        target_item: torch.Tensor,  # [B, D] 目标物品
        update_memory: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        前向传播

        Args:
            user_ids: 用户ID列表
            user_pref: 用户偏好向量
            item_desc: 物品描述向量
            item_image: 物品图像向量
            target_item: 目标物品向量
            update_memory: 是否更新用户记忆

        Returns:
            scores: [B] 匹配分数
            routing_weights: [B, num_experts] 路由权重
            memory_info: 记忆信息字典
        """
        batch_size = user_pref.size(0)

        # 1. 获取用户记忆
        memory_vectors = []
        for user_id in user_ids:
            user_memory = self.memory_bank.get_memory(user_id)
            if user_memory is not None:
                memory_vectors.append(user_memory)
            else:
                # 如果没有记忆，使用零向量
                memory_vectors.append(torch.zeros(1, self.embedding_dim, device=self.device))

        # 2. Expert routing - 计算每个专家的权重
        # 拼接所有模态信息用于路由决策
        user_pref_flat = user_pref.squeeze(1)  # [B, D]
        item_desc_flat = item_desc.squeeze(1)  # [B, D]
        item_image_flat = item_image.squeeze(1)  # [B, D]

        routing_input = torch.cat([user_pref_flat, item_desc_flat, item_image_flat], dim=-1)  # [B, 3D]
        routing_weights = self.router(routing_input)  # [B, num_experts]

        # 3. 加权融合专家输出
        expert_outputs = torch.stack([user_pref_flat, item_desc_flat, item_image_flat], dim=1)  # [B, 3, D]
        fused_repr = torch.sum(expert_outputs * routing_weights.unsqueeze(-1), dim=1)  # [B, D]

        # 4. 整合用户记忆
        if len(memory_vectors) > 0:
            # 使用注意力机制整合记忆
            max_mem_len = max(m.size(0) for m in memory_vectors)

            # Pad记忆到相同长度
            padded_memories = []
            memory_masks = []
            for mem in memory_vectors:
                if mem.size(0) < max_mem_len:
                    padding = torch.zeros(max_mem_len - mem.size(0), self.embedding_dim, device=self.device)
                    padded_mem = torch.cat([mem, padding], dim=0)
                    mask = torch.cat([
                        torch.zeros(mem.size(0), dtype=torch.bool, device=self.device),
                        torch.ones(max_mem_len - mem.size(0), dtype=torch.bool, device=self.device)
                    ])
                else:
                    padded_mem = mem
                    mask = torch.zeros(max_mem_len, dtype=torch.bool, device=self.device)

                padded_memories.append(padded_mem)
                memory_masks.append(mask)

            memory_tensor = torch.stack(padded_memories, dim=0)  # [B, max_mem_len, D]
            memory_mask = torch.stack(memory_masks, dim=0)  # [B, max_mem_len]

            # 使用记忆增强当前表示
            query = fused_repr.unsqueeze(1)  # [B, 1, D]
            memory_enhanced, _ = self.memory_attention(
                query,
                memory_tensor,
                memory_tensor,
                key_padding_mask=memory_mask
            )  # [B, 1, D]

            fused_repr = fused_repr + memory_enhanced.squeeze(1)  # 残差连接

        # 5. 输出投影
        final_repr = self.output_projection(fused_repr)  # [B, D]

        # 6. 计算与目标物品的相似度
        final_repr_norm = F.normalize(final_repr, p=2, dim=-1)
        target_item_norm = F.normalize(target_item, p=2, dim=-1)
        scores = torch.sum(final_repr_norm * target_item_norm, dim=-1)  # [B]

        # 7. 更新用户记忆
        if update_memory:
            with torch.no_grad():
                for i, user_id in enumerate(user_ids):
                    new_memory = final_repr[i].detach()
                    self.memory_bank.update_memory(user_id, new_memory)

        # 8. 返回结果
        memory_info = {
            'memory_stats': self.memory_bank.get_stats(),
            'fused_repr': final_repr
        }

        return scores, routing_weights, memory_info

    def get_memory_stats(self) -> Dict:
        """获取记忆统计信息"""
        return self.memory_bank.get_stats()

    def save_memories(self, save_path: str):
        """保存用户记忆"""
        self.memory_bank.save(save_path)

    def load_memories(self, load_path: str):
        """加载用户记忆"""
        self.memory_bank.load(load_path)
