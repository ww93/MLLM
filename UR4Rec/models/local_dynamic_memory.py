"""
FedMem: Local Dynamic Multimodal Memory Module

本地动态记忆模块 - 在客户端（用户）侧维护，基于"惊喜"（Surprise）机制动态更新
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import time


class LocalDynamicMemory:
    """
    本地动态多模态记忆

    功能：
    1. 存储用户的重要交互记忆（基于Surprise机制）
    2. 支持多模态表示（文本嵌入、图像嵌入、ID嵌入）
    3. 基于效用（Utility）的过期机制

    记忆结构：
    {
        item_id: {
            'text_emb': Tensor,      # LLM生成的文本嵌入
            'img_emb': Tensor,       # 图像特征（如CLIP）
            'id_emb': Tensor,        # ID嵌入
            'timestamp': float,      # 最后访问时间
            'frequency': int,        # 访问频率
            'surprise': float,       # 惊喜值（预测误差）
        }
    }
    """

    def __init__(
        self,
        capacity: int = 50,
        surprise_threshold: float = 0.5,
        recency_weight: float = 0.6,
        frequency_weight: float = 0.4,
        device: str = 'cpu'
    ):
        """
        Args:
            capacity: 记忆容量（最多存储的item数量）
            surprise_threshold: 惊喜阈值，超过此值才加入记忆
            recency_weight: 近期性权重（用于效用计算）
            frequency_weight: 频率权重（用于效用计算）
            device: 计算设备
        """
        self.capacity = capacity
        self.surprise_threshold = surprise_threshold
        self.alpha = recency_weight  # 近期性权重
        self.beta = frequency_weight  # 频率权重
        self.device = device

        # 记忆缓冲区
        self.memory_buffer: Dict[int, Dict] = {}

        # 统计信息
        self.total_accesses = 0
        self.total_updates = 0

    def query(
        self,
        target_item: int,
        k: int = 5,
        return_embeddings: bool = True
    ) -> Optional[Dict]:
        """
        查询记忆，获取与目标item相关的记忆条目

        Args:
            target_item: 目标item ID
            k: 返回Top-K个最相关的记忆
            return_embeddings: 是否返回嵌入向量

        Returns:
            相关记忆条目，如果target_item在记忆中则直接返回
        """
        self.total_accesses += 1

        # 如果目标item在记忆中，直接返回
        if target_item in self.memory_buffer:
            memory_entry = self.memory_buffer[target_item]
            # 更新访问时间和频率
            memory_entry['timestamp'] = time.time()
            memory_entry['frequency'] += 1

            if return_embeddings:
                return memory_entry
            else:
                return {
                    'item_id': target_item,
                    'frequency': memory_entry['frequency'],
                    'surprise': memory_entry['surprise']
                }

        # 否则，返回效用最高的Top-K记忆条目
        if len(self.memory_buffer) == 0:
            return None

        # 计算所有记忆条目的效用分数
        utilities = {}
        current_time = time.time()

        for item_id, memory in self.memory_buffer.items():
            utility = self._compute_utility(memory, current_time)
            utilities[item_id] = utility

        # 选择Top-K
        top_k_items = sorted(utilities.items(), key=lambda x: x[1], reverse=True)[:k]

        if return_embeddings:
            return {
                item_id: self.memory_buffer[item_id]
                for item_id, _ in top_k_items
            }
        else:
            return {
                'top_k_items': [item_id for item_id, _ in top_k_items],
                'utilities': [utility for _, utility in top_k_items]
            }

    def update(
        self,
        item_id: int,
        loss_val: float,
        text_emb: Optional[torch.Tensor] = None,
        img_emb: Optional[torch.Tensor] = None,
        id_emb: Optional[torch.Tensor] = None
    ):
        """
        基于Surprise机制更新记忆

        Surprise Logic:
        - 如果 loss_val > surprise_threshold，说明模型对此item的预测不准确
        - 这是一个"惊喜"信号，表明此item值得记忆
        - 将其加入或更新记忆缓冲区

        Expire Logic:
        - 如果记忆已满，移除效用最低的条目
        - 效用 = alpha * recency + beta * frequency

        Args:
            item_id: 物品ID
            loss_val: 该item的损失值（Surprise指标）
            text_emb: 文本嵌入
            img_emb: 图像嵌入
            id_emb: ID嵌入
        """
        # Surprise判断：只有当loss超过阈值时才加入记忆
        if loss_val < self.surprise_threshold:
            return

        current_time = time.time()

        # 如果item已在记忆中，更新其信息
        if item_id in self.memory_buffer:
            memory_entry = self.memory_buffer[item_id]
            memory_entry['timestamp'] = current_time
            memory_entry['frequency'] += 1
            memory_entry['surprise'] = max(memory_entry['surprise'], loss_val)

            # 更新嵌入（如果提供）
            if text_emb is not None:
                memory_entry['text_emb'] = text_emb.detach().cpu()
            if img_emb is not None:
                memory_entry['img_emb'] = img_emb.detach().cpu()
            if id_emb is not None:
                memory_entry['id_emb'] = id_emb.detach().cpu()

        else:
            # 新item加入记忆
            # 如果记忆已满，移除效用最低的条目
            if len(self.memory_buffer) >= self.capacity:
                self._expire_least_useful()

            # 添加新记忆条目
            self.memory_buffer[item_id] = {
                'text_emb': text_emb.detach().cpu() if text_emb is not None else None,
                'img_emb': img_emb.detach().cpu() if img_emb is not None else None,
                'id_emb': id_emb.detach().cpu() if id_emb is not None else None,
                'timestamp': current_time,
                'frequency': 1,
                'surprise': loss_val
            }

        self.total_updates += 1

    def _compute_utility(self, memory_entry: Dict, current_time: float) -> float:
        """
        计算记忆条目的效用分数

        Utility = alpha * recency_score + beta * frequency_score

        - recency_score: 归一化的时间衰减分数
        - frequency_score: 归一化的访问频率分数

        Args:
            memory_entry: 记忆条目
            current_time: 当前时间戳

        Returns:
            效用分数
        """
        # 计算时间差（秒）
        time_diff = current_time - memory_entry['timestamp']

        # 时间衰减：使用指数衰减，半衰期为1小时（3600秒）
        half_life = 3600.0
        recency_score = np.exp(-time_diff / half_life)

        # 频率分数：归一化到[0, 1]
        # 使用对数尺度来避免频率过大导致的偏差
        frequency_score = np.log1p(memory_entry['frequency']) / np.log1p(memory_entry['frequency'] + 10)

        # 综合效用
        utility = self.alpha * recency_score + self.beta * frequency_score

        return utility

    def _expire_least_useful(self):
        """
        移除效用最低的记忆条目（Expire Logic）
        """
        if len(self.memory_buffer) == 0:
            return

        current_time = time.time()

        # 计算所有条目的效用
        utilities = {}
        for item_id, memory in self.memory_buffer.items():
            utilities[item_id] = self._compute_utility(memory, current_time)

        # 找到效用最低的条目
        least_useful_item = min(utilities.items(), key=lambda x: x[1])[0]

        # 移除
        del self.memory_buffer[least_useful_item]

    def get_memory_prototypes(self, k: int = 5) -> Optional[torch.Tensor]:
        """
        获取记忆原型（Memory Prototypes）

        使用K-Means聚类方法，将记忆条目聚类为K个原型（中心点）
        这些原型将被上传到服务器进行聚合

        Args:
            k: 聚类中心数量

        Returns:
            [k, emb_dim] 的原型嵌入矩阵
        """
        if len(self.memory_buffer) == 0:
            return None

        # 如果记忆条目少于k，直接返回所有条目的平均嵌入
        if len(self.memory_buffer) < k:
            k = len(self.memory_buffer)

        # 收集所有嵌入（优先使用text_emb，如果没有则使用id_emb）
        embeddings = []
        for item_id, memory in self.memory_buffer.items():
            if memory['text_emb'] is not None:
                embeddings.append(memory['text_emb'])
            elif memory['id_emb'] is not None:
                embeddings.append(memory['id_emb'])

        if len(embeddings) == 0:
            return None

        # 转换为张量
        embeddings = torch.stack(embeddings)  # [N, emb_dim]

        # 简单的K-Means聚类（使用PyTorch实现）
        prototypes = self._kmeans(embeddings, k)

        return prototypes

    def _kmeans(self, embeddings: torch.Tensor, k: int, max_iters: int = 10) -> torch.Tensor:
        """
        简单的K-Means聚类实现

        Args:
            embeddings: [N, emb_dim]
            k: 聚类中心数量
            max_iters: 最大迭代次数

        Returns:
            [k, emb_dim] 的聚类中心
        """
        N, emb_dim = embeddings.shape

        # 随机初始化k个中心
        indices = torch.randperm(N)[:k]
        centroids = embeddings[indices].clone()

        for _ in range(max_iters):
            # 计算每个点到各中心的距离
            distances = torch.cdist(embeddings, centroids)  # [N, k]

            # 分配到最近的中心
            assignments = torch.argmin(distances, dim=1)  # [N]

            # 更新中心
            new_centroids = centroids.clone()
            for i in range(k):
                mask = (assignments == i)
                if mask.sum() > 0:
                    new_centroids[i] = embeddings[mask].mean(dim=0)

            # 检查收敛
            if torch.allclose(centroids, new_centroids, atol=1e-4):
                break

            centroids = new_centroids

        return centroids

    def set_global_abstract_memory(self, global_prototypes: torch.Tensor):
        """
        接收服务器下发的全局抽象记忆（Global Abstract Memory）

        这些全局原型可以用于辅助本地推荐

        Args:
            global_prototypes: [k, emb_dim] 全局原型嵌入
        """
        self.global_prototypes = global_prototypes.to(self.device)

    def get_statistics(self) -> Dict:
        """
        获取记忆统计信息

        Returns:
            统计信息字典
        """
        return {
            'memory_size': len(self.memory_buffer),
            'capacity': self.capacity,
            'total_accesses': self.total_accesses,
            'total_updates': self.total_updates,
            'utilization': len(self.memory_buffer) / self.capacity
        }

    def clear(self):
        """清空记忆缓冲区"""
        self.memory_buffer.clear()
        self.total_accesses = 0
        self.total_updates = 0

    def __len__(self):
        return len(self.memory_buffer)

    def retrieve_multimodal_memory_batch(
        self,
        batch_size: int,
        top_k: int = 20
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        【FedDMMR专用】批量检索多模态记忆张量

        对于批中的每个样本，返回效用最高的Top-K记忆的视觉和文本特征。
        此方法专门用于FedDMMR架构的三专家系统（序列、视觉、语义）。

        Args:
            batch_size: 批大小
            top_k: 每个样本返回的记忆数量

        Returns:
            memory_visual: [B, TopK, img_dim] 或 None（如果没有视觉特征）
            memory_text: [B, TopK, text_dim] 或 None（如果没有文本特征）

        Example:
            >>> memory = LocalDynamicMemory(capacity=50)
            >>> vis, txt = memory.retrieve_multimodal_memory_batch(batch_size=32, top_k=20)
            >>> # vis: [32, 20, 512], txt: [32, 20, 384]
        """
        if len(self.memory_buffer) == 0:
            return None, None

        # 计算效用最高的Top-K记忆
        current_time = time.time()
        utilities = {}
        for item_id, memory in self.memory_buffer.items():
            utilities[item_id] = self._compute_utility(memory, current_time)

        # 排序并选择Top-K（如果记忆不足top_k个，取所有）
        actual_k = min(top_k, len(utilities))
        sorted_items = sorted(utilities.items(), key=lambda x: x[1], reverse=True)[:actual_k]
        top_k_item_ids = [item_id for item_id, _ in sorted_items]

        # 收集视觉和文本特征
        visual_feats = []
        text_feats = []

        has_visual = False
        has_text = False

        for item_id in top_k_item_ids:
            memory_entry = self.memory_buffer[item_id]

            # 视觉特征 (CLIP)
            img_emb = memory_entry.get('img_emb')
            if img_emb is not None:
                visual_feats.append(img_emb)
                has_visual = True

            # 文本特征 (Sentence-BERT)
            text_emb = memory_entry.get('text_emb')
            if text_emb is not None:
                text_feats.append(text_emb)
                has_text = True

        # 构造批量张量
        memory_visual = None
        memory_text = None

        # 处理视觉特征
        if has_visual and len(visual_feats) > 0:
            # Stack所有特征 [K', img_dim]
            visual_stack = torch.stack(visual_feats).to(self.device)

            # 如果不足top_k，零填充
            if visual_stack.shape[0] < top_k:
                img_dim = visual_stack.shape[1]
                padding = torch.zeros(
                    top_k - visual_stack.shape[0],
                    img_dim,
                    device=self.device
                )
                visual_stack = torch.cat([visual_stack, padding], dim=0)  # [TopK, img_dim]

            # 复制到batch维度
            memory_visual = visual_stack.unsqueeze(0).expand(batch_size, -1, -1)  # [B, TopK, img_dim]

        # 处理文本特征
        if has_text and len(text_feats) > 0:
            # Stack所有特征 [K', text_dim]
            text_stack = torch.stack(text_feats).to(self.device)

            # 如果不足top_k，零填充
            if text_stack.shape[0] < top_k:
                text_dim = text_stack.shape[1]
                padding = torch.zeros(
                    top_k - text_stack.shape[0],
                    text_dim,
                    device=self.device
                )
                text_stack = torch.cat([text_stack, padding], dim=0)  # [TopK, text_dim]

            # 复制到batch维度
            memory_text = text_stack.unsqueeze(0).expand(batch_size, -1, -1)  # [B, TopK, text_dim]

        return memory_visual, memory_text

    def __repr__(self):
        return f"LocalDynamicMemory(size={len(self.memory_buffer)}/{self.capacity}, " \
               f"updates={self.total_updates}, accesses={self.total_accesses})"
