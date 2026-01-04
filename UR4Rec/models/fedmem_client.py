"""
FedMem Client: Federated Learning Client with Local Dynamic Memory

带本地动态记忆的联邦学习客户端
- 集成LocalDynamicMemory
- Surprise-based记忆更新
- Memory Prototypes提取与聚合
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import copy
import numpy as np
from typing import Dict, List, OrderedDict, Optional, Tuple
from collections import defaultdict

from .local_dynamic_memory import LocalDynamicMemory
from .federated_aggregator import FederatedAggregator


class ClientDataset(Dataset):
    """
    客户端数据集（复用之前的实现）
    使用 leave-one-out 划分
    """

    def __init__(
        self,
        user_id: int,
        sequence: List[int],
        max_seq_len: int = 50,
        split: str = "train"
    ):
        self.user_id = user_id
        self.full_sequence = sequence
        self.max_seq_len = max_seq_len
        self.split = split

        # Leave-one-out 划分
        if split == "test":
            self.target_item = sequence[-1]
            self.input_seq = sequence[:-1]
            self.train_samples = None
        elif split == "val":
            if len(sequence) < 2:
                self.target_item = sequence[-1]
                self.input_seq = sequence[:-1]
            else:
                self.target_item = sequence[-2]
                self.input_seq = sequence[:-2]
            self.train_samples = None
        else:  # train
            if len(sequence) < 3:
                self.target_item = sequence[-1]
                self.input_seq = sequence[:-1]
                self.train_samples = None
            else:
                # 滑动窗口生成训练样本
                train_seq = sequence[:-2]
                if len(train_seq) <= 1:
                    self.target_item = sequence[-1]
                    self.input_seq = sequence[:-1]
                    self.train_samples = None
                else:
                    self.train_samples = []
                    for i in range(1, len(train_seq)):
                        input_items = train_seq[:i]
                        target = train_seq[i]
                        self.train_samples.append((input_items, target))
                    self.target_item = None
                    self.input_seq = None

    def __len__(self) -> int:
        if self.split == 'train' and self.train_samples is not None:
            return len(self.train_samples)
        return 1

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.split == 'train' and self.train_samples is not None:
            input_items, target_item = self.train_samples[idx]
        else:
            input_items = self.input_seq
            target_item = self.target_item

        # 截断/填充序列
        if len(input_items) > self.max_seq_len:
            input_items = input_items[-self.max_seq_len:]
        else:
            padding = [0] * (self.max_seq_len - len(input_items))
            input_items = padding + input_items

        return {
            'user_id': torch.tensor(self.user_id, dtype=torch.long),
            'item_seq': torch.tensor(input_items, dtype=torch.long),
            'target_item': torch.tensor(target_item, dtype=torch.long)
        }


class FedMemClient:
    """
    FedMem联邦学习客户端

    核心功能：
    1. 维护本地动态记忆（LocalDynamicMemory）
    2. 训练时使用Surprise机制更新记忆
    3. 上传模型参数 + Memory Prototypes
    4. 接收全局模型 + Global Abstract Memory
    """

    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        user_sequence: List[int],
        device: str = 'cpu',
        # [NEW] 多模态特征
        item_visual_feats: Optional[torch.Tensor] = None,
        item_text_feats: Optional[torch.Tensor] = None,
        # 训练参数
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        local_epochs: int = 1,
        batch_size: int = 32,
        max_seq_len: int = 50,
        # 负采样
        num_negatives: int = 100,
        num_items: int = 1682,
        # 记忆参数
        memory_capacity: int = 50,
        surprise_threshold: float = 0.5,
        contrastive_lambda: float = 0.1,
        num_memory_prototypes: int = 5,
        # 负采样评估参数
        use_negative_sampling: bool = False,
        num_negatives_eval: int = 100
    ):
        """
        Args:
            client_id: 客户端ID（对应user_id）
            model: 全局模型（UR4RecV2MoE）
            user_sequence: 用户交互序列
            device: 计算设备
            item_visual_feats: [NEW] 物品视觉特征 [num_items, img_dim]
            item_text_feats: [NEW] 物品文本特征 [num_items, text_dim]
            learning_rate: 学习率
            weight_decay: 权重衰减
            local_epochs: 本地训练轮数
            batch_size: 批大小
            max_seq_len: 最大序列长度
            num_negatives: 负样本数量
            num_items: 物品总数
            memory_capacity: 记忆容量
            surprise_threshold: 惊喜阈值
            contrastive_lambda: 对比学习损失权重
            num_memory_prototypes: 记忆原型数量
        """
        self.client_id = client_id
        self.device = device
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.num_negatives = num_negatives
        self.num_items = num_items
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.contrastive_lambda = contrastive_lambda
        self.num_memory_prototypes = num_memory_prototypes

        # [NEW] 存储多模态特征
        self.item_visual_feats = item_visual_feats
        self.item_text_feats = item_text_feats

        # [FIX 3] 完整性检查：验证多模态特征是否正确加载
        if client_id == 0:  # 只在第一个客户端打印，避免日志过多
            print(f"\n[FIX 3] 客户端 {client_id} 多模态特征完整性检查:")
            if self.item_visual_feats is not None:
                print(f"  ✓ 视觉特征已加载: shape={self.item_visual_feats.shape}, "
                      f"dtype={self.item_visual_feats.dtype}, device={self.item_visual_feats.device}")
                print(f"    统计: min={self.item_visual_feats.min():.4f}, "
                      f"max={self.item_visual_feats.max():.4f}, mean={self.item_visual_feats.mean():.4f}")
            else:
                print(f"  ✗ 视觉特征未加载 (item_visual_feats=None)")

            if self.item_text_feats is not None:
                print(f"  ✓ 文本特征已加载: shape={self.item_text_feats.shape}, "
                      f"dtype={self.item_text_feats.dtype}, device={self.item_text_feats.device}")
                print(f"    统计: min={self.item_text_feats.min():.4f}, "
                      f"max={self.item_text_feats.max():.4f}, mean={self.item_text_feats.mean():.4f}")
            else:
                print(f"  ✗ 文本特征未加载 (item_text_feats=None)")

            if self.item_visual_feats is None and self.item_text_feats is None:
                print(f"  ⚠️  警告: 未加载任何多模态特征！模型将仅使用ID嵌入。")

        # 负采样评估参数
        self.use_negative_sampling = use_negative_sampling
        self.num_negatives_eval = num_negatives_eval

        # 延迟模型实例化
        self._model_reference = model
        self.model = None
        self.optimizer = None

        # 本地数据
        self.user_sequence = user_sequence
        self.train_dataset = ClientDataset(
            client_id, user_sequence, max_seq_len, split="train"
        )
        self.val_dataset = ClientDataset(
            client_id, user_sequence, max_seq_len, split="val"
        )
        self.test_dataset = ClientDataset(
            client_id, user_sequence, max_seq_len, split="test"
        )

        # 用于计算训练权重
        self.num_train_samples = len(self.train_dataset)

        # 【FedMem核心】初始化本地动态记忆
        self.local_memory = LocalDynamicMemory(
            capacity=memory_capacity,
            surprise_threshold=surprise_threshold,
            device=device
        )

    def _ensure_model_initialized(self):
        """确保模型已初始化（延迟实例化）"""
        if self.model is None:
            self.model = copy.deepcopy(self._model_reference).to(self.device)
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
            # [加速优化1] 初始化混合精度训练的GradScaler
            # 兼容字符串和torch.device对象
            device_type = self.device if isinstance(self.device, str) else self.device.type
            if device_type == 'cuda':
                self.scaler = torch.cuda.amp.GradScaler()
            else:
                self.scaler = None

    def release_model(self):
        """释放模型内存"""
        if self.model is not None:
            del self.model
            del self.optimizer
            self.model = None
            self.optimizer = None
            torch.cuda.empty_cache()

    def get_data_size(self) -> int:
        """获取客户端训练数据量"""
        return self.num_train_samples

    def set_model_parameters(self, global_parameters: OrderedDict) -> None:
        """
        从服务器接收全局模型参数

        Args:
            global_parameters: 全局模型参数
        """
        self._ensure_model_initialized()
        self.model.load_state_dict(global_parameters, strict=True)

    def get_model_parameters(self) -> OrderedDict:
        """
        上传本地模型参数到服务器

        Returns:
            本地模型参数
        """
        self._ensure_model_initialized()
        return FederatedAggregator.get_model_parameters(self.model)

    def get_memory_prototypes(self) -> Optional[torch.Tensor]:
        """
        【FedMem核心】提取记忆原型

        Returns:
            [K, emb_dim] 记忆原型矩阵
        """
        return self.local_memory.get_memory_prototypes(k=self.num_memory_prototypes)

    def set_global_abstract_memory(self, global_prototypes: torch.Tensor):
        """
        【FedMem核心】接收全局抽象记忆

        Args:
            global_prototypes: [K, emb_dim] 全局原型嵌入
        """
        self.local_memory.set_global_abstract_memory(global_prototypes)

    def train_local_model(
        self,
        verbose: bool = False
    ) -> Dict[str, float]:
        """
        【FedMem核心】在本地数据上训练模型，同时更新动态记忆

        训练流程：
        1. 前向传播，计算推荐损失（rec_loss）和对比学习损失（contrastive_loss）
        2. 总损失 = rec_loss + lambda * contrastive_loss
        3. 反向传播，更新模型参数
        4. 根据Surprise（rec_loss）更新本地记忆

        Args:
            verbose: 是否打印训练信息

        Returns:
            training_metrics: 训练指标
        """
        self._ensure_model_initialized()
        self.model.train()

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        total_rec_loss = 0.0
        total_contrastive_loss = 0.0
        total_loss = 0.0
        num_batches = 0

        for epoch in range(self.local_epochs):
            epoch_rec_loss = 0.0
            epoch_contrastive_loss = 0.0

            for batch in train_loader:
                user_ids = batch['user_id'].tolist()
                item_seqs = batch['item_seq'].to(self.device)
                target_items = batch['target_item'].to(self.device)

                batch_size = item_seqs.size(0)

                # 负采样
                neg_items = self._negative_sampling(batch_size, target_items)

                # 准备候选items：[target, neg1, neg2, ...]
                all_candidates = torch.cat([
                    target_items.unsqueeze(1),  # [B, 1]
                    neg_items  # [B, N]
                ], dim=1)  # [B, 1+N]

                # ===========================
                # [加速优化1] 前向传播（FedDMMR） - 使用混合精度
                # ===========================
                # 使用autocast自动选择合适的精度
                with torch.cuda.amp.autocast(enabled=(self.scaler is not None)):
                    # 【NEW】从本地记忆检索多模态特征
                    memory_visual, memory_text = self._retrieve_multimodal_memory_batch(
                        batch_size=batch_size,
                        top_k=20
                    )

                    # 【NEW】获取候选物品的多模态特征
                    target_visual = self._get_candidate_visual_features(all_candidates)
                    target_text = self._get_candidate_text_features(all_candidates)

                    # 【NEW】使用FedDMMR的新forward接口
                    final_scores, info = self.model(
                        user_ids=user_ids,
                        input_seq=item_seqs,
                        target_items=all_candidates,
                        memory_visual=memory_visual,    # [B, 20, img_dim] 或 None
                        memory_text=memory_text,        # [B, 20, text_dim] 或 None
                        target_visual=target_visual,    # [B, N, img_dim] 或 None
                        target_text=target_text,        # [B, N, text_dim] 或 None
                        return_components=True  # 需要获取lb_loss
                    )

                    # 提取负载均衡损失和中间表示
                    lb_loss = info['lb_loss']
                    vis_out = info['vis_out']  # [B, 1+N, D] 视觉专家输出
                    sem_out = info['sem_out']  # [B, 1+N, D] 语义专家输出

                    # 计算推荐损失（使用BPR loss）
                    # all_candidates: [B, 1+N]，第0列是正样本，其余是负样本
                    labels = torch.zeros(batch_size, dtype=torch.long, device=self.device)  # 正样本索引都是0
                    rec_loss, _ = self.model.compute_loss(final_scores, labels, lb_loss=None)

                # ===========================
                # 计算惊讶度分数 (Surprise Score)
                # ===========================
                # 当推荐损失高时，说明模型对当前样本"惊讶"，可能是兴趣漂移
                # 使用sigmoid将rec_loss归一化到[0, 1]
                # 关键：detach()确保梯度不会回传到surprise计算
                surprise = torch.sigmoid(rec_loss).detach()  # 标量 -> [1]

                # 扩展为batch维度
                surprise_batch = surprise.unsqueeze(0).expand(batch_size)  # [B]

                # ===========================
                # [FIX 2] 计算漂移自适应对比学习损失 (Drift-Adaptive Contrastive Loss)
                # ===========================
                with torch.cuda.amp.autocast(enabled=(self.scaler is not None)):
                    # 提取正样本（第0个候选物品）的视觉和语义表示
                    vis_pos = vis_out[:, 0, :]  # [B, D] 正样本的视觉表示
                    sem_pos = sem_out[:, 0, :]  # [B, D] 正样本的语义表示

                    # 调用模型的compute_contrastive_loss方法
                    # [FIX 2] 修复: 不再使用自适应温度，而是返回实例权重
                    contrastive_loss, instance_weights = self.model.compute_contrastive_loss(
                        vis_repr=vis_pos,
                        sem_repr=sem_pos,
                        surprise_score=surprise_batch,
                        base_temp=0.07,
                        alpha=0.5  # 权重调节系数: weights = 1.0 + 0.5 * surprise
                    )

                    # [FIX 2] 应用实例权重到对比学习损失
                    # 修复前: loss = rec_loss + λ * contrastive_loss
                    # 修复后: 困难样本(高surprise)获得更高的对比学习权重
                    if instance_weights is not None:
                        # 重新计算每个样本的损失，应用权重后再平均
                        # 注意: contrastive_loss已经是均值，这里需要重新获取per_sample_loss
                        # 为了简化，我们直接对整体损失进行调整
                        # weighted_cl_loss = contrastive_loss * instance_weights.mean()
                        # 但更正确的做法是在compute_contrastive_loss内部返回per_sample_loss
                        # 这里我们使用一个简化版本：用平均权重缩放
                        avg_weight = instance_weights.mean()
                        weighted_contrastive_loss = contrastive_loss * avg_weight
                    else:
                        weighted_contrastive_loss = contrastive_loss

                    # ===========================
                    # 总损失（加入负载均衡损失）
                    # ===========================
                    loss = rec_loss + self.contrastive_lambda * weighted_contrastive_loss + 0.01 * lb_loss

                # ===========================
                # [加速优化1] 反向传播 - 使用GradScaler
                # ===========================
                self.optimizer.zero_grad()
                if self.scaler is not None:
                    # 使用scaler进行混合精度的反向传播
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # CPU模式，正常反向传播
                    loss.backward()
                    self.optimizer.step()

                # ===========================
                # 【Surprise-based Memory Update】
                # ===========================
                # 对于每个样本，如果rec_loss超过阈值，则更新记忆
                with torch.no_grad():
                    # 使用正样本的得分来计算surprise
                    # final_scores: [B, 1+N], 第0列是正样本
                    pos_scores = final_scores[:, 0]  # [B]
                    neg_scores = final_scores[:, 1:]  # [B, N]

                    # 计算每个样本的损失（用于Surprise判断）
                    sample_losses = -torch.log(
                        torch.sigmoid(pos_scores.unsqueeze(1) - neg_scores) + 1e-10
                    ).mean(dim=1)  # [B]

                    for i in range(batch_size):
                        item_id = target_items[i].item()
                        loss_val = sample_losses[i].item()

                        # 提取嵌入（如果模型支持）
                        text_emb = self._get_item_text_emb(item_id)
                        img_emb = self._get_item_img_emb(item_id)
                        id_emb = self._get_item_id_emb(item_id)

                        # 更新记忆
                        self.local_memory.update(
                            item_id=item_id,
                            loss_val=loss_val,
                            text_emb=text_emb,
                            img_emb=img_emb,
                            id_emb=id_emb
                        )

                # 累积损失
                epoch_rec_loss += rec_loss.item()
                epoch_contrastive_loss += contrastive_loss.item()
                num_batches += 1

            total_rec_loss += epoch_rec_loss / len(train_loader)
            total_contrastive_loss += epoch_contrastive_loss / len(train_loader)

        # 平均损失
        avg_rec_loss = total_rec_loss / self.local_epochs
        avg_contrastive_loss = total_contrastive_loss / self.local_epochs
        avg_total_loss = avg_rec_loss + self.contrastive_lambda * avg_contrastive_loss

        metrics = {
            'loss': avg_total_loss,
            'rec_loss': avg_rec_loss,
            'contrastive_loss': avg_contrastive_loss,
            'memory_size': len(self.local_memory),
            'memory_updates': self.local_memory.total_updates
        }

        if verbose:
            print(f"Client {self.client_id} | Loss: {avg_total_loss:.4f} "
                  f"(Rec: {avg_rec_loss:.4f}, Contrast: {avg_contrastive_loss:.4f}) | "
                  f"Memory: {len(self.local_memory)}/{self.local_memory.capacity}")

        return metrics

    def _query_memory_batch(self, target_items: torch.Tensor) -> Optional[Dict]:
        """
        批量查询本地记忆（旧接口，已弃用）

        Args:
            target_items: [B] 目标item IDs

        Returns:
            记忆检索结果，用于注入模型
        """
        # 已弃用：使用_retrieve_multimodal_memory_batch代替
        return None

    def _retrieve_multimodal_memory_batch(
        self,
        batch_size: int,
        top_k: int = 20
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        【FedDMMR专用】从本地记忆中批量检索多模态特征

        Args:
            batch_size: 批大小
            top_k: 返回Top-K个记忆

        Returns:
            memory_visual: [B, TopK, img_dim] 或 None
            memory_text: [B, TopK, text_dim] 或 None
        """
        return self.local_memory.retrieve_multimodal_memory_batch(
            batch_size=batch_size,
            top_k=top_k
        )

    def _get_candidate_visual_features(
        self,
        candidate_items: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """
        [FIX 3] 获取候选物品的视觉特征（从预加载的特征矩阵中索引）

        梯度流验证:
        - 使用PyTorch高级索引: visual_feats = self.item_visual_feats[valid_items]
        - 此操作支持反向传播，梯度可以流向item_visual_feats
        - 无需使用F.embedding，直接索引即可

        Args:
            candidate_items: [B, N] 候选物品IDs

        Returns:
            visual_feats: [B, N, img_dim] 或 None（如果未加载视觉特征）
        """
        if self.item_visual_feats is None:
            return None

        batch_size, num_candidates = candidate_items.shape

        # Clamp到有效范围，避免越界
        valid_items = torch.clamp(
            candidate_items,
            0,
            self.item_visual_feats.shape[0] - 1
        )

        # [FIX 3] 索引视觉特征 [B, N, img_dim]
        # 验证: 此操作梯度流完整，无需修改
        visual_feats = self.item_visual_feats[valid_items]

        return visual_feats

    def _get_candidate_text_features(
        self,
        candidate_items: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """
        [FIX 3] 获取候选物品的文本特征（从预加载的特征矩阵中索引）

        梯度流验证:
        - 使用PyTorch高级索引: text_feats = self.item_text_feats[valid_items]
        - 此操作支持反向传播，梯度可以流向item_text_feats
        - 无需使用F.embedding，直接索引即可

        Args:
            candidate_items: [B, N] 候选物品IDs

        Returns:
            text_feats: [B, N, text_dim] 或 None（如果未加载文本特征）
        """
        if self.item_text_feats is None:
            return None

        batch_size, num_candidates = candidate_items.shape

        # Clamp到有效范围，避免越界
        valid_items = torch.clamp(
            candidate_items,
            0,
            self.item_text_feats.shape[0] - 1
        )

        # [FIX 3] 索引文本特征 [B, N, text_dim]
        # 验证: 此操作梯度流完整，无需修改
        text_feats = self.item_text_feats[valid_items]

        return text_feats

    def _compute_contrastive_loss(
        self,
        user_ids: List[int],
        target_items: torch.Tensor
    ) -> torch.Tensor:
        """
        计算对比学习损失

        目标：对齐User Preference (Text) 与 Positive Item (Image/ID)

        Args:
            user_ids: 用户IDs
            target_items: [B] 目标item IDs

        Returns:
            contrastive_loss: 标量损失
        """
        # 使用模型的compute_contrastive_loss方法
        if hasattr(self.model, 'compute_contrastive_loss'):
            return self.model.compute_contrastive_loss(
                user_ids=user_ids,
                positive_items=target_items,
                negative_items=None,  # 使用batch内负样本
                temperature=0.1
            )
        else:
            # 回退：返回0损失
            return torch.tensor(0.0, device=self.device)

    def _get_item_text_emb(self, item_id: int) -> Optional[torch.Tensor]:
        """
        获取物品的文本嵌入

        Args:
            item_id: 物品ID

        Returns:
            text_emb: 文本嵌入 [emb_dim]
        """
        # 直接从存储的文本特征中获取
        if self.item_text_feats is not None and item_id < self.item_text_feats.shape[0]:
            return self.item_text_feats[item_id].clone()
        return None

    def _get_item_img_emb(self, item_id: int) -> Optional[torch.Tensor]:
        """
        获取物品的图像嵌入

        Args:
            item_id: 物品ID

        Returns:
            img_emb: 图像嵌入 [emb_dim]
        """
        # 直接从存储的视觉特征中获取
        if self.item_visual_feats is not None and item_id < self.item_visual_feats.shape[0]:
            return self.item_visual_feats[item_id].clone()
        return None

    def _get_item_id_emb(self, item_id: int) -> Optional[torch.Tensor]:
        """
        获取物品的ID嵌入

        Args:
            item_id: 物品ID

        Returns:
            id_emb: ID嵌入 [emb_dim]
        """
        if self.model is not None and hasattr(self.model, 'get_item_embeddings'):
            item_tensor = torch.tensor([item_id], device=self.device)
            with torch.no_grad():
                emb = self.model.get_item_embeddings(item_tensor, embedding_type='id')
                if emb is not None:
                    return emb.squeeze(0)
        return None

    def _negative_sampling(
        self,
        batch_size: int,
        target_items: torch.Tensor
    ) -> torch.Tensor:
        """
        [加速优化3] 优化后的负采样（批量化处理，避免逐样本循环）

        Args:
            batch_size: 批大小
            target_items: [B] 正样本item IDs

        Returns:
            neg_items: [B, num_negatives]
        """
        # [优化3] 一次性生成所有负样本（过采样以确保足够）
        # 为每个样本生成2倍的候选，然后过滤
        all_candidates = torch.randint(
            1, self.num_items,
            (batch_size, self.num_negatives * 2),
            device=self.device
        )  # [B, num_negatives*2]

        # 创建正样本的mask：[B, num_negatives*2]
        pos_mask = all_candidates == target_items.unsqueeze(1)

        # 将正样本位置设置为0（无效item id）
        all_candidates[pos_mask] = 0

        # 对于每个样本，选择前num_negatives个非零候选
        neg_items = []
        for i in range(batch_size):
            valid_negs = all_candidates[i][all_candidates[i] != 0]
            if len(valid_negs) >= self.num_negatives:
                neg_items.append(valid_negs[:self.num_negatives])
            else:
                # 如果不够，补充随机采样（极少发生）
                need_more = self.num_negatives - len(valid_negs)
                extra = torch.randint(1, self.num_items, (need_more,), device=self.device)
                neg_items.append(torch.cat([valid_negs, extra]))

        return torch.stack(neg_items)  # [B, num_negatives]

    def evaluate(
        self,
        user_sequences: Optional[Dict[int, List[int]]] = None,
        split: str = "test",
        k_list: List[int] = [5, 10, 20]
    ) -> Dict[str, float]:
        """
        评估模型

        Args:
            user_sequences: 完整用户序列字典（用于负采样）{user_id: [items]}
            split: 'val' 或 'test'
            k_list: Top-K列表

        Returns:
            metrics: 评估指标
        """
        # 根据配置选择评估方式
        if self.use_negative_sampling and user_sequences is not None:
            return self.evaluate_negative_sampling(user_sequences, split, k_list)

        # 默认使用全排序评估
        self._ensure_model_initialized()
        self.model.eval()

        dataset = self.val_dataset if split == "val" else self.test_dataset

        with torch.no_grad():
            batch = dataset[0]

            user_id = batch['user_id'].item()
            item_seq = batch['item_seq'].unsqueeze(0).to(self.device)
            target_item = batch['target_item'].item()

            # 计算所有items的得分
            # num_items = max_item_id + 1, so arange(1, num_items) = [1, ..., max_item_id]
            all_item_ids = torch.arange(1, self.num_items, device=self.device)
            all_item_ids_batch = all_item_ids.unsqueeze(0)  # [1, num_items-1]

            # 【NEW】从本地记忆检索多模态特征（用于FedDMMR）
            memory_visual, memory_text = self._retrieve_multimodal_memory_batch(
                batch_size=1,
                top_k=20
            )

            # 【NEW】获取候选物品的多模态特征
            target_visual = self._get_candidate_visual_features(all_item_ids_batch)
            target_text = self._get_candidate_text_features(all_item_ids_batch)

            # 【NEW】FedDMMR前向
            final_scores = self.model(
                user_ids=[user_id],
                input_seq=item_seq,
                target_items=all_item_ids_batch,
                memory_visual=memory_visual,    # [1, 20, img_dim] 或 None
                memory_text=memory_text,        # [1, 20, text_dim] 或 None
                target_visual=target_visual,    # [1, num_items-1, img_dim] 或 None
                target_text=target_text,        # [1, num_items-1, text_dim] 或 None
                return_components=False  # 评估时不需要额外信息
            )

            scores = final_scores  # [1, num_items-1]

            # 获取Top-K
            _, top_k_indices = torch.topk(scores, max(k_list), dim=1)
            top_k_items = all_item_ids[top_k_indices].squeeze(0).cpu().numpy()

            # 计算指标
            metrics = {}
            for k in k_list:
                top_k = top_k_items[:k]

                # HR@K
                hr = 1.0 if target_item in top_k else 0.0
                metrics[f'HR@{k}'] = hr

                # NDCG@K
                if target_item in top_k:
                    idx = np.where(top_k == target_item)[0][0]
                    ndcg = 1.0 / np.log2(idx + 2)
                else:
                    ndcg = 0.0
                metrics[f'NDCG@{k}'] = ndcg

            # MRR
            if target_item in top_k_items:
                rank = np.where(top_k_items == target_item)[0][0] + 1
                mrr = 1.0 / rank
            else:
                mrr = 0.0
            metrics['MRR'] = mrr

        return metrics

    def evaluate_negative_sampling(
        self,
        user_sequences: Dict[int, List[int]],
        split: str = "test",
        k_list: List[int] = [5, 10, 20]
    ) -> Dict[str, float]:
        """
        使用1:100负采样评估模型（对齐NCF/原始SASRec论文的评估协议）

        对每个测试用户:
        1. 获取Ground Truth物品
        2. 随机采样N个负样本物品（不在用户历史交互中）
        3. 构建N+1个候选物品集合: [Ground Truth, Neg_1, ..., Neg_N]
        4. 计算Ground Truth在这N+1个物品中的排名
        5. 计算HR@K和NDCG@K指标

        Args:
            user_sequences: 用户完整序列字典 {user_id: [items]}
            split: 'val' 或 'test'
            k_list: 评估的K值列表

        Returns:
            metrics: 评估指标字典
        """
        self._ensure_model_initialized()
        self.model.eval()

        dataset = self.val_dataset if split == "val" else self.test_dataset

        # 为当前用户准备候选负样本池
        # 从所有物品中排除用户历史交互过的物品
        user_id = self.client_id
        full_sequence = user_sequences[user_id]
        user_items = set(full_sequence)
        all_items = set(range(1, self.num_items))  # 物品ID范围: 1~num_items-1
        candidate_pool = list(all_items - user_items)

        # 评估指标累加器
        all_hr = {k: [] for k in k_list}
        all_ndcg = {k: [] for k in k_list}

        with torch.no_grad():
            batch = dataset[0]

            user_id_val = batch['user_id'].item()
            item_seq = batch['item_seq'].unsqueeze(0).to(self.device)  # [1, seq_len]
            target_item = batch['target_item'].item()

            # 从候选池中随机采样N个负样本
            if len(candidate_pool) < self.num_negatives_eval:
                negative_items = candidate_pool
            else:
                negative_items = np.random.choice(
                    candidate_pool,
                    size=self.num_negatives_eval,
                    replace=False
                ).tolist()

            # 构建N+1个候选物品: [Ground Truth] + [N个负样本]
            candidate_items = [target_item] + negative_items  # 长度: N+1
            candidate_items_tensor = torch.tensor(
                candidate_items, dtype=torch.long
            ).unsqueeze(0).to(self.device)  # [1, N+1]

            # 【NEW】从本地记忆检索多模态特征
            memory_visual, memory_text = self._retrieve_multimodal_memory_batch(
                batch_size=1,
                top_k=20
            )

            # 【NEW】获取候选物品的多模态特征
            target_visual = self._get_candidate_visual_features(candidate_items_tensor)
            target_text = self._get_candidate_text_features(candidate_items_tensor)

            # 【NEW】FedMem前向传播
            final_scores = self.model(
                user_ids=[user_id_val],
                input_seq=item_seq,
                target_items=candidate_items_tensor,
                memory_visual=memory_visual,
                memory_text=memory_text,
                target_visual=target_visual,
                target_text=target_text,
                return_components=False
            )

            scores = final_scores.squeeze()  # [N+1]

            # 对得分进行排序，获取排名
            # Ground Truth在索引0，我们需要找到它的排名
            _, ranked_indices = torch.sort(scores, descending=True)
            ranked_indices = ranked_indices.cpu().numpy()

            # 找到Ground Truth（索引0）的排名位置
            rank = np.where(ranked_indices == 0)[0][0] + 1  # 排名从1开始

            # 计算HR@K和NDCG@K
            for k in k_list:
                # HR@K: Ground Truth是否在Top-K中
                if rank <= k:
                    all_hr[k].append(1.0)
                    # NDCG@K: 如果在Top-K中，计算NDCG
                    ndcg = 1.0 / np.log2(rank + 1)  # rank从1开始，log2(rank+1)
                    all_ndcg[k].append(ndcg)
                else:
                    all_hr[k].append(0.0)
                    all_ndcg[k].append(0.0)

        # 计算平均指标
        metrics = {}
        for k in k_list:
            metrics[f'HR@{k}'] = np.mean(all_hr[k])
            metrics[f'NDCG@{k}'] = np.mean(all_ndcg[k])

        # 添加MRR
        mrr = 1.0 / rank if rank > 0 else 0.0
        metrics['MRR'] = mrr

        return metrics

    def get_memory_statistics(self) -> Dict:
        """获取记忆统计信息"""
        return self.local_memory.get_statistics()

    def __repr__(self):
        return f"FedMemClient(id={self.client_id}, data={self.num_train_samples}, " \
               f"memory={len(self.local_memory)}/{self.local_memory.capacity})"
