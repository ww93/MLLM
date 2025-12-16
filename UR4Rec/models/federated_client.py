"""
Federated Learning Client for UR4Rec

实现联邦学习的客户端逻辑：
- 每个user作为一个client
- 本地训练 SASRec + UR4Rec MoE Retriever
- 上传模型参数到服务器
- 接收全局模型更新
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Optional, Tuple, Any
from collections import OrderedDict
import copy
import numpy as np
from tqdm import tqdm

from .ur4rec_v2_moe import UR4RecV2MoE
from .sasrec import SASRec
from .federated_aggregator import FederatedAggregator


class ClientDataset(Dataset):
    """
    单个客户端（用户）的数据集

    使用 leave-one-out 划分：
    - 训练集：除最后2个item外的所有历史，使用滑动窗口生成训练样本
    - 验证集：倒数第2个item
    - 测试集：最后1个item
    """

    def __init__(
        self,
        user_id: int,
        sequence: List[int],
        max_seq_len: int = 50,
        split: str = "train"  # 'train', 'val', 'test'
    ):
        """
        Args:
            user_id: 用户ID
            sequence: 用户交互序列（按时间排序）
            max_seq_len: 最大序列长度
            split: 数据集划分
        """
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
                # 数据太少，使用最后一个作为验证
                self.target_item = sequence[-1]
                self.input_seq = sequence[:-1]
            else:
                self.target_item = sequence[-2]
                self.input_seq = sequence[:-2]
            self.train_samples = None
        else:  # train
            # 训练时：排除最后2个（用于val/test），生成滑动窗口样本
            if len(sequence) < 3:
                # 数据太少，使用简单划分
                self.target_item = sequence[-1]
                self.input_seq = sequence[:-1]
                self.train_samples = None
            else:
                # 使用滑动窗口生成训练样本
                # 例如 [1,2,3,4,5,6,7,8]，排除[7,8]用于val/test
                # 生成：[1]→2, [1,2]→3, [1,2,3]→4, [1,2,3,4]→5, [1,2,3,4,5]→6
                train_seq = sequence[:-2]

                # 如果train_seq长度<=1，无法生成滑动窗口，使用简单划分
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
        """
        返回一个训练/测试样本

        Returns:
            {
                'user_id': user ID
                'item_seq': 输入序列 [max_seq_len]
                'target_item': 目标item
            }
        """
        if self.split == 'train' and self.train_samples is not None:
            # 训练集：返回第idx个样本
            input_items, target_item = self.train_samples[idx]
        else:
            # 验证/测试集：返回单个样本
            input_items = self.input_seq
            target_item = self.target_item

        # 截断/填充序列
        if len(input_items) > self.max_seq_len:
            input_items = input_items[-self.max_seq_len:]
        else:
            # 填充 0 (padding token)
            padding = [0] * (self.max_seq_len - len(input_items))
            input_items = padding + input_items

        return {
            'user_id': torch.tensor(self.user_id, dtype=torch.long),
            'item_seq': torch.tensor(input_items, dtype=torch.long),
            'target_item': torch.tensor(target_item, dtype=torch.long)
        }


class FederatedClient:
    """
    联邦学习客户端

    每个user对应一个client，拥有：
    - 本地模型（SASRec + Retriever）
    - 本地数据（该user的交互历史）
    - 本地优化器
    """

    def __init__(
        self,
        client_id: int,
        model: UR4RecV2MoE,
        user_sequence: List[int],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        # 训练参数
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        local_epochs: int = 1,
        batch_size: int = 32,
        max_seq_len: int = 50,
        # 负采样
        num_negatives: int = 100,
        num_items: int = 1682
    ):
        """
        Args:
            client_id: 客户端ID（对应user_id）
            model: 本地模型（存储模型引用，延迟复制）
            user_sequence: 该用户的交互序列
            device: 计算设备
            learning_rate: 学习率
            weight_decay: 权重衰减
            local_epochs: 本地训练轮数
            batch_size: 批大小
            max_seq_len: 最大序列长度
            num_negatives: 负样本数量
            num_items: 总item数（用于负采样）
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

        # 延迟模型实例化 - 仅存储模型引用
        # 实际的模型副本会在需要训练时创建
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

        # 用于计算训练权重（数据量）
        self.num_train_samples = len(self.train_dataset)

    def _ensure_model_initialized(self):
        """确保模型已初始化（延迟实例化）"""
        if self.model is None:
            # 创建模型副本
            self.model = copy.deepcopy(self._model_reference).to(self.device)
            # 创建优化器
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )

    def release_model(self):
        """释放模型内存（训练完成后）"""
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

    def train_local_model(
        self,
        verbose: bool = False
    ) -> Dict[str, float]:
        """
        在本地数据上训练模型

        Args:
            verbose: 是否打印训练信息

        Returns:
            training_metrics: 训练指标
        """
        self._ensure_model_initialized()
        self.model.train()

        # 创建DataLoader
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        total_loss = 0.0
        num_batches = 0

        for epoch in range(self.local_epochs):
            epoch_loss = 0.0

            for batch in train_loader:
                # 准备数据
                user_ids = batch['user_id'].to(self.device)
                item_seqs = batch['item_seq'].to(self.device)
                target_items = batch['target_item'].to(self.device)

                # 负采样
                batch_size = item_seqs.size(0)
                neg_items = self._negative_sampling(batch_size, target_items)

                # 前向传播
                # 获取序列表示
                seq_output = self.model(item_seqs)  # [B, L, D]
                seq_repr = seq_output[:, -1, :]  # 取最后一个位置 [B, D]

                # 计算正负样本的得分
                pos_scores = self._compute_scores(seq_repr, target_items)  # [B]
                neg_scores = self._compute_scores(seq_repr, neg_items)  # [B, N]

                # BPR 损失：max(0, 1 - pos + neg)
                # 对每个负样本计算损失
                loss = -torch.mean(
                    torch.log(torch.sigmoid(pos_scores.unsqueeze(1) - neg_scores) + 1e-10)
                )

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            total_loss += epoch_loss / len(train_loader)

            if verbose:
                print(f"Client {self.client_id} - Epoch {epoch+1}/{self.local_epochs} - Loss: {epoch_loss/len(train_loader):.4f}")

        avg_loss = total_loss / self.local_epochs

        return {
            'loss': avg_loss,
            'num_samples': self.num_train_samples
        }

    def _negative_sampling(
        self,
        batch_size: int,
        positive_items: torch.Tensor
    ) -> torch.Tensor:
        """
        负采样

        Args:
            batch_size: 批大小
            positive_items: 正样本item [B]

        Returns:
            negative_items: 负样本items [B, num_negatives]
        """
        neg_items = []

        for i in range(batch_size):
            pos_item = positive_items[i].item()

            # 随机采样负样本（排除正样本和padding）
            negs = []
            while len(negs) < self.num_negatives:
                neg = np.random.randint(1, self.num_items)  # 1 to num_items-1
                if neg != pos_item:
                    negs.append(neg)

            neg_items.append(negs)

        return torch.tensor(neg_items, dtype=torch.long, device=self.device)

    def _compute_scores(
        self,
        seq_repr: torch.Tensor,
        items: torch.Tensor
    ) -> torch.Tensor:
        """
        计算序列表示和items的相似度得分

        Args:
            seq_repr: 序列表示 [B, D]
            items: item IDs [B] 或 [B, N]

        Returns:
            scores: 相似度得分 [B] 或 [B, N]
        """
        # 获取item embeddings
        if items.dim() == 1:
            # 单个item: [B]
            item_embs = self.model.item_embedding(items)  # [B, D]
            scores = torch.sum(seq_repr * item_embs, dim=-1)  # [B]
        else:
            # 多个items: [B, N]
            item_embs = self.model.item_embedding(items)  # [B, N, D]
            scores = torch.sum(
                seq_repr.unsqueeze(1) * item_embs, dim=-1
            )  # [B, N]

        return scores

    def evaluate(
        self,
        split: str = "test",
        k_list: List[int] = [5, 10, 20]
    ) -> Dict[str, float]:
        """
        在验证/测试集上评估模型

        Args:
            split: 'val' 或 'test'
            k_list: Top-K列表

        Returns:
            metrics: 评估指标
        """
        self._ensure_model_initialized()
        self.model.eval()

        dataset = self.val_dataset if split == "val" else self.test_dataset

        with torch.no_grad():
            batch = dataset[0]  # 只有一个样本

            user_id = batch['user_id'].unsqueeze(0).to(self.device)
            item_seq = batch['item_seq'].unsqueeze(0).to(self.device)
            target_item = batch['target_item'].item()

            # 获取序列表示
            seq_output = self.model(item_seq)
            seq_repr = seq_output[:, -1, :]  # [1, D]

            # 计算所有items的得分（注意：item ID从1开始，0是padding）
            all_item_ids = torch.arange(1, self.num_items, device=self.device)  # [1, 2, ..., 1681]
            all_item_embs = self.model.item_embedding(all_item_ids)  # [1681, D]

            scores = torch.matmul(seq_repr, all_item_embs.T).squeeze(0)  # [1681]

            # 排序获取Top-K（scores的索引对应all_item_ids的索引）
            _, top_k_indices = torch.topk(scores, max(k_list))
            # 通过索引获取实际的item ID（all_item_ids已经是真实ID）
            top_k_items = all_item_ids[top_k_indices].cpu().numpy()

            # 计算指标
            metrics = {}
            for k in k_list:
                top_k = top_k_items[:k]

                # HR@K
                hr = 1.0 if target_item in top_k else 0.0
                metrics[f'HR@{k}'] = hr

                # NDCG@K
                if target_item in top_k:
                    rank = np.where(top_k == target_item)[0][0] + 1
                    ndcg = 1.0 / np.log2(rank + 1)
                else:
                    ndcg = 0.0
                metrics[f'NDCG@{k}'] = ndcg

                # MRR (只在K=最大时计算一次)
                if k == max(k_list):
                    if target_item in top_k:
                        rank = np.where(top_k == target_item)[0][0] + 1
                        mrr = 1.0 / rank
                    else:
                        mrr = 0.0
                    metrics['MRR'] = mrr

        return metrics

    def get_train_loss(self) -> float:
        """
        计算训练集上的损失（用于监控）

        Returns:
            average_loss: 平均损失
        """
        self._ensure_model_initialized()
        self.model.eval()

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in train_loader:
                user_ids = batch['user_id'].to(self.device)
                item_seqs = batch['item_seq'].to(self.device)
                target_items = batch['target_item'].to(self.device)

                batch_size = item_seqs.size(0)
                neg_items = self._negative_sampling(batch_size, target_items)

                seq_output = self.model(item_seqs)
                seq_repr = seq_output[:, -1, :]

                pos_scores = self._compute_scores(seq_repr, target_items)
                neg_scores = self._compute_scores(seq_repr, neg_items)

                loss = -torch.mean(
                    torch.log(torch.sigmoid(pos_scores.unsqueeze(1) - neg_scores) + 1e-10)
                )

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0
