"""
Federated SASRec Client: Pure SASRec in Federated Learning Setting

联邦学习版本的SASRec客户端
- 不包含memory、MoE等复杂组件
- 纯SASRec模型在联邦学习框架下训练
- 用于对比集中式vs联邦式的性能差异
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import copy
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from .sasrec_fixed import SASRecFixed
from .federated_aggregator import FederatedAggregator


class ClientDataset(Dataset):
    """
    客户端数据集（使用leave-one-out划分）
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


class FederatedSASRecClient:
    """
    联邦学习版本的SASRec客户端

    核心功能:
    1. 本地训练SASRec模型
    2. 上传模型参数到服务器
    3. 接收全局模型参数
    4. 支持1:100负采样评估
    """

    def __init__(
        self,
        client_id: int,
        user_sequence: List[int],
        num_items: int,
        device: str = 'cpu',
        # 模型参数
        hidden_dim: int = 128,
        num_blocks: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 50,
        # 训练参数
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        local_epochs: int = 1,
        batch_size: int = 32,
        # 负采样
        num_negatives: int = 50,
        # 评估参数
        use_negative_sampling: bool = False,
        num_negatives_eval: int = 100
    ):
        """
        Args:
            client_id: 客户端ID（对应user_id）
            user_sequence: 用户交互序列
            num_items: 物品总数
            device: 计算设备
            hidden_dim: 隐藏维度
            num_blocks: Transformer块数量
            num_heads: 注意力头数量
            dropout: Dropout率
            max_seq_len: 最大序列长度
            learning_rate: 学习率
            weight_decay: 权重衰减
            local_epochs: 本地训练轮数
            batch_size: 批大小
            num_negatives: 训练时负样本数量
            use_negative_sampling: 是否使用负采样评估
            num_negatives_eval: 评估时负样本数量
        """
        self.client_id = client_id
        self.user_sequence = user_sequence
        self.num_items = num_items
        self.device = device

        # 模型参数
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout = dropout
        self.max_seq_len = max_seq_len

        # 训练参数
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.num_negatives = num_negatives

        # 评估参数
        self.use_negative_sampling = use_negative_sampling
        self.num_negatives_eval = num_negatives_eval

        # 创建数据集
        self.train_dataset = ClientDataset(client_id, user_sequence, max_seq_len, split="train")
        self.val_dataset = ClientDataset(client_id, user_sequence, max_seq_len, split="val")
        self.test_dataset = ClientDataset(client_id, user_sequence, max_seq_len, split="test")

        # 模型(延迟初始化)
        self.model = None
        self.optimizer = None

    def _ensure_model_initialized(self):
        """确保模型已初始化"""
        if self.model is None:
            self.model = SASRecFixed(
                num_items=self.num_items,
                hidden_dim=self.hidden_dim,
                num_blocks=self.num_blocks,
                num_heads=self.num_heads,
                dropout=self.dropout,
                max_seq_len=self.max_seq_len
            ).to(self.device)

            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )

    def get_model_parameters(self) -> Dict[str, torch.Tensor]:
        """获取模型参数"""
        self._ensure_model_initialized()
        return FederatedAggregator.get_model_parameters(self.model)

    def set_model_parameters(self, parameters: Dict[str, torch.Tensor]):
        """设置模型参数"""
        self._ensure_model_initialized()
        FederatedAggregator.set_model_parameters(self.model, parameters)

    def release_model(self):
        """释放模型以节省内存"""
        if self.model is not None:
            del self.model
            del self.optimizer
            self.model = None
            self.optimizer = None
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def get_data_size(self) -> int:
        """获取客户端训练数据量"""
        return len(self.train_dataset)

    def train_local_model(self, verbose: bool = False) -> Dict[str, float]:
        """
        执行本地训练

        Returns:
            metrics: 训练指标
        """
        self._ensure_model_initialized()
        self.model.train()

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        total_loss = 0.0
        total_samples = 0

        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            epoch_samples = 0

            for batch in train_loader:
                item_seq = batch['item_seq'].to(self.device)
                target_item = batch['target_item'].to(self.device)

                # 生成负样本
                batch_size = item_seq.size(0)
                negative_items = torch.randint(
                    1, self.num_items + 1,
                    (batch_size, self.num_negatives),
                    device=self.device
                )

                # 创建padding mask
                padding_mask = (item_seq != 0)

                # 计算损失
                loss, metrics = self.model.compute_loss(
                    item_seq,
                    target_item,
                    negative_items,
                    padding_mask
                )

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() * batch_size
                epoch_samples += batch_size

            total_loss += epoch_loss
            total_samples += epoch_samples

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

        return {'loss': avg_loss}

    def evaluate(
        self,
        user_sequences: Optional[Dict[int, List[int]]] = None,
        split: str = "test",
        k_list: List[int] = [5, 10, 20]
    ) -> Dict[str, float]:
        """
        评估模型

        Args:
            user_sequences: 完整用户序列（用于负采样评估）
            split: 'val' 或 'test'
            k_list: Top-K列表

        Returns:
            metrics: 评估指标
        """
        # 根据配置选择评估方式
        if self.use_negative_sampling and user_sequences is not None:
            return self.evaluate_negative_sampling(user_sequences, split, k_list)
        else:
            return self.evaluate_full_ranking(split, k_list)

    def evaluate_full_ranking(
        self,
        split: str = "test",
        k_list: List[int] = [5, 10, 20]
    ) -> Dict[str, float]:
        """
        全排序评估（对所有物品进行排序）
        """
        self._ensure_model_initialized()
        self.model.eval()

        dataset = self.val_dataset if split == "val" else self.test_dataset

        all_hr = {k: [] for k in k_list}
        all_ndcg = {k: [] for k in k_list}

        with torch.no_grad():
            batch = dataset[0]
            item_seq = batch['item_seq'].unsqueeze(0).to(self.device)
            target_item = batch['target_item'].item()

            # 创建padding mask
            padding_mask = (item_seq != 0)

            # 预测所有物品的分数
            scores = self.model.predict(item_seq, padding_mask=padding_mask)
            scores = scores.squeeze(0).cpu().numpy()

            # 排序
            ranked_items = np.argsort(-scores)

            # 计算指标
            for k in k_list:
                top_k = ranked_items[:k]
                hit = int(target_item in top_k)
                all_hr[k].append(hit)

                if hit:
                    rank = np.where(top_k == target_item)[0][0]
                    ndcg = 1.0 / np.log2(rank + 2)
                else:
                    ndcg = 0.0
                all_ndcg[k].append(ndcg)

        # 计算平均指标
        metrics = {}
        for k in k_list:
            metrics[f'HR@{k}'] = np.mean(all_hr[k])
            metrics[f'NDCG@{k}'] = np.mean(all_ndcg[k])

        return metrics

    def evaluate_negative_sampling(
        self,
        user_sequences: Dict[int, List[int]],
        split: str = "test",
        k_list: List[int] = [5, 10, 20]
    ) -> Dict[str, float]:
        """
        使用1:100负采样评估模型（对齐SASRec论文的评估协议）

        Args:
            user_sequences: 完整用户序列字典
            split: 'val' 或 'test'
            k_list: Top-K列表

        Returns:
            metrics: 评估指标
        """
        self._ensure_model_initialized()
        self.model.eval()

        dataset = self.val_dataset if split == "val" else self.test_dataset

        # 准备候选池（排除用户历史）
        user_id = self.client_id
        full_sequence = user_sequences[user_id]
        user_items = set(full_sequence)
        all_items = set(range(1, self.num_items + 1))
        candidate_pool = list(all_items - user_items)

        all_hr = {k: [] for k in k_list}
        all_ndcg = {k: [] for k in k_list}

        with torch.no_grad():
            batch = dataset[0]
            item_seq = batch['item_seq'].unsqueeze(0).to(self.device)
            target_item = batch['target_item'].item()

            # 采样负样本
            if len(candidate_pool) < self.num_negatives_eval:
                negative_items = candidate_pool + list(np.random.choice(
                    candidate_pool,
                    self.num_negatives_eval - len(candidate_pool),
                    replace=True
                ))
            else:
                negative_items = list(np.random.choice(
                    candidate_pool,
                    self.num_negatives_eval,
                    replace=False
                ))

            # 候选物品 = 正样本 + 负样本
            candidate_items = [target_item] + negative_items
            candidate_tensor = torch.tensor(candidate_items, dtype=torch.long).unsqueeze(0).to(self.device)

            # 创建padding mask
            padding_mask = (item_seq != 0)

            # 预测候选物品分数
            scores = self.model.predict(item_seq, candidate_tensor, padding_mask)
            scores = scores.squeeze(0).cpu().numpy()

            # 排序（正样本在索引0）
            ranked_indices = np.argsort(-scores)
            rank = np.where(ranked_indices == 0)[0][0] + 1

            # 计算指标
            for k in k_list:
                hit = int(rank <= k)
                all_hr[k].append(hit)

                if hit:
                    ndcg = 1.0 / np.log2(rank + 1)
                else:
                    ndcg = 0.0
                all_ndcg[k].append(ndcg)

        # 计算平均指标
        metrics = {}
        for k in k_list:
            metrics[f'HR@{k}'] = np.mean(all_hr[k])
            metrics[f'NDCG@{k}'] = np.mean(all_ndcg[k])

        return metrics


if __name__ == "__main__":
    print("=" * 70)
    print("Testing FederatedSASRecClient")
    print("=" * 70)

    # 创建测试客户端
    user_sequence = [1, 5, 10, 20, 30, 50, 100, 150]
    client = FederatedSASRecClient(
        client_id=0,
        user_sequence=user_sequence,
        num_items=1000,
        device='cpu',
        hidden_dim=64,
        num_blocks=2,
        num_heads=2,
        max_seq_len=50,
        batch_size=4,
        local_epochs=1,
        num_negatives=10,
        use_negative_sampling=True,
        num_negatives_eval=100
    )

    print(f"\n✓ 创建客户端: user_id={client.client_id}")
    print(f"  序列长度: {len(user_sequence)}")
    print(f"  训练样本: {len(client.train_dataset)}")

    # 测试训练
    print(f"\n训练测试:")
    train_metrics = client.train_local_model(verbose=True)
    print(f"  训练损失: {train_metrics['train_loss']:.4f}")

    # 测试评估（负采样）
    print(f"\n评估测试（负采样）:")
    user_sequences = {0: user_sequence}
    eval_metrics = client.evaluate(user_sequences, split="test", k_list=[5, 10])
    for key, value in eval_metrics.items():
        print(f"  {key}: {value:.4f}")

    # 测试参数传输
    print(f"\n参数传输测试:")
    params = client.get_model_parameters()
    print(f"  参数数量: {len(params)}")
    client.set_model_parameters(params)
    print(f"  ✓ 参数设置成功")

    print("\n" + "=" * 70)
    print("所有测试通过!")
    print("=" * 70)
