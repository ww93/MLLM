"""
FedMem Client: Federated Learning Client with Local Dynamic Memory

带本地动态记忆的联邦学习客户端
- 集成LocalDynamicMemory
- Surprise-based记忆更新
- Memory Prototypes提取与聚合
"""

import os
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
        # 记忆参数 (Two-tier: ST + LT)
        memory_capacity: int = 200,         # LT (long-term) 容量 (推荐200 for ML-1M)
        surprise_threshold: float = 0.5,    # 兼容参数，新版本主要使用novelty
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
            memory_capacity: LT (long-term) 记忆容量，推荐200 (ML-1M)
            surprise_threshold: 兼容参数，新版本主要使用novelty-based写入
            contrastive_lambda: 对比学习损失权重
            num_memory_prototypes: 记忆原型数量（从LT提取）
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
        # 调试开关：默认关闭，可通过环境变量 FEDMEM_DEBUG=1 打开
        self._debug = bool(int(os.environ.get('FEDMEM_DEBUG', '0')))


        # [NEW] 存储多模态特征
        self.item_visual_feats = item_visual_feats
        self.item_text_feats = item_text_feats

        # [FIX 3] 完整性检查：验证多模态特征是否正确加载
        if getattr(self, '_debug', False) and client_id == 0:  # 只在第一个客户端打印，避免日志过多
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
        # [Critical Fix] 缓存用户历史交互集合，用于负采样时排除
        self.user_items = set(user_sequence)  # 快速查找O(1)
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

        # 【FedMem核心】初始化本地动态记忆 (Two-tier: ST + LT)
        # - ST (short-term): FIFO, capacity=50, 捕获最近兴趣
        # - LT (long-term): novelty-gated, capacity=memory_capacity, 稳定多样性存储

        # [FIX] 推断特征维度，用于empty memory时返回正确形状的零张量
        id_emb_dim = getattr(model, 'sasrec_hidden_dim', 128)  # 从模型获取ID嵌入维度
        visual_emb_dim = item_visual_feats.shape[1] if item_visual_feats is not None else 512
        text_emb_dim = item_text_feats.shape[1] if item_text_feats is not None else 384

        self.local_memory = LocalDynamicMemory(
            capacity=memory_capacity,           # LT容量 (推荐200)
            surprise_threshold=surprise_threshold,  # 兼容参数
            device=device,
            # [FIX] 传入特征维度，确保empty memory时返回正确形状
            id_emb_dim=id_emb_dim,
            visual_emb_dim=visual_emb_dim,
            text_emb_dim=text_emb_dim
            # 其他参数使用数据驱动的默认值 (见local_dynamic_memory.py)
        )

    def _ensure_model_initialized(self):
        """确保模型已初始化（延迟实例化）"""
        if self.model is None:
            self.model = copy.deepcopy(self._model_reference).to(self.device)

            # [方案2调试] 验证客户端模型的维度
            if self.client_id == 0 and hasattr(self.model, 'visual_expert'):
                print(f"\n[方案2调试] 客户端 {self.client_id} 模型维度验证:")
                print(f"  preserve_multimodal_dim: {self.model.preserve_multimodal_dim}")
                print(f"  visual_expert.output_dim: {self.model.visual_expert.output_dim}")
                print(f"  semantic_expert.output_dim: {self.model.semantic_expert.output_dim}")
                print(f"  vis_layernorm.normalized_shape: {self.model.vis_layernorm.normalized_shape}")
                print(f"  sem_layernorm.normalized_shape: {self.model.sem_layernorm.normalized_shape}")

            # [优化4] 冻结embeddings后，只优化requires_grad=True的参数
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = optim.Adam(
                trainable_params,
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

    def freeze_embeddings_for_alignment(self):
        """
        [新策略] 冻结ID Embedding，训练多模态投影层以对齐到ID空间

        核心思想:
        - 预训练的ID embedding已经学到了良好的物品表示空间
        - 冻结ID embedding，防止多模态特征破坏这个空间
        - 训练visual_proj和text_proj，让多模态特征对齐到ID空间

        冻结策略:
        - 冻结: item_embedding, positional_embedding (保持ID空间稳定)
        - 保持可训练: Transformer blocks, visual_proj, text_proj, Router, Experts

        适用场景: 有高质量的预训练ID embedding时使用

        调用时机: 在加载预训练权重后立即调用
        """
        self._ensure_model_initialized()

        frozen_params = []
        trainable_params = []

        for name, param in self.model.named_parameters():
            # 只冻结embedding层（ID空间）
            if 'item_emb' in name.lower() or 'positional_emb' in name.lower():
                param.requires_grad = False
                frozen_params.append(name)
            else:
                # 其他层全部保持可训练（包括投影层、Transformer、Router、Experts）
                param.requires_grad = True
                trainable_params.append(name)

        # 重新创建优化器（只包含可训练参数）
        trainable_params_list = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(
            trainable_params_list,
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        print(f"[对齐策略] 客户端 {self.client_id} - 冻结ID Embedding，训练投影层:")
        print(f"  ❄️  冻结参数数: {len(frozen_params)}")
        print(f"  🔥 可训练参数数: {len(trainable_params)}")
        if len(frozen_params) > 0:
            print(f"  冻结层: {', '.join(frozen_params[:5])}")

        # 统计可训练的投影层参数
        proj_params = [name for name in trainable_params if 'proj' in name.lower()]
        if proj_params:
            print(f"  ✓ 投影层可训练: {len(proj_params)}个 (用于对齐到ID空间)")

    def freeze_embeddings(self):
        """
        [已废弃] 完全冻结embedding层

        注意: 此方法已被 freeze_embeddings_for_alignment() 替代
        新方法允许投影层训练，效果更好
        """
        # 兼容性保留，调用新方法
        self.freeze_embeddings_for_alignment()

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
        在本地数据上训练模型，同时更新动态记忆。

        训练采用 **显式负采样**（与经典 SASRec / NCF 评估协议一致）：
        - 对每个正样本采样 num_negatives 个负样本，构造候选集 [pos + negs]
        - logits: [B, 1+N]，标签恒为 0
        - 训练时可选加入“模态对齐损失”（Stage 2/3）：让多模态融合表示对齐冻结的 ID embedding 空间

        Args:
            verbose: 是否打印训练信息（默认关闭；建议仅 client_id==0 打开）

        Returns:
            dict: 训练指标
        """
        self._ensure_model_initialized()
        self.model.train()

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        total_rec_loss = 0.0
        total_align_loss = 0.0
        num_batches = 0

        for _ in range(self.local_epochs):
            epoch_rec_loss = 0.0
            epoch_align_loss = 0.0

            for batch in train_loader:
                user_ids = batch['user_id'].tolist()
                item_seqs = batch['item_seq'].to(self.device)
                target_items = batch['target_item'].to(self.device)  # [B]
                bsz = target_items.size(0)

                # 1) 负采样：构造候选集 [B, 1+N]
                neg_items = self._negative_sampling(batch_size=bsz, target_items=target_items)  # [B, N]
                candidate_items = torch.cat([target_items.unsqueeze(1), neg_items], dim=1)      # [B, 1+N]
                labels = torch.zeros(bsz, dtype=torch.long, device=self.device)                # 正样本恒在第0列

                # 2) 记忆检索（可为空）
                memory_visual, memory_text = self._retrieve_multimodal_memory_batch(
                    batch_size=bsz,
                    top_k=20
                )

                # 3) 候选多模态特征（可为空）
                cand_visual = self._get_candidate_visual_features(candidate_items)
                cand_text = self._get_candidate_text_features(candidate_items)

                # 4) 前向 + 损失
                self.optimizer.zero_grad()

                with torch.amp.autocast('cuda', enabled=(self.scaler is not None)):
                    logits, info = self.model(
                        user_ids=user_ids,
                        input_seq=item_seqs,
                        target_items=candidate_items,   # [B, 1+N]
                        memory_visual=memory_visual,
                        memory_text=memory_text,
                        target_visual=cand_visual,
                        target_text=cand_text,
                        return_components=True,
                        training_mode=False             # 显式负采样：必须 False
                    )

                    # 推荐损失
                    lb_loss = info.get('lb_loss', None) if isinstance(info, dict) else None
                    rec_loss, _ = self.model.compute_loss(logits, labels, lb_loss=None)

                    # “模态对齐”损失（Stage 2/3）：默认用 contrastive_lambda 作为权重
                    # 取模型返回的 fused_repr / auxiliary_repr / seq_out (优先 fused_repr)
                    align_loss = torch.tensor(0.0, device=self.device)
                    if self.contrastive_lambda > 0.0 and isinstance(info, dict):
                        rep = info.get('fused_repr', None)
                        if rep is None:
                            rep = info.get('auxiliary_repr', None)
                        if rep is None:
                            rep = info.get('seq_out', None)

                        if rep is not None:
                            # rep: [B, 1+N, D] -> 正样本为第0列
                            pos_rep = rep[:, 0, :] if rep.dim() == 3 else rep  # [B, D]

                            # 冻结的 ID embedding 作为锚点
                            id_emb = self._get_item_id_emb_batch(target_items)  # [B, D] 或 None
                            if id_emb is not None:
                                pos_rep_n = torch.nn.functional.normalize(pos_rep, dim=-1)
                                id_emb_n = torch.nn.functional.normalize(id_emb, dim=-1)
                                cos = (pos_rep_n * id_emb_n).sum(dim=-1)  # [B]
                                # surprise 加权（困难样本更强调对齐）
                                # rec_loss 是 batch mean；这里用 per-sample 的 CE loss 作为 surprise 的近似
                                with torch.no_grad():
                                    per_sample_ce = -torch.log_softmax(logits.detach(), dim=1)[:, 0]
                                    surprise = torch.sigmoid(per_sample_ce)  # [B] in (0,1)
                                weights = 1.0 + 0.5 * surprise
                                align_loss = ((1.0 - cos) * weights).mean()

                    lb = lb_loss if lb_loss is not None else torch.tensor(0.0, device=self.device)
                    loss = rec_loss + self.contrastive_lambda * align_loss + 0.01 * lb

                # 5) 反向传播
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                # 6) Two-tier Memory Update（ST always, LT when novelty is high）
                with torch.no_grad():
                    per_sample_loss = -torch.log_softmax(logits, dim=1)[:, 0]  # [B]
                    for i in range(bsz):
                        item_id = int(target_items[i].item())
                        loss_val = float(per_sample_loss[i].item())
                        # [Memory Update] 新版本参数顺序: (item_id, id_emb, visual_emb, text_emb, loss_val)
                        self.local_memory.update(
                            item_id=item_id,
                            id_emb=self._get_item_id_emb(item_id),
                            visual_emb=self._get_item_img_emb(item_id),  # 参数名从 img_emb 改为 visual_emb
                            text_emb=self._get_item_text_emb(item_id),
                            loss_val=loss_val
                        )

                epoch_rec_loss += float(rec_loss.item())
                epoch_align_loss += float(align_loss.item())
                num_batches += 1

            epoch_rec_loss /= max(1, len(train_loader))
            epoch_align_loss /= max(1, len(train_loader))
            total_rec_loss += epoch_rec_loss
            total_align_loss += epoch_align_loss

        avg_rec_loss = total_rec_loss / max(1, self.local_epochs)
        avg_align_loss = total_align_loss / max(1, self.local_epochs)
        avg_total_loss = avg_rec_loss + self.contrastive_lambda * avg_align_loss

        metrics = {
            'loss': avg_total_loss,
            'rec_loss': avg_rec_loss,
            # 保持原字段名，避免 server 端日志/画图断掉
            'contrastive_loss': avg_align_loss,
            'memory_size': len(self.local_memory),
            # [Two-tier兼容] total_updates = ST updates + LT updates
            'memory_updates': self.local_memory.total_updates_st + self.local_memory.total_updates_lt
        }

        if verbose:
            print(
                f"Client {self.client_id} | Loss: {avg_total_loss:.4f} "
                f"(Rec: {avg_rec_loss:.4f}, Align: {avg_align_loss:.4f}) | "
                f"Memory: {len(self.local_memory)}/{self.local_memory.capacity}"
            )

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
        【FedDMMR专用】从本地记忆中批量检索多模态特征（Two-tier: ST + LT）

        Args:
            batch_size: 批大小
            top_k: 返回Top-K个记忆（默认从ST和LT混合检索）

        Returns:
            memory_visual: [B, TopK, img_dim] 或 None
            memory_text: [B, TopK, text_dim] 或 None

        Note:
            新版本memory返回4个值 (mem_vis, mem_txt, mem_id, mask)，
            此wrapper方法只返回前2个以保持向后兼容性。
        """
        # [Memory Retrieval] 新版本返回4个值：(mem_vis, mem_txt, mem_id, mask)
        mem_vis, mem_txt, mem_id, mask = self.local_memory.retrieve_multimodal_memory_batch(
            batch_size=batch_size,
            top_k=top_k
        )

        # 向后兼容：只返回visual和text（忽略mem_id和mask）
        # 如果需要mask或id_emb，可以扩展此接口
        return mem_vis, mem_txt

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

    def _get_item_id_emb_batch(self, item_ids: torch.Tensor) -> Optional[torch.Tensor]:
        """
        批量获取物品的 ID embedding（用于 Stage 2/3 的对齐损失）

        Args:
            item_ids: [B] 或 [B, 1] 的 item ids

        Returns:
            [B, D] 或 None
        """
        self._ensure_model_initialized()
        if item_ids is None:
            return None
        if item_ids.dim() > 1:
            item_ids = item_ids.view(-1)
        item_ids = item_ids.to(self.device)

        # 优先使用模型提供的接口
        if hasattr(self.model, 'get_item_embeddings'):
            with torch.no_grad():
                emb = self.model.get_item_embeddings(item_ids, embedding_type='id')
            if emb is not None:
                if emb.dim() == 3:
                    emb = emb.squeeze(1)
                return emb

        # 回退：访问 SASRec 内部 embedding
        try:
            if hasattr(self.model, 'sasrec') and hasattr(self.model.sasrec, 'item_embedding'):
                with torch.no_grad():
                    return self.model.sasrec.item_embedding(item_ids)
        except Exception:
            return None
        return None

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
        [Critical Fix] 负采样：排除用户历史交互的所有物品

        在联邦单用户客户端场景下，必须排除用户的完整历史交互，而不仅仅是target_item。
        否则会产生"伪负样本"：用户交互过的物品被当作负样本，破坏训练信号。

        Args:
            batch_size: 批大小
            target_items: [B] 正样本item IDs

        Returns:
            neg_items: [B, num_negatives] 保证不在用户历史中的负样本
        """
        # [Critical Fix] 一次性生成所有负样本（过采样10倍以确保足够）
        # 因为需要排除用户历史，可能需要多次采样
        all_candidates = torch.randint(
            1, self.num_items,
            (batch_size, self.num_negatives * 10),  # 10倍过采样
            device=self.device
        )  # [B, num_negatives*10]

        # [Critical Fix] 创建用户历史物品的mask
        # 对于联邦学习，batch内所有样本都来自同一用户，使用相同的user_items
        user_items_tensor = torch.tensor(list(self.user_items), device=self.device)  # [|history|]

        # 对于每个样本，选择不在用户历史中的负样本
        neg_items = []
        for i in range(batch_size):
            candidates = all_candidates[i]  # [num_negatives*10]

            # [Critical Fix] 排除用户历史：使用set membership check
            # 方法：将候选转为CPU numpy，快速过滤，再转回GPU
            candidates_np = candidates.cpu().numpy()
            valid_mask = np.array([item not in self.user_items for item in candidates_np])
            valid_negs = candidates[torch.from_numpy(valid_mask)]

            if len(valid_negs) >= self.num_negatives:
                # 有足够的有效负样本
                neg_items.append(valid_negs[:self.num_negatives])
            else:
                # [极少情况] 不够，继续采样直到足够
                # 这种情况在用户历史很长时可能发生
                collected = valid_negs.tolist()
                while len(collected) < self.num_negatives:
                    # 采样单个候选并检查
                    candidate = torch.randint(1, self.num_items, (1,), device=self.device).item()
                    if candidate not in self.user_items:
                        collected.append(candidate)

                neg_items.append(torch.tensor(collected[:self.num_negatives], device=self.device))

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
