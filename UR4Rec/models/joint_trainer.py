"""
Joint Trainer for UR4Rec V2

多阶段联合训练框架：
1. 预训练阶段：分别训练 SASRec 和检索器
2. 联合微调阶段：固定一部分，训练另一部分
3. 端到端优化：所有模块一起训练
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np

from .ur4rec_v2 import UR4RecV2
from .multimodal_retriever import MultiModalPreferenceRetriever
from .multimodal_loss import MultiModalRetrievalLoss, JointLoss, UncertaintyWeightedLoss


class JointTrainer:
    """
    联合训练器：协调 SASRec 和多模态检索器的训练

    训练阶段：
    1. Stage 1: 预训练 SASRec
    2. Stage 2: 预训练检索器（固定 SASRec）
    3. Stage 3: 联合微调（交替训练）
    4. Stage 4: 端到端优化
    """

    def __init__(
        self,
        model: Union[UR4RecV2, MultiModalPreferenceRetriever],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        # 优化器参数
        sasrec_lr: float = 1e-3,
        retriever_lr: float = 1e-4,
        weight_decay: float = 1e-5,
        # 损失函数参数
        use_multimodal: bool = False,
        retrieval_loss_weight: float = 1.0,
        consistency_weight: float = 0.1,
        contrastive_weight: float = 0.1,
        diversity_weight: float = 0.01,
        # 训练策略
        use_uncertainty_weighting: bool = False,
        gradient_clip: float = 1.0,
        warmup_steps: int = 100
    ):
        """
        Args:
            model: UR4RecV2 或 MultiModalPreferenceRetriever
            device: 设备
            sasrec_lr: SASRec 学习率
            retriever_lr: 检索器学习率
            weight_decay: 权重衰减
            use_multimodal: 是否使用多模态
            retrieval_loss_weight: 检索损失权重
            consistency_weight: 一致性损失权重
            contrastive_weight: 对比学习损失权重
            diversity_weight: 多样性损失权重
            use_uncertainty_weighting: 是否使用不确定性加权
            gradient_clip: 梯度裁剪
            warmup_steps: 预热步数
        """
        self.model = model
        self.device = device
        self.use_multimodal = use_multimodal
        self.gradient_clip = gradient_clip
        self.warmup_steps = warmup_steps

        # 分组参数（SASRec vs 检索器）
        self.sasrec_params = list(model.sasrec.parameters())

        if hasattr(model, 'preference_retriever'):
            self.retriever_params = list(model.preference_retriever.parameters())
        else:
            self.retriever_params = list(model.parameters())

        # 优化器
        self.sasrec_optimizer = optim.AdamW(
            self.sasrec_params,
            lr=sasrec_lr,
            weight_decay=weight_decay
        )

        self.retriever_optimizer = optim.AdamW(
            self.retriever_params,
            lr=retriever_lr,
            weight_decay=weight_decay
        )

        # 学习率调度器
        self.sasrec_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.sasrec_optimizer, T_max=1000
        )

        self.retriever_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.retriever_optimizer, T_max=1000
        )

        # 损失函数
        if use_uncertainty_weighting:
            # 自动任务加权
            loss_components = ['retrieval', 'sasrec']
            if use_multimodal:
                loss_components.extend(['consistency', 'contrastive', 'diversity'])

            self.criterion = UncertaintyWeightedLoss(
                loss_components=loss_components,
                device=device
            )
        else:
            # 手动权重
            if use_multimodal:
                self.criterion = MultiModalRetrievalLoss(
                    retrieval_loss_weight=retrieval_loss_weight,
                    consistency_weight=consistency_weight,
                    contrastive_weight=contrastive_weight,
                    diversity_weight=diversity_weight,
                    device=device
                )
            else:
                self.criterion = JointLoss(
                    sasrec_weight=0.5,
                    retriever_weight=0.5
                )

        # 训练状态
        self.current_stage = "pretrain_sasrec"
        self.global_step = 0
        self.best_metrics = {}

        # 历史记录
        self.train_history = {
            'loss': [],
            'sasrec_loss': [],
            'retriever_loss': [],
            'metrics': []
        }

    def set_training_stage(self, stage: str):
        """
        设置训练阶段

        Args:
            stage: 'pretrain_sasrec' | 'pretrain_retriever' | 'joint_finetune' | 'end_to_end'
        """
        self.current_stage = stage

        if stage == "pretrain_sasrec":
            # 只训练 SASRec
            for param in self.sasrec_params:
                param.requires_grad = True
            for param in self.retriever_params:
                param.requires_grad = False
            print("阶段：预训练 SASRec")

        elif stage == "pretrain_retriever":
            # 只训练检索器
            for param in self.sasrec_params:
                param.requires_grad = False
            for param in self.retriever_params:
                param.requires_grad = True
            print("阶段：预训练检索器")

        elif stage == "joint_finetune":
            # 交替训练（本 batch 决定训练哪个）
            print("阶段：联合微调（交替训练）")

        elif stage == "end_to_end":
            # 所有参数都训练
            for param in self.sasrec_params:
                param.requires_grad = True
            for param in self.retriever_params:
                param.requires_grad = True
            print("阶段：端到端训练")

        else:
            raise ValueError(f"未知的训练阶段: {stage}")

    def _warmup_lr(self, step: int, base_lr: float) -> float:
        """学习率预热"""
        if step < self.warmup_steps:
            return base_lr * (step + 1) / self.warmup_steps
        return base_lr

    def _update_learning_rate(self):
        """更新学习率（预热 + 调度）"""
        if self.global_step < self.warmup_steps:
            # 预热阶段
            warmup_sasrec_lr = self._warmup_lr(
                self.global_step,
                self.sasrec_optimizer.defaults['lr']
            )
            warmup_retriever_lr = self._warmup_lr(
                self.global_step,
                self.retriever_optimizer.defaults['lr']
            )

            for param_group in self.sasrec_optimizer.param_groups:
                param_group['lr'] = warmup_sasrec_lr
            for param_group in self.retriever_optimizer.param_groups:
                param_group['lr'] = warmup_retriever_lr
        else:
            # 调度器
            self.sasrec_scheduler.step()
            self.retriever_scheduler.step()

    def train_step(
        self,
        batch: Dict,
        alternate_training: bool = False
    ) -> Dict:
        """
        单步训练

        Args:
            batch: 批次数据
            alternate_training: 是否交替训练（仅在 joint_finetune 阶段）

        Returns:
            metrics: 训练指标
        """
        self.model.train()

        # 数据移到设备
        user_ids = batch['user_ids']
        input_seq = batch['input_seq'].to(self.device)
        target_items = batch['target_items'].to(self.device)
        negative_items = batch['negative_items'].to(self.device)
        seq_padding_mask = batch.get('seq_padding_mask', None)
        if seq_padding_mask is not None:
            seq_padding_mask = seq_padding_mask.to(self.device)

        # 决定训练哪个模块
        train_sasrec = True
        train_retriever = True

        if self.current_stage == "pretrain_sasrec":
            train_retriever = False
        elif self.current_stage == "pretrain_retriever":
            train_sasrec = False
        elif self.current_stage == "joint_finetune" and alternate_training:
            # 交替训练：根据步数奇偶性
            if self.global_step % 2 == 0:
                train_retriever = False
            else:
                train_sasrec = False

        # 前向传播
        if hasattr(self.model, 'sasrec'):
            # UR4RecV2
            # 构造候选物品（正样本 + 负样本）
            batch_size = target_items.size(0)
            num_negatives = negative_items.size(1)

            candidate_items = torch.cat([
                target_items.unsqueeze(1),  # [batch, 1]
                negative_items  # [batch, num_neg]
            ], dim=1)  # [batch, 1+num_neg]

            # 标签（第一个是正样本）
            labels = torch.zeros(batch_size, 1 + num_negatives, device=self.device)
            labels[:, 0] = 1.0

            # 前向传播
            final_scores, scores_dict = self.model(
                user_ids=user_ids,
                input_seq=input_seq,
                candidate_items=candidate_items,
                seq_padding_mask=seq_padding_mask
            )

            # 计算损失
            if isinstance(self.criterion, UncertaintyWeightedLoss):
                # 不确定性加权
                losses = {}

                # 检索损失
                retriever_scores = scores_dict['retriever_scores']
                losses['retrieval'] = nn.functional.binary_cross_entropy_with_logits(
                    retriever_scores, labels
                )

                # SASRec 损失（BPR）
                pos_scores = scores_dict['sasrec_scores'][:, 0]  # [batch]
                neg_scores = scores_dict['sasrec_scores'][:, 1:]  # [batch, num_neg]
                diff = pos_scores.unsqueeze(1) - neg_scores
                losses['sasrec'] = -torch.log(torch.sigmoid(diff) + 1e-10).mean()

                # 多模态损失（如果有）
                if self.use_multimodal and 'text_features' in scores_dict:
                    text_feat = scores_dict['text_features']
                    visual_feat = scores_dict['visual_features']

                    losses['consistency'] = nn.functional.mse_loss(text_feat, visual_feat)

                    # 对比学习
                    sim_matrix = text_feat @ visual_feat.T / 0.07
                    contrastive_labels = torch.arange(text_feat.size(0), device=self.device)
                    losses['contrastive'] = nn.functional.cross_entropy(sim_matrix, contrastive_labels)

                    # 多样性
                    norm_feat = nn.functional.normalize(text_feat, p=2, dim=-1)
                    similarity = norm_feat @ norm_feat.T
                    identity = torch.eye(similarity.size(0), device=self.device)
                    losses['diversity'] = -torch.logdet(similarity + 0.1 * identity)

                total_loss, loss_dict = self.criterion(losses)

            else:
                # 固定权重
                if isinstance(self.criterion, JointLoss):
                    # SASRec BPR 损失
                    pos_scores = scores_dict['sasrec_scores'][:, 0]
                    neg_scores = scores_dict['sasrec_scores'][:, 1:]
                    diff = pos_scores.unsqueeze(1) - neg_scores
                    sasrec_loss = -torch.log(torch.sigmoid(diff) + 1e-10).mean()

                    # 检索损失
                    retriever_scores = scores_dict['retriever_scores']
                    retriever_loss = nn.functional.binary_cross_entropy_with_logits(
                        retriever_scores, labels
                    )

                    total_loss = self.criterion(sasrec_loss, retriever_loss)
                    loss_dict = {
                        'sasrec_loss': sasrec_loss.item(),
                        'retriever_loss': retriever_loss.item()
                    }

                else:
                    # 多模态损失
                    retriever_scores = scores_dict['retriever_scores']

                    # 需要文本和视觉特征（如果有）
                    text_feat = scores_dict.get('text_features', None)
                    visual_feat = scores_dict.get('visual_features', None)
                    fused_feat = scores_dict.get('fused_features', None)

                    if text_feat is not None and visual_feat is not None:
                        total_loss, loss_dict = self.criterion(
                            similarity_scores=retriever_scores,
                            labels=labels,
                            text_features=text_feat,
                            visual_features=visual_feat,
                            fused_features=fused_feat
                        )
                    else:
                        # 只有检索损失
                        total_loss = nn.functional.binary_cross_entropy_with_logits(
                            retriever_scores, labels
                        )
                        loss_dict = {'retrieval_loss': total_loss.item()}

        else:
            # 纯检索器训练
            # TODO: 实现纯检索器的训练逻辑
            raise NotImplementedError("纯检索器训练尚未实现")

        # 反向传播
        if train_sasrec:
            self.sasrec_optimizer.zero_grad()
        if train_retriever:
            self.retriever_optimizer.zero_grad()

        total_loss.backward()

        # 梯度裁剪
        if train_sasrec:
            torch.nn.utils.clip_grad_norm_(self.sasrec_params, self.gradient_clip)
        if train_retriever:
            torch.nn.utils.clip_grad_norm_(self.retriever_params, self.gradient_clip)

        # 更新参数
        if train_sasrec:
            self.sasrec_optimizer.step()
        if train_retriever:
            self.retriever_optimizer.step()

        # 更新学习率
        self._update_learning_rate()

        self.global_step += 1

        # 返回指标
        metrics = {
            'total_loss': total_loss.item(),
            'step': self.global_step,
            'lr_sasrec': self.sasrec_optimizer.param_groups[0]['lr'],
            'lr_retriever': self.retriever_optimizer.param_groups[0]['lr']
        }
        metrics.update(loss_dict)

        return metrics

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int
    ) -> Dict:
        """
        训练一个 epoch

        Args:
            dataloader: 数据加载器
            epoch: 当前 epoch

        Returns:
            avg_metrics: 平均指标
        """
        self.model.train()

        total_metrics = {}
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            metrics = self.train_step(
                batch,
                alternate_training=(self.current_stage == "joint_finetune")
            )

            # 累积指标
            for key, value in metrics.items():
                if key not in total_metrics:
                    total_metrics[key] = 0
                total_metrics[key] += value

            num_batches += 1

            # 更新进度条
            pbar.set_postfix({
                'loss': f"{metrics['total_loss']:.4f}",
                'lr_s': f"{metrics['lr_sasrec']:.2e}",
                'lr_r': f"{metrics['lr_retriever']:.2e}"
            })

        # 计算平均
        avg_metrics = {
            key: value / num_batches
            for key, value in total_metrics.items()
        }

        # 记录历史
        self.train_history['loss'].append(avg_metrics['total_loss'])

        return avg_metrics

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        k_list: List[int] = [5, 10, 20]
    ) -> Dict:
        """
        评估模型

        Args:
            dataloader: 验证集数据加载器
            k_list: Top-K 列表

        Returns:
            metrics: 评估指标
        """
        self.model.eval()

        all_ranks = []
        all_scores = []

        for batch in tqdm(dataloader, desc="Evaluating"):
            user_ids = batch['user_ids']
            input_seq = batch['input_seq'].to(self.device)
            target_items = batch['target_items'].to(self.device)
            candidate_items = batch['candidate_items'].to(self.device)
            seq_padding_mask = batch.get('seq_padding_mask', None)
            if seq_padding_mask is not None:
                seq_padding_mask = seq_padding_mask.to(self.device)

            # 预测
            if hasattr(self.model, 'forward'):
                final_scores, _ = self.model(
                    user_ids=user_ids,
                    input_seq=input_seq,
                    candidate_items=candidate_items,
                    seq_padding_mask=seq_padding_mask
                )
            else:
                final_scores = self.model.predict(input_seq, candidate_items, seq_padding_mask)

            # 计算排名
            batch_size = target_items.size(0)
            for i in range(batch_size):
                target_idx = (candidate_items[i] == target_items[i]).nonzero(as_tuple=True)[0]
                if len(target_idx) > 0:
                    target_idx = target_idx[0]
                    scores = final_scores[i]
                    rank = (scores > scores[target_idx]).sum().item() + 1
                    all_ranks.append(rank)
                    all_scores.append(scores[target_idx].item())

        # 计算指标
        all_ranks = np.array(all_ranks)

        metrics = {}
        for k in k_list:
            metrics[f'hit@{k}'] = (all_ranks <= k).mean()
            metrics[f'ndcg@{k}'] = self._compute_ndcg(all_ranks, k)

        metrics['mrr'] = (1.0 / all_ranks).mean()
        metrics['avg_rank'] = all_ranks.mean()

        return metrics

    def _compute_ndcg(self, ranks: np.ndarray, k: int) -> float:
        """计算 NDCG@K"""
        dcg = np.where(ranks <= k, 1.0 / np.log2(ranks + 1), 0.0)
        idcg = 1.0 / np.log2(2)  # 理想情况：排名第1
        ndcg = (dcg / idcg).mean()
        return ndcg

    def save_checkpoint(
        self,
        save_path: str,
        epoch: int,
        metrics: Dict
    ):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'sasrec_optimizer_state_dict': self.sasrec_optimizer.state_dict(),
            'retriever_optimizer_state_dict': self.retriever_optimizer.state_dict(),
            'sasrec_scheduler_state_dict': self.sasrec_scheduler.state_dict(),
            'retriever_scheduler_state_dict': self.retriever_scheduler.state_dict(),
            'metrics': metrics,
            'current_stage': self.current_stage,
            'train_history': self.train_history
        }

        if isinstance(self.criterion, UncertaintyWeightedLoss):
            checkpoint['uncertainty_weights'] = self.criterion.log_vars.data

        torch.save(checkpoint, save_path)
        print(f"检查点已保存至: {save_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.sasrec_optimizer.load_state_dict(checkpoint['sasrec_optimizer_state_dict'])
        self.retriever_optimizer.load_state_dict(checkpoint['retriever_optimizer_state_dict'])
        self.sasrec_scheduler.load_state_dict(checkpoint['sasrec_scheduler_state_dict'])
        self.retriever_scheduler.load_state_dict(checkpoint['retriever_scheduler_state_dict'])

        self.global_step = checkpoint['global_step']
        self.current_stage = checkpoint['current_stage']
        self.train_history = checkpoint['train_history']

        if isinstance(self.criterion, UncertaintyWeightedLoss):
            self.criterion.log_vars.data = checkpoint['uncertainty_weights']

        print(f"检查点已加载: {checkpoint_path}")
        print(f"  Epoch: {checkpoint['epoch']}")
        print(f"  Step: {self.global_step}")
        print(f"  Stage: {self.current_stage}")


if __name__ == "__main__":
    print("测试 JointTrainer...")

    # 创建模型
    from .ur4rec_v2 import UR4RecV2

    model = UR4RecV2(
        num_items=1000,
        sasrec_hidden_dim=128,
        text_embedding_dim=384,
        retriever_output_dim=128
    )

    # 创建训练器
    trainer = JointTrainer(
        model=model,
        device='cpu',
        use_uncertainty_weighting=True
    )

    print(f"当前阶段: {trainer.current_stage}")
    print(f"SASRec 参数: {len(trainer.sasrec_params)}")
    print(f"检索器参数: {len(trainer.retriever_params)}")

    # 测试不同训练阶段
    for stage in ['pretrain_sasrec', 'pretrain_retriever', 'joint_finetune', 'end_to_end']:
        print(f"\n切换到阶段: {stage}")
        trainer.set_training_stage(stage)

        # 检查参数梯度
        sasrec_trainable = sum(p.requires_grad for p in trainer.sasrec_params)
        retriever_trainable = sum(p.requires_grad for p in trainer.retriever_params)

        print(f"  SASRec 可训练参数: {sasrec_trainable}/{len(trainer.sasrec_params)}")
        print(f"  检索器可训练参数: {retriever_trainable}/{len(trainer.retriever_params)}")
