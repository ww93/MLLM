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
from .training_strategies import (
    AdaptiveAlternatingTrainer,
    CurriculumWeightScheduler,
    MemoryBankContrastiveLoss,
    BidirectionalKnowledgeDistillation
)


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
        warmup_steps: int = 100,
        # 改进策略开关
        use_adaptive_alternating: bool = True,
        use_curriculum_learning: bool = False,
        use_memory_bank: bool = False,
        use_knowledge_distillation: bool = False,
        # 改进策略参数
        adaptive_switch_threshold: float = 0.01,
        adaptive_min_steps: int = 5,
        memory_bank_size: int = 65536,
        kd_temperature: float = 4.0,
        kd_weight: float = 0.1
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
            use_adaptive_alternating: 是否使用自适应交替训练
            use_curriculum_learning: 是否使用课程学习
            use_memory_bank: 是否使用 Memory Bank 对比学习
            use_knowledge_distillation: 是否使用知识蒸馏
            adaptive_switch_threshold: 自适应切换阈值
            adaptive_min_steps: 自适应最小步数
            memory_bank_size: Memory Bank 大小
            kd_temperature: 知识蒸馏温度
            kd_weight: 知识蒸馏权重
        """
        self.model = model
        self.device = device
        self.use_multimodal = use_multimodal
        self.gradient_clip = gradient_clip
        self.warmup_steps = warmup_steps

        # 改进策略开关
        self.use_adaptive_alternating = use_adaptive_alternating
        self.use_curriculum_learning = use_curriculum_learning
        self.use_memory_bank = use_memory_bank
        self.use_knowledge_distillation = use_knowledge_distillation
        self.kd_weight = kd_weight

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
                num_tasks=len(loss_components)
            )
        else:
            # 手动权重
            if use_multimodal:
                self.criterion = MultiModalRetrievalLoss(
                    temperature=0.07,
                    alpha_consistency=consistency_weight,
                    beta_contrastive=contrastive_weight,
                    gamma_diversity=diversity_weight
                )
            else:
                self.criterion = JointLoss(
                    sasrec_weight=0.5,
                    retriever_weight=0.5
                )

        # 改进策略模块
        if self.use_adaptive_alternating:
            self.adaptive_alternating = AdaptiveAlternatingTrainer(
                switch_threshold=adaptive_switch_threshold,
                min_steps_per_module=adaptive_min_steps
            )
            print(f"✓ 启用自适应交替训练 (阈值: {adaptive_switch_threshold})")
        else:
            self.adaptive_alternating = None

        if self.use_curriculum_learning:
            # 需要知道总步数，在训练时初始化
            self.curriculum_scheduler = None
            print(f"✓ 启用课程学习（将在训练开始时初始化）")
        else:
            self.curriculum_scheduler = None

        if self.use_memory_bank:
            # 获取特征维度
            if hasattr(model, 'preference_retriever'):
                embedding_dim = model.preference_retriever.output_dim
            elif hasattr(model, 'retriever_output_dim'):
                embedding_dim = model.retriever_output_dim
            else:
                embedding_dim = 256  # 默认值

            self.memory_bank = MemoryBankContrastiveLoss(
                memory_size=memory_bank_size,
                feature_dim=embedding_dim,
                device=device
            )
            print(f"✓ 启用 Memory Bank 对比学习 (大小: {memory_bank_size})")
        else:
            self.memory_bank = None

        if self.use_knowledge_distillation:
            self.kd_module = BidirectionalKnowledgeDistillation(
                temperature=kd_temperature
            )
            print(f"✓ 启用知识蒸馏 (温度: {kd_temperature}, 权重: {kd_weight})")
        else:
            self.kd_module = None

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
            # 自适应交替训练 vs 固定交替训练
            if self.use_adaptive_alternating and self.adaptive_alternating is not None:
                # 使用自适应策略（需要先进行一次前向传播获取损失）
                # 这里暂时使用历史损失决定，在反向传播后更新
                pass  # 在反向传播后处理
            else:
                # 传统方式：根据步数奇偶性
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
                target_items=candidate_items,
                seq_padding_mask=seq_padding_mask,
                return_components=True
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

                # SASRec 损失（BPR） - 使用数值稳定的版本
                pos_scores = scores_dict['sasrec_scores'][:, 0]  # [batch]
                neg_scores = scores_dict['sasrec_scores'][:, 1:]  # [batch, num_neg]

                # Debug: 检查分数范围
                if self.global_step % 100 == 0:
                    print(f"\n[DEBUG] Step {self.global_step}:")
                    print(f"  pos_scores: min={pos_scores.min().item():.4f}, max={pos_scores.max().item():.4f}, mean={pos_scores.mean().item():.4f}")
                    print(f"  neg_scores: min={neg_scores.min().item():.4f}, max={neg_scores.max().item():.4f}, mean={neg_scores.mean().item():.4f}")

                # 检查NaN
                if torch.isnan(pos_scores).any() or torch.isnan(neg_scores).any():
                    print(f"\n[ERROR] NaN detected in scores at step {self.global_step}")
                    print(f"  pos_scores has NaN: {torch.isnan(pos_scores).any()}")
                    print(f"  neg_scores has NaN: {torch.isnan(neg_scores).any()}")

                diff = pos_scores.unsqueeze(1) - neg_scores
                # 使用 logsigmoid 更稳定: -log(sigmoid(x)) = log(1 + exp(-x)) = softplus(-x)
                # F.softplus(-x) = log(1 + exp(-x))
                losses['sasrec'] = nn.functional.softplus(-diff).mean()

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

                # Convert dict to tensor in the order expected by UncertaintyWeightedLoss
                loss_components = ['retrieval', 'sasrec']
                if self.use_multimodal:
                    loss_components.extend(['consistency', 'contrastive', 'diversity'])
                losses_tensor = torch.stack([losses[key] for key in loss_components])

                total_loss, weights = self.criterion(losses_tensor)

                # Create loss_dict for logging
                loss_dict = {key: losses[key].item() for key in losses}
                loss_dict['total'] = total_loss.item()

            else:
                # 固定权重
                if isinstance(self.criterion, JointLoss):
                    # 获取scores
                    pos_scores = scores_dict['sasrec_scores'][:, 0]
                    neg_scores = scores_dict['sasrec_scores'][:, 1:]
                    retriever_scores = scores_dict['retriever_scores']

                    # 调用criterion计算损失（传递scores而不是loss）
                    total_loss, loss_dict = self.criterion(
                        sasrec_pos_scores=pos_scores,
                        sasrec_neg_scores=neg_scores,
                        retriever_similarity=retriever_scores,
                        retriever_labels=labels
                    )

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

        # 使用自适应交替训练决定训练哪个模块
        if self.current_stage == "joint_finetune" and alternate_training and \
           self.use_adaptive_alternating and self.adaptive_alternating is not None:
            # 根据损失历史决定训练哪个模块
            sasrec_loss_val = loss_dict.get('sasrec_loss', 0.0)
            retriever_loss_val = loss_dict.get('retriever_loss', 0.0)

            train_module = self.adaptive_alternating.update(
                sasrec_loss=sasrec_loss_val,
                retriever_loss=retriever_loss_val
            )

            train_sasrec = (train_module == "sasrec")
            train_retriever = (train_module == "retriever")

        # 反向传播
        if train_sasrec:
            self.sasrec_optimizer.zero_grad()
        if train_retriever:
            self.retriever_optimizer.zero_grad()

        # 检查loss是否有NaN
        if torch.isnan(total_loss):
            print(f"\n[ERROR] NaN loss detected at step {self.global_step}")
            print(f"  Loss dict: {loss_dict}")
            # 跳过这个batch
            return {
                'total_loss': float('nan'),
                'step': self.global_step,
                'lr_sasrec': self.sasrec_optimizer.param_groups[0]['lr'],
                'lr_retriever': self.retriever_optimizer.param_groups[0]['lr']
            }

        total_loss.backward()

        # 梯度裁剪前检查梯度
        if train_sasrec and self.global_step % 100 == 0:
            sasrec_grad_norm = torch.nn.utils.clip_grad_norm_(self.sasrec_params, float('inf'))
            print(f"  SASRec grad norm before clip: {sasrec_grad_norm:.4f}")

        if train_retriever and self.global_step % 100 == 0:
            retriever_grad_norm = torch.nn.utils.clip_grad_norm_(self.retriever_params, float('inf'))
            print(f"  Retriever grad norm before clip: {retriever_grad_norm:.4f}")

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

        # 添加自适应交替训练的监控信息
        if self.use_adaptive_alternating and self.adaptive_alternating is not None:
            adaptive_stats = self.adaptive_alternating.get_stats()
            metrics['adaptive_current_module'] = adaptive_stats['current_module']
            metrics['adaptive_switch_count'] = adaptive_stats['switch_count']
            metrics['adaptive_steps_since_switch'] = adaptive_stats['steps_since_switch']
            # 添加当前训练的模块标识
            if train_sasrec and not train_retriever:
                metrics['training_module'] = 'sasrec'
            elif train_retriever and not train_sasrec:
                metrics['training_module'] = 'retriever'
            else:
                metrics['training_module'] = 'both'

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

            # 累积指标（只累积数值类型）
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    if key not in total_metrics:
                        total_metrics[key] = 0
                    total_metrics[key] += value

            num_batches += 1

            # 更新进度条
            postfix_dict = {
                'loss': f"{metrics['total_loss']:.4f}",
                'lr_s': f"{metrics['lr_sasrec']:.2e}",
                'lr_r': f"{metrics['lr_retriever']:.2e}"
            }

            # 显示当前训练的模块（自适应交替训练）
            if 'training_module' in metrics:
                module_abbr = {
                    'sasrec': 'SAS',
                    'retriever': 'RET',
                    'both': 'ALL'
                }
                postfix_dict['train'] = module_abbr.get(metrics['training_module'], '???')

            # 显示切换次数
            if 'adaptive_switch_count' in metrics:
                postfix_dict['switch'] = metrics['adaptive_switch_count']

            pbar.set_postfix(postfix_dict)

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
        num_valid_batches = 0
        num_skipped_batches = 0

        for batch in tqdm(dataloader, desc="Evaluating"):
            user_ids = batch['user_ids']
            input_seq = batch['input_seq'].to(self.device)
            target_items = batch['target_items'].to(self.device)
            candidate_items = batch['candidate_items'].to(self.device)
            seq_padding_mask = batch.get('seq_padding_mask', None)
            if seq_padding_mask is not None:
                seq_padding_mask = seq_padding_mask.to(self.device)

            # 预测（不需要components，所以使用默认return_components=False）
            if hasattr(self.model, 'forward'):
                final_scores = self.model(
                    user_ids=user_ids,
                    input_seq=input_seq,
                    target_items=candidate_items,
                    seq_padding_mask=seq_padding_mask,
                    update_memory=False  # 评估时不更新记忆
                )
            else:
                final_scores = self.model.predict(input_seq, candidate_items, seq_padding_mask)

            # 检查NaN并跳过该batch
            if torch.isnan(final_scores).any():
                print(f"[WARNING] NaN detected in evaluation batch, skipping...")
                num_skipped_batches += 1
                continue

            # 计算排名
            num_valid_batches += 1
            batch_size = target_items.size(0)
            ranks_in_batch = 0
            for i in range(batch_size):
                target_idx = (candidate_items[i] == target_items[i]).nonzero(as_tuple=True)[0]
                if len(target_idx) > 0:
                    target_idx = target_idx[0]
                    scores = final_scores[i]
                    rank = (scores > scores[target_idx]).sum().item() + 1
                    all_ranks.append(rank)
                    all_scores.append(scores[target_idx].item())
                    ranks_in_batch += 1

            # Debug: 如果这个batch没有产生任何rank，打印警告
            if ranks_in_batch == 0:
                print(f"[WARNING] Batch processed but no valid ranks found (target not in candidates?)")

        # 计算指标
        print(f"\nEvaluation summary: {num_valid_batches} valid batches, {num_skipped_batches} skipped (NaN), {len(all_ranks)} total samples")

        if len(all_ranks) == 0:
            print(f"[WARNING] No valid evaluation samples. Returning NaN metrics.")
            metrics = {}
            for k in k_list:
                metrics[f'hit@{k}'] = float('nan')
                metrics[f'ndcg@{k}'] = float('nan')
            metrics['mrr'] = float('nan')
            metrics['avg_rank'] = float('nan')
            return metrics

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
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

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
