"""
多模态损失函数

包含多种损失组件用于训练多模态推荐系统
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math


class MultiModalRetrievalLoss(nn.Module):
    """
    多模态检索损失函数

    组合多个损失组件：
    1. 主检索损失（BPR/BCE）
    2. 模态一致性损失
    3. 对比学习损失（InfoNCE）
    4. 多样性正则
    """

    def __init__(
        self,
        temperature: float = 0.07,
        # 损失权重
        alpha_consistency: float = 0.1,
        beta_contrastive: float = 0.2,
        gamma_diversity: float = 0.05,
        # 其他参数
        diversity_eps: float = 1e-6
    ):
        """
        Args:
            temperature: 对比学习温度参数
            alpha_consistency: 模态一致性损失权重
            beta_contrastive: 对比学习损失权重
            gamma_diversity: 多样性正则权重
            diversity_eps: 数值稳定性参数
        """
        super().__init__()

        self.temperature = temperature
        self.alpha_consistency = alpha_consistency
        self.beta_contrastive = beta_contrastive
        self.gamma_diversity = gamma_diversity
        self.diversity_eps = diversity_eps

    def retrieval_loss(
        self,
        similarity_scores: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        主检索损失（二元交叉熵）

        Args:
            similarity_scores: [batch_size, num_candidates] - 相似度分数
            labels: [batch_size, num_candidates] - 标签（1=正样本，0=负样本）
            mask: [batch_size, num_candidates] - 有效位置掩码

        Returns:
            loss: 标量损失
        """
        # BCE with logits
        loss = F.binary_cross_entropy_with_logits(
            similarity_scores,
            labels.float(),
            reduction='none'
        )

        # 应用掩码
        if mask is not None:
            loss = loss * mask.float()
            loss = loss.sum() / mask.sum().clamp(min=1)
        else:
            loss = loss.mean()

        return loss

    def bpr_loss(
        self,
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        BPR (Bayesian Personalized Ranking) 损失

        Args:
            pos_scores: [batch_size] - 正样本分数
            neg_scores: [batch_size, num_negatives] - 负样本分数

        Returns:
            loss: 标量损失
        """
        # -log(sigmoid(pos_score - neg_score))
        diff = pos_scores.unsqueeze(1) - neg_scores  # [batch, num_neg]
        loss = -torch.log(torch.sigmoid(diff) + 1e-10).mean()

        return loss

    def modal_consistency_loss(
        self,
        text_features: torch.Tensor,
        visual_features: torch.Tensor,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        模态一致性损失

        确保文本和视觉偏好表示一致

        Args:
            text_features: [batch_size, dim] - 文本特征
            visual_features: [batch_size, dim] - 视觉特征
            normalize: 是否先归一化

        Returns:
            loss: 标量损失
        """
        if normalize:
            text_features = F.normalize(text_features, p=2, dim=-1)
            visual_features = F.normalize(visual_features, p=2, dim=-1)

        # MSE 损失
        loss = F.mse_loss(text_features, visual_features)

        return loss

    def contrastive_loss(
        self,
        text_features: torch.Tensor,
        visual_features: torch.Tensor,
        temperature: Optional[float] = None
    ) -> torch.Tensor:
        """
        对比学习损失（InfoNCE）

        同一用户的文本和视觉特征应该接近（正样本对）
        不同用户的特征应该远离（负样本对）

        Args:
            text_features: [batch_size, dim] - 文本特征
            visual_features: [batch_size, dim] - 视觉特征
            temperature: 温度参数

        Returns:
            loss: 标量损失
        """
        if temperature is None:
            temperature = self.temperature

        batch_size = text_features.size(0)

        # 归一化
        text_features = F.normalize(text_features, p=2, dim=-1)
        visual_features = F.normalize(visual_features, p=2, dim=-1)

        # 计算相似度矩阵
        # [batch, batch] - (i,j) 表示第i个文本与第j个视觉的相似度
        similarity_matrix = torch.matmul(
            text_features,
            visual_features.T
        ) / temperature

        # 标签：对角线为正样本（同一用户的文本和视觉）
        labels = torch.arange(batch_size, device=text_features.device)

        # 文本 -> 视觉的对比损失
        loss_t2v = F.cross_entropy(similarity_matrix, labels)

        # 视觉 -> 文本的对比损失
        loss_v2t = F.cross_entropy(similarity_matrix.T, labels)

        # 双向对比损失
        loss = (loss_t2v + loss_v2t) / 2

        return loss

    def diversity_loss(
        self,
        preference_vectors: torch.Tensor,
        method: str = "logdet"
    ) -> torch.Tensor:
        """
        多样性正则

        鼓励学习不同方面的用户偏好，避免所有偏好向量相似

        Args:
            preference_vectors: [batch_size, dim] - 偏好向量
            method: 多样性度量方法 ('logdet', 'cosine', 'distance')

        Returns:
            loss: 标量损失（负值，最小化时增加多样性）
        """
        # 归一化
        preference_vectors = F.normalize(preference_vectors, p=2, dim=-1)

        if method == "logdet":
            # 使用 log determinant
            # 相似度矩阵
            similarity_matrix = torch.matmul(
                preference_vectors,
                preference_vectors.T
            )

            # 添加对角线正则以确保正定
            batch_size = similarity_matrix.size(0)
            identity = torch.eye(batch_size, device=similarity_matrix.device)
            regularized_matrix = similarity_matrix + self.diversity_eps * identity

            # Log determinant (值越大，多样性越高)
            # 我们返回负值，使得最小化损失等价于最大化多样性
            try:
                loss = -torch.logdet(regularized_matrix)
            except RuntimeError:
                # 如果数值不稳定，使用替代方法
                eigenvalues = torch.linalg.eigvalsh(regularized_matrix)
                loss = -torch.log(eigenvalues + 1e-10).sum()

        elif method == "cosine":
            # 最小化平均余弦相似度
            similarity_matrix = torch.matmul(
                preference_vectors,
                preference_vectors.T
            )

            # 去除对角线（自己与自己的相似度）
            batch_size = similarity_matrix.size(0)
            mask = ~torch.eye(batch_size, dtype=torch.bool, device=similarity_matrix.device)

            # 平均非对角线相似度（值越大，多样性越低）
            loss = similarity_matrix[mask].mean()

        elif method == "distance":
            # 最大化平均距离
            # 计算两两距离
            distances = torch.cdist(preference_vectors, preference_vectors, p=2)

            # 去除对角线
            batch_size = distances.size(0)
            mask = ~torch.eye(batch_size, dtype=torch.bool, device=distances.device)

            # 平均距离（负值，最小化时增加距离）
            loss = -distances[mask].mean()

        else:
            raise ValueError(f"Unknown diversity method: {method}")

        return loss

    def forward(
        self,
        # 检索损失参数
        similarity_scores: torch.Tensor,
        labels: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None,
        # 模态特征
        text_features: Optional[torch.Tensor] = None,
        visual_features: Optional[torch.Tensor] = None,
        fused_features: Optional[torch.Tensor] = None,
        # 控制开关
        use_consistency: bool = True,
        use_contrastive: bool = True,
        use_diversity: bool = True
    ) -> Tuple[torch.Tensor, Dict]:
        """
        计算总损失

        Args:
            similarity_scores: [batch_size, num_candidates] - 相似度分数
            labels: [batch_size, num_candidates] - 标签
            candidate_mask: [batch_size, num_candidates] - 掩码
            text_features: [batch_size, dim] - 文本特征（可选）
            visual_features: [batch_size, dim] - 视觉特征（可选）
            fused_features: [batch_size, dim] - 融合特征（可选）
            use_*: 是否使用各个损失组件

        Returns:
            total_loss: 总损失
            loss_dict: 各组件损失字典
        """
        loss_dict = {}

        # 1. 主检索损失
        loss_retrieval = self.retrieval_loss(
            similarity_scores, labels, candidate_mask
        )
        loss_dict['retrieval'] = loss_retrieval.item()
        total_loss = loss_retrieval

        # 2. 模态一致性损失
        if use_consistency and text_features is not None and visual_features is not None:
            loss_consistency = self.modal_consistency_loss(
                text_features, visual_features
            )
            loss_dict['consistency'] = loss_consistency.item()
            total_loss = total_loss + self.alpha_consistency * loss_consistency

        # 3. 对比学习损失
        if use_contrastive and text_features is not None and visual_features is not None:
            loss_contrastive = self.contrastive_loss(
                text_features, visual_features
            )
            loss_dict['contrastive'] = loss_contrastive.item()
            total_loss = total_loss + self.beta_contrastive * loss_contrastive

        # 4. 多样性正则
        if use_diversity and fused_features is not None:
            loss_diversity = self.diversity_loss(fused_features, method="cosine")
            loss_dict['diversity'] = loss_diversity.item()
            total_loss = total_loss + self.gamma_diversity * loss_diversity

        loss_dict['total'] = total_loss.item()

        return total_loss, loss_dict


class JointLoss(nn.Module):
    """
    联合损失：SASRec + 多模态检索
    """

    def __init__(
        self,
        # 检索损失
        retrieval_temperature: float = 0.07,
        alpha_consistency: float = 0.1,
        beta_contrastive: float = 0.2,
        gamma_diversity: float = 0.05,
        # SASRec 损失
        sasrec_weight: float = 0.5,
        retriever_weight: float = 0.5
    ):
        """
        Args:
            retrieval_*: 检索损失参数
            sasrec_weight: SASRec 损失权重
            retriever_weight: 检索器损失权重
        """
        super().__init__()

        self.retrieval_loss_fn = MultiModalRetrievalLoss(
            temperature=retrieval_temperature,
            alpha_consistency=alpha_consistency,
            beta_contrastive=beta_contrastive,
            gamma_diversity=gamma_diversity
        )

        self.sasrec_weight = sasrec_weight
        self.retriever_weight = retriever_weight

    def forward(
        self,
        # SASRec 参数
        sasrec_pos_scores: Optional[torch.Tensor] = None,
        sasrec_neg_scores: Optional[torch.Tensor] = None,
        # 检索器参数
        retriever_similarity: Optional[torch.Tensor] = None,
        retriever_labels: Optional[torch.Tensor] = None,
        retriever_mask: Optional[torch.Tensor] = None,
        # 特征
        text_features: Optional[torch.Tensor] = None,
        visual_features: Optional[torch.Tensor] = None,
        fused_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        计算联合损失

        Args:
            sasrec_*: SASRec 相关参数
            retriever_*: 检索器相关参数
            *_features: 各模态特征

        Returns:
            total_loss: 总损失
            loss_dict: 损失字典
        """
        loss_dict = {}
        total_loss = 0.0

        # SASRec BPR 损失
        if sasrec_pos_scores is not None and sasrec_neg_scores is not None:
            diff = sasrec_pos_scores.unsqueeze(1) - sasrec_neg_scores
            sasrec_loss = -torch.log(torch.sigmoid(diff) + 1e-10).mean()

            loss_dict['sasrec'] = sasrec_loss.item()
            total_loss = total_loss + self.sasrec_weight * sasrec_loss

        # 检索器损失
        if retriever_similarity is not None and retriever_labels is not None:
            retriever_loss, retriever_loss_dict = self.retrieval_loss_fn(
                similarity_scores=retriever_similarity,
                labels=retriever_labels,
                candidate_mask=retriever_mask,
                text_features=text_features,
                visual_features=visual_features,
                fused_features=fused_features
            )

            # 合并损失字典
            for key, value in retriever_loss_dict.items():
                loss_dict[f'retriever_{key}'] = value

            total_loss = total_loss + self.retriever_weight * retriever_loss

        loss_dict['total'] = total_loss.item()

        return total_loss, loss_dict


class UncertaintyWeightedLoss(nn.Module):
    """
    不确定性加权的多任务损失

    根据任务的不确定性自动调整损失权重

    参考: Kendall et al. "Multi-task learning using uncertainty to weigh losses"
    """

    def __init__(self, num_tasks: int = 2):
        """
        Args:
            num_tasks: 任务数量
        """
        super().__init__()

        # 学习每个任务的不确定性（log variance）
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(
        self,
        losses: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            losses: [num_tasks] - 各任务的损失

        Returns:
            weighted_loss: 加权总损失
            weights: 各任务权重
        """
        # 根据不确定性计算权重
        # loss_weighted = loss / (2 * sigma^2) + log(sigma)
        # 其中 sigma^2 = exp(log_var)

        precision = torch.exp(-self.log_vars)  # 1 / sigma^2
        weighted_loss = torch.sum(
            precision * losses + self.log_vars
        )

        # 返回权重用于监控
        weights = precision / precision.sum()

        return weighted_loss, weights


if __name__ == "__main__":
    print("测试多模态损失函数...")

    # 创建损失函数
    loss_fn = MultiModalRetrievalLoss(
        temperature=0.07,
        alpha_consistency=0.1,
        beta_contrastive=0.2,
        gamma_diversity=0.05
    )

    # 模拟数据
    batch_size = 8
    num_candidates = 20
    dim = 128

    similarity_scores = torch.randn(batch_size, num_candidates)
    labels = torch.randint(0, 2, (batch_size, num_candidates))
    candidate_mask = torch.ones(batch_size, num_candidates, dtype=torch.bool)

    text_features = torch.randn(batch_size, dim)
    visual_features = torch.randn(batch_size, dim)
    fused_features = torch.randn(batch_size, dim)

    # 计算损失
    total_loss, loss_dict = loss_fn(
        similarity_scores=similarity_scores,
        labels=labels,
        candidate_mask=candidate_mask,
        text_features=text_features,
        visual_features=visual_features,
        fused_features=fused_features
    )

    print(f"\n总损失: {total_loss.item():.4f}")
    print("各组件损失:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value:.4f}")

    # 测试联合损失
    print("\n测试联合损失...")
    joint_loss_fn = JointLoss()

    sasrec_pos = torch.randn(batch_size)
    sasrec_neg = torch.randn(batch_size, 10)

    total_loss, loss_dict = joint_loss_fn(
        sasrec_pos_scores=sasrec_pos,
        sasrec_neg_scores=sasrec_neg,
        retriever_similarity=similarity_scores,
        retriever_labels=labels,
        retriever_mask=candidate_mask,
        text_features=text_features,
        visual_features=visual_features,
        fused_features=fused_features
    )

    print(f"\n联合总损失: {total_loss.item():.4f}")
    print("各组件损失:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value:.4f}")

    # 测试不确定性加权
    print("\n测试不确定性加权...")
    uncertainty_loss = UncertaintyWeightedLoss(num_tasks=2)

    losses = torch.tensor([1.5, 0.8])
    weighted_loss, weights = uncertainty_loss(losses)

    print(f"原始损失: {losses}")
    print(f"权重: {weights}")
    print(f"加权损失: {weighted_loss.item():.4f}")
