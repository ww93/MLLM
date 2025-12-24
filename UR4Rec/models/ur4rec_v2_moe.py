"""
FedDMMR: Federated Deep Multimodal Memory Recommendation

实现基于场景自适应异构混合专家(Scenario-Adaptive Heterogeneous MoE)的联邦推荐系统

核心组件:
1. SASRec: 序列模式建模骨干网络
2. VisualExpert: 视觉特征专家 (基于多模态记忆)
3. SemanticExpert: 语义特征专家 (基于文本记忆)
4. ItemCentricRouter: 以目标物品为中心的路由器

Reference: FedDMMR Framework for ACL Submission
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import math

from .sasrec import SASRec


class LightweightAttention(nn.Module):
    """
    轻量级注意力机制

    用于VisualExpert，计算Query与Key之间的注意力权重
    """

    def __init__(self, query_dim: int, key_dim: int, hidden_dim: int = 128):
        """
        Args:
            query_dim: Query向量维度
            key_dim: Key向量维度
            hidden_dim: 隐藏层维度
        """
        super().__init__()

        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.key_proj = nn.Linear(key_dim, hidden_dim)
        self.scale = math.sqrt(hidden_dim)

    def forward(
        self,
        query: torch.Tensor,      # [B, query_dim]
        keys: torch.Tensor,       # [B, TopK, key_dim]
        values: Optional[torch.Tensor] = None  # [B, TopK, value_dim]
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            query: 查询向量 [B, query_dim]
            keys: 键向量 [B, TopK, key_dim]
            values: 值向量 (可选) [B, TopK, value_dim]，默认使用keys

        Returns:
            output: 加权求和后的输出 [B, key_dim] or [B, value_dim]
        """
        if values is None:
            values = keys

        # 投影
        q = self.query_proj(query)  # [B, hidden_dim]
        k = self.key_proj(keys)     # [B, TopK, hidden_dim]

        # 计算注意力得分
        scores = torch.bmm(
            k,
            q.unsqueeze(2)
        ).squeeze(2) / self.scale  # [B, TopK]

        # Softmax归一化
        weights = F.softmax(scores, dim=1)  # [B, TopK]

        # 加权求和
        output = torch.bmm(
            weights.unsqueeze(1),  # [B, 1, TopK]
            values                  # [B, TopK, value_dim]
        ).squeeze(1)  # [B, value_dim]

        return output


class VisualExpert(nn.Module):
    """
    视觉特征专家

    使用轻量级注意力机制从视觉记忆中检索相关信息
    Query: 目标物品的视觉特征
    Keys/Values: 记忆中物品的视觉特征
    """

    def __init__(
        self,
        visual_dim: int,
        hidden_dim: int,
        dropout: float = 0.1
    ):
        """
        Args:
            visual_dim: 视觉特征维度
            hidden_dim: 输出隐藏层维度
            dropout: Dropout率
        """
        super().__init__()

        self.visual_dim = visual_dim
        self.hidden_dim = hidden_dim

        # 轻量级注意力
        self.attention = LightweightAttention(
            query_dim=visual_dim,
            key_dim=visual_dim,
            hidden_dim=min(visual_dim, 128)
        )

        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        target_visual: torch.Tensor,  # [B, visual_dim]
        memory_visual: torch.Tensor   # [B, TopK, visual_dim]
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            target_visual: 目标物品视觉特征 [B, visual_dim]
            memory_visual: 记忆中的视觉特征 [B, TopK, visual_dim]

        Returns:
            visual_emb: 视觉专家输出 [B, hidden_dim]
        """
        # 使用轻量级注意力聚合记忆
        aggregated = self.attention(
            query=target_visual,
            keys=memory_visual,
            values=memory_visual
        )  # [B, visual_dim]

        # 投影到隐藏维度
        visual_emb = self.output_proj(aggregated)  # [B, hidden_dim]

        return visual_emb


class SemanticExpert(nn.Module):
    """
    语义特征专家

    使用多头交叉注意力从文本记忆中检索相关信息
    Query: 目标物品的ID嵌入
    Keys/Values: 记忆中物品的文本特征
    """

    def __init__(
        self,
        id_dim: int,
        text_dim: int,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Args:
            id_dim: ID嵌入维度
            text_dim: 文本特征维度
            hidden_dim: 输出隐藏层维度
            num_heads: 多头注意力头数
            dropout: Dropout率
        """
        super().__init__()

        self.id_dim = id_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim

        # 输入投影 (使Query和Key维度一致)
        self.query_proj = nn.Linear(id_dim, hidden_dim)
        self.key_proj = nn.Linear(text_dim, hidden_dim)
        self.value_proj = nn.Linear(text_dim, hidden_dim)

        # 多头交叉注意力
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # 输出层
        self.output_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        target_id_emb: torch.Tensor,  # [B, id_dim]
        memory_text: torch.Tensor      # [B, TopK, text_dim]
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            target_id_emb: 目标物品ID嵌入 [B, id_dim]
            memory_text: 记忆中的文本特征 [B, TopK, text_dim]

        Returns:
            semantic_emb: 语义专家输出 [B, hidden_dim]
        """
        batch_size = target_id_emb.size(0)

        # 投影到hidden_dim
        query = self.query_proj(target_id_emb).unsqueeze(1)  # [B, 1, hidden_dim]
        keys = self.key_proj(memory_text)      # [B, TopK, hidden_dim]
        values = self.value_proj(memory_text)  # [B, TopK, hidden_dim]

        # 多头交叉注意力
        attn_output, _ = self.cross_attention(
            query=query,    # [B, 1, hidden_dim]
            key=keys,       # [B, TopK, hidden_dim]
            value=values    # [B, TopK, hidden_dim]
        )  # [B, 1, hidden_dim]

        # 去掉序列维度
        attn_output = attn_output.squeeze(1)  # [B, hidden_dim]

        # 输出层
        semantic_emb = self.output_layer(attn_output)  # [B, hidden_dim]

        return semantic_emb


class ItemCentricRouter(nn.Module):
    """
    以物品为中心的路由器（Residual Enhancement版本）

    根据目标物品的嵌入表示，动态决定辅助专家的权重:
    - 视觉专家 (Visual Expert)
    - 语义专家 (Semantic Expert)

    注意: SASRec不再参与路由竞争，而是作为骨干直接保留
    """

    def __init__(
        self,
        item_emb_dim: int,
        hidden_dim: int = 128,
        num_experts: int = 2,  # 现在只有2个辅助专家: Visual, Semantic
        dropout: float = 0.1
    ):
        """
        Args:
            item_emb_dim: 物品嵌入维度
            hidden_dim: 隐藏层维度
            num_experts: 专家数量 (现在固定为2: Visual, Semantic)
            dropout: Dropout率
        """
        super().__init__()

        self.num_experts = num_experts

        # 路由网络
        self.router = nn.Sequential(
            nn.Linear(item_emb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_experts),  # 输出2个权重
            nn.Softmax(dim=-1)
        )

    def forward(
        self,
        target_item_emb: torch.Tensor  # [B, item_emb_dim] or [B, N, item_emb_dim]
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            target_item_emb: 目标物品嵌入 [B, item_emb_dim] or [B, N, item_emb_dim]

        Returns:
            weights: 辅助专家权重 [B, 2] or [B, N, 2] (Visual, Semantic)
        """
        weights = self.router(target_item_emb)  # [B, 2] or [B, N, 2]
        return weights


class UR4RecV2MoE(nn.Module):
    """
    FedDMMR: Residual Enhancement with Heterogeneous MoE

    架构（Residual Enhancement版本）:
    1. SASRec骨干: 序列模式建模（保留为基础表示）
    2. VisualExpert: 视觉特征专家 (基于多模态记忆)
    3. SemanticExpert: 语义特征专家 (基于文本记忆)
    4. ItemCentricRouter: 路由器（仅控制辅助专家权重）
    5. Gating Weight: 可学习的门控权重，控制辅助信息注入

    创新点:
    - 残差增强: SASRec输出直接保留，辅助专家提供增量信息
    - 场景自适应: 路由器根据目标物品动态调整辅助专家权重
    - 可控融合: 通过可学习的gating weight控制多模态信息注入强度
    """

    def __init__(
        self,
        num_items: int,
        # SASRec参数
        sasrec_hidden_dim: int = 256,
        sasrec_num_blocks: int = 2,
        sasrec_num_heads: int = 4,
        sasrec_dropout: float = 0.1,
        max_seq_len: int = 50,
        # 多模态特征维度
        visual_dim: int = 512,      # CLIP特征维度
        text_dim: int = 384,        # Sentence-BERT特征维度
        # MoE参数
        moe_hidden_dim: int = 256,  # MoE输出维度 (应与sasrec_hidden_dim一致)
        moe_num_heads: int = 4,     # SemanticExpert的注意力头数
        moe_dropout: float = 0.1,
        router_hidden_dim: int = 128,
        # 残差增强参数
        gating_init: float = 0.1,   # Gating weight初始值
        # 负载均衡
        load_balance_lambda: float = 0.01,  # 负载均衡损失权重
        # 【已废弃】Router Bias Initialization (不再需要，因为SASRec不参与竞争)
        init_bias_for_sasrec: bool = False,
        sasrec_bias_value: float = 5.0,
        # 设备
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        初始化FedDMMR模型（Residual Enhancement版本）

        Args:
            num_items: 物品总数
            sasrec_*: SASRec相关参数
            max_seq_len: 最大序列长度
            visual_dim: 视觉特征维度 (CLIP: 512)
            text_dim: 文本特征维度 (SBERT: 384)
            moe_hidden_dim: MoE专家输出维度
            moe_num_heads: 语义专家的多头注意力头数
            moe_dropout: MoE模块的dropout率
            router_hidden_dim: 路由器隐藏层维度
            gating_init: Gating weight初始值（推荐0.0-0.1）
            load_balance_lambda: 负载均衡损失权重
            init_bias_for_sasrec: [已废弃] 不再使用
            sasrec_bias_value: [已废弃] 不再使用
            device: 运行设备
        """
        super().__init__()

        self.num_items = num_items
        self.sasrec_hidden_dim = sasrec_hidden_dim
        self.moe_hidden_dim = moe_hidden_dim
        self.visual_dim = visual_dim
        self.text_dim = text_dim
        self.load_balance_lambda = load_balance_lambda
        self.device = device

        # 确保MoE输出维度与SASRec一致
        if moe_hidden_dim != sasrec_hidden_dim:
            print(f"警告: moe_hidden_dim ({moe_hidden_dim}) 与 sasrec_hidden_dim ({sasrec_hidden_dim}) 不一致")
            print(f"      建议设置 moe_hidden_dim = sasrec_hidden_dim 以确保特征对齐")

        # ===========================
        # 1. 序列骨干: SASRec
        # ===========================
        self.sasrec = SASRec(
            num_items=num_items,
            hidden_dim=sasrec_hidden_dim,
            num_blocks=sasrec_num_blocks,
            num_heads=sasrec_num_heads,
            dropout=sasrec_dropout,
            max_seq_len=max_seq_len
        )

        # ===========================
        # 2. 视觉专家
        # ===========================
        self.visual_expert = VisualExpert(
            visual_dim=visual_dim,
            hidden_dim=moe_hidden_dim,
            dropout=moe_dropout
        )

        # ===========================
        # 3. 语义专家
        # ===========================
        self.semantic_expert = SemanticExpert(
            id_dim=sasrec_hidden_dim,  # 使用SASRec的item embedding作为query
            text_dim=text_dim,
            hidden_dim=moe_hidden_dim,
            num_heads=moe_num_heads,
            dropout=moe_dropout
        )

        # ===========================
        # 4. 以物品为中心的路由器（仅控制辅助专家）
        # ===========================
        self.router = ItemCentricRouter(
            item_emb_dim=sasrec_hidden_dim,
            hidden_dim=router_hidden_dim,
            num_experts=2,  # 只有2个辅助专家: Visual, Semantic
            dropout=moe_dropout
        )

        # ===========================
        # 5. LayerNorm for each expert output
        # ===========================
        self.seq_layernorm = nn.LayerNorm(sasrec_hidden_dim)
        self.vis_layernorm = nn.LayerNorm(moe_hidden_dim)
        self.sem_layernorm = nn.LayerNorm(moe_hidden_dim)

        # ===========================
        # 6. Gating Weight (可学习的门控参数)
        # ===========================
        self.gating_weight = nn.Parameter(torch.tensor(gating_init))

        print(f"✓ Residual Enhancement 架构初始化:")
        print(f"  Gating Weight 初始值: {gating_init}")
        print(f"  Router 控制专家数: 2 (Visual, Semantic)")
        print(f"  SASRec 作为骨干直接保留")

        self.to(device)

    def forward(
        self,
        user_ids: Optional[List[int]],
        input_seq: torch.Tensor,           # [B, L]
        target_items: torch.Tensor,        # [B] or [B, N]
        # 多模态特征 (可选)
        target_visual: Optional[torch.Tensor] = None,  # [B, visual_dim] or [B, N, visual_dim]
        target_text: Optional[torch.Tensor] = None,    # [B, text_dim] or [B, N, text_dim]
        # 多模态记忆 (可选)
        memory_visual: Optional[torch.Tensor] = None,  # [B, TopK, visual_dim]
        memory_text: Optional[torch.Tensor] = None,    # [B, TopK, text_dim]
        # 其他参数
        seq_padding_mask: Optional[torch.Tensor] = None,
        update_memory: bool = False,
        return_components: bool = False,
        retrieved_memory: Optional[Dict] = None  # 兼容旧接口
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        FedDMMR前向传播

        Args:
            user_ids: 用户ID列表 [batch_size] (兼容性参数)
            input_seq: 输入序列 [B, L]
            target_items: 目标物品ID [B] or [B, N]
            target_visual: 目标物品视觉特征 [B, visual_dim] or [B, N, visual_dim]
            target_text: 目标物品文本特征 [B, text_dim] or [B, N, text_dim]
            memory_visual: 记忆中的视觉特征 [B, TopK, visual_dim]
            memory_text: 记忆中的文本特征 [B, TopK, text_dim]
            seq_padding_mask: 序列padding mask [B, L]
            update_memory: 是否更新记忆 (兼容性参数)
            return_components: 是否返回各组件得分
            retrieved_memory: 兼容旧接口 (已废弃)

        Returns:
            scores: 预测得分 [B, N]
            (optional) info: 包含各组件信息的字典
        """
        batch_size = input_seq.size(0)

        # 确保target_items是2D: [B, N]
        if target_items.dim() == 1:
            target_items = target_items.unsqueeze(1)  # [B, 1]

        num_candidates = target_items.size(1)

        # ===========================
        # 1. 序列专家: SASRec
        # ===========================
        # 获取序列表示
        seq_output = self.sasrec(input_seq, padding_mask=seq_padding_mask)  # [B, L, D]
        seq_repr = seq_output[:, -1, :]  # 取最后一个时间步 [B, D]

        # 获取目标物品嵌入（用于路由和最终得分计算）
        target_item_embs = self.sasrec.item_embedding(target_items)  # [B, N, D]

        # 序列专家输出：扩展seq_repr为[B, N, D]（每个候选物品使用相同的序列表示）
        seq_out = seq_repr.unsqueeze(1).expand(-1, num_candidates, -1)  # [B, N, D]

        # ===========================
        # 【方案2修复】没有多模态特征时，直接使用SASRec
        # ===========================
        if memory_visual is None and target_visual is None and memory_text is None:
            # 没有任何多模态特征，直接使用标准SASRec计算得分
            # 计算内积得分: [B, 1, D] @ [B, N, D]^T = [B, 1, N]
            final_scores = torch.bmm(
                seq_repr.unsqueeze(1),  # [B, 1, D]
                target_item_embs.transpose(1, 2)  # [B, D, N]
            ).squeeze(1)  # [B, N]

            if return_components:
                return final_scores, {'lb_loss': torch.tensor(0.0, device=self.device)}
            return final_scores

        # ===========================
        # 2. 视觉专家
        # ===========================
        if memory_visual is not None and target_visual is not None:
            # 如果target_visual是[B, N, visual_dim]，需要分别处理每个候选物品
            if target_visual.dim() == 3:
                # 批量处理所有候选物品
                vis_embs_list = []
                for i in range(num_candidates):
                    vis_emb = self.visual_expert(
                        target_visual=target_visual[:, i, :],  # [B, visual_dim]
                        memory_visual=memory_visual            # [B, TopK, visual_dim]
                    )  # [B, moe_hidden_dim]
                    vis_embs_list.append(vis_emb)
                vis_out = torch.stack(vis_embs_list, dim=1)  # [B, N, moe_hidden_dim]
            else:
                # target_visual是[B, visual_dim]，所有候选物品使用相同的视觉特征
                vis_emb = self.visual_expert(
                    target_visual=target_visual,
                    memory_visual=memory_visual
                )  # [B, moe_hidden_dim]
                vis_out = vis_emb.unsqueeze(1).expand(-1, num_candidates, -1)  # [B, N, moe_hidden_dim]
        else:
            # 没有视觉特征时，使用零向量
            vis_out = torch.zeros(batch_size, num_candidates, self.moe_hidden_dim, device=self.device)

        # ===========================
        # 3. 语义专家
        # ===========================
        if memory_text is not None:
            # 获取目标物品的ID嵌入作为query
            target_id_embs = self.sasrec.item_embedding(target_items)  # [B, N, D]

            # 分别处理每个候选物品
            sem_embs_list = []
            for i in range(num_candidates):
                sem_emb = self.semantic_expert(
                    target_id_emb=target_id_embs[:, i, :],  # [B, D]
                    memory_text=memory_text                  # [B, TopK, text_dim]
                )  # [B, moe_hidden_dim]
                sem_embs_list.append(sem_emb)
            sem_out = torch.stack(sem_embs_list, dim=1)  # [B, N, moe_hidden_dim]
        else:
            # 没有文本特征时，使用零向量
            sem_out = torch.zeros(batch_size, num_candidates, self.moe_hidden_dim, device=self.device)

        # ===========================
        # 4. 残差增强融合 (Residual Enhancement Fusion)
        # ===========================
        # 应用LayerNorm到所有专家输出
        seq_out_norm = self.seq_layernorm(seq_out)    # [B, N, D]
        vis_out_norm = self.vis_layernorm(vis_out)    # [B, N, D]
        sem_out_norm = self.sem_layernorm(sem_out)    # [B, N, D]

        # 获取路由权重 (仅对辅助专家: Visual, Semantic)
        router_weights = self.router(target_item_embs)  # [B, N, 2]

        # 【修复】根据专家可用性调整权重，避免给零专家分配权重导致信息稀释
        expert_available = torch.tensor([
            1.0 if (memory_visual is not None and target_visual is not None) else 0.0,  # Visual
            1.0 if memory_text is not None else 0.0  # Semantic
        ], device=router_weights.device)

        # 将不可用专家的权重mask掉并重新归一化
        router_weights = router_weights * expert_available.view(1, 1, 2)
        router_weights = router_weights / (router_weights.sum(dim=-1, keepdim=True) + 1e-10)

        # 提取辅助专家的权重
        w_vis = router_weights[:, :, 0].unsqueeze(2)  # [B, N, 1]
        w_sem = router_weights[:, :, 1].unsqueeze(2)  # [B, N, 1]

        # 【残差增强融合】SASRec骨干保留，辅助专家提供增量信息
        # Formula: final_repr = seq_out + gating * (w_vis * vis_out + w_sem * sem_out)
        auxiliary_repr = w_vis * vis_out_norm + w_sem * sem_out_norm  # [B, N, D]
        fused_repr = seq_out_norm + self.gating_weight * auxiliary_repr  # [B, N, D]

        # L2归一化，使用余弦相似度计算得分
        fused_repr_norm = torch.nn.functional.normalize(fused_repr, p=2, dim=-1)  # [B, N, D]
        target_item_embs_norm = torch.nn.functional.normalize(target_item_embs, p=2, dim=-1)  # [B, N, D]

        # 计算最终得分：归一化后的余弦相似度
        scale = self.sasrec_hidden_dim ** 0.5
        final_scores = scale * (fused_repr_norm * target_item_embs_norm).sum(dim=-1)  # [B, N]

        # ===========================
        # 5. 负载均衡损失
        # ===========================
        # 计算每个辅助专家的平均使用率
        expert_usage = router_weights.mean(dim=[0, 1])  # [2]

        # 负载均衡损失: 鼓励辅助专家均匀使用
        # L_lb = Σ(usage_i - 1/N)^2
        target_usage = 1.0 / 2.0  # 均匀分配给2个辅助专家
        lb_loss = torch.sum((expert_usage - target_usage) ** 2)

        # ===========================
        # 6. 返回结果
        # ===========================
        if return_components:
            # 为了调试和分析，也计算各专家单独的分数
            seq_scores = (seq_out_norm * target_item_embs_norm).sum(dim=-1) * scale  # [B, N]
            vis_scores = (vis_out_norm * target_item_embs_norm).sum(dim=-1) * scale  # [B, N]
            sem_scores = (sem_out_norm * target_item_embs_norm).sum(dim=-1) * scale  # [B, N]

            info = {
                'seq_scores': seq_scores,           # 序列骨干单独得分（仅供分析）
                'vis_scores': vis_scores,           # 视觉专家单独得分（仅供分析）
                'sem_scores': sem_scores,           # 语义专家单独得分（仅供分析）
                'final_scores': final_scores,       # 融合后的最终得分
                'router_weights': router_weights,   # [B, N, 2] 辅助专家路由权重
                'expert_usage': expert_usage,       # [2] 辅助专家使用率
                'lb_loss': lb_loss,                 # 负载均衡损失
                'w_vis': w_vis.mean().item(),       # 视觉专家平均权重
                'w_sem': w_sem.mean().item(),       # 语义专家平均权重
                'gating_weight': self.gating_weight.item(),  # 门控权重
                # 表示向量信息
                'seq_out': seq_out,                 # [B, N, D] 序列骨干表示
                'vis_out': vis_out,                 # [B, N, D] 视觉专家表示
                'sem_out': sem_out,                 # [B, N, D] 语义专家表示
                'seq_out_norm': seq_out_norm,       # [B, N, D] 归一化后的序列表示
                'vis_out_norm': vis_out_norm,       # [B, N, D] 归一化后的视觉表示
                'sem_out_norm': sem_out_norm,       # [B, N, D] 归一化后的语义表示
                'auxiliary_repr': auxiliary_repr,   # [B, N, D] 辅助表示
                'fused_repr': fused_repr            # [B, N, D] 融合后的表示
            }
            return final_scores, info
        else:
            return final_scores

    def compute_loss(
        self,
        logits: torch.Tensor,      # [B, N]
        labels: torch.Tensor,      # [B] (正样本在第0个位置)
        lb_loss: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        计算总损失

        Args:
            logits: 模型输出得分 [B, N]
            labels: 标签 [B] (通常全0，表示正样本在第0个位置)
            lb_loss: 负载均衡损失 (如果已计算)

        Returns:
            total_loss: 总损失
            loss_dict: 各部分损失的字典
        """
        # 推荐损失 (CrossEntropy)
        rec_loss = F.cross_entropy(logits, labels)

        # 如果没有提供lb_loss，设为0
        if lb_loss is None:
            lb_loss = torch.tensor(0.0, device=self.device)

        # 总损失
        total_loss = rec_loss + self.load_balance_lambda * lb_loss

        loss_dict = {
            'total_loss': total_loss.item(),
            'rec_loss': rec_loss.item(),
            'lb_loss': lb_loss.item()
        }

        return total_loss, loss_dict

    def get_item_embeddings(
        self,
        item_ids: torch.Tensor,
        embedding_type: str = 'id'
    ) -> torch.Tensor:
        """
        获取物品嵌入

        Args:
            item_ids: 物品ID [B] or [B, N]
            embedding_type: 嵌入类型 ('id' - 从SASRec的item embedding获取)

        Returns:
            embeddings: 物品嵌入 [B, D] or [B, N, D]
        """
        if embedding_type == 'id':
            return self.sasrec.item_embedding(item_ids)
        else:
            # 其他类型的嵌入需要外部提供 (visual/text features)
            raise ValueError(f"不支持的embedding_type: {embedding_type}. 请使用 'id'")

    def get_sequence_representation(
        self,
        input_seq: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        获取序列表示 (用于对比学习等)

        Args:
            input_seq: 输入序列 [B, L]
            padding_mask: Padding mask [B, L]

        Returns:
            seq_repr: 序列表示 [B, D]
        """
        seq_output = self.sasrec(input_seq, padding_mask=padding_mask)  # [B, L, D]
        seq_repr = seq_output[:, -1, :]  # [B, D]
        return seq_repr


__all__ = ['UR4RecV2MoE', 'VisualExpert', 'SemanticExpert', 'ItemCentricRouter']
