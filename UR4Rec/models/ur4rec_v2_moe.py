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
    视觉特征专家（方案2：保持原始维度）

    使用轻量级注意力机制从视觉记忆中检索相关信息
    Query: 目标物品的视觉特征
    Keys/Values: 记忆中物品的视觉特征

    [方案2修改]: 不压缩特征维度，保持512维输出
    原因: 避免丢失预训练CLIP特征的信息
    """

    def __init__(
        self,
        visual_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        preserve_dim: bool = True  # [新增] 是否保持原始维度
    ):
        """
        Args:
            visual_dim: 视觉特征维度
            hidden_dim: 输出隐藏层维度（preserve_dim=False时使用）
            dropout: Dropout率
            preserve_dim: 是否保持原始维度（True=512维输出，False=hidden_dim输出）
        """
        super().__init__()

        self.visual_dim = visual_dim
        self.hidden_dim = hidden_dim
        self.preserve_dim = preserve_dim

        # 轻量级注意力
        self.attention = LightweightAttention(
            query_dim=visual_dim,
            key_dim=visual_dim,
            hidden_dim=min(visual_dim, 128)
        )

        # 输出投影
        if preserve_dim:
            # [方案2]: 保持原始维度 512 → 512
            self.output_proj = nn.Sequential(
                nn.Linear(visual_dim, visual_dim),
                nn.LayerNorm(visual_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.output_dim = visual_dim
        else:
            # 原版: 压缩维度 512 → hidden_dim
            self.output_proj = nn.Sequential(
                nn.Linear(visual_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.output_dim = hidden_dim

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
    语义特征专家（方案2：保持原始维度）

    使用多头交叉注意力从文本记忆中检索相关信息
    Query: 目标物品的ID嵌入
    Keys/Values: 记忆中物品的文本特征

    [方案2修改]: 不压缩特征维度，保持384维输出
    原因: 避免丢失预训练SBERT特征的信息
    """

    def __init__(
        self,
        id_dim: int,
        text_dim: int,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        preserve_dim: bool = True  # [新增] 是否保持原始维度
    ):
        """
        Args:
            id_dim: ID嵌入维度
            text_dim: 文本特征维度
            hidden_dim: 输出隐藏层维度（preserve_dim=False时使用）
            num_heads: 多头注意力头数
            dropout: Dropout率
            preserve_dim: 是否保持原始维度（True=384维输出，False=hidden_dim输出）
        """
        super().__init__()

        self.id_dim = id_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.preserve_dim = preserve_dim

        # 输入投影 (使Query和Key维度一致)
        if preserve_dim:
            # [方案2]: 保持文本特征维度 384
            self.query_proj = nn.Linear(id_dim, text_dim)
            self.key_proj = nn.Linear(text_dim, text_dim)
            self.value_proj = nn.Linear(text_dim, text_dim)
            self.attn_dim = text_dim
            self.output_dim = text_dim
        else:
            # 原版: 压缩到hidden_dim
            self.query_proj = nn.Linear(id_dim, hidden_dim)
            self.key_proj = nn.Linear(text_dim, hidden_dim)
            self.value_proj = nn.Linear(text_dim, hidden_dim)
            self.attn_dim = hidden_dim
            self.output_dim = hidden_dim

        # 多头交叉注意力
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.attn_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # 输出层
        self.output_layer = nn.Sequential(
            nn.LayerNorm(self.attn_dim),
            nn.Linear(self.attn_dim, self.output_dim),
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


class CrossModalFusion(nn.Module):
    """
    跨模态融合层（方案2：注意力融合）

    使用交叉注意力融合不同维度的多模态特征
    Query: SASRec输出 (128维)
    Keys/Values: [Visual特征 (512维), Semantic特征 (384维)]
    Output: 128维（与SASRec对齐）

    [方案2设计]:
    - 保持预训练特征的完整性（不压缩维度）
    - 使用注意力机制自适应加权融合
    - 输出维度与SASRec一致，便于残差连接
    """

    def __init__(
        self,
        query_dim: int,      # SASRec输出维度（128）
        visual_dim: int,     # 视觉特征维度（512）
        text_dim: int,       # 文本特征维度（384）
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Args:
            query_dim: Query维度（SASRec输出）
            visual_dim: 视觉特征维度
            text_dim: 文本特征维度
            num_heads: 多头注意力头数
            dropout: Dropout率
        """
        super().__init__()

        self.query_dim = query_dim
        self.visual_dim = visual_dim
        self.text_dim = text_dim

        # 为每种模态创建独立的注意力机制
        # Visual Attention
        self.visual_query_proj = nn.Linear(query_dim, visual_dim)
        self.visual_key_proj = nn.Linear(visual_dim, visual_dim)
        self.visual_value_proj = nn.Linear(visual_dim, visual_dim)
        self.visual_attn = nn.MultiheadAttention(
            embed_dim=visual_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.visual_out_proj = nn.Linear(visual_dim, query_dim)

        # Text Attention
        self.text_query_proj = nn.Linear(query_dim, text_dim)
        self.text_key_proj = nn.Linear(text_dim, text_dim)
        self.text_value_proj = nn.Linear(text_dim, text_dim)
        self.text_attn = nn.MultiheadAttention(
            embed_dim=text_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.text_out_proj = nn.Linear(text_dim, query_dim)

        # 最终融合层
        self.fusion_layer = nn.Sequential(
            nn.LayerNorm(query_dim * 3),  # SASRec + Visual + Text
            nn.Linear(query_dim * 3, query_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        seq_repr: torch.Tensor,      # [B, query_dim] SASRec输出
        visual_repr: torch.Tensor,   # [B, visual_dim] 视觉专家输出
        text_repr: torch.Tensor      # [B, text_dim] 语义专家输出
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            seq_repr: SASRec输出 [B, query_dim]
            visual_repr: 视觉专家输出 [B, visual_dim]
            text_repr: 语义专家输出 [B, text_dim]

        Returns:
            fused_repr: 融合后的表示 [B, query_dim]
        """
        batch_size = seq_repr.size(0)

        # Visual Attention: 使用SASRec输出查询视觉特征
        vis_query = self.visual_query_proj(seq_repr).unsqueeze(1)  # [B, 1, visual_dim]
        vis_key = self.visual_key_proj(visual_repr).unsqueeze(1)   # [B, 1, visual_dim]
        vis_value = self.visual_value_proj(visual_repr).unsqueeze(1)  # [B, 1, visual_dim]

        vis_attn_out, _ = self.visual_attn(
            query=vis_query,
            key=vis_key,
            value=vis_value
        )  # [B, 1, visual_dim]
        vis_attn_out = vis_attn_out.squeeze(1)  # [B, visual_dim]
        vis_out = self.visual_out_proj(vis_attn_out)  # [B, query_dim]

        # Text Attention: 使用SASRec输出查询文本特征
        text_query = self.text_query_proj(seq_repr).unsqueeze(1)  # [B, 1, text_dim]
        text_key = self.text_key_proj(text_repr).unsqueeze(1)     # [B, 1, text_dim]
        text_value = self.text_value_proj(text_repr).unsqueeze(1)  # [B, 1, text_dim]

        text_attn_out, _ = self.text_attn(
            query=text_query,
            key=text_key,
            value=text_value
        )  # [B, 1, text_dim]
        text_attn_out = text_attn_out.squeeze(1)  # [B, text_dim]
        text_out = self.text_out_proj(text_attn_out)  # [B, query_dim]

        # 拼接并融合: [SASRec, Visual, Text]
        concatenated = torch.cat([seq_repr, vis_out, text_out], dim=1)  # [B, query_dim*3]
        fused_repr = self.fusion_layer(concatenated)  # [B, query_dim]

        return fused_repr


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
        # [FIX 1] 残差增强参数 - 已移除可学习gating，确保完整梯度流
        gating_init: float = 1.0,   # 保留参数以保持向后兼容（不再使用）
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
        # 1.5. [Stage 2新增] 轻量级投影层（多模态对齐）
        # ===========================
        # 目标：将多模态特征投影到与SASRec相同的空间（128维）
        # Stage 2只训练这些投影层，冻结SASRec骨干
        self.visual_proj = nn.Linear(visual_dim, sasrec_hidden_dim)  # 512→128 (~65K params)
        self.text_proj = nn.Linear(text_dim, sasrec_hidden_dim)      # 384→128 (~49K params)

        # [Stage 2新增] Gating MLP：控制多模态信息注入强度
        # 输入：拼接的[SASRec, Visual, Text] 3×128=384维
        # 输出：2维权重（Visual weight, Text weight）
        self.align_gating = nn.Sequential(
            nn.Linear(sasrec_hidden_dim * 3, 64),  # 384→64
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2),  # 64→2
            nn.Softmax(dim=-1)
        )  # ~25K params

        print(f"  ✓ [Stage 2] 轻量级投影层:")
        print(f"    - visual_proj: {visual_dim}→{sasrec_hidden_dim} (~{visual_dim * sasrec_hidden_dim:,} params)")
        print(f"    - text_proj: {text_dim}→{sasrec_hidden_dim} (~{text_dim * sasrec_hidden_dim:,} params)")
        print(f"    - align_gating MLP: ~25K params")
        total_align_params = visual_dim * sasrec_hidden_dim + text_dim * sasrec_hidden_dim + 25_000
        print(f"    - 总计: ~{total_align_params:,} params (目标 <200K)")

        # ===========================
        # 2. 视觉专家（使用投影后的128维特征）
        # ===========================
        self.preserve_multimodal_dim = False  # [Stage 2修改] 使用投影后的统一维度
        self.visual_expert = VisualExpert(
            visual_dim=sasrec_hidden_dim,  # 修改：使用投影后的128维
            hidden_dim=moe_hidden_dim,
            dropout=moe_dropout,
            preserve_dim=self.preserve_multimodal_dim
        )

        # ===========================
        # 3. 语义专家（使用投影后的128维特征）
        # ===========================
        self.semantic_expert = SemanticExpert(
            id_dim=sasrec_hidden_dim,  # 使用SASRec的item embedding作为query
            text_dim=sasrec_hidden_dim,  # 修改：使用投影后的128维
            hidden_dim=moe_hidden_dim,
            num_heads=moe_num_heads,
            dropout=moe_dropout,
            preserve_dim=self.preserve_multimodal_dim
        )

        # ===========================
        # 3.5 跨模态融合层（Stage 3启用，使用统一的128维）
        # ===========================
        if self.preserve_multimodal_dim:
            # 旧方案：保持原始维度（已弃用）
            self.cross_modal_fusion = CrossModalFusion(
                query_dim=sasrec_hidden_dim,  # 128
                visual_dim=visual_dim,        # 512
                text_dim=text_dim,            # 384
                num_heads=moe_num_heads,
                dropout=moe_dropout
            )
            print(f"  ✓ 添加CrossModalFusion层（方案2：注意力融合）")
            print(f"    - SASRec: {sasrec_hidden_dim}维")
            print(f"    - Visual: {visual_dim}维（保持CLIP原始维度）")
            print(f"    - Text: {text_dim}维（保持SBERT原始维度）")
            print(f"    - 输出: {sasrec_hidden_dim}维（与SASRec对齐）")
        else:
            # 新方案：使用投影后的统一维度（Stage 2/3推荐）
            self.cross_modal_fusion = CrossModalFusion(
                query_dim=sasrec_hidden_dim,  # 128
                visual_dim=sasrec_hidden_dim,  # 128（投影后）
                text_dim=sasrec_hidden_dim,    # 128（投影后）
                num_heads=moe_num_heads,
                dropout=moe_dropout
            )
            print(f"  ✓ [Stage 3] CrossModalFusion层（统一维度）")
            print(f"    - SASRec: {sasrec_hidden_dim}维")
            print(f"    - Visual: {sasrec_hidden_dim}维（投影后）")
            print(f"    - Text: {sasrec_hidden_dim}维（投影后）")
            print(f"    - 输出: {sasrec_hidden_dim}维")

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
        # 5. LayerNorm for each expert output（统一使用128维）
        # ===========================
        self.seq_layernorm = nn.LayerNorm(sasrec_hidden_dim)     # 128
        self.vis_layernorm = nn.LayerNorm(sasrec_hidden_dim)     # 128
        self.sem_layernorm = nn.LayerNorm(sasrec_hidden_dim)     # 128

        # ===========================
        # 6. [STAGE 2/3 FIX] 可学习的门控权重
        # ===========================
        # 问题: Stage 2初期，未对齐的多模态表示会破坏训练好的SASRec输出
        # 解决: 添加可学习的gating_weight，初始化为很小的值（0.01）
        #       让辅助表示从微弱信号逐渐增强，避免破坏SASRec
        # 融合方式: seq_out + gating_weight * (w_vis * vis_out + w_sem * sem_out)
        self.gating_weight = nn.Parameter(torch.tensor(gating_init, dtype=torch.float32))

        print(f"✓ Residual Enhancement 架构初始化 [Stage 2/3 优化版]:")
        if self.preserve_multimodal_dim:
            print(f"  [方案2] 使用CrossModalFusion注意力融合")
            print(f"  Visual Expert输出维度: {self.visual_expert.output_dim}")
            print(f"  Semantic Expert输出维度: {self.semantic_expert.output_dim}")
        else:
            print(f"  融合方式: seq_out + gating_weight * (w_vis * vis_out + w_sem * sem_out)")
        print(f"  Router 控制专家数: 2 (Visual, Semantic)")
        print(f"  SASRec 作为骨干直接保留")
        print(f"  ✓ 添加可学习gating_weight，初始值={gating_init:.3f}")
        print(f"    - gating_init小时: 保护Stage 1的SASRec输出，避免被随机多模态破坏")
        print(f"    - 训练过程中: gating_weight逐渐增大，让多模态信息逐步融入")

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
        retrieved_memory: Optional[Dict] = None,  # 兼容旧接口
        training_mode: bool = False  # [优化5] 批内负采样模式
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
            training_mode: [优化5] 是否使用批内负采样模式
                          - True: target_items [B], 返回 [B, B] 分数矩阵
                          - False: target_items [B, N], 返回 [B, N] 分数

        Returns:
            scores: 预测得分 [B, N] or [B, B] (training_mode=True时)
            (optional) info: 包含各组件信息的字典
        """
        batch_size = input_seq.size(0)

        # ===========================
        # [优化5] 批内负采样模式处理
        # ===========================
        if training_mode:
            # 训练模式：target_items [B]，使用批内负采样
            # 不需要unsqueeze，保持[B]用于后续计算
            if target_items.dim() == 2:
                target_items = target_items.squeeze(1)  # [B, N] -> [B]
            num_candidates = batch_size  # 每个用户对batch内所有物品打分
        else:
            # 推理模式：target_items [B] or [B, N]
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
        if training_mode:
            # [优化5] 批内负采样：获取batch内所有物品的嵌入 [B, D]
            target_item_embs = self.sasrec.item_embedding(target_items)  # [B, D]
            # 序列专家输出：直接使用[B, D]
            seq_out = seq_repr  # [B, D]
        else:
            # 推理模式：target_items [B, N]
            target_item_embs = self.sasrec.item_embedding(target_items)  # [B, N, D]
            # 序列专家输出：扩展seq_repr为[B, N, D]
            seq_out = seq_repr.unsqueeze(1).expand(-1, num_candidates, -1)  # [B, N, D]

        # ===========================
        # 【方案2修复】没有多模态特征时，直接使用SASRec
        # ===========================
        if memory_visual is None and target_visual is None and memory_text is None:
            # 没有任何多模态特征，直接使用标准SASRec计算得分
            if training_mode:
                # [优化5] 批内负采样模式: [B, D] × [D, B] = [B, B]
                final_scores = torch.mm(seq_repr, target_item_embs.t())  # [B, B]

                if return_components:
                    # 创建零张量作为vis_out和sem_out（用于漂移自适应对比学习）
                    zero_out = torch.zeros(batch_size, self.moe_hidden_dim, device=self.device)  # [B, D]
                    return final_scores, {
                        'lb_loss': torch.tensor(0.0, device=self.device),
                        'vis_out': zero_out,  # [B, D] 零张量
                        'sem_out': zero_out   # [B, D] 零张量
                    }
                return final_scores
            else:
                # 推理模式: [B, 1, D] @ [B, N, D]^T = [B, 1, N]
                final_scores = torch.bmm(
                    seq_repr.unsqueeze(1),  # [B, 1, D]
                    target_item_embs.transpose(1, 2)  # [B, D, N]
                ).squeeze(1)  # [B, N]

                if return_components:
                    # 创建零张量作为vis_out和sem_out（用于漂移自适应对比学习）
                    zero_out = torch.zeros(batch_size, num_candidates, self.moe_hidden_dim, device=self.device)
                    return final_scores, {
                        'lb_loss': torch.tensor(0.0, device=self.device),
                        'vis_out': zero_out,  # [B, N, D] 零张量
                        'sem_out': zero_out   # [B, N, D] 零张量
                    }
                return final_scores

        # ===========================
        # 2. [Stage 2新增] 应用投影层 + 视觉专家
        # ===========================
        if memory_visual is not None and target_visual is not None:
            # [Stage 2关键] 先投影再送入Expert
            if training_mode:
                # [优化5] 批内负采样模式：target_visual [B, 1, visual_dim]
                # 提取单个物品的视觉特征 [B, visual_dim]
                if target_visual.dim() == 3:
                    target_visual_single = target_visual[:, 0, :]  # [B, visual_dim=512]
                else:
                    target_visual_single = target_visual  # [B, visual_dim=512]

                # [Stage 2] 投影: 512→128
                target_visual_proj = self.visual_proj(target_visual_single)  # [B, 128]

                # 投影memory_visual: [B, TopK, 512] → [B, TopK, 128]
                B, TopK, _ = memory_visual.shape
                memory_visual_flat = memory_visual.view(B * TopK, -1)  # [B*TopK, 512]
                memory_visual_proj = self.visual_proj(memory_visual_flat).view(B, TopK, -1)  # [B, TopK, 128]

                vis_out = self.visual_expert(
                    target_visual=target_visual_proj,  # [B, 128]
                    memory_visual=memory_visual_proj   # [B, TopK, 128]
                )  # [B, moe_hidden_dim=128]
            else:
                # 推理模式：处理[B, N, visual_dim]
                if target_visual.dim() == 3:
                    # [Stage 2] 投影所有候选: [B, N, 512] → [B, N, 128]
                    B, N, _ = target_visual.shape
                    target_visual_flat = target_visual.view(B * N, -1)  # [B*N, 512]
                    target_visual_proj = self.visual_proj(target_visual_flat).view(B, N, -1)  # [B, N, 128]

                    # 投影memory_visual
                    B, TopK, _ = memory_visual.shape
                    memory_visual_flat = memory_visual.view(B * TopK, -1)  # [B*TopK, 512]
                    memory_visual_proj = self.visual_proj(memory_visual_flat).view(B, TopK, -1)  # [B, TopK, 128]

                    # 批量处理所有候选物品
                    vis_embs_list = []
                    for i in range(num_candidates):
                        vis_emb = self.visual_expert(
                            target_visual=target_visual_proj[:, i, :],  # [B, 128]
                            memory_visual=memory_visual_proj            # [B, TopK, 128]
                        )  # [B, moe_hidden_dim=128]
                        vis_embs_list.append(vis_emb)
                    vis_out = torch.stack(vis_embs_list, dim=1)  # [B, N, moe_hidden_dim]
                else:
                    # target_visual是[B, visual_dim]，所有候选物品使用相同的视觉特征
                    target_visual_proj = self.visual_proj(target_visual)  # [B, 128]

                    # 投影memory_visual
                    B, TopK, _ = memory_visual.shape
                    memory_visual_flat = memory_visual.view(B * TopK, -1)
                    memory_visual_proj = self.visual_proj(memory_visual_flat).view(B, TopK, -1)

                    vis_emb = self.visual_expert(
                        target_visual=target_visual_proj,
                        memory_visual=memory_visual_proj
                    )  # [B, moe_hidden_dim=128]
                    vis_out = vis_emb.unsqueeze(1).expand(-1, num_candidates, -1)  # [B, N, moe_hidden_dim]
        else:
            # 没有视觉特征时，使用零向量（统一使用128维）
            if training_mode:
                vis_out = torch.zeros(batch_size, self.sasrec_hidden_dim, device=self.device)  # [B, 128]
            else:
                vis_out = torch.zeros(batch_size, num_candidates, self.sasrec_hidden_dim, device=self.device)  # [B, N, 128]

        # ===========================
        # 3. [Stage 2新增] 应用投影层 + 语义专家
        # ===========================
        if memory_text is not None:
            # [Stage 2关键] 先投影文本特征: 384→128
            B, TopK, text_dim_orig = memory_text.shape
            memory_text_flat = memory_text.view(B * TopK, -1)  # [B*TopK, 384]
            memory_text_proj = self.text_proj(memory_text_flat).view(B, TopK, -1)  # [B, TopK, 128]

            if training_mode:
                # [优化5] 批内负采样模式：target_item_embs [B, D=128]
                sem_out = self.semantic_expert(
                    target_id_emb=target_item_embs,  # [B, 128]
                    memory_text=memory_text_proj     # [B, TopK, 128]
                )  # [B, moe_hidden_dim=128]
            else:
                # 推理模式：target_item_embs [B, N, D=128]
                # 分别处理每个候选物品
                sem_embs_list = []
                for i in range(num_candidates):
                    sem_emb = self.semantic_expert(
                        target_id_emb=target_item_embs[:, i, :],  # [B, 128]
                        memory_text=memory_text_proj              # [B, TopK, 128]
                    )  # [B, moe_hidden_dim=128]
                    sem_embs_list.append(sem_emb)
                sem_out = torch.stack(sem_embs_list, dim=1)  # [B, N, moe_hidden_dim=128]
        else:
            # 没有文本特征时，使用零向量（统一使用128维）
            if training_mode:
                sem_out = torch.zeros(batch_size, self.sasrec_hidden_dim, device=self.device)  # [B, 128]
            else:
                sem_out = torch.zeros(batch_size, num_candidates, self.sasrec_hidden_dim, device=self.device)  # [B, N, 128]

        # ===========================
        # 4. 残差增强融合 (Residual Enhancement Fusion)
        # ===========================
        # 应用LayerNorm到所有专家输出
        seq_out_norm = self.seq_layernorm(seq_out)    # [B, D] or [B, N, D]
        vis_out_norm = self.vis_layernorm(vis_out)    # [B, D] or [B, N, D]
        sem_out_norm = self.sem_layernorm(sem_out)    # [B, D] or [B, N, D]

        # 获取路由权重 (仅对辅助专家: Visual, Semantic)
        if training_mode:
            router_weights = self.router(target_item_embs)  # [B, 2]
        else:
            router_weights = self.router(target_item_embs)  # [B, N, 2]

        # 【修复】根据专家可用性调整权重，避免给零专家分配权重导致信息稀释
        expert_available = torch.tensor([
            1.0 if (memory_visual is not None and target_visual is not None) else 0.0,  # Visual
            1.0 if memory_text is not None else 0.0  # Semantic
        ], device=router_weights.device)

        # 将不可用专家的权重mask掉并重新归一化
        if training_mode:
            router_weights = router_weights * expert_available.view(1, 2)
        else:
            router_weights = router_weights * expert_available.view(1, 1, 2)
        router_weights = router_weights / (router_weights.sum(dim=-1, keepdim=True) + 1e-10)

        # 提取辅助专家的权重
        if training_mode:
            w_vis = router_weights[:, 0].unsqueeze(1)  # [B, 1]
            w_sem = router_weights[:, 1].unsqueeze(1)  # [B, 1]
        else:
            w_vis = router_weights[:, :, 0].unsqueeze(2)  # [B, N, 1]
            w_sem = router_weights[:, :, 1].unsqueeze(2)  # [B, N, 1]

        # [方案2] 使用CrossModalFusion层进行注意力融合
        if self.preserve_multimodal_dim and self.cross_modal_fusion is not None:
            # 方案2: 使用注意力融合（保持预训练特征完整性）
            if training_mode:
                # 训练模式: [B, D] 所有维度都是batch维度
                fused_repr = self.cross_modal_fusion(
                    seq_repr=seq_out_norm,
                    visual_repr=vis_out_norm,
                    text_repr=sem_out_norm
                )  # [B, D]
            else:
                # [方案2修复] 推理模式: 使用CrossModalFusion处理异构维度
                # seq_out_norm: [B, N, D], vis_out_norm: [B, N, 512], sem_out_norm: [B, N, 384]
                # 需要对每个候选物品分别进行融合
                fused_list = []
                for i in range(num_candidates):
                    fused_i = self.cross_modal_fusion(
                        seq_repr=seq_out_norm[:, i, :],      # [B, 128]
                        visual_repr=vis_out_norm[:, i, :],   # [B, 512]
                        text_repr=sem_out_norm[:, i, :]      # [B, 384]
                    )  # [B, 128]
                    fused_list.append(fused_i)
                fused_repr = torch.stack(fused_list, dim=1)  # [B, N, 128]
        else:
            # 原版: 简单加权和融合
            # [STAGE 2/3 FIX] 残差增强融合：使用可学习的gating_weight控制辅助信号强度
            # 优点1: gating_weight初始化为0.01时，保护Stage 1的SASRec输出不被破坏
            # 优点2: gating_weight可训练，模型自动学习最优的融合强度
            # 优点3: 辅助专家仍然接收梯度，可以正常训练（梯度通过gating_weight缩放）
            auxiliary_repr = w_vis * vis_out_norm + w_sem * sem_out_norm  # [B, D] or [B, N, D]
            fused_repr = seq_out_norm + self.gating_weight * auxiliary_repr  # [B, D] or [B, N, D]

        # [FIX] 保证 fused_repr / auxiliary_repr 在所有分支中都有定义（避免 return_components 时 UnboundLocalError）
        # preserve_multimodal_dim=True 时可能没有 auxiliary_repr；此外历史补丁可能导致 fused_repr 未在某些路径赋值。
        if 'fused_repr' not in locals():
            fused_repr = seq_out_norm  # 回退为纯SASRec表示
        if 'auxiliary_repr' not in locals():
            auxiliary_repr = torch.zeros_like(fused_repr)

        # ===========================
        # 5. 计算最终得分
        # ===========================
        # [FIX] 当gating_weight很小时（<0.001），使用不归一化的点积（与Stage 1一致）
        # 关键：必须使用 fused_repr（确保梯度流向投影层），但用不归一化的点积计算得分
        use_pure_sasrec_scoring = (self.gating_weight.item() < 0.001)

        if use_pure_sasrec_scoring:
            # 使用 fused_repr，但用不归一化的点积（与Stage 1评分一致）
            # 重要：这样梯度仍然流向投影层！
            if training_mode:
                # 训练模式：fused_repr [B, D] × target_item_embs^T [D, B] = [B, B]
                final_scores = torch.mm(fused_repr, target_item_embs.t())  # [B, B]
            else:
                # 推理模式：fused_repr [B, N, D] * target_item_embs [B, N, D] -> sum(dim=-1) = [B, N]
                final_scores = (fused_repr * target_item_embs).sum(dim=-1)  # [B, N]
        else:
            # 使用归一化计分方式（gating_weight >= 0.001，Stage 3多模态融合时）
            if training_mode:
                # [优化5] 批内负采样模式：计算 [B, D] × [D, B] = [B, B]
                # L2归一化
                fused_repr_norm = torch.nn.functional.normalize(fused_repr, p=2, dim=-1)  # [B, D]
                target_item_embs_norm = torch.nn.functional.normalize(target_item_embs, p=2, dim=-1)  # [B, D]

                # 计算分数矩阵
                scale = self.sasrec_hidden_dim ** 0.5
                final_scores = scale * torch.mm(fused_repr_norm, target_item_embs_norm.t())  # [B, B]
            else:
                # 推理模式：计算 [B, N] 得分
                # L2归一化
                fused_repr_norm = torch.nn.functional.normalize(fused_repr, p=2, dim=-1)  # [B, N, D]
                target_item_embs_norm = torch.nn.functional.normalize(target_item_embs, p=2, dim=-1)  # [B, N, D]

                # 计算最终得分：归一化后的余弦相似度
                scale = self.sasrec_hidden_dim ** 0.5
                final_scores = scale * (fused_repr_norm * target_item_embs_norm).sum(dim=-1)  # [B, N]

        # ===========================
        # 6. 负载均衡损失
        # ===========================
        # 计算每个辅助专家的平均使用率
        if training_mode:
            expert_usage = router_weights.mean(dim=0)  # [2]
        else:
            expert_usage = router_weights.mean(dim=[0, 1])  # [2]

        # 负载均衡损失: 鼓励辅助专家均匀使用
        # L_lb = Σ(usage_i - 1/N)^2
        target_usage = 1.0 / 2.0  # 均匀分配给2个辅助专家
        lb_loss = torch.sum((expert_usage - target_usage) ** 2)

        # ===========================
        # 7. 返回结果
        # ===========================
        if return_components:
            if training_mode:
                # [优化5] 批内负采样模式：返回 [B, D] 的表示向量
                info = {
                    'lb_loss': lb_loss,                 # 负载均衡损失
                    'vis_out': vis_out,                 # [B, D] 视觉专家表示
                    'sem_out': sem_out,                 # [B, D] 语义专家表示
                    'seq_out': seq_out,                 # [B, D] 序列骨干表示
                    'router_weights': router_weights,   # [B, 2] 辅助专家路由权重
                    'expert_usage': expert_usage,       # [2] 辅助专家使用率
                    'w_vis': w_vis.mean().item(),       # 视觉专家平均权重
                    'w_sem': w_sem.mean().item(),       # 语义专家平均权重
                }
            else:
                # 推理模式：返回 [B, N, D] 的表示向量和各专家得分
                seq_scores = (seq_out_norm * target_item_embs_norm).sum(dim=-1) * scale  # [B, N]

                # [FIX] preserve_multimodal_dim=True 时 vis_out_norm/sem_out_norm 维度可能为 512/384，
                # 而 target_item_embs_norm 是 SASRec hidden_dim（如 128），不能直接点乘。
                # vis_scores/sem_scores 仅用于组件分析，不参与训练目标；维度不一致时置零以避免崩溃。
                if vis_out_norm.size(-1) == target_item_embs_norm.size(-1):
                    vis_scores = (vis_out_norm * target_item_embs_norm).sum(dim=-1) * scale  # [B, N]
                else:
                    vis_scores = torch.zeros_like(seq_scores)

                if sem_out_norm.size(-1) == target_item_embs_norm.size(-1):
                    sem_scores = (sem_out_norm * target_item_embs_norm).sum(dim=-1) * scale  # [B, N]
                else:
                    sem_scores = torch.zeros_like(seq_scores)

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
        logits: torch.Tensor,      # [B, N] or [B, B]
        labels: torch.Tensor,      # [B]
        lb_loss: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        计算总损失

        Args:
            logits: 模型输出得分
                   - [B, N]: 推理模式，N个候选物品
                   - [B, B]: 训练模式（批内负采样），对角线为正样本
            labels: 标签 [B]
                   - 推理模式: 通常全0，表示正样本在第0个位置
                   - 训练模式: torch.arange(B)，表示正样本在对角线上
            lb_loss: 负载均衡损失 (如果已计算)

        Returns:
            total_loss: 总损失
            loss_dict: 各部分损失的字典
        """
        # [优化5] 统一使用交叉熵损失
        # - 推理模式 [B, N]: labels指示正样本在第几个候选中
        # - 训练模式 [B, B]: labels为[0,1,2,...,B-1]，正样本在对角线
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

    def compute_contrastive_loss(
        self,
        vis_repr: torch.Tensor,
        sem_repr: torch.Tensor,
        surprise_score: Optional[torch.Tensor] = None,
        base_temp: float = 0.07,
        alpha: float = 0.5
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        [FIX 2] 计算漂移自适应对比学习损失 (Drift-Adaptive Contrastive Loss)

        核心修复:
        - 修复前: 高surprise导致高temperature，使损失变小，告诉模型"忽略"困难样本
        - 修复后: 固定temperature，高surprise增加实例权重，告诉模型"关注"困难样本

        Args:
            vis_repr: 视觉专家表示 [Batch, Dim]
            sem_repr: 语义专家表示 [Batch, Dim]
            surprise_score: 惊讶度分数 [Batch] (归一化的标量，表示漂移/错误程度)
                           如果为None，则返回None权重
            base_temp: 固定温度 (默认0.07)
            alpha: 权重调节系数 (默认0.5)

        Returns:
            contrastive_loss: InfoNCE对比学习损失 (标量)
            instance_weights: 实例级损失权重 [Batch] 或 None
                            weights = 1.0 + alpha * surprise_score
                            高surprise的样本获得更高权重

        修复原理:
            - 修复前: temp↑ → loss↓ → 梯度↓ → 困难样本被忽略
            - 修复后: weight↑ → loss×weight↑ → 梯度↑ → 困难样本被强化学习
        """
        batch_size = vis_repr.size(0)

        # [方案2修复] 处理异构维度（512 vs 384）
        # 如果维度不匹配，投影到共享空间
        vis_dim = vis_repr.size(-1)
        sem_dim = sem_repr.size(-1)

        if vis_dim != sem_dim:
            # 使用动态投影层将两者投影到相同维度（取较小维度）
            target_dim = min(vis_dim, sem_dim)

            # 创建或获取投影层（缓存在模型中）
            if not hasattr(self, '_contrastive_vis_proj') or self._contrastive_vis_proj_dim != (vis_dim, target_dim):
                self._contrastive_vis_proj = nn.Linear(vis_dim, target_dim).to(vis_repr.device)
                self._contrastive_vis_proj_dim = (vis_dim, target_dim)
            if not hasattr(self, '_contrastive_sem_proj') or self._contrastive_sem_proj_dim != (sem_dim, target_dim):
                self._contrastive_sem_proj = nn.Linear(sem_dim, target_dim).to(sem_repr.device)
                self._contrastive_sem_proj_dim = (sem_dim, target_dim)

            vis_repr = self._contrastive_vis_proj(vis_repr)  # [B, target_dim]
            sem_repr = self._contrastive_sem_proj(sem_repr)  # [B, target_dim]

        # 步骤1: L2归一化表示向量
        vis_repr_norm = F.normalize(vis_repr, p=2, dim=-1)  # [B, D]
        sem_repr_norm = F.normalize(sem_repr, p=2, dim=-1)  # [B, D]

        # 步骤2: 计算相似度矩阵 (余弦相似度)
        similarity = torch.mm(vis_repr_norm, sem_repr_norm.t())  # [B, B]

        # [FIX 2] 步骤3: 使用固定温度（不再自适应）
        logits = similarity / base_temp  # [B, B]

        # 步骤4: InfoNCE损失 (逐样本计算，不使用reduction='mean')
        labels = torch.arange(batch_size, device=vis_repr.device)  # [B]

        # 计算交叉熵损失（双向对比，返回每个样本的损失）
        loss_vis2sem = F.cross_entropy(logits, labels, reduction='none')  # [B]
        loss_sem2vis = F.cross_entropy(logits.t(), labels, reduction='none')  # [B]

        # 平均两个方向的损失（保持[B]维度）
        per_sample_loss = (loss_vis2sem + loss_sem2vis) / 2.0  # [B]

        # [FIX 2] 步骤5: 计算实例级权重
        if surprise_score is not None:
            # 确保surprise_score是[B]形状
            if surprise_score.dim() == 0:
                surprise_score = surprise_score.unsqueeze(0).expand(batch_size)

            # 计算权重: 高surprise → 高权重 → 强化学习
            instance_weights = 1.0 + alpha * surprise_score  # [B]
        else:
            instance_weights = None

        # 返回未加权的平均损失和实例权重（在外部应用）
        contrastive_loss = per_sample_loss.mean()  # 标量

        return contrastive_loss, instance_weights


__all__ = ['UR4RecV2MoE', 'VisualExpert', 'SemanticExpert', 'ItemCentricRouter']
