"""
FedDMMR: Federated Deep Multimodal Memory Recommendation

实现基于场景自适应异构混合专家(Scenario-Adaptive Heterogeneous MoE)的联邦推荐系统

核心组件:
1. SASRec: 序列模式建模骨干网络
2. VisualExpert: 视觉特征专家 (基于多模态记忆)
3. SemanticExpert: 语义特征专家 (基于文本记忆)
4. ItemCentricRouter: 以目标物品为中心的路由器

修改记录 (Optimization for FedSASRec alignment):
1. [Init] 增加 _init_weights 方法，强制 Embedding std=0.02
2. [Loss] compute_loss 改为 BPR Loss
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
        if values is None:
            values = keys

        q = self.query_proj(query)  # [B, hidden_dim]
        k = self.key_proj(keys)     # [B, TopK, hidden_dim]

        scores = torch.bmm(k, q.unsqueeze(2)).squeeze(2) / self.scale  # [B, TopK]
        weights = F.softmax(scores, dim=1)  # [B, TopK]

        output = torch.bmm(weights.unsqueeze(1), values).squeeze(1)  # [B, value_dim]
        return output


class VisualExpert(nn.Module):
    """
    视觉特征专家（方案2：保持原始维度）
    """

    def __init__(
        self,
        visual_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        preserve_dim: bool = True
    ):
        super().__init__()
        self.visual_dim = visual_dim
        self.hidden_dim = hidden_dim
        self.preserve_dim = preserve_dim

        self.attention = LightweightAttention(
            query_dim=visual_dim,
            key_dim=visual_dim,
            hidden_dim=min(visual_dim, 128)
        )

        if preserve_dim:
            self.output_proj = nn.Sequential(
                nn.Linear(visual_dim, visual_dim),
                nn.LayerNorm(visual_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.output_dim = visual_dim
        else:
            self.output_proj = nn.Sequential(
                nn.Linear(visual_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.output_dim = hidden_dim

    def forward(self, target_visual: torch.Tensor, memory_visual: torch.Tensor) -> torch.Tensor:
        aggregated = self.attention(
            query=target_visual,
            keys=memory_visual,
            values=memory_visual
        )
        visual_emb = self.output_proj(aggregated)
        return visual_emb


class SemanticExpert(nn.Module):
    """
    语义特征专家（方案2：保持原始维度）
    """

    def __init__(
        self,
        id_dim: int,
        text_dim: int,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        preserve_dim: bool = True
    ):
        super().__init__()
        self.id_dim = id_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.preserve_dim = preserve_dim

        if preserve_dim:
            self.query_proj = nn.Linear(id_dim, text_dim)
            self.key_proj = nn.Linear(text_dim, text_dim)
            self.value_proj = nn.Linear(text_dim, text_dim)
            self.attn_dim = text_dim
            self.output_dim = text_dim
        else:
            self.query_proj = nn.Linear(id_dim, hidden_dim)
            self.key_proj = nn.Linear(text_dim, hidden_dim)
            self.value_proj = nn.Linear(text_dim, hidden_dim)
            self.attn_dim = hidden_dim
            self.output_dim = hidden_dim

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.attn_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.output_layer = nn.Sequential(
            nn.LayerNorm(self.attn_dim),
            nn.Linear(self.attn_dim, self.output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, target_id_emb: torch.Tensor, memory_text: torch.Tensor) -> torch.Tensor:
        query = self.query_proj(target_id_emb).unsqueeze(1)
        keys = self.key_proj(memory_text)
        values = self.value_proj(memory_text)

        attn_output, _ = self.cross_attention(query=query, key=keys, value=values)
        attn_output = attn_output.squeeze(1)
        semantic_emb = self.output_layer(attn_output)

        return semantic_emb


class CrossModalFusion(nn.Module):
    """
    跨模态融合层
    """

    def __init__(
        self,
        query_dim: int,
        visual_dim: int,
        text_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.query_dim = query_dim
        self.visual_dim = visual_dim
        self.text_dim = text_dim

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

        # Fusion
        self.fusion_layer = nn.Sequential(
            nn.LayerNorm(query_dim * 3),
            nn.Linear(query_dim * 3, query_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, seq_repr: torch.Tensor, visual_repr: torch.Tensor, text_repr: torch.Tensor) -> torch.Tensor:
        vis_query = self.visual_query_proj(seq_repr).unsqueeze(1)
        vis_key = self.visual_key_proj(visual_repr).unsqueeze(1)
        vis_value = self.visual_value_proj(visual_repr).unsqueeze(1)
        vis_attn_out, _ = self.visual_attn(query=vis_query, key=vis_key, value=vis_value)
        vis_out = self.visual_out_proj(vis_attn_out.squeeze(1))

        text_query = self.text_query_proj(seq_repr).unsqueeze(1)
        text_key = self.text_key_proj(text_repr).unsqueeze(1)
        text_value = self.text_value_proj(text_repr).unsqueeze(1)
        text_attn_out, _ = self.text_attn(query=text_query, key=text_key, value=text_value)
        text_out = self.text_out_proj(text_attn_out.squeeze(1))

        concatenated = torch.cat([seq_repr, vis_out, text_out], dim=1)
        fused_repr = self.fusion_layer(concatenated)

        return fused_repr


class ItemCentricRouter(nn.Module):
    """
    以物品为中心的路由器
    """

    def __init__(
        self,
        item_emb_dim: int,
        hidden_dim: int = 128,
        num_experts: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_experts = num_experts
        self.router = nn.Sequential(
            nn.Linear(item_emb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, target_item_emb: torch.Tensor) -> torch.Tensor:
        weights = self.router(target_item_emb)
        return weights


class UR4RecV2MoE(nn.Module):
    """
    FedDMMR: Residual Enhancement with Heterogeneous MoE
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
        visual_dim: int = 512,
        text_dim: int = 384,
        # MoE参数
        moe_hidden_dim: int = 256,
        moe_num_heads: int = 4,
        moe_dropout: float = 0.1,
        router_hidden_dim: int = 128,
        # 残差增强参数
        gating_init: float = 1.0,
        # 负载均衡
        load_balance_lambda: float = 0.01,
        # 兼容性参数
        init_bias_for_sasrec: bool = False,
        sasrec_bias_value: float = 5.0,
        # 设备
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()

        self.num_items = num_items
        self.sasrec_hidden_dim = sasrec_hidden_dim
        self.moe_hidden_dim = moe_hidden_dim
        self.visual_dim = visual_dim
        self.text_dim = text_dim
        self.load_balance_lambda = load_balance_lambda
        self.device = device

        if moe_hidden_dim != sasrec_hidden_dim:
            print(f"警告: moe_hidden_dim ({moe_hidden_dim}) 与 sasrec_hidden_dim ({sasrec_hidden_dim}) 不一致")

        # 1. 序列骨干: SASRec
        self.sasrec = SASRec(
            num_items=num_items,
            hidden_dim=sasrec_hidden_dim,
            num_blocks=sasrec_num_blocks,
            num_heads=sasrec_num_heads,
            dropout=sasrec_dropout,
            max_seq_len=max_seq_len
        )

        # 1.5. 轻量级投影层
        self.visual_proj = nn.Linear(visual_dim, sasrec_hidden_dim)
        self.text_proj = nn.Linear(text_dim, sasrec_hidden_dim)

        self.align_gating = nn.Sequential(
            nn.Linear(sasrec_hidden_dim * 3, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2),
            nn.Softmax(dim=-1)
        )

        # 2. 视觉专家
        self.preserve_multimodal_dim = False
        self.visual_expert = VisualExpert(
            visual_dim=sasrec_hidden_dim,
            hidden_dim=moe_hidden_dim,
            dropout=moe_dropout,
            preserve_dim=self.preserve_multimodal_dim
        )

        # 3. 语义专家
        self.semantic_expert = SemanticExpert(
            id_dim=sasrec_hidden_dim,
            text_dim=sasrec_hidden_dim,
            hidden_dim=moe_hidden_dim,
            num_heads=moe_num_heads,
            dropout=moe_dropout,
            preserve_dim=self.preserve_multimodal_dim
        )

        # 3.5 跨模态融合层
        self.cross_modal_fusion = CrossModalFusion(
            query_dim=sasrec_hidden_dim,
            visual_dim=sasrec_hidden_dim,
            text_dim=sasrec_hidden_dim,
            num_heads=moe_num_heads,
            dropout=moe_dropout
        )

        # 4. 路由器
        self.router = ItemCentricRouter(
            item_emb_dim=sasrec_hidden_dim * 2,
            hidden_dim=router_hidden_dim,
            num_experts=2,
            dropout=moe_dropout
        )

        # 5. LayerNorm
        self.seq_layernorm = nn.LayerNorm(sasrec_hidden_dim)
        self.vis_layernorm = nn.LayerNorm(sasrec_hidden_dim)
        self.sem_layernorm = nn.LayerNorm(sasrec_hidden_dim)

        # 6. 可学习的门控权重
        self.gating_weight = nn.Parameter(torch.tensor(gating_init, dtype=torch.float32))

        # [NEW] 初始化权重：关键步骤，强制使用正确的初始化
        self.apply(self._init_weights)

        self.to(device)

    def _init_weights(self, module):
        """
        初始化权重 - 移植自 FedSASRec (sasrec_fixed.py)
        强制使用 std=0.02 初始化 Embedding，这对 Transformer 收敛至关重要
        """
        if isinstance(module, nn.Embedding):
            # [KEY FIX] 使用 std=0.02 而不是默认的 1.0
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, 'padding_idx') and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(
        self,
        user_ids: Optional[List[int]],
        input_seq: torch.Tensor,
        target_items: torch.Tensor,
        target_visual: Optional[torch.Tensor] = None,
        target_text: Optional[torch.Tensor] = None,
        memory_visual: Optional[torch.Tensor] = None,
        memory_text: Optional[torch.Tensor] = None,
        seq_padding_mask: Optional[torch.Tensor] = None,
        update_memory: bool = False,
        return_components: bool = False,
        retrieved_memory: Optional[Dict] = None,
        training_mode: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        
        batch_size = input_seq.size(0)

        if training_mode:
            # 训练模式：target_items [B]，使用批内负采样
            if target_items.dim() == 2:
                target_items = target_items.squeeze(1)
            num_candidates = batch_size
        else:
            # 推理模式：target_items [B, N] (FedMemClient通常使用这种模式进行显式负采样训练)
            if target_items.dim() == 1:
                target_items = target_items.unsqueeze(1)
            num_candidates = target_items.size(1)

        # 1. 序列专家: SASRec
        seq_output = self.sasrec(input_seq, padding_mask=seq_padding_mask)
        seq_repr = seq_output[:, -1, :]

        # 获取目标物品嵌入
        if training_mode:
            target_item_embs = self.sasrec.item_embedding(target_items)
            seq_out = seq_repr
        else:
            target_item_embs = self.sasrec.item_embedding(target_items)
            seq_out = seq_repr.unsqueeze(1).expand(-1, num_candidates, -1)

        # 如果没有多模态特征，直接返回SASRec结果（加速）
        if memory_visual is None and target_visual is None and memory_text is None:
            if training_mode:
                final_scores = torch.mm(seq_repr, target_item_embs.t())
                if return_components:
                    zero_out = torch.zeros(batch_size, self.moe_hidden_dim, device=self.device)
                    return final_scores, {'lb_loss': torch.tensor(0.0, device=self.device), 'vis_out': zero_out, 'sem_out': zero_out}
                return final_scores
            else:
                final_scores = torch.bmm(
                    seq_repr.unsqueeze(1),
                    target_item_embs.transpose(1, 2)
                ).squeeze(1)
                if return_components:
                    zero_out = torch.zeros(batch_size, num_candidates, self.moe_hidden_dim, device=self.device)
                    return final_scores, {'lb_loss': torch.tensor(0.0, device=self.device), 'vis_out': zero_out, 'sem_out': zero_out}
                return final_scores

        # 2. 视觉专家处理
        if memory_visual is not None and target_visual is not None:
            if training_mode:
                if target_visual.dim() == 3: target_visual_single = target_visual[:, 0, :]
                else: target_visual_single = target_visual
                target_visual_proj = self.visual_proj(target_visual_single)
                
                B, TopK, _ = memory_visual.shape
                memory_visual_proj = self.visual_proj(memory_visual.view(B * TopK, -1)).view(B, TopK, -1)
                vis_out = self.visual_expert(target_visual_proj, memory_visual_proj)
            else:
                if target_visual.dim() == 3:
                    B, N, _ = target_visual.shape
                    target_visual_proj = self.visual_proj(target_visual.view(B * N, -1)).view(B, N, -1)
                    
                    B, TopK, _ = memory_visual.shape
                    memory_visual_proj = self.visual_proj(memory_visual.view(B * TopK, -1)).view(B, TopK, -1)
                    
                    vis_embs_list = []
                    for i in range(num_candidates):
                        vis_emb = self.visual_expert(target_visual_proj[:, i, :], memory_visual_proj)
                        vis_embs_list.append(vis_emb)
                    vis_out = torch.stack(vis_embs_list, dim=1)
                else:
                    target_visual_proj = self.visual_proj(target_visual)
                    B, TopK, _ = memory_visual.shape
                    memory_visual_proj = self.visual_proj(memory_visual.view(B * TopK, -1)).view(B, TopK, -1)
                    vis_emb = self.visual_expert(target_visual_proj, memory_visual_proj)
                    vis_out = vis_emb.unsqueeze(1).expand(-1, num_candidates, -1)
        else:
            if training_mode: vis_out = torch.zeros(batch_size, self.sasrec_hidden_dim, device=self.device)
            else: vis_out = torch.zeros(batch_size, num_candidates, self.sasrec_hidden_dim, device=self.device)

        # 3. 语义专家处理
        if memory_text is not None:
            B, TopK, _ = memory_text.shape
            memory_text_proj = self.text_proj(memory_text.view(B * TopK, -1)).view(B, TopK, -1)
            
            if training_mode:
                sem_out = self.semantic_expert(target_item_embs, memory_text_proj)
            else:
                sem_embs_list = []
                for i in range(num_candidates):
                    sem_emb = self.semantic_expert(target_item_embs[:, i, :], memory_text_proj)
                    sem_embs_list.append(sem_emb)
                sem_out = torch.stack(sem_embs_list, dim=1)
        else:
            if training_mode: sem_out = torch.zeros(batch_size, self.sasrec_hidden_dim, device=self.device)
            else: sem_out = torch.zeros(batch_size, num_candidates, self.sasrec_hidden_dim, device=self.device)

        # 4. 融合与路由
        seq_out_norm = self.seq_layernorm(seq_out)
        vis_out_norm = self.vis_layernorm(vis_out)
        sem_out_norm = self.sem_layernorm(sem_out)

        # Router Input
        if training_mode:
            if target_visual is not None:
                visual_aligned = self.visual_proj(target_visual[:, 0, :] if target_visual.dim() == 3 else target_visual)
            else:
                visual_aligned = torch.zeros(batch_size, self.sasrec_hidden_dim, device=self.device)
                
            if target_text is not None:
                text_aligned = self.text_proj(target_text[:, 0, :] if target_text.dim() == 3 else target_text)
            else:
                text_aligned = torch.zeros(batch_size, self.sasrec_hidden_dim, device=self.device)
                
            router_input = torch.cat([visual_aligned, text_aligned], dim=-1)
            router_weights = self.router(router_input)
        else:
            if target_visual is not None and target_visual.dim() == 3:
                B, N, _ = target_visual.shape
                visual_aligned = self.visual_proj(target_visual.view(B * N, -1)).view(B, N, -1)
            elif target_visual is not None:
                visual_aligned = self.visual_proj(target_visual).unsqueeze(1).expand(-1, num_candidates, -1)
            else:
                visual_aligned = torch.zeros(batch_size, num_candidates, self.sasrec_hidden_dim, device=self.device)

            if target_text is not None and target_text.dim() == 3:
                B, N, _ = target_text.shape
                text_aligned = self.text_proj(target_text.view(B * N, -1)).view(B, N, -1)
            elif target_text is not None:
                text_aligned = self.text_proj(target_text).unsqueeze(1).expand(-1, num_candidates, -1)
            else:
                text_aligned = torch.zeros(batch_size, num_candidates, self.sasrec_hidden_dim, device=self.device)

            router_input = torch.cat([visual_aligned, text_aligned], dim=-1)
            router_weights = self.router(router_input)

        # 掩码和归一化Router权重
        expert_available = torch.tensor([
            1.0 if (memory_visual is not None and target_visual is not None) else 0.0,
            1.0 if memory_text is not None else 0.0
        ], device=router_weights.device)

        if training_mode: router_weights = router_weights * expert_available.view(1, 2)
        else: router_weights = router_weights * expert_available.view(1, 1, 2)
        router_weights = router_weights / (router_weights.sum(dim=-1, keepdim=True) + 1e-10)

        if training_mode:
            w_vis = router_weights[:, 0].unsqueeze(1)
            w_sem = router_weights[:, 1].unsqueeze(1)
        else:
            w_vis = router_weights[:, :, 0].unsqueeze(2)
            w_sem = router_weights[:, :, 1].unsqueeze(2)

        # 融合
        if self.cross_modal_fusion is not None:
            if training_mode:
                fused_repr = self.cross_modal_fusion(seq_out_norm, vis_out_norm, sem_out_norm)
            else:
                fused_list = []
                for i in range(num_candidates):
                    fused_i = self.cross_modal_fusion(seq_out_norm[:, i, :], vis_out_norm[:, i, :], sem_out_norm[:, i, :])
                    fused_list.append(fused_i)
                fused_repr = torch.stack(fused_list, dim=1)
        else:
            auxiliary_repr = w_vis * vis_out_norm + w_sem * sem_out_norm
            fused_repr = seq_out_norm + self.gating_weight * auxiliary_repr

        # 5. 计算得分
        use_pure_sasrec_scoring = (self.gating_weight.item() < 0.001)

        if use_pure_sasrec_scoring:
            if training_mode:
                final_scores = torch.mm(fused_repr, target_item_embs.t())
            else:
                final_scores = (fused_repr * target_item_embs).sum(dim=-1)
        else:
            if training_mode:
                fused_repr_norm = F.normalize(fused_repr, p=2, dim=-1)
                target_item_embs_norm = F.normalize(target_item_embs, p=2, dim=-1)
                scale = self.sasrec_hidden_dim ** 0.5
                final_scores = scale * torch.mm(fused_repr_norm, target_item_embs_norm.t())
            else:
                fused_repr_norm = F.normalize(fused_repr, p=2, dim=-1)
                target_item_embs_norm = F.normalize(target_item_embs, p=2, dim=-1)
                scale = self.sasrec_hidden_dim ** 0.5
                final_scores = scale * (fused_repr_norm * target_item_embs_norm).sum(dim=-1)

        # 6. 负载均衡损失
        if training_mode: expert_usage = router_weights.mean(dim=0)
        else: expert_usage = router_weights.mean(dim=[0, 1])
        lb_loss = torch.sum((expert_usage - 0.5) ** 2)

        if return_components:
            info = {
                'lb_loss': lb_loss,
                'vis_out': vis_out,
                'sem_out': sem_out,
                'seq_out': seq_out,
                'router_weights': router_weights,
                'fused_repr': fused_repr
            }
            return final_scores, info
        else:
            return final_scores

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lb_loss: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        计算总损失 - [KEY FIX] 使用 BPR Loss

        Args:
            logits: 模型输出得分
                   - [B, 1+N]: FedMemClient在training_mode=False时返回 (Pos + Negs)
                   - [B, B]: training_mode=True时返回 (Batch Negatives)
            labels: 标签 (BPR下通常不需要具体值，只通过位置判断)
        """
        # 1. 识别 Logits 结构并应用 BPR
        # FedMemClient 使用 training_mode=False, 传入 logits 形状为 [B, 1+N]
        # 其中 index 0 是正样本，index 1: 是负样本
        
        if logits.dim() == 2 and logits.size(1) > 1:
            # 显式负采样模式 (FedMemClient default)
            pos_scores = logits[:, 0]      # [B]
            neg_scores = logits[:, 1:]     # [B, N]
            
            # BPR Loss: -log_sigmoid(pos - neg)
            # 广播: [B, 1] - [B, N] -> [B, N]
            diff = pos_scores.unsqueeze(1) - neg_scores
            rec_loss = -F.logsigmoid(diff).mean()
            
        elif logits.dim() == 2 and logits.size(0) == logits.size(1):
            # 批内负采样模式 (training_mode=True)
            # 对角线是正样本，其他是负样本
            # 这种情况下 CrossEntropy 也是合理的，但为了对齐 FedSASRec，我们可以模拟 BPR
            pos_scores = torch.diag(logits)  # [B]
            # 生成mask: 非对角线元素为1
            mask = ~torch.eye(logits.size(0), dtype=torch.bool, device=logits.device)
            # 取出负样本 (flatten然后reshape) - 注意这在Batch较大时计算量大
            # 简单起见，这里回退到 CE，或者只计算部分负样本
            # 考虑到 FedMemClient 主要用上面那种模式，这里保留 CE 作为 fallback
            rec_loss = F.cross_entropy(logits, labels)
            
        else:
            # Fallback
            rec_loss = F.cross_entropy(logits, labels)

        # 2. 负载均衡损失
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

    # ... (其他辅助方法如 get_item_embeddings, get_sequence_representation, compute_contrastive_loss 保持不变) ...
    def get_item_embeddings(self, item_ids: torch.Tensor, embedding_type: str = 'id') -> torch.Tensor:
        if embedding_type == 'id':
            return self.sasrec.item_embedding(item_ids)
        raise ValueError(f"不支持的embedding_type: {embedding_type}")

    def get_sequence_representation(self, input_seq: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        seq_output = self.sasrec(input_seq, padding_mask=padding_mask)
        seq_repr = seq_output[:, -1, :]
        return seq_repr

    def compute_contrastive_loss(
        self,
        vis_repr: torch.Tensor,
        sem_repr: torch.Tensor,
        surprise_score: Optional[torch.Tensor] = None,
        base_temp: float = 0.07,
        alpha: float = 0.5
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        vis_dim = vis_repr.size(-1)
        sem_dim = sem_repr.size(-1)
        if vis_dim != sem_dim:
            target_dim = min(vis_dim, sem_dim)
            if not hasattr(self, '_contrastive_vis_proj') or self._contrastive_vis_proj_dim != (vis_dim, target_dim):
                self._contrastive_vis_proj = nn.Linear(vis_dim, target_dim).to(vis_repr.device)
                self._contrastive_vis_proj_dim = (vis_dim, target_dim)
            if not hasattr(self, '_contrastive_sem_proj') or self._contrastive_sem_proj_dim != (sem_dim, target_dim):
                self._contrastive_sem_proj = nn.Linear(sem_dim, target_dim).to(sem_repr.device)
                self._contrastive_sem_proj_dim = (sem_dim, target_dim)
            vis_repr = self._contrastive_vis_proj(vis_repr)
            sem_repr = self._contrastive_sem_proj(sem_repr)

        vis_repr_norm = F.normalize(vis_repr, p=2, dim=-1)
        sem_repr_norm = F.normalize(sem_repr, p=2, dim=-1)
        similarity = torch.mm(vis_repr_norm, sem_repr_norm.t())
        logits = similarity / base_temp
        
        batch_size = vis_repr.size(0)
        labels = torch.arange(batch_size, device=vis_repr.device)
        loss_vis2sem = F.cross_entropy(logits, labels, reduction='none')
        loss_sem2vis = F.cross_entropy(logits.t(), labels, reduction='none')
        per_sample_loss = (loss_vis2sem + loss_sem2vis) / 2.0

        if surprise_score is not None:
            if surprise_score.dim() == 0:
                surprise_score = surprise_score.unsqueeze(0).expand(batch_size)
            instance_weights = 1.0 + alpha * surprise_score
        else:
            instance_weights = None

        contrastive_loss = per_sample_loss.mean()
        return contrastive_loss, instance_weights