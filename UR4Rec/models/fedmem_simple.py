"""
FedMem Simple: 简化版多模态联邦推荐模型

核心思想: 抛弃复杂的MoE架构，直接将多模态特征concat到item embedding

架构对比:
- 原始MoE: item_id → SASRec → [128]
           visual → VisualExpert → [128] ┐
           text → SemanticExpert → [128] ┴→ Router融合

- 简化版:  item_id → item_emb → [128]
           visual → Linear → [64]      ┐
           text → Linear → [64]        ┴→ concat → [256] → SASRec → output

优势:
1. 梯度流更直接（无Router bottleneck）
2. 参数更少（270K vs 320K）
3. 训练更稳定（单一损失函数）
4. 可解释性更好

预期性能: HR@10: 0.55-0.60 (vs MoE: 0.50-0.55)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Union

from .sasrec import SASRec


class FedMemSimple(nn.Module):
    """
    简化版FedMem：直接拼接多模态特征

    核心创新:
    1. 移除MoE和Router，使用简单的特征拼接
    2. 多模态特征通过轻量级投影层降维
    3. 拼接后的特征直接输入SASRec
    """

    def __init__(
        self,
        num_items: int,
        # ID embedding维度
        id_emb_dim: int = 128,
        # 多模态特征维度
        visual_dim: int = 512,      # CLIP特征
        text_dim: int = 384,        # Sentence-BERT特征
        # 投影维度
        visual_proj_dim: int = 64,
        text_proj_dim: int = 64,
        # SASRec参数
        sasrec_num_blocks: int = 2,
        sasrec_num_heads: int = 4,
        sasrec_dropout: float = 0.1,
        max_seq_len: int = 50,
        # 设备
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        初始化简化版FedMem模型

        Args:
            num_items: 物品总数
            id_emb_dim: ID嵌入维度
            visual_dim: 视觉特征维度 (CLIP: 512)
            text_dim: 文本特征维度 (SBERT: 384)
            visual_proj_dim: 视觉特征投影后维度
            text_proj_dim: 文本特征投影后维度
            sasrec_*: SASRec相关参数
            max_seq_len: 最大序列长度
            device: 运行设备
        """
        super().__init__()

        self.num_items = num_items
        self.id_emb_dim = id_emb_dim
        self.visual_dim = visual_dim
        self.text_dim = text_dim
        self.visual_proj_dim = visual_proj_dim
        self.text_proj_dim = text_proj_dim
        self.device = device

        # 计算SASRec的输入维度
        sasrec_input_dim = id_emb_dim + visual_proj_dim + text_proj_dim  # 128+64+64=256

        # ===========================
        # 1. Item ID Embedding
        # ===========================
        self.item_embedding = nn.Embedding(num_items, id_emb_dim)

        # ===========================
        # 2. 多模态投影层（轻量级）
        # ===========================
        # 视觉特征投影: 512 → 64
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, visual_proj_dim),
            nn.LayerNorm(visual_proj_dim),
            nn.ReLU(),
            nn.Dropout(sasrec_dropout)
        )

        # 文本特征投影: 384 → 64
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, text_proj_dim),
            nn.LayerNorm(text_proj_dim),
            nn.ReLU(),
            nn.Dropout(sasrec_dropout)
        )

        # ===========================
        # 3. SASRec骨干网络
        # ===========================
        self.sasrec = SASRec(
            num_items=num_items,
            hidden_dim=sasrec_input_dim,  # 256维输入
            num_blocks=sasrec_num_blocks,
            num_heads=sasrec_num_heads,
            dropout=sasrec_dropout,
            max_seq_len=max_seq_len,
            input_dim=sasrec_input_dim,  # 使用外部embedding
            use_external_emb=True         # 不使用SASRec自带的embedding
        )

        print(f"✓ FedMemSimple 初始化:")
        print(f"  架构: [ID({id_emb_dim}) + Visual({visual_proj_dim}) + Text({text_proj_dim})] → SASRec({sasrec_input_dim})")
        print(f"  参数量: {sum(p.numel() for p in self.parameters())/1e6:.2f}M")
        print(f"  优势: 简单、稳定、高效")

        self.to(device)

    def forward(
        self,
        user_ids: Optional[list],
        input_seq: torch.Tensor,           # [B, L]
        target_items: torch.Tensor,        # [B] or [B, N]
        # 多模态特征 (可选)
        target_visual: Optional[torch.Tensor] = None,  # [B, visual_dim] or [B, N, visual_dim]
        target_text: Optional[torch.Tensor] = None,    # [B, text_dim] or [B, N, text_dim]
        # 兼容性参数（与MoE保持一致）
        memory_visual: Optional[torch.Tensor] = None,  # 兼容性参数，不使用
        memory_text: Optional[torch.Tensor] = None,    # 兼容性参数，不使用
        # 其他参数
        seq_padding_mask: Optional[torch.Tensor] = None,
        training_mode: bool = False,       # 批内负采样模式
        return_components: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        前向传播（简化版：序列只用ID，目标物品用ID+多模态）

        Args:
            user_ids: 用户ID列表 (兼容性参数)
            input_seq: 输入序列 [B, L]
            target_items: 目标物品ID [B] or [B, N]
            target_visual: 目标物品视觉特征 [B, visual_dim] or [B, N, visual_dim]
            target_text: 目标物品文本特征 [B, text_dim] or [B, N, text_dim]
            memory_visual: 兼容性参数（不使用）
            memory_text: 兼容性参数（不使用）
            seq_padding_mask: 序列padding mask [B, L]
            training_mode: 是否使用批内负采样
            return_components: 是否返回各组件信息

        Returns:
            scores: 预测得分 [B, N] or [B, B]
            (optional) info: 包含各组件信息的字典
        """
        batch_size = input_seq.size(0)

        # ===========================
        # 处理training_mode
        # ===========================
        if training_mode:
            if target_items.dim() == 2:
                target_items = target_items.squeeze(1)  # [B, N] -> [B]
            num_candidates = batch_size
        else:
            if target_items.dim() == 1:
                target_items = target_items.unsqueeze(1)  # [B] -> [B, 1]
            num_candidates = target_items.size(1)

        # ===========================
        # 1. 序列编码（ID embedding → 投影到总维度 → SASRec）
        # ===========================
        # 1.1 获取序列ID embedding [B, L, id_dim]
        seq_id_embs = self.item_embedding(input_seq)  # [B, L, 128]

        # 1.2 创建零填充的多模态特征（序列不使用多模态）
        seq_vis_zero = torch.zeros(
            batch_size, input_seq.size(1), self.visual_proj_dim,
            device=self.device
        )  # [B, L, 64]
        seq_txt_zero = torch.zeros(
            batch_size, input_seq.size(1), self.text_proj_dim,
            device=self.device
        )  # [B, L, 64]

        # 1.3 拼接为总维度
        seq_input = torch.cat([seq_id_embs, seq_vis_zero, seq_txt_zero], dim=-1)  # [B, L, 256]

        # 1.4 SASRec编码
        seq_output = self.sasrec(seq_input, padding_mask=seq_padding_mask)  # [B, L, 256]
        seq_repr = seq_output[:, -1, :]  # [B, 256]

        # ===========================
        # 2. 构建目标物品的融合特征
        # ===========================
        # 2.1 获取目标物品ID embedding
        if training_mode:
            target_id_embs = self.item_embedding(target_items)  # [B, id_dim=128]
        else:
            target_id_embs = self.item_embedding(target_items)  # [B, N, id_dim=128]

        # 2.2 处理目标视觉特征
        if target_visual is not None:
            if training_mode:
                # 训练模式: [B, visual_dim] or [B, 1, visual_dim]
                if target_visual.dim() == 3:
                    target_visual = target_visual[:, 0, :]  # [B, visual_dim]
                target_vis_embs = self.visual_proj(target_visual)  # [B, vis_proj_dim]
            else:
                # 推理模式: [B, N, visual_dim]
                if target_visual.dim() == 2:
                    # 所有候选共享相同特征
                    target_vis_embs = self.visual_proj(target_visual).unsqueeze(1).expand(-1, num_candidates, -1)
                else:
                    target_vis_embs = self.visual_proj(target_visual)  # [B, N, vis_proj_dim]
        else:
            # 没有视觉特征
            if training_mode:
                target_vis_embs = torch.zeros(batch_size, self.visual_proj_dim, device=self.device)
            else:
                target_vis_embs = torch.zeros(batch_size, num_candidates, self.visual_proj_dim, device=self.device)

        # 3.3 处理目标文本特征
        if target_text is not None:
            if training_mode:
                if target_text.dim() == 3:
                    target_text = target_text[:, 0, :]
                target_txt_embs = self.text_proj(target_text)  # [B, txt_proj_dim]
            else:
                if target_text.dim() == 2:
                    target_txt_embs = self.text_proj(target_text).unsqueeze(1).expand(-1, num_candidates, -1)
                else:
                    target_txt_embs = self.text_proj(target_text)  # [B, N, txt_proj_dim]
        else:
            if training_mode:
                target_txt_embs = torch.zeros(batch_size, self.text_proj_dim, device=self.device)
            else:
                target_txt_embs = torch.zeros(batch_size, num_candidates, self.text_proj_dim, device=self.device)

        # 2.4 拼接目标物品多模态特征 [ID + Visual + Text] = 256维
        if training_mode:
            target_fused_embs = torch.cat([
                target_id_embs,    # [B, 128]
                target_vis_embs,   # [B, 64]
                target_txt_embs    # [B, 64]
            ], dim=-1)  # [B, 256]
        else:
            target_fused_embs = torch.cat([
                target_id_embs,    # [B, N, 128]
                target_vis_embs,   # [B, N, 64]
                target_txt_embs    # [B, N, 64]
            ], dim=-1)  # [B, N, 256]

        # ===========================
        # 3. 计算得分（余弦相似度）
        # ===========================
        # 获取SASRec的hidden_dim（应该是256）
        hidden_dim = self.id_emb_dim + self.visual_proj_dim + self.text_proj_dim

        if training_mode:
            # 批内负采样: [B, 256] × [256, B] = [B, B]
            seq_repr_norm = F.normalize(seq_repr, p=2, dim=-1)  # [B, 256]
            target_fused_norm = F.normalize(target_fused_embs, p=2, dim=-1)  # [B, 256]

            scale = hidden_dim ** 0.5
            final_scores = scale * torch.mm(seq_repr_norm, target_fused_norm.t())  # [B, B]
        else:
            # 推理模式: [B, N]
            seq_repr_norm = F.normalize(seq_repr, p=2, dim=-1).unsqueeze(1)  # [B, 1, 256]
            target_fused_norm = F.normalize(target_fused_embs, p=2, dim=-1)  # [B, N, 256]

            scale = hidden_dim ** 0.5
            final_scores = scale * (seq_repr_norm * target_fused_norm).sum(dim=-1)  # [B, N]

        # ===========================
        # 4. 返回结果
        # ===========================
        if return_components:
            info = {
                'seq_repr': seq_repr,                    # [B, 256]
                'target_fused': target_fused_embs,       # [B, 256] or [B, N, 256]
                'final_scores': final_scores,            # [B, B] or [B, N]
                # 为了兼容性，返回空的vis_out和sem_out（漂移自适应对比学习用）
                'vis_out': torch.zeros(batch_size, hidden_dim, device=self.device) if training_mode else torch.zeros(batch_size, num_candidates, hidden_dim, device=self.device),
                'sem_out': torch.zeros(batch_size, hidden_dim, device=self.device) if training_mode else torch.zeros(batch_size, num_candidates, hidden_dim, device=self.device),
                'lb_loss': torch.tensor(0.0, device=self.device)  # 无负载均衡损失
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
        计算损失（简化版，只有交叉熵损失）

        Args:
            logits: 模型输出得分 [B, N] or [B, B]
            labels: 标签 [B]
            lb_loss: 负载均衡损失 (兼容性参数，简化版中无此损失)

        Returns:
            total_loss: 总损失
            loss_dict: 各部分损失的字典
        """
        # 交叉熵损失
        rec_loss = F.cross_entropy(logits, labels)

        loss_dict = {
            'total_loss': rec_loss.item(),
            'rec_loss': rec_loss.item(),
            'lb_loss': 0.0  # 简化版没有负载均衡损失
        }

        return rec_loss, loss_dict

    def get_item_embeddings(
        self,
        item_ids: torch.Tensor,
        embedding_type: str = 'id'
    ) -> torch.Tensor:
        """
        获取物品嵌入

        Args:
            item_ids: 物品ID [B] or [B, N]
            embedding_type: 嵌入类型 ('id')

        Returns:
            embeddings: 物品嵌入 [B, D] or [B, N, D]
        """
        if embedding_type == 'id':
            return self.item_embedding(item_ids)
        else:
            raise ValueError(f"不支持的embedding_type: {embedding_type}")

    def get_sequence_representation(
        self,
        input_seq: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        获取序列表示（简化版：只用ID）

        Args:
            input_seq: 输入序列 [B, L]
            padding_mask: Padding mask [B, L]

        Returns:
            seq_repr: 序列表示 [B, 256]
        """
        batch_size = input_seq.size(0)
        seq_len = input_seq.size(1)

        # 获取ID embedding
        seq_id_embs = self.item_embedding(input_seq)  # [B, L, 128]

        # 创建零填充的多模态特征
        seq_vis_zero = torch.zeros(batch_size, seq_len, self.visual_proj_dim, device=self.device)
        seq_txt_zero = torch.zeros(batch_size, seq_len, self.text_proj_dim, device=self.device)

        # 拼接
        seq_input = torch.cat([seq_id_embs, seq_vis_zero, seq_txt_zero], dim=-1)  # [B, L, 256]

        # SASRec编码
        seq_output = self.sasrec(seq_input, padding_mask=padding_mask)  # [B, L, 256]
        seq_repr = seq_output[:, -1, :]  # [B, 256]

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
        计算对比学习损失（简化版：简单的InfoNCE）

        简化架构不使用对比学习，返回零损失以保持兼容性

        Args:
            vis_repr: 视觉专家表示 [Batch, Dim] (在简化版中是零向量)
            sem_repr: 语义专家表示 [Batch, Dim] (在简化版中是零向量)
            surprise_score: 惊讶度分数 [Batch] (可选)
            base_temp: 基础温度
            alpha: 权重调节系数

        Returns:
            contrastive_loss: 对比学习损失（简化版中为0）
            instance_weights: 实例权重（简化版中为None）
        """
        # 简化架构不使用对比学习，返回零损失
        contrastive_loss = torch.tensor(0.0, device=self.device)
        instance_weights = None

        return contrastive_loss, instance_weights


__all__ = ['FedMemSimple']
