"""
多模态偏好检索器

结合文本和图像信息进行用户偏好检索
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np


class CLIPVisualEncoder(nn.Module):
    """
    CLIP 视觉编码器
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        output_dim: int = 256,
        freeze: bool = True
    ):
        """
        Args:
            model_name: CLIP 模型名称
            output_dim: 输出维度
            freeze: 是否冻结预训练权重
        """
        super().__init__()

        try:
            from transformers import CLIPVisionModel, CLIPProcessor
            self.vision_model = CLIPVisionModel.from_pretrained(model_name)
            self.processor = CLIPProcessor.from_pretrained(model_name)

            if freeze:
                for param in self.vision_model.parameters():
                    param.requires_grad = False

            # 获取 CLIP 输出维度
            self.clip_dim = self.vision_model.config.hidden_size

        except ImportError:
            print("警告: transformers 未安装 CLIP，使用模拟编码器")
            self.vision_model = None
            self.processor = None
            self.clip_dim = 512

        # 投影到统一空间
        self.projection = nn.Sequential(
            nn.Linear(self.clip_dim, output_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        编码图像

        Args:
            images: [batch_size, 3, H, W] - 图像张量

        Returns:
            image_features: [batch_size, output_dim]
        """
        if self.vision_model:
            with torch.no_grad():
                outputs = self.vision_model(pixel_values=images)
                # 使用 pooler_output
                image_features = outputs.pooler_output
        else:
            # 模拟特征
            batch_size = images.size(0)
            image_features = torch.randn(batch_size, self.clip_dim).to(images.device)

        # 投影
        image_features = self.projection(image_features)

        return image_features

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.encode_images(images)


class CrossModalAttention(nn.Module):
    """
    跨模态注意力机制

    允许文本和图像特征相互关注
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Args:
            embed_dim: 嵌入维度
            num_heads: 注意力头数
            dropout: Dropout 率
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # 文本->图像的注意力
        self.text_to_image = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

        # 图像->文本的注意力
        self.image_to_text = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Layer Norm
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            text_features: [batch_size, text_seq_len, embed_dim]
            image_features: [batch_size, image_seq_len, embed_dim]

        Returns:
            attended_text: [batch_size, text_seq_len, embed_dim]
            attended_image: [batch_size, image_seq_len, embed_dim]
        """
        # 文本关注图像
        attn_text, _ = self.text_to_image(
            text_features, image_features, image_features
        )
        attended_text = self.norm1(text_features + self.dropout(attn_text))

        # 图像关注文本
        attn_image, _ = self.image_to_text(
            image_features, text_features, text_features
        )
        attended_image = self.norm2(image_features + self.dropout(attn_image))

        return attended_text, attended_image


class MultiModalFusion(nn.Module):
    """
    多模态融合模块
    """

    def __init__(
        self,
        text_dim: int = 256,
        image_dim: int = 256,
        output_dim: int = 256,
        fusion_method: str = "concat"  # 'concat', 'add', 'gated'
    ):
        """
        Args:
            text_dim: 文本特征维度
            image_dim: 图像特征维度
            output_dim: 输出维度
            fusion_method: 融合方法
        """
        super().__init__()

        self.fusion_method = fusion_method

        if fusion_method == "concat":
            # 拼接融合
            self.fusion = nn.Sequential(
                nn.Linear(text_dim + image_dim, output_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(output_dim * 2, output_dim),
                nn.LayerNorm(output_dim)
            )

        elif fusion_method == "add":
            # 加法融合（需要维度一致）
            assert text_dim == image_dim == output_dim
            self.fusion = nn.LayerNorm(output_dim)

        elif fusion_method == "gated":
            # 门控融合
            assert text_dim == image_dim
            self.gate = nn.Sequential(
                nn.Linear(text_dim * 2, text_dim),
                nn.Sigmoid()
            )
            self.projection = nn.Linear(text_dim, output_dim)

    def forward(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            text_features: [batch_size, text_dim]
            image_features: [batch_size, image_dim]

        Returns:
            fused_features: [batch_size, output_dim]
        """
        if self.fusion_method == "concat":
            combined = torch.cat([text_features, image_features], dim=-1)
            fused = self.fusion(combined)

        elif self.fusion_method == "add":
            fused = self.fusion(text_features + image_features)

        elif self.fusion_method == "gated":
            # 计算门控权重
            combined = torch.cat([text_features, image_features], dim=-1)
            gate = self.gate(combined)

            # 门控融合
            gated = gate * text_features + (1 - gate) * image_features
            fused = self.projection(gated)

        return fused


class MultiModalPreferenceRetriever(nn.Module):
    """
    多模态用户偏好检索器

    结合文本偏好和视觉偏好进行检索
    """

    def __init__(
        self,
        num_items: int,
        # 文本编码器参数
        text_model_name: str = "all-MiniLM-L6-v2",
        text_embedding_dim: int = 384,
        # 视觉编码器参数
        vision_model_name: str = "openai/clip-vit-base-patch32",
        # 统一表示维度
        unified_dim: int = 256,
        # 跨模态注意力
        use_cross_attention: bool = True,
        num_attention_heads: int = 8,
        # 融合方法
        fusion_method: str = "concat",
        # 其他
        dropout: float = 0.1
    ):
        """
        Args:
            num_items: 物品数量
            text_*: 文本编码器参数
            vision_*: 视觉编码器参数
            unified_dim: 统一表示维度
            use_cross_attention: 是否使用跨模态注意力
            num_attention_heads: 注意力头数
            fusion_method: 融合方法
            dropout: Dropout 率
        """
        super().__init__()

        self.num_items = num_items
        self.unified_dim = unified_dim
        self.use_cross_attention = use_cross_attention

        # 文本编码器
        try:
            from sentence_transformers import SentenceTransformer
            self.text_encoder = SentenceTransformer(text_model_name)
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            text_output_dim = self.text_encoder.get_sentence_embedding_dimension()
        except ImportError:
            print("警告: sentence-transformers 未安装")
            self.text_encoder = None
            text_output_dim = text_embedding_dim

        # 文本投影
        self.text_projection = nn.Sequential(
            nn.Linear(text_output_dim, unified_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(unified_dim, unified_dim),
            nn.LayerNorm(unified_dim)
        )

        # 视觉编码器
        self.vision_encoder = CLIPVisualEncoder(
            model_name=vision_model_name,
            output_dim=unified_dim
        )

        # 跨模态注意力
        if use_cross_attention:
            self.cross_attention = CrossModalAttention(
                embed_dim=unified_dim,
                num_heads=num_attention_heads,
                dropout=dropout
            )

        # 多模态融合
        self.fusion = MultiModalFusion(
            text_dim=unified_dim,
            image_dim=unified_dim,
            output_dim=unified_dim,
            fusion_method=fusion_method
        )

        # 物品嵌入（多模态）
        # 初始化为文本+视觉的融合表示
        self.item_text_embeddings = nn.Embedding(
            num_items + 1, unified_dim, padding_idx=0
        )
        self.item_visual_embeddings = nn.Embedding(
            num_items + 1, unified_dim, padding_idx=0
        )

        # 初始化
        nn.init.xavier_uniform_(self.item_text_embeddings.weight[1:])
        nn.init.xavier_uniform_(self.item_visual_embeddings.weight[1:])

    def encode_text_preference(self, texts: List[str]) -> torch.Tensor:
        """
        编码文本偏好

        Args:
            texts: 偏好文本列表

        Returns:
            text_features: [batch_size, unified_dim]
        """
        if self.text_encoder:
            with torch.no_grad():
                text_embeds = self.text_encoder.encode(
                    texts, convert_to_tensor=True, show_progress_bar=False
                )
        else:
            # 模拟
            batch_size = len(texts)
            text_embeds = torch.randn(batch_size, 384)

        text_features = self.text_projection(text_embeds)
        text_features = F.normalize(text_features, p=2, dim=-1)

        return text_features

    def encode_visual_preference(self, images: torch.Tensor) -> torch.Tensor:
        """
        编码视觉偏好

        Args:
            images: [batch_size, 3, H, W]

        Returns:
            visual_features: [batch_size, unified_dim]
        """
        visual_features = self.vision_encoder(images)
        visual_features = F.normalize(visual_features, p=2, dim=-1)

        return visual_features

    def fuse_multimodal_preferences(
        self,
        text_features: torch.Tensor,
        visual_features: torch.Tensor
    ) -> torch.Tensor:
        """
        融合多模态偏好

        Args:
            text_features: [batch_size, unified_dim]
            visual_features: [batch_size, unified_dim]

        Returns:
            fused_preference: [batch_size, unified_dim]
        """
        if self.use_cross_attention:
            # 增加序列维度用于注意力
            text_feat = text_features.unsqueeze(1)  # [batch, 1, dim]
            visual_feat = visual_features.unsqueeze(1)  # [batch, 1, dim]

            # 跨模态注意力
            attended_text, attended_visual = self.cross_attention(
                text_feat, visual_feat
            )

            # 压缩回来
            text_feat = attended_text.squeeze(1)
            visual_feat = attended_visual.squeeze(1)
        else:
            text_feat = text_features
            visual_feat = visual_features

        # 融合
        fused = self.fusion(text_feat, visual_feat)
        fused = F.normalize(fused, p=2, dim=-1)

        return fused

    def encode_items(self, item_ids: torch.Tensor) -> torch.Tensor:
        """
        编码物品（多模态）

        Args:
            item_ids: [batch_size, num_items]

        Returns:
            item_features: [batch_size, num_items, unified_dim]
        """
        text_embeds = self.item_text_embeddings(item_ids)
        visual_embeds = self.item_visual_embeddings(item_ids)

        # 融合文本和视觉
        batch_size, num_items, dim = text_embeds.shape

        text_flat = text_embeds.view(-1, dim)
        visual_flat = visual_embeds.view(-1, dim)

        fused_flat = self.fusion(text_flat, visual_flat)
        item_features = fused_flat.view(batch_size, num_items, dim)

        # 归一化
        item_features = F.normalize(item_features, p=2, dim=-1)

        return item_features

    def forward(
        self,
        preference_texts: List[str],
        preference_images: Optional[torch.Tensor],
        candidate_ids: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None,
        text_weight: float = 0.5,
        image_weight: float = 0.5
    ) -> Tuple[torch.Tensor, Dict]:
        """
        前向传播

        Args:
            preference_texts: 用户偏好文本列表
            preference_images: [batch_size, 3, H, W] - 用户偏好图像（可选）
            candidate_ids: [batch_size, num_candidates]
            candidate_mask: [batch_size, num_candidates]
            text_weight: 文本权重
            image_weight: 图像权重

        Returns:
            similarity_scores: [batch_size, num_candidates]
            features_dict: 特征字典
        """
        # 编码文本偏好
        text_preference = self.encode_text_preference(preference_texts)

        # 编码视觉偏好
        if preference_images is not None:
            visual_preference = self.encode_visual_preference(preference_images)

            # 融合多模态偏好
            fused_preference = self.fuse_multimodal_preferences(
                text_preference, visual_preference
            )
        else:
            # 只使用文本
            fused_preference = text_preference

        # 编码候选物品
        item_features = self.encode_items(candidate_ids)

        # 计算相似度
        preference_expanded = fused_preference.unsqueeze(1)  # [batch, 1, dim]
        similarity_scores = torch.matmul(
            preference_expanded,
            item_features.transpose(1, 2)
        ).squeeze(1)  # [batch, num_candidates]

        # 应用掩码
        if candidate_mask is not None:
            similarity_scores = similarity_scores.masked_fill(
                ~candidate_mask, -1e9
            )

        features_dict = {
            'text_preference': text_preference,
            'visual_preference': visual_preference if preference_images is not None else None,
            'fused_preference': fused_preference,
            'item_features': item_features
        }

        return similarity_scores, features_dict


if __name__ == "__main__":
    print("测试多模态偏好检索器...")

    # 创建模型
    retriever = MultiModalPreferenceRetriever(
        num_items=1000,
        unified_dim=128,
        use_cross_attention=True
    )

    print(f"模型参数: {sum(p.numel() for p in retriever.parameters()):,}")

    # 测试数据
    batch_size = 4
    num_candidates = 10

    preference_texts = [
        "该用户喜欢动作和科幻电影",
        "该用户偏好浪漫喜剧",
        "该用户喜欢悬疑thriller电影",
        "该用户喜欢动画电影"
    ]

    # 模拟图像（随机张量）
    preference_images = torch.randn(batch_size, 3, 224, 224)

    candidate_ids = torch.randint(1, 1000, (batch_size, num_candidates))
    candidate_mask = torch.ones(batch_size, num_candidates, dtype=torch.bool)

    # 前向传播
    with torch.no_grad():
        similarity_scores, features = retriever(
            preference_texts,
            preference_images,
            candidate_ids,
            candidate_mask
        )

    print(f"\n相似度分数形状: {similarity_scores.shape}")
    print(f"文本偏好形状: {features['text_preference'].shape}")
    print(f"视觉偏好形状: {features['visual_preference'].shape}")
    print(f"融合偏好形状: {features['fused_preference'].shape}")

    print(f"\n示例相似度分数: {similarity_scores[0]}")
