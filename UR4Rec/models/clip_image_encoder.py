"""
CLIP Image Encoder for extracting visual features.

Uses OpenAI's CLIP model to extract image embeddings for items.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict
from pathlib import Path
from PIL import Image
import json


class CLIPImageEncoder(nn.Module):
    """
    CLIP-based image encoder for extracting visual features.

    Features:
    - Uses pretrained CLIP model (ViT-B/32 or ViT-L/14)
    - Extracts 512-dim or 768-dim image embeddings
    - Projects to desired output dimension
    - Supports batch processing and caching
    """

    def __init__(
        self,
        model_name: str = "ViT-B/32",
        output_dim: int = 512,
        dropout: float = 0.1,
        freeze_encoder: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Args:
            model_name: CLIP model variant ("ViT-B/32", "ViT-B/16", "ViT-L/14")
            output_dim: Output embedding dimension
            dropout: Dropout rate
            freeze_encoder: Whether to freeze CLIP encoder
            device: Device to run on
        """
        super().__init__()

        self.device = device
        self.output_dim = output_dim

        # Load CLIP model
        try:
            import clip
            self.clip_model, self.preprocess = clip.load(model_name, device=device)

            if freeze_encoder:
                for param in self.clip_model.parameters():
                    param.requires_grad = False

            # Get CLIP embedding dimension
            if "ViT-L" in model_name:
                self.clip_dim = 768
            else:
                self.clip_dim = 512

            print(f"✓ 加载 CLIP 模型: {model_name} (embedding_dim={self.clip_dim})")

        except ImportError:
            print("警告: clip 未安装")
            print("请安装: pip install git+https://github.com/openai/CLIP.git")
            self.clip_model = None
            self.preprocess = None
            self.clip_dim = 512

        # Projection layer
        self.projection = nn.Sequential(
            nn.Linear(self.clip_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim)
        )

    def encode_image_from_path(self, image_path: str) -> torch.Tensor:
        """
        Encode a single image from file path.

        Args:
            image_path: Path to image file

        Returns:
            embedding: [clip_dim] image embedding
        """
        if self.clip_model is None:
            return torch.randn(self.clip_dim, device=self.device)

        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_tensor)
                image_features = F.normalize(image_features, dim=-1)

            return image_features.squeeze(0)

        except Exception as e:
            print(f"警告: 无法加载图片 {image_path}: {e}")
            return torch.randn(self.clip_dim, device=self.device)

    def encode_images_batch(self, image_paths: List[str]) -> torch.Tensor:
        """
        Encode a batch of images.

        Args:
            image_paths: List of image file paths

        Returns:
            embeddings: [batch_size, clip_dim]
        """
        if self.clip_model is None:
            return torch.randn(len(image_paths), self.clip_dim, device=self.device)

        images = []
        for path in image_paths:
            try:
                image = Image.open(path).convert('RGB')
                images.append(self.preprocess(image))
            except Exception as e:
                print(f"警告: 无法加载图片 {path}: {e}")
                # Use blank image as fallback
                images.append(torch.zeros(3, 224, 224))

        image_tensor = torch.stack(images).to(self.device)

        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_tensor)
            image_features = F.normalize(image_features, dim=-1)

        return image_features

    def forward(self, clip_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Project CLIP embeddings to output dimension.

        Args:
            clip_embeddings: [batch_size, clip_dim] CLIP features

        Returns:
            projected: [batch_size, output_dim]
        """
        return self.projection(clip_embeddings)

    def precompute_item_features(
        self,
        item_image_paths: Dict[int, str],
        num_items: int,
        batch_size: int = 32
    ) -> torch.Tensor:
        """
        Precompute and cache image features for all items.

        Args:
            item_image_paths: Dict mapping item_id to image path
            num_items: Total number of items
            batch_size: Batch size for encoding

        Returns:
            item_features: [num_items + 1, output_dim] (index 0 is padding)
        """
        print(f"预计算 {num_items} 个物品的图片特征...")

        # Set to eval mode to disable dropout for deterministic features
        self.eval()

        # Initialize feature matrix
        item_features = torch.zeros(num_items + 1, self.output_dim, device=self.device)

        # Collect all image paths
        items_to_process = []
        for item_id in range(1, num_items + 1):
            if item_id in item_image_paths and Path(item_image_paths[item_id]).exists():
                items_to_process.append((item_id, item_image_paths[item_id]))

        if not items_to_process:
            print("警告: 没有找到有效的图片文件")
            return item_features

        # Process in batches
        total_batches = (len(items_to_process) + batch_size - 1) // batch_size

        for i in range(0, len(items_to_process), batch_size):
            batch_items = items_to_process[i:i + batch_size]
            item_ids = [item[0] for item in batch_items]
            image_paths = [item[1] for item in batch_items]

            # Encode batch
            clip_features = self.encode_images_batch(image_paths)
            projected_features = self.forward(clip_features)

            # Store features
            for idx, item_id in enumerate(item_ids):
                item_features[item_id] = projected_features[idx]

            if (i // batch_size + 1) % 10 == 0:
                print(f"  进度: {i // batch_size + 1}/{total_batches} batches")

        print(f"✓ 完成: {len(items_to_process)} 个物品的图片特征已提取")
        return item_features

    def save_features(self, features: torch.Tensor, save_path: str):
        """Save precomputed features to disk."""
        torch.save(features.cpu(), save_path)
        print(f"✓ 图片特征已保存至: {save_path}")

    def load_features(self, load_path: str) -> torch.Tensor:
        """Load precomputed features from disk."""
        features = torch.load(load_path, map_location=self.device)
        print(f"✓ 从 {load_path} 加载了图片特征")
        return features


__all__ = ["CLIPImageEncoder"]
