"""
Federated Aggregation Methods

实现联邦学习中的参数聚合算法，包括：
1. FedAvg: 经典的联邦平均算法
2. FedProx: 带近端项的联邦优化
3. 其他聚合策略的扩展接口

参考论文:
- McMahan et al. (2017). Communication-Efficient Learning of Deep Networks
  from Decentralized Data. AISTATS.
"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict
import copy


class FederatedAggregator:
    """
    联邦学习参数聚合器

    支持多种聚合策略：
    - FedAvg: 按客户端数据量加权平均
    - FedProx: 带近端正则化的聚合
    - Weighted: 自定义权重聚合
    """

    def __init__(
        self,
        aggregation_method: str = "fedavg",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Args:
            aggregation_method: 聚合方法 ('fedavg', 'fedprox', 'weighted')
            device: 计算设备
        """
        self.aggregation_method = aggregation_method
        self.device = device

    def fedavg(
        self,
        client_models: List[OrderedDict],
        client_weights: Optional[List[float]] = None
    ) -> OrderedDict:
        """
        FedAvg 聚合算法

        公式: w_global = Σ(n_k / n_total) * w_k

        Args:
            client_models: 客户端模型参数列表
            client_weights: 客户端权重（通常为数据量），None则均匀权重

        Returns:
            aggregated_model: 聚合后的全局模型参数
        """
        if not client_models:
            raise ValueError("No client models to aggregate")

        num_clients = len(client_models)

        # 如果没有指定权重，使用均匀权重
        if client_weights is None:
            client_weights = [1.0 / num_clients] * num_clients
        else:
            # 归一化权重
            total_weight = sum(client_weights)
            client_weights = [w / total_weight for w in client_weights]

        # 初始化聚合模型
        aggregated_model = OrderedDict()

        # 获取第一个客户端的参数作为模板
        first_model = client_models[0]

        # 对每个参数进行加权平均
        for key in first_model.keys():
            # 加权求和
            aggregated_param = torch.zeros_like(first_model[key], device=self.device)

            for client_model, weight in zip(client_models, client_weights):
                if key in client_model:
                    aggregated_param += weight * client_model[key].to(self.device)

            aggregated_model[key] = aggregated_param

        return aggregated_model

    def fedprox(
        self,
        client_models: List[OrderedDict],
        global_model: OrderedDict,
        client_weights: Optional[List[float]] = None,
        mu: float = 0.01
    ) -> OrderedDict:
        """
        FedProx 聚合算法（带近端项）

        在FedAvg基础上添加近端正则化，使客户端模型不要偏离全局模型太远

        Args:
            client_models: 客户端模型参数列表
            global_model: 当前全局模型
            client_weights: 客户端权重
            mu: 近端项系数

        Returns:
            aggregated_model: 聚合后的模型参数
        """
        # 先使用FedAvg聚合
        aggregated_model = self.fedavg(client_models, client_weights)

        # 添加近端项：向全局模型方向移动一点
        for key in aggregated_model.keys():
            if key in global_model:
                aggregated_model[key] = (
                    (1 - mu) * aggregated_model[key] +
                    mu * global_model[key].to(self.device)
                )

        return aggregated_model

    def weighted_aggregation(
        self,
        client_models: List[OrderedDict],
        custom_weights: List[float]
    ) -> OrderedDict:
        """
        自定义权重聚合

        允许用户指定任意权重（不一定基于数据量）

        Args:
            client_models: 客户端模型参数列表
            custom_weights: 自定义权重列表

        Returns:
            aggregated_model: 聚合后的模型参数
        """
        return self.fedavg(client_models, custom_weights)

    def aggregate(
        self,
        client_models: List[OrderedDict],
        client_weights: Optional[List[float]] = None,
        global_model: Optional[OrderedDict] = None,
        **kwargs
    ) -> OrderedDict:
        """
        统一聚合接口

        根据初始化时指定的方法选择聚合策略

        Args:
            client_models: 客户端模型参数列表
            client_weights: 客户端权重
            global_model: 全局模型（FedProx需要）
            **kwargs: 其他参数

        Returns:
            aggregated_model: 聚合后的模型参数
        """
        if self.aggregation_method == "fedavg":
            return self.fedavg(client_models, client_weights)

        elif self.aggregation_method == "fedprox":
            if global_model is None:
                raise ValueError("FedProx requires global_model")
            mu = kwargs.get("mu", 0.01)
            return self.fedprox(client_models, global_model, client_weights, mu)

        elif self.aggregation_method == "weighted":
            if client_weights is None:
                raise ValueError("Weighted aggregation requires client_weights")
            return self.weighted_aggregation(client_models, client_weights)

        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")

    def compute_model_diff(
        self,
        model_before: OrderedDict,
        model_after: OrderedDict
    ) -> OrderedDict:
        """
        计算模型参数差异（梯度或更新量）

        用于计算客户端的本地更新量: Δw = w_after - w_before

        Args:
            model_before: 训练前的模型
            model_after: 训练后的模型

        Returns:
            model_diff: 参数差异
        """
        model_diff = OrderedDict()

        for key in model_after.keys():
            if key in model_before:
                model_diff[key] = model_after[key] - model_before[key]
            else:
                model_diff[key] = model_after[key]

        return model_diff

    def apply_model_diff(
        self,
        base_model: OrderedDict,
        model_diff: OrderedDict,
        lr: float = 1.0
    ) -> OrderedDict:
        """
        应用模型差异更新

        w_new = w_base + lr * Δw

        Args:
            base_model: 基础模型
            model_diff: 模型差异
            lr: 学习率（更新步长）

        Returns:
            updated_model: 更新后的模型
        """
        updated_model = OrderedDict()

        for key in base_model.keys():
            if key in model_diff:
                updated_model[key] = base_model[key] + lr * model_diff[key]
            else:
                updated_model[key] = base_model[key]

        return updated_model

    @staticmethod
    def get_model_parameters(model: nn.Module) -> OrderedDict:
        """
        提取模型参数

        Args:
            model: PyTorch模型

        Returns:
            parameters: 参数字典
        """
        return OrderedDict(
            (name, param.detach().clone().cpu())
            for name, param in model.state_dict().items()
        )

    @staticmethod
    def set_model_parameters(
        model: nn.Module,
        parameters: OrderedDict
    ) -> None:
        """
        设置模型参数

        Args:
            model: PyTorch模型
            parameters: 参数字典
        """
        model.load_state_dict(parameters, strict=True)

    def compute_aggregation_quality(
        self,
        client_models: List[OrderedDict],
        aggregated_model: OrderedDict
    ) -> Dict[str, float]:
        """
        评估聚合质量

        计算指标：
        - 平均参数距离
        - 参数方差

        Args:
            client_models: 客户端模型列表
            aggregated_model: 聚合后的模型

        Returns:
            quality_metrics: 质量指标字典
        """
        total_distance = 0.0
        total_variance = 0.0
        num_params = 0

        for key in aggregated_model.keys():
            # 收集该参数在所有客户端的值
            param_values = [
                client_model[key].flatten()
                for client_model in client_models
                if key in client_model
            ]

            if not param_values:
                continue

            # 聚合后的参数值
            agg_param = aggregated_model[key].flatten()

            # 计算平均距离
            for param_val in param_values:
                distance = torch.norm(param_val - agg_param).item()
                total_distance += distance

            # 计算方差
            param_tensor = torch.stack(param_values)
            variance = torch.var(param_tensor, dim=0).mean().item()
            total_variance += variance

            num_params += 1

        return {
            "avg_distance": total_distance / (len(client_models) * num_params),
            "avg_variance": total_variance / num_params,
            "num_parameters": num_params
        }


class GradientAggregator(FederatedAggregator):
    """
    梯度聚合器

    直接聚合客户端计算的梯度，而不是模型参数
    适用于某些需要梯度压缩或稀疏化的场景
    """

    def aggregate_gradients(
        self,
        client_gradients: List[OrderedDict],
        client_weights: Optional[List[float]] = None
    ) -> OrderedDict:
        """
        聚合客户端梯度

        Args:
            client_gradients: 客户端梯度列表
            client_weights: 客户端权重

        Returns:
            aggregated_gradients: 聚合后的梯度
        """
        return self.fedavg(client_gradients, client_weights)

    def compress_gradients(
        self,
        gradients: OrderedDict,
        compression_ratio: float = 0.1
    ) -> OrderedDict:
        """
        梯度压缩（Top-K稀疏化）

        只保留最大的k个梯度值，减少通信量

        Args:
            gradients: 梯度字典
            compression_ratio: 压缩比例（保留比例）

        Returns:
            compressed_gradients: 压缩后的梯度
        """
        compressed_gradients = OrderedDict()

        for key, grad in gradients.items():
            flat_grad = grad.flatten()
            k = max(1, int(len(flat_grad) * compression_ratio))

            # 获取Top-K索引
            _, top_k_indices = torch.topk(torch.abs(flat_grad), k)

            # 创建稀疏梯度
            sparse_grad = torch.zeros_like(flat_grad)
            sparse_grad[top_k_indices] = flat_grad[top_k_indices]

            compressed_gradients[key] = sparse_grad.reshape(grad.shape)

        return compressed_gradients
