"""
FedMem Server: 支持原型聚合的联邦学习服务器

核心创新：
1. 聚合模型参数（FedAvg）
2. 聚合记忆原型（Memory Prototypes）→ 全局抽象记忆
3. 下发全局模型 + 全局抽象记忆
"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict
import copy
import numpy as np
from tqdm import tqdm

from .ur4rec_v2_moe import UR4RecV2MoE
from .fedmem_client import FedMemClient
from .federated_aggregator import FederatedAggregator


class FedMemServer:
    """
    FedMem联邦学习服务器

    职责：
    1. 维护全局模型
    2. 选择clients参与每轮训练
    3. 聚合客户端模型参数（FedAvg）
    4. 【FedMem核心】聚合客户端记忆原型 → 全局抽象记忆
    5. 下发全局模型 + 全局抽象记忆
    6. 评估全局模型性能
    """

    def __init__(
        self,
        global_model: UR4RecV2MoE,
        clients: List[FedMemClient],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        # 联邦学习参数
        aggregation_method: str = "fedavg",
        client_fraction: float = 0.1,  # 每轮参与的客户端比例
        # 训练参数
        num_rounds: int = 50,
        local_epochs: int = 1,
        patience: int = 10,  # Early stopping
        # FedMem参数
        enable_prototype_aggregation: bool = True,
        num_memory_prototypes: int = 5,
        # 【策略2】Partial Aggregation
        partial_aggregation_warmup_rounds: int = 0  # 0表示禁用，>0表示启用
    ):
        """
        Args:
            global_model: 全局模型（UR4RecV2MoE）
            clients: FedMem客户端列表
            device: 计算设备
            aggregation_method: 聚合方法 ('fedavg', 'fedprox')
            client_fraction: 每轮参与训练的客户端比例
            num_rounds: 联邦学习轮数
            local_epochs: 客户端本地训练轮数
            patience: 早停patience
            enable_prototype_aggregation: 是否启用原型聚合
            num_memory_prototypes: 记忆原型数量
            partial_aggregation_warmup_rounds: Warmup轮数，前N轮只聚合SASRec参数（策略2）
        """
        self.global_model = global_model.to(device)
        self.clients = clients
        self.device = device
        self.aggregation_method = aggregation_method
        self.client_fraction = client_fraction
        self.num_rounds = num_rounds
        self.local_epochs = local_epochs
        self.patience = patience
        self.enable_prototype_aggregation = enable_prototype_aggregation
        self.num_memory_prototypes = num_memory_prototypes
        self.partial_aggregation_warmup_rounds = partial_aggregation_warmup_rounds

        # 聚合器
        self.aggregator = FederatedAggregator(
            aggregation_method=aggregation_method,
            device=device
        )

        # 【FedMem】全局抽象记忆
        self.global_abstract_memory: Optional[torch.Tensor] = None

        # 训练历史
        self.train_history = {
            'round': [],
            'train_loss': [],
            'val_metrics': [],
            'test_metrics': [],
            'memory_stats': []  # 记忆统计
        }

        # 最佳模型
        self.best_val_metric = 0.0
        self.best_model_state = None
        self.best_global_memory = None
        self.rounds_without_improvement = 0

    def select_clients(self) -> List[FedMemClient]:
        """
        选择参与本轮训练的客户端

        Returns:
            selected_clients: 选中的客户端列表
        """
        num_clients = max(1, int(len(self.clients) * self.client_fraction))

        # 随机选择
        import random
        selected_clients = random.sample(self.clients, num_clients)

        return selected_clients

    def aggregate_prototypes(
        self,
        client_prototypes: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        【FedMem核心】聚合客户端记忆原型 → 全局抽象记忆

        聚合策略：
        1. 简单平均：对所有客户端原型取平均
        2. 加权平均：根据客户端数据量加权
        3. 聚类：对所有原型重新聚类

        Args:
            client_prototypes: 客户端原型列表 [List of [K, emb_dim]]

        Returns:
            global_prototypes: 全局原型 [K, emb_dim]
        """
        # 过滤掉None（某些客户端可能没有记忆）
        valid_prototypes = [p for p in client_prototypes if p is not None]

        if len(valid_prototypes) == 0:
            return None

        # 检查所有原型是否具有相同的数量
        prototype_shapes = [p.shape[0] for p in valid_prototypes]

        if len(set(prototype_shapes)) == 1:
            # 所有客户端返回相同数量的原型 - 直接平均
            global_prototypes = torch.stack(valid_prototypes).mean(dim=0)  # [K, emb_dim]
        else:
            # 不同客户端返回不同数量的原型 - 使用concat + re-cluster策略
            # 将所有原型连接在一起
            all_prototypes = torch.cat(valid_prototypes, dim=0)  # [sum(K_i), emb_dim]

            # 对所有原型重新聚类得到固定数量的全局原型
            target_num_prototypes = self.num_memory_prototypes

            if all_prototypes.shape[0] <= target_num_prototypes:
                # 如果总原型数不足目标数量，直接使用所有原型（零填充）
                global_prototypes = torch.zeros(
                    target_num_prototypes,
                    all_prototypes.shape[1],
                    device=all_prototypes.device
                )
                global_prototypes[:all_prototypes.shape[0]] = all_prototypes
            else:
                # 使用K-means重新聚类
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=target_num_prototypes, random_state=42, n_init=10)
                labels = kmeans.fit_predict(all_prototypes.cpu().numpy())

                # 计算每个簇的中心作为全局原型
                global_prototypes = torch.from_numpy(kmeans.cluster_centers_).float().to(all_prototypes.device)

        return global_prototypes

    def distribute_global_abstract_memory(self, selected_clients: List[FedMemClient]):
        """
        【FedMem核心】下发全局抽象记忆到选中的客户端

        Args:
            selected_clients: 选中的客户端列表
        """
        if self.global_abstract_memory is None:
            return

        for client in selected_clients:
            client.set_global_abstract_memory(self.global_abstract_memory)

    def train_round(
        self,
        round_idx: int,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        执行一轮FedMem联邦学习

        流程：
        1. 选择客户端
        2. 下发全局模型参数
        3. 【FedMem】下发全局抽象记忆
        4. 客户端本地训练（带记忆更新）
        5. 收集客户端参数 + 记忆原型
        6. 聚合模型参数（FedAvg）
        7. 【FedMem】聚合记忆原型
        8. 更新全局模型和全局抽象记忆

        Args:
            round_idx: 当前轮数
            verbose: 是否打印信息

        Returns:
            round_metrics: 本轮训练指标
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Round {round_idx + 1}/{self.num_rounds}")
            print(f"{'='*60}")

        # 1. 选择客户端
        selected_clients = self.select_clients()
        if verbose:
            print(f"选择了 {len(selected_clients)} / {len(self.clients)} 个客户端")

        # 2. 下发全局模型参数
        global_parameters = FederatedAggregator.get_model_parameters(self.global_model)

        for client in selected_clients:
            client.set_model_parameters(global_parameters)

        # 3. 【FedMem】下发全局抽象记忆
        if self.enable_prototype_aggregation:
            self.distribute_global_abstract_memory(selected_clients)

        # 4. 客户端本地训练
        client_models = []
        client_weights = []
        client_prototypes = []  # 【FedMem】收集记忆原型
        total_loss = 0.0
        total_memory_size = 0
        total_memory_updates = 0

        if verbose:
            client_iter = tqdm(selected_clients, desc="客户端训练")
        else:
            client_iter = selected_clients

        for client in client_iter:
            # 本地训练
            train_metrics = client.train_local_model(verbose=False)

            # 收集模型参数和权重
            client_models.append(client.get_model_parameters())
            client_weights.append(client.get_data_size())

            # 【FedMem】收集记忆原型
            if self.enable_prototype_aggregation:
                prototypes = client.get_memory_prototypes()
                client_prototypes.append(prototypes)

            # 累积指标
            total_loss += train_metrics['loss'] * client.get_data_size()
            total_memory_size += train_metrics.get('memory_size', 0)
            total_memory_updates += train_metrics.get('memory_updates', 0)

            # 释放客户端模型内存
            client.release_model()

        # 平均训练损失
        total_data_size = sum(client_weights)
        avg_train_loss = total_loss / total_data_size
        avg_memory_size = total_memory_size / len(selected_clients)
        avg_memory_updates = total_memory_updates / len(selected_clients)

        # 5. 聚合客户端参数（FedAvg）
        if verbose:
            print(f"聚合客户端模型参数...")

        # 【策略2】Partial Aggregation - 前N轮只聚合SASRec参数
        if self.partial_aggregation_warmup_rounds > 0 and round_idx < self.partial_aggregation_warmup_rounds:
            if verbose:
                print(f"  [Warmup阶段 {round_idx+1}/{self.partial_aggregation_warmup_rounds}] 只聚合SASRec参数")

            # 过滤客户端模型：只保留sasrec相关参数
            filtered_client_models = []
            for client_model in client_models:
                filtered_model = OrderedDict()
                for key, value in client_model.items():
                    # 只保留包含 "sasrec" 的参数，排除 router 和 expert
                    if "sasrec" in key and "router" not in key and "expert" not in key:
                        filtered_model[key] = value
                filtered_client_models.append(filtered_model)

            # 聚合过滤后的参数
            aggregated_sasrec_params = self.aggregator.aggregate(
                client_models=filtered_client_models,
                client_weights=client_weights,
                global_model=global_parameters if self.aggregation_method == "fedprox" else None
            )

            # 用聚合后的SASRec参数更新全局参数，保持其他参数不变
            aggregated_parameters = copy.deepcopy(global_parameters)
            for key, value in aggregated_sasrec_params.items():
                aggregated_parameters[key] = value

            if verbose:
                print(f"  聚合了 {len(aggregated_sasrec_params)} 个SASRec参数（共{len(global_parameters)}个参数）")
        else:
            # 正常全量聚合
            if self.partial_aggregation_warmup_rounds > 0 and verbose:
                print(f"  [正常阶段] 全量聚合所有参数")

            aggregated_parameters = self.aggregator.aggregate(
                client_models=client_models,
                client_weights=client_weights,
                global_model=global_parameters if self.aggregation_method == "fedprox" else None
            )

        # 6. 【FedMem】聚合记忆原型
        if self.enable_prototype_aggregation and len(client_prototypes) > 0:
            if verbose:
                print(f"聚合客户端记忆原型...")

            self.global_abstract_memory = self.aggregate_prototypes(client_prototypes)

            if self.global_abstract_memory is not None and verbose:
                print(f"全局抽象记忆形状: {self.global_abstract_memory.shape}")

        # 7. 更新全局模型
        FederatedAggregator.set_model_parameters(
            self.global_model,
            aggregated_parameters
        )

        # 计算聚合质量
        agg_quality = self.aggregator.compute_aggregation_quality(
            client_models, aggregated_parameters
        )

        if verbose:
            print(f"平均训练损失: {avg_train_loss:.4f}")
            print(f"平均记忆大小: {avg_memory_size:.1f}")
            print(f"平均记忆更新: {avg_memory_updates:.1f}")
            print(f"聚合质量 - 平均距离: {agg_quality['avg_distance']:.4f}, "
                  f"平均方差: {agg_quality['avg_variance']:.4f}")

        return {
            'train_loss': avg_train_loss,
            'agg_distance': agg_quality['avg_distance'],
            'agg_variance': agg_quality['avg_variance'],
            'avg_memory_size': avg_memory_size,
            'avg_memory_updates': avg_memory_updates
        }

    def evaluate_global_model(
        self,
        user_sequences: Optional[Dict[int, List[int]]] = None,
        split: str = "test",
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        在所有客户端上评估全局模型

        Args:
            user_sequences: 完整用户序列（用于负采样评估）
            split: 'val' 或 'test'
            verbose: 是否打印信息

        Returns:
            avg_metrics: 平均评估指标
        """
        # 下发全局模型参数
        global_parameters = FederatedAggregator.get_model_parameters(self.global_model)

        # 【FedMem】下发全局抽象记忆
        if self.enable_prototype_aggregation and self.global_abstract_memory is not None:
            for client in self.clients:
                client.set_global_abstract_memory(self.global_abstract_memory)

        # 在每个客户端上评估
        all_metrics = []

        for client in self.clients:
            # 设置模型参数
            client.set_model_parameters(global_parameters)

            # 评估（传递user_sequences用于负采样）
            client_metrics = client.evaluate(user_sequences=user_sequences, split=split)
            all_metrics.append(client_metrics)

            # 释放模型内存
            client.release_model()

        # 计算平均指标
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)

        if verbose:
            print(f"\n{split.capitalize()} 指标（在 {len(self.clients)} 个客户端上平均）:")
            for key, value in avg_metrics.items():
                print(f"  {key}: {value:.4f}")

        return avg_metrics

    def train(self, user_sequences: Optional[Dict[int, List[int]]] = None, verbose: bool = True) -> Dict[str, List]:
        """
        执行完整的FedMem联邦学习训练

        Args:
            user_sequences: 完整用户序列（用于负采样评估）
            verbose: 是否打印训练信息

        Returns:
            train_history: 训练历史
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"开始 FedMem 联邦学习训练")
            print(f"总轮数: {self.num_rounds}")
            print(f"客户端数量: {len(self.clients)}")
            print(f"客户端参与比例: {self.client_fraction}")
            print(f"聚合方法: {self.aggregation_method}")
            print(f"原型聚合: {'启用' if self.enable_prototype_aggregation else '禁用'}")
            print(f"{'='*60}\n")

        for round_idx in range(self.num_rounds):
            # 训练一轮
            round_metrics = self.train_round(round_idx, verbose=verbose)

            # [加速优化2] 动态验证频率
            # 前20轮：每5轮验证一次（Warmup阶段性能低，频繁验证浪费时间）
            # 后期：每2轮验证一次（精细调优阶段）
            should_validate = False
            if round_idx < 20:
                # Warmup阶段：每5轮或最后一轮验证
                should_validate = (round_idx % 5 == 0) or (round_idx == self.num_rounds - 1)
            else:
                # 后期：每2轮或最后一轮验证
                should_validate = (round_idx % 2 == 0) or (round_idx == self.num_rounds - 1)

            # 在验证集上评估（传递user_sequences）
            if should_validate:
                val_metrics = self.evaluate_global_model(user_sequences=user_sequences, split="val", verbose=verbose)
            else:
                # 不验证时，复用上一次的验证结果
                if len(self.train_history['val_metrics']) > 0:
                    val_metrics = self.train_history['val_metrics'][-1]
                    if verbose:
                        print(f"  ⏭️  跳过验证（将在Round {round_idx + (5 if round_idx < 20 else 2) - round_idx % (5 if round_idx < 20 else 2)}时验证）")
                else:
                    # 第一轮必须验证
                    val_metrics = self.evaluate_global_model(user_sequences=user_sequences, split="val", verbose=verbose)

            # 记录历史
            self.train_history['round'].append(round_idx)
            self.train_history['train_loss'].append(round_metrics['train_loss'])
            self.train_history['val_metrics'].append(val_metrics)
            self.train_history['memory_stats'].append({
                'avg_memory_size': round_metrics.get('avg_memory_size', 0),
                'avg_memory_updates': round_metrics.get('avg_memory_updates', 0)
            })

            # Early stopping 检查（只在实际验证时更新）
            if should_validate:
                current_val_metric = val_metrics['HR@10']  # 使用 HR@10 作为主要指标
            else:
                current_val_metric = None

            # 只在实际验证时才更新best model和early stopping计数
            if current_val_metric is not None:
                if current_val_metric > self.best_val_metric:
                    self.best_val_metric = current_val_metric
                    self.best_model_state = copy.deepcopy(
                        self.global_model.state_dict()
                    )
                    if self.global_abstract_memory is not None:
                        self.best_global_memory = self.global_abstract_memory.clone()
                    self.rounds_without_improvement = 0

                    if verbose:
                        print(f"✓ 新的最佳验证 HR@10: {self.best_val_metric:.4f}")
                else:
                    self.rounds_without_improvement += 1

                    if verbose:
                        print(f"  连续 {self.rounds_without_improvement} 轮无改进 "
                          f"(最佳: {self.best_val_metric:.4f})")

            # Early stopping
            if self.rounds_without_improvement >= self.patience:
                if verbose:
                    print(f"\n早停触发，在第 {round_idx + 1} 轮停止训练")
                break

        # 恢复最佳模型
        if self.best_model_state is not None:
            self.global_model.load_state_dict(self.best_model_state)
            if self.best_global_memory is not None:
                self.global_abstract_memory = self.best_global_memory
            if verbose:
                print(f"\n恢复最佳模型 (Val HR@10: {self.best_val_metric:.4f})")

        # 在测试集上最终评估
        if verbose:
            print(f"\n{'='*60}")
            print("在测试集上进行最终评估")
            print(f"{'='*60}")

        test_metrics = self.evaluate_global_model(user_sequences=user_sequences, split="test", verbose=verbose)
        self.train_history['test_metrics'] = test_metrics

        if verbose:
            print(f"\n{'='*60}")
            print("训练完成！")
            print(f"{'='*60}")

        return self.train_history

    def save_model(self, save_path: str) -> None:
        """
        保存全局模型和全局抽象记忆

        Args:
            save_path: 保存路径
        """
        checkpoint = {
            'model_state_dict': self.global_model.state_dict(),
            'train_history': self.train_history,
            'best_val_metric': self.best_val_metric,
            'global_abstract_memory': self.global_abstract_memory
        }

        torch.save(checkpoint, save_path)
        print(f"模型已保存到 {save_path}")

    def load_model(self, load_path: str) -> None:
        """
        加载全局模型和全局抽象记忆

        Args:
            load_path: 加载路径
        """
        checkpoint = torch.load(load_path, map_location=self.device)

        self.global_model.load_state_dict(checkpoint['model_state_dict'])
        self.train_history = checkpoint.get('train_history', {})
        self.best_val_metric = checkpoint.get('best_val_metric', 0.0)
        self.global_abstract_memory = checkpoint.get('global_abstract_memory', None)

        print(f"模型已从 {load_path} 加载")

    def get_best_metrics(self) -> Dict[str, float]:
        """
        获取最佳验证指标

        Returns:
            best_metrics: 最佳指标
        """
        if not self.train_history['val_metrics']:
            return {}

        best_round_idx = max(
            range(len(self.train_history['val_metrics'])),
            key=lambda i: self.train_history['val_metrics'][i]['HR@10']
        )

        best_metrics = {
            'round': self.train_history['round'][best_round_idx],
            'train_loss': self.train_history['train_loss'][best_round_idx],
            **self.train_history['val_metrics'][best_round_idx]
        }

        return best_metrics
