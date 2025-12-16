"""
Federated Learning Server for UR4Rec

实现联邦学习服务器端逻辑：
- 管理全局模型
- 协调客户端训练
- 聚合客户端参数（FedAvg）
- 下发全局模型
"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict
import copy
from tqdm import tqdm

from .ur4rec_v2_moe import UR4RecV2MoE
from .federated_client import FederatedClient
from .federated_aggregator import FederatedAggregator


class FederatedServer:
    """
    联邦学习服务器

    职责：
    - 维护全局模型
    - 选择clients参与每轮训练
    - 聚合客户端模型
    - 评估全局模型性能
    """

    def __init__(
        self,
        global_model: UR4RecV2MoE,
        clients: List[FederatedClient],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        # 联邦学习参数
        aggregation_method: str = "fedavg",
        client_fraction: float = 1.0,  # 每轮参与的客户端比例
        # 训练参数
        num_rounds: int = 100,
        local_epochs: int = 1,
        patience: int = 10  # Early stopping
    ):
        """
        Args:
            global_model: 全局模型
            clients: 客户端列表
            device: 计算设备
            aggregation_method: 聚合方法 ('fedavg', 'fedprox')
            client_fraction: 每轮参与训练的客户端比例
            num_rounds: 联邦学习轮数
            local_epochs: 客户端本地训练轮数
            patience: 早停patience
        """
        self.global_model = global_model.to(device)
        self.clients = clients
        self.device = device
        self.aggregation_method = aggregation_method
        self.client_fraction = client_fraction
        self.num_rounds = num_rounds
        self.local_epochs = local_epochs
        self.patience = patience

        # 聚合器
        self.aggregator = FederatedAggregator(
            aggregation_method=aggregation_method,
            device=device
        )

        # 训练历史
        self.train_history = {
            'round': [],
            'train_loss': [],
            'val_metrics': [],
            'test_metrics': []
        }

        # 最佳模型
        self.best_val_metric = 0.0
        self.best_model_state = None
        self.rounds_without_improvement = 0

    def select_clients(self) -> List[FederatedClient]:
        """
        选择参与本轮训练的客户端

        Returns:
            selected_clients: 选中的客户端列表
        """
        num_clients = max(1, int(len(self.clients) * self.client_fraction))

        # 随机选择（可以实现更复杂的选择策略）
        import random
        selected_clients = random.sample(self.clients, num_clients)

        return selected_clients

    def train_round(
        self,
        round_idx: int,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        执行一轮联邦学习

        流程：
        1. 选择客户端
        2. 下发全局模型
        3. 客户端本地训练
        4. 收集客户端参数
        5. 聚合更新全局模型

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
            print(f"Selected {len(selected_clients)} / {len(self.clients)} clients")

        # 2. 下发全局模型参数
        global_parameters = FederatedAggregator.get_model_parameters(self.global_model)

        for client in selected_clients:
            client.set_model_parameters(global_parameters)

        # 3. 客户端本地训练
        client_models = []
        client_weights = []
        total_loss = 0.0

        if verbose:
            client_iter = tqdm(selected_clients, desc="Client Training")
        else:
            client_iter = selected_clients

        for client in client_iter:
            # 本地训练
            train_metrics = client.train_local_model(verbose=False)

            # 收集模型参数和权重（数据量）
            client_models.append(client.get_model_parameters())
            client_weights.append(client.get_data_size())

            total_loss += train_metrics['loss'] * client.get_data_size()

            # 释放客户端模型内存（训练完成后）
            client.release_model()

        print(f"DEBUG: Client training loop completed", flush=True)
        import sys
        sys.stdout.flush()

        # 平均训练损失
        print(f"DEBUG: Calculating average training loss...", flush=True)
        print(f"DEBUG: client_weights length: {len(client_weights)}", flush=True)
        print(f"DEBUG: total_loss: {total_loss}", flush=True)
        sys.stdout.flush()
        try:
            print(f"DEBUG: About to sum client_weights...", flush=True)
            total_data_size = sum(client_weights)
            print(f"DEBUG: total_data_size: {total_data_size}", flush=True)
            avg_train_loss = total_loss / total_data_size
            print(f"DEBUG: Average training loss calculated: {avg_train_loss:.4f}", flush=True)
        except Exception as e:
            print(f"ERROR: Failed to calculate average loss: {e}")
            import traceback
            traceback.print_exc()
            raise

        # 4. 聚合客户端参数
        print(f"DEBUG: Starting parameter aggregation...")
        aggregated_parameters = self.aggregator.aggregate(
            client_models=client_models,
            client_weights=client_weights,
            global_model=global_parameters if self.aggregation_method == "fedprox" else None
        )
        print(f"DEBUG: Parameter aggregation complete")

        # 5. 更新全局模型
        FederatedAggregator.set_model_parameters(
            self.global_model,
            aggregated_parameters
        )

        # 计算聚合质量
        agg_quality = self.aggregator.compute_aggregation_quality(
            client_models, aggregated_parameters
        )

        if verbose:
            print(f"Avg Train Loss: {avg_train_loss:.4f}")
            print(f"Aggregation Quality - Avg Distance: {agg_quality['avg_distance']:.4f}, "
                  f"Avg Variance: {agg_quality['avg_variance']:.4f}")

        return {
            'train_loss': avg_train_loss,
            'agg_distance': agg_quality['avg_distance'],
            'agg_variance': agg_quality['avg_variance']
        }

    def evaluate_global_model(
        self,
        split: str = "test",
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        在所有客户端上评估全局模型

        Args:
            split: 'val' 或 'test'
            verbose: 是否打印信息

        Returns:
            avg_metrics: 平均评估指标
        """
        # 下发全局模型参数
        global_parameters = FederatedAggregator.get_model_parameters(self.global_model)

        # 在每个客户端上评估（逐个处理以节省内存）
        all_metrics = []

        for client in self.clients:
            # 设置模型参数（触发延迟实例化）
            client.set_model_parameters(global_parameters)

            # 评估
            client_metrics = client.evaluate(split=split)
            all_metrics.append(client_metrics)

            # 立即释放模型内存
            client.release_model()

        # 计算平均指标
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)

        if verbose:
            print(f"\n{split.capitalize()} Metrics (averaged over {len(self.clients)} clients):")
            for key, value in avg_metrics.items():
                print(f"  {key}: {value:.4f}")

        return avg_metrics

    def train(self, verbose: bool = True) -> Dict[str, List]:
        """
        执行完整的联邦学习训练

        Args:
            verbose: 是否打印训练信息

        Returns:
            train_history: 训练历史
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Starting Federated Learning Training")
            print(f"Total Rounds: {self.num_rounds}")
            print(f"Clients: {len(self.clients)}")
            print(f"Client Fraction: {self.client_fraction}")
            print(f"Aggregation Method: {self.aggregation_method}")
            print(f"{'='*60}\n")

        for round_idx in range(self.num_rounds):
            # 训练一轮
            round_metrics = self.train_round(round_idx, verbose=verbose)

            # 在验证集上评估
            val_metrics = self.evaluate_global_model(split="val", verbose=verbose)

            # 记录历史
            self.train_history['round'].append(round_idx)
            self.train_history['train_loss'].append(round_metrics['train_loss'])
            self.train_history['val_metrics'].append(val_metrics)

            # Early stopping 检查
            current_val_metric = val_metrics['HR@10']  # 使用 HR@10 作为主要指标

            if current_val_metric > self.best_val_metric:
                self.best_val_metric = current_val_metric
                self.best_model_state = copy.deepcopy(
                    self.global_model.state_dict()
                )
                self.rounds_without_improvement = 0

                if verbose:
                    print(f"✓ New best validation HR@10: {self.best_val_metric:.4f}")
            else:
                self.rounds_without_improvement += 1

                if verbose:
                    print(f"  No improvement for {self.rounds_without_improvement} rounds "
                          f"(best: {self.best_val_metric:.4f})")

            # Early stopping
            if self.rounds_without_improvement >= self.patience:
                if verbose:
                    print(f"\nEarly stopping triggered after {round_idx + 1} rounds")
                break

        # 恢复最佳模型
        if self.best_model_state is not None:
            self.global_model.load_state_dict(self.best_model_state)
            if verbose:
                print(f"\nRestored best model (Val HR@10: {self.best_val_metric:.4f})")

        # 在测试集上最终评估
        if verbose:
            print(f"\n{'='*60}")
            print("Final Evaluation on Test Set")
            print(f"{'='*60}")

        test_metrics = self.evaluate_global_model(split="test", verbose=verbose)
        self.train_history['test_metrics'] = test_metrics

        if verbose:
            print(f"\n{'='*60}")
            print("Training Complete!")
            print(f"{'='*60}")

        return self.train_history

    def save_model(self, save_path: str) -> None:
        """
        保存全局模型

        Args:
            save_path: 保存路径
        """
        torch.save({
            'model_state_dict': self.global_model.state_dict(),
            'train_history': self.train_history,
            'best_val_metric': self.best_val_metric
        }, save_path)

        print(f"Model saved to {save_path}")

    def load_model(self, load_path: str) -> None:
        """
        加载全局模型

        Args:
            load_path: 加载路径
        """
        checkpoint = torch.load(load_path, map_location=self.device)

        self.global_model.load_state_dict(checkpoint['model_state_dict'])
        self.train_history = checkpoint.get('train_history', {})
        self.best_val_metric = checkpoint.get('best_val_metric', 0.0)

        print(f"Model loaded from {load_path}")

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
