"""
Training Strategies for UR4Rec

包含各种协同训练策略：
1. 自适应交替训练
2. 课程学习权重调度
3. Memory Bank 对比学习
4. 双向知识蒸馏
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from collections import deque


class AdaptiveAlternatingTrainer:
    """自适应交替训练策略

    根据两个模块的损失变化动态调整训练频率：
    - 当前模块损失趋于稳定 → 切换到另一个模块
    - 避免某个模块训练过度或不足

    Example:
        >>> trainer = AdaptiveAlternatingTrainer(
        ...     switch_threshold=0.01,
        ...     min_steps_per_module=5
        ... )
        >>>
        >>> for step in range(1000):
        ...     # 获取当前应该训练的模块
        ...     train_module = trainer.update(
        ...         sasrec_loss=sasrec_loss.item(),
        ...         retriever_loss=retriever_loss.item()
        ...     )
        ...
        ...     if train_module == "sasrec":
        ...         # 只训练 SASRec
        ...         train_sasrec = True
        ...         train_retriever = False
        ...     else:
        ...         # 只训练 Retriever
        ...         train_sasrec = False
        ...         train_retriever = True
    """

    def __init__(
        self,
        switch_threshold: float = 0.01,
        min_steps_per_module: int = 5,
        history_window: int = 10,
        initial_module: str = "sasrec"
    ):
        """
        Args:
            switch_threshold: 损失变化率阈值，低于此值则认为收敛
            min_steps_per_module: 每个模块最少连续训练的步数
            history_window: 用于计算损失变化的历史窗口大小
            initial_module: 初始训练的模块 ('sasrec' | 'retriever')
        """
        self.switch_threshold = switch_threshold
        self.min_steps_per_module = min_steps_per_module
        self.history_window = history_window

        # 损失历史（使用 deque 自动维护窗口大小）
        self.sasrec_loss_history = deque(maxlen=history_window)
        self.retriever_loss_history = deque(maxlen=history_window)

        # 当前训练状态
        self.current_module = initial_module
        self.steps_since_switch = 0
        self.total_steps = 0
        self.switch_count = 0

        # 统计信息
        self.sasrec_training_steps = 0
        self.retriever_training_steps = 0

    def _compute_loss_change_rate(self, loss_history: deque) -> float:
        """计算损失变化率

        Args:
            loss_history: 损失历史队列

        Returns:
            change_rate: 相对变化率 |loss[t] - loss[t-k]| / loss[t-k]
        """
        if len(loss_history) < self.history_window:
            return float('inf')  # 历史不足，不切换

        # 比较最新值和窗口中点的值
        recent_loss = loss_history[-1]
        mid_loss = loss_history[self.history_window // 2]

        # 避免除零
        if abs(mid_loss) < 1e-8:
            return 0.0

        change_rate = abs(recent_loss - mid_loss) / (abs(mid_loss) + 1e-8)

        return change_rate

    def should_switch(self) -> Tuple[bool, str]:
        """判断是否应该切换训练模块

        Returns:
            (should_switch, reason): 是否切换和原因
        """
        # 1. 检查最少步数要求
        if self.steps_since_switch < self.min_steps_per_module:
            return False, f"未达到最少步数 ({self.steps_since_switch}/{self.min_steps_per_module})"

        # 2. 检查历史数据是否充足
        if self.current_module == "sasrec":
            current_history = self.sasrec_loss_history
        else:
            current_history = self.retriever_loss_history

        if len(current_history) < self.history_window:
            return False, f"历史数据不足 ({len(current_history)}/{self.history_window})"

        # 3. 计算当前模块的损失变化率
        change_rate = self._compute_loss_change_rate(current_history)

        # 4. 判断是否趋于稳定
        if change_rate < self.switch_threshold:
            return True, f"损失趋于稳定 (变化率: {change_rate:.4f} < {self.switch_threshold})"

        return False, f"损失仍在变化 (变化率: {change_rate:.4f})"

    def update(
        self,
        sasrec_loss: float,
        retriever_loss: float,
        force_module: Optional[str] = None
    ) -> str:
        """更新并决定下一步训练哪个模块

        Args:
            sasrec_loss: SASRec 当前步的损失
            retriever_loss: Retriever 当前步的损失
            force_module: 强制指定训练的模块（用于调试）

        Returns:
            train_module: 下一步应该训练的模块 ('sasrec' | 'retriever')
        """
        # 记录损失
        self.sasrec_loss_history.append(sasrec_loss)
        self.retriever_loss_history.append(retriever_loss)

        # 更新步数
        self.total_steps += 1
        self.steps_since_switch += 1

        # 强制指定模块（调试用）
        if force_module is not None:
            self.current_module = force_module
            return force_module

        # 检查是否应该切换
        should_switch, reason = self.should_switch()

        if should_switch:
            # 切换模块
            old_module = self.current_module
            self.current_module = "retriever" if self.current_module == "sasrec" else "sasrec"
            self.steps_since_switch = 0
            self.switch_count += 1

            print(f"\n[AdaptiveAlternating] Step {self.total_steps}: 切换训练模块")
            print(f"  从 {old_module} → {self.current_module}")
            print(f"  原因: {reason}")
            print(f"  总切换次数: {self.switch_count}")

        # 更新统计
        if self.current_module == "sasrec":
            self.sasrec_training_steps += 1
        else:
            self.retriever_training_steps += 1

        return self.current_module

    def get_stats(self) -> Dict:
        """获取训练统计信息

        Returns:
            stats: 统计字典
        """
        return {
            'total_steps': self.total_steps,
            'current_module': self.current_module,
            'steps_since_switch': self.steps_since_switch,
            'switch_count': self.switch_count,
            'sasrec_training_steps': self.sasrec_training_steps,
            'retriever_training_steps': self.retriever_training_steps,
            'sasrec_training_ratio': self.sasrec_training_steps / max(self.total_steps, 1),
            'retriever_training_ratio': self.retriever_training_steps / max(self.total_steps, 1),
            'recent_sasrec_loss': self.sasrec_loss_history[-1] if self.sasrec_loss_history else None,
            'recent_retriever_loss': self.retriever_loss_history[-1] if self.retriever_loss_history else None
        }

    def reset(self):
        """重置训练状态（用于新的训练阶段）"""
        self.sasrec_loss_history.clear()
        self.retriever_loss_history.clear()
        self.steps_since_switch = 0
        self.total_steps = 0
        self.switch_count = 0
        self.sasrec_training_steps = 0
        self.retriever_training_steps = 0
        print(f"[AdaptiveAlternating] 重置训练状态")


class CurriculumWeightScheduler:
    """课程学习权重调度器

    根据训练进度动态调整各损失组件的权重：
    - 训练初期：专注于简单任务（主检索损失）
    - 训练中期：逐渐引入辅助损失（一致性、对比学习）
    - 训练后期：引入正则项（多样性）

    Example:
        >>> scheduler = CurriculumWeightScheduler(
        ...     total_steps=10000,
        ...     warmup_steps=1000
        ... )
        >>>
        >>> for step in range(10000):
        ...     weights = scheduler.get_weights(step)
        ...     # 使用动态权重计算损失
        ...     total_loss = (
        ...         weights['retrieval'] * retrieval_loss +
        ...         weights['consistency'] * consistency_loss +
        ...         weights['contrastive'] * contrastive_loss +
        ...         weights['diversity'] * diversity_loss
        ...     )
    """

    def __init__(
        self,
        total_steps: int,
        warmup_steps: int = 1000,
        stage_ratios: Tuple[float, float, float] = (0.3, 0.3, 0.4)
    ):
        """
        Args:
            total_steps: 总训练步数
            warmup_steps: 预热步数（只用主损失）
            stage_ratios: (early, mid, late) 三个阶段的比例
        """
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.stage_ratios = stage_ratios

        # 计算各阶段的步数边界
        effective_steps = total_steps - warmup_steps
        self.early_end = warmup_steps + int(effective_steps * stage_ratios[0])
        self.mid_end = self.early_end + int(effective_steps * stage_ratios[1])

        # 最大权重配置
        self.max_weights = {
            'retrieval': 1.0,
            'consistency': 0.1,
            'contrastive': 0.2,
            'diversity': 0.05
        }

    def get_weights(self, current_step: int) -> Dict[str, float]:
        """根据训练进度返回损失权重

        Args:
            current_step: 当前训练步数

        Returns:
            weights: 各损失组件的权重字典
        """
        # 预热阶段：只用主检索损失
        if current_step < self.warmup_steps:
            return {
                'retrieval': 1.0,
                'consistency': 0.0,
                'contrastive': 0.0,
                'diversity': 0.0
            }

        # 早期阶段：逐渐引入一致性损失
        elif current_step < self.early_end:
            alpha = (current_step - self.warmup_steps) / (self.early_end - self.warmup_steps)
            return {
                'retrieval': 1.0,
                'consistency': self.max_weights['consistency'] * alpha,
                'contrastive': 0.0,
                'diversity': 0.0
            }

        # 中期阶段：引入对比学习
        elif current_step < self.mid_end:
            alpha = (current_step - self.early_end) / (self.mid_end - self.early_end)
            return {
                'retrieval': 1.0,
                'consistency': self.max_weights['consistency'],
                'contrastive': self.max_weights['contrastive'] * alpha,
                'diversity': 0.0
            }

        # 后期阶段：引入多样性正则
        else:
            alpha = (current_step - self.mid_end) / (self.total_steps - self.mid_end)
            return {
                'retrieval': 1.0,
                'consistency': self.max_weights['consistency'],
                'contrastive': self.max_weights['contrastive'],
                'diversity': self.max_weights['diversity'] * alpha
            }

    def get_current_stage(self, current_step: int) -> str:
        """获取当前训练阶段

        Returns:
            stage: 'warmup' | 'early' | 'mid' | 'late'
        """
        if current_step < self.warmup_steps:
            return 'warmup'
        elif current_step < self.early_end:
            return 'early'
        elif current_step < self.mid_end:
            return 'mid'
        else:
            return 'late'


class MemoryBankContrastiveLoss(nn.Module):
    """基于 Memory Bank 的对比学习损失

    维护一个大的特征队列，提供更多负样本进行对比学习。
    相比只在 batch 内对比，负样本数量从 batch_size-1 增加到 memory_size。

    参考: MoCo v2 (https://arxiv.org/abs/2003.04297)

    Example:
        >>> memory_bank = MemoryBankContrastiveLoss(
        ...     memory_size=65536,
        ...     feature_dim=256,
        ...     temperature=0.07
        ... )
        >>>
        >>> # 前向传播
        >>> loss = memory_bank(text_features, visual_features)
        >>> loss.backward()
    """

    def __init__(
        self,
        memory_size: int = 65536,
        feature_dim: int = 256,
        temperature: float = 0.07,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Args:
            memory_size: 特征队列大小（负样本数量）
            feature_dim: 特征维度
            temperature: 温度参数
            device: 设备
        """
        super().__init__()

        self.memory_size = memory_size
        self.feature_dim = feature_dim
        self.temperature = temperature
        self.device = device

        # 特征队列（FIFO）
        self.register_buffer("text_queue", torch.randn(memory_size, feature_dim))
        self.register_buffer("visual_queue", torch.randn(memory_size, feature_dim))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # 归一化初始队列
        self.text_queue = F.normalize(self.text_queue, p=2, dim=1)
        self.visual_queue = F.normalize(self.visual_queue, p=2, dim=1)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, text_keys: torch.Tensor, visual_keys: torch.Tensor):
        """更新特征队列（FIFO）

        Args:
            text_keys: [batch_size, dim] - 新的文本特征
            visual_keys: [batch_size, dim] - 新的视觉特征
        """
        batch_size = text_keys.size(0)
        ptr = int(self.queue_ptr)

        # 确保不超出队列大小
        assert batch_size <= self.memory_size, \
            f"Batch size ({batch_size}) > memory size ({self.memory_size})"

        # 环形队列更新
        if ptr + batch_size <= self.memory_size:
            # 直接替换
            self.text_queue[ptr:ptr + batch_size] = text_keys
            self.visual_queue[ptr:ptr + batch_size] = visual_keys
            ptr = (ptr + batch_size) % self.memory_size
        else:
            # 分段替换
            remaining = self.memory_size - ptr
            self.text_queue[ptr:] = text_keys[:remaining]
            self.visual_queue[ptr:] = visual_keys[:remaining]
            self.text_queue[:batch_size - remaining] = text_keys[remaining:]
            self.visual_queue[:batch_size - remaining] = visual_keys[remaining:]
            ptr = batch_size - remaining

        self.queue_ptr[0] = ptr

    def forward(
        self,
        text_features: torch.Tensor,
        visual_features: torch.Tensor
    ) -> torch.Tensor:
        """计算对比学习损失

        Args:
            text_features: [batch_size, dim] - 文本特征
            visual_features: [batch_size, dim] - 视觉特征

        Returns:
            loss: InfoNCE 对比损失
        """
        batch_size = text_features.size(0)

        # 归一化特征
        text_features = F.normalize(text_features, p=2, dim=1)
        visual_features = F.normalize(visual_features, p=2, dim=1)

        # 1. Positive pairs: 当前batch内的匹配（对角线）
        # [batch] - 同一用户的文本和视觉特征
        pos_sim = (text_features * visual_features).sum(dim=1) / self.temperature

        # 2. Negative pairs: batch + memory bank
        # Text -> Visual
        # [batch, batch] + [batch, memory_size]
        neg_sim_t2v_batch = torch.matmul(
            text_features, visual_features.T
        ) / self.temperature

        neg_sim_t2v_memory = torch.matmul(
            text_features, self.visual_queue.T
        ) / self.temperature

        # Visual -> Text
        neg_sim_v2t_batch = torch.matmul(
            visual_features, text_features.T
        ) / self.temperature

        neg_sim_v2t_memory = torch.matmul(
            visual_features, self.text_queue.T
        ) / self.temperature

        # 3. 构建 logits 和标签
        # Text -> Visual: 第一个位置是正样本
        logits_t2v = torch.cat([
            pos_sim.unsqueeze(1),    # [batch, 1]
            neg_sim_t2v_batch,       # [batch, batch]
            neg_sim_t2v_memory       # [batch, memory_size]
        ], dim=1)  # [batch, 1 + batch + memory_size]

        labels_t2v = torch.zeros(batch_size, dtype=torch.long, device=text_features.device)
        loss_t2v = F.cross_entropy(logits_t2v, labels_t2v)

        # Visual -> Text (对称)
        logits_v2t = torch.cat([
            pos_sim.unsqueeze(1),
            neg_sim_v2t_batch,
            neg_sim_v2t_memory
        ], dim=1)

        labels_v2t = torch.zeros(batch_size, dtype=torch.long, device=visual_features.device)
        loss_v2t = F.cross_entropy(logits_v2t, labels_v2t)

        # 4. 更新 memory bank
        self._dequeue_and_enqueue(text_features.detach(), visual_features.detach())

        # 5. 双向对比损失
        loss = (loss_t2v + loss_v2t) / 2

        return loss


class BidirectionalKnowledgeDistillation(nn.Module):
    """双向知识蒸馏

    让 SASRec 和 Retriever 互相学习：
    - Forward KD: SASRec 学习 Retriever 的偏好理解
    - Backward KD: Retriever 学习 SASRec 的序列模式

    Example:
        >>> kd_module = BidirectionalKnowledgeDistillation(temperature=4.0)
        >>>
        >>> # 前向传播
        >>> kd_loss = kd_module(
        ...     sasrec_scores=sasrec_output,
        ...     retriever_scores=retriever_output
        ... )
        >>> total_loss += 0.1 * kd_loss
    """

    def __init__(self, temperature: float = 4.0):
        """
        Args:
            temperature: 蒸馏温度（越大越软的分布，推荐 2-8）
        """
        super().__init__()
        self.temperature = temperature
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(
        self,
        sasrec_scores: torch.Tensor,
        retriever_scores: torch.Tensor
    ) -> torch.Tensor:
        """计算双向蒸馏损失

        Args:
            sasrec_scores: [batch_size, num_candidates] - SASRec 分数
            retriever_scores: [batch_size, num_candidates] - Retriever 分数

        Returns:
            kd_loss: 知识蒸馏损失
        """
        # Soft targets (teacher)
        sasrec_soft = F.softmax(sasrec_scores / self.temperature, dim=-1)
        retriever_soft = F.softmax(retriever_scores / self.temperature, dim=-1)

        # Forward KD: SASRec 学习 Retriever
        sasrec_log_probs = F.log_softmax(sasrec_scores / self.temperature, dim=-1)
        loss_s2r = self.kl_loss(sasrec_log_probs, retriever_soft.detach())

        # Backward KD: Retriever 学习 SASRec
        retriever_log_probs = F.log_softmax(retriever_scores / self.temperature, dim=-1)
        loss_r2s = self.kl_loss(retriever_log_probs, sasrec_soft.detach())

        # 双向损失
        kd_loss = (loss_s2r + loss_r2s) / 2

        # 温度缩放（恢复梯度尺度）
        return kd_loss * (self.temperature ** 2)


if __name__ == "__main__":
    print("测试训练策略模块...")

    # 测试1: 自适应交替训练
    print("\n" + "="*60)
    print("测试1: 自适应交替训练")
    print("="*60)

    adaptive_trainer = AdaptiveAlternatingTrainer(
        switch_threshold=0.01,
        min_steps_per_module=5
    )

    # 模拟训练过程
    print("\n模拟训练 50 步...")
    for step in range(50):
        # 模拟损失（SASRec 先收敛，然后切换到 Retriever）
        if step < 20:
            sasrec_loss = 1.0 - step * 0.04  # 快速下降
            retriever_loss = 0.8
        else:
            sasrec_loss = 0.2 + (step - 20) * 0.001  # 缓慢上升
            retriever_loss = 0.8 - (step - 20) * 0.02  # 开始下降

        train_module = adaptive_trainer.update(sasrec_loss, retriever_loss)

        if step % 10 == 0 or train_module != adaptive_trainer.current_module:
            print(f"Step {step}: 训练 {train_module} "
                  f"(SASRec loss: {sasrec_loss:.3f}, Retriever loss: {retriever_loss:.3f})")

    stats = adaptive_trainer.get_stats()
    print(f"\n训练统计:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # 测试2: 课程学习权重调度
    print("\n" + "="*60)
    print("测试2: 课程学习权重调度")
    print("="*60)

    scheduler = CurriculumWeightScheduler(
        total_steps=10000,
        warmup_steps=1000
    )

    test_steps = [0, 500, 1500, 3500, 7000, 9500]
    print("\n不同训练阶段的权重:")
    for step in test_steps:
        weights = scheduler.get_weights(step)
        stage = scheduler.get_current_stage(step)
        print(f"\nStep {step} ({stage}):")
        for key, value in weights.items():
            print(f"  {key}: {value:.3f}")

    # 测试3: Memory Bank 对比学习
    print("\n" + "="*60)
    print("测试3: Memory Bank 对比学习")
    print("="*60)

    memory_bank = MemoryBankContrastiveLoss(
        memory_size=128,  # 测试用小值
        feature_dim=64,
        temperature=0.07,
        device='cpu'
    )

    # 模拟数据
    batch_size = 8
    text_feat = torch.randn(batch_size, 64)
    visual_feat = torch.randn(batch_size, 64)

    loss = memory_bank(text_feat, visual_feat)
    print(f"\n对比损失: {loss.item():.4f}")
    print(f"队列指针: {memory_bank.queue_ptr.item()}")

    # 测试4: 双向知识蒸馏
    print("\n" + "="*60)
    print("测试4: 双向知识蒸馏")
    print("="*60)

    kd_module = BidirectionalKnowledgeDistillation(temperature=4.0)

    sasrec_scores = torch.randn(8, 20)
    retriever_scores = torch.randn(8, 20)

    kd_loss = kd_module(sasrec_scores, retriever_scores)
    print(f"\n蒸馏损失: {kd_loss.item():.4f}")

    print("\n" + "="*60)
    print("✓ 所有测试完成！")
    print("="*60)
