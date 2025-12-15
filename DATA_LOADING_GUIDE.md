# M_ML-100K 数据加载指南

## 📋 概述

本指南介绍如何加载和使用 M_ML-100K 多模态 MovieLens 数据集，以及如何适配代码以正确处理数据格式。

---

## 📁 数据集结构

### M_ML-100K 目录内容

```
UR4Rec/data/Multimodal_Datasets/M_ML-100K/
├── movies.dat          # 电影元数据
├── ratings.dat         # 用户评分
├── text.xls            # 电影文本描述（Excel格式）
├── user.dat            # 用户信息
└── image/              # 电影图片
    ├── 1.png
    ├── 2.png
    └── ...
```

### 数据格式说明

#### 1. movies.dat
```
格式: movie_id::title::genres
分隔符: ::
编码: latin-1

示例:
1::Toy Story (1995)::Animation|Children's|Comedy
2::GoldenEye (1995)::Action|Adventure|Thriller
```

#### 2. ratings.dat
```
格式: user_id::movie_id::rating::timestamp
分隔符: ::

示例:
196::242::3::881250949
186::302::3::891717742
```

#### 3. text.xls
```
格式: Excel 文件
列名: ['movie-id', 'review']

需要 pandas 和 xlrd 读取:
pip install pandas xlrd
```

#### 4. user.dat
```
格式: user_id::gender::age::occupation::zip_code

示例:
1::M::24::17::85711
2::F::53::0::94043
```

#### 5. image/
```
格式: {movie_id}.png
示例: 1.png, 2.png, ...
分辨率: 不固定，需要resize
```

---

## 🚀 快速开始

### 方法 1: 使用便捷函数（推荐）

```python
from UR4Rec.data.dataset_loader import load_ml_100k

# 一次性加载所有数据
item_metadata, user_sequences, users = load_ml_100k(
    data_dir="UR4Rec/data/Multimodal_Datasets",
    min_rating=4.0,      # 只保留高评分
    min_seq_len=5        # 最小序列长度
)

print(f"物品数: {len(item_metadata)}")
print(f"用户数: {len(user_sequences)}")

# 查看物品元数据
item = item_metadata[1]
print(f"标题: {item['title']}")
print(f"类型: {item['genres']}")
print(f"描述: {item['description']}")
```

### 方法 2: 分步加载（更灵活）

```python
from UR4Rec.data.dataset_loader import MovieLensDataLoader

# 创建加载器
loader = MovieLensDataLoader(
    data_dir="UR4Rec/data/Multimodal_Datasets",
    dataset_name="ml-100k"
)

# 分步加载
movies = loader.load_movies()
text_descriptions = loader.load_text_descriptions()
ratings = loader.load_ratings()
users = loader.load_users()

# 构建适配格式
item_metadata = loader.build_item_metadata(movies, text_descriptions)
user_sequences = loader.build_user_sequences(ratings, min_rating=4.0)
```

---

## 📊 数据格式适配

### 输出格式说明

加载器输出的数据格式已适配 `llm_generator` 和 `retriever` 的输入要求：

#### item_metadata 格式

```python
{
    item_id (int): {
        'title': str,                    # 电影标题
        'genres': List[str],             # 类型列表
        'genres_str': str,               # 类型字符串（用|分隔）
        'description': str,              # 文本描述
        'original_id': int               # 原始ID
    }
}
```

**示例**:
```python
item_metadata[1] = {
    'title': 'Toy Story (1995)',
    'genres': ['Animation', "Children's", 'Comedy'],
    'genres_str': "Animation|Children's|Comedy",
    'description': 'A cowboy doll is profoundly threatened...',
    'original_id': 1
}
```

#### user_sequences 格式

```python
{
    user_id (int): [item_id1, item_id2, item_id3, ...]  # 按时间排序
}
```

**示例**:
```python
user_sequences[298] = [286, 172, 588, 174, 69, 603, ...]
```

---

## 🔄 创建 PyTorch DataLoader

### 方法 1: 使用便捷函数

```python
from UR4Rec.data.multimodal_dataset import create_dataloaders

# 创建训练/验证/测试 DataLoader
train_loader, val_loader, test_loader = create_dataloaders(
    user_sequences=user_sequences,
    item_metadata=item_metadata,
    image_dir="UR4Rec/data/Multimodal_Datasets/M_ML-100K/image",
    batch_size=128,
    num_workers=4,
    load_images=False,    # 是否加载图片（训练时设为True）
    max_seq_len=50,
    num_negatives=5
)

# 训练循环
for epoch in range(10):
    for batch in train_loader:
        user_ids = batch['user_ids']
        input_seq = batch['input_seq']           # [batch, max_seq_len]
        target_items = batch['target_items']      # [batch]
        negative_items = batch['negative_items']  # [batch, num_neg]

        # 如果 load_images=True
        # target_images = batch['target_images']    # [batch, 3, H, W]
        # negative_images = batch['negative_images'] # [batch, num_neg, 3, H, W]

        # 训练代码...
```

### 方法 2: 手动创建

```python
from UR4Rec.data.multimodal_dataset import SequenceRecommendationDataset, MultimodalCollator
from torch.utils.data import DataLoader

# 创建数据集
dataset = SequenceRecommendationDataset(
    user_sequences=user_sequences,
    item_metadata=item_metadata,
    image_dir="UR4Rec/data/Multimodal_Datasets/M_ML-100K/image",
    max_seq_len=50,
    num_negatives=5,
    mode="train"
)

# 创建 collator
collator = MultimodalCollator(dataset, load_images=True)

# 创建 DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=128,
    shuffle=True,
    num_workers=4,
    collate_fn=collator,
    pin_memory=True
)
```

---

## 🤖 使用 LLM 生成

### 生成用户偏好

```python
from UR4Rec.models.llm_generator import LLMPreferenceGenerator
import os

# 创建生成器
generator = LLMPreferenceGenerator(
    llm_backend="openai",
    model_name="qwen-flash",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 生成单个用户偏好
user_id = 1
user_history = user_sequences[user_id]

preference = generator.generate_user_preference(
    user_id=user_id,
    user_history=user_history,
    item_metadata=item_metadata
)

print(f"用户偏好: {preference}")
```

### 批量生成

```python
# 准备用户数据
users_data = [
    {"user_id": uid, "user_history": seq}
    for uid, seq in user_sequences.items()
]

# 批量生成用户偏好
generator.batch_generate_user_preferences(
    users_data=users_data,
    item_metadata=item_metadata,
    save_path="data/llm_generated/user_preferences.json"
)

# 批量生成物品描述
generator.batch_generate_item_descriptions(
    item_metadata=item_metadata,
    save_path="data/llm_generated/item_descriptions.json"
)
```

---

## 📈 完整训练流程示例

```python
from UR4Rec.data.dataset_loader import load_ml_100k
from UR4Rec.data.multimodal_dataset import create_dataloaders
from UR4Rec.models.ur4rec_v2 import UR4RecV2
from UR4Rec.models.joint_trainer import JointTrainer

# 1. 加载数据
print("加载数据...")
item_metadata, user_sequences, users = load_ml_100k(
    min_rating=4.0,
    min_seq_len=5
)

# 2. 创建 DataLoaders
print("创建 DataLoaders...")
train_loader, val_loader, test_loader = create_dataloaders(
    user_sequences=user_sequences,
    item_metadata=item_metadata,
    image_dir="UR4Rec/data/Multimodal_Datasets/M_ML-100K/image",
    batch_size=128,
    load_images=True,  # 多模态训练
    max_seq_len=50,
    num_negatives=5
)

# 3. 创建模型
print("创建模型...")
model = UR4RecV2(
    num_items=len(item_metadata) + 1,  # +1 for padding
    sasrec_hidden_dim=256,
    text_embedding_dim=384,
    retriever_output_dim=256,
    device='cuda'
)

# 4. 创建训练器（启用自适应交替训练）
print("创建训练器...")
trainer = JointTrainer(
    model=model,
    device='cuda',
    sasrec_lr=1e-3,
    retriever_lr=1e-4,
    use_adaptive_alternating=True,
    adaptive_switch_threshold=0.01,
    adaptive_min_steps=5
)

# 5. 四阶段训练
print("\n=== 阶段1: 预训练 SASRec ===")
trainer.set_training_stage("pretrain_sasrec")
for epoch in range(1, 6):
    metrics = trainer.train_epoch(train_loader, epoch)
    print(f"Epoch {epoch} - Loss: {metrics['total_loss']:.4f}")

print("\n=== 阶段2: 预训练 Retriever ===")
trainer.set_training_stage("pretrain_retriever")
for epoch in range(6, 11):
    metrics = trainer.train_epoch(train_loader, epoch)
    print(f"Epoch {epoch} - Loss: {metrics['total_loss']:.4f}")

print("\n=== 阶段3: 联合微调（自适应交替） ===")
trainer.set_training_stage("joint_finetune")
for epoch in range(11, 21):
    metrics = trainer.train_epoch(train_loader, epoch)
    stats = trainer.adaptive_alternating.get_stats()
    print(f"Epoch {epoch}:")
    print(f"  Loss: {metrics['total_loss']:.4f}")
    print(f"  切换次数: {stats['switch_count']}")
    print(f"  当前模块: {stats['current_module']}")

    # 验证
    if epoch % 5 == 0:
        val_metrics = trainer.evaluate(val_loader)
        print(f"  Hit@10: {val_metrics['hit@10']:.4f}")

print("\n=== 阶段4: 端到端训练 ===")
trainer.set_training_stage("end_to_end")
for epoch in range(21, 26):
    metrics = trainer.train_epoch(train_loader, epoch)
    val_metrics = trainer.evaluate(val_loader)
    print(f"Epoch {epoch} - Loss: {metrics['total_loss']:.4f}, "
          f"Hit@10: {val_metrics['hit@10']:.4f}")

# 6. 最终测试
print("\n=== 最终测试 ===")
test_metrics = trainer.evaluate(test_loader)
print(f"Hit@10: {test_metrics['hit@10']:.4f}")
print(f"NDCG@10: {test_metrics['ndcg@10']:.4f}")
print(f"MRR: {test_metrics['mrr']:.4f}")

# 7. 保存模型
trainer.save_checkpoint("checkpoints/final_model.pt", epoch=25, metrics=test_metrics)
```

---

## 🔧 常见问题

### Q1: 如何处理缺失的文本描述？

**解决方法**：如果 text.xls 不存在或无法读取，加载器会自动使用标题和类型生成描述：

```python
# 如果没有描述，自动生成
description = f"{title}. Genres: {', '.join(genres)}"
```

### Q2: 图片加载失败怎么办？

**解决方法**：MultimodalCollator 会自动处理缺失的图片：

```python
if image is None:
    # 使用零张量代替
    image = torch.zeros(3, H, W)
```

### Q3: 如何只使用文本不使用图片？

**解决方法**：设置 `load_images=False`：

```python
train_loader, val_loader, test_loader = create_dataloaders(
    ...,
    load_images=False  # 不加载图片
)
```

### Q4: pandas 或 xlrd 未安装

**解决方法**：
```bash
pip install pandas xlrd
```

如果仍然无法读取 text.xls，加载器会跳过并使用标题+类型作为描述。

### Q5: 序列太长或太短

**解决方法**：调整参数：

```python
# 调整最小序列长度
user_sequences = loader.build_user_sequences(
    ratings,
    min_rating=4.0,
    min_seq_len=3  # 降低最小长度
)

# 调整最大序列长度
dataset = SequenceRecommendationDataset(
    ...,
    max_seq_len=100  # 增加最大长度
)
```

---

## 📝 数据统计

### M_ML-100K 数据集

运行示例后的统计数据：

```
物品数: 1659
用户数: 943
评分数: 99309
用户序列数: 938 (min_rating=4.0, min_seq_len=5)
高评分记录: 55024
图片可用性: 100% (1659/1659)
```

---

## 🎯 总结

✅ **数据加载适配器完成**
- [UR4Rec/data/dataset_loader.py](UR4Rec/data/dataset_loader.py)
- 支持 M_ML-100K 和 M_ML-1M 格式
- 自动处理 Excel、图片等多模态数据

✅ **PyTorch Dataset 完成**
- [UR4Rec/data/multimodal_dataset.py](UR4Rec/data/multimodal_dataset.py)
- 支持序列推荐、负样本采样
- 支持文本和图像特征

✅ **完全兼容现有代码**
- llm_generator 无需修改，直接使用
- retriever 无需修改，直接使用
- joint_trainer 无需修改，直接使用

✅ **完整测试通过**
- 数据加载测试：✓
- Dataset 测试：✓
- DataLoader 测试：✓
- 端到端示例：✓

---

## 📚 相关文件

- [UR4Rec/data/dataset_loader.py](UR4Rec/data/dataset_loader.py) - 数据加载适配器
- [UR4Rec/data/multimodal_dataset.py](UR4Rec/data/multimodal_dataset.py) - PyTorch Dataset
- [example_data_loading.py](example_data_loading.py) - 完整使用示例
- [UR4Rec/scripts/preprocess_multimodal_dataset.py](UR4Rec/scripts/preprocess_multimodal_dataset.py) - 数据预处理脚本
- [QWEN_FLASH_USAGE.md](QWEN_FLASH_USAGE.md) - LLM 生成指南
- [ADAPTIVE_TRAINING_GUIDE.md](ADAPTIVE_TRAINING_GUIDE.md) - 自适应训练指南

---

*创建时间: 2025-12-09*
