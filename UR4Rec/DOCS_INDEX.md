# 📚 UR4Rec V2 文档索引

本文档提供项目所有文档的快速导航。

---

## 🚀 入门文档

### [README.md](README.md) - 项目主页
**阅读时间**: 5 分钟

**内容**:
- 项目简介和核心思想
- 快速开始（4 步上手）
- 支持的功能列表
- 架构图解
- 实验结果对比
- 高级用法示例

**适合人群**: 所有用户，首次了解项目必读

---

### [WORKFLOW.md](WORKFLOW.md) - 完整工作流程
**阅读时间**: 15 分钟

**内容**:
- 总体流程图
- 5 个详细步骤（数据预处理 → 图片准备 → LLM 生成 → 训练 → 评估）
- 每个步骤的命令示例
- 完整流程示例
- 常见问题解答
- 数据流详解

**适合人群**: 希望了解完整流程的用户

---

## 🎓 教程文档

### [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - 训练指南
**阅读时间**: 20 分钟

**内容**:
- 训练流程概览
- 环境准备详解
- 数据准备步骤
- LLM 数据生成教程
- 4 个训练阶段详解
- 配置文件说明
- 9 个常见问题解答
- 最佳实践建议

**适合人群**: 准备训练模型的用户

---

## 🔬 技术文档

### [RETRIEVER_ANALYSIS.md](RETRIEVER_ANALYSIS.md) - 检索器设计分析
**阅读时间**: 25 分钟

**内容**:
- 论文 vs 当前实现对比
- 实现正确性检查
- 4 种多模态融合方案对比：
  - 早期融合（Early Fusion）
  - 晚期融合（Late Fusion）
  - 跨模态注意力（推荐）⭐
  - Token-Level 注意力
- 将图片 Token 加入 Retriever 的可行性分析
- 具体实现建议和代码示例
- 性能预期

**适合人群**: 深入研究检索器设计，或希望加入图像模态的用户

---

### [REFACTORING_PROGRESS.md](REFACTORING_PROGRESS.md) - 重构进度
**阅读时间**: 15 分钟

**内容**:
- 项目背景和正确架构理解
- Phase 1-5 的详细进度：
  - ✅ Phase 1: 重构为正确架构
  - ✅ Phase 2: 加入图片模态
  - ✅ Phase 3: 优化损失函数
  - ✅ Phase 4: 联合优化
  - ⏳ Phase 5: 文档更新（进行中）
- 新增文件总览
- 关键改进点和创新点
- 预期性能指标

**适合人群**: 了解项目演进历史，或参与开发的用户

---

## 📂 文件组织

```
UR4Rec/
├── README.md                    ⭐ 从这里开始
├── WORKFLOW.md                  📋 完整流程
├── TRAINING_GUIDE.md            🎓 训练教程
├── RETRIEVER_ANALYSIS.md        🔬 深入分析
├── REFACTORING_PROGRESS.md      📝 项目进展
└── DOCS_INDEX.md               📚 本文档
```

---

## 🎯 推荐阅读路径

### 路径 1: 快速上手（总时间 ~25 分钟）

1. [README.md](README.md) - 5 分钟
   - 了解项目是什么

2. [WORKFLOW.md](WORKFLOW.md) - 15 分钟
   - 学习完整流程

3. 开始实践！
   ```bash
   # 运行快速开始命令
   python scripts/preprocess_movielens.py ...
   python scripts/generate_llm_data.py ...
   python scripts/train_v2.py ...
   ```

4. 遇到问题时查阅 [TRAINING_GUIDE.md](TRAINING_GUIDE.md)

---

### 路径 2: 深入研究（总时间 ~60 分钟）

1. [README.md](README.md) - 5 分钟
   - 项目概览

2. [REFACTORING_PROGRESS.md](REFACTORING_PROGRESS.md) - 15 分钟
   - 理解正确架构

3. [RETRIEVER_ANALYSIS.md](RETRIEVER_ANALYSIS.md) - 25 分钟
   - 深入检索器设计

4. [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - 20 分钟
   - 详细训练细节

5. 阅读代码
   ```bash
   # 核心模块
   models/text_preference_retriever.py
   models/sasrec.py
   models/ur4rec_v2.py
   models/multimodal_retriever.py
   ```

---

### 路径 3: 实现多模态（总时间 ~40 分钟）

1. [README.md](README.md) - 5 分钟
   - 快速了解

2. [RETRIEVER_ANALYSIS.md](RETRIEVER_ANALYSIS.md) - 25 分钟
   - ⭐ 重点：4 种多模态方案对比
   - ⭐ 重点：Token-Level 跨模态注意力实现

3. [WORKFLOW.md](WORKFLOW.md) 中的"Step 2: 图片数据准备" - 10 分钟
   - 学习如何下载和处理图片

4. 实践
   ```bash
   # 下载图片
   python scripts/download_images.py ...

   # 提取特征
   python scripts/preprocess_images.py ...

   # 训练多模态模型
   python scripts/train_v2.py --use_multimodal ...
   ```

---

## 📝 文档维护

### 文档状态

| 文档 | 状态 | 最后更新 |
|------|------|----------|
| README.md | ✅ 最新 | 2025-11-27 |
| WORKFLOW.md | ✅ 最新 | 2025-11-27 |
| TRAINING_GUIDE.md | ✅ 最新 | 2025-11-27 |
| RETRIEVER_ANALYSIS.md | ✅ 最新 | 2025-11-27 |
| REFACTORING_PROGRESS.md | ✅ 最新 | 2025-11-27 |
| DOCS_INDEX.md | ✅ 最新 | 2025-11-27 |

### 已删除的过时文档

以下文档基于错误的架构理解，已被删除：
- ❌ `README.md` (旧版英文)
- ❌ `QUICKSTART.md` (旧版英文)
- ❌ `PROJECT_SUMMARY.md` (旧版英文)
- ❌ `TASK_COMPLETED_CN.md` (临时文件)
- ❌ `README_CN.md` (旧版中文)
- ❌ `QUICKSTART_CN.md` (旧版中文)
- ❌ `PROJECT_SUMMARY_CN.md` (旧版中文)

---

## 🤔 需要帮助？

### 找不到想要的信息？

1. **概念理解问题** → [README.md](README.md) + [REFACTORING_PROGRESS.md](REFACTORING_PROGRESS.md)
2. **流程不清楚** → [WORKFLOW.md](WORKFLOW.md)
3. **训练遇到问题** → [TRAINING_GUIDE.md](TRAINING_GUIDE.md) 的常见问题部分
4. **想实现多模态** → [RETRIEVER_ANALYSIS.md](RETRIEVER_ANALYSIS.md)
5. **代码实现细节** → 直接阅读相应的 Python 文件

### 报告问题

如果文档中有错误或不清楚的地方，欢迎：
- 提交 Issue
- 提交 Pull Request
- 联系项目维护者

---

## 💡 提示

- 📌 **首次使用**: 一定要从 [README.md](README.md) 开始
- ⚡ **快速验证**: 使用 Mock LLM 后端快速测试整个流程
- 🎯 **深入学习**: 按"路径 2"顺序阅读所有技术文档
- 🚀 **多模态研究**: 重点阅读 [RETRIEVER_ANALYSIS.md](RETRIEVER_ANALYSIS.md)

---

**最后更新**: 2025-11-27
**维护者**: UR4Rec V2 项目团队
