#!/bin/bash

# 测试环境设置脚本

echo "==================================="
echo "UR4Rec V2 环境测试"
echo "==================================="

# 检测 python 命令
if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
elif command -v python &> /dev/null; then
    PYTHON_CMD=python
else
    echo "错误: 找不到 python 或 python3 命令"
    exit 1
fi

echo "Python 命令: $PYTHON_CMD"
echo ""

# 测试 Python 版本
echo "1. 检查 Python 版本..."
$PYTHON_CMD --version

# 测试必要的包
echo ""
echo "2. 检查必要的 Python 包..."

packages=("torch" "numpy" "pandas" "yaml" "tqdm" "PIL")

for pkg in "${packages[@]}"; do
    if $PYTHON_CMD -c "import $pkg" 2>/dev/null; then
        echo "  ✅ $pkg"
    else
        echo "  ❌ $pkg (未安装)"
    fi
done

# 检查可选包
echo ""
echo "3. 检查可选的 Python 包..."

optional_packages=("transformers" "sentence_transformers")

for pkg in "${optional_packages[@]}"; do
    if $PYTHON_CMD -c "import $pkg" 2>/dev/null; then
        echo "  ✅ $pkg"
    else
        echo "  ⚠️  $pkg (未安装，多模态功能需要)"
    fi
done

# 检查数据目录
echo ""
echo "4. 检查数据目录..."

if [ -d "data/Multimodal_Datasets" ]; then
    echo "  ✅ data/Multimodal_Datasets"
    
    if [ -d "data/Multimodal_Datasets/M_ML-100K" ]; then
        echo "    ✅ M_ML-100K"
        echo "      - $(ls data/Multimodal_Datasets/M_ML-100K/image/*.png 2>/dev/null | wc -l | tr -d ' ') 张图片"
    fi
    
    if [ -d "data/Multimodal_Datasets/M_ML-1M" ]; then
        echo "    ✅ M_ML-1M"
        echo "      - $(ls data/Multimodal_Datasets/M_ML-1M/image/*.png 2>/dev/null | wc -l | tr -d ' ') 张图片"
    fi
else
    echo "  ⚠️  data/Multimodal_Datasets (不存在，可使用原始数据)"
fi

# 检查脚本文件
echo ""
echo "5. 检查核心脚本..."

scripts=(
    "scripts/preprocess_multimodal_dataset.py"
    "scripts/preprocess_movielens.py"
    "scripts/download_images.py"
    "scripts/preprocess_images.py"
    "scripts/generate_llm_data.py"
    "scripts/train_v2.py"
)

for script in "${scripts[@]}"; do
    if [ -f "$script" ]; then
        echo "  ✅ $script"
    else
        echo "  ❌ $script"
    fi
done

# 检查模型文件
echo ""
echo "6. 检查核心模型..."

models=(
    "models/llm_generator.py"
    "models/text_preference_retriever.py"
    "models/sasrec.py"
    "models/ur4rec_v2.py"
    "models/multimodal_retriever.py"
    "models/multimodal_loss.py"
    "models/joint_trainer.py"
)

for model in "${models[@]}"; do
    if [ -f "$model" ]; then
        echo "  ✅ $model"
    else
        echo "  ❌ $model"
    fi
done

# 检查文档
echo ""
echo "7. 检查文档..."

docs=(
    "README.md"
    "WORKFLOW.md"
    "TRAINING_GUIDE.md"
    "MULTIMODAL_DATA_GUIDE.md"
    "RETRIEVER_ANALYSIS.md"
    "DOCS_INDEX.md"
    "PROJECT_COMPLETE.md"
)

for doc in "${docs[@]}"; do
    if [ -f "$doc" ]; then
        echo "  ✅ $doc"
    else
        echo "  ❌ $doc"
    fi
done

echo ""
echo "==================================="
echo "测试完成"
echo "==================================="

# 给出建议
echo ""
echo "📋 下一步建议:"
echo ""

if [ -d "data/Multimodal_Datasets" ]; then
    echo "1. 预处理多模态数据:"
    echo "   $PYTHON_CMD scripts/preprocess_multimodal_dataset.py \\"
    echo "       --dataset ml-100k \\"
    echo "       --data_dir data/Multimodal_Datasets \\"
    echo "       --output_dir data/ml-100k-mm \\"
    echo "       --copy_images"
else
    echo "1. 下载并预处理原始数据:"
    echo "   $PYTHON_CMD scripts/preprocess_movielens.py \\"
    echo "       --dataset ml-100k \\"
    echo "       --output_dir data/ml-100k"
fi

echo ""
echo "2. 查看完整文档: cat WORKFLOW.md"
echo ""
echo "3. 获取帮助: $PYTHON_CMD scripts/train_v2.py --help"

