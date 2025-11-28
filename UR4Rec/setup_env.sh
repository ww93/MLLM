#!/bin/bash

# UR4Rec V2 环境配置脚本

echo "======================================"
echo "UR4Rec V2 环境配置"
echo "======================================"

# 检查是否已有虚拟环境
if [ -d "venv" ]; then
    echo "✓ 虚拟环境已存在"
else
    echo "创建虚拟环境..."
    python3 -m venv venv
    echo "✓ 虚拟环境创建完成"
fi

# 激活虚拟环境
echo ""
echo "激活虚拟环境..."
source venv/bin/activate

# 升级 pip
echo ""
echo "升级 pip..."
pip install --upgrade pip

# 安装依赖
echo ""
echo "======================================"
echo "安装依赖包"
echo "======================================"

# 1. 安装 PyTorch (CPU 版本，更快)
echo ""
echo "1. 安装 PyTorch (CPU 版本)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 2. 安装数据处理包
echo ""
echo "2. 安装数据处理包..."
pip install numpy pandas scipy

# 3. 安装 NLP 相关包
echo ""
echo "3. 安装 NLP 相关包..."
pip install transformers sentence-transformers tokenizers

# 4. 安装工具包
echo ""
echo "4. 安装工具包..."
pip install pyyaml tqdm requests Pillow

# 5. 安装 Excel 支持
echo ""
echo "5. 安装 Excel 支持..."
pip install openpyxl xlrd==1.2.0

echo ""
echo "======================================"
echo "安装完成！"
echo "======================================"

# 验证安装
echo ""
echo "验证安装..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python3 -c "import sentence_transformers; print(f'Sentence-Transformers: {sentence_transformers.__version__}')"
python3 -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python3 -c "import pandas; print(f'Pandas: {pandas.__version__}')"

echo ""
echo "======================================"
echo "使用说明"
echo "======================================"
echo ""
echo "激活虚拟环境:"
echo "  source venv/bin/activate"
echo ""
echo "运行测试脚本:"
echo "  ./test_setup.sh"
echo ""
echo "预处理数据:"
echo "  python scripts/preprocess_multimodal_dataset.py \\"
echo "      --dataset ml-100k \\"
echo "      --data_dir data/Multimodal_Datasets \\"
echo "      --output_dir data/ml-100k-mm \\"
echo "      --copy_images"
echo ""
echo "退出虚拟环境:"
echo "  deactivate"
echo ""

