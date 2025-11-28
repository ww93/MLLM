#!/bin/bash

# UR4Rec 虚拟环境激活脚本
# 可以从 MLLM 目录的任何位置运行

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$SCRIPT_DIR/UR4Rec/venv"

if [ -d "$VENV_PATH" ]; then
    echo "激活 UR4Rec 虚拟环境..."
    source "$VENV_PATH/bin/activate"
    echo "✅ 虚拟环境已激活"
    echo ""
    echo "当前 Python: $(which python)"
    python -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null || echo "PyTorch 未安装"
    echo ""
    echo "退出虚拟环境: deactivate"
else
    echo "❌ 错误: 找不到虚拟环境 $VENV_PATH"
    exit 1
fi
