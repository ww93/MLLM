#!/bin/bash

# UR4Rec Demo Script
# This script runs a quick demo of the UR4Rec framework

echo "=========================================="
echo "UR4Rec Demo Runner"
echo "=========================================="
echo ""

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Error: Python is not installed or not in PATH"
    exit 1
fi

# Create necessary directories
echo "[1/4] Creating directories..."
mkdir -p data outputs logs

# Install dependencies (optional - uncomment if needed)
# echo "[2/4] Installing dependencies..."
# pip install -r requirements.txt

echo "[2/4] Running demo script..."
python demo.py

# Check if demo was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Demo completed successfully!"
    echo "=========================================="
    echo ""
    echo "Generated files:"
    echo "  - data/demo_train.json (training data)"
    echo "  - data/demo_test.json (test data)"
    echo "  - outputs/demo_model.pt (trained model)"
    echo ""
    echo "Next steps:"
    echo "  1. Train on larger datasets:"
    echo "     python scripts/train.py --config configs/default_config.yaml"
    echo ""
    echo "  2. Evaluate with LLM reranking:"
    echo "     python scripts/evaluate.py --checkpoint outputs/best_model.pt --use_llm"
    echo ""
    echo "  3. Explore the Jupyter notebook:"
    echo "     jupyter notebook notebooks/quickstart.ipynb"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "Demo failed. Please check the error messages above."
    echo "=========================================="
    exit 1
fi
