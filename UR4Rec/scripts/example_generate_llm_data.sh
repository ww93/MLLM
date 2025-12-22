#!/bin/bash
# LLM 数据生成示例脚本（使用 qwen-flash）

# 设置 API 密钥
export DASHSCOPE_API_KEY="your-api-key-here"

# 数据路径
DATA_DIR="UR4Rec/data/Multimodal_Datasets"
OUTPUT_DIR="data/llm_generated"

echo "============================================"
echo "LLM 数据生成示例 (qwen-flash)"
echo "============================================"

# 示例 1: 生成用户偏好和物品描述（默认使用 qwen-flash + DashScope）
echo -e "\n[示例 1] 生成两者（用户偏好 + 物品描述）"
python UR4Rec/scripts/generate_llm_data.py \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --max_users 10 \
    --max_items 20

# 示例 2: 只生成用户偏好（从环境变量读取 API key）
echo -e "\n[示例 2] 只生成用户偏好"
python UR4Rec/scripts/generate_llm_data.py \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR/only_users \
    --only_users \
    --max_users 10

# 示例 3: 只生成物品描述（使用新的语义密集型 prompt）
echo -e "\n[示例 3] 只生成物品描述（使用新 prompt）"
python UR4Rec/scripts/generate_llm_data.py \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR/only_items \
    --only_items \
    --max_items 20 \
    --regenerate_descriptions

# 示例 4: 使用已有的物品描述（来自 text.xls）
echo -e "\n[示例 4] 使用 text.xls 中已有的描述"
python UR4Rec/scripts/generate_llm_data.py \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR/existing_desc \
    --only_items \
    --use_existing_descriptions

# 示例 5: 使用显式指定的 API key（不使用环境变量）
echo -e "\n[示例 5] 使用显式 API key"
python UR4Rec/scripts/generate_llm_data.py \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR/with_key \
    --api_key "your-explicit-api-key" \
    --only_items \
    --max_items 5

echo -e "\n============================================"
echo "完成！检查输出目录: $OUTPUT_DIR"
echo "============================================"
echo ""
echo "提示："
echo "  - 默认使用 qwen-flash 模型"
echo "  - 默认从环境变量 DASHSCOPE_API_KEY 读取密钥"
echo "  - 可以使用 --api_key 参数显式指定密钥"
echo "  - 使用 --model_name 可以切换模型"
