#!/bin/bash
# FedDMMR Project Cleanup Script
# 根据 FEDDMMR_CLEANUP_PLAN.md 自动删除不相关文件

# set -e disabled to continue even if some files are already deleted

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 统计变量
DELETED_COUNT=0
FAILED_COUNT=0

# 删除文件函数
delete_file() {
    local file=$1
    if [ -f "$file" ]; then
        rm -f "$file"
        echo -e "${GREEN}✓${NC} Deleted: $file"
        ((DELETED_COUNT++))
    elif [ -d "$file" ]; then
        rm -rf "$file"
        echo -e "${GREEN}✓${NC} Deleted directory: $file"
        ((DELETED_COUNT++))
    else
        echo -e "${YELLOW}⚠${NC}  Not found (skipped): $file"
    fi
}

echo "=========================================="
echo "FedDMMR Project Cleanup Script"
echo "=========================================="
echo ""
echo -e "${YELLOW}WARNING:${NC} This script will delete files permanently!"
echo "Please review FEDDMMR_CLEANUP_PLAN.md before proceeding."
echo ""
read -p "Do you want to continue? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Cleanup cancelled."
    exit 0
fi

echo ""
echo "Starting cleanup..."
echo ""

# ====================
# 阶段 1: 删除文档文件
# ====================
echo -e "${GREEN}[Stage 1/7]${NC} Deleting documentation files..."

# 根目录文档
delete_file "FEDMEM_ADAPTATION_COMPLETED.md"
delete_file "FEDMEM_PROJECT_SUMMARY.md"
delete_file "FEDMEM_NEG_SAMPLING_IMPLEMENTATION.md"
delete_file "FedMem_ADAPTATION_GUIDE.md"
delete_file "FedMem_README.md"
delete_file "DIAGNOSTIC_SUMMARY.md"
delete_file "FINAL_DIAGNOSIS.md"
delete_file "CONFIG_PARAMETER_REMOVED.md"
delete_file "FEDERATED_VS_CENTRALIZED_PERFORMANCE.md"
delete_file "DIAGNOSIS_REPORT.md"
delete_file "debug_training_issue.md"
delete_file "QWEN_FLASH_INTEGRATION.md"

# UR4Rec 文档
delete_file "UR4Rec/PROJECT_COMPLETE.md"
delete_file "UR4Rec/FEDMEM_IMPLEMENTATION.md"
delete_file "UR4Rec/MULTIMODAL_DATA_GUIDE.md"
delete_file "UR4Rec/CHANGELOG_TEXT_DESCRIPTIONS.md"
delete_file "UR4Rec/RETRIEVER_ANALYSIS.md"
delete_file "UR4Rec/WORKFLOW.md"
delete_file "UR4Rec/TRAINING_GUIDE.md"
delete_file "UR4Rec/DOCS_INDEX.md"

echo ""

# ====================
# 阶段 2: 删除旧模型文件
# ====================
echo -e "${GREEN}[Stage 2/7]${NC} Deleting obsolete model files..."

delete_file "UR4Rec/models/hierarchical_moe.py"
delete_file "UR4Rec/models/ur4rec_v2.py"
delete_file "UR4Rec/models/user_preference_retriever.py"
delete_file "UR4Rec/models/text_preference_retriever.py"
delete_file "UR4Rec/models/text_preference_retriever_moe.py"
delete_file "UR4Rec/models/retriever_moe_memory.py"
delete_file "UR4Rec/models/multimodal_retriever.py"
delete_file "UR4Rec/models/joint_trainer.py"
# KEEP: federated_aggregator.py - Required by fedmem_client.py and fedmem_server.py
delete_file "UR4Rec/models/federated_server.py"
delete_file "UR4Rec/models/federated_client.py"
delete_file "UR4Rec/models/federated_client_ur4rec.py"


echo ""

# ====================
# 阶段 3: 删除旧训练脚本
# ====================
echo -e "${GREEN}[Stage 3/7]${NC} Deleting obsolete training scripts..."

delete_file "UR4Rec/scripts/train_federated.py"
delete_file "UR4Rec/scripts/train_federated_ur4rec_moe.py"
delete_file "UR4Rec/scripts/train_ur4rec_moe.py"
delete_file "UR4Rec/scripts/train_v2.py"
delete_file "UR4Rec/scripts/train_sasrec_centralized.py"
# KEEP: generate_llm_data.py - Used to generate text features for multimodal training
# KEEP: test_llm_connection.py - Utility for testing LLM API connection
# KEEP: extract_clip_features.py - Used to generate CLIP visual features
# KEEP: generate_text_features.py - Used to generate text embeddings
delete_file "UR4Rec/scripts/extract_ml1m_descriptions.py"
delete_file "UR4Rec/scripts/LLM_DATA_GENERATION_README.md"
delete_file "UR4Rec/scripts/diagnose_training_eval_mismatch.py"
delete_file "UR4Rec/scripts/diagnostic_check_embedding_update.py"
delete_file "UR4Rec/scripts/test_item_pop_baseline.py"
delete_file "UR4Rec/scripts/process_ml100k_4star.py"

# KEEP: preprocess_movielens.py and process_ml100k.py - Data preprocessing scripts
delete_file "UR4Rec/scripts/preprocess_images.py"
delete_file "UR4Rec/scripts/preprocess_beauty.py"
delete_file "UR4Rec/scripts/download_images.py"


echo ""

# ====================
# 阶段 4: 删除根目录诊断脚本
# ====================
echo -e "${GREEN}[Stage 4/7]${NC} Deleting diagnostic scripts..."

delete_file "train_sasrec_fixed.py"
delete_file "diagnose_id_mapping_bug.py"
delete_file "diagnose_router_weights.py"
delete_file "diagnose_scoring.py"
delete_file "test_model_forward.py"
delete_file "test_negative_sampling.py"
delete_file "analyze_training.py"
delete_file "analyze_expert_contributions.py"

echo ""

# ====================
# 阶段 5: 删除配置文件
# ====================
echo -e "${GREEN}[Stage 5/7]${NC} Deleting obsolete config files..."

delete_file "UR4Rec/configs/ur4rec_moe_100k.yaml"
delete_file "UR4Rec/configs/ur4rec_hierarchical_balanced.yaml"
delete_file "UR4Rec/configs/ur4rec_federated.yaml"
delete_file "UR4Rec/config_ml100k.yaml"

echo ""

# ====================
# 阶段 6: 删除示例和其他
# ====================
echo -e "${GREEN}[Stage 6/7]${NC} Deleting examples and miscellaneous files..."

delete_file "UR4Rec/examples"
delete_file "UR4Rec/demo.py"
delete_file "UR4Rec/test_text_extraction.py"
delete_file "UR4Rec/setup.py"
delete_file "UR4Rec/data/multimodal_dataset.py"

echo ""

# ====================
# 阶段 7: 清理 Checkpoints
# ====================
echo -e "${GREEN}[Stage 7/7]${NC} Cleaning checkpoints..."

if [ -d "UR4Rec/checkpoints" ]; then
    # 删除所有 checkpoint 子目录，但保留 checkpoints 目录本身
    find UR4Rec/checkpoints -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} \;
    # 删除所有文件（除了 README.md）
    find UR4Rec/checkpoints -maxdepth 1 -type f ! -name "README.md" -exec rm -f {} \;

    # 创建 README.md
    cat > UR4Rec/checkpoints/README.md << 'EOF'
# FedDMMR Checkpoints Directory

Training checkpoints will be saved here.

## Usage

When training with `train_fedmem.py`, use the `--save_dir` argument to specify a checkpoint subdirectory:

```bash
python UR4Rec/scripts/train_fedmem.py \
    --data_dir UR4Rec/data \
    --data_file ml100k_ratings_processed.dat \
    --visual_file clip_features_fixed.pt \
    --text_file item_text_features.pt \
    --save_dir UR4Rec/checkpoints/my_experiment
```

## Structure

Each checkpoint directory contains:
- `config.json` - Training configuration
- `fedmem_model.pt` - Model weights
- `train_history.json` - Training history
EOF

    echo -e "${GREEN}✓${NC} Cleaned checkpoints directory"
    ((DELETED_COUNT++))
fi

echo ""

# ====================
# 总结
# ====================
echo "=========================================="
echo "Cleanup Complete!"
echo "=========================================="
echo ""
echo -e "${GREEN}Successfully deleted:${NC} $DELETED_COUNT items"
if [ $FAILED_COUNT -gt 0 ]; then
    echo -e "${RED}Failed:${NC} $FAILED_COUNT items"
fi
echo ""
echo "Remaining core files for FedDMMR:"
echo ""
echo "Models:"
echo "  ✓ UR4Rec/models/ur4rec_v2_moe.py (FedDMMR main model)"
echo "  ✓ UR4Rec/models/sasrec.py (SASRec backbone)"
echo "  ✓ UR4Rec/models/local_dynamic_memory.py (Dynamic memory)"
echo "  ✓ UR4Rec/models/fedmem_client.py (Federated client)"
echo "  ✓ UR4Rec/models/fedmem_server.py (Federated server)"
echo "  ✓ UR4Rec/models/federated_aggregator.py (Aggregation utils)"
echo ""
echo "Scripts:"
echo "  ✓ UR4Rec/scripts/train_fedmem.py (Main training script)"
echo "  ✓ UR4Rec/scripts/generate_llm_data.py (Text feature generation)"
echo "  ✓ UR4Rec/scripts/test_llm_connection.py (LLM API test)"
echo "  ✓ UR4Rec/scripts/extract_clip_features.py (CLIP feature extraction)"
echo "  ✓ UR4Rec/scripts/generate_text_features.py (Text embeddings)"
echo "  ✓ UR4Rec/scripts/preprocess_movielens.py (Data preprocessing)"
echo "  ✓ UR4Rec/scripts/process_ml100k.py (ML-100K preprocessing)"
echo ""
echo "Data & Config:"
echo "  ✓ UR4Rec/data/ (All dataset files preserved)"
echo "  ✓ UR4Rec/utils/metrics.py (Evaluation metrics)"
echo "  ✓ UR4Rec/configs/fedmem_config.yaml"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "  1. Review remaining files"
echo "  2. Test training script: python UR4Rec/scripts/train_fedmem.py --help"
echo "  3. Update README.md with FedDMMR documentation"
echo ""
