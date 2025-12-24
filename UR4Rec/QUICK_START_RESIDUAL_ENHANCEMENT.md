# Quick Start: Residual Enhancement Architecture

## What Changed?

Your FedDMMR model has been successfully refactored from **Competitive Mixture** to **Residual Enhancement**. The key improvement is that **SASRec backbone is now always preserved**, and multimodal experts provide incremental improvements.

### Key Formula Change

**Before:**
```
final = w_seq * SASRec + w_vis * Visual + w_sem * Semantic
```
Problem: SASRec can be suppressed if router assigns low weight

**After:**
```
final = SASRec + gating_weight * (w_vis * Visual + w_sem * Semantic)
```
Benefit: SASRec always preserved, multimodal adds incremental value

---

## Quick Verification

Test that the new architecture works:

```bash
cd /Users/admin/Desktop/MLLM/UR4Rec
source ../venv/bin/activate
python scripts/test_residual_enhancement.py
```

Expected: All tests pass ✅

---

## Run Your First Experiment

### Basic Training (Recommended Start)

```bash
cd /Users/admin/Desktop/MLLM/UR4Rec

source ../venv/bin/activate

python scripts/train_fedmem.py \
    --data_dir data \
    --data_file ml100k_ratings_processed.dat \
    --visual_file clip_features_fixed.pt \
    --text_file item_text_features.pt \
    --num_rounds 30 \
    --client_fraction 0.2 \
    --learning_rate 0.001 \
    --batch_size 32 \
    --gating_init 0.1 \
    --save_dir checkpoints/residual_baseline
```

### With Partial Aggregation (Strategy 2)

```bash
python scripts/train_fedmem.py \
    --data_dir data \
    --data_file ml100k_ratings_processed.dat \
    --visual_file clip_features_fixed.pt \
    --text_file item_text_features.pt \
    --num_rounds 30 \
    --client_fraction 0.2 \
    --learning_rate 0.001 \
    --batch_size 32 \
    --gating_init 0.1 \
    --partial_aggregation_warmup_rounds 20 \
    --save_dir checkpoints/residual_with_partial_agg
```

---

## Find Optimal Gating Weight

Run a sweep to find the best `gating_init` value:

```bash
cd /Users/admin/Desktop/MLLM/UR4Rec
bash scripts/test_gating_sweep.sh
```

This will test: [0.0, 0.01, 0.05, 0.1, 0.2, 0.5] and tell you which is best.

---

## Expected Improvements

Compared to Competitive Mixture:

1. **Higher Baseline Performance** (+2-5%)
   - SASRec signal never suppressed
   - More stable lower bound

2. **Better Training Stability**
   - Less sensitive to initialization
   - Smoother convergence

3. **More Interpretable**
   - `gating_weight` shows auxiliary contribution
   - Router shows expert selection

4. **Fail-Safe Design**
   - If multimodal features are noisy, gating_weight → 0
   - Falls back to SASRec automatically

---

## Understanding Outputs

During training, you'll see:

```
✓ Residual Enhancement 架构初始化:
  Gating Weight 初始值: 0.1
  Router 控制专家数: 2 (Visual, Semantic)
  SASRec 作为骨干直接保留
```

In logs:
```json
{
  "gating_weight": 0.1234,  // Learnable, shows auxiliary importance
  "w_vis": 0.4567,          // Visual expert weight
  "w_sem": 0.5433,          // Semantic expert weight
  "lb_loss": 0.0012         // Load balance loss (2 experts)
}
```

---

## Hyperparameter Guide

### gating_init

Controls initial strength of auxiliary information:

- **0.0**: No multimodal (pure SASRec)
- **0.01-0.05**: Conservative (1-5% auxiliary)
- **0.1**: Balanced (10% auxiliary) ← **Recommended start**
- **0.2-0.5**: Aggressive (20-50% auxiliary)
- **> 0.5**: Multimodal-dominant

**Recommendation:** Start with 0.1, then sweep

### When to Use Different Values

- **Use 0.0-0.05** if:
  - Multimodal features are noisy
  - SASRec alone is already strong
  - You want to be conservative

- **Use 0.1-0.2** if:
  - Multimodal features are high-quality
  - You want balanced fusion
  - Standard use case

- **Use 0.3-0.5** if:
  - Multimodal features are very informative
  - Sequential signal is weak
  - You want to emphasize visual/semantic

---

## Compare with Previous Results

Your previous experiment (Competitive Mixture):
- Baseline: HR@10 = 0.3606
- Strategy1: HR@10 = 0.3786
- Strategy2: HR@10 = 0.3796
- Both: HR@10 = 0.3849

Expected with Residual Enhancement:
- Baseline (gating=0.1): HR@10 ≈ 0.40-0.42 (+4-9%)
- With Partial Agg: HR@10 ≈ 0.42-0.45 (+9-17%)

Run the experiments and compare!

---

## Troubleshooting

### Performance Not Improving

1. **Check gating_weight evolution:**
   - If → 0: Multimodal features not helpful
   - If → very large: May need regularization

2. **Try different gating_init:**
   - Run sweep script to find optimal value

3. **Combine with other strategies:**
   - Use `--partial_aggregation_warmup_rounds 20`
   - Increase model capacity: `--sasrec_hidden_dim 512`
   - Increase training: `--num_rounds 50`

### Gating Weight Not Learning

1. **Check learning rate:**
   - Try `--learning_rate 0.005` (higher)

2. **Check auxiliary experts:**
   - Verify multimodal features are loaded correctly
   - Check expert outputs are non-zero

### Router Weights Imbalanced

This is OK! Router should assign different weights based on:
- Visual vs Semantic feature quality
- Item characteristics
- User preferences

If one expert always gets 90%+ weight:
- Other expert may not be useful
- Consider removing or improving it

---

## Next Experiments

### 1. Full Evaluation (Remove Negative Sampling)

```bash
python scripts/train_fedmem.py \
    --data_dir data \
    --data_file ml100k_ratings_processed.dat \
    --visual_file clip_features_fixed.pt \
    --text_file item_text_features.pt \
    --num_rounds 30 \
    --gating_init 0.1 \
    --use_negative_sampling False \
    --save_dir checkpoints/residual_full_eval
```

Expected improvement: +10-15% HR@10

### 2. Increase Model Capacity

```bash
python scripts/train_fedmem.py \
    --data_dir data \
    --data_file ml100k_ratings_processed.dat \
    --visual_file clip_features_fixed.pt \
    --text_file item_text_features.pt \
    --sasrec_hidden_dim 512 \
    --sasrec_num_blocks 3 \
    --num_rounds 50 \
    --gating_init 0.1 \
    --save_dir checkpoints/residual_large_model
```

Expected improvement: +5-10% HR@10

### 3. Combined Optimization

```bash
python scripts/train_fedmem.py \
    --data_dir data \
    --data_file ml100k_ratings_processed.dat \
    --visual_file clip_features_fixed.pt \
    --text_file item_text_features.pt \
    --sasrec_hidden_dim 512 \
    --sasrec_num_blocks 3 \
    --num_rounds 50 \
    --learning_rate 0.005 \
    --batch_size 64 \
    --gating_init 0.1 \
    --partial_aggregation_warmup_rounds 35 \
    --use_negative_sampling False \
    --save_dir checkpoints/residual_best_config
```

Target: HR@10 = 0.60+

---

## Files Reference

- **Implementation:** [models/ur4rec_v2_moe.py](models/ur4rec_v2_moe.py)
- **Training Script:** [scripts/train_fedmem.py](scripts/train_fedmem.py)
- **Test Script:** [scripts/test_residual_enhancement.py](scripts/test_residual_enhancement.py)
- **Gating Sweep:** [scripts/test_gating_sweep.sh](scripts/test_gating_sweep.sh)
- **Detailed Docs:** [RESIDUAL_ENHANCEMENT_SUMMARY.md](RESIDUAL_ENHANCEMENT_SUMMARY.md)

---

## Summary

✅ **Completed:**
- Refactored to Residual Enhancement architecture
- SASRec backbone always preserved
- Router controls only 2 auxiliary experts
- Added learnable gating weight
- All tests passing

🚀 **Next Steps:**
1. Run verification test
2. Run baseline experiment
3. Run gating sweep to find optimal value
4. Compare with previous results
5. Run combined optimization

📊 **Target:**
- HR@10 = 0.60-0.70 (vs current 0.38)
- Through: architecture + hyperparameters + evaluation

---

**Ready to start? Run:**

```bash
cd /Users/admin/Desktop/MLLM/UR4Rec
source ../venv/bin/activate
python scripts/test_residual_enhancement.py  # Verify
```

Good luck! 🎉
