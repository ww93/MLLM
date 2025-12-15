# UR4Rec Training Report - MovieLens-100K

## 📊 Training Summary

### Configuration
- Dataset: MovieLens-100K (943 users, 1,659 items, 99,309 ratings)
- Model: UR4Rec V2 with MoE (26.5M parameters)
- Training Stages: 4 stages × 25 epochs = 100 total epochs
- Fusion Weights: SASRec 0.4, Retriever 0.6

### Performance by Stage

| Stage | Epochs | Hit@10 | NDCG@10 | vs Previous |
|-------|--------|--------|---------|-------------|
| **1. Pretrain SASRec** | 18/25 | 0.3177 | 0.1510 | baseline |
| **2. Pretrain Retriever** | 16/25 | **0.4606** | **0.2527** | **+45%** / +67% |
| **3. Joint Finetune** | 7/25 | 0.4542 | 0.2500 | -1.4% / -1.1% |
| **4. End-to-End** | 6/25 | 0.4531 | 0.2516 | -0.2% / +0.6% |

**Final Best Performance:** Hit@10 = **0.4606**, NDCG@10 = **0.2527**

---

## ✅ What Worked

### 1. **Bug Fixes Were Critical**
- **Retriever Normalization Bug**: Fixed missing L2 normalization before dot product
  - Before: Loss = 89-142 (catastrophic)
  - After: Loss = 1.37 (**85-136x improvement**)
  - Files: `retriever_moe.py:118-122`, `retriever_moe_memory.py:304-310`

- **Memory Loading Bug**: Fixed user_id parsing to handle float strings
  - File: `retriever_moe_memory.py:392`

- **Test Evaluation Bug**: Added item ID clamping to handle OOV items
  - File: `sasrec.py:228`

### 2. **Retriever Brought Massive Improvement**
- Stage 2 (pretrain_retriever) achieved **+45% hit@10** vs Stage 1
- Confirms retriever can effectively leverage LLM-generated text
- Text-based matching complements sequential patterns well

### 3. **Multi-Stage Training Strategy**
- Separate pretraining allowed each module to specialize
- Early stopping prevented overfitting (patience=15)

---

## ❌ What Didn't Work

### 1. **Failed to Reach Target (0.7-0.8 Hit@10)**
- Current: **0.46**
- Target: **0.70-0.80**
- **Gap: 52-74% relative improvement still needed**

### 2. **Joint Training Degraded Performance**
- Stage 3 & 4 slightly worse than Stage 2
- Suggests optimization conflicts between SASRec and Retriever
- May need better learning rate scheduling or gradient balancing

### 3. **Training Stopped Early**
- Stage 2: Stopped at epoch 16/25 (patience limit)
- Stage 3: Stopped at epoch 7/25
- Stage 4: Stopped at epoch 6/25
- Suggests local optima or unstable training

---

## 🔍 Root Cause Analysis

### Why Performance Plateaued at 0.46?

#### **1. Insufficient Retriever Capacity**
- Current: 256-dim embeddings, 4 MoE heads, 4 proxies
- Text encoder frozen (all-MiniLM-L6-v2)
- **Problem**: May be too simple to capture complex preferences

#### **2. Training Instability**
- Frequent early stopping suggests:
  - Learning rates may be too high
  - Fusion weights (0.4/0.6) may cause gradient imbalance
  - Memory mechanism update frequency may be suboptimal

#### **3. Limited Text Signal**
- LLM-generated descriptions may be too generic
- User preferences may lack personalization detail
- Cold-start users have weak text signals

#### **4. Suboptimal Fusion**
- Fixed weighted fusion (0.4/0.6) is too rigid
- SASRec and Retriever may work better with learned gates
- Joint training degrades performance → optimization conflict

---

## 🚀 Recommended Improvements

### **Priority 1: Enhance Retriever Capacity** 🎯

**A. Scale Up Retriever**
```yaml
# Increase model capacity
retriever_output_dim: 256 → 512
moe_num_heads: 4 → 8
moe_num_proxies: 4 → 8

# Deeper text encoding
text_encoder_layers: +1-2 extra layers with fine-tuning
```

**B. Fine-tune Text Encoder**
```yaml
# Unfreeze text encoder
freeze_encoder: true → false
text_encoder_lr: 0.0001  # Lower LR for pretrained weights
```

---

### **Priority 2: Fix Training Instability** ⚙️

**A. Adaptive Learning Rates**
```yaml
# Use cosine annealing with warmup
scheduler: "cosine_with_warmup"
warmup_epochs: 5
min_lr: 1e-6

# Lower learning rates
sasrec_lr: 0.001 → 0.0005
retriever_lr: 0.003 → 0.001
```

**B. Gradient Balancing**
```yaml
# Add gradient clipping per-module
max_grad_norm_sasrec: 1.0
max_grad_norm_retriever: 1.0

# Use uncertainty weighting
use_uncertainty_weighting: false → true
```

**C. Longer Training**
```yaml
# Increase patience
patience: 15 → 25

# More epochs per stage
epochs_per_stage: 25 → 50
```

---

### **Priority 3: Improve Fusion Strategy** 🔗

**A. Learnable Fusion**
```yaml
fusion_method: "weighted" → "learned"

# Add fusion network
fusion_hidden_dim: 128
fusion_dropout: 0.1
```

**B. Dynamic Weighting**
```yaml
# User-specific weights
use_user_specific_fusion: true

# Time-varying weights (early: SASRec, late: Retriever)
use_curriculum_fusion: true
```

---

### **Priority 4: Enhance Text Quality** 📝

**A. Richer LLM Prompts**
- Generate multi-aspect preferences (genre, mood, theme, style)
- Include contrastive descriptions (likes vs dislikes)
- Add temporal evolution cues

**B. User-Item Co-Attention**
```python
# Replace simple dot product with cross-attention
cross_attention = nn.MultiheadAttention(dim, num_heads)
enhanced_score = cross_attention(user_pref, item_desc)
```

---

### **Priority 5: Better Negative Sampling** 🎲

```yaml
# Current: 20 random negatives
num_negatives: 20 → 50

# Add hard negatives
hard_negative_ratio: 0.3  # 30% hard negatives
hard_negative_method: "in_batch"  # or "mined"
```

---

## 📋 Action Plan

### **Immediate Next Steps (Quick Wins)**

1. **Increase Retriever Capacity** (Est. +10-15% Hit@10)
   ```yaml
   retriever_output_dim: 512
   moe_num_heads: 8
   moe_num_proxies: 8
   ```

2. **Fix Learning Rate Schedule** (Est. +5-8% Hit@10)
   ```yaml
   sasrec_lr: 0.0005
   retriever_lr: 0.001
   scheduler: "cosine_with_warmup"
   warmup_epochs: 5
   patience: 25
   ```

3. **Switch to Learned Fusion** (Est. +5-10% Hit@10)
   ```yaml
   fusion_method: "learned"
   use_uncertainty_weighting: true
   ```

**Expected Result:** Hit@10 = **0.55-0.60** (+20-30% improvement)

---

### **Medium-Term Improvements**

4. **Fine-tune Text Encoder** (Est. +8-12% Hit@10)
   - Unfreeze last 2 layers of text encoder
   - Use very low LR (1e-5)

5. **Improve Negative Sampling** (Est. +5-8% Hit@10)
   - Increase to 50-100 negatives
   - Add 30% hard negatives

**Expected Result:** Hit@10 = **0.63-0.68** (additional +15-20%)

---

### **Long-Term Research Directions**

6. **Advanced Architecture**
   - Multi-task learning (predict + explain)
   - Retrieval-augmented generation for recommendations
   - Graph neural networks for user-item-text relations

7. **Better Text Generation**
   - Fine-tune LLaMA/Mistral on recommendation data
   - Generate explanations for recommendations
   - Multi-modal fusion (images + text)

**Potential Result:** Hit@10 = **0.70-0.80** (reach target)

---

## 💾 Files Modified

1. `UR4Rec/models/retriever_moe.py` (lines 118-122) - L2 normalization fix
2. `UR4Rec/models/retriever_moe_memory.py` (lines 304-310, 392) - L2 norm + user_id parsing
3. `UR4Rec/models/sasrec.py` (line 228) - Item ID clamping
4. `UR4Rec/configs/ur4rec_moe_100k.yaml` - Training config

---

## 📈 Expected Roadmap

| Phase | Actions | Expected Hit@10 | Time |
|-------|---------|----------------|------|
| **Current** | Baseline + Bug Fixes | 0.46 | ✅ Done |
| **Phase 1** | Capacity + LR + Fusion | 0.55-0.60 | 1-2 days |
| **Phase 2** | Text Fine-tune + Hard Negatives | 0.63-0.68 | 3-5 days |
| **Phase 3** | Advanced Architecture | 0.70-0.80 | 1-2 weeks |

---

## 🎯 Key Takeaways

1. **Retriever works!** (+45% improvement proves concept)
2. **Bug fixes were critical** (85x loss reduction)
3. **Need more capacity** (model is underfitting, not overfitting)
4. **Training instability** (early stopping too aggressive)
5. **Fusion needs work** (joint training degrades performance)

**Bottom Line:** Current approach is fundamentally sound, but needs scaling up and training improvements to reach 0.7-0.8 target.
