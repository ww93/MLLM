#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_memory_para_fixed.py

A clean, reproducible analysis script to inspect multimodal embedding distributions
and data-driven hyperparameter hints for memory (capacity/topK/thresholds).

Key fixes vs common buggy prototypes:
- Robust loading for .pt/.pth/.npy (tensor or dict-like containers)
- Filters near-zero vectors to avoid "cosine==0 spikes" dominating statistics
- Uses sampling + chunked computation to avoid O(n^2) blow-ups on large catalogs
- Provides per-item nearest-neighbor (NN) max cosine percentiles
- Provides user novelty percentiles: novelty = 1 - maxcos(history window W)
- Provides a simple online cluster-count estimate per user (sanity check only)

Typical usage (MovieLens-1M subset style):
  python analyze_memory_para_fixed.py \
    --data_path UR4Rec/data/ml-1m/subset_ratings.dat \
    --visual_path UR4Rec/data/ml-1m/clip_features.pt \
    --text_path UR4Rec/data/ml-1m/text_features.pt \
    --num_items 3953 \
    --window 50 \
    --max_positions_per_user 400 \
    --sample_pairs 200000 \
    --plot_dir ./plots_mem

Notes:
- For very large num_items (e.g., >50k), keep --sample_pairs and --nn_chunk small.
- This script does NOT change your training code; it's only for analysis.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch


# -----------------------------
# Utilities
# -----------------------------

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _unwrap_feature_container(obj):
    """
    Handle common checkpoint container formats:
      - tensor
      - numpy array
      - dict with keys: 'features', 'emb', 'embedding', 'item_features', etc.
      - list/tuple containing tensor
    """
    if isinstance(obj, torch.Tensor) or isinstance(obj, np.ndarray):
        return obj
    if isinstance(obj, (list, tuple)) and len(obj) > 0:
        # try first element
        if isinstance(obj[0], (torch.Tensor, np.ndarray)):
            return obj[0]
    if isinstance(obj, dict):
        # try common keys
        for k in ["features", "feats", "emb", "embedding", "embeddings", "item_features", "item_emb"]:
            if k in obj and isinstance(obj[k], (torch.Tensor, np.ndarray)):
                return obj[k]
        # sometimes nested
        for v in obj.values():
            if isinstance(v, (torch.Tensor, np.ndarray)):
                return v
    raise ValueError(f"Unsupported feature container type: {type(obj)}")


def load_features(path: str, device: str = "cpu") -> torch.Tensor:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.endswith(".npy"):
        arr = np.load(path)
        t = torch.from_numpy(arr).float()
    elif path.endswith(".pt") or path.endswith(".pth"):
        obj = torch.load(path, map_location="cpu", weights_only=False)
        obj = _unwrap_feature_container(obj)
        if isinstance(obj, np.ndarray):
            t = torch.from_numpy(obj).float()
        else:
            t = obj.float()
    else:
        raise ValueError(f"Unsupported file: {path}")
    return t.to(device)


def load_sequences(data_path: str) -> Tuple[Dict[int, List[int]], int]:
    """
    Supports:
      - format A: user item1 item2 ...
      - format B: user item rating timestamp (one interaction per line)
    """
    user_sequences: Dict[int, List[int]] = {}
    max_item = 0

    with open(data_path, "r") as f:
        first = f.readline().strip()
        f.seek(0)
        parts = first.split()
        if len(parts) > 4:
            for line in f:
                p = line.strip().split()
                if len(p) < 2:
                    continue
                uid = int(p[0])
                items = [int(x) for x in p[1:]]
                if len(items) == 0:
                    continue
                user_sequences[uid] = items
                max_item = max(max_item, max(items))
        else:
            tmp: Dict[int, List[Tuple[int, int]]] = {}
            for line in f:
                p = line.strip().split()
                if len(p) < 2:
                    continue
                uid = int(p[0])
                iid = int(p[1])
                ts = int(p[3]) if len(p) >= 4 else 0
                tmp.setdefault(uid, []).append((ts, iid))
                max_item = max(max_item, iid)
            for uid, arr in tmp.items():
                arr.sort(key=lambda x: x[0])
                user_sequences[uid] = [iid for _, iid in arr]

    return user_sequences, max_item + 1


def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def valid_mask_nonzero(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # Treat near-zero vectors as invalid
    return x.norm(dim=-1) > eps


def percentiles(arr: np.ndarray, ps=(1, 5, 10, 25, 50, 75, 90, 95, 99)) -> Dict[int, float]:
    out = {}
    for p in ps:
        out[int(p)] = float(np.percentile(arr, p))
    return out


# -----------------------------
# Core computations
# -----------------------------

@torch.no_grad()
def sample_global_cosine(x: torch.Tensor, sample_pairs: int, chunk: int = 20000) -> np.ndarray:
    """
    Sample cosine similarities from random pairs (i, j), i != j.
    x must be L2-normalized.
    Returns a numpy array of sampled cosines.
    """
    n = x.shape[0]
    if sample_pairs <= 0:
        return np.array([], dtype=np.float32)

    # sample in chunks to avoid large temp tensors
    sims = []
    remain = sample_pairs
    while remain > 0:
        b = min(remain, chunk)
        i = torch.randint(0, n, (b,), device=x.device)
        j = torch.randint(0, n, (b,), device=x.device)
        # avoid i==j
        same = (i == j)
        if same.any():
            j[same] = (j[same] + 1) % n
        s = (x[i] * x[j]).sum(dim=-1)
        sims.append(s.cpu().numpy().astype(np.float32))
        remain -= b
    return np.concatenate(sims, axis=0)


@torch.no_grad()
def nn_max_cosine(x: torch.Tensor, nn_chunk: int = 512) -> np.ndarray:
    """
    Compute per-item max cosine similarity to any other item (approx exact, chunked).
    x must be L2-normalized.
    Complexity: O(n^2) but chunked. For n~4k it's fine.
    For n huge, consider reducing items or using ANN (not included here).
    """
    n, d = x.shape
    x_cpu = x  # keep device
    max_vals = torch.full((n,), -1.0, device=x.device)

    # chunked matrix multiplication
    for start in range(0, n, nn_chunk):
        end = min(n, start + nn_chunk)
        q = x_cpu[start:end]  # [b, d]
        # [b, n]
        sim = q @ x_cpu.T
        # exclude self
        idx = torch.arange(start, end, device=x.device)
        sim[torch.arange(end - start, device=x.device), idx] = -1.0
        max_vals[start:end] = sim.max(dim=1).values

    return max_vals.cpu().numpy().astype(np.float32)


@torch.no_grad()
def user_novelty(
    seqs: Dict[int, List[int]],
    emb: torch.Tensor,
    window: int,
    max_positions_per_user: int,
    device: str = "cpu",
) -> np.ndarray:
    """
    For each user, sample up to max_positions_per_user positions t where t>=1,
    compute max cosine between item_t and previous min(window, t) items.
    novelty = 1 - maxcos.
    emb must be L2-normalized.
    """
    novs = []
    for _, items in seqs.items():
        if len(items) < 2:
            continue
        # candidate positions (exclude first)
        positions = list(range(1, len(items)))
        if len(positions) > max_positions_per_user:
            # uniform subsample
            step = max(1, len(positions) // max_positions_per_user)
            positions = positions[::step][:max_positions_per_user]

        for t in positions:
            cur = items[t]
            if cur <= 0 or cur >= emb.shape[0]:
                continue
            hist = items[max(0, t - window):t]
            # filter invalid ids
            hist = [h for h in hist if 0 < h < emb.shape[0]]
            if len(hist) == 0:
                continue
            cur_vec = emb[cur]  # [d]
            hist_mat = emb[torch.tensor(hist, device=device)]  # [h, d]
            # max cosine
            maxcos = (hist_mat @ cur_vec).max().item()
            novs.append(1.0 - maxcos)
    return np.array(novs, dtype=np.float32)


@torch.no_grad()
def estimate_cluster_count(
    seqs: Dict[int, List[int]],
    emb: torch.Tensor,
    W: int,
    sim_th: float,
    users_limit: int = 200,
) -> np.ndarray:
    """
    Simple online clustering on a prefix of length W for each user:
      - keep one representative per cluster (first element assigned)
      - if maxcos(reps, x) >= sim_th => assign to best cluster
      - else create new cluster
    This is NOT a rigorous clustering algorithm; it's only a sanity indicator.
    """
    counts = []
    uids = list(seqs.keys())
    if users_limit > 0:
        uids = uids[:min(users_limit, len(uids))]
    for uid in uids:
        items = [i for i in seqs[uid] if 0 < i < emb.shape[0]]
        items = items[:W]
        reps: List[int] = []
        for iid in items:
            if len(reps) == 0:
                reps.append(iid)
                continue
            rep_mat = emb[torch.tensor(reps, device=emb.device)]  # [c, d]
            sim = (rep_mat @ emb[iid]).max().item()
            if sim < sim_th:
                reps.append(iid)
        counts.append(len(reps))
    return np.array(counts, dtype=np.int32)


def maybe_plot_hist(arr: np.ndarray, title: str, xlabel: str, outpath: str) -> None:
    import matplotlib.pyplot as plt
    plt.figure()
    plt.hist(arr, bins=60)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, default="UR4Rec/data/ml-1m/subset_ratings.dat", help="interaction file")
    ap.add_argument("--visual_path", type=str, default="UR4Rec/data/ml-1m/clip_features.pt", help="visual embedding file (.pt/.npy)")
    ap.add_argument("--text_path", type=str, default="UR4Rec/data/ml-1m/text_features.pt", help="text embedding file (.pt/.npy)")
    ap.add_argument("--num_items", type=int, default=-1, help="optional, verify shape[0]")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--sample_pairs", type=int, default=200000, help="sample size for global cosine")
    ap.add_argument("--nn_chunk", type=int, default=512, help="chunk size for NN max cosine (matmul)")
    ap.add_argument("--window", type=int, default=50, help="history window for novelty")
    ap.add_argument("--max_positions_per_user", type=int, default=400, help="subsample positions per user")
    ap.add_argument("--cluster_W", type=int, default=200)
    ap.add_argument("--cluster_sim_th", type=float, default=0.65)
    ap.add_argument("--cluster_users_limit", type=int, default=200)

    ap.add_argument("--plot_dir", type=str, default="UR4Rec/data/ml-1m/mm_memory_analysis", help="if set, save histograms to this dir")
    args = ap.parse_args()

    set_seed(args.seed)

    print(f"[1] Load sequences: {args.data_path}")
    seqs, inferred_num_items = load_sequences(args.data_path)
    print(f"  users={len(seqs)}  num_items={inferred_num_items}")

    print(f"[2] Load embeddings:")
    visual = load_features(args.visual_path, device=args.device)
    text = load_features(args.text_path, device=args.device)
    print(f"  visual: {tuple(visual.shape)}  text: {tuple(text.shape)}")

    if args.num_items > 0:
        if visual.shape[0] != args.num_items or text.shape[0] != args.num_items:
            print(f"  [WARN] num_items mismatch: arg={args.num_items} visual={visual.shape[0]} text={text.shape[0]}")

    # filter near-zero vectors (very important)
    vmask = valid_mask_nonzero(visual)
    tmask = valid_mask_nonzero(text)
    both_mask = vmask & tmask

    num_zero_v = int((~vmask).sum().item())
    num_zero_t = int((~tmask).sum().item())
    num_zero_any = int((~both_mask).sum().item())

    print(f"[2.1] Zero/near-zero vectors (norm<=1e-8):")
    print(f"  visual zero-like: {num_zero_v} / {visual.shape[0]}")
    print(f"  text   zero-like: {num_zero_t} / {text.shape[0]}")
    print(f"  either modality zero-like: {num_zero_any} / {visual.shape[0]}")
    if num_zero_any > 0:
        print("  NOTE: These items can create big cosine==0 spikes. We exclude them from cosine stats.")

    # normalize (only valid rows)
    visual_n = l2_normalize(visual)
    text_n = l2_normalize(text)

    # build filtered views
    valid_idx = torch.where(both_mask)[0]
    visual_f = visual_n[valid_idx]
    text_f = text_n[valid_idx]

    # [3] global cosine distribution via sampling
    print(f"[3] Global similarity distribution (sampled pairs={args.sample_pairs})")
    vis_s = sample_global_cosine(visual_f, args.sample_pairs)
    txt_s = sample_global_cosine(text_f, args.sample_pairs)
    print(f"  Visual cosine percentiles: {percentiles(vis_s)}")
    print(f"  Text   cosine percentiles: {percentiles(txt_s)}")

    # [4] per-item NN max cosine (exact chunked on filtered set)
    print(f"[4] Item nearest-neighbor similarity (max cosine to any other item)")
    vis_nn = nn_max_cosine(visual_f, nn_chunk=args.nn_chunk)
    txt_nn = nn_max_cosine(text_f, nn_chunk=args.nn_chunk)
    print(f"  Visual NN max cosine percentiles: {percentiles(vis_nn)}")
    print(f"  Text   NN max cosine percentiles: {percentiles(txt_nn)}")

    # [5] user novelty distribution
    # For novelty we need an embedding table aligned to original item ids.
    # We compute novelty only for items where both modalities are valid,
    # and ignore invalid items in history.
    print(f"[5] User novelty distribution (window={args.window}, max_positions_per_user={args.max_positions_per_user})")
    # We use "combined" cosine as average of (vis cos + text cos)/2 by building a combined normalized embedding.
    # This is simple and stable for gating writes; you can swap it to learned fusion later.
    comb = l2_normalize(torch.cat([visual_n, text_n], dim=-1))
    comb = comb.to(args.device)

    # novelty arrays
    vis_nov = user_novelty(seqs, visual_n, args.window, args.max_positions_per_user, device=args.device)
    txt_nov = user_novelty(seqs, text_n, args.window, args.max_positions_per_user, device=args.device)
    comb_nov = user_novelty(seqs, comb, args.window, args.max_positions_per_user, device=args.device)

    print(f"  Visual novelty percentiles: {percentiles(vis_nov)}")
    print(f"  Text   novelty percentiles: {percentiles(txt_nov)}")
    print(f"  Comb   novelty percentiles: {percentiles(comb_nov)}")

    # [6] cluster-count estimate (sanity)
    print(f"[6] Estimate per-user cluster count (W={args.cluster_W}, sim_th={args.cluster_sim_th}, users_limit={args.cluster_users_limit})")
    # cluster on combined embedding
    cc = estimate_cluster_count(seqs, comb, W=args.cluster_W, sim_th=args.cluster_sim_th, users_limit=args.cluster_users_limit)
    if cc.size > 0:
        print(f"  Cluster-count percentiles: {percentiles(cc.astype(np.float32))}")
    else:
        print("  Cluster-count: empty (check your sequences)")

    # Hyperparam hints (heuristic)
    if comb_nov.size > 0:
        median_n = float(np.percentile(comb_nov, 50))
        p90_n = float(np.percentile(comb_nov, 90))
        p95_n = float(np.percentile(comb_nov, 95))
        # write ratio mapping examples
        print("\n================ Suggested hyperparams (data-driven, heuristic) ================")
        print("Novelty-gated writes (recommended):")
        print(f"  target write_ratio≈10%  -> novelty_threshold≈p90≈{p90_n:.4f}")
        print(f"  target write_ratio≈5%   -> novelty_threshold≈p95≈{p95_n:.4f}")
        print(f"  target write_ratio≈50%  -> novelty_threshold≈median≈{median_n:.4f}")
        if cc.size > 0:
            cap_med = int(np.percentile(cc, 50))
            cap_p90 = int(np.percentile(cc, 90))
            cap_p95 = int(np.percentile(cc, 95))
            # cap suggestion: around p75~p90 clusters but clipped
            cap_suggest = int(np.clip(cap_p90, 50, 200))
            topk_suggest = int(np.clip(int(0.3 * cap_suggest), 8, 32))
            print("\nMemory capacity suggestion (sanity, depends on sim_th):")
            print(f"  cluster median≈{cap_med}, p90≈{cap_p90}, p95≈{cap_p95}")
            print(f"  capacity (clip[50,200]) ≈ {cap_suggest}")
            print(f"  TopK retrieval suggestion: ~{topk_suggest} (≈0.3*cap, clipped to [8,32])")
        print("Note: if you keep a loss-based surprise_threshold, tune it to match the desired write_ratio.\n")

    # plots
    if args.plot_dir:
        os.makedirs(args.plot_dir, exist_ok=True)
        maybe_plot_hist(cc.astype(np.float32), "Estimated cluster count per user", "cluster count", os.path.join(args.plot_dir, "hist_cluster_count.png"))
        maybe_plot_hist(vis_s, "Global cosine similarity (visual, sampled)", "cosine similarity", os.path.join(args.plot_dir, "hist_global_visual.png"))
        maybe_plot_hist(txt_s, "Global cosine similarity (text, sampled)", "cosine similarity", os.path.join(args.plot_dir, "hist_global_text.png"))
        maybe_plot_hist(vis_nn, "Per-item NN max cosine (visual)", "cosine similarity", os.path.join(args.plot_dir, "hist_item_nn_visual.png"))
        maybe_plot_hist(txt_nn, "Per-item NN max cosine (text)", "cosine similarity", os.path.join(args.plot_dir, "hist_item_nn_text.png"))
        # novelty plots
        # stack not required; separate for clarity
        maybe_plot_hist(vis_nov, f"Novelty = 1 - maxcos(history), visual, window={args.window}", "novelty", os.path.join(args.plot_dir, "hist_novelty_visual.png"))
        maybe_plot_hist(txt_nov, f"Novelty = 1 - maxcos(history), text, window={args.window}", "novelty", os.path.join(args.plot_dir, "hist_novelty_text.png"))
        maybe_plot_hist(comb_nov, f"Novelty = 1 - maxcos(history), combined, window={args.window}", "novelty", os.path.join(args.plot_dir, "hist_novelty_combined.png"))
        print(f"[plots] saved to: {args.plot_dir}")


if __name__ == "__main__":
    main()
