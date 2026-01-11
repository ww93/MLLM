# -*- coding: utf-8 -*-
"""
Two-tier (Short-Term + Long-Term) local dynamic memory for UR4Rec/FedMem.

Design goals:
- ST (short-term) captures *recent* interests and reacts quickly to drift (FIFO / recency-biased).
- LT (long-term) stores *diverse & stable* interests, updated sparsely via novelty-gated writes.

The novelty threshold defaults are data-driven for ML-1M based on analyze_memory_para_fixed.py:
- window W = 50 (ST capacity)
- LT write threshold (combined novelty) ~ p90 ≈ 0.583 for ~10% write ratio
- Retrieval topK suggestion ≈ 32
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict
from typing import Dict, Optional, Tuple, List

import math
import torch


# ----------------------------- helpers -----------------------------

def _l2_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if x is None:
        return None
    n = torch.norm(x, p=2, dim=-1, keepdim=True).clamp_min(eps)
    return x / n


def _is_zero_like(x: Optional[torch.Tensor], eps: float = 1e-8) -> bool:
    if x is None:
        return True
    return torch.norm(x, p=2).item() <= eps


def _build_combined_embedding(
    visual: Optional[torch.Tensor],
    text: Optional[torch.Tensor],
    eps: float = 1e-8
) -> Optional[torch.Tensor]:
    """
    Match analyze_memory_para_fixed.py's "combined" definition:
      combined = l2_norm( concat( l2_norm(visual), l2_norm(text) ) )
    If one modality is missing/zero-like, fallback to the other.
    """
    v_ok = (visual is not None) and (not _is_zero_like(visual, eps))
    t_ok = (text is not None) and (not _is_zero_like(text, eps))

    if not v_ok and not t_ok:
        return None

    if v_ok and not t_ok:
        return _l2_normalize(visual, eps)

    if t_ok and not v_ok:
        return _l2_normalize(text, eps)

    v = _l2_normalize(visual, eps)
    t = _l2_normalize(text, eps)
    comb = torch.cat([v, t], dim=-1)
    return _l2_normalize(comb, eps)


def _max_cosine_to_set(
    q: torch.Tensor,
    mem: List[torch.Tensor],
) -> float:
    """q and mem elements should already be L2-normalized."""
    if q is None or len(mem) == 0:
        return 0.0
    # Small set (<=50 for ST): loop is fine and avoids allocating big matrices.
    best = -1.0
    for m in mem:
        if m is None:
            continue
        s = float(torch.dot(q, m).item())
        if s > best:
            best = s
    return best if best > -1.0 else 0.0


def _exp_recency(age_steps: int, half_life_steps: float) -> float:
    if half_life_steps <= 0:
        return 0.0
    # exp(-age / tau), where tau = half_life / ln(2)
    tau = half_life_steps / math.log(2.0)
    return float(math.exp(-age_steps / tau))


# ----------------------------- entries -----------------------------

@dataclass
class MemoryEntry:
    item_id: int
    id_emb: torch.Tensor                      # [D_id]
    visual_emb: Optional[torch.Tensor] = None # [D_v]
    text_emb: Optional[torch.Tensor] = None   # [D_t]
    comb_emb: Optional[torch.Tensor] = None   # [D_v + D_t] normalized
    timestamp: int = 0                        # step index
    frequency: int = 1
    last_access: int = 0                      # step index


# ----------------------------- main memory -----------------------------

class LocalDynamicMemory:
    """
    Backward-compatible class name.

    - capacity: interpreted as LT capacity (recommended: ~200 for ML-1M based on your stats).
    - st_capacity: fixed by default to W=50 (from your novelty analysis window).
    """

    def __init__(
        self,
        capacity: int = 200,
        surprise_threshold: float = 0.3,   # kept for compatibility; LT uses novelty by default
        device: str = "cpu",
        *,
        st_capacity: int = 50,
        # data-driven defaults (ML-1M):
        lt_novelty_threshold: float = 0.5830,   # ~p90 combined novelty -> ~10% write ratio
        retrieve_topk: int = 32,
        st_retrieve_ratio: float = 0.25,        # 25% from ST, rest from LT
        lt_merge_sim_threshold: float = 0.74,   # heuristic from NN p95 (vis≈0.833, txt≈0.65)
        lt_recency_half_life_steps: int = 200,  # tie to W=200 cluster estimation window
        eps: float = 1e-8,
        # [FIX] 添加特征维度参数，用于empty memory时返回正确形状的零张量
        id_emb_dim: int = 128,              # ID嵌入维度
        visual_emb_dim: int = 512,          # 视觉特征维度 (CLIP)
        text_emb_dim: int = 384,            # 文本特征维度 (SBERT)
    ):
        self.device = device
        self.eps = eps

        # ST / LT capacities
        self.st_capacity = int(st_capacity)
        self.capacity = int(capacity)  # LT capacity (keep attribute name used elsewhere)

        # thresholds / knobs
        self.surprise_threshold = float(surprise_threshold)
        self.lt_novelty_threshold = float(lt_novelty_threshold)
        self.lt_merge_sim_threshold = float(lt_merge_sim_threshold)

        self.retrieve_topk = int(retrieve_topk)
        self.st_retrieve_ratio = float(st_retrieve_ratio)

        self.lt_recency_half_life_steps = int(lt_recency_half_life_steps)

        # [FIX] 存储特征维度，用于empty memory时返回正确形状
        self.id_emb_dim = int(id_emb_dim)
        self.visual_emb_dim = int(visual_emb_dim)
        self.text_emb_dim = int(text_emb_dim)

        # buffers
        self.st_buffer: "OrderedDict[int, MemoryEntry]" = OrderedDict()  # item_id -> entry (recency order)
        self.memory_buffer: Dict[int, MemoryEntry] = {}                  # LT memory (item_id -> entry)

        # stats
        self.total_updates_st = 0
        self.total_updates_lt = 0
        self.total_promotions = 0
        self.total_expires_lt = 0

        # optional: global abstract memory (server-level prototypes), not trained locally
        self.global_abstract_memory: Optional[torch.Tensor] = None

        # internal step counter (monotonic)
        self._step = 0

    def __len__(self) -> int:
        # by convention, memory_size = LT size (used in logs)
        return len(self.memory_buffer)

    # ----------------------------- update -----------------------------

    @torch.no_grad()
    def update(
        self,
        item_id: int,
        id_emb: torch.Tensor,
        visual_emb: Optional[torch.Tensor] = None,
        text_emb: Optional[torch.Tensor] = None,
        loss_val: Optional[float] = None,
    ) -> None:
        """
        Update ST always; update/promote into LT when novelty is high enough.
        - Novelty is computed against ST window (W=st_capacity) using combined embedding
          consistent with analyze_memory_para_fixed.py.
        """
        self._step += 1
        step = self._step

        # Ensure tensors are on CPU/device consistently
        id_emb = id_emb.detach().to(self.device)

        v = visual_emb.detach().to(self.device) if visual_emb is not None else None
        t = text_emb.detach().to(self.device) if text_emb is not None else None
        comb = _build_combined_embedding(v, t, eps=self.eps)

        # --- compute novelty vs ST (before inserting current) ---
        st_comb_list = [e.comb_emb for e in self.st_buffer.values() if e.comb_emb is not None]
        maxcos = _max_cosine_to_set(comb, st_comb_list) if comb is not None else 0.0
        novelty = 1.0 - maxcos

        # --- ST update (always) ---
        self._update_st(item_id, id_emb, v, t, comb, step)
        self.total_updates_st += 1

        # --- LT update condition ---
        # Primary: novelty gate (data-driven)
        write_to_lt = (comb is not None) and (novelty >= self.lt_novelty_threshold)

        # Fallback: if no multimodal embedding, keep old loss-based behavior
        if comb is None and (loss_val is not None):
            write_to_lt = float(loss_val) >= self.surprise_threshold

        if write_to_lt:
            self._update_lt(item_id, id_emb, v, t, comb, step)
            self.total_updates_lt += 1

    def _update_st(
        self,
        item_id: int,
        id_emb: torch.Tensor,
        v: Optional[torch.Tensor],
        t: Optional[torch.Tensor],
        comb: Optional[torch.Tensor],
        step: int,
    ) -> None:
        if item_id in self.st_buffer:
            e = self.st_buffer.pop(item_id)
            e.id_emb = id_emb
            e.visual_emb = v
            e.text_emb = t
            e.comb_emb = comb
            e.timestamp = step
            e.frequency += 1
            self.st_buffer[item_id] = e
        else:
            self.st_buffer[item_id] = MemoryEntry(
                item_id=item_id,
                id_emb=id_emb,
                visual_emb=v,
                text_emb=t,
                comb_emb=comb,
                timestamp=step,
                frequency=1,
                last_access=step,
            )

        # FIFO eviction to enforce ST capacity
        while len(self.st_buffer) > self.st_capacity:
            self.st_buffer.popitem(last=False)

    def _update_lt(
        self,
        item_id: int,
        id_emb: torch.Tensor,
        v: Optional[torch.Tensor],
        t: Optional[torch.Tensor],
        comb: Optional[torch.Tensor],
        step: int,
    ) -> None:
        # If already in LT: update stats and optionally refresh embeddings
        if item_id in self.memory_buffer:
            e = self.memory_buffer[item_id]
            e.timestamp = step
            e.frequency += 1
            # refresh embeddings (latest snapshot)
            e.id_emb = id_emb
            e.visual_emb = v
            e.text_emb = t
            e.comb_emb = comb
            return

        # Else: try merge into nearest LT entry if very similar (avoid storing near-duplicates)
        if comb is not None and len(self.memory_buffer) > 0:
            best_id, best_sim = self._find_most_similar_lt(comb)
            if best_id is not None and best_sim >= self.lt_merge_sim_threshold:
                e = self.memory_buffer[best_id]
                # lightweight merge: update frequency and recency; keep representative embedding as EMA
                e.frequency += 1
                e.timestamp = step
                # EMA on combined embedding if available
                if e.comb_emb is not None:
                    alpha = 0.1  # small update, keep stability
                    e.comb_emb = _l2_normalize((1 - alpha) * e.comb_emb + alpha * comb, eps=self.eps)
                # also update id_emb as EMA (optional)
                alpha_id = 0.1
                e.id_emb = (1 - alpha_id) * e.id_emb + alpha_id * id_emb
                return

        # Add as new LT entry (may trigger eviction)
        if len(self.memory_buffer) >= self.capacity:
            self._expire_least_useful_lt(step)

        self.memory_buffer[item_id] = MemoryEntry(
            item_id=item_id,
            id_emb=id_emb,
            visual_emb=v,
            text_emb=t,
            comb_emb=comb,
            timestamp=step,
            frequency=1,
            last_access=step,
        )
        self.total_promotions += 1

    def _find_most_similar_lt(self, comb_q: torch.Tensor) -> Tuple[Optional[int], float]:
        best_id = None
        best_sim = -1.0
        for iid, e in self.memory_buffer.items():
            if e.comb_emb is None:
                continue
            s = float(torch.dot(comb_q, e.comb_emb).item())
            if s > best_sim:
                best_sim = s
                best_id = iid
        if best_id is None:
            return None, 0.0
        return best_id, best_sim

    def _utility_lt(self, e: MemoryEntry, now_step: int) -> float:
        # frequency gives importance; recency helps adapt to drift
        freq_term = math.log1p(e.frequency)
        age = max(0, now_step - e.timestamp)
        rec = _exp_recency(age, self.lt_recency_half_life_steps)
        return float(freq_term + rec)

    def _expire_least_useful_lt(self, now_step: int) -> None:
        if not self.memory_buffer:
            return
        worst_id = None
        worst_u = float("inf")
        for iid, e in self.memory_buffer.items():
            u = self._utility_lt(e, now_step)
            if u < worst_u:
                worst_u = u
                worst_id = iid
        if worst_id is not None:
            self.memory_buffer.pop(worst_id, None)
            self.total_expires_lt += 1

    # ----------------------------- retrieval -----------------------------

    @torch.no_grad()
    def retrieve_multimodal_memory_batch(
        self,
        batch_size: int,
        top_k: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return memory tensors for a whole batch.

        Output shapes:
          mem_vis: [B, K, Dv]  (zeros if missing)
          mem_txt: [B, K, Dt]
          mem_id : [B, K, Did]
          mask  : [B, K]  1 for valid, 0 for padded
        """
        K = int(top_k) if top_k is not None else self.retrieve_topk
        if K <= 0:
            # empty
            return (
                torch.zeros(batch_size, 0, 1, device=self.device),
                torch.zeros(batch_size, 0, 1, device=self.device),
                torch.zeros(batch_size, 0, 1, device=self.device),
                torch.zeros(batch_size, 0, device=self.device),
            )

        # decide split
        k_st = min(max(0, int(round(K * self.st_retrieve_ratio))), len(self.st_buffer))
        k_lt = min(K - k_st, len(self.memory_buffer))

        # ST: most recent
        st_entries = list(self.st_buffer.values())[-k_st:] if k_st > 0 else []

        # LT: top utility
        if k_lt > 0:
            now_step = self._step
            lt_sorted = sorted(
                self.memory_buffer.values(),
                key=lambda e: self._utility_lt(e, now_step),
                reverse=True,
            )
            lt_entries = lt_sorted[:k_lt]
        else:
            lt_entries = []

        entries = st_entries + lt_entries
        # pad if not enough
        while len(entries) < K:
            entries.append(None)

        # [FIX] 优先使用存储的维度，确保empty memory时返回正确形状
        # 如果有非空entry，从第一个非空entry推断维度；否则使用存储的默认维度
        first_valid_entry = next((e for e in entries if e is not None), None)

        if first_valid_entry is not None:
            did = int(first_valid_entry.id_emb.numel())
            dv = int(first_valid_entry.visual_emb.numel()) if first_valid_entry.visual_emb is not None else self.visual_emb_dim
            dt = int(first_valid_entry.text_emb.numel()) if first_valid_entry.text_emb is not None else self.text_emb_dim
        else:
            # Empty memory: use stored dimensions
            did = self.id_emb_dim
            dv = self.visual_emb_dim
            dt = self.text_emb_dim

        mem_id = torch.zeros(K, did, device=self.device)
        mem_vis = torch.zeros(K, dv, device=self.device)
        mem_txt = torch.zeros(K, dt, device=self.device)
        mask = torch.zeros(K, device=self.device)

        for i, e in enumerate(entries):
            if e is None:
                continue
            mem_id[i] = e.id_emb.view(-1)
            if e.visual_emb is not None:
                mem_vis[i] = e.visual_emb.view(-1)
            if e.text_emb is not None:
                mem_txt[i] = e.text_emb.view(-1)
            mask[i] = 1.0
            e.last_access = self._step

        # expand to batch
        mem_id = mem_id.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        mem_vis = mem_vis.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        mem_txt = mem_txt.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        mask = mask.unsqueeze(0).expand(batch_size, -1).contiguous()

        return mem_vis, mem_txt, mem_id, mask

    # ----------------------------- prototypes / global memory -----------------------------

    @torch.no_grad()
    def get_memory_prototypes(self, k: int = 5) -> Optional[torch.Tensor]:
        """
        Return k prototypes from LT memory (for prototype aggregation on server).
        Strategy: choose top-k LT entries by utility and return their id_emb.
        (You can replace with k-means later; this is stable + cheap.)
        """
        if k <= 0 or len(self.memory_buffer) == 0:
            return None
        now_step = self._step
        lt_sorted = sorted(
            self.memory_buffer.values(),
            key=lambda e: self._utility_lt(e, now_step),
            reverse=True,
        )
        selected = lt_sorted[: min(k, len(lt_sorted))]
        protos = torch.stack([e.id_emb for e in selected], dim=0)  # [k, Did]
        return protos

    def set_global_abstract_memory(self, prototypes: Optional[torch.Tensor]) -> None:
        """
        Store server-provided global prototypes (optional).
        Not mixed into LT by default; you can retrieve/use it elsewhere if needed.
        """
        self.global_abstract_memory = prototypes.detach().to(self.device) if prototypes is not None else None

    def get_statistics(self) -> Dict[str, float]:
        return {
            "st_size": float(len(self.st_buffer)),
            "lt_size": float(len(self.memory_buffer)),
            "total_updates_st": float(self.total_updates_st),
            "total_updates_lt": float(self.total_updates_lt),
            "total_promotions": float(self.total_promotions),
            "total_expires_lt": float(self.total_expires_lt),
            "lt_novelty_threshold": float(self.lt_novelty_threshold),
            "lt_merge_sim_threshold": float(self.lt_merge_sim_threshold),
            "st_capacity": float(self.st_capacity),
            "lt_capacity": float(self.capacity),
            "retrieve_topk": float(self.retrieve_topk),
        }
