# -*- coding: utf-8 -*-
"""
gefe_functional.py
Utility-only functions and minimal type definitions for reuse by the ExtremeTrain class.

Includes:
1) Extreme extraction and event clustering (extreme_from_samples, cluster_extremes_event_from_res)
2) Frequency-domain metrics (PLV / Ck / Lk) and selection (select_contributing_freqs_*)
3) Pair summarization for training (summarize_event_list)
4) DDPM-related schedules and noising utilities (make_beta_schedule, sample_t_late_biased, forward_diffuse, forward_diffuse_eval_mean)
5) Evaluation helpers (hits_with_tol_unique, hitk_presence_single, coverage_atk)
6) Extra utilities (set_seed, etc.)

Note: This file does NOT include Dataset/Model/training loops or other training-facing classes.
"""

from typing import List, Tuple, Optional, Literal, TypedDict, Dict, Any
import math
import random
import numpy as np
import torch
import torch.nn.functional as F

# =========================
# Type definitions
# =========================

class SampleExtremePoint(TypedDict):
    sample_index: int
    extreme_indices: np.ndarray

class EventRep(TypedDict):
    t: int                  # representative timestamp
    s: int                  # polarity: +1 or -1
    W: float                # event weight (capped)
    cluster_indices: np.ndarray
    span: int               # cluster span
    r_dil: Optional[int]    # dilation radius for event mask

class SampleExtremeEvent(TypedDict):
    representatives: List[EventRep]
    event_mask: Optional[np.ndarray]

# =========================
# 1) Extreme extraction & event clustering
# =========================

def extreme_from_samples(
    samples: np.ndarray,
    p: float = 0.1,
    alpha_mean: float = 0.05,
    k_sigma: float = 1.5,
    k_mad: float = 2.5,
) -> List[np.ndarray]:
    """Extract extreme indices per sample using a mixed threshold. Input is (N, L, 1) or (N, L, C=1)."""
    x = np.asarray(samples, dtype=float)[..., 0]  # (N, L)
    N, L = x.shape
    results: List[np.ndarray] = []

    for i in range(N):
        w = x[i]
        mu = float(w.mean())
        sigma = float(w.std(ddof=0))
        med = float(np.median(w))
        mad = float(np.median(np.abs(w - med)))
        mad_scaled = 1.4826 * mad

        deviation = np.abs(w - mu)
        if not np.isfinite(deviation).all() or deviation.max() == 0:
            results.append(np.array([], dtype=int))
            continue

        rel_thr = max(alpha_mean * abs(mu), k_sigma * sigma, k_mad * mad_scaled)
        q_thr = float(np.quantile(deviation, 1.0 - p))
        thr = max(rel_thr, q_thr)

        idx_local = np.where(deviation >= thr)[0]
        results.append(idx_local.astype(int))

    return results


def cluster_extremes_event_from_res(
    samples: np.ndarray,
    res: List[np.ndarray],
    *,
    trend: Literal["none", "mean", "linear"] = "linear",
    rho_near: float = 0.02,
    rho_dil: Optional[float] = 0.02
) -> List[SampleExtremeEvent]:
    """Cluster extremes by polarity, derive representatives, event weights, and event masks."""
    CAP_SCALE = 2.5
    WEIGHT_MODE = "robust_z"

    x = np.asarray(samples, dtype=float)[..., 0]  # (N, L)
    N, L = x.shape
    out: List[SampleExtremeEvent] = []

    for i in range(N):
        y = x[i]
        ext = np.asarray(res[i], dtype=int)
        if ext.size == 0:
            out.append({"representatives": [], "event_mask": None})
            continue

        # detrend
        n = np.arange(L, dtype=float)
        if trend == "none":
            baseline = np.zeros_like(y)
        elif trend == "mean":
            baseline = np.full(L, y.mean())
        elif trend == "linear":
            a, b = np.polyfit(n, y, 1)
            baseline = a * n + b
        else:
            raise ValueError("trend must be in {'none','mean','linear'}")

        resid = y - baseline

        vals = resid[ext]
        signs_all = np.where(vals >= 0.0, +1, -1).astype(int)

        eps = 1e-12
        med = float(np.median(resid))
        mad = float(np.median(np.abs(resid - med)))
        sigma_r = 1.4826 * mad + eps
        if WEIGHT_MODE == "robust_z":
            w_all = np.abs(vals) / sigma_r
        else:
            raise ValueError("unsupported weight_mode")

        r = int(max(1, round(rho_near * L)))
        r_dil_base = int(max(1, round((rho_dil or 0.0) * L))) if rho_dil is not None else None

        ref = float(np.median(w_all)) if w_all.size > 0 else 0.0
        W_cap = max(ref * CAP_SCALE, 1e-6)

        reps: List[EventRep] = []

        for sgn in (+1, -1):
            sel = np.where(signs_all == sgn)[0]
            if sel.size == 0:
                continue

            t_loc = ext[sel]
            order = np.argsort(t_loc)
            t_loc = t_loc[order]
            w_sel = w_all[sel][order]

            # proximity-based clustering
            clusters = []
            cur = [0]
            for j in range(1, t_loc.size):
                if (t_loc[j] - t_loc[j - 1]) <= r:
                    cur.append(j)
                else:
                    clusters.append(np.array(cur, dtype=int))
                    cur = [j]
            clusters.append(np.array(cur, dtype=int))

            for inds in clusters:
                cg_loc = t_loc[inds]
                cg_w = w_sel[inds]

                rep_i = int(np.argmax(cg_w))
                t_rep = int(cg_loc[rep_i])

                W_raw = float(cg_w.sum())
                W_eff = float(W_cap * np.tanh(W_raw / W_cap))

                span = int(cg_loc.max() - cg_loc.min()) if cg_loc.size > 1 else 0
                r_dil_rep = None
                if r_dil_base is not None:
                    r_dil_rep = int(max(r_dil_base, span // 2))

                reps.append(EventRep(
                    t=t_rep, s=sgn, W=W_eff,
                    cluster_indices=cg_loc.copy(),
                    span=span, r_dil=r_dil_rep
                ))

        event_mask = None
        if rho_dil is not None:
            event_mask = np.zeros(L, dtype=bool)
            for rep in reps:
                rd = rep["r_dil"] or 0
                a = max(0, rep["t"] - rd)
                b = min(L - 1, rep["t"] + rd)
                if a <= b:
                    event_mask[a:b + 1] = True

        reps.sort(key=lambda d: d["t"])
        out.append({"representatives": reps, "event_mask": event_mask})

    return out

# =========================
# 2) Frequency-domain computations & metrics
# =========================

def _rfft_amp_phase(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float).reshape(-1)
    m = len(y)
    Y = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(m, d=1.0)   # normalized frequency in cycles/sample
    A = np.abs(Y)
    phi = np.angle(Y)
    return freqs, A, phi


def metric_plv_row(x_row: np.ndarray, events_item: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Weighted polarity phase-locking value. Q=0 -> zeros; Q=1 -> ones."""
    y = np.asarray(x_row, dtype=float).reshape(-1)
    freqs, _, phi = _rfft_amp_phase(y)
    K = freqs.size

    reps = events_item.get("representatives", [])
    Q = len(reps)
    if Q == 0:
        return freqs, np.zeros(K, dtype=float)
    if Q == 1:
        return freqs, np.ones(K, dtype=float)

    tau = np.array([rep["t"] for rep in reps], dtype=float)
    sgn = np.array([rep.get("s", 1.0) for rep in reps], dtype=float)
    W = np.array([rep.get("W", 1.0) for rep in reps], dtype=float)

    phi_k = phi[:, None]
    f_k = freqs[:, None]
    tau_q = tau[None, :]
    s_q = sgn[None, :]
    W_q = W[None, :]

    psi_kq = phi_k + 2.0 * np.pi * f_k * tau_q
    vec_kq = np.exp(1j * s_q * psi_kq)
    num = np.abs((W_q * vec_kq).sum(axis=1))
    den = W_q.sum(axis=1) + 1e-12
    plv = (num / den).astype(float)
    return freqs, plv


def metric_ck_row(x_row: np.ndarray, events_item: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Ck: non-negative cosine mean with phase alignment; zero out frequencies with zero amplitude."""
    y = np.asarray(x_row, dtype=float).reshape(-1)
    freqs, A, phi = _rfft_amp_phase(y)
    K = freqs.size

    reps = events_item.get("representatives", [])
    Q = len(reps)
    if Q == 0:
        return freqs, np.zeros(K, dtype=float)

    tau = np.array([rep["t"] for rep in reps], dtype=float)
    sgn = np.array([rep.get("s", 1.0) for rep in reps], dtype=float)
    W = np.array([rep.get("W", 1.0) for rep in reps], dtype=float)

    Wsum = W.sum() + 1e-12
    wtil = (W / Wsum)[None, :]

    phi_k = phi[:, None]
    f_k = freqs[:, None]
    tau_q = tau[None, :]
    s_q = sgn[None, :]

    psi_kq = phi_k + 2.0 * np.pi * f_k * tau_q
    cos_help = np.cos(s_q * psi_kq)
    cos_help = np.clip(cos_help, 0.0, None)

    Ck = (wtil * cos_help).sum(axis=1)
    Ck = np.where(A > 0.0, Ck, 0.0).astype(float)
    return freqs, Ck


def metric_lk_row(x_row: np.ndarray, events_item: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Lk: contrast between event and background (mean absolute value), normalized to [0,1]."""
    y = np.asarray(x_row, dtype=float).reshape(-1)
    N = y.size
    freqs, A, phi = _rfft_amp_phase(y)
    K = freqs.size

    mask = events_item.get("event_mask", None)
    if mask is None:
        return freqs, np.zeros(K, dtype=float)

    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if mask.size != N or mask.sum() == 0 or (~mask).sum() == 0:
        return freqs, np.zeros(K, dtype=float)

    t = np.arange(N, dtype=float)
    T = t[None, :]
    f_k = freqs[:, None]
    phi_k = phi[:, None]

    xk = A[:, None] * np.cos(2.0 * np.pi * f_k * T + phi_k)
    xk_abs = np.abs(xk)

    eps = 1e-12
    mu_evt = xk_abs[:, mask].mean(axis=1)
    mu_bg = xk_abs[:, ~mask].mean(axis=1)
    raw = (mu_evt - mu_bg) / (mu_evt + mu_bg + eps)
    Lk = 0.5 * (raw + 1.0)
    return freqs, Lk.astype(float)

# Aggregation of the three metrics

def add_freq_metrics_to_item_row(x_row: np.ndarray, events_item: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(events_item)
    freqs_plv, plv = metric_plv_row(x_row, events_item)
    freqs_ck, Ck = metric_ck_row(x_row, events_item)
    freqs_lk, Lk = metric_lk_row(x_row, events_item)
    assert freqs_plv.shape == freqs_ck.shape == freqs_lk.shape
    out["freq_metrics"] = {"freqs": freqs_plv, "plv": plv, "Ck": Ck, "Lk": Lk}
    return out


def add_freq_metrics_to_rows(
    X: np.ndarray,
    events_per_row: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Batch-add PLV/Ck/Lk to each row."""
    X = np.asarray(X)
    if X.ndim == 3 and X.shape[-1] == 1:
        X = X[..., 0]
    assert X.ndim == 2, f"X must be (M,N) or (M,N,1), got {X.shape}"
    M, _ = X.shape
    assert len(events_per_row) == M, "events_per_row length must match batch size M"

    out: List[Dict[str, Any]] = []
    for i in range(M):
        out.append(add_freq_metrics_to_item_row(X[i], events_per_row[i]))
    return out

# Selection & summarization

def _align4(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = min(len(a), len(b), len(c), len(d))
    return a[:n], b[:n], c[:n], d[:n]


def select_contributing_freqs_item(
    item: Dict[str, Any],
    *,
    thr_plv: float = 0.70,
    thr_ck: float = 0.60,
    thr_lk: float = 0.55,
    weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    top_k: int = 5
) -> Dict[str, Any]:
    """Select top-k contributing frequencies via thresholds and weighted scores."""
    out = dict(item)

    fm = out.get("freq_metrics", {})
    freqs = np.asarray(fm.get("freqs", []), dtype=float)
    plv = np.asarray(fm.get("plv", []), dtype=float)
    Ck = np.asarray(fm.get("Ck", []), dtype=float)
    Lk = np.asarray(fm.get("Lk", []), dtype=float)

    if freqs.size == 0:
        out["extre_freq"] = []
        return out

    freqs, plv, Ck, Lk = _align4(freqs, plv, Ck, Lk)

    mask_plv = (plv >= thr_plv)
    mask_ck = mask_plv & (Ck >= thr_ck)
    mask_lk = mask_ck & (Lk >= thr_lk)
    idx_pass = np.flatnonzero(mask_lk)

    a, b, c = weights
    denom = (a + b + c) if (a + b + c) > 0 else 1.0
    score_all = (a * plv + b * Ck + c * Lk) / denom

    if idx_pass.size >= top_k:
        order = np.argsort(-score_all[idx_pass])
        idx_sorted = idx_pass[order[:top_k]]
    else:
        order = np.argsort(-score_all)
        idx_sorted = order[:top_k]

    extre = []
    for k in idx_sorted:
        extre.append({
            "k": int(k),
            "f": float(freqs[k]),
            "plv": float(plv[k]),
            "Ck": float(Ck[k]),
            "Lk": float(Lk[k]),
            "score": float(score_all[k])
        })
    out["extre_freq"] = extre
    return out


def select_contributing_freqs_all(
    extre_event_list: List[Dict[str, Any]],
    **kwargs
) -> List[Dict[str, Any]]:
    return [select_contributing_freqs_item(item, **kwargs) for item in extre_event_list]


def summarize_event_list(X: np.ndarray, extreme_event_list: List[Dict[str, Any]], n: int = 5):
    """
    Return training pairs: [(x_row, top_freqs)] where len(top_freqs) == n.
    X: (M, N) or (M, N, 1)
    """
    X = np.asarray(X)
    if X.ndim == 3 and X.shape[-1] == 1:
        X = X[..., 0]
    assert X.ndim == 2, f"X must be (M,N) or (M,N,1), got {X.shape}"
    M, _ = X.shape
    assert len(extreme_event_list) == M, "extreme_event_list length must match rows of X"

    result = []
    for i, event in enumerate(extreme_event_list):
        x_row = X[i]
        extre_freqs = event.get("extre_freq", [])
        sorted_freqs = sorted(extre_freqs, key=lambda d: d["score"], reverse=True)
        top_freqs = [f["f"] for f in sorted_freqs[:n]]
        result.append((x_row.astype(np.float32), top_freqs))

    return result

# =========================
# 3) DDPM noising & time sampling utilities
# =========================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_beta_schedule(T: int, start=1e-4, end=0.02):
    """Linear beta schedule of shape (T,) as a torch tensor."""
    return torch.linspace(start, end, T)


def sample_t_late_biased(B: int, T: int, gamma: float):
    """Late-biased discrete t sampling in [1, T], returns LongTensor of shape (B,)."""
    u = torch.rand(B)
    t_cont = -(T / gamma) * torch.log(1 - u * (1 - math.exp(-gamma)))
    return torch.clamp(t_cont.round().long(), 1, T)


@torch.no_grad()
def forward_diffuse(x0: torch.Tensor, t: torch.Tensor, ALPHAS_CUM: torch.Tensor):
    """Sampling from q(x_t|x_0)."""
    a_bar = ALPHAS_CUM.to(x0.device)[t - 1].view(-1, 1, 1)
    noise = torch.randn_like(x0)
    return torch.sqrt(a_bar) * x0 + torch.sqrt(1 - a_bar) * noise


@torch.no_grad()
def forward_diffuse_eval_mean(x0: torch.Tensor, t: torch.Tensor, ALPHAS_CUM: torch.Tensor):
    """Mean of q(x_t|x_0) for deterministic noising in evaluation."""
    a_bar = ALPHAS_CUM.to(x0.device)[t - 1].view(-1, 1, 1)
    return torch.sqrt(a_bar) * x0

# =========================
# 4) Evaluation helpers (Top-k hit & coverage)
# =========================

def hits_with_tol_unique(pred_bins, true_bins, delta):
    """Unique matching with tolerance; returns number of hits."""
    used = [False] * len(pred_bins)
    hits = 0
    for t in sorted(true_bins):
        best = -1
        bestd = delta + 1
        for j, p in enumerate(sorted(pred_bins)):
            if used[j]:
                continue
            d = abs(p - t)
            if d <= delta and d < bestd:
                bestd = d
                best = j
                if d == 0:
                    break
        if best >= 0:
            used[best] = True
            hits += 1
    return hits


def hitk_presence_single(pred_bins, true_bins, delta: int):
    """Presence-based hit count within tolerance; returns (hits, recall)."""
    pset = set(pred_bins)
    hits = 0
    for t in true_bins:
        found = any((t + d) in pset for d in range(-delta, delta + 1))
        if found:
            hits += 1
    return hits, hits / max(len(true_bins), 1)


@torch.no_grad()
def coverage_atk(logits: torch.Tensor, true_bins_batch: torch.Tensor, delta: int, k: int):
    """Batch coverage@k with tolerance."""
    B, Kall = logits.shape
    hit_sum = 0.0
    for b in range(B):
        tb = true_bins_batch[b].tolist()
        pred = torch.topk(logits[b], k=min(k, Kall)).indices.tolist()
        hits = hits_with_tol_unique(pred, tb, delta)
        hit_sum += hits / max(len(tb), 1)
    return hit_sum / max(B, 1)


__all__ = [
    # types
    "SampleExtremePoint", "EventRep", "SampleExtremeEvent",
    # extremes & clustering
    "extreme_from_samples", "cluster_extremes_event_from_res",
    # frequency metrics & selection
    "_rfft_amp_phase", "metric_plv_row", "metric_ck_row", "metric_lk_row",
    "add_freq_metrics_to_item_row", "add_freq_metrics_to_rows",
    "select_contributing_freqs_item", "select_contributing_freqs_all",
    "summarize_event_list",
    # DDPM utils
    "set_seed", "make_beta_schedule", "sample_t_late_biased",
    "forward_diffuse", "forward_diffuse_eval_mean",
    # evaluation helpers
    "hits_with_tol_unique", "hitk_presence_single", "coverage_atk",
]
