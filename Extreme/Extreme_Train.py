# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
from types import SimpleNamespace
from typing import List, Tuple, Optional, Sequence, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from Extreme.Extreme_Util import (
    extreme_from_samples, cluster_extremes_event_from_res,
    add_freq_metrics_to_rows, select_contributing_freqs_all, summarize_event_list,
    set_seed, make_beta_schedule, sample_t_late_biased,
    forward_diffuse, forward_diffuse_eval_mean,
    coverage_atk,
)

class TimestepEmbed(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(1, d), nn.SiLU(), nn.Linear(d, d))
    def forward(self, t: torch.Tensor):
        return self.mlp(t.float().unsqueeze(-1) / 1000.0)

class Head(nn.Module):
    def __init__(self, L: int, d=128):
        super().__init__()
        self.K = L // 2 + 1
        self.temb = TimestepEmbed(d)
        self.backbone = nn.Sequential(
            nn.Conv1d(1, d, 5, padding=2, padding_mode='circular'),
            nn.GELU(),
            nn.Conv1d(d, d, 3, padding=1, padding_mode='circular'),
            nn.GELU(),
        )
        self.gamma = nn.Linear(d, d)
        self.beta  = nn.Linear(d, d)
        self.projK = nn.Conv1d(d, d, 1)
        self.head  = nn.Conv1d(d, 1, 1)
        if self.head.bias is not None:
            nn.init.constant_(self.head.bias, -2.0)
    def forward(self, x_t: torch.Tensor, t: torch.Tensor):
        h  = self.backbone(x_t.transpose(1, 2))
        te = self.temb(t)
        h  = h * (1 + self.gamma(te).unsqueeze(-1)) + self.beta(te).unsqueeze(-1)
        hK = F.interpolate(h, size=self.K, mode='linear', align_corners=False)
        return self.head(self.projK(hK)).squeeze(1)  # (B,K)

class FixedKFreqDataset(Dataset):
    def __init__(self, data_pairs: List[Tuple[np.ndarray, List[float]]], L: int, k_true: int):
        self.L = L
        self.K = L // 2 + 1
        self.k_true = int(k_true)
        self.data = []
        for x0_np, flist in data_pairs:
            assert len(flist) == self.k_true, f"Provide exactly {self.k_true} target frequencies per sample"
            x0 = np.asarray(x0_np, np.float32).reshape(L, 1)
            bins = sorted(set(self.freqs_to_bins(flist, L)))
            if len(bins) < self.k_true:
                need = self.k_true - len(bins)
                all_bins = list(range(self.K))
                cand = [b for b in all_bins if b not in bins]
                extra = np.random.choice(cand, size=need, replace=False).tolist()
                bins = sorted(bins + extra)
            y = np.zeros(self.K, dtype=np.float32)
            for k in bins: y[k] = 1.0
            self.data.append((x0, y, np.array(bins, dtype=np.int64)))
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        x0, y, bins = self.data[idx]
        return torch.from_numpy(x0), torch.from_numpy(y), torch.from_numpy(bins)
    @staticmethod
    def freqs_to_bins(freqs: Sequence[float], L: int):
        K = L // 2 + 1
        out = []
        for f in freqs:
            k = int(round(float(f) * L))
            k = max(0, min(K - 1, k))
            out.append(k)
        return out

def topk_margin_loss(logits: torch.Tensor, k: int, m: float):
    if k <= 0:
        return torch.zeros(logits.size(0), device=logits.device)
    vals, _ = torch.sort(logits, dim=1, descending=True)
    t, n = vals[:, :k], vals[:, k:]
    if n.numel() == 0:
        return torch.zeros(logits.size(0), device=logits.device)
    viol = F.relu(m - (t.unsqueeze(2) - n.unsqueeze(1)))
    return viol.mean(dim=(1, 2))

class ExtremeTrain:
    DEFAULTS: Dict[str, Any] = {
        "L": 200,
        "T": 1000,
        "batch_size": 64,
        "epochs": 5,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "delta": 2,
        "topk": 5,
        "topk_margin": 0.4,
        "lambda_topk": 2e-3,
        "pos_weight": 1.6,
        "late_bias_gamma": 5.0,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "seed": 42,
        "d_model": 128,
        "beta_start": 1e-4,
        "beta_end": 2e-2,
        "extreme_extract": {
            "p": 0.10,
            "alpha_mean": 0.10,
            "k_sigma": 2.0,
            "k_mad": 3.0,
            "rho_near": 0.02,
            "rho_dil": 0.02,
            "trend": "linear",
        },
    }

    def __init__(self, extreme_cfg: Dict[str, Any]):
        cfg_in = extreme_cfg or {}

        merged = dict(self.DEFAULTS)
        for k, v in cfg_in.items():
            if k == "extreme_extract":
                continue
            if v is not None:
                merged[k] = v

        ee_default = self.DEFAULTS["extreme_extract"]
        ee_user = (cfg_in.get("extreme_extract") or {})
        ee_merged = dict(ee_default)
        ee_merged.update({k: v for k, v in ee_user.items() if v is not None})

        try:
            ee_merged["p"] = float(ee_merged["p"])
            if not (0.0 < ee_merged["p"] < 1.0):
                raise ValueError("extreme_extract.p must be in (0,1)")
            ee_merged["alpha_mean"] = float(ee_merged["alpha_mean"])
            ee_merged["k_sigma"]    = float(ee_merged["k_sigma"])
            ee_merged["k_mad"]      = float(ee_merged["k_mad"])
            ee_merged["rho_near"]   = float(ee_merged["rho_near"])
            ee_merged["rho_dil"]    = float(ee_merged["rho_dil"])
            if ee_merged["trend"] not in ("none", "mean", "linear"):
                raise ValueError("extreme_extract.trend must be one of {'none','mean','linear'}")
        except Exception as e:
            raise ValueError(f"Invalid extreme_extract config: {e}")

        merged["extreme_extract"] = ee_merged

        dev = str(merged.get("device", "cuda"))
        if dev.lower() == "cuda" and not torch.cuda.is_available():
            dev = "cpu"
        merged["device"] = dev

        self.cfg = SimpleNamespace(**merged)

        set_seed(self.cfg.seed)

        self.BETAS = make_beta_schedule(self.cfg.T, start=self.cfg.beta_start, end=self.cfg.beta_end)
        self.ALPHAS = 1.0 - self.BETAS
        self.ALPHAS_CUM = torch.cumprod(self.ALPHAS, dim=0)

        self.model = Head(L=self.cfg.L, d=self.cfg.d_model).to(self.cfg.device)

    def make_pairs_from_dataset_cfg(self, samples: np.ndarray) -> List[Tuple[np.ndarray, List[float]]]:
        samples = np.asarray(samples)
        assert samples.ndim == 3 and samples.shape[-1] == 1, f"samples must be (M,N,1), got {samples.shape}"

        ee = self.cfg.extreme_extract

        res = extreme_from_samples(
            samples,
            p=ee["p"],
            alpha_mean=ee["alpha_mean"],
            k_sigma=ee["k_sigma"],
            k_mad=ee["k_mad"],
        )

        events = cluster_extremes_event_from_res(
            samples, res,
            trend=ee["trend"],
            rho_near=ee["rho_near"],
            rho_dil=ee["rho_dil"],
        )

        events = add_freq_metrics_to_rows(samples, events)
        events = select_contributing_freqs_all(events, top_k=self.cfg.topk)

        pairs = summarize_event_list(samples, events, n=self.cfg.topk)
        return pairs

    def fit(self,
            train_pairs: List[Tuple[np.ndarray, List[float]]],
            val_pairs: Optional[List[Tuple[np.ndarray, List[float]]]] = None) -> Dict[str, Any]:
        cfg = self.cfg
        device = cfg.device
        train_ds = FixedKFreqDataset(train_pairs, cfg.L, k_true=cfg.topk)
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
        val_loader = None
        if val_pairs is not None:
            val_loader = DataLoader(FixedKFreqDataset(val_pairs, cfg.L, k_true=cfg.topk),
                                    batch_size=cfg.batch_size, shuffle=False)
        opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        pos_w = torch.tensor(cfg.pos_weight, device=device)

        def run_epoch(loader, train_flag=True):
            self.model.train(train_flag)
            loss_sum = bce_sum = tk_sum = cov_sum = 0.0
            n_batches = 0
            for x0, y, bins in loader:
                x0 = x0.to(device); y = y.to(device); bins = bins.to(device)
                B = x0.size(0)
                t  = sample_t_late_biased(B, cfg.T, cfg.late_bias_gamma).to(device)
                xt = forward_diffuse(x0, t, self.ALPHAS_CUM)
                logits = self.model(xt, t)
                bce_el = F.binary_cross_entropy_with_logits(logits, y, reduction='none', pos_weight=pos_w)
                bce = bce_el.mean(dim=1)
                tkm = topk_margin_loss(logits, cfg.topk, cfg.topk_margin)
                loss = (bce + cfg.lambda_topk * tkm).mean()
                if train_flag:
                    opt.zero_grad(); loss.backward(); opt.step()
                loss_sum += float(loss.item())
                bce_sum  += float(bce.mean().item())
                tk_sum   += float(tkm.mean().item())
                cov_sum  += float(coverage_atk(logits, bins, cfg.delta, cfg.topk))
                n_batches += 1
            return dict(loss=loss_sum / n_batches, bce=bce_sum / n_batches,
                        topk=tk_sum / n_batches, cov=cov_sum / n_batches)

        history = {"train": [], "val": []}
        for ep in range(1, cfg.epochs + 1):
            tr = run_epoch(train_loader, True); history["train"].append(tr)
            msg = f"[Ep {ep:03d}] train: loss={tr['loss']:.4f} | bce={tr['bce']:.4f} | topk={tr['topk']:.4f} | Cov@{cfg.topk}={tr['cov']:.3f}"
            if val_loader is not None:
                with torch.no_grad():
                    val = run_epoch(val_loader, False); history["val"].append(val)
                msg += f" || val: loss={val['loss']:.4f} | bce={val['bce']:.4f} | topk={val['topk']:.4f} | Cov@{cfg.topk}={val['cov']:.3f}"
            # print(msg)
        return {"history": history}

    @torch.no_grad()
    def predict_bins(self, x_t: torch.Tensor, t: torch.Tensor, k: Optional[int] = None) -> List[List[int]]:
        self.model.eval()
        k = k or self.cfg.topk
        logits = self.model(x_t.to(self.cfg.device), t.to(self.cfg.device))
        return torch.topk(logits, k=min(k, logits.shape[1]), dim=1).indices.cpu().tolist()

    @torch.no_grad()
    def predict_freqs(self, x_t: torch.Tensor, t: torch.Tensor, k: Optional[int] = None) -> List[List[float]]:
        bins_batch = self.predict_bins(x_t, t, k=k)
        L = self.cfg.L
        return [[b / float(L) for b in bins] for bins in bins_batch]

    @torch.no_grad()
    def predict_from_clean(self, x0: torch.Tensor, t_val: int, k: Optional[int] = None, use_mean: bool = True):
        B = x0.size(0)
        t = torch.full((B,), int(t_val), device=self.cfg.device, dtype=torch.long)
        xt = forward_diffuse_eval_mean(x0.to(self.cfg.device), t, self.ALPHAS_CUM) if use_mean \
             else forward_diffuse(x0.to(self.cfg.device), t, self.ALPHAS_CUM)
        return self.predict_freqs(xt, t, k=k)

    def save(self, path: str):
        torch.save({"cfg": vars(self.cfg), "model": self.model.state_dict(), "betas": self.BETAS}, path)

    @classmethod
    def load(cls, path: str, map_location: Optional[str] = None) -> "ExtremeTrain":
        ckpt = torch.load(path, map_location=map_location or ("cuda" if torch.cuda.is_available() else "cpu"))
        obj = cls(ckpt["cfg"])
        obj.model.load_state_dict(ckpt["model"])
        if "betas" in ckpt:
            obj.BETAS = ckpt["betas"]; obj.ALPHAS = 1.0 - obj.BETAS; obj.ALPHAS_CUM = torch.cumprod(obj.ALPHAS, dim=0)
        return obj
