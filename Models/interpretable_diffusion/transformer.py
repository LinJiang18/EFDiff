import math
import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from einops import rearrange, reduce, repeat
from Models.interpretable_diffusion.model_utils import LearnablePositionalEncoding, Conv_MLP,\
                                                       AdaLayerNorm, Transpose, GELU2, series_decomp
from typing import Optional

class TrendBlock(nn.Module):
    """
    Model trend of time series using the polynomial regressor.
    """
    def __init__(self, in_dim, out_dim, in_feat, out_feat, act):
        super(TrendBlock, self).__init__()
        trend_poly = 3
        self.trend = nn.Sequential(
            nn.Conv1d(in_channels=in_dim, out_channels=trend_poly, kernel_size=3, padding=1), # Reduce the time steps into multi-dimensional trend components, each corresponding to a specific time step.
            act,
            Transpose(shape=(1, 2)),
            nn.Conv1d(in_feat, out_feat, 3, stride=1, padding=1)  # Reduce the high-dimensional feature space to a lower-dimensional one.
        )

        lin_space = torch.arange(1, out_dim + 1, 1) / (out_dim + 1)
        self.poly_space = torch.stack([lin_space ** float(p + 1) for p in range(trend_poly)], dim=0)

    def forward(self, input):
        b, c, h = input.shape
        x = self.trend(input).transpose(1, 2)
        trend_vals = torch.matmul(x.transpose(1, 2), self.poly_space.to(x.device))
        trend_vals = trend_vals.transpose(1, 2)
        return trend_vals
    

class MovingBlock(nn.Module):
    """
    Model trend of time series using the moving average.
    """
    def __init__(self, out_dim):
        super(MovingBlock, self).__init__()
        size = max(min(int(out_dim / 4), 24), 4)
        self.decomp = series_decomp(size)

    def forward(self, input):
        b, c, h = input.shape
        x, trend_vals = self.decomp(input)
        return x, trend_vals


import math
from typing import Optional
import torch
import torch.nn as nn

class FourierLayer(nn.Module):
    """
    Seasonality via inverse DFT using selected frequencies.
    - 先用 top-k ∪ ext_bins 选频率（去重且保持矩阵形状）
    - 再分块合成，避免 (B,K,T,D) 巨张量导致 OOM
    """
    def __init__(self, d_model: int, top_k_frequency: int, low_freq: int = 1, factor: int = 1, max_sel: int = 32):
        super().__init__()
        self.d_model = d_model
        self.top_k_frequency = int(top_k_frequency)
        self.factor = factor
        self.low_freq = int(low_freq)
        self.max_sel = int(max_sel)  # 选频上限，防 OOM

    # ----------- 工具：在频率维 gather（保持 (B,K,D)） -----------
    @staticmethod
    def _gather_lastdim(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        x:   (B, Lf, D) 在频率维 Lf 上 gather
        idx: (B, K,  D) 或 (B, K)（会自动扩到 (B,K,D)）
        返回: (B, K,  D)
        """
        B, Lf, D = x.shape
        if idx.dim() == 2:
            idx = idx.unsqueeze(-1).expand(B, idx.size(1), D)
        a = torch.arange(B, device=x.device).view(B, 1, 1).expand_as(idx)
        d = torch.arange(D, device=x.device).view(1, 1, D).expand_as(idx)
        return x[a, idx, d]

    # ----------- 选频率：top-k ∪ ext_bins（去重但保持规则形状） -----------
    def _select_indices(self, x_freq_abs: torch.Tensor, extra_bins: Optional[torch.Tensor]) -> torch.Tensor:
        """
        x_freq_abs: (B, Lf, D)
        extra_bins: (B, M) 或 (B, D, M) 或 None
        返回 combined_idx: (B, K_sel, D)，K_sel <= min(max_sel, K+M)
        """
        B, Lf, D = x_freq_abs.shape
        device = x_freq_abs.device
        K = min(self.top_k_frequency, Lf)

        # 1) 每个 (B,D) 的 top-k 索引
        _, topk_idx = torch.topk(x_freq_abs, k=K, dim=1, largest=True, sorted=True)  # (B, K, D)

        # 2) 无外部 bins：直接截断到 max_sel
        if extra_bins is None:
            return topk_idx[:, :min(self.max_sel, K), :]

        # 3) 规格化 extra_bins -> (B, M, D)
        add = extra_bins.to(device=device, dtype=torch.long)
        if add.dim() == 2:                   # (B, M) -> 每通道共用
            add = add.unsqueeze(1).expand(B, D, -1)  # (B, D, M)
        elif add.dim() == 3 and add.size(1) != D:    # 兜底
            add = add[:, :1, :].expand(B, D, -1)
        add = torch.clamp(add, 0, Lf - 1)            # 越界保护
        add = add.permute(0, 2, 1).contiguous()      # (B, M, D)
        M = add.size(1)

        # 4) 维护 seen，逐列处理 extra（去重但不改变规则形状）
        seen = torch.zeros((B, Lf, D), dtype=torch.bool, device=device)
        seen.scatter_(1, topk_idx, True)  # 标记 top-k 已见

        fill_col = topk_idx[:, :1, :].expand(B, 1, D)  # 占位列（老频率用占位替换，保持形状）
        appended = []
        for j in range(M):
            cand = add[:, j:j+1, :]                      # (B,1,D)
            is_new = ~seen.gather(1, cand)               # (B,1,D)
            new_col = torch.where(is_new, cand, fill_col)
            appended.append(new_col)
            seen.scatter_(1, cand, True)                 # 标记为已见

        extra_block = torch.cat(appended, dim=1) if appended else add.new_zeros((B, 0, D))
        combined_idx = torch.cat([topk_idx, extra_block], dim=1)  # (B, K+M, D)

        # 5) 截断上限
        K_sel = min(self.max_sel, combined_idx.size(1))
        return combined_idx[:, :K_sel, :]

    # ----------- 分块合成：避免 (B,K,T,D) 巨张量 -----------
    @staticmethod
    def _synth_chunked(x_amp: torch.Tensor, x_phase: torch.Tensor, f_sel: torch.Tensor, T: int,
                       chunk_k: int = 8, chunk_bd: int = 4096) -> torch.Tensor:
        """
        x_amp, x_phase, f_sel: (B, K_sel, D)  实数
        返回 y: (B, T, D) = sum_k amp * cos(2π f t + phase)
        """
        B, K, D = x_amp.shape
        device = x_amp.device
        dtype  = x_amp.dtype

        BD = B * D
        amp   = x_amp.permute(0, 2, 1).reshape(BD, K)      # (BD, K)
        phase = x_phase.permute(0, 2, 1).reshape(BD, K)
        f     = f_sel.permute(0, 2, 1).reshape(BD, K)

        t = torch.linspace(0, 1, steps=T, device=device, dtype=dtype)   # (T,)
        two_pi_t = (2.0 * math.pi) * t

        out = torch.zeros(BD, T, device=device, dtype=dtype)

        for bd0 in range(0, BD, chunk_bd):
            bd1 = min(bd0 + chunk_bd, BD)
            for k0 in range(0, K, chunk_k):
                k1 = min(k0 + chunk_k, K)
                a  = amp[bd0:bd1,   k0:k1]               # (bd', k')
                ff = f[bd0:bd1,     k0:k1]               # (bd', k')
                ph = phase[bd0:bd1, k0:k1]               # (bd', k')

                arg = torch.einsum('bk,t->bkt', ff, two_pi_t) + ph.unsqueeze(-1)  # (bd', k', T)
                out[bd0:bd1] += torch.einsum('bk,bkt->bt', a, torch.cos(arg))

        return out.view(B, D, T).permute(0, 2, 1)          # (B, T, D)

    # ----------- 前向：rFFT -> 选频 -> 合成 -----------
    def forward(self, x: torch.Tensor, ext_bins: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, T, D)
        ext_bins: (B, M) 或 (B, D, M)，可选
        返回: (B, T, D)
        """
        B, T, D = x.shape
        device, dtype = x.device, x.dtype

        # 频域
        x_freq = torch.fft.rfft(x, dim=1)                 # (B, Lf_full, D)
        if T % 2 == 0:
            x_freq = x_freq[:, self.low_freq:-1, :]
            f_vec = torch.fft.rfftfreq(T, d=1.0)[self.low_freq:-1]  # (Lf,)
        else:
            x_freq = x_freq[:, self.low_freq:, :]
            f_vec = torch.fft.rfftfreq(T, d=1.0)[self.low_freq:]    # (Lf,)
        Lf = x_freq.size(1)

        # 选频索引 (B, K_sel, D)
        idx_sel = self._select_indices(x_freq.abs(), ext_bins)

        # gather 频谱与频率值
        x_sel = self._gather_lastdim(x_freq, idx_sel)                            # (B, K_sel, D) complex
        f_all = f_vec.to(device=device, dtype=dtype).view(1, Lf, 1).expand(B, Lf, D)
        f_sel = self._gather_lastdim(f_all, idx_sel)                             # (B, K_sel, D) real

        # 幅相
        amp   = x_sel.abs().to(dtype)
        phase = torch.angle(x_sel).to(dtype)

        # 分块合成
        y = self._synth_chunked(amp, phase, f_sel, T,
                                chunk_k=min(8, amp.size(1)),
                                chunk_bd=4096)
        return y



class SeasonBlock(nn.Module):
    """
    Model seasonality of time series using the Fourier series.
    """
    def __init__(self, in_dim, out_dim, factor=1):
        super(SeasonBlock, self).__init__()
        season_poly = factor * min(32, int(out_dim // 2))
        self.season = nn.Conv1d(in_channels=in_dim, out_channels=season_poly, kernel_size=1, padding=0)
        fourier_space = torch.arange(0, out_dim, 1) / out_dim
        p1, p2 = (season_poly // 2, season_poly // 2) if season_poly % 2 == 0 \
            else (season_poly // 2, season_poly // 2 + 1)
        s1 = torch.stack([torch.cos(2 * np.pi * p * fourier_space) for p in range(1, p1 + 1)], dim=0)
        s2 = torch.stack([torch.sin(2 * np.pi * p * fourier_space) for p in range(1, p2 + 1)], dim=0)
        self.poly_space = torch.cat([s1, s2])

    def forward(self, input):
        b, c, h = input.shape
        x = self.season(input)
        season_vals = torch.matmul(x.transpose(1, 2), self.poly_space.to(x.device))
        season_vals = season_vals.transpose(1, 2)
        return season_vals


class FullAttention(nn.Module):
    def __init__(self,
                 n_embd, # the embed dim
                 n_head, # the number of heads
                 attn_pdrop=0.1, # attention dropout prob
                 resid_pdrop=0.1, # residual attention dropout prob
    ):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x, mask=None):
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, T, T)

        att = F.softmax(att, dim=-1) # (B, nh, T, T)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side, (B, T, C)
        att = att.mean(dim=1, keepdim=False) # (B, T, T)

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, att


class CrossAttention(nn.Module):
    def __init__(self,
                 n_embd, # the embed dim
                 condition_embd, # condition dim
                 n_head, # the number of heads
                 attn_pdrop=0.1, # attention dropout prob
                 resid_pdrop=0.1, # residual attention dropout prob
    ):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(condition_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(condition_embd, n_embd)
        
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x, encoder_output, mask=None):
        B, T, C = x.size()
        B, T_E, _ = encoder_output.size()
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(encoder_output).view(B, T_E, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(encoder_output).view(B, T_E, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, T, T)

        att = F.softmax(att, dim=-1) # (B, nh, T, T)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side, (B, T, C)
        att = att.mean(dim=1, keepdim=False) # (B, T, T)

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, att
    

class EncoderBlock(nn.Module):
    """ an unassuming Transformer block """
    def __init__(self,
                 n_embd=1024,
                 n_head=16,
                 attn_pdrop=0.1,
                 resid_pdrop=0.1,
                 mlp_hidden_times=4,
                 activate='GELU'
                 ):
        super().__init__()

        self.ln1 = AdaLayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = FullAttention(
                n_embd=n_embd,
                n_head=n_head,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
            )
        
        assert activate in ['GELU', 'GELU2']
        act = nn.GELU() if activate == 'GELU' else GELU2()

        self.mlp = nn.Sequential(
                nn.Linear(n_embd, mlp_hidden_times * n_embd),
                act,
                nn.Linear(mlp_hidden_times * n_embd, n_embd),
                nn.Dropout(resid_pdrop),
            )
        
    def forward(self, x, timestep, mask=None, label_emb=None):
        a, att = self.attn(self.ln1(x, timestep, label_emb), mask=mask)  # AdaLayerNorm: LayerNorm with Condition t
        x = x + a
        x = x + self.mlp(self.ln2(x))   # only one really use encoder_output
        return x, att


class Encoder(nn.Module):
    def __init__(
        self,
        n_layer=14,
        n_embd=1024,
        n_head=16,
        attn_pdrop=0.,
        resid_pdrop=0.,
        mlp_hidden_times=4,
        block_activate='GELU',
    ):
        super().__init__()

        self.blocks = nn.Sequential(*[EncoderBlock(
                n_embd=n_embd,
                n_head=n_head,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
                mlp_hidden_times=mlp_hidden_times,
                activate=block_activate,
        ) for _ in range(n_layer)])

    def forward(self, input, t, padding_masks=None, label_emb=None):
        x = input
        for block_idx in range(len(self.blocks)):
            x, _ = self.blocks[block_idx](x, t, mask=padding_masks, label_emb=label_emb)
        return x


class DecoderBlock(nn.Module):
    """ an unassuming Transformer block """
    def __init__(self,
                 n_channel,
                 n_feat,
                 top_k_frequency,
                 n_embd=1024,
                 n_head=16,
                 attn_pdrop=0.1,
                 resid_pdrop=0.1,
                 mlp_hidden_times=4,
                 activate='GELU',
                 condition_dim=1024,
                 ):
        super().__init__()
        
        self.ln1 = AdaLayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        self.attn1 = FullAttention(
                n_embd=n_embd,
                n_head=n_head,
                attn_pdrop=attn_pdrop, 
                resid_pdrop=resid_pdrop,
                )
        self.attn2 = CrossAttention(
                n_embd=n_embd,
                condition_embd=condition_dim,
                n_head=n_head,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
                )
        
        self.ln1_1 = AdaLayerNorm(n_embd)

        assert activate in ['GELU', 'GELU2']
        act = nn.GELU() if activate == 'GELU' else GELU2()

        self.trend = TrendBlock(n_channel, n_channel, n_embd, n_feat, act=act)
        # self.decomp = MovingBlock(n_channel)
        self.seasonal = FourierLayer(d_model=n_embd,top_k_frequency=top_k_frequency)
        # self.seasonal = SeasonBlock(n_channel, n_channel)

        self.mlp = nn.Sequential(
            nn.Linear(n_embd, mlp_hidden_times * n_embd),
            act,
            nn.Linear(mlp_hidden_times * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

        self.proj = nn.Conv1d(n_channel, n_channel * 2, 1)
        self.linear = nn.Linear(n_embd, n_feat)

    def forward(self, x, encoder_output, timestep, mask=None, label_emb=None,ext_bins=None):
        a, att = self.attn1(self.ln1(x, timestep, label_emb), mask=mask)
        x = x + a
        a, att = self.attn2(self.ln1_1(x, timestep), encoder_output, mask=mask)
        x = x + a
        x1, x2 = self.proj(x).chunk(2, dim=1)
        trend, season = self.trend(x1), self.seasonal(x2,ext_bins=ext_bins)
        x = x + self.mlp(self.ln2(x))
        m = torch.mean(x, dim=1, keepdim=True)
        return x - m, self.linear(m), trend, season
    

class Decoder(nn.Module):
    def __init__(
        self,
        n_channel,
        n_feat,
        top_k_frequency,
        n_embd=1024,
        n_head=16,
        n_layer=10,
        attn_pdrop=0.1,
        resid_pdrop=0.1,
        mlp_hidden_times=4,
        block_activate='GELU',
        condition_dim=512    
    ):
      super().__init__()
      self.d_model = n_embd
      self.n_feat = n_feat
      self.blocks = nn.Sequential(*[DecoderBlock(
                n_feat=n_feat,
                n_channel=n_channel,
                top_k_frequency=top_k_frequency,
                n_embd=n_embd,
                n_head=n_head,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
                mlp_hidden_times=mlp_hidden_times,
                activate=block_activate,
                condition_dim=condition_dim,
        ) for _ in range(n_layer)])
      
    def forward(self, x, t, enc, padding_masks=None, label_emb=None,ext_bins=None):
        b, c, _ = x.shape
        # att_weights = []
        mean = []
        season = torch.zeros((b, c, self.d_model), device=x.device)
        trend = torch.zeros((b, c, self.n_feat), device=x.device)
        for block_idx in range(len(self.blocks)):
            x, residual_mean, residual_trend, residual_season = \
                self.blocks[block_idx](x, enc, t, mask=padding_masks, label_emb=label_emb,ext_bins=ext_bins)
            season += residual_season
            trend += residual_trend
            mean.append(residual_mean)

        mean = torch.cat(mean, dim=1)
        return x, mean, trend, season


class Transformer(nn.Module):
    def __init__(
        self,
        n_feat,
        n_channel,
        top_k_frequency,
        n_layer_enc=5,
        n_layer_dec=14,
        n_embd=1024,
        n_heads=16,
        attn_pdrop=0.1,
        resid_pdrop=0.1,
        mlp_hidden_times=4,
        block_activate='GELU',
        max_len=2048,
        conv_params=None,
        **kwargs
    ):
        super().__init__()
        self.emb = Conv_MLP(n_feat, n_embd, resid_pdrop=resid_pdrop)
        self.inverse = Conv_MLP(n_embd, n_feat, resid_pdrop=resid_pdrop)

        if conv_params is None or conv_params[0] is None:
            if n_feat < 32 and n_channel < 64:
                kernel_size, padding = 1, 0
            else:
                kernel_size, padding = 5, 2
        else:
            kernel_size, padding = conv_params

        self.combine_s = nn.Conv1d(n_embd, n_feat, kernel_size=kernel_size, stride=1, padding=padding,
                                   padding_mode='circular', bias=False)
        self.combine_m = nn.Conv1d(n_layer_dec, 1, kernel_size=1, stride=1, padding=0,
                                   padding_mode='circular', bias=False)

        self.encoder = Encoder(n_layer_enc, n_embd, n_heads, attn_pdrop, resid_pdrop, mlp_hidden_times, block_activate)
        self.pos_enc = LearnablePositionalEncoding(n_embd, dropout=resid_pdrop, max_len=max_len)

        self.decoder = Decoder(n_channel, n_feat, top_k_frequency, n_embd, n_heads, n_layer_dec, attn_pdrop, resid_pdrop, mlp_hidden_times,
                               block_activate, condition_dim=n_embd)
        self.pos_dec = LearnablePositionalEncoding(n_embd, dropout=resid_pdrop, max_len=max_len)

    def forward(self, input, t, padding_masks=None, return_res=False,ext_bins=None):
        emb = self.emb(input)
        inp_enc = self.pos_enc(emb)
        enc_cond = self.encoder(inp_enc, t, padding_masks=padding_masks)

        inp_dec = self.pos_dec(emb)
        output, mean, trend, season = self.decoder(inp_dec, t, enc_cond, padding_masks=padding_masks,ext_bins=ext_bins)

        res = self.inverse(output)
        res_m = torch.mean(res, dim=1, keepdim=True)
        season_error = self.combine_s(season.transpose(1, 2)).transpose(1, 2) + res - res_m
        trend = self.combine_m(mean) + res_m + trend

        if return_res:
            return trend, self.combine_s(season.transpose(1, 2)).transpose(1, 2), res - res_m

        return trend, season_error


if __name__ == '__main__':
    pass