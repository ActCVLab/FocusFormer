import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numbers
import math
from models.detectors.PFA import PFABlock


Conv2d = nn.Conv2d

__all__ = ["TransformerBlock"]

# ---------- helpers ----------

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
    
class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))  # γ
        self.bias   = nn.Parameter(torch.zeros(self.normalized_shape)) # β

    def forward(self, x):
        mu  = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        xhat = (x - mu) / torch.sqrt(var + self.eps)
        return xhat * self.weight + self.bias

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))  # γ

    def forward(self, x):
        var  = x.var(dim=-1, keepdim=True, unbiased=False)
        xhat = x / torch.sqrt(var + self.eps)
        return xhat * self.weight

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type: str = 'WithBias', eps=1e-5):
        super().__init__()
        self.body = WithBias_LayerNorm(dim, eps) if LayerNorm_type != 'BiasFree' \
                    else BiasFree_LayerNorm(dim, eps)

    def forward(self, x):
        B, C, H, W = x.shape
        x3 = x.permute(0,2,3,1).contiguous().view(B, H*W, C)   # NCHW -> (N,HW,C)
        y3 = self.body(x3)
        y  = y3.view(B, H, W, C).permute(0,3,1,2).contiguous() # (N,HW,C) -> NCHW
        return y




from dataclasses import dataclass
@dataclass
class SIEM:
    scale: int = 2
    depth_ratio_357: tuple = (2, 3, 5)
    gate_share: "float | str" = "match"
    dilate: int = 2

def _make_divisible(v, divisor=4):
    return int(math.ceil(v / divisor) * divisor)

def compute_dgff_splits(hidden: int, cfg: SIEM):
    s2 = cfg.scale * cfg.scale
    assert (2 * hidden) % s2 == 0, f"2*hidden 必须能被 scale^2 整除以用于 PixelShuffle(r)（r^2 约束）"
    total_ps = (2 * hidden) // s2                  
    if cfg.gate_share == "match":
        ch_gate = total_ps // 2
        ch_depth = total_ps - ch_gate
    else:
        ch_gate = int(round(total_ps * float(cfg.gate_share)))
        ch_gate = max(1, min(total_ps - 1, ch_gate))
        ch_depth = total_ps - ch_gate

    r3, r5, r7 = cfg.depth_ratio_357
    rsum = max(1, (r3 + r5 + r7))
    c3 = (ch_depth * r3) // rsum
    c5 = (ch_depth * r5) // rsum
    c7 = ch_depth - c3 - c5  
    return {
        "total_ps": total_ps,
        "gate": ch_gate,
        "depth_total": ch_depth,
        "depth_3": c3, "depth_5": c5, "depth_7": c7,
        "need_align": (ch_gate != ch_depth)  
    }


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False, cfg: SIEM = SIEM(),
                 use_fp32_gate: bool = False, use_reflect_pad: bool = True):
        super().__init__()
        self.cfg = cfg
        self.use_fp32_gate = use_fp32_gate
        padmode = 'reflect' if use_reflect_pad else 'zeros'

        hidden = _make_divisible(dim * ffn_expansion_factor, divisor=4)
        self.hidden = hidden
        self.project_in = Conv2d(dim, hidden * 2, kernel_size=1, bias=bias)
        self.p_shuffle   = nn.PixelShuffle(cfg.scale)
        self.p_unshuffle = nn.PixelUnshuffle(cfg.scale)

        plan = compute_dgff_splits(hidden, cfg)
        self.plan = plan
        c3, c5, c7 = plan["depth_3"], plan["depth_5"], plan["depth_7"]
        cd = plan["depth_total"]
        cg = plan["gate"]

        # --- 深度分支： ---
        self.dw3 = Conv2d(c3, c3, 3, padding=1, groups=c3, bias=bias, padding_mode=padmode) if c3>0 else None
        self.dw5 = Conv2d(c5, c5, 5, padding=2, groups=c5, bias=bias, padding_mode=padmode) if c5>0 else None
        self.dw7 = Conv2d(c7, c7, 7, padding=3, groups=c7, bias=bias, padding_mode=padmode) if c7>0 else None
        self.depth_fuse = Conv2d(cd, cd, kernel_size=1, bias=bias)

        # --- 门控分支： ---
        self.dw_dil = Conv2d(cg, cg, 3, padding=cfg.dilate, dilation=cfg.dilate,
                             groups=cg, bias=bias, padding_mode=padmode)

        self.gate_align = (Conv2d(cg, cd, 1, bias=False) if plan["need_align"] else nn.Identity())

        self.project_out = Conv2d(hidden, dim, kernel_size=1, bias=bias)

        self._idx = (c3, c3 + c5, cd)  

    def forward(self, x):
        z = self.project_in(x)                       # (B, 2H, H, W)
        z = self.p_shuffle(z)                        # (B, total_ps, H*s, W*s)

        cd, cg = self.plan["depth_total"], self.plan["gate"]
        z_depth, z_gate = torch.split(z, [cd, cg], dim=1)

        b1, b2, bd = self._idx  # c3, c3+c5, cd
        parts = []
        if b1>0: parts += [self.dw3(z_depth[:, :b1])]
        if b2-b1>0: parts += [self.dw5(z_depth[:, b1:b2])]
        if bd-b2>0: parts += [self.dw7(z_depth[:, b2:bd])]
        depth = self.depth_fuse(torch.cat(parts, dim=1)) if len(parts)>1 else self.depth_fuse(parts[0])

        # 门控分支（可选 FP32 乘法更稳）
        gate = self.dw_dil(z_gate)
        gate = self.gate_align(gate)
        if self.use_fp32_gate:
            with torch.autocast(device_type='cuda', enabled=False):
                out = (F.mish(gate.float()) * depth.float()).to(depth.dtype)
                # out = (F.silu(gate.float()) * depth.float()).to(depth.dtype)
        else:
            out = F.mish(gate) * depth
            # out = F.silu(gate) * depth

        out = self.p_unshuffle(out)                 # (B, hidden, H, W)
        out = self.project_out(out)                 # (B, dim, H, W)
        return out


# ---------- Transformer Block ----------
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, ffn_expansion_factor=2.66, bias=False, layernorm_type: str = 'WithBias'):
        super().__init__()
        assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})"
        self.attn_g = PFABlock(channels=dim, num_heads=num_heads, dhsa_bias=bias,ifBox=True, factor=4,merge='mul')
        self.norm_g = LayerNorm(dim, layernorm_type)
        cfg = SIEM(
            scale=2,                 
            depth_ratio_357=(2,3,5), 
            gate_share='match',     
            dilate=2                 
        )
        self.ffn = FeedForward(dim, ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                               cfg=cfg, use_fp32_gate=True,use_reflect_pad=True)
        self.norm_ff1 = LayerNorm(dim, layernorm_type)
    def forward(self, x, score: torch.Tensor = None):
        x = x + self.attn_g(self.norm_g(x), score=score)   # 只影响 DHSA
        x = x + self.ffn(self.norm_ff1(x))
        return x
