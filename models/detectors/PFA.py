import torch
import torch.nn as nn
from einops import rearrange
from torch.autograd import Function
import torch.nn.functional as F
import math
import numpy as np
import cv2


class DWT_Function(Function):
    @staticmethod
    def forward(ctx, x, w_ll, w_lh, w_hl, w_hh):
        ctx.save_for_backward(w_ll, w_lh, w_hl, w_hh)
        ctx.shape = x.shape

        dim = x.shape[1]
        x_ll = torch.nn.functional.conv2d(x, w_ll.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_lh = torch.nn.functional.conv2d(x, w_lh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hl = torch.nn.functional.conv2d(x, w_hl.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hh = torch.nn.functional.conv2d(x, w_hh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            w_ll, w_lh, w_hl, w_hh = ctx.saved_tensors
            _, C, _, _ = ctx.shape
            dx_ll, dx_lh, dx_hl, dx_hh = dx[:, :C], dx[:, C:C * 2], dx[:, C * 2:C * 3], dx[:, C * 3:]
            dtype = dx.dtype
            device = dx.device
            w_ll = w_ll.to(device=device, dtype=dtype)
            w_lh = w_lh.to(device=device, dtype=dtype)
            w_hl = w_hl.to(device=device, dtype=dtype)
            w_hh = w_hh.to(device=device, dtype=dtype)
  
            dx_x_ll = torch.nn.functional.conv_transpose2d(dx_ll, w_ll.expand(C, -1, -1, -1) * 4, stride=2, groups=C)
            dx_x_lh = torch.nn.functional.conv_transpose2d(dx_lh, w_lh.expand(C, -1, -1, -1) * 4, stride=2, groups=C)
            dx_x_hl = torch.nn.functional.conv_transpose2d(dx_hl, w_hl.expand(C, -1, -1, -1) * 4, stride=2, groups=C)
            dx_x_hh = torch.nn.functional.conv_transpose2d(dx_hh, w_hh.expand(C, -1, -1, -1) * 4, stride=2, groups=C)
            return dx_x_ll + dx_x_lh + dx_x_hl + dx_x_hh, None, None, None, None
        else:
            return dx, None, None, None, None


class DWT_2D(nn.Module):
    def __init__(self):
        super(DWT_2D, self).__init__()
        w_ll = torch.tensor([[[[0.25, 0.25], [0.25, 0.25]]]], dtype=torch.float32, requires_grad=False)
        w_lh = torch.tensor([[[[0.25, 0.25], [-0.25, -0.25]]]], dtype=torch.float32, requires_grad=False)
        w_hl = torch.tensor([[[[0.25, -0.25], [0.25, -0.25]]]], dtype=torch.float32, requires_grad=False)
        w_hh = torch.tensor([[[[0.25, -0.25], [-0.25, 0.25]]]], dtype=torch.float32, requires_grad=False)

        self.register_buffer('w_ll', w_ll)
        self.register_buffer('w_lh', w_lh)
        self.register_buffer('w_hl', w_hl)
        self.register_buffer('w_hh', w_hh)


    def forward(self, x):
        return DWT_Function.apply(x, self.w_ll, self.w_lh, self.w_hl, self.w_hh)
    
class IDWT_Function(Function):
    @staticmethod
    def forward(ctx, x, filters):
        ctx.save_for_backward(filters)
        ctx.shape = x.shape

        _, C, _, _ = x.shape
        w_ll, w_lh, w_hl, w_hh = torch.unbind(filters, dim=0)
        x_ll, x_lh, x_hl, x_hh = x[:, :C // 4], x[:, C // 4:C * 2 // 4], x[:, C * 2 // 4:C * 3 // 4], x[:, C * 3 // 4:]
        x_1_ll = torch.nn.functional.conv_transpose2d(x_ll, w_ll.expand(C // 4, -1, -1, -1), stride=2, groups=C // 4)
        x_1_lh = torch.nn.functional.conv_transpose2d(x_lh, w_lh.expand(C // 4, -1, -1, -1), stride=2, groups=C // 4)
        x_1_hl = torch.nn.functional.conv_transpose2d(x_hl, w_hl.expand(C // 4, -1, -1, -1), stride=2, groups=C // 4)
        x_1_hh = torch.nn.functional.conv_transpose2d(x_hh, w_hh.expand(C // 4, -1, -1, -1), stride=2, groups=C // 4)
        return x_1_ll + x_1_lh + x_1_hl + x_1_hh

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            (filters,) = ctx.saved_tensors
            dtype = dx.dtype
            _, C, _, _ = ctx.shape
            C //= 4

            w_ll, w_lh, w_hl, w_hh = torch.unbind(filters, dim=0)
            w_ll = w_ll.to(dtype).unsqueeze(1).expand(C, -1, -1, -1) / 4
            w_lh = w_lh.to(dtype).unsqueeze(1).expand(C, -1, -1, -1) / 4
            w_hl = w_hl.to(dtype).unsqueeze(1).expand(C, -1, -1, -1) / 4
            w_hh = w_hh.to(dtype).unsqueeze(1).expand(C, -1, -1, -1) / 4
            x_ll = torch.nn.functional.conv2d(dx, w_ll, stride=2, groups=C)
            x_lh = torch.nn.functional.conv2d(dx, w_lh, stride=2, groups=C)
            x_hl = torch.nn.functional.conv2d(dx, w_hl, stride=2, groups=C)
            x_hh = torch.nn.functional.conv2d(dx, w_hh, stride=2, groups=C)
            dx = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return dx, None


class IDWT_2D(nn.Module):
    def __init__(self):
        super(IDWT_2D, self).__init__()
        w_ll = torch.tensor([[[[1, 1], [1, 1]]]], dtype=torch.float32, requires_grad=False)
        w_lh = torch.tensor([[[[1, 1], [-1, -1]]]], dtype=torch.float32, requires_grad=False)
        w_hl = torch.tensor([[[[1, -1], [1, -1]]]], dtype=torch.float32, requires_grad=False)
        w_hh = torch.tensor([[[[1, -1], [-1, 1]]]], dtype=torch.float32, requires_grad=False)

        filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0)
        self.register_buffer('filters', filters)
        self.filters = filters

    def forward(self, x):
        return IDWT_Function.apply(x, self.filters)
    

class MCDR(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, attn_drop: float = 0.,
                 gate_layer: str = "sigmoid", use_film: bool = True):
        super().__init__()
        assert dim % num_heads == 0, f"dim({dim}) must be divisible by num_heads({num_heads})"
        self.dim = dim
        self.h   = num_heads
        self.dk  = dim // num_heads

        self.dwt  = DWT_2D()
        self.idwt = IDWT_2D()

        self.fuse_k   = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.v_reduce = nn.Conv2d(4 * dim, dim, kernel_size=1, groups=4, bias=True)

        self.Wq = nn.Linear(dim, self.h * self.dk, bias=False)   # 共享 Q
        self.Wk = nn.Linear(dim, self.dk,           bias=False)  # 单套 K（共享）
        self.Wv = nn.Linear(dim, self.dk,           bias=False)  # 单套 V（共享）
        self.Wo = nn.Linear(self.h * self.dk, dim,  bias=False)  # 共享输出投影

        self.softmax      = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(attn_drop)
        self.proj_dropout = nn.Dropout(attn_drop)

        self.use_qk_layernorm = True
        if self.use_qk_layernorm:
            self.qk_ln = nn.LayerNorm(self.dk, elementwise_affine=True)
        self.center_logits = True

        self.use_film = use_film
        if self.use_film:
            self.film_scale = nn.Parameter(torch.ones(4, dim, 1, 1))   # [LL,LH,HL,HH]
            self.film_bias  = nn.Parameter(torch.zeros(4, dim, 1, 1))

        self.sa_dw3d3 = nn.Conv2d(self.dim, self.dim, 3, padding=3, dilation=3,
                                  groups=self.dim, bias=False)
        self.sa_act   = nn.SiLU(inplace=True)
        self.sa_conv1 = nn.Conv2d(dim, 1,   kernel_size=1, bias=True)  # PW: C→1
        self.sa_sig   = nn.Sigmoid()
        nn.init.constant_(self.sa_conv1.bias, 1.0)   
        self.g_scale = nn.Parameter(torch.zeros(1))

    # helpers
    def _to_seq(self, x):  # (B,C,H,W) -> (B,L,C)
        return rearrange(x, "b c h w -> b (h w) c")

    def _to_feat(self, x, h, w):  # (B,L,C) -> (B,C,H,W)
        return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)

    def _apply_film(self, band, idx):
        if not self.use_film:
            return band
        return band * self.film_scale[idx] + self.film_bias[idx]

    def _mhca(self, q_seq, k_seq, v_seq):
        """
        q_seq,k_seq,v_seq: (B,L,C)
        returns: (B,L,C)
        """
        B, L, C = q_seq.shape
        Q = self.Wq(q_seq).view(B, L, self.h, self.dk).permute(0, 2, 1, 3)  # (B,h,L,dk)
        K = self.Wk(k_seq).unsqueeze(1)                                     # (B,1,L,dk)
        V = self.Wv(v_seq).unsqueeze(1)                                     # (B,1,L,dk)
        if self.use_qk_layernorm:
             Q = self.qk_ln(Q)
             K = self.qk_ln(K)

        attn = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.dk)    # (B,h,L,L)
        if self.center_logits:
            attn = attn - attn.mean(dim=-1, keepdim=True)                   
        attn = self.softmax(attn)
        attn = self.attn_dropout(attn)

        ctx  = torch.matmul(attn, V)                                        # (B,h,L,dk)
        out  = ctx.permute(0, 2, 1, 3).reshape(B, L, self.h * self.dk)      # (B,L,C)
        out  = self.proj_dropout(self.Wo(out))                              # (B,L,C)
        return out

    def forward(self, x: torch.Tensor, score: torch.Tensor = None) -> torch.Tensor:
        B, C, H, W = x.shape

        x_dwt = self.dwt(x)                     # (B,4C,H/2,W/2)
        ll, lh, hl, hh = torch.split(x_dwt, C, dim=1)
        h2, w2 = ll.shape[2], ll.shape[3]


        k_base = self.fuse_k(ll) 
        v_base = self.v_reduce(torch.cat([ll, lh, hl, hh], dim=1))  # (B,C,h2,w2)
        k_seq  = self._to_seq(k_base)                           # (B,L,C)
        v_seq  = self._to_seq(v_base)                           # (B,L,C)

        ll_q = self._to_seq(self._apply_film(ll, 0))
        lh_q = self._to_seq(self._apply_film(lh, 1))
        hl_q = self._to_seq(self._apply_film(hl, 2))
        hh_q = self._to_seq(self._apply_film(hh, 3))

        out_ll = self._mhca(ll_q, k_seq, v_seq)
        out_lh = self._mhca(lh_q, k_seq, v_seq)
        out_hl = self._mhca(hl_q, k_seq, v_seq)
        out_hh = self._mhca(hh_q, k_seq, v_seq)


        out_ll = self._to_feat(out_ll, h2, w2)
        out_lh = self._to_feat(out_lh, h2, w2)
        out_hl = self._to_feat(out_hl, h2, w2)
        out_hh = self._to_feat(out_hh, h2, w2)
        feat   = self.idwt(torch.cat([out_ll, out_lh, out_hl, out_hh], dim=1))  # (B,C,H,W)

        gate = self.sa_sig(self.sa_conv1(self.sa_act(self.sa_dw3d3(feat))))      # (B,1,H,W)
        y = x + self.g_scale * (feat * gate)
        return y


class CFPM(nn.Module):
    def __init__(self, dim, num_heads, bias, ifBox=True, factor=None, merge='mul',
                 q_fg: float = 0.97, q_bg: float = 0.03,
                 min_k: int = 64, max_k_ratio: float = 0.25):
        super().__init__()
        self.ifBox = ifBox
        self.num_heads = num_heads
        self.factor = factor if factor is not None else num_heads

        self.q_fg = q_fg
        self.q_bg = q_bg
        self.min_k = int(min_k)
        self.max_k_ratio = float(max_k_ratio)

        self.qkv = nn.Conv2d(dim, dim * 5, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 5, dim * 5, kernel_size=3, stride=1, padding=1,
                                    groups=dim * 5, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        nn.init.zeros_(self.project_out.weight)
        if self.project_out.bias is not None:
            nn.init.zeros_(self.project_out.bias)

        self.vis_out_fg = None   
        self.vis_out_bg = None   
        self.vis_w_fg   = None   
    
    @staticmethod
    def _channel_self_attn(q, k, v):
        B, C, N = q.shape
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = torch.bmm(q, k.transpose(1, 2)) / (C ** 0.5)   # (B,C,C)
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(attn, v)                               # (B,C,N)
        return out

    @torch.no_grad()
    def build_fg_bg_mask_quantile(
        self,
        score: torch.Tensor,
        q_fg: float = None,
        q_bg: float = None,
        min_ratio: float = 0.001,
        assume_logit: bool = True,   
        eps_margin: float = 1e-6,    
    ):

        if q_fg is None: q_fg = getattr(self, "q_fg", 0.97)
        if q_bg is None: q_bg = getattr(self, "q_bg", 0.03)

        s = score
        if s.dtype not in (torch.float32, torch.float64):
            s = s.float()
        s = torch.nan_to_num(s, nan=0.0, posinf=1e4, neginf=-1e4)

        if assume_logit:
            s = torch.sigmoid(s)
        s = s.clamp_(0.0, 1.0)

        B, _, H, W = s.shape
        N = H * W
        min_keep = max(1, int(min_ratio * N))  

        s_flat = s.view(B, -1)  # (B,N)

        th_fg = torch.quantile(s_flat, q_fg, dim=1, keepdim=True)   # (B,1)
        th_bg = torch.quantile(s_flat, q_bg, dim=1, keepdim=True)   # (B,1)
        th_bg = torch.minimum(th_bg, th_fg - eps_margin)

        cand_fg = (s_flat >= th_fg)   # (B,N) bool
        cand_bg = (s_flat <= th_bg)   # (B,N) bool

        idx_high = torch.argsort(s_flat, dim=1, descending=True)  
        idx_low  = torch.argsort(s_flat, dim=1, descending=False)  

        need_fg = (min_keep - cand_fg.sum(dim=1)).clamp_min(0)   # (B,)
        need_bg = (min_keep - cand_bg.sum(dim=1)).clamp_min(0)   # (B,)

        for b in torch.nonzero(need_fg, as_tuple=False).squeeze(1).tolist():
            k = int(need_fg[b].item())
            cand_fg[b, idx_high[b, :k]] = True

        for b in torch.nonzero(need_bg, as_tuple=False).squeeze(1).tolist():
            k = int(need_bg[b].item())
            cand_bg[b, idx_low[b, :k]] = True

        overlap = cand_fg & cand_bg
        if overlap.any():
            cand_bg[overlap] = False

        mask_fg = cand_fg.view(B, 1, H, W)
        mask_bg = cand_bg.view(B, 1, H, W)
        return mask_fg, mask_bg




    def forward(self, x, score: torch.Tensor):
        
        if score is None:
            raise ValueError("Attention_histogram 需要 score=(B,1,H,W) 作为引导")
        score = score.float()
        score = torch.nan_to_num(score, nan=0.0, posinf=1.0, neginf=0.0)

        B, C, H, W = x.shape
        if score.shape[-2:] != (H, W):
            score = F.interpolate(score, size=(H, W), mode='bilinear', align_corners=False)
        s = score.detach()

        if (s.min() < 0) or (s.max() > 1):
            s = torch.sigmoid(s)
        s = torch.nan_to_num(s, nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)


        with torch.no_grad():
            mask_fg, mask_bg = self.build_fg_bg_mask_quantile(s)   # (B,1,H,W) bool
        w_fg = mask_fg.view(B, 1, H * W)
        w_bg = mask_bg.view(B, 1, H * W)

        qkv = self.qkv_dwconv(self.qkv(x))                 # (B,5C,H,W)
        q1, k1, q2, k2, v = qkv.chunk(5, dim=1)
        q1f = q1.view(B, C, -1); k1f = k1.view(B, C, -1)
        q2f = q2.view(B, C, -1); k2f = k2.view(B, C, -1)
        vf  = v.view(B, C, -1)                             

        out_fg_full = vf.new_zeros(B, C, H * W)
        out_bg_full = vf.new_zeros(B, C, H * W)

        for bi in range(B):
            idx_fg = torch.nonzero(w_fg[bi, 0], as_tuple=False).squeeze(1)
            if idx_fg.numel() > 0:
                y_fg = self._channel_self_attn(q1f[bi, :, idx_fg][None],
                                               k1f[bi, :, idx_fg][None],
                                               vf [bi, :, idx_fg][None]).squeeze(0)  # (C,N_fg)
                out_fg_full[bi].index_copy_(1, idx_fg, y_fg)

            idx_bg = torch.nonzero(w_bg[bi, 0], as_tuple=False).squeeze(1)
            if idx_bg.numel() > 0:
                y_bg = self._channel_self_attn(q2f[bi, :, idx_bg][None],
                                               k2f[bi, :, idx_bg][None],
                                               vf [bi, :, idx_bg][None]).squeeze(0)  # (C,N_bg)
                out_bg_full[bi].index_copy_(1, idx_bg, y_bg)

        out_fg = out_fg_full.view(B, C, H, W)
        out_bg = out_bg_full.view(B, C, H, W)

        eps = 1e-6
        fg_area = mask_fg.sum(dim=(2, 3), keepdim=True).clamp_min(eps)    # (B,1,1,1)
        bg_area = mask_bg.sum(dim=(2, 3), keepdim=True).clamp_min(eps)
        fg_avg  = (out_fg * mask_fg).sum(dim=(2, 3), keepdim=True) / fg_area  # (B,C,1,1)
        bg_avg  = (out_bg * mask_bg).sum(dim=(2, 3), keepdim=True) / bg_area

        g_fg = torch.sigmoid(fg_avg)        
        g_bg = torch.sigmoid(bg_avg)        

        feat_fg = g_fg * x
        feat_bg = (1.0 - g_bg) * x
        attn_feat = feat_fg + feat_bg

        out = x + self.project_out(attn_feat)


        return out


class PFABlock(nn.Module):
    def __init__(self,
                 channels: int,
                 num_heads: int = 8, dhsa_bias: bool = True, ifBox: bool = True,
                 factor: int = None, merge: str = 'mul',
                 sum_scale: float = 1.0,
                 ):
        super().__init__()
        self.register_buffer('sum_scale', torch.tensor(float(sum_scale)))

        self.mcdr = MCDR(dim=channels, gate_layer="sigmoid")
        self.cfpm = CFPM(dim=channels, num_heads=num_heads,
                                        bias=dhsa_bias, ifBox=ifBox,
                                        factor=factor, merge=merge)

    def forward(self, x, score: torch.Tensor = None):
        sa_x = self.cfpm(x, score) 
        ca_x = self.mcdr(sa_x)
        return ca_x