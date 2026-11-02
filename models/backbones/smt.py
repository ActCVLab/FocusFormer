import math, os
from functools import partial
import torch
import torch.nn as nn

try:
    from timm.layers import DropPath, to_2tuple, trunc_normal_
except ImportError:
    from timm.models.layers import DropPath, to_2tuple, trunc_normal_

__all__ = ["SMT", "smt_t", "smt_s", "smt_b", "build_smt", "load_smt_pretrained"]

class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=True)
    def forward(self, x, h, w):
        b, _, c = x.shape
        return self.dwconv(x.transpose(1, 2).reshape(b, c, h, w)).flatten(2).transpose(1, 2)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__(); out_features = out_features or in_features; hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features); self.dwconv = DWConv(hidden_features)
        self.act = act_layer(); self.fc2 = nn.Linear(hidden_features, out_features); self.drop = nn.Dropout(drop)
    def forward(self, x, h, w):
        x = self.fc1(x); x = self.act(x + self.dwconv(x, h, w)); x = self.drop(x)
        return self.drop(self.fc2(x))

class Attention(nn.Module):
    def __init__(self, dim, ca_num_heads=4, sa_num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., ca_attention=1, expand_ratio=2):
        super().__init__(); self.ca_attention = ca_attention; self.dim = dim
        self.ca_num_heads = ca_num_heads; self.sa_num_heads = sa_num_heads
        assert dim % ca_num_heads == 0 and dim % sa_num_heads == 0
        self.act = nn.GELU(); self.proj = nn.Linear(dim, dim); self.proj_drop = nn.Dropout(proj_drop)
        self.split_groups = dim // ca_num_heads
        if ca_attention:
            self.v = nn.Linear(dim, dim, bias=qkv_bias); self.s = nn.Linear(dim, dim, bias=qkv_bias)
            for i in range(ca_num_heads):
                c = dim // ca_num_heads
                setattr(self, f"local_conv_{i + 1}", nn.Conv2d(c, c, 3 + 2 * i, 1, 1 + i, groups=c))
            self.proj0 = nn.Conv2d(dim, dim * expand_ratio, 1, groups=self.split_groups)
            self.bn = nn.BatchNorm2d(dim * expand_ratio); self.proj1 = nn.Conv2d(dim * expand_ratio, dim, 1)
        else:
            head_dim = dim // sa_num_heads; self.scale = qk_scale or head_dim ** -0.5
            self.q = nn.Linear(dim, dim, bias=qkv_bias); self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
            self.attn_drop = nn.Dropout(attn_drop)
            self.local_conv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
    def forward(self, x, h, w):
        b, n, c = x.shape
        if self.ca_attention:
            v = self.v(x); s = self.s(x).reshape(b, h, w, self.ca_num_heads, c // self.ca_num_heads).permute(3, 0, 4, 1, 2)
            parts = []
            for i in range(self.ca_num_heads):
                y = getattr(self, f"local_conv_{i + 1}")(s[i])
                parts.append(y.reshape(b, self.split_groups, -1, h, w))
            s = torch.cat(parts, 2).reshape(b, c, h, w)
            s = self.proj1(self.act(self.bn(self.proj0(s)))).reshape(b, c, n).transpose(1, 2)
            x = s * v
        else:
            q = self.q(x).reshape(b, n, self.sa_num_heads, c // self.sa_num_heads).permute(0, 2, 1, 3)
            kv = self.kv(x).reshape(b, n, 2, self.sa_num_heads, c // self.sa_num_heads).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]; attn = self.attn_drop(((q @ k.transpose(-2, -1)) * self.scale).softmax(-1))
            local = self.local_conv(v.transpose(1, 2).reshape(b, n, c).transpose(1, 2).reshape(b, c, h, w))
            x = (attn @ v).transpose(1, 2).reshape(b, n, c) + local.flatten(2).transpose(1, 2)
        return self.proj_drop(self.proj(x))

class Block(nn.Module):
    def __init__(self, dim, ca_num_heads, sa_num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 use_layerscale=False, layerscale_value=1e-4, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, ca_attention=1, expand_ratio=2):
        super().__init__(); self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, ca_num_heads, sa_num_heads, qkv_bias, qk_scale, attn_drop, drop, ca_attention, expand_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity(); self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.gamma_1 = nn.Parameter(layerscale_value * torch.ones(dim)) if use_layerscale else 1.
        self.gamma_2 = nn.Parameter(layerscale_value * torch.ones(dim)) if use_layerscale else 1.
    def forward(self, x, h, w):
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), h, w))
        return x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x), h, w))

class OverlapPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=3, stride=2, in_chans=3, embed_dim=768):
        super().__init__(); patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, patch_size, stride, (patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, x):
        x = self.proj(x); h, w = x.shape[-2:]
        return self.norm(x.flatten(2).transpose(1, 2)), h, w

class Head(nn.Module):
    def __init__(self, head_conv, dim):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(3, dim, head_conv, 2, padding=3 if head_conv == 7 else 1, bias=False),
                                  nn.BatchNorm2d(dim), nn.ReLU(True), nn.Conv2d(dim, dim, 2, 2))
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        x = self.conv(x); h, w = x.shape[-2:]
        return self.norm(x.flatten(2).transpose(1, 2)), h, w

class SMT(nn.Module):
    def __init__(self, img_size=224, in_chans=3, num_classes=1000, embed_dims=(64, 128, 256, 512),
                 ca_num_heads=(4, 4, 4, -1), sa_num_heads=(-1, -1, 8, 16), mlp_ratios=(4, 4, 4, 2),
                 qkv_bias=True, qk_scale=None, use_layerscale=False, layerscale_value=1e-4,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=(2, 2, 8, 1),
                 ca_attentions=(1, 1, 1, 0), num_stages=4, head_conv=3, expand_ratio=2):
        super().__init__(); self.num_classes = num_classes; self.num_stages = num_stages
        self.embed_dims = tuple(embed_dims); self.depths = tuple(depths)
        dpr = torch.linspace(0, drop_path_rate, sum(depths)).tolist(); cur = 0
        for i in range(num_stages):
            patch = Head(head_conv, embed_dims[i]) if i == 0 else OverlapPatchEmbed(
                img_size=img_size // (2 ** (i + 1)), patch_size=3, stride=2,
                in_chans=embed_dims[i - 1], embed_dim=embed_dims[i])
            blocks = nn.ModuleList([Block(embed_dims[i], ca_num_heads[i], sa_num_heads[i], mlp_ratios[i],
                                          qkv_bias, qk_scale, use_layerscale, layerscale_value, drop_rate,
                                          attn_drop_rate, dpr[cur + j], norm_layer=norm_layer,
                                          ca_attention=0 if i == 2 and j % 2 else ca_attentions[i],
                                          expand_ratio=expand_ratio) for j in range(depths[i])])
            setattr(self, f"patch_embed{i + 1}", patch); setattr(self, f"block{i + 1}", blocks)
            setattr(self, f"norm{i + 1}", norm_layer(embed_dims[i])); cur += depths[i]
        self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)
    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels // m.groups
            nn.init.normal_(m.weight, 0, math.sqrt(2. / fan_out))
            if m.bias is not None: nn.init.zeros_(m.bias)
    def forward_features(self, x):
        b = x.shape[0]; outs = []
        for i in range(self.num_stages):
            x, h, w = getattr(self, f"patch_embed{i + 1}")(x)
            for blk in getattr(self, f"block{i + 1}"): x = blk(x, h, w)
            x = getattr(self, f"norm{i + 1}")(x)
            x = x.transpose(1, 2).reshape(b, self.embed_dims[i], h, w).contiguous()
            outs.append(x)
        return outs
    def forward(self, x): return self.forward_features(x)

def _unwrap_state_dict(ckpt):
    if not isinstance(ckpt, dict): return ckpt
    for key in ("model", "state_dict", "net"):
        if key in ckpt and isinstance(ckpt[key], dict): return ckpt[key]
    return ckpt

def load_smt_pretrained(model, path, model_name="SMT"):
    if not path: return model
    if not os.path.isfile(path): raise FileNotFoundError(f"{model_name}预训练权重不存在: {path}")
    try: ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError: ckpt = torch.load(path, map_location="cpu")
    state = _unwrap_state_dict(ckpt); cleaned = {}
    for k, v in state.items():
        for prefix in ("module.", "backbone.", "model."):
            if k.startswith(prefix): k = k[len(prefix):]
        cleaned[k] = v
    current = model.state_dict()
    matched = {k: v for k, v in cleaned.items() if k in current and current[k].shape == v.shape}
    msg = model.load_state_dict(matched, strict=False)
    print(f"[{model_name}] pretrained={path}, matched={len(matched)}/{len(current)}, "
          f"missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")
    return model

_COMMON = dict(embed_dims=(64, 128, 256, 512), ca_num_heads=(4, 4, 4, -1),
               sa_num_heads=(-1, -1, 8, 16), qkv_bias=True,
               ca_attentions=(1, 1, 1, 0), expand_ratio=2)

def smt_t(pretrained_path=None, **kwargs):
    kwargs.setdefault("drop_path_rate", 0.1)
    model = SMT(mlp_ratios=(4, 4, 4, 2), depths=(2, 2, 8, 1), head_conv=3, **_COMMON, **kwargs)
    return load_smt_pretrained(model, pretrained_path, "SMT-T")

def smt_s(pretrained_path=None, **kwargs):
    kwargs.setdefault("drop_path_rate", 0.2)
    model = SMT(mlp_ratios=(4, 4, 4, 2), depths=(3, 4, 18, 2), head_conv=3, **_COMMON, **kwargs)
    return load_smt_pretrained(model, pretrained_path, "SMT-S")

def smt_b(pretrained_path=None, **kwargs):
    kwargs.setdefault("drop_path_rate", 0.3)
    model = SMT(mlp_ratios=(8, 6, 4, 2), depths=(4, 6, 28, 2), head_conv=7, **_COMMON, **kwargs)
    return load_smt_pretrained(model, pretrained_path, "SMT-B")

def build_smt(name="smt_t", pretrained_path=None, **kwargs):
    name = name.lower().replace("-", "_")
    builders = {"smt_t": smt_t, "tiny": smt_t, "t": smt_t,
                "smt_s": smt_s, "small": smt_s, "s": smt_s,
                "smt_b": smt_b, "base": smt_b, "b": smt_b}
    if name not in builders:
        raise ValueError(f"不支持的SMT版本: {name}，可选 smt_t/smt_s/smt_b")
    return builders[name](pretrained_path=pretrained_path, **kwargs)