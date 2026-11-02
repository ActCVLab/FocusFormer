from models.backbones.smt import smt_s
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.detectors.Decoder import TransformerBlock

class CBR_SID(nn.Module):
    def __init__(self, in_c, out_c, k=3, p=1, s=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()
    def forward(self, x): return self.act(self.bn(self.conv(x)))

class DecoupleLayer_FG_BG(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.fg = nn.Sequential(CBR_SID(in_c, 512), CBR_SID(512, out_c), CBR_SID(out_c, out_c, k=1, p=0))
        self.bg = nn.Sequential(CBR_SID(in_c, 512), CBR_SID(512, out_c), CBR_SID(out_c, out_c, k=1, p=0))
    def forward(self, x): return self.fg(x), self.bg(x)

class AuxHeadCoarse_FG_BG(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.fg = nn.Sequential(CBR_SID(in_c, in_c), nn.Conv2d(in_c, 1, 1), nn.Sigmoid())
        self.bg = nn.Sequential(CBR_SID(in_c, in_c), nn.Conv2d(in_c, 1, 1), nn.Sigmoid())
    def forward(self, f_fg, f_bg):
        return self.fg(f_fg), self.bg(f_bg)   # (B,1,H,W), (B,1,H,W)

class SIDBranchFGbg(nn.Module):
    def __init__(self, in_c, out_c=None, smooth_kernel=3):
        super().__init__()
        out_c = in_c if out_c is None else out_c
        self.decouple = DecoupleLayer_FG_BG(in_c, out_c)
        self.aux = AuxHeadCoarse_FG_BG(out_c)
        self.k = smooth_kernel
    def forward(self, x):
        f_fg, f_bg = self.decouple(x)
        p_fg, p_bg = self.aux(f_fg, f_bg)                         
        score = (p_fg * (1.0 - p_bg)).clamp(0., 1.)               
        if self.k and self.k > 1:                                 
            score = F.avg_pool2d(score, self.k, 1, self.k // 2)
        return score.detach(), p_fg, p_bg



class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
    
class BlockStack(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
    def forward(self, x, score=None):
        for blk in self.blocks:
            x = blk(x, score=score)
        return x
    
@torch.no_grad()
def build_score_from_logit(prev_logit, target_hw, smooth=True):
    s = torch.sigmoid(prev_logit)                             # (B,1,Hp,Wp)
    s = F.interpolate(s, size=target_hw, mode='bilinear', align_corners=False)
    return s.detach()


class VSSNet(nn.Module):
    def __init__(self, in_channels=128, reduction=4, bias=False, act=nn.ReLU(inplace=True), backbone_pretrained='./pretrained/smt_small.pth'):
        super().__init__()
        self.backbone = smt_s(pretrained_path=backbone_pretrained, drop_path_rate=0.1)


        self.sid_fuse_reduce = nn.Conv2d(in_channels*4, in_channels, kernel_size=1, bias=False)
        self.sid_ms = SIDBranchFGbg(in_channels)   


        # decoder blocks
        self.decoder_level5 = BlockStack([TransformerBlock(in_channels)])
        self.decoder_level4 = BlockStack([TransformerBlock(in_channels*2)])
        self.decoder_level3 =  BlockStack([TransformerBlock(in_channels*3)])
        self.decoder_level2 = BlockStack([TransformerBlock(in_channels*4)])

        self.align_corners = False

        # heads
        self.out_IE_level5 = nn.Conv2d(in_channels, 1, 1)
        self.out_IE_level4 = nn.Conv2d(in_channels, 1, 1)
        self.out_IE_level3 = nn.Conv2d(in_channels, 1, 1)
        self.out_IE_level2 = nn.Conv2d(in_channels, 1, 1)  

        # Smooth layers
        self.smooth5 = BasicConv2d(1 * in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.smooth4 = BasicConv2d(2 * in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.smooth3 = BasicConv2d(3 * in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.smooth2 = BasicConv2d(4 * in_channels, in_channels, kernel_size=3, stride=1, padding=1)

        # Top & lateral layers 
        self.toplayer  = BasicConv2d(512, in_channels, kernel_size=3, stride=1, padding=1)
        self.latlayer4 = BasicConv2d(256, in_channels, kernel_size=3, stride=1, padding=1)
        self.latlayer3 = BasicConv2d(128, in_channels, kernel_size=3, stride=1, padding=1)
        self.latlayer2 = BasicConv2d( 64, in_channels, kernel_size=3, stride=1, padding=1)

       
    def _upsample_like(self, x, ref):
        return F.interpolate(x, size=ref.shape[2:], mode='bilinear', align_corners=self.align_corners)


    def forward(self, x):
    # ---------- backbone ----------
        features = self.backbone(x)
        x2, x3, x4, x5 = features

        stage_loss = []

        x5_t = self.toplayer(x5)    # (B,C,H5,W5)
        x4_t = self.latlayer4(x4)   # (B,C,H4,W4)
        x3_t = self.latlayer3(x3)   # (B,C,H3,W3)
        x2_t = self.latlayer2(x2)   # (B,C,H2,W2)

        H5W5 = x5_t.shape[-2:]

        x4_d = F.adaptive_avg_pool2d(x4_t, H5W5)
        x3_d = F.adaptive_avg_pool2d(x3_t, H5W5)
        x2_d = F.adaptive_avg_pool2d(x2_t, H5W5)

        sid_in = torch.cat((x2_d, x3_d, x4_d, x5_t), dim=1)          # (B,4C,H5,W5)
        sid_in = self.sid_fuse_reduce(sid_in)                        # (B,C,H5,W5)

        score5, p_fg5, p_bg5 = self.sid_ms(sid_in)


        x5_dec = self.decoder_level5(x5_t, score=score5)             # (B,C,H5,W5)

        sem5   = self.smooth5(x5_dec)
        logit5 = self.out_IE_level5(sem5)                            # (B,1,H5,W5)
        pred5  = F.interpolate(logit5, size=x.shape[-2:], mode='bilinear', align_corners=False)
        stage_loss.append(pred5)

        x4_cat = torch.cat((x4_t, F.interpolate(x5_dec, size=x4_t.shape[-2:], mode='bilinear', align_corners=False)), dim=1) # (B,2C,H4,W4)
        score4 = build_score_from_logit(logit5, target_hw=x4_cat.shape[-2:])
        x4_dec = self.decoder_level4(x4_cat, score=score4)

        sem4   = self.smooth4(x4_dec)
        logit4 = self.out_IE_level4(sem4)                            # (B,1,H4,W4)
        pred4  = F.interpolate(logit4, size=x.shape[-2:], mode='bilinear', align_corners=False)
        stage_loss.append(pred4)

        x3_cat = torch.cat((x3_t, F.interpolate(x4_dec, size=x3_t.shape[-2:], mode='bilinear', align_corners=False)), dim=1) # (B,3C,H3,W3)
        score3 = build_score_from_logit(logit4, target_hw=x3_cat.shape[-2:])
        x3_dec = self.decoder_level3(x3_cat, score=score3)

        sem3   = self.smooth3(x3_dec)
        logit3 = self.out_IE_level3(sem3)                            # (B,1,H3,W3)
        pred3  = F.interpolate(logit3, size=x.shape[-2:], mode='bilinear', align_corners=False)
        stage_loss.append(pred3)

        x2_cat = torch.cat((x2_t, F.interpolate(x3_dec, size=x2_t.shape[-2:], mode='bilinear', align_corners=False)), dim=1) # (B,4C,H2,W2)
        score2 = build_score_from_logit(logit3, target_hw=x2_cat.shape[-2:])
        x2_dec = self.decoder_level2(x2_cat, score=score2)

        sem2   = self.smooth2(x2_dec)
        logit2 = self.out_IE_level2(sem2)                            # (B,1,H2,W2)
        prediction = F.interpolate(logit2, size=x.shape[-2:], mode='bilinear', align_corners=False)

        stage_loss.append(prediction)
        return stage_loss, prediction, (p_fg5, p_bg5)
