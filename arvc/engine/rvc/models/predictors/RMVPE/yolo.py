import torch
import torch.nn as nn
import torch.nn.functional as F

def autopad(k, p=None):
    if p is None: p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class Conv(nn.Module):
    def __init__(
        self, 
        c1, 
        c2, 
        k=1, 
        s=1, 
        p=None, 
        g=1, 
        act=True
    ):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2) 
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(
            self.bn(
                self.conv(x)
            )
        )

class DSConv(nn.Module):
    def __init__(
        self, 
        c1, 
        c2, 
        k=3, 
        s=1, 
        p=None, 
        act=True
    ):
        super().__init__()
        self.dwconv = nn.Conv2d(c1, c1, k, s, autopad(k, p), groups=c1, bias=False)
        self.pwconv = nn.Conv2d(c1, c2, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(
            self.bn(
                self.pwconv(
                    self.dwconv(x)
                )
            )
        )

class DS_Bottleneck(nn.Module):
    def __init__(
        self, 
        c1, 
        c2, 
        k=3, 
        shortcut=True
    ):
        super().__init__()
        self.dsconv1 = DSConv(c1, c1, k=3, s=1)
        self.dsconv2 = DSConv(c1, c2, k=k, s=1)
        self.shortcut = shortcut and c1 == c2

    def forward(self, x):
        return x + self.dsconv2(self.dsconv1(x)) if self.shortcut else self.dsconv2(self.dsconv1(x))

class DS_C3k(nn.Module):
    def __init__(
        self, 
        c1, 
        c2, 
        n=1, 
        k=3, 
        e=0.5
    ):
        super().__init__()
        self.cv1 = Conv(c1, int(c2 * e), 1, 1)
        self.cv2 = Conv(c1, int(c2 * e), 1, 1)
        self.cv3 = Conv(2 * int(c2 * e), c2, 1, 1)
        self.m = nn.Sequential(
            *[
                DS_Bottleneck(
                    int(c2 * e), 
                    int(c2 * e), 
                    k=k, 
                    shortcut=True
                ) 
                for _ in range(n)
            ]
        )

    def forward(self, x):
        return self.cv3(
            torch.cat(
                (self.m(self.cv1(x)), self.cv2(x)), 
                dim=1
            )
        )

class DS_C3k2(nn.Module):
    def __init__(
        self, 
        c1, 
        c2, 
        n=1, 
        k=3, 
        e=0.5
    ):
        super().__init__()
        self.cv1 = Conv(c1, int(c2 * e), 1, 1)
        self.m = DS_C3k(int(c2 * e), int(c2 * e), n=n, k=k, e=1.0)
        self.cv2 = Conv(int(c2 * e), c2, 1, 1)

    def forward(self, x):
        return self.cv2(
            self.m(
                self.cv1(x)
            )
        )

class AdaptiveHyperedgeGeneration(nn.Module):
    def __init__(
        self, 
        in_channels, 
        num_hyperedges, 
        num_heads
    ):
        super().__init__()
        self.num_hyperedges = num_hyperedges
        self.num_heads = num_heads
        self.head_dim = max(1, in_channels // num_heads)
        self.global_proto = nn.Parameter(torch.randn(num_hyperedges, in_channels))
        self.context_mapper = nn.Linear(2 * in_channels, num_hyperedges * in_channels, bias=False)
        self.query_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        B, N, C = x.shape
        P = (
            self.global_proto.unsqueeze(0) + 
            self.context_mapper(
                torch.cat(
                    (
                        F.adaptive_avg_pool1d(x.permute(0, 2, 1), 1).squeeze(-1), 
                        F.adaptive_max_pool1d(x.permute(0, 2, 1), 1).squeeze(-1)
                    ), 
                    dim=1
                )
            ).view(B, self.num_hyperedges, C))

        return F.softmax((
            (self.query_proj(x).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3) @ P.view(B, self.num_hyperedges, self.num_heads, self.head_dim).permute(0, 2, 3, 1)) * self.scale
        ).mean(dim=1).permute(0, 2, 1), dim=-1)

class HypergraphConvolution(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels
    ):
        super().__init__()
        self.W_e = nn.Linear(in_channels, in_channels, bias=False)
        self.W_v = nn.Linear(in_channels, out_channels, bias=False)
        self.act = nn.SiLU()

    def forward(self, x, A):
        return x + self.act(self.W_v(A.transpose(1, 2).bmm(self.act(self.W_e(A.bmm(x))))))

class AdaptiveHypergraphComputation(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        num_hyperedges, 
        num_heads
    ):
        super().__init__()
        self.adaptive_hyperedge_gen = AdaptiveHyperedgeGeneration(in_channels, num_hyperedges, num_heads)
        self.hypergraph_conv = HypergraphConvolution(in_channels, out_channels)

    def forward(self, x):
        B, _, H, W = x.shape
        x_flat = x.flatten(2).permute(0, 2, 1)

        return self.hypergraph_conv(x_flat, self.adaptive_hyperedge_gen(x_flat)).permute(0, 2, 1).view(B, -1, H, W)

class C3AH(nn.Module):
    def __init__(
        self, 
        c1, 
        c2, 
        num_hyperedges, 
        num_heads, 
        e=0.5
    ):
        super().__init__()
        self.cv1 = Conv(c1, int(c1 * e), 1, 1)
        self.cv2 = Conv(c1, int(c1 * e), 1, 1)
        self.ahc = AdaptiveHypergraphComputation(int(c1 * e), int(c1 * e), num_hyperedges, num_heads)
        self.cv3 = Conv(2 * int(c1 * e), c2, 1, 1)

    def forward(self, x):
        return self.cv3(
            torch.cat(
                (self.ahc(self.cv2(x)), self.cv1(x)), 
                dim=1
            )
        )

class HyperACE(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        num_hyperedges=16, 
        num_heads=8, 
        k=2, 
        l=1, 
        c_h=0.5, 
        c_l=0.25
    ):
        super().__init__()
        c2, c3, c4, c5 = in_channels 
        c_mid = c4
        self.fuse_conv = Conv(c2 + c3 + c4 + c5, c_mid, 1, 1) 
        self.c_h = int(c_mid * c_h)
        self.c_l = int(c_mid * c_l)
        self.c_s = c_mid - self.c_h - self.c_l
        self.high_order_branch = nn.ModuleList([
            C3AH(
                self.c_h, 
                self.c_h, 
                num_hyperedges=num_hyperedges, 
                num_heads=num_heads, e=1.0
            ) 
            for _ in range(k)
        ])
        self.high_order_fuse = Conv(self.c_h * k, self.c_h, 1, 1)
        self.low_order_branch = nn.Sequential(
            *[
                DS_C3k(
                    self.c_l, 
                    self.c_l, 
                    n=1, 
                    k=3, 
                    e=1.0
                ) 
                for _ in range(l)
            ]
        )
        self.final_fuse = Conv(self.c_h + self.c_l + self.c_s, out_channels, 1, 1)

    def forward(self, x):
        B2, B3, B4, B5 = x 
        _, _, H4, W4 = B4.shape

        x_h, x_l, x_s = self.fuse_conv(
            torch.cat(
                (
                    F.interpolate(
                        B2, 
                        size=(H4, W4), 
                        mode='bilinear', 
                        align_corners=False
                    ), 
                    F.interpolate(
                        B3, 
                        size=(H4, W4), 
                        mode='bilinear', 
                        align_corners=False
                    ), 
                    B4, 
                    F.interpolate(
                        B5, 
                        size=(H4, W4), 
                        mode='bilinear', 
                        align_corners=False
                    )
                ), 
                dim=1
            )
        ).split([self.c_h, self.c_l, self.c_s], dim=1)

        return self.final_fuse(
            torch.cat(
                (
                    self.high_order_fuse(torch.cat([m(x_h) for m in self.high_order_branch], dim=1)), 
                    self.low_order_branch(x_l), 
                    x_s
                ), 
                dim=1
            )
        )

class GatedFusion(nn.Module):
    def __init__(
        self, 
        in_channels
    ):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, in_channels, 1, 1))

    def forward(self, f_in, h):
        return f_in + self.gamma * h

class YOLO13Encoder(nn.Module):
    def __init__(
        self, 
        in_channels, 
        base_channels=32
    ):
        super().__init__()
        self.stem = DSConv(
            in_channels, 
            base_channels, 
            k=3, 
            s=1
        )
        self.p2 = nn.Sequential(
            DSConv(
                base_channels, 
                base_channels*2, k=3, s=(2, 2)), 
            DS_C3k2(
                base_channels*2, 
                base_channels*2, 
                n=1
            )
        )
        self.p3 = nn.Sequential(
            DSConv(
                base_channels*2, 
                base_channels*4, 
                k=3, 
                s=(2, 2)
            ), 
            DS_C3k2(
                base_channels*4, 
                base_channels*4, 
                n=2
            )
        )
        self.p4 = nn.Sequential(
            DSConv(
                base_channels*4, 
                base_channels*8, 
                k=3, 
                s=(2, 2)
            ), 
            DS_C3k2(
                base_channels*8, 
                base_channels*8, 
                n=2
            )
        )
        self.p5 = nn.Sequential(
            DSConv(
                base_channels*8, 
                base_channels*16, 
                k=3, 
                s=(2, 2)
            ), 
            DS_C3k2(
                base_channels*16, 
                base_channels*16, 
                n=1
            )
        )
        
        self.out_channels = [base_channels*2, base_channels*4, base_channels*8, base_channels*16]

    def forward(self, x):
        p2 = self.p2(self.stem(x))
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)

        return [p2, p3, p4, p5]

class YOLO13FullPADDecoder(nn.Module):
    def __init__(self, encoder_channels, hyperace_out_c, out_channels_final):
        super().__init__()
        c_p2, c_p3, c_p4, c_p5 = encoder_channels
        c_d5, c_d4, c_d3, c_d2 = c_p5, c_p4, c_p3, c_p2
        
        self.h_to_d5 = Conv(
            hyperace_out_c, 
            c_d5, 
            1, 
            1
        )
        self.h_to_d4 = Conv(
            hyperace_out_c, 
            c_d4, 
            1, 
            1
        )
        self.h_to_d3 = Conv(
            hyperace_out_c, 
            c_d3, 
            1, 
            1
        )
        self.h_to_d2 = Conv(
            hyperace_out_c, 
            c_d2, 
            1, 
            1
        )

        self.fusion_d5 = GatedFusion(c_d5)
        self.fusion_d4 = GatedFusion(c_d4)
        self.fusion_d3 = GatedFusion(c_d3)
        self.fusion_d2 = GatedFusion(c_d2)

        self.skip_p5 = Conv(
            c_p5, 
            c_d5, 
            1, 
            1
        )
        self.skip_p4 = Conv(
            c_p4, 
            c_d4, 
            1, 
            1
        )
        self.skip_p3 = Conv(
            c_p3, 
            c_d3, 
            1, 
            1
        )
        self.skip_p2 = Conv(
            c_p2, 
            c_d2, 
            1, 
            1
        )

        self.up_d5 = DS_C3k2(
            c_d5, 
            c_d4, 
            n=1
        )
        self.up_d4 = DS_C3k2(
            c_d4, 
            c_d3, 
            n=1
        )
        self.up_d3 = DS_C3k2(
            c_d3, 
            c_d2, 
            n=1
        )
        
        self.final_d2 = DS_C3k2(
            c_d2, 
            c_d2, 
            n=1
        )
        self.final_conv = Conv(
            c_d2,
            out_channels_final, 
            1, 
            1
        )

    def forward(self, enc_feats, h_ace):
        p2, p3, p4, p5 = enc_feats
        d5 = self.skip_p5(p5)

        d4 = self.up_d5(
            F.interpolate(
                self.fusion_d5(d5, self.h_to_d5(F.interpolate(h_ace, size=d5.shape[2:], mode='bilinear', align_corners=False))), 
                size=p4.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
        ) + self.skip_p4(p4)

        d3 = self.up_d4(
            F.interpolate(
                self.fusion_d4(d4, self.h_to_d4(F.interpolate(h_ace, size=d4.shape[2:], mode='bilinear', align_corners=False))), 
                size=p3.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
        ) + self.skip_p3(p3)

        d2 = self.up_d3(
            F.interpolate(
                self.fusion_d3(d3, self.h_to_d3(F.interpolate(h_ace, size=d3.shape[2:], mode='bilinear', align_corners=False))), 
                size=p2.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
        ) + self.skip_p2(p2)

        return self.final_conv(
            self.final_d2(
                self.fusion_d2(
                    d2, 
                    self.h_to_d2(
                        F.interpolate(h_ace, size=d2.shape[2:], mode='bilinear', align_corners=False)
                    )
                )
            )
        )
