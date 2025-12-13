import torch.nn as nn
import torch.nn.functional as F
from advanced_rvc_inference.library.predictors.RMVPE.HPA.yolo import DSConv, DS_C3k2, HyperACE, GatedFusion, Conv

class YOLO13Encoder(nn.Module):
    def __init__(self, in_channels, base_channels=32):
        super().__init__()
        self.stem = DSConv(in_channels, base_channels, k=3, s=1) 
        
        self.p2 = nn.Sequential(
            DSConv(base_channels, base_channels*2, k=3, s=(2, 2)), 
            DS_C3k2(base_channels*2, base_channels*2, n=1)
        )
        
        self.p3 = nn.Sequential(
            DSConv(base_channels*2, base_channels*4, k=3, s=(2, 2)), 
            DS_C3k2(base_channels*4, base_channels*4, n=2)
        )
        
        self.p4 = nn.Sequential(
            DSConv(base_channels*4, base_channels*8, k=3, s=(2, 2)), 
            DS_C3k2(base_channels*8, base_channels*8, n=2)
        )
        
        self.p5 = nn.Sequential(
            DSConv(base_channels*8, base_channels*16, k=3, s=(2, 2)), 
            DS_C3k2(base_channels*16, base_channels*16, n=1)
        )
        
        self.out_channels = [base_channels*2, base_channels*4, base_channels*8, base_channels*16]

    def forward(self, x):
        x = self.stem(x)
        p2 = self.p2(x)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        return [p2, p3, p4, p5]

class YOLO13FullPADDecoder(nn.Module):
    def __init__(self, encoder_channels, hyperace_out_c, out_channels_final):
        super().__init__()
        c_p2, c_p3, c_p4, c_p5 = encoder_channels
        c_d5, c_d4, c_d3, c_d2 = c_p5, c_p4, c_p3, c_p2
        
        self.h_to_d5 = Conv(hyperace_out_c, c_d5, 1, 1)
        self.h_to_d4 = Conv(hyperace_out_c, c_d4, 1, 1)
        self.h_to_d3 = Conv(hyperace_out_c, c_d3, 1, 1)
        self.h_to_d2 = Conv(hyperace_out_c, c_d2, 1, 1)

        self.fusion_d5 = GatedFusion(c_d5)
        self.fusion_d4 = GatedFusion(c_d4)
        self.fusion_d3 = GatedFusion(c_d3)
        self.fusion_d2 = GatedFusion(c_d2)

        self.skip_p5 = Conv(c_p5, c_d5, 1, 1)
        self.skip_p4 = Conv(c_p4, c_d4, 1, 1)
        self.skip_p3 = Conv(c_p3, c_d3, 1, 1)
        self.skip_p2 = Conv(c_p2, c_d2, 1, 1)

        self.up_d5 = DS_C3k2(c_d5, c_d4, n=1)
        self.up_d4 = DS_C3k2(c_d4, c_d3, n=1)
        self.up_d3 = DS_C3k2(c_d3, c_d2, n=1)
        
        self.final_d2 = DS_C3k2(c_d2, c_d2, n=1)
        self.final_conv = Conv(c_d2, out_channels_final, 1, 1)

    def forward(self, enc_feats, h_ace):
        p2, p3, p4, p5 = enc_feats
        
        d5 = self.skip_p5(p5)
        h_d5 = self.h_to_d5(F.interpolate(h_ace, size=d5.shape[2:], mode='bilinear', align_corners=False))
        d5 = self.fusion_d5(d5, h_d5)

        d5_up = F.interpolate(d5, size=p4.shape[2:], mode='bilinear', align_corners=False)

        d4 = self.up_d5(d5_up) + self.skip_p4(p4)
        h_d4 = self.h_to_d4(F.interpolate(h_ace, size=d4.shape[2:], mode='bilinear', align_corners=False))
        d4 = self.fusion_d4(d4, h_d4)
        
        d4_up = F.interpolate(d4, size=p3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.up_d4(d4_up) + self.skip_p3(p3)
        h_d3 = self.h_to_d3(F.interpolate(h_ace, size=d3.shape[2:], mode='bilinear', align_corners=False))
        d3 = self.fusion_d3(d3, h_d3)

        d3_up = F.interpolate(d3, size=p2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.up_d3(d3_up) + self.skip_p2(p2)
        h_d2 = self.h_to_d2(F.interpolate(h_ace, size=d2.shape[2:], mode='bilinear', align_corners=False))
        d2 = self.fusion_d2(d2, h_d2)

        d2 = self.final_d2(d2)
        return self.final_conv(d2)

class DeepUnet(nn.Module):
    def __init__(self, 
                 in_channels=1, 
                 en_out_channels=16, 
                 base_channels=64, 
                 hyperace_k=2, 
                 hyperace_l=1, 
                 num_hyperedges=16, 
                 num_heads=8):
        super().__init__()
        
        self.encoder = YOLO13Encoder(in_channels, base_channels)
        enc_ch = self.encoder.out_channels
        
        self.hyperace = HyperACE(
            in_channels=enc_ch,
            out_channels=enc_ch[-1],
            num_hyperedges=num_hyperedges,
            num_heads=num_heads,
            k=hyperace_k, 
            l=hyperace_l
        )
        
        self.decoder = YOLO13FullPADDecoder(
            encoder_channels=enc_ch,
            hyperace_out_c=enc_ch[-1],
            out_channels_final=en_out_channels
        )

    def forward(self, x):
        original_size = x.shape[2:]
        
        features = self.encoder(x)
        h_ace = self.hyperace(features)
        x_dec = self.decoder(features, h_ace)
        
        x_out = F.interpolate(x_dec, size=original_size, mode='bilinear', align_corners=False)
        
        return x_out
