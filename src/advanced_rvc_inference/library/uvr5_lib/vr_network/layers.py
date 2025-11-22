import os
import sys
import torch

import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.getcwd())

from main.library.uvr5_lib import spec_utils

class Conv2DBNActiv(nn.Module):
    def __init__(self, nin, nout, ksize=3, stride=1, pad=1, dilation=1, activ=nn.ReLU):
        super(Conv2DBNActiv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(nin, nout, kernel_size=ksize, stride=stride, padding=pad, dilation=dilation, bias=False), nn.BatchNorm2d(nout), activ())

    def __call__(self, input_tensor):
        return self.conv(input_tensor)

class SeperableConv2DBNActiv(nn.Module):
    def __init__(self, nin, nout, ksize=3, stride=1, pad=1, dilation=1, activ=nn.ReLU):
        super(SeperableConv2DBNActiv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                nin,
                nin,
                kernel_size=ksize,
                stride=stride,
                padding=pad,
                dilation=dilation,
                groups=nin,
                bias=False,
            ),
            nn.Conv2d(
                nin,
                nout,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(nout),
            activ(),
        )

    def __call__(self, input_tensor):
        return self.conv(input_tensor)

class Encoder(nn.Module):
    def __init__(self, nin, nout, ksize=3, stride=1, pad=1, activ=nn.LeakyReLU):
        super(Encoder, self).__init__()
        self.conv1 = Conv2DBNActiv(nin, nout, ksize, 1, pad, activ=activ)
        self.conv2 = Conv2DBNActiv(nout, nout, ksize, stride, pad, activ=activ)

    def __call__(self, input_tensor):
        skip = self.conv1(input_tensor)
        hidden = self.conv2(skip)

        return hidden, skip

class Decoder(nn.Module):
    def __init__(self, nin, nout, ksize=3, stride=1, pad=1, activ=nn.ReLU, dropout=False):
        super(Decoder, self).__init__()
        self.conv = Conv2DBNActiv(nin, nout, ksize, 1, pad, activ=activ)
        self.dropout = nn.Dropout2d(0.1) if dropout else None

    def __call__(self, input_tensor, skip=None):
        input_tensor = F.interpolate(input_tensor, scale_factor=2, mode="bilinear", align_corners=True)
        if skip is not None:
            skip = spec_utils.crop_center(skip, input_tensor)
            input_tensor = torch.cat([input_tensor, skip], dim=1)

        output_tensor = self.conv(input_tensor)
        if self.dropout is not None:
            output_tensor = self.dropout(output_tensor)

        return output_tensor

class ASPPModule(nn.Module):
    def __init__(self, nn_architecture, nin, nout, dilations=(4, 8, 16), activ=nn.ReLU):
        super(ASPPModule, self).__init__()
        self.nn_architecture = nn_architecture
        self.six_layer = [129605]
        self.seven_layer = [537238, 537227, 33966]
        extra_conv = SeperableConv2DBNActiv(nin, nin, 3, 1, dilations[2], dilations[2], activ=activ)

        self.conv1 = nn.Sequential(nn.AdaptiveAvgPool2d((1, None)), Conv2DBNActiv(nin, nin, 1, 1, 0, activ=activ))
        self.conv2 = Conv2DBNActiv(nin, nin, 1, 1, 0, activ=activ)
        self.conv3 = SeperableConv2DBNActiv(nin, nin, 3, 1, dilations[0], dilations[0], activ=activ)
        self.conv4 = SeperableConv2DBNActiv(nin, nin, 3, 1, dilations[1], dilations[1], activ=activ)
        self.conv5 = SeperableConv2DBNActiv(nin, nin, 3, 1, dilations[2], dilations[2], activ=activ)

        if self.nn_architecture in self.six_layer:
            self.conv6 = extra_conv
            nin_x = 6
        elif self.nn_architecture in self.seven_layer:
            self.conv6 = extra_conv
            self.conv7 = extra_conv
            nin_x = 7
        else:
            nin_x = 5

        self.bottleneck = nn.Sequential(Conv2DBNActiv(nin * nin_x, nout, 1, 1, 0, activ=activ), nn.Dropout2d(0.1))

    def forward(self, input_tensor):
        _, _, h, w = input_tensor.size()

        feat1 = F.interpolate(self.conv1(input_tensor), size=(h, w), mode="bilinear", align_corners=True)
        feat2 = self.conv2(input_tensor)
        feat3 = self.conv3(input_tensor)
        feat4 = self.conv4(input_tensor)
        feat5 = self.conv5(input_tensor)

        if self.nn_architecture in self.six_layer:
            feat6 = self.conv6(input_tensor)
            out = torch.cat((feat1, feat2, feat3, feat4, feat5, feat6), dim=1)
        elif self.nn_architecture in self.seven_layer:
            feat6 = self.conv6(input_tensor)
            feat7 = self.conv7(input_tensor)
            out = torch.cat((feat1, feat2, feat3, feat4, feat5, feat6, feat7), dim=1)
        else:
            out = torch.cat((feat1, feat2, feat3, feat4, feat5), dim=1)

        bottleneck_output = self.bottleneck(out)
        return bottleneck_output