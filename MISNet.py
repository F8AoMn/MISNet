import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from MISNet.res2net_v1b_base import Res2Net_model
from ptflops import get_model_complexity_info


class BConv(nn.Module):
    """Conv-BN-ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, **kwargs):
        super(BConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


def upsample(x, size):
    return F.interpolate(x, size, mode='bilinear', align_corners=True)


class Gate(nn.Module):
    def __init__(self, in_channels):
        super(Gate, self).__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.gate(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.max_pool(x)  # [3, 512, 1, 1]
        # print(x1.shape)
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        # print(max_out.shape)
        out = max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)


class ASPP(nn.Module):
    def __init__(self, in_channel, depth):
        super(ASPP, self).__init__()
        # global average pooling : init nn.AdaptiveAvgPool2d ;also forward torch.mean(,,keep_dim=True)
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        # k=1 s=1 no pad
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)

        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)

    def forward(self, x):
        size = x.shape[2:]

        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.interpolate(image_features, size=size, mode='bilinear', align_corners=True)

        atrous_block1 = self.atrous_block1(x)

        atrous_block6 = self.atrous_block6(x)

        atrous_block12 = self.atrous_block12(x)

        atrous_block18 = self.atrous_block18(x)

        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))

        return net


class PyramidPooling(nn.Module):
    """Pyramid pooling module"""

    def __init__(self, in_channels, out_channels, **kwargs):
        super(PyramidPooling, self).__init__()
        inter_channels = int(in_channels / 4)
        self.conv1 = BConv(in_channels, inter_channels, 1, **kwargs)
        self.conv2 = BConv(in_channels, inter_channels, 1, **kwargs)
        self.conv3 = BConv(in_channels, inter_channels, 1, **kwargs)
        self.conv4 = BConv(in_channels, inter_channels, 1, **kwargs)
        self.out = BConv(in_channels * 2, out_channels, 1)

    def pool(self, x, size):
        avgpool = nn.AdaptiveAvgPool2d(size)
        return avgpool(x)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = upsample(self.conv1(self.pool(x, 1)), size)
        feat2 = upsample(self.conv2(self.pool(x, 2)), size)
        feat3 = upsample(self.conv3(self.pool(x, 3)), size)
        feat4 = upsample(self.conv4(self.pool(x, 6)), size)
        x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
        x = self.out(x)
        return x


class FM0(nn.Module):
    def __init__(self, in_dim, out_dim=128):
        super(FM0, self).__init__()
        self.CA = ChannelAttention(out_dim)
        self.SA = SpatialAttention()
        self.gate = Gate(out_dim)
        self.conv = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )
        # self.trans = BConv(64, 128, kernel_size=1, stride=1, padding=0)

    def forward(self, rgb, depth):
        rgb_w = (rgb + depth) * rgb  # [3, 64, 64, 64]
        # print('rgb_w:', rgb_w.shape)
        dep_w = (rgb + depth) * depth  # [3, 64, 64, 64]
        # print('dep_w:', dep_w.shape)
        rd_w = torch.cat((rgb_w, dep_w), dim=1)  # [3, 128, 64, 64]
        # print(rd_w.shape)
        # print('rd_w:', rd_w.shape)
        # print('self.gate(rd_w):', self.gate(rd_w).shape)  # [3, 1, 64, 64]
        rgb_h = rgb * self.gate(rd_w)  # [3, 64, 64, 64]
        # print('rgb_h:', rgb_h.shape)
        dep_h = depth * self.gate(rd_w)  # [3, 64, 64, 64]
        # print('dep_h:', dep_h.shape)
        rd_h = self.conv(torch.cat((rgb_h, dep_h), dim=1))  # [3, 128, 64, 64]
        # print('rd_h:', rd_h.shape)
        rd_c = self.CA(rd_h) * rd_h  # [3, 128, 64, 64]
        # print('rd_c', rd_c.shape)
        out1 = self.SA(rd_c) * rd_c  # [3, 128, 64, 64]
        # print('out1:', out1.shape)
        return out1


class FM1(nn.Module):
    def __init__(self, in_dim, out_dim, ratio=2):
        super(FM1, self).__init__()
        self.CA = ChannelAttention(in_dim)
        self.SA = SpatialAttention()
        self.gate = Gate(out_dim)
        self.conv = nn.Sequential(
            nn.Conv2d(out_dim, out_dim//ratio, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_dim//ratio),
            nn.ReLU(inplace=True)
        )

    def forward(self, rgb, depth, xx):
        rgb_w = (rgb + depth) * rgb  # [3, 256, 64, 64]
        # print('rgb_w:', rgb_w.shape)
        dep_w = (rgb + depth) * depth  # [3, 256, 64, 64]
        # print('dep_w:', dep_w.shape)
        rd_w = torch.cat((rgb_w, dep_w), dim=1)  # [3, 512, 64, 64]
        # print('rd_w:', rd_w.shape)
        rgb_h = rgb * self.gate(rd_w)  # [3, 256, 64, 64]
        # print('rgb_h:', rgb_h.shape)
        dep_h = depth * self.gate(rd_w)  # [3, 256, 64, 64]
        # print('dep_h:', dep_h.shape)
        rd_h = self.conv(torch.cat((rgb_h, dep_h), dim=1))  # [3, 256, 64, 64]
        # print('rd_h:', rd_h.shape)
        rd_c = self.CA(rd_h) * rd_h  # [3, 256, 64, 64]
        # print('rd_c', rd_c.shape)
        rd_a = self.SA(rd_c) * rd_c  # [3, 256, 64, 64]
        # print('rd_a:', rd_a.shape)
        # print(rd_a.shape)
        # print(xx.shape)
        out2 = rd_a + xx  # [3, 256, 64, 64]
        # print('out2:', out2.shape)
        return out2


class FM2(nn.Module):
    def __init__(self, in_dim, out_dim, ratio=2):
        super(FM2, self).__init__()
        self.CA = ChannelAttention(out_dim // ratio)
        self.SA = SpatialAttention()
        self.gate = Gate(out_dim)
        self.conv = nn.Sequential(
            nn.Conv2d(out_dim, out_dim // ratio, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_dim // ratio),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        # self.trans = BConv(256, 128, kernel_size=1, stride=2, padding=0)

    def forward(self, rgb, depth, xx):
        rgb_w = (rgb + depth) * rgb  # [3, 512, 32, 32]
        # print('rgb_w:', rgb_w.shape)
        dep_w = (rgb + depth) * depth  # [3, 512, 32, 32]
        # print('dep_w:', dep_w.shape)
        rd_w = torch.cat((rgb_w, dep_w), dim=1)  # [3, 1024, 32, 32]
        # print('rd_w:', rd_w.shape)
        rgb_h = rgb * self.gate(rd_w)  # [3, 512, 32, 32]
        # print('rgb_h:', rgb_h.shape)
        dep_h = depth * self.gate(rd_w)  # [3, 512, 32, 32]
        # print('dep_h:', dep_h.shape)
        rd_h = self.conv(torch.cat((rgb_h, dep_h), dim=1))  # [3, 512, 32, 32]
        # print('rd_h:', rd_h.shape)
        rd_c = self.CA(rd_h) * rd_h  # [3, 512, 32, 32]
        # print('rd_c', rd_c.shape)
        rd_a = self.SA(rd_c) * rd_c  # [3, 512, 32, 32]
        # print('rd_a:', rd_a.shape)
        # print('trans(xx):', self.trans(xx).shape)  # [3, 512, 32, 32]
        # print(rd_a.shape)
        # print(xx.shape)
        if xx.size()[2:] != rd_a.size()[2:]:
            xx = self.pool(xx)
        out2 = rd_a + xx  # [3, 512, 32, 32]

        # print('out2:', out2.shape)
        return out2


class FM3(nn.Module):
    def __init__(self, in_dim, out_dim, ratio=2):
        super(FM3, self).__init__()
        self.CA = ChannelAttention(out_dim // ratio)
        self.SA = SpatialAttention()
        self.gate = Gate(out_dim)
        self.conv = nn.Sequential(
            nn.Conv2d(out_dim, out_dim // ratio, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_dim // ratio),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

    def forward(self, rgb, depth, xx):
        rgb_w = (rgb + depth) * rgb  # [[3, 1024, 16, 16]
        # print('rgb_w:', rgb_w.shape)
        dep_w = (rgb + depth) * depth  # [3, 1024, 16, 16]
        # print('dep_w:', dep_w.shape)
        rd_w = torch.cat((rgb_w, dep_w), dim=1)  # [3, 2048, 16, 16]
        # print('rd_w:', rd_w.shape)
        rgb_h = rgb * self.gate(rd_w)  # [3, 1024, 16, 16]
        # print('rgb_h:', rgb_h.shape)
        dep_h = depth * self.gate(rd_w)  # [3, 1024, 16, 16]
        # print('dep_h:', dep_h.shape)
        rd_h = self.conv(torch.cat((rgb_h, dep_h), dim=1))  # [3, 1024, 16, 16]
        # print('rd_h:', rd_h.shape)
        rd_c = self.CA(rd_h) * rd_h  # [3, 1024, 16, 16]
        # print('rd_c', rd_c.shape)
        rd_a = self.SA(rd_c) * rd_c  # [3, 1024, 16, 16]
        # print('rd_a:', rd_a.shape)
        if xx.size()[2:] != rd_a.size()[2:]:
            xx = self.pool(xx)
        out2 = rd_a + xx # [3, 1024, 16, 16]
        # print('out2:', out2.shape)
        return out2


class FM4(nn.Module):
    def __init__(self, in_dim, out_dim, ratio=2):
        super(FM4, self).__init__()
        self.CA = ChannelAttention(out_dim // ratio)
        self.SA = SpatialAttention()
        self.gate = Gate(out_dim)
        self.conv = nn.Sequential(
            nn.Conv2d(out_dim, out_dim // ratio, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_dim // ratio),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(1024, 2048, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True)
        )

    def forward(self, rgb, depth, xx):
        rgb_w = (rgb + depth) * rgb  # [3, 2048, 8, 8]
        # print('rgb_w:', rgb_w.shape)
        dep_w = (rgb + depth) * depth  # [3, 2048, 8, 8]
        # print('dep_w:', dep_w.shape)
        rd_w = torch.cat((rgb_w, dep_w), dim=1)  # [3, 4096, 8, 8]
        # print('rd_w:', rd_w.shape)
        rgb_h = rgb * self.gate(rd_w)  # [3, 2048, 8, 8]
        # print('rgb_h:', rgb_h.shape)
        dep_h = depth * self.gate(rd_w)  # [3, 2048, 8, 8]
        # print('dep_h:', dep_h.shape)
        rd_h = self.conv(torch.cat((rgb_h, dep_h), dim=1))  # [3, 2048, 8, 8]
        # print('rd_h:', rd_h.shape)
        rd_c = self.CA(rd_h) * rd_h  # [3, 2048, 8, 8]
        # print('rd_c', rd_c.shape)
        rd_a = self.SA(rd_c) * rd_c  # [3, 2048, 8, 8]
        # print('rd_a:', rd_a.shape)
        if xx.size()[2:] != rd_a.size()[2:]:
            xx = self.pool(xx)
        out2 = rd_a + xx # [3, 2048, 8, 8]
        # print('out2:', out2.shape)
        return out2


class PDM(nn.Module):
    def __init__(self, in_channel, depth):
        super(PDM, self).__init__()

        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        self.conv1 = nn.Conv2d(in_channel, depth, 1, 1)
        # k=1 s=1 no pad
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block1_1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block6_1 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
        self.atrous_block12_1 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)
        self.atrous_block18_1 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)

        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)

    def forward(self, r5, d5, f5):
        size = f5.shape[2:]
        fr = f5 + r5
        fd = f5 + d5
        image_features = self.mean(fr)
        image_features = self.conv(image_features)
        image_features = F.interpolate(image_features, size=size, mode='bilinear', align_corners=True)
        image_features_1 = self.mean(fd)
        image_features_1 = self.conv(image_features_1)
        image_features_1 = F.interpolate(image_features_1, size=size, mode='bilinear', align_corners=True)

        atrous_block1 = self.atrous_block1(fr)
        atrous_block1_1 = self.atrous_block1_1(fd)
        atrous_block6 = self.atrous_block6(fr)
        atrous_block6_1 = self.atrous_block6_1(fd)
        atrous_block12 = self.atrous_block12(fr)
        atrous_block12_1 = self.atrous_block12_1(fd)
        atrous_block18 = self.atrous_block18(fr)
        atrous_block18_1 = self.atrous_block18_1(fd)

        net = self.conv_1x1_output(torch.cat([image_features+image_features_1, atrous_block1+atrous_block1_1, atrous_block6+atrous_block6_1,
                                              atrous_block12+atrous_block12_1, atrous_block18+atrous_block18_1], dim=1))

        return net

class Decoder0(nn.Module):
    def __init__(self, in_channel, out_channel):  # 1024 256
        super(Decoder0, self).__init__()
        self.SA = SpatialAttention()
        self.pool = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(128, 128, 1, 1)
        # k=1 s=1 no pad
        self.atrous_block1 = nn.Conv2d(1, 128, 1, 1)

        self.atrous_block2 = nn.Conv2d(in_channel, 128, 3, 1, padding=2, dilation=2)

        self.atrous_block4 = nn.Conv2d(in_channel, 128, 3, 1, padding=4, dilation=4)

        # self.conv_1x1_output = nn.Conv2d(out_channel * 5, out_channel, 1, 1)
        self.bconv = BConv(128, 256, 3)

    def forward(self, x):
        size = x.shape[2:]
        atrous_block1 = self.SA(x)  # [3, 1, 8, 8]
        # print(atrous_block1.shape)
        atrous_block1 = self.atrous_block1(atrous_block1)  # [3, 256, 16, 16]
        # print('atrous_block1:', atrous_block1.shape)

        atrous_block2 = self.atrous_block2(x)  # [3, 256, 14, 14]
        # print('atrous_block2:', atrous_block2.shape)

        atrous_block4 = self.atrous_block4(x)
        # print('atrous_block4:', atrous_block4.shape)

        d1 = nn.Softmax(dim=2)(atrous_block1 * atrous_block2)
        d2 = d1 * atrous_block4 + atrous_block1
        d3 = self.conv(d2)

        out3 = atrous_block4 + d3
        # print('out3:', out3.shape)
        return self.pool(out3)


class Decoder1(nn.Module):
    def __init__(self, in_channel, out_channel):  # 256 256
        super(Decoder1, self).__init__()
        self.SA = SpatialAttention()
        self.pool = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(128, 128, 1, 1)
        # k=1 s=1 no pad
        self.atrous_block1 = nn.Conv2d(1, 128, 1, 1)

        self.atrous_block2 = nn.Conv2d(128, 128, 3, 1, padding=2, dilation=2, bias=False)

        self.atrous_block4 = nn.Conv2d(128, 128, 3, 1, padding=4, dilation=4, bias=False)

        # self.conv_1x1_output = nn.Conv2d(out_channel * 5, out_channel, 1, 1)
        self.bconv = BConv(128, 256, 3)

    def forward(self, x):
        size = x.shape[2:]
        atrous_block1 = self.SA(x)
        atrous_block1 = self.atrous_block1(atrous_block1)  # [3, 256, 16, 16]
        # print('atrous_block1:', atrous_block1.shape)

        atrous_block2 = self.atrous_block2(x)  # [3, 256, 14, 14]
        # print('atrous_block2:', atrous_block2.shape)

        atrous_block4 = self.atrous_block4(x)
        # print('atrous_block4:', atrous_block4.shape)

        d1 = nn.Softmax(dim=2)(atrous_block1 * atrous_block2)
        d2 = d1 * atrous_block4 + atrous_block1
        d3 = self.conv(d2)

        out3 = atrous_block4 + d3
        # print('out3:', out3.shape)
        return self.pool(out3)


class Integraty_Enhanced_Module(nn.Module):
    def __init__(self, inchannels):
        super(Integraty_Enhanced_Module, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.trans_1 = nn.Conv2d(inchannels, 6, 1)
        self.trans_2 = nn.Conv2d(inchannels*2, 6, 1)
        self.up2x = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, features, last_predict):
        features_map = self.trans_1(features)
        similarity = torch.cosine_similarity(features_map, last_predict, dim=1).unsqueeze(dim=1)
        features_sly = features * similarity
        features_unsly = features * (1 - similarity)
        features_enhance = torch.cat([features_sly, features_unsly], dim=1)

        out = self.trans_2(features_enhance)
        return out


class FFNet(nn.Module):
    def __init__(self, num_classes=6, ind=50, pretrained=True):
        super(FFNet, self).__init__()
        # Backbone model
        self.layer_rgb = Res2Net_model(ind)
        self.layer_dep = Res2Net_model(ind)
        self.layer_dep0 = nn.Conv2d(1, 3, kernel_size=1)

        self.fu_0 = FM0(64, 128)
        self.trans = BConv(128, 256, kernel_size=1, stride=1, padding=0)
        self.fu_1 = FM1(256, 512)
        self.fu_2 = FM2(512, 1024)
        self.fu_3 = FM3(1024, 2048)
        self.fu_4 = FM4(2048, 4096)

        # self.aspp = ASPP(in_channel=256 + 128, depth=128)
        # self.ppm = PyramidPooling(in_channels=2048, out_channels=256)
        self.pd1 = PDM(in_channel=2048, depth=512)
        # self.pd2 = PDM(in_channel=2048, depth=512)
        # self.pd3 = PDM(in_channel=2048, depth=512)

        self.decoder4 = Decoder0(in_channel=512, out_channel=512)

        self.decoder3 = Decoder1(in_channel=256, out_channel=256)

        self.decoder2 = Decoder1(in_channel=256, out_channel=256)

        # self.decoder1 = Decoder1(in_channel=256, out_channel=256)

        self.decoder_bonv4 = BConv(1024, 256, 1, stride=1, padding=0)
        self.decoder_adjust4 = BConv(128+256, 128, 1, 1, 0)

        self.decoder_bonv3 = BConv(512, 256, 1, stride=1, padding=0)
        self.decoder_adjust3 = BConv(256+128, 128, 1, 1, 0)

        self.decoder_bonv2 = BConv(256, 256, 1, stride=1, padding=0)
        self.decoder_adjust2 = BConv(256+128, 128, 1, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        # self.bconv = BConv(256, 256, kernel_size=3, stride=1, padding=1)

        self.refine = Integraty_Enhanced_Module(256)

        self.aux_supervison = nn.Conv2d(128, 6, 1)

        self.predict = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(6, 6, kernel_size=3, stride=1, padding=1, bias=False),
        )

    def forward(self, rgb, depths):
        # x = torch.chunk(rgb, 4, dim=1)
        # rgb = torch.cat(x[0:3], dim=1)
        # depths = x[3]
        rgb_0, rgb_1, rgb_2, rgb_3, rgb_4 = self.layer_rgb(rgb)
        dep_0, dep_1, dep_2, dep_3, dep_4 = self.layer_dep(self.layer_dep0(depths))
        # rgb0,dep0: [3, 64, 64, 64]
        # rgb1,dep1: [3, 256, 64, 64]
        # rgb2,dep2: [3, 512, 32, 32]
        # rgb3,dep3: [3, 1024, 16, 16]
        # rgb4,dep4: [3, 2048, 8, 8]
        ful_0 = self.fu_0(rgb_0, dep_0)  # [3, 128, 64, 64])
        # print('ful_0:', ful_0.shape)
        ful_0 = self.trans(ful_0)  # [3, 256, 64, 64]
        # print('ful_0:', ful_0.shape)
        ful_1 = self.fu_1(rgb_1, dep_1, ful_0)  # [3, 256, 64, 64]
        # print('ful_1:', ful_1.shape)
        ful_2 = self.fu_2(rgb_2, dep_2, ful_1)  # [3, 512, 32, 32]
        # print('ful_2:', ful_2.shape)
        ful_3 = self.fu_3(rgb_3, dep_3, ful_2)  # [3, 1024, 16, 16]
        # print('ful_3:', ful_3.shape)
        ful_4 = self.fu_4(rgb_4, dep_4, ful_3)  # [3, 2048, 8, 8]
        # print('ful_4:', ful_4.shape)

        # print('pd:', self.pd(ful_4).shape)  # [3, 1024, 8, 8]
        # print('up:', self.upsample(self.pd(ful_4)).shape)  # [3, 1024, 16, 16]
        d_4 = self.pd1(rgb_4, dep_4, ful_4)
        # p1 = self.pd2(rgb_4, dep_4)
        # p2 = self.pd3(ful_4, dep_4)
        # d_4 = torch.cat([d_4, p1, p2], dim=1)
        # print('d_4:', d_4.shape)
        # print('self.decoder4(d_4):', (self.decoder4(d_4)).shape)  # [3, 128, 16, 16]
        # print('decoder_bonv4:', (self.decoder_bonv4(ful_3)).shape)  # [3, 256, 16, 16]

        d_3 = self.decoder_adjust4(torch.cat([self.decoder4(d_4), self.decoder_bonv4(ful_3)], dim=1))\
              + self.decoder4(d_4)  # [3, 128, 16, 16]
        # print('d_3:', d_3.shape)
        # print('self.decoder3(d_3):', (self.decoder3(d_3)).shape)  # [3, 128, 16, 16]
        # print('self.decoder_bonv3:', (self.decoder_bonv3(ful_2)).shape)  # [3, 256, 32, 32]

        d_2 = self.decoder_adjust3(torch.cat([self.decoder3(d_3), self.decoder_bonv3(ful_2)], dim=1))\
              + self.decoder3(d_3)   # [3, 128, 32, 32]
        # print('d_2:', d_2.shape)
        # print('self.decoder2(d_2):', (self.decoder2(d_2)).shape)  # [3, 128, 64, 64]
        # print('decoder_bonv2:', (self.decoder_bonv2(ful_1)).shape)  # [3, 256, 64, 64]

        d_1 = self.decoder_adjust2(torch.cat([self.decoder2(d_2), self.decoder_bonv2(ful_1)], dim=1))\
              + self.decoder2(d_2)  # [3, 128, 64, 64]
        # print('d_1:', d_1.shape)
        d_0 = self.aux_supervison(d_1)  # [3, 6, 64, 64]

        # preout = torch.cat([ful_1, d_1], dim=1)
        # preout = self.aspp(preout)
        # print('self.refine(ful_1, d_0):', self.refine(ful_1, d_0).shape)
        refine = self.refine(ful_1, d_0) * d_0 + d_0
        # refine1 = self.refine_1(ful_0, d_0) * d_0 + d_0
        # out = torch.cat([refine1, refine2], dim=1)
        out = self.predict(refine)  # [3, 6, 256, 256]
        # print('out:', out.shape)
        return out, d_0


if __name__ == "__main__":
    image = torch.randn(3, 3, 256, 256)
    depth = torch.randn(3, 1, 256, 256)
    model = FFNet()
    out = model(image, depth)
    print(out[0].shape)
    print(out[1].shape)
    # flops, params = get_model_complexity_info(model,
    #   (4, 256, 256), as_strings=True, print_per_layer_stat=True, verbose=True)
    # print(params)  # [3, 6, 256, 256]
    # print(flops)
