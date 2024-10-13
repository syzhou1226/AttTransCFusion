import torch
import torch.nn as nn
from torchvision.models import resnet34
import torch.nn.functional as F
from lib import pvt_v2
from functools import partial
from timm.models.vision_transformer import _cfg


class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        x = self.conv(x)
        return x


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class GAF(nn.Module):
    def __init__(self, ch_1, ch_2, ch_int, ch_out, drop_rate=0.):
        super(GAF, self).__init__()
        self.se = SE_Block(ch_out)
        self.W_g = Conv(ch_1, ch_int, 1, bn=True, relu=False)
        self.W_x = Conv(ch_2, ch_int, 1, bn=True, relu=False)
        self.W = DoubleConv(ch_int + ch_int, ch_out)

        self.relu = nn.ReLU(inplace=True)


        self.dropout = nn.Dropout2d(drop_rate)
        self.drop_rate = drop_rate

    def forward(self, g, x):
        W_g = self.W_g(g)
        W_x = self.W_x(x)
        fuse = self.W(torch.cat([W_g, W_x], 1))
        out = self.se(fuse)
        return out


class PVTModule(nn.Module):
    def __init__(self, size=352):

        super().__init__()

        backbone = pvt_v2.PyramidVisionTransformerV2(
            patch_size=16,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True,

            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            depths=[3, 4, 6, 3],
            sr_ratios=[8, 4, 2, 1],
        )

        checkpoint = torch.load("pvt_v2_b2.pth")
        backbone.default_cfg = _cfg()
        backbone.load_state_dict(checkpoint)
        self.backbone = torch.nn.Sequential(*list(backbone.children()))[:-1]
        self.up_tosize = nn.Upsample(size=size)
        for i in [1, 4, 7, 10]:
            self.backbone[i] = torch.nn.Sequential(*list(self.backbone[i].children()))

    def get_pyramid(self, x):
        pyramid = []
        B = x.shape[0]
        for i, module in enumerate(self.backbone):
            if i in [0, 3, 6, 9]:
                x, H, W = module(x)
            elif i in [1, 4, 7, 10]:
                for sub_module in module:
                    x = sub_module(x, H, W)
            else:
                x = module(x)
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                pyramid.append(x)

        return pyramid

    def forward(self, x):
        pyramid = self.get_pyramid(x)

        return pyramid


class FEABlock(nn.Module):

    def __init__(self, inplanes, planes, pool, fusions):
        super(FEABlock, self).__init__()
        assert pool in ['avg', 'att']
        assert all([f in ['channel_add', 'channel_mul'] for f in fusions])
        assert len(fusions) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.planes = planes
        self.pool = pool
        self.fusions = fusions
        if 'att' in pool:
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusions:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusions:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_mul_conv = None

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pool == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(3)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = x * channel_mul_term
        else:
            out = x
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term

        return out


class Decoder(nn.Module):

    def __init__(self, ch):
        super(Decoder, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2)
        self.gc = FEABlock(ch, ch, 'att', ['channel_add', 'channel_mul'])

    def forward(self, low, high):
        high = self.upsample(high)
        fusion = torch.mul(low, high)
        return self.gc(fusion)

class AttTransCFusion(nn.Module):
    def __init__(self, num_classes=1, drop_rate=0.2, normal_init=True, pretrained=False):
        super().__init__()

        self.p = PVTModule()

        self.resnet = resnet34()
        if pretrained:
            self.resnet.load_state_dict(
                torch.load('/home/amax/zsy/AttTransCFusion/pretrained/resnet34-333f7ec4.pth'))
        self.resnet.fc = nn.Identity()
        self.drop = nn.Dropout2d(drop_rate)
        self.gaf1 = GAF(64, 64, 64, 32)
        self.gaf2 = GAF(128, 128, 64, 32)
        self.gaf3 = GAF(320, 256, 64, 32)
        self.gaf4 = GAF(512, 512, 64, 32)
        self.decoder = Decoder(32)
        self.final = nn.Sequential(
            Conv(32, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
        )

    def forward(self, x):
        x_u = self.resnet.conv1(x)  # 16, 64, 128, 128
        x_u = self.resnet.bn1(x_u)
        x_u = self.resnet.relu(x_u)
        x_u = self.resnet.maxpool(x_u)

        x_c_3 = self.p(x)[0]
        x_c_3 = self.drop(x_c_3)  # 16, 64, 64, 64
        x_c_2 = self.p(x)[1]
        x_c_2 = self.drop(x_c_2)  # 16, 128, 32, 32
        x_c_1 = self.p(x)[2]
        x_c_1 = self.drop(x_c_1)  # 16, 320, 16, 16
        x_c_0 = self.p(x)[3]
        x_c_0 = self.drop(x_c_0)  # 16, 512, 8, 8

        x_u_3 = self.resnet.layer1(x_u)  # 16, 64, 64, 64
        x_u_3 = self.drop(x_u_3)

        x_u_2 = self.resnet.layer2(x_u_3)  # 16, 128, 32, 32
        x_u_2 = self.drop(x_u_2)

        x_u_1 = self.resnet.layer3(x_u_2)  # 16, 256, 16, 16
        x_u_1 = self.drop(x_u_1)

        x_u_0 = self.resnet.layer4(x_u_1)  # 16, 512, 8, 8
        x_u_0 = self.drop(x_u_0)
        x1 = self.gaf1(x_c_3, x_u_3)
        x2 = self.gaf2(x_c_2, x_u_2)
        x3 = self.gaf3(x_c_1, x_u_1)
        x4 = self.gaf4(x_c_0, x_u_0)

        d_1 = self.decoder(x3, x4)
        d_2 = self.decoder(x2, d_1)
        d_3 = self.decoder(x1, d_2)
        map = F.interpolate(self.final(d_3), scale_factor=4, mode='bilinear', align_corners=True)
        return map

