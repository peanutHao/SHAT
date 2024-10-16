import torch
import torch.nn as nn
import torch.nn.functional as F
from pvtv2 import pvt_v2_b2
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from torch.nn import Conv2d, UpsamplingBilinear2d
import warnings
import torch
from mmcv.cnn import constant_init, kaiming_init
from torch import nn

warnings.filterwarnings('ignore')


class DCAT(nn.Module):

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            proj_size: int = 64,
            num_heads: int = 8,
            dropout_rate: float = 0.1,
            pos_embed=True,
    ) -> None:
        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.norm = nn.LayerNorm(hidden_size)
        self.dca_block = DCA(input_size=input_size, hidden_size=hidden_size, proj_size=proj_size, num_heads=num_heads,
                             attn_drop=dropout_rate)

        if pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, input_size, hidden_size))
        else:
            self.pos_embed = None

        self.ffn = nn.Sequential(
            nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=1, stride=1),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # print(x.size())
        y = x
        x = x.reshape(B, C, H * W).permute(0, 2, 1)

        if self.pos_embed is not None:
            x = x + self.pos_embed

        attn = x + self.dca_block(self.norm(x))

        attn = attn.reshape(B, H, W, C).permute(0, 3, 1, 2)  # Reshape back to (B, C, H, W)

        # Apply Feed Forward Network (FFN) and add skip connection
        ffn_out = self.ffn(attn) + attn

        return ffn_out + y


class DCA(nn.Module):
    def __init__(self, input_size, hidden_size, proj_size, num_heads=4, qkv_bias=True, attn_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # 分别为通道注意力和空间注意力生成 Q, K, V
        self.qk = nn.Linear(hidden_size, hidden_size * 2, bias=qkv_bias)
        self.v = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)

        # 空间注意力的额外投影层
        self.proj_k_spatial = nn.Linear(input_size, proj_size)
        self.proj_v_spatial = nn.Linear(input_size, proj_size)

        # 注意力机制的dropout
        self.attn_drop = nn.Dropout(attn_drop)

        # 输出投影层
        self.proj_out = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x):
        B, N, C = x.shape
        # print(x.size())
        qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads)
        qk = qk.permute(2, 0, 3, 1, 4)
        q, k = qk[0], qk[1]
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q.transpose(-2, -1)
        v = v.transpose(-2, -1)

        # 通道注意力
        attn_channel = (q @ k) * self.temperature
        attn_channel = attn_channel.softmax(dim=-1)
        attn_channel = self.attn_drop(attn_channel)
        out_channel = (attn_channel @ v).transpose(-2, -1).reshape(B, N, C) + x

        # 空间注意力
        # print(k.size())
        k = k.transpose(-2, -1)
        # print(k.size())
        k_spatial = self.proj_k_spatial(k)  # (B, H, N, P)
        v_spatial = self.proj_v_spatial(v)  # (B, H, N, P)
        attn_spatial = (q.transpose(-2, -1) @ k_spatial) * self.temperature
        attn_spatial = attn_spatial.softmax(dim=-1)
        attn_spatial = self.attn_drop(attn_spatial)

        out_spatial = (attn_spatial @ v_spatial.transpose(-2, -1)).permute(0, 3, 1, 2).reshape(B, N, C) + x

        # 结合通道注意力和空间注意力的输出
        out = self.proj_out(torch.cat((out_channel, out_spatial), dim=-1))

        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out) * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        return self.sigmoid(out)


class SCA(nn.Module):

    def __init__(self, in_channels, kernel_size):
        super().__init__()

        self.ca = ChannelAttention(in_planes=in_channels)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        return self.sa(self.ca(x))


class SE(nn.Module):

    def __init__(self, in_channels, ratio=16):
        super(SE, self).__init__()

        self.gap = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // ratio, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // ratio, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, sub):
        b, c, h, w = x.size()
        sub = self.gap(sub).view(b, c)
        sub = self.fc(sub).view(b, c, 1, 1)

        return x * sub.expand_as(x)


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class CM(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=512, embed_dim=768, k_s=3):
        super().__init__()

        self.proj = nn.Sequential(nn.Conv2d(input_dim, embed_dim, 3, padding=1, bias=False), nn.ReLU(),
                                  nn.Conv2d(embed_dim, embed_dim, 3, padding=1, bias=False), nn.ReLU())

    def forward(self, x):
        x = self.proj(x)
        return x


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class SHAT(nn.Module):
    def __init__(self, dim=64, dims=[64, 128, 320, 512]):
        super(SHAT, self).__init__()

        insize = 88
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = '/home/data/qianhao/Polyp_PVT_main/pretrained_pth/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = dims[0], dims[1], dims[2], dims[3]

        self.dcat4 = DCAT(input_size=insize * insize, hidden_size=64, proj_size=64, pos_embed=False)
        self.dcat3 = DCAT(input_size=insize * insize // 4, hidden_size=128, proj_size=128, pos_embed=False)
        self.dcat2 = DCAT(input_size=insize * insize // 16, hidden_size=320, proj_size=320, pos_embed=False)
        self.dcat1 = DCAT(input_size=insize * insize // 64, hidden_size=512, proj_size=412, pos_embed=False)


        self.CM_c4 = CM(input_dim=c4_in_channels, embed_dim=dim)
        self.CM_c3 = CM(input_dim=c3_in_channels, embed_dim=dim)
        self.CM_c2 = CM(input_dim=c2_in_channels, embed_dim=dim)
        self.CM_c1 = CM(input_dim=c1_in_channels, embed_dim=dim)

        self.linear_fuse3 = ConvModule(in_channels=dim, out_channels=dim, kernel_size=1,
                                       norm_cfg=dict(type='BN', requires_grad=True))
        self.linear_fuse2 = ConvModule(in_channels=dim, out_channels=dim, kernel_size=1,
                                       norm_cfg=dict(type='BN', requires_grad=True))
        self.linear_fuse1 = ConvModule(in_channels=dim, out_channels=dim, kernel_size=1,
                                       norm_cfg=dict(type='BN', requires_grad=True))

        self.linear_pred = Conv2d(dim, 1, kernel_size=1)
        self.dropout = nn.Dropout(0.1)
        self.linear_pred1 = Conv2d(dim, 1, kernel_size=1)
        self.dropout1 = nn.Dropout(0.1)
        self.linear_pred2 = Conv2d(dim, 1, kernel_size=1)
        self.dropout2 = nn.Dropout(0.1)


        self.conv1 = nn.Conv2d(1024, c1_in_channels, 1, bias=False)
        self.conv2 = nn.Conv2d(1024, c2_in_channels, 1, bias=False)
        self.conv3 = nn.Conv2d(1024, c3_in_channels, 1, bias=False)
        self.conv4 = nn.Conv2d(1024, c4_in_channels, 1, bias=False)

        self.SE3 = SE(in_channels=64)
        self.SE2 = SE(in_channels=64)
        self.SE1 = SE(in_channels=64)

    def forward(self, x):
        pvt = self.backbone(x)

        _c1, _c2, _c3, _c4 = pvt
        n, _, h, w = _c4.shape


        f = 0

        if w != 11:
            _c4 = F.interpolate(_c4, size=11, mode='bilinear', align_corners=False)
            _c3 = F.interpolate(_c3, size=22, mode='bilinear', align_corners=False)
            _c2 = F.interpolate(_c2, size=44, mode='bilinear', align_corners=False)
            _c1 = F.interpolate(_c1, size=88, mode='bilinear', align_corners=False)
            f = 1

        a = resize(_c3, size=_c4.size()[2:], mode='bilinear', align_corners=False)
        b = resize(_c2, size=_c4.size()[2:], mode='bilinear', align_corners=False)
        c = resize(_c1, size=_c4.size()[2:], mode='bilinear', align_corners=False)
        _c4 = self.dcat1(self.conv4(torch.cat([_c4, a, b, c], dim=1)))  # [1, 64, 11, 11]

        a = resize(_c4, size=_c3.size()[2:], mode='bilinear', align_corners=False)
        b = resize(_c2, size=_c3.size()[2:], mode='bilinear', align_corners=False)
        c = resize(_c1, size=_c3.size()[2:], mode='bilinear', align_corners=False)
        _c3 = self.dcat2(self.conv3(torch.cat((_c3, a, b, c), dim=1)))  # [1, 64, 22, 22]

        a = resize(_c4, size=_c2.size()[2:], mode='bilinear', align_corners=False)
        b = resize(_c3, size=_c2.size()[2:], mode='bilinear', align_corners=False)
        c = resize(_c1, size=_c2.size()[2:], mode='bilinear', align_corners=False)
        _c2 = self.dcat3(self.conv2(torch.cat((_c2, a, b, c), dim=1)))  # [1, 64, 44, 44]

        a = resize(_c3, size=_c1.size()[2:], mode='bilinear', align_corners=False)
        b = resize(_c2, size=_c1.size()[2:], mode='bilinear', align_corners=False)
        c = resize(_c4, size=_c1.size()[2:], mode='bilinear', align_corners=False)
        _c1 = self.dcat4(self.conv1(torch.cat((_c1, a, b, c), dim=1)))

        if f == 1:
            _c4 = F.interpolate(_c4, size=w, mode='bilinear', align_corners=False)
            _c3 = F.interpolate(_c3, size=w * 2, mode='bilinear', align_corners=False)
            _c2 = F.interpolate(_c2, size=w * 4, mode='bilinear', align_corners=False)
            _c1 = F.interpolate(_c1, size=w * 8, mode='bilinear', align_corners=False)

        _c4 = self.CM_c4(_c4)
        _c3 = self.CM_c3(_c3)
        _c2 = self.CM_c2(_c2)
        _c1 = self.CM_c1(_c1)

        _c4 = resize(_c4, size=_c3.size()[2:], mode='bilinear', align_corners=False)

        sub = abs(_c3 - _c4)
        L34 = self.SE3(_c3, sub)

        L34 = self.linear_fuse3(L34)
        O34 = L34

        _c3 = resize(_c3, size=_c2.size()[2:], mode='bilinear', align_corners=False)
        sub = abs(_c2 - _c3)

        L2 = self.SE2(_c2, sub)
        L2 = self.linear_fuse2(L2)
        O2 = L2

        _c2 = resize(_c2, size=_c1.size()[2:], mode='bilinear', align_corners=False)
        sub = abs(_c1 - _c2)

        _c = self.SE1(_c1, sub)
        _c = self.linear_fuse1(_c)

        x = self.dropout(_c)
        x = self.linear_pred(x)
        O2 = self.dropout2(O2)
        O2 = self.linear_pred2(O2)
        O34 = self.dropout1(O34)
        O34 = self.linear_pred1(O34)
        return x, O2, O34


if __name__ == '__main__':
    model = SHAT().to('cuda:5')

    from thop import profile
    import torch

    input = torch.randn(1, 3, 352, 352).to('cuda:1')
    macs, params = profile(model, inputs=(input,))
    print('macs:', macs / 1000000000)
    print('params:', params / 1000000)
    prediction1 = model(input)
    print(prediction1[0].size())
    print(prediction1[1].size())
    print(prediction1[2].size())
#     print(prediction1[3].size())






