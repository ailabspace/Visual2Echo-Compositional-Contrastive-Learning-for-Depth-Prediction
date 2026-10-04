import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone import BinauralResNet18, BinauralResNet34
import numpy as np


def _adaptive_avg_pool2d_det(x, size):
    ih, iw = x.shape[-2:]
    oh, ow = size
    if (ih, iw) == (oh, ow):
        return x
    if ih % oh == 0 and iw % ow == 0:
        return F.avg_pool2d(x, kernel_size=(ih // oh, iw // ow))
    if oh % ih == 0 and ow % iw == 0:
        rh, rw = oh // ih, ow // iw
        b, c = x.shape[:2]
        return (x[:, :, :, None, :, None].expand(b, c, ih, rh, iw, rw)
                .reshape(b, c, oh, ow))
    return F.adaptive_avg_pool2d(x, size)

def _ec_double_conv(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


def _ec_down(in_ch, out_ch):
    return nn.Sequential(
        nn.MaxPool2d(2, stride=2),
        _ec_double_conv(in_ch, out_ch),
    )


def _malf_add_conv(in_ch, out_ch, ksize, stride, leaky=True):
    stage = nn.Sequential()
    pad = (ksize - 1) // 2
    stage.add_module("conv", nn.Conv2d(in_ch, out_ch, ksize, stride, pad, bias=False))
    stage.add_module("batch_norm", nn.BatchNorm2d(out_ch))
    if leaky:
        stage.add_module("leaky", nn.LeakyReLU(0.1))
    else:
        stage.add_module("relu6", nn.ReLU6(inplace=True))
    return stage


class _MALF_ASPP(nn.Module):
    def __init__(self, in_channel=512, depth=256):
        super().__init__()
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block1  = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block6  = nn.Conv2d(in_channel, depth, 3, 1, padding=6,  dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)
        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)

    def forward(self, x):
        size = x.shape[2:]
        img_feat = F.interpolate(self.conv(self.mean(x)), size=size, mode="bilinear", align_corners=False)
        net = self.conv_1x1_output(torch.cat([
            img_feat,
            self.atrous_block1(x),
            self.atrous_block6(x),
            self.atrous_block12(x),
            self.atrous_block18(x),
        ], dim=1))
        return net


class _MALF_ASFF(nn.Module):
    def __init__(self, level, rfb=False, vis=False):
        super().__init__()
        self.level = level
        self.dim = [512, 256, 128]
        self.inter_dim = self.dim[self.level]
        compress_c = 8 if rfb else 16

        if level == 0:
            self.stride_level_1 = _malf_add_conv(256, self.inter_dim, 3, 2)
            self.stride_level_2 = _malf_add_conv(128, self.inter_dim, 3, 2)
            self.expand = _malf_add_conv(self.inter_dim, 512, 3, 1)
        elif level == 1:
            self.compress_level_0 = _malf_add_conv(512, self.inter_dim, 1, 1)
            self.stride_level_2   = _malf_add_conv(128, self.inter_dim, 3, 2)
            self.expand = _malf_add_conv(self.inter_dim, 256, 3, 1)
        elif level == 2:
            self.compress_level_0 = _malf_add_conv(512, self.inter_dim, 1, 1)
            self.compress_level_1 = _malf_add_conv(256, self.inter_dim, 1, 1)
            self.expand = _malf_add_conv(self.inter_dim, 128, 3, 1)

        self.weight_level_0 = _malf_add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = _malf_add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = _malf_add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_levels  = nn.Conv2d(compress_c * 3, 3, 1, 1, 0)
        self.vis = vis

    def forward(self, x_level_0, x_level_1, x_level_2):
        if self.level == 0:
            l0 = x_level_0
            l1 = self.stride_level_1(x_level_1)
            l2 = self.stride_level_2(F.max_pool2d(x_level_2, 3, stride=2, padding=1))
        elif self.level == 1:
            l0 = F.interpolate(self.compress_level_0(x_level_0), scale_factor=2, mode="nearest")
            l1 = x_level_1
            l2 = self.stride_level_2(x_level_2)
        elif self.level == 2:
            l0 = F.interpolate(self.compress_level_0(x_level_0), scale_factor=4, mode="nearest")
            l1 = F.interpolate(self.compress_level_1(x_level_1), scale_factor=2, mode="nearest")
            l2 = x_level_2

        anchor = l0 if self.level in (0, 1) else l2
        h, w = anchor.shape[2], anchor.shape[3]
        if l0.shape[2:] != anchor.shape[2:]:
            l0 = F.interpolate(l0, size=(h, w), mode="nearest")
        if l1.shape[2:] != anchor.shape[2:]:
            l1 = F.interpolate(l1, size=(h, w), mode="nearest")
        if l2.shape[2:] != anchor.shape[2:]:
            l2 = F.interpolate(l2, size=(h, w), mode="nearest")

        w_scores = F.softmax(self.weight_levels(torch.cat([
            self.weight_level_0(l0),
            self.weight_level_1(l1),
            self.weight_level_2(l2),
        ], dim=1)), dim=1)
        fused = l0 * w_scores[:, 0:1] + l1 * w_scores[:, 1:2] + l2 * w_scores[:, 2:]
        out = self.expand(fused)
        return out


class _BackboneAvgPoolShim(nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


class MALFBackbone(nn.Module):
    def __init__(self, in_channels: int = 2, base_c: int = 64):
        super().__init__()

        self.in_conv = _ec_down(in_channels, base_c)
        self.down1   = _ec_down(base_c,      base_c * 2)
        self.down2   = _ec_down(base_c * 2,  base_c * 4)
        self.down3   = _ec_down(base_c * 4,  base_c * 8)

        self.aspp1 = _MALF_ASPP(base_c,      base_c)
        self.aspp2 = _MALF_ASPP(base_c * 2,  base_c * 2)
        self.aspp3 = _MALF_ASPP(base_c * 4,  base_c * 4)
        self.aspp4 = _MALF_ASPP(base_c * 8,  base_c * 8)

        self.asff_0 = _MALF_ASFF(level=0)
        self.asff_1 = _MALF_ASFF(level=1)
        self.asff_2 = _MALF_ASFF(level=2)

        self.backbone = _BackboneAvgPoolShim()

    def forward(self, x):
        x1 = self.in_conv(x)
        x1 = self.aspp1(x1)

        x2 = self.down1(x1)
        x2 = self.aspp2(x2)
        feat2 = x2

        x3 = self.down2(x2)
        x3 = self.aspp3(x3)
        feat1 = x3

        x4 = self.down3(x3)
        x4 = self.aspp4(x4)
        feat0 = x4

        fused0 = self.asff_0(feat0, feat1, feat2)
        fused1 = self.asff_1(feat0, feat1, feat2)
        fused2 = self.asff_2(feat0, feat1, feat2)

        return [x1, fused2, fused1, fused0]

_ACT_CLS = 'gelu'


def _make_act(inplace=True):
    if _ACT_CLS == 'silu':
        return nn.SiLU(inplace=inplace)
    if _ACT_CLS == 'gelu':
        return nn.GELU()
    return nn.ReLU(inplace=inplace)


class _ActivationScope:
    def __init__(self, act_cls):
        self._new = act_cls
        self._prev = None

    def __enter__(self):
        global _ACT_CLS
        self._prev = _ACT_CLS
        _ACT_CLS = self._new
        return self

    def __exit__(self, exc_type, exc, tb):
        global _ACT_CLS
        _ACT_CLS = self._prev
        return False


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8, identity_init=True):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, max(1, channels // reduction))
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(max(1, channels // reduction), channels)
        self.sigmoid = nn.Sigmoid()

        if identity_init:
            nn.init.zeros_(self.fc2.weight)
            nn.init.constant_(self.fc2.bias, 5.0)

    def forward(self, x):
        w = self.pool(x).flatten(1)
        w = self.sigmoid(self.fc2(self.relu(self.fc1(w))))
        w = w.view(-1, x.size(1), 1, 1)
        return x * w


def unet_conv(input_nc, output_nc, norm_layer=nn.BatchNorm2d, dropout=0, double=False):
    downconv = nn.Conv2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1)
    downrelu = nn.LeakyReLU(0.2, True)
    downnorm = norm_layer(output_nc)

    if dropout > 0:
        layers = [downconv, downrelu, nn.Dropout2d(dropout), downnorm]
    else:
        layers = [downconv, downnorm, downrelu]

    if double:
        conv = [nn.Conv2d(output_nc, output_nc, kernel_size=3, padding=1),
                downnorm, downrelu]
        layers.append(conv)
    return nn.Sequential(*layers)


def replace_batchnorm(module, norm_type='groupnorm', num_groups=8):
    if norm_type == 'batchnorm':
        return module
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            C = child.num_features
            if norm_type == 'groupnorm':
                g = min(num_groups, C)
                while C % g != 0 and g > 1:
                    g -= 1
                new_layer = nn.GroupNorm(num_groups=g, num_channels=C,
                                         eps=child.eps, affine=child.affine)
            elif norm_type == 'layernorm':
                new_layer = nn.GroupNorm(num_groups=1, num_channels=C,
                                         eps=child.eps, affine=child.affine)
            elif norm_type == 'instancenorm':
                new_layer = nn.GroupNorm(num_groups=C, num_channels=C,
                                         eps=child.eps, affine=child.affine)
            else:
                raise ValueError(f'unknown norm_type: {norm_type}')
            if child.affine and new_layer.affine:
                with torch.no_grad():
                    new_layer.weight.copy_(child.weight)
                    new_layer.bias.copy_(child.bias)
            setattr(module, name, new_layer)
        else:
            replace_batchnorm(child, norm_type=norm_type, num_groups=num_groups)
    return module


def unet_upconv(input_nc, output_nc, outermost=False, norm_layer=nn.BatchNorm2d,
                dropout=0., outermost_activation='sigmoid'):
    upconv = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1)
    uprelu = _make_act(inplace=True)
    upnorm = norm_layer(output_nc)

    if not outermost:
        if dropout > 0:
            return nn.Sequential(*[upconv, upnorm, uprelu, nn.Dropout2d(dropout)])
        else:
            return nn.Sequential(*[upconv, upnorm, uprelu])
    else:
        if outermost_activation == 'none':
            return nn.Sequential(upconv)
        return nn.Sequential(upconv, nn.Sigmoid())


def create_conv(input_channels, output_channels, kernel, paddings,
                batch_norm=True, Relu=True, stride=1, groups=1):
    model = [nn.Conv2d(input_channels, output_channels, kernel,
                       stride=stride, padding=paddings, groups=groups)]
    if batch_norm:
        model.append(nn.BatchNorm2d(output_channels))
    if Relu:
        model.append(_make_act(inplace=True))
    return nn.Sequential(*model)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)


class DoubleConv(nn.Module):
    def __init__(self, in_channel, out_channel, mid_channel=None):
        super(DoubleConv, self).__init__()
        if not mid_channel:
            mid_channel = out_channel
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channel),
            _make_act(inplace=True),
            nn.Conv2d(mid_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            _make_act(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class ResDoubleConv(nn.Module):
    def __init__(self, in_channel, out_channel, mid_channel=None):
        super(ResDoubleConv, self).__init__()
        if not mid_channel:
            mid_channel = out_channel
        self.conv1 = nn.Conv2d(in_channel, mid_channel, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channel)
        self.conv2 = nn.Conv2d(mid_channel, out_channel, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        if in_channel != out_channel:
            self.proj = nn.Conv2d(in_channel, out_channel, 1, bias=False)
        else:
            self.proj = nn.Identity()
        self.act = _make_act(inplace=True)

    def forward(self, x):
        h = self.act(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return self.act(h + self.proj(x))


class MaterialHead(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout=.5, h_class=23, norm=False):
        super(MaterialHead, self).__init__()
        layers = []
        for i, hidden_dim in enumerate(hidden_dims):
            in_dim = input_dim if i == 0 else hidden_dims[i - 1]
            layers.append(nn.Linear(in_dim, hidden_dim))
            if norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.LeakyReLU(0.2, True))
            if dropout > 0.:
                layers.append(nn.Dropout1d(dropout))
        layers.append(nn.Linear(hidden_dims[-1], h_class))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class ProjectionHead(nn.Module):
    def __init__(self, in_fc, out_fc, use_ln=False):
        super(ProjectionHead, self).__init__()
        self.use_ln = use_ln
        if use_ln:
            hidden = in_fc
            self.net = nn.Sequential(
                nn.Linear(in_fc, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Linear(hidden, out_fc),
                nn.LayerNorm(out_fc),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(in_fc, in_fc // 2),
                nn.BatchNorm1d(in_fc // 2),
                nn.ReLU(),
                nn.Linear(in_fc // 2, out_fc),
                nn.BatchNorm1d(out_fc)
            )
        self.__weight_init()

    def __weight_init(self):
        for _, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        return self.net(x)


class SimpleAudioDepthNet(nn.Module):
    def __init__(self, audio_shape, audio_feature_length=512, backbone='Resnet18', mode="base",
                 use_se_skips=False, decoder_spatial_entry=False, use_silu=False,
                 use_residual_decoder=False, wider_decoder=False, use_log_depth=False,
                 max_depth=10.0, decoder_dropout=0.1, target_hw=(128, 128),
                 audio_norm_type='batchnorm', audio_norm_groups=8, mat_nclass=23):
        super().__init__()
        self.mode = mode
        self._decoder_spatial_entry = bool(decoder_spatial_entry)
        self._use_se_skips = use_se_skips
        self._use_log_depth = bool(use_log_depth)
        self._max_depth = float(max_depth)
        self._target_hw = tuple(target_hw)
        n_in = audio_shape[0]
        d1, d2, d3, d4, d5 = (512, 384, 256, 192, 96) if wider_decoder else (512, 256, 128, 64, 32)
        s2, s3, s4 = 256, 128, 64
        _DC = ResDoubleConv if use_residual_decoder else DoubleConv

        with _ActivationScope('silu' if use_silu else 'relu'):
            if backbone == "Resnet18":
                self.feature_extraction = BinauralResNet18(in_channel=n_in, pretrained=True)
                self.echo_avgpool = nn.AdaptiveAvgPool2d((1, 1))
                self.conv1x1 = nn.Sequential(create_conv(512, audio_feature_length, 1, 0))
            elif backbone == "MALFNet":
                self.feature_extraction = MALFBackbone(in_channels=n_in)
                self.conv1x1 = nn.Sequential(create_conv(512, audio_feature_length, 1, 0))
            elif backbone == "Resnet34":
                self.feature_extraction = BinauralResNet34(in_channel=n_in, pretrained=True)
                self.conv1x1 = create_conv(512, audio_feature_length, 1, 0)
            else:
                raise ValueError(f'unknown backbone {backbone}')

            self.rgbdepth_conv2d_feat1 = nn.Conv2d(audio_feature_length, 512, kernel_size=1)
            if self._decoder_spatial_entry:
                self.film_gamma = nn.Conv2d(audio_feature_length, 512, kernel_size=1)
                self.film_beta = nn.Conv2d(audio_feature_length, 512, kernel_size=1)
                nn.init.zeros_(self.film_gamma.weight)
                nn.init.ones_(self.film_gamma.bias)
                nn.init.zeros_(self.film_beta.weight)
                nn.init.zeros_(self.film_beta.bias)

            p = float(decoder_dropout)
            self.rgbdepth_upconvlayer1 = unet_upconv(512 + 512, d1, dropout=p)
            self.rgbdepth_double_conv_layer2 = _DC(d1, d2)
            self.rgbdepth_upconvlayer2 = unet_upconv(d2 + s2, d2, dropout=p)
            self.rgbdepth_double_conv_layer3 = _DC(d2, d3)
            self.rgbdepth_upconvlayer3 = unet_upconv(d3 + s3, d3, dropout=p)
            self.rgbdepth_double_conv_layer4 = _DC(d3, d4)
            self.rgbdepth_upconvlayer4 = unet_upconv(d4 + s4, d4, dropout=p)
            self.rgbdepth_double_conv_layer5 = _DC(d4, d5)
            self.rgbdepth_upconvlayer5 = unet_upconv(
                d5, 1, outermost=True, outermost_activation='none' if self._use_log_depth else 'sigmoid')

            self.se_skip2 = SEBlock(s2, reduction=8, identity_init=True)
            self.se_skip3 = SEBlock(s3, reduction=8, identity_init=True)
            self.se_skip4 = SEBlock(s4, reduction=8, identity_init=True)

            if self.mode == "mat":
                self.matnet_head = MaterialHead(input_dim=audio_feature_length, hidden_dims=[256, 128, 64],
                                                h_class=mat_nclass, norm=False)
                self.matnet_softmax = nn.Softmax(dim=1)

        self._init_weights()
        if audio_norm_type and audio_norm_type != 'batchnorm':
            replace_batchnorm(self, norm_type=audio_norm_type, num_groups=int(audio_norm_groups))

    def _init_weights(self):
        for name, m in self.named_modules():
            if any(p in name for p in ("feature_extraction", "se_skip", "film_")):
                continue
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)
            if self.mode == "mat":
                for h in self.matnet_head.modules():
                    if isinstance(h, nn.Linear):
                        nn.init.kaiming_normal_(h.weight, mode='fan_in', nonlinearity='leaky_relu')
                        if h.bias is not None:
                            nn.init.constant_(h.bias, 0)

    @staticmethod
    def _resize(feat, ref):
        if feat.shape[2:] == ref.shape[2:]:
            return feat
        return F.interpolate(feat, size=ref.shape[2:], mode='bilinear', align_corners=False)

    def _forward_depth(self, entry, features, audio_feat):
        f4 = self.rgbdepth_conv2d_feat1(features[3])
        if self._decoder_spatial_entry:
            f4 = self.film_gamma(audio_feat) * f4 + self.film_beta(audio_feat)
        if entry.size(2) == 1 and entry.size(3) == 1:
            entry = entry.expand(-1, -1, f4.size(2), f4.size(3))
        u = self.rgbdepth_upconvlayer1(torch.cat((entry, f4), dim=1))
        for i, (feat, se) in enumerate(((features[2], self.se_skip2), (features[1], self.se_skip3),
                                        (features[0], self.se_skip4)), start=2):
            skip = self._resize(feat, u)
            if self._use_se_skips:
                skip = se(skip)
            dbl = getattr(self, f'rgbdepth_double_conv_layer{i}')(u)
            u = getattr(self, f'rgbdepth_upconvlayer{i}')(torch.cat((dbl, skip), dim=1))
        out = self.rgbdepth_upconvlayer5(self.rgbdepth_double_conv_layer5(u))
        if out.shape[-2:] != torch.Size(self._target_hw):
            out = F.interpolate(out, size=self._target_hw, mode='bilinear', align_corners=False)
        if self._use_log_depth:
            out = torch.exp(out.clamp(min=-5.0, max=math.log(max(self._max_depth, 1e-6))))
            out = out.clamp(min=0.0, max=self._max_depth) / self._max_depth
        return out

    def forward(self, x):
        features = self.feature_extraction(x)
        audio_feat = self.conv1x1(features[-1].mean((2, 3), keepdim=True))
        entry = features[-1] if self._decoder_spatial_entry else audio_feat
        if self.mode == "mat":
            mat = self.matnet_head(audio_feat.mean((2, 3)))
            return self._forward_depth(entry, features, audio_feat), mat, audio_feat
        return self._forward_depth(entry, features, audio_feat), audio_feat


class LegacyAudioDepthNet(nn.Module):
    _KERNEL_SIZES = [(8, 8), (4, 4), (3, 3)]
    _STRIDES = [(4, 4), (2, 2), (1, 1)]

    def __init__(self, audio_shape, audio_feature_length=512, output_nc=1, mode="base"):
        super().__init__()
        self.mode = mode
        n_ch = audio_shape[0]

        self.conv1 = create_conv(n_ch, 32, kernel=self._KERNEL_SIZES[0],
                                 paddings=0, stride=self._STRIDES[0])
        self.conv2 = create_conv(32, 64, kernel=self._KERNEL_SIZES[1],
                                 paddings=0, stride=self._STRIDES[1])
        self.conv3 = create_conv(64, 8, kernel=self._KERNEL_SIZES[2],
                                 paddings=0, stride=self._STRIDES[2])
        self.feature_extraction = nn.Sequential(self.conv1, self.conv2, self.conv3)

        flat_dim = self._compute_flat_dim(audio_shape[1:])
        self.conv1x1 = create_conv(flat_dim, audio_feature_length, 1, 0)

        self.rgbdepth_upconvlayer1 = unet_upconv(audio_feature_length, 512)
        self.rgbdepth_upconvlayer2 = unet_upconv(512, 256)
        self.rgbdepth_upconvlayer3 = unet_upconv(256, 128)
        self.rgbdepth_upconvlayer4 = unet_upconv(128, 64)
        self.rgbdepth_upconvlayer5 = unet_upconv(64, 32)
        self.rgbdepth_upconvlayer6 = unet_upconv(32, 16)
        self.rgbdepth_upconvlayer7 = unet_upconv(16, output_nc, outermost=True)

        if mode == "mat":
            self.matnet_head = MaterialHead(
                input_dim=audio_feature_length, hidden_dims=[256, 128, 64], norm=False)

    def _compute_flat_dim(self, spatial_shape):
        dims = np.array(spatial_shape, dtype=np.float32)
        for k, s in zip(self._KERNEL_SIZES, self._STRIDES):
            dims = np.floor((dims - np.array(k)) / np.array(s)) + 1
        return int(8 * dims[0] * dims[1])

    def forward(self, x):
        feat = self.feature_extraction(x)
        feat_flat = feat.view(feat.shape[0], -1, 1, 1)
        audio_feat = self.conv1x1(feat_flat)

        u1 = self.rgbdepth_upconvlayer1(audio_feat)
        u2 = self.rgbdepth_upconvlayer2(u1)
        u3 = self.rgbdepth_upconvlayer3(u2)
        u4 = self.rgbdepth_upconvlayer4(u3)
        u5 = self.rgbdepth_upconvlayer5(u4)
        u6 = self.rgbdepth_upconvlayer6(u5)
        depth = self.rgbdepth_upconvlayer7(u6)

        if self.mode == "mat":
            mat = self.matnet_head(audio_feat.flatten(start_dim=1))
            return depth, mat, audio_feat
        else:
            return depth, audio_feat


class RGBDepthNet(nn.Module):
    def __init__(self, ngf=64, input_nc=3, output_nc=1):
        super(RGBDepthNet, self).__init__()
        self.rgbdepth_convlayer1 = unet_conv(input_nc, ngf)
        self.rgbdepth_convlayer2 = unet_conv(ngf, ngf * 2)
        self.rgbdepth_convlayer3 = unet_conv(ngf * 2, ngf * 4)
        self.rgbdepth_convlayer4 = unet_conv(ngf * 4, ngf * 8)
        self.rgbdepth_convlayer5 = unet_conv(ngf * 8, ngf * 8)
        self.rgbdepth_upconvlayer1 = unet_upconv(512, ngf * 8)
        self.rgbdepth_upconvlayer2 = unet_upconv(ngf * 16, ngf * 4)
        self.rgbdepth_upconvlayer3 = unet_upconv(ngf * 8, ngf * 2)
        self.rgbdepth_upconvlayer4 = unet_upconv(ngf * 4, ngf)
        self.rgbdepth_upconvlayer5 = unet_upconv(ngf * 2, output_nc, True)

    def forward(self, x):
        rgbdepth_conv1feature = self.rgbdepth_convlayer1(x)
        rgbdepth_conv2feature = self.rgbdepth_convlayer2(rgbdepth_conv1feature)
        rgbdepth_conv3feature = self.rgbdepth_convlayer3(rgbdepth_conv2feature)
        rgbdepth_conv4feature = self.rgbdepth_convlayer4(rgbdepth_conv3feature)
        rgbdepth_conv5feature = self.rgbdepth_convlayer5(rgbdepth_conv4feature)

        rgbdepth_upconv1feature = self.rgbdepth_upconvlayer1(rgbdepth_conv5feature)
        rgbdepth_upconv2feature = self.rgbdepth_upconvlayer2(
            torch.cat((rgbdepth_upconv1feature, rgbdepth_conv4feature), dim=1))
        rgbdepth_upconv3feature = self.rgbdepth_upconvlayer3(
            torch.cat((rgbdepth_upconv2feature, rgbdepth_conv3feature), dim=1))
        rgbdepth_upconv4feature = self.rgbdepth_upconvlayer4(
            torch.cat((rgbdepth_upconv3feature, rgbdepth_conv2feature), dim=1))
        depth_prediction = self.rgbdepth_upconvlayer5(
            torch.cat((rgbdepth_upconv4feature, rgbdepth_conv1feature), dim=1))
        return depth_prediction, rgbdepth_conv5feature


class MaterialPropertyNet(nn.Module):
    def __init__(self, nclass, backbone):
        super(MaterialPropertyNet, self).__init__()
        self.pretrained = backbone
        self.pool = nn.AvgPool2d(4)
        self.fc = nn.Linear(512, nclass)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)
        x = self.pretrained.layer1(x)
        x = self.pretrained.layer2(x)
        x = self.pretrained.layer3(x)
        feat = self.pretrained.layer4(x)
        x = self.pool(feat)
        x = x.view(-1, 512)
        x = self.fc(x)
        return x, feat


class CompositionalEmbedding(nn.Module):
    def __init__(self, input_dim, out_nc=256, head=None,
                 normalization_sign=False, use_film=False, n_class=23):
        super().__init__()
        self.normalization_sign = normalization_sign
        self.head = head
        self.out_nc = out_nc
        self.use_film = use_film

        if self.use_film:
            self.film = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.GELU(),
                nn.Linear(input_dim, 2 * input_dim),
            )
            nn.init.zeros_(self.film[-1].weight)
            nn.init.zeros_(self.film[-1].bias)

        if self.head == "MatNet":
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, out_nc),
                nn.ReLU(inplace=True),
            )
            self.fc = nn.Linear(out_nc, n_class)
            self.mat_softmax = nn.Softmax(dim=1)
            if use_film:
                self.film_gamma = nn.Linear(input_dim, input_dim)
                self.film_beta = nn.Linear(input_dim, input_dim)

        elif self.head == "DepthNet":
            self.mlp = nn.Conv2d(input_dim, out_nc, kernel_size=1)
            self.upconv_layers = nn.Sequential(
                unet_upconv(out_nc, 256),
                unet_upconv(256, 128),
                unet_upconv(128, 64),
                unet_upconv(64, 32),
            )
            self.upconv_head = unet_upconv(32, 1, outermost=True)
            if use_film:
                self.film_gamma = nn.Conv2d(input_dim, input_dim, kernel_size=1)
                self.film_beta = nn.Conv2d(input_dim, input_dim, kernel_size=1)

        if use_film:
            nn.init.zeros_(self.film_gamma.weight)
            nn.init.zeros_(self.film_beta.weight)
            if self.film_gamma.bias is not None:
                nn.init.zeros_(self.film_gamma.bias)
            if self.film_beta.bias is not None:
                nn.init.zeros_(self.film_beta.bias)

    def _film_fuse(self, f_audio, f_teacher):
        gamma = self.film_gamma(f_audio)
        beta = self.film_beta(f_audio)
        return f_teacher * (1.0 + gamma) + beta

    def _fuse(self, f1, f2):
        if not self.use_film:
            return f1 + f2

        audio_global = f1.mean((2, 3))
        gamma_beta = self.film(audio_global)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = gamma[:, :, None, None]
        beta  = beta[:, :, None, None]
        return f2 * (1.0 + gamma) + beta

    def forward(self, f1, f2):
        if self.normalization_sign:
            f1 = F.normalize(f1, dim=1)
            f2 = F.normalize(f2, dim=1)

        if self.head == "MatNet":
            if self.use_film:
                fused = self._fuse(f1, f2)
                fused_p = fused.mean((2, 3))
                feat = self.mlp(fused_p)
            else:
                f1_p = f1.mean((2, 3))
                f2_p = f2.mean((2, 3))
                feat = self.mlp(f1_p + f2_p)
            out = self.mat_softmax(self.fc(feat))
            return out, feat

        elif self.head == "DepthNet":
            f1 = _adaptive_avg_pool2d_det(f1, (f2.shape[2], f2.shape[3]))
            fused = self._fuse(f1, f2)
            feat = F.relu(self.mlp(fused))
            out = _adaptive_avg_pool2d_det(
                self.upconv_head(self.upconv_layers(feat)), (128, 128))
            return out, feat
