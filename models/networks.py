import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone import BinauralResNet18, BinauralResNet34
import numpy as np
import os


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention on skip connections."""

    def __init__(self, channels, reduction=8):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(1, channels // reduction)),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, channels // reduction), channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.pool(x).flatten(1)
        w = self.fc(w).view(-1, x.size(1), 1, 1)
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


def unet_upconv(input_nc, output_nc, outermost=False, norm_layer=nn.BatchNorm2d, dropout=0.):
    upconv = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1)
    uprelu = nn.ReLU(True)
    upnorm = norm_layer(output_nc)

    if not outermost:
        if dropout > 0:
            return nn.Sequential(*[upconv, upnorm, uprelu, nn.Dropout2d(dropout)])
        else:
            return nn.Sequential(*[upconv, upnorm, uprelu])
    else:
        return nn.Sequential(*[upconv, nn.Sigmoid()])


def create_conv(input_channels, output_channels, kernel, paddings,
                batch_norm=True, Relu=True, stride=1, groups=1):
    model = [nn.Conv2d(input_channels, output_channels, kernel,
                       stride=stride, padding=paddings, groups=groups)]
    if batch_norm:
        model.append(nn.BatchNorm2d(output_channels))
    if Relu:
        model.append(nn.ReLU())
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


def freeze_weight(net):
    for m in net.parameters():
        m.requires_grad = False


class DoubleConv(nn.Module):
    def __init__(self, in_channel, out_channel, mid_channel=None):
        super(DoubleConv, self).__init__()
        if not mid_channel:
            mid_channel = out_channel
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


# ---------------------------------------------------------------------------
# Sub-networks
# ---------------------------------------------------------------------------

class MaterialHead(nn.Module):
    """MLP head for material classification (23 MINC classes)."""

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
    """Projection head for contrastive learning."""

    def __init__(self, in_fc, out_fc):
        super(ProjectionHead, self).__init__()
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


# ---------------------------------------------------------------------------
# Audio depth network (ResNet backbone + UNet decoder)
# ---------------------------------------------------------------------------

class SimpleAudioDepthNet(nn.Module):
    """Audio-to-depth network with ResNet encoder and UNet decoder with skip connections.

    Adapted from Visual Echoes [Gao et al., ECCV 2020].
    """

    def __init__(self, conv1x1_dim, audio_shape, audio_feature_length=512,
                 output_nc=1, backbone=None, mode="base"):
        super(SimpleAudioDepthNet, self).__init__()

        self.mode = mode
        self._backbone = backbone
        self._n_input_audio = audio_shape[0]
        self._output_nc = output_nc
        self._cnn_layers_kernel_size = [(8, 8), (4, 4), (3, 3)]
        self._cnn_layers_stride = [(4, 4), (2, 2), (1, 1)]

        cnn_dims = np.array(audio_shape[1:], dtype=np.float32)
        for kernel_size, stride in zip(
            self._cnn_layers_kernel_size, self._cnn_layers_stride
        ):
            cnn_dims = self._conv_output_dim(
                dimension=cnn_dims,
                padding=np.array([0, 0], dtype=np.float32),
                dilation=np.array([1, 1], dtype=np.float32),
                kernel_size=np.array(kernel_size, dtype=np.float32),
                stride=np.array(stride, dtype=np.float32),
            )

        if self._backbone == "Resnet18":
            self.feature_extraction = BinauralResNet18(
                in_channel=self._n_input_audio, pretrained=True)
            self.echo_avgpool = nn.AdaptiveAvgPool2d((1, 1))
            in_conv1x1 = 512
            self.conv1x1 = nn.Sequential(
                create_conv(in_conv1x1, audio_feature_length, 1, 0)
            )
        elif self._backbone == "Resnet34":
            self.feature_extraction = BinauralResNet34(
                in_channel=self._n_input_audio, pretrained=True)
            self.conv1x1 = create_conv(512, audio_feature_length, 1, 0)
        else:
            self.conv1 = create_conv(self._n_input_audio, 32,
                                     kernel=self._cnn_layers_kernel_size[0],
                                     paddings=0, stride=self._cnn_layers_stride[0])
            self.conv2 = create_conv(32, 64,
                                     kernel=self._cnn_layers_kernel_size[1],
                                     paddings=0, stride=self._cnn_layers_stride[1])
            self.conv3 = create_conv(64, conv1x1_dim,
                                     kernel=self._cnn_layers_kernel_size[2],
                                     paddings=0, stride=self._cnn_layers_stride[2])
            self.feature_extraction = nn.Sequential(self.conv1, self.conv2, self.conv3)
            self.conv1x1 = create_conv(2464, audio_feature_length, 1, 0)

        # UNet decoder with skip connections
        _dropout = 0.1
        self.rgbdepth_conv2d_feat1 = nn.Conv2d(
            in_channels=audio_feature_length, out_channels=512,
            kernel_size=1, stride=1, padding=0)
        self.rgbdepth_upconvlayer1 = unet_upconv(512 * 2, 512, dropout=_dropout)

        self.rgbdepth_double_conv_layer2 = DoubleConv(512, 256)
        self.rgbdepth_upconvlayer2 = unet_upconv(256 * 2, 256, dropout=_dropout)

        self.rgbdepth_double_conv_layer3 = DoubleConv(256, 128)
        self.rgbdepth_upconvlayer3 = unet_upconv(128 * 2, 128, dropout=_dropout)

        self.rgbdepth_double_conv_layer4 = DoubleConv(128, 64)
        self.rgbdepth_upconvlayer4 = unet_upconv(64 * 2, 64, dropout=_dropout)

        self.rgbdepth_double_conv_layer5 = DoubleConv(64, 32)
        self.rgbdepth_upconvlayer5 = unet_upconv(32, output_nc, True)

        # SE blocks for skip connection attention
        self._use_se_skips = False
        self.se_skip2 = SEBlock(256, reduction=8)
        self.se_skip3 = SEBlock(128, reduction=8)
        self.se_skip4 = SEBlock(64, reduction=8)

        if self.mode == "mat":
            hidden_dims = [256, 128, 64]
            self.matnet_head = MaterialHead(
                input_dim=audio_feature_length, hidden_dims=hidden_dims, norm=False)
            self.matnet_softmax = nn.Softmax(dim=1)

        self.__weight_init()

    def __weight_init(self):
        for name, m in self.named_modules():
            if "feature_extraction" in name:
                continue
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

            if self.mode == "mat":
                for m in self.matnet_head.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.kaiming_normal_(m.weight, mode='fan_in',
                                                nonlinearity='leaky_relu')
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)

    def _conv_output_dim(self, dimension, padding, dilation, kernel_size, stride):
        assert len(dimension) == 2
        out_dimension = []
        for i in range(len(dimension)):
            out_dimension.append(
                int(np.floor(((dimension[i] + 2 * padding[i]
                               - dilation[i] * (kernel_size[i] - 1) - 1) / stride[i]) + 1)))
        return tuple(out_dimension)

    @staticmethod
    def _resize_skip(feat, target_h, target_w):
        """Resize skip-connection feature to match decoder spatial dims."""
        if feat.size(2) == target_h and feat.size(3) == target_w:
            return feat
        return F.interpolate(feat, size=(target_h, target_w),
                             mode='bilinear', align_corners=False)

    def _forward_depth(self, audio_feat, features):
        feat4_conv2d = self.rgbdepth_conv2d_feat1(features[3])
        audio_feat = audio_feat.expand(-1, -1, feat4_conv2d.size(2), feat4_conv2d.size(3))
        rgbdepth_upconv1feature = self.rgbdepth_upconvlayer1(
            torch.cat((audio_feat, feat4_conv2d), dim=1))

        x_4 = self._resize_skip(features[2],
                                 rgbdepth_upconv1feature.size(2),
                                 rgbdepth_upconv1feature.size(3))
        if self._use_se_skips:
            x_4 = self.se_skip2(x_4)
        rgbdepth_upconv2feature_dbl = self.rgbdepth_double_conv_layer2(rgbdepth_upconv1feature)
        rgbdepth_upconv2feature = self.rgbdepth_upconvlayer2(
            torch.cat((rgbdepth_upconv2feature_dbl, x_4), dim=1))

        x_3 = self._resize_skip(features[1],
                                 rgbdepth_upconv2feature.size(2),
                                 rgbdepth_upconv2feature.size(3))
        if self._use_se_skips:
            x_3 = self.se_skip3(x_3)
        rgbdepth_upconv3feature_dbl = self.rgbdepth_double_conv_layer3(rgbdepth_upconv2feature)
        rgbdepth_upconv3feature = self.rgbdepth_upconvlayer3(
            torch.cat((rgbdepth_upconv3feature_dbl, x_3), dim=1))

        x_2 = self._resize_skip(features[0],
                                 rgbdepth_upconv3feature.size(2),
                                 rgbdepth_upconv3feature.size(3))
        if self._use_se_skips:
            x_2 = self.se_skip4(x_2)
        rgbdepth_upconv4feature_dbl = self.rgbdepth_double_conv_layer4(rgbdepth_upconv3feature)
        rgbdepth_upconv4feature = self.rgbdepth_upconvlayer4(
            torch.cat((rgbdepth_upconv4feature_dbl, x_2), dim=1))

        rgbdepth_upconv5feature_dbl = self.rgbdepth_double_conv_layer5(rgbdepth_upconv4feature)
        final_layer_out = self.rgbdepth_upconvlayer5(rgbdepth_upconv5feature_dbl)

        if final_layer_out.shape[-2:] != torch.Size([128, 128]):
            final_layer_out = F.interpolate(final_layer_out, size=(128, 128),
                                            mode='bilinear', align_corners=False)

        if self._output_nc == 2:
            depth_prediction = torch.unsqueeze(final_layer_out[:, 0, :, :], axis=1)
            sigma = torch.unsqueeze(final_layer_out[:, 1, :, :], axis=1)
            return depth_prediction, sigma
        else:
            return final_layer_out

    def enable_gradient_checkpointing(self):
        """Trade compute for VRAM via gradient checkpointing on backbone + decoder."""
        import torch.utils.checkpoint as cp

        _orig_backbone = self.feature_extraction.forward
        def _checkpointed_backbone(x):
            return cp.checkpoint(_orig_backbone, x, use_reentrant=False)
        self.feature_extraction.forward = _checkpointed_backbone

        _orig_depth = self._forward_depth
        def _checkpointed_depth(audio_feat, features):
            def _inner(audio_feat, *features_tuple):
                return _orig_depth(audio_feat, list(features_tuple))
            return cp.checkpoint(_inner, audio_feat, *features, use_reentrant=False)
        self._forward_depth = _checkpointed_depth

    def forward(self, x):
        features = self.feature_extraction(x)
        x_pool = self.feature_extraction.backbone.avgpool(features[-1])
        x = self.conv1x1(x_pool)
        audio_feat = x

        if self.mode == "mat":
            mat_prediction = self.matnet_head(torch.flatten(audio_feat, start_dim=1))
            if self._output_nc == 2:
                depth_prediction, sigma = self._forward_depth(audio_feat, features)
                return depth_prediction, sigma, mat_prediction, audio_feat
            else:
                depth_prediction = self._forward_depth(audio_feat, features)
                return depth_prediction, mat_prediction, audio_feat
        elif self.mode == "base":
            if self._output_nc == 2:
                depth_prediction, sigma = self._forward_depth(audio_feat, features)
                return depth_prediction, sigma, audio_feat
            else:
                depth_prediction = self._forward_depth(audio_feat, features)
                return depth_prediction, audio_feat


class LegacyAudioDepthNet(nn.Module):
    """Legacy audio depth network matching the pretrained audiodepth_replica.pth architecture.

    Uses a 3-layer CNN backbone and 7-layer upconv decoder without skip connections.
    audio_shape=[2,257,166] requires hop_length=16 at 44.1kHz / audio_length=0.06s.
    """

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

    def enable_gradient_checkpointing(self):
        pass


# ---------------------------------------------------------------------------
# RGB depth teacher network
# ---------------------------------------------------------------------------

class RGBDepthNet(nn.Module):
    """UNet-based RGB-to-depth teacher network."""

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


# ---------------------------------------------------------------------------
# Material property network
# ---------------------------------------------------------------------------

class MaterialPropertyNet(nn.Module):
    """ResNet-18 based material classification network (pretrained on MINC)."""

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
    """Compositional Contrastive Learning (CCL) embedding head.

    Composes audio and teacher features via elementwise addition,
    then decodes to the target domain (depth map or material class).

    Adapted from [Yang et al., CCL].
    """

    def __init__(self, input_dim, out_nc=256, head=None, normalization_sign=False):
        super().__init__()
        self.normalization_sign = normalization_sign
        self.head = head
        self.out_nc = out_nc

        if self.head == "MatNet":
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, out_nc),
                nn.ReLU(inplace=True),
            )
            self.fc = nn.Linear(out_nc, 23)
            self.mat_softmax = nn.Softmax(dim=1)

        elif self.head == "DepthNet":
            self.mlp = nn.Conv2d(input_dim, out_nc, kernel_size=1)
            self.upconv_layers = nn.Sequential(
                unet_upconv(out_nc, 256),
                unet_upconv(256, 128),
                unet_upconv(128, 64),
                unet_upconv(64, 32),
            )
            self.upconv_head = unet_upconv(32, 1, outermost=True)

    def forward(self, f1, f2):
        if self.normalization_sign:
            f1 = F.normalize(f1, dim=1)
            f2 = F.normalize(f2, dim=1)

        if self.head == "MatNet":
            f1_p = F.adaptive_avg_pool2d(f1, (1, 1)).flatten(1)
            f2_p = F.adaptive_avg_pool2d(f2, (1, 1)).flatten(1)
            feat = self.mlp(f1_p + f2_p)
            out = self.mat_softmax(self.fc(feat))
            return out, feat

        elif self.head == "DepthNet":
            f1 = F.adaptive_avg_pool2d(f1, (f2.shape[2], f2.shape[3]))
            feat = F.relu(self.mlp(f1 + f2))
            out = F.adaptive_avg_pool2d(
                self.upconv_head(self.upconv_layers(feat)), (128, 128))
            return out, feat


