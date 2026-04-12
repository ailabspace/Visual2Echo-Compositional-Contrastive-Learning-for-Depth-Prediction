import os
import torch
import torchvision
import torch.nn as nn
from collections import OrderedDict
from .networks import (RGBDepthNet, weights_init, SimpleAudioDepthNet,
                        LegacyAudioDepthNet, MaterialPropertyNet,
                        CompositionalEmbedding, ProjectionHead)


class ModelBuilder():
    """Factory for building all network components."""

    def build_audiodepth(self, audio_shape=[2, 257, 166], out_nc=1,
                         weights='', backbone=None, mode='base'):
        if backbone == "Legacy":
            net = LegacyAudioDepthNet(
                audio_shape=audio_shape, audio_feature_length=512,
                output_nc=out_nc, mode=mode)
        else:
            net = SimpleAudioDepthNet(
                8, audio_shape=audio_shape, audio_feature_length=512,
                output_nc=out_nc, backbone=backbone, mode=mode)
        if len(weights) > 0:
            print('Loading weights for audio stream')
            sd = torch.load(weights, map_location='cpu')
            model_sd = net.state_dict()
            compatible = {k: v for k, v in sd.items()
                          if k in model_sd and v.shape == model_sd[k].shape}
            incompatible_shapes = {k: (v.shape, model_sd[k].shape)
                                   for k, v in sd.items()
                                   if k in model_sd and v.shape != model_sd[k].shape}
            if incompatible_shapes:
                print(f'  Skipped {len(incompatible_shapes)} keys with shape mismatch')
            missing, unexpected = net.load_state_dict(compatible, strict=False)
            print(f'  Loaded {len(compatible)}/{len(model_sd)} keys '
                  f'({len(missing)} missing, {len(unexpected)} unexpected)')
        return net

    def build_rgbdepth(self, ngf=64, input_nc=3, output_nc=1, weights=''):
        net = RGBDepthNet(ngf, input_nc, output_nc)
        net.apply(weights_init)
        if len(weights) > 0:
            if not os.path.isfile(weights):
                print(f'[WARN] RGB teacher checkpoint not found: {weights}')
            else:
                print('Loading weights for visual stream')
                net.load_state_dict(torch.load(weights, map_location='cpu'))
        return net

    def build_ccl(self, head=None, input_dim=None, out_nc=256,
                  weights='', normalization_sign=False):
        assert input_dim is not None
        net = CompositionalEmbedding(
            head=head, input_dim=input_dim, out_nc=out_nc,
            normalization_sign=normalization_sign)
        net.apply(weights_init)
        if len(weights) > 0:
            print('Loading weights for CCL stream')
            net.load_state_dict(torch.load(weights))
        return net

    def build_material_property(self, nclass=10, weights='', init_weights=''):
        if len(init_weights) > 0:
            original_resnet = torchvision.models.resnet18(
                weights=torchvision.models.ResNet18_Weights.DEFAULT)
            net = MaterialPropertyNet(23, original_resnet)
            pre_trained_dict = torch.load(init_weights, weights_only=False)['state_dict']
            pre_trained_mod_dict = OrderedDict()
            for k, v in pre_trained_dict.items():
                new_key = '.'.join(k.split('.')[1:])
                pre_trained_mod_dict[new_key] = v
            pre_trained_mod_dict = {k: v for k, v in pre_trained_mod_dict.items()
                                    if k in net.state_dict()}
            net.load_state_dict(pre_trained_mod_dict, strict=False)
            print('Material Property Net loaded (pretrained)')
        else:
            original_resnet = torchvision.models.resnet18(weights=None)
            net = MaterialPropertyNet(nclass, original_resnet)
            net.apply(weights_init)

        if len(weights) > 0:
            print('Loading weights for material property stream')
            net.load_state_dict(torch.load(weights, True))
        return net

    def build_latent_proj_head(self, in_fc, out_fc):
        return ProjectionHead(in_fc, out_fc)
