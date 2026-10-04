import os
import torch
import torchvision
import torch.nn as nn
from collections import OrderedDict
from .networks import (RGBDepthNet, weights_init, SimpleAudioDepthNet, LegacyAudioDepthNet,
                       MaterialPropertyNet, CompositionalEmbedding, ProjectionHead)


class ModelBuilder():
    def build_audiodepth(self, audio_shape=[2, 257, 166], out_nc=1, weights='', backbone=None,
                         mode='base', **kwargs):
        if backbone == "Legacy":
            net = LegacyAudioDepthNet(audio_shape=audio_shape, audio_feature_length=512,
                                      output_nc=out_nc, mode=mode)
        else:
            net = SimpleAudioDepthNet(audio_shape=audio_shape, audio_feature_length=512,
                                      backbone=backbone, mode=mode, **kwargs)

        if len(weights) > 0:
            print('Loading weights for audio stream')
            sd = torch.load(weights, map_location='cpu')
            model_sd = net.state_dict()
            compatible = {k: v for k, v in sd.items() if k in model_sd and v.shape == model_sd[k].shape}
            n_bad = sum(1 for k, v in sd.items() if k in model_sd and v.shape != model_sd[k].shape)
            if n_bad:
                print(f'   Skipped {n_bad} keys with shape mismatch')
            if kwargs.get('use_log_depth'):
                for k in [k for k in compatible if 'rgbdepth_upconvlayer5' in k]:
                    compatible.pop(k)
            missing, unexpected = net.load_state_dict(compatible, strict=False)
            print(f'   Loaded {len(compatible)}/{len(model_sd)} keys '
                  f'({len(missing)} missing, {len(unexpected)} unexpected)')
        return net

    def build_rgbdepth(self, ngf=64, input_nc=3, output_nc=1, weights='',
                       teacher='rgbdepth_unet', teacher_max_depth=10.0,
                       moge_model_id='Ruicheng/moge-2-vitl-normal',
                       moge_num_tokens=None, moge_resolution_level=9, moge_use_fp16=False,
                       moge_cache_consumer_mode=False, moge_cache_enc_dim_out=None):
        if teacher == 'moge_v2':
            from .moge_teacher import MoGeRGBDepthNet
            print(f'Loading MoGe-2 teacher: {moge_model_id}')
            return MoGeRGBDepthNet(
                model_id=moge_model_id, teacher_max_depth=teacher_max_depth,
                num_tokens=moge_num_tokens, resolution_level=moge_resolution_level,
                use_fp16=moge_use_fp16, cache_consumer_mode=moge_cache_consumer_mode,
                cache_enc_dim_out=moge_cache_enc_dim_out)

        net = RGBDepthNet(ngf, input_nc, output_nc)
        net.apply(weights_init)
        if len(weights) > 0:
            if not os.path.isfile(weights):
                print(f'[WARN] RGB teacher checkpoint not found: {weights}')
            else:
                print('Loading weights for visual stream')
                net.load_state_dict(torch.load(weights, map_location='cpu'))
        return net

    def build_ccl(self, head=None, input_dim=None, out_nc=256, weights='',
                  normalization_sign=False, use_film=False, n_class=23):
        net = CompositionalEmbedding(head=head, input_dim=input_dim, out_nc=out_nc,
                                     normalization_sign=normalization_sign, use_film=use_film,
                                     n_class=n_class)
        net.apply(weights_init)
        if use_film:
            for name in ('film_gamma', 'film_beta'):
                m = getattr(net, name, None)
                if m is not None:
                    nn.init.zeros_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        if len(weights) > 0:
            print('Loading weights for CCL stream')
            sd = torch.load(weights, map_location='cpu')
            model_sd = net.state_dict()
            net.load_state_dict({k: v for k, v in sd.items()
                                 if k in model_sd and v.shape == model_sd[k].shape}, strict=False)
        return net

    def build_material_property(self, nclass=10, weights='', init_weights=''):
        if len(init_weights) > 0:
            net = MaterialPropertyNet(23, torchvision.models.resnet18(
                weights=torchvision.models.ResNet18_Weights.DEFAULT))
            sd = torch.load(init_weights, map_location='cpu', weights_only=False)['state_dict']
            sd = OrderedDict(('.'.join(k.split('.')[1:]), v) for k, v in sd.items())
            net.load_state_dict({k: v for k, v in sd.items() if k in net.state_dict()}, strict=False)
            print('Material Property Net loaded (pretrained)')
        else:
            net = MaterialPropertyNet(nclass, torchvision.models.resnet18(weights=None))
            net.apply(weights_init)
        if len(weights) > 0:
            print('Loading weights for material property stream')
            net.load_state_dict(torch.load(weights, map_location='cpu'))
        return net

    def build_latent_proj_head(self, in_fc, out_fc, use_ln=False):
        return ProjectionHead(in_fc, out_fc, use_ln=use_ln)
