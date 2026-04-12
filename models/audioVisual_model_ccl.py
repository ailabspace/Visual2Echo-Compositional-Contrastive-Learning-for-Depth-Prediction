import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioVisualModel(torch.nn.Module):
    """Cross-modal audio-visual model for CCL training.

    Combines audio depth estimation with compositional contrastive learning
    heads for material classification and depth alignment.
    """

    def name(self):
        return 'AudioVisualModel'

    def __init__(self, nets, opt, mode="depth"):
        super(AudioVisualModel, self).__init__()
        self.opt = opt
        self.mode = mode
        self.net_rgbdepth, self.net_audio, self.net_material, \
            self.ccl_audiomat, self.ccl_audiodepth, self.img_proj, self.aud_proj = nets

    def forward(self, input, compute_ccl=True, compute_ct=True):
        rgb_input = input['img']
        audio_input = input['audio']
        depth_gt = input['depth']

        audio_depth, audio_mat_class, audio_feat = self.net_audio(audio_input)

        _teacher_max = getattr(self.opt, 'teacher_max_depth', self.opt.max_depth)

        # Use cached teacher features if available, otherwise run live inference
        if 'img_feat' in input and 'material_feat' in input:
            img_feat = input['img_feat'].to(audio_feat.device)
            material_feat = input['material_feat'].to(audio_feat.device)
            material_class = input['material_class_teacher'].to(audio_feat.device)
            img_depth = input['img_depth'].to(audio_feat.device) if 'img_depth' in input else None
        else:
            with torch.no_grad():
                img_depth, img_feat = self.net_rgbdepth(rgb_input)
                material_class, material_feat = self.net_material(rgb_input)

        # CCL branches (stop-gradient on backbone)
        if compute_ccl:
            audio_feat_ccl = audio_feat.detach().expand(
                -1, -1, img_feat.shape[-2], img_feat.shape[-1]).contiguous()
            ccl_audiomat, ccl_mat_feat = self.ccl_audiomat(
                audio_feat_ccl, material_feat.detach())
            ccl_audiodepth, ccl_audiodepth_feat = self.ccl_audiodepth(
                audio_feat_ccl, img_feat.detach())
        else:
            ccl_audiomat = ccl_mat_feat = ccl_audiodepth = ccl_audiodepth_feat = None

        # Contrastive projection heads
        if compute_ct:
            aud_proj_feat = self.aud_proj(
                F.adaptive_avg_pool2d(audio_feat, (1, 1)))
            img_proj_feat = self.img_proj(
                F.adaptive_avg_pool2d(img_feat.detach(), (1, 1)))
        else:
            aud_proj_feat = img_proj_feat = None

        # Scale teacher output to meters
        if img_depth is not None:
            img_depth_m = (img_depth * _teacher_max).clamp(0, self.opt.max_depth)
        else:
            img_depth_m = None

        output = {
            'img_depth': img_depth_m,
            'img_feat': img_feat,
            'audio_depth': audio_depth * self.opt.max_depth,
            'audio_mat_class': audio_mat_class,
            'audio_feat': audio_feat,
            'ccl_audiomat': ccl_audiomat,
            'ccl_audiomat_feat': ccl_mat_feat,
            'ccl_audiodepth': ccl_audiodepth * self.opt.max_depth if ccl_audiodepth is not None else None,
            'ccl_audiodepth_feat': ccl_audiodepth_feat,
            'material_class': material_class,
            'material_feat': material_feat,
            'aud_proj_feat': aud_proj_feat,
            'img_proj_feat': img_proj_feat,
            'depth_gt': depth_gt,
        }
        return output
