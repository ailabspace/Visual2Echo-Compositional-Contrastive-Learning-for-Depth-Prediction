import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioVisualModel(torch.nn.Module):
    def __init__(self, nets, opt):
        super().__init__()
        self.opt = opt
        self.net_rgbdepth, self.net_audio, self.net_material, \
            self.ccl_audiomat, self.ccl_audiodepth, self.img_proj, self.aud_proj = nets
        self.ldc_head = None
        if opt.lambda_ldc > 0:
            self.ldc_head = nn.Sequential(nn.Linear(512, 256), nn.GELU(), nn.Linear(256, opt.ldc_grid ** 2))
        self.cc_head = None
        if opt.lambda_cc > 0:
            self.cc_head = nn.Sequential(nn.Linear(512, 256), nn.GELU(), nn.Linear(256, 256 * 16))

    def forward(self, input, compute_ccl=True, compute_ct=True):
        opt = self.opt
        rgb = input['img']
        audio_depth, audio_mat_class, audio_feat = self.net_audio(input['audio'])
        dev = audio_feat.device

        need_mat = compute_ccl and (opt.lambda_ccl_depth + opt.lambda_ccl_mat) > 0
        need_img_feat = need_mat or (compute_ct and opt.lambda_ct > 0)
        need_img_depth = opt.lambda_teacher_depth > 0 or self.ldc_head is not None

        if ('enc_feat' in input or 'enc_feat_multi' in input or 'img_feat' in input) and 'material_feat' in input:
            material_feat = input['material_feat'].to(dev) if need_mat else None
            material_class = input['material_class_teacher'].to(dev) if need_mat else None
            img_depth = input['img_depth'].to(dev) if (need_img_depth and 'img_depth' in input) else None
            if not need_img_feat:
                img_feat = None
            elif 'enc_feat_multi' in input or 'enc_feat' in input:
                key = 'enc_feat_multi' if 'enc_feat_multi' in input else 'enc_feat'
                enc_feat = torch.nan_to_num(input[key].to(dev), nan=0.0, posinf=0.0, neginf=0.0)
                if enc_feat.dim() == 5:
                    enc_feat = enc_feat.sum(1)
                enc_feat = F.interpolate(enc_feat, size=getattr(self.net_rgbdepth, 'feat_hw', (4, 4)),
                                         mode='bilinear', align_corners=False)
                img_feat = self.net_rgbdepth.feat_proj(enc_feat)
            else:
                img_feat = input['img_feat'].to(dev)
        else:
            rgb_trainable = any(p.requires_grad for p in self.net_rgbdepth.parameters())
            if not (rgb_trainable or compute_ccl or compute_ct or opt.lambda_teacher_depth > 0):
                img_depth = img_feat = None
                material_class = material_feat = None
                if need_mat:
                    with torch.no_grad():
                        material_class, material_feat = self.net_material(rgb)
            else:
                with torch.set_grad_enabled(rgb_trainable and torch.is_grad_enabled()):
                    img_depth, img_feat = self.net_rgbdepth(rgb)
                with torch.no_grad():
                    material_class, material_feat = self.net_material(rgb)

        ccl_audiomat = ccl_audiodepth = ccl_audiodepth_feat = None
        if compute_ccl:
            audio_feat_ccl = audio_feat.expand(-1, -1, img_feat.shape[-2], img_feat.shape[-1]).contiguous()
            ccl_audiomat, _ = self.ccl_audiomat(audio_feat_ccl, material_feat.detach())
            ccl_audiodepth, ccl_audiodepth_feat = self.ccl_audiodepth(audio_feat_ccl, img_feat.detach())

        aud_proj_feat = img_proj_feat = None
        if compute_ct:
            aud_proj_feat = self.aud_proj(audio_feat.mean((2, 3), keepdim=True))
            img_proj_feat = self.img_proj(img_feat.detach().mean((2, 3), keepdim=True))

        return {
            'img_depth': (img_depth * opt.max_depth).clamp(0, opt.max_depth) if img_depth is not None else None,
            'img_feat': img_feat,
            'audio_depth': audio_depth * opt.max_depth,
            'audio_mat_class': audio_mat_class,
            'audio_feat': audio_feat,
            'ccl_audiomat': ccl_audiomat,
            'ccl_audiodepth': ccl_audiodepth * opt.max_depth if ccl_audiodepth is not None else None,
            'ccl_audiodepth_feat': ccl_audiodepth_feat,
            'material_class': material_class,
            'material_feat': material_feat,
            'aud_proj_feat': aud_proj_feat,
            'img_proj_feat': img_proj_feat,
            'ldc_emb': self.ldc_head(audio_feat.float().flatten(2).mean(2)) if self.ldc_head is not None else None,
            'cc_emb': self.cc_head(audio_feat.float().flatten(2).mean(2)) if self.cc_head is not None else None,
            'depth_gt': input['depth'],
        }
