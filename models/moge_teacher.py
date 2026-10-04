import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ensure_moge_importable():
    try:
        import moge
        return
    except ImportError:
        pass
    here = Path(__file__).resolve().parent
    for cand in (here.parent.parent / "MoGe", here.parent.parent.parent / "MoGe"):
        if (cand / "moge").is_dir():
            sys.path.insert(0, str(cand))
            return
    raise ImportError("MoGe not found: pip install git+https://github.com/microsoft/MoGe.git")


class MoGeRGBDepthNet(nn.Module):
    def __init__(self, model_id="Ruicheng/moge-2-vitl-normal", teacher_max_depth=10.0, num_tokens=None,
                 resolution_level=9, feat_hw=(4, 4), pyramid_channels=(512, 512, 256, 128), use_fp16=False,
                 cache_consumer_mode=False, cache_enc_dim_out=None):
        super().__init__()
        if cache_consumer_mode:
            assert cache_enc_dim_out is not None, "cache_consumer_mode needs cache_enc_dim_out (384 ViT-S, 1024 ViT-L)"
            self.moge = None
            enc_dim_out = int(cache_enc_dim_out)
        else:
            _ensure_moge_importable()
            from moge.model.v2 import MoGeModel
            self.moge = MoGeModel.from_pretrained(model_id)
            for p in self.moge.parameters():
                p.requires_grad = False
            self.moge.eval()
            enc_dim_out = self.moge.encoder.output_projections[0].out_channels
        self.teacher_max = float(teacher_max_depth)
        self.num_tokens = num_tokens
        self.resolution_level = int(resolution_level)
        self.use_fp16 = bool(use_fp16)
        self.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("imagenet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.feat_hw = tuple(feat_hw)
        gen = torch.Generator(device="cpu").manual_seed(0xA5E)

        def _det_kaiming(weight):
            std = (1.0 / (weight.shape[1] * weight.shape[2] * weight.shape[3])) ** 0.5
            with torch.no_grad():
                weight.copy_(torch.empty_like(weight).normal_(0, std, generator=gen))

        self.feat_proj = nn.Conv2d(enc_dim_out, 512, kernel_size=1)
        _det_kaiming(self.feat_proj.weight)
        nn.init.zeros_(self.feat_proj.bias)
        self.pyramid_projs = nn.ModuleList([nn.Conv2d(enc_dim_out, c, kernel_size=1) for c in pyramid_channels])
        for proj in self.pyramid_projs:
            _det_kaiming(proj.weight)
            nn.init.zeros_(proj.bias)

    @torch.no_grad()
    def infer(self, x, multi=False):
        captured, per_layer = {}, []
        hooks = [self.moge.encoder.register_forward_hook(
            lambda m, i, o: captured.__setitem__("feat", o[0] if isinstance(o, tuple) else o))]
        if multi:
            for k, proj in enumerate(self.moge.encoder.output_projections):
                hooks.append(proj.register_forward_hook(
                    lambda m, i, o, k=k: per_layer.append((k, o.detach().to(torch.float16)))))
        det, det_warn = torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
        torch.use_deterministic_algorithms(False)
        try:
            x_raw = (x * self.imagenet_std + self.imagenet_mean).clamp(0.0, 1.0)
            out = self.moge.infer(x_raw, num_tokens=self.num_tokens, resolution_level=self.resolution_level,
                                  apply_mask=False, use_fp16=self.use_fp16, force_projection=True)
        finally:
            torch.use_deterministic_algorithms(det, warn_only=det_warn)
            for h in hooks:
                h.remove()
        depth = torch.nan_to_num(out["depth"], nan=0.0, posinf=self.teacher_max, neginf=0.0)
        res = {"depth_norm": (depth / self.teacher_max).clamp(0.0, 1.0).unsqueeze(1), "enc_feat": captured["feat"]}
        if multi:
            per_layer.sort(key=lambda t: t[0])
            res["enc_feat_multi"] = torch.stack([t.cpu() for _, t in per_layer], dim=1).contiguous()
        return res

    def forward(self, x):
        r = self.infer(x)
        feat = F.interpolate(r["enc_feat"].detach().clone(), size=self.feat_hw, mode="bilinear", align_corners=False)
        return r["depth_norm"], self.feat_proj(feat)
