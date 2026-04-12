"""
Precompute and cache teacher network features for train_ccl.py.

Run this once per dataset split before training. It saves teacher outputs
(img_feat, material_feat, material_class) indexed by dataset sample index to
an HDF5 file. Downstream, CachedLatentDataset loads these from disk so the
teacher networks never run on GPU during training.

Usage:
    python precompute_teacher_latents.py \
        --dataset mp3d \
        --img_path <path> --audio_path <path> --metadatapath <path> \
        --init_material_weight checkpoints_pretrained/material_pre_trained_minc.pth \
        --teacher_cache_path latents/teacher_mp3d_train.h5 \
        [--mode train]   # repeat with --mode val

Notes:
- Run with --mode train and --mode val (two separate runs).
- Teachers run on un-augmented RGB (mode=val disables augmentation).
  This is intentional: soft labels from clean images are more stable.
- Features are stored in float16 to halve disk/RAM usage.
- If the dataset is large, enable gzip compression via --compress.
"""

import os
import sys
import argparse
import torch
import h5py
import numpy as np
from options.train_options import TrainOptions
from models.models import ModelBuilder
from data_loader.custom_dataset_data_loader import CustomDatasetDataLoader


@torch.no_grad()
def precompute(opt, device, cache_path, compress):
    builder = ModelBuilder()
    net_rgbdepth = builder.build_rgbdepth(
        weights=os.path.join("checkpoints_pretrained", f"rgbdepth_{opt.dataset}.pth"))
    net_material = builder.build_material_property(init_weights=opt.init_material_weight)

    net_rgbdepth.to(device).eval()
    net_material.to(device).eval()

    # # Always load in val mode so augmentation is disabled → deterministic RGB
    # opt.mode = 'train'
    dataloader = CustomDatasetDataLoader()
    dataloader.initialize(opt)
    n_samples = len(dataloader.dataset)
    print(f"[precompute] Split: {opt.mode}, samples: {n_samples}")

    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)

    compression = "gzip" if compress else None

    with h5py.File(cache_path, 'w') as hf:
        # Pre-allocate datasets. We'll determine shapes from first batch.
        datasets = {}
        global_idx = 0

        for batch_idx, batch in enumerate(dataloader):
            if batch is None:
                continue

            rgb = batch['img'].to(device)
            B = rgb.shape[0]

            img_depth, img_feat = net_rgbdepth(rgb)
            material_class, material_feat = net_material(rgb)

            # Move to CPU and convert to fp16 for compact storage
            img_depth_np     = img_depth.half().cpu().numpy()      # [B, 1, H, W] sigmoid output
            img_feat_np      = img_feat.half().cpu().numpy()       # [B, 512, 8, 8]
            material_feat_np = material_feat.half().cpu().numpy()  # [B, 512, 4, 4]
            material_cls_np  = material_class.half().cpu().numpy() # [B, nclass]

            # Create pre-allocated datasets on first batch
            if not datasets:
                kwargs = dict(compression=compression, compression_opts=4 if compress else None)
                datasets['img_depth']      = hf.create_dataset('img_depth',
                    shape=(n_samples, *img_depth_np.shape[1:]),     dtype='float16', **kwargs)
                datasets['img_feat']       = hf.create_dataset('img_feat',
                    shape=(n_samples, *img_feat_np.shape[1:]),      dtype='float16', **kwargs)
                datasets['material_feat']  = hf.create_dataset('material_feat',
                    shape=(n_samples, *material_feat_np.shape[1:]), dtype='float16', **kwargs)
                datasets['material_class'] = hf.create_dataset('material_class',
                    shape=(n_samples, *material_cls_np.shape[1:]),  dtype='float16', **kwargs)
                print(f"[precompute] img_depth shape/sample: {img_depth_np.shape[1:]}")
                print(f"[precompute] img_feat shape/sample: {img_feat_np.shape[1:]}")
                print(f"[precompute] material_feat shape/sample: {material_feat_np.shape[1:]}")
                print(f"[precompute] material_class shape/sample: {material_cls_np.shape[1:]}")

            end = global_idx + B
            datasets['img_depth'][global_idx:end]      = img_depth_np
            datasets['img_feat'][global_idx:end]       = img_feat_np
            datasets['material_feat'][global_idx:end]  = material_feat_np
            datasets['material_class'][global_idx:end] = material_cls_np

            global_idx = end
            if batch_idx % 20 == 0:
                print(f"[precompute] {global_idx}/{n_samples} samples done")

        hf.attrs['total']   = global_idx
        hf.attrs['dataset'] = opt.dataset
        print(f"[precompute] Done. Cached {global_idx} samples → {cache_path}")

    size_mb = os.path.getsize(cache_path) / 1e6
    print(f"[precompute] Cache file size: {size_mb:.1f} MB")


if __name__ == '__main__':
    # Parse teacher_cache_path and compress from sys.argv before handing off to
    # TrainOptions (which uses argparse internally).
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument('--teacher_cache_path', type=str, required=True,
                       help='Path to write HDF5 cache file')
    extra.add_argument('--compress', action='store_true',
                       help='Enable gzip compression (slower write, smaller file)')
    extra.add_argument('--device', type=str, default='cuda')
    extra.add_argument('--mode_ext', type=str, default='train')
    known, remaining = extra.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    opt = TrainOptions().parse()
    opt.mode = known.mode_ext
    device = torch.device(known.device if torch.cuda.is_available() else 'cpu')
    precompute(opt, device, known.teacher_cache_path, known.compress)
