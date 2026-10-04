import argparse
import os
import sys

import h5py
import torch
import torch.utils.data as tud

from data_loader.custom_dataset_data_loader import CreateDataset
from models.models import ModelBuilder
from options.train_options import TrainOptions


@torch.no_grad()
def precompute(opt, device, cache_path, compress=False, multiscale=False):
    builder = ModelBuilder()
    is_moge = opt.rgb_teacher == 'moge_v2'
    net_rgbdepth = builder.build_rgbdepth(
        weights=os.path.join('checkpoints_pretrained', f'rgbdepth_{opt.dataset}.pth'), teacher=opt.rgb_teacher,
        teacher_max_depth=float(opt.teacher_max_depth), moge_model_id=opt.moge_model_id,
        moge_num_tokens=int(opt.moge_num_tokens) or None, moge_resolution_level=int(opt.moge_resolution_level),
        moge_use_fp16=opt.moge_use_fp16).to(device).eval()
    net_material = builder.build_material_property(init_weights=opt.init_material_weight).to(device).eval()

    dataset = CreateDataset(opt)
    loader = tud.DataLoader(dataset, batch_size=opt.batchSize, shuffle=False, drop_last=False,
                            num_workers=int(opt.nThreads))
    n = len(dataset)
    print(f'[precompute] split={opt.mode} samples={n} teacher={opt.rgb_teacher}')
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    kw = dict(compression='gzip', compression_opts=4) if compress else {}

    with h5py.File(cache_path, 'w') as hf:
        ds, i0 = {}, 0
        for batch in loader:
            rgb = batch['img'].to(device)
            if is_moge:
                full = net_rgbdepth.infer(rgb, multi=multiscale)
                key = 'enc_feat_multi' if multiscale else 'enc_feat'
                out = {'img_depth': full['depth_norm'], key: full[key]}
            else:
                img_depth, img_feat = net_rgbdepth(rgb)
                out = {'img_depth': img_depth, 'img_feat': img_feat}
            out['material_class'], out['material_feat'] = net_material(rgb)
            if not ds:
                for k, v in out.items():
                    ds[k] = hf.create_dataset(k, shape=(n, *v.shape[1:]), dtype='float16', **kw)
                    print(f'[precompute] {k}: {tuple(v.shape[1:])}')
            i1 = i0 + rgb.shape[0]
            for k, v in out.items():
                ds[k][i0:i1] = v.half().cpu().numpy()
            i0 = i1
        hf.attrs.update(total=i0, dataset=opt.dataset, teacher=opt.rgb_teacher)
        if is_moge:
            hf.attrs.update(moge_model_id=opt.moge_model_id, teacher_max_depth=float(opt.teacher_max_depth))
    assert i0 == n, f'cached {i0} rows != {n} samples'
    print(f'[precompute] {i0} samples -> {cache_path} ({os.path.getsize(cache_path) / 1e6:.1f} MB)')


if __name__ == '__main__':
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument('--teacher_cache_path', type=str, required=True)
    extra.add_argument('--compress', action='store_true')
    extra.add_argument('--device', type=str, default='cuda')
    extra.add_argument('--mode_ext', type=str, default='train')
    extra.add_argument('--cache_multiscale', action='store_true')
    known, rest = extra.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    opt = TrainOptions().parse()
    opt.mode = known.mode_ext
    precompute(opt, torch.device(known.device if torch.cuda.is_available() else 'cpu'),
               known.teacher_cache_path, known.compress, known.cache_multiscale)
