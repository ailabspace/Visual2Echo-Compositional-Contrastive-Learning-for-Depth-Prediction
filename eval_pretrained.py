import torch
import numpy as np
from torch.utils.data import DataLoader
from data_loader.audio_visual_dataset import AudioVisualDataset
from models.models import ModelBuilder
from util.util import compute_errors
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', required=True)
    parser.add_argument('--audio_path', required=True)
    parser.add_argument('--metadatapath', default='dataset/metadata/replica')
    parser.add_argument('--init_material_weight', default='checkpoints_pretrained/material_pre_trained_minc.pth')
    parser.add_argument('--weights', default='checkpoints_pretrained/audiodepth_replica.pth')
    parser.add_argument('--max_depth', type=float, default=5.0)
    parser.add_argument('--batchSize', type=int, default=64)
    parser.add_argument('--nThreads', type=int, default=4)
    parser.add_argument('--split', default='test', choices=['val', 'test'])
    parser.add_argument('--audio_hop_length', type=int, default=16,
                        help='STFT hop for Legacy backbone: 16 → [2,257,166]')
    return parser.parse_args()


def main():
    args = parse_args()

    class Opt:
        pass
    opt = Opt()
    opt.dataset        = 'replica'
    opt.audio_length   = 0.06
    opt.audio_hop_length = args.audio_hop_length
    opt.img_path       = args.img_path
    opt.audio_path     = args.audio_path
    opt.metadatapath   = args.metadatapath
    opt.max_depth      = args.max_depth
    opt.enable_img_augmentation = False
    opt.image_transform = True
    opt.use_ipd        = False
    opt.use_ild        = False
    opt.use_magdiff    = False
    opt.log_spectrogram = False
    opt.audio_nfft     = 512
    opt.audio_win_length = 256
    opt.no_audio_augment = True
    opt.use_specaugment = False
    opt.audio_normalize = False
    opt.mode           = args.split

    _sr   = 44100
    _hop  = args.audio_hop_length
    _n_frames = 1 + int(opt.audio_length * _sr) // _hop
    opt.audio_shape         = [2, 257, _n_frames]
    opt.audio_sampling_rate = _sr

    import os
    def _load_scenes(fname):
        with open(fname) as f:
            return [x.strip() for x in f.readlines()]
    opt.scenes = {
        'train': _load_scenes(os.path.join(args.metadatapath, 'replica_train.txt')),
        'val':   _load_scenes(os.path.join(args.metadatapath, 'replica_val.txt')),
        'test':  _load_scenes(os.path.join(args.metadatapath, 'replica_test.txt')),
    }

    print(f'audio_shape: {opt.audio_shape}')
    print(f'Evaluating on {args.split} split ({len(opt.scenes[args.split])} scenes)')

    dataset = AudioVisualDataset()
    dataset.initialize(opt)
    loader  = DataLoader(dataset, batch_size=args.batchSize, shuffle=False,
                         num_workers=args.nThreads, pin_memory=True)
    print(f'Dataset size: {len(dataset)} samples')

    builder = ModelBuilder()
    net = builder.build_audiodepth(
        audio_shape=opt.audio_shape, backbone='Legacy', mode='base',
        weights=args.weights
    )
    net = net.cuda().eval()

    all_preds, all_gts = [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            audio  = batch['audio'].cuda()
            depth_gt = batch['depth'].cuda()

            depth_pred, _ = net(audio)
            depth_pred = depth_pred * args.max_depth
            depth_pred = depth_pred.clamp(0, args.max_depth)

            all_preds.append(depth_pred.cpu().numpy())
            all_gts.append(depth_gt.cpu().numpy())
            if (i + 1) % 20 == 0:
                print(f'  Batch {i+1}/{len(loader)}')

    all_preds = np.concatenate(all_preds, axis=0)
    all_gts   = np.concatenate(all_gts, axis=0)

    pred_flat = all_preds.flatten()
    gt_flat   = all_gts.flatten()
    valid = gt_flat > 0
    pred_flat = pred_flat[valid]
    gt_flat   = gt_flat[valid]
    gt_flat   = np.clip(gt_flat, 0, args.max_depth)

    abs_rel, rmse, d1, d2, d3, log10, mae = compute_errors(gt_flat, pred_flat)
    print('\n=== Results ===')
    print(f'ABS_REL: {abs_rel:.4f}')
    print(f'RMSE:    {rmse:.4f}')
    print(f'LOG10:   {log10:.4f}')
    print(f'MAE:     {mae:.4f}')
    print(f'DELTA1:  {d1:.4f}')
    print(f'DELTA2:  {d2:.4f}')
    print(f'DELTA3:  {d3:.4f}')


if __name__ == '__main__':
    main()
