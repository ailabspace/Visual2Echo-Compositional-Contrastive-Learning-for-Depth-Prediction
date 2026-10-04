import os
import json
import math
import random

import h5py
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
from scipy.signal import resample as _fft_resample
from scipy.signal import butter, sosfiltfilt

from data_loader.audio_visual_dataset import generate_spectrogram, normalize
from util.augment import AddGaussianNoise


def _resize_depth(d, mask, target_hw, method='bilinear'):
    d_t = torch.from_numpy(d)[None, None]
    m_t = torch.from_numpy(mask)[None, None]
    if d.shape != target_hw:
        kw = {} if method == 'nearest' else {'align_corners': False}
        d_t = F.interpolate(d_t, size=target_hw, mode='nearest' if method == 'nearest' else 'bilinear', **kw)
        m_t = F.interpolate(m_t, size=target_hw, mode='nearest')
    return d_t.squeeze(0), m_t.squeeze(0)


class BiosonarDataset(data.Dataset):
    def __init__(self):
        super().__init__()
        self._h5 = {}
        self._rgb_handles = {}
        self._geom = {}

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.audio_path or opt.img_path
        with open(os.path.join(self.root, 'splits.json')) as f:
            self.splits = json.load(f)

        sz = int(opt.batvision_img_size)
        self.target_hw = tuple(int(x) for x in opt.depth_target_hw.split(',')) if opt.depth_target_hw else (sz, sz)
        self.depth_resize_method = opt.depth_resize_method
        self.rgb_dir = opt.biosonar_rgb_dir or ''
        self.teacher_img_size = int(opt.teacher_img_size)
        self._rgb_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self._rgb_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        self.audio_sr = int(opt.audio_sampling_rate)
        self.audio_len_samples = int(round(float(opt.audio_length) * self.audio_sr))
        self.crop_pre_roll = float(opt.audio_crop_pre_roll) if opt.audio_crop_pre_roll >= 0 else None

        n_fft = int(opt.audio_nfft)
        n_freq = n_fft // 2 + 1
        lo, hi = float(opt.audio_bandpass_lo), float(opt.audio_bandpass_hi)
        if hi > 0:
            bin_hz = self.audio_sr / n_fft
            self.bp_lo = max(0, int(round(lo / bin_hz)))
            self.bp_hi = min(n_freq, int(round(hi / bin_hz)) + 1)
        else:
            self.bp_lo, self.bp_hi = 0, n_freq
        order = int(opt.audio_butter_order)
        self._butter_sos = (butter(order, [lo, hi], btype='band', fs=self.audio_sr, output='sos')
                            if order > 0 and 0.0 < lo < hi < self.audio_sr * 0.5 else None)

        split = opt.mode if opt.mode in ('train', 'val', 'test') else 'train'
        split_keys = ('val', 'test') if split == 'val' and opt.val_include_test else (split,)

        vidx_path = os.path.join(self.root, 'valid_index.json')
        self.index = []
        if os.path.isfile(vidx_path):
            with open(vidx_path) as f:
                valid_splits = json.load(f)['splits']
            for sk in split_keys:
                for e in valid_splits.get(sk, []):
                    if os.path.isfile(os.path.join(self.root, e['h5'])):
                        self.index.append((os.path.join(self.root, e['h5']), int(e['win'])))
        else:
            for sk in split_keys:
                for fn in self.splits.get(sk, {}).get('files', []):
                    fpath = os.path.join(self.root, fn)
                    if os.path.isfile(fpath):
                        with h5py.File(fpath, 'r') as h:
                            self.index.extend((fpath, i) for i in range(int(h['depth'].shape[0])))
        self.audio_transform = AddGaussianNoise(mean=0, std=0.0005)
        print(f'[BiosonarDataset] split={split} windows={len(self.index)} sr={self.audio_sr} '
              f'T={self.audio_len_samples} img={self.target_hw} max_depth={opt.max_depth}')

    def _handle(self, fpath):
        if fpath not in self._h5:
            self._h5[fpath] = h5py.File(fpath, 'r')
        return self._h5[fpath]

    def _crop_window(self, fpath, a1, a2):
        if fpath not in self._geom:
            at = self._handle(fpath).attrs
            self._geom[fpath] = (float(at['fs']), float(at.get('pre_roll_s', 0.002)))
        fs, pre = self._geom[fpath]
        s0 = int(round((pre - self.crop_pre_roll) * fs))
        n = int(round(float(self.opt.audio_length) * fs))
        s0 = max(0, min(s0, a1.shape[0] - n)) if a1.shape[0] >= n else 0
        return a1[s0:s0 + n], a2[s0:s0 + n]

    def _load_rgb(self, fpath, row):
        rgb_path = os.path.join(self.rgb_dir, os.path.basename(fpath).replace('.h5', '_rgb.h5'))
        if rgb_path not in self._rgb_handles:
            self._rgb_handles[rgb_path] = h5py.File(rgb_path, 'r')
        h = self._rgb_handles[rgb_path]
        arr = np.asarray(h['rgb'][min(row, h['rgb'].shape[0] - 1)])
        t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        S = self.teacher_img_size
        t = F.interpolate(t.unsqueeze(0), size=(S, S), mode='bilinear', align_corners=False)[0]
        return (t - self._rgb_mean) / self._rgb_std

    def __len__(self):
        return len(self.index)

    def _spec(self, fpath, i):
        opt = self.opt
        h = self._handle(fpath)
        a1 = np.asarray(h['audio_mic1'][i], dtype=np.float32)
        a2 = np.asarray(h['audio_mic2'][i], dtype=np.float32)
        if self.crop_pre_roll is not None:
            a1, a2 = self._crop_window(fpath, a1, a2)
        if a1.shape[0] != self.audio_len_samples:
            a1 = _fft_resample(a1, self.audio_len_samples).astype(np.float32)
            a2 = _fft_resample(a2, self.audio_len_samples).astype(np.float32)
        if opt.audio_mono_mode == 'left':
            a2 = a1.copy()
        elif opt.audio_mono_mode == 'right':
            a1 = a2.copy()
        audio = np.stack([a1, a2], axis=0)
        if self._butter_sos is not None:
            audio = sosfiltfilt(self._butter_sos, audio, axis=-1).astype(np.float32)
        if opt.audio_normalize:
            audio = normalize(audio)
        train_aug = opt.mode == 'train' and not opt.no_audio_augment
        if train_aug and random.random() < 0.5:
            audio = self.audio_transform(torch.FloatTensor(audio)).numpy()

        spec = torch.FloatTensor(generate_spectrogram(
            audio[0, :], audio[1, :], opt.audio_win_length, hop_length=opt.audio_hop_length,
            use_ipd=opt.use_ipd, log_scale=opt.log_spectrogram, use_ild=opt.use_ild,
            use_magdiff=opt.use_magdiff, n_fft=opt.audio_nfft, ild_clip_db=float(opt.ild_clip_db or 0.0),
            log_gain=float(opt.spec_log_gain or 1.0)))
        if self.bp_lo != 0 or self.bp_hi != spec.shape[1]:
            spec = spec[:, self.bp_lo:self.bp_hi, :]

        if opt.spec_eq_db > 0 and train_aug and random.random() < opt.spec_eq_p:
            f = torch.linspace(0, math.pi, spec.shape[1])
            c = sum((random.uniform(-1, 1) / k) * torch.cos(k * f + random.uniform(0, 2 * math.pi))
                    for k in range(1, 5))
            c = c / c.abs().max().clamp_min(1e-6) * random.uniform(0, opt.spec_eq_db)
            g = (10.0 ** (c / 20.0))[None, :, None]
            if opt.log_spectrogram:
                spec[:2] = torch.log1p(torch.expm1(spec[:2]) * g)
            else:
                spec[:2] = spec[:2] * g
        return spec

    def __getitem__(self, index):
        fpath, i = self.index[index]
        h = self._handle(fpath)
        spec = self._spec(fpath, i)

        if self.opt.mode == 'train' and self.opt.use_specaugment:
            _, nf, nt = spec.shape
            for _ in range(2):
                f = random.randint(0, max(1, nf // 12))
                f0 = random.randint(0, max(0, nf - f))
                spec[:, f0:f0 + f, :] = 0.0
            for _ in range(2):
                t = random.randint(0, max(1, nt // 12))
                t0 = random.randint(0, max(0, nt - t))
                spec[:, :, t0:t0 + t] = 0.0

        d = np.asarray(h['depth'][i], dtype=np.float32) / 1000.0
        while d.ndim > 2 and d.shape[0] == 1:
            d = d[0]
        max_d = float(self.opt.max_depth)
        valid = np.isfinite(d) & (d > 0) & (d <= max_d)
        d = np.where(valid, d, 0.0).astype(np.float32)
        depth, mask = _resize_depth(d, valid.astype(np.float32), self.target_hw, self.depth_resize_method)
        img = self._load_rgb(fpath, i) if self.rgb_dir else torch.zeros(3, *self.target_hw)
        return {'img': img, 'depth': depth, 'depth_mask': mask, 'audio': spec, 'orn': 0, 'idx': int(index)}
