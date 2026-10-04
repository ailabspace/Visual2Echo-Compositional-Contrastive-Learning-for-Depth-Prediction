import os
import json
import pickle
import random

import librosa
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance

from util.augment import AddGaussianNoise

_worker_scene_cache = {}
_audio_cache = {}


def preload_audio_for_dataset(ds):
    scene_locs = {tuple(s.split('/')[:2]) for s in ds.data_idx}
    loaded = 0
    for scene, loc in sorted(scene_locs):
        for orn in ds._orientations:
            path = os.path.join(ds.base_audio_path, scene, ds.audio_type, str(orn), loc + '.wav')
            if path not in _audio_cache and os.path.isfile(path):
                try:
                    _audio_cache[path], _ = librosa.load(path, sr=ds.opt.audio_sampling_rate, mono=False,
                                                         duration=ds.opt.audio_length)
                    loaded += 1
                except Exception:
                    pass
    print(f"[DataLoader] Audio cache ready: {loaded} files loaded")


def _load_scene_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        raw = pickle.load(f)
    return {k: {'rgb': v['rgb'][:, :, :3], 'depth': v['depth']} for k, v in raw.items()}


def _get_scene_data(scenes_dir, scene):
    if scene not in _worker_scene_cache:
        _worker_scene_cache[scene] = _load_scene_pkl(os.path.join(scenes_dir, scene + '.pkl'))
    return _worker_scene_cache[scene]


def normalize(samples, desired_rms=0.1, eps=1e-4):
    rms = np.maximum(eps, np.sqrt(np.mean(samples ** 2)))
    return samples * (desired_rms / rms)


def generate_spectrogram(audioL, audioR, winl=32, hop_length=None, use_ipd=False, log_scale=False,
                         use_ild=False, use_magdiff=False, n_fft=512, ild_clip_db=0.0, log_gain=1.0):
    if hop_length is None:
        hop_length = n_fft // 4
    sL = librosa.stft(audioL, n_fft=n_fft, win_length=winl, hop_length=hop_length)
    sR = librosa.stft(audioR, n_fft=n_fft, win_length=winl, hop_length=hop_length)
    mL_raw, mR_raw = np.abs(sL), np.abs(sR)
    mL, mR = mL_raw.copy(), mR_raw.copy()
    if log_scale:
        mL = np.log1p(log_gain * mL)
        mR = np.log1p(log_gain * mR)
    channels = [mL[None], mR[None]]
    if use_ipd:
        ipd = np.angle(sL * np.conj(sR))
        channels += [np.sin(ipd).astype(np.float32)[None], np.cos(ipd).astype(np.float32)[None]]
    if use_ild:
        ild = 20 * np.log10((mL_raw + 1e-8) / (mR_raw + 1e-8))
        if ild_clip_db and ild_clip_db > 0:
            ild = np.clip(ild, -ild_clip_db, ild_clip_db) / 20.0
        channels.append(ild.astype(np.float32)[None])
    if use_magdiff:
        md = mL_raw - mR_raw
        if log_scale:
            md = np.sign(md) * np.log1p(np.abs(md))
        channels.append(md[None])
    return np.concatenate(channels, axis=0)


def process_image(rgb, augment):
    if augment:
        rgb = ImageEnhance.Brightness(rgb).enhance(random.random() * 0.6 + 0.7)
        rgb = ImageEnhance.Color(rgb).enhance(random.random() * 0.6 + 0.7)
        rgb = ImageEnhance.Contrast(rgb).enhance(random.random() * 0.6 + 0.7)
    return rgb


def parse_all_data(root_path, scenes):
    data_idx = []
    split_name = os.path.splitext(os.path.basename(root_path))[0]
    scenes_dir = os.path.join(os.path.dirname(root_path), 'scenes', split_name)
    if os.path.isdir(scenes_dir) and any(os.path.isfile(os.path.join(scenes_dir, s + '.pkl')) for s in scenes):
        index_path = os.path.join(scenes_dir, 'index.json')
        if os.path.isfile(index_path):
            with open(index_path) as f:
                full_index = json.load(f)
            for scene in scenes:
                if scene in full_index:
                    data_idx += [f"{scene}/{loc}/{ori}" for (loc, ori) in full_index[scene]]
            for scene in scenes:
                pkl = os.path.join(scenes_dir, scene + '.pkl')
                if os.path.isfile(pkl) and scene not in _worker_scene_cache:
                    _worker_scene_cache[scene] = _load_scene_pkl(pkl)
            return data_idx, {}, scenes_dir
        data_dict = {}
        for scene in scenes:
            pkl = os.path.join(scenes_dir, scene + '.pkl')
            if os.path.isfile(pkl):
                data_dict[scene] = _load_scene_pkl(pkl)
                data_idx += [f"{scene}/{loc}/{ori}" for (loc, ori) in data_dict[scene].keys()]
        return data_idx, data_dict, None
    with open(root_path, 'rb') as f:
        data_dict = pickle.load(f)
    for scene in scenes:
        data_idx += [f"{scene}/{loc}/{ori}" for (loc, ori) in data_dict[scene].keys()]
    return data_idx, data_dict, None


class AudioVisualDataset(data.Dataset):
    def initialize(self, opt):
        self.opt = opt
        root = os.path.join(opt.img_path, opt.mode + '.pkl') if opt.dataset == 'mp3d' else opt.img_path
        self.data_idx, self.data, self._scenes_dir = parse_all_data(root, opt.scenes[opt.mode])
        self.win_length = opt.audio_win_length
        self.hop_length = opt.audio_hop_length
        self.vision_transform = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.audio_transform = AddGaussianNoise(mean=0, std=0.0005)
        self.base_audio_path = opt.audio_path
        self.audio_type = '3ms_sweep_16khz' if opt.dataset == 'mp3d' else '3ms_sweep'
        self._orientations = [0, 90, 180, 270]

    def __getitem__(self, index):
        opt = self.opt
        scene, loc, orn = self.data_idx[index].split('/')
        audio_path = os.path.join(self.base_audio_path, scene, self.audio_type, orn, loc + '.wav')
        if audio_path in _audio_cache:
            audio = _audio_cache[audio_path]
        else:
            audio, _ = librosa.load(audio_path, sr=opt.audio_sampling_rate, mono=False, duration=opt.audio_length)
        if opt.audio_normalize:
            audio = normalize(audio)

        scene_dict = _get_scene_data(self._scenes_dir, scene) if self._scenes_dir is not None else self.data[scene]
        entry = scene_dict[(int(loc), int(orn))]
        img = Image.fromarray(entry['rgb']).convert('RGB')
        if opt.mode == "train":
            img = process_image(img, opt.enable_img_augmentation)
        if opt.image_transform:
            img = self.vision_transform(img)
        depth = torch.FloatTensor(entry['depth']).unsqueeze(0)

        if opt.mode == "train" and random.random() < 0.5 and not opt.no_audio_augment:
            audio = self.audio_transform(torch.FloatTensor(audio)).numpy()

        spec = torch.FloatTensor(generate_spectrogram(
            audio[0, :], audio[1, :], self.win_length, hop_length=self.hop_length, use_ipd=opt.use_ipd,
            log_scale=opt.log_spectrogram, use_ild=opt.use_ild, use_magdiff=opt.use_magdiff, n_fft=opt.audio_nfft))

        if opt.mode == 'train' and opt.use_specaugment:
            _, nf, nt = spec.shape
            for _ in range(2):
                f = random.randint(0, max(1, nf // 12))
                f0 = random.randint(0, max(0, nf - f))
                spec[:, f0:f0 + f, :] = 0.0
            for _ in range(2):
                t = random.randint(0, max(1, nt // 12))
                t0 = random.randint(0, max(0, nt - t))
                spec[:, :, t0:t0 + t] = 0.0

        return {'img': img, 'depth': depth, 'audio': spec, 'orn': int(orn), 'idx': int(index)}

    def __len__(self):
        return len(self.data_idx)
