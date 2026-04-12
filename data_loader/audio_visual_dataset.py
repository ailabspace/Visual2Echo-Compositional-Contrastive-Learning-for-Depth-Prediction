import os.path
import time
import librosa
import h5py
import random
import math
import numpy as np
import glob
import scipy.signal as signal
from scipy.signal import hilbert, resample
import torch
import torchaudio
from util.augment import (AddGaussianNoise, VolumeJitter, Deafening,
                          Trimming, RandomApplyTransform, ChannelFlip)
import pickle
from PIL import Image, ImageEnhance
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.utils.data as data
from medpy.filter.smoothing import anisotropic_diffusion  # type: ignore

# Per-worker scene cache to avoid COW-copying during DataLoader fork
_worker_scene_cache = {}


def _get_scene_data(scenes_dir, scene):
    """Load a scene pkl into the per-worker cache on first access."""
    if scene not in _worker_scene_cache:
        scene_pkl = os.path.join(scenes_dir, scene + '.pkl')
        with open(scene_pkl, 'rb') as f:
            _worker_scene_cache[scene] = pickle.load(f)
    return _worker_scene_cache[scene]


def normalize(samples, desired_rms=0.1, eps=1e-4):
    rms = np.maximum(eps, np.sqrt(np.mean(samples ** 2)))
    samples = samples * (desired_rms / rms)
    return samples


def generate_spectrogram(audioL, audioR, winl=32, hop_length=None,
                         use_ipd=False, log_scale=False, use_ild=False,
                         use_magdiff=False, n_fft=512):
    """Generate multi-channel spectrogram from binaural audio.

    Args:
        audioL, audioR: left and right audio channels
        winl: STFT window length
        hop_length: STFT hop length (None = n_fft // 4)
        use_ipd: add interaural phase difference channel
        log_scale: apply log1p compression
        use_ild: add interaural level difference channel
        use_magdiff: add signed magnitude difference channel
        n_fft: FFT size

    Returns:
        np.ndarray of shape [C, F, T] where C depends on feature flags
    """
    if hop_length is None:
        hop_length = n_fft // 4
    channel_1_spec = librosa.stft(audioL, n_fft=n_fft, win_length=winl,
                                  hop_length=hop_length)
    channel_2_spec = librosa.stft(audioR, n_fft=n_fft, win_length=winl,
                                  hop_length=hop_length)
    mag_L_raw = np.abs(channel_1_spec)
    mag_R_raw = np.abs(channel_2_spec)
    mag_L = mag_L_raw.copy()
    mag_R = mag_R_raw.copy()
    if log_scale:
        mag_L = np.log1p(mag_L)
        mag_R = np.log1p(mag_R)
    mag_L = np.expand_dims(mag_L, axis=0)
    mag_R = np.expand_dims(mag_R, axis=0)
    channels = [mag_L, mag_R]
    if use_ipd:
        ipd = np.angle(channel_1_spec * np.conj(channel_2_spec))
        channels.append(np.expand_dims(ipd, axis=0))
    if use_ild:
        ild = 20 * np.log10((mag_L_raw + 1e-8) / (mag_R_raw + 1e-8))
        channels.append(np.expand_dims(ild, axis=0))
    if use_magdiff:
        magdiff = mag_L_raw - mag_R_raw
        if log_scale:
            magdiff = np.sign(magdiff) * np.log1p(np.abs(magdiff))
        channels.append(np.expand_dims(magdiff, axis=0))
    return np.concatenate(channels, axis=0)


def scale_to_ultrasonic(chirp, sample_rate, target_sr, carrier_freq=50000):
    """Resample and modulate audio to ultrasonic range."""
    num_samples = chirp.shape[-1]
    scaling_factor = (target_sr / sample_rate)
    new_num_samples = int(num_samples * scaling_factor)

    ultrasonic_left = resample(chirp[0, :], new_num_samples)
    ultrasonic_right = resample(chirp[1, :], new_num_samples)

    t = np.arange(new_num_samples) / target_sr
    analytic_signal_L = hilbert(ultrasonic_left)
    analytic_signal_R = hilbert(ultrasonic_right)

    modulated_L = np.real(analytic_signal_L * np.exp(2j * np.pi * carrier_freq * t))
    modulated_R = np.real(analytic_signal_R * np.exp(2j * np.pi * carrier_freq * t))

    return np.vstack((modulated_L, modulated_R))


def process_image(rgb, augment):
    if augment:
        enhancer = ImageEnhance.Brightness(rgb)
        rgb = enhancer.enhance(random.random() * 0.6 + 0.7)
        enhancer = ImageEnhance.Color(rgb)
        rgb = enhancer.enhance(random.random() * 0.6 + 0.7)
        enhancer = ImageEnhance.Contrast(rgb)
        rgb = enhancer.enhance(random.random() * 0.6 + 0.7)
    return rgb


def parse_all_data(root_path, scenes):
    """Load image/depth data for the requested scenes.

    Supports two strategies:
    1. Per-scene pkl files: if a scenes/ sub-directory exists with {scene}.pkl files
    2. Monolithic pkl (legacy): loads the entire pkl file
    """
    data_idx_all = []

    split_name = os.path.splitext(os.path.basename(root_path))[0]
    scenes_dir = os.path.join(os.path.dirname(root_path), 'scenes', split_name)

    use_per_scene = os.path.isdir(scenes_dir) and any(
        os.path.isfile(os.path.join(scenes_dir, s + '.pkl')) for s in scenes
    )

    if use_per_scene:
        import json
        data_dict = {}
        index_path = os.path.join(scenes_dir, 'index.json')
        if os.path.isfile(index_path):
            print(f"[DataLoader] Loading index from: {index_path}")
            with open(index_path) as f:
                full_index = json.load(f)
            for scene in scenes:
                if scene not in full_index:
                    print(f"[WARN] Scene {scene} missing from index")
                    continue
                entries = full_index[scene]
                imgs = [f"{scene}/{loc}/{ori}" for (loc, ori) in entries]
                data_idx_all += imgs
                print(f"SCENE: {scene}, IMGS: {len(imgs)}")
            print(f"[DataLoader] Pre-loading {len(scenes)} scene pkls...")
            for scene in scenes:
                scene_pkl = os.path.join(scenes_dir, scene + '.pkl')
                if os.path.isfile(scene_pkl):
                    with open(scene_pkl, 'rb') as f:
                        raw = pickle.load(f)
                    data_dict[scene] = {k: {'rgb': v['rgb'], 'depth': v['depth']}
                                        for k, v in raw.items()}
                    del raw
        else:
            print(f"[DataLoader] Loading per-scene pkls...")
            for scene in scenes:
                scene_pkl = os.path.join(scenes_dir, scene + '.pkl')
                if not os.path.isfile(scene_pkl):
                    print(f"[WARN] Per-scene pkl missing for {scene}")
                    continue
                with open(scene_pkl, 'rb') as f:
                    raw = pickle.load(f)
                data_dict[scene] = {k: {'rgb': v['rgb'], 'depth': v['depth']}
                                    for k, v in raw.items()}
                del raw
                imgs = ['/'.join([scene, str(loc), str(ori)])
                        for (loc, ori) in list(data_dict[scene].keys())]
                data_idx_all += imgs
                print(f"SCENE: {scene}, IMGS: {len(imgs)}")
    else:
        print(f"[DataLoader] Loading monolithic pkl: {root_path}")
        with open(root_path, 'rb') as f:
            data_dict = pickle.load(f)
        for scene in scenes:
            imgs = ['/'.join([scene, str(loc), str(ori)])
                    for (loc, ori) in list(data_dict[scene].keys())]
            data_idx_all += imgs
            print(f"SCENE: {scene}, IMGS: {len(imgs)}")

    return data_idx_all, data_dict


class AudioVisualDataset(data.Dataset):
    """Audio-visual dataset for training with binaural echoes and images."""

    def __init__(self):
        super(AudioVisualDataset, self).__init__()

    def initialize(self, opt):
        self.opt = opt
        if self.opt.dataset == 'mp3d':
            self.data_idx, self.data = parse_all_data(
                os.path.join(self.opt.img_path, opt.mode + '.pkl'),
                self.opt.scenes[opt.mode])
            self._scenes_dir = None
            self.win_length = 32
            self.hop_length = None
        if self.opt.dataset == 'replica':
            self._scenes_dir = None
            self.data_idx, self.data = parse_all_data(
                self.opt.img_path, self.opt.scenes[opt.mode])
            _n_fft = getattr(opt, 'audio_nfft', 512)
            self.win_length = min(256, _n_fft)
            _hop_override = getattr(opt, 'audio_hop_length', 0)
            self.hop_length = _hop_override if _hop_override > 0 else 32

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        self.vision_transform = transforms.Compose(
            [transforms.ToTensor(), normalize])
        self.audio_transform = transforms.Compose([
            AddGaussianNoise(mean=0, std=0.0001),
        ])
        self.audio_vision_transform = RandomApplyTransform([
            Deafening(),
            ChannelFlip(),
        ], p=0.5)

        self.base_audio_path = self.opt.audio_path
        if self.opt.dataset == 'mp3d':
            self.audio_type = '3ms_sweep_16khz'
        if self.opt.dataset == 'replica':
            self.audio_type = '3ms_sweep'

        self._orientations = [0, 90, 180, 270]
        self.used_orientations = {}
        self._test_cls = {}
        self.white_pixel_threshold = 1.

    def _label_mapping(self, diff):
        return 0 if diff == 0 else 1

    def __getitem__(self, index):
        scene, loc, orn = self.data_idx[index].split('/')

        if (scene, loc) not in self.used_orientations:
            self.used_orientations[(scene, loc)] = []
            self._test_cls[(scene, loc)] = []

        unused_orientations = list(
            set(self._orientations) - set(self.used_orientations[(scene, loc)]))
        if not unused_orientations:
            self.used_orientations[(scene, loc)] = []
            self._test_cls[(scene, loc)] = []
            unused_orientations = self._orientations

        if getattr(self.opt, 'match_audio_orn', False):
            orn_aud = int(orn)
        elif random.random() > 0.5:
            orn_aud = int(orn)
        else:
            orn_aud = random.choice(unused_orientations) if len(unused_orientations) > 1 else unused_orientations[0]
        diff = int(orn) - orn_aud

        if diff == 270:
            orn_lbl = self._label_mapping(90)
        elif diff == -270:
            orn_lbl = self._label_mapping(-90)
        else:
            orn_lbl = self._label_mapping(diff)

        self.used_orientations[(scene, loc)].append(orn_aud)
        self._test_cls[(scene, loc)].append(orn_lbl)

        audio_path = os.path.join(
            self.base_audio_path, scene, self.audio_type, str(orn_aud), loc + '.wav')
        audio, _ = librosa.load(
            audio_path, sr=self.opt.audio_sampling_rate,
            mono=False, duration=self.opt.audio_length)

        if self.opt.audio_normalize:
            audio = normalize(audio)

        if self._scenes_dir is not None:
            scene_dict = _get_scene_data(self._scenes_dir, scene)
        else:
            scene_dict = self.data[scene]
        img = Image.fromarray(
            scene_dict[(int(loc), int(orn))]['rgb']).convert('RGB')

        if self.opt.mode == "train":
            img = process_image(img, self.opt.enable_img_augmentation)

        if self.opt.image_transform:
            img = self.vision_transform(img)

        depth = torch.FloatTensor(scene_dict[(int(loc), int(orn))]['depth'])
        depth = depth.unsqueeze(0)

        if self.opt.mode == "train":
            depth_norm = F.normalize(depth, dim=0)
            white_pixel_ratio = torch.sum(depth_norm > 0.9).item() / depth_norm.numel()
            if white_pixel_ratio >= self.white_pixel_threshold:
                return self.__getitem__((index + 1) % len(self))

            if self.opt.enable_cropping:
                RESOLUTION = self.opt.image_resolution
                w_offset = RESOLUTION - 128
                h_offset = RESOLUTION - 128
                left = random.randrange(0, w_offset + 1)
                upper = random.randrange(0, h_offset + 1)
                img = img[:, left:left + 128, upper:upper + 128]
                depth = depth[:, left:left + 128, upper:upper + 128]

            if random.random() < 0.5 and not getattr(self.opt, 'no_audio_augment', False):
                audio_t = torch.FloatTensor(audio)
                audio_t = self.audio_transform(audio_t)
                audio_t, depth = self.audio_vision_transform((audio_t, depth))
                audio = audio_t.numpy()

        audio_spec_both = torch.FloatTensor(generate_spectrogram(
            audio[0, :], audio[1, :], self.win_length,
            hop_length=self.hop_length,
            use_ipd=getattr(self.opt, 'use_ipd', False),
            log_scale=getattr(self.opt, 'log_spectrogram', False),
            use_ild=getattr(self.opt, 'use_ild', False),
            use_magdiff=getattr(self.opt, 'use_magdiff', False),
            n_fft=getattr(self.opt, 'audio_nfft', 512)))

        return {'img': img, 'depth': depth, 'audio': audio_spec_both, 'orn': orn_lbl}

    def __len__(self):
        return len(self.data_idx)

    def name(self):
        return 'AudioVisualDataset'
