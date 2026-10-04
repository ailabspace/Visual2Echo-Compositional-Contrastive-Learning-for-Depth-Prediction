import argparse
import datetime
import os

from util import util


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        p = self.parser
        p.add_argument('--img_path', type=str, default='')
        p.add_argument('--metadatapath', type=str, default='dataset/metadata/mp3d')
        p.add_argument('--audio_path', type=str, default='')
        p.add_argument('--nThreads', default=8, type=int)
        p.add_argument('--batchSize', type=int, default=64)
        p.add_argument('--audio_length', default=0.06, type=float)
        p.add_argument('--audio_normalize', action='store_true')
        p.add_argument('--biosonar_rgb_dir', type=str, default='')
        p.add_argument('--teacher_img_size', type=int, default=224)
        p.add_argument('--audio_mono_mode', type=str, default='none', choices=['none', 'left', 'right'])
        p.add_argument('--audio_crop_pre_roll', type=float, default=-1.0)
        p.add_argument('--audio_bandpass_lo', type=float, default=0.0)
        p.add_argument('--audio_bandpass_hi', type=float, default=0.0)
        p.add_argument('--spec_log_gain', type=float, default=1.0)
        p.add_argument('--audio_butter_order', type=int, default=-1)
        p.add_argument('--val_include_test', action='store_true')
        p.add_argument('--no_image_transform', dest='image_transform', action='store_false', default=True)
        p.add_argument('--dataset', default='mp3d', type=str)
        p.add_argument('--batvision_img_size', type=int, default=128)
        p.add_argument('--depth_target_hw', type=str, default='128,128')
        p.add_argument('--depth_resize_method', type=str, default='bilinear', choices=['bilinear', 'nearest'])
        p.add_argument('--max_depth', default=14.104, type=float)
        p.add_argument('--rgb_teacher', type=str, default='rgbdepth_unet', choices=['rgbdepth_unet', 'moge_v2'])
        p.add_argument('--moge_model_id', type=str, default='Ruicheng/moge-2-vitl-normal')
        p.add_argument('--moge_num_tokens', type=int, default=0)
        p.add_argument('--moge_resolution_level', type=int, default=9)
        p.add_argument('--moge_use_fp16', action='store_true')
        p.add_argument('--moge_cache_enc_dim_out', type=int, default=1024)
        p.add_argument('--exp_name', type=str, default=datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        p.add_argument('--use_ipd', action='store_true')
        p.add_argument('--use_ild', action='store_true')
        p.add_argument('--ild_clip_db', type=float, default=40.0)
        p.add_argument('--use_magdiff', action='store_true')
        p.add_argument('--audio_nfft', type=int, default=512)
        p.add_argument('--audio_hop_length', type=int, default=0)
        p.add_argument('--audio_win_length', type=int, default=0)
        p.add_argument('--log_spectrogram', action='store_true')
        p.add_argument('--use_specaugment', action='store_true')
        p.add_argument('--no_audio_augment', action='store_true')
        p.add_argument('--spec_eq_db', type=float, default=0.0)
        p.add_argument('--spec_eq_p', type=float, default=0.8)
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        opt = self.opt = self.parser.parse_args()
        opt.mode = self.mode
        opt.isTrain = self.isTrain
        opt.enable_img_augmentation = self.enable_data_augmentation
        n_fft = opt.audio_nfft
        n_freq = n_fft // 2 + 1

        if opt.dataset == 'mp3d':
            sr = 16000
            opt.audio_win_length = opt.audio_win_length if opt.audio_win_length > 0 else 32
            opt.audio_hop_length = opt.audio_hop_length if opt.audio_hop_length > 0 else 32
            opt.max_depth = 10.0
            opt.scenes = {s: self._scenes(f'mp3d_scenes_{s}.txt') for s in ('train', 'val', 'test')}
        elif opt.dataset == 'replica':
            sr = 44100
            opt.audio_hop_length = opt.audio_hop_length if opt.audio_hop_length > 0 else 32
            opt.audio_win_length = opt.audio_win_length if opt.audio_win_length > 0 else min(256, n_fft)
            opt.max_depth = 14.104
            opt.scenes = {s: self._scenes(f'replica_{s}.txt') for s in ('train', 'val', 'test')}
        elif opt.dataset == 'biosonar':
            sr = 320000
            opt.audio_hop_length = opt.audio_hop_length if opt.audio_hop_length > 0 else 64
            opt.audio_win_length = opt.audio_win_length if opt.audio_win_length > 0 else 128
            if abs(opt.audio_length - 0.06) < 1e-6:
                opt.audio_length = 0.075
            if opt.audio_bandpass_lo == 0.0 and opt.audio_bandpass_hi == 0.0:
                opt.audio_bandpass_lo, opt.audio_bandpass_hi = 20000.0, 100000.0
            if opt.audio_butter_order == -1:
                opt.audio_butter_order = 4
            if opt.audio_bandpass_hi > 0.0:
                bin_hz = sr / n_fft
                lo = max(0, int(round(opt.audio_bandpass_lo / bin_hz)))
                hi = min(n_freq, int(round(opt.audio_bandpass_hi / bin_hz)) + 1)
                n_freq = hi - lo
            if opt.max_depth == 14.104:
                opt.max_depth = 10.0
            opt.scenes = {s: ['biosonar'] for s in ('train', 'val', 'test')}
        else:
            raise ValueError(f'unknown dataset {opt.dataset}')
        opt.audio_sampling_rate = sr
        opt.teacher_max_depth = opt.max_depth
        opt.audio_shape = [2, n_freq, 1 + int(opt.audio_length * sr) // opt.audio_hop_length]
        if opt.use_ipd:
            opt.audio_shape[0] = 4
        if opt.use_ild:
            opt.audio_shape[0] += 1
        if opt.use_magdiff:
            opt.audio_shape[0] += 1

        args = sorted(vars(opt).items())
        print('------------ Options -------------')
        for k, v in args:
            print(f'{k}: {v}')
        print('-------------- End ----------------')
        opt.expr_dir = os.path.join('checkpoint', opt.exp_name, opt.dataset)
        util.mkdirs(opt.expr_dir)
        with open(os.path.join(opt.expr_dir, 'opt.txt'), 'wt') as f:
            f.write('------------ Options -------------\n')
            for k, v in args:
                f.write(f'{k}: {v}\n')
            f.write('-------------- End ----------------\n')
        return opt

    def _scenes(self, name):
        with open(os.path.join(self.opt.metadatapath, name)) as f:
            return [x.strip() for x in f.readlines()]
