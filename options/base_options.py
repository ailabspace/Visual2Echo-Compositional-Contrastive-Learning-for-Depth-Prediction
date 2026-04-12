#!/usr/bin/env python

import argparse
import os
from util import util
import torch
import datetime


class BaseOptions():
	def __init__(self):
		self.parser = argparse.ArgumentParser(
			formatter_class=argparse.ArgumentDefaultsHelpFormatter)
		self.initialized = False

	def initialize(self):
		# Data paths
		self.parser.add_argument('--img_path', type=str, default='',
			help='path to the .pkl file containing image/depth data')
		self.parser.add_argument('--metadatapath', type=str, default='dataset/metadata/mp3d',
			help='path to metadata file for different splits')
		self.parser.add_argument('--pkl_cache_dir', type=str, default='/tmp/visual_echoes_cache',
			help='directory to cache pkl files extracted from the dataset')
		self.parser.add_argument('--audio_path', type=str, default='',
			help='path to the folder containing echo responses')
		self.parser.add_argument('--checkpoints_dir', type=str, default='',
			help='path to save checkpoints')

		# Hardware
		self.parser.add_argument('--gpu_ids', type=str, default='0',
			help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
		self.parser.add_argument('--nThreads', default=8, type=int,
			help='number of data loading threads')

		# Data processing
		self.parser.add_argument('--batchSize', type=int, default=64)
		self.parser.add_argument('--audio_length', default=0.06, type=float,
			help='audio length in seconds')
		self.parser.add_argument('--audio_normalize', action='store_true',
			help='normalize audio to fixed RMS')
		self.parser.add_argument('--no_image_transform', dest='image_transform',
			action='store_false', default=True,
			help='disable image transforms')
		self.parser.add_argument('--image_resolution', default=128, type=int)
		self.parser.add_argument('--dataset', default='mp3d', type=str,
			help='dataset: replica or mp3d')
		self.parser.add_argument('--max_depth', default=10, type=float,
			help='maximum depth value')
		self.parser.add_argument('--exp_name', type=str,
			default=datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
			help="experiment name for logging")

		# Audio features
		self.parser.add_argument('--use_ipd', action='store_true',
			help='add interaural phase difference (IPD) channel')
		self.parser.add_argument('--use_ild', action='store_true',
			help='add interaural level difference (ILD) channel')
		self.parser.add_argument('--use_magdiff', action='store_true',
			help='add signed magnitude difference channel')
		self.parser.add_argument('--audio_nfft', type=int, default=512,
			help='n_fft for STFT spectrogram')
		self.parser.add_argument('--audio_hop_length', type=int, default=0,
			help='STFT hop length override (0=use dataset default)')
		self.parser.add_argument('--log_spectrogram', action='store_true',
			help='apply log1p to spectrogram magnitudes')
		self.parser.add_argument('--match_audio_orn', action='store_true',
			help='always use audio from the matched orientation')
		self.parser.add_argument('--use_specaugment', action='store_true',
			help='apply SpecAugment during training')
		self.parser.add_argument('--no_audio_augment', action='store_true',
			help='disable all audio augmentations')

		self.initialized = True

	def parse(self):
		if not self.initialized:
			self.initialize()
		self.opt = self.parser.parse_args()

		self.opt.mode = self.mode
		self.opt.isTrain = self.isTrain
		self.opt.enable_img_augmentation = self.enable_data_augmentation

		# Dataset-specific parameters
		self.opt.scenes = {}
		if self.opt.dataset == 'mp3d':
			self.opt.audio_shape = [2, 257, 121]
			self.opt.audio_sampling_rate = 16000
			self.opt.max_depth = min(self.opt.max_depth, 10)
			self.opt.teacher_max_depth = self.opt.max_depth

			for split in ['train', 'val', 'test']:
				scenes_file = os.path.join(
					self.opt.metadatapath, f'mp3d_scenes_{split}.txt')
				with open(scenes_file) as f:
					self.opt.scenes[split] = [x.strip() for x in f.readlines()]

		if self.opt.dataset in ('mp3d_custom', 'mp3d_map', 'mp3d_map_odom_1m'):
			self.opt.audio_shape = [2, 1025, 201]
			self.opt.audio_sampling_rate = 90000
			self.opt.audio_length = 0.3

			for split in ['train', 'val']:
				scenes_file = os.path.join(
					self.opt.metadatapath, f'{self.opt.dataset}_scenes_{split}.txt')
				with open(scenes_file) as f:
					self.opt.scenes[split] = [x.strip() for x in f.readlines()]

		if self.opt.dataset == 'replica':
			_sr = 44100
			_n_fft = getattr(self.opt, 'audio_nfft', 512)
			_hop = self.opt.audio_hop_length if self.opt.audio_hop_length > 0 else 32
			_n_frames = 1 + int(self.opt.audio_length * _sr) // _hop
			self.opt.audio_shape = [2, _n_fft // 2 + 1, _n_frames]
			self.opt.audio_sampling_rate = _sr
			self.opt.max_depth = min(self.opt.max_depth, 10.0)
			self.opt.teacher_max_depth = 14.104

			for split in ['train', 'val', 'test']:
				scenes_file = os.path.join(
					self.opt.metadatapath, f'replica_{split}.txt')
				with open(scenes_file) as f:
					self.opt.scenes[split] = [x.strip() for x in f.readlines()]

		# Update audio channels based on feature flags
		if getattr(self.opt, 'use_ipd', False):
			self.opt.audio_shape[0] = 3
		if getattr(self.opt, 'use_ild', False):
			self.opt.audio_shape[0] += 1
		if getattr(self.opt, 'use_magdiff', False):
			self.opt.audio_shape[0] += 1

		str_ids = self.opt.gpu_ids.split(',')
		self.opt.gpu_ids = []
		for str_id in str_ids:
			id = int(str_id)
			if id >= 0:
				self.opt.gpu_ids.append(id)

		args = vars(self.opt)
		print('------------ Options -------------')
		for k, v in sorted(args.items()):
			print('%s: %s' % (str(k), str(v)))
		print('-------------- End ----------------')

		# Save to disk
		expr_dir = os.path.join('checkpoint', self.opt.exp_name, self.opt.dataset)
		self.opt.expr_dir = expr_dir
		util.mkdirs(expr_dir)
		file_name = os.path.join(expr_dir, 'opt.txt')
		with open(file_name, 'wt') as opt_file:
			opt_file.write('------------ Options -------------\n')
			for k, v in sorted(args.items()):
				opt_file.write('%s: %s\n' % (str(k), str(v)))
			opt_file.write('-------------- End ----------------\n')

		return self.opt
