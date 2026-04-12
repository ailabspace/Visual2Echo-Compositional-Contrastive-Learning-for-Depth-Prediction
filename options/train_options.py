from .base_options import BaseOptions


class TrainOptions(BaseOptions):
	def initialize(self):
		BaseOptions.initialize(self)

		# Training schedule
		self.parser.add_argument('--display_freq', type=int, default=100,
			help='frequency of displaying average loss')
		self.parser.add_argument('--niter', type=int, default=300,
			help='number of training epochs')
		self.parser.add_argument('--learning_rate_decrease_itr', type=int, default=-1,
			help='epoch interval for LR decay (-1 = disabled)')
		self.parser.add_argument('--decay_factor', type=float, default=0.94)
		self.parser.add_argument('--validation_on', action='store_true')
		self.parser.add_argument('--validation_freq', type=int, default=400)
		self.parser.add_argument('--epoch_save_freq', type=int, default=5)
		self.parser.add_argument('--resume_dir', type=str, help='path to trained weights')
		self.parser.add_argument('--last_epoch', default=None, type=int)
		self.parser.add_argument('--freeze_nets', action='store_true',
			help='freeze teacher networks')
		self.parser.add_argument('--enable_cropping', action='store_true')

		# Teacher caching
		self.parser.add_argument('--teacher_cache_path', type=str, default='',
			help='HDF5 teacher cache (from precompute_teacher_latents.py)')
		self.parser.add_argument('--validation_cache_path', type=str, default='',
			help='HDF5 val-split teacher cache')

		# Training techniques
		self.parser.add_argument('--gradient_checkpointing', action='store_true',
			help='enable gradient checkpointing to save VRAM')
		self.parser.add_argument('--detect_anomaly', action='store_true',
			help='enable autograd anomaly detection (slow, for debugging)')
		self.parser.add_argument('--accumulation_steps', type=int, default=1,
			help='gradient accumulation steps')
		self.parser.add_argument('--use_fp16', action='store_true',
			help='enable fp16 mixed precision training')
		self.parser.add_argument('--seed', type=int, default=0,
			help='random seed (0 = no seed)')

		# LR scheduling
		self.parser.add_argument('--cosine_T_max', type=int, default=0,
			help='CosineAnnealingLR T_max (0 = disabled)')
		self.parser.add_argument('--lr_plateau_factor', type=float, default=0.0,
			help='ReduceLROnPlateau factor (0 = disabled)')
		self.parser.add_argument('--lr_plateau_patience', type=int, default=5)
		self.parser.add_argument('--warmup_steps', type=int, default=0)
		self.parser.add_argument('--ema_decay', type=float, default=0.0,
			help='EMA decay for audio net (0 = disabled)')
		self.parser.add_argument('--early_stop_patience', type=int, default=0,
			help='early stopping patience (0 = disabled)')

		# Loss weights
		self.parser.add_argument('--depth_loss_type', type=str, default='log',
			choices=['log', 'silog', 'l1', 'l2', 'berhu'])
		self.parser.add_argument('--lambda_teacher_depth', type=float, default=0.5)
		self.parser.add_argument('--lambda_depth', type=float, default=1.0)
		self.parser.add_argument('--lambda_mat', type=float, default=0.7)
		self.parser.add_argument('--lambda_ccl_depth', type=float, default=0.3)
		self.parser.add_argument('--lambda_ct', type=float, default=0.05)
		self.parser.add_argument('--lambda_multiscale', type=float, default=0.0)
		self.parser.add_argument('--lambda_laplacian', type=float, default=0.0)
		self.parser.add_argument('--lambda_ssim', type=float, default=0.0)
		self.parser.add_argument('--lambda_grad', type=float, default=0.0)
		self.parser.add_argument('--ccl_temperature', type=float, default=1.0)
		self.parser.add_argument('--berhu_threshold', type=float, default=0.2)
		self.parser.add_argument('--ccl_depth_loss', type=str, default='log',
			choices=['log', 'berhu'])
		self.parser.add_argument('--berhu_teacher', action='store_true')
		self.parser.add_argument('--use_se_skips', action='store_true',
			help='enable SE attention on skip connections')

		# Model architecture
		self.parser.add_argument('--backbone', type=str, default='Resnet18',
			choices=['Resnet18', 'Resnet34', 'Legacy'])
		self.parser.add_argument('--init_audiodepth_weight', type=str, default='',
			help='pretrained audio net weights')
		self.parser.add_argument('--init_material_weight', type=str, default='',
			help='pretrained material net weights')
		self.parser.add_argument('--unet_ngf', type=int, default=64)
		self.parser.add_argument('--unet_input_nc', type=int, default=3)
		self.parser.add_argument('--unet_output_nc', type=int, default=1)

		# Optimizer
		self.parser.add_argument('--lr_backbone_main', type=float, default=1e-4)
		self.parser.add_argument('--lr_audio', type=float, default=0.0001)
		self.parser.add_argument('--optimizer', default='adam', type=str)
		self.parser.add_argument('--beta1', default=0.9, type=float)
		self.parser.add_argument('--weight_decay', default=0.0005, type=float)

		self.mode = "train"
		self.isTrain = True
		self.enable_data_augmentation = True
