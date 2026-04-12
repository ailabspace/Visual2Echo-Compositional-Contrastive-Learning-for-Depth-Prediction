from .base_options import BaseOptions


class TestOptions(BaseOptions):
	def initialize(self):
		BaseOptions.initialize(self)

		self.mode = "test"
		self.isTrain = False
		self.enable_data_augmentation = False
		self.enable_cropping = True

		self.parser.add_argument('--audio_std_weight_dir', type=str,
			help='directory path to weights')
		self.parser.add_argument('--audio_std_weight_pth', type=str,
			help='weight file name')
		self.parser.add_argument('--init_material_weight', type=str, default='',
			help='path to pretrained material net')
