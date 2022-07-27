"""
Contains wrapper for torch.utils.data.Dataset derived classes
"""

import os
import numpy as np
import torch
import torch.utils.data
from albumentations import HorizontalFlip, VerticalFlip, Rotate, RandomRotate90, RandomBrightnessContrast, GaussNoise
from augmenter import Augmenter
from scipy.ndimage import gaussian_gradient_magnitude

class NpyDataset(torch.utils.data.Dataset):
	'''
	A supervised learning dataset class to handle serialised
	numpy data, for example images.

	Data consists of float `.npy` files of fixed shape.
	Observations and labels are given by different folders
	containing files with same names.
	'''
	def __init__(self, x_dir, y_dir):
		"""
		Instantiate .npy file dataset.

		:param x_dir: (str) observation directory
		:param y_dir: (str) label directory
		"""

		self.x_dir = x_dir
		self.y_dir = y_dir

		# sort is needed for order in data
		self.x_list = np.sort(os.listdir(x_dir))
		self.y_list = np.sort(os.listdir(y_dir))

		transforms = [
			VerticalFlip(p=.2),
			HorizontalFlip(p=.2),
			RandomRotate90(p=.3)]
			#RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=.2),
			#GaussNoise(var_limit=(0.0, 20.0), p=.2)]

		self.augmenter = Augmenter(list_of_transforms=transforms, p=.9)


	def __len__(self):
		return len(self.x_list)

	def __getitem__(self, idx: int) -> tuple:

		img_name = os.path.join(self.x_dir, self.x_list[idx])
		img = np.rollaxis(np.load(img_name), 0, 3)
		#print("img", img.shape)

		padding = 0
		img = np.pad(img, ((padding,padding),(padding,padding),(0,0)), "reflect") # pad to reach side of 2**n

		label_name = os.path.join(self.y_dir, self.y_list[idx])
		label = np.rollaxis(np.load(label_name), 0, 3)
		label = label-label.min()
		#print("label", label.shape)

		label = np.pad(label, ((padding,padding),(padding,padding),(0,0)), "reflect")

		# albumentations needs channel last
		img, label = self.augmenter(img, label)

		# pytorch needs channels first
		img_tensor = torch.Tensor(img).permute((2, 0, 1))
		label_tensor = torch.Tensor(label).permute((2, 0, 1))

		return img_tensor, label_tensor


class NpyPredictionDataset(torch.utils.data.Dataset):
	'''
	A dataset class to handle prediction on serialised numpy data,
	for example images.

	Data consists of float `.npy` files of fixed shape.
	'''
	def __init__(self, files):
		"""
		Instantiate .npy file dataset.

		:param files: (list) list of files to predict on
		"""

		self.files = files

	def __len__(self):
		return len(self.files)

	def __getitem__(self, idx: int) -> tuple:
		padding = 3
		img = np.rollaxis(np.load(self.files[idx]), 0, 3)
		img = np.pad(img, ((padding,padding),(padding,padding),(0,0)), "reflect")
		img = torch.Tensor(img).permute((2, 0, 1))
		return self.files[idx], img
